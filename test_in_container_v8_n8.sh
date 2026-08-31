#!/bin/bash
# set -euo pipefail

# Held-out evaluation for the v8 SELECTION arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8  @ global_step_200
#
# v8 keeps v7b's substrate verbatim (v6 outcome reward, v5 format gate,
# turns-aware budget with <tool_response> stripped, GRPO n=8) and spends its
# shaping budget one stage further down the funnel:
#   r_select  -- is the answer a CONTINUATION of something this user played,
#                per the ordered sequences already inside the retrieved docs
#                (+ grounding credit, - penalty for re-recommending history)
#   r_cover   -- exact GT-in-retrieved-titles earns full credit, with v7's
#                cosine kept as a dense tail so GRPO groups are not all-zero
#   length budget no longer charges for <tool_call> text either
#                (RTHINK_LEN_COUNT_CALLS=1 restores the v7b behavior)
# See verl/utils/reward_score/reward_retrieval/v8/{README.md,selection.py}.
#
# WHY step 200
#   The comparison step is LOCKED at 200 for every arm, decided before the run
#   (v8/README.md). It is not a "best checkpoint": data.val_files=test.parquet,
#   so every val-aux curve is computed on the TEST set and any step chosen by
#   looking at those curves would be selected on test. On v5 the held-out
#   metric also DECLINED after 300 while train reward kept climbing.
#
# WHY EVAL_REPEATS defaults to 6 (v7b used 3)
#   Sized in v8/README.md: 6 decodes give ~80% power at alpha=0.05 for the
#   pre-registered effects (coverage 3.13% -> 4.70%, selection 5.4% -> 15%,
#   HR@5 0.30% -> 0.90%) under the CONSERVATIVE unpaired test, and match the
#   baseline arm's 6 existing step-200 decodes 1:1 for the paired comparison.
#   Three decodes would not clear the selection gate. ~26 min per 3 decodes
#   (job 875723), so 6 fit inside the 12 h wall with a wide margin.
#
# WHAT TO READ, IN ORDER -- the three pre-registered gates. ALL must pass.
#   Gate 0 validity  : retr% >= 95, answer rate >= 95%, turns_max < 10,
#                      docs/ret >= 3.  An arm that "wins" by not retrieving
#                      has not won -- that is exactly how v7 produced an HR
#                      tie at a 5% retrieval rate. Fail here => DISCARD, do
#                      not interpret anything below.
#   Gate 1 mechanism : coverage >= 4.0% (vs 3.13% baseline) OR
#                      P(pick GT | GT in docs) >= 15% (vs 3.9% chance).
#                      At ~3 HR events per 1000 an HR swing WILL appear by
#                      chance; requiring the mechanism to move first is what
#                      stops a decode-noise draw from being read as a result.
#   Gate 2 outcome   : HR@5 >= 0.009, prompt-level PAIRED test against the six
#                      baseline decodes on the identical 1000 prompts.
#   Gates 0+1 without 2 is a real partial result. Gate 2 without 1 is not a
#   win; it is noise and must be re-decoded.
#
# PRE-REGISTERED CEILING (write it down before reading any number):
#   coverage 2.77% x successor-cue recall 13% = HR@1 ~0.0036, ~3x baseline.
#   A LARGER jump than that is more likely a measurement artifact than a win
#   and must be investigated before it is reported.
#
# Usage:
#   bash test_in_container_v8_n8.sh                      # step 200, 6 decodes
#   EVAL_REPEATS=1 bash test_in_container_v8_n8.sh       # quick smoke decode
#   CHECKPOINT_STEP=150 bash test_in_container_v8_n8.sh  # a different step
#
# Env vars: everything test_in_container_rthink.sh accepts works here
# (FORCE_MERGE, GEN_BATCH_SIZE, EVAL_CATEGORY, CUDA_VISIBLE_DEVICES, ...).
# This script only sets the arm's defaults and the post-decode audits; the
# merge -> generate -> json -> eval.py pipeline is delegated so every arm runs
# byte-identical evaluation code.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8"
export CHECKPOINT_STEP=${CHECKPOINT_STEP:-200}
export EVAL_REPEATS=${EVAL_REPEATS:-6}
export RTHINK_RUNS="${EXPERIMENT_NAME}:${CHECKPOINT_STEP}"

CKPT_DIR="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME/global_step_${CHECKPOINT_STEP}"
if [ ! -d "$CKPT_DIR" ]; then
    echo "[fatal] checkpoint not found: $CKPT_DIR" >&2
    echo "        available:" >&2
    ls -d "/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"/global_step_* >&2 2>/dev/null
    exit 1
fi

if [ "$CHECKPOINT_STEP" = "200" ]; then
    STEP_NOTE="(comparison step LOCKED at 200)"
else
    STEP_NOTE="(OFF the locked comparison step 200 -- cross-arm numbers below do NOT apply)"
fi

echo "######################################################################"
echo "# v8 arm: selection reward (r_select + r_cover)"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   checkpoint : global_step_$CHECKPOINT_STEP   $STEP_NOTE"
echo "#   decodes    : $EVAL_REPEATS  (unseeded, temperature=1.0)"
echo "#   prior arm  : test_in_container_v7b_n8.sh      (retrieval restored, HR tie)"
echo "#   control arm: test_in_container_baseline_n8.sh (outcome-only, 6 decodes)"
echo "######################################################################"

# The v8 checkpoint is frozen on disk, so this eval is safe to run while the v8
# training job is still going; the eval launcher also uses a job-scoped tool
# config, so it cannot repoint a running trainer's retriever.

unset EXPERIMENT_NAME
bash test_in_container_rthink.sh
CORE_RC=$?

EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8"
EVAL_ROOT="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/global_step_${CHECKPOINT_STEP}"
# shellcheck disable=SC2086
PREDS=$(ls "$EVAL_ROOT"/*/test_predictions.json "$EVAL_ROOT"/test_predictions.json 2>/dev/null)

if [ -z "$PREDS" ]; then
    echo "[warn] no test_predictions.json under $EVAL_ROOT -- generation likely failed" >&2
    exit $CORE_RC
fi

# ---------------------------------------------------------------------------
# [5/6] Gate 0 + the coverage branch of Gate 1.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [5/6] Decode behavior -- Gate 0 (validity) + coverage (Gate 1a)"
echo "######################################################################"
PYTHONPATH="$PROJECT_DIR" python3 \
    verl/utils/reward_score/reward_retrieval/audits/decode_behavior.py \
    --json "$EVAL_ROOT/decode_behavior.json" $PREDS

# ---------------------------------------------------------------------------
# [6/6] The selection branch of Gate 1 -- v8's actual hypothesis.
# decode_behavior.py stops at coverage; this is the stage v8 was built to move.
# It parses the successor relation with v8/selection.py, the same code the
# reward uses, so a gap against the training curve val-aux/*/is_successor is a
# real train/decode gap rather than two different definitions.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [6/6] Selection funnel -- Gate 1b (P(pick GT | GT in docs) >= 15%)"
echo "######################################################################"
python3 verl/utils/reward_score/reward_retrieval/audits/decode_selection.py \
    --json "$EVAL_ROOT/decode_selection.json" $PREDS

echo ""
echo "Reference levels @ global_step_200 (pooled, corrected metrics, 2026-07-30):"
echo "                    baseline-n8 (6 dec)   v7b (3 dec)"
echo "  retr%                    98.8-99.5         99.9-100.0"
echo "  answer rate              86.9-89.4%        99.3-99.6%"
echo "  coverage (exact)         3.13%             2.77%"
echo "  pick GT | GT in docs     0/172  (0.0%)     2/83  (2.4%)   chance 3.9-5.4%"
echo "  GT in successor set      15.4%             13.3%"
echo "  answer in successor set  2.9%              7.8%"
echo "  HR@1  (per decode)       0.001 x4, 0.000 x2   0.002 / 0.001 / 0.000"
echo "  HR@5  (per decode)       0.002-0.004       0.003 / 0.003 / 0.000"
echo ""
echo "DECISION:"
echo "  Gate 0 fails            -> DISCARD the run, do not compare (v7 rule)."
echo "  succ% up, pickGT flat   -> the rule was learned but does not carry the"
echo "                             GT; the cue is exhausted, not mis-weighted."
echo "  coverage up, pickGT flat-> coverage-only gain; 4.7% x 5.4% ~ HR@1 0.0025,"
echo "                             UNDER the bar -- must not be claimed as a win."
echo "  HR up, Gate 1 flat      -> decode noise. Re-decode; do not report."
echo ""
echo "Append the result as a new row in the workspace-root TEST_OUTPUT.md."

exit $CORE_RC
