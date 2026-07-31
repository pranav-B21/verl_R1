#!/bin/bash
# set -euo pipefail

# Held-out evaluation for the M3 v7b arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7b-n8
#
# v7b is the CORRECTED relaunch of v7. Same reward family (r_retqual at
# tau=0.20, W_COVGAIN=0, GRPO n=8); what changed is the length penalty:
#   Fix 1 -- _length_penalty no longer counts <tool_response> document text as
#            model output (it was ingesting retrieved docs and charging the
#            rollout for them).
#   Fix 2 -- turns-aware budget:
#            budget = LEN_SOFT + LEN_PER_TURN * min(credited_turns, 3)
#            with LEN_SOFT raised 600 -> 1200, and a credited turn requiring a
#            non-empty tool response so bare tool-call spam earns no budget.
#
# WHY this matters for reading the eval
#   Under v7's penalty the retrieve-minus-abstain reward gap was -0.231, so GRPO
#   correctly learned to abstain: decode-time retrieval rate collapsed to ~5%
#   against the baseline's ~99%. Offline replay put the gap at -0.107 with Fix 1
#   + LEN_SOFT=1200. If v7b's retrieval rate in [5/5] is still near v7's ~5%,
#   the length penalty was NOT the whole story and the fix did not take. That
#   comparison, not HR, is the primary read.
#
# WHY step 200 is the default
#   Comparison step is LOCKED at 200 for every arm, so v7b lines up with the v7
#   and baseline decodes already on disk. Later checkpoints are not a fairer
#   read: on v5 the held-out metric DECLINED after 300 while train reward kept
#   climbing, so a late checkpoint flatters the train curve.
#
# WHY EVAL_REPEATS defaults to 3
#   Decode is unseeded at temperature=1.0. Three decodes of the SAME v5
#   checkpoint straddled the baseline (HR@1 spread 0.000-0.005), wider than any
#   arm-to-arm gap we are trying to detect. One decode is a sample, not a
#   measurement. The spread across repeats IS the significance bar.
#
# WHY HR is NOT the headline here
#   GT-in-retrieved-docs coverage on test is ~4.6%, so ~95% of eval samples are
#   unwinnable regardless of reward quality, and HR differences of 1-3 hits out
#   of 1000 sit inside decode noise. Read coverage rate first (win: >=7%,
#   strong: >=10%), retrieval rate second, HR third.
#
# Usage:
#   bash test_in_container_v7b_n8.sh                      # step 200, 3 decodes
#   CHECKPOINT_STEP=150 bash test_in_container_v7b_n8.sh  # a different step
#   EVAL_REPEATS=1 bash test_in_container_v7b_n8.sh       # quick single decode
#
# Env vars: every variable accepted by test_in_container_rthink.sh works here
# (FORCE_MERGE, GEN_BATCH_SIZE, EVAL_CATEGORY, CUDA_VISIBLE_DEVICES, ...).
# This script only changes the defaults; it delegates the merge -> generate ->
# json -> eval.py pipeline to that script so all arms run byte-identical code.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7b-n8"
export CHECKPOINT_STEP=${CHECKPOINT_STEP:-200}
export EVAL_REPEATS=${EVAL_REPEATS:-3}
export RTHINK_RUNS="${EXPERIMENT_NAME}:${CHECKPOINT_STEP}"

CKPT_DIR="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME/global_step_${CHECKPOINT_STEP}"
if [ ! -d "$CKPT_DIR" ]; then
    echo "[fatal] checkpoint not found: $CKPT_DIR" >&2
    echo "        available:" >&2
    ls -d "/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"/global_step_* >&2 2>/dev/null
    exit 1
fi

echo "######################################################################"
echo "# M3 arm: v7b retrieval-quality reward (length-penalty fixes 1+2)"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   checkpoint : global_step_$CHECKPOINT_STEP   (comparison step LOCKED at 200)"
echo "#   decodes    : $EVAL_REPEATS  (unseeded, temperature=1.0)"
echo "#   prior arm  : test_in_container_v7_n8.sh      (buggy length penalty)"
echo "#   control arm: test_in_container_baseline_n8.sh (outcome-only)"
echo "######################################################################"

# Delegate merge/generate/json/eval to the shared core, so all arms run
# byte-identical evaluation code. RTHINK_RUNS already wins over the core's
# EXPERIMENT_NAME back-compat branch; unsetting it is belt-and-braces so the
# core cannot fall through to its v6 default if RTHINK_RUNS is ever cleared.
unset EXPERIMENT_NAME
bash test_in_container_rthink.sh
CORE_RC=$?

# ---------------------------------------------------------------------------
# [5/5] Retrieval behavior -- the axis the fix was built to restore.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [5/5] Decode-time retrieval behavior (v7b arm)"
echo "######################################################################"
EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7b-n8"
EVAL_ROOT="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/global_step_${CHECKPOINT_STEP}"
BEHAVIOR_JSON="$EVAL_ROOT/decode_behavior.json"

# shellcheck disable=SC2086
PREDS=$(ls "$EVAL_ROOT"/*/test_predictions.json "$EVAL_ROOT"/test_predictions.json 2>/dev/null)
if [ -z "$PREDS" ]; then
    echo "[warn] no test_predictions.json under $EVAL_ROOT -- generation likely failed" >&2
else
    PYTHONPATH="$PROJECT_DIR" python3 \
        verl/utils/reward_score/reward_retrieval/audits/decode_behavior.py \
        --json "$BEHAVIOR_JSON" $PREDS
fi

echo ""
echo "Reference levels @ global_step_200 (both arms, 3 decodes, 2026-07-21/25):"
echo "  baseline-n8   HR@1 0.001 / 0.001 / 0.001   HR@5 0.002 / 0.003 / 0.004"
echo "                retrieval rate 99.3% / 99.2% / 98.8%"
echo "  v7-n8         HR@1 ~0.002                  retrieval rate ~5%"
echo ""
echo "READ ORDER for this arm -- the fix is a retrieval-behavior claim, not an"
echo "HR claim:"
echo "  1. Retrieval rate. v7 collapsed to ~5%; baseline sits ~99%. If v7b is"
echo "     still in single digits the length-penalty fix did not take, and"
echo "     nothing downstream is worth interpreting."
echo "  2. GT-in-retrieved-docs coverage rate on test. This is the primary"
echo "     metric now. Corpus ceiling is ~4.6%; >=7% is a win, >=10% strong."
echo "  3. HR@1/@5 LAST, and only as a tie-check. At ~4.6% coverage, ~95% of"
echo "     eval samples are unwinnable and a 1-3 hit difference out of 1000 is"
echo "     decode noise. An HR gain counts only if the coverage rate moved with"
echo "     it -- otherwise it is not attributable to the retrieval mechanism."
echo ""
echo "Recovering retrieval rate WITHOUT moving coverage means the model retrieves"
echo "again but retrieves no better -- that is a fixed reward, not a working one."

exit $CORE_RC