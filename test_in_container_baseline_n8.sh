#!/bin/bash
# set -euo pipefail

# Held-out evaluation for the M3 control arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8
#
# Outcome-only reward (reward_SPRec.py, USE_RTHINK=0), GRPO n=8, same launcher
# and same substrate as the v7 arm. Its treatment partner is
# test_in_container_v7_n8.sh -- run BOTH at the same step and compare.
#
# WHY step 200 is the default
#   The comparison step is LOCKED at 200 for every arm, even though this arm has
#   checkpoints through 300. Using its later checkpoints against v7's step 200
#   would compare unequal amounts of training, and the v5 run already showed the
#   held-out metric DECLINING past 300 while train reward rose.
#
# WHY EVAL_REPEATS defaults to 3
#   Decode is unseeded at temperature=1.0. This arm's own three decodes of step
#   200 returned HR@5 0.002 / 0.003 / 0.004 -- a spread as large as any effect
#   we are trying to detect. The spread across repeats IS the significance bar.
#
# KNOWN ARM ASYMMETRY (disclose in any writeup)
#   This arm trained BEFORE the tool_call_repair JSON fix landed (2026-07-20
#   18:55); the v7 arm trained with it, so ~2% of v7 rollouts retrieve where
#   they would previously have retrieved nothing. Accepted deliberately rather
#   than spending a SLURM window on a re-baseline. The planned
#   RTHINK_RETRIEVAL_ONLY=1 ablation runs on the fixed substrate and is the
#   parity-matched control when it happens.
#
# Usage:
#   bash test_in_container_baseline_n8.sh                      # step 200, 3 decodes
#   CHECKPOINT_STEP=300 bash test_in_container_baseline_n8.sh  # this arm's last ckpt
#   EVAL_REPEATS=1 bash test_in_container_baseline_n8.sh       # quick single decode
#
# Env vars: every variable accepted by test_in_container_rthink.sh works here.
# This script only changes the defaults; it delegates the merge -> generate ->
# json -> eval.py pipeline to that script so both arms run byte-identical code.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8"
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
echo "# M3 arm: outcome-only baseline (control)"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   checkpoint : global_step_$CHECKPOINT_STEP   (comparison step LOCKED at 200)"
echo "#   decodes    : $EVAL_REPEATS  (unseeded, temperature=1.0)"
echo "#   treatment  : test_in_container_v7_n8.sh (run at the same step)"
echo "######################################################################"

# Delegate merge/generate/json/eval to the shared core, so both arms run
# byte-identical evaluation code. RTHINK_RUNS already wins over the core's
# EXPERIMENT_NAME back-compat branch; unsetting it is belt-and-braces so the
# core cannot fall through to its v6 default if RTHINK_RUNS is ever cleared.
unset EXPERIMENT_NAME
bash test_in_container_rthink.sh
CORE_RC=$?

# ---------------------------------------------------------------------------
# [5/5] Retrieval behavior -- same measurement as the v7 arm, so the two tables
# line up. For this arm it is the reference level, not the finding.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [5/5] Decode-time retrieval behavior (baseline arm)"
echo "######################################################################"
EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8"
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
echo "This arm is the retrieval reference level: it never received retrieval"
echo "shaping, so its rate (~99% at step 200) is what the v7 arm's rate should"
echo "be read against."

exit $CORE_RC
