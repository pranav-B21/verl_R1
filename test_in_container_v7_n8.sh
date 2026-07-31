#!/bin/bash
# set -euo pipefail

# Held-out evaluation for the M3 v7 arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8
#
# This is the treatment arm of the M3 A/B (retrieval-quality reward, r_retqual
# at tau=0.20, W_COVGAIN=0, GRPO n=8). Its control is
# test_in_container_baseline_n8.sh -- run BOTH at the same step and compare.
#
# WHY step 200 is the default
#   The comparison step is LOCKED at 200 for every arm. Later checkpoints are not
#   a fairer read: on the v5 run the held-out metric DECLINED after 300 while the
#   train reward kept climbing, so a late checkpoint flatters the train curve and
#   misreports generalization.
#
# WHY EVAL_REPEATS defaults to 3
#   Decode is unseeded at temperature=1.0. Three independent decodes of the SAME
#   v5 checkpoint straddled the baseline (HR@1 spread 0.000-0.005, wider than the
#   arm-to-arm gap). One decode is a sample, not a measurement. The spread across
#   repeats IS the significance bar -- if the arms' ranges overlap, they tie.
#
# WHY this script also reports retrieval behavior
#   During training this arm collapsed its retrieval rate from ~0.80 to ~0.02
#   tool-call turns per rollout: the +-0.06 retrieval shaping was outweighed
#   ~51x by the format+length penalty that retrieving indirectly triggers, so
#   GRPO learned to abstain. HR alone cannot show that. Step [5/5] below
#   measures whether the collapse survives at decode time, which is the finding
#   that actually decides what to do next -- see
#   verl/utils/reward_score/reward_retrieval/ROADMAP_v7.md.
#
# Usage:
#   bash test_in_container_v7_n8.sh                     # step 200, 3 decodes
#   CHECKPOINT_STEP=150 bash test_in_container_v7_n8.sh # a different step
#   EVAL_REPEATS=1 bash test_in_container_v7_n8.sh      # quick single decode
#
# Env vars: every variable accepted by test_in_container_rthink.sh works here
# (FORCE_MERGE, GEN_BATCH_SIZE, EVAL_CATEGORY, CUDA_VISIBLE_DEVICES, ...).
# This script only changes the defaults; it delegates the merge -> generate ->
# json -> eval.py pipeline to that script so both arms run byte-identical code.

set -u

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
cd "$PROJECT_DIR"

EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8"
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
echo "# M3 arm: v7 retrieval-quality reward"
echo "#   experiment : $EXPERIMENT_NAME"
echo "#   checkpoint : global_step_$CHECKPOINT_STEP   (comparison step LOCKED at 200)"
echo "#   decodes    : $EVAL_REPEATS  (unseeded, temperature=1.0)"
echo "#   control arm: test_in_container_baseline_n8.sh (run at the same step)"
echo "######################################################################"

# Delegate merge/generate/json/eval to the shared core, so both arms run
# byte-identical evaluation code. RTHINK_RUNS already wins over the core's
# EXPERIMENT_NAME back-compat branch; unsetting it is belt-and-braces so the
# core cannot fall through to its v6 default if RTHINK_RUNS is ever cleared.
unset EXPERIMENT_NAME
bash test_in_container_rthink.sh
CORE_RC=$?

# ---------------------------------------------------------------------------
# [5/5] Retrieval behavior -- the axis this arm was built to move.
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# [5/5] Decode-time retrieval behavior (v7 arm)"
echo "######################################################################"
EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8"
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
echo "Reference -- baseline-n8 @ global_step_200, 3 decodes (2026-07-21):"
echo "  HR@1 0.001 / 0.001 / 0.001    HR@5 0.002 / 0.003 / 0.004"
echo "  retrieval rate 99.3% / 99.2% / 98.8%"
echo ""
echo "Read the two together. HR parity with the baseline at a near-zero retrieval"
echo "rate is NOT a tie -- it means the shaping removed retrieval without paying"
echo "for it, and r_retqual is being scored on rollouts that never retrieve."

exit $CORE_RC
