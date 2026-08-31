#!/bin/bash
# Greedy held-out sweep for the repaired-metadata v8 selection-reward arm.
# The underlying v8 sweep owns checkpoint discovery, merge/generate/eval,
# decode_behavior.py (including META%), decode_selection.py, and the per-step
# accuracy and selection-funnel summaries.

set -u

export EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8-metafix"
export CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"200 250 300 350 400 450 500 550 600"}
export EVAL_REPEATS=${EVAL_REPEATS:-1}
export DECODE_TEMPERATURE=${DECODE_TEMPERATURE:-0}

echo "[metafix sweep] arm=v8 steps=$CHECKPOINT_STEPS repeats=$EVAL_REPEATS temperature=$DECODE_TEMPERATURE"

exec bash /work/11138/pranavbelligundu/vista/verl_R1/test_in_container_v8_n8_sweep.sh
