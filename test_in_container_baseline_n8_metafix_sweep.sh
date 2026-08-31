#!/bin/bash
# Greedy held-out sweep for the repaired-metadata outcome-only arm.
# The step grid and decode regime are fixed here so the job directly implements
# PREREG_metafix_2026-08-24.md. The underlying baseline sweep owns checkpoint
# discovery, merge/generate/eval, decode_behavior.py (including META%), and the
# per-step summary table.

set -u

export EXPERIMENT_NAME="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix"
export CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"200 250 300 350 400"}
export EVAL_REPEATS=${EVAL_REPEATS:-1}
export DECODE_TEMPERATURE=${DECODE_TEMPERATURE:-0}

echo "[metafix sweep] arm=baseline steps=$CHECKPOINT_STEPS repeats=$EVAL_REPEATS temperature=$DECODE_TEMPERATURE"

exec bash /work/11138/pranavbelligundu/vista/verl_R1/test_in_container_baseline_n8_sweep.sh
