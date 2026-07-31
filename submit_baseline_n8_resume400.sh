#!/bin/bash
# Resume the M1 baseline arm (outcome-only, USE_RTHINK=0) from global_step_300 → 400+.
#
# Run this from a LOGIN node (sbatch does not exist on compute nodes):
#   bash submit_baseline_n8_resume400.sh
#
# Why these settings:
#   USE_RTHINK=0        - baseline arm: reward dispatcher routes to plain reward_SPRec (InTop@n).
#   EXPERIMENT_NAME     - MUST match the original run; the trainer tracks ABSOLUTE steps and
#                         resolves default_local_dir from this name, so a rename restarts at 0.
#   CHECKPOINT_PATH     - pinned to step 300 explicitly. resume_mode=auto would also pick 300,
#                         but pinning documents intent and is immune to a stray newer dir.
#   VAL_BEFORE_TRAIN    - false: step 300 was already validated; skips a redundant eval pass.
#   total_epochs=22     - LEFT ALONE inside run_in_container_rthink.sh. Do NOT cap
#                         trainer.total_training_steps at 400 to make the job stop early:
#                         lr_warmup_steps_ratio=0.285 is applied to total_training_steps
#                         (0.285*1606 = 458), so capping it reshapes the LR schedule and the
#                         resumed steps stop being comparable to steps 1-300.
#
# save_freq=50, so checkpoints land at global_step_350 and global_step_400. The job runs to
# the 48h wall clock; ~100 steps took roughly a day on the last n=8 run, so 400 should land
# comfortably. Once global_step_400 exists you can scancel the job rather than burn the rest:
#   ls /scratch/11138/pranavbelligundu/verl/$EXP/global_step_400 && scancel <jobid>

set -euo pipefail

EXP=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8
CKPT_ROOT=/scratch/11138/pranavbelligundu/verl/$EXP
RESUME_STEP=${RESUME_STEP:-300}

cd /work/11138/pranavbelligundu/vista/verl_R1

if [[ ! -d "$CKPT_ROOT/global_step_${RESUME_STEP}/actor" ]]; then
  echo "ERROR: no checkpoint at $CKPT_ROOT/global_step_${RESUME_STEP}/actor" >&2
  exit 1
fi

echo "[resume] experiment : $EXP"
echo "[resume] from step  : $RESUME_STEP (tracker says $(cat "$CKPT_ROOT/latest_checkpointed_iteration.txt"))"

USE_RTHINK=0 \
EXPERIMENT_NAME=$EXP \
CHECKPOINT_PATH=$CKPT_ROOT/global_step_${RESUME_STEP} \
VAL_BEFORE_TRAIN=false \
  sbatch -o output_baseline_n8_resume400.log sbatch_run_dual_gpu_rthink.sh
