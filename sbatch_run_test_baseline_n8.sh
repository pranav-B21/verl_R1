#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_baseline_n8
#SBATCH -o output_test_baseline_n8_%j.log

# Held-out eval for the M3 CONTROL arm (outcome-only reward_SPRec, USE_RTHINK=0):
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 @ global_step_200
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_baseline_n8.sh
#   CHECKPOINT_STEP=300 sbatch sbatch_run_test_baseline_n8.sh   # this arm's last ckpt
#
# NOTE -- this arm's step-200 decodes ALREADY EXIST (job 854449, 2026-07-21):
#     HR@1 0.001 / 0.001 / 0.001    HR@5 0.002 / 0.003 / 0.004
#     retrieval rate 99.3% / 99.2% / 98.8%    docs/ret ~7.3
# under outputs/eval/<experiment>/global_step_200/decode{1,2,3}_2026-07-21/.
# Re-running costs ~1h and reproduces numbers you already have. Do it only if you
# want same-day parity with a v7 decode, or you changed the eval path. Decode dirs
# are date-tagged, so a rerun today lands beside the old ones without clobbering.
#
# Default step is 200 even though checkpoints exist through 300: the comparison
# step is LOCKED at 200 for every arm. Using 300 here against v7's 200 would
# compare unequal training, and held-out performance was already observed to
# DECLINE past 300 on the v5 run while train reward kept rising.
#
# This is a thin wrapper: it owns the SLURM directives and the arm's defaults,
# then hands off to sbatch_run_test_rthink.sh, which pins node roles, starts the
# retriever with a job-scoped tool config, waits for readiness, restarts it on
# crash, and runs TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_baseline_n8.sh

# Deliberately does NOT set CHECKPOINT_STEP or EVAL_REPEATS: the inner script owns
# those defaults (step 200, 3 decodes). Setting them here would silently override
# it -- the same trap that made the launcher's old CHECKPOINT_STEP=latest default
# beat the step-200 lock. Anything exported at submit time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
