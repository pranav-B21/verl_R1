#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_v7_n8
#SBATCH -o output_test_v7_n8_%j.log

# Held-out eval for the M3 TREATMENT arm (v7 retrieval-quality reward):
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8 @ global_step_200
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_v7_n8.sh
#   EVAL_REPEATS=1 sbatch sbatch_run_test_v7_n8.sh    # behavior-only, ~1 decode
#   CHECKPOINT_STEP=150 sbatch sbatch_run_test_v7_n8.sh
#
# The %j in -o means each submission gets its OWN log. The old shared
# output_test_rthink.log had no %j, so every run clobbered the previous one and a
# two-day-old failure was indistinguishable from a fresh run. Match the job id in
# the filename against `sacct` before reading a log.
#
# Control arm: sbatch_run_test_baseline_n8.sh -- run at the SAME step.
# Its step-200 decodes already exist from 2026-07-21 (job 854449), so it only
# needs re-running if you want same-day parity.
#
# This is a thin wrapper: it owns the SLURM directives and the arm's defaults,
# then hands off to sbatch_run_test_rthink.sh, which pins node roles, starts the
# retriever with a job-scoped tool config, waits for readiness, restarts it on
# crash, and runs TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_v7_n8.sh

# Deliberately does NOT set CHECKPOINT_STEP or EVAL_REPEATS: the inner script owns
# those defaults (step 200, 3 decodes). Setting them here would silently override
# it -- the same trap that made the launcher's old CHECKPOINT_STEP=latest default
# beat the step-200 lock. Anything exported at submit time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
