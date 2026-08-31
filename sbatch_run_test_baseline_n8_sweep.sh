#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 24:00:00
#SBATCH -J test_bn8_sweep
#SBATCH -o output_test_baseline_n8_sweep_%j.log

# Held-out eval SWEEP of the M3 CONTROL arm (outcome-only reward_SPRec, USE_RTHINK=0):
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 @ steps 200,250,...,500
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_baseline_n8_sweep.sh
#   EVAL_REPEATS=1 sbatch sbatch_run_test_baseline_n8_sweep.sh                  # quick pass
#   CHECKPOINT_STEPS="300 400 500" sbatch sbatch_run_test_baseline_n8_sweep.sh  # subset
#
# WALL TIME. 7 checkpoints x 3 decodes = 21 decodes at ~20 min each, plus six
# first-time checkpoint merges, so budget ~8-9 h; 24 h is deliberate headroom
# because a decode that trips the retriever costs a restart cycle. EVAL_REPEATS=1
# is ~2.5 h but is NOT a measurement -- see the entrypoint's header for why.
#
# %j in the output name is REQUIRED: without it every submission overwrites the
# same log and a stale failure reads as current (that already happened once, see
# sbatch_run_test_rthink.sh:8).
#
# Thin wrapper, same shape as sbatch_run_test_baseline_n8.sh: it owns the SLURM
# directives and hands off to sbatch_run_test_rthink.sh, which pins node roles,
# starts the retriever with a job-scoped tool config, waits for readiness,
# restarts it on crash, and runs TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_baseline_n8_sweep.sh

# Deliberately does NOT set CHECKPOINT_STEPS or EVAL_REPEATS: the entrypoint owns
# those defaults. Setting them here would silently override it -- the same trap
# that made the launcher's old CHECKPOINT_STEP=latest default beat the step-200
# lock. Anything exported at submit time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
