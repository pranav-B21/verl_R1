#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 04:00:00
#SBATCH -J test_bn8_v2
#SBATCH -o output_test_baseline_n8_v2_%j.log

# Held-out eval for the PARITY-MATCHED control arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-v2 @ global_step_200
#
# This is gpu-baseline-n8's twin, trained 2026-07-25 AFTER the tool_call_repair
# JSON fix (e2a492e8, 07-20 19:03) that gpu-baseline-n8's step-200 checkpoint
# predates. Same config, same reward (USE_RTHINK=0), never decoded. It answers
# one question: does that substrate asymmetry move the locked step-200 bar that
# v7/v7b/v8 are measured against? See the entrypoint header for the
# pre-registered expectation.
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_baseline_n8_v2.sh
#
# ONE COMMAND, not a menu -- any variant below is an ALTERNATIVE to that line,
# so run at most one of them. Two eval jobs on the same experiment and step write
# the same decode<i>_<date> directories and silently overwrite each other:
#   EVAL_REPEATS=1 sbatch sbatch_run_test_baseline_n8_v2.sh      # ~25 min, not a measurement
#   CHECKPOINT_STEP=150 sbatch sbatch_run_test_baseline_n8_v2.sh # earlier ckpt
#
# WALL TIME. 3 decodes at ~20 min plus one first-time checkpoint merge is ~1.5 h;
# 4 h covers a retriever restart cycle.
#
# %j in the output name is REQUIRED: without it every submission overwrites the
# same log and a stale failure reads as current (see sbatch_run_test_rthink.sh:8).
#
# Thin wrapper: it owns the SLURM directives and hands off to
# sbatch_run_test_rthink.sh, which pins node roles, starts the retriever with a
# job-scoped tool config, waits for readiness, restarts it on crash, and runs
# TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_baseline_n8_v2.sh

# Deliberately does NOT set CHECKPOINT_STEP or EVAL_REPEATS: the entrypoint owns
# those defaults (step 200, 3 decodes). Setting them here would silently override
# it -- the same trap that made the launcher's old CHECKPOINT_STEP=latest default
# beat the step-200 lock. Anything exported at submit time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
