#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_v8_n8
#SBATCH -o output_test_v8_n8_%j.log

# Held-out eval for the v8 SELECTION arm:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8 @ global_step_200
#
# Submit from a LOGIN node (sbatch does not exist on compute nodes):
#   sbatch sbatch_run_test_v8_n8.sh
#   EVAL_REPEATS=1 sbatch sbatch_run_test_v8_n8.sh     # quick smoke decode
#   CHECKPOINT_STEP=150 sbatch sbatch_run_test_v8_n8.sh
#
# Six decodes at ~9 min each land well inside the 12 h wall (job 875723 did 3
# in 26 min); the wall is sized for retriever restarts, not decode time.
#
# SAFE TO RUN WHILE v8 IS STILL TRAINING. global_step_200 is frozen on disk, and
# sbatch_run_test_rthink.sh writes a job-scoped copy of search_tool_config.yaml
# under /scratch/.../_tool_configs/, so it cannot repoint the running trainer's
# retriever (the race CLAUDE.md warns about applies to the OTHER launchers).
# It does need its own 2 nodes on top of the training job's 2.
#
# Comparison arms at the SAME step 200, already decoded -- do not re-run them:
#   sbatch_run_test_baseline_n8.sh  -- outcome-only control, 6 decodes
#   sbatch_run_test_v7b_n8.sh       -- retrieval restored, 3 decodes
#   sbatch_run_test_v7_n8.sh        -- retrieval collapsed to ~5%, 3 decodes
#
# The %j means each submission gets its OWN log; match the job id against
# `sacct` before reading one.
#
# Thin wrapper: it owns the SLURM directives, then hands off to
# sbatch_run_test_rthink.sh, which pins node roles, starts the retriever with a
# job-scoped tool config, waits for readiness, restarts it on crash, and runs
# TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_v8_n8.sh

# Deliberately does NOT set CHECKPOINT_STEP or EVAL_REPEATS: the inner script
# owns those defaults (step 200, 6 decodes -- 6 is the pre-registered power
# requirement, see verl/utils/reward_score/reward_retrieval/v8/README.md).
# Setting them here would silently override it. Anything exported at submit
# time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
