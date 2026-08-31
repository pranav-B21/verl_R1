#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_v8_sweep
#SBATCH -o output_test_v8_n8_sweep_%j.log

# Held-out eval SWEEP of the v8 SELECTION arm (RTHINK_MODE=v8, USE_RTHINK=1):
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8 @ steps 200,250,...,600
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_v8_n8_sweep.sh
#   EVAL_REPEATS=1 sbatch sbatch_run_test_v8_n8_sweep.sh                  # quick pass
#   CHECKPOINT_STEPS="600 650 700" sbatch sbatch_run_test_v8_n8_sweep.sh  # past the control
#
# WALL TIME. 9 checkpoints x 3 decodes = 27 decodes. This arm decodes at ~7 min
# (job 885095 did 3 decodes plus one merge in 22 min), so budget ~3.5-4 h
# including nine first-time checkpoint merges. 12 h is deliberate headroom
# because a decode that trips the retriever costs a restart cycle.
#
# DISK. The merges leave ~3.8 GB per step under
# /scratch/.../merged_models/<arm>/global_step_<n>, i.e. ~34 GB for the default
# nine steps. Steps 200 and 250 are already merged from the earlier jobs and
# will be reused. Pass PURGE_MERGED=1 to delete them on the way out.
#
# %j in the output name is REQUIRED: without it every submission overwrites the
# same log and a stale failure reads as current (that already happened once, see
# sbatch_run_test_rthink.sh:8).
#
# WHY THIS IS A SEPARATE SCRIPT, not an edit to sbatch_run_test_v8_n8.sh
#   That script reproduces the pre-registered step-200 cross-arm comparison and
#   must keep doing so byte-for-byte. This is the within-arm trajectory question
#   over 200-600 and gets its own entrypoint and its own log. Same rationale as
#   sbatch_run_test_v8_n8_s250.sh.
#
# READ BEFORE INTERPRETING ANY NUMBER FROM THIS JOB
#   1. The wandb val-core panel is NOT the result. For v8 that curve contains
#      the shaping bonus it pays itself (r_think ~+0.049 of a +0.054 total at
#      step 700); the baseline's contains no shaping at all. On the comparable
#      outcome term the two arms are level: v8 r_answer +0.0074 @700 vs baseline
#      val-core +0.0107 @550. The entrypoint's header carries the full table.
#   2. data.val_files=test.parquet, so every val-aux curve in wandb is on the
#      TEST set. Picking a step because a val-aux curve rose is selecting a
#      checkpoint on test. Report the whole curve, not its argmax.
#   3. Gates are unchanged and evaluated per step (see test_in_container_v8_n8.sh):
#      Gate 0 validity -> Gate 1 mechanism -> Gate 2 outcome, in that order.
#      Gate 0 failing at a late step -- retrieval collapse deep in training is
#      exactly the v7 failure mode -- means DISCARD that step, not "compare anyway".
#   4. Only step 200 is the PRE-REGISTERED comparator. Steps 250-500 overlay the
#      baseline sweep at equal decode budget (3 vs 3) and are a fair but POST-HOC
#      comparison; label them that way. 550-600 have no control decode at all.
#
# SAFE TO RUN WHILE v8 IS STILL TRAINING: the checkpoints are frozen on disk and
# sbatch_run_test_rthink.sh writes a job-scoped copy of search_tool_config.yaml
# under /scratch/.../_tool_configs/, so it cannot repoint a running trainer's
# retriever. It does need its own 2 nodes on top of the training job's 2.
#
# Thin wrapper, same shape as sbatch_run_test_baseline_n8_sweep.sh: it owns the
# SLURM directives and hands off to sbatch_run_test_rthink.sh, which pins node
# roles, starts the retriever with a job-scoped tool config, waits for readiness,
# restarts it on crash, and runs TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_v8_n8_sweep.sh

# Deliberately does NOT set CHECKPOINT_STEPS or EVAL_REPEATS: the entrypoint owns
# those defaults. Setting them here would silently override it -- the same trap
# that made the launcher's old CHECKPOINT_STEP=latest default beat the step-200
# lock. Anything exported at submit time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
