#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_greedy
#SBATCH -o output_test_greedy_sweep_%j.log

# GREEDY (temperature=0) held-out eval sweep, all arms in ONE job.
#
#   sbatch sbatch_run_test_greedy_sweep.sh
#   STEPS="200 300 400 500" sbatch sbatch_run_test_greedy_sweep.sh
#   ARMS="gpu-baseline-n8 gpu-rthink-v8-n8 gpu-rthink-v7b-n8" \
#     sbatch sbatch_run_test_greedy_sweep.sh
#
# Submit from a LOGIN node -- sbatch does not exist on compute nodes.
#
# WHY GREEDY, in one line: inference is served at temperature 0, and at
# temperature 1.0 the within-checkpoint decode spread (HR@5 0.006/0.000/0.001 on
# one frozen checkpoint) is wider than every arm gap we have measured. The
# trainer's own val pass is already greedy (rollout.val_kwargs.temperature=0), so
# this makes the held-out table and the wandb curves the same measurement.
# Full rationale in the entrypoint's header.
#
# WALL TIME. Default is 2 arms x 7 steps + 2 probe decodes = 16 decodes at
# ~7 min, so ~2 h plus first-time checkpoint merges. 12 h is headroom for a
# retriever restart cycle.
#
# DISK. ~3.8 GB per merged checkpoint under
# /scratch/.../merged_models/<arm>/global_step_<n>. Most of the default grid is
# already merged from the temperature-1.0 sweeps and gets reused. PURGE_MERGED=1
# deletes this sweep's merges on the way out.
#
# %j in the output name is REQUIRED: without it every submission overwrites the
# same log and a stale failure reads as current (see sbatch_run_test_rthink.sh:8).
#
# SAFE TO RUN WHILE TRAINING IS LIVE: checkpoints are frozen on disk and
# sbatch_run_test_rthink.sh writes a job-scoped copy of search_tool_config.yaml
# under /scratch/.../_tool_configs/, so it cannot repoint a running trainer's
# retriever. It does need its own 2 nodes on top of the training job's.
#
# Thin wrapper, same shape as sbatch_run_test_v8_n8_sweep.sh: it owns the SLURM
# directives and hands off to sbatch_run_test_rthink.sh, which pins node roles,
# starts the retriever with a job-scoped tool config, waits for readiness,
# restarts it on crash, and runs TEST_SCRIPT on the eval node.

export TEST_SCRIPT=test_in_container_greedy_sweep.sh

# Deliberately does NOT set ARMS / STEPS / PROBE_REPEATS: the entrypoint owns
# those defaults, and anything exported at submit time still wins.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
