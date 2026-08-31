#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_bn8_meta_sweep
#SBATCH -o output_test_baseline_n8_metafix_sweep_%j.log

# Repaired-metadata baseline: greedy held-out sweep at steps 200,250,...,400.
# Submit from a login node:
#   sbatch sbatch_run_test_baseline_n8_metafix_sweep.sh
# Optional subset (the entrypoint still defaults to the preregistered grid):
#   CHECKPOINT_STEPS="250 300 350" sbatch sbatch_run_test_baseline_n8_metafix_sweep.sh
#
# Two nodes are intentional: node 0 serves the repaired corpus/index pair and
# node 1 performs checkpoint merge and evaluation. The common launcher creates
# a job-scoped tool config, so concurrent evaluation jobs cannot repoint one
# another's retrievers.

export TEST_SCRIPT=test_in_container_baseline_n8_metafix_sweep.sh

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
