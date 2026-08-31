#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_v8_meta_sweep
#SBATCH -o output_test_v8_n8_metafix_sweep_%j.log

# Repaired-metadata v8: greedy held-out sweep at steps 200,250,...,600.
# Submit from a login node:
#   sbatch sbatch_run_test_v8_n8_metafix_sweep.sh
# Optional subset:
#   CHECKPOINT_STEPS="300 400 500 600" sbatch sbatch_run_test_v8_n8_metafix_sweep.sh
#
# The sweep reports META%/METAdoc%, retrieval health, selection behavior, and
# HR/NDCG for every checkpoint. Missing requested checkpoints are explicitly
# reported and skipped by the shared v8 sweep entrypoint.

export TEST_SCRIPT=test_in_container_v8_n8_metafix_sweep.sh

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
