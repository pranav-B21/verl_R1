#!/bin/bash
set -eo pipefail

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
DATA_DIR="$PROJECT_DIR/data/amazon_data"

module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

singularity exec \
  --bind /work:/work \
  --pwd "$PROJECT_DIR/data" \
  "$PROJECT_DIR/sglang_25.10-py3-tls-fixed.sif" \
  python3 build_rrcm_validation.py \
    --valid-json "$DATA_DIR/CDs_and_Vinyl_valid.json" \
    --train-json "$DATA_DIR/CDs_and_Vinyl_train_sampled_orig.json" \
    --test-json "$DATA_DIR/CDs_and_Vinyl_test.json" \
    --output "$DATA_DIR/val_rrcm_512_seed43.parquet" \
    "$@"
