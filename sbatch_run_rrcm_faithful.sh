#!/bin/bash

#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 48:00:00
#SBATCH -J rrcm_faithful
#SBATCH -o output_rrcm_faithful_%j.log

# Examples:
#   SEED=41 DECISION_ENTROPY_ENABLED=0 sbatch sbatch_run_rrcm_faithful.sh
#   SEED=41 DECISION_ENTROPY_ENABLED=1 sbatch sbatch_run_rrcm_faithful.sh

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
export RUN_SCRIPT=run_in_container_rrcm_faithful.sh
export TOOL_CONFIG_TEMPLATE="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config_rrcm_faithful.yaml"
export RETRIEVAL_SCRIPT=retrieval_launch.sh
export CORPUS_TAG=_v2

exec bash "$PROJECT_DIR/sbatch_run_dual_gpu_rthink.sh"
