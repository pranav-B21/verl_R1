#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J rrcm_eval
#SBATCH -o output_test_rrcm_faithful_%j.log

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
export TEST_SCRIPT=test_in_container_rrcm_faithful.sh
export TOOL_CONFIG_TEMPLATE="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config_rrcm_faithful.yaml"
export RETRIEVAL_SCRIPT=retrieval_launch.sh

exec bash "$PROJECT_DIR/sbatch_run_test_rthink.sh"
