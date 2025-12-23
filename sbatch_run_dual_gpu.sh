#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each
#SBATCH -n 2
#SBATCH -t 36:00:00
#SBATCH -o output_dual_gpu.log

# Vista nodes are single-GPU; this script uses two nodes: one for the retriever, one for training.

source ~/.bashrc

cleanup() {
  if [[ -f "${CONFIG_BACKUP:-}" && -n "${CONFIG_FILE:-}" ]]; then
    cp "${CONFIG_BACKUP}" "${CONFIG_FILE}" 2>/dev/null || true
  fi
  if [[ -n "${retrieval_pid:-}" ]] && kill -0 "$retrieval_pid" 2>/dev/null; then
    kill "$retrieval_pid" 2>/dev/null || true
    wait "$retrieval_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PROJECT_DIR="/work/09585/shijunli4527/vista/Project/verl_R1"
CONFIG_FILE="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"
CONFIG_BACKUP=$(mktemp)
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

# Enumerate allocated nodes and pin roles
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
retriever_host="${nodes[0]}"
training_host="${nodes[1]:-${nodes[0]}}"

# Point tool config at the retriever host for this job
retrieval_url="http://${retriever_host}:8000/retrieve"
sed -i "s#^\\( *retrieval_service_url: \\).*#\\1${retrieval_url}#" "$CONFIG_FILE"

# Launch retriever on its own node/GPU
conda activate retriever
srun --nodelist="${retriever_host}" --nodes=1 --ntasks=1 --exclusive bash retrieval_launch.sh &
retrieval_pid=$!
conda deactivate

sleep 120  # wait for retriever to finish loading index/model

# Launch training on the second node/GPU; it will read the updated config
srun --nodelist="${training_host}" --nodes=1 --ntasks=1 --exclusive bash run_in_container.sh
