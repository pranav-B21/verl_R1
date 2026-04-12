#!/bin/bash

#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 2                # two nodes, one GPU each
#SBATCH -n 2
#SBATCH -t 24:00:00
#SBATCH -o output_dual_gpu.log

# Vista nodes are single-GPU; this script uses two nodes: one for the retriever, one for training.

source ~/.bashrc

cleanup() {
  if [[ -f "${CONFIG_BACKUP:-}" && -n "${CONFIG_FILE:-}" ]]; then
    cp "${CONFIG_BACKUP}" "${CONFIG_FILE}" 2>/dev/null || true
  fi
  if [[ -n "${training_pid:-}" ]] && kill -0 "$training_pid" 2>/dev/null; then
    kill "$training_pid" 2>/dev/null || true
    wait "$training_pid" 2>/dev/null || true
  fi
  if [[ -n "${retrieval_pid:-}" ]] && kill -0 "$retrieval_pid" 2>/dev/null; then
    kill "$retrieval_pid" 2>/dev/null || true
    wait "$retrieval_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONFIG_FILE="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"
CONFIG_BACKUP=$(mktemp)
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

MAX_RETRIEVER_RESTARTS="${MAX_RETRIEVER_RESTARTS:-3}"
RETRIEVER_STARTUP_TIMEOUT_S="${RETRIEVER_STARTUP_TIMEOUT_S:-180}"

# Optional: set this to resume training from a specific checkpoint folder like ".../global_step_500".
# /scratch/11138/pranavbelligundu/verl/nq-search-r1-grpo-qwen3-1.7b-mt4/global_step_300
# Set to a valid checkpoint folder (e.g., ".../global_step_500") to resume when a certain experiment is actually in motion
CHECKPOINT_PATH=""

# Enumerate allocated nodes and pin roles
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
retriever_host="${nodes[0]}"
training_host="${nodes[1]:-${nodes[0]}}"

# Point tool config at the retriever host for this job
retrieval_url="http://${retriever_host}:8000/retrieve"
sed -i "s#^\\( *retrieval_service_url: \\).*#\\1${retrieval_url}#" "$CONFIG_FILE"

start_retriever() {
  echo "Starting retriever on ${retriever_host}..."
  srun --nodelist="${retriever_host}" --nodes=1 --ntasks=1 --exclusive bash -lc \
    "source ~/.bashrc && conda activate retriever && cd ${PROJECT_DIR} && bash retrieval_launch.sh" &
  retrieval_pid=$!
}

stop_retriever() {
  if [[ -n "${retrieval_pid:-}" ]] && kill -0 "$retrieval_pid" 2>/dev/null; then
    kill "$retrieval_pid" 2>/dev/null || true
    # Give srun a moment to terminate its remote step cleanly.
    for _ in {1..30}; do
      if ! kill -0 "$retrieval_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$retrieval_pid" 2>/dev/null; then
      kill -9 "$retrieval_pid" 2>/dev/null || true
    fi
    wait "$retrieval_pid" 2>/dev/null || true
  fi
  retrieval_pid=""
}

wait_for_retriever_ready() {
  local deadline=$((SECONDS + RETRIEVER_STARTUP_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if [[ -n "${retrieval_pid:-}" ]] && ! kill -0 "$retrieval_pid" 2>/dev/null; then
      wait "$retrieval_pid" 2>/dev/null || true
      return 1
    fi
    if (echo >"/dev/tcp/${retriever_host}/8000") >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

restart_retriever_or_fail() {
  local last_status="${1:-1}"
  for attempt in $(seq 1 "$MAX_RETRIEVER_RESTARTS"); do
    echo "Retriever restart attempt ${attempt}/${MAX_RETRIEVER_RESTARTS} (last exit=${last_status})..."
    stop_retriever
    start_retriever
    if wait_for_retriever_ready; then
      echo "Retriever restarted successfully."
      return 0
    fi
    if [[ -n "${retrieval_pid:-}" ]] && ! kill -0 "$retrieval_pid" 2>/dev/null; then
      wait "$retrieval_pid" 2>/dev/null
      last_status=$?
    else
      last_status=1
    fi
  done
  echo "Retriever failed to restart after ${MAX_RETRIEVER_RESTARTS} attempts." >&2
  return "${last_status:-1}"
}

start_retriever
if ! wait_for_retriever_ready; then
  wait "$retrieval_pid" 2>/dev/null
  status=$?
  echo "Retriever failed to start (exit code ${status})." >&2
  exit "${status:-1}"
fi

# Launch training on the second node/GPU; it will read the updated config
training_args=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  training_args+=(--checkpoint "${CHECKPOINT_PATH}")
fi
srun --nodelist="${training_host}" --nodes=1 --ntasks=1 --exclusive --chdir="${PROJECT_DIR}" bash run_in_container.sh "${training_args[@]}" &
training_pid=$!

# Monitor both jobs; restart retriever on crash; fail job if it can't be restarted.
while true; do
  if ! kill -0 "$training_pid" 2>/dev/null; then
    wait "$training_pid" 2>/dev/null
    training_status=$?
    stop_retriever
    exit "${training_status:-0}"
  fi

  if [[ -n "${retrieval_pid:-}" ]] && ! kill -0 "$retrieval_pid" 2>/dev/null; then
    wait "$retrieval_pid" 2>/dev/null
    retriever_status=$?
    echo "Retriever exited unexpectedly (exit code ${retriever_status}); attempting restart..." >&2
    if ! restart_retriever_or_fail "$retriever_status"; then
      final_status=$?
      echo "Retriever restart failed; terminating training." >&2
      if kill -0 "$training_pid" 2>/dev/null; then
        kill "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
      fi
      exit "${final_status:-1}"
    fi
  fi

  sleep 5
done
