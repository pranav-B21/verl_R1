#!/bin/bash

#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 2                # two nodes, one GPU each
#SBATCH -n 2
#SBATCH -t 48:00:00
#SBATCH -o output_dual_gpu_rthink.log

# Dual-node training: node[0] = retriever, node[1] = training.
# The training script is run_in_container_rthink.sh, currently the v7 retrieval-quality
# reward (r_retqual: cosine of the best retrieved doc to GT, per turn) at rollout.n=8.
# RUN_SCRIPT overrides the training script if needed.
#
#   node[0] → retriever (retrieval_launch.sh)
#   node[1] → training  (run_in_container_rthink.sh, USE_RTHINK=1, RTHINK_MODE=v7)
#
# MUST be submitted from a LOGIN node — sbatch is unavailable on compute nodes.
#
# THE A/B PAIR (both arms MUST be n=8; older -baseline (n=1) and -baseline-n5 runs are
# NOT valid comparators — n=1 disabled GRPO's group-relative baseline entirely):
#   M1 baseline: USE_RTHINK=0 EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 \
#                  sbatch -o output_baseline_n8.log sbatch_run_dual_gpu_rthink.sh
#   M3 v7 run:   sbatch -o output_rthink_v7_n8.log sbatch_run_dual_gpu_rthink.sh
# Using THIS script for both arms is deliberate: it keeps the substrate identical
# (patch_torch mount, gpu_mem 0.65, response 2048, n=8). Do NOT use
# run_in_container_baseline.sh — it omits rollout.n (inherits rollout.yaml n=1).
#
# Other invocations:
#   Wiring ablation:  RTHINK_RETRIEVAL_ONLY=1 EXPERIMENT_NAME=...-rthink-v7-shapingoff \
#                       sbatch sbatch_run_dual_gpu_rthink.sh    # must reproduce the baseline
#   Group-size ladder: ROLLOUT_N=12 (then 16) — the launcher pre-flights batch divisibility
#   Reproduce v6:     RTHINK_MODE=v6 EXPERIMENT_NAME=...-rthink-v6 sbatch ...
#
# Monitor: tail -f output_baseline_n8.log
#          tail -f nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8.log
#          grep -m1 "'n':" <log>            # confirm n=8 resolved
#          grep "\[pre-flight\]" <log>      # confirm batch geometry accepted
# WandB (project Search-R1-CF): compare r_answer + format_ok, NOT the shaped score.
#          Health: retrieval-rate must not drift down; best_sim distribution sane.

source ~/.bashrc

cleanup() {
  # The per-job tool config is disposable (see below) — just remove it. Nothing is
  # restored, because this job never edits the shared config in the first place.
  if [[ -n "${CONFIG_FILE:-}" && "${CONFIG_FILE}" == *"/_tool_configs/"* ]]; then
    rm -f "${CONFIG_FILE}" 2>/dev/null || true
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
SHARED_CONFIG="${TOOL_CONFIG_TEMPLATE:-$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml}"

# PER-JOB TOOL CONFIG — do NOT sed the shared file.
# retrieval_service_url has to be rewritten to this job's retriever host, but the
# shared config is a single file that every launcher here (sbatch_run_test_rthink.sh,
# sbatch_run_dual_gpu_good.sh, ...) also sed-edits and then restores on exit. Two
# concurrent jobs therefore race: the later sed points the earlier job's rollout at the
# wrong retriever, and the first job to exit restores the stale URL under the other.
# The window is real — it spans retriever startup (<=180s) plus trainer init, which is
# when the rollout actually reads this path. So this job copies the config to a
# job-scoped file on /scratch (shared FS, visible from the training node, outside the
# hydra config tree) and edits only that. The shared file is never modified.
CONFIG_DIR="/scratch/11138/pranavbelligundu/verl/_tool_configs"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/search_tool_config.${SLURM_JOB_ID:-$$}.yaml"
cp "$SHARED_CONFIG" "$CONFIG_FILE"
export TOOL_CONFIG="$CONFIG_FILE"   # consumed by run_in_container_rthink.sh

MAX_RETRIEVER_RESTARTS="${MAX_RETRIEVER_RESTARTS:-3}"
RETRIEVER_STARTUP_TIMEOUT_S="${RETRIEVER_STARTUP_TIMEOUT_S:-180}"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
retriever_host="${nodes[0]}"
training_host="${nodes[1]:-${nodes[0]}}"

retrieval_url="http://${retriever_host}:8000/retrieve"
sed -i "s#^\\( *retrieval_service_url: \\).*#\\1${retrieval_url}#" "$CONFIG_FILE"
echo "[tool-config] job-scoped copy: $CONFIG_FILE -> ${retrieval_url}"

start_retriever() {
  echo "Starting retriever on ${retriever_host}..."
  srun --nodelist="${retriever_host}" --nodes=1 --ntasks=1 --exclusive bash -lc \
    "source ~/.bashrc && conda activate retriever && cd ${PROJECT_DIR} && bash ${RETRIEVAL_SCRIPT:-retrieval_launch.sh}" &
  retrieval_pid=$!
}

stop_retriever() {
  if [[ -n "${retrieval_pid:-}" ]] && kill -0 "$retrieval_pid" 2>/dev/null; then
    kill "$retrieval_pid" 2>/dev/null || true
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

# Training script to run inside the container. run_in_container_rthink.sh is the
# current reasoning-reward iteration (v6); to A/B an earlier reward, override
# RTHINK_MODE on the same script (e.g. RTHINK_MODE=v5 ...).
RUN_SCRIPT="${RUN_SCRIPT:-run_in_container_rthink.sh}"
srun --nodelist="${training_host}" --nodes=1 --ntasks=1 --exclusive --chdir="${PROJECT_DIR}" \
  bash "${RUN_SCRIPT}" &
training_pid=$!

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
