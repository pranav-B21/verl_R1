#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -o output_test_rthink_%j.log
# %j (job id) is REQUIRED here. Without it every submission overwrites the same
# output_test_rthink.log, so a stale failure is indistinguishable from a fresh run
# -- on 2026-07-23 a two-day-old log from failed job 854435 was read as current,
# even though the bug it showed had been fixed minutes later (854449 COMPLETED).
# The filename now carries the id you can look up with `sacct -j <id>`.

# Vista nodes are single-GPU; this script uses two nodes: one for the retriever, one for evaluation.
#
# Held-out eval for the amazon (CDs_and_Vinyl) runs. Derived from
# sbatch_run_test_good.sh, but NOTE: the "_good" suffix there means *goodreads* --
# it launches retrieval_launch_good.sh (data/goodreads_data). This script uses
# retrieval_launch.sh (data/amazon_data), matching what training served, so the
# retriever corpus and EVAL_CATEGORY agree.
#
# Usage (3 independent decodes to get a mean/spread):
#   EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 \
#   CHECKPOINT_STEP=200 EVAL_REPEATS=3 sbatch sbatch_run_test_rthink.sh
#
# Or point TEST_SCRIPT at a per-arm wrapper, which carries that arm's defaults
# (step, repeats) and adds the decode-time retrieval-behavior report:
#   TEST_SCRIPT=test_in_container_v7_n8.sh EVAL_REPEATS=1 sbatch sbatch_run_test_rthink.sh
#   TEST_SCRIPT=test_in_container_baseline_n8.sh EVAL_REPEATS=3 sbatch sbatch_run_test_rthink.sh
#
# TEST_SCRIPT is the in-container entrypoint run on the eval node; it must live in
# PROJECT_DIR and accept the eval knobs as environment variables.
TEST_SCRIPT="${TEST_SCRIPT:-test_in_container_rthink.sh}"

source ~/.bashrc

cleanup() {
  # The per-job tool config is disposable (see below) -- just remove it. Nothing is
  # restored, because this job never edits the shared config in the first place.
  if [[ -n "${CONFIG_FILE:-}" && "${CONFIG_FILE}" == *"/_tool_configs/"* ]]; then
    rm -f "${CONFIG_FILE}" 2>/dev/null || true
  fi
  if [[ -n "${test_pid:-}" ]] && kill -0 "$test_pid" 2>/dev/null; then
    kill "$test_pid" 2>/dev/null || true
    wait "$test_pid" 2>/dev/null || true
  fi
  if [[ -n "${retrieval_pid:-}" ]] && kill -0 "$retrieval_pid" 2>/dev/null; then
    kill "$retrieval_pid" 2>/dev/null || true
    wait "$retrieval_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"

# Fail before allocating a retriever if the entrypoint is missing -- otherwise the
# typo surfaces ~3 minutes in, after node pinning and retriever startup.
if [[ ! -f "$PROJECT_DIR/$TEST_SCRIPT" ]]; then
  echo "[fatal] TEST_SCRIPT not found: $PROJECT_DIR/$TEST_SCRIPT" >&2
  echo "        available:" >&2
  ls "$PROJECT_DIR"/test_in_container_*.sh >&2 2>/dev/null
  exit 1
fi
echo "[entrypoint] $TEST_SCRIPT"

SHARED_CONFIG="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"
# PER-JOB TOOL CONFIG -- do NOT sed the shared file. Every sbatch_run_*.sh used to
# rewrite retrieval_service_url in the one shared yaml and restore a backup on exit,
# so two concurrent jobs raced: the later sed repointed the earlier job's rollout at
# the wrong retriever, and the first job to exit restored a stale URL under the other.
# The window spans retriever startup + trainer init -- exactly when the rollout reads
# the path. Training (sbatch_run_dual_gpu_rthink.sh) was fixed this way on 2026-07-21;
# this is the eval side of the same fix, so an eval can run alongside a training job.
CONFIG_DIR="/scratch/11138/pranavbelligundu/verl/_tool_configs"
mkdir -p "$CONFIG_DIR"
CONFIG_FILE="$CONFIG_DIR/search_tool_config.${SLURM_JOB_ID:-$$}.yaml"
cp "$SHARED_CONFIG" "$CONFIG_FILE"
export TOOL_CONFIG="$CONFIG_FILE"   # consumed by the inner test_in_container_*.sh

MAX_RETRIEVER_RESTARTS="${MAX_RETRIEVER_RESTARTS:-3}"
RETRIEVER_STARTUP_TIMEOUT_S="${RETRIEVER_STARTUP_TIMEOUT_S:-180}"

# Enumerate allocated nodes and pin roles
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
retriever_host="${nodes[0]}"
test_host="${nodes[1]:-${nodes[0]}}"

# Point tool config at the retriever host for this job
retrieval_url="http://${retriever_host}:8000/retrieve"
sed -i "s#^\\( *retrieval_service_url: \\).*#\\1${retrieval_url}#" "$CONFIG_FILE"

start_retriever() {
  echo "Starting retriever on ${retriever_host}..."
  # The `cd` is REQUIRED: retrieval_launch.sh refers to both the corpus
  # (data/amazon_data) and the server (examples/.../retrieval_server.py) by
  # RELATIVE path, so invoking it by absolute path from elsewhere dies with
  # "can't open file '.../vista/examples/...'". This matches how the training
  # sbatch launches it (sbatch_run_dual_gpu_rthink.sh:77).
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

echo "Starting evaluation on ${test_host}..."
# Forward the eval knobs explicitly: the remote step runs a login shell that
# sources ~/.bashrc, so relying on inherited exports is fragile.
#
# Forward them EMPTY when unset rather than substituting a default here. A default
# applied at this layer is indistinguishable from a user-supplied value downstream,
# so it silently overrides the entrypoint's own default -- e.g. CHECKPOINT_STEP
# defaulting to "latest" here would beat test_in_container_baseline_n8.sh's 200 and
# evaluate step 300, breaking the locked comparison step. `${VAR:-default}` treats
# empty as unset, so every inner script still resolves its own default correctly.
srun --nodelist="${test_host}" --nodes=1 --ntasks=1 --exclusive bash -lc \
  "source ~/.bashrc && \
   EXPERIMENT_NAME='${EXPERIMENT_NAME:-}' \
   CHECKPOINT_STEP='${CHECKPOINT_STEP:-}' \
   RTHINK_RUNS='${RTHINK_RUNS:-}' \
   EVAL_REPEATS='${EVAL_REPEATS:-}' \
   FORCE_MERGE='${FORCE_MERGE:-}' \
   GEN_BATCH_SIZE='${GEN_BATCH_SIZE:-}' \
   TOOL_CONFIG='${TOOL_CONFIG}' \
   bash ${PROJECT_DIR}/${TEST_SCRIPT}" &
test_pid=$!

# Monitor both jobs; restart retriever on crash; fail job if it can't be restarted.
while true; do
  if ! kill -0 "$test_pid" 2>/dev/null; then
    wait "$test_pid" 2>/dev/null
    test_status=$?
    stop_retriever
    exit "${test_status:-0}"
  fi

  if [[ -n "${retrieval_pid:-}" ]] && ! kill -0 "$retrieval_pid" 2>/dev/null; then
    wait "$retrieval_pid" 2>/dev/null
    retriever_status=$?
    echo "Retriever exited unexpectedly (exit code ${retriever_status}); attempting restart..." >&2
    if ! restart_retriever_or_fail "$retriever_status"; then
      final_status=$?
      echo "Retriever restart failed; terminating evaluation." >&2
      if kill -0 "$test_pid" 2>/dev/null; then
        kill "$test_pid" 2>/dev/null || true
        wait "$test_pid" 2>/dev/null || true
      fi
      exit "${final_status:-1}"
    fi
  fi

  sleep 5
done

