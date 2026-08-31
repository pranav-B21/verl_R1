#!/bin/bash

# Frozen-corpus, paper-protocol held-out evaluator. The underlying evaluator is
# shared with historical runs; this wrapper changes only protocol defaults.
set -o pipefail

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
DATA_DIR="$PROJECT_DIR/data/amazon_data"
TOOL_CONFIG="${TOOL_CONFIG:-$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config_rrcm_faithful.yaml}"

python "$PROJECT_DIR/scripts/verify_rrcm_frozen_inputs.py" \
  --data-dir "$DATA_DIR" \
  --manifest "$DATA_DIR/rrcm_frozen_manifest.json" \
  --require-validation || exit 1
grep -q '^ *topk: 1$' "$TOOL_CONFIG" || { echo "FATAL: evaluation must use topk: 1" >&2; exit 1; }
grep -q '^ *max_queries_per_call: 1$' "$TOOL_CONFIG" || {
  echo "FATAL: evaluation must enforce one query per tool call" >&2
  exit 1
}

export DATA_DIR
export TRAIN_DATA_DIR="$DATA_DIR"
export TEST_DATA_DIR="$DATA_DIR"
export TOOL_CONFIG
export MAX_ASSISTANT_TURNS=5
export DECODE_TEMPERATURE="${DECODE_TEMPERATURE:-0}"
export DECODE_TOP_P="${DECODE_TOP_P:-1}"
export DECODE_TOP_K="${DECODE_TOP_K:-1}"
export EVAL_REPEATS="${EVAL_REPEATS:-3}"

exec bash "$PROJECT_DIR/test_in_container_rthink.sh"
