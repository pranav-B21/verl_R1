#!/bin/bash
PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
set -eo pipefail
cd "$PROJECT_DIR"

# This script evaluates a trained checkpoint for Ablation B (interaction only) inside the Singularity container.
# Usage: bash test_in_container_ablationB.sh

cd /work/11138/pranavbelligundu/vista/verl_R1

# Load required modules (set +u to tolerate unbound vars in module/apptainer scripts)
set +u
module load tacc-apptainer
set -u

# GPU / data locations that will be passed to the container
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DATA_DIR=${DATA_DIR:-'./data/amazon_data'}
export TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-'./data/amazon_data/ablations/ablation_b_interaction_only'}
export TEST_DATA_DIR=${TEST_DATA_DIR:-'./data/amazon_data/ablations/ablation_b_interaction_only'}

export SSL_CERT_FILE=${SSL_CERT_FILE:-/work/11138/pranavbelligundu/vista/Software/cacert.pem}

# Training config defaults (can be overridden before calling the script)
export BASE_MODEL=${BASE_MODEL:-'Qwen/Qwen3-1.7B'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nq-search-r1-grpo-qwen3-1.7b-sbatch-ablationB}
export WAND_PROJECT=${WAND_PROJECT:-'Search-R1-CF'}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2048}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

# Evaluation specific overrides (customize as needed)
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-"global_step_350"} # Accepts "latest", a number, or "global_step_*"
FORCE_MERGE=${FORCE_MERGE:-0}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-16}
EVAL_CATEGORY=${EVAL_CATEGORY:-'../data/amazon_data/CDs_and_Vinyl'}

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "Checkpoint root not found: $CHECKPOINT_ROOT" >&2
    exit 1
fi

if [ "$CHECKPOINT_STEP" = "latest" ]; then
    tracker="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
    if [ ! -f "$tracker" ]; then
        echo "Unable to locate latest checkpoint tracker at $tracker" >&2
        exit 1
    fi
    step_id=$(tr -d '[:space:]' < "$tracker")
    CHECKPOINT_SUBDIR="global_step_${step_id}"
else
    if [[ "$CHECKPOINT_STEP" == global_step_* ]]; then
        CHECKPOINT_SUBDIR="$CHECKPOINT_STEP"
    else
        CHECKPOINT_SUBDIR="global_step_${CHECKPOINT_STEP}"
    fi
fi

CHECKPOINT_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_SUBDIR"
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT_PATH" >&2
    exit 1
fi

DEFAULT_MERGED_DIR="/scratch/11138/pranavbelligundu/verl/merged_models/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR"
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-$DEFAULT_MERGED_DIR}
DEFAULT_EVAL_DIR="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR"
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-$DEFAULT_EVAL_DIR}
GEN_OUTPUT_PARQUET=${GEN_OUTPUT_PARQUET:-$EVAL_OUTPUT_DIR/test_predictions.parquet}
PRED_JSON=${PRED_JSON:-$EVAL_OUTPUT_DIR/test_predictions.json}
METRICS_JSON=${METRICS_JSON:-$EVAL_OUTPUT_DIR/test_metrics.json}

mkdir -p "$MERGED_MODEL_DIR" "$EVAL_OUTPUT_DIR"

echo "Evaluating checkpoint: $CHECKPOINT_PATH"
echo "Merged HF model will be stored at: $MERGED_MODEL_DIR"
echo "Prediction parquet: $GEN_OUTPUT_PARQUET"
echo "Prediction json: $PRED_JSON"
echo "Metrics json: $METRICS_JSON"

singularity exec --nv --writable-tmpfs \
    --bind /work:/work \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env GLIBC_TUNABLES=$GLIBC_TUNABLES \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --env VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND \
    --env SSL_CERT_FILE=$SSL_CERT_FILE \
    --env FORCE_MERGE=$FORCE_MERGE \
    --env GEN_BATCH_SIZE=$GEN_BATCH_SIZE \
    --env CHECKPOINT_PATH=$CHECKPOINT_PATH \
    --env MERGED_MODEL_DIR=$MERGED_MODEL_DIR \
    --env GEN_OUTPUT_PARQUET=$GEN_OUTPUT_PARQUET \
    --env PRED_JSON=$PRED_JSON \
    --env METRICS_JSON=$METRICS_JSON \
    --env TOOL_CONFIG=$TOOL_CONFIG \
    --env TEST_PARQUET="$TEST_DATA_DIR/test.parquet" \
    --env EVAL_CATEGORY=$EVAL_CATEGORY \
    --env EXPERIMENT_NAME=$EXPERIMENT_NAME \
    --env PROJECT_DIR=$PROJECT_DIR \
    --pwd $PROJECT_DIR \
    sglang_25.10-py3-tls-fixed.sif \
    bash -c '
        set -euo pipefail
        cd "$PROJECT_DIR"

        # ---- FIX: Strip CUDA compat lib from LD_LIBRARY_PATH so host driver (CUDA 13.1) is used ----
        export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v "cuda/compat" | paste -sd ":" -)
        echo "[DEBUG] LD_LIBRARY_PATH after stripping cuda/compat: $LD_LIBRARY_PATH"
        echo "[DEBUG] GPU check:" && python3 -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}\")" || echo "[WARN] torch GPU check failed"
        # ---- END FIX ----

        echo "[1/4] Converting checkpoint to HuggingFace format..."
        if [ "$FORCE_MERGE" = "1" ] || [ ! -s "$MERGED_MODEL_DIR/config.json" ]; then
            python3 -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$CHECKPOINT_PATH/actor" \
                --target_dir "$MERGED_MODEL_DIR"
        else
            echo "Merged model already exists, skipping (set FORCE_MERGE=1 to rebuild)."
        fi

        echo "[2/4] Generating predictions on the test set..."
        python3 -m verl.trainer.main_generation \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=1 \
            data.path="$TEST_PARQUET" \
            data.prompt_key=prompt \
            data.batch_size="$GEN_BATCH_SIZE" \
            data.n_samples=1 \
            data.output_path="$GEN_OUTPUT_PARQUET" \
            model.path="$MERGED_MODEL_DIR" \
            rollout.name=sglang \
            rollout.temperature=1.0 \
            rollout.top_k=1 \
            rollout.top_p=0.95 \
            rollout.prompt_length=3072 \
            rollout.response_length=2048 \
            rollout.tensor_model_parallel_size=1 \
            +rollout.pipeline_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.8 \
            +rollout.multi_turn._target_=verl.workers.config.MultiTurnConfig \
            +rollout.multi_turn.enable=True \
            +rollout.multi_turn.max_assistant_turns=2 \
            +rollout.multi_turn.format=qwen \
            +rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
            +rollout.multi_turn.use_inference_chat_template=True

        echo "[3/4] Converting parquet outputs to eval.json format..."
        python3 - <<'"'"'PY'"'"'
import json
import os
import pyarrow.parquet as pq

parquet_path = os.environ["GEN_OUTPUT_PARQUET"]
json_path = os.environ["PRED_JSON"]

table = pq.read_table(parquet_path)
data = table.to_pydict()

prompts = data.get("prompt") or []
extra_info = data.get("extra_info") or []
responses = data.get("responses")
reward_model = data.get("reward_model")

if responses is None or reward_model is None:
    raise RuntimeError("Missing required columns (responses/reward_model) in parquet output.")

records = []
for idx in range(len(responses)):
    preds = responses[idx] or []
    if isinstance(preds, str):
        preds_list = [preds]
    else:
        preds_list = list(preds)

    # Prefer explicitly stored question, fall back to the last user message.
    question = None
    if idx < len(extra_info):
        info = extra_info[idx] or {}
        if isinstance(info, dict):
            question = info.get("question")
    if question is None and idx < len(prompts):
        prompt_turns = prompts[idx] or []
        if isinstance(prompt_turns, list) and prompt_turns:
            question = prompt_turns[-1].get("content", "")
    question = question or ""

    reward_info = reward_model[idx] or {}
    ground_truth = reward_info.get("ground_truth", {}) if isinstance(reward_info, dict) else {}
    target = ground_truth.get("target", "")

    records.append(
        {
            "input": question,
            "output": target,
            "predict": preds_list,
        }
    )

os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
PY

        echo "[4/4] Running eval.py for top-1 and top-5 metrics..."
        for topk in 1 5; do
            python3 eval.py \
                --input_dir "$PRED_JSON" \
                --model "$EXPERIMENT_NAME" \
                --output_dir "$METRICS_JSON" \
                --topk "$topk" \
                --category "$EVAL_CATEGORY"
        done

        echo "Evaluation complete. Metrics written to $METRICS_JSON"
    ' \
    2>&1 | tee "$EVAL_OUTPUT_DIR/test_eval.log"
