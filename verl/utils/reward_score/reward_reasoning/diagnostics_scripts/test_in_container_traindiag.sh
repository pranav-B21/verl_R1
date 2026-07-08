#!/bin/bash
# set -euo pipefail

# TRAIN-vs-TEST GREEDY DIAGNOSTIC
#
# Fills in the missing cell of the 2x2 (train/test x sampled/greedy): decodes a
# fixed 1000-prompt sample of TRAIN prompts (data/amazon_data/train_diag_1000.parquet,
# seed 42) and the full TEST set with GREEDY decoding (temperature=0, top_p=1),
# then scores both with the standard eval.py HR@1/HR@5 pipeline.
#
# Interpretation:
#   train-greedy HR >> test-greedy HR  -> generalization gap (RL levers: KL,
#                                         early stop, more data). Reward is fine.
#   train-greedy HR ~= 0 (same floor)  -> representation ceiling; the climbing
#                                         train reward was temp-1 sampling luck.
#                                         Reward iteration is provably wasted.
#
# Pipeline per (run, split): merge (skipped if cached) -> greedy generate ->
# to-json -> eval.py topk 1/5. Outputs land in a NEW subdir per split so prior
# test evals are never overwritten:
#   outputs/eval/<experiment>/<global_step_N>/greedy_<split>_<date>/
#
# Usage:
#   bash test_in_container_traindiag.sh          # v5@300 + v6@300, train+test
#   DIAG_RUNS="nq-...-rthink-v5:300" DIAG_SPLITS="train" bash test_in_container_traindiag.sh
#
# Env vars:
#   DIAG_RUNS        — space-separated "experiment:step" entries
#                      (default: rthink-v5:300 rthink-v6:300)
#   DIAG_SPLITS      — subset of "train test" (default both)
#   TRAIN_DIAG_PARQUET — train-sample parquet (default data/amazon_data/train_diag_1000.parquet)
#   FORCE_MERGE, GEN_BATCH_SIZE, CUDA_VISIBLE_DEVICES — as in test_in_container_rthink.sh

cd /work/11138/pranavbelligundu/vista/verl_R1

module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DATA_DIR=${DATA_DIR:-'./data/amazon_data'}
export SSL_CERT_FILE=${SSL_CERT_FILE:-/work/11138/pranavbelligundu/vista/Software/cacert.pem}
export BASE_MODEL=${BASE_MODEL:-'Qwen/Qwen3-1.7B'}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2048}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

FORCE_MERGE=${FORCE_MERGE:-0}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}
EVAL_CATEGORY=${EVAL_CATEGORY:-'CDs_and_Vinyl'}
EVAL_DATASET_DIR=${EVAL_DATASET_DIR:-"$PROJECT_DIR/data/amazon_data/CDs_and_Vinyl"}
TRAIN_DIAG_PARQUET=${TRAIN_DIAG_PARQUET:-"$PROJECT_DIR/data/amazon_data/train_diag_1000.parquet"}
TEST_PARQUET_PATH=${TEST_PARQUET_PATH:-"$PROJECT_DIR/data/amazon_data/test.parquet"}
DATE_TAG=$(date +%F)

DIAG_RUNS=${DIAG_RUNS:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5:300 nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6:300"}
DIAG_SPLITS=${DIAG_SPLITS:-"train test"}

SUMMARY_ENTRIES=()

run_one() {
    local EXPERIMENT_NAME="$1"
    local CHECKPOINT_STEP="$2"
    local SPLIT="$3"

    local CHECKPOINT_ROOT="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"
    local CHECKPOINT_SUBDIR="global_step_${CHECKPOINT_STEP}"
    local CHECKPOINT_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_SUBDIR"
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "[skip] Checkpoint directory does not exist: $CHECKPOINT_PATH" >&2
        return 1
    fi

    local INPUT_PARQUET
    if [ "$SPLIT" = "train" ]; then
        INPUT_PARQUET="$TRAIN_DIAG_PARQUET"
    else
        INPUT_PARQUET="$TEST_PARQUET_PATH"
    fi
    if [ ! -f "$INPUT_PARQUET" ]; then
        echo "[skip] Input parquet not found: $INPUT_PARQUET" >&2
        return 1
    fi

    local MERGED_MODEL_DIR="/scratch/11138/pranavbelligundu/verl/merged_models/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR"
    local EVAL_OUTPUT_DIR="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR/greedy_${SPLIT}_${DATE_TAG}"
    local GEN_OUTPUT_PARQUET="$EVAL_OUTPUT_DIR/predictions.parquet"
    local PRED_JSON="$EVAL_OUTPUT_DIR/predictions.json"
    local METRICS_JSON="$EVAL_OUTPUT_DIR/metrics.json"

    mkdir -p "$MERGED_MODEL_DIR" "$EVAL_OUTPUT_DIR"

    echo ""
    echo "=================================================================="
    echo ">>> Greedy diagnostic: $EXPERIMENT_NAME @ $CHECKPOINT_SUBDIR [$SPLIT]"
    echo "=================================================================="
    echo "Input parquet:     $INPUT_PARQUET"
    echo "Output dir:        $EVAL_OUTPUT_DIR"

    singularity exec --nv \
        --bind /work:/work \
        --bind /scratch:/scratch \
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
        --env INPUT_PARQUET="$INPUT_PARQUET" \
        --env EVAL_CATEGORY=$EVAL_CATEGORY \
        --env EVAL_DATASET_DIR=$EVAL_DATASET_DIR \
        --env EXPERIMENT_NAME=$EXPERIMENT_NAME \
        --env PROJECT_DIR=$PROJECT_DIR \
        --pwd $PROJECT_DIR \
        sglang_25.10-py3-tls-fixed.sif \
        bash -c '
            set -euo pipefail
            # Strip cuda/compat from LD_LIBRARY_PATH: container CUDA compat libs conflict with host driver (Error 803)
            export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v "cuda/compat" | paste -sd ":" -)
            cd "$PROJECT_DIR"

            echo "[1/4] Converting checkpoint to HuggingFace format..."
            if [ "$FORCE_MERGE" = "1" ] || [ ! -s "$MERGED_MODEL_DIR/config.json" ]; then
                python3 -m verl.model_merger merge \
                    --backend fsdp \
                    --local_dir "$CHECKPOINT_PATH/actor" \
                    --target_dir "$MERGED_MODEL_DIR"
            else
                echo "Merged model already exists, skipping (set FORCE_MERGE=1 to rebuild)."
            fi

            echo "[2/4] Generating GREEDY predictions (temperature=0)..."
            python3 -m verl.trainer.main_generation \
                trainer.nnodes=1 \
                trainer.n_gpus_per_node=1 \
                data.path="$INPUT_PARQUET" \
                data.prompt_key=prompt \
                +data.return_raw_chat=True \
                data.batch_size="$GEN_BATCH_SIZE" \
                data.n_samples=1 \
                data.output_path="$GEN_OUTPUT_PARQUET" \
                model.path="$MERGED_MODEL_DIR" \
                rollout.name=sglang \
                rollout.temperature=0.0 \
                rollout.top_k=-1 \
                rollout.top_p=1.0 \
                rollout.prompt_length=1024 \
                rollout.response_length=3072 \
                rollout.tensor_model_parallel_size=1 \
                +rollout.pipeline_model_parallel_size=1 \
                rollout.gpu_memory_utilization=0.9 \
                +rollout.multi_turn._target_=verl.workers.config.MultiTurnConfig \
                +rollout.multi_turn.enable=True \
                +rollout.multi_turn.max_assistant_turns=4 \
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
                echo "--- eval.py topk=$topk ---"
                python3 eval.py \
                    --input_dir "$PRED_JSON" \
                    --model "$EXPERIMENT_NAME" \
                    --output_dir "${METRICS_JSON%.json}_top${topk}.json" \
                    --topk "$topk" \
                    --category "$EVAL_CATEGORY" \
                    --dataset_dir "$EVAL_DATASET_DIR"
            done

            echo "Diagnostic pass complete."
        ' \
        2>&1 | tee "$EVAL_OUTPUT_DIR/eval.log"

    SUMMARY_ENTRIES+=("${EXPERIMENT_NAME}|${CHECKPOINT_SUBDIR}|${SPLIT}|${EVAL_OUTPUT_DIR}/eval.log")
}

for entry in $DIAG_RUNS; do
    exp="${entry%%:*}"
    step="${entry#*:}"
    for split in $DIAG_SPLITS; do
        run_one "$exp" "$step" "$split" || echo "[warn] pass failed/skipped: $entry [$split]" >&2
    done
done

echo ""
echo "######################################################################"
echo "# TRAIN-vs-TEST GREEDY DIAGNOSTIC — summary (greedy decode, temp=0)"
echo "######################################################################"
printf "%-30s %-14s %-6s %8s %8s %8s %10s\n" "experiment" "checkpoint" "split" "HR@1" "HR@5" "NDCG@5" "ORRatio@1"
printf "%-30s %-14s %-6s %8s %8s %8s %10s\n" "----------" "----------" "-----" "----" "----" "------" "---------"
for s in "${SUMMARY_ENTRIES[@]}"; do
    IFS='|' read -r exp sub split log <<< "$s"
    hr1=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n '1p')
    hr5=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n '2p')
    ndcg5=$(grep -aoE 'NDCG:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/NDCG:\[\(.*\)\]/\1/' | sed -n '2p')
    orr1=$(grep -aoE 'ORRatio:[-0-9.e]+' "$log" 2>/dev/null | sed 's/ORRatio://' | sed -n '1p')
    printf "%-30s %-14s %-6s %8s %8s %8s %10s\n" \
        "${exp#nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-}" "$sub" "$split" \
        "${hr1:-n/a}" "${hr5:-n/a}" "${ndcg5:-n/a}" "${orr1:-n/a}"
done
echo ""
echo "Reading the fork:"
echo "  train HR@5 >> test HR@5  -> generalization gap: fix with KL/early-stop/data, reward is fine"
echo "  train HR@5 ~= test ~= 0  -> representation ceiling: train reward was sampling luck;"
echo "                              stop reward iteration, fix decoding/embeddings/retrieval"
