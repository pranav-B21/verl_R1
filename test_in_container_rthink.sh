#!/bin/bash
# set -euo pipefail

# Evaluates trained rthink checkpoints inside the Singularity container, using the
# same 4-step pipeline as test_in_container.sh: merge -> generate predictions ->
# to-json -> eval.py.
#
# By DEFAULT this evaluates the v5 run on the true held-out metric
# (eval.py HR@1/HR@5):
#   nq-...-rthink-v5 @ global_step_300
# A final summary table prints HR@1 / HR@5 / NDCG@5 / ORRatio@1, and (when
# available) the baseline / v3 numbers for context.
# To evaluate a different run/step, set RTHINK_RUNS explicitly (see Usage).
#
# NOTE on the reward: training used R_total = r_dense + shaping (v4) or
# R_answer + shaping (v3). The *evaluation* here does NOT use the training reward
# at all — eval.py ranks the generated <answer> against the item-embedding catalog
# for top-1/top-5, so this measures the same generalization metric (answer
# correctness) as every other test_in_container_*.sh variant and is directly
# comparable to the baseline.
#
# Usage:
#   bash test_in_container_rthink.sh                       # v5 @ step 300
#   RTHINK_RUNS="...-rthink-v5:300 ...-rthink-v5:400" bash test_in_container_rthink.sh
#   EXPERIMENT_NAME=...-rthink-v3 CHECKPOINT_STEP=300 bash test_in_container_rthink.sh  # single run
#
# Overrideable env vars (set before running):
#   RTHINK_RUNS       — space-separated list of "experiment[:step]" entries to
#                       evaluate in order. step accepts "latest", a number, or
#                       "global_step_*"; if omitted it falls back to CHECKPOINT_STEP.
#                       Default: rthink-v5 @ step 300.
#   EXPERIMENT_NAME   — if set, overrides RTHINK_RUNS with a single run (back-compat).
#   CHECKPOINT_STEP   — default step for entries without an explicit ":step"
#                       (default "latest"). For v3 use 300 (pre-collapse peak).
#   FORCE_MERGE       — set to 1 to re-merge even if the merged model exists
#   GEN_BATCH_SIZE    — generation batch size (default 32)
#   CUDA_VISIBLE_DEVICES

cd /work/11138/pranavbelligundu/vista/verl_R1

# Load required modules
module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

# GPU / data locations that will be passed to the container
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DATA_DIR=${DATA_DIR:-'./data/amazon_data'}
export TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-'./data/amazon_data'}
export TEST_DATA_DIR=${TEST_DATA_DIR:-'./data/amazon_data'}

export SSL_CERT_FILE=${SSL_CERT_FILE:-/work/11138/pranavbelligundu/vista/Software/cacert.pem}

# Training config defaults (can be overridden before calling the script)
export BASE_MODEL=${BASE_MODEL:-'Qwen/Qwen3-1.7B'}
export WAND_PROJECT=${WAND_PROJECT:-'Search-R1-CF'}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2048}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

# Evaluation specific overrides (customize as needed)
CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}  # default step for entries w/o ":step"
FORCE_MERGE=${FORCE_MERGE:-0}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}
EVAL_CATEGORY=${EVAL_CATEGORY:-'CDs_and_Vinyl'}
# Optional explicit dataset resource dir (contains id2name.json, name2id.json, embeddings.pt, ...).
# If unset, it is derived from EVAL_CATEGORY under $PROJECT_DIR/data/amazon_data/<category>.
EVAL_DATASET_DIR=${EVAL_DATASET_DIR:-''}

# Which runs to evaluate. A single explicit EXPERIMENT_NAME wins (back-compat);
# otherwise default to the two v4 runs at their latest checkpoint.
if [ -n "${EXPERIMENT_NAME:-}" ]; then
    RTHINK_RUNS=${RTHINK_RUNS:-"${EXPERIMENT_NAME}:${CHECKPOINT_STEP}"}
else
    RTHINK_RUNS=${RTHINK_RUNS:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5:300"}
fi

# Collected (experiment, subdir, eval_log) per run for the final summary.
SUMMARY_ENTRIES=()

# ---------------------------------------------------------------------------
# run_one EXPERIMENT_NAME CHECKPOINT_STEP
#   Resolves the checkpoint path and runs merge -> generate -> json -> eval.py
#   inside the container for one checkpoint.
# ---------------------------------------------------------------------------
run_one() {
    local EXPERIMENT_NAME="$1"
    local CHECKPOINT_STEP="$2"

    local CHECKPOINT_ROOT="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME"
    if [ ! -d "$CHECKPOINT_ROOT" ]; then
        echo "[skip] Checkpoint root not found: $CHECKPOINT_ROOT" >&2
        return 1
    fi

    local CHECKPOINT_SUBDIR
    if [ "$CHECKPOINT_STEP" = "latest" ]; then
        local tracker="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
        if [ ! -f "$tracker" ]; then
            echo "[skip] Unable to locate latest checkpoint tracker at $tracker" >&2
            return 1
        fi
        local step_id
        step_id=$(tr -d '[:space:]' < "$tracker")
        CHECKPOINT_SUBDIR="global_step_${step_id}"
    elif [[ "$CHECKPOINT_STEP" == global_step_* ]]; then
        CHECKPOINT_SUBDIR="$CHECKPOINT_STEP"
    else
        CHECKPOINT_SUBDIR="global_step_${CHECKPOINT_STEP}"
    fi

    local CHECKPOINT_PATH="$CHECKPOINT_ROOT/$CHECKPOINT_SUBDIR"
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "[skip] Checkpoint directory does not exist: $CHECKPOINT_PATH" >&2
        return 1
    fi

    local MERGED_MODEL_DIR="/scratch/11138/pranavbelligundu/verl/merged_models/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR"
    local EVAL_OUTPUT_DIR="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR"
    local GEN_OUTPUT_PARQUET="$EVAL_OUTPUT_DIR/test_predictions.parquet"
    local PRED_JSON="$EVAL_OUTPUT_DIR/test_predictions.json"
    local METRICS_JSON="$EVAL_OUTPUT_DIR/test_metrics.json"

    mkdir -p "$MERGED_MODEL_DIR" "$EVAL_OUTPUT_DIR"

    echo ""
    echo "=================================================================="
    echo ">>> Evaluating: $EXPERIMENT_NAME @ $CHECKPOINT_SUBDIR"
    echo "=================================================================="
    echo "Checkpoint:        $CHECKPOINT_PATH"
    echo "Merged HF model:   $MERGED_MODEL_DIR"
    echo "Prediction parquet:$GEN_OUTPUT_PARQUET"
    echo "Prediction json:   $PRED_JSON"
    echo "Metrics json:      $METRICS_JSON"

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
        --env TEST_PARQUET="$TEST_DATA_DIR/test.parquet" \
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

            echo "[2/4] Generating predictions on the test set..."
            python3 -m verl.trainer.main_generation \
                trainer.nnodes=1 \
                trainer.n_gpus_per_node=1 \
                data.path="$TEST_PARQUET" \
                data.prompt_key=prompt \
                +data.return_raw_chat=True \
                data.batch_size="$GEN_BATCH_SIZE" \
                data.n_samples=1 \
                data.output_path="$GEN_OUTPUT_PARQUET" \
                model.path="$MERGED_MODEL_DIR" \
                rollout.name=sglang \
                rollout.temperature=1.0 \
                rollout.top_k=-1 \
                rollout.top_p=0.95 \
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
            # eval.py expects:
            # - --category: a category name (e.g. "CDs_and_Vinyl")
            # - --dataset_dir: a directory containing id2name.json, name2id.json, embeddings.pt, ...
            #
            # NOTE: eval.py writes its JSON to --output_dir (a file path), so the topk=5
            # pass overwrites the topk=1 JSON. The authoritative record of BOTH passes is
            # this run-s test_eval.log (the printed "HR:" / "NDCG:" / "ORRatio:" lines),
            # which the outer summary parses. We therefore also keep a per-topk JSON.
            if [[ -n "${EVAL_DATASET_DIR:-}" ]]; then
                _eval_dataset_dir="$EVAL_DATASET_DIR"
                _eval_category_name="$(basename "$_eval_dataset_dir")"
            elif [[ "$EVAL_CATEGORY" == /* ]]; then
                _eval_dataset_dir="$EVAL_CATEGORY"
                _eval_category_name="$(basename "$EVAL_CATEGORY")"
            elif [[ "$EVAL_CATEGORY" == */* ]]; then
                _eval_category_name="$(basename "$EVAL_CATEGORY")"
                _candidate_dir="$PROJECT_DIR/$EVAL_CATEGORY"
                if [[ -d "$_candidate_dir" ]]; then
                    _eval_dataset_dir="$_candidate_dir"
                else
                    _eval_dataset_dir="$PROJECT_DIR/data/amazon_data/$_eval_category_name"
                fi
            else
                _eval_category_name="$EVAL_CATEGORY"
                _eval_dataset_dir="$PROJECT_DIR/data/amazon_data/$EVAL_CATEGORY"
            fi

            for topk in 1 5; do
                echo "--- eval.py topk=$topk ---"
                python3 eval.py \
                    --input_dir "$PRED_JSON" \
                    --model "$EXPERIMENT_NAME" \
                    --output_dir "${METRICS_JSON%.json}_top${topk}.json" \
                    --topk "$topk" \
                    --category "$_eval_category_name" \
                    --dataset_dir "$_eval_dataset_dir"
            done

            echo "Evaluation complete. Per-topk metrics under ${METRICS_JSON%.json}_top{1,5}.json"
        ' \
        2>&1 | tee "$EVAL_OUTPUT_DIR/test_eval.log"

    SUMMARY_ENTRIES+=("${EXPERIMENT_NAME}|${CHECKPOINT_SUBDIR}|${EVAL_OUTPUT_DIR}/test_eval.log")
}

# ---------------------------------------------------------------------------
# Run every requested checkpoint.
# ---------------------------------------------------------------------------
for entry in $RTHINK_RUNS; do
    exp="${entry%%:*}"
    step="${entry#*:}"
    [ "$step" = "$entry" ] && step="$CHECKPOINT_STEP"   # no ":step" given
    run_one "$exp" "$step" || echo "[warn] run failed/skipped: $entry" >&2
done

# ---------------------------------------------------------------------------
# Side-by-side summary parsed from each run's tee'd eval log.
# eval.py prints, per topk pass:  HR:[x]   NDCG:[y]   ORRatio:z
# (topk=1 pass first, then topk=5).
# ---------------------------------------------------------------------------
echo ""
echo "######################################################################"
echo "# Held-out eval summary (true HR/NDCG/ORRatio from eval.py)"
echo "######################################################################"
printf "%-46s %-14s %8s %8s %8s %10s\n" "experiment" "checkpoint" "HR@1" "HR@5" "NDCG@5" "ORRatio@1"
printf "%-46s %-14s %8s %8s %8s %10s\n" "----------" "----------" "----" "----" "------" "---------"
for s in "${SUMMARY_ENTRIES[@]}"; do
    IFS='|' read -r exp sub log <<< "$s"
    # HR/NDCG are printed once per topk pass, in order [topk=1, topk=5].
    hr1=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n '1p')
    hr5=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n '2p')
    ndcg5=$(grep -aoE 'NDCG:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/NDCG:\[\(.*\)\]/\1/' | sed -n '2p')
    orr1=$(grep -aoE 'ORRatio:[-0-9.e]+' "$log" 2>/dev/null | sed 's/ORRatio://' | sed -n '1p')
    printf "%-46s %-14s %8s %8s %8s %10s\n" \
        "${exp#nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-}" "$sub" \
        "${hr1:-n/a}" "${hr5:-n/a}" "${ndcg5:-n/a}" "${orr1:-n/a}"
done
echo ""
echo "Context (from REWARD_REASONING_ANALYSIS.md TEST_OUTPUT, prior eval):"
echo "  baseline  global_step_500   HR@1 0.004  HR@5 0.007  NDCG@5 0.0054  ORRatio@1 0.035"
echo "  rthink-v3 global_step_300   HR@1 0.002  HR@5 0.004  NDCG@5 0.0033  ORRatio@1 0.093"
echo ""
echo "Pass bar (Part 1): a v4 run must beat baseline HR@5 (0.007) to earn its place;"
echo "watch whether dense-only lifts HR or only ORRatio/diversity like v3 did."
