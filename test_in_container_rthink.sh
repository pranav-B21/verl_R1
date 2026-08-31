#!/bin/bash
# set -euo pipefail

# Evaluates trained rthink checkpoints inside the Singularity container, using the
# same 4-step pipeline as test_in_container.sh: merge -> generate predictions ->
# to-json -> eval.py.
#
# By DEFAULT this evaluates the v6 run on the true held-out metric
# (eval.py HR@1/HR@5):
#   nq-...-rthink-v6 @ global_step_300
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
#   bash test_in_container_rthink.sh                       # v6 @ step 300
#   RTHINK_RUNS="...-rthink-v6:300 ...-rthink-v6:400" bash test_in_container_rthink.sh
#   EXPERIMENT_NAME=...-rthink-v3 CHECKPOINT_STEP=300 bash test_in_container_rthink.sh  # single run
#
# Overrideable env vars (set before running):
#   RTHINK_RUNS       — space-separated list of "experiment[:step]" entries to
#                       evaluate in order. step accepts "latest", a number, or
#                       "global_step_*"; if omitted it falls back to CHECKPOINT_STEP.
#                       Default: rthink-v6 @ step 300.
#   EXPERIMENT_NAME   — if set, overrides RTHINK_RUNS with a single run (back-compat).
#   CHECKPOINT_STEP   — default step for entries without an explicit ":step"
#                       (default "latest"). For v3 use 300 (pre-collapse peak).
#   FORCE_MERGE       — set to 1 to re-merge even if the merged model exists
#   GEN_BATCH_SIZE    — generation batch size (default 32)
#   EVAL_REPEATS      — independent decodes per checkpoint (default 1). Decode is
#                       unseeded at temperature=1.0, so one decode is a sample, not
#                       a measurement; use 3 to get a mean/spread. Each repeat lands
#                       in its own <step>/decode<i>_<date>/ dir (no clobbering) and
#                       gets its own row in the summary table.
#   DECODE_TEMPERATURE— 1.0 (default, historical) or 0 for a greedy/argmax decode.
#   DECODE_TOP_P      — default 0.95; forced to 1.0 when temperature=0.
#   DECODE_TOP_K      — default -1;   forced to 1   when temperature=0.
#   DECODE_TAG        — decode subdir prefix (default "decode", or "greedy" when
#                       temperature=0), so the two regimes never pool.
#   MAX_ASSISTANT_TURNS — rollout turn cap (historical default 4).
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
# Honor an externally supplied TOOL_CONFIG so a launcher can hand us a job-scoped
# copy instead of the shared file (see sbatch_run_test_rthink.sh). Concurrent jobs
# that all sed the shared config repoint each other's retriever mid-run; falling
# back to the shared path keeps standalone invocations working.
TOOL_CONFIG="${TOOL_CONFIG:-$CONFIG_PATH/tool_config/search_tool_config.yaml}"
echo "[tool-config] using $TOOL_CONFIG -> $(grep -m1 'retrieval_service_url' "$TOOL_CONFIG" 2>/dev/null)"

# Evaluation specific overrides (customize as needed)
CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}  # default step for entries w/o ":step"
FORCE_MERGE=${FORCE_MERGE:-0}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-32}
# Number of independent decodes per checkpoint. Decode is unseeded and sampled at
# temperature=1.0, so re-decoding the same checkpoint yields a different HR each
# time; a single decode is not a measurement. >1 nests each repeat in its own
# output subdir (see run_one's RUN_TAG) and the final summary prints one row per
# repeat, giving the mean/spread directly.
EVAL_REPEATS=${EVAL_REPEATS:-1}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-4}

# ---------------------------------------------------------------------------
# Decode sampling parameters.
#
# DEFAULTS ARE UNCHANGED (1.0 / 0.95 / -1) so every table already in
# TEST_OUTPUT.md keeps reproducing byte-for-byte. Set DECODE_TEMPERATURE=0 for a
# greedy (argmax) decode, which is what inference normally looks like once a
# policy is trained and is the only way to get the sampling-noise term out of a
# cross-arm comparison: at temperature 1.0 three decodes of ONE frozen v8
# checkpoint gave HR@5 0.005/0.004/0.002, wider than any arm gap measured so far.
#
# Greedy is also what the TRAINER already uses for its own validation pass
# (rollout.val_kwargs: temperature=0, do_sample=False, n=1 -- see
# verl/trainer/config/rollout/rollout.yaml), so a greedy decode here is the
# apples-to-apples partner of the wandb val curves. The temperature-1.0 harness
# was the outlier, not the val curves.
#
# temperature=0 forces top_p=1 / top_k=1: sglang's SamplingParams normalizes
# temperature 0 to greedy internally, but leaving top_p=0.95 in the command line
# makes the log read as if nucleus sampling were still active. Setting them
# explicitly keeps the recorded config honest.
#
# GREEDY IS NOT AUTOMATICALLY BIT-DETERMINISTIC on sglang: batching and the radix
# cache can change the reduction order, and this is a MULTI-TURN rollout, so one
# flipped token in a <search> query changes the retrieved docs and can change the
# answer. Run EVAL_REPEATS=2 once per arm to measure the residual spread instead
# of assuming it is zero; if it is zero, drop to EVAL_REPEATS=1 for the sweep.
DECODE_TEMPERATURE=${DECODE_TEMPERATURE:-1.0}
DECODE_TOP_P=${DECODE_TOP_P:-0.95}
DECODE_TOP_K=${DECODE_TOP_K:--1}
if [ "$(echo "$DECODE_TEMPERATURE" | awk '{print ($1 == 0) ? "greedy" : "sample"}')" = "greedy" ]; then
    DECODE_TEMPERATURE=0.0
    DECODE_TOP_P=1.0
    DECODE_TOP_K=1
    DECODE_MODE=greedy
else
    DECODE_MODE=sample
fi
# Decode subdir prefix. Greedy decodes must NOT land in decode<i>_<date> next to
# the temperature-1.0 decodes: decode_behavior.py / decode_selection.py /
# paired_arm_test.py glob those dirs and would silently pool two different
# sampling regimes into one "mean over decodes".
DECODE_TAG=${DECODE_TAG:-$([ "$DECODE_MODE" = greedy ] && echo greedy || echo decode)}

# Which prompts to decode. Defaults to the held-out test set -- every table in
# TEST_OUTPUT.md is that set and the default must never move. Overridable so the
# SAME instrument can decode the TRAIN sample, which is the only way to observe
# the distribution GRPO actually computes gradients on: the 1-in-64 training
# prints show 54.8% of TRAIN rollouts rank the GT #1 and 59.9% retrieve it,
# against 0.1% / 3.1% held-out, so every "no within-group gradient" statement
# derived from test decodes needs re-deriving here before it can be believed.
# ALWAYS pass a distinct DECODE_TAG with this, or train decodes land in
# decode<i>_<date> and get pooled with test decodes by the audit globs.
DECODE_PARQUET=${DECODE_PARQUET:-"$TEST_DATA_DIR/test.parquet"}
echo "[decode] mode=$DECODE_MODE temperature=$DECODE_TEMPERATURE top_p=$DECODE_TOP_P top_k=$DECODE_TOP_K tag=${DECODE_TAG}<i>_<date>"
echo "[decode] prompts=$DECODE_PARQUET"
if [ "$DECODE_PARQUET" != "$TEST_DATA_DIR/test.parquet" ] && \
   [ "$DECODE_TAG" = "decode" -o "$DECODE_TAG" = "greedy" ]; then
    echo "FATAL: non-default DECODE_PARQUET with the default DECODE_TAG='$DECODE_TAG'." >&2
    echo "       That would pool these decodes with the held-out tables. Set DECODE_TAG." >&2
    exit 1
fi

EVAL_CATEGORY=${EVAL_CATEGORY:-'CDs_and_Vinyl'}
# Optional explicit dataset resource dir (contains id2name.json, name2id.json, embeddings.pt, ...).
# If unset, it is derived from EVAL_CATEGORY under $PROJECT_DIR/data/amazon_data/<category>.
EVAL_DATASET_DIR=${EVAL_DATASET_DIR:-''}

# Which runs to evaluate. A single explicit EXPERIMENT_NAME wins (back-compat);
# otherwise default to the two v4 runs at their latest checkpoint.
if [ -n "${EXPERIMENT_NAME:-}" ]; then
    RTHINK_RUNS=${RTHINK_RUNS:-"${EXPERIMENT_NAME}:${CHECKPOINT_STEP}"}
else
    RTHINK_RUNS=${RTHINK_RUNS:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6:300"}
fi

# Collected (experiment, subdir, eval_log) per run for the final summary.
SUMMARY_ENTRIES=()

# ---------------------------------------------------------------------------
# run_one EXPERIMENT_NAME CHECKPOINT_STEP [RUN_TAG]
#   Resolves the checkpoint path and runs merge -> generate -> json -> eval.py
#   inside the container for one checkpoint.
#
#   RUN_TAG (optional) nests all outputs under an extra subdirectory. Decode is
#   stochastic (temperature=1.0, top_p=0.95) and unseeded, so repeated decodes of
#   the SAME checkpoint give different HR -- that spread is the measurement bar.
#   Without a tag every repeat writes the same test_predictions.json and clobbers
#   its predecessor, so EVAL_REPEATS>1 always passes one.
# ---------------------------------------------------------------------------
run_one() {
    local EXPERIMENT_NAME="$1"
    local CHECKPOINT_STEP="$2"
    local RUN_TAG="${3:-}"

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

    # The merged HF model is decode-invariant, so repeats share it (merged once).
    local MERGED_MODEL_DIR="/scratch/11138/pranavbelligundu/verl/merged_models/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR"
    local EVAL_OUTPUT_DIR="$PROJECT_DIR/outputs/eval/$EXPERIMENT_NAME/$CHECKPOINT_SUBDIR${RUN_TAG:+/$RUN_TAG}"
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
        --env DECODE_TEMPERATURE=$DECODE_TEMPERATURE \
        --env DECODE_TOP_P=$DECODE_TOP_P \
        --env DECODE_TOP_K=$DECODE_TOP_K \
        --env DECODE_MODE=$DECODE_MODE \
        --env MAX_ASSISTANT_TURNS=$MAX_ASSISTANT_TURNS \
        --env CHECKPOINT_PATH=$CHECKPOINT_PATH \
        --env MERGED_MODEL_DIR=$MERGED_MODEL_DIR \
        --env GEN_OUTPUT_PARQUET=$GEN_OUTPUT_PARQUET \
        --env PRED_JSON=$PRED_JSON \
        --env METRICS_JSON=$METRICS_JSON \
        --env TOOL_CONFIG=$TOOL_CONFIG \
        --env TEST_PARQUET="$DECODE_PARQUET" \
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

            echo "[2/4] Generating predictions on the test set ($DECODE_MODE: T=$DECODE_TEMPERATURE top_p=$DECODE_TOP_P top_k=$DECODE_TOP_K)..."
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
                rollout.temperature="$DECODE_TEMPERATURE" \
                rollout.top_k="$DECODE_TOP_K" \
                rollout.top_p="$DECODE_TOP_P" \
                rollout.prompt_length=1024 \
                rollout.response_length=3072 \
                rollout.tensor_model_parallel_size=1 \
                +rollout.pipeline_model_parallel_size=1 \
                rollout.gpu_memory_utilization=0.9 \
                +rollout.multi_turn._target_=verl.workers.config.MultiTurnConfig \
                +rollout.multi_turn.enable=True \
                +rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
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

    SUMMARY_ENTRIES+=("${EXPERIMENT_NAME}|${CHECKPOINT_SUBDIR}${RUN_TAG:+ [$RUN_TAG]}|${EVAL_OUTPUT_DIR}/test_eval.log")
}

# ---------------------------------------------------------------------------
# Run every requested checkpoint.
# ---------------------------------------------------------------------------
for entry in $RTHINK_RUNS; do
    exp="${entry%%:*}"
    step="${entry#*:}"
    [ "$step" = "$entry" ] && step="$CHECKPOINT_STEP"   # no ":step" given
    # A greedy decode is ALWAYS tagged, even at EVAL_REPEATS=1. The untagged path
    # writes test_metrics_top{1,5}.json at the step-dir root, which is where the
    # historical temperature-1.0 single decodes live -- an untagged greedy run
    # would overwrite them and leave no record of which regime produced the file.
    if [ "$EVAL_REPEATS" -le 1 ] && [ "$DECODE_MODE" = "sample" ]; then
        run_one "$exp" "$step" || echo "[warn] run failed/skipped: $entry" >&2
    else
        for i in $(seq 1 "$EVAL_REPEATS"); do
            echo ""
            echo ">>> $DECODE_MODE decode $i/$EVAL_REPEATS for $entry"
            run_one "$exp" "$step" "${DECODE_TAG}${i}_$(date +%F)" \
                || echo "[warn] run failed/skipped: $entry (repeat $i)" >&2
        done
    fi
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
printf "%-46s %-34s %8s %8s %8s %10s\n" "experiment" "checkpoint" "HR@1" "HR@5" "NDCG@5" "ORRatio@1"
printf "%-46s %-34s %8s %8s %8s %10s\n" "----------" "----------" "----" "----" "------" "---------"
for s in "${SUMMARY_ENTRIES[@]}"; do
    IFS='|' read -r exp sub log <<< "$s"
    # HR/NDCG are printed once per topk pass, in order [topk=1, topk=5].
    hr1=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n '1p')
    hr5=$(grep -aoE 'HR:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/HR:\[\(.*\)\]/\1/' | sed -n '2p')
    ndcg5=$(grep -aoE 'NDCG:\[[-0-9.e]+\]' "$log" 2>/dev/null | sed 's/NDCG:\[\(.*\)\]/\1/' | sed -n '2p')
    orr1=$(grep -aoE 'ORRatio:[-0-9.e]+' "$log" 2>/dev/null | sed 's/ORRatio://' | sed -n '1p')
    printf "%-46s %-34s %8s %8s %8s %10s\n" \
        "${exp#nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-}" "$sub" \
        "${hr1:-n/a}" "${hr5:-n/a}" "${ndcg5:-n/a}" "${orr1:-n/a}"
done
echo ""
echo "No control curve is baked into this evaluator; compare only with a matched audited arm."
