#!/bin/bash

# Paper-protocol RRCM training on the immutable repaired corpus/index pair.
# The control and decision-entropy arm use this exact entrypoint; the only A/B
# variable is DECISION_ENTROPY_ENABLED.
set -o pipefail

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
DATA_DIR="$PROJECT_DIR/data/amazon_data"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

cd "$PROJECT_DIR" || exit 1

module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SSL_CERT_FILE="$PROJECT_DIR/Software/cacert.pem"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
export WAND_PROJECT="${WAND_PROJECT:-Search-R1-CF}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
export TOKENIZERS_PARALLELISM=false
export RRCM_REWARD_MODE=paper
export USE_RTHINK=0

SEED="${SEED:-41}"
DECISION_ENTROPY_ENABLED="${DECISION_ENTROPY_ENABLED:-0}"
DECISION_ENTROPY_COEFF="${DECISION_ENTROPY_COEFF:-0.001}"
DECISION_ENTROPY_HOLD_STEPS="${DECISION_ENTROPY_HOLD_STEPS:-300}"
DECISION_ENTROPY_DECAY_END_STEPS="${DECISION_ENTROPY_DECAY_END_STEPS:-500}"
TOTAL_STEPS="${TOTAL_STEPS:-1300}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH="${TRAIN_BATCH:-56}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-56}"
PPO_MICRO_BATCH=8

if [[ "$DECISION_ENTROPY_ENABLED" == "1" ]]; then
  entropy_flag=true
  default_suffix="decision-entropy"
else
  entropy_flag=false
  default_suffix="control"
fi
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rrcm-paper-frozen-${default_suffix}-seed${SEED}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME//[^A-Za-z0-9_-]/-}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rrcm-paper-frozen-${default_suffix}}"

python "$PROJECT_DIR/scripts/verify_rrcm_frozen_inputs.py" \
  --data-dir "$DATA_DIR" \
  --manifest "$DATA_DIR/rrcm_frozen_manifest.json" \
  --require-validation || exit 1

real_train_batch=$(( TRAIN_BATCH * ROLLOUT_N ))
if (( real_train_batch % PPO_MINI_BATCH != 0 )); then
  echo "FATAL: ${TRAIN_BATCH}*${ROLLOUT_N} is not divisible by ppo mini-batch ${PPO_MINI_BATCH}" >&2
  exit 1
fi
if (( PPO_MINI_BATCH % PPO_MICRO_BATCH != 0 || TRAIN_BATCH < PPO_MINI_BATCH )); then
  echo "FATAL: invalid PPO mini/micro/train batch geometry" >&2
  exit 1
fi
echo "[pre-flight] batch geometry OK: ${TRAIN_BATCH} x n=${ROLLOUT_N} = ${real_train_batch}"

TOOL_CONFIG="${TOOL_CONFIG:-$CONFIG_PATH/tool_config/search_tool_config_rrcm_faithful.yaml}"
grep -q '^ *topk: 1$' "$TOOL_CONFIG" || { echo "FATAL: faithful tool config must set topk: 1" >&2; exit 1; }
grep -q '^ *max_queries_per_call: 1$' "$TOOL_CONFIG" || {
  echo "FATAL: faithful tool config must enforce one query per call" >&2
  exit 1
}
echo "[tool-config] $TOOL_CONFIG"

# Keep reward catalog reads away from the shared /work filesystem hot path.
STAGE_DATA="${STAGE_DATA:-1}"
REC_DATA_ROOT="$PROJECT_DIR/data"
if [[ "$STAGE_DATA" == "1" ]]; then
  stage_root="${SCRATCH:-/scratch/11138/pranavbelligundu}/rec_data_stage"
  mkdir -p "$stage_root/amazon_data/CDs_and_Vinyl"
  staged_ok=1
  for asset in embeddings.pt name2id.json; do
    source_asset="$DATA_DIR/CDs_and_Vinyl/$asset"
    staged_asset="$stage_root/amazon_data/CDs_and_Vinyl/$asset"
    if [[ ! -f "$staged_asset" || "$source_asset" -nt "$staged_asset" ]]; then
      cp -f "$source_asset" "$staged_asset" || staged_ok=0
    fi
  done
  if [[ "$staged_ok" == "1" ]]; then
    REC_DATA_ROOT="$stage_root"
  fi
fi
export REC_DATA_ROOT
echo "[stage-data] reward catalogs: $REC_DATA_ROOT"

extra_overrides=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  extra_overrides+=(trainer.resume_mode=resume_path)
  extra_overrides+=(trainer.resume_from_path="$CHECKPOINT_PATH")
else
  extra_overrides+=(trainer.resume_mode=auto)
fi

singularity exec --nv \
  --bind /work:/work \
  --bind /scratch:/scratch \
  --bind "$PROJECT_DIR/overrides/patch_torch.py:/usr/local/lib/python3.12/dist-packages/sglang/srt/patch_torch.py" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env GLIBC_TUNABLES="$GLIBC_TUNABLES" \
  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  --env VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  --env RRCM_REWARD_MODE="$RRCM_REWARD_MODE" \
  --env USE_RTHINK=0 \
  --env REC_DATA_ROOT="$REC_DATA_ROOT" \
  --pwd "$PROJECT_DIR" \
  "$PROJECT_DIR/sglang_25.10-py3-tls-fixed.sif" \
  bash -c 'export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v "cuda/compat" | paste -sd ":" -) && exec python3 "$@"' \
  -- \
  -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=search_multiturn_grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val_rrcm_512_seed43.parquet" \
    +data.seed="$SEED" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size=40 \
    data.validation_shuffle=false \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.tokenizer_path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH" \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.actor.decision_entropy_enabled="$entropy_flag" \
    actor_rollout_ref.actor.decision_entropy_coeff="$DECISION_ENTROPY_COEFF" \
    actor_rollout_ref.actor.decision_entropy_hold_steps="$DECISION_ENTROPY_HOLD_STEPS" \
    actor_rollout_ref.actor.decision_entropy_decay_end_steps="$DECISION_ENTROPY_DECAY_END_STEPS" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.seed="$SEED" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    trainer.logger="['wandb']" \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=22 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME" \
    "${extra_overrides[@]}" \
  2>&1 | tee "$EXPERIMENT_NAME.log"
