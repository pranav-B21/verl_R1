#!/bin/bash

# ############################################################################
# ##  DEPRECATED — DO NOT USE FOR AN A/B BASELINE. Runs at rollout.n = 1.   ##
# ############################################################################
#
# This script never passes actor_rollout_ref.rollout.n, so it inherits the default
# from verl/trainer/config/rollout/rollout.yaml -> n: 1. At n=1 verl's GRPO takes the
# len==1 branch (core_algos.py:312) and sets mean=0/std=1, so the advantage IS the raw
# reward: REINFORCE with NO group-relative baseline. Every run produced by this script
# (and all of v2-v6) is therefore confounded and is NOT a valid comparator. The PI has
# confirmed n=1 was a bug in the original code and that the canonical value is 8 — the
# published Table-1 baseline was itself produced at n=8.
#
# It also lacks the hardened substrate the current runs use: no patch_torch bind-mount
# (pidfd_getfd crash on resume), no gpu_memory_utilization=0.65 / max_response_length=2048
# OOM fixes, and no RTHINK_* --env passthrough.
#
# ==> THE CORRECT OUTCOME-ONLY BASELINE, on an identical substrate to the v7 arm:
#
#       USE_RTHINK=0 EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 \
#         sbatch -o output_baseline_n8.log sbatch_run_dual_gpu_rthink.sh
#
#     run_in_container_rthink.sh honours USE_RTHINK=0 and routes to the same
#     outcome-only reward_SPRec reward, at n=8, with every substrate fix applied.
#
# This file is kept only to reproduce the historical n=1 runs. To run it anyway:
#       ALLOW_N1_BASELINE=1 bash run_in_container_baseline.sh
#
# ----------------------------------------------------------------------------
# OUTCOME-ONLY BASELINE (no reasoning reward).
# Reward = R_answer only, via reward_SPRec.compute_score.
# With USE_RTHINK=0 and USE_REWARD_B=0, reward_score/__init__.py routes amazon
# data to the old reward_SPRec (outcome-based embedding-rank reward).
#
# Resume behaviour (auto by default):
#   - Leave CHECKPOINT_PATH unset → resume_mode=auto. For a brand-new run this
#     starts fresh from BASE_MODEL; on a re-launch it picks the latest checkpoint
#     from trainer.default_local_dir automatically.
#   - Set CHECKPOINT_PATH=/path/to/global_step_NNN to resume from a specific step.

if [[ "${ALLOW_N1_BASELINE:-0}" != "1" ]]; then
  echo "REFUSING TO RUN: this script trains at rollout.n=1 (no GRPO group baseline)," >&2
  echo "which produces an INVALID comparator that looks legitimate in WandB." >&2
  echo "" >&2
  echo "For the real outcome-only baseline at n=8, on the identical substrate:" >&2
  echo "  USE_RTHINK=0 EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8 \\" >&2
  echo "    sbatch -o output_baseline_n8.log sbatch_run_dual_gpu_rthink.sh" >&2
  echo "" >&2
  echo "To reproduce the historical n=1 run anyway: ALLOW_N1_BASELINE=1 bash $0" >&2
  exit 1
fi

cd /work/11138/pranavbelligundu/vista/verl_R1

module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='./data/amazon_data'
export TRAIN_DATA_DIR='./data/amazon_data'
export TEST_DATA_DIR='./data/amazon_data'

export SSL_CERT_FILE=/work/11138/pranavbelligundu/vista/verl_R1/Software/cacert.pem

export BASE_MODEL='Qwen/Qwen3-1.7B'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline

export WAND_PROJECT='Search-R1-CF'
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
export TOKENIZERS_PARALLELISM=false

# --- Reward switches: reasoning reward OFF (outcome-only baseline) ---
export USE_RTHINK=0
export USE_REWARD_B=0

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

# Build resume overrides. If CHECKPOINT_PATH is set (e.g. via sbatch env), pin to
# that specific step; otherwise fall back to resume_mode=auto which scans
# default_local_dir for the latest global_step_* checkpoint automatically.
extra_overrides=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  extra_overrides+=(trainer.resume_mode=resume_path)
  extra_overrides+=(trainer.resume_from_path="${CHECKPOINT_PATH}")
else
  extra_overrides+=(trainer.resume_mode=auto)
fi

singularity exec --nv \
    --bind /work:/work \
    --bind /scratch:/scratch \
    --bind $PROJECT_DIR/overrides/patch_torch.py:/usr/local/lib/python3.12/dist-packages/sglang/srt/patch_torch.py \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env GLIBC_TUNABLES=$GLIBC_TUNABLES \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --env VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND \
    --env USE_RTHINK=$USE_RTHINK \
    --env USE_REWARD_B=$USE_REWARD_B \
    --pwd $PROJECT_DIR \
    sglang_25.10-py3-tls-fixed.sif \
    bash -c \
    'export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v "cuda/compat" | paste -sd ":" -) && exec python3 "$@"' \
    -- \
    -m verl.trainer.main_ppo \
        --config-path=$PROJECT_DIR/examples/sglang_multiturn/config \
        --config-name=search_multiturn_grpo \
        data.train_files=$TRAIN_DATA_DIR/train.parquet \
        data.val_files=$TEST_DATA_DIR/test.parquet \
        data.train_batch_size=56 \
        data.val_batch_size=40 \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.model.tokenizer_path=$BASE_MODEL \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size=40 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
        trainer.logger=['wandb'] \
        trainer.val_only=false \
        trainer.val_before_train=true \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.project_name=$WAND_PROJECT \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=22 \
        trainer.default_local_dir=/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml \
        "${extra_overrides[@]}" \
    2>&1 | tee $EXPERIMENT_NAME.log