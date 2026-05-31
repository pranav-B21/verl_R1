#!/bin/bash

# Full reasoning reward: R_total = R_answer + lambda * R_think
#   R_think = alpha * info_gain - beta * redundancy + gamma * exploration_bonus
# Routes amazon data to reward_SPRec_rthink.compute_score via USE_RTHINK=1.
#
# Compare in WandB:
#   Baseline:        nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline
#   RewardB-v2:      nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rewardB-v2
#   This run:        nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink

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
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-test

export WAND_PROJECT='Search-R1-CF'
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
export TOKENIZERS_PARALLELISM=false

# --- Rthink reward switches ---
export USE_RTHINK=1
export RTHINK_LAMBDA=0.3   # weight of R_think in R_total
export RTHINK_ALPHA=0.5    # info_gain weight
export RTHINK_BETA=0.5     # redundancy weight
export RTHINK_GAMMA=0.3    # exploration_bonus weight

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

singularity exec --nv \
    --bind /work:/work \
    --bind /scratch:/scratch \
    --bind $PROJECT_DIR/overrides/patch_torch.py:/usr/local/lib/python3.12/dist-packages/sglang/srt/patch_torch.py \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env GLIBC_TUNABLES=$GLIBC_TUNABLES \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --env VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND \
    --env USE_RTHINK=$USE_RTHINK \
    --env RTHINK_LAMBDA=$RTHINK_LAMBDA \
    --env RTHINK_ALPHA=$RTHINK_ALPHA \
    --env RTHINK_BETA=$RTHINK_BETA \
    --env RTHINK_GAMMA=$RTHINK_GAMMA \
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
    2>&1 | tee $EXPERIMENT_NAME.log
