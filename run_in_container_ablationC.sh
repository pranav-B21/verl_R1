#!/bin/bash

# This script runs the training inside the Singularity container
# Usage: bash run_in_container_ablationC.sh [--checkpoint /path/to/global_step_XXX]

CHECKPOINT_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint|--checkpoint-path)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Error: --checkpoint requires a non-empty path argument." >&2
        exit 2
      fi
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash run_in_container_ablationC.sh [--checkpoint /path/to/global_step_XXX]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd /work/11138/pranavbelligundu/vista/verl_R1

# Load required modules
module reset
module load nvidia/25.5 cuda/12.9 gcc/15
module load tacc-apptainer

# Set environment variables that will be passed to the container
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='./data/amazon_data'
export TRAIN_DATA_DIR='./data/amazon_data/ablations/ablation_c_no_reasoning'
export TEST_DATA_DIR='./data/amazon_data/ablations/ablation_c_no_reasoning'

export SSL_CERT_FILE=/work/11138/pranavbelligundu/vista/verl_R1/Software/cacert.pem


# export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
# export BASE_MODEL='meta-llama/Llama-3.2-1B-Instruct'
# export EXPERIMENT_NAME=amazon-search-r1-grpo-llama-3.2-1b-data-new-4
export BASE_MODEL='Qwen/Qwen3-1.7B'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-ablationC
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em


export WAND_PROJECT='Search-R1-CF'
export VLLM_ATTENTION_BACKEND=XFORMERS

# Fix for glibc TLS exhaustion error
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
export TOKENIZERS_PARALLELISM=false

# Additional fix: Reduce dynamic library loading
# export OMP_NUM_THREADS=1
# export MALLOC_TRIM_THRESHOLD_=0

: '
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env MALLOC_TRIM_THRESHOLD_=$MALLOC_TRIM_THRESHOLD_ \
    --env PYTHONUNBUFFERED=1 \
    --env NCCL_ASYNC_ERROR_HANDLING=0 \
    --env NCCL_IB_DISABLE=1 \
    --env NCCL_P2P_DISABLE=1 \
    --env PYTORCH_JIT=0 \
    --env TORCH_COMPILE_DISABLE=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
'

PROJECT_DIR="/work/11138/pranavbelligundu/vista/verl_R1"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"

extra_overrides=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  extra_overrides+=(trainer.resume_mode=resume_path)
  extra_overrides+=(trainer.resume_from_path="${CHECKPOINT_PATH}")
fi

# Run using singularity exec instead of shell to avoid nested environment issues
singularity exec --nv \
    --bind /work:/work \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env GLIBC_TUNABLES=$GLIBC_TUNABLES \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --env VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND \
    --pwd $PROJECT_DIR \
    sglang_25.10-py3-tls-fixed.sif \
    python3 -m verl.trainer.main_ppo \
        --config-path=$PROJECT_DIR/examples/sglang_multiturn/config \
        --config-name=search_multiturn_grpo \
        data.train_files=$TRAIN_DATA_DIR/train.parquet \
        data.val_files=$TEST_DATA_DIR/test.parquet \
        data.train_batch_size=56 \
        data.val_batch_size=40 \
        data.max_prompt_length=1024 \
        data.max_response_length=3072 \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=$BASE_MODEL \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
        trainer.total_epochs=35 \
        trainer.default_local_dir=/scratch/11138/pranavbelligundu/verl/$EXPERIMENT_NAME \
        actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml \
        "${extra_overrides[@]}" \
    2>&1 | tee $EXPERIMENT_NAME.log
