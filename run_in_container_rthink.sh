#!/bin/bash

# Reasoning reward (current iteration: v5) — FORMAT-GATED, HR-TARGETED DENSE REWARD.
#   r_dense = (1 - log(rankId)/log(N))**p           # N = catalog size, p = RTHINK_DENSE_P (v5 default 2.0)
#   r_dense = 0                if output MALFORMED   # v5 anti-hack gate (!= 1 <answer> tag)
#   R_total = r_dense + clip(SCALE*R_think_raw, -CAP, +CAP) - format_penalty - length_penalty
#
# USE_RTHINK=1 routes amazon data to the versioned reasoning-reward dispatcher
# (verl/utils/reward_score/reward_reasoning/); RTHINK_MODE selects the iteration:
#   v5 (default) — format-gated dense reward (kills v4's multi-answer hack)
#   v4           — dense rank reward (peaks ~step250 then declines: reward-hacked)
#   v3           — evidence-grounded process reward
#   v2 / legacy  — alpha*info_gain - beta*redundancy + gamma*exploration_bonus
# See verl/utils/reward_score/reward_reasoning/v5/README.md for the rationale.
#
# Motivation for v5: v4's held-out curve climbs, peaks ~step 250, then DECLINES below
# its start. Cause (TEST_OUTPUT.md / REWARD_REASONING_ANALYSIS.md Part 3.3): the dense
# reward reads rankId only and never the float `match`, so the base multi-answer
# penalty is bypassed -> 73.9% of v4 outputs spam multiple <answer> tags and ramble
# (median 1203 words), inflating temp-1 train reward while greedy held-out drops.
# v5 closes both holes: (A) zero r_dense + penalty on malformed output, (B) capped
# length penalty, (C) sharper p=2 so the dense reward targets top-k (where HR lives)
# while staying nonzero everywhere.
#
# FULL v5 (this script's default): RTHINK_DENSE_ONLY=0 (process shaping ON). For a
# clean ablation of the gate alone, run dense-only:
#     RTHINK_DENSE_ONLY=1 \
#     EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5-denseonly \
#       sbatch sbatch_run_dual_gpu_rthink.sh
# Reproduce v4 exactly: RTHINK_FORMAT_GATE=0 RTHINK_DENSE_P=1.0 RTHINK_MODE=v4 sbatch ...
#
# Compare in WandB (use r_answer + the new format_ok, NOT score):
#   Baseline:   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline   (reward/mean@1)
#   v4:         nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v4
#   This run:   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5
#   Health check: format_ok should climb to ~1.0 (multi-answer hack dies).
#
# Resume behaviour (auto by default):
#   - Leave CHECKPOINT_PATH unset → resume_mode=auto picks the latest checkpoint
#     from trainer.default_local_dir automatically.
#   - Set CHECKPOINT_PATH=/path/to/global_step_NNN to resume from a specific step.

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
# Fresh v5 experiment name (new dir => starts from base model, not a v4 resume).
# Override with EXPERIMENT_NAME=... before sbatch to resume/rename.
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5}

export WAND_PROJECT='Search-R1-CF'
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
export TOKENIZERS_PARALLELISM=false

# --- Rthink reward switches ---
export USE_RTHINK=1
export RTHINK_MODE=${RTHINK_MODE:-v5}   # format-gated, HR-targeted dense reward

# v5 dense answer reward hyperparameters
export RTHINK_DENSE_P=${RTHINK_DENSE_P:-2.0}        # sharpness exponent (v5 default 2.0; targets top-k)
export RTHINK_RANK_FLOOR=${RTHINK_RANK_FLOOR:-0}    # zero r_dense when rankId > floor (0 = off)
export RTHINK_DENSE_ONLY=${RTHINK_DENSE_ONLY:-0}    # 0 = full v5; 1 = gate-only ablation (no shaping)

# v5 format-integrity gate (the anti-hack core)
export RTHINK_FORMAT_GATE=${RTHINK_FORMAT_GATE:-1}        # 1 = enable malformed-output gate
export RTHINK_FORMAT_PENALTY=${RTHINK_FORMAT_PENALTY:-0.5} # penalty for malformed output

# v5 length discipline
export RTHINK_LEN_SOFT=${RTHINK_LEN_SOFT:-600}      # word budget before penalty
export RTHINK_LEN_W=${RTHINK_LEN_W:-0.0005}         # penalty per word over budget
export RTHINK_LEN_CAP=${RTHINK_LEN_CAP:-0.2}        # max length penalty

# v3 process-shaping hyperparameters (only used when RTHINK_DENSE_ONLY=0)
export RTHINK_SCALE=${RTHINK_SCALE:-0.10}   # scale of R_think_raw before clipping
export RTHINK_CAP=${RTHINK_CAP:-0.08}       # absolute cap on applied shaping (< 0.1 tier gap)
export RTHINK_W_TOOL=${RTHINK_W_TOOL:-0.30}    # weight of tool_use
export RTHINK_W_GROUND=${RTHINK_W_GROUND:-0.40} # weight of grounding
export RTHINK_W_SYNTH=${RTHINK_W_SYNTH:-0.30}   # weight of synthesis
export RTHINK_W_REP=${RTHINK_W_REP:-0.50}       # weight of self_rep penalty

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

# Optional stability knobs (opt-in; default run is trainer-identical to v4 so the
# A/B is clean). Appended only when the env var is set.
if [[ -n "${ENTROPY_COEFF:-}" ]]; then
  extra_overrides+=(actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}")
fi
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

singularity exec --nv \
    --bind /work:/work \
    --bind /scratch:/scratch \
    --bind $PROJECT_DIR/overrides/patch_torch.py:/usr/local/lib/python3.12/dist-packages/sglang/srt/patch_torch.py \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env GLIBC_TUNABLES=$GLIBC_TUNABLES \
    --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
    --env VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND \
    --env USE_RTHINK=$USE_RTHINK \
    --env RTHINK_MODE=$RTHINK_MODE \
    --env RTHINK_DENSE_P=$RTHINK_DENSE_P \
    --env RTHINK_RANK_FLOOR=$RTHINK_RANK_FLOOR \
    --env RTHINK_DENSE_ONLY=$RTHINK_DENSE_ONLY \
    --env RTHINK_FORMAT_GATE=$RTHINK_FORMAT_GATE \
    --env RTHINK_FORMAT_PENALTY=$RTHINK_FORMAT_PENALTY \
    --env RTHINK_LEN_SOFT=$RTHINK_LEN_SOFT \
    --env RTHINK_LEN_W=$RTHINK_LEN_W \
    --env RTHINK_LEN_CAP=$RTHINK_LEN_CAP \
    --env RTHINK_SCALE=$RTHINK_SCALE \
    --env RTHINK_CAP=$RTHINK_CAP \
    --env RTHINK_W_TOOL=$RTHINK_W_TOOL \
    --env RTHINK_W_GROUND=$RTHINK_W_GROUND \
    --env RTHINK_W_SYNTH=$RTHINK_W_SYNTH \
    --env RTHINK_W_REP=$RTHINK_W_REP \
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
        actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
        trainer.logger=['wandb'] \
        trainer.val_only=false \
        trainer.val_before_train=${VAL_BEFORE_TRAIN:-true} \
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
