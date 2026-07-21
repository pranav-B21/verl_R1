#!/bin/bash

# Retrieval-quality reward (current iteration: v7) — r_retqual SHAPING ON v6's OUTCOME REWARD.
#   r_outcome = 1/log2(rankId+1)                     if rankId <= K   # mirrors eval.py NDCG@K
#   r_outcome = TAIL_W*(1-log(rankId)/log(N))**TAIL_P  otherwise      # weak, steep "get warmer"
#   r_outcome = 0                if output MALFORMED  # v5 anti-hack gate (kept; != 1 <answer> tag)
#   R_think_raw = W_RETQUAL*retqual_agg + W_COVGAIN*covgain_agg       # v7: retrieval, not reasoning
#   R_total   = r_outcome + clip(SCALE*R_think_raw, -CAP, +CAP) - format_penalty - length_penalty
#
# USE_RTHINK=1 routes amazon data to the versioned reward dispatcher
# (verl/utils/reward_score/reward_reasoning/); RTHINK_MODE selects the iteration:
#   v7 (default here) — retrieval-quality shaping (lives in reward_retrieval/v7/)
#   v6                — HR-faithful top-K reward (NDCG@K credit; mid-rank mush removed)
#   v5                — format-gated dense reward (kills v4's multi-answer hack)
#   v4                — dense rank reward (peaks ~step250 then declines: reward-hacked)
#   v3                — evidence-grounded process reward
#   v2 / legacy       — alpha*info_gain - beta*redundancy + gamma*exploration_bonus
# See verl/utils/reward_score/reward_retrieval/retrieval_reward_design.md for the rationale
# and reward_retrieval/ROADMAP_v7.md for the strategy this run sits inside.
#
# Motivation for v7: v2-v6 all trained at rollout.n=1, where verl's GRPO sets mean=0/std=1
# (core_algos.py:312) => advantage IS the raw reward: REINFORCE with NO group baseline. So
# "shaping never beat baseline" across v2-v6 is CONFOUNDED, not a negative result. n=8 is
# PI-canonical and turns the group-relative mechanism on for the first time. v7 also changes
# WHAT is shaped: v6 scored reasoning process, v7 scores RETRIEVAL quality — cosine of the
# best retrieved doc to GT, per turn. This is the only term to have passed the correlation
# audit (pearson +0.112, AUC 0.85 vs InTop@10) before ever training.
#
# FIRST v7 RUN (M3): r_retqual ONLY. RTHINK_W_COVGAIN=0 — covgain is DEFERRED (it scored
# AUC 0.499 = chance, because the multi-turn behavior it rewards barely exists in the logs;
# resolve that offline in M2 first). One variable per run.
#   Outcome-only ablation: RTHINK_RETRIEVAL_ONLY=1  (NOTE: 1 DISABLES shaping — the name
#                          reads backwards; it is v6's DENSE_ONLY renamed. Must reproduce
#                          the baseline curve exactly, else the wiring adds something spurious.)
#   Reproduce v6:          RTHINK_MODE=v6 RUN_SCRIPT=run_in_container_rthink.sh sbatch ...
#   Group-size ladder:     ROLLOUT_N=12 (then 16) if n=8 under-performs. Keep the batch
#                          divisibility invariant below intact when you do.
#
# BATCH DIVISIBILITY (unasserted — read before changing n or the batch sizes):
#   train_batch_size * rollout.n MUST be divisible by ppo_mini_batch_size. NOTHING in verl
#   checks this: dp_actor.py:388 splits with a non-strict chunker, so a bad combo silently
#   trains a ragged final mini-batch with a mis-scaled loss and no error. 56*8=448, 448/56=8.
#   Also required: ppo_mini_batch % ppo_micro_batch_per_gpu == 0 (fsdp_workers.py:257 asserts)
#   and train_batch >= ppo_mini_batch (actor.py:153 raises). 56/56/8 satisfies all three.
#
# Compare in WandB (use r_answer + format_ok, NOT the shaped score — score isn't comparable):
#   Baseline:   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8  (USE_RTHINK=0, THIS script)
#   This run:   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8
#   NOTE the older -baseline and -baseline-n5 runs are NOT valid comparators (n=1 and n=5).
#   North-star: val r_answer/mean@1 must clear baseline AVERAGED OVER >=3 decode seeds
#   (one eval swings HR@1 0.000<->0.005 on a frozen ckpt, so a single number proves nothing).
#   Health checks: format_ok ~1.0; retrieval-rate must NOT drift down (collapse guard);
#   best_sim distribution sane (tau=0.30 is an uncalibrated guess — check it against reality).
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
# Fresh v7/n8 experiment name (new dir => starts from base model, not a v6 resume).
# Override with EXPERIMENT_NAME=... before sbatch to resume/rename.
# The -n8 suffix is load-bearing: n=5 and n=1 runs of the same reward are NOT comparable.
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8}

export WAND_PROJECT='Search-R1-CF'
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
export TOKENIZERS_PARALLELISM=false

# --- Rthink reward switches ---
# USE_RTHINK=0 routes the reward dispatcher to plain reward_SPRec (InTop@n), i.e.
# the outcome-only RRCM baseline, through this same hardened script (patch_torch
# mount, gpu_mem 0.65, response 2048, rollout.n=8) so the baseline is an identical
# substrate control for the v7 shaping run. Default stays 1 (rthink on).
# M1 baseline: USE_RTHINK=0 EXPERIMENT_NAME=...-gpu-baseline-n8 sbatch ...
# Use THIS script for the baseline arm, NOT run_in_container_baseline.sh — that one
# omits rollout.n entirely (inherits rollout.yaml:102 n=1) and the patch_torch mount.
export USE_RTHINK=${USE_RTHINK:-1}
export RTHINK_MODE=${RTHINK_MODE:-v7}   # retrieval-quality shaping (reward_retrieval/v7)

# v6 HR-faithful outcome reward hyperparameters
export RTHINK_HIT_K=${RTHINK_HIT_K:-10}            # top-K window scored exactly like eval NDCG@K
export RTHINK_TAIL_W=${RTHINK_TAIL_W:-0.10}        # weight of the weak "get warmer" tail gradient
export RTHINK_TAIL_P=${RTHINK_TAIL_P:-4.0}         # tail steepness; large => mid-rank reward negligible
export RTHINK_REAL_BONUS=${RTHINK_REAL_BONUS:-0.0} # decisiveness bonus: exact catalog item (0 = off)
export RTHINK_DENSE_ONLY=${RTHINK_DENSE_ONLY:-0}   # 0 = full v6; 1 = gate-only ablation (no shaping)

# v5 format-integrity gate (the anti-hack core)
export RTHINK_FORMAT_GATE=${RTHINK_FORMAT_GATE:-1}        # 1 = enable malformed-output gate
export RTHINK_FORMAT_PENALTY=${RTHINK_FORMAT_PENALTY:-0.5} # penalty for malformed output

# v5 length discipline
export RTHINK_LEN_SOFT=${RTHINK_LEN_SOFT:-600}      # word budget before penalty
export RTHINK_LEN_W=${RTHINK_LEN_W:-0.0005}         # penalty per word over budget
export RTHINK_LEN_CAP=${RTHINK_LEN_CAP:-0.2}        # max length penalty

# Shaping scale/cap — shared by v3-v7. CAP < the 0.1 smallest answer-tier gap, so shaping
# can only break ties WITHIN a correctness band, never rank a wrong answer above a right one.
export RTHINK_SCALE=${RTHINK_SCALE:-0.10}   # scale of R_think_raw before clipping
export RTHINK_CAP=${RTHINK_CAP:-0.08}       # absolute cap on applied shaping (< 0.1 tier gap)

# v7 retrieval-quality shaping (RTHINK_MODE=v7). Per turn: sim = max cosine(retrieved doc, GT);
# above tau it pays up to +1, below tau it penalizes down to -floor. A rollout that never
# retrieves scores 0 (neutral) — abstention is not scored by this term.
#
# tau CALIBRATED OFFLINE 2026-07-16 against 400 real v3@300 rollouts (was 0.30, the design's
# admittedly uncalibrated guess). Measured best_sim: mean 0.217, median 0.203, p75 0.291.
#   tau   %rewarded   mean retqual
#   0.30      23.5%       -0.056    <- shipped default: abstain (0.0) BEATS retrieval
#   0.22      44.5%       +0.004       break-even
#   0.20      51.5%       +0.022    <- chosen: tau = the measured median
# Why this matters: tau=0.30 sat at the ~77th percentile, so it penalized 3 of every 4
# retrievals. GRPO centers by group mean, so a uniform negative offset cancels — but WITHIN
# a group the contrast between an abstaining rollout (exactly 0.0) and a retrieving one
# (<0) does NOT cancel, so tau=0.30 actively taught "never retrieve". That is the
# retrieval-collapse mode RTHINK_RETQUAL_FLOOR exists to prevent, in a system where RRCM's
# RQ3 already reports retrieval declining over training and ~10% of baseline rollouts
# already abstain. tau=0.20 makes retrieval beat abstention on average.
# This does NOT invalidate Audit B: retqual is monotone in best_sim, so the AUC 0.85 is
# identical for any tau, and raw best_sim alone scored +0.102 vs +0.112 shaped — tau is a
# calibration knob, not the signal. Re-check against the training-time best_sim log: if the
# distribution shifts as the policy learns, the median moves and tau should follow.
export RTHINK_RETQUAL_TAU=${RTHINK_RETQUAL_TAU:-0.20}     # neutral cosine threshold (= measured median)
export RTHINK_RETQUAL_FLOOR=${RTHINK_RETQUAL_FLOOR:-0.25} # damps the below-tau penalty
export RTHINK_W_RETQUAL=${RTHINK_W_RETQUAL:-0.6}          # weight of retqual_agg
# DEFERRED to 0 for the first run: covgain scored AUC 0.499 (chance) in the correlation audit
# because it only fires when a LATER turn beats an earlier one, and 99% of rollouts issue a
# single query. Audit B is the wrong instrument for a term whose target behavior is absent
# from the data. M2 measures offline whether multi-turn raises union coverage at all; only
# then does this get a weight, judged on the behavior it induces, not on correlation.
export RTHINK_W_COVGAIN=${RTHINK_W_COVGAIN:-0.0}          # weight of covgain_agg (DEFERRED)
# 1 = DISABLE retrieval shaping (outcome-only ablation). Name reads backwards; it is v6's
# RTHINK_DENSE_ONLY renamed. orchestrator.py:213 reads `if not _RETRIEVAL_ONLY`.
export RTHINK_RETRIEVAL_ONLY=${RTHINK_RETRIEVAL_ONLY:-0}

# v3 process-shaping hyperparameters (only used by RTHINK_MODE=v3..v6 with DENSE_ONLY=0;
# inert under v7, kept so RTHINK_MODE=v6 still reproduces the old runs from this script)
export RTHINK_W_TOOL=${RTHINK_W_TOOL:-0.30}    # weight of tool_use
export RTHINK_W_GROUND=${RTHINK_W_GROUND:-0.40} # weight of grounding
export RTHINK_W_SYNTH=${RTHINK_W_SYNTH:-0.30}   # weight of synthesis
export RTHINK_W_REP=${RTHINK_W_REP:-0.50}       # weight of self_rep penalty

# --- Batch geometry (GRPO group size + the batch invariant verl does NOT check) ---
export ROLLOUT_N=${ROLLOUT_N:-8}        # GRPO group size. PI-canonical 8. Ladder: 8 -> 12 -> 16.
export TRAIN_BATCH=${TRAIN_BATCH:-56}   # prompts per training step
export PPO_MINI_BATCH=${PPO_MINI_BATCH:-56}  # sequences per optimizer mini-batch; 56*8=448, /56=8

# Pre-flight the three batch constraints. Only ONE of them is enforced inside verl, and it is
# not the one that matters: dp_actor.py:388 splits the rollout buffer with a NON-STRICT chunker
# (protocol.py:912), so an indivisible combo silently yields a short final mini-batch whose loss
# is scaled by the wrong denominator (dp_actor.py:400/418) — a quietly wrong gradient, no error.
# Fail loudly here instead of discovering it in a 22h run's curves.
PPO_MICRO_BATCH=8   # keep in sync with ppo_micro_batch_size_per_gpu below
real_train_batch=$(( TRAIN_BATCH * ROLLOUT_N ))
if (( real_train_batch % PPO_MINI_BATCH != 0 )); then
  echo "FATAL: train_batch($TRAIN_BATCH) * rollout.n($ROLLOUT_N) = $real_train_batch is not" >&2
  echo "       divisible by ppo_mini_batch($PPO_MINI_BATCH). verl will NOT catch this; it would" >&2
  echo "       train a ragged final mini-batch with a mis-scaled loss." >&2
  exit 1
fi
if (( PPO_MINI_BATCH % PPO_MICRO_BATCH != 0 )); then
  echo "FATAL: ppo_mini_batch($PPO_MINI_BATCH) % ppo_micro_batch_per_gpu($PPO_MICRO_BATCH) != 0" >&2
  echo "       (fsdp_workers.py:257 asserts this at runtime)." >&2
  exit 1
fi
if (( TRAIN_BATCH < PPO_MINI_BATCH )); then
  echo "FATAL: train_batch($TRAIN_BATCH) < ppo_mini_batch($PPO_MINI_BATCH)" >&2
  echo "       (actor.py:153 raises ValueError). NB: this is why ppo_mini_batch=64, which the" >&2
  echo "       roadmap suggests for n=8, does not work at train_batch=56." >&2
  exit 1
fi
echo "[pre-flight] batch geometry OK: ${TRAIN_BATCH} x n=${ROLLOUT_N} = ${real_train_batch}" \
     "= $(( real_train_batch / PPO_MINI_BATCH )) mini-batches of ${PPO_MINI_BATCH}"

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
    --env RTHINK_HIT_K=$RTHINK_HIT_K \
    --env RTHINK_TAIL_W=$RTHINK_TAIL_W \
    --env RTHINK_TAIL_P=$RTHINK_TAIL_P \
    --env RTHINK_REAL_BONUS=$RTHINK_REAL_BONUS \
    --env RTHINK_DENSE_ONLY=$RTHINK_DENSE_ONLY \
    --env RTHINK_FORMAT_GATE=$RTHINK_FORMAT_GATE \
    --env RTHINK_FORMAT_PENALTY=$RTHINK_FORMAT_PENALTY \
    --env RTHINK_LEN_SOFT=$RTHINK_LEN_SOFT \
    --env RTHINK_LEN_W=$RTHINK_LEN_W \
    --env RTHINK_LEN_CAP=$RTHINK_LEN_CAP \
    --env RTHINK_SCALE=$RTHINK_SCALE \
    --env RTHINK_CAP=$RTHINK_CAP \
    --env RTHINK_RETQUAL_TAU=$RTHINK_RETQUAL_TAU \
    --env RTHINK_RETQUAL_FLOOR=$RTHINK_RETQUAL_FLOOR \
    --env RTHINK_W_RETQUAL=$RTHINK_W_RETQUAL \
    --env RTHINK_W_COVGAIN=$RTHINK_W_COVGAIN \
    --env RTHINK_RETRIEVAL_ONLY=$RTHINK_RETRIEVAL_ONLY \
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
        data.train_batch_size=${TRAIN_BATCH:-56} \
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
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH:-56} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
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
