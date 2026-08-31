#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J g0a2_traindiag
#SBATCH -o output_test_traindiag_g0a2_%j.log

# G0-A2 -- the measurement REWARD_REASONING_ANALYSIS.md Part 14.4 pre-declares
# "before any reward work resumes", and which has never been run (no traindiag
# decode directory exists anywhere under /scratch/.../verl/ as of 2026-08-30).
#
# WHAT IT IS FOR
#   Parts 11-13 closed the reward axis on one number: 95% of rollouts land past
#   GT rank 500, so there is no within-group outcome variance for GRPO to
#   descend. That number was measured on HELD-OUT decodes. GRPO computes
#   gradients on TRAINING rollouts, where the 1-in-64 orchestrator prints show
#   54.8% of rollouts rank the GT first and 59.9% retrieve it. Part 14 therefore
#   marks every "no gradient" claim in the document as measured on the wrong
#   distribution -- and marks its own refutation as incomplete, because marginal
#   spread is not within-group spread and the 1-in-64 prints cannot be grouped
#   by prompt.
#
#   This job decodes the SAME prompt 8 times so the repeats stand in for a real
#   n=8 GRPO group (advantage_mass.py:46 documents that substitution), on the
#   training distribution. It settles Part 12's central claim in either
#   direction. It is the gate on whether a v9 reward is worth building at all.
#
# THESE ARE TRAINING PROMPTS. THE HR/NDCG NUMBERS ARE NOT A RESULT.
#   eval.py still runs and still prints HR@1/HR@5/NDCG@5 at the end of each
#   decode. Those are TRAIN metrics against a policy that has seen these
#   prompts. They must NEVER be appended to TEST_OUTPUT.md and must never be
#   compared to any held-out row. The only output that matters here is
#   test_predictions.json, consumed by advantage_mass.py.
#
#   The DECODE_TAG below is what keeps this honest: test_in_container_rthink.sh:149
#   hard-fails a non-default DECODE_PARQUET under the default tag, so these
#   decodes cannot land in decode<i>_/greedy<i>_ and be swept up by the audit
#   globs. Do not "simplify" the tag away.
#
#   train_diag_1000.parquet was verified 2026-08-30 to be a clean subset of the
#   training split: 997 unique prompts, 997/997 present in train.parquet, 1
#   overlapping test.parquet, and 98.4% of its ground truths present in the eval
#   catalog (test.parquet is 98.3%), so the GT-ranking step is not biased
#   relative to the held-out measurement it is being compared against.
#
# WHY TEMPERATURE 1.0, NOT GREEDY
#   Training rollouts are sampled at temperature 1.0
#   (actor_rollout_ref.rollout.temperature=1.0). A greedy group would have
#   almost no within-group spread by construction and would answer a question
#   nobody asked. This job must match the training decode regime, not the
#   evaluation one.
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_traindiag_g0a2.sh
#   EVAL_REPEATS=2 sbatch sbatch_run_test_traindiag_g0a2.sh   # plumbing check only
#
# WALL TIME. 2 arms x 8 repeats = 16 decodes of 1000 prompts at temperature 1.0.
# Both step-200 checkpoints are ALREADY MERGED under
# /scratch/.../merged_models/<arm>/global_step_200, so there is no merge cost.
# Historical temp-1.0 decodes run ~20 min (baseline) and ~7 min (v8), so budget
# ~4-5 h; 12 h is headroom for a retriever restart cycle.
#
# WHY STEP 200. It is the pre-registered cross-arm comparator for this project
# and the step every other metafix measurement is anchored to, so the resulting
# component attribution can be read directly against the held-out numbers in
# Part 14.1 (outcome 1.6% / shaping 14.4% / length 43.7% / format 40.3%) and
# against the repaired-corpus held-out control measured 2026-08-30
# (outcome 4.8% / shaping 20.1% / length 52.0% / format 23.2%).
#
# AFTER THE JOB, run the audit (it needs sentence-transformers + torch, which
# NEITHER the login-node python3 NOR the `retriever` conda env has -- it only
# runs inside the container; scripts/run_audit_in_container.sh wraps that):
#
#   bash scripts/run_audit_in_container.sh advantage_mass \
#     --label baseline200-TRAIN \
#     outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix/global_step_200/traindiag*/test_predictions.json \
#     --json verl/utils/reward_score/reward_retrieval/audits/advantage_mass_baseline200_TRAIN_$(date +%F).json
#
#   bash scripts/run_audit_in_container.sh advantage_mass \
#     --label v8-200-TRAIN \
#     outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8-metafix/global_step_200/traindiag*/test_predictions.json \
#     --json verl/utils/reward_score/reward_retrieval/audits/advantage_mass_v8_200_TRAIN_$(date +%F).json
#
# READ: advantage_mass.py prints its own pre-registered verdict at a >=40%
# dead-group threshold. Read the THRESHOLD-FREE component attribution as the
# headline instead -- Part 14.1 records that this script's first pre-registered
# statistic was mis-specified (exact-equality on a reward that is continuous in
# rank) and returned a definitional null. The tolerance form is the fix, but the
# attribution table is the number that does not depend on a threshold at all.

export DECODE_PARQUET="/work/11138/pranavbelligundu/vista/verl_R1/data/amazon_data/train_diag_1000.parquet"
export DECODE_TAG=traindiag
export DECODE_TEMPERATURE=1.0
export EVAL_REPEATS=${EVAL_REPEATS:-8}
export RTHINK_RUNS=${RTHINK_RUNS:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix:200 nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8-metafix:200"}

# NOT set: TEST_SCRIPT. The stock test_in_container_rthink.sh already takes
# DECODE_PARQUET/DECODE_TAG and carries the guard; a per-arm entrypoint would
# only be another copy to keep in sync.
#
# NOT set: EXPERIMENT_NAME. test_in_container_rthink.sh:163 lets a single
# EXPERIMENT_NAME override RTHINK_RUNS entirely, which would silently drop one
# of the two arms.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
