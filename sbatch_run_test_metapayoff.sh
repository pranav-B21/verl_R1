#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J metapayoff
#SBATCH -o output_test_metapayoff_%j.log

# STAGE A2 -- power the metadata-payoff estimator that REWARD_REASONING_ANALYSIS.md
# Part 15.5 flags as UNDERPOWERED, NOT NEGATIVE.
#
# WHAT IT IS FOR
#   Part 15 killed the corpus-repair hypothesis: repairing SalesRank/Categories
#   moved META% up at every early step in both arms and moved HR@5 by less than
#   one hit per 1000. But it could not answer the question underneath it --
#   whether reading a metadata doc pays at all -- because the paired estimator
#   ran out of pairs:
#
#     n = 136 paired prompts at baseline-metafix@200
#     n =  36 at baseline-metafix@250
#     n =   5 at v8-metafix@200
#     n = 960 for the 9-decode broken-corpus prior it is being compared to
#
#   The step-200 figure (HR@5 -0.00368, CI [-0.01103, +0.00000]) is approximately
#   ONE EVENT. Part 15.5 states in terms it "must not be quoted as a negative
#   result". This job gets a real number.
#
#   That number is the gate on the only reward lever Part 15.3 left standing.
#   The mechanism finding there is that the metadata memory dies with the SECOND
#   QUERY, not on its merits: once docs/ret hits 3.0 (one query), META% is 0-3%
#   in every arm and both corpus versions. "Force a second, metadata-directed
#   query" is the reward change that follows -- and it is only worth building if
#   reading metadata pays. If this comes back null at n >= 500 pairs, that whole
#   family is dead and Stage C of the plan is cancelled.
#
# WHY TEMPERATURE 1.0 -- THIS IS THE POINT, NOT AN OVERSIGHT
#   memtype_payoff.py pairs WITHIN PROMPT. It needs prompts where some rollouts
#   read a metadata doc and some did not. Greedy decoding suppresses exactly that
#   discordance: two greedy decodes of one frozen checkpoint already reproduce
#   62.0% of answers identically. Greedy is the right regime for measuring
#   accuracy and the wrong regime for this estimator, which is why the
#   3-decodes-per-step greedy sweep left n=5 at v8@200.
#
#   CONSEQUENCE FOR THE TABLES: these decodes write to decode<i>_<date>, the
#   temperature-1.0 path. Their HR/NDCG rows go into TEST_OUTPUT.md LABELLED
#   temp-1.0 and must not be pooled with, or compared against, the greedy
#   metafix rows recorded 2026-08-26. Greedy and temp-1.0 rows are never
#   comparable; see TEST_OUTPUT.md's regime column.
#
# WHY ONLY STEPS 200 AND 250
#   They are the only steps where META% is still high enough to produce
#   discordant pairs at all. On the repaired corpus the baseline arm reads
#   52.1% META at 200 and 20.8% at 250, then 2.9% at 300 and <=1.0% after. A
#   step-300+ decode would spend GPU to add pairs at a rate of ~1 in 35 rollouts.
#
# WHY BASELINE-METAFIX AND NOT v8
#   v8 was already down to one query by step 200 (META% 10.6% repaired), so it
#   has the fewest pairs of any arm -- it is where n=5 came from. The estimator
#   asks a question about the memory, not about the reward, so it should run
#   where the memory is actually being used.
#
# Submit from a LOGIN node (sbatch is unavailable on compute nodes):
#   sbatch sbatch_run_test_metapayoff.sh
#   EVAL_REPEATS=2 sbatch sbatch_run_test_metapayoff.sh    # plumbing check only
#
# WALL TIME. 2 steps x 9 repeats = 18 decodes at temperature 1.0. Both
# checkpoints are ALREADY MERGED under /scratch/.../merged_models/, so there is
# no merge cost. Historical baseline temp-1.0 decodes run ~20 min, so budget
# ~6 h; 12 h is headroom for a retriever restart cycle.
#
# AFTER THE JOB (the audits need sentence-transformers + torch, which neither
# the login-node python3 nor the `retriever` conda env has -- container only):
#
#   bash scripts/run_audit_in_container.sh memtype_payoff \
#     outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix/global_step_200/decode*/test_predictions.json \
#     --label base200-metafix-9dec \
#     --json verl/utils/reward_score/reward_retrieval/audits/memtype_payoff_metafix_base200_9dec_$(date +%F).json
#
#   bash scripts/run_audit_in_container.sh memtype_payoff \
#     outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix/global_step_250/decode*/test_predictions.json \
#     --label base250-metafix-9dec \
#     --json verl/utils/reward_score/reward_retrieval/audits/memtype_payoff_metafix_base250_9dec_$(date +%F).json
#
# READING THE RESULT. Check the paired n FIRST, before the effect size. Below
# ~500 pairs this reproduces the same underpowered non-answer Part 15.5 already
# recorded, and the correct write-up is "still underpowered", not "null".

export DECODE_TEMPERATURE=1.0
export EVAL_REPEATS=${EVAL_REPEATS:-9}
export RTHINK_RUNS=${RTHINK_RUNS:-"nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix:200 nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline-n8-metafix:250"}

# NOT set: DECODE_PARQUET/DECODE_TAG. This one IS a held-out measurement and
# belongs on the default test.parquet under the default decode<i>_ tag -- unlike
# sbatch_run_test_traindiag_g0a2.sh, which must be tagged away from the tables.
#
# NOT set: EXPERIMENT_NAME. test_in_container_rthink.sh:163 lets a single
# EXPERIMENT_NAME override RTHINK_RUNS entirely, which would silently drop the
# second step.

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
