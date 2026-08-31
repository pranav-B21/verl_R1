#!/bin/bash

#SBATCH -p gh
#SBATCH -N 2                # two nodes, one GPU each: node[0] retriever, node[1] eval
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH -J test_v8_n8_s250
#SBATCH -o output_test_v8_n8_s250_%j.log

# Held-out eval for the v8 SELECTION arm at the LATER checkpoint:
#   nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v8-n8 @ global_step_250
#
# Submit from a LOGIN node (sbatch does not exist on compute nodes):
#   sbatch sbatch_run_test_v8_n8_s250.sh                   # 3 decodes, ~30 min
#   EVAL_REPEATS=1 sbatch sbatch_run_test_v8_n8_s250.sh    # quick smoke decode
#   EVAL_REPEATS=6 sbatch sbatch_run_test_v8_n8_s250.sh    # full power, matches step 200
#
# Identical to sbatch_run_test_v8_n8.sh except CHECKPOINT_STEP, EVAL_REPEATS and
# the log name -- same entrypoint, same audits -- so the two steps are compared
# on byte-identical evaluation code.
#
# WHY THIS IS A SEPARATE SCRIPT, not an edit to sbatch_run_test_v8_n8.sh
#   Step 200 is the pre-registered cross-arm comparison point (baseline-n8, v7,
#   v7b and v8 all have step-200 decodes). That script must keep reproducing it.
#   This one is the within-arm 250-vs-200 trajectory question, which is a
#   different comparison and gets its own log.
#
# READ BEFORE INTERPRETING THE RESULT
#   1. A higher TRAIN reward curve at 250 is not evidence of a held-out gain.
#      On v5 the held-out metric DECLINED after step 300 while train reward kept
#      climbing (the F9 train<->val gap widens with optimization). 250-vs-200 is
#      a real question, but the train curve is not the answer to it.
#   2. data.val_files=test.parquet, so every val-aux curve in wandb is computed
#      on the TEST set. Picking 250 because a val-aux curve rose would be
#      selecting a checkpoint on test. Pick the step for a stated reason, decode
#      it, and report both steps -- do not report only the better one.
#   3. Gates are unchanged (see test_in_container_v8_n8.sh header):
#      Gate 0 validity -> Gate 1 mechanism -> Gate 2 outcome, in that order.
#      Gate 0 failing at 250 (retrieval collapse late in training is exactly the
#      v7 failure mode) means DISCARD, not "compare anyway".
#   4. HR@1/HR@5 differences between 200 and 250 that are not accompanied by a
#      coverage or pick-GT move are decode noise. Re-decoding one FROZEN
#      checkpoint at temperature 1.0 already swings HR@1 across 0.000-0.005.
#
# Six decodes at ~9 min each land well inside the 12 h wall.
#
# SAFE TO RUN WHILE v8 IS STILL TRAINING: global_step_250 is frozen on disk, and
# sbatch_run_test_rthink.sh writes a job-scoped copy of search_tool_config.yaml
# under /scratch/.../_tool_configs/, so it cannot repoint a running trainer's
# retriever. It does need its own 2 nodes on top of the training job's 2.

export TEST_SCRIPT=test_in_container_v8_n8.sh
export CHECKPOINT_STEP=250

# 3 decodes, overriding the inner script's default of 6. 6 is the pre-registered
# power requirement for the step-200 CROSS-ARM gates (v8/README.md); this is the
# cheaper within-arm 250-vs-200 look, so it is deliberately underpowered:
#   - 3 decodes at temperature 1.0 span roughly the width of the decode noise
#     band itself (HR@1 0.000-0.005 on a FROZEN checkpoint), so a 250-vs-200 HR
#     difference measured at n=3 is NOT separable from noise. Read the mechanism
#     metrics (coverage, pick-GT) and Gate 0, which are far less noisy per decode.
#   - Do not report a 250 "win" from these 3 decodes. If the mechanism moves,
#     re-run at EVAL_REPEATS=6 to match the six step-200 decodes 1:1, which is
#     what the paired comparison needs.
# Anything exported at submit time still wins.
export EVAL_REPEATS=${EVAL_REPEATS:-3}

exec bash /work/11138/pranavbelligundu/vista/verl_R1/sbatch_run_test_rthink.sh
