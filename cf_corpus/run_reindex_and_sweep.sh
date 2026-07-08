#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# GPU stage of the CF-corpus pipeline (steps 4-5): encode each candidate corpus
# into a FAISS index with the ONLINE e5 encoder, then run the offline coverage
# sweep with the history-tail query variants. Run on the retriever node in the
# retriever conda env (torch/transformers/faiss). CPU-only steps 1-3 (the
# builder + realized-transition check) already ran on the login node.
#
#   conda activate retriever
#   bash cf_corpus/run_reindex_and_sweep.sh
#
# This is the OFFLINE GATE: do not spend GPU on retraining until the headline
# number (tail5_items coverage@20 on the CF+meta corpus) clears the target in
# cf_corpus/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."   # verl_R1 root

CF=data/amazon_data/cf
PREDS=outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6/global_step_300/greedy_test_2026-07-02/predictions.json

# Candidate corpora to A/B (assemble_corpus.py already produced these):
#   corpora_cf_meta       : CF docs + item metadata  (recommended primary)
#   corpora_cf_meta_hist  : + old history docs       (do they help or crowd?)
for name in corpora_cf_meta corpora_cf_meta_hist; do
  echo "=== build index: $name ==="
  python cf_corpus/build_index.py \
      --corpus "$CF/$name.jsonl" \
      --index  "$CF/e5_Flat_${name}.index"
done

echo "=== coverage sweep (model / history / gt / tail{3,5}_{items,concat}) ==="
for name in corpora_cf_meta corpora_cf_meta_hist; do
  python verl/utils/reward_score/reward_reasoning/diagnostics_scripts/coverage_sweep.py \
      --preds "$PREDS" \
      --label "$name" \
      --corpus "$CF/$name.jsonl" \
      --index  "$CF/e5_Flat_${name}.index" \
      --tails 3 5 \
      --out "$CF/coverage_sweep_${name}.json"
done

echo
echo "Headline: tail5_items coverage@20 on corpora_cf_meta."
echo "Predicted test HR@1 ~= that coverage * 0.5 (train-side selection existence proof)."
echo "Compare to the pre-registered ceiling in cf_corpus/README.md before retraining."
