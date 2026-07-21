# Train-vs-Test Greedy Diagnostic (2026-07-02)

## Question

Train reward climbs (v5 got `r_ans` to +0.38) while held-out HR sits at ~0. Two
mutually exclusive explanations, demanding opposite responses:

- **A — generalization gap**: the model genuinely learned the task on training
  users but it doesn't transfer. Fix with RL levers (KL, early stopping, data).
  Reward is fine.
- **B — representation ceiling**: rising train reward is temperature-1 sampling
  luck harvested by GRPO; the model's greedy best guess is as bad on train as on
  test. No reward shape can help.

Every reward iteration (v2–v6) implicitly bet on A without checking. This
diagnostic fills the missing cell of the 2×2 (train/test × sampled/greedy):
greedy decode on a fixed sample of **training** prompts, scored with the
standard `eval.py` HR pipeline.

## Setup

- Checkpoints: `rthink-v5 @ global_step_300` (the run whose train reward
  climbed; step 300 is its best per prior evals) and `rthink-v6 @ global_step_300`.
- Train sample: `data/amazon_data/train_diag_1000.parquet` — 1000 of 4096
  train prompts, numpy seed 42, sorted indices. Same schema as `test.parquet`.
- Decode: greedy (`temperature=0.0, top_p=1.0, top_k=-1, n_samples=1`),
  otherwise identical to the standard eval pipeline (multi-turn qwen tool
  format, max 4 assistant turns, retriever on amazon corpus, topk as in
  `retrieval_launch.sh`).
- Scripts (in `verl_R1/verl/utils/reward_score/reward_reasoning/diagnostics_scripts/`):
  `test_in_container_traindiag.sh` (per-pass: merge→generate→to-json→
  `eval.py` top1/top5), `sbatch_run_test_traindiag.sh` (2-node wrapper, unused
  this time — ran directly on an active idev allocation, retriever already live
  on `c642-042:8000`).
- Outputs: `outputs/eval/<experiment>/global_step_300/greedy_<split>_2026-07-02/`
  (never overwrites prior test evals).

## Results

| checkpoint | split | decode | HR@1 | HR@5 | NDCG@5 | ORRatio@1 |
|---|---|---|---|---|---|---|
| rthink-v5 @300 | train (n=1000) | greedy | **0.197** | **0.206** | 0.202 | 0.028 |
| rthink-v5 @300 | test (n=1000)  | greedy | 0.002 | 0.002 | 0.002 | 0.036 |
| rthink-v6 @300 | train (n=1000) | greedy | **0.391** | **0.400** | 0.396 | 0.013 |
| rthink-v6 @300 | test (n=1000)  | greedy | 0.005 | 0.007 | 0.006 | 0.020 |

Reference (temp-1 single-sample, prior evals): baseline @500 HR@1 0.004 /
HR@5 0.007; v5/v6 @300 HR@1 0.000–0.005 across seeds.

**Verdict: outcome A-shaped — train-greedy ≫ test-greedy by ~80–100× on both
checkpoints (v6 even more extreme: 0.391 vs 0.005).** The model
did learn the task; there is no representation ceiling. The +0.38 train reward
was real skill on training users, not sampling luck. But the grounding analysis
below shows the gap is *not* classic parameter overfitting — it is almost
entirely a **retrieval-coverage gap plus a selection-cue gap**, which redirects
the fix away from both reward shaping *and* generic RL regularization.

## Grounding forensics (greedy rollouts)

Exact-match of the final `<answer>` against ground truth, conditioned on
whether the GT title literally appears in the `<tool_response>` docs of the
same rollout:

| | v5 train | v5 test | v6 train | v6 test |
|---|---|---|---|---|
| rollouts that searched (`<tool_call>`) | 96.4% | 95.2% | 99.8% | 99.7% |
| **GT present in retrieved docs** | **44.7%** | **4.6%** | **69.4%** | **5.0%** |
| P(exact hit \| GT in docs) | **42.7%** | 2.2% (1/46) | **54.5%** | 6.0% (3/50) |
| P(exact hit \| GT **not** in docs) | 0.2% | 0.1% | 1.6% | 0.1% |
| overall exact-match | 19.2% | 0.2% | 38.3% | 0.4% |

Four facts fall out:

1. **The model is a doc-selector, not a knowledge-recommender.** Hits without
   doc support are ~0 on both splits (0.1–1.6%). Every correct answer is copied
   from a retrieved doc. Consistent with the earlier v6 forensics ("89% of
   answers grounded in docs").
2. **Coverage is ~10× higher on train than test (v5: 44.7% vs 4.6%; v6: 69.4%
   vs 5.0%).** The corpus is built from training users' histories, so for a
   train user the retriever can surface a doc containing that user's own
   continuation. Test users' targets are mostly absent — the ~5% is incidental
   overlap via other users' docs. This is the hard ceiling the 2026-07-02
   forensics identified, now measured on train too.
3. **Selection also collapses on test (v5: 42.7% → 2.2%; v6: 54.5% → 6.0%).**
   Even when GT *is* in the test docs the model rarely picks it. On train the
   GT-bearing doc is (likely) the user's own history doc, where "title in doc
   but not in prompt" is a strong positional cue for the continuation; on test
   the GT sits uncued inside another user's doc. So the learned selection
   strategy leans on a train-only cue and does not transfer. (Upper bound if
   selection *did* transfer: ~5% × ~50% ≈ 2.5% test HR@1 — an order of
   magnitude above current, but still capped by coverage.)
4. **Query formulation is learnable and moves train coverage, but test coverage
   is corpus-limited.** v6 pushed train GT-in-docs from v5's 44.7% to 69.4%
   (and search rate to ~100%) — GRPO taught it to write queries that recover
   the GT-bearing doc. Test coverage barely moved (4.6% → 5.0%): no query can
   retrieve a continuation the corpus doesn't associate with the query user.
   So offline GT-coverage@k must be measured on **test** users, and the fix is
   corpus content, not query quality.

## Implications

- **Stop reward-shape iteration** (confirms the standing conclusion). The
  reward successfully taught search-then-select; it cannot teach the retriever
  to surface the answer, and GRPO gets no gradient on test-style prompts where
  the answer is absent from the docs 95% of the time.
- **Generic RL regularization (stronger KL, early stop) is also not the fix.**
  The gap isn't parametric memorization — unsupported hits are ~0 even on
  train. The transferable skill ceiling is set by (a) coverage and (b) whether
  docs carry a *transferable* continuation cue.
- Optimization order stands, now with measured targets:
  1. Raise retriever topk 3→20+ and measure GT-coverage@k offline on **test**
     users (coverage is the binding constraint: 4.6% @ current setup).
  2. Restructure corpus docs to expose continuations as an explicit,
     user-independent CF signal ("users who played X next played Y") so the
     selection cue transfers; the train-side 42.7% selection shows the model
     can exploit such a cue once it exists.
  3. Un-collapse multi-turn search (99% single-query one-shot), selector
     credit, JSON-quote fix — as previously ranked.

## Follow-up: offline coverage sweep (2026-07-05)

The coverage@k measurement recommended above was run offline via
`coverage_sweep.py` — full report in **`offline_coverage_diagnostic.md`**.
Headline: coverage@∞ = 98.1% (test targets *exist* in the corpus) but
history-shaped queries reach only 24–26% even at k=100, with v6's learned
queries already at that ceiling → the fix is corpus **restructuring**
(user-independent CF-continuation docs), not expansion, not topk, not query
policy.

## Reproduce

```bash
# on a node with GPU + retriever reachable (or via sbatch_run_test_traindiag.sh)
DIAG_RUNS="nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5:300" \
DIAG_SPLITS="train test" \
  bash verl_R1/verl/utils/reward_score/reward_reasoning/diagnostics_scripts/test_in_container_traindiag.sh
```

Each 1000-prompt greedy pass takes ~6–7 min on one GH200 (merged model cached).
