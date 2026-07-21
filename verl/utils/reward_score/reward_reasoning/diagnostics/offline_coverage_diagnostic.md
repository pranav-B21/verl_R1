# Offline GT-Coverage@k Diagnostic (2026-07-05)

## Question

The 2026-07-02 greedy diagnostic (`train_test_greedy_diagnostic.md`) showed the
train/test gap is a retrieval-coverage gap (GT in retrieved docs: 45–69% train
vs ~5% test), not a reward or regularization problem. That left one fork with
very different costs:

- **Corpus expansion** — test targets are simply *absent* from the corpus, so
  no retrieval setup can find them. Fix: add documents.
- **Corpus restructuring** — test targets are *present* but unreachable from a
  test user's history-shaped query. Fix: reorganize how continuations are
  exposed in the docs.

Secondary questions the same sweep answers for free:

1. How much does raising retriever `topk` (3 → 20 → 50 → 100) buy on test?
2. Is there headroom left in *query formulation* (could a better RL-learned
   query lift test coverage), or is the model already at the ceiling?

All of this is measurable offline — no training, no rollout, one GH200 for a
few minutes of e5 encoding.

## Setup

- Script: `verl_R1/verl/utils/reward_score/reward_reasoning/diagnostics_scripts/coverage_sweep.py`
  (run in the `retriever` conda env). Replays
  queries against the **exact online retrieval stack**: `intfloat/e5-base-v2`,
  `query: ` prefix, mean pooling, L2-normalize, fp16, max_length 1024,
  `data/amazon_data/e5_Flat.index` (85,302 docs, inner product).
- Inputs: saved greedy predictions from the 2026-07-02 diagnostic
  (`outputs/eval/<exp>/global_step_300/greedy_{test,train}_2026-07-02/predictions.json`),
  1000 samples each.
- **Corpus composition** (found during this sweep): `corpora.jsonl` is *not*
  homogeneous — 72,191 user-history docs (`A user played ... "Title1",
  "Title2", ...`) **plus 13,111 item-metadata docs** (`Title: X, Price: ...,
  Brand: ..., Categories: ...`). GT-in-doc matching handles both forms:
  quote-delimited title in history docs, `Title:` field in metadata docs.
  (First pass matched only quoted titles; an implausible oracle-@1 of 0.001
  exposed the metadata docs.)
- Three query variants per sample:
  - **model** — queries extracted from the rollout's `<tool_call>` JSON
    (`query_list`), union of retrieved docs when a rollout made >1 query.
    Realistic coverage under the learned policy.
  - **history** — the raw user-history string from the prompt, phrased like
    the corpus history docs. Ceiling for what any query *policy* could do
    while still querying from the user's history.
  - **gt** — the ground-truth title itself as the query. Oracle probe: is a
    GT-bearing doc reachable in embedding space by *any* single query?
- **coverage@∞** — does the GT title exist anywhere in the corpus at all
  (quoted in any history doc or as any metadata `Title:`), ignoring retrieval.
- Metric: GT-coverage@k = fraction of samples where a GT-bearing doc appears
  in the top-k. Per-sample best ranks saved in
  `diagnostics_scripts/coverage_sweep_results.json` for re-slicing at any k.

## Results

**Test users (n=1000):**

| k | model (v6) | model (v5) | history | gt (oracle) |
|---|---|---|---|---|
| 1   | 0.032 | 0.023 | 0.032 | 0.894 |
| 3   | 0.045 | 0.038 | 0.052 | 0.974 |
| 5   | 0.061 | 0.050 | 0.067 | 0.989 |
| 10  | 0.087 | 0.074 | 0.088 | 0.999 |
| 20  | 0.116 | 0.103 | 0.120 | 0.999 |
| 50  | 0.175 | 0.155 | 0.192 | 1.000 |
| 100 | 0.243 | 0.206 | 0.258 | 1.000 |

**Train users (n=1000, v6 model queries):** @3 = 0.717, @20 = 0.800,
@100 = 0.851; history @3 = 0.789; gt oracle @1 = 0.917.

**coverage@∞: 0.981 test / 0.975 train.**

Replay fidelity: offline model-query coverage@3 reproduces the online
grounding forensics (test 4.5% vs 4.6–5.0% measured live; train 71.7% vs
69.4%), so the offline pipeline faithfully mirrors the served retriever.

## Verdict: restructuring, not expansion

**98.1% of test targets exist in the corpus, and 89.4% are retrievable at
rank 1 by a title-shaped query** (usually the item's own metadata doc). The
targets are present and perfectly embeddable — they are just unreachable
*from the user's history*. History-shaped queries top out at ~26% even at
k=100, because the only path from a test user's history to their target is
incidental co-occurrence in some other user's history doc. Corpus expansion
would add documents to a corpus that already contains the answers; the
binding problem is that nothing in the corpus *associates* a history with its
continuation in a user-independent way.

## Secondary findings

1. **Query policy is exhausted on test.** v6's learned queries sit at the
   history-as-query ceiling at every k (4.5% vs 5.2% @3; 24.3% vs 25.8%
   @100), and v6 > v5 at every k (the query-formulation skill GRPO taught is
   real and transfers — there's just almost nothing left to gain). No reward
   or RL lever acting on query text can close a 26% → 100% gap that lives in
   corpus structure.
2. **Raising topk helps but saturates far below the wall.** 3→20 gives 2.6×
   coverage (4.5% → 11.6%); 3→100 gives 5.4× (24.3%). Even a *perfect*
   selector at k=100 caps test HR@1 at ~24%. At the current non-transferring
   selection rate (~6%, per the greedy diagnostic), topk=20 alone predicts
   ~11.6% × 6% ≈ **0.7% HR@1** — barely above today's 0.5%. topk is a
   multiplier on the real fixes, not a fix.
3. **Practical k is context-limited with the current doc format.** History
   docs run to dozens of quoted titles; k=50–100 of them cannot fit a
   2048-token response budget (docs are already truncated at k=3). Any plan
   that raises k must also shrink docs.

## The fix: CF-continuation docs

Restructure the corpus so continuations are an explicit, user-independent
signal — e.g. sliding-window docs of the form *"users who played X, Y, Z next
played W"* built from train-user histories. One change, three effects:

- **Reachability**: history-shaped queries now embed close to
  continuation-bearing docs, attacking the 26% → 100% oracle gap directly.
- **Transferable selection cue**: "next played W" marks the answer in a way
  that works identically for train and test users. The greedy diagnostic
  proved the model exploits such a cue when it exists (train P(hit | GT in
  docs) = 54.5%).
- **Short docs**: compact continuation docs make k=20+ fit the response
  budget, letting the topk multiplier actually apply.

This is a data-generation script plus a re-index — and its effect is
measurable with this same sweep (re-run against the new index) *before*
spending any GPU-time on retraining.

## Pre-build check: do train behaviors encode test transitions? (2026-07-05)

Before building CF docs, one more leakage-boundary number: a train-derived
transition corpus can only contain (X → target) pairs that occur in *train*
users' sequences. `diagnostics_scripts/transition_coverage.py` reconstructs
the 58,492 full train sequences (avg length 9.9, max 16; 13,109 distinct
items; 111,752 distinct adjacent transitions) from the cumulative-prefix
structure of `CDs_and_Vinyl_train.json` and checks, for each of the same
1000 test samples: does the target appear in train data within *w* steps
after an *anchor* item the test user played?

Fraction of test (history → target) pairs reachable:

| anchor set | w=1 (adjacent) | w=3 | w=5 | w=10 | w=∞ (co-occurrence) |
|---|---|---|---|---|---|
| full history | 0.249 | 0.353 | 0.404 | **0.457** | 0.458 |
| tail-5 | 0.183 | 0.254 | 0.286 | 0.325 | 0.325 |
| tail-3 | 0.136 | 0.175 | 0.194 | 0.220 | 0.220 |
| tail-1 | 0.002 | 0.003 | 0.013 | 0.015 | 0.015 |

(Popularity floor — target appears as *some* continuation in train: 0.991,
consistent with coverage@∞ 0.981.)

Design consequences for the doc builder:

1. **The train-behavior ceiling is ~46%, not 98%.** Roughly half of test
   targets are never preceded by anything the test user played, anywhere in
   train sequences. A pure train-derived transition corpus caps
   GT-coverage at 45.8% no matter how it's chunked or embedded — the ~2×
   remaining gap is where item-similarity backoff (or metadata-doc synergy)
   would have to act. Still: 46% is **9×** the current 5% retrieval
   coverage, and 46% × the proven ~50% selection rate ≈ 23% test HR@1
   upper bound (~45× today's 0.5%).
2. **Use whole-sequence windows, not adjacent pairs.** Strict (X → next)
   transitions encode only 24.9%; widening to w=10 nearly doubles it to
   45.7% and saturates there (sequences max out at 16 items). Docs should
   say "users who played X later played …", aggregating everything after X
   in each sequence.
3. **Anchor on all history items, not just the recent tail.** tail-1 is
   useless (1.5%) and tail-5 gives up a third of the ceiling (32.5% vs
   45.7%). Retrieval must let *any* of the user's items match an anchor doc
   — which favors one aggregated doc per anchor item (~13k docs, one per
   distinct item) and queries that name several history items, over
   prefix-shaped docs keyed to recent context.
4. **Aggregate per anchor.** ~13k anchor items → ~13k compact docs
   ("users who played X later played: W₁ (n₁), W₂ (n₂), …", capped at
   top-N by frequency) instead of 111k+ raw transition docs: smaller
   index, frequency signal for the selector, and short docs so k=20+ fits
   the 2048-token budget.

## Optimization order (updated)

1. ~~Measure whether train behavior encodes test transitions~~ — **done,
   45.8% ceiling** (table above); proceed with aggregated per-anchor,
   whole-sequence-window CF docs.
2. Build CF-continuation corpus + re-index; re-run `coverage_sweep.py`
   against it with added history-tail query variants; **pre-register the
   predicted ceiling** (new coverage@20 × ~50% selection) before training.
3. Raise served `topk` (tool config / `retrieval_launch.sh`) to whatever the
   context budget allows under the new doc format.
4. Then (and only then) retrain; reward work resumes where it has a
   mechanism: grounded-selection credit, multi-turn coverage-gain credit,
   multi-turn un-collapse, JSON-quote fix — as previously ranked.

## Reproduce

```bash
conda activate retriever
cd verl_R1   # run from repo root: the script's --corpus/--index defaults are relative
DIAG=verl/utils/reward_score/reward_reasoning/diagnostics_scripts
python $DIAG/coverage_sweep.py \
  --preds outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6/global_step_300/greedy_test_2026-07-02/predictions.json \
          outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v5/global_step_300/greedy_test_2026-07-02/predictions.json \
          outputs/eval/nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v6/global_step_300/greedy_train_2026-07-02/predictions.json \
  --label v6-test v5-test v6-train \
  --out $DIAG/coverage_sweep_results.json
```

~3 min total on one GH200 (encoding ~3000 queries dominates; FAISS Flat
search over 85k docs is instant). Runs on CPU too, just slower.
