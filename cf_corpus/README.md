# CF-continuation corpus build

> **STATUS 2026-08-30 — DEAD, not paused. Do not build on this.**
> Superseded twice. (1) The PI rejected the diagnosis on 2026-07-11: the corpus is not to
> change, and 4.6% (since corrected to ~3.1%) coverage is a property of the benchmark, not a
> bug. (2) The paper is a journal extension on a frozen corpus/benchmark, so corpus edits
> would break the comparability the Table-1 comparison depends on.
> The claim below that the system is "retrieval-bound, not reward-bound" was also
> **superseded on its own terms**: `coverage × P(pick GT | covered)` decomposes the ceiling,
> and the selection factor is at chance (`REWARD_REASONING_ANALYSIS.md` §11.2), so widening
> retrieval is measurably *harmful* at a chance-level selector (§10.8). Kept for provenance
> and for the diagnostics it produced.

Restructure train-user behaviour into **user-independent CF-continuation docs**
so a test user can reach the continuation of their history. This is the single
mandatory fix identified by three diagnostics: the system is **retrieval-bound,
not reward-bound**.

- Reward is not the constraint — train-greedy HR@1 reached 0.39 (v6).
- Model is not the constraint — same.
- Corpus *content* is not the constraint — coverage@∞ 98.1%, GT-as-query oracle @1 = 89%.
- **The constraint:** nothing connects a *history* to its *continuation* in a
  reachable way. History-shaped queries top out at ~26% coverage@100 against the
  100% oracle. Test HR@1 sits at 0.002–0.005 (≈ decode noise).

The fix surfaces transitions in a form a test user's recent history can retrieve:

```
Users who played "Kind of Blue" next played: "Bitches Brew" (41), "In a Silent Way" (17), ...
```

## Pipeline

| Step | Script | Where | Status |
|------|--------|-------|--------|
| 1–3 extract → aggregate → render | `build_cf_corpus.py` | login node (CPU, secs) | **done** |
| 4a assemble parts + renumber ids | `assemble_corpus.py` | login node (CPU) | **done** |
| 4b re-index (encode + FAISS) | `build_index.py` | retriever node (GPU) | script ready |
| 5 coverage sweep (offline gate) | `diagnostics_scripts/coverage_sweep.py` (+`--tails`) | retriever node (GPU) | script ready |
| 4b+5 driver | `run_reindex_and_sweep.sh` | retriever node (GPU) | script ready |

### Step 1–3: build CF docs (already run)

```bash
python cf_corpus/build_cf_corpus.py \
    --anchor-len 1 --window 10 --min-support 2 --max-cont 30 \
    --out data/amazon_data/cf/corpora_cf.jsonl \
    --report data/amazon_data/cf/build_report.json
```

Design (each choice justified in the docstring): **train users only** (no test
leakage); **aggregate with counts** (bounds length, gives the selector a
frequency cue, avoids raw-co-occurrence popularity); **single-item anchor**
(`--anchor-len 1`, maximises coverage, matches a recency-shaped query); **windowed
continuations** (`--window 10 ≈ inf` since train seqs avg ~10, max 16); **min-support
+ top-M pruning** for length.

**Builder validated:** with pruning off (`--min-support 1 --max-cont 100000`)
it reproduces `transition_coverage_results.json` — tail-1 = 0.015 exact, tail-3
0.206 / tail-5 0.313 vs 0.220 / 0.325 (residual = self-loop skip + canonical
title merging, both benign).

### Step 4: assemble + re-index

```bash
# one-time split of the existing corpus into item-metadata / history parts
python cf_corpus/assemble_corpus.py --split-base \
    --base data/amazon_data/corpora.jsonl --out-dir data/amazon_data/cf

# primary corpus: CF docs + item metadata (kept as oracle-reachable item reps)
python cf_corpus/assemble_corpus.py \
    --inputs data/amazon_data/cf/corpora_cf.jsonl data/amazon_data/cf/corpora_meta.jsonl \
    --out data/amazon_data/cf/corpora_cf_meta.jsonl
# A/B variant that also keeps the 72k history docs (do they help or crowd?)
python cf_corpus/assemble_corpus.py \
    --inputs .../corpora_cf.jsonl .../corpora_meta.jsonl .../corpora_history.jsonl \
    --out data/amazon_data/cf/corpora_cf_meta_hist.jsonl

# GPU: encode + index + sweep both candidates
conda activate retriever
bash cf_corpus/run_reindex_and_sweep.sh
```

`build_index.py` reproduces the **online** doc encoding exactly (e5-base-v2,
`passage: ` prefix, mean-pool, L2-norm, fp16, `IndexFlatIP` 768-d). Line order of
the assembled JSONL == vector order of the index — build both from the same file.

## Assembled artifacts (produced, on disk)

| File | Docs | Note |
|------|------|------|
| `cf/corpora_cf.jsonl` | 12,607 | CF-continuation docs only (headline config) |
| `cf/corpora_meta.jsonl` | 13,111 | item-metadata docs sliced from base corpus |
| `cf/corpora_history.jsonl` | 72,191 | old per-user history docs sliced from base |
| `cf/corpora_cf_meta.jsonl` | 25,718 | **recommended primary** (CF + metadata) |
| `cf/corpora_cf_meta_hist.jsonl` | 97,909 | A/B (CF + metadata + history) |

Headline config docs: mean 673 chars ≈ **168 tokens**, p95 1120 chars; k=10
retrieved ≈ 1.7k tokens.

## Realized transition coverage — the intrinsic (pre-retrieval) ceiling

Fraction of the 1000 test `(history → target)` pairs whose target appears as a
"next played" continuation in a **tail-k-reachable** CF doc. This is the
embedding-free ceiling; the coverage sweep measures how much e5 recovers.

| config (support / max-cont) | docs | ≈tok/doc | tail-3 | tail-5 |
|---|---|---|---|---|
| raw uncapped ceiling | — | — | 0.206 | 0.313 |
| **2 / 30 (headline)** | 12,607 | 168 | **0.104** | **0.158** |
| 2 / 50 | 12,607 | 208 | 0.125 | 0.184 |
| 2 / 100 | 12,607 | 249 | 0.140 | 0.207 |
| 2 / 15 | 12,607 | 111 | 0.085 | 0.132 |

vs the **old** corpus: history-query coverage@100 ≈ 0.12 and test HR@1 ≈ 0.002–0.005.
`max_cont` is the coverage↔length↔embedding-tightness knob; the GPU sweep
(retrieval precision per doc length) is its real arbiter — regenerate any config
in seconds.

## Pre-registered ceiling (write this down BEFORE retraining)

```
predicted test HR@1  =  coverage@k(new corpus, tail5_items)  ×  selection_rate
                     ≈  coverage@20                           ×  0.50
```

`0.50` is the **train-side selection existence proof** (P(hit | GT in docs) ≈
0.54 when the cue exists); the CF cue ("next played") is designed to transfer
because it is user-independent.

- **Intrinsic ceiling caps** `tail5_items` coverage@20 at **0.158** (headline) /
  **0.207** (max-cont 100).
- **Pre-registered HR@1 ceiling:** `0.158 × 0.50 ≈ 0.079` (headline) up to
  `0.207 × 0.50 ≈ 0.10` (max-cont 100) — a **~16–50× lift** over today's
  0.002–0.005.
- After the GPU sweep, restate the prediction with the **measured** coverage@20
  (retrieval recovers ≤ intrinsic). A trained model that then approaches this
  pre-registered number is the empirical payoff; a post-hoc explanation is not.

**Gate:** if `tail5_items` coverage@20 on `corpora_cf_meta` does not clear ~0.10,
fix the *builder* (raise `--max-cont`, longer `--window`, or item-similarity
backoff), not the index — GPU retraining stays blocked until it clears.

## Known follow-ups (not blocking the corpus build)

- **Title canonicalization** is the fiddly part (flagged: apostrophes/quotes
  bite). Junk titles (`","`, giant concatenated metadata strings) exist in the
  **source** name2id/corpus, not introduced here; one shared normaliser across
  sequences ↔ name2id ↔ doc text ↔ exact-match reward avoids silent undercounting.
- **Serving (retraining time, after the gate):** raise served `--topk` in
  `retrieval_launch.sh` / `search_tool_config.yaml` to what the new doc length
  affords; fix the JSON-quote bug in the query path (deletes coverage per failed
  call); un-collapse multi-turn (a follow-up query can now raise coverage).
- **Reward work resumes only after retrieval surfaces candidates:** grounded-
  selection term (credit only when the answer is literally in a retrieved doc),
  then multi-turn coverage-gain credit, then any outcome reshaping — now pointed
  at a measured bottleneck with an existence proof.
