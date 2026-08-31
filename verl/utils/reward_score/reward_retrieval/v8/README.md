# v8 — the selection reward

**Status (2026-07-30):** implemented, not yet run. `RTHINK_MODE=v8`.

## Why v8 exists

v7b confirmed its behavioral claim and produced a null on its scientific one.
The funnel measured at `global_step_200` says why:

| stage | baseline-n8 (6 decodes) | v7b (3 decodes) |
|---|---|---|
| GT reaches the retrieved docs (coverage) | 3.13% | 2.77% |
| …and the model answers *from* the docs | 30% | 64% |
| …and picks the **right** item | **0 / 57** | **2 / 53** |
| chance, given ~26 / ~18.5 retrieved items | 3.8% | 5.4% |

**Selection is at chance.** Coverage multiplies against ~0, which is why
`best_sim` could rise 58% during v7b training (0.144 → 0.227) with no movement
in HR or in binary coverage. v8 therefore spends its budget on selection.

## What v8 changes

1. **`r_select` (new)** — credits the *structure* of the answer rather than its
   identity: is it a **continuation** of something this user actually played,
   according to the ordered sequences already present in the retrieved docs?
   Plus a small grounding credit and a penalty for re-recommending an item the
   user has already played. See `selection.py` for the measured support.
2. **`r_cover` replaces soft-only `r_retqual`** — exact GT-in-retrieved-titles
   earns full credit (the quantity the dev metric measures), with v7's cosine
   retained as a dense tail so GRPO groups are not all-zero at a ~3% base rate.
3. **The length budget stops charging for `<tool_call>` text.** `docs/ret ≈
   queries/call × topk`, and v7b's queries/call is 1.47 against the baseline's
   2.26 at an identical 100% retrieval rate — the budget was taxing retrieval
   breadth. Restore the old behavior with `RTHINK_LEN_COUNT_CALLS=1`.

Everything else is v7b verbatim: v6's outcome reward, v5's format gate, the
turns-aware budget with `<tool_response>` stripped.

## PRE-REGISTERED prediction — write this down before the run

The successor cue contains the GT in **13%** of retrievable cases. So v8's
ceiling, at *perfect* selection within the successor set, is:

```
coverage 2.77%  ×  13%  =  HR@1 ~0.0036      (~3× the baseline's ~0.001)
```

still below the Table-1 HR@5 target of 0.0102. **A larger HR jump than ~3× is
more likely a measurement artifact than a win** and must be investigated before
being reported. If coverage rises but selection stays at chance, predicted HR@1
is 0.047 × 0.054 ≈ 0.0025 — under the bar below — so a coverage-only gain must
not be claimed as a win.

## How we decide v8 beats baseline-n8

Three gates, evaluated in order. **All three must pass.** Comparison step is
locked at **200**, matching the 6 existing baseline decodes; `EVAL_REPEATS=6`
for the new arm (see power below).

**Gate 0 — validity.** The run is discarded, not compared, unless:
`retr% ≥ 95` · `answer rate ≥ 95%` · `turns_max < 10` (no runaway loops) ·
`docs/ret ≥ 3` (retriever was alive). v7's collapse would fail this gate, which
is the point: an arm that wins by not retrieving has not won.

**Gate 1 — mechanism.** At least one funnel stage must move, tested on the
corrected metric:
  * coverage ≥ **4.0%** (vs 3.13%), **or**
  * P(pick GT | GT in docs) ≥ **15%** (vs 5.4% chance).

This gate exists because at ~3 HR events per 1000 an HR swing will eventually
appear by chance. Requiring the mechanism to move first is what stops the v7
failure — "HR ties the baseline" at a 5% retrieval rate — from being read as a
tie ever again.

**Gate 2 — outcome.** HR@5 ≥ **0.009** (3× baseline's 0.003), by a
prompt-level **paired** test across the 6 decodes. Pairing matters: both arms
decode the identical 1000 prompts, so a paired test is materially more powerful
than the unpaired two-proportion test used to size the table below.

Passing Gates 0–1 but not 2 is a real partial result — "the mechanism moved,
the outcome is not yet resolvable" — and is worth continuing. Passing 2 without
1 is not a win; it is a decode-noise draw and should be re-decoded.

### Why 6 decodes (80% power, α=0.05, unpaired ⇒ conservative)

| endpoint | baseline | detect | prompt-decodes | decodes |
|---|---|---|---|---|
| coverage | 3.13% | +50% rel (4.70%) | 2,392 | **2.4** |
| coverage | 3.13% | +25% rel (3.91%) | 8,758 | 8.8 |
| selection \| covered | 5.4% chance | 15% | 152 winnable | **5.4** |
| HR@5 | 0.30% | 3× (0.90%) | 2,597 | **2.6** |
| HR@5 | 0.30% | 2× (0.60%) | 7,810 | 7.8 |

Six decodes clears every gate threshold with margin. It does **not** power a
2× HR@5 detection (~8 needed), which is deliberate: a 2× HR move is below what
this task can resolve, and Gate 1 is what carries the scientific claim.

### Caveats that must travel with any result

* **Validation is the test set.** `data.val_files=test.parquet`, so every
  `val-aux/*` curve is computed on held-out test data and any checkpoint chosen
  by it is selected on test. This is why the comparison step is fixed at 200 in
  advance rather than picked post hoc.
* **Coverage is dominated by popular items.** Splitting held-out targets by how
  many corpus documents contain them, coverage is 5.21% (baseline) on items in
  ≥50 documents versus **1.06%** on the long tail, and ~80% of all coverage
  events come from the popular bucket. On the tail the two arms are
  indistinguishable (1.06% vs 1.20%, p=0.69). Report coverage split by bucket,
  or a "win" may be nothing more than a shift toward popular items — which
  `ORRatio`/`DivRatio` would also register.
* **~1.5% of held-out targets are corrupt** (15/1000 are the scraped string
  `<span class="a-size-medium a-color-secondary a-text-normal`). That is
  3–15× the size of the HR signal. It affects both arms equally so it does not
  bias the A/B, but it caps achievable HR and should be stated.

## Ablations worth running if v8 passes

* `RTHINK_W_SELECT=0` — isolates how much of any gain is the successor cue.
* `RTHINK_LEN_COUNT_CALLS=1` — isolates the query-breadth change.
* `RTHINK_RETRIEVAL_ONLY=1` — shaping off; must reproduce the baseline.
