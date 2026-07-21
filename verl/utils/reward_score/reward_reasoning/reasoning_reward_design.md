# Reasoning Reward (`R_think`) — Design & Status

The base document for the reasoning-reward track: what the math is, how the
system is wired up, what has been tried, what is currently live, and what is
still planned. Treat this as the entry point; it summarizes and links out to
the deeper per-version READMEs, the full `REWARD_REASONING_ANALYSIS.md`
postmortem, and the `diagnostics/` folder rather than repeating them in full.

Its sibling document is
[`../reward_retrieval/retrieval_reward_design.md`](../reward_retrieval/retrieval_reward_design.md),
which covers the retrieval-quality reward that this track's diagnostics led to.

> **Updated 2026-07-16** to fold in the PI's reply: canonical GRPO group size is
> **`n=8`** (not the `n=5` earlier drafts used). The `n=1` was a bug in the
> *original* code; the PI confirmed it, fixed it to 8 in his own experiments, and
> "didn't push the newer code" — so **get his newer code**, and note the RRCM
> Table-1 baseline was itself produced at `n=8` (the local `n=1` baseline was never
> valid). Escalation ladder if under-performing: **8 → 12 → 16**. See §6.5.

---

## 0. Where this started: the old outcome-only reward (`reward_SPRec.py`)

Before any reasoning reward existed, REC-R1 trained on `../reward_SPRec.py`
alone — a small, single-purpose function with **no tunable hyperparameters
at all**:

```
extract_solution(solution_str)          # regex: last <answer>...</answer>, or None
similarity_match(...):
    title = extract_solution(...)
    if not title: return 0.0

    embed the predicted title            (paraphrase-MiniLM-L3-v2)
    load precomputed catalog embeddings + name2id     (per data_source)
    rank = argsort( L2 distance from predicted embedding to every catalog item )
    rankId = position of ground_truth["target"] in that ranking

    match = 1.0    if rankId == 1
          = 0.8    if rankId <= 5
          = 0.5    if rankId <= 10
          = 0.1    if rankId <= 100
          = 0.001  if rankId <= 500
          = 0.0    otherwise

    if more than one <answer> tag: match = match / 4   (or -0.5 if that's 0)   # anti-spam
    return match
```

Characteristics worth naming explicitly, since everything else in this doc is
a delta from this baseline:

- **No env vars, no config surface.** Every threshold (`5, 10, 100, 500`) and
  every payout (`0.8, 0.5, 0.1, 0.001`) is a numeric literal in the source.
  Changing the reward shape means editing the file, not setting a variable —
  the opposite end of the spectrum from the `RTHINK_*`-knob versions in §2–§3.
- **Outcome-only, trajectory-blind.** It reads only the final `<answer>`
  block. Two rollouts that land the same title via wildly different (or no)
  reasoning get identical reward — this blindness to *process* is precisely
  the gap the reasoning reward was created to fill.
- **Stateless, single-pass.** One embedding call, one distance computation,
  no memory across samples (the SentenceTransformer + embedding table are
  reloaded per call — an inefficiency noted but never fixed by any later
  version).
- **Still the ground truth for `R_answer`.** Every later version (`v2`–`v6`
  here, and `v7` in `reward_retrieval/`) calls this file's `compute_score`
  **unmodified** and layers shaping on top of it; it is never edited, which
  is what keeps the baseline reproducible across every experiment in this doc.

### 0.1 What the new goal is

`reward_SPRec.py` only answers "did the final title land near the target's
embedding" — it has no mechanism to reward *how the model got there*. The
reasoning-reward goal, unchanged in spirit since v1, is to layer a second
signal on top of that outcome score:

```
R_total = R_answer  +  (shaping derived from R_think)
```

so the policy is pushed toward good agentic *behavior along the way* —
searching when it should, grounding its final pick in what it actually
retrieved, synthesizing evidence instead of parroting or looping — rather
than only toward a lucky final embedding match. In practice this goal has
run through two phases:

- **v1–v6 (this track, §2–§3):** reward *reasoning quality* directly (tool
  use, grounding, synthesis vs. self-repetition), on the hypothesis that
  reasoning quality was the binding constraint on held-out accuracy.
- **Now (post-diagnosis, §4–§6):** the diagnostics converged on retrieval
  coverage, not reasoning quality, being the actual constraint, so the
  *immediate* effort moved to a sibling reward that scores retrieval itself
  (`reward_retrieval/v7`). This track's original goal is **paused, not
  abandoned** — once retrieval is addressed, §6 queues resuming it pointed at
  **selection** (`P(hit | GT in docs)`), the "how it got there" piece that
  fixing retrieval alone still won't capture.

---

## 1. What this reward is for

REC-R1 trains an LLM agent (GRPO) to recommend items via an agentic
search → reason → answer loop. The **outcome reward** `R_answer` (unchanged
across every iteration, `../reward_SPRec.py`) only scores the final
`<answer>` — it says nothing about *how* the model got there. The
**reasoning reward** `R_think` is a *process* signal layered on top, meant to
reward good agentic behavior (searching, grounding claims in retrieved
evidence, synthesizing rather than parroting) so the policy learns *useful*
reasoning, not just a lucky final guess:

```
R_total = R_answer  +  (shaping derived from R_think)
```

Wiring: `verl/utils/reward_score/__init__.py::default_compute_score` routes
any `amazon`/`goodreads`/`movie` example with `USE_RTHINK=1` (or legacy
`USE_REWARD_B=1`) to `reward_reasoning.compute_score`, whose dispatcher
(`__init__.py`) reads `RTHINK_MODE` and picks a version's `compute_score`.
`RTHINK_MODE` unset falls back to `v3`; the training launcher
(`run_in_container_rthink.sh`) exports `RTHINK_MODE=${RTHINK_MODE:-v6}`, so
**v6 is the operational default** in practice.

## 2. The general math pattern, and the current (v6) reward in full

Every version shares the same skeleton — an outcome term plus a **capped,
asymmetric shaping term** derived from process signals:

```
shaping   = clip( SCALE · R_think_raw , −CAP , +CAP )       # CAP < smallest answer-tier gap
applied   = max(0, shaping)   if R_answer ≥ 0.5              # LongPAS: never punish a correct, well-formed answer
          = shaping           otherwise
R_total   = r_outcome  +  applied  −  (format_penalty + length_penalty)
```

`CAP` is always kept below the smallest gap between answer tiers so shaping
can only break ties *within* a correctness band, never reorder a wrong
answer above a right one. What changes version to version is (a) how
`r_outcome` is computed from the answer's embedding rank, and (b) what
`R_think_raw` rewards.

**The current default, v6** (`v6/orchestrator.py`) makes `r_outcome` a
*faithful* surrogate of the held-out eval metric itself:

```
r_outcome = 1 / log2(rankId + 1)                          if rankId ≤ K       # = eval.py's NDCG@K term
r_outcome = TAIL_W · (1 − log(rankId)/log(N)) ** TAIL_P    if rankId > K       # weak, steep "get warmer" tail
r_outcome = 0                                              if MALFORMED        # format-integrity gate
```

with `K=10`, `TAIL_W=0.10`, `TAIL_P=4.0`, `N=13,079` (catalog size).
`rankId` is the 1-based position of the ground-truth title in the catalog,
sorted by cosine distance from the predicted title's embedding
(`paraphrase-MiniLM-L3-v2`) — the exact same computation `eval.py` uses for
held-out HR/NDCG, so the training reward and the eval metric read the same
number (Part 5 of `REWARD_REASONING_ANALYSIS.md` has the full derivation, a
worked 3-rollout GRPO-group example, and every env knob).

`R_think_raw` (v3's four components, kept verbatim through v6, computed by
`v3/reasoning.py::compute_process_components`):

```
R_think_raw = 0.30·tool_use + 0.40·grounding + 0.30·synthesis − 0.50·self_rep
```

- **`tool_use`** — issued a search, got results, reasoned after (not just before).
- **`grounding`** — max cosine similarity of the answer to retrieved/history evidence.
- **`synthesis`** — did the retrieved evidence actually change the reasoning (vs. copy or ramble).
- **`self_rep`** — the one penalty: intra-`<think>` looping/repetition.

Full per-version formulas (v1–v5, including the v4 dense-rank reward and the
v5 format gate/length cap it built on) are in Part 2 of
`REWARD_REASONING_ANALYSIS.md`; only the current v6 math is repeated in full
here.

> **All of v2–v6 trained at `rollout.n=1`** (a bug in the original code, since
> confirmed by the PI and fixed to **8**). At `n=1` GRPO has no group-relative
> baseline, so the whole per-version history below is confounded — see §6.5.

## 3. What has been done — version history

| Ver | One-line change | Result | Detail |
|---|---|---|---|
| **v1** | `α·info_gain − β·redundancy + γ·explore` | Tier-1/CF-domain bug fired the redundancy penalty on ~everything → flat reward. | superseded |
| **v2** | Same, bug fixed, weaker β | Good terms stayed dead; penalty ⊥ correctness → **lost to baseline** (held-out `r_answer` went negative). | [`v2/README.md`](v2/README.md) |
| **v3** | Process terms `tool_use+grounding+synthesis−self_rep`, stateless, capped shaping | Live, two-sided, benign-shaping gradient — but **ties, doesn't beat, baseline**, and collapses (entropy → 0.008) after ~step 300 if run long. | [`v3/README.md`](v3/README.md) |
| **v4** | **Dense** rank-based outcome reward `(1−log r/log N)^p` | Fixed the ~87%-all-wrong dead-gradient problem and v3's instability, but **still ≤ baseline** on held-out; opened a multi-`<answer>` hack (73.9% of samples). | [`v4/README.md`](v4/README.md) |
| **v5** | v4 + format-integrity gate + length cap + `p=2` | Killed the hack (`format_ok`→1.0) and verbosity cleanly, train `r_ans`→+0.38 — but held-out **still ≈ baseline**, worst run on HR@1/NDCG@5 at step 300; 3 independent re-decodes of the *same* checkpoint swing HR@1 0.000↔0.005, wider than any baseline↔v5 gap. | [`v5/README.md`](v5/README.md) |
| **v6** | Outcome reward reshaped to be the eval metric itself (NDCG@K inside top-K, weak tail outside) | Fixes v4/v5's "semantic mush" (mid-rank credit HR never scores). GPU-run: HR@1/HR@5/NDCG@5 mean below baseline but still inside single-seed decode noise. **Launcher's operational default.** | [`v6/README.md`](v6/README.md) |

**Important caveat on this whole table:** every row was trained at `rollout.n=1`
(§6.5), so GRPO's group-relative baseline was inoperative throughout. "v2 lost,
v3–v6 tied baseline" is **confounded** — it is not valid evidence that reasoning
shaping doesn't help; it is evidence produced with the optimizer's core mechanism
switched off. Full head-to-head numbers, the GRPO-variance argument, and the
citation list are in
[`../../../../../REWARD_REASONING_ANALYSIS.md`](../../../../../REWARD_REASONING_ANALYSIS.md).

## 4. The diagnosis that ended linear (v1→v6) iteration

After v6, three diagnostics (now in [`diagnostics/`](diagnostics/)) converged
on one story: **the system is retrieval-bound, not reward-bound.**

| Diagnostic | Finding | Rules out |
|---|---|---|
| [`diagnostics/train_test_greedy_diagnostic.md`](diagnostics/train_test_greedy_diagnostic.md) | train-greedy HR@1 reaches 0.20–0.39, vs test 0.002–0.005 | a representation ceiling — the model *can* learn the task |
| [`diagnostics/offline_coverage_diagnostic.md`](diagnostics/offline_coverage_diagnostic.md) | answer exists in the corpus 98.1% of the time, but a realistic history-shaped query only reaches it ~26% @ top-100; GT-in-retrieved-docs is 4.6% test vs 45–69% train | a reward-shape / hacking problem |
| [`diagnostics/corpus_analysis.md`](diagnostics/corpus_analysis.md) | plain-language walkthrough of *why*: the corpus stores raw per-user history, so a test user's continuation is only reachable by retrieving a *stranger's* record | — (explains the mechanism behind the coverage number above) |

The organizing frame that came out of this (see
[`../reward_retrieval/v7/ROADMAP_v7.md`](../reward_retrieval/v7/ROADMAP_v7.md) §1):

```
test HR  ≈  P(GT in retrieved docs)   ×   P(model picks GT | GT in docs)
              └─── COVERAGE ───┘             └──── SELECTION ────┘
              a RETRIEVAL problem            a REASONING problem
```

With coverage at ~5%, GRPO gets essentially zero gradient on the ~95% of test
prompts where the answer was never retrievable — no amount of reward
reshaping on `R_think` can fix that. This is why the reasoning-reward track
(v1–v6, this document) paused and the next reward experiment pivoted to
scoring *retrieval itself*.

## 5. The pivot: retrieval reward moved to its own package

An earlier `ROADMAP_v7.md` (the CF-corpus plan, since superseded) originally
proposed fixing coverage by restructuring the retrieval corpus into
user-independent CF-continuation documents (`verl_R1/cf_corpus/`), then
resuming reasoning-reward work once coverage was fixed. **The PI overrode that
plan**: don't touch the corpus — the observed 4.6% coverage is correct, not a
bug — and build a reward that scores retrieval quality directly on the
*unmodified* corpus instead.

That reward (`v7`) scores *retrieval behavior*, not reasoning process, so it
now lives in the sibling package
[`../reward_retrieval/`](../reward_retrieval/retrieval_reward_design.md)
rather than as a `v2`–`v6`-style subpackage here. `reward_reasoning/__init__.py`
still routes `RTHINK_MODE=v7` there for backward compatibility.

The current [`ROADMAP_v7.md`](../reward_retrieval/v7/ROADMAP_v7.md) (now living
in `reward_retrieval/v7/`) is the **corpus-fixed RRCM journal-extension plan**
that replaced the CF-corpus roadmap. It keeps the corpus frozen and recasts the
whole effort as a reward-only contribution over **two axes** — a retrieval-policy
axis (owned by the sibling doc's §9) and a **reasoning/selection axis (owned by
this doc, §6)**. So this track is no longer merely "paused pending coverage": it
now has a concrete *corpus-fixed* path via `r_infer`/`r_select`. `r_infer` uses no
ground truth as input (GT only scores), so it does not depend on the corpus
question ever reopening.

## 6. Phased work plan (RRCM-v7 roadmap): the selection axis

This section turns [`../reward_retrieval/v7/ROADMAP_v7.md`](../reward_retrieval/v7/ROADMAP_v7.md)
into concrete, staged work for the **reasoning/selection axis** — the second
factor of the roadmap's decomposition:

```
test HR ≈ P(GT in retrieved docs) × P(model picks GT | GT in docs)
          └── COVERAGE ~5% (corpus-fixed, off-limits) ──┘   └── SELECTION (this axis's lever) ──┘
```

Coverage is a corpus property and stays frozen. **Selection** is where a process
reward has real headroom: the model already picks GT correctly **~54% on train**
but only **~6% on test** — a generalization gap, not a capability gap (§4's
greedy diagnostic). This axis targets that gap with two terms, `r_select` (the
better-evidenced, on-track term) and `r_infer` (the PI's infer-from-similar-items
idea, which **failed its correlation gate as first defined** and needs redesign),
as **capped LongPAS shaping on the frozen `InTop@n`** outcome reward. The sibling
[retrieval doc §9](../reward_retrieval/retrieval_reward_design.md) owns the
retrieval-policy axis and the shared evaluation protocol.

### Milestone ↔ doc map

| Roadmap milestone | Reasoning doc (this §6) | Retrieval doc (§9) |
|---|---|---|
| M0 Audit A (selection headroom) | **owns** — done, gated (§6.1) | — |
| M0 Audit B (per-term correlation) | `r_infer` correlation | `r_retqual`/`r_covgain` |
| M1 implement lead term | `r_retqual` (retrieval) leads; reasoning follows | `r_retqual` (done) |
| M2 A/B vs baseline | `r_select` / redesigned `r_infer` | v7 first GPU run |
| M3 add second axis | `r_select` | `r_memtype`, `r_when` |
| M4 full sweep + RQ2/RQ3 | shared | owns eval protocol |

### 6.1 Milestone 0 — Audit A (run + statistically resolved; qualified positive)

Audit A (`../reward_retrieval/audits/audit_a_selection.py`) is the roadmap's
**decisive gate**: on the winnable set (GT *is* in the retrieved docs), is GT
distinguishable from the other candidates by a **transferable** feature
(frequency / co-occurrence with the user's history / CF-adjacency) rather than
only by the train-only positional cue the model overfits?

**First pass (single v6@300 checkpoint) was underpowered and misleading.** On the
realistic test view (UNSEEN items only, n=**25**) the point estimates put
`random` at 16% — *above* the transferable features — which read as "≈ random."
That was small-sample noise. The gate was resolved on **2026-07-14** by (a)
wiring a **paired bootstrap** into the script — each feature's hit@1 vs the
per-sample uniform-chance floor `1/#candidates`, 95% CI excluding 0 ⇒ beats
chance — and (b) **pooling 8 held-out test decodes** (v6/v5/baseline/v4, greedy +
temp-1 seeds) to grow the winnable set, with a **prompt-clustered** bootstrap so
8 decodes of the same test prompt don't fake 8× independence. Results (`hit@1`;
lift-over-chance 95% CI in brackets; `sig` = CI excludes 0), in
`audits/audit_a_ci_2026-07-14.json`:

| view (pooled) | set | freq | cooc_hist | cooc_tail3 | adjacency | pos_first *(control)* | random *(floor)* |
|---|---|---|---|---|---|---|---|
| **ALL items** | n=246 / **76 prompts** | 19.4% *** | **22.4% ***** | 18.2% *** | 11.8% *** | 9.3% (ns) | 4.1% |
| **UNSEEN only** | n=191 / **63 prompts** | 13.1% (ns) | 14.0% (ns) | 12.7% (ns) | 15.6% (ns) | 13.1% (ns) | 8.9% |

(*** = clustered 95% CI lift over chance excludes 0; ns = includes 0. `cooc_hist`
lift on UNSEEN was `[−0.001, +0.088]` — borderline. TRAIN-UNSEEN, for contrast,
is decisively feature-driven: adjacency hit@1 95.4% at ~3.4 candidates.)

**Verdict — qualified positive, held back by a coverage-starved test set.**

- In the **ALL-items view** the transferable co-occurrence/frequency features
  **robustly beat chance** (survive prompt clustering); the positional control
  does **not**. Co-occurrence genuinely separates GT among retrieved candidates.
- In the **realistic UNSEEN view** every transferable feature's point estimate
  sits above the chance floor (13–16% vs 9%) and above the positional control,
  and `cooc_hist`/`cooc_tail3` are **borderline** — but the clustered 95% CI
  still **includes 0**. The binding limit is now visible and structural: only
  **~63 distinct held-out prompts** have GT in unseen retrieved docs at all
  (coverage ≈ 3–5%), so the winnable *test* set is too small to certify the
  effect at 95%, even pooled.
- **The key qualitative result is robust and favorable:** the positional cue
  never out-separates the transferable features, so whatever selection signal
  exists is *transferable*, not the train-only positional artifact that would
  have doomed the axis (roadmap §8's failure mode). This rules out the worst case.

**Gating follow-ups — status:**

1. ✅ **Bootstrap 95% CIs on per-feature hit@1** — done (`paired_boot_ci`,
   `cluster_paired_boot_ci` in the script).
2. ✅ **Pool checkpoints/seeds** — done (n 25→191 on UNSEEN via 8 decodes); this
   is as far as pooling *existing* logs reaches. Growing it further needs more
   test decodes (cheap GPU), which would raise the distinct-winnable-prompt count
   toward the fraction of test prompts for which GT is ever retrievable.
3. ✅ **Audit B for `r_infer` (the decisive gate) — RUN, and `r_infer` FAILS it.**
   Built as `audits/correlation_audit.py` (named by function to avoid clashing with
   the train-vs-test greedy diagnostic that notes informally call "Audit B"; that
   one is already run and decisive). Log-only, single-thread CPU, no corpus, no GPU;
   run in the `retriever` conda env. It scores each candidate term on the **full**
   pooled sample and correlates it with the `InTop@n` correctness label
   (reproduced exactly from `reward_SPRec.py`: raw mean-pool embedding, `cdist` L2,
   rank of GT id). Run command:

   ```bash
   PYTHONPATH=verl/utils/reward_score/reward_retrieval/audits OMP_NUM_THREADS=1 \
   MKL_NUM_THREADS=1 <retriever-python> \
     verl/utils/reward_score/reward_retrieval/audits/correlation_audit.py \
     --preds <6 held-out test_predictions.json: v6/300, v6/seed1, v5/350, v5/300, \
              baseline/500, v4/200> \
     --data-source amazon --label pooled-heldout-6decode
   ```

   **Result (6000 pooled held-out rollouts, 2026-07-15;
   `audits/correlation_audit_2026-07-15.jsonl`):** `r_infer` = cos(answer, centroid
   of retrieved CF-neighbour items) correlates with `InTop@10` hit at pearson
   **+0.023** (spearman +0.027, AUC 0.61, Q4/Q1 hit 0.006/0.002) — **FAIL** on all
   three labels (exactHR@1 +0.008, InTop@1 +0.017, InTop@10 +0.023). This is the
   **v2 tripwire firing**: a term this weak (≪ the +0.10 bar; recall v2's redundancy
   term was +0.10 and still *hurt*) does not track correctness and must **not** train
   as-is. The negative is honest and important — grounding the answer in the
   retrieved CF centroid, as currently defined, does not predict a hit on this
   frozen substrate. (Contrast the retrieval-axis gate `r_retqual`, which **PASSES**
   the same audit at pearson +0.112 / AUC 0.85 — see retrieval doc §9.2 P0.)

**Decision.** Audit A no longer *blocks* the axis (the transferable-vs-positional
result is favorable and robust), **but Audit B is now the binding negative for
`r_infer` as specified.** The centroid-consistency formulation does not correlate
with correctness, so it does not earn GPU as-is. Two forks, in priority order:
(a) **redesign `r_infer` before it trains** — the co-occurrence–weighted variants
that Audit A found *transferable* (`freq`/`cooc_hist`/`cooc_tail3` all beat chance in
the ALL-view clustered CI) are the obvious next candidates to route through
`correlation_audit.py`; a term only trains once it clears the +0.10 bar there; and
(b) if no `r_infer`/`r_select` variant clears Audit B, **re-scope with the PI toward
the retrieval-policy axis** (which *did* clear, §9.2 — and which the PI endorsed)
**+ the evaluation-methodology contribution**, rather than forcing this axis. Treat
train-side selection-rate and Audit-B correlation as the early signals, never the
underpowered held-out winnable hit@1, exactly as roadmap §8 anticipates.

### 6.2 Phase table (selection axis)

| Phase | Work | Status | Gate to advance |
|---|---|---|---|
| **M0** | Audit A + its CI/Audit-B follow-ups (§6.1) | Audit A **resolved** (qualified positive: transferable > positional, robust in ALL-view/train, borderline on coverage-starved held-out UNSEEN-view); **Audit B RUN** (`correlation_audit.py`) | done — see per-term gates below |
| **P1** | **`r_infer`** — answer↔retrieved-CF-evidence consistency (roadmap §3.2) | Audit B **FAIL** as-defined (centroid-consistency pearson +0.023 vs `InTop@10`); **needs redesign before GPU** | a redesigned `r_infer` variant clears `corr(term, hit) > 0.10`; `SHAPING_OFF` reproduces baseline |
| **P2** | **`r_select`** — transferable, non-positional selection of GT on the winnable set (roadmap §3.2) | **not built — on the implement track**, justified by the grounding forensics (P(hit\|GT in docs) 54.5% train / 6% test), **not** gated on Audit B | operationalize the Audit-A transferable feature (`cooc_hist`/`freq`); ship with parse/position guards |
| **P3** | **v7a A/B** — RRCM-reward vs RRCM+reasoning term, Amazon first, ≥3 seeds, pre-decline ckpt | **not built** | **`rollout.n=8` on BOTH arms** (baseline re-run at n=8, not the old n=1 run); test selection ↑ and/or HR ↑ beyond the seed spread |
| **P4** | Full sweep across all 3 datasets + RQ2/RQ3 analysis (protocol in §9 of the retrieval doc) | **not built** | consistent lift over outcome-only RRCM across datasets |

### 6.3 Phase detail

**P1 — `r_infer` (the infer-from-similar-items idea; failed as first defined).**
Let `E` = the items appearing in the retrieved **collaborative** docs (the "users
who did X also did Y, Z" neighbors). Credit an answer that is *consistent with the
collaborative evidence for this user*: `r_infer = consistency(ŷ, E | user history)`
— e.g. similarity of `ŷ` to the CF-neighbor centroid, weighted by co-occurrence of
`E`-items with the user's history. Properties:

- **Uses no GT as input** — it scores grounding in *retrieved evidence*; GT enters
  only through the outcome `InTop@n` term, exactly as in RRCM. Clean, corpus-fixed.
- **Meant to provide gradient on dead prompts.** On the ~87% of prompts where no
  rollout hits GT, `InTop@n` is a flat 0 → zero GRPO advantage; `r_infer` is meant
  to give within-group variance there — **but this presupposes `rollout.n > 1` (a
  real group), which every past run lacked** (see §6.5 guardrails; now fixed to
  **n=8**). At n=1 there was no within-group variance for `r_infer` to create, so
  fixing the group size is a hard prerequisite to even testing it — **and it only
  helps if Audit B confirms evidence-consistency correlates with correctness**
  (§6.1). That proviso is what separates this from v2's failed penalty — **and, as
  run 2026-07-15, the plain centroid-consistency form does NOT clear it** (pearson
  +0.023 vs `InTop@10`, FAIL). So the version that earns GPU is a **redesigned**
  `r_infer` — e.g. weighting evidence items by their co-occurrence with the user's
  history (the `cooc_hist`/`freq` features Audit A found transferable), then
  re-running `correlation_audit.py` until it clears +0.10. The flat centroid variant
  is the negative result, not the shippable term.
- **Pushes copier → inferencer** — it credits a CF-supported pick even when the
  exact GT is *absent* from the docs, the only path to exceeding the coverage
  ceiling on a fixed corpus.
- **Guard against the v4-mush relapse (critical):** `r_infer` must credit
  closeness to the *retrieved* CF evidence, **not** closeness to GT in global
  catalog-embedding space. v4 rewarded the latter and taught "semantic mush"
  (land in a broad neighborhood, never commit) that didn't transfer. Keep it
  grounded strictly in the retrieved CF items; audit that it credits
  evidence-supported picks, not embedding-central hedges.

**P2 — `r_select` (targets the 6%→54% gap directly).** On the winnable subset,
credit selecting GT by a **transferable feature**, *not* by position — co-occurrence
(`cooc_hist`/`freq`) is the pick: Audit A's ALL-view clustered CI shows those beat
chance while the positional control never does (bounds in §6.1). **`r_select` is on
the implement track and is NOT gated on Audit B.** Its justification is the
already-established grounding forensics — `P(hit | GT in docs)` = 54.5% train vs 6%
test — which is direct evidence that a *transferable* selection signal exists and is
under-exploited on held-out data; that evidence stands on its own and does not need
the correlation gate that `r_infer` (a GT-free consistency term with no such prior)
does. Because it fires only on the ~5% winnable subset (sparse gradient), ride it
*alongside* the retrieval axis's dense `r_retqual`, not alone. Ship it with
parse/position anti-hacking guards; drop it only if training shows the positional
artifact re-emerges, not on an offline correlation threshold.

**P3 — v7a A/B experiment.** RRCM-reward vs RRCM+reasoning term, Amazon first, ≥3
decode seeds, **`rollout.n=8` on both arms** (the baseline re-run at n=8 — the old
n=1 baseline is void), pre-decline checkpoint (held-out declines after ~step 300 as
train reward climbs — the gap widens with optimization). Success = test selection
and/or HR rising beyond the seed spread. **P4** shares the retrieval doc's §9.4
protocol (all 3 datasets, RQ2 ablations, RQ3 behavior analysis) — cross-linked,
not duplicated.

### 6.4 Composition with the RRCM outcome reward

Blend, never replace — keep `InTop@n` as the answer signal (it *is* the
baseline), add the reasoning terms as capped, LongPAS-gated shaping, reusing the
exact v3–v6 machinery from §2:

```
R_total = R_rank(InTop@n) + clip( SCALE · (w_s·r_select + w_i·r_infer), ±CAP ) − parse_penalty
```

`CAP` must sit **under the 0.02 smallest-tier gap** (rank-100 hit vs miss) *or*
rely on the **LongPAS asymmetry** — positive-only shaping when the answer is a
hit, full two-sided shaping on misses (preferred: it lets shaping fully order the
all-miss prompts where the gradient is needed while never pulling a correct
answer down). With two axes live, cap the *sum* of retrieval + reasoning shaping.
A `SHAPING_OFF` toggle must reproduce the outcome-only RRCM curve exactly (the v4
`DENSE_ONLY` discipline).

### 6.5 Standing guardrails (roadmap §7 — do not re-earn these)

- **Compare `r_answer`/`InTop@n`, never the shaped `score`** — the shaped score
  isn't comparable to the outcome-only baseline.
- **≥3 decode seeds, pre-decline checkpoint**, before trusting any held-out
  number (one temp-1 decode swings HR@1 0.000↔0.005).
- **Keep `CAP` below the smallest tier gap**, or use LongPAS; a well-reasoned
  wrong pick must never outrank a correct one.
- **LongPAS never penalizes a correct, well-formed answer.**
- **`rollout.n = 8` — the group-size fix (PI-confirmed, 2026-07-16).** Every past
  run (baseline + v3–v6 + ablations) trained at `rollout.n=1`: the value fell
  through `search_multiturn_grpo → ppo_trainer → rollout.yaml (n:1)` and was never
  overridden. In this verl the `len==1` branch sets `mean=0,std=1` (not mean=score),
  so advantage `= raw reward` — **REINFORCE with no baseline, not literally zero** —
  but the group-relative mechanism is absent and the within-group-variance rationale
  for `r_infer` (§6.3) is **void at n=1**. **The PI confirmed this was a bug in the
  original code, that he fixed it to `n=8` in his own experiments, and that he never
  pushed the newer code — so obtain his version, and note the RRCM Table-1 baseline
  was itself produced at `n=8`.** Canonical `rollout.n=8`; escalation ladder if
  under-performing: **8 → 12 → 16**. Batch-divisibility pre-flight: `train_batch × n`
  must divide `ppo_mini_batch` (56×8=448 is **not** divisible by 40 — adjust
  `ppo_mini_batch`→64 or `train_batch`→60 before launch). `n=8` is GPU-safe (peak KV
  is n-independent, bounded by the sglang pool); only wall-clock rises. A valid A/B
  needs the **baseline arm re-run at `n=8`** — never compare against the old `n=1`
  baseline. See memory `grpo-rollout-n1-no-baseline`.
- **Every GT-free consistency term (`r_infer`, `r_retqual`, …) passes Audit B
  (correlation) before it trains** — that is the class the v2 tripwire watches.
  `r_select` is the one deliberate exception: it is justified by the grounding
  forensics (54.5%/6% `P(hit|GT in docs)`), not the correlation gate (§6.3 P2).
  Ablate one variable per run; never touch the corpus or the `InTop@n` weights
  (they define the baseline).

### 6.6 North-star metric

Test-set `P(hit | GT in docs)` (selection) rising toward the train-demonstrated
rate — **not** raw HR, which stays coverage-limited and noisy at the floor.
Selection rate on the winnable set is far less noisy than raw HR@1 and is the
right early signal (roadmap §1/§8).

## 7. Where the code and docs live

```
reward_reasoning/
├── __init__.py                  # dispatcher: RTHINK_MODE -> v2..v6 (v7 -> reward_retrieval.v7)
├── README.md                    # quick per-version index
├── reasoning_reward_design.md   # this file
├── diagnostics/                  # the 3 diagnostics that settled the pivot (§4)
├── diagnostics_scripts/          # scripts + result JSONs behind those diagnostics
├── v2/ … v6/                     # versioned implementations (§3)
└── (v7 + current ROADMAP_v7.md moved to ../reward_retrieval/v7/)
```

For the exhaustive per-run numbers, wandb run IDs, and citations behind every
design choice: [`../../../../../REWARD_REASONING_ANALYSIS.md`](../../../../../REWARD_REASONING_ANALYSIS.md).