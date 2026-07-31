# Retrieval-Quality Reward Design (v7)

The base document for the retrieval-reward track: what problem it solves,
the concrete math, the code it's grounded in, and what's done vs. still
open. Its sibling document is
[`../reward_reasoning/reasoning_reward_design.md`](../reward_reasoning/reasoning_reward_design.md),
which covers the earlier process/reasoning-reward track (v1–v6) that this
one superseded for the "reward what R_think does" slot.

> **Updated 2026-07-16** to fold in the PI's reply: canonical GRPO group size
> is **`n=8`** (not the `n=5` earlier drafts used); the `n=1` was a bug in the
> *original* code that the PI had already fixed to 8 in his own runs but never
> pushed — so **get his newer code**, and the RRCM Table-1 baseline was itself
> produced at `n=8` (the local `n=1` baseline was never valid). The PI endorsed
> this reward's design ("good design") and top-k widening as exploration, but is
> **explicitly uncertain whether pushing multi-turn querying helps** — so
> `r_covgain` stays deferred and is judged on the behavior it induces.

This is the **retrieval-policy half** of the reward: the second of the two axes
in the RRCM journal-extension roadmap
[`v7/ROADMAP_v7.md`](v7/ROADMAP_v7.md). That roadmap superseded the earlier
CF-corpus-restructuring plan and **freezes the whole RRCM system** (corpus,
retriever, grounding, GRPO) so the only thing that moves is the reward: every
term here is **capped LongPAS shaping** layered on the frozen `InTop@n` outcome
reward, and the bar to beat is RRCM's outcome-only Table 1 (HR@5 0.0102,
NDCG@5 0.0073) with a *consistent, multi-seed* lift — not solving recommendation.
The full phased work plan for this axis is in §9; the sibling
[reasoning doc](../reward_reasoning/reasoning_reward_design.md) §6 owns the
reasoning/selection axis.

It implements the PI's literal proposal, given after they overrode the CF-corpus
diagnosis. Stripped down, the PI's guidance was two things:

1. **Don't change the corpus** — it's already right. Leave the ground-truth
   item out of the retrieval corpus. The observed ~4.6% GT-in-docs rate on
   held-out (test) data is *correct*, not a bug to engineer away.
2. **Build a reward for retrieval, not reasoning.** Every time the model
   retrieves, score how similar the retrieved items are to the correct
   answer. Reward good retrieval, penalize bad retrieval. That teaches the
   model *when* to retrieve and *how* to write a good query.

The PI has since reviewed this design and endorsed it ("good design. If
retrieving similar items is the most important thing to improve the performance,
I would expect this reward design will work"), with one flagged uncertainty:
he is "not exactly sure whether pushing multi-turn querying will help" — which
is exactly why `r_covgain` is deferred and watched (§9.2), not trained blind.

This document is the concrete design; the implementation is in `v7/`.

## Status: done vs. planned

**Done:**
- `v7/` implemented — `r_retqual` (per-turn retrieval-quality shaping,
  cosine similarity of best-matching retrieved doc to GT) and `r_covgain`
  (credits only a later retrieval that beats the running-best similarity),
  composed as capped shaping on top of v6's outcome reward, format gate, and
  length discipline (§4–§6 below).
- Verified against the running code (not assumed) exactly what a training
  rollout's `solution_str` contains — Qwen3-native tool calling,
  `<tool_call>`/`<tool_response>` tags, not the `<search>`/`<info>` fallback
  (§2).
- `r_retqual` **PASSED Audit B** (pearson +0.112, AUC 0.85) — the retrieval axis
  is green-lit for its first GPU run (§9.2 P0).
- **M2 / Phase-1 offline coverage sweep DONE (2026-07-20):** `r_covgain` retired
  to `W=0` permanently (later-turn marginal coverage = 0.000, §4); `r_retqual`
  headroom quantified (model queries reach GT @20 = 0.111 vs a trivial tail5
  heuristic's 0.197 — ~2× room, all inside the frozen corpus, §4b); top-k 3→20
  carved out as a separate single-variable experiment (~0.7% HR@1, §8).
- **M1 baseline at `n=8` — trained, comparison step LOCKED at 200.** The first
  attempt died at SLURM wall-time at step 100; resumed as job `848159` from
  `global_step_100` (verified n=8, outcome-only `reward_SPRec` routing) and died
  again at wall-time having reached step 220, with a checkpoint at
  `global_step_200`. **The ≥300 comparison point in earlier drafts is superseded:
  every arm is read at step 200.** Eval decode is job `854449`.
- **M3 / P1 — READ 2026-07-24: retrieval COLLAPSED ❌ (SLURM `854501`).** v7 read at
  ckpt 200 vs baseline: retr% **~5%** vs ~99%, HR a tie inside decode noise. Two
  collapse pressures found + fixed: (1) `len_penalty` counted the retriever's docs
  and the model's post-retrieval reasoning → now docs-stripped + turns-aware budget
  (`RTHINK_LEN_PER_TURN`, `RTHINK_LEN_TURN_CAP`); (2) the below-τ junk penalty →
  removed, `r_retqual` now one-sided (advisor, see §4 formula note). Fix offline-
  audited (`audits/offline_reward_replay.py`). **Pending A2** (pre-collapse ckpt
  100/150 decodes, jobs 865934/865935) to lock the length increment + size the
  boost, then **v7b** (more steps, multi-node — 200 was undertrained). Full writeup:
  root `REWARD_REASONING_ANALYSIS.md` Part 8 / `TEST_OUTPUT.md`.

**Planned / open (§8 has the full list):**
- ~~**Set `rollout.n=8`**~~ **done** — both arms train at `n=8`; the old `n=1`
  and `n=5` runs remain invalid comparators and must never be quoted.
- ~~**Re-centre τ on the training-time `best_sim` median.**~~ **SUPERSEDED
  2026-07-25.** M3 showed the collapse was *not* a τ problem — a ±0.08-capped
  shaping term cannot out-bid a 0.2 length penalty at any τ. With `r_retqual` now
  one-sided (no below-τ penalty), τ is only a boost threshold; keep τ=0.20. The
  binding fixes were the length penalty + removing the junk penalty, not τ.
- **Run the `RTHINK_RETRIEVAL_ONLY=1` wiring ablation.** It is a P1 gate item
  ("must reproduce the outcome-only curve exactly"), still unrun, and needs its
  own SLURM window. It is also the parity-matched control for the JSON-fix arm
  asymmetry below.
- **Arm asymmetry to disclose:** the M1 baseline trained *before* the
  `verl/utils/tool_call_repair.py` fix landed (2026-07-20 18:55); M3 trains
  *with* it, so ~2% of rollouts retrieve where they previously retrieved
  nothing. Accepted deliberately over spending a window on a re-baseline, but it
  lands on the retrieval axis — exactly what v7 claims to improve — so it must
  be stated in any writeup, not buried.
- ~~Confirm turn/response pairing assumptions hold in practice~~ **done offline**
  in the M0 pass (§8): the `DOC` terminator bug that silently dropped the last
  doc of every response was found and fixed there; parser recall is now 100% and
  the reward and Audit-B parsers agree exactly.
- If `v7` alone doesn't move held-out HR, the roadmap's **reasoning/selection
  axis** (`r_infer`, `r_select`; `v7/ROADMAP_v7.md` §3) is the queued follow-on.
  Under the corpus-fixed roadmap this is *no longer* corpus-dependent — `r_infer`
  scores answer↔retrieved-evidence consistency with GT used only to score — so it
  does not wait on the PI reopening the corpus. It is tracked in the sibling
  [`../reward_reasoning/reasoning_reward_design.md`](../reward_reasoning/reasoning_reward_design.md)
  §6 since it targets reasoning/selection; the phased plan for *this*
  (retrieval-policy) axis is §9 below.

---

## 1. Hard constraint: no corpus interaction

`v7/retrieval.py` never opens `data/amazon_data/corpora.jsonl`,
`e5_Flat.index`, or any `cf_corpus/` artifact. The ground-truth answer
(`ground_truth["target"]`) is used **only to score** documents the model
already retrieved during its own rollout — recovered entirely from
`solution_str`, the decoded text of what the model produced (including the
tool's response, which becomes part of that text; see §2). The GT is never
written anywhere, never surfaces as model-visible input, and the corpus on
disk is untouched by this reward. This mirrors the guardrail already present
in `v7/ROADMAP_v7.md` ("the reward uses GT only to *score*, never as
model-visible input"). The current corpus-fixed roadmap shares that guardrail;
what the *earlier* CF-corpus roadmap would have changed — the corpus's
*reachability* — this design leaves untouched too.

## 2. What a training rollout's `solution_str` actually contains

This mattered enough to verify against the running code rather than assume it:

- **Retrieval is Qwen3-native tool calling** (`search_tool_config.yaml`:
  `type: native`), not the SearchR1-style `<search>`/`<info>` prompted
  format. The task prompt itself instructs the model to call
  `<tool_call>{"name": "search", "arguments": {"query_list": [...]}}</tool_call>`
  and that results come back inside
  `<tool_response>...</tool_response>` (`data/preprocess_amazon_recommendation_dataset.py`).
  Some reward code elsewhere in this package (`_extract_info_blocks`,
  `v3/reasoning.py`) still carries `<search>`/`<info>` as an OR-branch
  fallback — that branch is dead at runtime under the current config and
  model; `v7` parses `<tool_call>`/`<tool_response>` as primary.
- **The retrieved documents' raw text is present, not a summary.**
  `verl/tools/utils/search_r1_like_utils.py::_passages2string` builds, per
  query, `"Doc {i} (Title: {title})\n{body}\n\n"`, joins multiple queries'
  results with `"\n---\n"`, and the whole thing is wrapped as
  `json.dumps({"result": ...})` — that JSON string is what ends up inside
  `<tool_response>...</tool_response>` in the decoded rollout
  (`verl/workers/rollout/schemas.py::add_tool_response_messages` appends it
  as a `role="tool"` message, which the Qwen3 chat template renders inside
  those tags).
- **The reward function sees the entire decoded rollout**, all turns
  concatenated: `verl/workers/reward_manager/naive.py` decodes
  `data.batch["responses"]` into `solution_str` and passes it straight into
  `compute_score(...)`, so every `<tool_call>`/`<tool_response>` pair across
  every turn is available in one string.

`v7/retrieval.py`'s parsing (`TOOL_CALL`, `TOOL_RESPONSE`, `DOC` regexes)
follows this exactly: pair each `<tool_call>` with the `<tool_response>` that
follows it (by position, 1:1, truncating to the overlapping prefix if a
turn is malformed), `json.loads` the response body's `result` field
(tolerating malformed JSON the same way `coverage_sweep.py` does), and split
on the `"Doc N (Title: ...)"` headers to recover each document's text.

## 3. Embedding model: reuse, don't reload

The outcome reward (`reward_SPRec.py`) re-instantiates
`SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2')` on
every call — expensive, but that's existing, unrelated behavior. The
process-reward code already solved this with a module-level cache,
`reward_SPRec_reasoning._get_sentence_model()`, reused by `v3/reasoning.py`.
`v7/retrieval.py` imports and reuses that same cached instance rather than
loading a third copy.

Two similarity conventions exist in this codebase: `reward_SPRec.py` uses
raw L2 distance (`torch.cdist`) against a **precomputed corpus embedding
file** (off-limits here — that *is* a corpus artifact); `v3/reasoning.py`
uses L2-normalized cosine similarity computed fresh, which is what `v7`
follows, since it never has (and must never build) a cached corpus embedding
table.

## 4. The reward components

### `r_retqual` — per-turn retrieval quality

For each `(tool_call, tool_response)` turn with at least one recovered
document, compute the cosine similarity of the ground-truth answer to the
**best-matching** retrieved document (`max` over the turn's docs, not mean —
one good hit in a batch of 3 is what "good retrieval" means here, matching
how the outcome reward already only needs the GT to appear *once*).

> **⚠️ SUPERSEDED 2026-07-25 (advisor Shijun Li) — `r_retqual` is now ONE-SIDED.**
> The below-τ penalty branch shown below is **removed**: a below-τ retrieval scores
> **0, never negative**. The current code (`v7/retrieval.py::_retqual`) is:
> ```
> r_retqual(sim) = (sim − τ) / (1 − τ)   if sim ≥ τ
>                = 0                       if sim <  τ
> ```
> Reason: this term scores only the *collaborative* axis (cosine to the GT next
> item), but retrieval also serves legitimate item-*attribute* lookup, which
> scores low — penalizing it would suppress attribute retrieval. `RTHINK_RETQUAL_FLOOR`
> is now **INERT** (kept for back-compat). This also removes one of the two
> retrieval-collapse pressures found in M3 (the other was the length penalty —
> see the ROADMAP 07-25 update + `REWARD_REASONING_ANALYSIS.md` Part 8). The
> two-sided text below is retained only as design history.

```
sim_t = max_i cos( emb(doc_i), emb(GT) )        for turn t

r_retqual(sim) = (sim − τ) / (1 − τ)             if sim ≥ τ
               = −floor · (τ − sim) / τ          if sim <  τ    # REMOVED 2026-07-25 -> 0

retqual_agg = mean over turns with ≥1 doc of r_retqual(sim_t)
```

`τ` (`RTHINK_RETQUAL_TAU`, default 0.30) is the neutral threshold: above it,
retrieval is rewarded up to +1 at a perfect match; below it, retrieval is
~~penalized down to `−floor` at similarity 0~~ **now scored 0 (one-sided; see the
superseded note above)**. `floor` (`RTHINK_RETQUAL_FLOOR`, default 0.25) ~~damps
the downside~~ **is inert** — the below-τ penalty it damped no longer exists. A
rollout that never retrieves anything gets `retqual_agg = 0.0` (neutral, not
rewarded, not penalized) — abstention is not scored by this term; whether
retrieval was *needed* is already reflected in `r_answer`.

### 4a. τ calibration — DONE offline (2026-07-16), and the shipped default was harmful

The original text here said τ=0.30 and floor=0.25 were "starting points, not calibrated values"
to be checked "once a run produces real cosine-similarity distributions." **That check has now
been done offline instead — no GPU needed** — by scoring 400 real saved rollouts through
`compute_retrieval_components`. It should have been done before shipping the default:

```
best_sim over 400 real rollouts:  mean 0.217   median 0.203   p75 0.291   p90 0.380
                                  min -0.104   max 0.684

  τ      %rewarded   mean retqual
  0.30      23.5%       -0.056     <- SHIPPED DEFAULT: abstain (0.0) BEATS retrieval
  0.25      35.8%       -0.020        abstain still beats retrieval
  0.22      44.5%       +0.004        break-even
  0.20      51.5%       +0.022     <- NOW THE DEFAULT (τ = the measured median)
  0.15      68.5%       +0.068
```

**τ=0.30 sat at roughly the 77th percentile of the actual distribution**, so it penalized 3 of
every 4 retrievals, and mean `retqual` was **negative (−0.056)**. Since a rollout that never
retrieves scores exactly **0.0**, that inverts the term's intent:

> GRPO centres by group mean, so a *uniform* negative offset cancels. What does **not** cancel is
> the **within-group** contrast between an abstaining rollout (0.0) and a retrieving one (<0).
> With τ=0.30, GRPO would have actively pushed the policy toward **never retrieving** — the exact
> retrieval-collapse mode `RTHINK_RETQUAL_FLOOR` exists to prevent (§9.5), in a system where
> RRCM's own RQ3 reports retrieval count already declining over training and ~10% of baseline
> rollouts already abstain.

**This does not invalidate Audit B.** `r_retqual` is monotone in `best_sim`, so the **AUC 0.85 is
identical for any τ**, and raw `best_sim` alone scored +0.102 vs +0.112 shaped. τ is genuinely a
calibration knob, not the signal — unlike the `DOC` parser bug (§2), which changed *which
documents were seen at all*.

**Ongoing:** τ = the median is self-referential ("beat the current median retrieval"). If the
policy's `best_sim` distribution shifts as it learns, the median moves — so re-check τ against
the training-time `best_sim` log rather than assuming 0.20 is permanent. `floor = 0.25` remains
uncalibrated, but it only scales the downside and is now much less load-bearing.

### 4b. Offline headroom for `r_retqual` (M2 coverage sweep, 2026-07-20)

The same M2 sweep that retired `r_covgain` also **quantifies the room `r_retqual`
has to help**, which is the mechanistic case for the P1 run (beyond the Audit-B
correlation). Replaying held-out rollout queries against the frozen index:

| query source | GT-coverage @3 | @20 | @100 |
|---|---|---|---|
| **model's own queries** | 0.039 | **0.111** | 0.229 |
| history string (raw prompt) | 0.052 | 0.120 | 0.258 |
| **tail5_items** (query each of last 5 titles) | 0.072 | **0.197** | 0.368 |
| GT-as-query (oracle) | 0.894 | 0.999 | 1.000 |

Two facts matter. (1) **The model's learned queries retrieve *worse* than a
trivial heuristic** — one query per recent title (`tail5_items`) reaches GT at
**@20 = 0.197 vs the model's 0.111** (~1.8×). The query policy is leaving
reachable evidence on the table. (2) **The ceiling is a query problem, not a
corpus problem** — GT-as-query reaches GT 89% @1 and coverage@∞ is 0.981 (the
target exists in the corpus almost always). So a reward that pushes queries
toward GT-similar retrieval — exactly what `r_retqual` does — has a concrete
~2× coverage target to move toward, entirely inside the frozen corpus the PI
mandated. This is the strongest offline evidence that `r_retqual` can lift the
`P(GT in docs)` coverage factor without touching the corpus.

### `r_covgain` — reward multi-turn retrieval that actually helps

```
covgain_agg = Σ_t  max(0, sim_t − running_best_before_t)
```

Only a turn whose best similarity **exceeds every earlier turn's** best
contributes, and only the positive delta. A second query that finds nothing
better than the first contributes 0 — it doesn't get punished (spam isn't
directly penalized by this term either, keeping the two components
orthogonal: `r_retqual` grades each turn on its own merits, `r_covgain`
grades only whether continuing to search paid off). This directly targets
the measured 99% single-query collapse: right now there's no reason for the
policy to issue a second query, so it (almost) never does; `r_covgain` gives
it one, without requiring the corpus to change.

**The PI is explicitly uncertain multi-turn helps** ("I'm not exactly sure
whether pushing multi-turn querying will help or not. You can keep an eye on
this when doing experiments"). So `r_covgain` is deferred (`W_COVGAIN=0` on the
first run), and the question is resolved two ways before it ever trains at
weight: (a) an **offline** Phase-1 measurement of whether more turns / higher
top-k materially raise union coverage on the frozen corpus; and (b) if trained
later, judged on the **multi-turn / retrieval-count rate it induces** (an
RQ3-style behavioral metric), not on Audit-B correctness correlation (§9.2).

> **RESOLVED — `W_COVGAIN=0` is now PERMANENT, not just deferred (M2 / Phase-1,
> 2026-07-20).** The offline measurement (a) has been run: the cross-turn
> cumulative-coverage variant added to `coverage_sweep.py`, replayed on 1,000
> baseline and 1,000 v6 held-out rollouts against the frozen e5+FAISS index. The
> marginal GT-coverage added by any turn beyond the first is **exactly 0.000 at
> both top-k 3 and top-k 20** — turns/rollout averages **0.97** (baseline dist:
> 878 one-turn, 43 two-turn, 2 with ≥3; v6 is fully collapsed to `max=1`). The
> behavior `r_covgain` rewards essentially does not occur, and where it does
> (the 45 multi-turn baseline rollouts) it buys zero extra reachability. This is
> the same signature as its Audit-B AUC 0.499 (§9.2), now confirmed structurally
> rather than inferred. **Decision: `r_covgain` is retired to `W=0` unless a
> future policy change makes multi-turn retrieval both frequent and coverage-
> additive** — re-run this measurement before ever giving it weight. Results:
> `reward_reasoning/diagnostics/coverage_sweep_m2_2026-07-20.json`.

### Combining into shaping

```
shaping_raw = W_RETQUAL · retqual_agg + W_COVGAIN · covgain_agg
applied     = clip( SCALE · shaping_raw, −CAP, +CAP )
```

Same discipline as every prior version: capped well below the smallest
answer-tier gap (`CAP = 0.08 < 0.1`), scaled down before clipping, and
subject to the LongPAS asymmetry (a correct, well-formed answer never has
shaping subtracted — only added).

## 5. Integration

`v7/orchestrator.py` implements the standard
`compute_score(solution_str, ground_truth, data_source, method, format_score,
score, extra_info) -> dict` contract every version since v3 implements
(confirmed identical across v3–v6's orchestrators); `reward_reasoning/__init__.py`
dispatches to it under `RTHINK_MODE=v7` (or `7`). Every other version (v2–v6)
is untouched — this is purely additive. Everything but the shaping term is
v6's outcome reward, format gate, and length discipline, copied verbatim (not
re-derived) so v7 is a clean, single-variable change from v6, in keeping with
this package's "one substantive change per version" convention.

Returned dict keys, on top of the v2–v6 baseline set
(`score`, `r_answer`, `r_dense`, `rank`, `r_think`, `format_ok`,
`format_penalty`, `len_penalty`): `retqual`, `covgain`, `n_turns`, `n_docs`,
`best_sim` — all logged automatically by the reward manager
(`naive.py` iterates every dict key into `reward_extra_info`), so they show
up in training metrics/dashboards for free.

## 6. Env vars (follows the existing `RTHINK_` convention)

| Var | Default | Meaning |
|---|---|---|
| `RTHINK_MODE=v7` \| `7` | — | selects this version |
| `RTHINK_RETQUAL_TAU` | **0.20** | neutral cosine-similarity threshold. **Calibrated 2026-07-16** from 0.30 → 0.20 = the measured `best_sim` median (§4a); 0.30 penalized 77% of retrievals and made abstention win inside a GRPO group. |
| `RTHINK_RETQUAL_FLOOR` | 0.25 | **INERT since 2026-07-25** — `r_retqual` is one-sided (no below-τ penalty), so this no longer scales anything. Kept for back-compat. |
| `RTHINK_W_RETQUAL` | 0.6 | weight of `retqual_agg` in the shaping sum |
| `RTHINK_W_COVGAIN` | **0.0** | weight of `covgain_agg` in the shaping sum. **Deferred → the launcher now defaults it to 0** (was 0.4): PI-uncertain, AUC 0.499, and real rollouts average ~1.0 turns so the term is near-constant zero by construction. |
| `RTHINK_SCALE` | 0.10 | overall shaping scale (kept from v6) |
| `RTHINK_CAP` | 0.08 | absolute shaping cap (kept from v6) |
| `RTHINK_RETRIEVAL_ONLY` | 0 | 1 → disable retrieval shaping (outcome-only ablation) |
| `RTHINK_LEN_PER_TURN` | **400** (= measured 396 words/turn OLS slope on clean pre-collapse A2 data, 2026-07-25) | **v7b:** word-budget added per **doc-returning** turn (`len` counts model-generated words only, `<tool_response>` stripped). Fixes the M3 length-driven collapse; the clean gate PASSES with it (retrieve−abstain −0.029→+0.023). See ROADMAP 07-25 + REWARD_REASONING_ANALYSIS §8.5. |
| `RTHINK_LEN_TURN_CAP` | 3 | max credited turns that add budget (anti-hack: a few turns can't unlock unlimited budget) |
| `RTHINK_HIT_K`, `RTHINK_TAIL_W`, `RTHINK_TAIL_P`, `RTHINK_REAL_BONUS`, `RTHINK_FORMAT_GATE`, `RTHINK_FORMAT_PENALTY`, `RTHINK_LEN_SOFT`, `RTHINK_LEN_W`, `RTHINK_LEN_CAP` | (v6 defaults) | outcome/format machinery unchanged from v6; **`RTHINK_LEN_SOFT` is now the *base* budget** for the turns-aware total above |

Note: an earlier roadmap draft sketched a different naming scheme (`RET_SPACE`,
`R_GROUND_W`, `R_RETQUAL_W`, ...) for the now-superseded CF-corpus-dependent
v7a/v7b design. That naming was never implemented; this implementation
follows the `RTHINK_` prefix every shipped version (v2–v6) actually uses.

> **GRPO group size is a training-config setting, not a reward env var.** It lives
> in `run_in_container_rthink.sh` / the trainer config as `rollout.n`, now set to
> **8** (PI-canonical; see §9.2 P1). It is called out here only so it isn't
> forgotten: the reward math above is inert unless `rollout.n > 1`.

## 7. What this deliberately does *not* do

- No corpus file is read or written, per the PI's directive.
- No change to `reward_SPRec.py` — the baseline stays reproducible.
- No change to the **reward code's** view of `topk` — this is a reward-only
  change. (Serving-side top-k widening 3→20 and the JSON/apostrophe fix are
  PI-endorsed *exploration*, but they live in `retrieval_launch.sh` /
  `search_tool_config.yaml` and are owned by the roadmap's Phase 1, not by this
  reward term.)
- `cf_corpus/` and the earlier v7a/v7b CF-corpus design are left in place but
  superseded (see `v7/ROADMAP_v7.md`), not deleted, in case the PI's position
  changes.

## 8. Open questions before trusting a training run — MOSTLY SETTLED (2026-07-16)

All of the below were resolved by **executing the reward offline on 3 saved decodes**
(3,058 tool calls / 9,348 retrieved docs). Three of the four turned up real defects that code
review had missed. Run this class of check before spending GPU.

- ✅ **`rollout.n` is 8; batch geometry fixed.** `train_batch=56`, `ppo_mini_batch=**56**`
  (448/56 = 8 clean). **The earlier advice to use `ppo_mini_batch=64` crashes** —
  `verl/workers/config/actor.py:153` raises when `train_batch < ppo_mini_batch`. And an
  indivisible combo does **not** crash: `dp_actor.py:388` splits with a non-strict chunker, so it
  silently trains a ragged final mini-batch with a mis-scaled loss. The launcher now pre-flights
  all three constraints. *(The "PI's newer code" prerequisite was dropped by the user.)*
- ✅ **`τ` is calibrated → 0.20** (§4a). The shipped 0.30 was not merely uncalibrated, it was
  **harmful**: it penalized 77% of retrievals, so abstention beat retrieval inside a GRPO group.
  `floor=0.25` remains uncalibrated but only scales the downside.
- ❗ **FIXED — the `DOC` regex silently dropped one doc per response, always the last.**
  It required a literal `)\n`, but the joined result is `.strip()`-ed so the final doc ends at `)`
  with no trailing newline. Measured recall **67.2%** (6,108 headers − 4,102 recovered = 2,006 =
  exactly the response count). Because `r_retqual` takes a **max** over a turn's docs, a third of
  the evidence was missing and `sim` could only be depressed — and since the Audit-B script parses
  at 100%, **the shipped term was not the term Audit B validated**. Terminator is now
  `\)(?:\n|\Z)`; recall 100%, parsers agree. Docs/turn 2.03 → **3.03**.
  *Do not "simplify" it to `\)\s*\n?`* — `\s*` eats the `\n\n\n` separator and each doc swallows
  its successor (recall collapses to 67%).
- ✅ **Turn↔response pairing measured: 1–4.6% of `<tool_call>`s are dropped as unpaired**, not the
  0.1–0.7% this section guessed — roughly 5× higher. This is *correct* behaviour (no response ⇒ no
  evidence to score), and the dominant cause is now known and largely fixed: **2.13% of tool calls
  emit unparseable JSON** (apostrophe-leading titles like `'70s Gold` colliding with the model's
  quoting), after which `sglang_rollout.py:960` discarded the call so the search never fired.
  `verl/utils/tool_call_repair.py` now repairs these from inside that existing `except` branch
  (55.4% recovered → **0.95% residual**, 0 regressions on 947 good calls).
- ✅ **Multi-query-per-turn is rare and not currently a gaming surface: 0.2–3.2% of turns.**
  The whole-turn scoring simplification stands. Revisit only if that rate climbs once `r_retqual`
  is actually training — a policy that learns to batch many queries per turn would hide junk
  queries behind one good one.
- 📌 **New — a stale installed verl** at `~/.local/lib/python3.12/site-packages/verl` has **no**
  `reward_reasoning` package. Training is safe (`python3 -m verl.trainer.main_ppo` with cwd=project
  dir resolves the working tree), but any script run as `python3 foo.py` imports the stale copy and
  silently returns a bare `0.0`. **Pin `PYTHONPATH=/work/11138/pranavbelligundu/vista/verl_R1`.**
- ✅ **Multi-turn coverage question CLOSED (M2 / Phase-1, 2026-07-20).** The offline union-coverage
  measurement the P1 plan promised is done (§4a's sibling result, §4 `r_covgain` box): later turns
  add **0.000** marginal GT-coverage at @3 and @20; turns/rollout ≈ 0.97. **`W_COVGAIN=0` is now
  permanent**, so P1 stays a genuine single-variable (`r_retqual`-only) run with no covgain question
  hanging over it. Results in `reward_reasoning/diagnostics/coverage_sweep_m2_2026-07-20.json`.
- ✅ **Top-k 3→20 is a SEPARATE experiment, not part of P1 (M2 quantified it).** Widening effective
  top-k raises model-query coverage @3 0.039 → @20 0.111 (~2.8×) — material, but selection-bounded:
  0.111 × ~6% P(hit|in-docs) ≈ **0.7% HR@1**. Worth its own single-variable arm (add `topk:` in
  `search_tool_config.yaml`), never folded into the `r_retqual` A/B. §7 already carves this out; M2
  puts numbers on it.

---

## 9. Phased work plan (RRCM-v7 roadmap): the retrieval-policy axis

This section turns [`v7/ROADMAP_v7.md`](v7/ROADMAP_v7.md) into concrete,
staged work for the **retrieval-policy axis** — the "when / what / how to
retrieve" decisions (roadmap §3). The sibling
[reasoning doc](../reward_reasoning/reasoning_reward_design.md) §6 owns the
reasoning/selection axis (`r_infer`, `r_select`). Every term below is **capped
LongPAS shaping on the frozen `InTop@n` outcome reward**; the corpus, retriever,
grounding, and GRPO substrate are never touched (roadmap §0). The honest ceiling
the whole plan lives under (roadmap §1):

```
test HR ≈ P(GT in retrieved docs) × P(model picks GT | GT in docs)
          └── COVERAGE ~5% (corpus-fixed, off-limits) ──┘   └── SELECTION (the reward's lever) ──┘
```

The retrieval-policy axis does **not** try to raise coverage by changing the
corpus. It improves the *retrieval action* — writing queries that surface useful
evidence, choosing the right memory, retrieving only when it helps — which is the
precondition for the reasoning axis's `r_infer` to have good evidence to reason
over. (Serving-side top-k widening, PI-endorsed as exploration, is a separate
roadmap Phase-1 lever measured offline, not a reward term.)

### 9.0 Milestone ↔ doc map

| Roadmap milestone | Reasoning doc (§6) | Retrieval doc (this §9) |
|---|---|---|
| M0 Audit A (selection headroom) | **owns** — done, gated | — |
| M0 Audit B (per-term correlation) | `r_infer` correlation | `r_retqual`/`r_covgain` correlation |
| M1 implement lead term | `r_infer` (redesign) | `r_retqual` + `r_covgain` — **done** (`v7/`) |
| M2 A/B vs baseline | `r_select` / `r_infer` run | v7 first GPU run |
| M3 add second axis | `r_select` | `r_memtype`, `r_when` |
| M4 full sweep + RQ2/RQ3 analysis | shared | **owns** the eval protocol (§9.4) |

### 9.1 Phase table (retrieval-policy axis)

| Phase | Work | Status | Gate to advance |
|---|---|---|---|
| **P0** | **Audit B** for `r_retqual`/`r_covgain`: score on logged rollouts, correlate with `InTop@n` hit | **BUILT + RUN** (`audits/correlation_audit.py`, 2026-07-15) — `r_retqual` **PASS** (pearson **+0.112**, AUC 0.85 vs `InTop@10`); PI endorsed the design | ✅ cleared the ~0.1 v2-tripwire bar; retrieval axis green-lit for the P1 GPU run |
| **P1** | **First GPU run of the shipped `v7`** (`r_retqual` only, `W_COVGAIN=0`): calibrate `τ`/`floor`, verify wiring | 🔄 **RUNNING — SLURM `854501`, launched 2026-07-21** (τ=0.20, `n=8`, experiment `...-gpu-rthink-v7-n8`, log `output_rthink_v7_n8.log`). The `RTHINK_RETRIEVAL_ONLY=1` wiring ablation in the gate column is **still unrun** and needs its own window. | **`rollout.n=8` (PI-canonical; every past run was `n=1` = no GRPO baseline — a bug the PI already fixed to 8 but never pushed; get his code, re-check batch divisibility, escalation 8→12→16)**; **baseline re-run at `n=8`** (reproduces the paper's real Table-1 baseline; never compare to the old `n=1` run); `RTHINK_RETRIEVAL_ONLY=1` reproduces the outcome-only curve exactly; `best_sim` distribution sane; parsing assumptions (§8) confirmed on real logs |
| **P2** | **`r_memtype`** — credit retrieving the memory type that actually surfaced GT-relevant evidence for the instance (roadmap §3.1) | **not built** | passes Audit B; memory-type parse validated; attributable in ablation |
| **P3** | **`r_when`** — small credit for not retrieving when correct without it / retrieving when needed (roadmap §3.1) | **not built** | passes Audit B; retrieval-rate stays healthy (no collapse) |
| **P4** | **Eval + ablations** for the policy axis (roadmap §6): RQ2-style per-term ablation, RQ3-style behavior analysis, ≥3 seeds, all 3 datasets | **not built** | consistent lift over outcome-only RRCM across datasets, beyond seed spread |

### 9.2 Phase detail

**P0 — Audit B (the gate before any GPU) — BUILT + RUN.** Shipped as
`audits/correlation_audit.py`, a sibling of `audits/audit_a_selection.py` (it
reuses that file's validated log parsing) — same discipline: reads only saved
rollout logs (`test_predictions.json`), never opens a corpus file, single-thread
CPU, one cached mean-pool MiniLM embedding pass. It scores each candidate term on
the logged rollout and correlates it with the rollout's `InTop@n` hit, where the
`InTop@n` label is reproduced exactly from `reward_SPRec.py` (raw mean-pool
embedding, `cdist` L2, rank of GT id) so the audit's correctness label *is* the
outcome reward's. Named by function, not "audit_b", to avoid colliding with the
train-vs-test greedy diagnostic that some notes informally call "Audit B".

Run command (in the `retriever` conda env; pooled over the held-out test decodes
for enough positives given ~0.5% coverage):

```bash
PYTHONPATH=verl/utils/reward_score/reward_retrieval/audits OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 <retriever-python> \
  verl/utils/reward_score/reward_retrieval/audits/correlation_audit.py \
  --preds <6 held-out test_predictions.json> \
  --data-source amazon --label pooled-heldout-6decode
```

**Result (6000 pooled held-out rollouts, 2026-07-15;
`audits/correlation_audit_2026-07-15.jsonl`):**

| term | vs `InTop@10` | AUC | Q4/Q1 hit | verdict |
|---|---|---|---|---|
| `r_retqual` (shaped) | pearson **+0.112** | 0.85 | 0.015 / 0.001 | **PASS** |
| `best_sim` (raw, unshaped) | pearson +0.102 | 0.85 | 0.015 / 0.001 | PASS |
| `r_covgain` | pearson −0.003 | **0.499** | 0.004 / 0.004 | FAIL (see note) |
| `r_infer` (reasoning-axis term) | pearson +0.023 | 0.61 | 0.006 / 0.002 | FAIL |

`r_retqual` **clears the v2-tripwire bar**: it separates hits from misses strongly
(AUC 0.85; top-quartile hit-rate 15× the bottom quartile), so the retrieval axis is
green-lit for the P1 GPU run — and the PI independently endorsed the design. Two
caveats worth carrying forward: (1) positives are sparse (25 `InTop@10` hits even
pooled — a hard coverage ceiling, so read the AUC with wide error bars and confirm
on the first real training curve); (2) the shaping adds almost nothing over the raw
`best_sim` (+0.112 vs +0.102) — the doc↔GT proximity carries the signal, so
`τ`/`floor` are calibration knobs, not the source of the correlation.

**`r_covgain` scored (2026-07-15) and FAILS Audit B — but is DEFERRED, not killed.**
It correlates with `InTop@10` at pearson −0.003, AUC **exactly 0.499** (chance), Q4/Q1
hit identical (0.004/0.004). That signature = the term is **near-constant zero across
almost all rollouts**, which is the known 99% single-query collapse: `r_covgain` only
fires when a *later* turn beats an earlier turn's best-sim, and multi-turn rollouts
barely exist in the logs. Audit B — a correlate-with-correctness test — is the **wrong
instrument for an exploration term whose target behavior is absent from the data it's
scored on** (unlike `r_infer`, which varies plenty and still doesn't predict, so its
FAIL is real). **This aligns with the PI's own stated uncertainty about multi-turn.
Decision:** P1's first GPU run sets `RTHINK_W_COVGAIN=0` (isolate the validated
`r_retqual`, one variable per run); the multi-turn question is answered first
**offline in Phase 1** (does higher top-k / more turns raise union coverage at all?);
`r_covgain` is only introduced as a *later*, deliberate second run and judged on the
**multi-turn / retrieval-count rate it induces** (an RQ3-style behavioral metric),
never on Audit-B correctness correlation. Re-run Audit B on `r_covgain` only *after* a
run where multi-turn behavior actually appears.

> **Offline Phase-1 measurement now DONE (M2, 2026-07-20) → `r_covgain` retired to
> `W=0` permanently.** The "does higher top-k / more turns raise union coverage"
> question is answered: cross-turn marginal coverage is **0.000** at @3 and @20
> (§4 `r_covgain` box, §8). So `r_covgain` does not get a second run "later" by
> default — it stays off unless a policy change first makes multi-turn retrieval
> both common and coverage-additive. P1 is unaffected (it was already `W_COVGAIN=0`).

**P1 — first GPU run of `v7` (the already-implemented lead term).** The reward
code (`r_retqual`, `r_covgain`) is done and offline-checked (`py_compile`,
dispatcher resolves `v7`); it has **never trained**. **Hard prerequisites (PI
reply, 2026-07-16):**
- **`rollout.n = 8`, and get the PI's newer code.** Every prior run trained at
  `rollout.n=1`, i.e. GRPO with no group-relative baseline (the `len==1` branch
  makes advantage = raw reward — see reasoning doc guardrails and memory
  `grpo-rollout-n1-no-baseline`). The PI confirmed this was a bug in the *original*
  code, that he fixed it to **8** in his own experiments, and that he "didn't push
  the newer code" — so obtain his version before building on the stale one.
  `r_retqual`'s gradient at `n=1` was raw-reward REINFORCE, not the group-relative
  signal it was designed for. Canonical `rollout.n=8`; escalation ladder if
  under-performing: **8 → 12 → 16**.
- **Batch divisibility:** `train_batch_size × n` must divide `ppo_mini_batch_size`
  cleanly. The 56/40 setup was tuned for `n=5` (280/40=7); **`n=8` gives 448, and
  40 does not divide it** (448/40=11.2) — adjust `ppo_mini_batch` (e.g. 64 →
  448/64=7) or `train_batch` (e.g. 60 → 480/40=12) before launch.
- **Re-baseline at `n=8`.** RRCM's Table-1 baseline was produced by the PI's `n=8`
  runs, so the A/B baseline must be re-run at `n=8` — this reproduces the paper's
  real baseline for the first time; never compare against the old `n=1` run.
- **`n=8` is GPU-safe** (peak KV is n-independent, bounded by the sglang pool);
  only wall-clock rises. Still watch the first ~15 steps given prior OOMs.

Beyond those gates, this run does four things:
(0) **set `RTHINK_W_COVGAIN=0`** so the only shaping term is the Audit-B-validated
`r_retqual` (covgain deferred; one variable per run);
(1) calibrate `RTHINK_RETQUAL_TAU`/`RTHINK_RETQUAL_FLOOR` against the logged
`best_sim` distribution — they are starting points, not fit values (§4, §8);
(2) prove the wiring adds nothing spurious — `RTHINK_RETRIEVAL_ONLY=1` must
reproduce the outcome-only RRCM curve exactly (the `SHAPING_OFF`/`DENSE_ONLY`
discipline); (3) confirm the parsing assumptions in §8 (1:1 turn↔response
pairing via logged `n_turns` vs. a raw `<tool_call>` count; multi-query-per-turn
scoring) hold on real logs. **Watch retrieval-rate** as a health metric from
this run onward.

**P2 — `r_memtype` (what to retrieve).** RRCM's ablations show the useful memory
differs by instance/dataset (CF matters most on Amazon, meta most on Goodreads).
Credit retrieving the memory type that actually surfaced GT-relevant evidence for
*this* instance. **Open work to resolve first:** a single `Retrieve(q)` response
mixes both memories, so the reward must distinguish a collaborative-memory doc
(user-history play sequences) from a meta-memory doc (item metadata) inside one
`<tool_response>` — decide the parse (doc-shape heuristic vs. a tag the retriever
could emit) before implementing. One env knob, Audit-B'd before it trains.

**P3 — `r_when` (when to retrieve).** Small credit for *not* retrieving when the
model answers correctly without it, and for retrieving when it needs to — the
on-demand behavior RRCM reports in RQ3, now explicitly incentivized. Keep it
small (the outcome reward already applies weak pressure here). **Collapse guard,
mandatory:** RRCM's RQ3 already shows retrieval count *declines* over training;
floor any penalty and monitor retrieval-rate every run so this term doesn't
accelerate the drift into "never retrieve." (Same guard `r_retqual`'s
`RTHINK_RETQUAL_FLOOR` already applies — see §4.)

### 9.3 Per-term work checklist

Every new term (`r_memtype`, `r_when`; and retroactively `r_retqual`/`r_covgain`)
must ship with:

1. **Audit-B correlation** with `InTop@n` hit on logged rollouts (P0) — the gate.
2. **Env knob + default** on the `RTHINK_` prefix (§6), documented in the §6 table.
3. **Returned-dict key** so `naive.py` logs it for free (as `retqual`, `covgain`,
   `best_sim` already are — §5).
4. **An `*_ONLY` / off toggle** for a single-variable ablation (mirrors
   `RTHINK_RETRIEVAL_ONLY`).
5. **An anti-hacking gate** for its specific farm: retrieval-spam for `r_retqual`
   (guarded already — `r_covgain` credits only *improving* turns, spam earns 0),
   memory-type-parroting for `r_memtype`, retrieval-avoidance for `r_when`
   (guarded by the collapse floor).

New terms go in `reward_retrieval/v7/retrieval.py`, **not** a new
`reward_reasoning/v7/` — the roadmap's code-layout note supersedes the earlier
sketch: retrieval reward is its own `reward_retrieval/` package, dispatched from
`reward_reasoning/__init__.py` under `RTHINK_MODE=v7`. Keep `reward_SPRec.py` and
the `InTop@n` weights untouched so the baseline stays reproducible.

### 9.4 Evaluation protocol (owned here; cross-linked from the reasoning doc)

A reward extension lives or dies on whether the gain is *real* at these sparse
magnitudes (roadmap §6). This axis owns the protocol for both tracks:

1. **All three datasets** (Goodreads, MovieLens, Amazon CDs&Vinyl) — a
   single-dataset win won't survive review.
2. **≥3 decode seeds, report mean ± std.** One temp-1 decode swings HR@1
   0.000↔0.005 — wider than every between-method gap at these magnitudes; the win
   must exceed the seed spread. (This is itself a **methodology contribution**:
   single-decode eval is unreliable in this regime.) **Both arms at `rollout.n=8`.**
3. **RQ2-style ablations:** report the reward with each term removed
   (`w/o r_retqual`, `w/o r_memtype`, …) so each term is attributable.
4. **RQ3-style behavior analysis:** retrieval count/type over training, the
   single→multi-query shift `r_covgain` targets, and the copier→inferencer shift
   (hits-without-doc-support) the reasoning axis targets.
5. **Pre-register the expected effect** (from Audit A's selection headroom, §6 of
   the reasoning doc; and Phase-1 union-coverage) before the full runs, so a
   positive result is confirmatory.

### 9.5 Risks specific to this axis (roadmap §8)

- **Retrieval collapse.** The safe policy can become "never retrieve," worsened
  by RRCM's own RQ3 downward drift. Floor every penalty (`RTHINK_RETQUAL_FLOOR`
  already; `r_when` likewise) and monitor retrieval-rate every run.
- **Multi-turn may not help** (the PI's explicit doubt) — resolved offline in
  Phase 1 before any `r_covgain` training; if union coverage barely moves, drop it.
- **`r_retqual` gaming by batching many queries per turn** (§8) — a turn is
  scored as a whole, so one good query hides many junk ones. Revisit per-query
  scoring if the model starts batching to farm the term.
- **Decode noise dominates at the floor.** Until a term lifts HR meaningfully off
  ~0.005, every comparison sits inside the ≥3-seed error bar — which is why the
  Audit-B correlation and behavioral metrics (retrieval-rate, `best_sim`) are the
  right early signals, not raw HR@1.