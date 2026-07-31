# v7 — retrieval-quality reward (the PI's proposal)

## Status (2026-07-21)

| | |
|---|---|
| **Live term** | `r_retqual` only, at `RTHINK_RETQUAL_TAU=0.20` |
| **Retired** | `r_covgain` — `W=0` **permanently** (M2 measurement, see below) |
| **First GPU run** | **M3 launched 2026-07-21, SLURM `854501`** — `RTHINK_MODE=v7`, `W_COVGAIN=0`, τ=0.20, `n=8`, experiment `nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-rthink-v7-n8` |
| **A/B partner** | `...-gpu-baseline-n8` (outcome-only `reward_SPRec`, same launcher, same substrate). **Compare at step 200** — both arms locked there. |
| **Not built** | `r_memtype` (P2), `r_when` (P3), ablation/behavior tooling (P4), and the `RTHINK_RETRIEVAL_ONLY=1` wiring-ablation *run* |

⚠️ **Known arm asymmetry:** the baseline arm trained *before* the
`tool_call_repair` JSON fix landed (2026-07-20 18:55); this v7 run trains
*with* it, so ~2% of rollouts retrieve where they previously retrieved
nothing. Accepted deliberately rather than spending a window on a re-baseline.
The planned `RTHINK_RETRIEVAL_ONLY=1` ablation runs on this same fixed
substrate and is the parity-matched control when it happens.

## Why this exists

The v6 plateau was diagnosed as retrieval-bound, and an earlier roadmap
prescribed restructuring the retrieval corpus into user-independent
CF-continuation documents (`cf_corpus/`) to raise how often the ground-truth
item is reachable at all. The PI reviewed that diagnosis and rejected it (the
current `ROADMAP_v7.md` in this folder is the corpus-fixed plan that resulted):

> Don't change the corpus — it's already right. Leave the ground-truth item
> out of the retrieval corpus. That ~4.6% GT-in-docs rate on held-out data is
> correct, not a bug. Build a reward for retrieval, not reasoning: every time
> the model retrieves, check how similar the retrieved items are to the
> correct answer. Reward good retrieval, penalize bad retrieval. That teaches
> the model when to retrieve and how to write a good query.

v7 is that literal proposal, implemented. It never reads or writes the
corpus (`data/amazon_data/corpora.jsonl` / `e5_Flat.index`) — see
`../retrieval_reward_design.md` for the full design and the code research it
is grounded in (exact tag formats at train time, embedding reuse, the
`RTHINK_MODE` dispatch contract).

## What changed vs v6

Everything except the shaping term is v6 verbatim: the HR-faithful top-K
outcome reward (`r_outcome`), the format-integrity gate, the length
discipline, the optional decisiveness bonus, and the LongPAS asymmetry (a
correct + well-formed answer is only ever helped by shaping, never hurt).

The one substantive change: v3's reasoning-quality process shaping
(`tool_use` / `grounding` / `synthesis` / `self_rep`) is replaced by
`retrieval.py`'s retrieval-quality shaping:

- **`r_retqual`** — **the live term.** Per retrieval turn, cosine similarity
  between the best-matching retrieved document and the ground-truth answer,
  two-sided: rewarded above a neutral threshold (`RTHINK_RETQUAL_TAU`),
  penalized below it but damped (`RTHINK_RETQUAL_FLOOR`) so a junk retrieval
  never costs more than simply not retrieving would have — this guards
  against the "safe policy = never retrieve" collapse called out in
  `ROADMAP_v7.md` §4.1. **τ was re-calibrated 0.30 → 0.20** (the measured
  `best_sim` median) before the first GPU run: the shipped 0.30 sat at the
  ~77th percentile, gave a mean `retqual` of −0.056, and therefore made an
  abstaining rollout (exactly 0.0) beat a retrieving one *inside its own
  GRPO group* — i.e. the default actively taught retrieval collapse. This
  does not affect Audit B, since `retqual` is monotone in `best_sim` and the
  AUC is τ-invariant.
- **`r_covgain`** — **IMPLEMENTED BUT RETIRED TO `W=0` PERMANENTLY (M2,
  2026-07-20). Do not re-enable without new evidence.** It credits a *later*
  turn only when it beats the running-best similarity from earlier turns in
  the same rollout, which was meant to target the measured 99% single-query
  collapse without rewarding retrieval-spam. The M2 offline coverage sweep
  killed it on measurement, not on theory: rollouts average **~0.97 turns**,
  and the marginal coverage of a later turn is **0.000** at both k=3 and
  k=20 — the behavior it pays for essentially does not occur, and widening
  the turn budget does not make it occur. It is left in the code (inert at
  `RTHINK_W_COVGAIN=0`) rather than deleted, so the ablation story stays
  reproducible. See `ROADMAP_v7.md` M2 and
  `../retrieval_reward_design.md` §4.

Both are computed purely from `solution_str` — the text the model already
produced, including what the retrieval tool already returned into the
rollout. No corpus file is opened.

## Tags parsed

Confirmed to be what training rollouts actually use (Qwen3-native tool
calling, `type: native` in `search_tool_config.yaml`), not the `<search>`/
`<info>` fallback branches that exist elsewhere in this codebase but are dead
at runtime under the current config:

```
<tool_call>{"name": "search", "arguments": {"query_list": [...]}}</tool_call>
<tool_response>{"result": "Doc 1 (Title: ...)\n...\n\nDoc 2 (Title: ...)\n...\n\n---\n..."}</tool_response>
```

`retrieval.py` pairs each `<tool_call>` with the `<tool_response>` that
follows it, in order, and recovers per-document text from the
`"Doc N (Title: ...)"` blocks inside the (JSON-escaped) `result` string —
exactly what `verl/tools/utils/search_r1_like_utils.py::_passages2string`
produces.

## Ablation

`RTHINK_RETRIEVAL_ONLY=1` disables the retrieval shaping entirely (mirrors
v6's `RTHINK_DENSE_ONLY`), for an outcome-only baseline to compare against.
