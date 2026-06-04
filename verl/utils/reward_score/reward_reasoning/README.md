# Reasoning Reward (R_think) — versioned iterations

This package holds the successive iterations of the REC-R1 **reasoning reward**.
The reasoning reward augments the outcome reward `R_answer` with a *process*
signal so the policy is rewarded for *how* it reasons, not only *whether* the
final recommendation is correct:

```
R_total = R_answer + (shaping derived from R_think)
```

`R_answer` is unchanged across all iterations — it is the embedding-rank outcome
reward in [`../reward_SPRec.py`](../reward_SPRec.py). Only `R_think` (and how it
is combined with `R_answer`) changes between iterations.

## Layout

```
reward_reasoning/
├── __init__.py        # version dispatcher (RTHINK_MODE -> v2 | v3)
├── README.md          # this file
├── v2/                # info_gain / redundancy / exploration (underperformed)
│   ├── __init__.py    # re-exports the canonical v2 implementation
│   └── README.md      # v2 design + postmortem
└── v3/                # evidence-grounded process reward (current default)
    ├── __init__.py
    ├── reasoning.py   # tool_use / grounding / synthesis / self_rep components
    ├── orchestrator.py# R_think -> shaping -> R_total (dict return)
    └── README.md      # v3 design rationale
```

> **Note on v2 file locations.** The v2 *implementation* still lives at
> `../reward_SPRec_reasoning.py` (components) and `../reward_SPRec_rthink.py`
> (orchestrator), because the unit tests and `CLAUDE.md` reference those modules
> by name. `v2/__init__.py` simply re-exports them so the dispatcher can select
> v2 with the same interface as v3. The v3 implementation is fully contained in
> `v3/`.

## How to select an iteration

Routing happens in [`../__init__.py`](../__init__.py): for an
`amazon`/`goodreads`/`movie` data source, if `USE_RTHINK=1` (or legacy
`USE_REWARD_B=1`) is set, the reward is computed by this package's dispatcher,
which reads `RTHINK_MODE`:

| `RTHINK_MODE`        | Iteration used            |
| -------------------- | ------------------------- |
| unset / `v3` / `3`   | **v3** (default)          |
| `v2` / `legacy` / `2`| v2                        |

If `USE_RTHINK` is **not** set, the plain outcome reward (`reward_SPRec`,
i.e. the *baseline*) is used and this package is never imported.

## Iteration history

| Version | Idea | Result |
| ------- | ---- | ------ |
| **v1**  | `R_think = α·info_gain − β·redundancy + γ·exploration`, β=1.0 | `red=1.0` on ~71% of samples (Tier-1/CF-domain bug); near-zero gradient variance. |
| **v2**  | Same components; Tier-1 fix (prompt-only) + β=0.5 | `red=1.0` dropped to 8%, but `R_think` still **negative ~99%** of the time and `info_gain` dead (`>0.1` in 0% of samples). Trained reward stayed flat (~0.25) while the **baseline climbed to 0.53**; held-out reward went **negative** and answer quality *declined*. See [`v2/README.md`](v2/README.md). |
| **v3**  | Replace the components with **evidence-grounded process signals** (`tool_use`, `grounding`, `synthesis`, `self_rep`); two-sided shaping, capped below the answer-tier gaps, with corrected LongPAS asymmetry; dict return for clean metrics. | See [`v3/README.md`](v3/README.md). |

## Why v2 lost to "no reasoning" (the one fact that drove v3)

Under GRPO the advantage is `(r_i − mean_group)/std_group`, so **any reward
component that is constant across a group cancels — only its intra-group
*variance* survives.** A process reward therefore only helps if that surviving
variance is a **leading indicator of correctness**.

In v2 the positive terms were dead or per-group-constant, so the only surviving
variance came from the redundancy penalty — and in the rthink-v2 run that
penalty was **uncorrelated with correctness** (`corr(redundancy, R_answer) =
+0.10`, even slightly *anti*-helpful). With base accuracy low, the majority of
GRPO groups are all-wrong; in the baseline those groups have zero reward
variance and are harmlessly ignored, but v2 gave them redundancy-driven
variance, training the model to *reduce* overlap with the user's history — i.e.
to stop grounding its reasoning — which is exactly the wrong direction.

**v3 makes the surviving variance point toward correctness instead.** Its
process signals reward issuing a real search, recommending an item supported by
the retrieved evidence, and synthesising the results — behaviours that *precede*
correct recommendations — so injecting that signal into all-wrong groups becomes
a feature (process bootstrapping) rather than a bug.

## Evidence (from the 2026-06-02/03 dual-GPU runs)

Parsed from `nq-search-r1-grpo-qwen3-1.7b-sbatch-gpu-baseline.log` and
`...-rthink-v2.log`, and the corresponding wandb histories:

| Metric | baseline | rthink-v2 (this package's v2) |
| --- | --- | --- |
| training reward (`critic/score/mean`) | **0.013 → 0.526** (rising) | ~0.29 → 0.25 (flat) |
| `R_answer` (decoupled, debug lines) | rises to ~0.53 | **0.256 → 0.133 (falling)** |
| held-out `val-core reward@1` | +0.003 → +0.009 | **−0.081 (stuck, negative)** |
| `info_gain` | — | mean 0.008; `>0.1` in **0%** of samples |
| `redundancy` | — | mean 0.72; `=1.0` in 35%; `corr` w/ correctness **+0.10** |
| fraction with `R_think > 0` | — | **0%** |

A full prose postmortem is in
[`../../../../../REWARD_REASONING_ANALYSIS.md`](../../../../../REWARD_REASONING_ANALYSIS.md).

## Recommended A/B protocol

Run three jobs that differ only in the reward, same seed/steps/config:

1. **baseline** — `USE_RTHINK` unset (outcome reward only).
2. **v2** — `USE_RTHINK=1 RTHINK_MODE=v2`.
3. **v3** — `USE_RTHINK=1 RTHINK_MODE=v3` (default).

Compare on **`r_answer`**, not on `score`/`reward`: v3 returns a dict, so the
clean outcome metric is logged as `val-core`/`val-aux .../r_answer/mean@1`,
which is directly comparable to the baseline's `reward/mean@1`. The headline
`score` includes the shaping term and is therefore *not* comparable to the
baseline across runs.

Watch in wandb: `r_answer` (vs baseline `reward`), `tool_use`, `grounding`,
`synthesis`, `self_rep`, `actor/entropy` (v2 collapsed early), and the fraction
of GRPO groups with non-zero reward variance.
