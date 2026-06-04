# Reasoning Reward — iteration v2 (postmortem)

> **Status:** superseded by [v3](../v3/README.md). Kept for ablation
> (`RTHINK_MODE=v2`) and as the reference for *what not to do*.

## Where the code lives

The v2 implementation was not moved (tests and `CLAUDE.md` import it by name):

- Components: [`../../reward_SPRec_reasoning.py`](../../reward_SPRec_reasoning.py)
- Orchestrator: [`../../reward_SPRec_rthink.py`](../../reward_SPRec_rthink.py)

`v2/__init__.py` re-exports the orchestrator's `compute_score` so the dispatcher
can select it uniformly.

## Design

```
R_think = α·info_gain − β·redundancy + γ·exploration_bonus
R_total = R_answer + λ·R_think        (asymmetric: correct -> only +R_think)
defaults: λ=0.3, α=0.5, β=0.5, γ=0.3
```

- **info_gain** — novelty of `<think>` vs prompt+search, vs a global buffer of
  past think blocks, minus an n-gram staleness term. Combined as
  `min(sentence_novelty, past_novelty)·(1 − stale)`.
- **redundancy** — tiered copy detector: Tier 0 intra-think repetition →
  Tier 1 exact substring vs prompt → Tier 2 n-gram overlap → Tier 3 sentence
  cosine vs prompt/search. `redundancy = max(tiers)`, floored at 0.15.
- **exploration_bonus** — sigmoid decay as the *global* cumulative reward rises,
  EMA-smoothed, times a length sweet-spot factor.

## Why it underperformed the no-reasoning baseline

Empirical fingerprints from the rthink-v2 run (β=0.5, 211 debug samples) and its
wandb history, against the baseline:

| Symptom | Measurement |
| --- | --- |
| `info_gain` never pays | mean 0.008, `>0.1` in **0%** of samples |
| `redundancy` dominates, but is noise | mean 0.72, `=1.0` in 35%; `corr(redundancy, R_answer)=+0.10` |
| `R_think` is one-sided | **0%** of samples had `R_think > 0` (pure penalty) |
| training reward stalls | `critic/score/mean` ~0.29 → 0.25 (baseline: 0.013 → **0.526**) |
| answer quality declines | decoupled `R_answer` 0.256 → **0.133** |
| held-out reward goes negative | `val-core reward@1` ≈ **−0.081** (baseline +0.007) |
| entropy collapses early | 0.11 → 0.057 on a bad policy |

### Root causes

1. **`info_gain` is structurally dead.** `min(sentence_novelty, past_novelty)`
   requires novelty on *both* axes; the global cross-prompt past-think buffer
   makes `past_novelty ≈ 0` for every templated rec rollout, zeroing the term.
   So the only live term was the redundancy *penalty*.
2. **`redundancy` penalises grounding, not copying.** It fires on the model
   restating the user's history / framing the task — necessary working memory —
   so it is ~uncorrelated with correctness and pushes the policy to *stop*
   grounding.
3. **GRPO sees only intra-group variance.** `info_gain` (≈const 0) and
   `exploration` (driven by *global* cumulative reward → per-group constant)
   cancel; the surviving variance is redundancy noise. In the majority all-wrong
   groups, the baseline contributes zero gradient (harmless) while v2 injects a
   misleading one.
4. **The asymmetric gate became a no-op on correct, penalty-only on wrong.**
   Because `R_think` was negative ~99% of the time, `max(0, R_think)=0` for
   correct answers (no bonus) and the full negative term hit wrong answers.
5. **Module-global state is not resume-safe.** The past-think buffer, staleness
   n-grams, and cumulative-reward/EMA reset on every checkpoint resume, so the
   exploration schedule and staleness vocabulary were wrong after each resume.

The full prose analysis is in
[`../../../../../../REWARD_REASONING_ANALYSIS.md`](../../../../../../REWARD_REASONING_ANALYSIS.md).

## What v3 changed in response

- Dropped `info_gain`/`redundancy`/`exploration` for **evidence-grounded**
  signals that correlate with correctness (`tool_use`, `grounding`,
  `synthesis`), keeping only Tier-0 self-repetition as `self_rep`.
- Made shaping **two-sided and mostly positive**, **capped below the
  answer-tier gaps**, and fixed the asymmetric gate so the positive half pays.
- Removed all module-global state (resume-safe, GRPO-visible).
- Returns a dict so `r_answer` is logged separately for clean comparison.
