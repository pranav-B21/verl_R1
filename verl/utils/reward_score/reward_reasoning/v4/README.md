# Reasoning Reward — iteration v4 ("dense answer reward")

**Status:** implemented, *ablation-first*. Run dense-only and beat the baseline
HR before re-enabling the v3 process shaping.

## Why v4 exists

v2 and v3 both shaped the *process*; neither moved held-out HR. The held-out eval
(`TEST_OUTPUT.md`) and the GRPO analysis (`REWARD_REASONING_ANALYSIS.md`, §4 + the
v4 Proposal) located the real blocker — and it is **not** in the shaping:

> In the ~87% of GRPO groups where every rollout is wrong, the *tiered* outcome
> reward `r_answer` (1.0/0.8/0.5/0.1/0.001/0) collapses everything past rank-500
> to a flat **0.0**. So correctness has **zero within-group variance → zero
> gradient** there. v3 filled that vacuum with process shaping, whose surviving
> variance correlates with correctness at only ~**+0.1** (noise). The model
> learned to "look agentic / diverse," not to "get closer to the answer."

Tuning the shaping (what v2→v3 did) cannot fix a *sparsity* problem in the
outcome reward. v4 makes the outcome reward itself dense.

## The change

`reward_SPRec` already computes `rankId` = the position of the true target among
all `N` (≈13,079) catalog items, by embedding distance to the predicted title. v3
threw it into coarse cliffs; v4 keeps the whole signal:

```
r_dense = ( 1 - log(rankId) / log(N) ) ** p        # p = RTHINK_DENSE_P (>=1)
R_total = r_dense + clip( SCALE * R_think_raw, -CAP, +CAP )   # shaping = v3 terms
```

| target rank | v3 tiered r_answer | v4 r_dense (p=1) | v4 r_dense (p=2) |
|---|---|---|---|
| 1     | 1.0   | 1.00 | 1.00 |
| 10    | 0.5   | 0.76 | 0.57 |
| 100   | 0.1   | 0.51 | 0.26 |
| 500   | 0.001 | 0.34 | 0.12 |
| 1000  | 0.0   | 0.27 | 0.07 |
| 5000  | 0.0   | 0.10 | 0.01 |
| 13079 | 0.0   | 0.00 | 0.00 |

Now an all-wrong group ranking the target at 600/3000/9000 has **monotonic,
correctness-aligned variance** — GRPO finally gets a "name something closer"
gradient exactly where baseline and v3 had none. Because `r_dense` is a monotonic
transform of the same rank HR thresholds, pushing it up pushes targets toward the
top-k: it is a dense surrogate of the metric we are judged on.

**Coupling: additive, dense dominates.** The v3 components and the ±0.08 cap are
unchanged but added *on top of* `r_dense`, whose within-group variance (~0.1–0.3)
outweighs the ±0.08 shaping. The gradient points at correctness first; reasoning
quality only breaks ties among rollouts that landed similarly close — so
exploration becomes **targeted** while v3's diversity / ORRatio gains (which came
from the process terms, not penalised here) are retained. (A multiplicative gate
`shaping × f(r_dense)` is the deferred v4.1.)

## What changed in the code

| File | Change |
|---|---|
| `reward_SPRec.py` | `similarity_match` / `compute_score` gain `return_rank=True` → return `{match, rankId, N, target_found}`. **Default float path is byte-for-byte unchanged**, so baseline and v3 are unaffected. |
| `reward_reasoning/v4/orchestrator.py` | new — computes `r_dense`, reuses `v3.reasoning` for shaping, returns the dict. |
| `reward_reasoning/__init__.py` | `RTHINK_MODE=v4` (alias `4`) dispatches here. |
| `run_in_container_rthink_v4.sh` | training launcher (defaults to dense-only ablation). |
| `test_reward_system.py --v4` | pure-math demo that `r_dense` revives the dead all-wrong groups. |

**F9 leak fix (built in):** if the target title is absent from `name2id`,
`reward_SPRec` falls back to `target_id=0` (= "River of Dreams") and hands out a
spurious rank-1. v4 detects this (`target_found=False`) and returns **0** for both
`r_answer` and `r_dense`, so the now-everywhere-nonzero reward does not amplify the
leak. The baseline keeps the old fallback (comparability preserved).

## Hyperparameters (env vars)

| Var | Default | Meaning |
|---|---|---|
| `RTHINK_DENSE_P`    | `1.0` | sharpness exponent `p`; raise (e.g. 2) if the tail rewards far guesses too much |
| `RTHINK_RANK_FLOOR` | `0`   | zero `r_dense` when `rankId > floor` (0 = off) |
| `RTHINK_DENSE_ONLY` | `1`*  | `1` = no shaping (ablation); `0` = full v4 |
| `RTHINK_SCALE/_CAP/_W_*` | v3 defaults | process-shaping knobs (only when `DENSE_ONLY=0`) |

\* The run script defaults `RTHINK_DENSE_ONLY=1`; the dispatcher/orchestrator
default is `0` (full v4) if you call it directly.

## Logging & how to judge it

Returns a dict; the reward manager logs each key:
`score` (= R_total), **`r_answer` (tiered, baseline-comparable north-star)**,
`r_dense`, `rank`, `r_think` (applied shaping), and the four process components.
**Compare `r_answer` against the baseline's `reward/mean@1`, never `score`** —
`score` includes the dense term and is not comparable across runs.

## Build & validation plan (ablation-first)

1. **Dense-only** (`RTHINK_DENSE_ONLY=1`, the script default): prove `r_dense`
   *alone* lifts held-out HR@5 above the baseline. Attributes any gain to the
   dense reward, not the process terms.
2. **Simulator check:** `python test_reward_system.py --v4` →
   `corr(r_dense, -rank)` strongly positive (+0.92 on the sample) and all-wrong
   groups carry correctness-driven advantage, not shaping-driven.
3. **Full v4** (`RTHINK_DENSE_ONLY=0`): require HR@5 > baseline *while* ORRatio /
   diversity stay better than baseline — accuracy **and** the exploration v3 bought.
4. **Stability** (trainer config, not reward code): the optional `ENTROPY_COEFF` /
   `KL_LOSS_COEF` hooks in the run script head off the v3 post-step-300 collapse;
   evaluate the pre-collapse checkpoint, not `latest`.

## Risks to watch

- **The embedding-rank proxy is still the reward (F9).** Dense changes its
  resolution, not the proxy; it could be gamed without lifting *true* HR. `eval.py`
  measures true HR independently — watch the train-vs-val `r_answer` gap.
- **Tail too generous.** If p=1 over-rewards far guesses, raise `p` or set
  `RTHINK_RANK_FLOOR`.
- **Dense vs shaping scale.** If shaping still steers toward diversity at HR's
  expense, drop `RTHINK_CAP` or move to the v4.1 multiplicative gate.
