# Reasoning Reward — iteration v3 (evidence-grounded process reward)

> **Status:** current default (`RTHINK_MODE=v3`, or unset).

v3 replaces v2's `info_gain / redundancy / exploration` with process signals
that are **leading indicators of a correct recommendation** in this search-agent
setup, so that — unlike v2 — the reward's GRPO-surviving intra-group variance
points *toward* correctness. See [../v2/README.md](../v2/README.md) for why v2
failed and [../README.md](../README.md) for the cross-iteration evidence.

## Files

- [`reasoning.py`](reasoning.py) — the four components.
- [`orchestrator.py`](orchestrator.py) — combines them into `R_total` and
  returns a metrics dict. Entry point: `compute_score`.

## The reward

```
R_think_raw = w_tool·tool_use + w_ground·grounding + w_synth·synthesis − w_rep·self_rep
shaping     = clip( SCALE · R_think_raw , −CAP , +CAP )
R_total     = R_answer + ( max(0, shaping)  if R_answer ≥ 0.5   # correct
                           else shaping )                       # wrong
```

### Components (each in `[0, 1]`, per rollout, no global state)

| Component | What it measures | Why it correlates with correctness |
| --- | --- | --- |
| **tool_use** | `0.3·searched + 0.3·got results + 0.4·reasoned after` | The agent cannot retrieve the right item without actually searching and reading results. |
| **grounding** | max cosine(`<answer>` title, retrieved-evidence sentence) | Recommending an item retrieval actually surfaced beats hallucinating one. |
| **synthesis** | `0.5·(post-think uses evidence) + 0.5·(post-think differs from pre-think)` | Genuinely incorporating the search results into the final reasoning. |
| **self_rep** *(penalty)* | Tier-0 intra-`<think>` degeneration | Looping/parroting is a real failure mode; the *only* redundancy kept from v2. |

`grounding` and `synthesis` use the cached `paraphrase-MiniLM-L3-v2` encoder
(shared with `reward_SPRec`). Evidence = retrieved `<tool_response>`/`<info>`
sentences first, then the user's history from the prompt (capped at 40
sentences). **Crucially, overlap with evidence is now *rewarded*** (grounding),
the opposite of v2's redundancy penalty.

## How v3 fixes each v2 failure

| v2 failure | v3 fix |
| --- | --- |
| `info_gain` dead (positive term never paid) | `tool_use`/`grounding`/`synthesis` are easy to earn for good behaviour → `R_think_raw` is routinely **positive**. |
| redundancy penalised grounding | grounding is **rewarded**; only Tier-0 self-repetition is penalised. |
| GRPO saw only redundancy-noise variance | surviving variance now tracks retrieval + grounding + synthesis = correctness precursors. |
| asymmetric gate was a silent no-op on correct | positive half is live, so correct answers do get a (small) bonus. |
| `λ·R_think` reached −0.13 and could reorder answer tiers | `|shaping| ≤ CAP = 0.08 < 0.1` (smallest tier gap) → can differentiate within a tier, never across. |
| module-global state broke on resume | zero global state; everything is per-rollout. |
| `score` contaminated the held-out metric | returns a dict; `r_answer` is logged separately for a clean baseline comparison. |

## Hyperparameters (env vars, all optional)

| Var | Default | Meaning |
| --- | --- | --- |
| `RTHINK_SCALE` | `0.10` | scale of `R_think_raw` before clipping |
| `RTHINK_CAP` | `0.08` | absolute cap on applied shaping (keep `< 0.1`) |
| `RTHINK_W_TOOL` | `0.30` | weight of `tool_use` |
| `RTHINK_W_GROUND` | `0.40` | weight of `grounding` |
| `RTHINK_W_SYNTH` | `0.30` | weight of `synthesis` |
| `RTHINK_W_REP` | `0.50` | weight of `self_rep` penalty |

With the defaults, a strong rollout (`tool=1, grnd≈0.5, syn≈0.6, rep=0`) gives
`R_think_raw ≈ 0.3 + 0.2 + 0.18 = 0.68` → `shaping ≈ +0.068`; a lazy/no-search
rollout gives `shaping ≈ 0`; a degenerate one is pushed negative. So within an
all-wrong group the advantage gradient flows toward searching, grounding, and
synthesising — without ever letting a well-reasoned wrong answer outrank a
correct one.

## Returned dict

`compute_score` returns:

```python
{"score": R_total,        # reward used by the trainer
 "r_answer": R_answer,    # clean outcome reward  (compare to baseline reward@1)
 "r_think": shaping,      # the shaping actually applied
 "tool_use": ..., "grounding": ..., "synthesis": ..., "self_rep": ...}
```

The reward manager logs every key, so training and validation both expose
per-component curves (`.../r_answer/mean@1`, `.../grounding/mean@1`, ...).

## Running it

```bash
# v3 is the default whenever rthink is on:
USE_RTHINK=1 ...            # RTHINK_MODE defaults to v3

# explicit / tuned:
USE_RTHINK=1 RTHINK_MODE=v3 RTHINK_SCALE=0.10 RTHINK_CAP=0.08 ...
```

`run_in_container_rthink.sh` exports and forwards these vars. For the
baseline/v2/v3 A/B protocol see [../README.md](../README.md#recommended-ab-protocol).

## Caveats / things to watch

- **`grounding` rewards evidence-supported answers, not necessarily *correct*
  ones.** It is a process proxy; the cap keeps it from overriding `R_answer`.
- **Cosine magnitudes are modest** (short title vs sentence ≈ 0.2–0.5); that is
  fine — GRPO only needs *relative* per-rollout variation, and the weights/scale
  map it into the cap band.
- If you later add explicit `<info>` tags (instead of `<tool_response>`), no
  change needed — both are recognised.
