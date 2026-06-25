"""Combined Reasoning Reward orchestrator — iteration v4 ("dense answer reward").

    r_dense  = ( 1 - log(rankId) / log(N) ) ** p            # N = catalog size, p >= 1
    shaping  = clip( SCALE * R_think_raw , -CAP, +CAP )     # identical to v3
    R_total  = r_dense + ( max(0, shaping)  if r_answer >= 0.5   # correct: never punished
                           shaping          otherwise )

Why v4 (the problem v3 could not solve)
---------------------------------------
v3's outcome reward is the *tiered* `r_answer` (1.0/0.8/0.5/0.1/0.001/0). In the
~87% of GRPO groups where every rollout is wrong, the tiers collapse everything
past rank-500 to a flat 0.0, so correctness contributes **zero within-group
variance -> zero gradient** there. v3 filled that vacuum with process *shaping*,
whose surviving variance correlates with correctness at only ~+0.1 (noise). The
model therefore learned "look agentic / diverse" instead of "get closer to the
answer", and HR did not move (see REWARD_REASONING_ANALYSIS.md §4 + v4 Proposal).

The fix is to make the *outcome* reward itself dense. `rankId` (position of the
true target among all N catalog items, by embedding distance to the predicted
title) is already computed inside `reward_SPRec`; v3 threw it away into coarse
cliffs. v4 keeps the full signal:

    rank 1 -> 1.00,  10 -> 0.76,  100 -> 0.51,  1000 -> 0.27,  5000 -> 0.10  (p=1)

Now an all-wrong group whose rollouts rank the target at 600 / 3,000 / 9,000 has
**real, monotonic, correctness-aligned variance** — GRPO finally gets a gradient
that says "name something closer to the answer" exactly where baseline and v3 had
nothing. Because `r_dense` is a monotonic transform of the same rank HR
thresholds, pushing it up pushes targets toward the top-k: it is a dense
surrogate of the metric we are judged on.

Coupling: additive, dense dominates
-----------------------------------
The v3 process components and the ±CAP cap are kept unchanged, but added on top
of `r_dense` whose within-group variance (~0.1–0.3) now *outweighs* the ±0.08
shaping. So the gradient points at correctness first and reasoning quality only
breaks ties among rollouts that landed similarly close — exploration becomes
*targeted*, while the diversity / ORRatio gains v3 bought (which came from the
process terms, and which v4 does not penalise) are retained.

Logging & comparability
-----------------------
The dict still logs the **tiered** `r_answer` separately from `score`, so the val
`r_answer/mean@1` stays directly comparable to the baseline's `reward/mean@1`
(the §1 north-star). `r_dense` and `rank` are logged too so the dense signal is
auditable.

F9 leak fix
-----------
When the target title is absent from `name2id`, `reward_SPRec` would fall back to
`target_id=0` (= "River of Dreams") and hand out a spurious rank-1. With a
now-everywhere-nonzero reward that leak would be amplified, so v4 detects it
(`target_found=False`) and returns **0** for both `r_answer` and `r_dense`.

Hyperparameters (env vars, all optional)
----------------------------------------
    RTHINK_DENSE_P      sharpness exponent p on r_dense              (default 1.0)
    RTHINK_RANK_FLOOR   zero r_dense when rankId > floor (0=off)     (default 0)
    RTHINK_DENSE_ONLY   1 -> disable shaping (ablation)              (default 0)
    RTHINK_SCALE        overall scale of R_think_raw before clipping (default 0.10)
    RTHINK_CAP          absolute cap on the applied shaping          (default 0.08)
    RTHINK_W_TOOL/_W_GROUND/_W_SYNTH/_W_REP  process weights         (v3 defaults)
"""

import math
import os
import random


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


_DENSE_P    = _envf("RTHINK_DENSE_P",    1.0)
_RANK_FLOOR = _envi("RTHINK_RANK_FLOOR", 0)
_DENSE_ONLY = _envi("RTHINK_DENSE_ONLY", 0)

_SCALE    = _envf("RTHINK_SCALE",    0.10)
_CAP      = _envf("RTHINK_CAP",      0.08)
_W_TOOL   = _envf("RTHINK_W_TOOL",   0.30)
_W_GROUND = _envf("RTHINK_W_GROUND", 0.40)
_W_SYNTH  = _envf("RTHINK_W_SYNTH",  0.30)
_W_REP    = _envf("RTHINK_W_REP",    0.50)


def _dense_from_rank(rank_id, n_items) -> float:
    """r_dense = (1 - log(rankId)/log(N))**p, clamped to [0, 1]; 0 if no rank."""
    if not rank_id or not n_items or n_items <= 1:
        return 0.0
    if _RANK_FLOOR > 0 and rank_id > _RANK_FLOOR:
        return 0.0
    # rankId in [1, N]; clamp so log is well defined and the head saturates at 1.
    r = min(max(int(rank_id), 1), int(n_items))
    base = 1.0 - (math.log(r) / math.log(n_items))
    base = min(1.0, max(0.0, base))
    return base ** _DENSE_P


def compute_score(solution_str: str, ground_truth: dict, data_source: str,
                  method: str = "strict", format_score: float = 0.0,
                  score: float = 1.0, extra_info: dict = None) -> dict:
    """Drop-in replacement for reward_SPRec.compute_score (returns a dict).

    The reward manager uses ``score["score"]`` as the reward and logs the rest:
    ``r_answer`` (tiered, baseline-comparable), ``r_dense``, ``rank``,
    ``r_think`` (applied shaping) and the four process components.
    """
    # ------------------------------------------------------------------ #
    # Outcome — tiered r_answer (baseline-comparable) + raw rank for v4.  #
    # ------------------------------------------------------------------ #
    from ... import reward_SPRec
    outcome = reward_SPRec.compute_score(
        solution_str, ground_truth, data_source,
        method=method, format_score=format_score, score=score,
        return_rank=True,
    )
    rank_id = outcome["rankId"]
    n_items = outcome["N"]
    target_found = outcome["target_found"]

    # F9: target missing from name2id -> the tiered match is a spurious rank-1
    # off item-0; drop it (and the dense reward) to 0 instead of leaking reward.
    if not target_found:
        r_answer = 0.0
        r_dense = 0.0
    else:
        r_answer = float(outcome["match"])
        r_dense = _dense_from_rank(rank_id, n_items)

    # ------------------------------------------------------------------ #
    # Process shaping — reuse v3 components verbatim (unless DENSE_ONLY). #
    # ------------------------------------------------------------------ #
    tool_use = grounding = synthesis = self_rep = 0.0
    r_think_raw = 0.0
    applied = 0.0
    if not _DENSE_ONLY:
        prompt_text = ""
        if extra_info and extra_info.get("prompt_str"):
            prompt_text = extra_info["prompt_str"]

        from ..v3 import reasoning
        comp = reasoning.compute_process_components(
            solution_str, data_source, prompt_text=prompt_text, extra_info=extra_info,
        )
        tool_use, grounding = comp["tool_use"], comp["grounding"]
        synthesis, self_rep = comp["synthesis"], comp["self_rep"]

        r_think_raw = (
            _W_TOOL * tool_use
            + _W_GROUND * grounding
            + _W_SYNTH * synthesis
            - _W_REP * self_rep
        )
        shaping = max(-_CAP, min(_CAP, _SCALE * r_think_raw))

        # Asymmetric (LongPAS): a correct answer is never penalised for minor
        # repetition. "Correct" = the genuine tiered notion (rank <= 10).
        if r_answer >= 0.5:
            applied = max(0.0, shaping)
        else:
            applied = shaping

    r_total = r_dense + applied

    # Debug logging (~1 in 64).
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[Rthink-v4] r_ans={r_answer:.3f}  rank={rank_id}  N={n_items}  "
              f"r_dense={r_dense:.3f}  tool={tool_use:.3f}  grnd={grounding:.3f}  "
              f"syn={synthesis:.3f}  rep={self_rep:.3f}  shaping={applied:+.4f}  "
              f"R_total={r_total:.3f}  (p={_DENSE_P} floor={_RANK_FLOOR} "
              f"dense_only={_DENSE_ONLY} scale={_SCALE} cap={_CAP})")
        print(f"Solution string: {solution_str[:1200]}")

    return {
        "score": float(r_total),
        "r_answer": float(r_answer),
        "r_dense": float(r_dense),
        "rank": float(rank_id) if rank_id else -1.0,
        "r_think": float(applied),
        "tool_use": float(tool_use),
        "grounding": float(grounding),
        "synthesis": float(synthesis),
        "self_rep": float(self_rep),
    }
