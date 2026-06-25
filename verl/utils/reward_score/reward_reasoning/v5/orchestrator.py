"""Combined Reasoning Reward orchestrator — iteration v5 ("format-gated dense reward").

    r_dense  = ( 1 - log(rankId) / log(N) ) ** p            # N = catalog size, p >= 1 (v5 default 2.0)
    r_dense  = 0                if the output is MALFORMED   # v5 anti-hack gate
    shaping  = clip( SCALE * R_think_raw , -CAP, +CAP )      # identical to v3/v4
    penalty  = format_penalty (malformed) + length_penalty   # v5, LongPAS-gated
    R_total  = r_dense + applied_shaping - penalty

Why v5 (the problem v4 left open)
---------------------------------
v4 made the outcome reward *dense* (fixing v3's flat-zero gradient) and trained
stably, but the held-out curve **peaks ~step 250 then declines below its start**
(REWARD_REASONING_ANALYSIS.md Part 3.3 / Part 4). Two concrete causes, found in the
v4 @ step-250 predictions (TEST_OUTPUT.md):

  1. REWARD HACK — 73.9% of outputs emit MULTIPLE `<answer>` tags (vs 0% for
     baseline and v3). Root cause: the base multi-answer penalty (`match/4` /
     `-0.5`) lives on the float `match` in `reward_SPRec.similarity_match`, but v4
     trains on `r_dense = f(rankId)`, computed from the *last* `<answer>` and never
     reading `match`. So the only formatting discipline in the pipeline is
     BYPASSED — emitting many `<answer>` tags is free under the dense reward, and
     the policy drifts into that degenerate region after ~step 250.

  2. VERBOSITY INFLATION — median response 1203 words (vs 754 baseline / 974 v3),
     the longest of any run. The dense reward exerts no length pressure.

Both inflate the temp-1 *train* reward while degrading greedy held-out top-k — the
F9 train↔val gap, amplified by an unguarded reward.

What v5 changes (three things; everything else is v4)
-----------------------------------------------------
  A. FORMAT-INTEGRITY GATE (the core fix). The dense reward is the proxy for
     correctness, so it must only be paid on a *well-formed* answer. v5 zeroes
     `r_dense` and subtracts `RTHINK_FORMAT_PENALTY` whenever the output is not
     exactly one `<answer>...</answer>` (multi-answer spam OR no answer). This
     carries the base reward's multi-answer penalty into the term v4 actually
     trains on, killing the loophole at its source.

  B. LENGTH DISCIPLINE. A gentle, capped penalty for responses past a soft word
     budget (`RTHINK_LEN_SOFT`), so the dense reward can't be farmed with rambling
     `<think>`. Capped at `RTHINK_LEN_CAP` so it can never overpower correctness.

  C. HR-TARGETED DENSE DEFAULT. v5 defaults `RTHINK_DENSE_P=2.0` (was 1.0). HR
     cares only about the extreme top of the rank distribution; squaring the curve
     concentrates reward near the top (rank 10: 0.76->0.57, 100: 0.51->0.26) while
     STAYING NONZERO EVERYWHERE — so it targets top-k without bringing back v4's
     dead-gradient (which a hard rank floor would). `RTHINK_RANK_FLOOR` remains
     available (default off) for the harder-targeting ablation.

LongPAS (kept from v3/v4): a genuinely correct *and well-formed* answer
(`r_answer >= 0.5`, only reachable when there is exactly one answer at rank <= 5,
since multi-answer divides `match` by 4 to < 0.5) is never penalised — positive
shaping only, no format/length penalty.

Logging & comparability (kept from v4)
--------------------------------------
The dict still logs the **tiered** `r_answer` separately from `score`, so val
`r_answer/mean@1` stays directly comparable to the baseline's `reward/mean@1` (the
north-star). `r_dense`, `rank`, the applied shaping, the process components, and
the new `format_ok` / `len_penalty` are all logged for auditing.

Hyperparameters (env vars, all optional)
----------------------------------------
    RTHINK_DENSE_P       sharpness exponent p on r_dense            (v5 default 2.0)
    RTHINK_RANK_FLOOR    zero r_dense when rankId > floor (0=off)   (default 0)
    RTHINK_DENSE_ONLY    1 -> disable process shaping (ablation)    (default 0)
    RTHINK_FORMAT_GATE   1 -> enable the malformed-output gate      (default 1)
    RTHINK_FORMAT_PENALTY  penalty subtracted for malformed output  (default 0.5)
    RTHINK_LEN_SOFT      word budget before the length penalty      (default 600)
    RTHINK_LEN_W         penalty per word over the budget           (default 0.0005)
    RTHINK_LEN_CAP       max length penalty                         (default 0.2)
    RTHINK_SCALE/_CAP/_W_*  process-shaping knobs (only when DENSE_ONLY=0; v3 defaults)
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


# --- dense reward (v5 default p=2.0 to target the top of the rank distribution) ---
_DENSE_P    = _envf("RTHINK_DENSE_P",    2.0)
_RANK_FLOOR = _envi("RTHINK_RANK_FLOOR", 0)
_DENSE_ONLY = _envi("RTHINK_DENSE_ONLY", 0)

# --- v5 format-integrity gate (the anti-hack core) ---
_FORMAT_GATE    = _envi("RTHINK_FORMAT_GATE",    1)
_FORMAT_PENALTY = _envf("RTHINK_FORMAT_PENALTY", 0.5)

# --- v5 length discipline ---
_LEN_SOFT = _envi("RTHINK_LEN_SOFT", 600)
_LEN_W    = _envf("RTHINK_LEN_W",    0.0005)
_LEN_CAP  = _envf("RTHINK_LEN_CAP",  0.2)

# --- v3 process shaping (only when DENSE_ONLY=0) ---
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


def _length_penalty(text: str) -> float:
    """Gentle, capped penalty for responses past the soft word budget."""
    if _LEN_W <= 0 or _LEN_CAP <= 0:
        return 0.0
    n_words = len(text.split())
    excess = max(0, n_words - _LEN_SOFT)
    return min(_LEN_CAP, _LEN_W * excess)


def compute_score(solution_str: str, ground_truth: dict, data_source: str,
                  method: str = "strict", format_score: float = 0.0,
                  score: float = 1.0, extra_info: dict = None) -> dict:
    """Drop-in replacement for reward_SPRec.compute_score (returns a dict).

    The reward manager uses ``score["score"]`` as the reward and logs the rest:
    ``r_answer`` (tiered, baseline-comparable), ``r_dense``, ``rank``,
    ``r_think`` (applied shaping), ``format_ok``, ``len_penalty`` and the four
    process components.
    """
    # ------------------------------------------------------------------ #
    # Outcome — tiered r_answer (baseline-comparable) + raw rank.        #
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

    # v5 format integrity: a well-formed answer has EXACTLY one <answer>...</answer>.
    # 0 tags = no answer; >1 = the multi-answer reward hack. Either is malformed.
    open_count, close_count = reward_SPRec.count_answer_tags(solution_str)
    well_formed = (open_count == 1 and close_count == 1)

    # F9: target missing from name2id -> the tiered match is a spurious rank-1
    # off item-0; drop it (and the dense reward) to 0 instead of leaking reward.
    if not target_found:
        r_answer = 0.0
        r_dense = 0.0
    else:
        r_answer = float(outcome["match"])
        r_dense = _dense_from_rank(rank_id, n_items)

    # ------------------------------------------------------------------ #
    # v5 FORMAT-INTEGRITY GATE — the anti-hack core.                     #
    # The dense reward is the correctness proxy v4 actually trains on, so #
    # it must only be paid on a well-formed answer. Malformed output also #
    # carries an explicit penalty so the hack is actively discouraged.   #
    # ------------------------------------------------------------------ #
    format_penalty = 0.0
    if _FORMAT_GATE and not well_formed:
        r_dense = 0.0
        format_penalty = _FORMAT_PENALTY

    # v5 length discipline.
    len_penalty = _length_penalty(solution_str)

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
        applied = max(-_CAP, min(_CAP, _SCALE * r_think_raw))

    # ------------------------------------------------------------------ #
    # LongPAS asymmetry: a genuinely correct *and well-formed* answer     #
    # (r_answer >= 0.5 is only reachable with exactly one answer at rank  #
    # <= 5, since multi-answer divides match by 4 to < 0.5) is never      #
    # punished — positive shaping only, no format/length penalty.         #
    # ------------------------------------------------------------------ #
    if r_answer >= 0.5:
        applied = max(0.0, applied)
        format_penalty = 0.0
        len_penalty = 0.0

    penalty = format_penalty + len_penalty
    r_total = r_dense + applied - penalty

    # Debug logging (~1 in 64).
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[Rthink-v5] r_ans={r_answer:.3f}  rank={rank_id}  N={n_items}  "
              f"r_dense={r_dense:.3f}  fmt_ok={int(well_formed)}  "
              f"fmt_pen={format_penalty:.3f}  len_pen={len_penalty:.3f}  "
              f"tool={tool_use:.3f}  grnd={grounding:.3f}  syn={synthesis:.3f}  "
              f"rep={self_rep:.3f}  shaping={applied:+.4f}  R_total={r_total:.3f}  "
              f"(p={_DENSE_P} floor={_RANK_FLOOR} dense_only={_DENSE_ONLY} "
              f"gate={_FORMAT_GATE} fpen={_FORMAT_PENALTY} len_soft={_LEN_SOFT})")
        print(f"Solution string: {solution_str[:1200]}")

    return {
        "score": float(r_total),
        "r_answer": float(r_answer),
        "r_dense": float(r_dense),
        "rank": float(rank_id) if rank_id else -1.0,
        "r_think": float(applied),
        "format_ok": float(well_formed),
        "format_penalty": float(format_penalty),
        "len_penalty": float(len_penalty),
        "tool_use": float(tool_use),
        "grounding": float(grounding),
        "synthesis": float(synthesis),
        "self_rep": float(self_rep),
    }
