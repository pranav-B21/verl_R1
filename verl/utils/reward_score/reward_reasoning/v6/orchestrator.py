"""Combined Reasoning Reward orchestrator — iteration v6 ("HR-faithful, top-K reward").

    r_outcome = 1 / log2(rankId + 1)                 if rankId <= K   # mirrors eval.py NDCG@K
    r_outcome = TAIL_W * (1 - log(rankId)/log(N))**TAIL_P  otherwise   # weak, steep "get warmer"
    r_outcome = 0                if the output is MALFORMED            # v5 anti-hack gate (kept)
    shaping   = clip( SCALE * R_think_raw , -CAP, +CAP )               # v3 process terms (kept)
    penalty   = format_penalty (malformed) + length_penalty           # v5, LongPAS-gated (kept)
    R_total   = r_outcome + applied_shaping - penalty  (+ optional decisiveness bonus)

Why v6 (the problem v4/v5 left open)
------------------------------------
v5 closed v4's two HACKS (multi-answer spam, verbosity) and trained cleanly:
``format_ok`` -> 1.0, train ``r_answer`` -> ~0.38. But held-out it only TIES
baseline within decode noise (the same step-300 checkpoint re-decodes to HR@1
0.000 then 0.005; baseline HR@5 0.007, with random chance ~0.0004 — the metric
sits near its floor and per-decode variance spans the whole gap). The decisive
facts:

  1. eval.py computes held-out HR/NDCG with the EXACT SAME embedding-rank proxy as
     the training reward (same SentenceTransformer, same ``cdist``, same ``argsort``
     rank of the target). So the train<->val gap is NOT a metric mismatch.

  2. The gap is therefore decoding + the SHAPE of the reward. v4/v5 paid a large
     partial credit across the whole rank tail: ``(1 - log(r)/log(N))**2`` is ~0.11
     at rank 600 and ~0.26 at rank 100. HR@5 only scores rank <= 5. So the bulk of
     the "core reward" the policy was climbing lived in the MID-RANK region, which
     does not move HR. The model learned to land the target in a broad semantic
     neighborhood (rank ~hundreds) — "semantic mush" — without ever committing to a
     top-K item. That inflates temp-1 train reward while greedy held-out HR sits at
     the floor.

What v6 changes (one substantive thing; everything else is v5)
-------------------------------------------------------------
v6 makes the outcome reward a FAITHFUL surrogate of the metric we are judged on,
at the resolution we are judged on. It pays NDCG@K credit (``1/log2(rank+1)``,
exactly eval.py's term) inside the top-K window, and only a WEAK, STEEP tail term
outside it — enough to keep an all-far GRPO group from going fully dead, but far
too small to be farmed by mid-rank mush:

    | rank | v5 r_dense (p=2) | v6 r_outcome (K=10, tail_w=0.10, tail_p=4) |
    |------|------------------|--------------------------------------------|
    | 1    | 1.00             | 1.00                                       |
    | 5    | 0.66             | 0.39                                       |
    | 10   | 0.57             | 0.29                                       |
    | 100  | 0.26             | 0.0026                                     |
    | 600  | 0.11             | 0.00011                                    |
    | 3000 | 0.03             | ~0                                         |

The GRPO advantage is intra-group variance; v6 concentrates that variance where
HR lives (rank <= K) instead of spreading it across the mush region, so climbing
the reward MUST push targets toward the top-K — the only thing eval measures.

Optional decisiveness bonus (default OFF; the v6.1 lever for the greedy gap)
----------------------------------------------------------------------------
The residual train(temp-1)<->val(greedy) gap is consistent with a DIFFUSE policy:
temp-1 sampling occasionally lands a hit while the greedy mode is a bland hedge.
``RTHINK_REAL_BONUS`` (default 0) adds a small reward when the predicted title is
an EXACT catalog item (a key of name2id), pushing the policy to commit to concrete,
greedy-transferable items rather than embedding-central phrases. Off by default so
v6's headline stays a clean single-variable change from v5; turn it on to ablate.

Everything below — the format-integrity gate, the capped length penalty, the v3
process shaping, the LongPAS asymmetry, and the dict logging (r_answer stays the
baseline-comparable north-star) — is v5 verbatim.

Hyperparameters (env vars, all optional)
----------------------------------------
    RTHINK_HIT_K         top-K window scored as NDCG (mirrors eval HR@K)  (default 10)
    RTHINK_TAIL_W        weight of the weak tail-guidance term            (default 0.10)
    RTHINK_TAIL_P        steepness of the tail term (large => negligible) (default 4.0)
    RTHINK_REAL_BONUS    bonus if predicted title is an exact catalog key (default 0.0)
    RTHINK_DENSE_ONLY    1 -> disable process shaping (ablation)          (default 0)
    RTHINK_FORMAT_GATE   1 -> enable the malformed-output gate            (default 1)
    RTHINK_FORMAT_PENALTY  penalty subtracted for malformed output        (default 0.5)
    RTHINK_LEN_SOFT      word budget before the length penalty            (default 600)
    RTHINK_LEN_W         penalty per word over the budget                 (default 0.0005)
    RTHINK_LEN_CAP       max length penalty                               (default 0.2)
    RTHINK_SCALE/_CAP/_W_*  process-shaping knobs (only when DENSE_ONLY=0; v3 defaults)
"""

import json
import math
import os
import random
import re


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


# --- v6 HR-faithful outcome reward ---
_HIT_K   = _envi("RTHINK_HIT_K",  10)     # top-K window scored exactly like eval NDCG@K
_TAIL_W  = _envf("RTHINK_TAIL_W", 0.10)   # weight of the weak "get warmer" tail gradient
_TAIL_P  = _envf("RTHINK_TAIL_P", 4.0)    # steepness; large => mid-rank reward is negligible
_DENSE_ONLY = _envi("RTHINK_DENSE_ONLY", 0)

# --- v6 optional decisiveness bonus (default OFF; the v6.1 greedy-gap lever) ---
_REAL_BONUS = _envf("RTHINK_REAL_BONUS", 0.0)

# --- v5 format-integrity gate (the anti-hack core, kept verbatim) ---
_FORMAT_GATE    = _envi("RTHINK_FORMAT_GATE",    1)
_FORMAT_PENALTY = _envf("RTHINK_FORMAT_PENALTY", 0.5)

# --- v5 length discipline (kept verbatim) ---
_LEN_SOFT = _envi("RTHINK_LEN_SOFT", 600)
_LEN_W    = _envf("RTHINK_LEN_W",    0.0005)
_LEN_CAP  = _envf("RTHINK_LEN_CAP",  0.2)

# --- v3 process shaping (only when DENSE_ONLY=0; kept verbatim) ---
_SCALE    = _envf("RTHINK_SCALE",    0.10)
_CAP      = _envf("RTHINK_CAP",      0.08)
_W_TOOL   = _envf("RTHINK_W_TOOL",   0.30)
_W_GROUND = _envf("RTHINK_W_GROUND", 0.40)
_W_SYNTH  = _envf("RTHINK_W_SYNTH",  0.30)
_W_REP    = _envf("RTHINK_W_REP",    0.50)


def _ndcg_reward(rank_id, n_items) -> float:
    """HR-faithful outcome reward.

    Inside the top-K window: the exact eval.py NDCG term ``1/log2(rank+1)``
    (rank 1 -> 1.0, rank 5 -> 0.387, rank 10 -> 0.289). Outside it: a weak, steep
    tail (``TAIL_W * (1 - log(r)/log(N))**TAIL_P``) so an all-far GRPO group keeps a
    nonzero, monotonic "get warmer" gradient without paying mid-rank mush.
    """
    if not rank_id or not n_items or n_items <= 1:
        return 0.0
    r = min(max(int(rank_id), 1), int(n_items))
    if r <= _HIT_K:
        return 1.0 / math.log2(r + 1)
    base = 1.0 - (math.log(r) / math.log(n_items))
    base = min(1.0, max(0.0, base))
    return _TAIL_W * (base ** _TAIL_P)


def _length_penalty(text: str) -> float:
    """Gentle, capped penalty for responses past the soft word budget (v5)."""
    if _LEN_W <= 0 or _LEN_CAP <= 0:
        return 0.0
    n_words = len(text.split())
    excess = max(0, n_words - _LEN_SOFT)
    return min(_LEN_CAP, _LEN_W * excess)


# --- optional decisiveness bonus: is the predicted title a real catalog item? ---
_NAME_KEYS_CACHE = {}


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().strip('"').lower())


def _catalog_keys(data_source: str):
    """Module-cached set of normalized name2id keys for the active catalog."""
    if _REAL_BONUS <= 0:
        return None
    if "goodreads" in data_source:
        path = "./data/goodreads_data/Goodreads/name2id.json"
    else:
        path = "./data/amazon_data/CDs_and_Vinyl/name2id.json"
    if path not in _NAME_KEYS_CACHE:
        try:
            with open(path) as f:
                _NAME_KEYS_CACHE[path] = {_norm_title(k) for k in json.load(f).keys()}
        except (OSError, ValueError):
            _NAME_KEYS_CACHE[path] = set()
    return _NAME_KEYS_CACHE[path]


def _decisiveness_bonus(solution_str: str, data_source: str) -> float:
    """Small reward if the predicted title is an EXACT catalog item (commit, don't hedge)."""
    if _REAL_BONUS <= 0:
        return 0.0
    from ... import reward_SPRec
    title = reward_SPRec.extract_solution(solution_str)
    if not title:
        return 0.0
    m = re.search(r'"([^"]*)', title)
    pred = m.group(1) if m else title
    keys = _catalog_keys(data_source)
    return _REAL_BONUS if (keys and _norm_title(pred) in keys) else 0.0


def compute_score(solution_str: str, ground_truth: dict, data_source: str,
                  method: str = "strict", format_score: float = 0.0,
                  score: float = 1.0, extra_info: dict = None) -> dict:
    """Drop-in replacement for reward_SPRec.compute_score (returns a dict).

    Identical plumbing to v5; the only substantive difference is ``r_dense`` is now
    the HR-faithful top-K reward (``_ndcg_reward``) instead of v5's log-tail term.
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
    open_count, close_count = reward_SPRec.count_answer_tags(solution_str)
    well_formed = (open_count == 1 and close_count == 1)

    # F9: target missing from name2id -> spurious rank-1 off item-0; drop to 0.
    if not target_found:
        r_answer = 0.0
        r_outcome = 0.0
    else:
        r_answer = float(outcome["match"])
        r_outcome = _ndcg_reward(rank_id, n_items)

    # ------------------------------------------------------------------ #
    # v5 FORMAT-INTEGRITY GATE — the anti-hack core (kept verbatim).      #
    # ------------------------------------------------------------------ #
    format_penalty = 0.0
    if _FORMAT_GATE and not well_formed:
        r_outcome = 0.0
        format_penalty = _FORMAT_PENALTY

    # v5 length discipline.
    len_penalty = _length_penalty(solution_str)

    # v6 optional decisiveness bonus (only paid on a well-formed answer).
    real_bonus = _decisiveness_bonus(solution_str, data_source) if well_formed else 0.0

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
    # (r_answer >= 0.5) is never punished — positive shaping only.        #
    # ------------------------------------------------------------------ #
    if r_answer >= 0.5:
        applied = max(0.0, applied)
        format_penalty = 0.0
        len_penalty = 0.0

    penalty = format_penalty + len_penalty
    r_total = r_outcome + real_bonus + applied - penalty

    # Debug logging (~1 in 64).
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[Rthink-v6] r_ans={r_answer:.3f}  rank={rank_id}  N={n_items}  "
              f"r_out={r_outcome:.3f}  real={real_bonus:.3f}  fmt_ok={int(well_formed)}  "
              f"fmt_pen={format_penalty:.3f}  len_pen={len_penalty:.3f}  "
              f"tool={tool_use:.3f}  grnd={grounding:.3f}  syn={synthesis:.3f}  "
              f"rep={self_rep:.3f}  shaping={applied:+.4f}  R_total={r_total:.3f}  "
              f"(K={_HIT_K} tail_w={_TAIL_W} tail_p={_TAIL_P} real_bonus={_REAL_BONUS} "
              f"dense_only={_DENSE_ONLY} gate={_FORMAT_GATE})")
        print(f"Solution string: {solution_str[:1200]}")

    return {
        "score": float(r_total),
        "r_answer": float(r_answer),
        "r_dense": float(r_outcome),   # key kept for dashboard continuity (now the top-K reward)
        "real_bonus": float(real_bonus),
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
