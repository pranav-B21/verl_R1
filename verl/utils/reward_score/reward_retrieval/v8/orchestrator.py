# SPDX-License-Identifier: Apache-2.0
"""Retrieval reward orchestrator — iteration v8 ("selection reward").

What changed from v7b and why
-----------------------------
v7b confirmed its behavioral claim (retrieval 5% -> 100%, runaway tool-call
loops eliminated, answer rate ~88% -> ~99.5%) but was a null result on
retrieval *quality*: corrected GT-in-docs coverage 2.77% vs the baseline's
3.13% (p=0.34). Measuring the rest of the funnel explained why the outcome
never moved:

    HR@1 = coverage x P(pick GT | GT in docs) = ~3% x ~chance = ~0.001

Selection is at chance (baseline 0/57, v7b 2/53, against 3.8-5.4% chance).
Coverage work multiplies against ~0, so v8 attacks selection.

Three changes, all deliberately small so attribution stays possible:

  1. r_select (NEW, ../v8/selection.py) — credits the *structure* of the
     answer, not its identity: is it a continuation of something this user
     actually played, per the ordered sequences in the retrieved docs?
  2. r_cover replaces v7's soft-only r_retqual — exact GT-in-retrieved-titles
     gets full credit, with v7's cosine kept as a dense tail. v7's held-out
     `best_sim` rose 0.144 -> 0.227 (+58%) while binary coverage did not move
     at all: the reward was optimizing a functional the metric does not read.
  3. The length budget no longer counts <tool_call> bodies. v7b's queries/call
     is 1.47 vs the baseline's 2.26, and docs/ret ~= queries/call x topk, so
     charging for query text taxes retrieval breadth directly.

Kept verbatim from v7b (proven, orthogonal, and not worth re-litigating):
v6's HR-faithful outcome reward, v5's format-integrity gate, and the
turns-aware length budget with <tool_response> stripped.

PRE-REGISTERED EXPECTATION — read ./README.md before interpreting a run.
The successor cue holds the GT in 13% of retrievable cases, so v8's ceiling is
coverage 2.77% x 13% = HR@1 ~0.0036 (~3x baseline), still under the Table-1
HR@5 target of 0.0102. A larger HR jump than that is more likely a measurement
artifact than a win, and should be investigated before it is reported.

Hyperparameters (env vars, RTHINK_ prefix per the v2-v7 convention)
------------------------------------------------------------------
    RTHINK_W_SELECT      weight of the successor term                (default 0.5)
    RTHINK_W_GROUND      weight of the "answered a retrieved item" term (default 0.1)
    RTHINK_W_REPEAT      penalty weight for re-recommending a played item (default 0.3)
    RTHINK_W_COVER       weight of r_cover in the shaping sum        (default 0.4)
    RTHINK_COVER_TAIL    weight of the cosine tail inside r_cover    (default 0.3)
    RTHINK_LEN_COUNT_CALLS  1 -> count <tool_call> text in the budget (default 0)
    plus every v7b knob (RTHINK_HIT_K, _FORMAT_GATE, _LEN_*, _SCALE, _CAP,
    _RETQUAL_TAU, _RETRIEVAL_ONLY), unchanged in name and meaning.
"""

import math
import os
import random
import re

from . import selection
from ..v7 import retrieval as v7_retrieval


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


# --- v6 HR-faithful outcome reward (kept verbatim from v7b) ---
_HIT_K = _envi("RTHINK_HIT_K", 10)
_TAIL_W = _envf("RTHINK_TAIL_W", 0.10)
_TAIL_P = _envf("RTHINK_TAIL_P", 4.0)

# --- v5 format-integrity gate (kept verbatim) ---
_FORMAT_GATE = _envi("RTHINK_FORMAT_GATE", 1)
_FORMAT_PENALTY = _envf("RTHINK_FORMAT_PENALTY", 0.5)

# --- length discipline (v7b base; v8 also exempts the query text) ---
_LEN_SOFT = _envi("RTHINK_LEN_SOFT", 600)
_LEN_W = _envf("RTHINK_LEN_W", 0.0005)
_LEN_CAP = _envf("RTHINK_LEN_CAP", 0.2)
_LEN_PER_TURN = _envi("RTHINK_LEN_PER_TURN", 400)
_LEN_TURN_CAP = _envi("RTHINK_LEN_TURN_CAP", 3)
_LEN_COUNT_CALLS = _envi("RTHINK_LEN_COUNT_CALLS", 0)

# --- shaping scale/cap (kept verbatim from v6/v7) ---
_SCALE = _envf("RTHINK_SCALE", 0.10)
_CAP = _envf("RTHINK_CAP", 0.08)
_RETRIEVAL_ONLY = _envi("RTHINK_RETRIEVAL_ONLY", 0)

# --- v8 selection shaping (NEW) ---
_W_SELECT = _envf("RTHINK_W_SELECT", 0.5)
_W_GROUND = _envf("RTHINK_W_GROUND", 0.1)
_W_REPEAT = _envf("RTHINK_W_REPEAT", 0.3)

# --- v8 coverage shaping (replaces v7's soft-only retqual) ---
_W_COVER = _envf("RTHINK_W_COVER", 0.4)
_COVER_TAIL = _envf("RTHINK_COVER_TAIL", 0.3)
_RETQUAL_TAU = _envf("RTHINK_RETQUAL_TAU", 0.20)

_TOOL_RESPONSE_SPAN = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
_TOOL_CALL_SPAN = re.compile(r"<tool_call>.*?</tool_call>", re.S)
_DOC_MARKER = re.compile(r"Doc\s+\d+\s+\(Title:")


def _ndcg_reward(rank_id, n_items) -> float:
    """HR-faithful outcome reward — identical to v6/v7 (kept verbatim)."""
    if not rank_id or not n_items or n_items <= 1:
        return 0.0
    r = min(max(int(rank_id), 1), int(n_items))
    if r <= _HIT_K:
        return 1.0 / math.log2(r + 1)
    base = 1.0 - (math.log(r) / math.log(n_items))
    base = min(1.0, max(0.0, base))
    return _TAIL_W * (base ** _TAIL_P)


def _retrieved_spans(text: str):
    """Inner text of each <tool_response>, bounded by its closing tag."""
    return _TOOL_RESPONSE_SPAN.findall(text)


def _policy_generated_text(text: str) -> str:
    """Words the length budget is allowed to charge for.

    v7b removed the retriever's returned documents (they collapsed retrieval in
    M3 -- see ../v7/orchestrator.py). v8 additionally removes the <tool_call>
    bodies: docs/ret ~= queries/call x topk, so every word of query text the
    budget charges for is direct pressure to issue fewer queries and retrieve
    less. Measured consequence in v7b: queries/call 1.47 vs the baseline's 2.26
    at an identical 100% retrieval rate. Set RTHINK_LEN_COUNT_CALLS=1 to restore
    the v7b behavior for an ablation.
    """
    stripped = _TOOL_RESPONSE_SPAN.sub(" ", text)
    if not _LEN_COUNT_CALLS:
        stripped = _TOOL_CALL_SPAN.sub(" ", stripped)
    return stripped


def _credited_turns(text: str) -> int:
    """Turns that RETURNED documents; empty/failed responses buy no budget."""
    return sum(1 for span in _retrieved_spans(text) if _DOC_MARKER.search(span))


def _length_penalty(text: str) -> float:
    if _LEN_W <= 0 or _LEN_CAP <= 0:
        return 0.0
    budget = _LEN_SOFT + _LEN_PER_TURN * min(_credited_turns(text), _LEN_TURN_CAP)
    n_words = len(_policy_generated_text(text).split())
    return min(_LEN_CAP, _LEN_W * max(0, n_words - budget))


def _cover_reward(solution_str: str, ground_truth: dict, target: str) -> tuple:
    """Graded GT-in-retrieved-docs credit.

    Full credit when the GT title is EXACTLY one of the retrieved items -- the
    quantity the held-out dev metric actually measures
    (audits/decode_behavior.py). Falls back to v7's one-sided cosine tail so the
    term still has a gradient on the ~97% of rollouts that never retrieve the
    GT; without that tail the signal is binary at a ~3% base rate, which at
    rollout.n=8 leaves most GRPO groups all-zero and gradient-free (the failure
    mode v6 hit).
    """
    spans = _retrieved_spans(solution_str)
    if not spans:
        return 0.0, 0.0, 0
    seqs = selection.doc_sequences(spans)
    items = {i for s in seqs for i in s}
    if target and selection.norm_title(target) in items:
        return 1.0, 1.0, len(items)

    comp = v7_retrieval.compute_retrieval_components(
        solution_str, ground_truth, _RETQUAL_TAU, 0.0,
    )
    return _COVER_TAIL * float(comp["retqual_agg"]), float(comp["best_sim"]), len(items)


def compute_score(solution_str: str, ground_truth: dict, data_source: str,
                  method: str = "strict", format_score: float = 0.0,
                  score: float = 1.0, extra_info: dict = None) -> dict:
    """Drop-in replacement for reward_SPRec.compute_score (returns a dict)."""
    from ... import reward_SPRec

    outcome = reward_SPRec.compute_score(
        solution_str, ground_truth, data_source,
        method=method, format_score=format_score, score=score,
        return_rank=True,
    )
    rank_id, n_items = outcome["rankId"], outcome["N"]
    target_found = outcome["target_found"]

    open_count, close_count = reward_SPRec.count_answer_tags(solution_str)
    well_formed = (open_count == 1 and close_count == 1)

    if not target_found:
        r_answer = r_outcome = 0.0
    else:
        r_answer = float(outcome["match"])
        r_outcome = _ndcg_reward(rank_id, n_items)

    format_penalty = 0.0
    if _FORMAT_GATE and not well_formed:
        r_outcome = 0.0
        format_penalty = _FORMAT_PENALTY

    len_penalty = _length_penalty(solution_str)

    # ------------------------------------------------------------------ #
    # v8 shaping: selection (NEW) + graded coverage (replaces retqual).   #
    # Never touches the corpus -- ground_truth["target"] only SCORES what #
    # the rollout already retrieved, exactly as in v7.                    #
    # ------------------------------------------------------------------ #
    target = (ground_truth or {}).get("target", "").strip().strip('"')
    r_cover = best_sim = 0.0
    is_succ = is_grnd = is_rep = 0.0
    n_succ = n_items_ret = 0
    applied = 0.0

    if not _RETRIEVAL_ONLY:
        r_cover, best_sim, n_items_ret = _cover_reward(solution_str, ground_truth, target)

        pred_title = reward_SPRec.extract_solution(solution_str) or ""
        m = re.search(r'"([^"]*)', pred_title)
        pred_title = m.group(1) if m else pred_title
        sel = selection.compute_selection_components(
            _retrieved_spans(solution_str),
            (extra_info or {}).get("prompt_str", ""),
            pred_title,
        )
        is_succ, is_grnd, is_rep = sel["is_successor"], sel["is_grounded"], sel["is_repeat"]
        n_succ = sel["n_successors"]

        r_select = _W_SELECT * is_succ + _W_GROUND * is_grnd - _W_REPEAT * is_rep
        raw = r_select + _W_COVER * r_cover
        applied = max(-_CAP, min(_CAP, _SCALE * raw))

    # LongPAS asymmetry: a correct AND well-formed answer is never punished.
    if r_answer >= 0.5:
        applied = max(0.0, applied)
        format_penalty = 0.0
        len_penalty = 0.0

    penalty = format_penalty + len_penalty
    r_total = r_outcome + applied - penalty

    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[Rthink-v8] r_ans={r_answer:.3f}  rank={rank_id}  N={n_items}  "
              f"r_out={r_outcome:.3f}  fmt_ok={int(well_formed)}  "
              f"fmt_pen={format_penalty:.3f}  len_pen={len_penalty:.3f}  "
              f"cover={r_cover:.3f}  best_sim={best_sim:.3f}  "
              f"succ={int(is_succ)}  grnd={int(is_grnd)}  rep={int(is_rep)}  "
              f"n_succ={n_succ}  n_items={n_items_ret}  "
              f"shaping={applied:+.4f}  R_total={r_total:.3f}  "
              f"(w_sel={_W_SELECT} w_grnd={_W_GROUND} w_rep={_W_REPEAT} "
              f"w_cov={_W_COVER} count_calls={_LEN_COUNT_CALLS})")
        print(f"Solution string: {solution_str[:1200]}")

    return {
        "score": float(r_total),
        "r_answer": float(r_answer),
        "r_dense": float(r_outcome),
        "rank": float(rank_id) if rank_id else -1.0,
        "r_think": float(applied),
        "format_ok": float(well_formed),
        "format_penalty": float(format_penalty),
        "len_penalty": float(len_penalty),
        # v8 dev metrics — these are the ones to watch in WandB, not `score`
        "cover": float(r_cover),
        "best_sim": float(best_sim),
        "is_successor": float(is_succ),
        "is_grounded": float(is_grnd),
        "is_repeat": float(is_rep),
        "n_successors": float(n_succ),
        "n_retrieved_items": float(n_items_ret),
    }
