# SPDX-License-Identifier: Apache-2.0
"""Combined Reasoning Reward orchestrator — iteration v7 ("retrieval-quality reward").

Why v7 (the PI's override of the v2-v6 reasoning-shaping line)
----------------------------------------------------------------
v6 made the outcome reward faithful to held-out HR/NDCG but still only tied
baseline (see ./ROADMAP_v7.md). A follow-on diagnosis concluded the system
was retrieval-bound and prescribed restructuring the retrieval corpus itself
(``cf_corpus/``) to raise how often the ground-truth item is even reachable.
The PI reviewed that diagnosis and gave the opposite call:

  1. Don't change the corpus. The observed ~4.6% GT-in-retrieved-docs rate on
     held-out data is CORRECT, not a bug — leave the ground-truth item out of
     the retrieval corpus.
  2. Build a reward for retrieval, not reasoning: score how similar each
     retrieval's results are to the correct answer, reward good retrieval,
     penalize junk. Teach the policy WHEN to retrieve and HOW to query.

v7 is that literal proposal. It keeps v6's outcome reward, format-integrity
gate, and length discipline VERBATIM (all proven, all orthogonal to this
change), and replaces v3's process-quality shaping (tool_use / grounding /
synthesis / self_rep — reasoning-quality proxies) with retrieval-quality
shaping computed directly from what the rollout actually retrieved:

    r_outcome = v6's HR-faithful top-K outcome reward                 (unchanged)
    r_retqual = per-turn cosine(best retrieved doc, GT answer)         (see retrieval.py)
    r_covgain = credit for a LATER turn beating the running-best sim   (see retrieval.py)
    shaping   = clip( SCALE * (W_RETQUAL*r_retqual + W_COVGAIN*r_covgain), -CAP, +CAP )
    R_total   = r_outcome + real_bonus + applied_shaping - penalty

CRITICAL: this reward never reads or writes the retrieval corpus
(data/amazon_data/corpora.jsonl / e5_Flat.index). ``ground_truth["target"]``
is used only to SCORE documents the model already retrieved during its own
rollout (recovered from ``solution_str``) — never as model-visible input, and
never inserted into any corpus file. See ./ROADMAP_v7.md and
../retrieval_reward_design.md for the full rationale and the research this is
grounded in (exact tag formats, embedding reuse, dispatch contract).

Hyperparameters (env vars, all optional; RTHINK_ prefix per the existing v2-v6
convention -- NOT the R_*/RET_* naming sketched in the now-superseded parts of
ROADMAP_v7.md)
----------------------------------------------------------------------------
    RTHINK_HIT_K            top-K window scored as NDCG (mirrors eval HR@K)   (default 10)
    RTHINK_TAIL_W/_TAIL_P   weak tail-guidance term, kept verbatim from v6    (defaults 0.10 / 4.0)
    RTHINK_REAL_BONUS       exact-catalog-item bonus, kept verbatim from v6   (default 0.0)
    RTHINK_FORMAT_GATE/_FORMAT_PENALTY   kept verbatim from v5                (defaults 1 / 0.5)
    RTHINK_LEN_SOFT/_LEN_W/_LEN_CAP      kept verbatim from v5                (defaults 600 / 0.0005 / 0.2)
    RTHINK_RETQUAL_TAU      neutral similarity threshold (cosine, [-1,1])     (default 0.30)
    RTHINK_RETQUAL_FLOOR    damping on the below-tau penalty (in [0,1];      (default 0.25)
                            0 = no penalty for junk, 1 = symmetric penalty)
    RTHINK_W_RETQUAL        weight of r_retqual in the shaping sum            (default 0.6)
    RTHINK_W_COVGAIN        weight of r_covgain in the shaping sum            (default 0.4)
    RTHINK_SCALE/_CAP       overall shaping scale/cap, kept verbatim from v6  (defaults 0.10 / 0.08)
    RTHINK_RETRIEVAL_ONLY   1 -> disable retrieval shaping (ablation: outcome-only, mirrors v6 DENSE_ONLY) (default 0)
"""

import math
import os
import random
import re

from . import retrieval


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


# --- v6 HR-faithful outcome reward (kept verbatim) ---
_HIT_K   = _envi("RTHINK_HIT_K",  10)
_TAIL_W  = _envf("RTHINK_TAIL_W", 0.10)
_TAIL_P  = _envf("RTHINK_TAIL_P", 4.0)

# --- v6 optional decisiveness bonus (kept verbatim, default OFF) ---
_REAL_BONUS = _envf("RTHINK_REAL_BONUS", 0.0)

# --- v5 format-integrity gate (kept verbatim) ---
_FORMAT_GATE    = _envi("RTHINK_FORMAT_GATE",    1)
_FORMAT_PENALTY = _envf("RTHINK_FORMAT_PENALTY", 0.5)

# --- length discipline (v5 base; v7b multi-turn-aware budget) ---
# v7b: the word budget grows with the number of retrieval turns that actually
# returned documents, because a legitimate retrieving rollout reasons twice
# (before AND after retrieval) and so needs more model-generated words than an
# abstaining one. Sizing (REWARD_REASONING_ANALYSIS.md Part 8 / audits):
#   abstain (0 turns): ~239 median gen-words (docs stripped), p90 ~402 -> the
#     600 base already covers it with headroom.
#   +1 credited turn : jumps to ~999 median / ~1566 p90 -> needs the per-turn add.
# Only turns that RETURNED docs are credited (a bare <tool_call> buys no budget,
# so tool-call spam can't purchase length), capped so a few turns can't unlock
# unlimited budget. LEN_PER_TURN is sized empirically, not to the raw marginal
# (that would re-create the "retrieve once, ramble free" cliff of the exempt
# option) -- see the increment discussion in Part 8. Set via env at launch.
_LEN_SOFT     = _envi("RTHINK_LEN_SOFT", 600)
_LEN_W        = _envf("RTHINK_LEN_W",    0.0005)
_LEN_CAP      = _envf("RTHINK_LEN_CAP",  0.2)
_LEN_PER_TURN = _envi("RTHINK_LEN_PER_TURN", 400)   # budget += this * credited turns
_LEN_TURN_CAP = _envi("RTHINK_LEN_TURN_CAP", 3)     # max credited turns that add budget

# --- v7 retrieval-quality shaping (NEW; replaces v3 process shaping) ---
_RETRIEVAL_ONLY = _envi("RTHINK_RETRIEVAL_ONLY", 0)
_RETQUAL_TAU    = _envf("RTHINK_RETQUAL_TAU",   0.30)
_RETQUAL_FLOOR  = _envf("RTHINK_RETQUAL_FLOOR", 0.25)
_W_RETQUAL      = _envf("RTHINK_W_RETQUAL",     0.6)
_W_COVGAIN      = _envf("RTHINK_W_COVGAIN",     0.4)
_SCALE          = _envf("RTHINK_SCALE",         0.10)
_CAP            = _envf("RTHINK_CAP",           0.08)


def _ndcg_reward(rank_id, n_items) -> float:
    """HR-faithful outcome reward — identical to v6 (kept verbatim)."""
    if not rank_id or not n_items or n_items <= 1:
        return 0.0
    r = min(max(int(rank_id), 1), int(n_items))
    if r <= _HIT_K:
        return 1.0 / math.log2(r + 1)
    base = 1.0 - (math.log(r) / math.log(n_items))
    base = min(1.0, max(0.0, base))
    return _TAIL_W * (base ** _TAIL_P)


# Tool output injected into the rollout by the retriever. It must NOT count
# toward the length budget: the retriever returns 3-7 documents (~50-100 words
# each) per <tool_response>, so counting it penalizes the policy for the tool's
# verbosity, not its own. In v7 (M3) that inflated every retrieving rollout past
# the 600-word budget, saturating len_penalty at its cap while the retrieval
# bonus was only ~+0.004 -> retrieving became net -0.2 vs abstaining's 0 and
# GRPO drove retrieval to extinction (retr% 99% -> 5%). See ROADMAP_v7.md and
# REWARD_REASONING_ANALYSIS.md Part 8. The gate/self-repetition penalties are
# tag-based, not length-based, so only this term had the ingestion bug.
_TOOL_RESPONSE_SPAN = re.compile(r"<tool_response>.*?</tool_response>", re.S)
# A <tool_response> that actually returned documents (vs. an empty result) --
# _passages2string emits "Doc N (Title: ...)" per returned passage. Only these
# credit budget, so a bare/empty tool call cannot purchase length.
_DOC_MARKER = re.compile(r"Doc\s+\d+\s+\(Title:")


def _policy_generated_text(text: str) -> str:
    """Strip tool-injected <tool_response> spans so the length budget scores only
    what the policy actually generated (<think>/<tool_call>/<answer>)."""
    return _TOOL_RESPONSE_SPAN.sub(" ", text)


def _credited_turns(text: str) -> int:
    """Number of retrieval turns that RETURNED documents (empty/failed tool
    responses do not count -- otherwise tool-call spam would buy length budget)."""
    return sum(1 for m in _TOOL_RESPONSE_SPAN.finditer(text)
               if _DOC_MARKER.search(m.group(0)))


def _length_penalty(text: str) -> float:
    """Gentle, capped penalty for responses past a turns-aware word budget.

    v7b changes vs v5: (1) counts only policy-generated words -- the retriever's
    returned documents are excluded (they inflate every retrieving rollout past
    the budget and collapsed retrieval in M3); (2) the budget grows by
    ``_LEN_PER_TURN`` for each turn that returned docs (capped at
    ``_LEN_TURN_CAP``), because a legitimate multi-turn rollout reasons before
    AND after each retrieval. Net effect: abstaining and legitimately-retrieving
    rollouts are treated even-handedly, while genuine rambling (beyond the
    per-turn budget) is still penalized.
    """
    if _LEN_W <= 0 or _LEN_CAP <= 0:
        return 0.0
    budget = _LEN_SOFT + _LEN_PER_TURN * min(_credited_turns(text), _LEN_TURN_CAP)
    n_words = len(_policy_generated_text(text).split())
    excess = max(0, n_words - budget)
    return min(_LEN_CAP, _LEN_W * excess)


_NAME_KEYS_CACHE = {}


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().strip('"').lower())


def _catalog_keys(data_source: str):
    if _REAL_BONUS <= 0:
        return None
    if "goodreads" in data_source:
        path = "./data/goodreads_data/Goodreads/name2id.json"
    else:
        path = "./data/amazon_data/CDs_and_Vinyl/name2id.json"
    if path not in _NAME_KEYS_CACHE:
        try:
            import json
            with open(path) as f:
                _NAME_KEYS_CACHE[path] = {_norm_title(k) for k in json.load(f).keys()}
        except (OSError, ValueError):
            _NAME_KEYS_CACHE[path] = set()
    return _NAME_KEYS_CACHE[path]


def _decisiveness_bonus(solution_str: str, data_source: str) -> float:
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

    Identical plumbing to v6; the only substantive difference is the process
    shaping is now retrieval-quality (r_retqual / r_covgain) instead of v3's
    reasoning-quality proxies.
    """
    # ------------------------------------------------------------------ #
    # Outcome — tiered r_answer (baseline-comparable) + raw rank.        #
    # (identical to v6, kept verbatim)                                   #
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

    open_count, close_count = reward_SPRec.count_answer_tags(solution_str)
    well_formed = (open_count == 1 and close_count == 1)

    if not target_found:
        r_answer = 0.0
        r_outcome = 0.0
    else:
        r_answer = float(outcome["match"])
        r_outcome = _ndcg_reward(rank_id, n_items)

    format_penalty = 0.0
    if _FORMAT_GATE and not well_formed:
        r_outcome = 0.0
        format_penalty = _FORMAT_PENALTY

    len_penalty = _length_penalty(solution_str)
    real_bonus = _decisiveness_bonus(solution_str, data_source) if well_formed else 0.0

    # ------------------------------------------------------------------ #
    # v7 retrieval-quality shaping — replaces v3's process shaping.       #
    # Never touches the corpus; only scores docs the rollout retrieved.   #
    # ------------------------------------------------------------------ #
    retqual_agg = covgain_agg = 0.0
    n_turns = n_docs = 0
    best_sim = 0.0
    applied = 0.0
    if not _RETRIEVAL_ONLY:
        comp = retrieval.compute_retrieval_components(
            solution_str, ground_truth, _RETQUAL_TAU, _RETQUAL_FLOOR,
        )
        retqual_agg, covgain_agg = comp["retqual_agg"], comp["covgain_agg"]
        n_turns, n_docs, best_sim = comp["n_turns"], comp["n_docs"], comp["best_sim"]

        r_think_raw = _W_RETQUAL * retqual_agg + _W_COVGAIN * covgain_agg
        applied = max(-_CAP, min(_CAP, _SCALE * r_think_raw))

    # ------------------------------------------------------------------ #
    # LongPAS asymmetry: a genuinely correct *and well-formed* answer     #
    # (r_answer >= 0.5) is never punished — positive shaping only.        #
    # (identical to v6, kept verbatim)                                    #
    # ------------------------------------------------------------------ #
    if r_answer >= 0.5:
        applied = max(0.0, applied)
        format_penalty = 0.0
        len_penalty = 0.0

    penalty = format_penalty + len_penalty
    r_total = r_outcome + real_bonus + applied - penalty

    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[Rthink-v7] r_ans={r_answer:.3f}  rank={rank_id}  N={n_items}  "
              f"r_out={r_outcome:.3f}  real={real_bonus:.3f}  fmt_ok={int(well_formed)}  "
              f"fmt_pen={format_penalty:.3f}  len_pen={len_penalty:.3f}  "
              f"n_turns={n_turns}  n_docs={n_docs}  best_sim={best_sim:.3f}  "
              f"retqual={retqual_agg:+.3f}  covgain={covgain_agg:+.3f}  "
              f"shaping={applied:+.4f}  R_total={r_total:.3f}  "
              f"(K={_HIT_K} tau={_RETQUAL_TAU} floor={_RETQUAL_FLOOR} "
              f"w_rq={_W_RETQUAL} w_cg={_W_COVGAIN} retrieval_only={_RETRIEVAL_ONLY})")
        print(f"Solution string: {solution_str[:1200]}")

    return {
        "score": float(r_total),
        "r_answer": float(r_answer),
        "r_dense": float(r_outcome),
        "real_bonus": float(real_bonus),
        "rank": float(rank_id) if rank_id else -1.0,
        "r_think": float(applied),
        "format_ok": float(well_formed),
        "format_penalty": float(format_penalty),
        "len_penalty": float(len_penalty),
        "retqual": float(retqual_agg),
        "covgain": float(covgain_agg),
        "n_turns": float(n_turns),
        "n_docs": float(n_docs),
        "best_sim": float(best_sim),
    }
