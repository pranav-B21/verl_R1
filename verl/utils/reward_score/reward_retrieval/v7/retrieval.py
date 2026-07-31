# SPDX-License-Identifier: Apache-2.0
"""v7 retrieval-quality components — the PI's literal proposal.

Score each retrieval the model made during a rollout by how similar the
retrieved documents are to the ground-truth answer. Reward good retrieval,
penalize junk. This does NOT touch the corpus (data/amazon_data/corpora.jsonl
/ e5_Flat.index) at all -- it only scores documents the model already
retrieved, recovered from ``solution_str``. See ./ROADMAP_v7.md and
../retrieval_reward_design.md for the full design rationale.

Tags parsed are the ones confirmed to actually appear in TRAINING rollouts
(not just eval): Qwen-native ``<tool_call>...</tool_call>`` (the query) is
followed by ``<tool_response>...</tool_response>`` (a JSON string of the form
``{"result": "Doc 1 (Title: ...)\\n...\\n\\nDoc 2 (Title: ...)\\n...\\n\\n---\\n..."}``,
built by ``verl/tools/utils/search_r1_like_utils.py::_passages2string`` — one
``\\n---\\n``-separated segment per query in that call's ``query_list``).

Two components:
  r_retqual : per-turn, how close the BEST retrieved doc is to the GT answer
              (ONE-SIDED as of 2026-07-25 per advisor: rewards a near-GT
              retrieval, gives 0 -- never a penalty -- for a below-tau one, so
              legitimate item-attribute retrieval isn't discouraged and the
              term cannot teach "never retrieve". See ./ROADMAP_v7.md sec 4.1).
  r_covgain : rewards a LATER turn only when it beats the running-best
              similarity of earlier turns -- targets the observed 99%
              single-query collapse without rewarding retrieval-spam.
"""

import json
import re
from typing import List, Optional, Tuple

import torch

from ...reward_SPRec_reasoning import _get_sentence_model

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
# Matches `_passages2string`'s "Doc N (Title: ...)\n<body>" blocks; a doc's
# body runs until the next "Doc N (Title: ..." header or end of string, which
# also correctly bounds it across a "\n---\n" query-segment separator.
#
# The title terminator must accept END-OF-STRING as well as "\n": upstream builds
# each doc as "Doc {i} (Title: {title})\n{body}\n\n" but the joined result is
# .strip()-ed, so the FINAL doc of every response ends at ")" with no trailing
# newline. Requiring a literal ")\n" therefore dropped exactly one doc — always
# the last — from 100% of responses (measured: 67% recall over 9,348 docs in
# 3 saved decodes; the Audit-B script's header-split parser saw 100%). Since
# `r_retqual` takes a max over a turn's docs, a dropped doc could only depress
# sim, and it meant the shipped term was not the term Audit B validated.
# Do NOT "simplify" this to r"\)\s*\n?": \s* eats the "\n\n\n" separator, after
# which the next-header lookahead can no longer match and each doc swallows its
# successor (measured: recall drops to 67%). Keep the alternation explicit — it
# also forces correct backtracking on titles that contain parentheses, e.g.
# "Greatest Hits (Deluxe Edition)".
DOC = re.compile(
    r"Doc\s+\d+\s+\(Title:\s*(.*?)\)(?:\n|\Z)(.*?)(?=\nDoc\s+\d+\s+\(Title:|\Z)", re.S
)


def _extract_turns(solution_str: str) -> List[Tuple[str, str]]:
    """Return [(tool_call_text, tool_response_text), ...] in rollout order.

    Calls and responses alternate 1:1 in a well-formed multi-turn rollout. If
    counts mismatch (a malformed/truncated turn), pair only the overlapping
    prefix rather than guessing at alignment.
    """
    calls = TOOL_CALL.findall(solution_str)
    responses = TOOL_RESPONSE.findall(solution_str)
    n = min(len(calls), len(responses))
    return list(zip(calls[:n], responses[:n]))


def _docs_in_response(tool_response_text: str) -> List[str]:
    """Recover retrieved documents' text from one <tool_response> block."""
    text = tool_response_text.strip()
    try:
        result = json.loads(text).get("result", "")
    except (json.JSONDecodeError, AttributeError):
        result = text  # tolerate malformed JSON, same spirit as coverage_sweep.py
    if not isinstance(result, str) or not result.strip():
        return []
    docs = []
    for m in DOC.finditer(result):
        title = m.group(1).strip()
        # strip a trailing "---" query-segment separator off the last doc
        # in a segment, left over from the "\n---\n".join(...) upstream
        body = re.sub(r"-{2,}\s*$", "", m.group(2)).strip()
        doc = f"{title}. {body}".strip(" .")
        if doc:
            docs.append(doc)
    return docs


def _best_similarity(docs: List[str], gt_emb: torch.Tensor, model, device) -> Optional[float]:
    """Cosine similarity of the GT answer to the closest retrieved doc."""
    if not docs:
        return None
    emb = model.encode(docs, convert_to_tensor=True, device=device)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    emb = torch.nn.functional.normalize(emb, dim=1)
    sims = (emb @ gt_emb).clamp(-1.0, 1.0)
    return float(sims.max())


def compute_retrieval_components(
    solution_str: str, ground_truth: dict, retqual_tau: float, retqual_floor: float
) -> dict:
    """Per-rollout retrieval-quality signal.

    The corpus is never touched here -- ``ground_truth["target"]`` is only
    ever used to SCORE documents the model already retrieved, never surfaced
    as model-visible input.

    Returns a dict:
      retqual_agg : mean per-turn r_retqual over turns that retrieved >=1 doc
                    (0.0 if the rollout never retrieved anything -- neutral,
                    not penalized, so "never retrieve" isn't inadvertently
                    made the safe policy)
      covgain_agg : sum of positive similarity improvements across turns
                    (0.0 with <=1 productive turn)
      n_turns, n_docs, best_sim : diagnostics for logging
    """
    target = (ground_truth or {}).get("target", "").strip().strip('"')
    turns = _extract_turns(solution_str)
    empty = {"retqual_agg": 0.0, "covgain_agg": 0.0, "n_turns": len(turns),
             "n_docs": 0, "best_sim": 0.0}
    if not target or not turns:
        return empty

    model = _get_sentence_model()
    device = getattr(model, "device", "cpu")
    gt_emb = torch.nn.functional.normalize(
        model.encode([target], convert_to_tensor=True, device=device), dim=1
    )[0]

    per_turn_sims, n_docs = [], 0
    for _, resp_text in turns:
        docs = _docs_in_response(resp_text)
        n_docs += len(docs)
        per_turn_sims.append(_best_similarity(docs, gt_emb, model, device))

    productive = [s for s in per_turn_sims if s is not None]
    if not productive:
        return empty

    def _retqual(sim: float) -> float:
        # ONE-SIDED (advisor Shijun Li, 2026-07-25): reward a near-GT retrieval,
        # give exactly 0 for a below-tau retrieval -- never penalize it.
        # Rationale: r_retqual scores only the COLLABORATIVE axis (cosine to the
        # GT *next item*). A below-tau retrieval is often a legitimate item-
        # ATTRIBUTE lookup (genre/artist/description), which this term cannot
        # credit; penalizing it would discourage attribute retrieval wholesale.
        # Removing the penalty also removes a retrieval-collapse pressure (M3).
        # `retqual_floor` is now INERT -- kept in the signature for back-compat.
        if sim >= retqual_tau:
            return (sim - retqual_tau) / max(1e-6, 1.0 - retqual_tau)
        return 0.0

    retqual_agg = sum(_retqual(s) for s in productive) / len(productive)

    covgain_agg, running_best = 0.0, None
    for s in per_turn_sims:
        if s is None:
            continue
        if running_best is not None:
            covgain_agg += max(0.0, s - running_best)
        running_best = s if running_best is None else max(running_best, s)

    return {
        "retqual_agg": float(retqual_agg),
        "covgain_agg": float(covgain_agg),
        "n_turns": len(turns),
        "n_docs": n_docs,
        "best_sim": float(max(productive)),
    }
