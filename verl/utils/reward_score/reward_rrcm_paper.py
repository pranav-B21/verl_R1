"""Paper-protocol RRCM ranking reward.

This module is deliberately separate from ``reward_SPRec`` and the versioned
reasoning/retrieval rewards.  Enabling it cannot change historical runs.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from .reward_SPRec import _get_catalog, _get_model

RANK_CUTOFFS = (1, 5, 10, 50, 100)
RANK_WEIGHTS = (0.5, 0.3, 0.1, 0.08, 0.02)
PARSE_PENALTY = -1.0


def extract_single_quoted_answer(solution_str: str) -> str | None:
    """Return the title only for one well-formed, double-quoted answer."""

    if solution_str.count("<answer>") != 1 or solution_str.count("</answer>") != 1:
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)
    if len(matches) != 1:
        return None
    payload = matches[0].strip()
    if len(payload) < 3 or not payload.startswith('"') or not payload.endswith('"'):
        return None
    if payload.count('"') != 2:
        return None
    title = payload[1:-1].strip()
    return title or None


def ranking_reward(rank: int | None) -> float:
    """Weighted cumulative InTop@n reward from the RRCM paper."""

    if rank is None or rank < 1:
        return 0.0
    return float(sum(weight for cutoff, weight in zip(RANK_CUTOFFS, RANK_WEIGHTS) if rank <= cutoff))


def _catalog_rank(title: str, ground_truth: dict[str, Any], data_source: str) -> tuple[int | None, bool]:
    model = _get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings, name2id = _get_catalog(data_source, device)

    target_name = ground_truth["target"].strip().strip('"')
    if target_name not in name2id:
        return None, False

    predicted = torch.as_tensor(model.encode(title), device=device)
    if predicted.ndim == 1:
        predicted = predicted.unsqueeze(0)
    distances = torch.cdist(predicted, embeddings, p=2).squeeze(0)
    order = distances.argsort()
    target_id = name2id[target_name]
    position = (order == target_id).nonzero(as_tuple=False)
    if position.numel() == 0:
        return None, True
    return int(position.item()) + 1, True


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any],
    data_source: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return the exact ranking objective plus strict format diagnostics."""

    del extra_info
    title = extract_single_quoted_answer(solution_str)
    if title is None:
        return {
            "score": PARSE_PENALTY,
            "r_answer": 0.0,
            "rrcm_rank": -1.0,
            "format_ok": 0.0,
            "target_found": 0.0,
        }

    rank, target_found = _catalog_rank(title, ground_truth, data_source)
    answer_reward = ranking_reward(rank)
    return {
        "score": answer_reward,
        "r_answer": answer_reward,
        "rrcm_rank": float(rank) if rank is not None else -1.0,
        "format_ok": 1.0,
        "target_found": float(target_found),
    }

