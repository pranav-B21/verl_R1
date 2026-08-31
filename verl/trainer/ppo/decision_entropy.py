"""Utilities for entropy regularization at post-retrieval action decisions.

The RRCM extension intentionally does not reward retrieval, a memory type, or a
particular answer.  It only applies entropy at the point where a trajectory
that has already retrieved chooses between another ``<tool_call>`` and an
``<answer>``.  Keeping the mask construction here makes the behavior testable
without running a model forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DecisionActionStats:
    """Token mask and per-sample counts for recognized RRCM actions."""

    mask: torch.Tensor
    initial_tool_calls: torch.Tensor
    post_retrieval_tool_calls: torch.Tensor
    post_retrieval_answers: torch.Tensor
    action_token_ids: tuple[int, int]


def decision_entropy_coefficient(step: int, coefficient: float, hold_steps: int, decay_end_steps: int) -> float:
    """Return a hold-then-linear-decay coefficient.

    ``coefficient`` is used through ``hold_steps`` (inclusive), linearly
    decays to zero, and is exactly zero at and after ``decay_end_steps``.
    """

    if coefficient <= 0:
        return 0.0
    if hold_steps < 0:
        raise ValueError("hold_steps must be non-negative")
    if decay_end_steps <= hold_steps:
        raise ValueError("decay_end_steps must be greater than hold_steps")
    if step <= hold_steps:
        return float(coefficient)
    if step >= decay_end_steps:
        return 0.0
    remaining = (decay_end_steps - step) / (decay_end_steps - hold_steps)
    return float(coefficient) * float(remaining)


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    width = len(pattern)
    target = list(pattern)
    return [start for start in range(len(sequence) - width + 1) if list(sequence[start : start + width]) == target]


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    prefix = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix += 1
    if prefix >= min(len(left), len(right)):
        raise ValueError("tool-call and answer tag tokenizations must diverge")
    return prefix


def binary_action_entropy(logits: torch.Tensor, action_token_ids: Sequence[int]) -> torch.Tensor:
    """Entropy of the normalized retrieve/answer choice, excluding all other tokens."""

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if len(action_token_ids) != 2 or action_token_ids[0] == action_token_ids[1]:
        raise ValueError("action_token_ids must contain two distinct token ids")
    if min(action_token_ids) < 0 or max(action_token_ids) >= logits.shape[-1]:
        raise ValueError("action token id is outside the logits vocabulary")

    pair_logits = logits[..., list(action_token_ids)].float()
    log_probabilities = torch.log_softmax(pair_logits, dim=-1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1)


def build_post_retrieval_decision_stats(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tool_call_token_ids: Sequence[int],
    answer_token_ids: Sequence[int],
    tool_call_recognition_patterns: Sequence[Sequence[int]] | None = None,
    answer_recognition_patterns: Sequence[Sequence[int]] | None = None,
) -> DecisionActionStats:
    """Locate recognized actions and mask post-retrieval decision tokens.

    The first tool call creates the post-retrieval state and is deliberately
    excluded.  Every later recognized tool-call or answer tag is eligible.
    For each eligible action, only the first token where the two opening-tag
    tokenizations diverge is selected.  Tool-response and padding tokens remain
    excluded because every matched tag must lie wholly inside ``response_mask``.
    """

    if responses.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("responses and response_mask must both be rank-2 tensors")
    if responses.shape != response_mask.shape:
        raise ValueError("responses and response_mask must have identical shapes")
    if not tool_call_token_ids or not answer_token_ids:
        raise ValueError("action-tag tokenizations must be non-empty")

    divergence_offset = _common_prefix_length(tool_call_token_ids, answer_token_ids)
    tool_patterns = list(tool_call_recognition_patterns or [tool_call_token_ids])
    answer_patterns = list(answer_recognition_patterns or [answer_token_ids])
    if not tool_patterns or not answer_patterns or any(not pattern for pattern in tool_patterns + answer_patterns):
        raise ValueError("action recognition patterns must be non-empty")
    if any(len(pattern) <= divergence_offset for pattern in tool_patterns + answer_patterns):
        raise ValueError("each recognition pattern must include the divergent action token")
    batch_size, response_length = responses.shape
    decision_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    initial_tool_calls = torch.zeros(batch_size, dtype=torch.long, device=responses.device)
    post_tool_calls = torch.zeros(batch_size, dtype=torch.long, device=responses.device)
    post_answers = torch.zeros(batch_size, dtype=torch.long, device=responses.device)

    for sample_idx in range(batch_size):
        token_ids = responses[sample_idx].detach().cpu().tolist()
        valid = response_mask[sample_idx].detach().cpu().bool().tolist()
        recognized: dict[tuple[int, str], int] = {}
        for action_type, patterns in (("tool", tool_patterns), ("answer", answer_patterns)):
            for pattern in patterns:
                for start in _find_subsequence(token_ids, pattern):
                    key = (start, action_type)
                    recognized[key] = max(recognized.get(key, 0), len(pattern))
        actions = [(start, action_type, width) for (start, action_type), width in recognized.items()]
        actions.sort(key=lambda action: action[0])

        has_retrieved = False
        for start, action_type, width in actions:
            if start + width > response_length or not all(valid[start : start + width]):
                continue
            if action_type == "tool" and not has_retrieved:
                initial_tool_calls[sample_idx] += 1
                has_retrieved = True
                continue
            if not has_retrieved:
                continue

            decision_index = start + divergence_offset
            if valid[decision_index]:
                decision_mask[sample_idx, decision_index] = True
                if action_type == "tool":
                    post_tool_calls[sample_idx] += 1
                else:
                    post_answers[sample_idx] += 1

    return DecisionActionStats(
        mask=decision_mask,
        initial_tool_calls=initial_tool_calls,
        post_retrieval_tool_calls=post_tool_calls,
        post_retrieval_answers=post_answers,
        action_token_ids=(tool_call_token_ids[divergence_offset], answer_token_ids[divergence_offset]),
    )
