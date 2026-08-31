import math

import pytest
import torch

from verl.trainer.ppo.decision_entropy import (
    binary_action_entropy,
    build_post_retrieval_decision_stats,
    decision_entropy_coefficient,
)


def test_hold_then_linear_decay_schedule():
    coefficient = 0.001
    assert decision_entropy_coefficient(0, coefficient, 300, 500) == pytest.approx(coefficient)
    assert decision_entropy_coefficient(300, coefficient, 300, 500) == pytest.approx(coefficient)
    assert decision_entropy_coefficient(400, coefficient, 300, 500) == pytest.approx(0.0005)
    assert decision_entropy_coefficient(500, coefficient, 300, 500) == 0.0
    assert decision_entropy_coefficient(900, coefficient, 300, 500) == 0.0


def test_invalid_schedule_is_rejected():
    with pytest.raises(ValueError, match="greater than"):
        decision_entropy_coefficient(1, 0.001, 300, 300)
    with pytest.raises(ValueError, match="non-negative"):
        decision_entropy_coefficient(1, 0.001, -1, 500)


def test_binary_entropy_only_uses_the_two_action_logits():
    logits = torch.zeros(2, 3, 7, requires_grad=True)
    entropy = binary_action_entropy(logits, (2, 5))
    assert entropy.shape == (2, 3)
    assert torch.allclose(entropy, torch.full_like(entropy, math.log(2)))

    entropy.sum().backward()
    untouched = [index for index in range(logits.shape[-1]) if index not in (2, 5)]
    assert torch.count_nonzero(logits.grad[..., untouched]) == 0


def test_binary_entropy_falls_when_action_choice_is_certain():
    balanced = binary_action_entropy(torch.tensor([[0.0, 0.0, 0.0]]), (0, 2))
    certain = binary_action_entropy(torch.tensor([[10.0, 0.0, -10.0]]), (0, 2))
    assert certain.item() < balanced.item()


def test_direct_answer_has_no_post_retrieval_decision():
    responses = torch.tensor([[9, 1, 4, 5, 0, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    stats = build_post_retrieval_decision_stats(responses, response_mask, [1, 2, 3], [1, 4, 5])

    assert not stats.mask.any()
    assert stats.initial_tool_calls.tolist() == [0]
    assert stats.post_retrieval_answers.tolist() == [0]
    assert stats.action_token_ids == (2, 4)


def test_first_retrieval_is_excluded_and_later_action_is_masked():
    responses = torch.tensor([[9, 1, 2, 3, 8, 1, 4, 5, 0]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]])
    stats = build_post_retrieval_decision_stats(responses, response_mask, [1, 2, 3], [1, 4, 5])

    assert stats.mask.nonzero(as_tuple=False).tolist() == [[0, 6]]
    assert stats.initial_tool_calls.tolist() == [1]
    assert stats.post_retrieval_tool_calls.tolist() == [0]
    assert stats.post_retrieval_answers.tolist() == [1]


def test_second_retrieval_and_final_answer_are_both_decisions():
    responses = torch.tensor([[9, 1, 2, 3, 8, 1, 2, 3, 8, 1, 4, 5]])
    response_mask = torch.ones_like(responses)
    stats = build_post_retrieval_decision_stats(responses, response_mask, [1, 2, 3], [1, 4, 5])

    assert stats.mask.nonzero(as_tuple=False).tolist() == [[0, 6], [0, 10]]
    assert stats.initial_tool_calls.tolist() == [1]
    assert stats.post_retrieval_tool_calls.tolist() == [1]
    assert stats.post_retrieval_answers.tolist() == [1]


def test_action_touching_tool_or_padding_tokens_is_ignored():
    responses = torch.tensor([[1, 2, 3, 8, 1, 4, 5]])
    response_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    stats = build_post_retrieval_decision_stats(responses, response_mask, [1, 2, 3], [1, 4, 5])

    assert not stats.mask.any()
    assert stats.initial_tool_calls.tolist() == [1]
    assert stats.post_retrieval_answers.tolist() == [0]


def test_qwen_contextual_answer_suffix_still_matches():
    # Qwen3 tokenizes standalone <answer> as [27, 9217, 29], but in
    # <answer>"title" the final angle bracket merges with the quote (9877).
    responses = torch.tensor(
        [[151657, 4913, 1631, 2019, 36799, 87, 92446, 151658, 198, 27, 9217, 9877, 88, 21522, 9217, 29]]
    )
    stats = build_post_retrieval_decision_stats(
        responses,
        torch.ones_like(responses),
        tool_call_token_ids=[151657],
        answer_token_ids=[27, 9217, 29],
        tool_call_recognition_patterns=[[151657], [27, 14506, 13735]],
        answer_recognition_patterns=[[27, 9217, 29], [27, 9217]],
    )

    assert stats.action_token_ids == (151657, 27)
    assert stats.initial_tool_calls.tolist() == [1]
    assert stats.post_retrieval_answers.tolist() == [1]
    assert stats.mask.nonzero(as_tuple=False).tolist() == [[0, 9]]
