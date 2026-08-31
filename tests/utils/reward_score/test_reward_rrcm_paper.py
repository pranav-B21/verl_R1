import pytest

from verl.utils.reward_score import reward_rrcm_paper


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (1, 1.0),
        (2, 0.5),
        (5, 0.5),
        (6, 0.2),
        (10, 0.2),
        (11, 0.1),
        (50, 0.1),
        (51, 0.02),
        (100, 0.02),
        (101, 0.0),
        (None, 0.0),
    ],
)
def test_paper_ranking_reward(rank, expected):
    assert reward_rrcm_paper.ranking_reward(rank) == pytest.approx(expected)


@pytest.mark.parametrize(
    "response",
    [
        "<answer>Title</answer>",
        "<answer>'Title'</answer>",
        '<answer>"Title" and text</answer>',
        '<answer>"Title"</answer><answer>"Other"</answer>',
        '<answer>"Title</answer>',
        '<answer>""</answer>',
    ],
)
def test_answer_parser_rejects_non_paper_format(response):
    assert reward_rrcm_paper.extract_single_quoted_answer(response) is None


def test_answer_parser_accepts_reasoning_before_one_quoted_answer():
    response = 'reasoning and retrieved context\n<answer>  "A Title"  </answer>'
    assert reward_rrcm_paper.extract_single_quoted_answer(response) == "A Title"


def test_invalid_format_receives_parse_penalty():
    result = reward_rrcm_paper.compute_score("<answer>unquoted</answer>", {"target": "x"}, "amazon")
    assert result == {
        "score": -1.0,
        "r_answer": 0.0,
        "rrcm_rank": -1.0,
        "format_ok": 0.0,
        "target_found": 0.0,
    }


def test_valid_answer_receives_only_paper_rank_reward(monkeypatch):
    monkeypatch.setattr(reward_rrcm_paper, "_catalog_rank", lambda *args: (6, True))
    result = reward_rrcm_paper.compute_score('<answer>"A Title"</answer>', {"target": "x"}, "amazon")

    assert result["score"] == pytest.approx(0.2)
    assert result["r_answer"] == pytest.approx(0.2)
    assert result["rrcm_rank"] == 6.0
    assert result["format_ok"] == 1.0
    assert result["target_found"] == 1.0
