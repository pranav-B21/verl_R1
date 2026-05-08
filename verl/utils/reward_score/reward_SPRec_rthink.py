"""
Combined Reasoning Reward Orchestrator (R_think).

R_total = R_answer + lambda * R_think
R_think = alpha * info_gain - beta * redundancy + gamma * exploration_bonus

Components (computed in reward_SPRec_reasoning.py):
  Part A: info_gain        — novelty of reasoning vs prompt, search, and past thinking
  Part B: redundancy       — self-repetition, copying, parroting penalty
  Part C: exploration_bonus — encourages diverse reasoning early, exploits later

Asymmetric application (LongPAS):
  R_answer >= 0.5: R_total = R_answer + lambda * max(0, R_think)
  R_answer <  0.5: R_total = R_answer + lambda * R_think

Hyperparameters (env vars):
  RTHINK_LAMBDA  — weight for R_think in R_total            (default 0.3)
  RTHINK_ALPHA   — weight for info_gain in R_think          (default 0.5)
  RTHINK_BETA    — weight for redundancy in R_think         (default 1.0)
  RTHINK_GAMMA   — weight for exploration_bonus in R_think  (default 0.3)

Activated by USE_RTHINK=1 (or legacy USE_REWARD_B=1).
"""

import os
import random


_LAMBDA = float(os.environ.get("RTHINK_LAMBDA", os.environ.get("REWARD_B_LAMBDA", "0.3")))
_ALPHA  = float(os.environ.get("RTHINK_ALPHA",  os.environ.get("REWARD_B_ALPHA",  "0.5")))
_BETA   = float(os.environ.get("RTHINK_BETA",   os.environ.get("REWARD_B_BETA",   "1.0")))
_GAMMA  = float(os.environ.get("RTHINK_GAMMA",  os.environ.get("REWARD_B_GAMMA",  "0.3")))


def compute_score(
    solution_str: str,
    ground_truth: dict,
    data_source: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    extra_info: dict = None,
) -> float:
    """
    Drop-in replacement for reward_SPRec.compute_score.
    Returns R_total combining outcome reward with reasoning reward.
    """
    # ------------------------------------------------------------------ #
    # R_answer: outcome reward (unchanged from baseline)                  #
    # ------------------------------------------------------------------ #
    from . import reward_SPRec
    r_answer = reward_SPRec.compute_score(
        solution_str, ground_truth, data_source,
        method=method, format_score=format_score, score=score,
    )

    # ------------------------------------------------------------------ #
    # Reasoning components: info_gain, redundancy, exploration_bonus      #
    # ------------------------------------------------------------------ #
    from . import reward_SPRec_reasoning
    components = reward_SPRec_reasoning.compute_reasoning_components(
        solution_str, data_source, r_answer=r_answer, extra_info=extra_info,
    )

    info_gain = components["info_gain"]
    redundancy = components["redundancy"]
    exploration = components["exploration_bonus"]

    # ------------------------------------------------------------------ #
    # R_think = alpha * info_gain - beta * redundancy                     #
    #         + gamma * exploration_bonus                                  #
    # ------------------------------------------------------------------ #
    r_think = (
        _ALPHA * info_gain
        - _BETA * redundancy
        + _GAMMA * exploration
    )

    # ------------------------------------------------------------------ #
    # Asymmetric application (LongPAS):                                   #
    # Correct answers: only add positive R_think (never penalize)         #
    # Wrong answers: apply full R_think including penalties               #
    # ------------------------------------------------------------------ #
    if r_answer >= 0.5:
        r_total = r_answer + _LAMBDA * max(0.0, r_think)
    else:
        r_total = r_answer + _LAMBDA * r_think

    # Debug logging
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"[Rthink] R_ans={r_answer:.3f}  "
              f"ig={info_gain:.3f}  red={redundancy:.3f}  "
              f"exp={exploration:.3f}  R_think={r_think:.3f}  "
              f"R_total={r_total:.3f}  "
              f"(λ={_LAMBDA} α={_ALPHA} β={_BETA} γ={_GAMMA})")
        print(f"Solution string: {solution_str[:500]}")

    return float(r_total)