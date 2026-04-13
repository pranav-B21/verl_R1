"""
B. REDUNDANCY PENALITY

Combined reward for Part B ablation: R_total = R_answer + lambda * R_think
where R_think = -beta * redundancy 

Activated by setting USE_REWARD_B=1 in the container environment.
Routed from __init__.py's default_compute_score for amazon/goodreads data sources.

Hyperparameters (tunable via env vars):
  REWARD_B_LAMBDA  — weight for R_think relative to R_answer  (default 0.3)
  REWARD_B_BETA    — weight for redundancy within R_think      (default 1.0)

So: R_total = R_answer - LAMBDA * BETA * redundancy
           ∈ [-0.3, 1.0]  with defaults

WandB: total reward (R_total) is logged as the primary metric.
       R_answer and redundancy components are printed to the log file
       (~1/64 of calls) for debugging without WandB API overhead.

When Parts A and C are implemented, update the R_think line here.
"""

import os
import random


# Hyperparameters — override via env vars for sweep experiments
_LAMBDA = float(os.environ.get("REWARD_B_LAMBDA", "0.3"))
_BETA   = float(os.environ.get("REWARD_B_BETA",   "1.0"))


def compute_score(
    solution_str: str,
    ground_truth: dict,
    data_source: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """
    Drop-in replacement for reward_SPRec.compute_score with redundancy penalty.

    Returns R_total = R_answer - LAMBDA * BETA * redundancy
    """
    # ------------------------------------------------------------------ #
    # Part 0: Outcome reward (unchanged from baseline)                    #
    # ------------------------------------------------------------------ #
    from . import reward_SPRec
    r_answer = reward_SPRec.compute_score(
        solution_str, ground_truth, data_source,
        method=method, format_score=format_score, score=score,
    )

    # ------------------------------------------------------------------ #
    # Part B: Redundancy penalty                                          #
    # ------------------------------------------------------------------ #
    from . import reward_SPRec_reasoning
    redundancy = reward_SPRec_reasoning.compute_score(
        solution_str, ground_truth, data_source,
    )

    # ------------------------------------------------------------------ #
    # Combine: R_total = R_answer + lambda * R_think                      #
    # R_think = -beta * redundancy   (Parts A and C are TODO)            #
    # ------------------------------------------------------------------ #
    r_think = -_BETA * redundancy
    r_total = r_answer + _LAMBDA * r_think

    # Debug logging (matches do_print pattern from reward_SPRec.py)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"[RewardB] R_answer={r_answer:.3f}  redundancy={redundancy:.3f}  "
              f"R_think={r_think:.3f}  R_total={r_total:.3f}  "
              f"(lambda={_LAMBDA}, beta={_BETA})")
        print(f"Solution string: {solution_str}")

    return float(r_total)
