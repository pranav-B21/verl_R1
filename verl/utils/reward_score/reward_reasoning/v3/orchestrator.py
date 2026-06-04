"""
Combined Reasoning Reward orchestrator — iteration v3.

    R_total = R_answer + shaping
    shaping = clip( SCALE * R_think_raw , -CAP, +CAP )
    R_think_raw = w_tool*tool_use + w_ground*grounding
                + w_synth*synthesis - w_rep*self_rep

R_answer is the unchanged outcome reward (reward_SPRec.compute_score). The
process components come from reward_reasoning.v3.reasoning.

Why this shape (contrast with v2)
---------------------------------
* **Two-sided & mostly positive.** v2's R_think was negative ~100% of the time
  (a pure penalty that dragged R_total down and only ever punished). v3's
  positive terms (tool_use/grounding/synthesis) are leading indicators of
  correctness, so good agentic behaviour is *rewarded*, giving GRPO a climbable
  signal — including inside the majority all-wrong groups, where it now points
  toward retrieval + grounding + synthesis instead of away from grounding.
* **Capped below the answer-tier gaps.** R_answer tiers are
  1.0 / 0.8 / 0.5 / 0.1 / 0.001 / 0. With CAP = 0.08 the shaping can never push
  a top-100 hit (0.1) below a fully-wrong answer (0.0), so it can differentiate
  *within* a tier but never reorder correctness. (v2's lambda*R_think reached
  -0.13 and could reorder tiers.)
* **Asymmetric (LongPAS), correctly.** Correct answers (R_answer >= 0.5) only
  ever receive the *positive* part of the shaping (never penalised for minor
  repetition); wrong answers receive the full clipped shaping. Unlike v2 — where
  the positive half was dead so the "correct branch" was a silent no-op — here
  the positive half actually pays out.

Returns a dict so the reward manager logs per-component metrics; in particular
`r_answer` is logged separately, giving a clean val/train metric that is
directly comparable to the baseline's `reward/mean@1` (the headline `score`
includes shaping, so it is *not* comparable to baseline on its own).

Hyperparameters (env vars, all optional)
----------------------------------------
    RTHINK_SCALE     overall scale of R_think_raw before clipping   (default 0.10)
    RTHINK_CAP       absolute cap on the applied shaping            (default 0.08)
    RTHINK_W_TOOL    weight of tool_use   in R_think_raw            (default 0.30)
    RTHINK_W_GROUND  weight of grounding  in R_think_raw            (default 0.40)
    RTHINK_W_SYNTH   weight of synthesis  in R_think_raw            (default 0.30)
    RTHINK_W_REP     weight of self_rep   penalty in R_think_raw    (default 0.50)
"""

import os
import random


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_SCALE   = _envf("RTHINK_SCALE",   0.10)
_CAP     = _envf("RTHINK_CAP",     0.08)
_W_TOOL  = _envf("RTHINK_W_TOOL",  0.30)
_W_GROUND = _envf("RTHINK_W_GROUND", 0.40)
_W_SYNTH = _envf("RTHINK_W_SYNTH", 0.30)
_W_REP   = _envf("RTHINK_W_REP",   0.50)


def compute_score(solution_str: str, ground_truth: dict, data_source: str,
                  method: str = "strict", format_score: float = 0.0,
                  score: float = 1.0, extra_info: dict = None) -> dict:
    """Drop-in replacement for reward_SPRec.compute_score.

    Returns a dict: ``{"score": R_total, "r_answer": ..., "r_think": ...,
    "tool_use": ..., "grounding": ..., "synthesis": ..., "self_rep": ...}``.
    The reward manager uses ``score["score"]`` as the reward and logs the rest.
    """
    # ------------------------------------------------------------------ #
    # R_answer — outcome reward, identical to the baseline.              #
    # ------------------------------------------------------------------ #
    from ... import reward_SPRec
    r_answer = reward_SPRec.compute_score(
        solution_str, ground_truth, data_source,
        method=method, format_score=format_score, score=score,
    )

    # ------------------------------------------------------------------ #
    # Process components — evidence-grounded, per-rollout.               #
    # ------------------------------------------------------------------ #
    prompt_text = ""
    if extra_info and extra_info.get("prompt_str"):
        prompt_text = extra_info["prompt_str"]

    from . import reasoning
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

    # Scale then clip so |shaping| <= CAP < smallest answer-tier gap.
    shaping = max(-_CAP, min(_CAP, _SCALE * r_think_raw))

    # Asymmetric application (LongPAS): never penalise a correct answer.
    if r_answer >= 0.5:
        applied = max(0.0, shaping)
    else:
        applied = shaping
    r_total = r_answer + applied

    # Debug logging (full breakdown; longer solution snippet than v2).
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[Rthink-v3] R_ans={r_answer:.3f}  "
              f"tool={tool_use:.3f}  grnd={grounding:.3f}  syn={synthesis:.3f}  "
              f"rep={self_rep:.3f}  R_think_raw={r_think_raw:.3f}  "
              f"shaping={applied:+.4f}  R_total={r_total:.3f}  "
              f"(scale={_SCALE} cap={_CAP} "
              f"w=[{_W_TOOL},{_W_GROUND},{_W_SYNTH},{_W_REP}])")
        print(f"Solution string: {solution_str[:1200]}")

    return {
        "score": float(r_total),
        "r_answer": float(r_answer),
        "r_think": float(applied),
        "tool_use": float(tool_use),
        "grounding": float(grounding),
        "synthesis": float(synthesis),
        "self_rep": float(self_rep),
    }
