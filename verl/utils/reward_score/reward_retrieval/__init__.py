"""Retrieval-quality reward — versioned iterations, sibling to ``reward_reasoning``.

This package holds reward work that scores *retrieval*, not process/reasoning
quality: how similar what the policy retrieved is to the ground-truth answer,
rather than how well it reasoned over what it retrieved. It exists as a
separate top-level package from ``reward_reasoning`` because it targets a
different axis of the rollout (retrieval behavior vs. `<think>` content), even
though it reuses ``reward_reasoning``'s v6 outcome reward as its base.

    reward_retrieval/
      retrieval_reward_design.md   # base design doc: math, setup, done/planned
      v7/                          # retrieval-quality reward (the PI's proposal)

See ``retrieval_reward_design.md`` for the full design and
``v7/ROADMAP_v7.md`` (the RRCM journal-extension roadmap) for the diagnosis,
the PI pivot, and the phased reward plan that motivate it.
"""

from .v7 import compute_score

__all__ = ["compute_score"]
