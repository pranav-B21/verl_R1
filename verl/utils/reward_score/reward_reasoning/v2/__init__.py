"""Reasoning-reward iteration v2 — info_gain / redundancy / exploration.

The canonical v2 implementation lives at (and is still imported by name from
tests and CLAUDE.md):

    verl/utils/reward_score/reward_SPRec_reasoning.py   (components)
    verl/utils/reward_score/reward_SPRec_rthink.py      (orchestrator)

This subpackage re-exports the v2 orchestrator's ``compute_score`` so the
version dispatcher (../__init__.py) can select it uniformly via
``RTHINK_MODE=v2`` (alias ``legacy``). See ./README.md for the v2 design and the
postmortem explaining why it underperformed the no-reasoning baseline.
"""

from ...reward_SPRec_rthink import compute_score

__all__ = ["compute_score"]
