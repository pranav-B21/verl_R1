"""Reasoning-reward iteration v5 — format-gated, HR-targeted dense reward.

v5 = v4 (dense rank reward) + a format-integrity gate that kills the multi-answer
reward hack, a capped length penalty, and an HR-targeted dense default (p=2.0).

See ./README.md for the design rationale and ./orchestrator.py for the entry
point ``compute_score``.
"""

from .orchestrator import compute_score

__all__ = ["compute_score"]
