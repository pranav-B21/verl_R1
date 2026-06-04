"""Reasoning-reward iteration v3 — evidence-grounded process reward.

See ./README.md for the design rationale and ./orchestrator.py for the entry
point ``compute_score``.
"""

from .orchestrator import compute_score

__all__ = ["compute_score"]
