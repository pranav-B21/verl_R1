"""Reasoning-reward iteration v4 — dense, rank-based answer reward.

See ./README.md for the design rationale and ./orchestrator.py for the entry
point ``compute_score``.
"""

from .orchestrator import compute_score

__all__ = ["compute_score"]
