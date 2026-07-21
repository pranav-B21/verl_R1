"""Reasoning-reward iteration v6 — HR-faithful, top-K outcome reward.

v6 = v5 (format gate + length cap + v3 shaping + LongPAS) with ONE substantive
change: the outcome reward is now a faithful surrogate of the held-out metric.
It pays the exact eval.py NDCG term (``1/log2(rank+1)``) inside the top-K window
and only a weak, steep tail outside it — concentrating the GRPO gradient where
HR@K is actually scored, instead of paying v4/v5's large mid-rank partial credit
that never transferred to greedy held-out HR.

See ./README.md for the design rationale and the diagnosis it is built on, and
./orchestrator.py for the entry point ``compute_score``.
"""

from .orchestrator import compute_score

__all__ = ["compute_score"]
