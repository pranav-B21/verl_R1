"""
Reasoning-reward system — versioned iterations.

This package holds the successive iterations of the REC-R1 reasoning reward
(R_think), one subpackage per version, so each can be documented, run, and
ablated independently:

    reward_reasoning/
      v2/   info_gain / redundancy / exploration   (underperformed baseline)
      v3/   evidence-grounded process reward        (current default)

Entry point
-----------
``compute_score(...)`` dispatches to the iteration selected by the ``RTHINK_MODE``
environment variable (default ``v3``):

    RTHINK_MODE=v3            -> v3.compute_score   (default)
    RTHINK_MODE=v2 | legacy   -> v2.compute_score

It is wired in from ``verl/utils/reward_score/__init__.py`` whenever
``USE_RTHINK=1`` (or the legacy ``USE_REWARD_B=1``) is set for an
amazon/goodreads/movie data source.

See ./README.md for the full iteration history and the analysis behind v3.
"""

import os

_MODE_ALIASES = {
    "v2": "v2", "legacy": "v2", "2": "v2",
    "v3": "v3", "3": "v3",
}


def _resolve_mode() -> str:
    raw = os.environ.get("RTHINK_MODE", "v3").strip().lower()
    return _MODE_ALIASES.get(raw, "v3")


def compute_score(solution_str, ground_truth, data_source,
                  method="strict", format_score=0.0, score=1.0, extra_info=None):
    """Dispatch to the reasoning-reward iteration named by ``RTHINK_MODE``."""
    mode = _resolve_mode()
    if mode == "v2":
        from .v2 import compute_score as _impl
    else:
        from .v3 import compute_score as _impl
    return _impl(
        solution_str, ground_truth, data_source,
        method=method, format_score=format_score, score=score,
        extra_info=extra_info,
    )


__all__ = ["compute_score"]
