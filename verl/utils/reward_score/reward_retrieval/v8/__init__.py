# SPDX-License-Identifier: Apache-2.0
"""v8 — selection reward. See ./README.md for the design and the decision rule."""

from .orchestrator import compute_score

__all__ = ["compute_score"]
