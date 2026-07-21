# SPDX-License-Identifier: Apache-2.0
"""Reasoning-reward iteration v7 — retrieval-quality reward (the PI's proposal).

v7 = v6 (HR-faithful outcome + format gate + length cap + LongPAS) with ONE
substantive change: v3's reasoning-quality process shaping is replaced by a
retrieval-quality shaping term that scores how similar the model's own
retrieved documents are to the ground-truth answer. It does not touch the
retrieval corpus in any way.

See ./README.md for the design rationale, ./ROADMAP_v7.md (the RRCM
journal-extension roadmap) for why the corpus-restructuring line of work
(v7a/v7b, as originally sketched) was superseded by this, and ./orchestrator.py
for the entry point ``compute_score``.
"""

from .orchestrator import compute_score

__all__ = ["compute_score"]
