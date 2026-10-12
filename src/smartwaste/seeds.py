"""Reproducible sub-seed derivation.

Lives in its own module because both the Phase 1 fusion evaluation and the Phase 2 city benchmark
need it, and two copies of a seed-derivation function is exactly the kind of duplication that
silently de-synchronises published numbers.

``smartwaste.fusion_eval`` re-exports :func:`child_seed`, so the Phase 1 import path and the
values pinned in ``tests/test_fusion_eval.py`` are unchanged.
"""

from __future__ import annotations

import hashlib


def child_seed(seed: int, label: str) -> int:
    """Derive a reproducible sub-seed from a master seed and a label.

    Uses sha256 rather than :func:`hash`, because ``hash()`` of a ``str`` is salted per process
    (``PYTHONHASHSEED``). With ``hash()`` the same command and seed would silently produce
    different numbers in different runs, which is exactly what this project promises not to do.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
