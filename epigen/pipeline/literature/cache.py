"""Disk cache for per-accession Paperclip lookups.

Keyed by UniProt accession (not by construct sequence): the UniProt sequence
and its feature list are properties of the accession, so caching at that
level lets any construct of the same protein (mature chain, tagged variant,
...) reuse one Paperclip round-trip. Alignment of construct numbering to
UniProt numbering happens fresh in `annotations.py` on every call -- it's
free, and caching it would tie the cache to one specific construct.

Demo-day motivated: avoids depending on Paperclip's availability/latency for
repeated runs on the same scaffold (e.g. lysozyme) during a live demo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(accession: str) -> Path:
    return CACHE_DIR / f"{accession}.json"


def load(accession: str) -> dict[str, Any] | None:
    """Cached `{"sequence": ..., "features": [...]}` for `accession`, or `None` on a miss."""
    path = _cache_path(accession)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Ignoring unreadable literature cache entry {path}: {exc}")
        return None


def save(accession: str, *, sequence: str, features: list[dict[str, str]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(accession).write_text(json.dumps({"sequence": sequence, "features": features}, indent=2))
