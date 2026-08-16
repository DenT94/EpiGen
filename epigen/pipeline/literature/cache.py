"""Disk cache for Paperclip lookups: which accession a construct resolves to, and
that accession's sequence/features/metadata.

Both are properties of the *protein*, not of any particular MCMC run's edit/window/
seed settings -- once looked up for a given (sequence, pdb_id) or accession, they're
good forever, so this is a plain on-disk JSON cache, entirely separate from
`st.cache_data`. That matters for two reasons: it survives the app's "Clear cache"
button (which only clears `st.cache_data`-backed caches -- the point of that button is
forcing a pipeline recompute, not re-querying Paperclip for facts about a protein that
haven't changed), and it survives an app restart, unlike an in-memory `lru_cache`.

`load`/`save` (keyed by accession) cache the sequence/feature-list fetch -- any
construct of the same protein (mature chain, tagged variant, ...) reuses one Paperclip
round-trip; alignment of construct numbering to UniProt numbering happens fresh in
`annotations.py` on every call, since that's free and caching it would tie the cache
to one specific construct.

`load_accession`/`save_accession` (keyed by `(pdb_id, sequence)`) cache the earlier
step -- resolving which accession a construct even *is* -- which used to run a fresh
Paperclip SQL query on every single call, uncached, even for the exact same protein.

`load_paper_search`/`save_paper_search` (keyed by the exact search query string, plus
`n`) cache `literature.papers`' full-text search results -- a protein's supporting
literature for a given annotation doesn't change between clicks, but the "Find
supporting papers" button used to re-run every underlying Paperclip search subprocess
call from scratch every time it was clicked.

Demo-day motivated: avoids depending on Paperclip's availability/latency for
repeated runs on the same scaffold (e.g. lysozyme) during a live demo.
"""

from __future__ import annotations

import hashlib
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


def save(
    accession: str, *, sequence: str, features: list[dict[str, str]], metadata: dict[str, str] | None = None
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"sequence": sequence, "features": features, "metadata": metadata or {}}
    _cache_path(accession).write_text(json.dumps(payload, indent=2))


def _accession_lookup_cache_path(pdb_id: str | None, sequence: str) -> Path:
    # Hashed, not a readable filename like `load`/`save` use -- the key here is a
    # (pdb_id, sequence) pair, and a full construct sequence isn't a reasonable filename.
    key = hashlib.sha1(f"{pdb_id or ''}|{sequence}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"accession_lookup_{key}.json"


def load_accession(pdb_id: str | None, sequence: str) -> tuple[bool, str | None]:
    """Returns `(cache_hit, accession)`. `accession` is `None` both on a cache miss
    (`cache_hit=False` -- caller should resolve and `save_accession`) and on a cached
    *negative* result (`cache_hit=True`, `accession=None` -- resolution already failed
    for this exact (pdb_id, sequence) once; a negative result is cached too, same
    permanence as a positive one, so a caller doesn't re-pay for the same miss)."""
    path = _accession_lookup_cache_path(pdb_id, sequence)
    if not path.exists():
        return False, None
    try:
        return True, json.loads(path.read_text())["accession"]
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning(f"Ignoring unreadable accession-lookup cache entry {path}: {exc}")
        return False, None


def save_accession(pdb_id: str | None, sequence: str, accession: str | None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _accession_lookup_cache_path(pdb_id, sequence).write_text(json.dumps({"accession": accession}))


def _paper_search_cache_path(query: str, n: int) -> Path:
    key = hashlib.sha1(f"{query}|{n}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"paper_search_{key}.json"


def load_paper_search(query: str, n: int) -> tuple[bool, list[dict[str, str]] | None]:
    """Returns `(cache_hit, results)`, `results` a list of `PaperReference`-shaped dicts
    (or `None` on a miss). Same negative-caching rationale as `load_accession`: a search
    that found nothing is still cached, so a repeat click doesn't re-run it."""
    path = _paper_search_cache_path(query, n)
    if not path.exists():
        return False, None
    try:
        return True, json.loads(path.read_text())["results"]
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning(f"Ignoring unreadable paper-search cache entry {path}: {exc}")
        return False, None


def save_paper_search(query: str, n: int, results: list[dict[str, str]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _paper_search_cache_path(query, n).write_text(json.dumps({"query": query, "results": results}))
