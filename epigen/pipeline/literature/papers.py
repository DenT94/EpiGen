"""Supporting-literature search for functional annotations, via Paperclip full-text search.

Best-effort provenance, not verified citations: for each functional
`AnnotationRange`, searches PMC for papers likely to discuss that residue or
site, using the protein's name/gene plus the feature type and position as
the query. Results are candidate supporting literature for a human/agent to
read and judge -- not run through Paperclip's opt-in claim-verification
workflow (out of scope for this hackathon step; see the `paperclip` skill's
repo/`git commit` verification if that's ever wanted for a specific claim).

Structural rows (helix/strand/turn/coiled-coil) are left alone: their real
provenance is the determining structure itself (already available as
`FoldedStructure.pdb_id` / `pdb_v.structures_by_accession`), not a per-
residue literature claim worth a separate search.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, replace

from epigen.pipeline.literature import cache
from epigen.pipeline.literature.annotations import AccessionMetadata, AnnotationRange, PaperReference
from epigen.pipeline.literature.paperclip_client import PaperclipError, run_paperclip

logger = logging.getLogger(__name__)

PAPERCLIP_SEARCH_TIMEOUT_S = 30
DEFAULT_PAPERS_PER_FEATURE = 2
# Caps total Paperclip round-trips for a densely-annotated protein -- each search is a
# separate subprocess call (~0.3-1s), and this list is for display, not something the
# oracle/MCMC loop waits on, so it's fine to just cover the first N and say so.
DEFAULT_MAX_FEATURES = 10

# One result block: "  1. <title>\n     <authors>\n     <doc_id> · <source> · <date>\n     <url>"
# (see live `paperclip search` output; a trailing quoted snippet line, if present, is
# ignored). Line-based rather than one big regex because `<title>` can itself wrap
# across multiple lines for a long title -- there's no fixed line count to anchor on,
# only the doc_id/source/date line's distinct "X · Y · Z" shape.
_ENTRY_START_RE = re.compile(r"^\s*\d+\.\s+(?P<first_line>.*)$")
# Anchored on the literal "·" separators, not token counts -- `source` (a journal name
# like "Journal of Medicinal Chemistry") can itself contain spaces, so a token-counting
# pattern misfires and swallows lines up to the next single-word-source entry.
_ID_LINE_RE = re.compile(r"^\s*(?P<doc_id>\S+)\s+·\s+(?P<source>.+?)\s+·\s+(?P<date>\S+)\s*$")
_URL_LINE_RE = re.compile(r"^\s*(?P<url>https?://\S+)\s*$")


def _parse_search_results(stdout: str) -> list[PaperReference]:
    lines = stdout.splitlines()
    results = []
    i = 0
    while i < len(lines):
        start = _ENTRY_START_RE.match(lines[i])
        if not start:
            i += 1
            continue
        block = [start["first_line"]]
        i += 1
        id_match = None
        while i < len(lines):
            id_match = _ID_LINE_RE.match(lines[i])
            if id_match:
                break
            block.append(lines[i].strip())
            i += 1
        if id_match is None:
            break  # ran off the end without a doc_id line -- malformed tail, stop
        i += 1
        title = " ".join(block[:-1]).strip() if len(block) > 1 else block[0].strip()
        authors = block[-1].strip() if len(block) > 1 else ""
        url_match = _URL_LINE_RE.match(lines[i]) if i < len(lines) else None
        url = url_match["url"] if url_match else ""
        if url_match:
            i += 1
        results.append(
            PaperReference(
                title=title, authors=authors,
                doc_id=id_match["doc_id"], source=id_match["source"], date=id_match["date"], url=url,
            )
        )
    return results


def _search(query: str, *, n: int) -> list[PaperReference]:
    """Disk-cached by `(query, n)` (`literature.cache.load_paper_search`/
    `save_paper_search`) -- the same protein/feature/position query returns the same
    literature every time, so a repeat "Find supporting papers" click for an
    already-searched protein is a cache hit, not another round of Paperclip subprocess
    calls. Cached independently of `st.cache_data` -- see `literature.cache`'s
    docstring for why."""
    cache_hit, cached_results = cache.load_paper_search(query, n)
    if cache_hit:
        return [PaperReference(**r) for r in cached_results]

    try:
        stdout = run_paperclip(["search", "-s", "pmc", "-n", str(n), query], timeout=PAPERCLIP_SEARCH_TIMEOUT_S)
    except PaperclipError as exc:
        logger.warning(f"Paperclip search failed for {query!r} (after retries): {exc}")
        return []
    results = _parse_search_results(stdout)
    cache.save_paper_search(query, n, [asdict(r) for r in results])
    return results


def _query_for(metadata: AccessionMetadata, annotation: AnnotationRange) -> str:
    name = metadata.gene_name or metadata.protein_name
    position = f"{annotation.start}" if annotation.start == annotation.end else f"{annotation.start}-{annotation.end}"
    return f"{metadata.protein_name} {name} {annotation.feature_type} residue {position}".strip()


def attach_papers(
    ranges: list[AnnotationRange],
    metadata: AccessionMetadata,
    *,
    n_per_feature: int = DEFAULT_PAPERS_PER_FEATURE,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> list[AnnotationRange]:
    """`ranges` with `.papers` filled in for up to `max_features` functional entries
    (structural rows pass through with `papers=[]`, as documented above). Best-effort:
    a search failure for one feature just leaves that entry's `papers` empty rather than
    aborting the rest.
    """
    functional = [r for r in ranges if r.kind == "functional"]
    if len(functional) > max_features:
        logger.info(
            f"{len(functional)} functional annotations found; searching literature for the first {max_features}."
        )
    search_targets = {id(r) for r in functional[:max_features]}

    return [
        replace(r, papers=_search(_query_for(metadata, r), n=n_per_feature)) if id(r) in search_targets else r
        for r in ranges
    ]
