"""Step 6: tile MCMC candidates' coding sequences into a synthesis-ready oligo library.

mypipelinethoughts.md: "Transform the candidate list into a library, with
OLIGO_LEN(default=350) and OVERLAP_LEN(default=15) to send directly for
experiments."

Each candidate's `nt_sequence` (the compensated coding sequence -- see
`oracle.mcmc.MCMCCandidate`, only populated when Evo2 scoring was on) is cut
into a sliding window of `OLIGO_LEN`-nt fragments, consecutive fragments
overlapping by `OVERLAP_LEN` nt so a synthesis vendor's fragments (or a
downstream Gibson-style assembly) can be unambiguously stitched back
together in order. No Modal calls -- pure string slicing over data
`orchestrate.run_end_to_end` already produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from epigen.pipeline.oracle.mcmc import MCMCCandidate

OLIGO_LEN = 350
OVERLAP_LEN = 15


@dataclass(frozen=True)
class OligoFragment:
    """One synthesis-ready fragment of one candidate's coding sequence."""

    candidate_id: str  # e.g. "candidate_01" -- stable/unique within one build_library() call
    fragment_index: int  # 0-indexed position of this fragment within its candidate
    fragment_count: int  # total fragments for this candidate (e.g. "2 of 3")
    start: int  # 0-indexed start offset into the candidate's nt_sequence (inclusive)
    end: int  # 0-indexed end offset into the candidate's nt_sequence (exclusive)
    sequence: str  # candidate.nt_sequence[start:end]
    name: str  # human-readable, unique within the export -- use as the order-sheet ID


def tile_sequence(nt_sequence: str, *, oligo_len: int = OLIGO_LEN, overlap_len: int = OVERLAP_LEN) -> list[tuple[int, int]]:
    """0-indexed half-open `(start, end)` windows tiling `nt_sequence`: each `oligo_len`
    nt except possibly the last (clipped to the sequence's actual end, not padded),
    consecutive windows overlapping by exactly `overlap_len` nt.

    A sequence no longer than `oligo_len` needs no tiling -- returns a single
    whole-sequence window. Raises `ValueError` for a nonsensical `overlap_len`
    (`>= oligo_len` would never advance past the first window; negative doesn't mean
    anything).
    """
    if oligo_len <= 0:
        raise ValueError(f"oligo_len must be positive, got {oligo_len}")
    if not (0 <= overlap_len < oligo_len):
        raise ValueError(f"overlap_len must be in [0, oligo_len) -- got overlap_len={overlap_len}, oligo_len={oligo_len}")

    n = len(nt_sequence)
    if n <= oligo_len:
        return [(0, n)]

    step = oligo_len - overlap_len
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + oligo_len, n)
        windows.append((start, end))
        if end == n:
            return windows
        start += step


def build_library(
    candidates: list[MCMCCandidate],
    *,
    oligo_len: int = OLIGO_LEN,
    overlap_len: int = OVERLAP_LEN,
) -> tuple[list[OligoFragment], list[MCMCCandidate]]:
    """Tile every candidate that has an `nt_sequence` into `OligoFragment`s.

    Returns `(fragments, skipped)`: `skipped` holds candidates with
    `nt_sequence is None` (Evo2 was off for this run, so there's no coding
    sequence to tile) -- handled gracefully, not an error, since not every
    run enables Evo2. `candidate_id`s (`"candidate_01"`, `"candidate_02"`,
    ...) are assigned by position in `candidates`, skipped ones included in
    the count, so an id always points back to the same original candidate
    regardless of how many were skipped.
    """
    fragments: list[OligoFragment] = []
    skipped: list[MCMCCandidate] = []
    for i, candidate in enumerate(candidates):
        candidate_id = f"candidate_{i + 1:02d}"
        if candidate.nt_sequence is None:
            skipped.append(candidate)
            continue
        windows = tile_sequence(candidate.nt_sequence, oligo_len=oligo_len, overlap_len=overlap_len)
        for frag_idx, (start, end) in enumerate(windows):
            fragments.append(
                OligoFragment(
                    candidate_id=candidate_id,
                    fragment_index=frag_idx,
                    fragment_count=len(windows),
                    start=start,
                    end=end,
                    sequence=candidate.nt_sequence[start:end],
                    name=f"{candidate_id}_frag{frag_idx + 1:02d}of{len(windows):02d}",
                )
            )
    return fragments, skipped


def to_fasta(fragments: list[OligoFragment]) -> str:
    """FASTA text, one record per fragment -- header carries enough metadata
    (candidate/position/length) to route each fragment back to its candidate
    without needing the CSV export alongside it."""
    lines = []
    for f in fragments:
        lines.append(f">{f.name} candidate={f.candidate_id} start={f.start} end={f.end} len={len(f.sequence)}")
        lines.append(f.sequence)
    return "\n".join(lines) + ("\n" if fragments else "")


def to_csv_rows(fragments: list[OligoFragment]) -> list[dict[str, str | int]]:
    """One dict per fragment, column order matching a typical synthesis-vendor order
    sheet (name, sequence, then metadata) -- feed straight to `csv.DictWriter` or
    `pandas.DataFrame`/`st.dataframe`."""
    return [
        {
            "name": f.name,
            "sequence": f.sequence,
            "candidate_id": f.candidate_id,
            "fragment_index": f.fragment_index + 1,
            "fragment_count": f.fragment_count,
            "start": f.start,
            "end": f.end,
            "length": len(f.sequence),
        }
        for f in fragments
    ]
