"""Position alignment between wild-type (WT) numbering and an edited sequence.

The insertion site is a user-specified edit (from the Streamlit input step),
not something that needs sequence alignment/BLAST to discover -- so this is
closed-form arithmetic, not an alignment algorithm. Substitution-only edits
use `identity_map()`; insertions use `insertion_map()` so downstream diffs
(contact_diff, sae_diff) can skip positions with no WT counterpart instead of
silently comparing mismatched residues across the insertion boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionMap:
    """Maps an edited sequence's 1-indexed positions back to WT positions.

    `to_wt(edited_position)` returns `None` when the position falls inside an
    inserted span with no WT counterpart.
    """

    insertion_start: int | None = None  # 1-indexed, first inserted position
    insertion_length: int = 0

    def to_wt(self, edited_position: int) -> int | None:
        if self.insertion_start is None or edited_position < self.insertion_start:
            return edited_position
        if edited_position < self.insertion_start + self.insertion_length:
            return None  # inside the inserted span -- no WT counterpart
        return edited_position - self.insertion_length

    def to_edited(self, wt_position: int) -> int:
        """Inverse of `to_wt`, for WT positions (which are never inside the inserted span).

        `insertion_start` is expressed in edited-sequence numbering, but
        every position before it is identity-mapped, so the same threshold
        value is valid in WT numbering too.
        """
        if self.insertion_start is None or wt_position < self.insertion_start:
            return wt_position
        return wt_position + self.insertion_length


def identity_map() -> PositionMap:
    """Substitution-only edits: every position maps to itself."""
    return PositionMap()


def insertion_map(insertion_start: int, insertion_length: int) -> PositionMap:
    """An insertion of `insertion_length` residues starting at `insertion_start` (1-indexed,
    position in the *edited* sequence's numbering)."""
    if insertion_length <= 0:
        raise ValueError(f"insertion_length must be positive, got {insertion_length}")
    if insertion_start < 1:
        raise ValueError(f"insertion_start must be >= 1, got {insertion_start}")
    return PositionMap(insertion_start=insertion_start, insertion_length=insertion_length)
