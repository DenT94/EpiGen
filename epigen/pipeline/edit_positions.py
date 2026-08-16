"""Picking multiple candidate edit positions, for a "try the edit at N different spots"
run (num_starting_points > 1 on the Landing page -- see app_pages/landing.py).
"""

from __future__ import annotations

import random


def valid_edit_starts(sequence_length: int, edit_length: int) -> list[int]:
    """1-indexed positions where an edit of `edit_length` residues fits entirely within a
    sequence of `sequence_length` residues (i.e. `start + edit_length - 1 <= sequence_length`)."""
    return list(range(1, sequence_length - edit_length + 2))


def pick_random_edit_starts(sequence_length: int, edit_length: int, n: int, *, seed: int | None = None) -> list[int]:
    """`n` distinct 1-indexed edit-start positions, uniformly sampled without replacement
    from every position the edit fits at. Sorted for a stable, readable order.

    Raises ValueError if `n` exceeds the number of valid positions -- silently capping
    would return fewer positions than the caller asked for without saying so.
    """
    candidates = valid_edit_starts(sequence_length, edit_length)
    if n > len(candidates):
        raise ValueError(
            f"Requested {n} distinct edit positions, but only {len(candidates)} valid position(s) "
            f"exist for an edit of length {edit_length} in a sequence of length {sequence_length}."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, n))
