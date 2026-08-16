"""Compact mutation notation for a candidate sequence vs. a reference (e.g. WT).

Single chain: consecutive differing positions are grouped into one block and
written as `{original_block}{start_position}{mutant_block}`. Blocks separated
by a run of `<= max_gap` unchanged residues (default 5) are then merged into
one combined block spanning both, including the unchanged residues in
between -- matching the merge rule in EnViDash's canonical naming module
(`src/alignment/get_name_string.py`'s `full_string()`), the sibling lab
codebase this notation is meant to stay consistent with. Blocks still
farther apart than `max_gap` stay separate, joined by `_`.

E.g. reference "KVFGRCELAAAM" vs candidate "KVFWHCESAPAM" (mutations at
positions 4-5, 8, 10 -- gaps of 2 and 1 residues, both `<= max_gap=5`):

    -> "GRCELAA4WHCESAP"  (one merged block, positions 4-10)

whereas with `max_gap=0` (strict adjacency only, no merging) the same input
gives three separate blocks: "GR4WH_L8S_A10P".

Multichain: one parenthesized single-chain notation per chain, prefixed with
its chain ID and joined by `-`: "A:(...)-B:(...)".

Substitution-only (matches CLAUDE.md's MVP scope): both sequences compared
must be the same length. Insertions would shift every downstream position,
which this notation doesn't attempt to express -- see `alignment.py`'s
`PositionMap` for that problem instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationBlock:
    """One maximal run of consecutive differing positions."""

    start_position: int  # 1-indexed
    original: str  # one or more residues, contiguous in `reference`
    mutant: str  # same length as `original`, contiguous in `candidate`


def diff_blocks(reference: str, candidate: str) -> list[MutationBlock]:
    """Maximal runs of consecutive differing positions between two same-length sequences.

    Raises:
        ValueError: if `reference` and `candidate` differ in length.
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"diff_blocks is substitution-only: reference has {len(reference)} residues, "
            f"candidate has {len(candidate)} -- lengths must match."
        )
    blocks: list[MutationBlock] = []
    i, n = 0, len(reference)
    while i < n:
        if reference[i] == candidate[i]:
            i += 1
            continue
        start = i
        while i < n and reference[i] != candidate[i]:
            i += 1
        blocks.append(MutationBlock(start_position=start + 1, original=reference[start:i], mutant=candidate[start:i]))
    return blocks


def _merge_nearby_blocks(blocks: list[MutationBlock], reference: str, candidate: str, max_gap: int) -> list[MutationBlock]:
    """Merge blocks separated by `<= max_gap` unchanged residues into one combined block
    spanning both (including the unchanged residues in between) -- EnViDash's
    `full_string()` merge rule, ported here so the two codebases' naming stays consistent.
    """
    if not blocks:
        return []
    merged = [blocks[0]]
    for block in blocks[1:]:
        prev = merged[-1]
        prev_end = prev.start_position + len(prev.original)  # 1-indexed, one past prev's last residue
        gap = block.start_position - prev_end
        if gap <= max_gap:
            end = block.start_position - 1 + len(block.original)  # 1-indexed, inclusive end of block
            start0 = prev.start_position - 1
            merged[-1] = MutationBlock(
                start_position=prev.start_position,
                original=reference[start0:end],
                mutant=candidate[start0:end],
            )
        else:
            merged.append(block)
    return merged


def mutation_name(reference: str, candidate: str, *, identical_label: str = "identical", max_gap: int = 5) -> str:
    """Compact single-chain mutation notation, e.g. "GRCELAA4WHCESAP" (default
    `max_gap=5` merges nearby mutations into one block -- see module docstring).

    Returns `identical_label` if `candidate` is identical to `reference`. The
    default is the generic "identical", not "WT" -- callers here routinely
    name against `edit_only.sequence`, not true WT (e.g. an MCMC candidate
    whose trajectory never found an accepted compensatory move), and "WT"
    would misleadingly claim that sequence is wild-type when it still
    carries the fixed edit. Pass `identical_label="WT"` explicitly at a call
    site that really does compare against true WT.
    """
    blocks = _merge_nearby_blocks(diff_blocks(reference, candidate), reference, candidate, max_gap)
    if not blocks:
        return identical_label
    return "_".join(f"{b.original}{b.start_position}{b.mutant}" for b in blocks)


def multichain_mutation_name(
    chains: dict[str, tuple[str, str]], *, identical_label: str = "identical", max_gap: int = 5
) -> str:
    """Compact multichain mutation notation: "A:(...)-B:(...)".

    `chains` maps chain_id -> (reference, candidate), in the order chains
    should appear (dict insertion order) -- pass only the chains you want
    named; a chain with no differences renders as `A:({identical_label})`.
    """
    return "-".join(
        f"{chain_id}:({mutation_name(reference, candidate, identical_label=identical_label, max_gap=max_gap)})"
        for chain_id, (reference, candidate) in chains.items()
    )
