"""Compact mutation notation for a candidate sequence vs. a reference (e.g. WT).

Single chain: consecutive differing positions are grouped into one block and
written as `{original_block}{start_position}{mutant_block}`, blocks joined by
`_`. E.g. reference "KVFGRCELAAAM" vs candidate "KVFWHCESAPAM":

    positions 4-5 (G,R -> W,H) are adjacent  -> one block "GR4WH"
    position 8    (L -> S) is isolated       -> "L8S"
    position 10   (A -> P) is isolated       -> "A10P"
    -> "GR4WH_L8S_A10P"

Multichain: one parenthesized single-chain notation per chain, prefixed with
its chain ID and joined by `-`: "A:(GR4WH_L8S_A10P)-B:(...)".

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


def mutation_name(reference: str, candidate: str) -> str:
    """Compact single-chain mutation notation, e.g. "GR4WH_L8S_A10P".

    Returns "WT" if `candidate` is identical to `reference`.
    """
    blocks = diff_blocks(reference, candidate)
    if not blocks:
        return "WT"
    return "_".join(f"{b.original}{b.start_position}{b.mutant}" for b in blocks)


def multichain_mutation_name(chains: dict[str, tuple[str, str]]) -> str:
    """Compact multichain mutation notation: "A:(...)-B:(...)".

    `chains` maps chain_id -> (reference, candidate), in the order chains
    should appear (dict insertion order) -- pass only the chains you want
    named; a chain with no differences still renders as "A:(WT)".
    """
    return "-".join(f"{chain_id}:({mutation_name(reference, candidate)})" for chain_id, (reference, candidate) in chains.items())
