"""Coarse pairwise contact-energy proxy.

CLAUDE.md allows "Miyazawa-Jernigan or coarse hydrophobicity/charge
complementarity" for Δcontact_energy. This module implements the latter:
hand-transcribing the full 20x20 MJ (1996) matrix from memory risks silent
numeric errors in a tool whose whole point is grounding claims in trustworthy
deltas, so we use the well-known Kyte-Doolittle hydrophobicity scale plus a
simple charge model instead. Swap in a verified, citable MJ table later if
there's time -- `contact_energy()` is the only function callers need to
change.

Convention matches MJ: negative = favorable contact, positive = unfavorable.
"""

from __future__ import annotations

# Kyte & Doolittle (1982) hydrophobicity scale, keyed by one-letter code.
_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}  # fmt: skip

# Formal side-chain charge at physiological pH.
_CHARGE: dict[str, float] = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

_HYDROPHOBIC_WEIGHT = 0.1  # 1/10, keeps hydrophobic term roughly MJ-scale (units of ~kT)
_CHARGE_WEIGHT = 2.0

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


def contact_energy(aa1: str, aa2: str) -> float:
    """Coarse contact energy between two residues (one-letter or three-letter codes).

    Negative = favorable (both hydrophobic, or opposite charges attracting).
    Positive = unfavorable (like charges repelling).
    """
    a1 = THREE_TO_ONE.get(aa1.upper(), aa1.upper())
    a2 = THREE_TO_ONE.get(aa2.upper(), aa2.upper())
    h1, h2 = _HYDROPHOBICITY.get(a1, 0.0), _HYDROPHOBICITY.get(a2, 0.0)
    q1, q2 = _CHARGE.get(a1, 0.0), _CHARGE.get(a2, 0.0)
    return -_HYDROPHOBIC_WEIGHT * (h1 * h2) + _CHARGE_WEIGHT * (q1 * q2)
