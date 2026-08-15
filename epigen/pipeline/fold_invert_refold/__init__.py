"""Stage 1: Fold -> invert -> refold loop.

- Fold candidate sequence (ESMFold2 via Biohub API)
- Inverse-fold structure to propose compensatory mutation candidates
  (ESM-IF / ProteinMPNN)
- Refold candidates, check self-consistency (TM-score vs target structure)
- Gate everything on pLDDT/PAE confidence before trusting a structure enough
  to invert on
"""
