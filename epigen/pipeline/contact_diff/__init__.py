"""Stage 2: Contact microenvironment diff.

For each position near the edit, build a neighbor list (<10 A, prefer CB-CB
or sidechain heavy-atom distance over CA-CA). Per neighbor pair, compute
real numeric deltas -- not categorical labels:
- delta_distance (original vs candidate structure)
- delta_contact_energy (Miyazawa-Jernigan or coarse hydrophobicity/charge
  complementarity)
- local delta_pLDDT / delta_PAE
"""
