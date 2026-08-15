"""Stage 3: ESMC SAE feature diff.

Run original / edit-only / compensated sequences through ESMC, use Biohub's
ESMC SAE model to get interpretable feature activations, diff feature
vectors across the three states, and surface top-k highest-delta features
(scoped to top-k only for hackathon time budget -- no full feature atlas).
"""
