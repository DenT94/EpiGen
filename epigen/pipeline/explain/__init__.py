"""Stage 4: Agent explanation layer.

Receives contact deltas, SAE feature deltas, self-consistency TM-score, and
motif accessibility check. Must ground every claim in the numeric deltas
provided -- flag internally if a claimed compensation contradicts the
numbers (e.g. claims rescue but delta_contact_energy is unfavorable and
TM-score dropped). Outputs a plain-language explanation keyed to specific
evidence.
"""
