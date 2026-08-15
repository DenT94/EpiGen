"""Stage 4 (mypipelinethoughts.md): mixture-of-experts oracle + MCMC mutation search.

Scores every window position with two independent experts (ESM2 sequence
likelihood, ProteinMPNN structure-conditioned likelihood) from a single call
each, sanity-checks the two experts agree, then runs Metropolis-Hastings
chains over the edit window using the combined per-position score as a cheap
energy function -- no per-step model calls.
"""
