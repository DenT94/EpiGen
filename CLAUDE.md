# EpiGen

Protease-gated selective antibiotic design tool with agentic compensatory-mutation
explanation. Built for re:AGENT (GXL/Anthropic/Biohub hackathon, Aug 15-16 2026, SF).

## Core idea

Broad-spectrum antibiotic proteins kill commensal ("good") bacteria alongside
pathogens. EpiGen engineers a cleavage motif for a commensal-secreted protease into
the antibiotic protein's surface loop, so nearby commensals locally inactivate the
drug while it stays active against everything else. The insertion destabilizes the
scaffold; EpiGen finds and explains compensatory mutations that restore structure
while preserving the inserted motif's surface accessibility.

## Demo case

- Scaffold: hen egg-white lysozyme (129 aa, well-characterized, fast-folding)
- Insertion: lactocepin (PrtP) substrate motif, surface-exposed loop
- Goal: pathogen-killing activity preserved; activity locally silenced near
  *Lactobacillus* spp. secreting lactocepin

## Pipeline

1. **Fold → invert → refold loop**
   - Fold candidate sequence (ESMFold2 via Biohub API)
   - Inverse-fold structure to propose compensatory mutation candidates
     (ESM-IF / ProteinMPNN)
   - Refold candidates, check self-consistency (TM-score vs target structure)
   - Gate everything on pLDDT/PAE confidence before trusting a structure enough
     to invert on

2. **Contact microenvironment diff**
   - For each position near the edit, build neighbor list (<10 Å, prefer
     CB-CB or sidechain heavy-atom distance over CA-CA)
   - Per neighbor pair, compute real deltas — not just identity labels:
     - Δdistance (original vs candidate structure)
     - Δcontact energy (Miyazawa-Jernigan or coarse hydrophobicity/charge
       complementarity)
     - local ΔpLDDT/ΔPAE
   - These numeric deltas are what get passed to the explanation agent —
     never raw categorical "neighbor changed" labels alone

3. **ESMC SAE feature diff**
   - Run original / edit-only / compensated sequences through ESMC
   - Use Biohub's ESMC SAE model to get interpretable feature activations
   - Diff feature vectors across the three states; surface top-k
     highest-delta features (don't attempt a full feature atlas — scope to
     top-k for hackathon time budget)
   - Watch specifically for features associated with solvent-exposed/
     disordered loop regions, since that's what determines whether the
     lactocepin motif stays cleavable

4. **Agent explanation layer**
   - Agent receives: contact deltas, SAE feature deltas, self-consistency
     TM-score, motif accessibility check
   - Must ground every claim in the numeric deltas provided — flag internally
     if a claimed compensation contradicts the numbers (e.g. claims rescue
     but Δcontact_energy is unfavorable and TM-score dropped)
   - Output: plain-language explanation of whether/how the compensatory
     mutation rescues stability, keyed to specific evidence

## Stack

- Streamlit for UI (input: edit position + insertion sequence; output:
  ranked compensatory candidates + agent explanations)
- ESMFold2, ESMC, ESMC SAE model via `proto-tools` + Modal (`device="modal"`,
  `proto-env` environment) rather than a hosted Biohub API — proto-tools
  downloads Biohub's open weights and runs them on Modal directly, so no API
  key is needed. `.env`/`BIOHUB_API_KEY` is left in place but unused.
- Local: contact-map diff + energy scoring, orchestration logic
- Proto for orchestrating the design part
- Modal for compute
- Paperclip for literature search

## Validation / ground truth

- Confirm lactocepin substrate motif stays surface-accessible in
  compensated structure (not buried by repacking)
- Self-consistency TM-score as top-level sanity check before trusting any
  local signal
- If time allows, sanity-check against any published lysozyme stability
  mutation data

## Scope discipline (2-day build)

- Lock architecture now; do not add a 4th signal type mid-weekend
- SAE step is scoped to top-k delta features only, no full interpretability
  sweep
- Demo narrative: "here's the contact/energy evidence, here's the SAE
  interpretability read, here's the agent's grounded explanation" — three
  legs, don't overbuild any single one at the expense of finishing the loop
  end to end
