# Session recap

re:AGENT hackathon (Track C: bio design), Aug 15 2026. Two Claude Code
sessions worked this repo in parallel: this one (scoring/design/oracle
stack), and a second ("epigen-15") on the literature pipeline. Recap below
covers this session's work; `epigen/pipeline/literature/` was the other
session's, noted here only where it integrates with this one's code.

## Environment & infra

- `epigen` conda env (Python 3.10) with `proto-tools`
  (github.com/evo-design/proto-tools) installed.
- Modal (`proto-env` environment) authenticated and deployed:
  `esmfold2`, `esmc`, `esmif1`, `proteinmpnn`, `esm2`, `evo2`.
- `py2Dmol` installed for structure visualization.
- Git repo (`DenT94/EpiGen` on GitHub) with a launchd auto-commit agent
  (`scripts/auto_commit.sh`, every 25 min, commits + pushes if anything's
  dirty) as a safety net through the whole session.

## Pipeline built (`epigen/pipeline/`)

**Structure sourcing** (`fold_invert_refold/structure_source.py`) — always
tries the real PDB first via RCSB's sequence-search API, verifying identity
against each candidate's actual fetched sequence (not trusting RCSB's
relevance score alone); only falls back to ESMFold2 if nothing clears the
identity threshold. Verified live: correctly found `193L` (exact match) for
the CLAUDE.md lysozyme sequence, rejecting `132L` (96.1%, a point mutant)
along the way.

**Fold / invert / refold** (`fold_invert_refold/run.py`) — `fold_sequence`
(ESMFold2 + pLDDT confidence gate), `propose_compensatory_mutations`
(ProteinMPNN restricted to a window via `fixed_positions`), `refold_and_gate`
(refold + TMalign self-consistency gate).

**Oracle / MCMC search** (`oracle/`) — the mixture-of-experts mutation
search, per `mypipelinethoughts.md`:
- `scoring.py` — ESM2 (single-pass pseudo-likelihood, an approximation of
  the true masked-marginal PLL that trades a small accuracy cost for 1
  forward pass instead of L) and ProteinMPNN (structure-conditioned,
  already single-pass) per-position AA log-probs, batched across sequences.
- `evo2_scoring.py` + `codon.py` — Evo2 as a third, DNA-level expert.
  Because Evo2 needs nucleotides, `codon.py` carries a WT-derived nt
  sequence alongside the AA sequence and applies the matching codon
  substitution whenever MCMC proposes an AA change (`apply_aa_substitutions_to_nt`
  handles multi-residue edits too).
- `mcmc.py` — round-synchronized Metropolis-Hastings: every chain proposes
  one substitution per round, and all chains' proposals are scored together
  in one batched call per expert per round (3 calls/round regardless of
  chain count), recomputing fresh every round rather than reusing a stale
  precomputed table. Combined energy = weighted sum of all three experts.
- `correlation.py` — expert-agreement (Pearson correlation between ESM2 and
  ProteinMPNN) and fraction-below-WT sanity checks.
- `modal_app.py` — an *optional* accelerated path that runs the whole MCMC
  loop inside a custom Modal function (container-to-container calls to the
  already-deployed esm2/proteinmpnn/evo2 services, near-zero per-round
  latency instead of laptop↔Modal round trips). Built and the nested-call
  mechanics de-risked live (needed an explicit credentials Secret since
  ambient container identity doesn't satisfy proto-tools' own check, and
  needed deploying into `proto-env` specifically since environment
  resolution reads Modal's own ambient env var) — not yet deployed/exercised
  end-to-end, since the laptop-orchestrated path already works.

**Contact microenvironment diff** (`contact_diff/`) — Biotite `CellList`
neighbor search (CB-CB, CA for Gly) mirroring proto-tools' own internal
pattern; walks every position that actually changed (not just one nominal
edit site); a coarse hydrophobicity/charge contact-energy proxy
(`contact_energy.py`) stands in for a real Miyazawa-Jernigan table,
deliberately, to avoid hand-transcription errors — flagged as a follow-up,
not fixed this session.

**SAE feature diff** (`sae_diff/`) — the interpretability side, built out
fully this session:
- `run.py` — `diff_many_candidates` scores every MCMC candidate (not just
  the winner) in 3 total Modal calls regardless of candidate count (2 fixed
  original/edit-only + 1 batched call for all candidates).
- `pca.py` — unions each candidate's top-k ΔΔSAE features into a shared
  K-dim space, PCA-projects to 2D for a cross-candidate scatterplot.
- `describe.py` — human-readable feature labels via the one SAE
  configuration Biohub published descriptions for (`esmc_6b`/layer60),
  deliberately kept separate/on-demand from the cheap broad `esmc_300m`
  pass (hybrid strategy). Verified live: correctly identified a test K1A
  substitution by name ("Alanine and small-residue enrichment" /
  "Lysine residues and KK/KR motifs").
- `structural_viz.py` — colors a candidate's structure by one chosen SAE
  feature's per-residue activation, via `py2Dmol`. Resolved an open risk:
  `py2Dmol.view.show()` is IPython-coupled, but the HTML it builds
  (`_display_viewer`) can be grabbed directly and handed to Streamlit's
  `st.html(..., unsafe_allow_javascript=True)`.

**Alignment** (`alignment.py`) — `PositionMap` for WT↔edited numbering,
built for eventual insertion support; `contact_diff`/`sae_diff` both
exclude/reindex through it so an insertion's own residues never get
compared 1:1 against mismatched WT positions. (Insertions themselves stay
out of scope this session — substitution-only MVP, see `todo.md`.)

**Orchestration** (`orchestrate.py`) — `run_end_to_end` ties the whole chain
together: structure sourcing → apply the (now multi-residue-capable) edit →
oracle scoring/correlation → MCMC → refold+gate → contact diff → SAE diff
across all candidates. Also wires in the literature session's
`annotation_ranges`/`annotation_conflicts` (advisory flags when the edit or
compensatory window overlaps a known functional/structural site).

## Streamlit app (`epigen/app/streamlit_app.py`)

Input form → full pipeline run → structure source, literature annotation
map, oracle sanity checks, MCMC candidates table, top-candidate refold/TM-
score, contact deltas, SAE PCA scatterplot, per-candidate feature
descriptions, structure colored by a chosen SAE feature. Cached with
`st.cache_data` (identical settings load instantly instead of re-spending
several minutes of Modal calls); `developing-with-streamlit` skill (added
mid-session) caught two deprecated-API issues (`use_container_width`,
`st.components.v1.html`) that got fixed.

## Real bugs found and fixed this session

- `Structure.per_residue_plddt` is a `@property`, not a method — was
  calling it with `()`.
- `ResidueSelection({chain_id: fixed})` is invalid pydantic v2 usage
  (`BaseModel.__init__` only accepts keyword args); this was a genuine
  latent bug never caught by background testing because every test used
  `num_starting_points=1`, which skips the affected code path — the
  Streamlit form's actual default is `2`. Found via a user-reported
  traceback from a real app run, fixed and verified live.

## What's NOT done

- `oracle/modal_app.py` (whole-loop-on-Modal) — written, mechanics proven,
  never deployed/exercised end-to-end.
- Real Miyazawa-Jernigan contact-energy table (currently a coarse proxy).
- Insertions (conceptual only — see `todo.md`'s running list).
- `plot_annotation_map`'s edit marker only takes a single point/range, not
  arbitrary annotation-aware layout (peer-owned file, minor optional ask
  left with them).
- Stage 4 from CLAUDE.md, the actual grounded natural-language agent
  explanation layer tying contact deltas + SAE features + literature
  annotations into a demo-ready narrative — flagged early as the next big
  piece, not started.
- Everything explicitly deferred in the original plan: PDB/nt/multichain
  input beyond what exists, BLAST refinements, the oligo-pool library
  stage (mypipelinethoughts.md step 6).

## Notable findings along the way

- Expert correlation (ESM2 vs ProteinMPNN) dropped from ~0.84 (5-position
  toy window) to 0.67 (122-position full-protein window) once tested on a
  properly-sized sample — the earlier number was likely a small-sample
  artifact.
- A live WHSPRAL (residues 20:27) multi-residue edit ran successfully
  end-to-end with the full-protein compensatory window and all three
  experts, confirming the multi-residue-edit generalization works
  correctly (verified the exact substituted range post-edit).
