"""Custom Modal app: runs the MCMC search loop entirely inside Modal's network.

Wraps epigen's own `oracle.mcmc.run_mcmc_search` (reused, not reimplemented)
so each round's calls to the esm2/proteinmpnn/evo2 services (deployed
separately via `proto-tools deploy`) happen container-to-container instead
of laptop-to-Modal -- cutting per-round latency to near zero. This is an
*optional* accelerated path: `orchestrate.run_end_to_end`'s default
(laptop-orchestrated, per-round Modal calls) already works and is verified
end-to-end; this module only matters once step counts get large enough that
round-trip latency dominates.

Two things this needed, discovered by trial and error (see chat history):
- `_require_modal_credentials` inside a container doesn't see ambient Modal
  identity -- MODAL_TOKEN_ID/MODAL_TOKEN_SECRET must be injected explicitly
  via a Secret (`epigen-nested-modal-auth`, created with
  `modal secret create ... --env proto-env`).
- proto-tools' target-environment resolution reads Modal's own ambient
  `MODAL_ENVIRONMENT`, which reflects *this app's own* deployment
  environment -- overriding it via `env={}` doesn't stick. The fix is
  deploying this app itself into `proto-env` (matching where esm2/
  proteinmpnn/evo2 live), not overriding the variable.

Deploy:  modal deploy -e proto-env epigen/pipeline/oracle/modal_app.py
Call:    see `run_mcmc_search_on_modal()` below (local-side convenience
         wrapper) or `modal run -e proto-env ...::run_mcmc_search_remote`
         directly for a one-off.
"""

from __future__ import annotations

from typing import Any

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("proto-tools @ git+https://github.com/evo-design/proto-tools.git")
    .add_local_python_source("epigen")
)

app = modal.App("epigen-mcmc")

CREDENTIALS_SECRET = modal.Secret.from_name("epigen-nested-modal-auth", environment_name="proto-env")


@app.function(image=image, timeout=1800, secrets=[CREDENTIALS_SECRET])
def run_mcmc_search_remote(
    sequence: str,
    structure_pdb: str,
    window_positions: list[int],
    *,
    chain_id: str = "A",
    num_starting_points: int = 2,
    chains_per_start: int = 2,
    steps: int = 50,
    temperature: float = 1.0,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
    nt_sequence: str | None = None,
    weight_evo2: float = 0.5,
    refold_every: int | None = None,
    candidate_num: int = 5,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Runs the whole MCMC search server-side. JSON-safe args only (Modal
    functions need serializable arguments) -- `structure_pdb` is raw PDB
    text, reconstructed into a real `Structure` inside the container.
    """
    from proto_tools.entities.structures import Structure

    from epigen.pipeline.fold_invert_refold.run import FoldedStructure
    from epigen.pipeline.oracle.mcmc import run_mcmc_search

    structure = Structure(structure=structure_pdb, structure_format="pdb")
    folded = FoldedStructure(
        sequence=sequence,
        structure=structure,
        plddt=1.0,
        avg_pae=0.0,
        passed_confidence_gate=True,
    )
    candidates = run_mcmc_search(
        folded,
        window_positions,
        chain_id=chain_id,
        num_starting_points=num_starting_points,
        chains_per_start=chains_per_start,
        steps=steps,
        temperature=temperature,
        weight_esm2=weight_esm2,
        weight_pmpnn=weight_pmpnn,
        nt_sequence=nt_sequence,
        weight_evo2=weight_evo2,
        refold_every=refold_every,
        candidate_num=candidate_num,
        seed=seed,
    )
    return [
        {
            "sequence": c.sequence,
            "combined_score": c.combined_score,
            "passed_structural_check": c.passed_structural_check,
            "nt_sequence": c.nt_sequence,
        }
        for c in candidates
    ]


def run_mcmc_search_on_modal(folded, window_positions: list[int], **kwargs) -> list[dict[str, Any]]:
    """Local-side convenience wrapper: call the deployed `run_mcmc_search_remote`
    function from ordinary (non-Modal) Python, e.g. from `orchestrate.py`.

    Requires `modal deploy -e proto-env epigen/pipeline/oracle/modal_app.py`
    to have been run first.
    """
    remote_fn = modal.Function.from_name("epigen-mcmc", "run_mcmc_search_remote", environment_name="proto-env")
    return remote_fn.remote(
        sequence=folded.sequence,
        structure_pdb=folded.structure.structure_pdb,
        window_positions=window_positions,
        **kwargs,
    )
