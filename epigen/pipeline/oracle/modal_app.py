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

import hashlib
import json
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

# Checkpoints (oracle/checkpoint.py) live here, one subdirectory per run_id, so they survive
# a crashed/timed-out container -- see `run_mcmc_search_remote`'s docstring for how run_id is
# derived and how resume works.
CHECKPOINT_VOLUME = modal.Volume.from_name("epigen-mcmc-checkpoints", environment_name="proto-env", create_if_missing=True)
CHECKPOINT_MOUNT = "/checkpoints"


MCMC_SEARCH_TIMEOUT_S = 6 * 3600  # 6h -- was 1800s (30min), which silently killed a real ~2500-chain
# job mid-search with *no* partial results: run_mcmc_search held all chain state in memory
# and only returned once, at the very end, so any timeout/crash lost the entire run's compute,
# not just the tail. Checkpointing (below) now buys real resilience to that, not just runway --
# but this still stays generous since a run that resumes still has to re-enter the container and
# re-pay whatever latency got it here. Modal's own hard cap is 24h.


def _derive_run_id(**config: Any) -> str:
    """Deterministic run_id from a call's own config, so retrying/resuming after a crash needs
    no extra bookkeeping on the caller's side -- the *same* Streamlit "Run" click (same sequence,
    edit, window, MCMC settings, seed) always maps to the same checkpoint directory, the same way
    `pipeline_cache.cached_run_end_to_end` already keys its cache off exact call args."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


@app.function(image=image, timeout=MCMC_SEARCH_TIMEOUT_S, secrets=[CREDENTIALS_SECRET], volumes={CHECKPOINT_MOUNT: CHECKPOINT_VOLUME})
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
    checkpoint_every: int = 5,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Runs the whole MCMC search server-side. JSON-safe args only (Modal
    functions need serializable arguments) -- `structure_pdb` is raw PDB
    text, reconstructed into a real `Structure` inside the container.

    Returns a dict (not a bare candidate list) so the chain-level
    `starting_sequences`/`ending_sequences` from `MCMCSearchResult` survive
    the round trip too -- `orchestrate.py` uses those for a free (no extra
    Modal call) WT-vs-chain score comparison; see `oracle.mcmc.window_score`.

    Checkpointed to `CHECKPOINT_VOLUME` every `checkpoint_every` rounds (see
    `oracle.checkpoint`/`oracle.mcmc.run_mcmc_search`'s `checkpoint_dir`).
    `run_id` names the checkpoint subdirectory; when omitted (the normal
    case) it's derived from every other argument here, so calling this
    function again with the *exact same* arguments after a timeout/crash
    transparently resumes instead of restarting -- pass an explicit
    `run_id` only to force a fresh search under otherwise-identical config.
    The returned `run_id` is echoed back so a caller can tell them apart.
    """
    from proto_tools.entities.structures import Structure

    from epigen.pipeline.fold_invert_refold.run import FoldedStructure
    from epigen.pipeline.oracle.mcmc import run_mcmc_search

    if run_id is None:
        run_id = _derive_run_id(
            sequence=sequence,
            window_positions=window_positions,
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
    checkpoint_dir = f"{CHECKPOINT_MOUNT}/{run_id}"

    # Pick up any checkpoint another (e.g. crashed) container already committed for this
    # run_id -- a Volume's local view is only as fresh as its last reload.
    CHECKPOINT_VOLUME.reload()

    structure = Structure(structure=structure_pdb, structure_format="pdb")
    folded = FoldedStructure(
        sequence=sequence,
        structure=structure,
        plddt=1.0,
        avg_pae=0.0,
        passed_confidence_gate=True,
    )
    result = run_mcmc_search(
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
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        on_checkpoint=CHECKPOINT_VOLUME.commit,
    )
    return {
        "candidates": [
            {
                "sequence": c.sequence,
                "combined_score": c.combined_score,
                "passed_structural_check": c.passed_structural_check,
                "nt_sequence": c.nt_sequence,
            }
            for c in result.candidates
        ],
        "starting_sequences": result.starting_sequences,
        "ending_sequences": result.ending_sequences,
        "starting_nt_sequences": result.starting_nt_sequences,
        "ending_nt_sequences": result.ending_nt_sequences,
        "wt_score": result.wt_score,
        "rounds_run": result.rounds_run,
        "converged_chain_count": result.converged_chain_count,
        "run_id": run_id,
    }


def run_mcmc_search_on_modal(folded, window_positions: list[int], **kwargs) -> dict[str, Any]:
    """Local-side convenience wrapper: call the deployed `run_mcmc_search_remote`
    function from ordinary (non-Modal) Python, e.g. from `orchestrate.py`.

    Requires `modal deploy -e proto-env epigen/pipeline/oracle/modal_app.py`
    to have been run (again, after this function's return shape changed --
    most recently to add "starting_nt_sequences"/"ending_nt_sequences",
    which `orchestrate.py` uses to give the chain-starting/ending histogram
    bars a real Evo2 term instead of an implicit 0.0) before this actually
    returns the new shape. Until redeployed, the live function's dict is
    missing those two keys; `orchestrate.py` falls back to an Evo2-free
    `window_score` for the bars in that case rather than raising.

    Pass `run_id` explicitly (forwarded via `**kwargs`) to force a fresh
    search instead of resuming a same-config run's checkpoint -- see
    `run_mcmc_search_remote`'s docstring.
    """
    remote_fn = modal.Function.from_name("epigen-mcmc", "run_mcmc_search_remote", environment_name="proto-env")
    return remote_fn.remote(
        sequence=folded.sequence,
        structure_pdb=folded.structure.structure_pdb,
        window_positions=window_positions,
        **kwargs,
    )
