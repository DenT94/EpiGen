# Modal setup notes

- Auth: `modal token info` shows workspace `dent94` (already authenticated on
  this machine via cached `~/.modal.toml`).
- Environment: `proto-env` (created for this project; proto-tools' default).
- Deployed apps (via `proto-tools deploy --apps <name> --env proto-env`):
  - `esmif1` — ESM-IF1 inverse folding
  - `proteinmpnn` — ProteinMPNN inverse folding
  - `esmfold2` — all-atom structure prediction (stage 1 fold/refold)
  - `esmc` — embeddings + SAE features (`esmc-sae-features` reuses this same
    deployed app, just a different `operation` in the dispatch payload)
  - `esm2` — sequence log-likelihood scoring (oracle stage's ESM2 expert)
  - `evo2` — DNA-level causal sequence scoring (oracle stage's third expert;
    see epigen/pipeline/oracle/evo2_scoring.py + codon.py). Deploy of this
    one took unusually long (still `initializing` after ~50 min on 2026-08-15
    ~14:26 PDT vs. 1-3 min for the other apps) -- plausibly a large weight
    download for the deploy's warmup pass; check `modal app list --env
    proto-env` for current state before assuming something's wrong.
  - `epigen-mcmc` (a separate custom app, not a `proto-tools deploy` target)
    -- runs the MCMC loop server-side, see oracle/modal_app.py. Deploy with
    `modal deploy -e proto-env epigen/pipeline/oracle/modal_app.py`. Needs
    the `epigen-nested-modal-auth` Secret (already created, in `proto-env`)
    for its nested calls to esm2/proteinmpnn/evo2 -- ambient container
    identity does NOT satisfy proto-tools' own credential check, and its
    target-environment resolution reads Modal's ambient `MODAL_ENVIRONMENT`
    (this app's own deploy environment), not an injected env var -- that's
    why this app must itself be deployed into `proto-env`, not just given
    `env={"MODAL_ENVIRONMENT": "proto-env"}`.

  Decision: ESMFold2/ESMC/ESMC-SAE run through proto-tools+Modal like
  everything else, not the hosted Biohub API — see CLAUDE.md's Stack
  section. `.env`/`BIOHUB_API_KEY` is unused as a result.

## Using a deployed tool from code

```python
from proto_tools import run_proteinmpnn, ProteinMPNNInput, ProteinMPNNConfig

output = run_proteinmpnn(
    ProteinMPNNInput(...),
    ProteinMPNNConfig(device="modal"),
)
```

## Cost hygiene

- Cached weights persist on a Modal volume and accrue storage cost — remove
  deployments after the hackathon if not reused (`modal app list` /
  Modal dashboard).
- `PROTO_MODAL_SCALEDOWN_WINDOW` (seconds, default 30) controls how long a
  container stays warm after a call. Bump it during active dev to avoid
  cold-start latency:
  ```bash
  export PROTO_MODAL_SCALEDOWN_WINDOW=300
  ```
