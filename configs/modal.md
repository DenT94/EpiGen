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
