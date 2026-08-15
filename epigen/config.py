"""Central config loader.

Loads `.env` (gitignored, see .env.example) and exposes the settings the
pipeline needs. Import `settings` rather than reading os.environ directly,
so there's one place that knows what's required vs optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    biohub_api_key: str | None = os.environ.get("BIOHUB_API_KEY") or None
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY") or None

    def require_biohub_api_key(self) -> str:
        if not self.biohub_api_key:
            raise RuntimeError(
                "BIOHUB_API_KEY is not set. Copy .env.example to .env and "
                "fill it in."
            )
        return self.biohub_api_key

    def require_anthropic_api_key(self) -> str:
        # Stage 4 (epigen.pipeline.explain) needs this to call Claude.
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
                "fill it in."
            )
        return self.anthropic_api_key


settings = Settings()
