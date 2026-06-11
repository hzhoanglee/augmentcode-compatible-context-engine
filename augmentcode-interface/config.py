"""Environment-driven configuration, read once at import."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ce_url: str = field(
        default_factory=lambda: os.environ.get("AUGMENT_CE_URL", "http://127.0.0.1:6699").rstrip("/")
    )
    host: str = field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8787")))
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATA_DIR", "~/.augment-ce")).expanduser()
    )
    # Max simultaneous HTTP connections to the context engine.
    ce_max_connections: int = field(
        default_factory=lambda: int(os.environ.get("CE_MAX_CONNECTIONS", "200"))
    )
    # Retrievals running against CE at once; the rest queue (FIFO) instead of
    # overwhelming the engine or exhausting the connection pool.
    retrieve_concurrency: int = field(
        default_factory=lambda: int(os.environ.get("RETRIEVE_CONCURRENCY", "8"))
    )
    # Coalescing window for CE index triggers: a burst of batch-uploads sends
    # one trigger instead of one per request.
    index_debounce_secs: float = field(
        default_factory=lambda: float(os.environ.get("INDEX_DEBOUNCE_SECS", "2.0"))
    )

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"


CONFIG = Config()
