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

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"


CONFIG = Config()
