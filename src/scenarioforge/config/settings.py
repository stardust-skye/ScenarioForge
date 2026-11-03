"""Centralized environment-based settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    project_root: Path
    dataset_root: Path
    dataset_version: str
    database_url: str
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        """Create settings without requiring a settings-file dependency."""
        root = PROJECT_ROOT
        dataset_root = Path(
            os.getenv("NUSCENES_ROOT", str(root / "data" / "raw" / "nuscenes"))
        ).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = root / dataset_root

        default_db_path = root / "data" / "processed" / "scenarioforge.db"
        database_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path.as_posix()}")
        return cls(
            project_root=root,
            dataset_root=dataset_root.resolve(),
            dataset_version=os.getenv("NUSCENES_VERSION", "v1.0-mini"),
            database_url=database_url,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
