"""Read-only nuScenes lookup tables that Part 1 ingestion does not persist.

The Part 1 database stores visibility and attribute *tokens*, ego-pose *tokens*, and log
tokens. The human-readable labels and ego-pose coordinates live in small JSON tables inside
the dataset metadata directory. This reader loads them without touching the database schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NuScenesLookupError(RuntimeError):
    """Raised when a required nuScenes metadata table cannot be read."""


@dataclass(frozen=True)
class NuScenesLookups:
    """Token-to-value mappings read directly from nuScenes metadata JSON files."""

    visibility_levels: dict[str, str] = field(default_factory=dict)
    attribute_names: dict[str, str] = field(default_factory=dict)
    log_locations: dict[str, str] = field(default_factory=dict)
    # token -> {"timestamp": int, "translation": [x, y, z], "rotation": [w, x, y, z]}
    ego_poses: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, dataroot: Path, version: str = "v1.0-mini") -> NuScenesLookups:
        """Load lookup tables; a missing or malformed file raises instead of being skipped."""
        metadata_dir = Path(dataroot).expanduser().resolve() / version
        return cls(
            visibility_levels={
                row["token"]: row["level"] for row in _read_table(metadata_dir, "visibility")
            },
            attribute_names={
                row["token"]: row["name"] for row in _read_table(metadata_dir, "attribute")
            },
            log_locations={
                row["token"]: row["location"] for row in _read_table(metadata_dir, "log")
            },
            ego_poses={
                row["token"]: {
                    "timestamp": row["timestamp"],
                    "translation": row["translation"],
                    "rotation": row["rotation"],
                }
                for row in _read_table(metadata_dir, "ego_pose")
            },
        )


def _read_table(metadata_dir: Path, table: str) -> list[dict[str, Any]]:
    path = metadata_dir / f"{table}.json"
    try:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NuScenesLookupError(f"Cannot read nuScenes metadata table {path}: {exc}") from exc
    if not isinstance(rows, list):
        raise NuScenesLookupError(f"Expected a JSON list in {path}")
    try:
        for row in rows:
            row["token"]
    except (KeyError, TypeError) as exc:
        raise NuScenesLookupError(f"Malformed record without token in {path}") from exc
    logger.debug("Loaded %d rows from %s", len(rows), path)
    return rows
