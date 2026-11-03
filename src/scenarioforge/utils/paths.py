"""Filesystem path helpers."""

from pathlib import Path


def ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
