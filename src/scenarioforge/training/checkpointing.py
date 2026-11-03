"""Atomic experiment metadata and checkpoint persistence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from scenarioforge.training.model_factory import DetectionModelAdapter
from scenarioforge.training.seed import capture_rng_state, validate_rng_state


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def save_checkpoint(
    path: Path,
    *,
    model: DetectionModelAdapter,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_validation_ap: float | None,
    metadata: dict[str, Any],
    global_step: int = 0,
    best_validation_epoch: int | None = None,
    stale_epochs: int = 0,
    rng_state: dict[str, Any] | None = None,
    train_loader_generator_state: torch.Tensor | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_ap": best_validation_ap,
        "best_validation_epoch": best_validation_epoch,
        "stale_epochs": stale_epochs,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
        "train_loader_generator_state": train_loader_generator_state,
        "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: DetectionModelAdapter,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: torch.device | str = "cpu",
    expected_metadata: Mapping[str, Any] | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid detection checkpoint: {path}")
    saved_metadata = checkpoint.get("metadata", {})
    if not isinstance(saved_metadata, dict):
        raise ValueError(f"Invalid metadata in detection checkpoint: {path}")
    for key, expected_value in (expected_metadata or {}).items():
        if saved_metadata.get(key) != expected_value:
            raise ValueError(f"Checkpoint {key} does not match this experiment")
    if restore_rng and (
        checkpoint.get("checkpoint_version") != 2
        or not isinstance(checkpoint.get("rng_state"), dict)
    ):
        raise ValueError("Checkpoint does not contain resumable RNG state")
    if restore_rng:
        validate_rng_state(checkpoint["rng_state"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is None:
        raise ValueError("Checkpoint is missing optimizer state")
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is None:
        raise ValueError("Checkpoint is missing scheduler state")
    if scaler is not None and checkpoint.get("scaler_state_dict") is None:
        raise ValueError("Checkpoint is missing GradScaler state")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if restore_rng:
        from scenarioforge.training.seed import restore_rng_state

        restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _atomic_write_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
