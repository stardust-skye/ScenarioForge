"""Reproducible, detection-compatible PyTorch DataLoader helpers."""

from __future__ import annotations

import csv
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from scenarioforge.vision_data.dataset import CameraDetectionDataset, load_dataset_version
from scenarioforge.vision_data.splits import (
    SPLIT_NAMES,
    validate_split_manifest,
    validate_training_selection,
)


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Keep variable-sized detection targets as a batch of individual tensors."""
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)


def make_detection_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a seeded DataLoader with safe Windows defaults."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker if num_workers else None,
        generator=generator,
    )


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the worker-specific PyTorch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_split_datasets(
    *,
    dataset_root: str,
    annotations_csv: str,
    image_manifest_csv: str,
    split_manifest: dict[str, Any],
    transforms: Mapping[str, Callable[..., Any]] | None = None,
    category_mapping_json: str | None = None,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, CameraDetectionDataset]:
    """Create independent datasets restricted to each persisted split membership."""
    image_manifest_path = Path(image_manifest_csv)
    with image_manifest_path.open(newline="", encoding="utf-8") as handle:
        image_rows = list(csv.DictReader(handle))
    dataset_version = load_dataset_version(
        image_manifest_path.with_name("dataset_manifest.json")
    )
    validate_split_manifest(
        split_manifest,
        image_rows,
        dataset_version=dataset_version,
    )
    requested_splits = tuple(splits)
    if not requested_splits or len(set(requested_splits)) != len(requested_splits):
        raise ValueError("splits must contain unique split names")
    unknown_splits = set(requested_splits) - set(SPLIT_NAMES)
    if unknown_splits:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown_splits)}")
    transforms = transforms or {}
    return {
        split: CameraDetectionDataset(
            dataset_root,
            annotations_csv,
            image_manifest_csv,
            category_mapping_json=category_mapping_json,
            sample_tokens=split_manifest["sample_tokens"][split],
            image_inventory=[
                row for row in image_rows
                if row["sample_token"] in set(split_manifest["sample_tokens"][split])
            ],
            image_inventory_provenance=image_rows,
            known_sample_tokens=[row["sample_token"] for row in image_rows],
            expected_dataset_version=dataset_version,
            transform=transforms.get(split),
        )
        for split in requested_splits
    }


def select_training_subset(
    training_dataset: CameraDetectionDataset,
    split_manifest: dict[str, Any],
    selected_sample_tokens: list[str],
    *,
    image_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Subset:
    """Create a training-only subset, rejecting any held-out or unknown sample token."""
    if not training_dataset.dataset_version:
        raise ValueError(
            "Cannot validate training selection without an authoritative dataset manifest"
        )
    audit = validate_training_selection(
        split_manifest,
        selected_sample_tokens,
        image_rows=image_rows or training_dataset.inventory,
        dataset_version=training_dataset.dataset_version,
    )
    train_samples = set(split_manifest["sample_tokens"]["train"])
    dataset_samples = set(training_dataset.sample_tokens)
    if not dataset_samples <= train_samples:
        raise ValueError("Curation input dataset contains samples outside the training split")
    indices = {token: index for index, token in enumerate(training_dataset.sample_tokens)}
    missing = sorted(set(selected_sample_tokens) - set(indices))
    if missing:
        raise ValueError(f"Selected samples are absent from the training dataset: {missing[:10]}")
    training_dataset.training_selection_audit = audit
    return Subset(training_dataset, [indices[token] for token in selected_sample_tokens])
