"""Deterministic scene-grouped dataset splits and leakage checks."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def build_split_manifest(
    image_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_version: str,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
    scene_assignments: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Create a reproducible manifest assigning complete scenes to exactly one split."""
    if not dataset_version:
        raise ValueError("dataset_version must not be empty")
    scene_samples = _index_eligible_images(image_rows)
    scenes = sorted(scene_samples)
    if scene_assignments is None:
        normalized_ratios = _validate_ratios(ratios or DEFAULT_SPLIT_RATIOS)
        if len(scenes) < sum(value > 0 for value in normalized_ratios.values()):
            raise ValueError(
                f"Cannot populate all requested splits with {len(scenes)} scenes; "
                f"{sum(value > 0 for value in normalized_ratios.values())} non-empty splits requested"
            )
        shuffled = scenes.copy()
        random.Random(seed).shuffle(shuffled)
        counts = _allocate_scene_counts(len(scenes), normalized_ratios)
        scene_tokens: dict[str, list[str]] = {}
        start = 0
        for split in SPLIT_NAMES:
            scene_tokens[split] = sorted(shuffled[start:start + counts[split]])
            start += counts[split]
        split_strategy = "seeded_scene_grouped"
    else:
        normalized_ratios = None
        scene_tokens = _validate_explicit_assignments(scene_assignments, scenes)
        split_strategy = "explicit_scene_assignments"

    sample_tokens = {
        split: sorted(
            sample
            for scene in scene_tokens[split]
            for sample in scene_samples[scene]
        )
        for split in SPLIT_NAMES
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "dataset_version": dataset_version,
        "inventory_digest": _inventory_digest(image_rows),
        "strategy": split_strategy,
        "seed": seed if scene_assignments is None else None,
        "ratios": normalized_ratios,
        "scene_tokens": scene_tokens,
        "sample_tokens": sample_tokens,
    }
    manifest["manifest_digest"] = _manifest_digest(manifest)
    validate_split_manifest(manifest, image_rows, dataset_version=dataset_version)
    return manifest


def _index_eligible_images(
    image_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    if not image_rows:
        raise ValueError("Image manifest has no eligible image rows")
    scene_samples: dict[str, list[str]] = defaultdict(list)
    seen_samples: set[str] = set()
    seen_image_ids: set[str] = set()
    for row in image_rows:
        sample = str(row.get("sample_token") or "")
        scene = str(row.get("scene_token") or "")
        image_id = str(row.get("image_id") or "")
        if not sample or not scene or not image_id:
            raise ValueError("Every image row must have non-empty image_id, sample_token, and scene_token")
        if row.get("image_status") != "ok" or not row.get("image_path"):
            raise ValueError(
                f"Sample {sample} is not eligible for splitting: image status/path is invalid"
            )
        if sample in seen_samples:
            raise ValueError(f"Duplicate sample token in image manifest: {sample}")
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image_id in image manifest: {image_id}")
        seen_samples.add(sample)
        seen_image_ids.add(image_id)
        scene_samples[scene].append(sample)
    return dict(scene_samples)


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must have exactly these keys: {SPLIT_NAMES}")
    normalized = {split: float(ratios[split]) for split in SPLIT_NAMES}
    if any(not math.isfinite(value) or value <= 0 for value in normalized.values()):
        raise ValueError("Every requested split ratio must be positive")
    if abs(sum(normalized.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    return normalized


def _allocate_scene_counts(
    scene_count: int, ratios: Mapping[str, float]
) -> dict[str, int]:
    quotas = {split: scene_count * ratios[split] for split in SPLIT_NAMES}
    counts = {split: int(quotas[split]) for split in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        counts[split] = max(counts[split], 1)
    if sum(counts.values()) > scene_count:
        raise ValueError(
            f"{scene_count} scenes cannot populate all non-empty splits for ratios {dict(ratios)}"
        )
    while sum(counts.values()) < scene_count:
        split = max(SPLIT_NAMES, key=lambda name: (quotas[name] - counts[name], -SPLIT_NAMES.index(name)))
        counts[split] += 1
    return counts


def _validate_explicit_assignments(
    assignments: Mapping[str, Iterable[str]], scenes: Sequence[str]
) -> dict[str, list[str]]:
    if set(assignments) != set(SPLIT_NAMES):
        raise ValueError(f"Explicit scene assignments must have exactly these keys: {SPLIT_NAMES}")
    normalized = {split: [str(scene) for scene in assignments[split]] for split in SPLIT_NAMES}
    if any(not values for values in normalized.values()):
        raise ValueError("Explicit assignments must include at least one scene in every split")
    all_assigned = [scene for values in normalized.values() for scene in values]
    duplicates = sorted(scene for scene in set(all_assigned) if all_assigned.count(scene) > 1)
    if duplicates:
        raise ValueError(f"Scene tokens assigned more than once: {duplicates}")
    unknown = sorted(set(all_assigned) - set(scenes))
    missing = sorted(set(scenes) - set(all_assigned))
    if unknown or missing:
        raise ValueError(f"Explicit scene assignments mismatch image manifest; unknown={unknown}, unassigned={missing}")
    return {split: sorted(values) for split, values in normalized.items()}


def validate_split_manifest(
    manifest: Mapping[str, Any],
    image_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_version: str,
) -> dict[str, Any]:
    """Validate a split against the authoritative dataset version and image inventory."""
    _validate_manifest_structure(manifest)
    if not dataset_version:
        raise ValueError("Authoritative dataset_version must not be empty")
    if manifest.get("dataset_version") != dataset_version:
        raise ValueError(
            f"Split manifest dataset_version {manifest.get('dataset_version')!r} "
            f"does not match authoritative dataset_version {dataset_version!r}"
        )
    if manifest.get("inventory_digest") != _inventory_digest(image_rows):
        raise ValueError("Split manifest inventory provenance does not match image inventory")

    scene_samples = _index_eligible_images(image_rows)
    scene_tokens = manifest["scene_tokens"]
    sample_tokens = manifest["sample_tokens"]
    scene_sets = {split: set(scene_tokens[split]) for split in SPLIT_NAMES}
    sample_sets = {split: set(sample_tokens[split]) for split in SPLIT_NAMES}
    expected_scenes = set(scene_samples)
    assigned_scenes = set().union(*scene_sets.values())
    unknown_scenes = assigned_scenes - expected_scenes
    unassigned_scenes = expected_scenes - assigned_scenes
    if unknown_scenes or unassigned_scenes:
        raise ValueError(
            f"Scene coverage mismatch; unknown={sorted(unknown_scenes)}, "
            f"unassigned={sorted(unassigned_scenes)}"
        )
    for split in SPLIT_NAMES:
        expected_samples = {
            sample
            for scene in scene_tokens[split]
            for sample in scene_samples.get(scene, [])
        }
        if expected_samples != sample_sets[split]:
            raise ValueError(
                f"Sample membership does not match scene assignment for {split}: "
                f"missing={sorted(expected_samples - sample_sets[split])}, "
                f"unexpected={sorted(sample_sets[split] - expected_samples)}"
            )
    return {
        "valid": True,
        "scene_overlap_count": 0,
        "sample_overlap_count": 0,
        "scene_count_by_split": {split: len(scene_sets[split]) for split in SPLIT_NAMES},
        "sample_count_by_split": {split: len(sample_sets[split]) for split in SPLIT_NAMES},
        "manifest_digest": manifest["manifest_digest"],
        "dataset_version": dataset_version,
        "inventory_digest": manifest["inventory_digest"],
        "image_manifest_coverage_checked": True,
    }


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported split manifest schema_version")
    if manifest.get("manifest_digest") != _manifest_digest(manifest):
        raise ValueError("Split manifest digest mismatch")
    if not isinstance(manifest.get("dataset_version"), str) or not manifest["dataset_version"]:
        raise ValueError("Split manifest must include a dataset_version")
    if not isinstance(manifest.get("inventory_digest"), str) or not manifest["inventory_digest"]:
        raise ValueError("Split manifest must include an inventory_digest")
    scene_tokens = manifest.get("scene_tokens")
    sample_tokens = manifest.get("sample_tokens")
    if not isinstance(scene_tokens, Mapping) or not isinstance(sample_tokens, Mapping):
        raise ValueError("Split manifest must include scene_tokens and sample_tokens mappings")
    if set(scene_tokens) != set(SPLIT_NAMES) or set(sample_tokens) != set(SPLIT_NAMES):
        raise ValueError(f"Split manifest must include exactly {SPLIT_NAMES}")
    for split in SPLIT_NAMES:
        if not scene_tokens[split] or not sample_tokens[split]:
            raise ValueError(f"Split {split} must contain at least one scene and sample")
        if len(scene_tokens[split]) != len(set(scene_tokens[split])):
            raise ValueError(f"Split {split} contains duplicate scene tokens")
        if len(sample_tokens[split]) != len(set(sample_tokens[split])):
            raise ValueError(f"Split {split} contains duplicate sample tokens")
    scene_sets = {split: set(scene_tokens[split]) for split in SPLIT_NAMES}
    sample_sets = {split: set(sample_tokens[split]) for split in SPLIT_NAMES}
    scene_overlaps = _overlap_pairs(scene_sets)
    sample_overlaps = _overlap_pairs(sample_sets)
    if scene_overlaps:
        raise ValueError(f"Scenes overlap between splits: {scene_overlaps}")
    if sample_overlaps:
        raise ValueError(f"Samples overlap between splits: {sample_overlaps}")


def validate_training_selection(
    manifest: Mapping[str, Any],
    selected_sample_tokens: Iterable[str],
    *,
    image_rows: Sequence[Mapping[str, Any]],
    dataset_version: str,
) -> dict[str, Any]:
    """Require a curated training selection to be a unique subset of training samples only."""
    validate_split_manifest(manifest, image_rows, dataset_version=dataset_version)
    selected = [str(token) for token in selected_sample_tokens]
    duplicates = sorted(token for token, count in _counts(selected).items() if count > 1)
    if duplicates:
        raise ValueError(f"Training selection contains duplicate sample tokens: {duplicates}")
    train = set(manifest["sample_tokens"]["train"])
    heldout = set(manifest["sample_tokens"]["validation"]) | set(manifest["sample_tokens"]["test"])
    selected_set = set(selected)
    heldout_selected = sorted(selected_set & heldout)
    unknown = sorted(selected_set - train - heldout)
    if heldout_selected or unknown:
        raise ValueError(
            f"Training selection contains non-training samples; heldout={heldout_selected}, "
            f"unknown={unknown}"
        )
    return {
        "valid": True,
        "selected_sample_count": len(selected),
        "training_budget": len(selected),
        "heldout_samples_unchanged": True,
        "heldout_sample_count": len(heldout),
        "selection_digest": hashlib.sha256(
            "\n".join(sorted(selected)).encode("utf-8")
        ).hexdigest(),
    }


def validate_equal_training_budgets(
    manifest: Mapping[str, Any],
    selections_by_strategy: Mapping[str, Iterable[str]],
    *,
    image_rows: Sequence[Mapping[str, Any]],
    dataset_version: str,
) -> dict[str, Any]:
    """Validate training-only selections and require the same sample budget per strategy."""
    if not selections_by_strategy:
        raise ValueError("At least one strategy selection is required")
    reports = {
        strategy: validate_training_selection(
            manifest,
            selected,
            image_rows=image_rows,
            dataset_version=dataset_version,
        )
        for strategy, selected in sorted(selections_by_strategy.items())
    }
    budgets = {strategy: report["training_budget"] for strategy, report in reports.items()}
    if len(set(budgets.values())) != 1:
        raise ValueError(f"Curation strategies have unequal training budgets: {budgets}")
    return {
        "valid": True,
        "common_training_budget": next(iter(budgets.values())),
        "strategies": reports,
    }


def write_split_manifest(
    manifest: Mapping[str, Any],
    path: Path,
    *,
    image_rows: Sequence[Mapping[str, Any]],
    dataset_version: str,
) -> Path:
    """Validate against the authoritative inventory and write a stable JSON split manifest."""
    validate_split_manifest(manifest, image_rows, dataset_version=dataset_version)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_split_manifest(
    path: Path,
    *,
    image_rows: Sequence[Mapping[str, Any]],
    dataset_version: str,
) -> dict[str, Any]:
    """Load a persisted split manifest and validate it against the authoritative inventory."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest, image_rows, dataset_version=dataset_version)
    return manifest


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _inventory_digest(image_rows: Sequence[Mapping[str, Any]]) -> str:
    canonical_rows = [
        {str(key): str(value) for key, value in row.items()}
        for row in image_rows
    ]
    canonical_rows.sort(key=lambda row: row.get("sample_token", ""))
    canonical = json.dumps(
        canonical_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _overlap_pairs(values: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    overlaps = []
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            intersection = sorted(values[left] & values[right])
            if intersection:
                overlaps.append({"left": left, "right": right, "tokens": intersection})
    return overlaps


def _counts(values: Sequence[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    result.update(values)
    return result
