"""Split-level dataset quality and leakage statistics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from scenarioforge.vision_data.splits import (
    SPLIT_NAMES,
    validate_split_manifest,
    validate_training_selection,
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_split_statistics(
    image_rows: list[dict[str, str]],
    annotation_rows: list[dict[str, str]],
    split_manifest: dict[str, Any],
    dataset_root: Path,
    *,
    dataset_version: str,
    selected_training_samples: list[str] | None = None,
) -> dict[str, Any]:
    """Report per-split counts, distributions, path/annotation errors and leakage checks."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    integrity = validate_split_manifest(
        split_manifest,
        image_rows,
        dataset_version=dataset_version,
    )
    sample_split = {
        sample: split
        for split in SPLIT_NAMES
        for sample in split_manifest["sample_tokens"][split]
    }
    image_by_sample = {row["sample_token"]: row for row in image_rows}
    annotations_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_annotation_count = Counter()
    unassigned_annotations: list[str] = []
    seen_annotation_tokens: set[str] = set()
    for row in annotation_rows:
        annotation_token = row.get("source_annotation_token", "")
        split = sample_split.get(row.get("sample_token", ""))
        if split is None:
            unassigned_annotations.append(annotation_token)
            continue
        if not annotation_token or annotation_token in seen_annotation_tokens:
            invalid_annotation_count[split] += 1
            continue
        seen_annotation_tokens.add(annotation_token)
        try:
            _validate_annotation_row(row, image_by_sample[row["sample_token"]])
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_annotation_count[split] += 1
            continue
        annotations_by_sample[row["sample_token"]].append(row)

    split_reports: dict[str, dict[str, Any]] = {}
    scene_reports: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        sample_tokens = split_manifest["sample_tokens"][split]
        scene_tokens = split_manifest["scene_tokens"][split]
        images = [image_by_sample[token] for token in sample_tokens]
        missing_paths = [
            row["sample_token"]
            for row in images
            if not _image_path_exists(dataset_root, row.get("image_path", ""))
        ]
        class_counts: Counter[str] = Counter()
        widths: list[float] = []
        heights: list[float] = []
        areas: list[float] = []
        scene_images: Counter[str] = Counter()
        scene_annotations: Counter[str] = Counter()
        empty_images = 0
        annotation_count = 0
        for image in images:
            sample = image["sample_token"]
            scene = image["scene_token"]
            scene_images[scene] += 1
            rows = annotations_by_sample[sample]
            if not rows:
                empty_images += 1
            for row in rows:
                class_counts[row["detection_category"]] += 1
                width = float(row["x_max"]) - float(row["x_min"])
                height = float(row["y_max"]) - float(row["y_min"])
                widths.append(width)
                heights.append(height)
                areas.append(width * height)
                scene_annotations[scene] += 1
                annotation_count += 1
        for scene in sorted(scene_tokens):
            scene_reports.append({
                "split": split,
                "scene_token": scene,
                "image_count": scene_images[scene],
                "annotation_count": scene_annotations[scene],
            })
        split_reports[split] = {
            "scene_count": len(scene_tokens),
            "image_count": len(images),
            "annotation_count": annotation_count,
            "empty_image_count": empty_images,
            "class_distribution": dict(sorted(class_counts.items())),
            "images_per_scene": dict(sorted(scene_images.items())),
            "annotations_per_scene": dict(sorted(scene_annotations.items())),
            "bounding_box_size_distribution": {
                "width_px": _distribution(widths),
                "height_px": _distribution(heights),
                "area_px": _distribution(areas),
            },
            "missing_image_path_count": len(missing_paths),
            "missing_image_sample_tokens": missing_paths,
            "invalid_annotation_count": invalid_annotation_count[split],
        }
    leakage = dict(integrity)
    leakage["unassigned_annotation_count"] = len(unassigned_annotations)
    leakage["valid"] = integrity["valid"] and not unassigned_annotations
    leakage["training_selection"] = (
        validate_training_selection(
            split_manifest,
            selected_training_samples,
            image_rows=image_rows,
            dataset_version=dataset_version,
        )
        if selected_training_samples is not None
        else None
    )
    return {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "split_manifest_digest": split_manifest["manifest_digest"],
        "splits": split_reports,
        "scene_statistics": scene_reports,
        "unassigned_annotation_count": len(unassigned_annotations),
        "unassigned_annotation_tokens": unassigned_annotations,
        "leakage_validation": leakage,
    }


def write_split_statistics(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write JSON split/leakage reports and flat split/scene CSV summaries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "split_statistics.json",
        "csv": output_dir / "split_statistics.csv",
        "scene_csv": output_dir / "split_scene_statistics.csv",
        "leakage": output_dir / "leakage_report.json",
    }
    paths["json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["leakage"].write_text(
        json.dumps(report["leakage_validation"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        columns = (
            "split", "scene_count", "image_count", "annotation_count", "empty_image_count",
            "class_distribution", "bounding_box_size_distribution",
            "missing_image_path_count", "invalid_annotation_count",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for split, values in report["splits"].items():
            writer.writerow({
                "split": split,
                **{
                    key: json.dumps(values[key], sort_keys=True)
                    if isinstance(values[key], (dict, list))
                    else values[key]
                    for key in columns
                    if key != "split" and key in values
                },
            })
    with paths["scene_csv"].open("w", newline="", encoding="utf-8") as handle:
        columns = ("split", "scene_token", "image_count", "annotation_count")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["scene_statistics"])
    return paths


def _validate_annotation_row(
    row: dict[str, str], image: dict[str, str]
) -> None:
    required_equal = (
        "image_id", "sample_token", "scene_token", "image_path",
        "image_width", "image_height",
    )
    for key in required_equal:
        image_key = {"image_width": "width", "image_height": "height"}.get(key, key)
        if row.get(key) != image.get(image_key):
            raise ValueError(f"Annotation/image {key} mismatch")
    class_id = int(row["class_id"])
    if class_id <= 0 or not row.get("detection_category"):
        raise ValueError("Annotation class ID/category is invalid")
    coordinates = [float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Annotation coordinates must be finite")
    x_min, y_min, x_max, y_max = coordinates
    width, height = int(image["width"]), int(image["height"])
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError("Annotation box is outside image bounds or has non-positive area")


def _image_path_exists(dataset_root: Path, relative_path: str) -> bool:
    if not relative_path:
        return False
    path = (dataset_root / relative_path).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError:
        return False
    return path.is_file()


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": fmean(values),
        "max": max(values),
    }
