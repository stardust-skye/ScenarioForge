"""Lazy PyTorch detection dataset over the Part 3A CSV exports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from scenarioforge.vision_data.transforms import build_eval_transforms


class CameraDetectionDataset(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Read one image and its accepted 2D annotations lazily for each dataset item."""

    def __init__(
        self,
        dataset_root: Path,
        annotations_csv: Path,
        image_manifest_csv: Path,
        *,
        category_mapping_json: Path | None = None,
        sample_tokens: Sequence[str] | None = None,
        image_inventory: Sequence[Mapping[str, str]] | None = None,
        image_inventory_provenance: Sequence[Mapping[str, str]] | None = None,
        known_sample_tokens: Sequence[str] | None = None,
        expected_dataset_version: str | None = None,
        transform: Callable[[Any, dict[str, Tensor]], tuple[Any, dict[str, Tensor]]] | None = None,
        target_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        validate_image_paths: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.annotations_csv = Path(annotations_csv)
        self.image_manifest_csv = Path(image_manifest_csv)
        self.dataset_manifest_path = self.image_manifest_csv.with_name("dataset_manifest.json")
        stored_dataset_version = (
            load_dataset_version(self.dataset_manifest_path)
            if self.dataset_manifest_path.is_file()
            else None
        )
        if expected_dataset_version is not None and stored_dataset_version != expected_dataset_version:
            raise ValueError(
                "Dataset version changed or is unavailable during split dataset construction"
            )
        self.dataset_version = expected_dataset_version or stored_dataset_version
        self.transform = transform or build_eval_transforms()
        self.target_transform = target_transform
        self.images = (
            [dict(row) for row in image_inventory]
            if image_inventory is not None
            else _read_csv(self.image_manifest_csv)
        )
        self.inventory = (
            [dict(row) for row in image_inventory_provenance]
            if image_inventory_provenance is not None
            else list(self.images)
        )
        self._known_sample_tokens = set(known_sample_tokens or ())
        self._image_by_sample: dict[str, dict[str, str]] = {}
        self._image_ids: dict[str, int] = {}
        seen_ids: set[int] = set()
        for row in self.images:
            sample = _required(row, "sample_token", "image manifest")
            image_id = _required(row, "image_id", "image manifest")
            if sample in self._image_by_sample:
                raise ValueError(f"Duplicate sample_token in image manifest: {sample}")
            if image_id in self._image_ids:
                raise ValueError(f"Duplicate image_id in image manifest: {image_id}")
            if row.get("image_status") != "ok":
                raise ValueError(f"Image {sample} is not valid: status={row.get('image_status')!r}")
            path = self._resolve_image_path(row.get("image_path", ""), sample)
            if validate_image_paths and not path.is_file():
                raise FileNotFoundError(f"Image for sample {sample} does not exist: {path}")
            width, height = _positive_int(row.get("width"), f"{sample}.width"), _positive_int(
                row.get("height"), f"{sample}.height"
            )
            if row.get("scene_token") is None or not row["scene_token"]:
                raise ValueError(f"Image {sample} has no scene_token")
            if validate_image_paths:
                try:
                    with Image.open(path) as image:
                        image.verify()
                        if image.size != (width, height):
                            raise ValueError(
                                f"Image {sample} dimensions {image.size} do not match "
                                f"manifest {(width, height)}"
                            )
                except (OSError, Image.DecompressionBombError) as exc:
                    raise ValueError(f"Image for sample {sample} cannot be decoded: {path}") from exc
            stable_id = _stable_image_id(image_id)
            if stable_id in seen_ids:
                raise ValueError(f"Stable integer image_id collision for {image_id}")
            seen_ids.add(stable_id)
            self._image_ids[image_id] = stable_id
            self._image_by_sample[sample] = row
            self._known_sample_tokens.add(sample)

        self.source_id_to_model_label, self.model_label_to_source_id = _load_label_mapping(
            category_mapping_json or self.annotations_csv.parent / "category_mapping.json",
            self.annotations_csv,
        )
        self._annotations: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._load_annotations()

        if sample_tokens is None:
            selected = set(self._image_by_sample)
        else:
            requested = [str(token) for token in sample_tokens]
            if len(requested) != len(set(requested)):
                raise ValueError("sample_tokens contains duplicates")
            unknown = sorted(set(requested) - set(self._image_by_sample))
            if unknown:
                raise ValueError(f"Requested samples are absent from image manifest: {unknown[:10]}")
            selected = set(requested)
        self.image_records = tuple(
            row for row in self.images if row["sample_token"] in selected
        )
        if len(self.image_records) != len(selected):
            raise ValueError("Selected sample tokens did not resolve to unique image records")
        self.sample_tokens = tuple(row["sample_token"] for row in self.image_records)
        self.scene_tokens = tuple(row["scene_token"] for row in self.image_records)
        self.annotation_counts = {
            sample: len(self._annotations.get(sample, []))
            for sample in self.sample_tokens
        }
        self.training_selection_audit: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.image_records)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        row = self.image_records[index]
        sample = row["sample_token"]
        path = self._resolve_image_path(row["image_path"], sample)
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Image for sample {sample} cannot be loaded: {path}") from exc
        boxes_rows = self._annotations.get(sample, [])
        boxes = torch.tensor(
            [
                [float(item["x_min"]), float(item["y_min"]), float(item["x_max"]), float(item["y_max"])]
                for item in boxes_rows
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        labels = torch.tensor(
            [
                self.source_id_to_model_label[int(item["class_id"])]
                for item in boxes_rows
            ],
            dtype=torch.int64,
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(self._image_ids[row["image_id"]], dtype=torch.int64),
            "area": _box_areas(boxes),
            "iscrowd": torch.zeros((len(boxes_rows),), dtype=torch.int64),
        }
        image, target = self.transform(image, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        self._validate_output(image, target, row)
        target["area"] = _box_areas(target["boxes"])
        return image, target

    def _load_annotations(self) -> None:
        source_tokens: set[str] = set()
        with self.annotations_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample = _required(row, "sample_token", "annotation")
                image = self._image_by_sample.get(sample)
                if image is None:
                    if sample in self._known_sample_tokens:
                        continue
                    raise ValueError(f"Annotation references unknown sample_token {sample}")
                for key, image_key in (
                    ("image_id", "image_id"),
                    ("scene_token", "scene_token"),
                    ("image_path", "image_path"),
                    ("image_width", "width"),
                    ("image_height", "height"),
                ):
                    if row.get(key) != image.get(image_key):
                        raise ValueError(
                            f"Annotation {row.get('source_annotation_token')} has {key} that "
                            f"does not match image manifest for {sample}"
                        )
                token = _required(row, "source_annotation_token", "annotation")
                if token in source_tokens:
                    raise ValueError(f"Duplicate source annotation token: {token}")
                source_tokens.add(token)
                if row.get("conversion_status") != "accepted":
                    raise ValueError(f"Annotation {token} is not marked accepted")
                try:
                    source_class_id = int(row["class_id"])
                    model_label = self.source_id_to_model_label[source_class_id]
                    coordinates = [
                        float(row[key])
                        for key in ("x_min", "y_min", "x_max", "y_max")
                    ]
                    width, height = int(image["width"]), int(image["height"])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"Annotation {token} has invalid class/box values") from exc
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(f"Annotation {token} has non-finite box coordinates")
                x_min, y_min, x_max, y_max = coordinates
                if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                    raise ValueError(f"Annotation {token} box is invalid or outside image bounds")
                if model_label <= 0:
                    raise ValueError(f"Annotation {token} maps to background label 0")
                self._annotations[sample].append(row)

    def _resolve_image_path(self, relative_path: str, sample: str) -> Path:
        if not relative_path:
            raise ValueError(f"Image {sample} has an empty image_path")
        path = (self.dataset_root / relative_path).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as exc:
            raise ValueError(f"Image path for {sample} escapes dataset root: {relative_path}") from exc
        return path

    def _validate_output(self, image: Any, target: dict[str, Tensor], row: dict[str, str]) -> None:
        if not isinstance(image, Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Image transform must return a float tensor with shape [3, H, W]")
        if image.dtype != torch.float32 or not torch.isfinite(image).all():
            raise ValueError("Image transform must return finite float32 values")
        required = {"boxes", "labels", "image_id", "area", "iscrowd"}
        if not required <= target.keys():
            raise ValueError(f"Target transform removed required fields: {sorted(required - target.keys())}")
        boxes, labels = target["boxes"], target["labels"]
        if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Target boxes must be a float32 tensor with shape [N, 4]")
        if labels.dtype != torch.int64 or labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
            raise ValueError("Target labels must be int64 with shape [N]")
        if target["iscrowd"].dtype != torch.int64 or target["iscrowd"].shape != labels.shape:
            raise ValueError("Target iscrowd must be int64 with shape [N]")
        if target["image_id"].dtype != torch.int64 or target["image_id"].numel() != 1:
            raise ValueError("Target image_id must be a unique int64 tensor with one element")
        if boxes.numel():
            if not torch.isfinite(boxes).all():
                raise ValueError("Transformed boxes contain non-finite values")
            height, width = image.shape[-2:]
            if not (
                torch.all(boxes[:, 0] >= 0)
                and torch.all(boxes[:, 1] >= 0)
                and torch.all(boxes[:, 2] <= width)
                and torch.all(boxes[:, 3] <= height)
                and torch.all(boxes[:, 2] > boxes[:, 0])
                and torch.all(boxes[:, 3] > boxes[:, 1])
            ):
                raise ValueError(
                    f"Transformed boxes for sample {row['sample_token']} are invalid for "
                    f"image dimensions {(height, width)}"
                )
        if labels.numel() and (torch.any(labels <= 0) or torch.any(labels > len(self.model_label_to_source_id))):
            raise ValueError("Target labels must be valid positive foreground model labels")


def _load_label_mapping(
    mapping_path: Path, annotations_csv: Path
) -> tuple[dict[int, int], dict[int, int]]:
    source_to_name: dict[int, str] = {}
    if Path(mapping_path).is_file():
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        for item in payload.get("classes", []):
            source_id = int(item["class_id"])
            name = str(item["name"])
            if source_id <= 0 or source_id in source_to_name:
                raise ValueError(f"Invalid source class ID in mapping: {source_id}")
            source_to_name[source_id] = name
    else:
        for row in _read_csv(annotations_csv):
            source_id = int(row["class_id"])
            name = row["detection_category"]
            if source_id in source_to_name and source_to_name[source_id] != name:
                raise ValueError(f"Source class ID {source_id} maps to multiple categories")
            source_to_name[source_id] = name
    if not source_to_name:
        raise ValueError(f"No foreground classes found in mapping or annotations: {mapping_path}")
    ordered_source_ids = sorted(source_to_name)
    source_to_model = {source_id: index + 1 for index, source_id in enumerate(ordered_source_ids)}
    return source_to_model, {model_label: source_id for source_id, model_label in source_to_model.items()}


def load_dataset_version(dataset_manifest_path: Path) -> str:
    """Read the authoritative version associated with a processed image inventory."""
    payload = json.loads(Path(dataset_manifest_path).read_text(encoding="utf-8"))
    dataset_version = payload.get("dataset_version") if isinstance(payload, dict) else None
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError(f"Dataset manifest has no valid dataset_version: {dataset_manifest_path}")
    return dataset_version


def _stable_image_id(image_id: str) -> int:
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _box_areas(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _required(row: dict[str, str], key: str, source: str) -> str:
    value = row.get(key)
    if value is None or not value:
        raise ValueError(f"{source} row is missing required field {key!r}")
    return value


def _positive_int(value: str | None, field: str) -> int:
    try:
        result = int(value or "")
    except ValueError as exc:
        raise ValueError(f"Image manifest field {field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"Image manifest field {field} must be a positive integer")
    return result
