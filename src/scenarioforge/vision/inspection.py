"""Render saved annotation manifests over real camera images for manual review."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

INSPECTION_MODES = ("random", "many", "partial", "empty", "rejected")


def render_inspection(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    mode: str = "random",
    sample_token: str | None = None,
    count: int = 4,
    seed: int = 42,
) -> list[Path]:
    """Select manifest images deterministically and save category-colored inspection overlays."""
    if mode not in INSPECTION_MODES:
        raise ValueError(f"Unknown inspection mode {mode!r}; choose from {INSPECTION_MODES}")
    if count < 1:
        raise ValueError("count must be at least one")
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotation_rows = payload.get("annotations", [])
    image_rows = payload.get("images", [])
    rejected_path = manifest_path.parent / "rejected_annotations.csv"
    rejected_rows: list[dict[str, str]] = []
    if rejected_path.is_file():
        with rejected_path.open(newline="", encoding="utf-8") as handle:
            rejected_rows = list(csv.DictReader(handle))
    accepted_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotation_rows:
        accepted_by_image[row["image_id"]].append(row)
    for row in rejected_rows:
        rejected_by_image[row["image_id"]].append(row)

    candidates = [
        row for row in image_rows
        if row.get("image_status") == "ok" and row.get("image_path")
    ]
    if sample_token is not None:
        candidates = [row for row in candidates if row["sample_token"] == sample_token]
        if not candidates:
            raise ValueError(f"No valid image found for sample token {sample_token}")
    elif mode == "many":
        candidates = sorted(
            candidates,
            key=lambda row: (-len(accepted_by_image[row["image_id"]]), row["sample_token"]),
        )
        candidates = [row for row in candidates if accepted_by_image[row["image_id"]]]
    elif mode == "partial":
        candidates = [
            row for row in candidates
            if any(annotation.get("is_clipped") for annotation in accepted_by_image[row["image_id"]])
        ]
    elif mode == "empty":
        candidates = [row for row in candidates if not accepted_by_image[row["image_id"]]]
    elif mode == "rejected":
        candidates = [
            row for row in candidates if rejected_by_image[row["image_id"]]
        ]
    if mode != "many":
        candidates.sort(key=lambda row: row["sample_token"])
    if sample_token is None:
        random.Random(seed).shuffle(candidates)
    selected = candidates[:count]
    if not selected:
        raise ValueError(f"No images available for inspection mode {mode!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, row in enumerate(selected, start=1):
        image_path = (dataset_root / row["image_path"]).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"Manifest image path escapes dataset root: {row['image_path']}") from exc
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        figure, axis = plt.subplots(figsize=(12, 7))
        axis.imshow(image)
        for annotation in accepted_by_image[row["image_id"]]:
            color = _category_color(str(annotation["detection_category"]))
            x_min, y_min = float(annotation["x_min"]), float(annotation["y_min"])
            width = float(annotation["x_max"]) - x_min
            height = float(annotation["y_max"]) - y_min
            axis.add_patch(Rectangle((x_min, y_min), width, height, fill=False, edgecolor=color, linewidth=2))
            label = (
                f"{annotation['detection_category']} | "
                f"{str(annotation['source_annotation_token'])[:8]}"
            )
            if annotation.get("visibility_level"):
                label += f" | {annotation['visibility_level']}"
            axis.text(
                x_min,
                max(0, y_min - 3),
                label,
                color="white",
                fontsize=8,
                bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
            )
        reasons = Counter(row["rejection_reason"] for row in rejected_by_image[row["image_id"]])
        axis.set_title(
            f"sample {row['sample_token']} | scene {row['scene_token']} | "
            f"accepted {len(accepted_by_image[row['image_id']])} | rejected {sum(reasons.values())}"
        )
        axis.axis("off")
        if reasons:
            axis.text(
                0.01,
                0.01,
                "Rejected: " + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())),
                transform=axis.transAxes,
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.7, "pad": 4},
            )
        path = output_dir / f"{index:02d}_{row['sample_token']}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=120)
        plt.close(figure)
        saved.append(path)
    return saved


def _category_color(category: str) -> tuple[float, float, float]:
    palette = {
        "car": "#00d4ff",
        "truck": "#ff9f1c",
        "bus": "#f15bb5",
        "trailer": "#fee440",
        "construction_vehicle": "#9b5de5",
        "pedestrian": "#00f5d4",
        "motorcycle": "#f15b2a",
        "bicycle": "#70e000",
        "traffic_cone": "#ff006e",
        "barrier": "#8d99ae",
    }
    color = palette.get(category, "#ffffff")
    return tuple(int(color[index:index + 2], 16) / 255 for index in (1, 3, 5))
