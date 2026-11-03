"""Write portable, deterministic annotation and image manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scenarioforge.vision.converter import ConversionResult

ANNOTATION_COLUMNS = (
    "annotation_id",
    "source_annotation_token",
    "sample_token",
    "scene_token",
    "camera_sample_data_token",
    "image_id",
    "image_path",
    "image_width",
    "image_height",
    "original_category",
    "detection_category",
    "class_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "visibility_token",
    "visibility_level",
    "center_3d",
    "dimensions_3d",
    "instance_token",
    "conversion_status",
    "rejection_reason",
    "is_clipped",
    "projected_point_count",
)

IMAGE_COLUMNS = (
    "image_id",
    "sample_token",
    "scene_token",
    "scene_name",
    "sample_data_token",
    "image_path",
    "width",
    "height",
    "image_status",
    "annotation_count",
    "rejected_annotation_count",
)


def write_conversion(result: ConversionResult, output_dir: Path) -> dict[str, Path]:
    """Write all products beneath output_dir without copying source images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "annotations_json": output_dir / "annotations.json",
        "annotations_csv": output_dir / "annotations.csv",
        "conversion_report": output_dir / "conversion_report.json",
        "category_mapping": output_dir / "category_mapping.json",
        "image_manifest": output_dir / "image_manifest.csv",
        "rejected_annotations": output_dir / "rejected_annotations.csv",
        "dataset_manifest": output_dir / "dataset_manifest.json",
    }
    manifest = result.to_manifest()
    files["annotations_json"].write_text(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "dataset_version": result.dataset_version,
                "camera_channel": result.config.camera_channel,
                "images": manifest["images"],
                "annotations": manifest["annotations"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files["conversion_report"].write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files["category_mapping"].write_text(
        json.dumps(result.category_mapping.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files["dataset_manifest"].write_text(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "dataset_version": result.dataset_version,
                "camera_channel": result.config.camera_channel,
                "config": result.config.to_dict(),
                "image_count": len(result.images),
                "annotation_count": len(result.accepted_annotations),
                "rejected_annotation_count": len(result.rejected_annotations),
                "image_manifest": files["image_manifest"].name,
                "annotations": files["annotations_json"].name,
                "category_mapping": files["category_mapping"].name,
                "conversion_report": files["conversion_report"].name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        files["annotations_csv"],
        ANNOTATION_COLUMNS,
        [item.to_dict() for item in result.accepted_annotations],
    )
    _write_csv(
        files["rejected_annotations"],
        ANNOTATION_COLUMNS,
        [item.to_dict() for item in result.rejected_annotations],
    )
    _write_csv(
        files["image_manifest"],
        IMAGE_COLUMNS,
        [item.to_dict() for item in result.images],
    )
    return files


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":"), sort_keys=True)
                    if isinstance(value, (tuple, list, dict))
                    else value
                    for key, value in row.items()
                }
            )
