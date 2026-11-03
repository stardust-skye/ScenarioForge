"""Convert normalized nuScenes camera frames and 3D annotations into 2D boxes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from scenarioforge.vision.category_mapping import CategoryMapping
from scenarioforge.vision.config import ProjectionConfig
from scenarioforge.vision.projection import project_annotation
from scenarioforge.vision.schemas import (
    CameraFrame,
    ConvertedAnnotation,
    ImageRecord,
    SourceAnnotation,
)


@dataclass(frozen=True)
class ConversionResult:
    dataset_version: str
    dataset_root: Path
    config: ProjectionConfig
    category_mapping: CategoryMapping
    images: tuple[ImageRecord, ...]
    accepted_annotations: tuple[ConvertedAnnotation, ...]
    rejected_annotations: tuple[ConvertedAnnotation, ...]
    report: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset_version": self.dataset_version,
            "camera_channel": self.config.camera_channel,
            "config": self.config.to_dict(),
            "category_mapping": self.category_mapping.to_dict(),
            "images": [image.to_dict() for image in self.images],
            "annotations": [annotation.to_dict() for annotation in self.accepted_annotations],
            "report": self.report,
        }


def convert_camera_samples(
    samples: Sequence[dict[str, Any]],
    *,
    dataset_root: Path,
    dataset_version: str,
    visibility_levels: dict[str, str] | None = None,
    config: ProjectionConfig | None = None,
    category_mapping: CategoryMapping | None = None,
    processing_timestamp: str | None = None,
) -> ConversionResult:
    """Convert input frames deterministically; rejected source annotations are retained by reason."""
    config = config or ProjectionConfig()
    category_mapping = category_mapping or CategoryMapping()
    dataset_root = Path(dataset_root).expanduser().resolve()
    visibility_levels = visibility_levels or {}
    ordered_samples = sorted(
        samples,
        key=lambda item: (
            str(item.get("scene_token") or ""),
            str(item.get("sample_token") or ""),
        ),
    )
    images: list[ImageRecord] = []
    accepted: list[ConvertedAnnotation] = []
    rejected: list[ConvertedAnnotation] = []
    image_annotation_counts: Counter[str] = Counter()
    image_rejection_counts: Counter[str] = Counter()
    image_status_counts: Counter[str] = Counter()

    for frame_data in ordered_samples:
        frame = _camera_frame(frame_data, visibility_levels)
        relative_path, image_status, actual_size = _validate_image(frame, dataset_root)
        width, height = actual_size or (frame.width, frame.height)
        if width is None or height is None or width <= 0 or height <= 0:
            image_status = "invalid_image_dimensions"
            width = height = None
        image_id = frame.sample_token
        for source in frame.annotations:
            converted, reason = _convert_one(
                source,
                frame,
                image_id,
                relative_path,
                width,
                height,
                image_status,
                category_mapping,
                config,
            )
            if reason is None:
                accepted.append(converted)
                image_annotation_counts[image_id] += 1
            else:
                rejected.append(converted)
                image_rejection_counts[image_id] += 1
        images.append(
            ImageRecord(
                image_id=image_id,
                sample_token=frame.sample_token,
                scene_token=frame.scene_token,
                scene_name=frame.scene_name,
                sample_data_token=frame.sample_data_token,
                image_path=relative_path,
                width=width,
                height=height,
                image_status=image_status,
                annotation_count=image_annotation_counts[image_id],
                rejected_annotation_count=image_rejection_counts[image_id],
            )
        )
        image_status_counts[image_status] += 1

    images.sort(key=lambda image: (image.scene_token, image.sample_token))
    accepted.sort(key=lambda item: (item.scene_token, item.sample_token, item.source_annotation_token))
    rejected.sort(key=lambda item: (item.scene_token, item.sample_token, item.source_annotation_token))
    report = _build_report(
        ordered_samples,
        images,
        accepted,
        rejected,
        dataset_version,
        config,
        category_mapping,
        processing_timestamp or datetime.now(timezone.utc).isoformat(),
    )
    return ConversionResult(
        dataset_version=dataset_version,
        dataset_root=dataset_root,
        config=config,
        category_mapping=category_mapping,
        images=tuple(images),
        accepted_annotations=tuple(accepted),
        rejected_annotations=tuple(rejected),
        report=report,
    )


def _camera_frame(
    payload: dict[str, Any], visibility_levels: dict[str, str]
) -> CameraFrame:
    return CameraFrame(
        sample_token=str(payload["sample_token"]),
        scene_token=str(payload["scene_token"]),
        scene_name=payload.get("scene_name"),
        sample_data_token=payload.get("sample_data_token"),
        image_path=Path(payload["image_path"]) if payload.get("image_path") is not None else None,
        width=payload.get("width"),
        height=payload.get("height"),
        calibration=payload.get("calibration"),
        ego_pose=payload.get("ego_pose"),
        camera_error=payload.get("camera_error"),
        annotations=tuple(
            item if isinstance(item, SourceAnnotation)
            else SourceAnnotation.from_dict(item, visibility_levels)
            for item in payload.get("annotations", [])
        ),
    )


def _validate_image(
    frame: CameraFrame, dataset_root: Path
) -> tuple[str | None, str, tuple[int, int] | None]:
    if frame.sample_data_token is None:
        return None, "missing_camera_sample_data", None
    if frame.image_path is None:
        return None, "missing_image", None
    try:
        relative = frame.image_path.expanduser().resolve().relative_to(dataset_root)
    except ValueError:
        return None, "image_outside_dataset_root", None
    if not frame.image_path.is_file():
        return relative.as_posix(), "missing_image", None
    if (
        not isinstance(frame.width, int)
        or not isinstance(frame.height, int)
        or frame.width <= 0
        or frame.height <= 0
    ):
        return relative.as_posix(), "invalid_image_dimensions", None
    try:
        with Image.open(frame.image_path) as image:
            image.load()
            actual_size = image.size
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return relative.as_posix(), "image_decode_error", None
    if actual_size != (frame.width, frame.height):
        return relative.as_posix(), "image_dimension_mismatch", actual_size
    if frame.camera_error and frame.camera_error.startswith("missing_"):
        return relative.as_posix(), frame.camera_error.split(":", 1)[0], actual_size
    if frame.camera_error:
        return relative.as_posix(), "camera_data_error", actual_size
    if frame.calibration is None or not {
        "translation", "rotation", "camera_intrinsic"
    } <= frame.calibration.keys():
        return relative.as_posix(), "missing_calibration", actual_size
    if frame.ego_pose is None or not {"translation", "rotation"} <= frame.ego_pose.keys():
        return relative.as_posix(), "missing_ego_pose", actual_size
    return relative.as_posix(), "ok", actual_size


def _convert_one(
    source: SourceAnnotation,
    frame: CameraFrame,
    image_id: str,
    relative_path: str | None,
    width: int | None,
    height: int | None,
    image_status: str,
    mapping: CategoryMapping,
    config: ProjectionConfig,
) -> tuple[ConvertedAnnotation, str | None]:
    detection_class = mapping.by_source.get(source.category or "")
    reason: str | None = None
    projection = None
    if source.source_error:
        reason = "invalid_source_annotation"
    elif source.category is None:
        reason = "missing_category"
    elif detection_class is None:
        reason = "unsupported_category"
    elif frame.sample_data_token is None:
        reason = "missing_camera_sample_data"
    elif frame.camera_error:
        reason = (
            frame.camera_error.split(":", 1)[0]
            if frame.camera_error.startswith("missing_")
            else "camera_data_error"
        )
    elif frame.calibration is None:
        reason = "missing_calibration"
    elif frame.ego_pose is None:
        reason = "missing_ego_pose"
    elif image_status != "ok":
        reason = image_status
    else:
        projection, reason = project_annotation(
            source,
            frame.ego_pose,
            frame.calibration,
            width or 0,
            height or 0,
            config,
        )
    if reason is not None:
        projected = (None, None, None, None, False, 0)
    else:
        assert projection is not None and detection_class is not None
        projected = (
            projection.x_min,
            projection.y_min,
            projection.x_max,
            projection.y_max,
            projection.is_clipped,
            projection.projected_point_count,
        )
    annotation = ConvertedAnnotation(
        annotation_id=source.source_annotation_token,
        source_annotation_token=source.source_annotation_token,
        sample_token=frame.sample_token,
        scene_token=frame.scene_token,
        camera_sample_data_token=frame.sample_data_token,
        image_id=image_id,
        image_path=relative_path,
        image_width=width,
        image_height=height,
        original_category=source.category,
        detection_category=detection_class.name if detection_class else None,
        class_id=detection_class.class_id if detection_class else None,
        x_min=projected[0],
        y_min=projected[1],
        x_max=projected[2],
        y_max=projected[3],
        visibility_token=source.visibility_token,
        visibility_level=source.visibility_level,
        center_3d=source.translation,
        dimensions_3d=source.size,
        instance_token=source.instance_token,
        conversion_status="accepted" if reason is None else "rejected",
        rejection_reason=reason,
        is_clipped=projected[4],
        projected_point_count=projected[5],
    )
    return annotation, reason


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": fmean(values),
        "max": max(values),
    }


def _build_report(
    source_samples: Sequence[dict[str, Any]],
    images: Sequence[ImageRecord],
    accepted: Sequence[ConvertedAnnotation],
    rejected: Sequence[ConvertedAnnotation],
    dataset_version: str,
    config: ProjectionConfig,
    mapping: CategoryMapping,
    processing_timestamp: str,
) -> dict[str, Any]:
    rejection_counts = Counter(item.rejection_reason for item in rejected)
    rejection_by_category: dict[str, Counter[str | None]] = defaultdict(Counter)
    for item in rejected:
        rejection_by_category[item.rejection_reason or "unknown"][item.detection_category] += 1
    mapped_distribution = Counter(item.detection_category for item in accepted)
    visibility_counts = Counter(item.visibility_level or "unknown" for item in (*accepted, *rejected))
    scene_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"images": 0, "accepted_annotations": 0, "rejected_annotations": 0}
    )
    for image in images:
        scene_summary[image.scene_token]["images"] += 1
    for item in accepted:
        scene_summary[item.scene_token]["accepted_annotations"] += 1
    for item in rejected:
        scene_summary[item.scene_token]["rejected_annotations"] += 1
    widths = [item.x_max - item.x_min for item in accepted if item.x_min is not None and item.x_max is not None]
    heights = [item.y_max - item.y_min for item in accepted if item.y_min is not None and item.y_max is not None]
    areas = [width * height for width, height in zip(widths, heights)]
    source_ids = [item.source_annotation_token for item in (*accepted, *rejected)]
    duplicate_ids = len(source_ids) - len(set(source_ids))
    image_without_annotations = sum(image.annotation_count == 0 for image in images)
    out_of_bounds = sum(
        item.x_min is None
        or item.y_min is None
        or item.x_max is None
        or item.y_max is None
        or item.image_width is None
        or item.image_height is None
        or item.x_min < 0
        or item.y_min < 0
        or item.x_max > item.image_width
        or item.y_max > item.image_height
        for item in accepted
    )
    zero_area = sum(
        item.x_min is None
        or item.y_min is None
        or item.x_max is None
        or item.y_max is None
        or item.x_max <= item.x_min
        or item.y_max <= item.y_min
        for item in accepted
    )
    expected_class_ids = {item.class_id for item in mapping.classes}
    unexpected_class_ids = sum(item.class_id not in expected_class_ids for item in accepted)
    return {
        "dataset_version": dataset_version,
        "camera_channel": config.camera_channel,
        "processing_timestamp_utc": processing_timestamp,
        "config": config.to_dict(),
        "category_mapping_version": mapping.version,
        "total_samples_processed": len(source_samples),
        "total_camera_images_processed": len(images),
        "total_source_annotations_considered": len(accepted) + len(rejected),
        "successfully_projected_annotations": len(accepted),
        "rejected_annotations": len(rejected),
        "rejected_annotations_by_reason": dict(sorted(rejection_counts.items())),
        "rejected_annotations_by_reason_and_category": {
            reason: {
                category or "unmapped": count
                for category, count in sorted(
                    categories.items(), key=lambda pair: pair[0] or ""
                )
            }
            for reason, categories in sorted(rejection_by_category.items())
        },
        "image_status_counts": dict(sorted(Counter(image.image_status for image in images).items())),
        "missing_file_count": sum(image.image_status == "missing_image" for image in images),
        "missing_calibration_count": sum(image.image_status == "missing_calibration" for image in images),
        "missing_ego_pose_count": sum(image.image_status == "missing_ego_pose" for image in images),
        "missing_camera_sample_data_count": sum(
            image.image_status == "missing_camera_sample_data" for image in images
        ),
        "category_distribution": dict(sorted(mapped_distribution.items())),
        "visibility_distribution": dict(sorted(visibility_counts.items())),
        "bounding_box_distributions": {
            "width_px": _distribution(widths),
            "height_px": _distribution(heights),
            "area_px": _distribution(areas),
        },
        "per_scene_annotation_counts": {
            scene: counts for scene, counts in sorted(scene_summary.items())
        },
        "sanity_checks": {
            "accepted_boxes_out_of_image_bounds": out_of_bounds,
            "accepted_zero_or_negative_area_boxes": zero_area,
            "accepted_annotations_with_unexpected_class_ids": unexpected_class_ids,
            "duplicate_source_annotation_ids": duplicate_ids,
            "images_without_valid_annotations": image_without_annotations,
            "abnormally_large_boxes_rejected": rejection_counts["box_too_large"],
        },
        "deterministic_payload_note": (
            "Processing timestamp varies per run; image, annotation, and configuration records "
            "are deterministically ordered for identical source inputs."
        ),
    }
