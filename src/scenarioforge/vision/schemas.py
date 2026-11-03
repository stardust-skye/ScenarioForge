"""Validated, model-independent records for camera images and projected annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceAnnotation:
    source_annotation_token: str
    category: str | None
    translation: tuple[float, float, float] | None
    size: tuple[float, float, float] | None
    rotation: tuple[float, float, float, float] | None
    instance_token: str | None = None
    visibility_token: str | None = None
    visibility_level: str | None = None
    source_error: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], visibility: dict[str, str]) -> SourceAnnotation:
        def vector(name: str, length: int) -> tuple[float, ...] | None:
            value = payload.get(name)
            if value is None:
                return None
            try:
                if len(value) != length:
                    return None
                return tuple(float(item) for item in value)
            except (TypeError, ValueError, OverflowError):
                return None

        visibility_token = payload.get("visibility_token")
        return cls(
            source_annotation_token=str(payload["source_annotation_token"]),
            category=payload.get("category"),
            translation=vector("translation", 3),  # type: ignore[arg-type]
            size=vector("size", 3),  # type: ignore[arg-type]
            rotation=vector("rotation", 4),  # type: ignore[arg-type]
            instance_token=payload.get("instance_token"),
            visibility_token=visibility_token,
            visibility_level=visibility.get(visibility_token) if visibility_token else None,
            source_error=payload.get("source_error"),
        )


@dataclass(frozen=True)
class CameraFrame:
    sample_token: str
    scene_token: str
    scene_name: str | None
    sample_data_token: str | None
    image_path: Path | None
    width: int | None
    height: int | None
    calibration: dict[str, Any] | None
    ego_pose: dict[str, Any] | None
    camera_error: str | None
    annotations: tuple[SourceAnnotation, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    sample_token: str
    scene_token: str
    scene_name: str | None
    sample_data_token: str | None
    image_path: str | None
    width: int | None
    height: int | None
    image_status: str
    annotation_count: int
    rejected_annotation_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvertedAnnotation:
    annotation_id: str
    source_annotation_token: str
    sample_token: str
    scene_token: str
    camera_sample_data_token: str | None
    image_id: str
    image_path: str | None
    image_width: int | None
    image_height: int | None
    original_category: str | None
    detection_category: str | None
    class_id: int | None
    x_min: float | None
    y_min: float | None
    x_max: float | None
    y_max: float | None
    visibility_token: str | None
    visibility_level: str | None
    center_3d: tuple[float, float, float] | None
    dimensions_3d: tuple[float, float, float] | None
    instance_token: str | None
    conversion_status: str
    rejection_reason: str | None
    is_clipped: bool
    projected_point_count: int

    def __post_init__(self) -> None:
        if self.conversion_status not in {"accepted", "rejected"}:
            raise ValueError("conversion_status must be accepted or rejected")
        if self.conversion_status == "accepted":
            if self.rejection_reason is not None or self.class_id is None:
                raise ValueError("Accepted annotations require a class and no rejection reason")
            coords = (self.x_min, self.y_min, self.x_max, self.y_max)
            if any(value is None or not isfinite(value) for value in coords):
                raise ValueError("Accepted annotations require finite box coordinates")
            if self.x_max <= self.x_min or self.y_max <= self.y_min:
                raise ValueError("Accepted annotations require positive-area boxes")
        elif not self.rejection_reason:
            raise ValueError("Rejected annotations require a reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
