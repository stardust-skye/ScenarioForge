"""Configuration for deterministic nuScenes camera annotation conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class ProjectionConfig:
    camera_channel: str = "CAM_FRONT"
    near_plane_m: float = 0.1
    min_projected_points: int = 3
    min_box_area_px: float = 1.0
    max_box_area_fraction: float = 0.98

    def __post_init__(self) -> None:
        if not self.camera_channel:
            raise ValueError("camera_channel must not be empty")
        if not isfinite(self.near_plane_m) or self.near_plane_m <= 0:
            raise ValueError("near_plane_m must be finite and positive")
        if self.min_projected_points < 1:
            raise ValueError("min_projected_points must be at least 1")
        if not isfinite(self.min_box_area_px) or self.min_box_area_px < 0:
            raise ValueError("min_box_area_px must be finite and non-negative")
        if (
            not isfinite(self.max_box_area_fraction)
            or not 0 < self.max_box_area_fraction <= 1
        ):
            raise ValueError("max_box_area_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)
