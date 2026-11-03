"""Plain data structures shared by all scenario-intelligence modules.

These dataclasses deliberately do not import SQLAlchemy models so the algorithms stay
independent of the storage layer and can later consume other datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MISSING_BIN = "missing"


@dataclass(frozen=True)
class FeatureSpec:
    """Documentation and provenance for one scenario feature."""

    name: str
    description: str
    source: str
    kind: str  # count | continuous | categorical | mapping
    unit: str | None = None


@dataclass(frozen=True)
class AnnotationInput:
    """One annotated object as supplied by a data source."""

    source_id: str
    category: str
    instance_source_id: str | None
    translation: tuple[float, float, float] | None
    visibility_level: str | None
    attribute_names: tuple[str, ...] | None  # None = labels not resolved (distinct from "none")
    num_lidar_pts: int | None = None
    num_radar_pts: int | None = None


@dataclass(frozen=True)
class SensorRefInput:
    """One keyframe sensor-data reference."""

    source_id: str
    channel: str
    modality: str | None
    file_exists: bool


@dataclass(frozen=True)
class EgoPose:
    """Ego pose in the dataset's global frame; rotation is a (w, x, y, z) quaternion."""

    timestamp: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class SampleInput:
    """Everything needed to extract features for one keyframe sample."""

    sample_id: str
    scene_id: str
    scene_name: str
    timestamp: int
    scene_sample_count: int
    scene_duration_s: float | None
    location: str | None
    annotations: tuple[AnnotationInput, ...]
    sensor_refs: tuple[SensorRefInput, ...]
    ego_pose: EgoPose | None
    expected_channels: tuple[str, ...] = ()


@dataclass
class ScenarioRecord:
    """A sample-level scenario: identifiers plus a flat feature mapping.

    A feature value of ``None`` means "not available"; it is never replaced by a default.
    """

    sample_id: str
    scene_id: str
    scene_name: str
    timestamp: int
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def missing_features(self) -> list[str]:
        return sorted(name for name, value in self.features.items() if value is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "scene_name": self.scene_name,
            "timestamp": self.timestamp,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScenarioRecord:
        try:
            return cls(
                sample_id=payload["sample_id"],
                scene_id=payload["scene_id"],
                scene_name=payload["scene_name"],
                timestamp=payload["timestamp"],
                features=dict(payload["features"]),
            )
        except KeyError as exc:
            raise ValueError(f"Scenario record is missing field {exc}") from exc
