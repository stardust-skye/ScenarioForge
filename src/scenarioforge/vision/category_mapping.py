"""Stable mapping from nuScenes leaf categories to 2D detection classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectionClass:
    class_id: int
    name: str
    source_categories: tuple[str, ...]


DEFAULT_CLASSES = (
    DetectionClass(1, "car", ("vehicle.car",)),
    DetectionClass(2, "truck", ("vehicle.truck",)),
    DetectionClass(3, "bus", ("vehicle.bus.bendy", "vehicle.bus.rigid")),
    DetectionClass(4, "trailer", ("vehicle.trailer",)),
    DetectionClass(5, "construction_vehicle", ("vehicle.construction",)),
    DetectionClass(
        6,
        "pedestrian",
        (
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.police_officer",
        ),
    ),
    DetectionClass(7, "motorcycle", ("vehicle.motorcycle",)),
    DetectionClass(8, "bicycle", ("vehicle.bicycle",)),
    DetectionClass(9, "traffic_cone", ("movable_object.trafficcone",)),
    DetectionClass(10, "barrier", ("movable_object.barrier",)),
)


@dataclass(frozen=True)
class CategoryMapping:
    classes: tuple[DetectionClass, ...] = DEFAULT_CLASSES
    version: str = "nuscenes-2d-v1"

    def __post_init__(self) -> None:
        ids = [item.class_id for item in self.classes]
        names = [item.name for item in self.classes]
        sources = [source for item in self.classes for source in item.source_categories]
        if not self.version:
            raise ValueError("Category mapping version must not be empty")
        if any(class_id < 1 for class_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Detection class IDs must be unique positive integers")
        if len(set(names)) != len(names) or len(set(sources)) != len(sources):
            raise ValueError("Detection class names and source categories must be unique")
        if any(not item.name or not item.source_categories for item in self.classes):
            raise ValueError("Every detection class needs a name and source categories")

    @property
    def by_source(self) -> dict[str, DetectionClass]:
        return {
            source: item
            for item in self.classes
            for source in item.source_categories
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "background_class_id": 0,
            "classes": [
                {
                    "class_id": item.class_id,
                    "name": item.name,
                    "source_categories": list(item.source_categories),
                }
                for item in self.classes
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CategoryMapping:
        try:
            classes = tuple(
                DetectionClass(
                    class_id=int(item["class_id"]),
                    name=str(item["name"]),
                    source_categories=tuple(item["source_categories"]),
                )
                for item in payload["classes"]
            )
            return cls(classes=classes, version=str(payload["version"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid category mapping: {exc}") from exc


def class_rows(mapping: CategoryMapping) -> list[dict[str, Any]]:
    return [asdict(item) for item in mapping.classes]
