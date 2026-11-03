"""Feature engineering for sample-level scenarios.

Every feature is declared in ``FEATURE_SPECS`` with its meaning and provenance. A feature that
cannot be computed from available data is ``None``; nothing is imputed.

Category grouping rules (nuScenes category taxonomy, first matching rule wins):

  =============================  ==================
  nuScenes category              group
  =============================  ==================
  human.pedestrian.*             pedestrian
  animal                         animal
  vehicle.bicycle / motorcycle   two_wheeler
  vehicle.* (all others)         vehicle
  movable_object.*               traffic_related
  static_object.*                static_object
  anything else                  other
  =============================  ==================

``two_wheeler`` annotations are the bicycle/motorcycle object itself; a rider is annotated as a
separate pedestrian-class object. ``traffic_related`` covers barriers, traffic cones, debris and
pushable/pullable objects; it is a grouping convenience, not a statement about traffic law.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from scenarioforge.scenario_intelligence.metrics import normalized_entropy
from scenarioforge.scenario_intelligence.schemas import (
    EgoPose,
    FeatureSpec,
    SampleInput,
)

CATEGORY_GROUPS = (
    "pedestrian",
    "vehicle",
    "two_wheeler",
    "traffic_related",
    "static_object",
    "animal",
    "other",
)

_TWO_WHEELERS = ("vehicle.bicycle", "vehicle.motorcycle")
_SENSOR_MODALITIES = ("camera", "lidar", "radar")

FEATURE_SPECS: dict[str, FeatureSpec] = {
    spec.name: spec
    for spec in (
        FeatureSpec("scene_sample_count", "Keyframe samples in the sample's scene.", "scenes/samples tables", "count"),
        FeatureSpec("scene_duration_s", "Last minus first keyframe timestamp in the scene.", "samples.timestamp (microseconds)", "continuous", "s"),
        FeatureSpec("location", "Log location of the scene.", "nuScenes log.json via scenes.log_source_id", "categorical"),
        FeatureSpec("n_objects", "Annotated objects in this keyframe.", "annotations table", "count"),
        FeatureSpec("n_pedestrians", "Annotated objects in group 'pedestrian'.", "annotations.category", "count"),
        FeatureSpec("n_vehicles", "Annotated objects in group 'vehicle' (excludes two-wheelers).", "annotations.category", "count"),
        FeatureSpec("n_two_wheelers", "Annotated bicycles and motorcycles.", "annotations.category", "count"),
        FeatureSpec("n_traffic_related", "Annotated movable objects (barrier, cone, debris, pushable).", "annotations.category", "count"),
        FeatureSpec("n_static_objects", "Annotated static objects (bicycle rack).", "annotations.category", "count"),
        FeatureSpec("category_counts", "Annotated object count per nuScenes category.", "annotations.category", "mapping"),
        FeatureSpec("group_counts", "Annotated object count per category group.", "derived from category_counts", "mapping"),
        FeatureSpec("category_combination", "Sorted '|'-joined set of categories present.", "annotations.category", "categorical"),
        FeatureSpec("group_combination", "Sorted '|'-joined set of category groups present.", "derived", "categorical"),
        FeatureSpec("n_distinct_categories", "Distinct categories present.", "annotations.category", "count"),
        FeatureSpec("n_unique_instances", "Distinct object instances annotated in this keyframe.", "annotations.instance_source_id", "count"),
        FeatureSpec("visibility_counts", "Annotated objects per nuScenes visibility bin.", "annotations.attributes.visibility_token + visibility.json", "mapping"),
        FeatureSpec("fraction_low_visibility", "Share of objects in the lowest visibility bin (v0-40).", "visibility_counts", "continuous"),
        FeatureSpec("attribute_counts", "Annotated objects per nuScenes attribute name.", "annotations.attributes.attribute_tokens + attribute.json", "mapping"),
        FeatureSpec("n_objects_near_ego", "Annotated objects whose xy centre is within the configured radius of the ego xy.", "annotation translation + ego_pose", "count"),
        FeatureSpec("ego_x", "Ego x in the global map frame.", "ego_pose via LIDAR_TOP sample_data", "continuous", "m"),
        FeatureSpec("ego_y", "Ego y in the global map frame.", "ego_pose via LIDAR_TOP sample_data", "continuous", "m"),
        FeatureSpec("ego_yaw_deg", "Ego heading (yaw) from the pose quaternion.", "ego_pose rotation", "continuous", "deg"),
        FeatureSpec("ego_displacement_m", "Planar distance from the previous keyframe's ego pose.", "ego_pose", "continuous", "m"),
        FeatureSpec("ego_speed_mps", "Planar displacement divided by time since previous keyframe (coarse keyframe-to-keyframe estimate).", "ego_pose + timestamps", "continuous", "m/s"),
        FeatureSpec("n_camera_refs", "Keyframe camera references whose file exists.", "sensor_data/sensors", "count"),
        FeatureSpec("n_lidar_refs", "Keyframe LiDAR references whose file exists.", "sensor_data/sensors", "count"),
        FeatureSpec("n_radar_refs", "Keyframe radar references whose file exists.", "sensor_data/sensors", "count"),
        FeatureSpec("n_missing_sensor_files", "Sensor references whose file is missing.", "sensor_data.file_exists", "count"),
        FeatureSpec("sensor_completeness", "Distinct available channels / channels seen anywhere in the dataset. Availability only, not quality.", "sensor_data", "continuous"),
    )
}


def category_group(category: str) -> str:
    """Map a nuScenes category name to a documented group."""
    if category.startswith("human.pedestrian."):
        return "pedestrian"
    if category == "animal":
        return "animal"
    if category in _TWO_WHEELERS:
        return "two_wheeler"
    if category.startswith("vehicle."):
        return "vehicle"
    if category.startswith("movable_object."):
        return "traffic_related"
    if category.startswith("static_object."):
        return "static_object"
    return "other"


def yaw_degrees(rotation: Sequence[float]) -> float:
    """Yaw (rotation about z) from a (w, x, y, z) quaternion, in degrees."""
    w, x, y, z = rotation
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def composition_features(sample: SampleInput) -> dict[str, object]:
    """Object-composition and complexity features."""
    annotations = sample.annotations
    category_counts = Counter(a.category for a in annotations)
    group_counts = Counter(category_group(a.category) for a in annotations)
    instances = {a.instance_source_id for a in annotations if a.instance_source_id}
    has_instances = all(a.instance_source_id for a in annotations)
    return {
        "n_objects": len(annotations),
        "n_pedestrians": group_counts["pedestrian"],
        "n_vehicles": group_counts["vehicle"],
        "n_two_wheelers": group_counts["two_wheeler"],
        "n_traffic_related": group_counts["traffic_related"],
        "n_static_objects": group_counts["static_object"],
        "category_counts": dict(sorted(category_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "category_combination": "|".join(sorted(category_counts)),
        "group_combination": "|".join(sorted(group_counts)),
        "n_distinct_categories": len(category_counts),
        "n_unique_instances": len(instances) if has_instances else None,
    }


def visibility_attribute_features(sample: SampleInput) -> dict[str, object]:
    """Visibility and attribute distributions; None when labels were not resolved."""
    annotations = sample.annotations
    levels = [a.visibility_level for a in annotations]
    visibility = None
    low_fraction = None
    if annotations and all(level is not None for level in levels):
        visibility = dict(sorted(Counter(levels).items()))
        low_fraction = sum(1 for level in levels if level == "v0-40") / len(levels)
    attributes = None
    if annotations and all(a.attribute_names is not None for a in annotations):
        attributes = dict(
            sorted(Counter(name for a in annotations for name in a.attribute_names).items())
        )
    return {
        "visibility_counts": visibility,
        "fraction_low_visibility": low_fraction,
        "attribute_counts": attributes,
    }


def motion_features(
    sample: SampleInput, previous_pose: EgoPose | None, near_ego_radius_m: float
) -> dict[str, object]:
    """Ego pose, displacement, and speed. Speed needs a valid previous pose and dt > 0."""
    pose = sample.ego_pose
    features: dict[str, object] = {
        "ego_x": None,
        "ego_y": None,
        "ego_yaw_deg": None,
        "ego_displacement_m": None,
        "ego_speed_mps": None,
        "n_objects_near_ego": None,
    }
    if pose is None:
        return features
    features["ego_x"], features["ego_y"] = pose.translation[0], pose.translation[1]
    features["ego_yaw_deg"] = yaw_degrees(pose.rotation)
    located = [a.translation for a in sample.annotations]
    if all(t is not None for t in located):
        features["n_objects_near_ego"] = sum(
            1
            for t in located
            if math.hypot(t[0] - pose.translation[0], t[1] - pose.translation[1])
            <= near_ego_radius_m
        )
    if previous_pose is not None:
        displacement = math.hypot(
            pose.translation[0] - previous_pose.translation[0],
            pose.translation[1] - previous_pose.translation[1],
        )
        elapsed_s = (pose.timestamp - previous_pose.timestamp) / 1_000_000
        features["ego_displacement_m"] = displacement
        if elapsed_s > 0:
            features["ego_speed_mps"] = displacement / elapsed_s
    return features


def sensor_features(sample: SampleInput) -> dict[str, object]:
    """Sensor availability counts. Availability is not a statement about data quality."""
    refs = sample.sensor_refs
    available = [r for r in refs if r.file_exists]
    by_modality = Counter(r.modality for r in available)
    expected = set(sample.expected_channels)
    completeness = (
        len({r.channel for r in available} & expected) / len(expected) if expected else None
    )
    return {
        "n_camera_refs": by_modality["camera"],
        "n_lidar_refs": by_modality["lidar"],
        "n_radar_refs": by_modality["radar"],
        "n_missing_sensor_files": sum(1 for r in refs if not r.file_exists),
        "sensor_completeness": completeness,
    }


def within_sample_entropy(category_counts: dict[str, int]) -> float:
    """Normalised entropy of the category mix inside one sample (0 for <=1 category)."""
    return normalized_entropy(category_counts)
