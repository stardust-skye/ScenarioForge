"""Turn normalized sample inputs into :class:`ScenarioRecord` objects."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from scenarioforge.scenario_intelligence.schemas import (
    EgoPose,
    SampleInput,
    ScenarioRecord,
)
from scenarioforge.scenario_intelligence.scenario_features import (
    composition_features,
    motion_features,
    sensor_features,
    visibility_attribute_features,
)

logger = logging.getLogger(__name__)


def extract_scenarios(
    samples: Iterable[SampleInput], near_ego_radius_m: float = 50.0
) -> list[ScenarioRecord]:
    """Extract one scenario per keyframe sample, in deterministic (scene, timestamp) order.

    The previous keyframe's pose within the same scene is used for displacement/speed; the first
    keyframe of a scene therefore has no motion features. Duplicate sample ids are an error
    rather than being silently merged.
    """
    ordered = sorted(samples, key=lambda s: (s.scene_id, s.timestamp, s.sample_id))
    seen: set[str] = set()
    last_pose_by_scene: dict[str, EgoPose | None] = defaultdict(lambda: None)
    records: list[ScenarioRecord] = []
    for sample in ordered:
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate sample id in extraction input: {sample.sample_id}")
        seen.add(sample.sample_id)
        features: dict[str, object] = {
            "scene_sample_count": sample.scene_sample_count,
            "scene_duration_s": sample.scene_duration_s,
            "location": sample.location,
        }
        features.update(composition_features(sample))
        features.update(visibility_attribute_features(sample))
        features.update(motion_features(sample, last_pose_by_scene[sample.scene_id], near_ego_radius_m))
        features.update(sensor_features(sample))
        last_pose_by_scene[sample.scene_id] = sample.ego_pose
        records.append(
            ScenarioRecord(
                sample_id=sample.sample_id,
                scene_id=sample.scene_id,
                scene_name=sample.scene_name,
                timestamp=sample.timestamp,
                features=features,
            )
        )
    logger.info("Extracted %d scenarios from %d scenes", len(records), len({s.scene_id for s in ordered}))
    return records
