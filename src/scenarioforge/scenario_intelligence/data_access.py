"""Data access: build :class:`SampleInput` objects from the Part 1 SQLite/PostgreSQL database.

This is the only scenario-intelligence module that imports SQLAlchemy models. Ego pose,
visibility labels, attribute names, and log locations are not stored in the Part 1 database, so
they are resolved from optional :class:`NuScenesLookups`; without them those features are None.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from scenarioforge.database.models import (
    Annotation,
    Dataset,
    Sample,
    Scene,
    Sensor,
    SensorData,
)
from scenarioforge.datasets.nuscenes_lookups import NuScenesLookups
from scenarioforge.scenario_intelligence.schemas import (
    AnnotationInput,
    EgoPose,
    SampleInput,
    SensorRefInput,
)

logger = logging.getLogger(__name__)


class ScenarioDataSource(Protocol):
    """Anything that can yield normalized samples; other datasets implement this."""

    def iter_samples(self) -> Iterator[SampleInput]: ...


class SqlScenarioDataSource:
    """Read samples, annotations, and sensor references for one ingested dataset."""

    def __init__(
        self,
        session: Session,
        dataset_source_id: str,
        lookups: NuScenesLookups | None = None,
        ego_pose_channel: str = "LIDAR_TOP",
    ) -> None:
        self.session = session
        self.dataset_source_id = dataset_source_id
        self.lookups = lookups
        self.ego_pose_channel = ego_pose_channel

    def iter_samples(self) -> Iterator[SampleInput]:
        dataset = self.session.scalar(
            select(Dataset).where(Dataset.source_id == self.dataset_source_id)
        )
        if dataset is None:
            raise ValueError(f"Dataset has not been ingested: {self.dataset_source_id}")

        scenes = {s.id: s for s in self.session.scalars(select(Scene).where(Scene.dataset_id == dataset.id))}
        samples = list(
            self.session.scalars(
                select(Sample).where(Sample.dataset_id == dataset.id).order_by(Sample.scene_id, Sample.timestamp, Sample.source_id)
            )
        )
        annotations_by_sample: dict[int, list[AnnotationInput]] = defaultdict(list)
        for annotation in self.session.scalars(
            select(Annotation).where(Annotation.dataset_id == dataset.id).order_by(Annotation.source_id)
        ):
            annotations_by_sample[annotation.sample_id].append(self._annotation_input(annotation))

        refs_by_sample: dict[int, list[SensorRefInput]] = defaultdict(list)
        pose_token_by_sample: dict[int, str] = {}
        rows = self.session.execute(
            select(SensorData, Sensor)
            .join(Sensor, SensorData.sensor_id == Sensor.id, isouter=True)
            .where(SensorData.dataset_id == dataset.id)
            .order_by(SensorData.source_id)
        ).all()
        for sensor_data, sensor in rows:
            channel = (sensor.channel if sensor else None) or sensor_data.metadata_json.get("channel")
            if channel is None:
                logger.warning("Sensor data %s has no channel; skipped from sensor features", sensor_data.source_id)
                continue
            refs_by_sample[sensor_data.sample_id].append(
                SensorRefInput(
                    source_id=sensor_data.source_id,
                    channel=channel,
                    modality=sensor.modality if sensor else None,
                    file_exists=bool(sensor_data.file_exists),
                )
            )
            if channel == self.ego_pose_channel:
                token = sensor_data.metadata_json.get("ego_pose_source_id")
                if token:
                    pose_token_by_sample[sensor_data.sample_id] = token

        expected_channels = tuple(sorted({r.channel for refs in refs_by_sample.values() for r in refs}))
        timestamps_by_scene: dict[int, list[int]] = defaultdict(list)
        for sample in samples:
            timestamps_by_scene[sample.scene_id].append(sample.timestamp)

        for sample in samples:
            scene = scenes[sample.scene_id]
            stamps = timestamps_by_scene[scene.id]
            yield SampleInput(
                sample_id=sample.source_id,
                scene_id=scene.source_id,
                scene_name=scene.name,
                timestamp=sample.timestamp,
                scene_sample_count=len(stamps),
                scene_duration_s=(max(stamps) - min(stamps)) / 1_000_000 if len(stamps) > 1 else None,
                location=self._location(scene.log_source_id),
                annotations=tuple(annotations_by_sample.get(sample.id, ())),
                sensor_refs=tuple(refs_by_sample.get(sample.id, ())),
                ego_pose=self._ego_pose(pose_token_by_sample.get(sample.id)),
                expected_channels=expected_channels,
            )

    def _location(self, log_source_id: str | None) -> str | None:
        if self.lookups is None or log_source_id is None:
            return None
        return self.lookups.log_locations.get(log_source_id)

    def _ego_pose(self, token: str | None) -> EgoPose | None:
        if self.lookups is None or token is None:
            return None
        raw = self.lookups.ego_poses.get(token)
        if raw is None:
            logger.warning("Ego pose token %s not found in lookups", token)
            return None
        return EgoPose(raw["timestamp"], tuple(raw["translation"]), tuple(raw["rotation"]))

    def _annotation_input(self, annotation: Annotation) -> AnnotationInput:
        attrs = annotation.attributes or {}
        visibility = attrs.get("visibility_token")
        translation = attrs.get("translation")
        names: tuple[str, ...] | None = None
        if self.lookups is not None:
            tokens = attrs.get("attribute_tokens") or []
            names = tuple(sorted(self.lookups.attribute_names.get(t, f"unknown:{t}") for t in tokens))
        return AnnotationInput(
            source_id=annotation.source_id,
            category=annotation.category,
            instance_source_id=annotation.instance_source_id,
            translation=tuple(translation) if translation else None,
            visibility_level=(
                self.lookups.visibility_levels.get(visibility) if self.lookups and visibility else None
            ),
            attribute_names=names,
            num_lidar_pts=attrs.get("num_lidar_pts"),
            num_radar_pts=attrs.get("num_radar_pts"),
        )
