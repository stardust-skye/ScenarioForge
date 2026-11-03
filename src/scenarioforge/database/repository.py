"""Idempotent persistence operations for normalized metadata records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from scenarioforge.database.models import (
    Annotation,
    Dataset,
    IngestionIssue,
    Sample,
    Scene,
    Sensor,
    SensorData,
)
from scenarioforge.schemas.dataset import (
    AnnotationSchema,
    DatasetSchema,
    SampleSchema,
    SceneSchema,
    SensorDataSchema,
    SensorSchema,
)


class MetadataRepository:
    """Upsert metadata by stable source IDs scoped to a dataset."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_dataset(self, record: DatasetSchema) -> Dataset:
        entity = self.session.scalar(select(Dataset).where(Dataset.source_id == record.source_id))
        if entity is None:
            entity = Dataset(source_id=record.source_id)
            self.session.add(entity)
        entity.name = record.name
        entity.version = record.version
        entity.root_path = record.root_path
        self.session.flush()
        return entity

    def upsert_scene(self, dataset_id: int, record: SceneSchema) -> Scene:
        entity = self._find(Scene, dataset_id, record.source_id)
        if entity is None:
            entity = Scene(dataset_id=dataset_id, source_id=record.source_id)
            self.session.add(entity)
        entity.name = record.name
        entity.description = record.description
        entity.log_source_id = record.log_source_id
        entity.declared_sample_count = record.declared_sample_count
        self.session.flush()
        return entity

    def upsert_sample(self, dataset_id: int, scene_id: int, record: SampleSchema) -> Sample:
        self._require_owned(Scene, scene_id, dataset_id, "scene")
        entity = self._find(Sample, dataset_id, record.source_id)
        if entity is None:
            entity = Sample(dataset_id=dataset_id, source_id=record.source_id)
            self.session.add(entity)
        entity.scene_id = scene_id
        entity.timestamp = record.timestamp
        entity.previous_source_id = record.previous_source_id
        entity.next_source_id = record.next_source_id
        self.session.flush()
        return entity

    def upsert_sensor(self, dataset_id: int, record: SensorSchema) -> Sensor:
        entity = self._find(Sensor, dataset_id, record.source_id)
        if entity is None:
            entity = Sensor(dataset_id=dataset_id, source_id=record.source_id)
            self.session.add(entity)
        entity.channel = record.channel
        entity.modality = record.modality
        entity.calibration = record.calibration
        self.session.flush()
        return entity

    def upsert_sensor_data(
        self,
        dataset_id: int,
        sample_id: int,
        sensor_id: int | None,
        sensor_source_id: str,
        record: SensorDataSchema,
        file_exists: bool,
    ) -> SensorData:
        self._require_owned(Sample, sample_id, dataset_id, "sample")
        if sensor_id is not None:
            self._require_owned(Sensor, sensor_id, dataset_id, "sensor")
        entity = self._find(SensorData, dataset_id, record.source_id)
        if entity is None:
            entity = SensorData(dataset_id=dataset_id, source_id=record.source_id)
            self.session.add(entity)
        entity.sample_id = sample_id
        entity.sensor_id = sensor_id
        entity.sensor_source_id = sensor_source_id
        entity.timestamp = record.timestamp
        entity.file_path = record.file_path
        entity.is_key_frame = record.is_key_frame
        entity.file_exists = file_exists
        entity.metadata_json = record.metadata
        self.session.flush()
        return entity

    def upsert_annotation(
        self, dataset_id: int, sample_id: int, record: AnnotationSchema
    ) -> Annotation:
        self._require_owned(Sample, sample_id, dataset_id, "sample")
        entity = self._find(Annotation, dataset_id, record.source_id)
        if entity is None:
            entity = Annotation(dataset_id=dataset_id, source_id=record.source_id)
            self.session.add(entity)
        entity.sample_id = sample_id
        entity.category = record.category
        entity.instance_source_id = record.instance_source_id
        entity.attributes = record.attributes
        self.session.flush()
        return entity

    def record_issue(
        self,
        dataset_id: int,
        entity_type: str,
        source_id: str | None,
        code: str,
        message: str,
        severity: str = "error",
    ) -> IngestionIssue:
        stable_source_id = source_id or ""
        statement = select(IngestionIssue).where(
            IngestionIssue.dataset_id == dataset_id,
            IngestionIssue.entity_type == entity_type,
            IngestionIssue.source_id == stable_source_id,
            IngestionIssue.code == code,
        )
        entity = self.session.scalar(statement)
        if entity is None:
            entity = IngestionIssue(
                dataset_id=dataset_id,
                entity_type=entity_type,
                source_id=stable_source_id,
                code=code,
                severity=severity,
                message=message,
            )
            self.session.add(entity)
        else:
            entity.severity = severity
            entity.message = message
            entity.is_resolved = False
        self.session.flush()
        return entity

    def resolve_issue(
        self, dataset_id: int, entity_type: str, source_id: str | None, code: str
    ) -> None:
        entity = self.session.scalar(
            select(IngestionIssue).where(
                IngestionIssue.dataset_id == dataset_id,
                IngestionIssue.entity_type == entity_type,
                IngestionIssue.source_id == (source_id or ""),
                IngestionIssue.code == code,
            )
        )
        if entity is not None:
            entity.is_resolved = True

    def resolve_all_issues(self, dataset_id: int) -> None:
        for issue in self.session.scalars(
            select(IngestionIssue).where(
                IngestionIssue.dataset_id == dataset_id,
                IngestionIssue.is_resolved.is_(False),
            )
        ):
            issue.is_resolved = True

    def _require_owned(
        self, model: type[Any], entity_id: int, dataset_id: int, entity_name: str
    ) -> None:
        entity = self.session.scalar(
            select(model).where(model.id == entity_id, model.dataset_id == dataset_id)
        )
        if entity is None:
            raise ValueError(
                f"{entity_name.title()} {entity_id} does not belong to dataset {dataset_id}"
            )

    def _find(self, model: type[Any], dataset_id: int, source_id: str) -> Any | None:
        return self.session.scalar(
            select(model).where(model.dataset_id == dataset_id, model.source_id == source_id)
        )
