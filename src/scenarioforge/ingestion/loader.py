"""Idempotent ingestion of normalized dataset metadata."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from scenarioforge.datasets.base import DatasetAdapter
from scenarioforge.database.repository import MetadataRepository
from scenarioforge.ingestion.metadata_parser import parse_record
from scenarioforge.ingestion.validators import validation_error_message, validate_sensor_path
from scenarioforge.schemas.dataset import (
    AnnotationSchema,
    DatasetSchema,
    SampleSchema,
    SceneSchema,
    SensorDataSchema,
    SensorSchema,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionSummary:
    """Counts for one ingestion run; counts are processed records, not DB deltas."""

    dataset_source_id: str
    status: Literal[
        "running", "completed", "completed_with_issues", "failed"
    ] = "running"
    scenes: int = 0
    samples: int = 0
    sensors: int = 0
    sensor_data: int = 0
    annotations: int = 0
    issues: int = 0
    missing_sensor_files: int = 0
    invalid_timestamps: int = 0
    duplicate_identifiers: int = 0
    unresolved_references: int = 0
    fatal_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable summary values."""
        return asdict(self)


def ingest_dataset(
    adapter: DatasetAdapter,
    session_factory: sessionmaker[Session],
) -> IngestionSummary:
    """Validate and upsert dataset metadata while recording row-level problems."""
    dataset = parse_record(DatasetSchema, adapter.dataset_record)
    summary = IngestionSummary(dataset_source_id=dataset.source_id)

    try:
        with session_factory() as session, session.begin():
            repository = MetadataRepository(session)
            dataset_entity = repository.upsert_dataset(dataset)
            dataset_id = dataset_entity.id
            repository.resolve_all_issues(dataset_id)

            sensor_ids = _ingest_sensors(adapter, repository, dataset_id, summary)
            seen_scenes: set[str] = set()
            seen_samples: set[str] = set()
            seen_sensor_data: set[str] = set()
            seen_annotations: set[str] = set()
            for scene_index, scene_raw in enumerate(adapter.iter_scenes()):
                scene_source_id = _source_id(scene_raw)
                if _is_duplicate(
                    repository,
                    dataset_id,
                    "scene",
                    scene_source_id,
                    seen_scenes,
                    summary,
                    f"scene[{scene_index}]",
                ):
                    continue
                try:
                    scene = parse_record(SceneSchema, scene_raw)
                    scene_entity = repository.upsert_scene(dataset_id, scene)
                except ValidationError as exc:
                    _record_validation_issue(
                        repository, dataset_id, "scene", scene_source_id, exc, summary
                    )
                    continue
                summary.scenes += 1
                observed_samples = 0
                for sample_index, sample_raw in enumerate(adapter.iter_samples(scene_raw)):
                    observed_samples += 1
                    _ingest_sample(
                        adapter,
                        sample_raw,
                        scene_source_id,
                        scene_entity.id,
                        dataset_id,
                        sensor_ids,
                        repository,
                        summary,
                        seen_samples,
                        seen_sensor_data,
                        seen_annotations,
                        f"sample[{sample_index}]",
                    )
                if observed_samples != scene.declared_sample_count:
                    _record_issue(
                        repository,
                        dataset_id,
                        "scene",
                        scene_source_id,
                        "sample_count_mismatch",
                        f"Scene declares {scene.declared_sample_count} samples but traversal "
                        f"found {observed_samples}.",
                        summary,
                    )
    except Exception as exc:
        summary.status = "failed"
        summary.fatal_error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "Ingestion failed; metadata transaction rolled back",
            extra={
                "event": "ingestion_failed",
                "dataset_source_id": dataset.source_id,
                "summary": summary.to_dict(),
            },
        )
        return summary

    summary.status = "completed_with_issues" if summary.issues else "completed"
    logger.info(
        "Ingestion completed",
        extra={
            "event": "ingestion_completed",
            "dataset_source_id": dataset.source_id,
            "summary": summary.to_dict(),
        },
    )
    return summary


def _ingest_sensors(
    adapter: DatasetAdapter,
    repository: MetadataRepository,
    dataset_id: int,
    summary: IngestionSummary,
) -> dict[str, int]:
    sensor_ids: dict[str, int] = {}
    seen: set[str] = set()
    for sensor_index, raw in enumerate(adapter.iter_sensors()):
        source_id = _source_id(raw)
        if _is_duplicate(
            repository,
            dataset_id,
            "sensor",
            source_id,
            seen,
            summary,
            f"sensor[{sensor_index}]",
        ):
            continue
        try:
            record = parse_record(SensorSchema, raw)
        except ValidationError as exc:
            _record_validation_issue(repository, dataset_id, "sensor", source_id, exc, summary)
            continue
        entity = repository.upsert_sensor(dataset_id, record)
        sensor_ids[record.source_id] = entity.id
        summary.sensors += 1
    return sensor_ids


def _ingest_sample(
    adapter: DatasetAdapter,
    sample_raw: dict[str, Any],
    expected_scene_source_id: str,
    scene_id: int,
    dataset_id: int,
    sensor_ids: dict[str, int],
    repository: MetadataRepository,
    summary: IngestionSummary,
    seen_samples: set[str],
    seen_sensor_data: set[str],
    seen_annotations: set[str],
    occurrence_id: str,
) -> None:
    source_id = _source_id(sample_raw)
    if _is_duplicate(
        repository, dataset_id, "sample", source_id, seen_samples, summary, occurrence_id
    ):
        return
    try:
        sample = parse_record(SampleSchema, sample_raw)
    except ValidationError as exc:
        _record_validation_issue(repository, dataset_id, "sample", source_id, exc, summary)
        return
    if sample.scene_source_id != expected_scene_source_id:
        _record_issue(
            repository,
            dataset_id,
            "sample",
            source_id,
            "missing_scene_reference",
            f"Sample references scene {sample.scene_source_id}; expected {expected_scene_source_id}.",
            summary,
        )
        summary.unresolved_references += 1
        return

    sample_entity = repository.upsert_sample(dataset_id, scene_id, sample)
    summary.samples += 1

    for sensor_data_index, raw in enumerate(adapter.iter_sensor_data(sample_raw)):
        _ingest_sensor_data(
            adapter,
            raw,
            sample.source_id,
            sample_entity.id,
            dataset_id,
            sensor_ids,
            repository,
            summary,
            seen_sensor_data,
            f"sample_data[{sensor_data_index}]",
        )

    for annotation_index, raw in enumerate(adapter.iter_annotations(sample_raw)):
        _ingest_annotation(
            raw,
            sample.source_id,
            sample_entity.id,
            dataset_id,
            repository,
            summary,
            seen_annotations,
            f"annotation[{annotation_index}]",
        )


def _ingest_sensor_data(
    adapter: DatasetAdapter,
    raw: dict[str, Any],
    expected_sample_source_id: str,
    sample_id: int,
    dataset_id: int,
    sensor_ids: dict[str, int],
    repository: MetadataRepository,
    summary: IngestionSummary,
    seen: set[str],
    occurrence_id: str,
) -> None:
    source_id = _source_id(raw)
    if _is_duplicate(
        repository, dataset_id, "sensor_data", source_id, seen, summary, occurrence_id
    ):
        return
    try:
        record = parse_record(SensorDataSchema, raw)
    except ValidationError as exc:
        _record_validation_issue(repository, dataset_id, "sensor_data", source_id, exc, summary)
        return
    if record.sample_source_id != expected_sample_source_id:
        _record_issue(
            repository,
            dataset_id,
            "sensor_data",
            source_id,
            "missing_sample_reference",
            f"Sensor data references sample {record.sample_source_id}; "
            f"expected {expected_sample_source_id}.",
            summary,
        )
        summary.unresolved_references += 1
        return
    sensor_id = sensor_ids.get(record.sensor_source_id)
    if sensor_id is None:
        _record_issue(
            repository,
            dataset_id,
            "sensor_data",
            source_id,
            "missing_sensor_reference",
            f"Sensor data references unknown calibrated sensor {record.sensor_source_id}.",
            summary,
        )
        summary.unresolved_references += 1
    path, exists = validate_sensor_path(record.file_path)
    if exists:
        exists = adapter.sensor_file_exists(path)
    if not exists:
        repository.record_issue(
            dataset_id,
            "sensor_data",
            source_id,
            "missing_sensor_file",
            f"Sensor file is missing or inaccessible: {record.file_path}",
            severity="warning",
        )
        summary.issues += 1
        summary.missing_sensor_files += 1
        logger.warning(
            "Missing or inaccessible sensor file",
            extra={
                "event": "missing_sensor_file",
                "entity_type": "sensor_data",
                "source_id": source_id,
                "issue_code": "missing_sensor_file",
            },
        )
    else:
        repository.resolve_issue(dataset_id, "sensor_data", source_id, "missing_sensor_file")
    repository.upsert_sensor_data(
        dataset_id, sample_id, sensor_id, record.sensor_source_id, record, exists
    )
    summary.sensor_data += 1


def _ingest_annotation(
    raw: dict[str, Any],
    expected_sample_source_id: str,
    sample_id: int,
    dataset_id: int,
    repository: MetadataRepository,
    summary: IngestionSummary,
    seen: set[str],
    occurrence_id: str,
) -> None:
    source_id = _source_id(raw)
    if _is_duplicate(
        repository, dataset_id, "annotation", source_id, seen, summary, occurrence_id
    ):
        return
    try:
        record = parse_record(AnnotationSchema, raw)
    except ValidationError as exc:
        _record_validation_issue(repository, dataset_id, "annotation", source_id, exc, summary)
        return
    if record.sample_source_id != expected_sample_source_id:
        _record_issue(
            repository,
            dataset_id,
            "annotation",
            source_id,
            "missing_sample_reference",
            f"Annotation references sample {record.sample_source_id}; "
            f"expected {expected_sample_source_id}.",
            summary,
        )
        summary.unresolved_references += 1
        return
    repository.upsert_annotation(dataset_id, sample_id, record)
    summary.annotations += 1


def _source_id(record: dict[str, Any]) -> str | None:
    value = record.get("source_id")
    return value if isinstance(value, str) else None


def _is_duplicate(
    repository: MetadataRepository,
    dataset_id: int,
    entity_type: str,
    source_id: str | None,
    seen: set[str],
    summary: IngestionSummary,
    occurrence_id: str,
) -> bool:
    if source_id is None:
        _record_issue(
            repository,
            dataset_id,
            entity_type,
            f"<missing-id:{entity_type}:{occurrence_id}>",
            "missing_identifier",
            f"Record at {occurrence_id} has no string source_id.",
            summary,
        )
        return True
    if source_id in seen:
        _record_issue(
            repository,
            dataset_id,
            entity_type,
            source_id,
            "duplicate_identifier",
            f"Duplicate {entity_type} source identifier: {source_id}",
            summary,
        )
        summary.duplicate_identifiers += 1
        return True
    seen.add(source_id)
    return False


def _record_validation_issue(
    repository: MetadataRepository,
    dataset_id: int,
    entity_type: str,
    source_id: str | None,
    error: ValidationError,
    summary: IngestionSummary,
) -> None:
    message = validation_error_message(error)
    invalid_timestamp = any(
        error_item["loc"] and error_item["loc"][0] == "timestamp" for error_item in error.errors()
    )
    code = "invalid_timestamp" if invalid_timestamp else "invalid_record"
    if invalid_timestamp:
        summary.invalid_timestamps += 1
    _record_issue(repository, dataset_id, entity_type, source_id, code, message, summary)


def _record_issue(
    repository: MetadataRepository,
    dataset_id: int,
    entity_type: str,
    source_id: str | None,
    code: str,
    message: str,
    summary: IngestionSummary,
) -> None:
    repository.record_issue(dataset_id, entity_type, source_id, code, message)
    summary.issues += 1
    logger.warning(
        "%s issue: %s",
        entity_type,
        message,
        extra={
            "event": "ingestion_issue",
            "entity_type": entity_type,
            "source_id": source_id or "<unknown>",
            "issue_code": code,
        },
    )
