"""Basic descriptive statistics for ingested dataset metadata."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
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


def generate_statistics(session: Session, dataset_source_id: str) -> dict[str, Any]:
    """Aggregate persisted metadata for one dataset, without inferring absent records."""
    dataset = session.scalar(select(Dataset).where(Dataset.source_id == dataset_source_id))
    if dataset is None:
        raise ValueError(f"Dataset has not been ingested: {dataset_source_id}")

    scene_count = _count(session, Scene, dataset.id)
    sample_count = _count(session, Sample, dataset.id)
    sensor_data_count = _count(session, SensorData, dataset.id)
    annotation_count = _count(session, Annotation, dataset.id)

    sensor_types = dict(
        session.execute(
            select(Sensor.channel, func.count(Sensor.id))
            .where(Sensor.dataset_id == dataset.id)
            .group_by(Sensor.channel)
            .order_by(Sensor.channel)
        ).all()
    )
    sensor_modalities = dict(
        session.execute(
            select(Sensor.modality, func.count(Sensor.id))
            .where(Sensor.dataset_id == dataset.id)
            .group_by(Sensor.modality)
            .order_by(Sensor.modality)
        ).all()
    )
    categories = dict(
        session.execute(
            select(Annotation.category, func.count(Annotation.id))
            .where(Annotation.dataset_id == dataset.id)
            .group_by(Annotation.category)
            .order_by(Annotation.category)
        ).all()
    )
    min_timestamp, max_timestamp = session.execute(
        select(func.min(Sample.timestamp), func.max(Sample.timestamp)).where(
            Sample.dataset_id == dataset.id
        )
    ).one()
    durations = _scene_durations(session, dataset.id)
    missing_sensor_files = session.scalar(
        select(func.count(SensorData.id)).where(
            SensorData.dataset_id == dataset.id,
            SensorData.file_exists.is_(False),
        )
    ) or 0
    unresolved_issues = dict(
        session.execute(
            select(IngestionIssue.code, func.count(IngestionIssue.id))
            .where(
                IngestionIssue.dataset_id == dataset.id,
                IngestionIssue.is_resolved.is_(False),
            )
            .group_by(IngestionIssue.code)
            .order_by(IngestionIssue.code)
        ).all()
    )

    return {
        "dataset": {
            "source_id": dataset.source_id,
            "name": dataset.name,
            "version": dataset.version,
            "root_path": dataset.root_path,
        },
        "counts": {
            "scenes": scene_count,
            "samples": sample_count,
            "sensors": _count(session, Sensor, dataset.id),
            "sensor_data": sensor_data_count,
            "annotations": annotation_count,
        },
        "sensor_channels": sensor_types,
        "sensor_modalities": sensor_modalities,
        "annotation_categories": categories,
        "missing_or_invalid_references": {
            "missing_sensor_files": missing_sensor_files,
            "unresolved_sensor_references": unresolved_issues.get("missing_sensor_reference", 0),
            "unresolved_sample_or_scene_references": sum(
                count
                for code, count in unresolved_issues.items()
                if code in {"missing_scene_reference", "missing_sample_reference"}
            ),
            "invalid_timestamps": unresolved_issues.get("invalid_timestamp", 0),
        },
        "unresolved_ingestion_issues": unresolved_issues,
        "sensor_data_coverage": {
            "scope": "keyframe references linked from nuScenes sample.data",
            "sweeps_included": False,
        },
        "timestamp_coverage": {
            "minimum_unix_microseconds": min_timestamp,
            "maximum_unix_microseconds": max_timestamp,
        },
        "scene_duration_seconds": _distribution(durations),
    }


def export_statistics(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    """Write a full JSON report and a two-column flattened CSV report."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "value"))
        for key, value in _flatten(report):
            serialized = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            writer.writerow((key, serialized))


def _count(session: Session, model: type[Any], dataset_id: int) -> int:
    return session.scalar(
        select(func.count(model.id)).where(model.dataset_id == dataset_id)
    ) or 0


def _scene_durations(session: Session, dataset_id: int) -> list[float]:
    bounds = session.execute(
        select(func.min(Sample.timestamp), func.max(Sample.timestamp))
        .join(Scene, Sample.scene_id == Scene.id)
        .where(Scene.dataset_id == dataset_id)
        .group_by(Scene.id)
    ).all()
    return [(last - first) / 1_000_000 for first, last in bounds if first is not None and last is not None]


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _flatten(value: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.extend(_flatten(item, name))
        else:
            rows.append((name, item))
    return rows
