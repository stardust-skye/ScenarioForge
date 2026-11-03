from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from scenarioforge.analytics.basic_statistics import export_statistics, generate_statistics
from scenarioforge.config.settings import Settings
from scenarioforge.database.connection import (
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from scenarioforge.database.models import (
    Annotation,
    IngestionIssue,
    Sample,
    SensorData,
    Scene,
)
from scenarioforge.datasets.base import DatasetAdapter
from scenarioforge.ingestion.loader import ingest_dataset


class FixtureAdapter(DatasetAdapter):
    def __init__(
        self,
        file_path: Path,
        *,
        sample_timestamp: int = 1_000_000,
        reference_override: str | None = None,
        duplicate_sample: bool = False,
    ) -> None:
        self.file_path = file_path
        self.sample_timestamp = sample_timestamp
        self.reference_override = reference_override
        self.duplicate_sample = duplicate_sample

    @property
    def dataset_record(self) -> dict[str, Any]:
        return {
            "source_id": "fixture:mini",
            "name": "fixture",
            "version": "mini",
            "root_path": str(self.file_path.parent),
        }

    def validate_accessibility(self) -> list[str]:
        return []

    def iter_scenes(self):
        yield {
            "source_id": "scene-1",
            "name": "scene-1",
            "description": "",
            "declared_sample_count": 2 if self.duplicate_sample else 1,
            "first_sample_source_id": "sample-1",
        }

    def iter_samples(self, scene: dict[str, Any]):
        sample = {
            "source_id": "sample-1",
            "scene_source_id": scene["source_id"],
            "timestamp": self.sample_timestamp,
            "previous_source_id": None,
            "next_source_id": None,
            "data": {},
            "annotations": ["annotation-1"],
        }
        yield sample
        if self.duplicate_sample:
            yield sample

    def iter_sensors(self):
        yield {
            "source_id": "calibrated-sensor-1",
            "channel": "CAM_FRONT",
            "modality": "camera",
            "calibration": {"translation": [0, 0, 0]},
        }

    def iter_sensor_data(self, sample: dict[str, Any]):
        yield {
            "source_id": "sensor-data-1",
            "sample_source_id": self.reference_override or sample["source_id"],
            "sensor_source_id": "calibrated-sensor-1",
            "timestamp": self.sample_timestamp,
            "file_path": str(self.file_path),
            "is_key_frame": True,
            "metadata": {"channel": "CAM_FRONT"},
        }

    def iter_annotations(self, sample: dict[str, Any]):
        yield {
            "source_id": "annotation-1",
            "sample_source_id": self.reference_override or sample["source_id"],
            "category": "vehicle.car",
            "instance_source_id": "instance-1",
            "attributes": {"translation": [1, 2, 3]},
        }

    def sensor_file_exists(self, file_path: Path) -> bool:
        return file_path.is_file()


def test_ingestion_is_idempotent_and_reports_missing_sensor_files(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    adapter = FixtureAdapter(tmp_path / "missing.bin")

    try:
        first = ingest_dataset(adapter, factory)
        second = ingest_dataset(adapter, factory)
        assert first.scenes == second.scenes == 1
        assert first.samples == second.samples == 1
        assert first.sensor_data == second.sensor_data == 1
        assert first.annotations == second.annotations == 1
        assert second.missing_sensor_files == 1

        with factory() as session:
            assert session.scalar(select(func.count(Sample.id))) == 1
            assert session.scalar(select(func.count(SensorData.id))) == 1
            assert session.scalar(select(func.count(Annotation.id))) == 1
            issue = session.scalar(select(IngestionIssue))
            assert issue is not None
            assert issue.code == "missing_sensor_file"
            assert issue.is_resolved is False

            report = generate_statistics(session, "fixture:mini")
            assert report["counts"]["scenes"] == 1
            assert report["counts"]["sensor_data"] == 1
            assert report["sensor_channels"] == {"CAM_FRONT": 1}
            assert report["sensor_modalities"] == {"camera": 1}
            assert report["annotation_categories"] == {"vehicle.car": 1}
            assert report["missing_or_invalid_references"]["missing_sensor_files"] == 1

        json_path = tmp_path / "report.json"
        csv_path = tmp_path / "report.csv"
        export_statistics(report, json_path, csv_path)
        assert json_path.is_file()
        assert csv_path.read_text(encoding="utf-8").startswith("metric,value")
    finally:
        engine.dispose()


def test_ingestion_records_invalid_timestamps(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        summary = ingest_dataset(
            FixtureAdapter(tmp_path / "sensor.bin", sample_timestamp=-1), factory
        )

        assert summary.samples == 0
        assert summary.invalid_timestamps == 1
        with factory() as session:
            issue = session.scalar(select(IngestionIssue))
            assert issue is not None
            assert issue.code == "invalid_timestamp"
    finally:
        engine.dispose()


def test_ingestion_records_dangling_sample_references(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        summary = ingest_dataset(
            FixtureAdapter(tmp_path / "sensor.bin", reference_override="unknown-sample"),
            factory,
        )

        assert summary.unresolved_references == 2
        with factory() as session:
            assert session.scalar(select(func.count(SensorData.id))) == 0
            assert session.scalar(
                select(func.count(IngestionIssue.id)).where(
                    IngestionIssue.code == "missing_sample_reference"
                )
            ) == 2
    finally:
        engine.dispose()


def test_ingestion_detects_duplicate_source_identifiers(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        summary = ingest_dataset(
            FixtureAdapter(tmp_path / "sensor.bin", duplicate_sample=True), factory
        )

        assert summary.samples == 1
        assert summary.duplicate_identifiers == 1
        with factory() as session:
            assert session.scalar(select(func.count(Sample.id))) == 1
    finally:
        engine.dispose()


def test_failed_traversal_rolls_back_and_preserves_previous_issues(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        initial = ingest_dataset(FixtureAdapter(tmp_path / "missing.bin"), factory)
        assert initial.status == "completed_with_issues"

        class FailingAdapter(FixtureAdapter):
            def iter_scenes(self):
                yield from super().iter_scenes()
                raise ValueError("fixture traversal failed after scene processing")

        existing_file = tmp_path / "existing.bin"
        existing_file.write_bytes(b"fixture")
        failed = ingest_dataset(FailingAdapter(existing_file), factory)
        assert failed.status == "failed"
        assert "fixture traversal failed" in failed.fatal_error

        with factory() as session:
            assert session.scalar(select(func.count(Scene.id))) == 1
            assert session.scalar(select(func.count(Sample.id))) == 1
            assert session.scalar(select(func.count(SensorData.id))) == 1
            assert session.scalar(select(func.count(Annotation.id))) == 1
            sensor_data = session.scalar(select(SensorData))
            assert sensor_data is not None
            assert sensor_data.file_exists is False
            issue = session.scalar(
                select(IngestionIssue).where(IngestionIssue.code == "missing_sensor_file")
            )
            assert issue is not None
            assert issue.is_resolved is False
            assert session.scalar(select(func.count(IngestionIssue.id))) == 1
    finally:
        engine.dispose()


def test_missing_source_ids_create_distinct_issues(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        class MissingIdentifiersAdapter(FixtureAdapter):
            def iter_scenes(self):
                yield {"name": "missing-scene-one"}
                yield {"name": "missing-scene-two"}

        summary = ingest_dataset(
            MissingIdentifiersAdapter(tmp_path / "sensor.bin"), factory
        )
        assert summary.status == "completed_with_issues"

        with factory() as session:
            issues = session.scalars(
                select(IngestionIssue).where(IngestionIssue.code == "missing_identifier")
            ).all()
            assert len(issues) == 2
            assert issues[0].source_id != issues[1].source_id
    finally:
        engine.dispose()


def test_completed_run_records_scene_sample_count_mismatch(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        class MismatchedAdapter(FixtureAdapter):
            def iter_scenes(self):
                scene = next(super().iter_scenes())
                scene["declared_sample_count"] = 2
                yield scene

        summary = ingest_dataset(MismatchedAdapter(tmp_path / "sensor.bin"), factory)
        assert summary.status == "completed_with_issues"

        with factory() as session:
            issue = session.scalar(
                select(IngestionIssue).where(IngestionIssue.code == "sample_count_mismatch")
            )
            assert issue is not None
            assert issue.is_resolved is False
    finally:
        engine.dispose()


def test_statistics_include_every_unresolved_issue_code(tmp_path: Path) -> None:
    engine, factory = create_database(tmp_path)
    try:
        ingest_dataset(FixtureAdapter(tmp_path / "missing.bin"), factory)
        with factory() as session:
            from scenarioforge.database.repository import MetadataRepository

            dataset_id = session.scalar(select(IngestionIssue.dataset_id))
            assert dataset_id is not None
            repository = MetadataRepository(session)
            for code in (
                "sample_count_mismatch",
                "invalid_timestamp",
                "duplicate_identifier",
                "sample_traversal_error",
                "missing_sample_reference",
                "other_recorded_error",
            ):
                repository.record_issue(
                    dataset_id, "scene", "scene-1", code, f"Recorded {code}."
                )
            session.commit()

            report = generate_statistics(session, "fixture:mini")
            assert report["unresolved_ingestion_issues"] == {
                "duplicate_identifier": 1,
                "invalid_timestamp": 1,
                "missing_sensor_file": 1,
                "missing_sample_reference": 1,
                "other_recorded_error": 1,
                "sample_count_mismatch": 1,
                "sample_traversal_error": 1,
            }
            assert report["sensor_data_coverage"]["sweeps_included"] is False
    finally:
        engine.dispose()


def create_database(tmp_path: Path):
    settings = Settings(
        project_root=tmp_path,
        dataset_root=tmp_path / "raw",
        dataset_version="mini",
        database_url=f"sqlite:///{(tmp_path / 'metadata.db').as_posix()}",
        log_level="INFO",
    )
    engine = create_database_engine(settings)
    initialize_database(engine)
    return engine, create_session_factory(engine)
