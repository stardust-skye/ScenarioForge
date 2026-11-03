from pathlib import Path

import pytest
from sqlalchemy import func, select

from scenarioforge.config.settings import Settings
from scenarioforge.database.connection import (
    create_database_engine,
    create_session_factory,
    initialize_database,
)
from scenarioforge.database.models import Annotation, Dataset, Sample, Scene, Sensor
from scenarioforge.database.repository import MetadataRepository
from scenarioforge.schemas.dataset import (
    AnnotationSchema,
    DatasetSchema,
    SampleSchema,
    SceneSchema,
    SensorDataSchema,
    SensorSchema,
)


def test_repository_upserts_source_records(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dataset_root=tmp_path / "raw",
        dataset_version="v1.0-mini",
        database_url=f"sqlite:///{(tmp_path / 'metadata.db').as_posix()}",
        log_level="INFO",
    )
    engine = create_database_engine(settings)
    initialize_database(engine)
    factory = create_session_factory(engine)

    try:
        with factory() as session, session.begin():
            repository = MetadataRepository(session)
            dataset = repository.upsert_dataset(
                DatasetSchema(
                    source_id="nuscenes:v1.0-mini",
                    name="nuscenes",
                    version="v1.0-mini",
                    root_path=str(tmp_path / "raw"),
                )
            )
            scene = repository.upsert_scene(
                dataset.id,
                SceneSchema(
                    source_id="scene-token",
                    name="scene-001",
                    declared_sample_count=1,
                ),
            )
            sample_record = SampleSchema(
                source_id="sample-token",
                scene_source_id="scene-token",
                timestamp=100,
            )
            repository.upsert_sample(dataset.id, scene.id, sample_record)
            repository.upsert_sample(dataset.id, scene.id, sample_record)

        with factory() as session:
            assert session.scalar(select(func.count(Dataset.id))) == 1
            assert session.scalar(select(func.count(Scene.id))) == 1
            assert session.scalar(select(func.count(Sample.id))) == 1
    finally:
        engine.dispose()


def test_repository_rejects_cross_dataset_parent_associations(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        dataset_root=tmp_path / "raw",
        dataset_version="v1.0-mini",
        database_url=f"sqlite:///{(tmp_path / 'ownership.db').as_posix()}",
        log_level="INFO",
    )
    engine = create_database_engine(settings)
    initialize_database(engine)
    factory = create_session_factory(engine)

    try:
        with factory() as session, session.begin():
            repository = MetadataRepository(session)
            datasets = [
                repository.upsert_dataset(
                    DatasetSchema(
                        source_id=f"dataset-{index}",
                        name=f"dataset-{index}",
                        version="mini",
                        root_path=str(tmp_path),
                    )
                )
                for index in (1, 2)
            ]
            scenes = [
                repository.upsert_scene(
                    dataset.id,
                    SceneSchema(
                        source_id=f"scene-{index}",
                        name=f"scene-{index}",
                        declared_sample_count=1,
                    ),
                )
                for index, dataset in enumerate(datasets, start=1)
            ]
            samples = [
                repository.upsert_sample(
                    dataset.id,
                    scene.id,
                    SampleSchema(
                        source_id=f"sample-{index}",
                        scene_source_id=scene.source_id,
                        timestamp=100,
                    ),
                )
                for index, (dataset, scene) in enumerate(zip(datasets, scenes), start=1)
            ]
            sensor = repository.upsert_sensor(
                datasets[1].id,
                SensorSchema(
                    source_id="sensor-2",
                    channel="CAM_FRONT",
                    modality="camera",
                ),
            )

        with factory() as session, session.begin():
            repository = MetadataRepository(session)
            with pytest.raises(ValueError, match="does not belong to dataset"):
                repository.upsert_sample(
                    datasets[0].id,
                    scenes[1].id,
                    SampleSchema(
                        source_id="cross-sample",
                        scene_source_id=scenes[1].source_id,
                        timestamp=200,
                    ),
                )
            with pytest.raises(ValueError, match="does not belong to dataset"):
                repository.upsert_sensor_data(
                    datasets[0].id,
                    samples[0].id,
                    sensor.id,
                    sensor.source_id,
                    SensorDataSchema(
                        source_id="cross-sensor-data",
                        sample_source_id=samples[0].source_id,
                        sensor_source_id=sensor.source_id,
                        timestamp=200,
                        file_path=str(tmp_path / "sensor.bin"),
                        is_key_frame=True,
                    ),
                    file_exists=False,
                )
            with pytest.raises(ValueError, match="does not belong to dataset"):
                repository.upsert_annotation(
                    datasets[0].id,
                    samples[1].id,
                    AnnotationSchema(
                        source_id="cross-annotation",
                        sample_source_id=samples[1].source_id,
                        category="vehicle.car",
                    ),
                )

        with factory() as session:
            assert session.scalar(select(func.count(Sample.id))) == 2
            assert session.scalar(select(func.count(Annotation.id))) == 0
    finally:
        engine.dispose()
