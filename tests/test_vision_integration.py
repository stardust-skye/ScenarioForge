from __future__ import annotations

from pathlib import Path

import pytest

from scenarioforge.config.settings import Settings
from scenarioforge.database.connection import create_database_engine
from scenarioforge.database.models import Dataset
from scenarioforge.datasets.nuscenes_adapter import NuScenesAdapter
from scenarioforge.vision.converter import convert_camera_samples
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session


def _dataset_root(settings: Settings) -> Path | None:
    if settings.dataset_root.is_dir():
        return settings.dataset_root
    engine = create_database_engine(settings)
    try:
        if not inspect(engine).has_table(Dataset.__tablename__):
            return None
        with Session(engine) as session:
            root = session.scalar(
                select(Dataset.root_path).where(
                    Dataset.source_id == f"nuscenes:{settings.dataset_version}"
                )
            )
        return Path(root) if root and Path(root).is_dir() else None
    finally:
        engine.dispose()


def test_convert_actual_nuscenes_mini_records_and_geometry():
    settings = Settings.from_env()
    root = _dataset_root(settings)
    if root is None:
        pytest.skip("nuScenes mini is not available in the configured root or ingested database")
    adapter = NuScenesAdapter(root, settings.dataset_version)
    frames = list(adapter.iter_camera_samples("CAM_FRONT"))
    result = convert_camera_samples(
        frames,
        dataset_root=root,
        dataset_version=settings.dataset_version,
        visibility_levels=adapter.visibility_levels,
    )

    assert result.report["total_samples_processed"] == len(adapter._nusc.sample)
    assert result.report["total_camera_images_processed"] == len(adapter._nusc.sample)
    assert result.report["total_source_annotations_considered"] == len(adapter._nusc.sample_annotation)
    assert result.report["total_source_annotations_considered"] == (
        len(result.accepted_annotations) + len(result.rejected_annotations)
    )
    assert all(image.image_status == "ok" for image in result.images)
    assert all(image.width == 1600 and image.height == 900 for image in result.images)
    assert all(
        annotation.scene_token == next(
            image.scene_token for image in result.images if image.sample_token == annotation.sample_token
        )
        for annotation in result.accepted_annotations
    )
    assert all(
        0 <= annotation.x_min < annotation.x_max <= annotation.image_width
        and 0 <= annotation.y_min < annotation.y_max <= annotation.image_height
        for annotation in result.accepted_annotations
    )
