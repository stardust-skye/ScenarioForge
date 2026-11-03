from pathlib import Path

import pytest
from pydantic import ValidationError

from scenarioforge.ingestion.validators import validate_record, validate_sensor_path
from scenarioforge.schemas.dataset import SampleSchema


def test_sample_schema_accepts_nonnegative_microsecond_timestamp() -> None:
    record = validate_record(
        SampleSchema,
        {"source_id": "sample-token", "scene_source_id": "scene-token", "timestamp": 123},
    )

    assert record.timestamp == 123


def test_sample_schema_rejects_negative_timestamp() -> None:
    with pytest.raises(ValidationError):
        validate_record(
            SampleSchema,
            {"source_id": "sample-token", "scene_source_id": "scene-token", "timestamp": -1},
        )


def test_validate_sensor_path_reports_missing_file(tmp_path: Path) -> None:
    path, exists = validate_sensor_path(str(tmp_path / "not-present.bin"))

    assert path == tmp_path / "not-present.bin"
    assert exists is False
