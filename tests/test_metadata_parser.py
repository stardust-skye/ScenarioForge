import pytest
from pydantic import ValidationError

from scenarioforge.ingestion.metadata_parser import parse_record
from scenarioforge.schemas.dataset import SampleSchema


def test_parser_allows_supported_adapter_traversal_fields() -> None:
    parsed = parse_record(
        SampleSchema,
        {
            "source_id": "sample",
            "scene_source_id": "scene",
            "timestamp": 10,
            "data": {"CAM_FRONT": "sensor-data"},
            "annotations": ["annotation"],
        },
    )

    assert parsed.source_id == "sample"
    assert not hasattr(parsed, "data")


def test_parser_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        parse_record(
            SampleSchema,
            {
                "source_id": "sample",
                "scene_source_id": "scene",
                "timestamp": 10,
                "timestmap": 11,
            },
        )


def test_parser_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        parse_record(SampleSchema, {"source_id": "sample"})
