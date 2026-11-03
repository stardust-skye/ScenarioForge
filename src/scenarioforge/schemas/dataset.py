"""Pydantic schemas for dataset-independent metadata records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecordSchema(BaseModel):
    """Base schema requiring stable source identifiers."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)


class DatasetSchema(RecordSchema):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    root_path: str = Field(min_length=1)


class SceneSchema(RecordSchema):
    name: str = Field(min_length=1)
    description: str = ""
    log_source_id: str | None = None
    declared_sample_count: int = Field(ge=0)


class SampleSchema(RecordSchema):
    scene_source_id: str = Field(min_length=1)
    timestamp: int = Field(ge=0, description="Unix timestamp in microseconds")
    previous_source_id: str | None = None
    next_source_id: str | None = None


class SensorSchema(RecordSchema):
    channel: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    calibration: dict[str, Any] = Field(default_factory=dict)


class SensorDataSchema(RecordSchema):
    sample_source_id: str = Field(min_length=1)
    sensor_source_id: str = Field(min_length=1)
    timestamp: int = Field(ge=0, description="Unix timestamp in microseconds")
    file_path: str = Field(min_length=1)
    is_key_frame: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationSchema(RecordSchema):
    sample_source_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    instance_source_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
