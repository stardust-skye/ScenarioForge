"""SQLAlchemy relational schema for dataset metadata and ingestion issues."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base shared by all ScenarioForge database entities."""


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    root_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    scenes: Mapped[list[Scene]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    sensors: Mapped[list[Sensor]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class Scene(Base):
    __tablename__ = "scenes"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_id", name="uq_scenes_dataset_source"),
        Index("ix_scenes_dataset_name", "dataset_id", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    log_source_id: Mapped[str | None] = mapped_column(String(255))
    declared_sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    dataset: Mapped[Dataset] = relationship(back_populates="scenes")
    samples: Mapped[list[Sample]] = relationship(back_populates="scene", cascade="all, delete-orphan")


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_id", name="uq_samples_dataset_source"),
        Index("ix_samples_scene_timestamp", "scene_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    scene_id: Mapped[int] = mapped_column(
        ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False, comment="Unix microseconds")
    previous_source_id: Mapped[str | None] = mapped_column(String(255))
    next_source_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    scene: Mapped[Scene] = relationship(back_populates="samples")
    sensor_data: Mapped[list[SensorData]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )


class Sensor(Base):
    __tablename__ = "sensors"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_id", name="uq_sensors_dataset_source"),
        Index("ix_sensors_dataset_channel", "dataset_id", "channel"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    channel: Mapped[str] = mapped_column(String(100), nullable=False)
    modality: Mapped[str] = mapped_column(String(100), nullable=False)
    calibration: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    dataset: Mapped[Dataset] = relationship(back_populates="sensors")
    sensor_data: Mapped[list[SensorData]] = relationship(back_populates="sensor")


class SensorData(Base):
    __tablename__ = "sensor_data"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_id", name="uq_sensor_data_dataset_source"),
        Index("ix_sensor_data_sample_timestamp", "sample_id", "timestamp"),
        Index("ix_sensor_data_sensor_timestamp", "sensor_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    sample_id: Mapped[int] = mapped_column(
        ForeignKey("samples.id", ondelete="CASCADE"), nullable=False
    )
    sensor_id: Mapped[int | None] = mapped_column(ForeignKey("sensors.id", ondelete="SET NULL"))
    sensor_source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False, comment="Unix microseconds")
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_key_frame: Mapped[bool] = mapped_column(Boolean, nullable=False)
    file_exists: Mapped[bool] = mapped_column(Boolean, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    sample: Mapped[Sample] = relationship(back_populates="sensor_data")
    sensor: Mapped[Sensor | None] = relationship(back_populates="sensor_data")


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint("dataset_id", "source_id", name="uq_annotations_dataset_source"),
        Index("ix_annotations_sample_category", "sample_id", "category"),
        Index("ix_annotations_dataset_category", "dataset_id", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    sample_id: Mapped[int] = mapped_column(
        ForeignKey("samples.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    instance_source_id: Mapped[str | None] = mapped_column(String(255))
    attributes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    sample: Mapped[Sample] = relationship(back_populates="annotations")


class IngestionIssue(Base):
    __tablename__ = "ingestion_issues"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "entity_type",
            "source_id",
            "code",
            name="uq_issues_dataset_entity_source_code",
        ),
        Index("ix_issues_dataset_severity", "dataset_id", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    code: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
