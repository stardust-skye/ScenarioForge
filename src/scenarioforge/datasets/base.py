"""Dataset-independent adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class DatasetAdapter(ABC):
    """Interface implemented by each dataset-specific metadata adapter."""

    @property
    @abstractmethod
    def dataset_record(self) -> dict[str, Any]:
        """Return the dataset identity and source location."""

    @abstractmethod
    def validate_accessibility(self) -> list[str]:
        """Return actionable metadata-root problems; an empty list means accessible."""

    @abstractmethod
    def iter_scenes(self) -> Iterator[dict[str, Any]]:
        """Yield normalized scene records."""

    @abstractmethod
    def iter_samples(self, scene: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield normalized samples belonging to a scene."""

    @abstractmethod
    def iter_sensors(self) -> Iterator[dict[str, Any]]:
        """Yield normalized calibrated sensor records."""

    @abstractmethod
    def iter_sensor_data(self, sample: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield file references for the sensors attached to a sample."""

    @abstractmethod
    def iter_annotations(self, sample: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield normalized annotations attached to a sample."""

    @abstractmethod
    def sensor_file_exists(self, file_path: Path) -> bool:
        """Check whether a referenced sensor file is locally accessible."""
