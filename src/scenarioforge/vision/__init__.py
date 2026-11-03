"""Model-independent camera annotation projection and dataset preparation."""

from scenarioforge.vision.config import ProjectionConfig
from scenarioforge.vision.converter import ConversionResult, convert_camera_samples

__all__ = ["ConversionResult", "ProjectionConfig", "convert_camera_samples"]
