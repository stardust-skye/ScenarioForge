"""Reusable object-detection training and evaluation infrastructure."""

from scenarioforge.training.config import ExperimentConfig, ModelConfig
from scenarioforge.training.experiment import DetectionExperiment
from scenarioforge.training.model_factory import create_detection_model

__all__ = [
    "DetectionExperiment",
    "ExperimentConfig",
    "ModelConfig",
    "create_detection_model",
]
