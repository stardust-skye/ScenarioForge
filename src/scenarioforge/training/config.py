"""Validated, serializable configuration for detection experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    pretrained_weights: str | None = "COCO_V1"
    freeze_backbone_epochs: int = 0

    def __post_init__(self) -> None:
        if self.architecture != "fasterrcnn_mobilenet_v3_large_320_fpn":
            raise ValueError(f"Unsupported detector architecture: {self.architecture}")
        if self.pretrained_weights not in (None, "COCO_V1"):
            raise ValueError("pretrained_weights must be 'COCO_V1' or null")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs must be non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path = Path("data/downloaded-dataset")
    vision_dir: Path = Path("data/processed/vision")
    split_manifest: Path = Path(
        "data/processed/vision/splits/scene_split_seed42.json"
    )
    output_dir: Path = Path("experiments")
    training_sample_tokens_file: Path | None = None
    training_sample_limit: int | None = None
    validation_sample_limit: int | None = None
    image_size: tuple[int, int] | None = None
    batch_size: int = 1
    num_workers: int = 0
    seed: int = 42
    epochs: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    mixed_precision: bool = False
    reproducibility_mode: str = "strict"
    validation_interval: int = 1
    early_stopping_patience: int | None = None
    scheduler_step_size: int = 3
    scheduler_gamma: float = 0.1
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        for field_name in ("dataset_root", "vision_dir", "split_manifest", "output_dir"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if self.training_sample_tokens_file is not None:
            object.__setattr__(
                self, "training_sample_tokens_file", Path(self.training_sample_tokens_file)
            )
        if self.image_size is not None:
            if len(self.image_size) != 2 or any(value < 1 for value in self.image_size):
                raise ValueError("image_size must contain positive (height, width)")
            object.__setattr__(self, "image_size", tuple(self.image_size))
        if self.training_sample_limit is not None and self.training_sample_limit < 1:
            raise ValueError("training_sample_limit must be positive")
        if self.validation_sample_limit is not None and self.validation_sample_limit < 1:
            raise ValueError("validation_sample_limit must be positive")
        if self.training_sample_limit and self.training_sample_tokens_file:
            raise ValueError("Use either a training sample file or a sample limit, not both")
        if self.batch_size < 1 or self.num_workers < 0 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive; num_workers cannot be negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when configured")
        if self.validation_interval < 1 or self.scheduler_step_size < 1:
            raise ValueError("validation_interval and scheduler_step_size must be positive")
        if not 0 < self.scheduler_gamma <= 1:
            raise ValueError("scheduler_gamma must be in (0, 1]")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive when configured")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.reproducibility_mode not in ("strict", "best_effort"):
            raise ValueError("reproducibility_mode must be 'strict' or 'best_effort'")
        if not isinstance(self.model, ModelConfig):
            object.__setattr__(self, "model", ModelConfig(**self.model))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("dataset_root", "vision_dir", "split_manifest", "output_dir"):
            payload[key] = str(payload[key])
        if payload["training_sample_tokens_file"] is not None:
            payload["training_sample_tokens_file"] = str(payload["training_sample_tokens_file"])
        if payload["image_size"] is not None:
            payload["image_size"] = list(payload["image_size"])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        data = dict(payload)
        legacy_deterministic = data.pop("deterministic", None)
        if legacy_deterministic is not None and "reproducibility_mode" not in data:
            data["reproducibility_mode"] = "best_effort"
        model = data.pop("model", {})
        if not isinstance(model, dict):
            raise ValueError("model configuration must be a JSON object")
        data["model"] = ModelConfig(**model)
        if data.get("image_size") is not None:
            data["image_size"] = tuple(data["image_size"])
        expected = set(cls.__dataclass_fields__)
        unknown = set(data) - expected
        if unknown:
            raise ValueError(f"Unknown experiment configuration fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Experiment configuration must be a JSON object")
        return cls.from_dict(payload)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
