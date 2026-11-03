"""Model adapters and construction for supported detection architectures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from scenarioforge.training.config import ModelConfig


@runtime_checkable
class DetectionModelAdapter(Protocol):
    """Small contract separating detector-specific calls from experiment logic."""

    architecture: str
    model_label_to_source_id: Mapping[int, int]

    def parameters(self, recurse: bool = True): ...

    def train(self, mode: bool = True): ...

    def eval(self): ...

    def to(self, device: torch.device): ...

    def state_dict(self): ...

    def load_state_dict(self, state_dict, strict: bool = True): ...

    def compute_loss(
        self, images: Sequence[Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]: ...

    def predict(self, images: Sequence[Tensor]) -> list[dict[str, Tensor]]: ...


class FasterRCNNAdapter(nn.Module):
    """Torchvision Faster R-CNN wrapped behind the common detector contract."""

    architecture = "fasterrcnn_mobilenet_v3_large_320_fpn"

    def __init__(
        self,
        detector: nn.Module,
        model_label_to_source_id: Mapping[int, int],
        *,
        pretrained_weights: str | None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.model_label_to_source_id = dict(model_label_to_source_id)
        self.pretrained_weights = pretrained_weights
        self._backbone_trainable = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self._backbone_trainable:
            for module in self.detector.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(
        self,
        images: Sequence[Tensor],
        targets: Sequence[dict[str, Tensor]] | None = None,
    ):
        return self.detector(images, targets)

    def compute_loss(
        self, images: Sequence[Tensor], targets: Sequence[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        losses = self.detector(list(images), list(targets))
        validate_loss_dictionary(losses)
        return losses

    @torch.inference_mode()
    def predict(self, images: Sequence[Tensor]) -> list[dict[str, Tensor]]:
        return self.detector(list(images))

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_trainable = trainable
        for parameter in self.detector.backbone.parameters():
            parameter.requires_grad_(trainable)
        for module in self.detector.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if not trainable:
                    module.eval()
                elif self.training:
                    module.train()


def create_detection_model(
    config: ModelConfig,
    model_label_to_source_id: Mapping[int, int],
) -> FasterRCNNAdapter:
    """Construct Faster R-CNN and replace only its class-specific prediction head."""
    if not model_label_to_source_id:
        raise ValueError("At least one foreground class mapping is required")
    labels = sorted(int(label) for label in model_label_to_source_id)
    if labels != list(range(1, len(labels) + 1)):
        raise ValueError("Model labels must be contiguous foreground IDs starting at 1")
    if len(set(model_label_to_source_id.values())) != len(labels):
        raise ValueError("Source category IDs in the model mapping must be unique")

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = None
    if config.pretrained_weights == "COCO_V1":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    arguments = {
        "weights": weights,
        "weights_backbone": None,
        "num_classes": None if weights is not None else len(labels) + 1,
    }
    if weights is not None:
        arguments["trainable_backbone_layers"] = 6
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(**arguments)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(labels) + 1)
    adapter = FasterRCNNAdapter(
        detector,
        model_label_to_source_id,
        pretrained_weights=config.pretrained_weights,
    )
    validate_model_class_count(adapter, len(labels))
    return adapter


def validate_model_class_count(model: FasterRCNNAdapter, foreground_classes: int) -> None:
    """Check that the detector head includes exactly one background class."""
    actual = model.detector.roi_heads.box_predictor.cls_score.out_features
    expected = foreground_classes + 1
    if actual != expected:
        raise ValueError(f"Detector predicts {actual} classes; expected {expected}")


def validate_loss_dictionary(losses: object) -> None:
    """Reject malformed or non-finite detection loss dictionaries."""
    if not isinstance(losses, dict) or not losses:
        raise ValueError("Detection model must return a non-empty loss dictionary")
    for name, value in losses.items():
        if not isinstance(name, str) or not isinstance(value, Tensor) or value.ndim != 0:
            raise ValueError("Every detection loss must be a named scalar tensor")
        if not torch.isfinite(value).item():
            raise ValueError(f"Detection loss {name!r} is not finite")


def validate_predictions(
    predictions: object, *, foreground_classes: int
) -> list[dict[str, Tensor]]:
    """Validate the standard torchvision detection output shape and label range."""
    if not isinstance(predictions, list):
        raise ValueError("Detector predictions must be a list of dictionaries")
    for prediction in predictions:
        if not isinstance(prediction, dict) or not {"boxes", "labels", "scores"} <= prediction.keys():
            raise ValueError("Each prediction must contain boxes, labels, and scores")
        boxes, labels, scores = (
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
        )
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Predicted boxes must have shape [N, 4]")
        if labels.dtype != torch.int64 or labels.shape != (len(boxes),):
            raise ValueError("Predicted labels must be int64 with shape [N]")
        if scores.ndim != 1 or scores.shape != (len(boxes),):
            raise ValueError("Prediction scores must have shape [N]")
        if boxes.numel() and not torch.isfinite(boxes).all().item():
            raise ValueError("Predicted boxes contain non-finite values")
        if scores.numel() and not torch.isfinite(scores).all().item():
            raise ValueError("Prediction scores contain non-finite values")
        if labels.numel() and (
            torch.any(labels < 1).item() or torch.any(labels > foreground_classes).item()
        ):
            raise ValueError("Predicted labels include background or an unknown foreground class")
    return predictions
