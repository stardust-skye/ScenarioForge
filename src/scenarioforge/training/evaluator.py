"""TorchMetrics COCO-style evaluation and fixed-threshold precision/recall."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torchvision.ops import box_iou

from scenarioforge.training.model_factory import (
    DetectionModelAdapter,
    validate_predictions,
)


class DetectionEvaluator:
    """Evaluate detection outputs using TorchMetrics with Faster-COCO-Eval."""

    def __init__(
        self,
        *,
        model_label_to_name: Mapping[int, str],
        score_threshold: float = 0.5,
    ) -> None:
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        self.model_label_to_name = dict(model_label_to_name)
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def evaluate(
        self,
        model: DetectionModelAdapter,
        data_loader,
        *,
        device: torch.device,
        split_name: str,
    ) -> dict[str, Any]:
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ImportError as exc:
            raise RuntimeError(
                "Object-detection metrics are unavailable; install requirements-ml.txt "
                "to use TorchMetrics with Faster-COCO-Eval."
            ) from exc

        metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True,
            extended_summary=False,
            backend="faster_coco_eval",
        )
        previous_training = getattr(model, "training", False)
        model.eval()
        class_counts: dict[int, dict[str, int]] = {
            label: {
                "ground_truth": 0,
                "predictions": 0,
                "predictions_at_score_threshold": 0,
            }
            for label in self.model_label_to_name
        }
        precision_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
        image_count = 0
        try:
            for images, targets in data_loader:
                images_on_device = [
                    image.to(device, non_blocking=device.type == "cuda") for image in images
                ]
                predictions = validate_predictions(
                    model.predict(images_on_device),
                    foreground_classes=len(self.model_label_to_name),
                )
                if len(predictions) != len(targets):
                    raise ValueError("Detector returned a different number of predictions than inputs")
                metric.update(
                    [_to_metric_prediction(item) for item in predictions],
                    [_to_metric_target(item) for item in targets],
                )
                for prediction, target in zip(predictions, targets):
                    image_count += 1
                    _count_class_objects(
                        prediction,
                        target,
                        class_counts,
                        precision_counts,
                        score_threshold=self.score_threshold,
                    )
            if image_count == 0:
                raise ValueError(f"Cannot evaluate an empty {split_name} data loader")
            result = metric.compute()
        finally:
            if previous_training:
                model.train()
        return _metric_payload(
            result,
            class_counts,
            precision_counts,
            self.model_label_to_name,
            image_count=image_count,
            split_name=split_name,
            score_threshold=self.score_threshold,
        )


def _to_metric_prediction(prediction: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        "boxes": prediction["boxes"].detach().cpu(),
        "scores": prediction["scores"].detach().cpu(),
        "labels": prediction["labels"].detach().cpu() - 1,
    }


def _to_metric_target(target: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu() - 1,
        "area": target["area"].detach().cpu(),
        "iscrowd": target["iscrowd"].detach().cpu(),
    }


def _count_class_objects(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    class_counts: dict[int, dict[str, int]],
    precision_counts: dict[int, list[int]],
    *,
    score_threshold: float,
) -> None:
    for label, counts in class_counts.items():
        pred_mask = (prediction["labels"] == label) & (
            prediction["scores"] >= score_threshold
        )
        all_predicted = prediction["labels"] == label
        gt_mask = target["labels"] == label
        predicted_boxes = prediction["boxes"][pred_mask].detach().cpu()
        predicted_scores = prediction["scores"][pred_mask].detach().cpu()
        ground_truth_boxes = target["boxes"][gt_mask].detach().cpu()
        counts["ground_truth"] += len(ground_truth_boxes)
        counts["predictions"] += int(all_predicted.sum().item())
        counts["predictions_at_score_threshold"] += len(predicted_boxes)
        tp, fp, fn = _greedy_matches(predicted_boxes, predicted_scores, ground_truth_boxes)
        totals = precision_counts[label]
        totals[0] += tp
        totals[1] += fp
        totals[2] += fn


def _greedy_matches(
    predicted_boxes: Tensor,
    scores: Tensor,
    ground_truth_boxes: Tensor,
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    if not len(predicted_boxes):
        return 0, 0, len(ground_truth_boxes)
    if not len(ground_truth_boxes):
        return 0, len(predicted_boxes), 0
    overlaps = box_iou(predicted_boxes, ground_truth_boxes)
    matched: set[int] = set()
    true_positives = 0
    for pred_index in torch.argsort(scores, descending=True).tolist():
        unmatched = [index for index in range(len(ground_truth_boxes)) if index not in matched]
        if not unmatched:
            break
        available_ious = overlaps[pred_index, unmatched]
        best_iou, best_offset = available_ious.max(dim=0)
        best_gt = unmatched[int(best_offset.item())]
        if best_iou.item() >= iou_threshold:
            matched.add(best_gt)
            true_positives += 1
    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(ground_truth_boxes) - len(matched)
    return true_positives, false_positives, false_negatives


def _metric_payload(
    result: dict[str, Tensor],
    class_counts: dict[int, dict[str, int]],
    precision_counts: dict[int, list[int]],
    model_label_to_name: Mapping[int, str],
    *,
    image_count: int,
    split_name: str,
    score_threshold: float,
) -> dict[str, Any]:
    classes = [
        int(value) for value in torch.as_tensor(result["classes"]).reshape(-1).tolist()
    ]
    ap_by_class = (
        torch.as_tensor(result["map_per_class"]).reshape(-1).tolist()
    )
    per_class: dict[str, Any] = {}
    map_values: list[float] = []
    ap50 = _scalar_metric(result["map_50"])
    ap75 = _scalar_metric(result["map_75"])
    for label, name in model_label_to_name.items():
        metric_class = label - 1
        count = class_counts[label]
        ap_value: float | None = None
        if count["ground_truth"] > 0 and metric_class in classes:
            ap_value = _optional_metric(ap_by_class[classes.index(metric_class)])
            if ap_value is not None:
                map_values.append(ap_value)
        tp, fp, fn = precision_counts[label]
        per_class[name] = {
            "average_precision": ap_value,
            "ground_truth_objects": count["ground_truth"],
            "predictions": count["predictions"],
            "predictions_at_score_threshold": count["predictions_at_score_threshold"],
            "precision_iou50": _ratio(tp, tp + fp),
            "recall_iou50": _ratio(tp, tp + fn),
        }
    tp_total = sum(values[0] for values in precision_counts.values())
    fp_total = sum(values[1] for values in precision_counts.values())
    fn_total = sum(values[2] for values in precision_counts.values())
    return {
        "split": split_name,
        "image_count": image_count,
        "metric_implementation": "torchmetrics.detection.MeanAveragePrecision/faster_coco_eval",
        "iou_thresholds": [round(0.5 + 0.05 * index, 2) for index in range(10)],
        "map_50_95": _scalar_metric(result["map"]),
        "map_50": ap50,
        "map_75": ap75,
        "per_class": per_class,
        "precision_iou50_score_threshold": score_threshold,
        "precision_iou50": _ratio(tp_total, tp_total + fp_total),
        "recall_iou50": _ratio(tp_total, tp_total + fn_total),
    }


def _scalar_metric(value: Tensor) -> float | None:
    return _optional_metric(float(value.detach().cpu().item()))


def _optional_metric(value: float) -> float | None:
    if value < 0 or not torch.isfinite(torch.tensor(value)).item():
        return None
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
