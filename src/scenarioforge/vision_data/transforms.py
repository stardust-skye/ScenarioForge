"""Synchronized, lightweight image and XYXY box transforms."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


class ComposeDetectionTransforms:
    """Apply transforms that each receive and return an image/target pair."""

    def __init__(self, transforms: Sequence[Callable[[Any, dict[str, Tensor]], tuple[Any, dict[str, Tensor]]]]):
        self.transforms = tuple(transforms)

    def __call__(self, image: Any, target: dict[str, Tensor]) -> tuple[Any, dict[str, Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Resize:
    """Resize a PIL image to (height, width) and scale XYXY boxes identically."""

    def __init__(self, size: tuple[int, int]) -> None:
        if len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
            raise ValueError("size must be a pair of positive integers (height, width)")
        self.height, self.width = size

    def __call__(self, image: Image.Image, target: dict[str, Tensor]) -> tuple[Image.Image, dict[str, Tensor]]:
        old_width, old_height = image.size
        if old_width <= 0 or old_height <= 0:
            raise ValueError("Cannot resize an image with invalid dimensions")
        resized = image.resize((self.width, self.height), resample=Image.Resampling.BILINEAR)
        updated = dict(target)
        boxes = target["boxes"].clone()
        boxes[:, (0, 2)] *= self.width / old_width
        boxes[:, (1, 3)] *= self.height / old_height
        updated["boxes"] = boxes
        updated["area"] = _box_areas(boxes)
        return resized, updated


class RandomHorizontalFlip:
    """Optionally flip an image and its boxes together; disabled by default."""

    def __init__(self, probability: float = 0.0) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability

    def __call__(self, image: Image.Image, target: dict[str, Tensor]) -> tuple[Image.Image, dict[str, Tensor]]:
        if random.random() >= self.probability:
            return image, target
        width, _ = image.size
        updated = dict(target)
        boxes = target["boxes"].clone()
        old_x_min = boxes[:, 0].clone()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - old_x_min
        updated["boxes"] = boxes
        updated["area"] = _box_areas(boxes)
        return ImageOps.mirror(image), updated


class ToFloatTensor:
    """Convert RGB PIL pixels to float32 CHW tensors in [0, 1], without normalization."""

    def __call__(self, image: Image.Image, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().to(dtype=torch.float32)
        return tensor.div_(255.0), target


class Normalize:
    """Apply explicit per-channel normalization to a float image tensor."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError("mean and std must have three channels and positive std values")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Tensor, target: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(image, Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Normalize expects a [3, H, W] tensor")
        return (image - self.mean.to(image.device)) / self.std.to(image.device), target


def build_train_transforms(
    size: tuple[int, int] | None = None,
    *,
    horizontal_flip_probability: float = 0.0,
) -> ComposeDetectionTransforms:
    """Build training transforms; semantic-changing flips remain opt-in."""
    transforms: list[Callable[[Any, dict[str, Tensor]], tuple[Any, dict[str, Tensor]]]] = []
    if size is not None:
        transforms.append(Resize(size))
    if horizontal_flip_probability:
        transforms.append(RandomHorizontalFlip(horizontal_flip_probability))
    transforms.append(ToFloatTensor())
    return ComposeDetectionTransforms(transforms)


def build_eval_transforms(
    size: tuple[int, int] | None = None,
) -> ComposeDetectionTransforms:
    """Build deterministic evaluation transforms."""
    transforms: list[Callable[[Any, dict[str, Tensor]], tuple[Any, dict[str, Tensor]]]] = []
    if size is not None:
        transforms.append(Resize(size))
    transforms.append(ToFloatTensor())
    return ComposeDetectionTransforms(transforms)


def _box_areas(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=boxes.device)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
