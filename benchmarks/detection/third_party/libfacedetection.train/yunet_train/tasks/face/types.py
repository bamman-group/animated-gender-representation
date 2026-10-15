from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class FaceAnnotation:
    bbox: np.ndarray
    keypoints: np.ndarray
    ignore: bool
    label: int = 0


@dataclass(frozen=True)
class FaceRecord:
    filename: str
    width: int
    height: int
    annotations: tuple[FaceAnnotation, ...]

    @property
    def boxes(self) -> np.ndarray:
        boxes = [ann.bbox for ann in self.annotations if not ann.ignore]
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(boxes, ndmin=2, dtype=np.float32)

    @property
    def labels(self) -> np.ndarray:
        labels = [ann.label for ann in self.annotations if not ann.ignore]
        return np.array(labels, dtype=np.int64)

    @property
    def keypoints(self) -> np.ndarray:
        keypoints = [ann.keypoints for ann in self.annotations if not ann.ignore]
        if not keypoints:
            return np.zeros((0, 5, 3), dtype=np.float32)
        return np.array(keypoints, ndmin=3, dtype=np.float32)

    @property
    def ignored_boxes(self) -> np.ndarray:
        boxes = [ann.bbox for ann in self.annotations if ann.ignore]
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(boxes, ndmin=2, dtype=np.float32)

    @property
    def ignored_labels(self) -> np.ndarray:
        labels = [ann.label for ann in self.annotations if ann.ignore]
        return np.array(labels, dtype=np.int64)


@dataclass
class FaceSample:
    image: np.ndarray | torch.Tensor
    boxes: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor
    keypoints: np.ndarray | torch.Tensor
    ignored_boxes: np.ndarray | torch.Tensor
    ignored_labels: np.ndarray | torch.Tensor
    filename: str
    original_shape: tuple[int, int, int]
    image_shape: tuple[int, int, int]
    scale_factor: np.ndarray | None = None
    flip: bool = False
    flip_direction: str | None = None
    pad_shape: tuple[int, int, int] | None = None
    image_norm: dict[str, Any] | None = None


@dataclass(frozen=True)
class FaceBatch:
    images: torch.Tensor
    boxes: list[torch.Tensor]
    labels: list[torch.Tensor]
    keypoints: list[torch.Tensor]
    ignored_boxes: list[torch.Tensor]
    ignored_labels: list[torch.Tensor]
    metas: list[dict[str, Any]]
