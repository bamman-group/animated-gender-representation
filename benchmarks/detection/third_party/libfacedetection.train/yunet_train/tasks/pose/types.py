from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PoseRecord:
    image_path: Path
    label_path: Path
    filename: str


@dataclass
class PoseSample:
    image: np.ndarray | torch.Tensor
    boxes: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor
    keypoints: np.ndarray | torch.Tensor
    filename: str
    original_shape: tuple[int, int, int]
    image_shape: tuple[int, int, int]
    kpt_shape: tuple[int, int]
    scale_factor: np.ndarray | None = None
    flip: bool = False
    flip_direction: str | None = None
    pad_shape: tuple[int, int, int] | None = None
    image_norm: dict[str, Any] | None = None


@dataclass(frozen=True)
class PoseBatch:
    images: torch.Tensor
    boxes: list[torch.Tensor]
    labels: list[torch.Tensor]
    keypoints: list[torch.Tensor]
    metas: list[dict[str, Any]]
