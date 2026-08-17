from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import COCO8_POSE_ROOT

COCO17_FLIP_IDX: tuple[int, ...] = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
COCO17_OKS_SIGMA: np.ndarray = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float32,
    )
    / 10.0
)


@dataclass(frozen=True)
class PoseDatasetConfig:
    root: Path = COCO8_POSE_ROOT
    train_images: str = "images/train"
    val_images: str = "images/val"
    train_labels: str = "labels/train"
    val_labels: str = "labels/val"
    kpt_shape: tuple[int, int] = (17, 3)
    flip_idx: tuple[int, ...] = COCO17_FLIP_IDX
    names: dict[int, str] = field(default_factory=lambda: {0: "person"})
