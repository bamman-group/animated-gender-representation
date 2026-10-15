from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from .types import PoseSample

COCO17_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def render_pose_sample(
    sample: PoseSample,
    *,
    skeleton: Sequence[tuple[int, int]] = COCO17_SKELETON,
    keypoint_radius: int = 3,
) -> np.ndarray:
    image = _image_to_uint8_bgr(sample.image).copy()
    boxes = _as_numpy(sample.boxes).reshape(-1, 4)
    labels = _as_numpy(sample.labels).reshape(-1)
    keypoints = _as_numpy(sample.keypoints).reshape(-1, sample.kpt_shape[0], sample.kpt_shape[1])

    for idx, box in enumerate(boxes):
        color = _color_for_index(idx)
        x1, y1, x2, y2 = np.round(box).astype(int).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = str(int(labels[idx])) if idx < labels.shape[0] else "person"
        cv2.putText(image, label, (x1, max(y1 - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        kpts = keypoints[idx]
        for start, end in skeleton:
            if _visible(kpts[start]) and _visible(kpts[end]):
                p1 = tuple(np.round(kpts[start, :2]).astype(int).tolist())
                p2 = tuple(np.round(kpts[end, :2]).astype(int).tolist())
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        for point_idx, point in enumerate(kpts):
            if _visible(point):
                point_color = (0, 255, 255) if point_idx == 0 else color
                center = tuple(np.round(point[:2]).astype(int).tolist())
                cv2.circle(image, center, keypoint_radius, point_color, -1, cv2.LINE_AA)
    return image


def pose_sample_annotation_text(sample: PoseSample) -> str:
    boxes = _as_numpy(sample.boxes).reshape(-1, 4)
    labels = _as_numpy(sample.labels).reshape(-1)
    keypoints = _as_numpy(sample.keypoints).reshape(-1, sample.kpt_shape[0], sample.kpt_shape[1])
    lines = [
        f"filename: {sample.filename}",
        f"original_shape: {sample.original_shape}",
        f"image_shape: {sample.image_shape}",
        f"pad_shape: {sample.pad_shape}",
        f"scale_factor: {sample.scale_factor}",
        f"flip: {sample.flip}",
        f"flip_direction: {sample.flip_direction}",
        f"objects: {boxes.shape[0]}",
    ]
    for idx, box in enumerate(boxes):
        visible = int((keypoints[idx, :, 2] > 0).sum()) if sample.kpt_shape[1] >= 3 else sample.kpt_shape[0]
        label = int(labels[idx]) if idx < labels.shape[0] else -1
        lines.append(f"[{idx}] label={label} box={box.tolist()} visible_keypoints={visible}")
        lines.append(f"[{idx}] keypoints={keypoints[idx].tolist()}")
    return "\n".join(lines) + "\n"


def _image_to_uint8_bgr(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
        if array.ndim == 3 and array.shape[0] in {1, 3}:
            array = array.transpose(1, 2, 0)
    else:
        array = image
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(array)


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _visible(point: np.ndarray) -> bool:
    return point.shape[0] < 3 or point[2] > 0


def _color_for_index(index: int) -> tuple[int, int, int]:
    palette = (
        (40, 220, 40),
        (60, 180, 255),
        (255, 140, 40),
        (220, 80, 220),
        (80, 220, 220),
    )
    return palette[index % len(palette)]
