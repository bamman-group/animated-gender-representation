from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch

from .config import COCO17_FLIP_IDX
from .types import PoseSample


class Compose:
    def __init__(self, transforms: Sequence[Callable[[PoseSample], PoseSample]]):
        self.transforms = tuple(transforms)

    def __call__(self, sample: PoseSample) -> PoseSample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Resize:
    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        keep_ratio: bool = False,
        clip_border: bool = True,
    ):
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.clip_border = clip_border

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        old_h, old_w = image.shape[:2]
        target_w, target_h = self.image_size
        if self.keep_ratio:
            scale = min(target_w / old_w, target_h / old_h)
            new_w = int(old_w * scale + 0.5)
            new_h = int(old_h * scale + 0.5)
        else:
            new_w, new_h = target_w, target_h

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        w_scale = new_w / old_w
        h_scale = new_h / old_h
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        sample.image = resized
        sample.image_shape = resized.shape
        sample.pad_shape = resized.shape
        sample.scale_factor = scale_factor
        sample.boxes = _scale_boxes(_ensure_numpy_array(sample.boxes), scale_factor, resized.shape, self.clip_border)
        sample.keypoints = _scale_keypoints(
            _ensure_numpy_array(sample.keypoints),
            w_scale,
            h_scale,
            resized.shape,
            self.clip_border,
        )
        return sample


class RandomSquareCrop:
    """Random square crop (face-style): keeps instances whose box center falls inside the crop."""

    def __init__(
        self,
        *,
        crop_ratio_range: tuple[float, float] | None = None,
        crop_choice: Sequence[float] | None = None,
        clip_border: bool = True,
        pad_value: float = 114,
    ):
        if (crop_ratio_range is None) == (crop_choice is None):
            raise ValueError("exactly one of crop_ratio_range or crop_choice must be set")
        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = tuple(crop_choice) if crop_choice is not None else None
        self.clip_border = clip_border
        self.pad_value = pad_value

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        boxes = _ensure_numpy_array(sample.boxes)
        if boxes.shape[0] == 0:
            return sample

        height, width = image.shape[:2]
        scale_retry = 0
        max_scale = self.crop_ratio_range[1] if self.crop_ratio_range is not None else max(self.crop_choice)
        scale = None

        while True:
            scale_retry += 1
            if scale_retry == 1 or max_scale > 1.0:
                if self.crop_ratio_range is not None:
                    scale = np.random.uniform(*self.crop_ratio_range)
                else:
                    scale = np.random.choice(self.crop_choice)
            else:
                scale = float(scale) * 1.2

            for _ in range(250):
                crop_w = int(scale * min(width, height))
                crop_h = crop_w

                if width == crop_w:
                    left = 0
                elif width > crop_w:
                    left = np.random.randint(0, width - crop_w)
                else:
                    left = np.random.randint(width - crop_w, 0)

                if height == crop_h:
                    top = 0
                elif height > crop_h:
                    top = np.random.randint(0, height - crop_h)
                else:
                    top = np.random.randint(height - crop_h, 0)

                patch = np.array((left, top, left + crop_w, top + crop_h), dtype=np.int64)
                keep = _centers_in_patch(boxes, patch)
                if not keep.any():
                    continue

                sample.boxes = _crop_boxes_after_patch(boxes, patch, keep, self.clip_border)
                sample.labels = _ensure_numpy_array(sample.labels)[keep]
                sample.keypoints = _crop_keypoints_after_patch(_ensure_numpy_array(sample.keypoints), patch, keep)

                sample.image = _crop_image_with_padding_pose(image, patch, crop_h, crop_w, self.pad_value)
                sample.image_shape = sample.image.shape
                sample.pad_shape = sample.image.shape
                return sample


class FilterSmallBoxes:
    def __init__(self, min_size: float):
        if min_size < 0:
            raise ValueError("min_size must be non-negative")
        self.min_size = min_size

    def __call__(self, sample: PoseSample) -> PoseSample:
        if self.min_size <= 0:
            return sample
        boxes = _ensure_numpy_array(sample.boxes)
        keep = _box_size_mask(boxes, self.min_size)
        sample.boxes = boxes[keep].reshape(-1, 4)
        sample.labels = _ensure_numpy_array(sample.labels)[keep]
        kpt = _ensure_numpy_array(sample.keypoints)
        num_kpt = kpt.shape[1] if kpt.ndim == 3 else sample.kpt_shape[0]
        last = kpt.shape[-1] if kpt.ndim == 3 else 3
        sample.keypoints = kpt[keep].reshape(-1, num_kpt, last)
        return sample


class RandomHorizontalFlip:
    def __init__(
        self,
        probability: float = 0.5,
        *,
        flip_idx: Sequence[int] = COCO17_FLIP_IDX,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("flip probability must be in [0, 1]")
        self.probability = probability
        self.flip_idx = tuple(flip_idx)

    def __call__(self, sample: PoseSample) -> PoseSample:
        if np.random.random() >= self.probability:
            sample.flip = False
            sample.flip_direction = None
            return sample

        image = _ensure_numpy_image(sample.image)
        width = image.shape[1]
        sample.image = np.flip(image, axis=1).copy()
        sample.boxes = _flip_boxes(_ensure_numpy_array(sample.boxes), width)
        sample.keypoints = _flip_keypoints(_ensure_numpy_array(sample.keypoints), width, self.flip_idx)
        sample.flip = True
        sample.flip_direction = "horizontal"
        return sample


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float], *, to_rgb: bool = False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image).astype(np.float32)
        if self.to_rgb:
            image = image[..., ::-1]
        sample.image = (image - self.mean) / self.std
        sample.image_norm = {
            "mean": self.mean,
            "std": self.std,
            "to_rgb": self.to_rgb,
        }
        return sample


class Pad:
    def __init__(
        self,
        *,
        size: tuple[int, int] | None = None,
        size_divisor: int | None = None,
        pad_value: int | float = 114,
    ):
        if (size is None) == (size_divisor is None):
            raise ValueError("exactly one of size or size_divisor must be set")
        self.size = size
        self.size_divisor = size_divisor
        self.pad_value = pad_value

    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        height, width = image.shape[:2]
        if self.size is not None:
            target_w, target_h = self.size
        else:
            assert self.size_divisor is not None
            target_h = int(np.ceil(height / self.size_divisor)) * self.size_divisor
            target_w = int(np.ceil(width / self.size_divisor)) * self.size_divisor

        if target_h < height or target_w < width:
            raise ValueError(f"pad target {(target_w, target_h)} is smaller than image {(width, height)}")

        padded = np.full((target_h, target_w, image.shape[2]), self.pad_value, dtype=image.dtype)
        padded[:height, :width] = image
        sample.image = padded
        sample.pad_shape = padded.shape
        return sample


class ToTensor:
    def __call__(self, sample: PoseSample) -> PoseSample:
        image = _ensure_numpy_image(sample.image)
        sample.image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        sample.boxes = torch.from_numpy(np.ascontiguousarray(sample.boxes)).float()
        sample.labels = torch.from_numpy(np.ascontiguousarray(sample.labels)).long()
        sample.keypoints = torch.from_numpy(np.ascontiguousarray(sample.keypoints)).float()
        return sample


def build_pose_train_transforms(
    image_size: int = 640,
    *,
    flip_idx: Sequence[int] = COCO17_FLIP_IDX,
    random_crop: bool = True,
    crop_choice: Sequence[float] = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    min_box_size: float = 10.0,
    crop_pad_value: float = 114,
) -> Compose:
    blocks: list[Callable[[PoseSample], PoseSample]] = []
    if random_crop:
        blocks.append(RandomSquareCrop(crop_choice=tuple(crop_choice), pad_value=crop_pad_value))
    blocks.extend(
        [
            Resize((image_size, image_size), keep_ratio=True),
            Pad(size=(image_size, image_size), pad_value=114),
            FilterSmallBoxes(min_size=min_box_size),
            RandomHorizontalFlip(0.5, flip_idx=flip_idx),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), to_rgb=False),
            ToTensor(),
        ]
    )
    return Compose(tuple(blocks))


def build_pose_eval_transforms(image_size: int = 640) -> Compose:
    return Compose(
        (
            Resize((image_size, image_size), keep_ratio=True),
            Pad(size=(image_size, image_size), pad_value=114),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), to_rgb=False),
            ToTensor(),
        )
    )


def _ensure_numpy_image(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        raise TypeError("image is already a tensor; ToTensor should be the final transform")
    return image


def _ensure_numpy_array(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def _scale_boxes(boxes: np.ndarray, scale_factor: np.ndarray, image_shape: tuple[int, int, int], clip: bool) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes *= scale_factor
    if clip:
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image_shape[1])
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image_shape[0])
    return boxes


def _scale_keypoints(
    keypoints: np.ndarray,
    w_scale: float,
    h_scale: float,
    image_shape: tuple[int, int, int],
    clip: bool,
) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)
    if keypoints.size == 0:
        return keypoints.reshape(0, keypoints.shape[1] if keypoints.ndim == 3 else 0, keypoints.shape[-1] if keypoints.ndim else 3)
    keypoints[..., 0] *= w_scale
    keypoints[..., 1] *= h_scale
    if clip:
        keypoints = _clip_keypoints(keypoints, image_shape[1], image_shape[0])
    return keypoints


def _clip_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    outside = (keypoints[..., 0] < 0) | (keypoints[..., 0] > width) | (keypoints[..., 1] < 0) | (keypoints[..., 1] > height)
    if keypoints.shape[-1] >= 3:
        keypoints[..., 2] = np.where(outside, 0, keypoints[..., 2])
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0, width)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0, height)
    return keypoints


def _flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    flipped = boxes.copy()
    flipped[..., 0::4] = width - boxes[..., 2::4]
    flipped[..., 2::4] = width - boxes[..., 0::4]
    return flipped


def _flip_keypoints(keypoints: np.ndarray, width: int, flip_idx: Sequence[int]) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)
    if keypoints.size == 0:
        return keypoints.reshape(0, len(flip_idx), keypoints.shape[-1] if keypoints.ndim else 3)
    if len(flip_idx) != keypoints.shape[1]:
        raise ValueError(f"flip_idx length {len(flip_idx)} does not match keypoints {keypoints.shape[1]}")
    flipped = keypoints[:, flip_idx, :].copy()
    flipped[..., 0] = width - flipped[..., 0]
    return flipped


def _centers_in_patch(boxes: np.ndarray, patch: np.ndarray) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return (
        (centers[:, 0] > patch[0])
        & (centers[:, 1] > patch[1])
        & (centers[:, 0] < patch[2])
        & (centers[:, 1] < patch[3])
    )


def _crop_boxes_after_patch(boxes: np.ndarray, patch: np.ndarray, keep: np.ndarray, clip: bool) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)[keep]
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if clip:
        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
    boxes -= np.tile(patch[:2], 2)
    return boxes


def _crop_keypoints_after_patch(
    keypoints: np.ndarray,
    patch: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)[keep]
    if keypoints.size == 0:
        nk = keypoints.shape[1] if keypoints.ndim >= 2 else 17
        return keypoints.reshape(0, nk, 3)
    pl, pt, pr, pb = float(patch[0]), float(patch[1]), float(patch[2]), float(patch[3])
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            x = float(keypoints[i, j, 0])
            y = float(keypoints[i, j, 1])
            vis = float(keypoints[i, j, 2])
            if vis <= 0:
                keypoints[i, j, 0] = 0.0
                keypoints[i, j, 1] = 0.0
                continue
            if x < pl or x > pr or y < pt or y > pb:
                keypoints[i, j, 0] = 0.0
                keypoints[i, j, 1] = 0.0
                keypoints[i, j, 2] = 0.0
            else:
                keypoints[i, j, 0] = x - pl
                keypoints[i, j, 1] = y - pt
    return keypoints


def _crop_image_with_padding_pose(
    image: np.ndarray,
    patch: np.ndarray,
    crop_h: int,
    crop_w: int,
    pad_value: float,
) -> np.ndarray:
    if image.dtype == np.uint8:
        fill: int | float = int(round(np.clip(pad_value, 0, 255)))
    else:
        fill = pad_value
    cropped = np.full((crop_h, crop_w, image.shape[2]), fill, dtype=image.dtype)
    patch_from = patch.copy()
    patch_from[0] = max(0, patch_from[0])
    patch_from[1] = max(0, patch_from[1])
    patch_from[2] = min(image.shape[1], patch_from[2])
    patch_from[3] = min(image.shape[0], patch_from[3])

    patch_to = patch.copy()
    patch_to[0] = max(0, patch_to[0] * -1)
    patch_to[1] = max(0, patch_to[1] * -1)
    patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
    patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])

    cropped[patch_to[1] : patch_to[3], patch_to[0] : patch_to[2], :] = image[
        patch_from[1] : patch_from[3],
        patch_from[0] : patch_from[2],
        :,
    ]
    return cropped


def _box_size_mask(boxes: np.ndarray, min_size: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return (widths >= min_size) & (heights >= min_size)
