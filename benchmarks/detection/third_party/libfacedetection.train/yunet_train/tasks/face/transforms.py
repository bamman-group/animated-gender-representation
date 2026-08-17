from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
import torch

from .types import FaceSample


class Compose:
    def __init__(self, transforms: Sequence[Callable[[FaceSample], FaceSample]]):
        self.transforms = tuple(transforms)

    def __call__(self, sample: FaceSample) -> FaceSample:
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

    def __call__(self, sample: FaceSample) -> FaceSample:
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
        sample.ignored_boxes = _scale_boxes(
            _ensure_numpy_array(sample.ignored_boxes),
            scale_factor,
            resized.shape,
            self.clip_border,
        )
        sample.keypoints = _scale_keypoints(
            _ensure_numpy_array(sample.keypoints),
            w_scale,
            h_scale,
            resized.shape,
            self.clip_border,
        )
        return sample


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("flip probability must be in [0, 1]")
        self.probability = probability

    def __call__(self, sample: FaceSample) -> FaceSample:
        if np.random.random() >= self.probability:
            sample.flip = False
            sample.flip_direction = None
            return sample

        image = _ensure_numpy_image(sample.image)
        width = image.shape[1]
        sample.image = np.flip(image, axis=1).copy()
        sample.boxes = _flip_boxes(_ensure_numpy_array(sample.boxes), width)
        sample.ignored_boxes = _flip_boxes(_ensure_numpy_array(sample.ignored_boxes), width)
        sample.keypoints = _flip_keypoints(_ensure_numpy_array(sample.keypoints), width)
        sample.flip = True
        sample.flip_direction = "horizontal"
        return sample


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float], *, to_rgb: bool = False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, sample: FaceSample) -> FaceSample:
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
        pad_value: int | float = 0,
    ):
        if (size is None) == (size_divisor is None):
            raise ValueError("exactly one of size or size_divisor must be set")
        self.size = size
        self.size_divisor = size_divisor
        self.pad_value = pad_value

    def __call__(self, sample: FaceSample) -> FaceSample:
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
    def __call__(self, sample: FaceSample) -> FaceSample:
        image = _ensure_numpy_image(sample.image)
        sample.image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sample.boxes = torch.from_numpy(np.ascontiguousarray(sample.boxes)).float()
        sample.labels = torch.from_numpy(np.ascontiguousarray(sample.labels)).long()
        sample.keypoints = torch.from_numpy(np.ascontiguousarray(sample.keypoints)).float()
        sample.ignored_boxes = torch.from_numpy(np.ascontiguousarray(sample.ignored_boxes)).float()
        sample.ignored_labels = torch.from_numpy(np.ascontiguousarray(sample.ignored_labels)).long()
        return sample


class FilterSmallBoxes:
    def __init__(self, min_size: float):
        if min_size < 0:
            raise ValueError("min_size must be non-negative")
        self.min_size = min_size

    def __call__(self, sample: FaceSample) -> FaceSample:
        if self.min_size <= 0:
            return sample

        boxes = _ensure_numpy_array(sample.boxes)
        keep = _box_size_mask(boxes, self.min_size)
        sample.boxes = boxes[keep].reshape(-1, 4)
        sample.labels = _ensure_numpy_array(sample.labels)[keep]
        sample.keypoints = _ensure_numpy_array(sample.keypoints)[keep].reshape(-1, 5, 3)

        ignored_boxes = _ensure_numpy_array(sample.ignored_boxes)
        ignored_keep = _box_size_mask(ignored_boxes, self.min_size)
        sample.ignored_boxes = ignored_boxes[ignored_keep].reshape(-1, 4)
        sample.ignored_labels = _ensure_numpy_array(sample.ignored_labels)[ignored_keep]
        return sample


class RandomSquareCrop:
    def __init__(
        self,
        *,
        crop_ratio_range: tuple[float, float] | None = None,
        crop_choice: Sequence[float] | None = None,
        clip_border: bool = True,
    ):
        if (crop_ratio_range is None) == (crop_choice is None):
            raise ValueError("exactly one of crop_ratio_range or crop_choice must be set")
        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = tuple(crop_choice) if crop_choice is not None else None
        self.clip_border = clip_border

    def __call__(self, sample: FaceSample) -> FaceSample:
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
                scale = scale * 1.2

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

                sample.boxes = _crop_boxes(boxes, patch, keep, self.clip_border)
                sample.labels = _ensure_numpy_array(sample.labels)[keep]
                sample.keypoints = _crop_keypoints(_ensure_numpy_array(sample.keypoints), patch, keep, self.clip_border)

                ignored_boxes = _ensure_numpy_array(sample.ignored_boxes)
                ignored_keep = _centers_in_patch(ignored_boxes, patch)
                sample.ignored_boxes = _crop_boxes(ignored_boxes, patch, ignored_keep, self.clip_border)
                sample.ignored_labels = _ensure_numpy_array(sample.ignored_labels)[ignored_keep]

                sample.image = _crop_image_with_padding(image, patch, crop_h, crop_w)
                sample.image_shape = sample.image.shape
                return sample


def build_train_transforms(
    image_size: int = 640,
    crop_choice: Sequence[float] = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    min_box_size: float = 10.0,
) -> Compose:
    return Compose(
        (
            RandomSquareCrop(crop_choice=crop_choice),
            Resize((image_size, image_size), keep_ratio=False),
            FilterSmallBoxes(min_size=min_box_size),
            RandomHorizontalFlip(0.5),
            Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), to_rgb=False),
            ToTensor(),
        )
    )


def build_eval_transforms(image_size: int | None = 640, *, size_divisor: int | None = None) -> Compose:
    if image_size is None:
        if size_divisor is None:
            raise ValueError("size_divisor must be set when image_size is None")
        return Compose(
            (
                Pad(size_divisor=size_divisor, pad_value=0),
                Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), to_rgb=False),
                ToTensor(),
            )
        )
    return Compose(
        (
            Resize((image_size, image_size), keep_ratio=True),
            Pad(size=(image_size, image_size), pad_value=0),
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
        return keypoints.reshape(0, 5, 3)
    keypoints[:, :, 0] *= w_scale
    keypoints[:, :, 1] *= h_scale
    if clip:
        keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, image_shape[1])
        keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, image_shape[0])
    return keypoints


def _box_size_mask(boxes: np.ndarray, min_size: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return (widths >= min_size) & (heights >= min_size)


def _flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    flipped = boxes.copy()
    flipped[..., 0::4] = width - boxes[..., 2::4]
    flipped[..., 2::4] = width - boxes[..., 0::4]
    return flipped


def _flip_keypoints(keypoints: np.ndarray, width: int) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)
    if keypoints.size == 0:
        return keypoints.reshape(0, 5, 3)
    flip_order = [1, 0, 2, 4, 3]
    flipped = keypoints[:, flip_order, :].copy()
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


def _crop_boxes(boxes: np.ndarray, patch: np.ndarray, keep: np.ndarray, clip: bool) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)[keep]
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if clip:
        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
    boxes -= np.tile(patch[:2], 2)
    return boxes


def _crop_keypoints(keypoints: np.ndarray, patch: np.ndarray, keep: np.ndarray, clip: bool) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=True)[keep]
    if keypoints.size == 0:
        return keypoints.reshape(0, 5, 3)
    if clip:
        keypoints[:, :, :2] = keypoints[:, :, :2].clip(max=patch[2:])
        keypoints[:, :, :2] = keypoints[:, :, :2].clip(min=patch[:2])
    keypoints[:, :, 0] -= patch[0]
    keypoints[:, :, 1] -= patch[1]
    return keypoints


def _crop_image_with_padding(image: np.ndarray, patch: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    cropped = np.ones((crop_h, crop_w, image.shape[2]), dtype=image.dtype) * 128
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
