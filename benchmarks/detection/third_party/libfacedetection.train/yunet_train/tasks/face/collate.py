from __future__ import annotations

from typing import Any

import torch

from .types import FaceBatch, FaceSample


def collate_face_samples(samples: list[FaceSample]) -> FaceBatch:
    images = torch.stack([_as_image_tensor(sample.image) for sample in samples], dim=0)

    return FaceBatch(
        images=images,
        boxes=[_as_float_tensor(sample.boxes) for sample in samples],
        labels=[_as_long_tensor(sample.labels) for sample in samples],
        keypoints=[_as_float_tensor(sample.keypoints) for sample in samples],
        ignored_boxes=[_as_float_tensor(sample.ignored_boxes) for sample in samples],
        ignored_labels=[_as_long_tensor(sample.ignored_labels) for sample in samples],
        metas=[_meta_from_sample(sample) for sample in samples],
    )


def _as_image_tensor(image: Any) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("FaceSample.image must be a tensor before collation")
    return image.float()


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.long()
    return torch.as_tensor(value, dtype=torch.long)


def _meta_from_sample(sample: FaceSample) -> dict[str, Any]:
    return {
        "filename": sample.filename,
        "ori_shape": sample.original_shape,
        "img_shape": sample.image_shape,
        "pad_shape": sample.pad_shape,
        "scale_factor": sample.scale_factor,
        "flip": sample.flip,
        "flip_direction": sample.flip_direction,
        "img_norm_cfg": sample.image_norm,
    }

