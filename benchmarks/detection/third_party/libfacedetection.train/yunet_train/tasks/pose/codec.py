from __future__ import annotations

import torch


def pose_keypoints_decode(
    priors: torch.Tensor,
    keypoint_preds: torch.Tensor,
    *,
    kpt_shape: tuple[int, int] = (17, 3),
) -> torch.Tensor:
    kpt_num, kpt_dim = kpt_shape
    keypoints = keypoint_preds.reshape(*keypoint_preds.shape[:-1], kpt_num, kpt_dim).clone()
    keypoints[..., :2] = keypoints[..., :2] * priors[..., None, 2:] + priors[..., None, :2]
    return keypoints


def pose_keypoints_encode(
    priors: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    kpt_shape: tuple[int, int] = (17, 3),
) -> torch.Tensor:
    kpt_num, kpt_dim = kpt_shape
    if keypoints.shape[-2:] != (kpt_num, kpt_dim):
        raise ValueError(f"Expected keypoints shape (..., {kpt_num}, {kpt_dim}), got {tuple(keypoints.shape)}")
    encoded = keypoints.clone()
    encoded[..., :2] = (encoded[..., :2] - priors[..., None, :2]) / priors[..., None, 2:]
    return encoded.reshape(*keypoints.shape[:-2], kpt_num * kpt_dim)
