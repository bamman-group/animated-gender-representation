from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import COCO17_OKS_SIGMA


def oks_keypoint_loss(
    pred_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    target_areas: torch.Tensor,
    *,
    sigmas: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    if pred_keypoints.shape[:2] != target_keypoints.shape[:2]:
        raise ValueError("pred_keypoints and target_keypoints must share the first two dimensions")
    if target_keypoints.shape[-1] < 3:
        raise ValueError("target_keypoints must include visibility in the last dimension")

    if sigmas is None:
        sigmas = torch.as_tensor(COCO17_OKS_SIGMA, dtype=pred_keypoints.dtype, device=pred_keypoints.device)
    else:
        sigmas = sigmas.to(dtype=pred_keypoints.dtype, device=pred_keypoints.device)

    visible = target_keypoints[..., 2] > 0
    squared_distance = (pred_keypoints[..., :2] - target_keypoints[..., :2]).square().sum(dim=-1)
    area = target_areas.to(dtype=pred_keypoints.dtype, device=pred_keypoints.device).clamp_min(eps).unsqueeze(-1)
    normalized_error = squared_distance / (((2.0 * sigmas).square() * area) * 2.0 + eps)
    visible_count = visible.sum(dim=1).clamp_min(1).to(dtype=pred_keypoints.dtype)
    instance_weight = pred_keypoints.shape[1] / visible_count
    loss = (1.0 - torch.exp(-normalized_error)) * visible.to(dtype=pred_keypoints.dtype)
    return (loss * instance_weight.unsqueeze(-1)).mean()


def keypoint_visibility_loss(
    pred_visibility_logits: torch.Tensor,
    target_keypoints: torch.Tensor,
) -> torch.Tensor:
    if target_keypoints.shape[-1] < 3:
        raise ValueError("target_keypoints must include visibility in the last dimension")

    target = (target_keypoints[..., 2] > 0).to(dtype=pred_visibility_logits.dtype)
    return F.binary_cross_entropy_with_logits(pred_visibility_logits, target)
