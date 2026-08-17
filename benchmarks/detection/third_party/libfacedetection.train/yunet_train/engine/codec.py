from __future__ import annotations

import torch


def bbox_decode(priors: torch.Tensor, bbox_preds: torch.Tensor) -> torch.Tensor:
    xys = (bbox_preds[..., :2] * priors[..., 2:]) + priors[..., :2]
    whs = bbox_preds[..., 2:].exp() * priors[..., 2:]

    tl_x = xys[..., 0] - whs[..., 0] / 2
    tl_y = xys[..., 1] - whs[..., 1] / 2
    br_x = xys[..., 0] + whs[..., 0] / 2
    br_y = xys[..., 1] + whs[..., 1] / 2

    return torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)


def kps_decode(priors: torch.Tensor, kps_preds: torch.Tensor) -> torch.Tensor:
    num_points = kps_preds.shape[-1] // 2
    return torch.cat(
        [
            (kps_preds[..., [2 * i, 2 * i + 1]] * priors[..., 2:]) + priors[..., :2]
            for i in range(num_points)
        ],
        dim=-1,
    )


def kps_encode(priors: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    num_points = keypoints.shape[-1] // 2
    return torch.cat(
        [
            (keypoints[..., [2 * i, 2 * i + 1]] - priors[..., :2]) / priors[..., 2:]
            for i in range(num_points)
        ],
        dim=-1,
    )
