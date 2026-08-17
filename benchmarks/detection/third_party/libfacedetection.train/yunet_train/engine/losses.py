from __future__ import annotations

import torch


def eiou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth_point: float = 0.1,
    eps: float = 1e-7,
    reduction: str = "none",
) -> torch.Tensor:
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    intersection = (
        (ix2 - ex1) * (iy2 - ey1)
        + (xmin - ex1) * (ymin - ey1)
        - (ix1 - ex1) * (ymax - ey1)
        - (xmax - ex1) * (iy1 - ey1)
    )
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - intersection + eps
    ious = 1 - (intersection / union)

    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point
    )
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def bbox_overlaps(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if bboxes1.numel() == 0 or bboxes2.numel() == 0:
        return bboxes1.new_zeros((bboxes1.size(0), bboxes2.size(0)))

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, :, 0] * wh[:, :, 1]

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2 - overlap
    return overlap / (union + eps)

