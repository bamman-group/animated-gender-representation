from __future__ import annotations

import torch


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep_indices = []
    for label in torch.unique(labels):
        label_indices = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        label_keep = nms(boxes[label_indices], scores[label_indices], iou_threshold)
        keep_indices.append(label_indices[label_keep])

    keep = torch.cat(keep_indices, dim=0)
    _, order = scores[keep].sort(descending=True)
    return keep[order]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        ious = inter / union.clamp(min=torch.finfo(boxes.dtype).eps)
        order = rest[ious <= iou_threshold]

    return torch.stack(keep)
