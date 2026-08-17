from __future__ import annotations

from dataclasses import dataclass

import torch

from yunet_train.engine.nms import batched_nms
from yunet_train.engine.priors import MlvlPointGenerator

from .codec import bbox_decode, kps_decode


@dataclass(frozen=True)
class DetectionResult:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    keypoints: torch.Tensor


class YuNetPostprocessor:
    def __init__(
        self,
        *,
        strides: tuple[int, ...] = (8, 16, 32),
        score_threshold: float = 0.02,
        nms_threshold: float = 0.45,
        max_detections: int = -1,
    ):
        self.prior_generator = MlvlPointGenerator(strides=strides, offset=0)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections

    @torch.no_grad()
    def __call__(
        self,
        preds: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
    ) -> list[DetectionResult]:
        cls_scores, bbox_preds, objectnesses, kps_preds = preds
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [tuple(cls_score.shape[2:]) for cls_score in cls_scores]
        priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )
        flatten_priors = torch.cat(priors, dim=0)
        flatten_cls_scores = _flatten_preds(cls_scores, num_imgs, cls_scores[0].shape[1]).sigmoid()
        flatten_bbox_preds = _flatten_preds(bbox_preds, num_imgs, 4)
        flatten_objectness = _flatten_preds(objectnesses, num_imgs, 1).squeeze(-1).sigmoid()
        flatten_kps_preds = _flatten_preds(kps_preds, num_imgs, kps_preds[0].shape[1])

        expanded_priors = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        decoded_boxes = bbox_decode(expanded_priors, flatten_bbox_preds)
        decoded_keypoints = kps_decode(expanded_priors, flatten_kps_preds)

        results = []
        for img_idx in range(num_imgs):
            cls_score = flatten_cls_scores[img_idx]
            max_scores, labels = torch.max(cls_score, dim=1)
            scores = max_scores * flatten_objectness[img_idx]
            keep = scores >= self.score_threshold
            boxes = decoded_boxes[img_idx][keep]
            scores = scores[keep]
            labels = labels[keep]
            keypoints = decoded_keypoints[img_idx][keep]

            keep_indices = batched_nms(boxes, scores, labels, self.nms_threshold)
            if self.max_detections > 0:
                keep_indices = keep_indices[: self.max_detections]
            results.append(
                DetectionResult(
                    boxes=boxes[keep_indices],
                    scores=scores[keep_indices],
                    labels=labels[keep_indices],
                    keypoints=keypoints[keep_indices],
                )
            )
        return results


def _flatten_preds(preds: list[torch.Tensor], num_imgs: int, channels: int) -> torch.Tensor:
    flattened = [
        pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, channels)
        for pred in preds
    ]
    return torch.cat(flattened, dim=1)
