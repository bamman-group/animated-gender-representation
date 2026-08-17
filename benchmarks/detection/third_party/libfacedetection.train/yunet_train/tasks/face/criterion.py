from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from yunet_train.engine.codec import bbox_decode, kps_encode

from yunet_train.engine.assigners import SimOTAAssigner
from yunet_train.engine.losses import eiou_loss
from yunet_train.engine.priors import MlvlPointGenerator


@dataclass(frozen=True)
class YuNetLossWeights:
    cls: float = 1.0
    bbox: float = 5.0
    obj: float = 1.0
    kps: float = 0.1


class YuNetCriterion:
    def __init__(
        self,
        *,
        num_classes: int = 1,
        strides: tuple[int, ...] = (8, 16, 32),
        kps_num: int = 5,
        assigner: SimOTAAssigner | None = None,
        loss_weights: YuNetLossWeights | None = None,
        smooth_l1_beta: float = 1.0 / 9.0,
    ):
        self.num_classes = num_classes
        self.kps_num = kps_num
        self.prior_generator = MlvlPointGenerator(strides=strides, offset=0)
        self.assigner = assigner or SimOTAAssigner(center_radius=2.5)
        self.loss_weights = loss_weights or YuNetLossWeights()
        self.smooth_l1_beta = smooth_l1_beta

    def __call__(
        self,
        preds: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        *,
        boxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        keypoints: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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

        flatten_cls_preds = _flatten_preds(cls_scores, num_imgs, self.num_classes)
        flatten_bbox_preds = _flatten_preds(bbox_preds, num_imgs, 4)
        flatten_objectness = _flatten_preds(objectnesses, num_imgs, 1).squeeze(-1)
        flatten_kps_preds = _flatten_preds(kps_preds, num_imgs, self.kps_num * 2)

        expanded_priors = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        flatten_bboxes = bbox_decode(expanded_priors, flatten_bbox_preds)

        targets = [
            self._get_target_single(
                cls_preds=flatten_cls_preds[img_idx].detach(),
                objectness=flatten_objectness[img_idx].detach(),
                priors=expanded_priors[img_idx],
                decoded_bboxes=flatten_bboxes[img_idx].detach(),
                gt_bboxes=boxes[img_idx],
                gt_labels=labels[img_idx],
                gt_keypoints=keypoints[img_idx],
            )
            for img_idx in range(num_imgs)
        ]

        pos_masks = torch.cat([target["pos_mask"] for target in targets], dim=0)
        cls_targets = torch.cat([target["cls_target"] for target in targets], dim=0)
        obj_targets = torch.cat([target["obj_target"] for target in targets], dim=0)
        bbox_targets = torch.cat([target["bbox_target"] for target in targets], dim=0)
        kps_targets = torch.cat([target["kps_target"] for target in targets], dim=0)
        kps_weights = torch.cat([target["kps_weight"] for target in targets], dim=0)
        num_pos = sum(target["num_pos"] for target in targets)
        num_total_samples = max(float(num_pos), 1.0)

        flatten_cls_preds_all = flatten_cls_preds.reshape(-1, self.num_classes)
        flatten_bboxes_all = flatten_bboxes.reshape(-1, 4)
        flatten_objectness_all = flatten_objectness.reshape(-1, 1)
        flatten_kps_preds_all = flatten_kps_preds.reshape(-1, self.kps_num * 2)
        flatten_priors_all = expanded_priors.reshape(-1, 4)

        loss_obj = (
            F.binary_cross_entropy_with_logits(flatten_objectness_all, obj_targets, reduction="sum")
            / num_total_samples
            * self.loss_weights.obj
        )

        if pos_masks.any():
            loss_cls = (
                F.binary_cross_entropy_with_logits(
                    flatten_cls_preds_all[pos_masks],
                    cls_targets,
                    reduction="sum",
                )
                / num_total_samples
                * self.loss_weights.cls
            )
            loss_bbox = (
                eiou_loss(
                    flatten_bboxes_all[pos_masks],
                    bbox_targets,
                    reduction="sum",
                )
                / num_total_samples
                * self.loss_weights.bbox
            )
            encoded_kps = kps_encode(flatten_priors_all[pos_masks], kps_targets)
            loss_kps = _smooth_l1_loss(
                flatten_kps_preds_all[pos_masks],
                encoded_kps,
                weight=kps_weights,
                beta=self.smooth_l1_beta,
            ) * self.loss_weights.kps
        else:
            zero = flatten_cls_preds_all.sum() * 0
            loss_cls = zero
            loss_bbox = zero
            loss_kps = zero

        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_obj": loss_obj,
            "loss_kps": loss_kps,
        }

    @torch.no_grad()
    def _get_target_single(
        self,
        *,
        cls_preds: torch.Tensor,
        objectness: torch.Tensor,
        priors: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_keypoints: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        num_priors = priors.size(0)
        device = cls_preds.device
        gt_bboxes = gt_bboxes.to(device=device, dtype=decoded_bboxes.dtype)
        gt_labels = gt_labels.to(device=device, dtype=torch.long)
        gt_keypoints = gt_keypoints.to(device=device, dtype=decoded_bboxes.dtype)

        if gt_labels.numel() == 0:
            return {
                "pos_mask": cls_preds.new_zeros(num_priors).bool(),
                "cls_target": cls_preds.new_zeros((0, self.num_classes)),
                "obj_target": cls_preds.new_zeros((num_priors, 1)),
                "bbox_target": cls_preds.new_zeros((0, 4)),
                "kps_target": cls_preds.new_zeros((0, self.kps_num * 2)),
                "kps_weight": cls_preds.new_zeros((0, 1)),
                "num_pos": 0,
            }

        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
        )

        pos_mask = assign_result.gt_inds > 0
        pos_inds = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        num_pos = pos_inds.numel()

        if num_pos == 0:
            return {
                "pos_mask": pos_mask,
                "cls_target": cls_preds.new_zeros((0, self.num_classes)),
                "obj_target": cls_preds.new_zeros((num_priors, 1)),
                "bbox_target": cls_preds.new_zeros((0, 4)),
                "kps_target": cls_preds.new_zeros((0, self.kps_num * 2)),
                "kps_weight": cls_preds.new_zeros((0, 1)),
                "num_pos": 0,
            }

        pos_labels = gt_labels[pos_assigned_gt_inds]
        pos_ious = assign_result.max_overlaps[pos_inds]
        cls_target = F.one_hot(pos_labels, self.num_classes).to(cls_preds.dtype) * pos_ious.unsqueeze(-1)

        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1

        bbox_target = gt_bboxes[pos_assigned_gt_inds]
        kps_target = gt_keypoints[pos_assigned_gt_inds, :, :2].reshape(-1, self.kps_num * 2)
        kps_weight = torch.mean(gt_keypoints[pos_assigned_gt_inds, :, 2], dim=1, keepdim=True)

        return {
            "pos_mask": pos_mask,
            "cls_target": cls_target,
            "obj_target": obj_target,
            "bbox_target": bbox_target,
            "kps_target": kps_target,
            "kps_weight": kps_weight,
            "num_pos": num_pos,
        }


def _flatten_preds(preds: list[torch.Tensor], num_imgs: int, channels: int) -> torch.Tensor:
    flattened = [
        pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, channels)
        for pred in preds
    ]
    return torch.cat(flattened, dim=1)


def _smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum() * 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    weighted_loss = loss * weight
    avg_factor = weight.sum()
    if avg_factor <= 0:
        return pred.sum() * 0
    return weighted_loss.sum() / (avg_factor + torch.finfo(torch.float32).eps)

