from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from yunet_train.engine.assigners import SimOTAAssigner
from yunet_train.engine.codec import bbox_decode
from yunet_train.engine.losses import eiou_loss
from yunet_train.engine.priors import MlvlPointGenerator

from .codec import pose_keypoints_decode
from .losses import keypoint_visibility_loss, oks_keypoint_loss


@dataclass(frozen=True)
class YuNetPoseLossWeights:
    cls: float = 1.0
    bbox: float = 5.0
    obj: float = 1.0
    kpt: float = 10.0
    kpt_vis: float = 1.0


class YuNetPoseCriterion:
    def __init__(
        self,
        *,
        num_classes: int = 1,
        strides: tuple[int, ...] = (8, 16, 32),
        kpt_shape: tuple[int, int] = (17, 3),
        assigner: SimOTAAssigner | None = None,
        loss_weights: YuNetPoseLossWeights | None = None,
    ):
        if kpt_shape[1] < 3:
            raise ValueError("YuNetPoseCriterion currently expects keypoints with visibility, e.g. kpt_shape=(17, 3)")
        self.num_classes = num_classes
        self.kpt_shape = kpt_shape
        self.kpt_out_channels = kpt_shape[0] * kpt_shape[1]
        self.prior_generator = MlvlPointGenerator(strides=strides, offset=0)
        self.assigner = assigner or SimOTAAssigner(center_radius=2.5)
        self.loss_weights = loss_weights or YuNetPoseLossWeights()

    def __call__(
        self,
        preds: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        *,
        boxes: list[torch.Tensor],
        labels: list[torch.Tensor],
        keypoints: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cls_scores, bbox_preds, objectnesses, kpt_preds = preds
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
        flatten_kpt_preds = _flatten_preds(kpt_preds, num_imgs, self.kpt_out_channels)

        expanded_priors = flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1)
        flatten_bboxes = bbox_decode(expanded_priors, flatten_bbox_preds)
        decoded_keypoints = pose_keypoints_decode(expanded_priors, flatten_kpt_preds, kpt_shape=self.kpt_shape)
        raw_keypoints = flatten_kpt_preds.reshape(num_imgs, -1, *self.kpt_shape)

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
        kpt_targets = torch.cat([target["kpt_target"] for target in targets], dim=0)
        area_targets = torch.cat([target["area_target"] for target in targets], dim=0)
        num_pos = sum(target["num_pos"] for target in targets)
        num_total_samples = max(float(num_pos), 1.0)

        flatten_cls_preds_all = flatten_cls_preds.reshape(-1, self.num_classes)
        flatten_bboxes_all = flatten_bboxes.reshape(-1, 4)
        flatten_objectness_all = flatten_objectness.reshape(-1, 1)
        decoded_keypoints_all = decoded_keypoints.reshape(-1, *self.kpt_shape)
        raw_keypoints_all = raw_keypoints.reshape(-1, *self.kpt_shape)

        loss_obj = (
            F.binary_cross_entropy_with_logits(flatten_objectness_all, obj_targets, reduction="sum")
            / num_total_samples
            * self.loss_weights.obj
        )

        if pos_masks.any():
            loss_cls = (
                F.binary_cross_entropy_with_logits(flatten_cls_preds_all[pos_masks], cls_targets, reduction="sum")
                / num_total_samples
                * self.loss_weights.cls
            )
            loss_bbox = (
                eiou_loss(flatten_bboxes_all[pos_masks], bbox_targets, reduction="sum")
                / num_total_samples
                * self.loss_weights.bbox
            )
            loss_kpt = (
                oks_keypoint_loss(decoded_keypoints_all[pos_masks][..., :2], kpt_targets, area_targets)
                * self.loss_weights.kpt
            )
            loss_kpt_vis = (
                keypoint_visibility_loss(raw_keypoints_all[pos_masks][..., 2], kpt_targets)
                * self.loss_weights.kpt_vis
            )
        else:
            zero = flatten_cls_preds_all.sum() * 0
            loss_cls = zero
            loss_bbox = zero
            loss_kpt = zero
            loss_kpt_vis = zero

        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_obj": loss_obj,
            "loss_kpt": loss_kpt,
            "loss_kpt_vis": loss_kpt_vis,
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
            return self._empty_target(cls_preds, objectness, num_priors)

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
            return self._empty_target(cls_preds, objectness, num_priors, pos_mask=pos_mask)

        pos_labels = gt_labels[pos_assigned_gt_inds]
        pos_ious = assign_result.max_overlaps[pos_inds]
        cls_target = F.one_hot(pos_labels, self.num_classes).to(cls_preds.dtype) * pos_ious.unsqueeze(-1)

        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1

        bbox_target = gt_bboxes[pos_assigned_gt_inds]
        kpt_target = gt_keypoints[pos_assigned_gt_inds]
        widths = (bbox_target[:, 2] - bbox_target[:, 0]).clamp_min(0)
        heights = (bbox_target[:, 3] - bbox_target[:, 1]).clamp_min(0)
        area_target = (widths * heights).clamp_min(1.0)

        return {
            "pos_mask": pos_mask,
            "cls_target": cls_target,
            "obj_target": obj_target,
            "bbox_target": bbox_target,
            "kpt_target": kpt_target,
            "area_target": area_target,
            "num_pos": num_pos,
        }

    def _empty_target(
        self,
        cls_preds: torch.Tensor,
        objectness: torch.Tensor,
        num_priors: int,
        *,
        pos_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        if pos_mask is None:
            pos_mask = cls_preds.new_zeros(num_priors).bool()
        return {
            "pos_mask": pos_mask,
            "cls_target": cls_preds.new_zeros((0, self.num_classes)),
            "obj_target": cls_preds.new_zeros((num_priors, 1)),
            "bbox_target": cls_preds.new_zeros((0, 4)),
            "kpt_target": cls_preds.new_zeros((0, *self.kpt_shape)),
            "area_target": cls_preds.new_zeros((0,)),
            "num_pos": 0,
        }


def _flatten_preds(preds: list[torch.Tensor], num_imgs: int, channels: int) -> torch.Tensor:
    flattened = [pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, channels) for pred in preds]
    return torch.cat(flattened, dim=1)
