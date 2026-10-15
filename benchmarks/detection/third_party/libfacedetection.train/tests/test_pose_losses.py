from __future__ import annotations

import torch

from yunet_train.tasks.pose import YuNetPoseCriterion, keypoint_visibility_loss, oks_keypoint_loss


def test_oks_keypoint_loss_is_zero_for_perfect_visible_points() -> None:
    target = torch.zeros(1, 17, 3)
    target[..., 2] = 2
    pred = target[..., :2].clone()
    areas = torch.tensor([1000.0])

    loss = oks_keypoint_loss(pred, target, areas)

    assert loss.item() == 0.0


def test_oks_keypoint_loss_ignores_invisible_points() -> None:
    target = torch.zeros(1, 17, 3)
    target[..., 2] = 0
    pred = torch.ones(1, 17, 2) * 100
    areas = torch.tensor([1000.0])

    loss = oks_keypoint_loss(pred, target, areas)

    assert loss.item() == 0.0


def test_keypoint_visibility_loss_trains_visible_and_invisible_targets() -> None:
    target = torch.zeros(1, 2, 3)
    target[:, 0, 2] = 2
    good_logits = torch.tensor([[8.0, -8.0]])
    bad_logits = torch.tensor([[-8.0, 8.0]])

    assert keypoint_visibility_loss(good_logits, target) < keypoint_visibility_loss(bad_logits, target)


def test_pose_criterion_runs_forward_and_backward_on_synthetic_batch() -> None:
    cls_scores = [torch.zeros(1, 1, 2, 2, requires_grad=True)]
    bbox_preds = [torch.zeros(1, 4, 2, 2, requires_grad=True)]
    objectnesses = [torch.zeros(1, 1, 2, 2, requires_grad=True)]
    kpt_preds = [torch.zeros(1, 51, 2, 2, requires_grad=True)]
    boxes = [torch.tensor([[0.0, 0.0, 16.0, 16.0]])]
    labels = [torch.tensor([0])]
    keypoints = [torch.zeros(1, 17, 3)]
    keypoints[0][..., 0] = 4.0
    keypoints[0][..., 1] = 4.0
    keypoints[0][..., 2] = 2.0
    criterion = YuNetPoseCriterion(strides=(8,), kpt_shape=(17, 3))

    losses = criterion(
        (cls_scores, bbox_preds, objectnesses, kpt_preds),
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
    )
    total_loss = sum(losses.values())
    total_loss.backward()

    assert set(losses) == {"loss_cls", "loss_bbox", "loss_obj", "loss_kpt", "loss_kpt_vis"}
    assert torch.isfinite(total_loss)
    assert cls_scores[0].grad is not None
    assert bbox_preds[0].grad is not None
    assert objectnesses[0].grad is not None
    assert kpt_preds[0].grad is not None
