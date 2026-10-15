from __future__ import annotations

import torch

from yunet_train.tasks.face import YuNetCriterion


def _single_level_preds() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    cls_scores = [torch.tensor([[[[4.0]]]], requires_grad=True)]
    bbox_preds = [
        torch.tensor(
            [
                [
                    [[0.5]],
                    [[0.5]],
                    [[0.0]],
                    [[0.0]],
                ]
            ],
            requires_grad=True,
        )
    ]
    objectnesses = [torch.tensor([[[[4.0]]]], requires_grad=True)]
    kps_preds = [torch.zeros(1, 10, 1, 1, requires_grad=True)]
    return cls_scores, bbox_preds, objectnesses, kps_preds


def test_yunet_criterion_returns_finite_losses_and_backpropagates() -> None:
    criterion = YuNetCriterion(strides=(8,), kps_num=5)
    preds = _single_level_preds()
    boxes = [torch.tensor([[0.0, 0.0, 8.0, 8.0]])]
    labels = [torch.tensor([0])]
    keypoints = [
        torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],
                    [8.0, 0.0, 1.0],
                    [4.0, 4.0, 1.0],
                    [0.0, 8.0, 1.0],
                    [8.0, 8.0, 1.0],
                ]
            ]
        )
    ]

    losses = criterion(preds, boxes=boxes, labels=labels, keypoints=keypoints)
    total_loss = sum(losses.values())
    total_loss.backward()

    assert set(losses) == {"loss_cls", "loss_bbox", "loss_obj", "loss_kps"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert preds[0][0].grad is not None
    assert preds[1][0].grad is not None
    assert preds[2][0].grad is not None
    assert preds[3][0].grad is not None


def test_yunet_criterion_handles_images_without_ground_truth() -> None:
    criterion = YuNetCriterion(strides=(8,), kps_num=5)
    preds = _single_level_preds()

    losses = criterion(
        preds,
        boxes=[torch.zeros((0, 4))],
        labels=[torch.zeros((0,), dtype=torch.long)],
        keypoints=[torch.zeros((0, 5, 3))],
    )
    total_loss = sum(losses.values())
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert losses["loss_cls"].item() == 0.0
    assert losses["loss_bbox"].item() == 0.0
    assert losses["loss_kps"].item() == 0.0
    assert losses["loss_obj"].item() > 0.0
    assert preds[2][0].grad is not None

