from __future__ import annotations

import torch

from yunet_train.tasks.face import bbox_decode, kps_decode, kps_encode
from yunet_train.engine import MlvlPointGenerator, SimOTAAssigner, eiou_loss
from yunet_train.engine.losses import bbox_overlaps


def test_mlvl_point_generator_uses_legacy_offset_zero_order() -> None:
    generator = MlvlPointGenerator(strides=(8,), offset=0)

    priors = generator.grid_priors([(2, 3)])[0]

    expected = torch.tensor(
        [
            [0, 0, 8, 8],
            [8, 0, 8, 8],
            [16, 0, 8, 8],
            [0, 8, 8, 8],
            [8, 8, 8, 8],
            [16, 8, 8, 8],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(priors, expected)


def test_bbox_decode_matches_yunet_formula() -> None:
    priors = torch.tensor([[8.0, 8.0, 8.0, 8.0]])
    bbox_preds = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

    decoded = bbox_decode(priors, bbox_preds)

    torch.testing.assert_close(decoded, torch.tensor([[4.0, 4.0, 12.0, 12.0]]))


def test_keypoint_encode_decode_roundtrip() -> None:
    priors = torch.tensor([[8.0, 8.0, 8.0, 8.0]])
    keypoints = torch.tensor([[4.0, 4.0, 12.0, 4.0, 8.0, 8.0, 4.0, 12.0, 12.0, 12.0]])

    encoded = kps_encode(priors, keypoints)
    decoded = kps_decode(priors, encoded)

    torch.testing.assert_close(decoded, keypoints)


def test_eiou_loss_is_zero_for_equal_boxes() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    loss = eiou_loss(boxes, boxes)

    torch.testing.assert_close(loss, torch.zeros(1), atol=1e-6, rtol=0)


def test_bbox_overlaps_pairwise_iou() -> None:
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

    overlaps = bbox_overlaps(boxes1, boxes2)

    torch.testing.assert_close(overlaps, torch.tensor([[25.0 / 175.0]]), rtol=1e-5, atol=1e-6)


def test_simota_assigner_matches_foreground_prior() -> None:
    assigner = SimOTAAssigner(center_radius=2.5, candidate_topk=2)
    priors = torch.tensor(
        [
            [8.0, 8.0, 8.0, 8.0],
            [40.0, 40.0, 8.0, 8.0],
        ]
    )
    decoded_bboxes = torch.tensor(
        [
            [4.0, 4.0, 12.0, 12.0],
            [36.0, 36.0, 44.0, 44.0],
        ]
    )
    pred_scores = torch.tensor([[0.9], [0.1]])
    gt_bboxes = torch.tensor([[4.0, 4.0, 12.0, 12.0]])
    gt_labels = torch.tensor([0])

    result = assigner.assign(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels)

    assert result.gt_inds.tolist() == [1, 0]
    assert result.labels.tolist() == [0, -1]
    assert result.max_overlaps[0] > 0.99

