from __future__ import annotations

import torch

from yunet_train.engine import batched_nms, nms
from yunet_train.tasks.face import YuNetPostprocessor


def test_nms_suppresses_overlapping_boxes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])

    keep = nms(boxes, scores, 0.5)

    assert keep.tolist() == [0, 2]


def test_batched_nms_keeps_different_labels() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])

    keep = batched_nms(boxes, scores, labels, 0.5)

    assert keep.tolist() == [0, 1]


def test_yunet_postprocessor_decodes_and_filters_predictions() -> None:
    postprocessor = YuNetPostprocessor(strides=(8,), score_threshold=0.5, nms_threshold=0.45)
    cls_scores = [torch.tensor([[[[10.0]]]])]
    bbox_preds = [
        torch.tensor(
            [
                [
                    [[0.5]],
                    [[0.5]],
                    [[0.0]],
                    [[0.0]],
                ]
            ]
        )
    ]
    objectnesses = [torch.tensor([[[[10.0]]]])]
    kps_preds = [torch.zeros(1, 10, 1, 1)]

    result = postprocessor((cls_scores, bbox_preds, objectnesses, kps_preds))[0]

    torch.testing.assert_close(result.boxes, torch.tensor([[0.0, 0.0, 8.0, 8.0]]))
    assert result.scores.shape == (1,)
    assert result.labels.tolist() == [0]
    assert result.keypoints.shape == (1, 10)

