from __future__ import annotations

import torch

from yunet_train.tasks.pose import YuNetPosePostprocessor


def test_pose_postprocessor_decodes_boxes_scores_and_keypoints() -> None:
    cls_scores = [torch.full((1, 1, 1, 1), 10.0)]
    bbox_preds = [torch.zeros(1, 4, 1, 1)]
    objectnesses = [torch.full((1, 1, 1, 1), 10.0)]
    kpt_preds = [torch.zeros(1, 51, 1, 1)]
    postprocessor = YuNetPosePostprocessor(strides=(8,), score_threshold=0.1, kpt_shape=(17, 3))

    results = postprocessor((cls_scores, bbox_preds, objectnesses, kpt_preds))

    assert len(results) == 1
    result = results[0]
    assert tuple(result.boxes.shape) == (1, 4)
    assert tuple(result.keypoints.shape) == (1, 17, 3)
    assert result.scores.item() > 0.99
    assert result.labels.tolist() == [0]
    torch.testing.assert_close(result.keypoints[..., :2], torch.zeros(1, 17, 2))
    torch.testing.assert_close(result.keypoints[..., 2], torch.full((1, 17), 0.5))
