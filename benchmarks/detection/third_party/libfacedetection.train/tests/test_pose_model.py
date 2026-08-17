from __future__ import annotations

import torch

from yunet_train.tasks.pose import build_yunet_pose


def test_yunet_pose_forward_shapes() -> None:
    model = build_yunet_pose("yunet_n", kpt_shape=(17, 3)).eval()
    image = torch.zeros(1, 3, 128, 128)

    with torch.no_grad():
        cls_scores, bbox_preds, objectnesses, kpt_preds = model(image)

    expected_spatial_shapes = [(16, 16), (8, 8), (4, 4)]
    assert len(cls_scores) == 3
    assert len(bbox_preds) == 3
    assert len(objectnesses) == 3
    assert len(kpt_preds) == 3

    for level, (height, width) in enumerate(expected_spatial_shapes):
        assert tuple(cls_scores[level].shape) == (1, 1, height, width)
        assert tuple(bbox_preds[level].shape) == (1, 4, height, width)
        assert tuple(objectnesses[level].shape) == (1, 1, height, width)
        assert tuple(kpt_preds[level].shape) == (1, 51, height, width)
