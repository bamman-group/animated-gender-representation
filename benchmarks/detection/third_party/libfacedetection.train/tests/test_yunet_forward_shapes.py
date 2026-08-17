from __future__ import annotations

import sys

import pytest
import torch

from yunet_train.tasks.face import build_yunet


@pytest.mark.parametrize(
    "variant",
    ["yunet_n", "yunet_s"],
)
def test_yunet_forward_shapes(variant: str) -> None:
    model = build_yunet(variant).eval()
    image = torch.zeros(1, 3, 640, 640)

    with torch.no_grad():
        cls_scores, bbox_preds, objectnesses, kps_preds = model(image)

    expected_spatial_shapes = [(80, 80), (40, 40), (20, 20)]

    assert len(cls_scores) == 3
    assert len(bbox_preds) == 3
    assert len(objectnesses) == 3
    assert len(kps_preds) == 3

    for level, (height, width) in enumerate(expected_spatial_shapes):
        assert tuple(cls_scores[level].shape) == (1, 1, height, width)
        assert tuple(bbox_preds[level].shape) == (1, 4, height, width)
        assert tuple(objectnesses[level].shape) == (1, 1, height, width)
        assert tuple(kps_preds[level].shape) == (1, 10, height, width)


def test_yunet_model_does_not_import_mmdet_or_mmcv() -> None:
    imported_names = {module_name.split(".")[0] for module_name in sys.modules}

    assert "mmdet" not in imported_names
    assert "mmcv" not in imported_names
