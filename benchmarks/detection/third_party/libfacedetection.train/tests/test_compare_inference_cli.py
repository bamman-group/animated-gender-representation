from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yunet_train.cli.compare_inference import SCRFD, _parse_input_size, resize_image, xyxy_score_to_xywh_score


def test_resize_image_auto_pads_to_stride_32() -> None:
    image = np.ones((33, 65, 3), dtype=np.uint8)

    resized, scale = resize_image(image, "AUTO")

    assert resized.shape == (64, 96, 3)
    assert scale == 1.0
    np.testing.assert_array_equal(resized[:33, :65], image)


def test_resize_image_vga_keeps_aspect_ratio_inside_canvas() -> None:
    image = np.ones((100, 200, 3), dtype=np.uint8)

    resized, scale = resize_image(image, "VGA")

    assert resized.shape == (480, 640, 3)
    assert scale == pytest.approx(3.2)


def test_xyxy_score_to_xywh_score() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 45.0, 0.8]], dtype=np.float32)

    converted = xyxy_score_to_xywh_score(boxes)

    np.testing.assert_allclose(converted, [[10.0, 20.0, 20.0, 25.0, 0.8]])


def test_parse_input_size_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        _parse_input_size("640")


def test_scrfd_anchor_centers_returns_uncached_centers_after_cache_limit() -> None:
    detector = SCRFD.__new__(SCRFD)
    detector.center_cache = {(idx, idx, 8): np.zeros((1, 2), dtype=np.float32) for idx in range(100)}
    detector.num_anchors = 2

    centers = detector._anchor_centers(68, 64, 16)

    assert centers.shape == (68 * 64 * 2, 2)
    assert (68, 64, 16) not in detector.center_cache


def test_compare_inference_module_path_is_importable() -> None:
    assert Path("yunet_train/cli/compare_inference.py").as_posix()
