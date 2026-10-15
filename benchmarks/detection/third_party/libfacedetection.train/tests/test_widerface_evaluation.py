from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from yunet_train.tasks.face.evaluation import detections_to_widerface, image_eval, norm_score, voc_ap, write_widerface_predictions
import yunet_train.tasks.face.evaluation as widerface_eval
from yunet_train.tasks.face import DetectionResult


def test_detections_to_widerface_handles_origin_size_without_scale_factor() -> None:
    result = DetectionResult(
        boxes=torch.tensor([[10.0, 20.0, 50.0, 80.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
        keypoints=torch.zeros((1, 10)),
    )
    meta = {
        "scale_factor": None,
        "ori_shape": (200, 300, 3),
    }

    boxes = detections_to_widerface(result, meta)

    np.testing.assert_allclose(boxes, [[10.0, 20.0, 40.0, 60.0, 0.9]], rtol=1e-6)


def test_detections_to_widerface_rescales_to_original_xywh() -> None:
    result = DetectionResult(
        boxes=torch.tensor([[10.0, 20.0, 50.0, 80.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
        keypoints=torch.zeros((1, 10)),
    )
    meta = {
        "scale_factor": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "ori_shape": (200, 300, 3),
    }

    boxes = detections_to_widerface(result, meta)

    np.testing.assert_allclose(boxes, [[20.0, 40.0, 80.0, 120.0, 0.9]], rtol=1e-6)


def test_detections_to_widerface_keeps_out_of_image_boxes_like_legacy_eval() -> None:
    result = DetectionResult(
        boxes=torch.tensor([[-5.0, 2.0, 12.0, 18.0]]),
        scores=torch.tensor([0.7]),
        labels=torch.tensor([0]),
        keypoints=torch.zeros((1, 10)),
    )
    meta = {
        "scale_factor": np.ones((4,), dtype=np.float32),
        "ori_shape": (10, 10, 3),
    }

    boxes = detections_to_widerface(result, meta)

    np.testing.assert_allclose(boxes, [[-5.0, 2.0, 17.0, 16.0, 0.7]], rtol=1e-6)


def test_widerface_image_eval_counts_single_match() -> None:
    pred = np.array([[0.0, 0.0, 10.0, 10.0, 0.9]], dtype=np.float32)
    gt = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
    ignore = np.array([1], dtype=np.int64)

    pred_recall, proposal_list = image_eval(pred, gt, ignore, 0.5)

    np.testing.assert_allclose(pred_recall, [1])
    np.testing.assert_allclose(proposal_list, [1])


def test_widerface_setting_eval_keeps_fractional_coordinates(monkeypatch) -> None:
    predictions = {"event": {"image": np.array([[0.6, 0.6, 9.0, 9.0, 1.0]], dtype=np.float32)}}
    facebox_list = np.empty((1, 1), dtype=object)
    event_list = np.empty((1, 1), dtype=object)
    file_list = np.empty((1, 1), dtype=object)
    gt_list = np.empty((1, 1), dtype=object)
    facebox_list[0, 0] = np.array([[np.array([[0.6, 0.6, 9.0, 9.0]], dtype=np.float32)]], dtype=object)
    event_list[0, 0] = np.array(["event"], dtype=object)
    file_list[0, 0] = np.array([[np.array(["image"], dtype=object)]], dtype=object)
    gt_list[0, 0] = np.array([[np.array([1], dtype=np.int64)]], dtype=object)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy WIDERFace evaluation does not round coordinates")

    monkeypatch.setattr(widerface_eval.np, "round", fail_if_called)

    ap = widerface_eval._evaluate_setting(predictions, facebox_list, event_list, file_list, gt_list, 0.5, thresh_num=10)

    assert ap == 1.0


def test_norm_score_and_voc_ap() -> None:
    predictions = {"event": {"image": np.array([[0, 0, 1, 1, 0.2], [0, 0, 1, 1, 0.6]], dtype=np.float32)}}

    normalized = norm_score(predictions)

    np.testing.assert_allclose(normalized["event"]["image"][:, 4], [0.0, 1.0])
    assert voc_ap(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == 1.0


def test_write_widerface_predictions() -> None:
    output_dir = Path(__file__).resolve().parent / "output" / "widerface_predictions"
    predictions = {"0--Parade": {"sample": np.array([[1, 2, 3, 4, 0.5]], dtype=np.float32)}}

    write_widerface_predictions(predictions, output_dir)

    text = (output_dir / "0--Parade" / "sample.txt").read_text(encoding="utf-8")
    assert "0--Parade/sample.jpg" in text
    assert "1.00000 2.00000 3.00000 4.00000 0.50000000" in text
