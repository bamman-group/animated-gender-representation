from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import rmtree

import cv2
import numpy as np
import torch

from yunet_train.cli.eval_pose_coco import eval_pose_coco
from yunet_train.tasks.pose import PoseSample, build_yunet_pose
from yunet_train.tasks.pose.coco_eval import _result_to_coco_predictions, evaluate_coco_keypoints
from yunet_train.tasks.pose.postprocess import PoseDetectionResult


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "pose_coco_eval"


def test_coco_prediction_format_rescales_to_original_image() -> None:
    sample = PoseSample(
        image=np.zeros((50, 100, 3), dtype=np.uint8),
        boxes=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        keypoints=np.zeros((0, 17, 3), dtype=np.float32),
        filename="synthetic.jpg",
        original_shape=(100, 200, 3),
        image_shape=(50, 100, 3),
        pad_shape=(64, 64, 3),
        kpt_shape=(17, 3),
        scale_factor=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    keypoints = torch.zeros(1, 17, 3)
    keypoints[..., 0] = 20
    keypoints[..., 1] = 30
    keypoints[..., 2] = 0.75
    result = PoseDetectionResult(
        boxes=torch.tensor([[10.0, 15.0, 50.0, 45.0]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
        keypoints=keypoints,
    )

    predictions = _result_to_coco_predictions(123, sample, result, category_id=1)

    assert len(predictions) == 1
    assert predictions[0]["image_id"] == 123
    assert predictions[0]["category_id"] == 1
    assert predictions[0]["bbox"] == [20.0, 30.0, 80.0, 60.0]
    assert predictions[0]["keypoints"][:6] == [40.0, 60.0, 0.75, 40.0, 60.0, 0.75]


def test_evaluate_coco_keypoints_empty_predictions_writes_zero_metrics() -> None:
    work_dir = OUTPUT_ROOT / "empty_predictions"
    if work_dir.exists():
        rmtree(work_dir)
    work_dir.mkdir(parents=True)
    ann_file = work_dir / "person_keypoints_val.json"
    ann_file.write_text(json.dumps({"images": [], "annotations": [], "categories": []}), encoding="utf-8")

    result = evaluate_coco_keypoints(
        ann_file=ann_file,
        predictions=[],
        results_file=work_dir / "results.json",
    )

    assert result.num_predictions == 0
    assert result.metrics["AP"] == 0.0
    assert json.loads((work_dir / "results.json").read_text(encoding="utf-8")) == []
    rmtree(work_dir)


def test_eval_pose_coco_cli_smoke_with_empty_predictions() -> None:
    work_dir = OUTPUT_ROOT / "cli_smoke"
    if work_dir.exists():
        rmtree(work_dir)
    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000000000001.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    ann_file = work_dir / "person_keypoints_val.json"
    ann_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 32, "height": 32}],
                "annotations": [],
                "categories": [{"id": 1, "name": "person", "keypoints": [], "skeleton": []}],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = _checkpoint(work_dir / "yunet_pose_n.pth")

    result = eval_pose_coco(
        argparse.Namespace(
            checkpoint=checkpoint,
            ann_file=ann_file,
            image_dir=image_dir,
            variant="yunet_n",
            image_size=64,
            batch_size=1,
            workers=0,
            device="cpu",
            limit_samples=1,
            score_threshold=2.0,
            nms_threshold=0.45,
            max_detections=20,
            category_id=1,
            out_dir=work_dir / "eval",
        )
    )

    assert result.num_predictions == 0
    assert (work_dir / "eval" / "pose_coco_metrics.csv").exists()
    rmtree(work_dir)


def _checkpoint(path: Path) -> Path:
    model = build_yunet_pose("yunet_n", kpt_shape=(17, 3))
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "config": {"variant": "yunet_n"},
            "metrics": {},
        },
        path,
    )
    return path
