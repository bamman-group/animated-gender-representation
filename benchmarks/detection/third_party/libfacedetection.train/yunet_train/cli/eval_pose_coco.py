from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch

from yunet_train.tasks.pose import COCO_PERSON_KEYPOINTS_VAL2017, COCO_VAL_IMAGE_DIR, build_yunet_pose
from yunet_train.tasks.pose.coco_eval import COCOPoseEvalDataset, collect_coco_keypoint_predictions, evaluate_coco_keypoints
from yunet_train.engine import load_checkpoint

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YuNet pose with official COCO keypoint AP.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ann-file", type=Path, default=COCO_PERSON_KEYPOINTS_VAL2017)
    parser.add_argument("--image-dir", type=Path, default=COCO_VAL_IMAGE_DIR)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--category-id", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/pose_coco_eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.out_dir)
    _LOG.info(
        "eval_pose_coco started checkpoint=%s device=%s ann_file=%s image_dir=%s",
        args.checkpoint.resolve(),
        args.device,
        args.ann_file.resolve(),
        args.image_dir.resolve(),
    )
    _LOG.info(
        "options variant=%s image_size=%s batch_size=%s workers=%s limit_samples=%s "
        "score_threshold=%s nms_threshold=%s max_detections=%s",
        args.variant,
        args.image_size,
        args.batch_size,
        args.workers,
        args.limit_samples,
        args.score_threshold,
        args.nms_threshold,
        args.max_detections,
    )
    started = time.perf_counter()
    result = eval_pose_coco(args)
    elapsed = time.perf_counter() - started
    metrics = " ".join(f"{key}={value:.6f}" for key, value in result.metrics.items())
    _LOG.info("COCO keypoint AP %s", metrics)
    _LOG.info("eval_pose_coco finished in %.2f s (predictions=%s)", elapsed, result.num_predictions)


def eval_pose_coco(args: argparse.Namespace):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.ann_file.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.ann_file.resolve()}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir.resolve()}")

    _LOG.info("Loading checkpoint %s", args.checkpoint.resolve())
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    _LOG.info("Building model variant=%s", variant)
    device = torch.device(args.device)
    model = build_yunet_pose(variant, kpt_shape=(17, 3))
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.to(device).eval()

    dataset = COCOPoseEvalDataset(
        ann_file=args.ann_file,
        image_dir=args.image_dir,
        image_size=args.image_size,
        limit_samples=args.limit_samples,
    )
    n_images = len(dataset)
    _LOG.info("COCO val images to run: %s", n_images)
    if n_images == 0:
        _LOG.warning(
            "Dataset is empty. Check that %s lists images and that --limit-samples is not 0.",
            args.ann_file,
        )

    predictions = collect_coco_keypoint_predictions(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        category_id=args.category_id,
    )
    _LOG.info("Collected %s raw detection records for COCOeval", len(predictions))

    results_path = args.out_dir / "pose_coco_results.json"
    metrics_path = args.out_dir / "pose_coco_metrics.csv"
    result = evaluate_coco_keypoints(
        ann_file=args.ann_file,
        predictions=predictions,
        results_file=results_path,
    )
    _write_metrics(metrics_path, result.metrics, result.num_predictions)
    _LOG.info("Wrote metrics %s", metrics_path.resolve())
    return result


def _configure_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval_pose_coco.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _write_metrics(path: Path, metrics: dict[str, float], num_predictions: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ("num_predictions", *metrics.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"num_predictions": num_predictions, **metrics})


if __name__ == "__main__":
    main()
