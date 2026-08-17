from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from yunet_train.tasks.pose import (
    COCO8_POSE_ROOT,
    YOLOPoseDataset,
    YuNetPoseCriterion,
    YuNetPosePostprocessor,
    build_pose_eval_transforms,
    build_yunet_pose,
    collate_pose_samples,
    evaluate_pose_loss,
)
from yunet_train.tasks.pose.visualize import render_pose_sample
from yunet_train.engine import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YuNet pose checkpoint on YOLO-pose validation data.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=COCO8_POSE_ROOT)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/pose_eval"))
    parser.add_argument("--save-visualizations", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    stats = eval_pose(parse_args())
    print(
        f"Pose val loss={stats.loss:.6f} cls={stats.loss_cls:.6f} bbox={stats.loss_bbox:.6f} "
        f"obj={stats.loss_obj:.6f} kpt={stats.loss_kpt:.6f} kpt_vis={stats.loss_kpt_vis:.6f}"
    )


def eval_pose(args: argparse.Namespace):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    device = torch.device(args.device)
    model = build_yunet_pose(variant, kpt_shape=(17, 3))
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.to(device).eval()

    dataset = YOLOPoseDataset(
        args.data_root,
        split="val",
        transform=build_pose_eval_transforms(args.image_size),
        kpt_shape=(17, 3),
    )
    if args.limit_samples is not None:
        dataset.records = dataset.records[: args.limit_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pose_samples,
        pin_memory=device.type == "cuda",
    )
    criterion = YuNetPoseCriterion(strides=(8, 16, 32), kpt_shape=(17, 3))
    stats = evaluate_pose_loss(model=model, criterion=criterion, data_loader=loader, device=device)
    _write_metrics(args.out_dir / "pose_eval_metrics.csv", stats)
    if args.save_visualizations > 0:
        _save_visualizations(args, dataset, model, device)
    return stats


@torch.no_grad()
def _save_visualizations(args: argparse.Namespace, dataset: YOLOPoseDataset, model: torch.nn.Module, device: torch.device) -> None:
    postprocessor = YuNetPosePostprocessor(
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        kpt_shape=(17, 3),
    )
    vis_dir = args.out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(args.save_visualizations, len(dataset))):
        sample = dataset[idx]
        image = sample.image.unsqueeze(0).to(device) if isinstance(sample.image, torch.Tensor) else None
        if image is None:
            raise TypeError("eval pose visualization expects tensor samples")
        result = postprocessor(model(image))[0]
        pred_sample = _prediction_sample(sample, result)
        rendered = render_pose_sample(pred_sample)
        cv2.imwrite(str(vis_dir / f"{idx:04d}_{Path(sample.filename).stem}.jpg"), rendered)


def _prediction_sample(sample, result):
    from yunet_train.tasks.pose import PoseSample

    return PoseSample(
        image=sample.image,
        boxes=result.boxes.detach().cpu(),
        labels=result.labels.detach().cpu(),
        keypoints=result.keypoints.detach().cpu(),
        filename=sample.filename,
        original_shape=sample.original_shape,
        image_shape=sample.image_shape,
        pad_shape=sample.pad_shape,
        kpt_shape=sample.kpt_shape,
        scale_factor=sample.scale_factor,
    )


def _write_metrics(path: Path, stats) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("loss", "loss_cls", "loss_bbox", "loss_obj", "loss_kpt", "loss_kpt_vis", "steps"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "loss": stats.loss,
                "loss_cls": stats.loss_cls,
                "loss_bbox": stats.loss_bbox,
                "loss_obj": stats.loss_obj,
                "loss_kpt": stats.loss_kpt,
                "loss_kpt_vis": stats.loss_kpt_vis,
                "steps": stats.steps,
            }
        )


if __name__ == "__main__":
    main()
