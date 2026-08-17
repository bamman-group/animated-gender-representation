from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from yunet_train.engine import load_checkpoint
from yunet_train.tasks.pose import PoseSample, build_pose_eval_transforms, build_yunet_pose
from yunet_train.tasks.pose.coco_eval import _rescale_result_to_original
from yunet_train.tasks.pose.postprocess import YuNetPosePostprocessor
from yunet_train.tasks.pose.visualize import render_pose_sample

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YuNet pose on a single image and save a visualization (original image resolution)."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: work_dirs/pose_vis_one/<stem>_pose.jpg)",
    )
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    out_path = visualize_pose_image(args)
    _LOG.info("Wrote %s", out_path)


@torch.no_grad()
def visualize_pose_image(args: argparse.Namespace) -> Path:
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {args.image.resolve()}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    device = torch.device(args.device)
    model = build_yunet_pose(variant, kpt_shape=(17, 3))
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.to(device).eval()

    transform = build_pose_eval_transforms(args.image_size)
    sample = PoseSample(
        image=image_bgr,
        boxes=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        keypoints=np.zeros((0, 17, 3), dtype=np.float32),
        filename=args.image.name,
        original_shape=image_bgr.shape,
        image_shape=image_bgr.shape,
        pad_shape=image_bgr.shape,
        kpt_shape=(17, 3),
    )
    t_sample = transform(sample)
    postprocessor = YuNetPosePostprocessor(
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        kpt_shape=(17, 3),
    )
    batch = t_sample.image.unsqueeze(0).to(device)
    result = postprocessor(model(batch))[0]
    boxes, keypoints = _rescale_result_to_original(t_sample, result)
    scores_np = result.scores.detach().cpu().numpy()
    labels_np = result.labels.detach().cpu().numpy()
    _log_inference_settings(args, variant)
    _log_detections(args.image, boxes, scores_np, labels_np, keypoints)
    n = int(boxes.shape[0])
    vis = PoseSample(
        image=image_bgr,
        boxes=torch.from_numpy(boxes) if n else torch.zeros(0, 4),
        labels=torch.zeros(n, dtype=torch.int64),
        keypoints=torch.from_numpy(keypoints) if n else torch.zeros(0, 17, 3),
        filename=args.image.name,
        original_shape=image_bgr.shape,
        image_shape=image_bgr.shape,
        pad_shape=image_bgr.shape,
        kpt_shape=(17, 3),
    )
    rendered = render_pose_sample(vis)
    out_path = args.out
    if out_path is None:
        out_dir = Path("work_dirs") / "pose_vis_one"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.image.stem}_pose.jpg"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rendered)
    return out_path


def _log_inference_settings(args: argparse.Namespace, variant: str) -> None:
    _LOG.info(
        "Settings: score_threshold=%.4f nms_threshold=%.4f max_detections=%d image_size=%d variant=%s device=%s",
        args.score_threshold,
        args.nms_threshold,
        args.max_detections,
        args.image_size,
        variant,
        args.device,
    )


def _log_detections(
    image_path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    keypoints: np.ndarray,
) -> None:
    n = int(boxes.shape[0])
    _LOG.info(
        "Image=%s num_detections=%d (after score_threshold, nms, max_detections)",
        image_path.resolve(),
        n,
    )
    if n == 0:
        return
    kpt_vis = None
    if keypoints.size and keypoints.shape[-1] >= 3:
        vis = keypoints[..., 2]
        # kpt visibility channel is sigmoid probability in [0, 1]
        kpt_vis = (vis >= 0.5).sum(axis=-1)
    for i in range(n):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        extra = ""
        if kpt_vis is not None:
            extra = f" keypoints_conf_ge_0.5={int(kpt_vis[i])}/{keypoints.shape[1]}"
        _LOG.info(
            "  det[%d] label=%d score=%.6f bbox_xyxy=(%.2f, %.2f, %.2f, %.2f)%s",
            i,
            int(labels[i]),
            float(scores[i]),
            x1,
            y1,
            x2,
            y2,
            extra,
        )


if __name__ == "__main__":
    main()
