from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .postprocess import PoseDetectionResult, YuNetPosePostprocessor
from .transforms import build_pose_eval_transforms
from .types import PoseSample

_LOGGER = logging.getLogger(__name__)

COCO_KEYPOINT_METRIC_NAMES: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "APM",
    "APL",
    "AR",
    "AR50",
    "AR75",
    "ARM",
    "ARL",
)


@dataclass(frozen=True)
class CocoPoseItem:
    image_id: int
    sample: PoseSample


@dataclass(frozen=True)
class CocoPoseBatch:
    images: torch.Tensor
    image_ids: list[int]
    samples: list[PoseSample]


@dataclass(frozen=True)
class CocoKeypointEvalResult:
    stats: np.ndarray
    metrics: dict[str, float]
    results_file: Path
    num_predictions: int


class COCOPoseEvalDataset(Dataset):
    def __init__(
        self,
        ann_file: str | Path,
        image_dir: str | Path,
        *,
        image_size: int = 640,
        limit_samples: int | None = None,
    ):
        self.ann_file = Path(ann_file)
        self.image_dir = Path(image_dir)
        self.images = _load_coco_images(self.ann_file)
        if limit_samples is not None:
            self.images = self.images[:limit_samples]
        self.transform = build_pose_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> CocoPoseItem:
        info = self.images[index]
        image_path = self.image_dir / info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read COCO image: {image_path}")
        sample = PoseSample(
            image=image,
            boxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            keypoints=np.zeros((0, 17, 3), dtype=np.float32),
            filename=info["file_name"],
            original_shape=image.shape,
            image_shape=image.shape,
            pad_shape=image.shape,
            kpt_shape=(17, 3),
        )
        return CocoPoseItem(image_id=int(info["id"]), sample=self.transform(sample))


@torch.no_grad()
def collect_coco_keypoint_predictions(
    *,
    model: torch.nn.Module,
    dataset: COCOPoseEvalDataset,
    device: torch.device | str,
    batch_size: int = 1,
    workers: int = 0,
    score_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 20,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_coco_pose_items,
        pin_memory=torch.device(device).type == "cuda",
    )
    postprocessor = YuNetPosePostprocessor(
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        max_detections=max_detections,
        kpt_shape=(17, 3),
    )
    model.eval()
    num_batches = len(loader)
    num_images = len(dataset)
    _LOGGER.info(
        "Inference: images=%s batch_size=%s batches=%s device=%s",
        num_images,
        batch_size,
        num_batches,
        device,
    )
    log_every = max(1, num_batches // 10) if num_batches > 10 else max(1, num_batches)
    predictions: list[dict[str, Any]] = []
    infer_started = time.perf_counter()
    if num_batches == 0:
        _LOGGER.warning("DataLoader has zero batches; skipping inference.")
    for batch_idx, batch in enumerate(loader):
        images = batch.images.to(device, non_blocking=True)
        results = postprocessor(model(images))
        for image_id, sample, result in zip(batch.image_ids, batch.samples, results):
            predictions.extend(_result_to_coco_predictions(image_id, sample, result, category_id=category_id))
        done = batch_idx + 1
        if done == 1 or done == num_batches or done % log_every == 0:
            _LOGGER.info(
                "Inference progress: batch %s/%s (elapsed %.1fs)",
                done,
                num_batches,
                time.perf_counter() - infer_started,
            )
    _LOGGER.info("Inference done in %.2fs, total detection records=%s", time.perf_counter() - infer_started, len(predictions))
    return predictions


def collate_coco_pose_items(items: list[CocoPoseItem]) -> CocoPoseBatch:
    images = torch.stack([_as_image_tensor(item.sample.image) for item in items], dim=0)
    return CocoPoseBatch(
        images=images,
        image_ids=[item.image_id for item in items],
        samples=[item.sample for item in items],
    )


def evaluate_coco_keypoints(
    *,
    ann_file: str | Path,
    predictions: list[dict[str, Any]],
    results_file: str | Path,
) -> CocoKeypointEvalResult:
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Writing COCO detection JSON (%s entries) -> %s", len(predictions), results_path.resolve())
    results_path.write_text(json.dumps(predictions), encoding="utf-8")
    if not predictions:
        _LOGGER.warning(
            "No predictions produced (empty results). COCO AP/AR will be zero. "
            "Check score threshold, data paths, and that the dataset is not empty."
        )
        stats = np.zeros((len(COCO_KEYPOINT_METRIC_NAMES),), dtype=np.float32)
        return CocoKeypointEvalResult(
            stats=stats,
            metrics=_stats_to_metrics(stats),
            results_file=results_path,
            num_predictions=0,
        )

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO keypoint AP. "
            "Install it with `python -m pip install -r requirements-pose.txt`."
        ) from exc

    _LOGGER.info("Running official COCO keypoint evaluation (pycocotools COCOeval)...")
    eval_started = time.perf_counter()
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(str(results_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    _LOGGER.info("COCOeval finished in %.2fs", time.perf_counter() - eval_started)
    stats = np.array(coco_eval.stats, dtype=np.float32)
    return CocoKeypointEvalResult(
        stats=stats,
        metrics=_stats_to_metrics(stats),
        results_file=results_path,
        num_predictions=len(predictions),
    )


def _result_to_coco_predictions(
    image_id: int,
    sample: PoseSample,
    result: PoseDetectionResult,
    *,
    category_id: int,
) -> list[dict[str, Any]]:
    boxes, keypoints = _rescale_result_to_original(sample, result)
    scores = result.scores.detach().cpu().numpy()
    predictions = []
    for idx in range(boxes.shape[0]):
        box = boxes[idx]
        xywh = [
            float(box[0]),
            float(box[1]),
            float(max(box[2] - box[0], 0.0)),
            float(max(box[3] - box[1], 0.0)),
        ]
        predictions.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": xywh,
                "score": float(scores[idx]),
                "keypoints": keypoints[idx].reshape(-1).astype(float).tolist(),
            }
        )
    return predictions


def _rescale_result_to_original(
    sample: PoseSample,
    result: PoseDetectionResult,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = result.boxes.detach().cpu().numpy().astype(np.float32, copy=True)
    keypoints = result.keypoints.detach().cpu().numpy().astype(np.float32, copy=True)
    if sample.scale_factor is None:
        scale_x = scale_y = 1.0
    else:
        scale_x = float(sample.scale_factor[0])
        scale_y = float(sample.scale_factor[1])
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(f"Invalid scale_factor: {sample.scale_factor}")

    original_h, original_w = sample.original_shape[:2]
    if boxes.size:
        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_h)
    if keypoints.size:
        keypoints[..., 0] /= scale_x
        keypoints[..., 1] /= scale_y
        keypoints[..., 0] = np.clip(keypoints[..., 0], 0, original_w)
        keypoints[..., 1] = np.clip(keypoints[..., 1], 0, original_h)
    return boxes, keypoints


def _stats_to_metrics(stats: np.ndarray) -> dict[str, float]:
    return {
        name: float(stats[idx]) if idx < stats.shape[0] else 0.0
        for idx, name in enumerate(COCO_KEYPOINT_METRIC_NAMES)
    }


def _load_coco_images(ann_file: Path) -> list[dict[str, Any]]:
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO annotation file, images must be a list: {ann_file}")
    return sorted(images, key=lambda item: int(item["id"]))


def _as_image_tensor(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("COCO pose samples must be tensors before collation")
    return image.float()
