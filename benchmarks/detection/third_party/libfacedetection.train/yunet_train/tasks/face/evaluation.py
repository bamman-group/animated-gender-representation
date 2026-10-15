from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .postprocess import DetectionResult

PredictionDict = dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class WiderFaceAP:
    easy: float
    medium: float
    hard: float

    def as_dict(self) -> dict[str, float]:
        return {"easy": self.easy, "medium": self.medium, "hard": self.hard}


def wider_evaluation(predictions: PredictionDict, gt_dir: str | Path, iou_thresh: float = 0.5) -> WiderFaceAP:
    predictions = norm_score(predictions)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_dir)
    aps = []
    for gt_list in (easy_gt_list, medium_gt_list, hard_gt_list):
        aps.append(_evaluate_setting(predictions, facebox_list, event_list, file_list, gt_list, iou_thresh))
    return WiderFaceAP(easy=aps[0], medium=aps[1], hard=aps[2])


def detections_to_widerface(result: DetectionResult, meta: dict[str, Any]) -> np.ndarray:
    boxes = result.boxes.detach().cpu().float().numpy().copy()
    scores = result.scores.detach().cpu().float().numpy().reshape(-1, 1)
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    scale_factor = meta.get("scale_factor")
    if scale_factor is None:
        scale_factor = np.ones((4,), dtype=np.float32)
    else:
        scale_factor = np.asarray(scale_factor, dtype=np.float32)
    boxes[:, 0::2] /= scale_factor[0::2]
    boxes[:, 1::2] /= scale_factor[1::2]

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    xywh = np.stack((boxes[:, 0], boxes[:, 1], widths, heights), axis=1)
    return np.concatenate((xywh, scores), axis=1).astype(np.float32)


def add_prediction(predictions: PredictionDict, filename: str, boxes: np.ndarray) -> None:
    image_path = Path(filename)
    event_name = image_path.parent.as_posix()
    image_name = image_path.stem
    predictions.setdefault(event_name, {})[image_name] = boxes.astype(np.float32, copy=False)


def write_widerface_predictions(predictions: PredictionDict, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for event_name, event_predictions in predictions.items():
        event_dir = root / event_name
        event_dir.mkdir(parents=True, exist_ok=True)
        for image_name, boxes in event_predictions.items():
            path = event_dir / f"{image_name}.txt"
            with path.open("w", encoding="utf-8") as file:
                file.write(f"{event_name}/{image_name}.jpg\n")
                file.write(f"{boxes.shape[0]}\n")
                for box in boxes:
                    file.write(
                        f"{box[0]:.5f} {box[1]:.5f} {box[2]:.5f} {box[3]:.5f} {box[4]:.8f}\n"
                    )


def get_gt_boxes(gt_dir: str | Path) -> tuple[Any, Any, Any, Any, Any, Any]:
    root = Path(gt_dir)
    gt_mat = loadmat(root / "wider_face_val.mat")
    hard_mat = loadmat(root / "wider_hard_val.mat")
    medium_mat = loadmat(root / "wider_medium_val.mat")
    easy_mat = loadmat(root / "wider_easy_val.mat")
    return (
        gt_mat["face_bbx_list"],
        gt_mat["event_list"],
        gt_mat["file_list"],
        hard_mat["gt_list"],
        medium_mat["gt_list"],
        easy_mat["gt_list"],
    )


def norm_score(predictions: PredictionDict) -> PredictionDict:
    min_score = np.inf
    max_score = -np.inf
    for event_predictions in predictions.values():
        for boxes in event_predictions.values():
            if boxes.size == 0:
                continue
            min_score = min(min_score, float(np.min(boxes[:, -1])))
            max_score = max(max_score, float(np.max(boxes[:, -1])))

    if not np.isfinite(min_score) or not np.isfinite(max_score):
        return predictions
    diff = max_score - min_score
    for event_predictions in predictions.values():
        for boxes in event_predictions.values():
            if boxes.size == 0:
                continue
            if diff <= np.finfo(np.float32).eps:
                boxes[:, -1] = 1.0
            else:
                boxes[:, -1] = (boxes[:, -1] - min_score) / diff
    return predictions


def image_eval(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    pred = pred.copy()
    gt = gt.copy()
    pred_recall = np.zeros(pred.shape[0])
    recall_list = np.zeros(gt.shape[0])
    proposal_list = np.ones(pred.shape[0])

    pred[:, 2] += pred[:, 0]
    pred[:, 3] += pred[:, 1]
    gt[:, 2] += gt[:, 0]
    gt[:, 3] += gt[:, 1]

    for pred_idx in range(pred.shape[0]):
        gt_overlap = bbox_overlap(gt, pred[pred_idx])
        max_overlap = gt_overlap.max()
        max_idx = gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[pred_idx] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        pred_recall[pred_idx] = np.count_nonzero(recall_list == 1)
    return pred_recall, proposal_list


def bbox_overlap(boxes: np.ndarray, query_box: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], query_box[0])
    y1 = np.maximum(boxes[:, 1], query_box[1])
    x2 = np.minimum(boxes[:, 2], query_box[2])
    y2 = np.minimum(boxes[:, 3], query_box[3])
    widths = x2 - x1 + 1
    heights = y2 - y1 + 1
    inter = widths * heights
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_area = (query_box[2] - query_box[0] + 1) * (query_box[3] - query_box[1] + 1)
    union = box_areas + query_area - inter
    overlaps = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
    overlaps[widths <= 0] = 0
    overlaps[heights <= 0] = 0
    return overlaps


def img_pr_info(thresh_num: int, pred_info: np.ndarray, proposal_list: np.ndarray, pred_recall: np.ndarray) -> np.ndarray:
    pr_info = np.zeros((thresh_num, 2), dtype=np.float64)
    for threshold_idx in range(thresh_num):
        thresh = 1 - (threshold_idx + 1) / thresh_num
        pred_indices = np.where(pred_info[:, 4] >= thresh)[0]
        if len(pred_indices) == 0:
            continue
        pred_idx = pred_indices[-1]
        valid_pred_indices = np.where(proposal_list[: pred_idx + 1] == 1)[0]
        pr_info[threshold_idx, 0] = len(valid_pred_indices)
        pr_info[threshold_idx, 1] = pred_recall[pred_idx]
    return pr_info


def dataset_pr_info(thresh_num: int, pr_curve: np.ndarray, count_face: int) -> np.ndarray:
    output = np.zeros((thresh_num, 2), dtype=np.float64)
    output[:, 0] = np.divide(pr_curve[:, 1], pr_curve[:, 0], out=np.zeros(thresh_num), where=pr_curve[:, 0] > 0)
    if count_face > 0:
        output[:, 1] = pr_curve[:, 1] / count_face
    return output


def voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = np.maximum(mpre[idx - 1], mpre[idx])
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1]))


def _evaluate_setting(
    predictions: PredictionDict,
    facebox_list: Any,
    event_list: Any,
    file_list: Any,
    gt_list: Any,
    iou_thresh: float,
    thresh_num: int = 1000,
) -> float:
    event_num = len(event_list)
    count_face = 0
    pr_curve = np.zeros((thresh_num, 2), dtype=np.float64)

    for event_idx in range(event_num):
        event_name = str(event_list[event_idx][0][0])
        event_predictions = predictions.get(event_name, {})
        image_list = file_list[event_idx][0]
        sub_gt_list = gt_list[event_idx][0]
        gt_bbx_list = facebox_list[event_idx][0]
        for image_idx in range(len(image_list)):
            image_name = str(image_list[image_idx][0][0])
            pred_info = event_predictions.get(image_name, np.zeros((0, 5), dtype=np.float32))
            gt_boxes = gt_bbx_list[image_idx][0].astype(np.float32)
            keep_index = np.asarray(sub_gt_list[image_idx][0], dtype=np.int64).reshape(-1)
            count_face += len(keep_index)
            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue

            ignore = np.zeros(gt_boxes.shape[0], dtype=np.int64)
            if len(keep_index) != 0:
                ignore[keep_index - 1] = 1
            pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
            pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
    return voc_ap(pr_curve[:, 1], pr_curve[:, 0])
