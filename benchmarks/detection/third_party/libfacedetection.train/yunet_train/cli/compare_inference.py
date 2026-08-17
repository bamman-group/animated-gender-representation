from __future__ import annotations

import argparse
from itertools import product
from math import ceil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnx
import onnxruntime
from tqdm import tqdm

from yunet_train.tasks.face import (
    PredictionDict,
    WIDERFaceDataset,
    WIDER_VAL_ANN_FILE,
    WIDER_VAL_GT_DIR,
    WIDER_VAL_IMAGE_DIR,
    add_prediction,
    wider_evaluation,
    write_widerface_predictions,
)


class Timer:
    def __init__(self) -> None:
        self.total = 0.0
        self.value = 0.0
        self.count = 0
        self.running = False

    def tic(self) -> None:
        if self.running:
            raise RuntimeError("timer is already running")
        self.running = True
        self.value = perf_counter()

    def toc(self) -> None:
        if not self.running:
            raise RuntimeError("timer is not running")
        self.running = False
        self.count += 1
        self.total += perf_counter() - self.value
        self.value = 0.0

    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class TimeEngine:
    def __init__(self) -> None:
        self.container: dict[str, Timer] = {}

    def tic(self, key: str) -> None:
        self.container.setdefault(key, Timer()).tic()

    def toc(self, key: str) -> None:
        self.container[key].toc()

    def total_second(self) -> float:
        return sum(timer.total for timer in self.container.values())

    def reset(self) -> None:
        self.container = {}

    def summary_lines(self) -> list[str]:
        lines = []
        for key, timer in self.container.items():
            lines.append(f"{key}: {timer.average():.6f}s")
        total = self.total_second()
        count = max((timer.count for timer in self.container.values()), default=0)
        if count > 0:
            lines.append(f"total: {total / count:.6f}s")
            if total > 0:
                lines.append(f"fps: {count / total:.2f}")
        return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ONNX face detector inference.")
    parser.add_argument("model_file", type=Path, help="ONNX model file path")
    parser.add_argument("--eval", action="store_true", help="evaluate on WIDER Face val")
    parser.add_argument("--image", type=Path, default=None, help="image path for single-image inference")
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/sample"))
    parser.add_argument(
        "--mode",
        default="640,640",
        help='input resize mode: ORIGIN, AUTO, VGA, or "width,height"',
    )
    parser.add_argument("--ann-file", type=Path, default=WIDER_VAL_ANN_FILE)
    parser.add_argument("--img-prefix", type=Path, default=WIDER_VAL_IMAGE_DIR)
    parser.add_argument("--gt-dir", type=Path, default=WIDER_VAL_GT_DIR)
    parser.add_argument("--nms-thresh", "--nms_thresh", type=float, default=0.45)
    parser.add_argument("--score-thresh", "--score_thresh", type=float, default=0.02)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = build_detector(args.model_file, nms_thresh=args.nms_thresh)
    prefix = detector.taskname
    if args.eval:
        aps = evaluate_onnx_detector(
            detector,
            ann_file=args.ann_file,
            img_prefix=args.img_prefix,
            gt_dir=args.gt_dir,
            output_dir=args.out_dir,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            iou_thresh=args.iou_thresh,
            mode=args.mode,
            limit_samples=args.limit_samples,
            save_preds=args.save_preds,
        )
        print(f"WIDERFace AP easy={aps.easy:.6f} medium={aps.medium:.6f} hard={aps.hard:.6f}")
        return

    if args.image is None:
        raise ValueError("--image is required unless --eval is set")
    run_single_image(
        detector,
        image_path=args.image,
        output_dir=args.out_dir,
        prefix=prefix,
        score_thresh=args.score_thresh,
        mode=args.mode,
    )


def build_detector(model_file: Path, *, nms_thresh: float) -> "OnnxDetector":
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    name = model_file.name.lower()
    if name.startswith("scrfd"):
        return SCRFD(model_file, nms_thresh=nms_thresh)
    if name.startswith("yunet"):
        return YUNET(model_file, nms_thresh=nms_thresh)
    if name.startswith("yolo5face"):
        return YOLO5FACE(model_file, nms_thresh=nms_thresh)
    if name.startswith("retinaface"):
        return RETINAFACE(model_file, nms_thresh=nms_thresh)
    raise ValueError(f"Unknown detector type from filename: {model_file.name}")


def evaluate_onnx_detector(
    detector: "OnnxDetector",
    *,
    ann_file: Path,
    img_prefix: Path,
    gt_dir: Path,
    output_dir: Path,
    score_thresh: float,
    nms_thresh: float,
    iou_thresh: float,
    mode: str,
    limit_samples: int | None,
    save_preds: bool,
):
    detector.nms_thresh = nms_thresh
    dataset = WIDERFaceDataset(ann_file=ann_file, img_prefix=img_prefix, test_mode=True)
    if limit_samples is not None:
        dataset.records = dataset.records[:limit_samples]

    collect_start = perf_counter()
    predictions: PredictionDict = {}
    for sample in tqdm(dataset, desc="ONNX WIDERFace val"):
        boxes, _ = detector.detect(sample.image, score_thresh=score_thresh, mode=mode)
        widerface_boxes = xyxy_score_to_xywh_score(boxes)
        add_prediction(predictions, sample.filename, widerface_boxes)
    collect_seconds = perf_counter() - collect_start
    print(f"Collected predictions for {len(dataset)} images in {collect_seconds:.2f}s", flush=True)
    print_timing(detector)

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_preds:
        write_start = perf_counter()
        write_widerface_predictions(predictions, output_dir / "predictions")
        print(f"Wrote WIDER Face predictions in {perf_counter() - write_start:.2f}s", flush=True)

    eval_start = perf_counter()
    print("Computing WIDER Face AP for easy/medium/hard splits...", flush=True)
    aps = wider_evaluation(predictions, gt_dir, iou_thresh=iou_thresh)
    print(f"Computed WIDER Face AP in {perf_counter() - eval_start:.2f}s", flush=True)
    (output_dir / "aps.txt").write_text(
        f"{aps.easy:.6f},{aps.medium:.6f},{aps.hard:.6f}\n",
        encoding="utf-8",
    )

    return aps


def run_single_image(
    detector: "OnnxDetector",
    *,
    image_path: Path,
    output_dir: Path,
    prefix: str,
    score_thresh: float,
    mode: str,
) -> Path:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    print(f"The original shape is: {image.shape[:-1]}")
    detector.detect(image, score_thresh=score_thresh, mode=mode)
    detector.time_engine.reset()
    start = perf_counter()
    boxes, keypoints = detector.detect(image, score_thresh=score_thresh, mode=mode)
    wall_time = perf_counter() - start
    print_timing(detector)
    if wall_time > 0:
        print(f"wall_time: {wall_time:.6f}s")
        print(f"wall_fps: {1 / wall_time:.2f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{prefix}_{mode.replace(',', '_')}_{image_path.name}"
    draw_detections(image, boxes, keypoints, output_file)
    print(f"output: {output_file}")
    return output_file


def print_timing(detector: "OnnxDetector") -> None:
    for line in detector.time_engine.summary_lines():
        print(line)


def resize_image(image: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    mode = mode.upper()
    if mode == "ORIGIN":
        return image, 1.0
    if mode == "AUTO":
        assign_h = ((image.shape[0] - 1) & -32) + 32
        assign_w = ((image.shape[1] - 1) & -32) + 32
        det_image = np.zeros((assign_h, assign_w, 3), dtype=np.uint8)
        det_image[: image.shape[0], : image.shape[1], :] = image
        return det_image, 1.0

    input_size = (640, 480) if mode == "VGA" else _parse_input_size(mode)
    long_side, short_side = max(input_size), min(input_size)
    if image.shape[1] > image.shape[0]:
        input_size = (long_side, short_side)
    else:
        input_size = (short_side, long_side)

    image_ratio = image.shape[0] / image.shape[1]
    model_ratio = input_size[1] / input_size[0]
    if image_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / image_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * image_ratio)

    det_scale = new_height / image.shape[0]
    resized = cv2.resize(image, (new_width, new_height))
    det_image = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_image[:new_height, :new_width, :] = resized
    return det_image, det_scale


def _parse_input_size(mode: str) -> tuple[int, int]:
    values = [int(value) for value in mode.split(",")]
    if len(values) != 2:
        raise ValueError(f"Invalid mode: {mode!r}")
    return values[0], values[1]


def xyxy_score_to_xywh_score(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xywh = boxes.astype(np.float32, copy=True)
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    keypoints: np.ndarray | None,
    output_file: Path,
) -> None:
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2, _ = box.astype(np.int32)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if keypoints is None:
            continue
        for point in keypoints[idx].reshape(-1, 2):
            x, y = point.astype(np.int32)
            cv2.circle(image, (x, y), 1, (255, 0, 0), 2)
    cv2.imwrite(str(output_file), image)


def nms(boxes: np.ndarray, thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    cv_boxes = boxes[:, :4].copy()
    cv_boxes[:, 2] = cv_boxes[:, 2] - cv_boxes[:, 0]
    cv_boxes[:, 3] = cv_boxes[:, 3] - cv_boxes[:, 1]
    keep = cv2.dnn.NMSBoxes(
        bboxes=cv_boxes.tolist(),
        scores=boxes[:, -1].tolist(),
        score_threshold=0.0,
        nms_threshold=thresh,
        eta=1,
        top_k=5000,
    )
    if len(keep) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(keep, dtype=np.int64).reshape(-1)


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for idx in range(0, distance.shape[1], 2):
        px = points[:, idx % 2] + distance[:, idx]
        py = points[:, idx % 2 + 1] + distance[:, idx + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class OnnxDetector:
    taskname = "detector"

    def __init__(self, model_file: Path, *, nms_thresh: float) -> None:
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        model = onnx.load(str(model_file))
        onnx.checker.check_model(model)
        self.session = onnxruntime.InferenceSession(str(model_file), providers=onnxruntime.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.time_engine = TimeEngine()

    def detect(self, image: np.ndarray, *, score_thresh: float, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError


class YUNET(OnnxDetector):
    taskname = "yunet"

    def __init__(self, model_file: Path, *, nms_thresh: float) -> None:
        super().__init__(model_file, nms_thresh=nms_thresh)
        self.strides = (8, 16, 32)
        self.keypoint_count = 5

    def forward(self, image: np.ndarray, score_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.time_engine.tic("forward_calc")
        input_size = tuple(image.shape[0:2][::-1])
        blob = np.transpose(image, (2, 0, 1)).astype(np.float32)[np.newaxis, ...].copy()
        self.time_engine.toc("forward_calc")

        self.time_engine.tic("forward_run")
        outputs = self.session.run(None, {self.input_name: blob})
        self.time_engine.toc("forward_run")

        self.time_engine.tic("forward_calc")
        scores, boxes, keypoints = [], [], []
        for level_idx, stride in enumerate(self.strides):
            cls_pred = outputs[level_idx].reshape(-1, 1)
            obj_pred = outputs[level_idx + len(self.strides)].reshape(-1, 1)
            reg_pred = outputs[level_idx + len(self.strides) * 2].reshape(-1, 4)
            kps_pred = outputs[level_idx + len(self.strides) * 3].reshape(-1, self.keypoint_count * 2)

            anchor_centers = np.stack(
                np.mgrid[: input_size[1] // stride, : input_size[0] // stride][::-1],
                axis=-1,
            )
            anchor_centers = (anchor_centers * stride).astype(np.float32).reshape(-1, 2)

            bbox_center = reg_pred[:, :2] * stride + anchor_centers
            bbox_wh = np.exp(reg_pred[:, 2:]) * stride
            boxes.append(
                np.stack(
                    [
                        bbox_center[:, 0] - bbox_wh[:, 0] / 2,
                        bbox_center[:, 1] - bbox_wh[:, 1] / 2,
                        bbox_center[:, 0] + bbox_wh[:, 0] / 2,
                        bbox_center[:, 1] + bbox_wh[:, 1] / 2,
                    ],
                    axis=-1,
                )
            )
            keypoints.append(
                np.concatenate(
                    [
                        kps_pred[:, [2 * idx, 2 * idx + 1]] * stride + anchor_centers
                        for idx in range(self.keypoint_count)
                    ],
                    axis=-1,
                )
            )
            scores.append(cls_pred * obj_pred)

        scores_array = np.concatenate(scores, axis=0).reshape(-1)
        boxes_array = np.concatenate(boxes, axis=0)
        keypoints_array = np.concatenate(keypoints, axis=0)
        keep = scores_array > score_thresh
        self.time_engine.toc("forward_calc")
        return boxes_array[keep], scores_array[keep], keypoints_array[keep]

    def detect(self, image: np.ndarray, *, score_thresh: float, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
        self.time_engine.tic("preprocess")
        det_image, det_scale = resize_image(image, mode)
        self.time_engine.toc("preprocess")

        boxes, scores, keypoints = self.forward(det_image, score_thresh)

        self.time_engine.tic("postprocess")
        boxes = boxes / det_scale
        keypoints = keypoints / det_scale
        detections = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_thresh)
        self.time_engine.toc("postprocess")
        return detections[keep], keypoints[keep]


class SCRFD(OnnxDetector):
    taskname = "scrfd"

    def __init__(self, model_file: Path, *, nms_thresh: float) -> None:
        super().__init__(model_file, nms_thresh=nms_thresh)
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self.batched = len(self.session.get_outputs()[0].shape) == 3
        self.use_kps = False
        self.num_anchors = 1
        output_count = len(self.session.get_outputs())
        if output_count == 6:
            self.fmc = 3
            self.strides = (8, 16, 32)
            self.num_anchors = 2
        elif output_count == 9:
            self.fmc = 3
            self.strides = (8, 16, 32)
            self.num_anchors = 2
            self.use_kps = True
        elif output_count == 10:
            self.fmc = 5
            self.strides = (8, 16, 32, 64, 128)
        elif output_count == 15:
            self.fmc = 5
            self.strides = (8, 16, 32, 64, 128)
            self.use_kps = True
        else:
            raise ValueError(f"Unsupported SCRFD output count: {output_count}")

    def forward(self, image: np.ndarray, score_thresh: float) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        self.time_engine.tic("forward_calc")
        input_size = tuple(image.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(image, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        self.time_engine.toc("forward_calc")

        self.time_engine.tic("forward_run")
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        self.time_engine.toc("forward_run")

        self.time_engine.tic("forward_calc")
        scores_list, boxes_list, keypoints_list = [], [], []
        input_height, input_width = blob.shape[2], blob.shape[3]
        for idx, stride in enumerate(self.strides):
            scores = outputs[idx][0] if self.batched else outputs[idx]
            bbox_preds = (outputs[idx + self.fmc][0] if self.batched else outputs[idx + self.fmc]) * stride
            if self.use_kps:
                kps_preds = (outputs[idx + self.fmc * 2][0] if self.batched else outputs[idx + self.fmc * 2]) * stride

            anchor_centers = self._anchor_centers(input_height // stride, input_width // stride, stride)
            pos_indices = np.where(scores >= score_thresh)[0]
            scores_list.append(scores[pos_indices])
            boxes_list.append(distance2bbox(anchor_centers, bbox_preds)[pos_indices])
            if self.use_kps:
                keypoints_list.append(distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))[pos_indices])

        self.time_engine.toc("forward_calc")
        return scores_list, boxes_list, keypoints_list

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        centers = self.center_cache.get(key)
        if centers is not None:
            return centers

        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
        if len(self.center_cache) < 100:
            self.center_cache[key] = centers
        return centers

    def detect(self, image: np.ndarray, *, score_thresh: float, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
        self.time_engine.tic("preprocess")
        det_image, det_scale = resize_image(image, mode)
        self.time_engine.toc("preprocess")

        scores_list, boxes_list, keypoints_list = self.forward(det_image, score_thresh)

        self.time_engine.tic("postprocess")
        scores = np.vstack(scores_list)
        boxes = np.vstack(boxes_list) / det_scale
        detections = np.hstack((boxes, scores)).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_thresh)
        keypoints = np.vstack(keypoints_list) / det_scale if self.use_kps and keypoints_list else None
        self.time_engine.toc("postprocess")
        return detections[keep], None if keypoints is None else keypoints[keep]


class YOLO5FACE(OnnxDetector):
    taskname = "yolo5face"

    def forward(self, image: np.ndarray, score_thresh: float) -> tuple[np.ndarray, np.ndarray]:
        self.time_engine.tic("forward_calc")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = np.transpose(image[np.newaxis, ...], (0, 3, 1, 2)).copy().astype(np.float32) / 255.0
        self.time_engine.toc("forward_calc")

        self.time_engine.tic("forward_run")
        outputs = self.session.run(None, {self.input_name: blob})[0].squeeze(0)
        self.time_engine.toc("forward_run")

        self.time_engine.tic("forward_calc")
        outputs = outputs[outputs[:, 4] > score_thresh]
        outputs[:, 15:] *= outputs[:, 4:5]
        outputs = outputs[outputs[:, 15] > score_thresh]
        boxes = outputs[:, :15].copy()
        boxes[:, 0] = outputs[:, 0] - outputs[:, 2] / 2
        boxes[:, 1] = outputs[:, 1] - outputs[:, 3] / 2
        boxes[:, 2] = outputs[:, 0] + outputs[:, 2] / 2
        boxes[:, 3] = outputs[:, 1] + outputs[:, 3] / 2
        scores = outputs[:, 15]
        self.time_engine.toc("forward_calc")
        return boxes, scores

    def detect(self, image: np.ndarray, *, score_thresh: float, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
        self.time_engine.tic("preprocess")
        det_image, det_scale = resize_image(image, mode)
        self.time_engine.toc("preprocess")

        boxes, scores = self.forward(det_image, score_thresh)

        self.time_engine.tic("postprocess")
        boxes = boxes / det_scale
        detections = np.hstack((boxes[:, :4], scores[:, None])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_thresh)
        self.time_engine.toc("postprocess")
        return detections[keep], boxes[keep, 4:]


class RETINAFACE(OnnxDetector):
    taskname = "retinaface"

    def __init__(self, model_file: Path, *, nms_thresh: float) -> None:
        super().__init__(model_file, nms_thresh=nms_thresh)
        self.priors_cache: np.ndarray | None = None

    def anchor_fn(self, shape: tuple[int, int]) -> np.ndarray:
        min_sizes_cfg = ((16, 32), (64, 128), (256, 512))
        steps = (8, 16, 32)
        feature_maps = [[ceil(shape[0] / step), ceil(shape[1] / step)] for step in steps]
        anchors = []
        for idx, feature_map in enumerate(feature_maps):
            for row, col in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes_cfg[idx]:
                    anchors.extend([(col + 0.5) * steps[idx] / shape[1], (row + 0.5) * steps[idx] / shape[0]])
                    anchors.extend([min_size / shape[1], min_size / shape[0]])
        return np.array(anchors, dtype=np.float32).reshape(-1, 4)

    def decode(self, loc: np.ndarray, priors: np.ndarray, variances: tuple[float, float]) -> np.ndarray:
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ),
            axis=1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landmarks(self, pred: np.ndarray, priors: np.ndarray, variances: tuple[float, float]) -> np.ndarray:
        return np.concatenate(
            [priors[:, :2] + pred[:, i : i + 2] * variances[0] * priors[:, 2:] for i in range(0, 10, 2)],
            axis=1,
        )

    def forward(self, image: np.ndarray, score_thresh: float, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.time_engine.tic("forward_calc")
        blob = image.astype(np.float32)
        blob -= (104, 117, 123)
        blob = np.transpose(blob[np.newaxis, ...], (0, 3, 1, 2)).copy()
        self.time_engine.toc("forward_calc")

        self.time_engine.tic("forward_run")
        loc, conf, landmarks = self.session.run(None, {self.input_name: blob})
        self.time_engine.toc("forward_run")

        self.time_engine.tic("forward_calc")
        scores = conf.squeeze(0)[:, 1]
        boxes = self.decode(loc.squeeze(0), priors, variances=(0.1, 0.2))
        landmarks = self.decode_landmarks(landmarks.squeeze(0), priors, variances=(0.1, 0.2))
        boxes = np.concatenate((boxes, landmarks), axis=1)
        _, _, height, width = blob.shape
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height
        keep = scores > score_thresh
        self.time_engine.toc("forward_calc")
        return boxes[keep], scores[keep]

    def detect(self, image: np.ndarray, *, score_thresh: float, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
        self.time_engine.tic("preprocess")
        det_image, det_scale = resize_image(image, mode)
        if mode.upper() in {"ORIGIN", "AUTO"}:
            priors = self.anchor_fn(det_image.shape[:2])
        else:
            if self.priors_cache is None:
                self.priors_cache = self.anchor_fn(det_image.shape[:2])
            priors = self.priors_cache
        self.time_engine.toc("preprocess")

        boxes, scores = self.forward(det_image, score_thresh, priors)

        self.time_engine.tic("postprocess")
        boxes = boxes / det_scale
        detections = np.hstack((boxes[:, :4], scores[:, None])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_thresh)
        self.time_engine.toc("postprocess")
        return detections[keep], boxes[keep, 4:]


if __name__ == "__main__":
    main()
