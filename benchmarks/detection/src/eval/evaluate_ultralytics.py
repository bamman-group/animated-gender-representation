"""Evaluate an Ultralytics checkpoint (YOLO26, RT-DETR, or a YOLOE
open-vocabulary model via --text-prompt) with the shared protocol.

TTA is implemented explicitly (multi-scale 640/1100/1600 + horizontal flip,
merged with one NMS) - the same recipe as the RetinaFace and YuNet
evaluators - because ultralytics' built-in `augment=True` is unsupported by
some architectures (RT-DETR, YOLOE) and silently reverts to single-scale.

Examples (from the repo root):
    python -m src.eval.evaluate_ultralytics \
        --weights runs/ultralytics/yolo26_wf_icf/weights/best.pt \
        --eval-set icartoon_test --tag yolo26_wf-icf_icartoon_test
    python -m src.eval.evaluate_ultralytics --weights yoloe-26l-seg.pt \
        --text-prompt "cartoon face" --eval-set film --tag yoloe_film
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.eval.eval_common import add_eval_set_args, load_gt, report, resolve_eval_set  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Ultralytics .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="inference size; matches RetinaFace --max-size 640")
    parser.add_argument("--tta", action="store_true",
                        help="explicit multi-scale + horizontal-flip TTA, merged NMS")
    parser.add_argument("--tta-scales", default="640,1100,1600")
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--text-prompt", default=None,
                        help="comma-separated class prompt(s) for a YOLOE checkpoint; "
                             "any detection of any prompted class counts as a face")
    add_eval_set_args(parser, REPO_ROOT)
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    resolve_eval_set(args, REPO_ROOT)

    multi_prompt = False
    if args.text_prompt:
        from ultralytics import YOLOE
        model = YOLOE(args.weights)
        names = [c.strip() for c in args.text_prompt.split(",")]
        model.set_classes(names, model.get_text_pe(names))
        # with several prompts the same face can match more than one class;
        # class-agnostic NMS collapses those duplicates (which would
        # otherwise count as false positives)
        multi_prompt = len(names) > 1
    else:
        from ultralytics import YOLO
        model = YOLO(args.weights)

    gt = load_gt(args.val_csv)
    filenames = sorted(gt.keys())
    if args.num_images > 0:
        filenames = filenames[:args.num_images]
        gt = {f: gt[f] for f in filenames}

    from src.eval.eval_common import nms

    def one_pass(img, size, flip):
        """One predict call; returns (N,5) in original image coords."""
        src = cv2.flip(img, 1) if flip else img
        r = model.predict(src, imgsz=size,
                          conf=args.confidence_threshold,
                          iou=args.nms_threshold,
                          agnostic_nms=multi_prompt,
                          max_det=1000, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()[:, None]
        if flip:
            w = img.shape[1]
            x1 = w - 1 - xyxy[:, 2]
            x2 = w - 1 - xyxy[:, 0]
            xyxy[:, 0], xyxy[:, 2] = x1.copy(), x2.copy()
        return np.hstack([xyxy, conf]).astype(np.float32)

    sizes = ([int(v) for v in args.tta_scales.split(",")]
             if args.tta else [args.imgsz])
    flips = [False, True] if args.tta else [False]

    all_dets = {}
    t0 = time.time()
    for i, fname in enumerate(filenames):
        img = cv2.imread(os.path.join(args.images_root, fname), cv2.IMREAD_COLOR)
        if img is None:
            print("WARNING: unreadable image", fname)
            all_dets[fname] = np.zeros((0, 5), dtype=np.float32)
            continue
        passes = [one_pass(img, s_, f_) for s_ in sizes for f_ in flips]
        dets = np.vstack(passes)
        if len(dets) > 1 and len(passes) > 1:
            dets = dets[nms(dets, args.nms_threshold)]
        all_dets[fname] = dets
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(filenames)} images "
                  f"({(i + 1) / (time.time() - t0):.1f} im/s)")

    elapsed = time.time() - t0

    size_desc = (args.tta_scales + "+flip") if args.tta else str(args.imgsz)
    if args.text_prompt:
        size_desc += f" prompt='{args.text_prompt}'"
    report(args.tag, args.weights, filenames, all_dets, gt,
           args.iou_threshold, args.confidence_threshold, size_desc,
           args.results_file, args.save_dets, elapsed=elapsed,
           cluster=args.cluster, n_boot=args.bootstrap)


if __name__ == "__main__":
    main()
