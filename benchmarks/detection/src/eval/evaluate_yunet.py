"""Evaluate a YuNet checkpoint (libfacedetection.train .pth) with the shared
protocol. The yunet_train package is imported directly from the vendored
third_party copy (no pip install needed).

Inference mirrors the upstream eval pipeline: BGR input, no mean/std
normalization, keep-ratio resize + zero padding, YuNetPostprocessor decode.
TTA = multi-scale (640/1100/1600 longest side) + horizontal flip, merged with
one NMS — the same recipe as evaluate_retinaface.py.

Examples (from the repo root):
    python -m src.eval.evaluate_yunet \
        --checkpoint third_party/libfacedetection.train/weights/yunet_n.pth \
        --eval-set icartoon_test --tag yunet_wf_icartoon_test
    python -m src.eval.evaluate_yunet --checkpoint runs/yunet/wf_icf/best_loss.pth \
        --eval-set film --tta --tag yunet_wf-icf_film-tta
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
# import yunet_train straight from the vendored copy; its pyproject is not
# installable with setuptools >= 77 (deprecated license classifier)
sys.path.insert(0, str(REPO_ROOT / "third_party" / "libfacedetection.train"))

from src.eval.eval_common import add_eval_set_args, load_gt, nms, report, resolve_eval_set  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="YuNet .pth checkpoint")
    parser.add_argument("--variant", default=None, choices=[None, "yunet_n", "yunet_s"])
    parser.add_argument("--max-size", type=int, default=640,
                        help="longest image side after resize (0 = original size)")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-scales", default="640,1100,1600")
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--cpu", action="store_true")
    add_eval_set_args(parser, REPO_ROOT)
    return parser.parse_args()


SIZE_DIVISOR = 32


def load_model(device):
    from yunet_train.tasks.face import build_yunet

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = (args.variant
               or checkpoint.get("config", {}).get("variant", "yunet_n"))
    model = build_yunet(variant)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), variant


def pad_to_divisor(img):
    h, w = img.shape[:2]
    ph = (SIZE_DIVISOR - h % SIZE_DIVISOR) % SIZE_DIVISOR
    pw = (SIZE_DIVISOR - w % SIZE_DIVISOR) % SIZE_DIVISOR
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return img


def forward_pass(model, postprocessor, device, img_raw, max_size, flip):
    orig_w = img_raw.shape[1]
    img = cv2.flip(img_raw, 1) if flip else img_raw
    resize = 1.0
    if max_size > 0:
        resize = max_size / max(img.shape[0], img.shape[1])
        if resize != 1.0:
            img = cv2.resize(img, None, fx=resize, fy=resize,
                             interpolation=cv2.INTER_LINEAR)
    img = pad_to_divisor(img)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        result = postprocessor(model(tensor))[0]
    boxes = result.boxes.detach().cpu().float().numpy()  # xyxy, resized coords
    scores = result.scores.detach().cpu().float().numpy()
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes /= resize
    if flip:
        x1 = orig_w - 1 - boxes[:, 2]
        x2 = orig_w - 1 - boxes[:, 0]
        boxes[:, 0], boxes[:, 2] = x1.copy(), x2.copy()
    return np.hstack([boxes, scores[:, None]]).astype(np.float32)


def main():
    global args
    args = parse_args()
    resolve_eval_set(args, REPO_ROOT)
    from yunet_train.tasks.face import YuNetPostprocessor

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, variant = load_model(device)
    print(f"Loaded {args.checkpoint} ({variant}) on {device}")

    postprocessor = YuNetPostprocessor(
        score_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=-1,
    )

    gt = load_gt(args.val_csv)
    filenames = sorted(gt.keys())
    if args.num_images > 0:
        filenames = filenames[:args.num_images]
        gt = {f: gt[f] for f in filenames}

    sizes = ([int(s) for s in args.tta_scales.split(",")]
             if args.tta else [args.max_size])
    flips = [False, True] if args.tta else [False]

    all_dets = {}
    t0 = time.time()
    for i, fname in enumerate(filenames):
        img = cv2.imread(os.path.join(args.images_root, fname), cv2.IMREAD_COLOR)
        if img is None:
            print("WARNING: unreadable image", fname)
            all_dets[fname] = np.zeros((0, 5), dtype=np.float32)
            continue
        passes = [forward_pass(model, postprocessor, device, img, s, f)
                  for s in sizes for f in flips]
        dets = np.vstack(passes)
        if len(dets) > 1 and len(passes) > 1:
            dets = dets[nms(dets, args.nms_threshold)]
        all_dets[fname] = dets
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(filenames)} images "
                  f"({(i + 1) / (time.time() - t0):.1f} im/s)")

    elapsed = time.time() - t0

    size_desc = args.tta_scales + "+flip" if args.tta else str(args.max_size)
    report(args.tag, args.checkpoint, filenames, all_dets, gt,
           args.iou_threshold, args.confidence_threshold, size_desc,
           args.results_file, args.save_dets, elapsed=elapsed,
           cluster=args.cluster, n_boot=args.bootstrap)


if __name__ == "__main__":
    main()
