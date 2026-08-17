"""Evaluate a RetinaFace checkpoint (shared protocol: VOC AP@0.5, 0.02 floor).

Examples (from the repo root):
    python -m src.eval.evaluate_retinaface --trained-model weights/Resnet50_Final.pth \
        --eval-set icartoon_test --tag retinaface_wf_icartoon_test
    python -m src.eval.evaluate_retinaface --trained-model runs/retinaface/wf_icf/Resnet50_Final.pth \
        --eval-set film --tta --tag retinaface_wf-icf_film-tta
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
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Pytorch_Retinaface"))

from src.eval.eval_common import add_eval_set_args, load_gt, report, resolve_eval_set  # noqa: E402
from data import cfg_mnet, cfg_re50  # noqa: E402
from layers.functions.prior_box import PriorBox  # noqa: E402
from models.retinaface import RetinaFace  # noqa: E402
from utils.box_utils import decode  # noqa: E402
from utils.nms.py_cpu_nms import py_cpu_nms  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--network", default="resnet50", choices=["mobile0.25", "resnet50"])
    parser.add_argument("--max-size", type=int, default=640,
                        help="longest image side after resize (0 = original size)")
    parser.add_argument("--tta", action="store_true",
                        help="multi-scale (640/1100/1600) + horizontal-flip TTA")
    parser.add_argument("--tta-scales", default="640,1100,1600")
    parser.add_argument("--nms-threshold", type=float, default=0.4)
    parser.add_argument("--cpu", action="store_true")
    add_eval_set_args(parser, REPO_ROOT)
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def forward_pass(net, device, cfg, img_raw, max_size, flip):
    """One pass; returns pre-NMS (N,5) [x1,y1,x2,y2,score] in original coords."""
    orig_w = img_raw.shape[1]
    if flip:
        img_raw = cv2.flip(img_raw, 1)
    img = np.float32(img_raw)
    resize = 1.0
    if max_size > 0:
        resize = max_size / max(img.shape[0], img.shape[1])
        if resize != 1.0:
            img = cv2.resize(img, None, fx=resize, fy=resize,
                             interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.tensor([im_width, im_height, im_width, im_height],
                         dtype=torch.float32, device=device)
    img -= (104, 117, 123)
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    loc, conf, _ = net(img)

    priors = PriorBox(cfg, image_size=(im_height, im_width)).forward().to(device)
    boxes = decode(loc.data.squeeze(0), priors.data, cfg["variance"])
    boxes = (boxes * scale / resize).cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    keep = scores > args.confidence_threshold
    boxes, scores = boxes[keep], scores[keep]
    if flip:
        x1 = orig_w - 1 - boxes[:, 2]
        x2 = orig_w - 1 - boxes[:, 0]
        boxes[:, 0], boxes[:, 2] = x1, x2
    return np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)


def detect_image(net, device, cfg, img_raw):
    sizes = ([int(s) for s in args.tta_scales.split(",")]
             if args.tta else [args.max_size])
    flips = [False, True] if args.tta else [False]
    passes = [forward_pass(net, device, cfg, img_raw, size, flip)
              for size in sizes for flip in flips]
    dets = np.vstack(passes)
    dets = dets[dets[:, 4].argsort()[::-1]]
    keep = py_cpu_nms(dets, args.nms_threshold)
    return dets[keep]


def main():
    global args
    args = parse_args()
    torch.set_grad_enabled(False)
    resolve_eval_set(args, REPO_ROOT)

    cfg = dict(cfg_mnet if args.network == "mobile0.25" else cfg_re50)
    cfg["pretrain"] = False

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net = RetinaFace(cfg=cfg, phase="test")
    state = torch.load(args.trained_model, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(strip_module_prefix(state))
    net.eval().to(device)
    print(f"Loaded {args.trained_model} on {device}")

    gt = load_gt(args.val_csv)
    filenames = sorted(gt.keys())
    if args.num_images > 0:
        filenames = filenames[:args.num_images]
        gt = {f: gt[f] for f in filenames}

    all_dets = {}
    t0 = time.time()
    for i, fname in enumerate(filenames):
        img = cv2.imread(os.path.join(args.images_root, fname), cv2.IMREAD_COLOR)
        if img is None:
            print("WARNING: unreadable image", fname)
            all_dets[fname] = np.zeros((0, 5), dtype=np.float32)
            continue
        all_dets[fname] = detect_image(net, device, cfg, img)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(filenames)} images "
                  f"({(i + 1) / (time.time() - t0):.1f} im/s)")

    elapsed = time.time() - t0

    size_desc = args.tta_scales + "+flip" if args.tta else str(args.max_size)
    report(args.tag, args.trained_model, filenames, all_dets, gt,
           args.iou_threshold, args.confidence_threshold, size_desc,
           args.results_file, args.save_dets, elapsed=elapsed,
           cluster=args.cluster, n_boot=args.bootstrap)


if __name__ == "__main__":
    main()
