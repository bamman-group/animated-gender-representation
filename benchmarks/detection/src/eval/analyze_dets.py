"""Failure analysis for saved detections (--save_dets output) against a GT CSV.

Reports recall and AP broken down by GT face size, the score threshold that
maximizes F1, and precision/recall/F1 swept across every score threshold (step
--pr-step, default 0.01) so an operating point can be chosen by eye. Points at
what the model is missing (e.g. small faces) rather than giving one pooled
number.

Usage:
    python -m src.eval.analyze_dets \
        --dets results/dets_retinaface_icf_icartoon_test.json \
        --val-csv data/icartoonface/detval.csv
"""
import argparse
import json
from collections import defaultdict

import numpy as np

BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 100000)]


def load_gt(csv_path):
    gt = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 5:
                continue
            x1, y1, x2, y2 = (float(v) for v in p[1:5])
            if x2 > x1 and y2 > y1:
                gt[p[0]].append([x1, y1, x2, y2])
    return gt


def match_all(all_dets, gt, iou_thresh):
    """Greedy match like the evaluator; returns per-GT matched score (or None)
    and per-detection (score, is_tp)."""
    records = []
    for fname, dets in all_dets.items():
        for d in dets:
            records.append((float(d[4]), fname, np.asarray(d[:4], dtype=np.float64)))
    records.sort(key=lambda r: -r[0])

    gt_match_score = {f: [None] * len(b) for f, b in gt.items()}
    det_flags = []
    for score, fname, box in records:
        boxes = np.asarray(gt.get(fname, []), dtype=np.float64)
        if boxes.size == 0:
            det_flags.append((score, False))
            continue
        ix1 = np.maximum(boxes[:, 0], box[0]); iy1 = np.maximum(boxes[:, 1], box[1])
        ix2 = np.minimum(boxes[:, 2], box[2]); iy2 = np.minimum(boxes[:, 3], box[3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area_g = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_d = (box[2] - box[0]) * (box[3] - box[1])
        ious = inter / np.maximum(area_g + area_d - inter, 1e-9)
        # best unmatched GT
        order = np.argsort(-ious)
        hit = False
        for j in order:
            if ious[j] < iou_thresh:
                break
            if gt_match_score[fname][j] is None:
                gt_match_score[fname][j] = score
                hit = True
                break
        det_flags.append((score, hit))
    return gt_match_score, det_flags


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dets", required=True, help="json from evaluate --save-dets")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--pr-step", type=float, default=0.01,
                    help="score-threshold step for the precision/recall sweep")
    args = ap.parse_args()

    with open(args.dets) as f:
        all_dets = json.load(f)
    gt = load_gt(args.val_csv)

    # the dets file may cover a subset (evaluate --num_images); restrict GT to
    # the images that were actually evaluated
    gt = {f: b for f, b in gt.items() if f in all_dets}
    print(f"analyzing {len(gt)} images "
          f"({sum(len(b) for b in gt.values())} GT faces)\n")

    gt_match_score, det_flags = match_all(all_dets, gt, args.iou_threshold)

    # ---- recall by GT size (sqrt of box area, px at native resolution) ------
    print(f"{'GT size (px)':<14}{'faces':>8}{'detected':>10}{'recall':>9}"
          f"{'med score':>11}")
    for lo, hi in BUCKETS:
        n = hit = 0
        scores = []
        for fname, boxes in gt.items():
            for j, b in enumerate(boxes):
                size = ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5
                if not (lo <= size < hi):
                    continue
                n += 1
                s = gt_match_score[fname][j]
                if s is not None:
                    hit += 1
                    scores.append(s)
        med = f"{np.median(scores):.2f}" if scores else "-"
        label = f"{lo}-{hi if hi < 100000 else 'inf'}"
        print(f"{label:<14}{n:>8}{hit:>10}{hit / max(n, 1):>9.1%}{med:>11}")

    # ---- best-F1 operating point --------------------------------------------
    n_gt = sum(len(b) for b in gt.values())
    flags = sorted(det_flags, key=lambda r: -r[0])
    tp = np.cumsum([1 if h else 0 for _, h in flags])
    fp = np.cumsum([0 if h else 1 for _, h in flags])
    scores = np.array([s for s, _ in flags])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    i = int(np.argmax(f1))
    print(f"\ntotal GT faces: {n_gt}   detections: {len(flags)}")
    print(f"missed faces (no match at any score): "
          f"{n_gt - int(tp[-1])} ({(n_gt - int(tp[-1])) / n_gt:.1%})")
    print(f"best F1 = {f1[i]:.3f} at score >= {scores[i]:.3f} "
          f"(P={prec[i]:.3f}, R={rec[i]:.3f})")

    # ---- precision/recall/F1 at every score threshold -----------------------
    # scores is descending, so the k highest-scored detections are exactly
    # those with score >= t; tp[k-1] is their true-positive count. Reading the
    # cumulative curve at each step avoids re-matching per threshold.
    print(f"\nthresh   precision  recall     F1     TP  dets>=t")
    asc = scores[::-1]  # ascending, for searchsorted
    max_score = float(scores[0]) if len(scores) else 0.0
    # Step by integer index (not repeated += ) so the threshold lands exactly
    # on the 0.01 grid; repeated addition drifts and can mis-place a score
    # sitting exactly on a grid line.
    for si in range(int(max_score / args.pr_step) + 1):
        t = round(si * args.pr_step, 10)
        k = len(scores) - int(np.searchsorted(asc, t, side="left"))  # dets with score >= t
        if k > 0:
            tp_k = int(tp[k - 1])
            p = tp_k / k
            r = tp_k / n_gt
            fscore = 2 * p * r / max(p + r, 1e-9)
            print(f"{t:>6.2f}{p:>11.3f}{r:>9.3f}{fscore:>8.3f}{tp_k:>7}{k:>9}")


if __name__ == "__main__":
    main()
