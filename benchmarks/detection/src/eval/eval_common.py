"""Shared evaluation code so every detector (RetinaFace, YOLO, ...) is scored
by exactly the same AP implementation and reporting format.

Used by evaluate_retinaface.py, evaluate_ultralytics.py, and evaluate_yunet.py.
"""
import json
import os
from collections import defaultdict

import numpy as np


def load_gt(csv_path):
    """CSV rows: filename,x1,y1,x2,y2[,label] -> {filename: [[x1,y1,x2,y2],...]}"""
    gt = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = (float(v) for v in parts[1:5])
            if x2 <= x1 or y2 <= y1:
                continue
            gt[parts[0]].append([x1, y1, x2, y2])
    return gt


def voc_ap(recall, precision):
    """All-points interpolated AP (VOC2010+ style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap(all_dets, gt, iou_thresh):
    """all_dets: {filename: (N,5) array-like [x1,y1,x2,y2,score]};
    gt: {filename: [[x1,y1,x2,y2],...]} -> (ap, recall, precision, n_gt)"""
    n_gt = sum(len(v) for v in gt.values())
    records = []  # (score, filename, box)
    for fname, dets in all_dets.items():
        for d in dets:
            records.append((float(d[4]), fname, d[:4]))
    records.sort(key=lambda r: -r[0])

    matched = {f: np.zeros(len(b), dtype=bool) for f, b in gt.items()}
    tp = np.zeros(len(records))
    fp = np.zeros(len(records))
    for i, (score, fname, box) in enumerate(records):
        gt_boxes = np.asarray(gt.get(fname, []), dtype=np.float64)
        if gt_boxes.size == 0:
            fp[i] = 1
            continue
        ix1 = np.maximum(gt_boxes[:, 0], box[0])
        iy1 = np.maximum(gt_boxes[:, 1], box[1])
        ix2 = np.minimum(gt_boxes[:, 2], box[2])
        iy2 = np.minimum(gt_boxes[:, 3], box[3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        area_d = max((box[2] - box[0]) * (box[3] - box[1]), 0)
        ious = inter / np.maximum(area_g + area_d - inter, 1e-9)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thresh and not matched[fname][j]:
            matched[fname][j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = voc_ap(recall, precision)
    return ap, recall, precision, n_gt


def per_image_match(all_dets, gt, iou_thresh):
    """Greedy per-image matching (identical to compute_ap, which only ever
    matches within an image) -> {fname: (scores desc, tp flags, n_gt)}."""
    per = {}
    for fname in set(gt) | set(all_dets):
        dets = np.asarray(all_dets.get(fname, []), dtype=np.float64).reshape(-1, 5)
        gt_boxes = np.asarray(gt.get(fname, []), dtype=np.float64).reshape(-1, 4)
        order = dets[:, 4].argsort()[::-1]
        dets = dets[order]
        matched = np.zeros(len(gt_boxes), dtype=bool)
        tp = np.zeros(len(dets))
        for i, d in enumerate(dets):
            if not len(gt_boxes):
                break
            ix1 = np.maximum(gt_boxes[:, 0], d[0])
            iy1 = np.maximum(gt_boxes[:, 1], d[1])
            ix2 = np.minimum(gt_boxes[:, 2], d[2])
            iy2 = np.minimum(gt_boxes[:, 3], d[3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            area_d = max((d[2] - d[0]) * (d[3] - d[1]), 0)
            ious = inter / np.maximum(area_g + area_d - inter, 1e-9)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thresh and not matched[j]:
                matched[j] = True
                tp[i] = 1
        per[fname] = (dets[:, 4], tp, len(gt_boxes))
    return per


def bootstrap_ap(all_dets, gt, iou_thresh, cluster_of, n_boot=1000, seed=0):
    """95%% bootstrap CI for AP, resampling clusters (e.g. films) with
    replacement. cluster_of: fname -> cluster id."""
    per = per_image_match(all_dets, gt, iou_thresh)
    clusters = defaultdict(lambda: [[], [], 0])
    for fname, (scores, tp, n_gt) in per.items():
        c = clusters[cluster_of(fname)]
        c[0].append(scores)
        c[1].append(tp)
        c[2] += n_gt
    packed = [(np.concatenate(s) if s else np.zeros(0),
               np.concatenate(t) if t else np.zeros(0), n)
              for s, t, n in clusters.values()]

    rng = np.random.default_rng(seed)
    n_clusters = len(packed)
    aps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_clusters, n_clusters)
        n_gt = sum(packed[i][2] for i in idx)
        if n_gt == 0:
            continue
        scores = np.concatenate([packed[i][0] for i in idx])
        tp = np.concatenate([packed[i][1] for i in idx])
        order = scores.argsort()[::-1]
        tp = tp[order]
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(voc_ap(recall, precision))
    if not aps:
        return None
    return float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


def report(tag, model_path, filenames, all_dets, gt, iou_thresh, conf_floor,
           size_desc, results_file, save_dets=None, elapsed=None,
           cluster=None, n_boot=1000):
    """Print the standard results block and append a row to the results table.

    elapsed: wall seconds spent in the detection loop (includes image
    read/decode, excludes model load); reported as images/second so speed
    sits next to accuracy in the table.
    cluster: bootstrap resampling unit - "dir" (top-level directory of the
    filename, e.g. one film) or "image"; None/n_boot=0 disables the CI.
    """
    ap, recall, precision, n_gt = compute_ap(all_dets, gt, iou_thresh)

    ci = None
    if cluster and n_boot:
        cluster_of = ((lambda f: f.split("/")[0]) if cluster == "dir"
                      else (lambda f: f))
        ci = bootstrap_ap(all_dets, gt, iou_thresh, cluster_of, n_boot=n_boot)
    final_r = float(recall[-1]) if recall.size else 0.0
    final_p = float(precision[-1]) if precision.size else 0.0
    ims = (len(filenames) / elapsed) if elapsed and len(filenames) else None

    print("=" * 60)
    print(f"[{tag}] {model_path}")
    print(f"images: {len(filenames)}  gt faces: {n_gt}  "
          f"detections: {sum(len(d) for d in all_dets.values())}")
    if ci:
        print(f"AP@{iou_thresh:.2f} = {ap * 100:.2f}%  "
              f"(95% CI [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}], "
              f"cluster={cluster}, n={n_boot})")
    else:
        print(f"AP@{iou_thresh:.2f} = {ap * 100:.2f}%")
    print(f"(at conf>={conf_floor}: recall {final_r * 100:.2f}%, "
          f"precision {final_p * 100:.2f}%)")
    if ims is not None:
        print(f"inference: {ims:.1f} im/s ({1000.0 / ims:.1f} ms/im, "
              f"{elapsed:.1f}s total)")
    print("=" * 60)

    ci_desc = (f"[{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]" if ci else "-")
    if not os.path.isfile(results_file) or os.path.getsize(results_file) == 0:
        with open(results_file, "w") as f:
            f.write("| tag | checkpoint | images | AP@0.5 (%) | 95% CI | "
                    "recall@0.02 (%) | im/s | protocol |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
    with open(results_file, "a") as f:
        f.write(f"| {tag} | {os.path.basename(str(model_path))} | "
                f"{len(filenames)} | {ap * 100:.2f} | {ci_desc} | "
                f"{final_r * 100:.2f} | "
                f"{f'{ims:.1f}' if ims is not None else '-'} | "
                f"{size_desc} |\n")

    if save_dets:
        serializable = {f: np.asarray(d).tolist() for f, d in all_dets.items()}
        with open(save_dets, "w") as f:
            json.dump(serializable, f)
        print("Saved detections to", save_dets)
    return ap


def nms(dets, thresh):
    """Greedy NMS over (N,5) [x1,y1,x2,y2,score]; returns kept row indices."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


# ---------------------------------------------------------------------------
# Standard eval sets: pass --eval_set icartoon|film to any evaluator
# ---------------------------------------------------------------------------

def add_eval_set_args(parser, repo_root):
    parser.add_argument("--eval-set", default=None,
                        choices=["icartoon_val", "icartoon_test", "film"],
                        help="icartoon_val: 5000 images held out from the "
                             "training set (checkpoint selection); "
                             "icartoon_test: the full 10000-image detval "
                             "release, held out for final numbers; "
                             "film: animated-film frames")
    parser.add_argument("--val-csv", default=str(repo_root / "data/icartoonface/detval.csv"))
    parser.add_argument("--images-root", default=str(repo_root / "data/icartoonface/detval"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.02)
    parser.add_argument("--num-images", type=int, default=0,
                        help="evaluate only the first N images (0 = all)")
    parser.add_argument("--test", action="store_true",
                        help="smoke-test mode: evaluate only 1000 images")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="bootstrap resamples for the AP 95%% CI (0 = off)")
    parser.add_argument("--cluster", default="auto",
                        choices=["auto", "image", "dir"],
                        help="bootstrap resampling unit; auto = dir (film) "
                             "for the film set, image otherwise")
    parser.add_argument("--tag", default="model")
    parser.add_argument("--save-dets", default=None)
    parser.add_argument("--results-file", default=str(repo_root / "results/results.md"))


def resolve_eval_set(args, repo_root):
    if getattr(args, "test", False) and args.num_images == 0:
        args.num_images = 1000
        print("--test: evaluating 1000 images")
    if args.cluster == "auto":
        args.cluster = "dir" if args.eval_set == "film" else "image"
    if args.eval_set == "icartoon_val":
        args.val_csv = str(repo_root / "labels/icf_val.csv")
        args.images_root = str(repo_root / "data/icartoonface/dettrain")
    elif args.eval_set == "icartoon_test":
        args.val_csv = str(repo_root / "data/icartoonface/detval.csv")
        args.images_root = str(repo_root / "data/icartoonface/detval")
    elif args.eval_set == "film":
        args.val_csv = str(repo_root / "labels/film_val.csv")
        args.images_root = str(repo_root / "data/film/images")
    import os as _os
    _os.makedirs(_os.path.dirname(args.results_file), exist_ok=True)
