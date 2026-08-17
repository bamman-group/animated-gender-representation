#!/usr/bin/env python3
"""
extract_cooccurrence_faces.py

Find frames where 2+ faces appear simultaneously, then for each such frame
extract face crops sampled at ~1fps from every track visible in that frame.

Output structure:
  {output_dir}/{movie_id}/{top_frame}/{track_no}/{frame_of_sample}.png
  (movie_id = the mp4's basename without extension)

Rules:
  - If any track from a candidate top_frame has already been used by an
    earlier top_frame, skip that top_frame entirely.
  - Crops are saved as uncompressed PNG (OpenCV cv2.imwrite).
  - Crop size: bbox expanded by --crop-padding fraction of its width/height on
    each side (0.0 = tight crop; 0.25 = 25% padding). This is baked into the
    output pixels, matching src/datasets/icartoonface.py's --crop-padding
    convention - the fine-tuning scripts read these already-cropped images and
    can't re-pad, so one extraction run per crop amount (e.g. the 0-crop and
    25-crop track dirs the SLURM jobs consume).

Usage
-----
python3 extract_cooccurrence_faces.py \
    --mp4          /path/to/{movie_id}.mp4 \
    --data-dir     /path/to/pipeline_data \
    --output-dir   paired_face_tracks_0crop \
    --crop-padding 0.0
"""

import os, sys, argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CROP_PADDING = 0.0   # fraction of bbox w/h added to each side


# ── loaders ───────────────────────────────────────────────────────────────────

def load_tracks(tracks_path):
    """
    Returns:
      by_frame : {frame_no: [(tid, face_no, x1, y1, x2, y2), ...]}
      by_track : {tid: [(frame_no, face_no, x1, y1, x2, y2), ...]}  sorted
    """
    by_frame = defaultdict(list)
    by_track = defaultdict(list)
    with open(tracks_path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            try:
                tid, fno, fac = int(p[0]), int(p[1]), int(p[2])
                x1, y1, x2, y2 = int(p[3]), int(p[4]), int(p[5]), int(p[6])
            except ValueError:
                continue
            by_frame[fno].append((tid, fac, x1, y1, x2, y2))
            by_track[tid].append((fno, fac, x1, y1, x2, y2))
    by_track = {tid: sorted(v) for tid, v in by_track.items()}
    return dict(by_frame), by_track


def load_fps(fps_path):
    with open(fps_path) as fh:
        return float(fh.read().strip().split("\t")[4])


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_1fps(track_frames, fps):
    """Evenly sample ~1 frame per second from track_frames."""
    n = len(track_frames)
    n_samples = max(1, round(n / fps))
    if n_samples >= n:
        return list(track_frames)
    indices = {round(i * (n - 1) / (n_samples - 1))
               for i in range(n_samples)} if n_samples > 1 else {0}
    return [track_frames[i] for i in sorted(indices)]


# ── cropping ──────────────────────────────────────────────────────────────────

def padded_crop(frame, x1, y1, x2, y2, pad=DEFAULT_CROP_PADDING):
    h, w  = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 - pad * bw))
    py1 = max(0, int(y1 - pad * bh))
    px2 = min(w, int(x2 + pad * bw))
    py2 = min(h, int(y2 + pad * bh))
    crop = frame[py1:py2, px1:px2]
    return crop if crop.size > 0 else frame


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mp4",        required=True,
                    help="Full path to the movie; its basename (minus .mp4) is the movie_id "
                         "and the stem for the fps/tracks files")
    ap.add_argument("--data-dir",   required=True,
                    help="Pipeline data root (fps/, tracks/ subdirs)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--crop-padding", type=float, default=DEFAULT_CROP_PADDING,
                    help="Fraction of the bbox width/height added on each side before cropping "
                         "(0.0 = tight; 0.25 = 25%% padding), baked into the saved PNGs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Crop padding: {args.crop_padding}")

    stem        = Path(args.mp4).stem
    fps_path    = os.path.join(args.data_dir, "fps",    f"{stem}.fps.txt")
    tracks_path = os.path.join(args.data_dir, "tracks", f"{stem}.tracks.txt")

    for label, path in [("fps",    fps_path),
                        ("tracks", tracks_path)]:
        if not os.path.exists(path):
            sys.exit(f"Missing {label} file: {path}")

    fps = load_fps(fps_path)
    print(f"FPS: {fps:.3f}")

    by_frame, by_track = load_tracks(tracks_path)
    print(f"Tracks: {len(by_track)}  Frames with faces: {len(by_frame)}")

    # Find frames with 2+ faces, sorted by frame number
    multi_frames = sorted(fno for fno, faces in by_frame.items()
                          if len(faces) >= 2)
    print(f"Frames with 2+ faces: {len(multi_frames)}")

    # Select top_frames, skipping any whose tracks overlap with already-used tracks
    used_tracks  = set()
    top_frames   = []   # (top_frame, [tid, ...])
    for fno in multi_frames:
        tids = {face[0] for face in by_frame[fno]}
        if tids & used_tracks:
            continue
        top_frames.append((fno, tids))
        used_tracks.update(tids)

    print(f"Selected top_frames: {len(top_frames)}")

    # Collect all (frame_no → list of (out_path, x1, y1, x2, y2)) for one video pass
    frame_targets = defaultdict(list)
    for top_frame, tids in top_frames:
        for tid in tids:
            if tid not in by_track:
                continue
            for fno, fac, x1, y1, x2, y2 in sample_1fps(by_track[tid], fps):
                out_dir  = os.path.join(args.output_dir, stem,
                                        str(top_frame), str(tid))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{fno}.png")
                frame_targets[fno].append((out_path, x1, y1, x2, y2))

    if not frame_targets:
        print("No crops to extract.")
        return

    # Single sequential video pass
    max_frame = max(frame_targets)
    cap       = cv2.VideoCapture(args.mp4)
    fno       = 0
    saved     = 0
    while fno <= max_frame:
        ok, bgr = cap.read()
        if not ok:
            break
        if fno in frame_targets:
            for out_path, x1, y1, x2, y2 in frame_targets[fno]:
                crop = padded_crop(bgr, x1, y1, x2, y2, pad=args.crop_padding)
                # PNG_COMPRESSION=0 → uncompressed
                cv2.imwrite(out_path, crop,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                saved += 1
        fno += 1
    cap.release()

    print(f"Saved {saved} crops to {args.output_dir}")


if __name__ == "__main__":
    main()
