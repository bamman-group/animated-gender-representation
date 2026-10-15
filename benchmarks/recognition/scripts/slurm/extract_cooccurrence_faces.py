#!/usr/bin/env python3
import os, sys, argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CROP_PADDING = 0.0



def load_tracks(tracks_path):
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



def sample_1fps(track_frames, fps):
    n = len(track_frames)
    n_samples = max(1, round(n / fps))
    if n_samples >= n:
        return list(track_frames)
    indices = {round(i * (n - 1) / (n_samples - 1))
               for i in range(n_samples)} if n_samples > 1 else {0}
    return [track_frames[i] for i in sorted(indices)]



def padded_crop(frame, x1, y1, x2, y2, pad=DEFAULT_CROP_PADDING):
    h, w  = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 - pad * bw))
    py1 = max(0, int(y1 - pad * bh))
    px2 = min(w, int(x2 + pad * bw))
    py2 = min(h, int(y2 + pad * bh))
    crop = frame[py1:py2, px1:px2]
    return crop if crop.size > 0 else frame



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

    multi_frames = sorted(fno for fno, faces in by_frame.items()
                          if len(faces) >= 2)
    print(f"Frames with 2+ faces: {len(multi_frames)}")

    used_tracks  = set()
    top_frames   = []
    for fno in multi_frames:
        tids = {face[0] for face in by_frame[fno]}
        if tids & used_tracks:
            continue
        top_frames.append((fno, tids))
        used_tracks.update(tids)

    print(f"Selected top_frames: {len(top_frames)}")

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
                cv2.imwrite(out_path, crop,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                saved += 1
        fno += 1
    cap.release()

    print(f"Saved {saved} crops to {args.output_dir}")


if __name__ == "__main__":
    main()
