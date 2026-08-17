#!/usr/bin/env python3
"""Recognize characters in one animated film: segment it into shots, detect
faces in every frame, link them into within-shot tracks, embed each track,
name it from the film's cast list, and optionally write an annotated mp4.

Everything runs from this one script. Shot detection (TransNetV2 / TensorFlow)
runs in an isolated subprocess (see detect_shots / _SHOT_WORKER) because
TensorFlow and this script's PyTorch segfault when they share one process and
both touch CUDA; the subprocess is spawned with `python -c`, so no shell
wrapper is needed.

Given a still image instead of a video (any of .jpg/.jpeg/.png/.bmp/.webp/
.tif), it runs the shorter detect -> embed -> recognize path on the one frame -
no shots, no tracking, each detected face named on its own - and can write an
annotated image (same box/label style). --transnet is not needed for an image.

The stages:

  shots       TransNetV2 (isolated subprocess)                ($TRANSNET)
  detection   ultralytics RT-DETR fine-tuned on iCartoonFace  ($DETR_MODEL)
  embedding   DINOv2 ViT-L/14 fine-tuned on tracks            ($DINO_CHECKPOINT)
  crop pad    0.25                                            ($DINO_CROP_PADDING)
  min conf    0.5                                             ($MIN_TRACK_CONF)

The tracker is the Bochinski et al. 2017 IOU tracker, with a 3-frame
look-back, shot-boundary resets, a max-confidence track filter, and num_best
representation. The matcher takes a mean embedding per character, scores by
cosine similarity, and emits the top-k at/above a score floor as
"label:score". A per-frame duplicate-box filter (dedupe_detections) is
applied because RT-DETR is NMS-free; it is a detector cleanup, not part of
the tracking logic.

The video is decoded twice - once to detect (bboxes are small, so all of them
fit in memory), once to embed the faces of surviving tracks. Holding a crop
per face at full frame rate would not fit. --annotate adds a third pass.

Usage:
  python recognize_characters.py MOVIE.mp4 \
      --transnet models/transnetv2-weights \
      --detr models/rtdetr.icf.FINAL.pt \
      --dino models/dino_vitl14_crop25.FINAL.pth \
      --dinov2-dir third_party/dinov2 \
      --out out/

Each file is written as soon as its stage finishes, so a long run can be
inspected while it is still going (and a crash in a later stage does not throw
away the earlier ones). Under --out, for clip id <c>, in order:

  shots/<c>.scenes.txt              "<start> <end>" per shot
  faces_detected/<c>.faces_detected.txt
                                    frame_no, face_no, "x1 y1 w h conf"
                                    (width/height, conf last)
  tracks/<c>.tracks.txt             track_no, frame_no, face_no, x1, y1, x2, y2
  track_reps/<c>.insightface.txt    track_no, frame_no, face_no, <vector>
  recog/<c>.recog.txt               track_no, frame_no, face_no,
                                    "label:score label:score ..."
  annotated_movies/<c>.annotated.mp4  with --annotate

Requires: torch, ultralytics, opencv-python, pillow, numpy, transnetv2, and a
local clone of facebookresearch/dinov2 (--dinov2-dir). See
pipeline/environment.yml and pipeline/scripts/setup.sh.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Roughly how many progress lines each long stage prints over a whole run,
# regardless of film length (so a feature isn't hundreds of lines).
PROGRESS_REPORTS = 20
ARCH_EMBED_DIM = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}

_START = time.time()


def log(message, end="\n"):
    """Progress line stamped with elapsed wall time. Flushed on every call so
    output still arrives in order when stdout is a pipe or a log file."""
    print(f"[{time.time() - _START:7.1f}s] {message}", end=end, flush=True)


def wrote(path):
    size = path.stat().st_size
    log(f"  -> wrote {path.name} ({size:,} bytes)")


# --------------------------------------------------------------------------
# Embedding model (mirrors benchmarks/recognition/src/models/dino_backbone.py)
# --------------------------------------------------------------------------
class DinoFaceModel(nn.Module):
    def __init__(self, backbone, embed_dim, proj_dim):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = proj_dim if proj_dim > 0 else embed_dim
        self.proj = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, proj_dim)
            )
            if proj_dim > 0
            else None
        )

    def forward(self, x):
        feats = self.backbone(x)
        if self.proj is not None:
            feats = self.proj(feats)
        return F.normalize(feats, dim=1)


def load_dino(checkpoint_path, dinov2_dir, device):
    """Loads a self-contained checkpoint written by
    benchmarks/recognition/src/train_dino.py -- the fine-tuned weights are in
    the checkpoint, so no pretrain .pth download is needed."""
    if dinov2_dir not in sys.path:
        sys.path.insert(0, dinov2_dir)
    import hubconf  # from the local dinov2 clone

    state = torch.load(checkpoint_path, map_location="cpu")
    arch, proj_dim = state["arch"], state["proj_dim"]
    backbone = getattr(hubconf, f"dinov2_{arch}")(pretrained=False)
    model = DinoFaceModel(backbone, ARCH_EMBED_DIM[arch], proj_dim)
    model.load_state_dict(state["model"])
    return model.eval().to(device), state


def pad_bbox(bbox, width, height, padding):
    """Grows a bbox by `padding` fraction of its own width/height on each
    side, clipped to the image (matches
    benchmarks/recognition/src/datasets/icartoonface.py)."""
    x1, y1, x2, y2 = bbox
    pad_w, pad_h = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, int(round(x1 - pad_w))),
        max(0, int(round(y1 - pad_h))),
        min(width, int(round(x2 + pad_w))),
        min(height, int(round(y2 + pad_h))),
    )


# --------------------------------------------------------------------------
# Stage 1: shots
# --------------------------------------------------------------------------
def read_scenes(scenes_file):
    """Reads back a "<start> <end>" per line scenes file, so an existing one
    can be reused instead of recomputed."""
    shots = []
    with open(scenes_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                shots.append((int(parts[0]), int(parts[1])))
    return shots


def read_faces_detected(faces_file):
    """Reads back a faces_detected file ("<frame>\\t<face_no>\\t<x1 y1 w h conf>")
    written by an earlier run. Returns (detections, face_nos) in the same shape
    detect_faces produces, so a resumed run is indistinguishable from a fresh
    one. Boxes are stored as x1 y1 w h on disk and converted back to x1 y1 x2 y2
    here. These detections were already deduped before being written."""
    detections, face_nos = [], []
    with open(faces_file, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            x1, y1, w, h, conf = cols[2].split()
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            detections.append((int(cols[0]), (x1, y1, x1 + w, y1 + h), float(conf)))
            face_nos.append(int(cols[1]))
    return detections, face_nos


def read_tracks_file(tracks_file):
    """Reads back a tracks file (track_no, frame_no, face_no, x1, y1, x2, y2).
    Returns {(frame_no, face_no): track_no}. Only surviving tracks were
    written, so any detection missing from this map was dropped by the
    tracker's filters."""
    assignment = {}
    with open(tracks_file, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            assignment[(int(cols[1]), int(cols[2]))] = int(cols[0])
    return assignment


def read_track_reps(reps_file):
    """Reads back a track reps file (track_no, frame_no, face_no, <floats>).
    Returns (ordered_track_nos, embeddings) with embeddings L2-normalized."""
    track_nos, vectors = [], []
    with open(reps_file, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            track_nos.append(int(cols[0]))
            vectors.append(np.fromstring(cols[3], dtype=np.float32, sep=" "))
    if not vectors:
        raise SystemExit(f"No track reps read from {reps_file}")
    return track_nos, F.normalize(torch.from_numpy(np.stack(vectors)), dim=1)


# Shot detection runs TransNetV2 (TensorFlow) in a SUBPROCESS, never in this
# process: TensorFlow and PyTorch segfault when they share one process and both
# initialize CUDA. The subprocess below imports only TensorFlow and TransNetV2 - never
# torch - runs shot detection on the GPU (memory growth on, so it does not
# reserve the whole card), writes the scenes file, and exits, leaving this
# process TF-free for detection and embedding. It is spawned with `python -c`,
# so the whole pipeline is still one script and needs no shell wrapper.
_SHOT_WORKER = r'''
import sys
import tensorflow as tf
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass
from transnetv2 import TransNetV2
video_path, weights, scenes_out = sys.argv[1], sys.argv[2], sys.argv[3]
model = TransNetV2(model_dir=weights)
_, single_frame_predictions, _ = model.predict_video(video_path)
scenes = model.predictions_to_scenes(single_frame_predictions)
with open(scenes_out, "w", encoding="utf-8") as f:
    for start, end in scenes:
        f.write("%d %d\n" % (int(start), int(end)))
'''


def detect_shots(video_path, transnet_weights, scenes_out):
    """Runs TransNetV2 shot detection in a subprocess (see _SHOT_WORKER), which
    writes scenes_out directly; returns the parsed [(start, end), ...]. Keeping
    TensorFlow in its own process is what prevents the TF/PyTorch CUDA segfault."""
    subprocess.run(
        [sys.executable, "-c", _SHOT_WORKER,
         str(video_path), str(transnet_weights), str(scenes_out)],
        check=True,
    )
    return read_scenes(scenes_out)


# --------------------------------------------------------------------------
# Stage 2: detect faces in every frame
# --------------------------------------------------------------------------
def detect_faces(video_path, detr_model, min_conf, device, batch_size, stride=1, max_frames=0,
                 progress_every=500):
    """Runs the detector over the video, keeping only bboxes and scores.
    Returns (detections, n_frames) where detections is
    [(frame_index, (x1, y1, x2, y2), conf), ...] in frame order.

    stride defaults to 1 (every frame). Raising it trades recall for speed."""
    from ultralytics import RTDETR

    detector = RTDETR(detr_model)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total_frames = min(total_frames, max_frames * stride)

    detections = []
    batch, batch_indices = [], []

    def flush():
        if not batch:
            return
        for frame_index, result in zip(
            batch_indices, detector.predict(batch, conf=min_conf, device=device, verbose=False)
        ):
            for box in result.boxes:
                x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
                detections.append((frame_index, (x1, y1, x2, y2), float(box.conf[0])))
        batch.clear()
        batch_indices.clear()

    started = time.time()
    # Report ~PROGRESS_REPORTS times over the whole run, not on a fixed cadence,
    # so a feature-length film prints a couple dozen lines rather than hundreds.
    report_every = max(progress_every, (total_frames // stride) // PROGRESS_REPORTS) \
        if total_frames else progress_every
    frame_index, sampled, last_report = 0, 0, 0
    while True:
        if not cap.grab():
            break
        if frame_index % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            batch.append(frame)
            batch_indices.append(frame_index)
            if len(batch) >= batch_size:
                flush()
            sampled += 1
            if sampled - last_report >= report_every:
                last_report = sampled
                rate = sampled / max(time.time() - started, 1e-6)
                eta = (total_frames / stride - sampled) / rate if total_frames else 0
                log(f"  frame {frame_index:,}"
                    + (f"/{total_frames:,}" if total_frames else "")
                    + f"  {len(detections):,} faces  {rate:.1f} frame/s"
                    + (f"  eta {eta / 60:.1f} min" if total_frames else ""))
            if max_frames and sampled >= max_frames:
                break
        frame_index += 1
    flush()
    cap.release()
    return detections, frame_index


# --------------------------------------------------------------------------
# Stage 3: link detections into tracks (greedy IoU, never across a shot)
# --------------------------------------------------------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def dedupe_detections(detections, iou_threshold=0.5, containment_threshold=0.7):
    """Drops duplicate boxes on the same face within a frame, keeping the
    highest-confidence one.

    RT-DETR is NMS-free (one-to-one label assignment), so ultralytics applies
    no duplicate suppression, and on animated faces it readily emits two
    nested boxes for one face - a tight one and a taller one including hair or
    chin. Left in, they alternate frame to frame and the tracker splits one
    character into two flip-flopping tracks.

    Two boxes are duplicates if their IoU is >= iou_threshold OR one is mostly
    inside the other (intersection / smaller area >= containment_threshold);
    the containment test is what catches the nested case, whose IoU can sit
    below any threshold loose enough to be safe.
    """
    by_frame = {}
    for detection in detections:
        by_frame.setdefault(detection[0], []).append(detection)

    kept = []
    for frame_index in sorted(by_frame):
        candidates = sorted(by_frame[frame_index], key=lambda d: -d[2])  # confident first
        survivors = []
        for frame_idx, bbox, conf in candidates:
            duplicate = False
            for _, kept_bbox, _ in survivors:
                if iou(bbox, kept_bbox) >= iou_threshold:
                    duplicate = True
                    break
                ax1, ay1, ax2, ay2 = bbox
                bx1, by1, bx2, by2 = kept_bbox
                inter = (max(0, min(ax2, bx2) - max(ax1, bx1))
                         * max(0, min(ay2, by2) - max(ay1, by1)))
                smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
                if smaller > 0 and inter / smaller >= containment_threshold:
                    duplicate = True
                    break
            if not duplicate:
                survivors.append((frame_idx, bbox, conf))
        kept.extend(sorted(survivors, key=lambda d: (d[1][0], d[1][1])))
    return kept


def build_tracks(detections, boundaries, min_iou, min_track_length, min_track_conf):
    """The Bochinski et al. 2017 "Tracking-by-Detection Without Using Image
    Information" IOU tracker.

    Returns a per-detection track id, or None where the detection's track was
    dropped (None tracks are skipped when the tracks file is written).

    Rules:
      - Faces are processed in contiguous frame order (every-frame detection is
        assumed).
      - Frame 0 and every shot-start frame in `boundaries` start fresh: each
        face there begins a new track.
      - Otherwise a face joins the track of the highest-IoU face (> min_iou)
        found by looking back up to 3 frames - i-1 always, i-2 if i-1 is not a
        boundary, i-3 if neither i-1 nor i-2 is - else it starts a new track.
        (This bounded look-back is what bridges a frame or two of missed
        detection; there is no persisted box or fixed gap window.)
      - A track is kept only if it has >= min_track_length faces AND its single
        highest-confidence face >= min_track_conf (note: max, not mean).
      - Surviving tracks are renumbered densely by order of first appearance.
    """
    by_frame = {}
    for i, (frame_index, _, _) in enumerate(detections):
        by_frame.setdefault(frame_index, []).append(i)
    if not by_frame:
        return []
    max_frame = max(by_frame)
    faces = [by_frame.get(f, []) for f in range(max_frame + 1)]  # det indices per frame

    def bbox(f, j):
        return detections[faces[f][j]][1]

    def conf(f, j):
        return detections[faces[f][j]][2]

    tracks = {}                 # (frame, position) -> raw track id
    highest_conf = {}           # raw track id -> max face confidence seen
    next_id = 0

    def best_match(f, j, back):
        best, best_iou = None, min_iou
        for k in range(len(faces[f - back])):
            score = iou(bbox(f, j), bbox(f - back, k))
            if score > best_iou:
                best, best_iou = (f - back, k), score
        return best, best_iou

    for i in range(len(faces)):
        if i == 0 or i in boundaries:
            for j in range(len(faces[i])):
                tracks[(i, j)] = next_id
                highest_conf[next_id] = conf(i, j)
                next_id += 1
            continue
        for j in range(len(faces[i])):
            match, match_iou = best_match(i, j, 1)
            if i > 1 and (i - 1) not in boundaries:
                cand, cand_iou = best_match(i, j, 2)
                if cand is not None and cand_iou > match_iou:
                    match, match_iou = cand, cand_iou
            if i > 2 and (i - 1) not in boundaries and (i - 2) not in boundaries:
                cand, cand_iou = best_match(i, j, 3)
                if cand is not None and cand_iou > match_iou:
                    match, match_iou = cand, cand_iou

            if match is not None:
                tracks[(i, j)] = tracks[match]
            else:
                tracks[(i, j)] = next_id
                next_id += 1
            t, c = tracks[(i, j)], conf(i, j)
            highest_conf[t] = c if t not in highest_conf else max(highest_conf[t], c)

    members = {}
    for key in tracks:  # insertion order = frame/position order = creation order
        members.setdefault(tracks[key], []).append(key)
    mapper = {}
    for raw in members:
        if len(members[raw]) < min_track_length:
            continue
        if highest_conf[raw] < min_track_conf:
            continue
        mapper[raw] = len(mapper)

    track_ids = [None] * len(detections)
    for (i, j), raw in tracks.items():
        track_ids[faces[i][j]] = mapper.get(raw)  # None if track was dropped
    return track_ids


# --------------------------------------------------------------------------
# Stage 4: embed the faces of surviving tracks (second decode pass)
# --------------------------------------------------------------------------
@torch.no_grad()
def embed_tracks(video_path, detections, rows_by_track, model, device, padding, img_size,
                 batch_size, progress_every=200):
    """Decodes the video once more, embedding only the faces named in
    rows_by_track, and returns one L2-normalized mean embedding per track."""
    wanted = {}  # frame_index -> [(detection_index, track_position), ...]
    for track_position, indices in enumerate(rows_by_track):
        for det_index in indices:
            wanted.setdefault(detections[det_index][0], []).append((det_index, track_position))

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    sums = [None] * len(rows_by_track)
    counts = [0] * len(rows_by_track)
    crops, positions = [], []

    def flush():
        if not crops:
            return
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        embeddings = model((batch - mean) / std).cpu()
        for embedding, track_position in zip(embeddings, positions):
            sums[track_position] = (
                embedding if sums[track_position] is None else sums[track_position] + embedding
            )
            counts[track_position] += 1
        crops.clear()
        positions.clear()

    cap = cv2.VideoCapture(str(video_path))
    started = time.time()
    total_wanted = sum(len(v) for v in wanted.values())
    report_every = max(progress_every, total_wanted // PROGRESS_REPORTS)
    embedded, last_report = 0, 0
    frame_index = 0
    remaining = len(wanted)
    while remaining:
        if not cap.grab():
            break
        if frame_index in wanted:
            ok, frame = cap.retrieve()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for det_index, track_position in wanted[frame_index]:
                bbox = detections[det_index][1]
                crop = image.crop(pad_bbox(bbox, image.width, image.height, padding))
                crops.append(np.asarray(crop.resize((img_size, img_size), Image.BILINEAR),
                                        dtype=np.uint8))
                positions.append(track_position)
                embedded += 1
            if len(crops) >= batch_size:
                flush()
            if embedded - last_report >= report_every:
                last_report = embedded
                rate = embedded / max(time.time() - started, 1e-6)
                log(f"  face {embedded:,}/{total_wanted:,}  {rate:.1f} face/s"
                    f"  eta {(total_wanted - embedded) / rate / 60:.1f} min")
            remaining -= 1
        frame_index += 1
    flush()
    cap.release()

    if any(c == 0 for c in counts):
        raise SystemExit("Some tracks got no embedding (video shorter on the second pass?)")
    return F.normalize(torch.stack([s / c for s, c in zip(sums, counts)]), dim=1)


# --------------------------------------------------------------------------
# Single-image path: detect + embed straight from one loaded image, no shots,
# no tracking (each face stands alone).
# --------------------------------------------------------------------------
def detect_faces_image(image_bgr, detr_model, min_conf, device):
    """Runs the detector on one image. Returns [(0, (x1, y1, x2, y2), conf), ...]
    - the frame index is always 0, so the same downstream code (dedup, the
    faces file) works unchanged."""
    from ultralytics import RTDETR

    result = RTDETR(detr_model).predict(image_bgr, conf=min_conf, device=device,
                                        verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        detections.append((0, (x1, y1, x2, y2), float(box.conf[0])))
    return detections


@torch.no_grad()
def embed_faces_image(image_bgr, detections, model, device, padding, img_size, batch_size):
    """Embeds every detection's face crop from a single in-memory image.
    Returns one L2-normalized embedding per detection, in order (no averaging -
    each face is its own identity here)."""
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    embeddings = []
    for start in range(0, len(detections), batch_size):
        crops = []
        for _, bbox, _ in detections[start:start + batch_size]:
            crop = image.crop(pad_bbox(bbox, image.width, image.height, padding))
            crops.append(np.asarray(crop.resize((img_size, img_size), Image.BILINEAR),
                                    dtype=np.uint8))
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
        embeddings.append(model((batch - mean) / std).cpu())
    return F.normalize(torch.cat(embeddings), dim=1)


# --------------------------------------------------------------------------
# Stage 5: match tracks against the cast list
# --------------------------------------------------------------------------
def load_cast(cast_file):
    """Reads a cast_list/<movie_id>.tsv: one row per labelled exemplar face,

        movie_id  character_id  character_name  <1024 floats>  track_id  frame  x1 y1 x2 y2

    Returns (names, ids, embeddings) with embeddings L2-normalized (the
    released vectors already are, but normalizing again is free and makes the
    dot product below a cosine similarity regardless)."""
    names, ids, vectors = [], [], []
    with open(cast_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            ids.append(parts[1])
            names.append(parts[2])
            vectors.append(np.fromstring(parts[3], dtype=np.float32, sep=" "))
    if not vectors:
        raise SystemExit(f"No exemplars read from {cast_file}")
    embeddings = torch.from_numpy(np.stack(vectors))
    return names, ids, F.normalize(embeddings, dim=1)


def average_cast(cast):
    """Collapses the exemplars to one mean vector per character_id. The mean is
    deliberately left unnormalized; the cosine similarity below divides by the
    norms anyway."""
    names, ids, embeddings = cast
    by_character = {}
    for i, character_id in enumerate(ids):
        by_character.setdefault(character_id, [names[i], []])[1].append(embeddings[i])
    characters = []
    for character_id, (name, vectors) in by_character.items():
        characters.append((character_id, name, torch.stack(vectors).mean(dim=0)))
    return characters


def recognize_tracks(track_embeddings, cast, top_k=10, min_score=0.18):
    """Scores every track against every character's mean embedding and keeps
    the top_k at or above min_score.

    Returns [[(name, score), ...], ...] - one ranked list per track, highest
    first, possibly empty when nothing clears min_score."""
    characters = average_cast(cast)
    gallery = F.normalize(torch.stack([c[2] for c in characters]), dim=1)
    similarity = F.normalize(track_embeddings, dim=1) @ gallery.T  # cosine

    ranked_per_track = []
    for row in similarity:
        scored = [(characters[i][1], float(score)) for i, score in enumerate(row.tolist())]
        scored.sort(key=lambda x: -x[1])
        ranked_per_track.append([(n, s) for n, s in scored if s >= min_score][:top_k])
    return ranked_per_track


# --------------------------------------------------------------------------
# Stage 6: annotated mp4
# --------------------------------------------------------------------------
# Visual style: orange rounded boxes, filled label backgrounds. The label font
# is Pillow's built-in default; pass --font <path.ttf> for a specific typeface.
BOX_COLOR = (255, 147, 0)  # orange, RGB (PIL, not BGR)
LABEL_FILL_ALPHA = 90      # 0-255
TEXT_COLOR = (255, 255, 255)
BOX_WIDTH = 2
LABEL_PADDING = 4
DEFAULT_FONT = None        # None -> Pillow built-in font (ImageFont.load_default)
DEFAULT_FONT_SIZE = 18
DEFAULT_SMOOTH_WINDOW = 9


def smooth_track(frames, window):
    """frames: sorted [(frame_index, x1, y1, x2, y2), ...] for one track.
    Returns {frame_index: (x1, y1, x2, y2)} moving-averaged over `window`
    frames, so boxes don't jitter frame to frame."""
    boxes = np.array([f[1:] for f in frames], dtype=np.float32)
    half = window // 2
    smoothed = {}
    for i, (frame_index, *_) in enumerate(frames):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        smoothed[frame_index] = tuple(boxes[lo:hi].mean(axis=0).tolist())
    return smoothed


def build_frame_index(detections, track_ids, keep_tracks, labels, window):
    """Returns {frame_index: [(label, x1, y1, x2, y2), ...]} with each track's
    boxes smoothed and carrying its character label."""
    by_track = {}
    for i, (frame_index, (x1, y1, x2, y2), _) in enumerate(detections):
        if track_ids[i] in keep_tracks:
            by_track.setdefault(track_ids[i], []).append((frame_index, x1, y1, x2, y2))

    index = {}
    for track, frames in by_track.items():
        frames.sort()
        name, score = labels.get(track, (None, None))
        # Same label text parse_recog() builds: "<name> (<score>)", underscores
        # back to spaces.
        label = f"{name.replace('_', ' ')} ({score:.2f})" if name else ""
        for frame_index, box in smooth_track(frames, window).items():
            index.setdefault(frame_index, []).append((label, *box))
    return index


def _default_font(size):
    # Pillow >= 10.1 sizes the built-in font; older versions ignore the arg.
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def load_font(font_path, size):
    if font_path is None:
        return _default_font(size)
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        log(f"      (could not load {font_path} - using Pillow's default font)")
        return _default_font(size)


def draw_frame(frame_bgr, boxes_for_frame, font):
    """Draws orange rounded boxes with translucent filled label backgrounds,
    compositing through an RGBA overlay so the label fill is see-through."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for label, x1, y1, x2, y2 in boxes_for_frame:
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=4, outline=BOX_COLOR, width=BOX_WIDTH)
        if label:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lh = th + 2 * LABEL_PADDING
            lt = y1 - lh if y1 - lh > 0 else y2  # above the box, else below it
            lb = [x1, lt, x1 + tw + 2 * LABEL_PADDING, lt + lh]
            draw.rectangle(lb, fill=(*BOX_COLOR, LABEL_FILL_ALPHA))
            draw.text((lb[0] + LABEL_PADDING, lb[1] + LABEL_PADDING - tb[1]),
                      label, fill=TEXT_COLOR, font=font)

    return cv2.cvtColor(np.array(Image.alpha_composite(pil, overlay).convert("RGB")),
                        cv2.COLOR_RGB2BGR)


def has_audio_stream(video_path):
    """True if the source has at least one audio stream (ffprobe)."""
    if shutil.which("ffprobe") is None:
        return False
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return "audio" in probe.stdout


def mux_audio(silent_video, source_video, output_path):
    """Copies the annotated video and the source's audio into one file.

    cv2.VideoWriter writes video only, so the annotated frames are muxed here
    with the original soundtrack. The video stream is stream-copied (no
    re-encode, so the drawn boxes are untouched); audio is copied too, falling
    back to re-encoding as AAC if the source codec is not mp4-compatible.
    -shortest guards against an audio track that outruns the frames we wrote.
    """
    base = ["ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_video), "-i", str(source_video),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-shortest"]
    for audio_args in (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]):
        result = subprocess.run(base + audio_args + [str(output_path)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return audio_args[1]
    raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def annotate_video(video_path, output_path, detections, track_ids, keep_tracks, labels,
                   font_path=DEFAULT_FONT, font_size=DEFAULT_FONT_SIZE,
                   window=DEFAULT_SMOOTH_WINDOW, progress_every=500, with_audio=True):
    """Writes an mp4 with a box per detected face, labelled with the matched
    character name. Unmatched tracks get a box and no label.

    OpenCV draws the boxes into a silent temporary file, then ffmpeg muxes the
    source's audio onto it (see mux_audio). With --no-audio, a source that has
    no audio, or no ffmpeg on PATH, the silent file is kept as-is."""
    frame_index_map = build_frame_index(detections, track_ids, keep_tracks, labels, window)
    font = load_font(font_path, font_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    mux = with_audio and has_audio_stream(video_path) and shutil.which("ffmpeg") is not None
    if with_audio and not mux:
        log("      (no audio muxed: source has no audio track, or ffmpeg/ffprobe not on PATH)")
    # Draw into a sibling temp file when muxing, so the final path only ever
    # holds the finished (audio-bearing) video.
    write_target = (output_path.with_name(output_path.stem + ".silent" + output_path.suffix)
                    if mux else output_path)

    writer = cv2.VideoWriter(str(write_target), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {write_target}")

    report_every = max(progress_every, total // PROGRESS_REPORTS) if total else progress_every
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = frame_index_map.get(frame_index)
        if boxes:
            frame = draw_frame(frame, boxes, font)
        writer.write(frame)
        frame_index += 1
        if frame_index % report_every == 0:
            log(f"  annotated frame {frame_index:,}" + (f"/{total:,}" if total else ""))
    writer.release()
    cap.release()

    if mux:
        log("      muxing source audio with ffmpeg ...")
        try:
            codec = mux_audio(write_target, video_path, output_path)
            write_target.unlink()
            log(f"      audio muxed (-c:a {codec})")
        except RuntimeError as e:
            write_target.replace(output_path)  # keep the silent video rather than nothing
            log(f"      !! audio mux failed, keeping silent video: {e}")
    return frame_index


# --------------------------------------------------------------------------
# Single-image pipeline: detect -> embed -> recognize -> annotate. No shots,
# no tracking (each detected face is its own identity).
# --------------------------------------------------------------------------
def process_image(args):
    image_id = args.video.stem
    faces_path = args.out / "faces_detected" / f"{image_id}.faces_detected.txt"
    recog_path = args.out / "recog" / f"{image_id}.recog.txt"
    annotated_path = args.annotated_out or (
        args.out / "annotated_images" / f"{image_id}.annotated{args.video.suffix}")
    for path in (faces_path, recog_path, annotated_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    log(f"image_id={image_id}  device={args.device}  out={args.out}")
    image_bgr = cv2.imread(str(args.video))
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.video}")
    h, w = image_bgr.shape[:2]
    log(f"image {w}x{h}")

    # --- [1/4] detection --------------------------------------------------
    log(f"[1/4] Detecting faces (conf >= {args.min_conf}) ...")
    detections = detect_faces_image(image_bgr, args.detr, args.min_conf, args.device)
    if not detections:
        raise SystemExit("No faces detected.")
    n_raw = len(detections)
    detections = dedupe_detections(detections, args.dedup_iou, args.dedup_containment)
    if n_raw != len(detections):
        log(f"      dropped {n_raw - len(detections):,} duplicate boxes "
            f"({n_raw} -> {len(detections)})")
    # face_no is the index of the face within the (single) frame.
    with open(faces_path, "w", encoding="utf-8") as f:
        for face_no, (_, (x1, y1, x2, y2), conf) in enumerate(detections):
            f.write(f"0\t{face_no}\t{x1} {y1} {x2 - x1} {y2 - y1} {conf:.4f}\n")
    log(f"      {len(detections)} faces")
    wrote(faces_path)

    # --- [2/4] embeddings -------------------------------------------------
    log(f"[2/4] Embedding {len(detections)} faces ...")
    model, state = load_dino(args.dino, args.dinov2_dir, args.device)
    log(f"      {state['arch']}, trained on {state.get('data_source')}, "
        f"checkpoint crop_padding {state.get('crop_padding')} (using {args.crop_padding})")
    face_embeddings = embed_faces_image(image_bgr, detections, model, args.device,
                                        args.crop_padding, args.img_size, args.batch_size)

    # --- [3/4] recognition ------------------------------------------------
    cast_file = args.cast
    lookup_id = image_id.split("__")[0]
    if cast_file is None:
        candidates = sorted(args.cast_dir.glob(f"{lookup_id}*.tsv"))
        if candidates:
            cast_file = candidates[0]

    labels = {}  # face_no -> (name, score)
    if cast_file is None:
        log(f"[3/4] No cast list for '{lookup_id}' in {args.cast_dir} "
            f"- skipping recognition (pass --cast to name one).")
    else:
        log(f"[3/4] Recognizing faces against {Path(cast_file).name} ...")
        cast = load_cast(cast_file)
        log(f"      {len(cast[0]):,} exemplars, {len(set(cast[0]))} characters")
        # Same matcher as video; each face plays the role of one "track".
        ranked_per_face = recognize_tracks(face_embeddings, cast, args.top_k, args.min_score)
        with open(recog_path, "w", encoding="utf-8") as f:
            for face_no, ranked in enumerate(ranked_per_face):
                if ranked:
                    labels[face_no] = ranked[0]
                ranked_str = " ".join(f"{name.replace(' ', '_')}:{score:.4f}"
                                      for name, score in ranked)
                f.write(f"{face_no}\t0\t{face_no}\t{ranked_str}\n")
        log(f"      named {len(labels)}/{len(detections)} faces "
            f"(top {args.top_k} at score >= {args.min_score})")
        for face_no, ranked in enumerate(ranked_per_face):
            top = f"{ranked[0][0]} {ranked[0][1]:.3f}" if ranked else "(none)"
            runner_up = f"  next {ranked[1][0]} {ranked[1][1]:.3f}" if len(ranked) > 1 else ""
            log(f"        face {face_no:>3}  {top:<28}{runner_up}")
        wrote(recog_path)

    # --- [4/4] annotated image --------------------------------------------
    if not args.annotate:
        log("[4/4] Annotation not requested (--annotate to write a labelled image).")
    else:
        log(f"[4/4] Writing annotated image -> {annotated_path.name} ...")
        boxes = []
        for face_no, (_, (x1, y1, x2, y2), _) in enumerate(detections):
            name, score = labels.get(face_no, (None, None))
            label = f"{name.replace('_', ' ')} ({score:.2f})" if name else ""
            boxes.append((label, x1, y1, x2, y2))
        annotated = draw_frame(image_bgr, boxes, load_font(args.font, args.font_size))
        if not cv2.imwrite(str(annotated_path), annotated):
            raise SystemExit(f"Could not write annotated image: {annotated_path}")
        wrote(annotated_path)

    log(f"Done in {time.time() - _START:.1f}s.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path,
                   help="a video (.mp4/.mkv/...) or a still image (.jpg/.png/...); "
                        "image input runs detect + recognize on the one frame")
    p.add_argument("--transnet", default=None,
                   help="TransNetV2 weights dir (required for video, unused for an image)")
    p.add_argument("--detr", required=True, help="RT-DETR face detector checkpoint")
    p.add_argument("--dino", required=True, help="fine-tuned DINOv2 checkpoint")
    p.add_argument("--dinov2-dir", required=True, help="local clone of facebookresearch/dinov2")
    p.add_argument("--out", type=Path, default=Path("."), help="output directory")
    p.add_argument("--min-conf", type=float, default=0.5,
                   help="detection confidence floor for the detector")
    p.add_argument("--crop-padding", type=float, default=0.25,
                   help="bbox padding before embedding (DINO_CROP_PADDING)")
    p.add_argument("--img-size", type=int, default=224, help="embedder input size (train_dino default)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dedup-iou", type=float, default=0.5,
                   help="within a frame, suppress the lower-confidence of two boxes at this IoU")
    p.add_argument("--dedup-containment", type=float, default=0.7,
                   help="...or when this fraction of the smaller box lies inside the larger")
    # Tracker: FaceTracker(minIOU=0.3, min_track_length=2, min_track_conf, num_best=1).
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="minIOU to continue a track (FaceTracker minIOU)")
    p.add_argument("--min-track-length", type=int, default=2,
                   help="drop tracks with fewer faces (FaceTracker min_track_length)")
    p.add_argument("--min-track-conf", type=float, default=0.5,
                   help="drop tracks whose best face is below this confidence (MIN_TRACK_CONF)")
    p.add_argument("--num-best", type=int, default=1,
                   help="representative faces averaged per track (FaceTracker num_best=1)")
    p.add_argument("--cast", type=Path, default=None,
                   help="cast list TSV; default: cast_list/<id>*.tsv, where <id> is the "
                        "clip name up to the first '__' (as local_complete_mov.sh does)")
    p.add_argument("--cast-dir", type=Path, default=Path(__file__).parent / "cast_list",
                   help="directory searched for the cast list")
    p.add_argument("--top-k", type=int, default=10,
                   help="characters listed per track in the recog file")
    p.add_argument("--min-score", type=float, default=0.18,
                   help="minimum cosine similarity to list a character")
    p.add_argument("--annotate", action="store_true",
                   help="write an annotated mp4 (video) or image with boxes + names")
    p.add_argument("--annotated-out", type=Path, default=None,
                   help="path for the annotated output (default under <out>/)")
    p.add_argument("--font", type=Path, default=DEFAULT_FONT,
                   help="label font .ttf (default: Pillow's built-in font)")
    p.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    p.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW,
                   help="moving-average window (frames) for bbox smoothing")
    p.add_argument("--no-audio", action="store_true",
                   help="skip muxing the source audio into the annotated mp4")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute every stage even if its output file exists; by default "
                        "existing shots/faces/tracks/track_reps are reused and the stage "
                        "is skipped, as proc_one_mov.sh does")
    p.add_argument("--stride", type=int, default=1, help="detect every Nth frame (1 = all)")
    p.add_argument("--max-frames", type=int, default=0, help="cap frames scanned (0 = whole film)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.video.suffix.lower() in IMAGE_EXTS:
        process_image(args)
        return
    if args.transnet is None:
        p.error("--transnet is required for video input")

    clip_id = args.video.stem

    scenes_path = args.out / "shots" / f"{clip_id}.scenes.txt"
    faces_path = args.out / "faces_detected" / f"{clip_id}.faces_detected.txt"
    tracks_path = args.out / "tracks" / f"{clip_id}.tracks.txt"
    # Named .insightface.txt even for the DINO backend: one filename is used
    # for whichever recognizer produced the vectors.
    track_reps_path = args.out / "track_reps" / f"{clip_id}.insightface.txt"
    recog_path = args.out / "recog" / f"{clip_id}.recog.txt"
    annotated_path = args.annotated_out or (
        args.out / "annotated_movies" / f"{clip_id}.annotated.mp4")
    for path in (scenes_path, faces_path, tracks_path, track_reps_path, recog_path,
                 annotated_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    log(f"clip_id={clip_id}  device={args.device}  out={args.out}")

    # --- [1/6] shots ------------------------------------------------------
    # Reuse an existing scenes file; otherwise run shot detection in its
    # isolated subprocess. Reusing also keeps TensorFlow out of the process
    # entirely on a rerun, which matters on a shared GPU.
    if scenes_path.exists() and not args.overwrite:
        shots = read_scenes(scenes_path)
        log(f"[1/6] Scenes file exists, reusing {scenes_path.name} "
            f"({len(shots)} shots). --overwrite to redo.")
    else:
        log(f"[1/6] Shot detection (TransNetV2, isolated subprocess) on {args.video} ...")
        # The subprocess writes scenes_path directly in the "<start> <end>"
        # format; see detect_shots / _SHOT_WORKER.
        shots = detect_shots(args.video, args.transnet, scenes_path)
        wrote(scenes_path)
    lengths = [end - start + 1 for start, end in shots]
    log(f"      {len(shots)} shots, median length {int(np.median(lengths))} frames"
        if shots else "      0 shots")

    # --- [2/6] detection --------------------------------------------------
    # Detection is the most expensive stage - every frame through RT-DETR - so
    # an existing faces file is reused rather than recomputed.
    if faces_path.exists() and not args.overwrite:
        detections, face_nos = read_faces_detected(faces_path)
        if not detections:
            raise SystemExit(f"{faces_path} is empty - rerun with --overwrite.")
        log(f"[2/6] Faces file exists, reusing {faces_path.name} "
            f"({len(detections):,} faces) - detector not loaded. --overwrite to redo.")
    else:
        log(f"[2/6] Detecting faces (conf >= {args.min_conf}, "
            f"every {args.stride} frame(s)) ...")
        detections, n_frames = detect_faces(
            args.video, args.detr, args.min_conf, args.device, args.batch_size,
            args.stride, args.max_frames,
        )
        if not detections:
            raise SystemExit("No faces detected.")
        n_raw = len(detections)
        detections = dedupe_detections(detections, args.dedup_iou, args.dedup_containment)
        if n_raw != len(detections):
            log(f"      dropped {n_raw - len(detections):,} duplicate boxes "
                f"({n_raw:,} -> {len(detections):,})")

        # face_no is the index of a face within its own frame; every downstream
        # file keys on (frame_no, face_no), so it is assigned once, here.
        face_nos, seen_per_frame = [], {}
        for frame_index, _, _ in detections:
            face_nos.append(seen_per_frame.get(frame_index, 0))
            seen_per_frame[frame_index] = face_nos[-1] + 1

        # Format: "<frame_no>\t<face_no>\t<x1 y1 w h ... conf>" - width/height,
        # not x2/y2, and confidence last. No landmarks, as with box-only
        # detectors.
        with open(faces_path, "w", encoding="utf-8") as f:
            for i, (frame_index, (x1, y1, x2, y2), conf) in enumerate(detections):
                f.write(f"{frame_index}\t{face_nos[i]}\t"
                        f"{x1} {y1} {x2 - x1} {y2 - y1} {conf:.4f}\n")
        log(f"      {len(detections):,} faces in {n_frames:,} frames")
        wrote(faces_path)

    # --- [3/6] tracks -----------------------------------------------------
    if tracks_path.exists() and not args.overwrite:
        assignment = read_tracks_file(tracks_path)
        track_ids = [assignment.get((detections[i][0], face_nos[i]))
                     for i in range(len(detections))]
        log(f"[3/6] Tracks file exists, reusing {tracks_path.name}. "
            f"--overwrite to redo.")
    else:
        log(f"[3/6] Building tracks (FaceTracker: minIOU={args.iou_threshold}, "
            f"min_track_length={args.min_track_length}, "
            f"min_track_conf={args.min_track_conf}) ...")
        # boundaries = shot-start frames, as read_shots builds them.
        boundaries = {start for start, _ in shots}
        track_ids = build_tracks(detections, boundaries, args.iou_threshold,
                                 args.min_track_length, args.min_track_conf)

    # track_ids[i] is None for detections whose track FaceTracker dropped; the
    # survivors are already renumbered densely, so sorted() gives 0..n-1.
    rows_by_track_all = {}
    for i, track in enumerate(track_ids):
        if track is not None:
            rows_by_track_all.setdefault(track, []).append(i)
    ordered_tracks = sorted(rows_by_track_all)
    if not ordered_tracks:
        raise SystemExit(
            f"No track passes min_track_length={args.min_track_length} and "
            f"min_track_conf={args.min_track_conf}.")
    n_dropped = sum(1 for t in track_ids if t is None)
    log(f"      {len(ordered_tracks):,} tracks, {n_dropped:,} faces not in a kept track")

    # write_tracks() format: track_no, frame_no, face_no, x1, y1, x2, y2 (ints).
    keep_tracks = set(ordered_tracks)
    if not (tracks_path.exists() and not args.overwrite):
        with open(tracks_path, "w", encoding="utf-8") as f:
            for track in ordered_tracks:
                for i in rows_by_track_all[track]:
                    _, (x1, y1, x2, y2), _ = detections[i]
                    f.write(f"{track}\t{detections[i][0]}\t{face_nos[i]}\t"
                            f"{x1}\t{y1}\t{x2}\t{y2}\n")
        wrote(tracks_path)

    # Representation: --num-best defaults to 1, giving a single representative
    # face per track - one embedding per track, not an average over the track.
    # --num-best > 1 averages the N highest-confidence faces if a more robust
    # vector is wanted. "Best" here means highest detection confidence.
    reps_by_track = []  # representative detection rows per track (to embed)
    for track in ordered_tracks:
        rows = sorted(rows_by_track_all[track], key=lambda i: -detections[i][2])
        reps_by_track.append(rows[: max(1, args.num_best)])

    # --- [4/6] embeddings -------------------------------------------------
    # Reusing existing track files also means DINOv2 is never loaded, so
    # iterating on recognition/annotation costs nothing on the GPU.
    if track_reps_path.exists() and not args.overwrite:
        rep_track_nos, track_embeddings = read_track_reps(track_reps_path)
        if rep_track_nos != ordered_tracks:
            raise SystemExit(
                f"{track_reps_path.name} has {len(rep_track_nos)} tracks but the tracks "
                f"file has {len(ordered_tracks)}; they disagree. Rerun with --overwrite.")
        log(f"[4/6] Track reps exist, reusing {track_reps_path.name} "
            f"({tuple(track_embeddings.shape)}) - DINOv2 not loaded. --overwrite to redo.")
    else:
        n_to_embed = sum(len(r) for r in reps_by_track)
        log(f"[4/6] Embedding {n_to_embed:,} representative faces from "
            f"{len(reps_by_track):,} tracks (num_best={args.num_best}) ...")
        model, state = load_dino(args.dino, args.dinov2_dir, args.device)
        log(f"      {state['arch']}, trained on {state.get('data_source')}, "
            f"checkpoint crop_padding {state.get('crop_padding')} (using {args.crop_padding})")
        track_embeddings = embed_tracks(
            args.video, detections, reps_by_track, model, args.device,
            args.crop_padding, args.img_size, args.batch_size,
        )

        # Track reps file: "track_no, frame_no, face_no, <space-separated
        # floats>", one row per track, carrying the representative
        # (highest-confidence) face's (frame_no, face_no) - the same shape
        # get_reps_dino writes.
        with open(track_reps_path, "w", encoding="utf-8") as f:
            for position, track in enumerate(ordered_tracks):
                best = reps_by_track[position][0]
                vector = " ".join(str(x) for x in track_embeddings[position].tolist())
                f.write(f"{track}\t{detections[best][0]}\t{face_nos[best]}\t{vector}\n")
        log(f"      {tuple(track_embeddings.shape)} track embeddings")
        wrote(track_reps_path)

    # --- [5/6] recognition against the cast list --------------------------
    # The embeddings lookup id is derived from the clip name: everything
    # before the first "__".
    cast_file = args.cast
    lookup_id = clip_id.split("__")[0]
    if cast_file is None:
        candidates = sorted(args.cast_dir.glob(f"{lookup_id}*.tsv"))
        if candidates:
            cast_file = candidates[0]

    labels = {}
    if cast_file is None:
        log(f"[5/6] No cast list for '{lookup_id}' in {args.cast_dir} "
            f"- skipping recognition (pass --cast to name one).")
    else:
        log(f"[5/6] Recognizing tracks against {Path(cast_file).name} ...")
        cast = load_cast(cast_file)
        log(f"      {len(cast[0]):,} exemplars, {len(set(cast[0]))} characters")
        ranked_per_track = recognize_tracks(track_embeddings, cast, args.top_k,
                                            args.min_score)

        # Format: track_no, frame_no, face_no, then the ranked "label:score"
        # list, spaces in names replaced by underscores.
        with open(recog_path, "w", encoding="utf-8") as f:
            for position, track in enumerate(ordered_tracks):
                ranked = ranked_per_track[position]
                if ranked:
                    labels[track] = ranked[0]  # (name, score) - top match
                best = reps_by_track[position][0]  # representative face of the track
                ranked_str = " ".join(f"{name.replace(' ', '_')}:{score:.4f}"
                                      for name, score in ranked)
                f.write(f"{track}\t{detections[best][0]}\t{face_nos[best]}\t{ranked_str}\n")
        log(f"      named {len(labels)}/{len(ordered_tracks)} tracks "
            f"(top {args.top_k} at score >= {args.min_score}; per-track results in "
            f"{recog_path.name})")
        wrote(recog_path)

    # --- [6/6] annotated mp4 ----------------------------------------------
    if not args.annotate:
        log("[6/6] Annotation not requested (--annotate to write a labelled mp4).")
    else:
        log(f"[6/6] Writing annotated video -> {annotated_path.name} ...")
        n_written = annotate_video(args.video, annotated_path, detections, track_ids,
                                   keep_tracks, labels, args.font, args.font_size,
                                   args.smooth_window, with_audio=not args.no_audio)
        log(f"      {n_written:,} frames")
        wrote(annotated_path)

    log(f"Done in {time.time() - _START:.1f}s.")


if __name__ == "__main__":
    main()
