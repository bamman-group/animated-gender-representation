"""Fine-tunes a DINOv2 backbone on iCartoonFace. The default objective is an
ArcFace margin-softmax classifier over the training identities (--loss
arcface); --loss triplet selects a triplet loss instead, and --data-source
tracks always uses triplet (its negatives must be co-occurring faces - no
global identity labels there). Two training data sources are supported
(--data-source):

  "identity" (default) - iCartoonFace rectrain, one identity per subdirectory:
    {train_dir}/{identity_dir}/{image}.jpg
    {train_det_file}: "<identity_dir>/<image>.jpg  x1  y1  x2  y2" per image
    Positive pairs — two images of the same identity
    Negative pairs — one image from a different, randomly chosen identity
                     (no "same scene" constraint - this data has no
                     video/frame/track structure to exploit)

  "tracks" - paired face-track data:
    {data_dir}/{mov_id}/{top_frame}/{track_no}/*.png
    (already-cropped face images - no separate bbox file, unlike "identity")
    Positive pairs — two images from the same {mov_id}/{track_no} (same
                     identity, possibly from different top_frames)
    Negative pairs — two images from different track_nos within the same
                     {mov_id}/{top_frame} (co-occurring, so guaranteed
                     different people)

Evaluation: mid-training checkpoint selection uses a held-out **dev split**
carved out of the training data itself (--dev-fraction, default 0.1) - NOT
the official iCartoonFace rectest split or data/film, both of which are test
data held out entirely for final, post-hoc comparison (src/evaluate_baseline.py,
src/evaluate_film.py, or this script's own --eval-only against rectest) and
never touched during training. Selection is on MRR (smoother than Rank@1);
both are logged. For --data-source identity, a fraction of *identities* is
held out from _resample()'s triplet sampling entirely and used for a
leave-one-out eval over the whole dev set. For --data-source tracks, a
fraction of co-occurrence *frames* (scenes with >=2 tracks, capped at
MAX_DEV_FRAMES) is held out and every track in a held-out frame is excluded
from training; the eval treats each frame as one unit - each track's
first/last-frame pair must rank above the faces of the *other* tracks in that
same frame. Frames (not tracks) are the holdout unit precisely so the
distractors are guaranteed different identities: tracks co-occurring in one
frame are two faces on screen at once, so they can't be the same person -
which "other tracks in the same movie" (a recurring character's separate
tracks) would not guarantee. src/models/dino_backbone.py's DinoEvalAdapter
makes the DINO model pluggable into the shared evaluation pipeline either way.

Usage:
    python -m src.train_dino \\
        --data-source identity \\
        --train-dir data/raw/personai_icartoonface_rectrain/icartoonface_rectrain \\
        --train-det-file data/raw/personai_icartoonface_rectrain/icartoonface_rectrain_det.txt \\
        --dinov2-dir /path/to/dinov2 \\
        --weights /path/to/dinov2_vitl14_pretrain.pth \\
        --arch vitl14 \\
        --output-dir outputs/dino_vitl14 \\
        --epochs 20 --batch-size 64 --gpu 0 --fp16

    python -m src.train_dino \\
        --data-source tracks \\
        --data-dir /path/to/paired_face_tracks \\
        --dinov2-dir /path/to/dinov2 \\
        --weights /path/to/dinov2_vitl14_pretrain.pth \\
        --arch vitl14 \\
        --output-dir outputs/dino_tracks \\
        --epochs 20 --batch-size 64 --gpu 0 --fp16

    # Eval only
    python -m src.train_dino \\
        --eval-only \\
        --resume outputs/dino_vitl14/backbone_best.pth \\
        --dinov2-dir /path/to/dinov2 \\
        --weights /path/to/dinov2_vitl14_pretrain.pth \\
        --arch vitl14
"""
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.datasets.icartoonface import DEFAULT_CROP_PADDING, crop_face, parse_det_file
from src.evaluate import EmbeddingTimer, count_rank1_and_mrr, count_rank1_correct, embed_paths
from src.evaluate import rank1_identification_accuracy as icartoonface_rank1
from src.evaluate_film import rank1_identification_accuracy as film_rank1
from src.models.arcface import ArcFaceHead
from src.models.dino_backbone import ARCH_EMBED_DIM, DinoEvalAdapter, load_model
from src.report import (
    EvalResult,
    append_json_record,
    append_markdown_result,
    result_to_record,
    write_timing_json,
)

# Cap on the number of held-out co-occurrence frames used for the tracks dev
# eval (--data-source tracks), to keep mid-training evals fast.
MAX_DEV_FRAMES = 1000

# A handful of images in the large training sets (track PNGs from the
# external extraction pipeline especially) can be truncated on disk; without
# this, PIL raises "OSError: broken data stream" and kills a multi-hour run
# mid-epoch. Truncated files decode with the missing region blanked instead.
# (Process-global PIL state - also covers the dev eval and, via import, the
# datasets train_buffalo.py shares from this module. Files that are corrupt
# beyond truncation still raise; see TripletTrackDataset.__getitem__'s
# substitute-and-continue handling.)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dinov2-dir", required=True, help="Path to the dinov2 repo root (contains hubconf.py)")
    p.add_argument("--weights", required=True, help="Path to local DINOv2 pretrain .pth (e.g. dinov2_vitl14_pretrain.pth)")
    p.add_argument("--arch", default="vitl14", choices=list(ARCH_EMBED_DIM), help="DINOv2 architecture")
    p.add_argument("--proj-dim", type=int, default=256, help="Output dimension of projection head (0 = no projection, use raw CLS)")

    p.add_argument(
        "--data-source",
        choices=["identity", "tracks"],
        default="identity",
        help="'identity': iCartoonFace rectrain identity folders (--train-dir/--train-det-file). "
        "'tracks': the original paired face-track data (--data-dir), positive=same track, "
        "negative=co-occurring different track.",
    )
    p.add_argument(
        "--train-dir",
        default="data/raw/personai_icartoonface_rectrain/icartoonface_rectrain",
        help="[--data-source identity] iCartoonFace rectrain root: one identity per subdirectory",
    )
    p.add_argument(
        "--train-det-file",
        default="data/raw/personai_icartoonface_rectrain/icartoonface_rectrain_det.txt",
        help="[--data-source identity] Per-image face bbox file matching --train-dir",
    )
    p.add_argument(
        "--crop-padding",
        type=float,
        default=None,
        help="Fraction of the face bbox's width/height added on each side before cropping "
        "(matches src/datasets/icartoonface.py). Applies identically to [--data-source identity] "
        "training images, the dev eval, and both test protocols, so train and eval preprocessing "
        "always match. When not given: the checkpoint's recorded training value "
        "(--resume/--init-weights/--eval-only), else 0.0 (tight crop). Not used for "
        "--data-source tracks - those images are already cropped face tracks.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="[--data-source tracks] Root of paired face-track data: "
        "{data_dir}/{mov_id}/{top_frame}/{track_no}/*.png",
    )
    p.add_argument(
        "--only-movies-file",
        default=None,
        help="[--data-source tracks] Path to a TSV/text file whose first column lists movie IDs "
        "(matching {mov_id} directory names under --data-dir) - training uses ONLY these movies; "
        "every other movie is excluded entirely, as if its directory did not exist (not scanned, "
        "not sampled, not eligible for the dev split either). A header row named 'id' is skipped "
        "automatically. Use this to train only on movies known to be safe (e.g. "
        "data/animated.only.titles.unannotated.no_char_overlap.tsv - unannotated, no character "
        "overlap with data/film's eval set), excluding every other movie that might leak into "
        "final evaluation.",
    )
    p.add_argument(
        "--dev-fraction",
        type=float,
        default=0.1,
        help="Fraction held out from training for mid-training checkpoint selection: identities "
        "(--data-source identity) or co-occurrence frames (--data-source tracks; tracks appearing "
        "in a held-out frame are excluded from training, and the frame count is capped at "
        "MAX_DEV_FRAMES). This is the ONLY data used to pick a checkpoint - never the rectest "
        "split or data/film below, both test data reserved for final, post-hoc evaluation only.",
    )
    p.add_argument(
        "--dev-probes-per-identity",
        type=int,
        default=2,
        help="[--data-source identity] Cap how many images per dev identity take a turn as "
        "anchor/probe in the Rank@1 dev eval (e.g. 2 -> 2 trials/identity instead of n*(n-1)). "
        "The distractor gallery still embeds every dev image regardless, so trial difficulty is "
        "unchanged - only the number of trials shrinks. Pass 0 to use every image (old behavior).",
    )
    p.add_argument("--output-dir", default="outputs/dino_vitl14")
    p.add_argument(
        "--resume",
        default=None,
        help="Resume an interrupted run from a checkpoint saved by this script - continues the "
        "epoch counter and LR schedule from where it left off (start_epoch = the checkpoint's "
        "saved epoch). Mutually exclusive with --init-weights.",
    )
    p.add_argument(
        "--init-weights",
        default=None,
        help="Load a checkpoint's weights as the starting point for a fresh training run - "
        "epoch counter, LR warmup/cosine schedule, and best-Rank@1/early-stopping all restart "
        "at 0, exactly as if these were the pretrained weights. Use this (not --resume) to "
        "chain fine-tuning stages, e.g. continuing an --data-source identity checkpoint on "
        "--data-source tracks with its own full schedule. Mutually exclusive with --resume.",
    )
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--eval-dataset",
        choices=["rectest", "film", "both"],
        default="both",
        help="[--eval-only] Which test set(s) to report Rank@1 on (matches evaluate_baseline.py's "
        "--dataset). Both are test data, read only here - never during train()/mid-training "
        "checkpoint selection (--dev-fraction).",
    )

    # TEST DATA ONLY - used exclusively by --eval-only for a final, post-hoc
    # accuracy check against a saved checkpoint. Never read anywhere inside
    # train() / the training loop; mid-training checkpoint selection uses
    # --dev-fraction above instead.
    p.add_argument(
        "--test-dir",
        default="data/raw/personai_icartoonface_rectest/icartoonface_rectest",
    )
    p.add_argument("--test-info-file", default="data/raw/icartoonface_rectest_info.txt")
    p.add_argument("--film-images-dir", default="data/film/images")
    p.add_argument("--film-annotations", default="data/film/annotations.json")
    p.add_argument(
        "--output", type=str, default=None, help="Append eval results as rows to this markdown file"
    )
    p.add_argument(
        "--timing-output",
        type=str,
        default=None,
        help="Append accuracy/timing as JSON lines to this file (for later analysis)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Name for this run in the results.md / timing.jsonl 'Model' column and timing.json "
        "(default: dino_<arch>_<data-source> when training; dino_<arch>[_finetuned] for --eval-only). "
        "Set a clean, stable tag so src/collect_results.py groups rows sensibly.",
    )

    # Training
    p.add_argument(
        "--loss",
        choices=["arcface", "triplet"],
        default="arcface",
        help="Training objective for --data-source identity. 'arcface' (default): margin-softmax "
        "classification over the training identities - denser signal than one random negative per "
        "anchor. 'triplet': the original random-negative triplet loss. --data-source tracks always "
        "uses triplet (no global identity labels there - negatives must be co-occurring faces), so "
        "this flag is ignored / auto-forced to triplet for tracks.",
    )
    p.add_argument("--arc-scale", type=float, default=64.0, help="[--loss arcface] logit scale (s)")
    p.add_argument("--arc-margin", type=float, default=0.5, help="[--loss arcface] angular margin (m)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--triplets-per-epoch", type=int, default=100_000, help="[--loss triplet] triplets sampled per epoch")
    p.add_argument("--margin", type=float, default=0.3, help="[--loss triplet] triplet margin")
    p.add_argument(
        "--lr-backbone", type=float, default=1e-5, help="LR for the DINOv2 backbone parameters"
    )
    p.add_argument(
        "--lr-head", type=float, default=1e-3,
        help="LR for the projection head (--proj-dim > 0) and/or the ArcFace classifier head "
        "(--loss arcface). Both start from a random init and adapt faster than the pretrained "
        "backbone. Unused for --proj-dim 0 with --loss triplet (no head at all).",
    )
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-batches", type=int, default=500, help="Linear LR warmup steps before cosine decay")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0, help="Primary GPU index (-1 for CPU)")
    p.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Wrap the model in nn.DataParallel across every CUDA device visible to this "
        "process. Off by default - without this flag, only --gpu is used, even if more GPUs "
        "are visible (e.g. on a shared server where you were given access to just one).",
    )
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="torch.set_num_threads() - caps CPU-side parallelism (BLAS/intra-op) for this "
        "process. PyTorch otherwise defaults to using every core on the machine, which is "
        "unnecessary for GPU training and can be a problem on a shared server.",
    )
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--eval-every-batches", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=224, help="Input image size (DINOv2 native is 518, 224 is faster)")
    p.add_argument(
        "--limit-identities",
        type=int,
        default=0,
        help="[--data-source identity] Use only the first N identities (0 = all) - for quick "
        "debugging / smoke tests without scanning the full 5,013-identity training set.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test mode: 1 epoch, a handful of identities/triplets, frequent dev evals, and "
        "tiny --eval-only gallery slices - verifies the whole pipeline runs end to end in minutes. "
        "Numbers produced are NOT valid results.",
    )
    return p.parse_args()


def apply_test_mode(args):
    """Shrink a run to a fast end-to-end smoke test (see --test)."""
    args.epochs = 1
    args.triplets_per_epoch = min(args.triplets_per_epoch, 200)
    args.warmup_batches = min(args.warmup_batches, 5)
    args.eval_every_batches = 20
    if getattr(args, "limit_identities", 0) == 0:
        args.limit_identities = 60


def resolve_crop_padding(args, ckpt_state: dict | None, context: str) -> None:
    """--crop-padding defaults to None so "not specified" is distinguishable
    from an explicit 0.0. Precedence: an explicit flag always wins (with a
    warning when it disagrees with what the checkpoint was trained with) >
    the checkpoint's recorded training value > DEFAULT_CROP_PADDING. Keeping
    eval preprocessing matched to training preprocessing matters: a model
    fine-tuned on 25%-padded crops scores differently on tight crops, and
    without the checkpoint remembering its padding that mismatch is silent.
    Shared by both fine-tuning scripts."""
    saved = ckpt_state.get("crop_padding") if ckpt_state else None
    if args.crop_padding is None:
        args.crop_padding = saved if saved is not None else DEFAULT_CROP_PADDING
        if saved is not None:
            print(f"Using the checkpoint's crop padding ({saved:g}) for {context}.")
    elif saved is not None and abs(args.crop_padding - saved) > 1e-9:
        print(
            f"[warning] --crop-padding {args.crop_padding:g} differs from the checkpoint's "
            f"training value ({saved:g}); {context} uses {args.crop_padding:g} - the numbers "
            f"will reflect a train/eval preprocessing mismatch."
        )


# ---------------------------------------------------------------------------
# Transform (ImageNet normalization for DINOv2 - different from this repo's
# shared EVAL_TRANSFORM / buffalo_l convention, which uses mean=std=0.5;
# DINOv2 is pretrained with standard ImageNet stats and expects them at
# fine-tune time too). Only needed for training - evaluation goes through
# src/datasets/icartoonface.py's EVAL_TRANSFORM via DinoEvalAdapter instead,
# to share the exact same preprocessing every other model's Rank@1 eval uses.
# ---------------------------------------------------------------------------


def make_train_transform(img_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset - identity-based triplets: positive = two images of the same
# identity, negative = one image from any other identity. Used where the data
# has no video-frame/track structure to exploit.
# ---------------------------------------------------------------------------


def split_identity_images(
    train_dir: str, dev_fraction: float, seed: int, limit_identities: int = 0
) -> tuple[list[Path], dict[str, list[Path]], dict[str, list[Path]]]:
    """Deterministic (seeded) train/dev split of iCartoonFace rectrain
    identity folders. Returns (all_identity_dirs, train_images, dev_images)
    where train_images/dev_images map identity name -> sorted list of its
    image paths. A fraction (dev_fraction) of *identities* is held out
    entirely for mid-training checkpoint selection; callers apply their own
    minimum-image filter. Shared by TripletIdentityDataset and
    IdentityClassificationDataset so both hold out the same identities for a
    given --seed/--dev-fraction."""
    root = Path(train_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run scripts/download_data.sh and scripts/prepare_data.py first."
        )
    all_identity_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if limit_identities and limit_identities > 0:
        # smoke-test / debugging cap (see --test / --limit-identities)
        all_identity_dirs = all_identity_dirs[:limit_identities]
    shuffled = all_identity_dirs[:]
    random.Random(seed).shuffle(shuffled)
    n_dev = int(len(shuffled) * dev_fraction)
    dev_names = {d.name for d in shuffled[:n_dev]}

    train_images: dict[str, list[Path]] = {}
    dev_images: dict[str, list[Path]] = {}
    for identity_dir in all_identity_dirs:
        images = sorted(identity_dir.glob("*.jpg"))
        (dev_images if identity_dir.name in dev_names else train_images)[identity_dir.name] = images
    return all_identity_dirs, train_images, dev_images


class TripletIdentityDataset(Dataset):
    def __init__(
        self,
        train_dir: str,
        det_file: str,
        triplets_per_epoch: int = 100_000,
        padding: float = DEFAULT_CROP_PADDING,
        dev_fraction: float = 0.1,
        transform=None,
        seed: int = 42,
        limit_identities: int = 0,
    ):
        self.transform = transform
        self.triplets_per_epoch = triplets_per_epoch
        self.padding = padding
        self.rng = random.Random(seed)
        self.bboxes = parse_det_file(Path(det_file))

        # Held-out dev identities are never sampled by _resample() below - they
        # are used only for evaluate_identity_dev_rank1's mid-training
        # checkpoint-selection eval. Both pools need >=2 images (a positive
        # pair for training, a leave-one-out trial for dev).
        all_identity_dirs, train_images, dev_images = split_identity_images(
            train_dir, dev_fraction, seed, limit_identities
        )
        self.by_identity = {k: v for k, v in train_images.items() if len(v) >= 2}
        self.dev_by_identity = {k: v for k, v in dev_images.items() if len(v) >= 2}
        self.identities = list(self.by_identity.keys())

        print(f"  Identities total             : {len(all_identity_dirs)}")
        print(f"  Dev identities (held out)    : {len(self.dev_by_identity)}")
        print(f"  Train identities w/ >=2 imgs : {len(self.identities)}")
        print(f"  Triplets per epoch           : {triplets_per_epoch}")

        self._resample()

    def _resample(self):
        rng = self.rng
        self.triplets = []
        for _ in range(self.triplets_per_epoch):
            anchor_id = rng.choice(self.identities)
            a_path, p_path = rng.sample(self.by_identity[anchor_id], 2)
            neg_id = rng.choice(self.identities)
            while neg_id == anchor_id:
                neg_id = rng.choice(self.identities)
            n_path = rng.choice(self.by_identity[neg_id])
            self.triplets.append((a_path, p_path, n_path))

    def __len__(self):
        return len(self.triplets)

    def _load_crop(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        bbox = self.bboxes.get(f"{path.parent.name}/{path.name}")
        if bbox is not None:
            image = crop_face(image, bbox, self.padding)
        return image

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        return (
            self.transform(self._load_crop(a_path)),
            self.transform(self._load_crop(p_path)),
            self.transform(self._load_crop(n_path)),
        )


# ---------------------------------------------------------------------------
# Dataset - identity classification (--data-source identity --loss arcface,
# the default). Yields (image, class_index) over every training image, for a
# margin-softmax (ArcFace) classifier over the training identities - a much
# denser gradient than one random negative per anchor when there are
# thousands of identities. Shares the exact same held-out dev identities as
# TripletIdentityDataset (via split_identity_images) so checkpoint selection
# is identical regardless of --loss.
# ---------------------------------------------------------------------------


class IdentityClassificationDataset(Dataset):
    def __init__(
        self,
        train_dir: str,
        det_file: str,
        padding: float = DEFAULT_CROP_PADDING,
        dev_fraction: float = 0.1,
        transform=None,
        seed: int = 42,
        limit_identities: int = 0,
    ):
        self.transform = transform
        self.padding = padding
        self.bboxes = parse_det_file(Path(det_file))

        all_identity_dirs, train_images, dev_images = split_identity_images(
            train_dir, dev_fraction, seed, limit_identities
        )
        # Dev needs >=2 images (leave-one-out); training classes need >=1.
        self.dev_by_identity = {k: v for k, v in dev_images.items() if len(v) >= 2}
        self.identities = [k for k, v in train_images.items() if len(v) >= 1]
        self.label_of = {name: i for i, name in enumerate(self.identities)}
        self.num_classes = len(self.identities)
        self.samples: list[tuple[Path, int]] = [
            (p, self.label_of[name]) for name in self.identities for p in train_images[name]
        ]

        print(f"  Identities total             : {len(all_identity_dirs)}")
        print(f"  Dev identities (held out)    : {len(self.dev_by_identity)}")
        print(f"  Train identities (classes)   : {self.num_classes}")
        print(f"  Train images                 : {len(self.samples)}")

    def _resample(self):
        pass  # a full pass over all images is one epoch; nothing to resample

    def __len__(self):
        return len(self.samples)

    def _load_crop(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        bbox = self.bboxes.get(f"{path.parent.name}/{path.name}")
        if bbox is not None:
            image = crop_face(image, bbox, self.padding)
        return image

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(self._load_crop(path)), label


# ---------------------------------------------------------------------------
# Dataset - paired face-track triplets (--data-source tracks). Scans
# {data_dir}/{mov_id}/{top_frame}/{track_no}/*.png, already-cropped face
# images produced by a separate extraction step - no bbox file here, unlike
# TripletIdentityDataset above.
# ---------------------------------------------------------------------------


def _load_track_img(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_movie_id_list(path: str) -> set[str]:
    """Parses --only-movies-file: a TSV/text file whose first (tab- or
    whitespace-separated) column is a movie ID matching {mov_id} directory
    names under --data-dir (e.g. data/animated.only.titles.unannotated.no_char_overlap.tsv,
    id column like "zootopia_tt2948356"). A header row literally named "id"
    is skipped; blank lines are ignored."""
    ids = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        first_col = line.split("\t")[0].split()[0]
        if first_col == "id":
            continue  # header row
        ids.add(first_col)
    return ids


class TripletTrackDataset(Dataset):
    """
    Scans {data_dir}/{mov_id}/{top_frame}/{track_no}/*.png.

    Each __getitem__ returns (anchor, positive, negative):
      anchor + positive : different images from the same (mov_id, track_no)
      negative          : image from a different track_no in the same
                          (mov_id, top_frame) as the anchor

    Tracks with only 1 image can serve as anchor/negative but not as
    the positive source - they are used only on the negative side.

    Dev split: a fraction of co-occurrence *frames* (--dev-fraction of the
    (mov_id, top_frame) scenes with >=2 tracks, capped at MAX_DEV_FRAMES) is
    held out, and every track appearing in a held-out frame is excluded from
    training entirely. Stored in dev_by_frame for evaluate_track_dev_rank1 -
    frames (not tracks) are the holdout unit so the dev eval's distractors are
    guaranteed different identities (tracks co-occurring in one frame are two
    faces on screen at once).
    """

    def __init__(
        self,
        data_dir: str,
        triplets_per_epoch: int = 100_000,
        dev_fraction: float = 0.1,
        transform=None,
        seed: int = 42,
        only_movie_ids: set[str] | None = None,
    ):
        self.transform = transform
        self.triplets_per_epoch = triplets_per_epoch
        self.rng = random.Random(seed)

        # by_track[(mov_id, track_no)] = [Path, ...]
        # by_cooccur[(mov_id, top_frame)] = {track_no: [Path, ...]}
        by_track: dict[tuple[str, str], list[Path]] = defaultdict(list)
        by_cooccur: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

        data_root = Path(data_dir)
        if not data_root.exists():
            raise FileNotFoundError(f"{data_root} not found.")
        excluded_movies_found: list[str] = []
        for mov_dir in sorted(data_root.iterdir()):
            if not mov_dir.is_dir():
                continue
            mov_id = mov_dir.name
            if only_movie_ids is not None and mov_id not in only_movie_ids:
                # --only-movies-file was given: every movie NOT in that list
                # is pretended not to exist at all - never scanned into
                # by_track/by_cooccur, so it can never be sampled for
                # training OR held out for the dev split.
                excluded_movies_found.append(mov_id)
                continue
            for tf_dir in sorted(mov_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                top_frame = tf_dir.name
                for track_dir in sorted(tf_dir.iterdir()):
                    if not track_dir.is_dir():
                        continue
                    track_no = track_dir.name
                    imgs = sorted(track_dir.glob("*.png"))
                    if not imgs:
                        continue
                    by_track[(mov_id, track_no)].extend(imgs)
                    by_cooccur[(mov_id, top_frame)][track_no].extend(imgs)

        # Images are named by frame number (e.g. 39684.png) - sort each
        # track numerically by that frame number rather than lexicographically
        # by filename, so imgs[0]/imgs[-1] below are the true first/last
        # frames (lexicographic sort breaks once frame numbers have
        # different digit counts, e.g. "984.png" would sort after
        # "39684.png").
        for imgs in by_track.values():
            imgs.sort(key=lambda p: int(p.stem))

        # Dev split: hold out --dev-fraction of co-occurrence FRAMES (scenes
        # with >=2 tracks), capped at MAX_DEV_FRAMES, deterministic given
        # --seed. Every track appearing in a held-out frame becomes a dev
        # track and is excluded from training entirely. The point of holding
        # out *frames* (not tracks) is a clean distractor set for the dev eval:
        # tracks that co-occur in one frame are two faces on screen at once, so
        # they are guaranteed different people - unlike "other tracks in the
        # same movie", where a recurring character's separate tracks could be
        # the same identity and pollute the gallery. Each dev track contributes
        # its first and last frame (the hardest same-identity pair); see
        # evaluate_track_dev_rank1.
        eligible_frames = sorted(k for k, tracks in by_cooccur.items() if len(tracks) >= 2)
        shuffled_frames = eligible_frames[:]
        random.Random(seed).shuffle(shuffled_frames)
        n_dev = min(int(len(shuffled_frames) * dev_fraction), MAX_DEV_FRAMES)
        dev_frames = shuffled_frames[:n_dev]

        dev_track_keys = set()
        self.dev_by_frame: dict[tuple[str, str], dict[str, tuple[Path, Path]]] = {}
        for key in dev_frames:
            frame_tracks = {}
            for track_no in by_cooccur[key]:
                dev_track_keys.add((key[0], track_no))
                imgs = by_track[(key[0], track_no)]
                frame_tracks[track_no] = (imgs[0], imgs[-1])
            self.dev_by_frame[key] = frame_tracks

        # Training pools exclude every dev track entirely (from all its frames,
        # not just the held-out ones) - never sampled as anchor/positive/negative.
        train_by_track = {k: v for k, v in by_track.items() if k not in dev_track_keys}
        train_by_cooccur: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(dict)
        for (mov_id, top_frame), tracks in by_cooccur.items():
            for track_no, imgs in tracks.items():
                if (mov_id, track_no) not in dev_track_keys:
                    train_by_cooccur[(mov_id, top_frame)][track_no] = imgs

        # Only top_frames with >=2 distinct tracks can produce negatives
        self.cooccur_keys = [
            (mov_id, top_frame) for (mov_id, top_frame), tracks in train_by_cooccur.items() if len(tracks) >= 2
        ]
        # Only tracks with >=2 images can be anchor+positive sources
        self.positive_tracks = {k: v for k, v in train_by_track.items() if len(v) >= 2}
        self.by_track = train_by_track
        self.by_cooccur = train_by_cooccur

        # Full movie-level listing, printed so a SLURM log can be grepped/read
        # to confirm exactly which movies a given run trained on (and which
        # were excluded via --only-movies-file, if any).
        all_movies_scanned = sorted({mov_id for mov_id, _ in by_track})
        movies_used_for_training = sorted({mov_id for mov_id, _ in train_by_track})
        print(f"  Movies excluded (--only-movies-file, not in the list): {len(excluded_movies_found)}")
        if excluded_movies_found:
            for mov_id in sorted(excluded_movies_found):
                print(f"    excluded: {mov_id}")
        print(f"  Movies scanned (not excluded): {len(all_movies_scanned)}")
        print(f"  Movies used for training ({len(movies_used_for_training)}):")
        for mov_id in movies_used_for_training:
            print(f"    train: {mov_id}")
        print(f"  Tracks total                 : {len(by_track)}")
        print(f"  Dev frames (held out)        : {len(dev_frames)}")
        print(f"  Dev tracks (held out)        : {len(dev_track_keys)}")
        print(f"  Train tracks with >=2 images : {len(self.positive_tracks)}")
        print(f"  Co-occurrence scenes (train) : {len(self.cooccur_keys)}")
        print(f"  Triplets per epoch           : {triplets_per_epoch}")

        self._resample()

    def _resample(self):
        """Pre-sample triplets for one epoch."""
        self.triplets = []
        rng = self.rng
        attempts = 0
        while len(self.triplets) < self.triplets_per_epoch and attempts < self.triplets_per_epoch * 10:
            attempts += 1
            # Pick a co-occurrence scene that has a track with >=2 images
            mov_id, top_frame = rng.choice(self.cooccur_keys)
            tracks_here = self.by_cooccur[(mov_id, top_frame)]
            anchor_candidates = [t for t in tracks_here if len(self.by_track[(mov_id, t)]) >= 2]
            if not anchor_candidates:
                continue
            anchor_track = rng.choice(anchor_candidates)
            neg_tracks = [t for t in tracks_here if t != anchor_track]
            if not neg_tracks:
                continue
            neg_track = rng.choice(neg_tracks)

            # anchor + positive from same (mov_id, track_no) - possibly across top_frames
            a, pos = rng.sample(self.by_track[(mov_id, anchor_track)], 2)
            neg = rng.choice(self.by_track[(mov_id, neg_track)])
            self.triplets.append((a, pos, neg))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # An unreadable image (corrupt beyond PIL's truncated-file tolerance)
        # substitutes a different sampled triplet rather than killing the
        # whole run - a rare event, so the sampling distribution is unaffected
        # in practice. Each worker's `random` is seeded by the DataLoader.
        last_err = None
        for _ in range(10):
            a_path, p_path, n_path = self.triplets[idx]
            try:
                return (
                    self.transform(_load_track_img(a_path)),
                    self.transform(_load_track_img(p_path)),
                    self.transform(_load_track_img(n_path)),
                )
            except OSError as err:
                last_err = err
                print(
                    f"[warning] unreadable track image in triplet "
                    f"({a_path}, {p_path}, {n_path}): {err} - substituting another triplet"
                )
                idx = random.randrange(len(self.triplets))
        raise last_err


# ---------------------------------------------------------------------------
# Triplet loss
# ---------------------------------------------------------------------------


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = 1 - (anchor * positive).sum(dim=1)
        d_neg = 1 - (anchor * negative).sum(dim=1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0)
        return loss.mean(), (loss > 0).float().mean()


# ---------------------------------------------------------------------------
# Evaluation
#
# evaluate_dev_rank1() (and its two data-source-specific implementations
# below) is the ONLY evaluation used for mid-training checkpoint selection -
# it runs Rank@1 on data held out from training itself (--dev-fraction),
# never on test data.
#
# evaluate_icartoonface_rank1() is TEST DATA - the official iCartoonFace
# rectest Rank@1 protocol (src/evaluate.py) - used exclusively by
# --eval-only for a final, post-hoc accuracy check against a saved
# checkpoint, and never called from train()/the training loop.
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_identity_dev_rank1(
    model,
    device,
    dataset: "TripletIdentityDataset",
    padding: float,
    batch_size: int = 128,
    num_workers: int = 4,
    probes_per_identity: int | None = None,
) -> EvalResult:
    """Rank@1 over dataset.dev_by_identity - the identities --dev-fraction
    held out entirely from _resample()'s triplet sampling. Every dev image
    is embedded once (full pool, so the distractor gallery is unaffected by
    probes_per_identity); for each dev identity, every image belonging to a
    *different* dev identity is a distractor (the same leave-one-out
    construction src/evaluate.py/evaluate_film.py use, just grouped over the
    whole dev set rather than a fixed pool or per-movie).

    probes_per_identity caps how many of each identity's images take a turn
    as anchor/probe (trials = sum of min(n_i, k)*(min(n_i, k)-1), not n_i*
    (n_i-1)) - trades trial count for gallery difficulty being held fixed.
    None or <=0 uses every image (the original, uncapped behavior)."""
    if probes_per_identity is not None and probes_per_identity <= 0:
        probes_per_identity = None
    dev_items = list(dataset.dev_by_identity.items())
    if not dev_items:
        raise ValueError("No dev identities available - check --dev-fraction and --train-dir.")

    paths: list[Path] = []
    bboxes = []
    for _, imgs in dev_items:
        for p in imgs:
            paths.append(p)
            bboxes.append(dataset.bboxes.get(f"{p.parent.name}/{p.name}"))

    timer = EmbeddingTimer()
    with timer.track(len(paths)):
        embeddings = embed_paths(
            model, paths, bboxes, device, batch_size=batch_size, num_workers=num_workers,
            padding=padding, desc="Dev eval (identity)",
        )

    # NOT count_rank1_correct: that function reconstructs the distractor
    # gallery (a torch.cat + fresh matmul against every embedding) on every
    # single leave-one-out trial - fine for evaluate.py's fixed 2,501-image
    # rectest gallery or evaluate_film.py's small per-movie galleries, but
    # here the "gallery" is the entire dev pool (tens of thousands of
    # images), so that pattern is O(num_trials x pool_size x embed_dim), which
    # dominates the runtime (the embedding step itself is fast).
    # Instead: one (n, N) matmul per identity against the whole pool -
    # already computes every similarity this identity's images will ever
    # need - then mask+argmax per trial operates on precomputed scalars
    # (no embed_dim factor, no reallocating the pool each trial).
    total_correct, total_total, total_mrr, offset = 0, 0, 0.0, 0
    for _, imgs in dev_items:
        n = len(imgs)
        idxs = torch.arange(offset, offset + n)
        offset += n

        # sel = the images that take a turn as anchor/probe this identity;
        # idxs (the full identity, not just sel) is still what gets masked
        # out of the candidate columns below, so images belonging to this
        # identity but NOT in sel can never win a trial by accident.
        sel = idxs[:probes_per_identity] if probes_per_identity is not None else idxs
        m = sel.shape[0]
        if m < 2:
            continue  # need at least an anchor + one probe to form a trial

        sel_embeddings = embeddings[sel]
        sims_sel = sel_embeddings @ embeddings.T  # (m, N)

        for gallery_idx in range(m):
            col = sel[gallery_idx].item()
            probe_mask = torch.ones(m, dtype=torch.bool)
            probe_mask[gallery_idx] = False
            probe_sims = sims_sel[probe_mask]  # (m-1, N), boolean indexing already copies
            probe_sims[:, idxs[idxs != col]] = float("-inf")  # exclude every other own-identity column
            total_correct += (probe_sims.argmax(dim=1) == col).sum().item()
            # Reciprocal rank of the true match (col): masked -inf columns are
            # never strictly greater, so they don't inflate the rank.
            true_sims = probe_sims[:, col : col + 1]
            ranks = 1 + (probe_sims > true_sims).sum(dim=1)
            total_mrr += (1.0 / ranks.float()).sum().item()
            total_total += probe_sims.shape[0]

    accuracy = total_correct / total_total
    mrr = total_mrr / total_total
    print(timer.report())
    print(f"Dev (identity): Rank@1 {accuracy:.4f} | MRR {mrr:.4f} ({total_correct}/{total_total} trials)")
    return EvalResult(accuracy, total_correct, total_total, timer.elapsed, timer.count, mrr=mrr)


@torch.no_grad()
def evaluate_track_dev_rank1(
    model, device, dataset: "TripletTrackDataset", batch_size: int = 128, num_workers: int = 4
) -> EvalResult:
    """Rank@1/MRR over dataset.dev_by_frame - the held-out co-occurrence
    frames (see TripletTrackDataset / the module docstring). Each dev frame is
    one unit: its tracks co-occur, so they are guaranteed different identities
    (two faces on screen at once). For every track in the frame that has a
    first/last pair, that pair must rank above the faces of the *other* tracks
    in the same frame. Distractors therefore never come from a different frame
    or a non-co-occurring track, so - unlike a per-movie gallery - a recurring
    character's separate tracks cannot pollute the gallery with same-identity
    'distractors'. Every distinct dev face is embedded once and reused across
    the frames it appears in."""
    all_paths: list[Path] = []
    row_of: dict[Path, int] = {}
    for frame_tracks in dataset.dev_by_frame.values():
        for first, last in frame_tracks.values():
            for p in (first, last):
                if p not in row_of:
                    row_of[p] = len(all_paths)
                    all_paths.append(p)
    if not all_paths:
        raise ValueError("No dev frames available - check --dev-fraction and --data-dir.")

    timer = EmbeddingTimer()
    with timer.track(len(all_paths)):
        embeddings = embed_paths(
            model, all_paths, [None] * len(all_paths), device,
            batch_size=batch_size, num_workers=num_workers, desc="Dev eval (tracks)",
        )

    total_correct, total_total, total_mrr, frames_used = 0, 0, 0.0, 0
    for frame_tracks in dataset.dev_by_frame.values():
        track_nos = list(frame_tracks.keys())
        if len(track_nos) < 2:
            continue  # need >=1 other (co-occurring) track as a distractor

        # Flatten the frame's faces (dedup a single-image track's first==last)
        # with their owning track, then embed-index into the global pool.
        rows: list[int] = []
        owners: list[str] = []
        for track_no in track_nos:
            first, last = frame_tracks[track_no]
            for p in ([first] if first == last else [first, last]):
                rows.append(row_of[p])
                owners.append(track_no)
        frame_emb = embeddings[torch.tensor(rows)]

        used = False
        for track_no in track_nos:
            own = torch.tensor([o == track_no for o in owners])
            if own.sum().item() < 2 or (~own).sum().item() == 0:
                continue  # need a first/last pair and >=1 different-identity distractor
            correct, total, mrr_sum = count_rank1_and_mrr(frame_emb[~own], frame_emb[own])
            total_correct += correct
            total_total += total
            total_mrr += mrr_sum
            used = True
        frames_used += int(used)

    if total_total == 0:
        raise ValueError(
            "No dev frame yielded a track with a first/last pair - check --dev-fraction/--data-dir."
        )

    accuracy = total_correct / total_total
    mrr = total_mrr / total_total
    print(f"\nUsed {frames_used}/{len(dataset.dev_by_frame)} dev frames.")
    print(timer.report())
    print(f"Dev (tracks): Rank@1 {accuracy:.4f} | MRR {mrr:.4f} ({total_correct}/{total_total} trials)")
    return EvalResult(accuracy, total_correct, total_total, timer.elapsed, timer.count, mrr=mrr)


def evaluate_dev_rank1(dino_model, device, args, dataset) -> EvalResult:
    eval_model = DinoEvalAdapter(dino_model, img_size=args.img_size).to(device)
    eval_model.eval()
    if args.data_source == "tracks":
        return evaluate_track_dev_rank1(eval_model, device, dataset)
    return evaluate_identity_dev_rank1(
        eval_model, device, dataset, padding=args.crop_padding,
        probes_per_identity=args.dev_probes_per_identity,
    )


def evaluate_icartoonface_rank1(dino_model, device, args) -> EvalResult:
    """TEST DATA - the official iCartoonFace rectest split. Only called by
    --eval-only (main()), never from train()."""
    eval_model = DinoEvalAdapter(dino_model, img_size=args.img_size).to(device)
    eval_model.eval()
    caps = dict(max_distractors=200, max_probe_identities=50) if getattr(args, "test", False) else {}
    return icartoonface_rank1(
        eval_model,
        device,
        test_dir=Path(args.test_dir),
        info_file=Path(args.test_info_file),
        padding=args.crop_padding,
        **caps,
    )


def evaluate_film_rank1(dino_model, device, args) -> EvalResult:
    """TEST DATA - data/film's per-movie protocol. Only called by --eval-only
    --eval-dataset film (main()), never from train()."""
    eval_model = DinoEvalAdapter(dino_model, img_size=args.img_size).to(device)
    eval_model.eval()
    caps = dict(max_movies=10) if getattr(args, "test", False) else {}
    return film_rank1(
        eval_model,
        device,
        images_dir=Path(args.film_images_dir),
        annotations_file=Path(args.film_annotations),
        padding=args.crop_padding,
        **caps,
    )


def _log_result(
    args, result: EvalResult, model_name: str, dataset_label: str, batch: int | None = None
) -> None:
    if args.output:
        append_markdown_result(args.output, model_name, dataset_label, result)
    if args.timing_output:
        extra = {"batch": batch} if batch is not None else {}
        append_json_record(args.timing_output, result_to_record(model_name, dataset_label, result, **extra))


def default_eval_tag(base: str, ckpt_state: dict | None) -> str:
    """Default results.md model name for --eval-only when no --tag is given,
    derived from the checkpoint's recorded provenance so identity-trained,
    tracks-trained, and chained (identity -> tracks via --init-weights)
    checkpoints never collide under one name. Shared by both fine-tuning
    scripts; pass an explicit --tag for a custom name."""
    if ckpt_state is None:
        return f"{base} (baseline)"  # pretrained weights, no fine-tuning
    source = ckpt_state.get("data_source")
    if source == "tracks" and ckpt_state.get("init_weights"):
        trained = "identity+tracks"
    elif source:
        trained = source
    else:
        trained = "fine-tuned"  # no data_source key recorded in the checkpoint
    if ckpt_state.get("only_movies_file"):
        # Trained on ONLY the movies listed in --only-movies-file (every other
        # movie excluded entirely) - flag this distinctly so it never collides
        # with a normal tracks/identity+tracks run in results.md. Matches
        # src/collect_results.py's VARIANT_COLUMNS key for the "I+T/Train" column.
        trained = f"{trained}/train"
    parts = [trained]
    padding = ckpt_state.get("crop_padding")
    if padding is not None:
        parts.append(f"crop {round(padding * 100)}%")
    return f"{base} ({', '.join(parts)})"


def dev_selection_score(result: EvalResult) -> float:
    """The scalar checkpoint selection / early-stopping tracks. MRR when the
    dev eval computed it (a smoother, lower-variance signal than binary Rank@1
    for single-relevant-item retrieval), else Rank@1. Shared by both
    fine-tuning scripts."""
    return result.mrr if result.mrr is not None else result.accuracy


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args):
    torch.set_num_threads(args.cpu_threads)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    if args.fp16 and device.type == "cpu":
        print("[warning] --fp16 requires CUDA; disabling.")
        args.fp16 = False

    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  ({n_gpus} GPU(s) available, --multi-gpu={args.multi_gpu})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ArcFace needs global identity labels, which only --data-source identity
    # has: for tracks, positives come from within one track and negatives must
    # be co-occurring faces (two tracks in *different* frames may be the same
    # person - unlabeled), so a classification objective isn't defined. Force
    # triplet there.
    loss_type = args.loss
    if args.data_source == "tracks" and loss_type == "arcface":
        print("[note] --data-source tracks has no global identity labels; using --loss triplet.")
        loss_type = "triplet"
    args.loss = loss_type

    if args.resume and args.init_weights:
        sys.exit("--resume and --init-weights are mutually exclusive")
    resume_state = None
    init_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu")
    elif args.init_weights:
        init_state = torch.load(args.init_weights, map_location="cpu")
    # Resolved before the dataset is built - training crops, the dev eval, and
    # the saved checkpoints' recorded value all use the same padding.
    resolve_crop_padding(args, resume_state or init_state, "this training run")

    print(f"Loading dataset (--data-source {args.data_source}, --loss {loss_type}) ...")
    if args.data_source == "tracks":
        if not args.data_dir:
            sys.exit("--data-dir is required for --data-source tracks")
        only_movie_ids = load_movie_id_list(args.only_movies_file) if args.only_movies_file else None
        if only_movie_ids is not None:
            print(f"Training ONLY on {len(only_movie_ids)} movie(s) from {args.only_movies_file}")
        dataset = TripletTrackDataset(
            args.data_dir,
            triplets_per_epoch=args.triplets_per_epoch,
            dev_fraction=args.dev_fraction,
            transform=make_train_transform(args.img_size),
            seed=args.seed,
            only_movie_ids=only_movie_ids,
        )
    elif loss_type == "arcface":
        dataset = IdentityClassificationDataset(
            args.train_dir,
            args.train_det_file,
            padding=args.crop_padding,
            dev_fraction=args.dev_fraction,
            transform=make_train_transform(args.img_size),
            seed=args.seed,
            limit_identities=args.limit_identities,
        )
    else:
        dataset = TripletIdentityDataset(
            args.train_dir,
            args.train_det_file,
            triplets_per_epoch=args.triplets_per_epoch,
            padding=args.crop_padding,
            dev_fraction=args.dev_fraction,
            transform=make_train_transform(args.img_size),
            seed=args.seed,
            limit_identities=args.limit_identities,
        )

    print(f"Loading DINOv2 {args.arch} from {args.weights} ...")
    model = load_model(args.dinov2_dir, args.arch, args.weights, args.proj_dim)

    start_epoch = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        start_epoch = resume_state.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")
    elif init_state is not None:
        model.load_state_dict(init_state["model"])
        print(f"Initialized weights from {args.init_weights} (fresh schedule, epoch 0)")

    model = model.to(device)
    if args.multi_gpu and n_gpus > 1:
        # DataParallel scatters --batch-size across n_gpus devices; BatchNorm
        # in training mode needs >1 sample per device (torch.nn.functional
        # .batch_norm's "Expected more than 1 value per channel" otherwise) -
        # fail fast here with an actionable message instead of that cryptic
        # error surfacing deep inside a replica's forward pass.
        if args.batch_size < 2 * n_gpus:
            sys.exit(
                f"--multi-gpu splits --batch-size ({args.batch_size}) across {n_gpus} visible "
                f"GPUs; each per-GPU shard needs >=2 samples for BatchNorm during training. Use "
                f"--batch-size >= {2 * n_gpus}, or drop --multi-gpu. (If you only requested one "
                f"GPU from the scheduler, {n_gpus} visible devices also means the container isn't "
                f"restricted to the GPU(s) you were granted - worth checking that separately.)"
            )
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    bare_model = model.module if isinstance(model, nn.DataParallel) else model

    # Head params trained at --lr-head: the optional projection head and, for
    # --loss arcface, the classifier prototypes (both start from a random
    # init). The ArcFace head is a training-time classifier only - it is not
    # part of the saved backbone checkpoint (eval uses the embedding directly).
    head_params = list(bare_model.proj.parameters()) if bare_model.proj is not None else []
    arc_head = None
    if args.loss == "arcface":
        arc_head = ArcFaceHead(
            bare_model.embedding_dim, dataset.num_classes, scale=args.arc_scale, margin=args.arc_margin
        ).to(device)
        head_params += list(arc_head.parameters())
        criterion = nn.CrossEntropyLoss().to(device)
        print(f"ArcFace head: {dataset.num_classes} classes, scale={args.arc_scale}, margin={args.arc_margin}")
        # --resume must restore the head too: continuing mid-schedule with a
        # freshly randomized classifier would wreck the backbone at low LR.
        # (--init-weights deliberately does not - it starts a fresh run, and
        # the checkpoint may come from a different class set entirely.)
        if resume_state is not None:
            if "arc_head" in resume_state:
                arc_head.load_state_dict(resume_state["arc_head"])
                print("Restored ArcFace head state from the checkpoint.")
            else:
                print(
                    "[warning] --resume checkpoint has no ArcFace head state (saved by an "
                    "older version?) - the head restarts from random init."
                )
    else:
        criterion = TripletLoss(margin=args.margin).to(device)

    def checkpoint_state(epoch_num: int) -> dict:
        state = {
            "model": bare_model.state_dict(),
            "epoch": epoch_num,
            "arch": args.arch,
            "proj_dim": args.proj_dim,
            "crop_padding": args.crop_padding,
            # Provenance, so --eval-only can name results rows unambiguously
            # (identity- vs tracks-trained vs chained) without a manual --tag.
            "data_source": args.data_source,
            "init_weights": args.init_weights,
            "only_movies_file": args.only_movies_file,
        }
        if arc_head is not None:
            state["arc_head"] = arc_head.state_dict()
        return state

    bb_params = list(bare_model.backbone.parameters())
    optimizer = optim.AdamW(
        [
            {"params": bb_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = max(len(dataset) // args.batch_size, 1)
    total_batches = args.epochs * batches_per_epoch
    warmup = args.warmup_batches

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_batches - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # LambdaLR with last_epoch != -1 (--resume) requires 'initial_lr' on every
    # param group and raises a KeyError otherwise; set what last_epoch=-1
    # would have set (a no-op for fresh runs).
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch * batches_per_epoch - 1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    trainable_params = list(model.parameters()) + (list(arc_head.parameters()) if arc_head else [])
    model_name = args.tag or f"dino_{args.arch}_{args.data_source}"
    dev_label = f"dev ({args.data_source})"
    metric_name = "acc" if args.loss == "arcface" else "active"

    print(f"LR backbone: {args.lr_backbone:.2e}  LR head: {args.lr_head:.2e}  loss: {args.loss}")
    print("\n--- Baseline evaluation (dev split) ---")
    baseline = evaluate_dev_rank1(bare_model, device, args, dataset)
    _log_result(args, baseline, model_name, dev_label, batch=0)
    # Select checkpoints on MRR when the dev eval provides it (smoother than
    # binary Rank@1), else Rank@1.
    sel_name = "MRR" if baseline.mrr is not None else "Rank@1"
    best_score = dev_selection_score(baseline)
    evals_no_gain = 0
    stop_training = False
    train_start = time.perf_counter()
    steps_done = 0
    epoch = start_epoch - 1  # defined even if the loop body never runs

    for epoch in range(start_epoch, args.epochs):
        if stop_training:
            break

        model.train()
        dataset._resample()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        total_loss, total_active, n_batches = 0.0, 0.0, 0
        global_batch = epoch * batches_per_epoch
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}", unit="batch", dynamic_ncols=True)

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                if args.loss == "arcface":
                    images, labels = batch
                    labels = labels.to(device)
                    embeddings = model(images.to(device))
                    logits = arc_head(embeddings, labels)
                    loss = criterion(logits, labels)
                    active = (logits.argmax(dim=1) == labels).float().mean()  # train acc
                else:
                    anchors, positives, negatives = batch
                    a = model(anchors.to(device))
                    p = model(positives.to(device))
                    n = model(negatives.to(device))
                    loss, active = criterion(a, p, n)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_active += active.item()
            n_batches += 1
            global_batch += 1
            steps_done += 1
            scheduler.step()
            pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", **{metric_name: f"{total_active/n_batches:.2%}"})

            if args.eval_every_batches > 0 and global_batch % args.eval_every_batches == 0:
                pbar.write(f"  --- Dev eval @ batch {global_batch} ---")
                result = evaluate_dev_rank1(bare_model, device, args, dataset)
                mrr_str = f" MRR {result.mrr:.4f}" if result.mrr is not None else ""
                pbar.write(f"  Rank@1 {result.accuracy:.4f}{mrr_str} ({result.correct}/{result.total})")
                _log_result(args, result, model_name, dev_label, batch=global_batch)
                model.train()
                score = dev_selection_score(result)
                if score > best_score:
                    best_score = score
                    evals_no_gain = 0
                    ckpt = out_dir / "backbone_best.pth"
                    torch.save(checkpoint_state(epoch + 1), str(ckpt))
                    pbar.write(f"  New best {sel_name} {best_score:.4f} -> saved {ckpt.name}")
                else:
                    evals_no_gain += 1
                    pbar.write(f"  No improvement ({evals_no_gain}/{args.patience}), best {sel_name}={best_score:.4f}")
                    if args.patience > 0 and evals_no_gain >= args.patience:
                        pbar.write("  Early stopping.")
                        stop_training = True
                        break

        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"  epoch {epoch+1:3d}  loss={avg_loss:.4f}  "
            f"{metric_name}={total_active/max(n_batches,1):.2%}  "
            f'bb_lr={optimizer.param_groups[0]["lr"]:.2e}'
        )

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt = out_dir / f"backbone_epoch{epoch+1:03d}.pth"
            torch.save(checkpoint_state(epoch + 1), str(ckpt))
            print(f"  Saved {ckpt.name}")

    elapsed = time.perf_counter() - train_start
    epochs_run = max(epoch + 1 - start_epoch, 1)
    imgs_per_step = args.batch_size if args.loss == "arcface" else 3 * args.batch_size
    write_timing_json(
        str(out_dir / "timing.json"),
        {
            "model": model_name,
            "data_source": args.data_source,
            "loss": args.loss,
            "dev_selection_metric": sel_name,
            "best_dev_score": best_score,
            "train_seconds": elapsed,
            "epochs_run": epochs_run,
            "seconds_per_epoch": elapsed / epochs_run,
            "steps": steps_done,
            "batch_size": args.batch_size,
            "images_per_sec": (steps_done * imgs_per_step / elapsed) if elapsed > 0 else None,
            "device": str(device),
            "gpu": args.gpu,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.test:
        apply_test_mode(args)

    if args.eval_only:
        device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")
        model = load_model(args.dinov2_dir, args.arch, args.weights, args.proj_dim)
        state = None
        if args.resume:
            state = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(state["model"])
            print(f"Loaded {args.resume}")
        resolve_crop_padding(args, state, "evaluation")
        model = model.to(device)
        model_name = args.tag or default_eval_tag(f"dino_{args.arch}", state)
        if args.eval_dataset in ("rectest", "both"):
            print("=== iCartoonFace rectest ===")
            result = evaluate_icartoonface_rank1(model, device, args)
            print(f"Rank@1: {result.accuracy:.4f} ({result.correct}/{result.total})")
            _log_result(args, result, model_name, "iCartoonFace rectest")
        if args.eval_dataset in ("film", "both"):
            print("\n=== data/film (per-movie) ===")
            result = evaluate_film_rank1(model, device, args)
            print(f"Rank@1: {result.accuracy:.4f} ({result.correct}/{result.total})")
            _log_result(args, result, model_name, "data/film (per-movie)")
        return

    train(args)


if __name__ == "__main__":
    main()
