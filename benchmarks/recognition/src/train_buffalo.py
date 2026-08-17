"""Fine-tunes InsightFace buffalo_l's w600k_r50.onnx recognition backbone. The
default objective is an ArcFace margin-softmax classifier over the training
identities (--loss arcface, the same objective w600k_r50 itself was trained
with); --loss triplet selects a triplet loss instead, and --data-source tracks
always uses triplet (its negatives must be co-occurring faces - no global
identity labels there). Shares its training data handling and evaluation
protocol with src/train_dino.py's DINOv2 fine-tuning script. Two training data
sources are supported (--data-source):

  "identity" (default) - iCartoonFace rectrain, one identity per subdirectory:
    {train_dir}/{identity_dir}/{image}.jpg
    {train_det_file}: "<identity_dir>/<image>.jpg  x1  y1  x2  y2" per image
    Positive pairs — two images of the same identity
    Negative pairs — one image from a different, randomly chosen identity

  "tracks" - paired face-track data:
    {data_dir}/{mov_id}/{top_frame}/{track_no}/*.png
    (already-cropped face images - no separate bbox file)
    Positive pairs — two images from the same {mov_id}/{track_no}
    Negative pairs — two images from different track_nos within the same
                     {mov_id}/{top_frame} (co-occurring, guaranteed
                     different people). See src/train_dino.py's
                     TripletTrackDataset (imported here) for details.

Evaluation: mid-training checkpoint selection uses a held-out **dev split**
carved out of the training data itself (--dev-fraction, default 0.1;
train/dev splitting and the dev Rank@1 eval functions live in
src/train_dino.py, shared by both scripts) - NOT the official iCartoonFace
rectest split or data/film, both of which are test data held out entirely for
final, post-hoc comparison (src/evaluate_baseline.py, src/evaluate_film.py,
or this script's own --eval-only against rectest) and never touched during
training. See src/train_dino.py's module docstring for the full split
design (identity: held-out identities; tracks: held-out co-occurrence
frames - every track appearing in a held-out frame is excluded from
training, and each track's first/last-frame pair is ranked against the
other, guaranteed-different-identity tracks in the same frame).

Usage:
    python -m src.train_buffalo \\
        --data-source identity \\
        --train-dir data/raw/personai_icartoonface_rectrain/icartoonface_rectrain \\
        --train-det-file data/raw/personai_icartoonface_rectrain/icartoonface_rectrain_det.txt \\
        --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx \\
        --output-dir outputs/buffalo_finetuned \\
        --epochs 20 --batch-size 64 --gpu 0 --fp16

    python -m src.train_buffalo \\
        --data-source tracks \\
        --data-dir /path/to/paired_face_tracks \\
        --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx \\
        --output-dir outputs/buffalo_tracks \\
        --epochs 20 --batch-size 64 --gpu 0 --fp16

    # Eval only
    python -m src.train_buffalo --eval-only \\
        --resume outputs/buffalo_finetuned/backbone_best.pth \\
        --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx

    # Export a fine-tuned checkpoint back to ONNX (for use with insightface/onnxruntime)
    python -m src.train_buffalo --export-only \\
        --resume outputs/buffalo_finetuned/backbone_best.pth \\
        --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx \\
        --output-dir outputs/buffalo_finetuned
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.evaluate import rank1_identification_accuracy as icartoonface_rank1
from src.evaluate_film import rank1_identification_accuracy as film_rank1
from src.models.arcface import ArcFaceHead
from src.models.buffalo_backbone import (
    INPUT_SIZE,
    BuffaloEvalAdapter,
    BuffaloTorchModel,
    export_onnx,
    load_backbone_from_onnx,
)
from src.report import (
    EvalResult,
    append_json_record,
    append_markdown_result,
    result_to_record,
    write_timing_json,
)
from src.train_dino import (
    IdentityClassificationDataset,
    TripletIdentityDataset,
    TripletTrackDataset,
    apply_test_mode,
    default_eval_tag,
    dev_selection_score,
    evaluate_identity_dev_rank1,
    evaluate_track_dev_rank1,
    load_movie_id_list,
    resolve_crop_padding,
)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--onnx",
        default="~/.insightface/models/buffalo_l/w600k_r50.onnx",
        help="Path to buffalo_l's w600k_r50.onnx (downloaded by src/models/insightface_backbone.py on first use)",
    )

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
    p.add_argument("--output-dir", default="outputs/buffalo_finetuned")
    p.add_argument(
        "--resume",
        default=None,
        help="Resume an interrupted run from a checkpoint saved by this script - continues the "
        "epoch counter and LR schedule from where it left off (start_epoch = the checkpoint's "
        "saved epoch). Also used by --eval-only/--export-only to pick which checkpoint to load. "
        "Mutually exclusive with --init-weights for training.",
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
    p.add_argument("--export-only", action="store_true", help="Export --resume checkpoint back to ONNX and exit")
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
        "(default: buffalo_l_finetuned_<data-source> when training; buffalo_l[_finetuned] for "
        "--eval-only). Set a clean, stable tag so src/collect_results.py groups rows sensibly.",
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
    p.add_argument("--triplets-per-epoch", type=int, default=10_000, help="[--loss triplet] triplets sampled per epoch")
    p.add_argument("--margin", type=float, default=0.3, help="[--loss triplet] triplet margin")
    p.add_argument(
        "--lr-backbone", type=float, default=1e-5,
        help="LR for the buffalo_l backbone parameters (the whole recognition network)",
    )
    p.add_argument(
        "--lr-head", type=float, default=1e-3,
        help="LR for the ArcFace classifier head (--loss arcface); it starts from a random init "
        "and adapts faster than the pretrained backbone. Unused for --loss triplet.",
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


# ---------------------------------------------------------------------------
# Transform (mean=std=0.5 @ 112x112 - buffalo_l's native convention, same
# normalization src/datasets/icartoonface.py's EVAL_TRANSFORM already uses,
# just a different resolution). Only needed for training - evaluation goes
# through EVAL_TRANSFORM via BuffaloEvalAdapter instead, to share the exact
# same preprocessing every other model's Rank@1 eval uses.
# ---------------------------------------------------------------------------


def make_train_transform(img_size: int = INPUT_SIZE):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


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
# evaluate_dev_rank1() is the ONLY evaluation used for mid-training
# checkpoint selection - it runs Rank@1 on data held out from training
# itself (--dev-fraction, via src/train_dino.py's evaluate_identity_dev_rank1
# / evaluate_track_dev_rank1), never on test data.
#
# evaluate_icartoonface_rank1() is TEST DATA - the official iCartoonFace
# rectest Rank@1 protocol (src/evaluate.py) - used exclusively by
# --eval-only for a final, post-hoc accuracy check against a saved
# checkpoint, and never called from train()/the training loop.
# ---------------------------------------------------------------------------


def evaluate_dev_rank1(buffalo_model: BuffaloTorchModel, device, args, dataset) -> EvalResult:
    eval_model = BuffaloEvalAdapter(buffalo_model).to(device)
    eval_model.eval()
    if args.data_source == "tracks":
        return evaluate_track_dev_rank1(eval_model, device, dataset)
    return evaluate_identity_dev_rank1(
        eval_model, device, dataset, padding=args.crop_padding,
        probes_per_identity=args.dev_probes_per_identity,
    )


def evaluate_icartoonface_rank1(buffalo_model: BuffaloTorchModel, device, args) -> EvalResult:
    """TEST DATA - the official iCartoonFace rectest split. Only called by
    --eval-only (main()), never from train()."""
    eval_model = BuffaloEvalAdapter(buffalo_model).to(device)
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


def evaluate_film_rank1(buffalo_model: BuffaloTorchModel, device, args) -> EvalResult:
    """TEST DATA - data/film's per-movie protocol. Only called by --eval-only
    --eval-dataset film (main()), never from train()."""
    eval_model = BuffaloEvalAdapter(buffalo_model).to(device)
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
    # has: for tracks, negatives must be co-occurring faces (two tracks in
    # different frames may be the same, unlabeled person), so a classification
    # objective isn't defined. Force triplet there.
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
            transform=make_train_transform(INPUT_SIZE),
            seed=args.seed,
            only_movie_ids=only_movie_ids,
        )
    elif loss_type == "arcface":
        dataset = IdentityClassificationDataset(
            args.train_dir,
            args.train_det_file,
            padding=args.crop_padding,
            dev_fraction=args.dev_fraction,
            transform=make_train_transform(INPUT_SIZE),
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
            transform=make_train_transform(INPUT_SIZE),
            seed=args.seed,
            limit_identities=args.limit_identities,
        )

    raw_backbone = load_backbone_from_onnx(str(Path(args.onnx).expanduser()))
    model = BuffaloTorchModel(raw_backbone)

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

    # For --loss arcface, a training-time classifier head over the training
    # identities (not saved with the backbone checkpoint - eval uses the
    # embedding directly), trained at --lr-head.
    arc_head = None
    param_groups = [{"params": bare_model.parameters(), "lr": args.lr_backbone}]
    if args.loss == "arcface":
        arc_head = ArcFaceHead(
            bare_model.embedding_dim, dataset.num_classes, scale=args.arc_scale, margin=args.arc_margin
        ).to(device)
        param_groups.append({"params": arc_head.parameters(), "lr": args.lr_head})
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

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

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
    model_name = args.tag or f"buffalo_l_finetuned_{args.data_source}"
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
            f'lr={optimizer.param_groups[0]["lr"]:.2e}'
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

    # Export the dev-selected best checkpoint, not whatever weights the loop
    # ended on - the .onnx artifact should match backbone_best.pth.
    best_ckpt = out_dir / "backbone_best.pth"
    if best_ckpt.exists():
        bare_model.load_state_dict(torch.load(str(best_ckpt), map_location="cpu")["model"])
        print(f"\nExporting to ONNX (from {best_ckpt.name}) ...")
    else:
        print("\nExporting to ONNX (no backbone_best.pth - using final weights) ...")
    export_onnx(bare_model.backbone, str(out_dir / "buffalo_finetuned.onnx"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.test:
        apply_test_mode(args)
    onnx_path = str(Path(args.onnx).expanduser())

    if args.export_only:
        if not args.resume:
            sys.exit("--export-only requires --resume")
        raw_backbone = load_backbone_from_onnx(onnx_path)
        model = BuffaloTorchModel(raw_backbone)
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_onnx(model.backbone, str(out_dir / "buffalo_finetuned.onnx"))
        return

    if args.eval_only:
        device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")
        raw_backbone = load_backbone_from_onnx(onnx_path)
        model = BuffaloTorchModel(raw_backbone)
        state = None
        if args.resume:
            state = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(state["model"])
            print(f"Loaded {args.resume}")
        resolve_crop_padding(args, state, "evaluation")
        model = model.to(device)
        model_name = args.tag or default_eval_tag("buffalo_l", state)
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
