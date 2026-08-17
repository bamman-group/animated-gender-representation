"""Train YuNet via the vendored libfacedetection.train code on a dataset
setting built by src/prepare/prepare_yunet_dataset.py.

Runs the upstream trainer in-process and injects MosaicWIDERFaceDataset
(src/train/yunet_mosaic.py) in place of the upstream dataset class, giving
YuNet the same mosaic augmentation as the other stacks WITHOUT modifying the
vendored code. --native-aug (or --mosaic 0) restores the stock upstream pipeline.

Training initializes from the released WIDER-trained checkpoint
(third_party/libfacedetection.train/weights/<variant>.pth) for parity with the other stacks
(RetinaFace fine-tunes from its WIDER checkpoint, YOLO26/RT-DETR from COCO);
--pretrained_weights none trains from scratch (upstream's own recipe, which
expects ~640 epochs).

The upstream WIDER recipe is 640 epochs with lr steps at 400/544; the
defaults here scale that schedule to --epochs.

The wf setting needs no training: evaluate the released checkpoint
third_party/libfacedetection.train/weights/yunet_n.pth directly.

Example (from the repo root):
    python -m src.train.train_yunet --setting wf_icf --epochs 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "libfacedetection.train"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=["icf", "wf_icf"])
    parser.add_argument("--variant", default="yunet_n", choices=["yunet_n", "yunet_s"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002,
                        help="fine-tuning LR (upstream's 0.01 is a from-scratch "
                             "rate and diverges when fine-tuning with mosaic)")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="gradient-norm clip (upstream trains unclipped; "
                             "crowd-mosaic batches spike gradients). 0 = off")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="mosaic probability (aligned default 1.0; 0 = off)")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="approx. mosaic-free final epochs (per-worker estimate)")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                        help="save an epoch checkpoint every N epochs (for "
                             "best-checkpoint selection); 1 = every epoch, "
                             "matching the other stacks")
    parser.add_argument("--pretrained-weights", default="released",
                        help="'released' = third_party/libfacedetection.train/"
                             "weights/<variant>.pth; "
                             "a path; or 'none' for from-scratch")
    parser.add_argument("--native-aug", action="store_true",
                        help="stock upstream augmentation (implies --mosaic 0)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--test", action="store_true",
                        help="smoke-test mode: 10 training images, <=2 epochs")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra args passed through to yunet_train.cli.train")
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    if args.test:
        args.epochs = min(args.epochs, 2)
        print("--test: 10 training images, epochs", args.epochs)
    if args.native_aug:
        args.mosaic = 0.0

    ann_file = REPO_ROOT / "datasets" / f"yunet_{args.setting}" / "labelv2.txt"
    img_prefix = REPO_ROOT / "datasets" / f"yunet_{args.setting}" / "images"
    if not ann_file.is_file():
        raise SystemExit(f"{ann_file} not found - run scripts/setup.sh "
                         f"(or src/prepare/prepare_yunet_dataset.py) first")

    work_dir = REPO_ROOT / "runs" / "yunet" / args.setting
    # scale the upstream 640-epoch recipe (steps at 400/544) to --epochs
    lr_steps = [max(1, round(args.epochs * 400 / 640)),
                max(2, round(args.epochs * 544 / 640))]

    import yunet_train.cli.train as upstream_train
    from yunet_train.engine import load_model_weights_only
    from src.train.yunet_mosaic import MosaicWIDERFaceDataset

    # inject gradient clipping: the engine loop supports grad_clip_norm but
    # the face trainer never passes it, and unclipped training diverges to
    # NaN on crowd-mosaic batches (surfaces as a CUDA assert in the assigner)
    if args.grad_clip > 0:
        _orig_train_one_epoch = upstream_train.train_one_epoch

        def _clipped_train_one_epoch(**kwargs):
            kwargs.setdefault("grad_clip_norm", args.grad_clip)
            return _orig_train_one_epoch(**kwargs)

        upstream_train.train_one_epoch = _clipped_train_one_epoch
        print(f"gradient clipping injected (max norm {args.grad_clip})")

    # initialize from the released checkpoint (fine-tuning parity with the
    # other stacks) by wrapping the model factory the trainer calls by name
    pretrained = args.pretrained_weights
    if pretrained == "released":
        pretrained = str(REPO_ROOT / "third_party" / "libfacedetection.train"
                         / "weights" / f"{args.variant}.pth")
    if pretrained and pretrained.lower() != "none":
        if not Path(pretrained).is_file():
            raise SystemExit(f"pretrained weights not found: {pretrained}")
        upstream_build = upstream_train.build_yunet

        def build_and_load(variant):
            model = upstream_build(variant)
            load_model_weights_only(pretrained, model=model, map_location="cpu")
            print(f"initialized from {pretrained}")
            return model

        upstream_train.build_yunet = build_and_load
    else:
        print("training from scratch (no face-detection pretraining)")

    MosaicWIDERFaceDataset.MOSAIC_PROB = args.mosaic
    MosaicWIDERFaceDataset.IMAGE_SIZE = 640
    MosaicWIDERFaceDataset.TOTAL_EPOCHS = args.epochs
    MosaicWIDERFaceDataset.CLOSE_EPOCHS = args.close_mosaic
    MosaicWIDERFaceDataset.WORKERS = max(args.workers, 1)
    upstream_train.WIDERFaceDataset = MosaicWIDERFaceDataset
    print(f"mosaic p={args.mosaic}"
          + (f" (close_mosaic ~{args.close_mosaic} epochs)" if args.mosaic else ""))

    cli = ["--variant", args.variant,
           "--ann-file", str(ann_file),
           "--img-prefix", str(img_prefix),
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--lr", str(args.lr),
           "--lr-steps", *[str(s) for s in lr_steps],
           "--workers", str(args.workers),
           # in-training eval expects WIDER val under the vendored repo's data/;
           # push it past the horizon and evaluate with src/eval instead
           "--eval-interval", str(args.epochs + 1),
           "--checkpoint-interval", str(args.checkpoint_interval),
           "--work-dir", str(work_dir)]
    if args.test:
        cli += ["--limit-samples", "10"]
    if args.resume:
        cli += ["--resume", args.resume]
    cli += args.extra

    old_argv = sys.argv
    sys.argv = ["yunet_train.cli.train"] + cli
    try:
        upstream_args = upstream_train.parse_args()
    finally:
        sys.argv = old_argv

    t0 = time.time()
    upstream_train.run_training(upstream_args)
    train_seconds = time.time() - t0

    work_dir.mkdir(parents=True, exist_ok=True)
    timing = {
        "model": f"yunet ({args.variant})",
        "setting": args.setting,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "native_aug": args.native_aug,
        "pretrained_weights": args.pretrained_weights,
        "lr": args.lr, "grad_clip": args.grad_clip,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / max(args.epochs, 1), 1),
    }
    with open(work_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print("Training timing:", timing)
    print(f"\nCheckpoints in {work_dir} (select with src/eval/select_best.py)")


if __name__ == "__main__":
    main()
