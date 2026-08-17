"""Train an Ultralytics model (YOLO26 or RT-DETR) on a dataset setting built
by src/prepare/prepare_yolo_dataset.py.

Augmentation is aligned with the RetinaFace recipe by default (see
configs/augmentation.yaml): mosaic p=1.0 with a 10-epoch close-mosaic
wind-down, scale/translate, horizontal flip, HSV color jitter; mixup and
rotations disabled. Pass --native-aug for the framework defaults instead.

Examples (from the repo root):
    python -m src.train.train_ultralytics --arch yolo26 --setting wf_icf
    python -m src.train.train_ultralytics --arch rtdetr --setting icf --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ARCH_WEIGHTS = {
    "yolo26": "yolo26l.pt",
    "rtdetr": "rtdetr-l.pt",
}

# aligned with the RetinaFace recipe (crop zoom + expand, hflip 0.5,
# photometric distortion + extras, mosaic on with a close-mosaic wind-down);
# mixup/rotations off. documented in configs/augmentation.yaml
ALIGNED_AUG = dict(
    mosaic=1.0,
    close_mosaic=10,   # framework default: plain images for the last 10 epochs
    mixup=0.0,
    copy_paste=0.0,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    translate=0.1,
    scale=0.5,       # random resize 0.5-1.5x ~ RetinaFace crop zoom range
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    erasing=0.0,
)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=sorted(ARCH_WEIGHTS))
    parser.add_argument("--setting", required=True, choices=["wf", "icf", "wf_icf"])
    parser.add_argument("--model", default=None,
                        help="override checkpoint (default per --arch)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--native-aug", action="store_true",
                        help="use framework-default augmentation instead of the aligned recipe")
    parser.add_argument("--mosaic", type=float, default=None,
                        help="override mosaic probability (aligned default: 1.0; "
                             "match the value used for train_retinaface.py --mosaic)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-period", type=int, default=1,
                        help="save an epoch checkpoint every N epochs (for "
                             "best-checkpoint selection); 1 = every epoch, "
                             "matching the other stacks")
    parser.add_argument("--test", action="store_true",
                        help="smoke-test mode: ~10 training images (via dataset "
                             "fraction), <=2 epochs")
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    from ultralytics import RTDETR, YOLO

    data_yaml = REPO_ROOT / "datasets" / args.setting / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"{data_yaml} not found - run scripts/setup.sh "
                         f"(or src/prepare/prepare_yolo_dataset.py) first")

    weights = args.model or ARCH_WEIGHTS[args.arch]
    model = RTDETR(weights) if args.arch == "rtdetr" else YOLO(weights)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(REPO_ROOT / "runs" / "ultralytics"),
        name=f"{args.arch}_{args.setting}",
        exist_ok=True,
        single_cls=True,
        resume=args.resume,
        save_period=args.save_period,
    )
    if not args.native_aug:
        train_kwargs.update(ALIGNED_AUG)
    if args.mosaic is not None:
        train_kwargs["mosaic"] = args.mosaic
    if args.arch == "rtdetr":
        from src.train.rtdetr_cap import CappedRTDETRTrainer
        train_kwargs["trainer"] = CappedRTDETRTrainer
    if args.test:
        n_train = sum(1 for _ in (REPO_ROOT / "datasets" / args.setting
                                  / "images" / "train").iterdir())
        train_kwargs["fraction"] = min(1.0, max(10 / n_train, 1e-6))
        train_kwargs["epochs"] = args.epochs = min(args.epochs, 2)
        print(f"--test: fraction {train_kwargs['fraction']:.6f} "
              f"(~10 of {n_train} images), epochs {args.epochs}")

    import json
    import time

    import torch

    t0 = time.time()
    model.train(**train_kwargs)
    train_seconds = time.time() - t0

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    timing = {
        "model": f"{args.arch} ({weights})",
        "setting": args.setting,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / max(args.epochs, 1), 1),
        "native_aug": args.native_aug,
        "mosaic": train_kwargs.get("mosaic", "framework-default"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    out = (Path(save_dir) if save_dir
           else REPO_ROOT / "runs" / "ultralytics" / f"{args.arch}_{args.setting}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    print("Training timing:", timing)
    if save_dir:
        print(f"\nBest weights: {save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
