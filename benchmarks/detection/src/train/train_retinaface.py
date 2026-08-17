"""Fine-tune RetinaFace (ResNet-50) from the WIDER Face-pretrained checkpoint.

Settings:
    icf     iCartoonFace only
    wf_icf  iCartoonFace + WIDER Face mixed (uniformly shuffled)
    (wf     = the released checkpoint itself; no training run — evaluate
              weights/Resnet50_Final.pth directly)

Example (from the repo root):
    python -m src.train.train_retinaface --setting wf_icf --epochs 40 \
        --batch-size 16 --amp
"""
from __future__ import annotations

import argparse
import datetime
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Pytorch_Retinaface"))

from data import cfg_mnet, cfg_re50, detection_collate, preproc  # noqa: E402
from data.icartoonface import ICartoonFaceDetection  # noqa: E402
from layers.functions.prior_box import PriorBox  # noqa: E402
from layers.modules import MultiBoxLoss  # noqa: E402
from models.retinaface import RetinaFace  # noqa: E402

SETTINGS = {
    "icf": [("labels/icf_train45.txt", "data/icartoonface/dettrain")],
    "wf_icf": [("labels/icf_train45.txt", "data/icartoonface/dettrain"),
               ("labels/wf_train.txt", "data/widerface/WIDER_train/images")],
}

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    parser.add_argument("--network", default="resnet50", choices=["mobile0.25", "resnet50"])
    parser.add_argument("--pretrained-weights",
                        default=str(REPO_ROOT / "weights/Resnet50_Final.pth"),
                        help="checkpoint to fine-tune from; 'none' = ImageNet backbone only")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--decay-epochs", type=int, nargs="*", default=None,
                        help="lr decay epochs; default 60%% and 85%% of --epochs")
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-images", type=int, default=0,
                        help="use only the first N images of each source (quick tests)")
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="probability of YOLO-style 4-image mosaic per sample "
                             "(aligned default 1.0, matching ultralytics; 0 = off)")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="disable mosaic for the last N epochs (matches "
                             "ultralytics close_mosaic)")
    parser.add_argument("--expand", type=float, default=0.5,
                        help="zoom-out probability (paste on a larger mean canvas); "
                             "aligns with the ultralytics scale 0.5-1.5 range. 0 = off")
    parser.add_argument("--expand-max", type=float, default=2.0,
                        help="max zoom-out canvas ratio")
    parser.add_argument("--extras", type=float, default=0.01,
                        help="per-op probability of blur/median-blur/grayscale/CLAHE "
                             "(mirrors ultralytics+albumentations). 0 = off")
    parser.add_argument("--native-aug", action="store_true",
                        help="use the stock Pytorch_Retinaface augmentation only "
                             "(crop/distort/flip); disables mosaic, expand, and "
                             "the photometric extras")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-net", default=None)
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument("--save-folder", default=None,
                        help="default: runs/retinaface/<setting>/")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="run icartoon validation every N epochs (0 = off)")
    parser.add_argument("--eval-num-images", type=int, default=2000)
    parser.add_argument("--test", action="store_true",
                        help="smoke-test mode: 10 images per source, <=2 epochs, "
                             "batch <=4, in-training validation on 1000 images")
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def main():
    global args
    args = parse_args()
    if args.test:
        args.num_images = 10
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 4)
        args.eval_num_images = 1000
        print("--test: 10 images/source, epochs", args.epochs,
              ", batch", args.batch_size)

    save_folder = args.save_folder or str(REPO_ROOT / "runs" / "retinaface" / args.setting)
    os.makedirs(save_folder, exist_ok=True)
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50

    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    img_dim = cfg["image_size"]

    net = RetinaFace(cfg=cfg)
    if args.resume_net:
        print("Resuming from", args.resume_net)
        net.load_state_dict(strip_module_prefix(
            torch.load(args.resume_net, map_location="cpu")))
    elif args.pretrained_weights and args.pretrained_weights.lower() != "none":
        print("Fine-tuning from", args.pretrained_weights)
        net.load_state_dict(strip_module_prefix(
            torch.load(args.pretrained_weights, map_location="cpu")))
    else:
        print("No face-detection pretraining; ImageNet backbone only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=args.amp)  # noqa: E731
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=args.amp)  # noqa: E731

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward().to(device)

    if args.native_aug:
        if args.mosaic or args.expand or args.extras:
            print("--native-aug set: ignoring --mosaic/--expand/--extras")
        args.mosaic, args.expand, args.extras = 0.0, 0.0, 0.0
        pp = preproc(img_dim, rgb_mean)
    else:
        from data.extra_augment import ExtendedPreproc
        pp = ExtendedPreproc(img_dim, rgb_mean, expand_prob=args.expand,
                             expand_max=args.expand_max, extras_prob=args.extras)
    parts = []
    for label_file, images_root in SETTINGS[args.setting]:
        # with mosaic, datasets yield raw samples and preproc runs on the
        # composite inside MosaicDetection
        ds = ICartoonFaceDetection(str(REPO_ROOT / label_file),
                                   str(REPO_ROOT / images_root),
                                   None if args.mosaic > 0 else pp)
        if args.num_images > 0:
            ds.imgs_path = ds.imgs_path[:args.num_images]
            ds.annotations = ds.annotations[:args.num_images]
        print(f"  {label_file}: {len(ds)} images")
        parts.append(ds)
    dataset = parts[0] if len(parts) == 1 else data.ConcatDataset(parts)
    if args.mosaic > 0:
        from data.mosaic import MosaicDetection
        dataset = MosaicDetection(dataset, pp, img_dim, prob=args.mosaic)
        print(f"Mosaic augmentation enabled (p={args.mosaic})")
    print(f"Loaded {len(dataset)} training images total")

    loader = data.DataLoader(dataset, args.batch_size, shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=detection_collate,
                             pin_memory=True, drop_last=True)

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    max_iter = args.epochs * epoch_size
    warmup_iters = min(args.warmup_iters, max(1, max_iter // 10))
    decay_epochs = args.decay_epochs
    if not decay_epochs:
        decay_epochs = sorted({max(1, round(args.epochs * 0.6)),
                               max(2, round(args.epochs * 0.85))})
    print(f"lr decays x{args.gamma} at epochs {decay_epochs}")

    def lr_at(iteration, epoch):
        if iteration < warmup_iters:
            return args.lr * (iteration + 1) / warmup_iters
        step = sum(1 for e in decay_epochs if epoch >= e)
        return args.lr * (args.gamma ** step)

    net.train()
    iteration = args.resume_epoch * epoch_size
    train_seconds = 0.0
    t_run = time.time()
    mosaic_closed = False
    for epoch in range(args.resume_epoch, args.epochs):
        # close-mosaic wind-down: plain-augmentation epochs at the end
        # (workers respawn per epoch, so mutating the dataset takes effect)
        if (args.mosaic > 0 and not mosaic_closed
                and epoch >= args.epochs - args.close_mosaic):
            if hasattr(dataset, "prob"):
                dataset.prob = 0.0
            mosaic_closed = True
            print(f"close_mosaic: mosaic off from epoch {epoch + 1}")
        t_epoch = time.time()
        for images, targets in loader:
            t0 = time.time()
            lr = lr_at(iteration, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = images.to(device, non_blocking=True)
            targets = [t.to(device) for t in targets]

            with autocast():
                out = net(images)
                # no landmark supervision in this benchmark: every face is
                # written with flag -1, so the criterion's landmark term is
                # identically 0 and is left out of the objective entirely.
                # (The model keeps its landmark head; it is simply never
                # supervised, and the evaluator already discards its output.)
                loss_l, loss_c, _ = criterion(out, priors, targets)
                loss = cfg["loc_weight"] * loss_l + loss_c

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iteration += 1
            if iteration % 50 == 0:
                eta = int((time.time() - t0) * (max_iter - iteration))
                print("Epoch {}/{} Iter {}/{} || Loc: {:.4f} Cla: {:.4f} "
                      "|| LR: {:.6f} || {:.2f} s/it || ETA {}".format(
                          epoch + 1, args.epochs, iteration, max_iter,
                          loss_l.item(), loss_c.item(),
                          lr, time.time() - t0,
                          str(datetime.timedelta(seconds=eta))))

        state = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
        ckpt = os.path.join(save_folder, f"epoch_{epoch + 1}.pth")
        torch.save(state, ckpt)
        epoch_seconds = time.time() - t_epoch
        train_seconds += epoch_seconds
        print(f"Epoch {epoch + 1} done in {epoch_seconds:.0f}s, saved {ckpt}")

        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"--- validation after epoch {epoch + 1} ---", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "src.eval.evaluate_retinaface",
                 "--trained-model", ckpt, "--network", args.network,
                 "--eval-set", "icartoon_val",
                 "--num-images", str(args.eval_num_images),
                 "--tag", f"{args.setting}-epoch{epoch + 1}",
                 "--results-file", os.path.join(save_folder, "val_curve.md")],
                cwd=str(REPO_ROOT))
            if result.returncode != 0:
                print("validation failed (training continues)", flush=True)

    final = os.path.join(save_folder, "Resnet50_Final.pth")
    state = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
    torch.save(state, final)
    print("Saved final model to", final)

    n_epochs = args.epochs - args.resume_epoch
    timing = {
        "model": f"retinaface-{args.network}",
        "setting": args.setting,
        "epochs": n_epochs,
        "batch_size": args.batch_size,
        "images_per_epoch": len(dataset),
        "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / max(n_epochs, 1), 1),
        "images_per_second": round(len(dataset) * n_epochs / max(train_seconds, 1e-9), 1),
        "wall_seconds_incl_validation": round(time.time() - t_run, 1),
        "amp": args.amp,
        "native_aug": args.native_aug,
        "mosaic": args.mosaic, "close_mosaic": args.close_mosaic,
        "expand": args.expand, "extras": args.extras,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    import json
    with open(os.path.join(save_folder, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print("Training timing:", timing)


if __name__ == "__main__":
    main()
