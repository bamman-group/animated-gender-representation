from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yunet_train.tasks.face import (
    PredictionDict,
    WIDERFaceDataset,
    WIDER_VAL_ANN_FILE,
    WIDER_VAL_GT_DIR,
    WIDER_VAL_IMAGE_DIR,
    YuNetPostprocessor,
    add_prediction,
    build_eval_transforms,
    build_yunet,
    collate_face_samples,
    detections_to_widerface,
    move_batch_to_device,
    wider_evaluation,
    write_widerface_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YuNet checkpoint on WIDER Face val.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--variant", choices=("yunet_n", "yunet_s"), default=None)
    parser.add_argument("--ann-file", type=Path, default=WIDER_VAL_ANN_FILE)
    parser.add_argument("--img-prefix", type=Path, default=WIDER_VAL_IMAGE_DIR)
    parser.add_argument("--gt-dir", type=Path, default=WIDER_VAL_GT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/widerface_eval"))
    parser.add_argument("--mode", choices=("origin", "resize"), default="origin")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--size-divisor", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=-1)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--save-preds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aps = run_evaluation(args)
    print(f"WIDERFace AP easy={aps.easy:.6f} medium={aps.medium:.6f} hard={aps.hard:.6f}")


def run_evaluation(args: argparse.Namespace):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    variant = args.variant or checkpoint.get("config", {}).get("variant", "yunet_n")
    model = build_yunet(variant)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(_clean_state_dict(state_dict), strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint does not match model variant "
            f"{variant!r}; missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model.to(device).eval()

    dataset = WIDERFaceDataset(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        transform=_build_transform(args),
        test_mode=True,
    )
    if args.limit_samples is not None:
        dataset.records = dataset.records[: args.limit_samples]
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_face_samples,
        pin_memory=device.type == "cuda",
    )
    postprocessor = YuNetPostprocessor(
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
    )
    collect_start = perf_counter()
    predictions = collect_predictions(model, postprocessor, data_loader, device)
    print(f"Collected predictions for {len(dataset)} images in {perf_counter() - collect_start:.2f}s", flush=True)
    if args.save_preds:
        write_start = perf_counter()
        write_widerface_predictions(predictions, args.output_dir / "predictions")
        print(f"Wrote WIDER Face predictions in {perf_counter() - write_start:.2f}s", flush=True)
    eval_start = perf_counter()
    print("Computing WIDER Face AP for easy/medium/hard splits...", flush=True)
    aps = wider_evaluation(predictions, args.gt_dir, iou_thresh=args.iou_threshold)
    print(f"Computed WIDER Face AP in {perf_counter() - eval_start:.2f}s", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aps.txt").write_text(
        f"{aps.easy:.6f},{aps.medium:.6f},{aps.hard:.6f}\n",
        encoding="utf-8",
    )
    return aps


def _build_transform(args: argparse.Namespace):
    if args.mode == "origin":
        return build_eval_transforms(image_size=None, size_divisor=args.size_divisor)
    return build_eval_transforms(image_size=args.image_size)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    postprocessor: YuNetPostprocessor,
    data_loader: DataLoader,
    device: torch.device,
) -> PredictionDict:
    predictions: PredictionDict = {}
    for batch in tqdm(data_loader, desc="WIDERFace val"):
        batch = move_batch_to_device(batch, device)
        results = postprocessor(model(batch.images))
        for result, meta in zip(results, batch.metas):
            boxes = detections_to_widerface(result, meta)
            add_prediction(predictions, meta["filename"], boxes)
    return predictions


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


if __name__ == "__main__":
    main()
