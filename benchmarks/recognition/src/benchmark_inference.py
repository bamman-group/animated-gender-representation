import argparse
import datetime
import json
import random
import time
from pathlib import Path

import torch

from src.datasets.icartoonface import DEFAULT_CROP_PADDING
from src.evaluate import BBox, embed_paths
from src.evaluate_baseline import has_nvidia_gpu
from src.evaluate_film import collect_movie_faces
from src.models.buffalo_backbone import BuffaloEvalAdapter, BuffaloTorchModel, load_backbone_from_onnx
from src.models.dino_backbone import DinoEvalAdapter, load_model as load_dino_model, load_model_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--film-images-dir", default="data/film/images")
    p.add_argument("--film-annotations", default="data/film/annotations.json")
    p.add_argument(
        "--crop-padding", type=float, default=DEFAULT_CROP_PADDING,
        help="Fraction of each face bbox's width/height added on each side before cropping "
        "(matches src/datasets/icartoonface.py's crop_face/DEFAULT_CROP_PADDING) - same padding "
        "applied identically for every model benchmarked.",
    )
    p.add_argument(
        "--max-faces", type=int, default=0,
        help="Sample this many faces out of every face found in --film-annotations (0 = use all "
        "~3,700 faces across all 100 movies)",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for the face sample when --max-faces > 0")
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="SAME batch size used for every model - the whole point of this benchmark is a fair comparison.",
    )
    p.add_argument(
        "--warmup-batches", type=int, default=2,
        help="Batches run (untimed) before the timed pass, to absorb cudnn/onnxruntime warmup cost.",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0, help="GPU index for the torch-based models (buffalo fine-tuned, DINOv2)")
    p.add_argument("--cpu-threads", type=int, default=4)

    p.add_argument("--buffalo-onnx", default="~/.insightface/models/buffalo_l/w600k_r50.onnx")
    p.add_argument("--skip-buffalo-baseline", action="store_true")
    p.add_argument(
        "--buffalo-checkpoint", default=None,
        help="Fine-tuned buffalo_l backbone_best.pth (src/train_buffalo.py) - if given, also benchmarks this",
    )

    p.add_argument("--dinov2-dir", default="third_party/dinov2")
    p.add_argument("--dino-vitb14-weights", default="data/dinov2_weights/dinov2_vitb14_pretrain.pth")
    p.add_argument("--dino-vitb14-checkpoint", default=None, help="Fine-tuned dino_vitb14 backbone_best.pth")
    p.add_argument("--skip-dino-vitb14", action="store_true")
    p.add_argument("--dino-vitl14-weights", default="data/dinov2_weights/dinov2_vitl14_pretrain.pth")
    p.add_argument("--dino-vitl14-checkpoint", default=None, help="Fine-tuned dino_vitl14 backbone_best.pth")
    p.add_argument("--skip-dino-vitl14", action="store_true")
    p.add_argument("--dino-img-size", type=int, default=224)

    p.add_argument(
        "--variant-label", default="fine-tuned",
        help="Text used inside the parens for every fine-tuned row's name, e.g. 'buffalo_l "
        "(<variant-label>)' - matches this repo's \"<model> (<variant>, crop N%%)\" --tag convention "
        "(src/collect_results.py's --table variants). Lets one invocation's --*-checkpoint rows stay "
        "distinguishable from another's when benchmarking several variants/crops in separate runs "
        "against the same --output file.",
    )
    p.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip every baseline (buffalo_l and both DINOv2 archs) in this run - useful when "
        "benchmarking several fine-tuned variants/crops against one shared --output file, since the "
        "(checkpoint-independent) baselines only need measuring once.",
    )

    p.add_argument("--output", default=None, help="Append a markdown timing table row per model to this file")
    return p.parse_args()


def collect_film_faces(
    images_dir: str, annotations_file: str, max_faces: int, seed: int
) -> tuple[list[Path], list[BBox]]:
    root = Path(images_dir)
    ann_path = Path(annotations_file)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found.")
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} not found.")

    with open(ann_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    paths: list[Path] = []
    bboxes: list[BBox] = []
    for movie in movies:
        movie_id = movie["movie_id"]
        movie_dir = root / movie_id
        for frame_filename, _cluster_id, bbox in collect_movie_faces(movie):
            paths.append(movie_dir / frame_filename)
            bboxes.append(bbox)

    if not paths:
        raise FileNotFoundError(f"No faces found in {ann_path}")
    if max_faces and max_faces < len(paths):
        idx = random.Random(seed).sample(range(len(paths)), max_faces)
        paths = [paths[i] for i in idx]
        bboxes = [bboxes[i] for i in idx]
    return paths, bboxes


@torch.no_grad()
def benchmark_model(
    name: str, model: torch.nn.Module, device: torch.device, paths: list, bboxes: list,
    padding: float, batch_size: int, num_workers: int, warmup_batches: int,
) -> tuple[float, float]:
    model = model.to(device).eval()

    warmup_n = min(len(paths), warmup_batches * batch_size)
    if warmup_n:
        embed_paths(
            model, paths[:warmup_n], bboxes[:warmup_n], device,
            batch_size=batch_size, num_workers=num_workers, padding=padding, desc=f"{name} (warmup)",
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    embeddings = embed_paths(
        model, paths, bboxes, device,
        batch_size=batch_size, num_workers=num_workers, padding=padding, desc=name,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    images_per_sec = len(paths) / elapsed if elapsed > 0 else float("nan")
    print(
        f"  {name}: {len(paths)} faces in {elapsed:.2f}s "
        f"({images_per_sec:.1f} images/sec, embedding_dim={embeddings.shape[1]})"
    )
    return elapsed, images_per_sec


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)

    paths, bboxes = collect_film_faces(args.film_images_dir, args.film_annotations, args.max_faces, args.seed)
    print(
        f"Benchmarking on {len(paths)} faces from {args.film_images_dir} "
        f"(crop-padding={args.crop_padding}), batch_size={args.batch_size}"
    )
    print("(same faces, bboxes, and batch size for every model below)\n")

    results: list[tuple[str, float, float]] = []

    if not args.skip_buffalo_baseline and not args.skip_baselines:
        try:
            from src.models.insightface_backbone import InsightFaceBuffaloL

            ctx_id = args.gpu if (args.gpu >= 0 and has_nvidia_gpu()) else -1
            model = InsightFaceBuffaloL(ctx_id=ctx_id)
            cpu_device = torch.device("cpu")
            print("buffalo_l (baseline):")
            elapsed, ims = benchmark_model(
                "buffalo_l (baseline)", model, cpu_device, paths, bboxes, args.crop_padding,
                args.batch_size, args.num_workers, args.warmup_batches,
            )
            results.append(("buffalo_l (baseline)", elapsed, ims))
        except Exception as e:
            print(f"[skip] buffalo_l (baseline): {e}")

    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")
    print(f"\nTorch device for the remaining models: {device}\n")

    if args.buffalo_checkpoint:
        try:
            onnx_path = str(Path(args.buffalo_onnx).expanduser())
            backbone = load_backbone_from_onnx(onnx_path)
            torch_model = BuffaloTorchModel(backbone)
            state = torch.load(args.buffalo_checkpoint, map_location="cpu")
            torch_model.load_state_dict(state["model"])
            model = BuffaloEvalAdapter(torch_model)
            tag = f"buffalo_l ({args.variant_label})"
            print(f"{tag}:")
            elapsed, ims = benchmark_model(
                tag, model, device, paths, bboxes, args.crop_padding,
                args.batch_size, args.num_workers, args.warmup_batches,
            )
            results.append((tag, elapsed, ims))
        except Exception as e:
            print(f"[skip] buffalo_l ({args.variant_label}): {e}")

    dino_archs = [
        ("vitb14", args.dino_vitb14_weights, args.dino_vitb14_checkpoint, args.skip_dino_vitb14),
        ("vitl14", args.dino_vitl14_weights, args.dino_vitl14_checkpoint, args.skip_dino_vitl14),
    ]
    for arch, weights_path, checkpoint_path, skip in dino_archs:
        if skip:
            continue
        if not args.skip_baselines:
            try:
                dino_model = load_dino_model(args.dinov2_dir, arch, weights_path, proj_dim=0)
                model = DinoEvalAdapter(dino_model, img_size=args.dino_img_size)
                print(f"dino_{arch} (baseline):")
                elapsed, ims = benchmark_model(
                    f"dino_{arch} (baseline)", model, device, paths, bboxes, args.crop_padding,
                    args.batch_size, args.num_workers, args.warmup_batches,
                )
                results.append((f"dino_{arch} (baseline)", elapsed, ims))
            except Exception as e:
                print(f"[skip] dino_{arch} (baseline): {e}")

        if checkpoint_path:
            try:
                dino_model = load_model_from_checkpoint(checkpoint_path, args.dinov2_dir)
                model = DinoEvalAdapter(dino_model, img_size=args.dino_img_size)
                tag = f"dino_{arch} ({args.variant_label})"
                print(f"{tag}:")
                elapsed, ims = benchmark_model(
                    tag, model, device, paths, bboxes, args.crop_padding,
                    args.batch_size, args.num_workers, args.warmup_batches,
                )
                results.append((tag, elapsed, ims))
            except Exception as e:
                print(f"[skip] dino_{arch} ({args.variant_label}): {e}")

    print("\n=== Inference throughput (same faces, same batch size, every model) ===")
    print(f"{'Model':<28} {'Faces':>8} {'Seconds':>10} {'im/s':>10}")
    for name, elapsed, ims in results:
        print(f"{name:<28} {len(paths):>8} {elapsed:>10.2f} {ims:>10.1f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(out_path, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("# Inference Throughput Benchmark\n\n")
                f.write("| Model | Images | Batch Size | Seconds | im/s | Timestamp |\n|---|---|---|---|---|---|\n")
            for name, elapsed, ims in results:
                f.write(f"| {name} | {len(paths)} | {args.batch_size} | {elapsed:.2f} | {ims:.1f} | {timestamp} |\n")
        print(f"\nAppended to {out_path}")


if __name__ == "__main__":
    main()
