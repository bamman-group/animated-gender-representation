"""Evaluates the pretrained InsightFace buffalo_l model as a baseline (no
training in this repo) using the exact same Rank@1 protocols implemented for
every model in this repo: src/evaluate.py (official iCartoonFace rectest
protocol) and src/evaluate_film.py (data/film, per-movie protocol). Useful as
a reference point before/after fine-tuning src/train_buffalo.py or
src/train_dino.py.

Usage:
    python -m src.evaluate_baseline --dataset icartoonface
    python -m src.evaluate_baseline --dataset film
    python -m src.evaluate_baseline --dataset both
    python -m src.evaluate_baseline --dataset both --output results.md --timing-output timing.jsonl
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from src.datasets.icartoonface import DEFAULT_CROP_PADDING
from src.evaluate import rank1_identification_accuracy as icartoonface_rank1
from src.evaluate_film import rank1_identification_accuracy as film_rank1
from src.models.insightface_backbone import InsightFaceBuffaloL
from src.report import append_json_record, append_markdown_result, result_to_record

MODEL_NAME = "buffalo_l (baseline)"


def has_nvidia_gpu() -> bool:
    """Checks for a usable NVIDIA GPU via `nvidia-smi` rather than
    `torch.cuda.is_available()`. This script never runs PyTorch tensor ops on
    GPU (embeddings come from onnxruntime; `device` is always "cpu" below) -
    the only reason to know GPU presence at all is to pick onnxruntime's
    `ctx_id`. Avoiding any `torch.cuda.*` call keeps PyTorch from
    initializing its own CUDA context / loading its own cuBLAS in this
    process, which was winning a load-order race against onnxruntime's own
    (newer, required) cuBLAS and causing a hard crash - see the GPU setup
    notes in README.md."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=10, check=False
        )
        return result.returncode == 0 and b"GPU" in result.stdout
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["icartoonface", "film", "both"], default="both")
    parser.add_argument(
        "--test-dir",
        default="data/raw/personai_icartoonface_rectest/icartoonface_rectest",
    )
    parser.add_argument("--test-info-file", default="data/raw/icartoonface_rectest_info.txt")
    parser.add_argument("--film-images-dir", default="data/film/images")
    parser.add_argument("--film-annotations", default="data/film/annotations.json")
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=DEFAULT_CROP_PADDING,
        help="Fraction of the face bbox's width/height added on each side before cropping "
        "(0.0 = tight crop; 0.25 = 25%% padding on every side)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index for onnxruntime's CUDA provider (-1 = CPU). Lets parallel runs "
        "(e.g. the two crop modes) target different GPUs.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=f"Name for the results rows (default: '{MODEL_NAME}'). Include the crop mode when "
        "comparing --crop-padding settings, so the two modes' rows don't collide in results.md / "
        "src/collect_results.py (which keeps the latest row per model name).",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Append results as rows to this markdown file"
    )
    parser.add_argument(
        "--timing-output",
        type=str,
        default=None,
        help="Append accuracy/timing as JSON lines to this file (for later analysis)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test mode: evaluate a tiny slice (a few hundred gallery images / a handful of "
        "movies) so the whole pipeline runs in seconds. The Rank@1 it prints is NOT a valid number.",
    )
    args = parser.parse_args()

    # Smoke-test caps passed through to both protocols.
    rectest_caps = dict(max_distractors=200, max_probe_identities=50) if args.test else {}
    film_caps = dict(max_movies=10) if args.test else {}

    ctx_id = args.gpu if (args.gpu >= 0 and has_nvidia_gpu()) else -1
    model = InsightFaceBuffaloL(ctx_id=ctx_id)
    model.eval()
    device = torch.device("cpu")  # embeddings run via onnxruntime, not torch autograd
    model_name = args.tag or MODEL_NAME

    if args.dataset in ("icartoonface", "both"):
        print("=== iCartoonFace rectest (buffalo_l baseline) ===")
        result = icartoonface_rank1(
            model,
            device,
            test_dir=Path(args.test_dir),
            info_file=Path(args.test_info_file),
            padding=args.crop_padding,
            **rectest_caps,
        )
        if args.output:
            append_markdown_result(args.output, model_name, "iCartoonFace rectest", result)
        if args.timing_output:
            append_json_record(
                args.timing_output,
                result_to_record(model_name, "iCartoonFace rectest", result),
            )

    if args.dataset in ("film", "both"):
        print("\n=== data/film per-movie (buffalo_l baseline) ===")
        result = film_rank1(
            model,
            device,
            images_dir=Path(args.film_images_dir),
            annotations_file=Path(args.film_annotations),
            padding=args.crop_padding,
            **film_caps,
        )
        if args.output:
            append_markdown_result(args.output, model_name, "data/film (per-movie)", result)
        if args.timing_output:
            append_json_record(
                args.timing_output,
                result_to_record(model_name, "data/film (per-movie)", result),
            )


if __name__ == "__main__":
    main()
