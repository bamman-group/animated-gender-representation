"""Select the best checkpoint of a training run by AP@0.5 on the icartoon_val
split (labels/icf_val.csv, 5000 images held out from the training set) — the same criterion and code path for
every stack, so selection is uniform across models.

Evaluates every candidate checkpoint in the run directory (single-scale 640)
via the stack's evaluator, records per-checkpoint APs in <run_dir>/selection.md,
and writes <run_dir>/selection.json with the winner. Idempotent: checkpoints
already listed in selection.md are not re-evaluated (delete selection.md to
redo from scratch).

Examples (from the repo root):
    python -m src.eval.select_best --stack retinaface  --run-dir runs/retinaface/wf_icf
    python -m src.eval.select_best --stack ultralytics --run-dir runs/ultralytics/yolo26_icf
    python -m src.eval.select_best --stack yunet       --run-dir runs/yunet/icf
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

STACKS = {
    "retinaface": dict(evaluator="src.eval.evaluate_retinaface", flag="--trained-model",
                       patterns=["epoch_*.pth", "Resnet50_Final.pth"]),
    "ultralytics": dict(evaluator="src.eval.evaluate_ultralytics", flag="--weights",
                        patterns=["weights/best.pt", "weights/last.pt",
                                  "weights/epoch*.pt"]),
    "yunet": dict(evaluator="src.eval.evaluate_yunet", flag="--checkpoint",
                  patterns=["epoch_*.pth", "latest.pth", "best_loss.pth"]),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True, choices=sorted(STACKS))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-images", type=int, default=0,
                        help="cap selection images (0 = full icartoon_val)")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="extra args for the evaluator (e.g. --network mobile0.25)")
    return parser.parse_args()


def parse_selection_md(path):
    """selection.md rows -> {checkpoint_name: ap}"""
    done = {}
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0].startswith("sel:"):
            try:
                done[cells[0][4:]] = float(cells[3])
            except ValueError:
                pass
    return done


def main():
    global args
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"run dir not found: {run_dir}")
    spec = STACKS[args.stack]

    candidates = sorted({p for pat in spec["patterns"] for p in run_dir.glob(pat)})
    if not candidates:
        sys.exit(f"no checkpoints matching {spec['patterns']} in {run_dir}")

    selection_md = run_dir / "selection.md"
    done = parse_selection_md(selection_md)

    scores = {}
    for ckpt in candidates:
        name = str(ckpt.relative_to(run_dir))
        if name in done:
            scores[name] = done[name]
            print(f"cached  {name}: AP {done[name]:.2f}")
            continue
        cmd = [sys.executable, "-m", spec["evaluator"],
               spec["flag"], str(ckpt),
               "--eval-set", "icartoon_val",
               "--tag", f"sel:{name}",
               "--bootstrap", "0",
               "--results-file", str(selection_md)] + args.extra
        if args.num_images > 0:
            cmd += ["--num-images", str(args.num_images)]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"WARNING: evaluation failed for {name}; skipping")
            continue
        after = parse_selection_md(selection_md)
        if name in after:
            scores[name] = after[name]
            print(f"scored  {name}: AP {after[name]:.2f}")

    if not scores:
        sys.exit("no checkpoint could be evaluated")

    best_name = max(scores, key=scores.get)
    selection = {
        "best": str(run_dir / best_name),
        "best_ap_icartoon_val": scores[best_name],
        "criterion": "AP@0.5, icartoon_val (5k held out from train), single-scale 640",
        "candidates": scores,
    }
    with open(run_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    print(f"\nBEST: {selection['best']} (AP {scores[best_name]:.2f} on icartoon_val)")


if __name__ == "__main__":
    main()
