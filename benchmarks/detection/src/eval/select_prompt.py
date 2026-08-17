"""Select the best YOLOE text prompt by AP@0.5 on icartoon_val — prompts are
treated exactly like checkpoint candidates: scored on the dev split, the
winner is used for the reported icartoon_test / film evaluations.

Per-prompt scores are cached in <out_dir>/selection.md (idempotent; delete to
redo) and the winner is written to <out_dir>/selection.json.

Example (from the repo root):
    python -m src.eval.select_prompt
    python -m src.eval.select_prompt --prompts "face" "cartoon face" "animated face"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROMPTS = [
    "face",
    "cartoon face",
    "animated face",
    "cartoon face, animated face, face",
]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yoloe-26l-seg.pt")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--num-images", type=int, default=0,
                        help="cap selection images (0 = full icartoon_val)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "runs/yoloe"))
    return parser.parse_args()


def parse_selection_md(path):
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_md = out_dir / "selection.md"
    done = parse_selection_md(selection_md)

    scores = {}
    for prompt in args.prompts:
        if prompt in done:
            scores[prompt] = done[prompt]
            print(f"cached  {prompt!r}: AP {done[prompt]:.2f}")
            continue
        cmd = [sys.executable, "-m", "src.eval.evaluate_ultralytics",
               "--weights", args.weights,
               "--text-prompt", prompt,
               "--eval-set", "icartoon_val",
               "--bootstrap", "0",
               "--tag", f"sel:{prompt}",
               "--results-file", str(selection_md)]
        if args.num_images > 0:
            cmd += ["--num-images", str(args.num_images)]
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"WARNING: evaluation failed for prompt {prompt!r}; skipping")
            continue
        after = parse_selection_md(selection_md)
        if prompt in after:
            scores[prompt] = after[prompt]
            print(f"scored  {prompt!r}: AP {after[prompt]:.2f}")

    if not scores:
        sys.exit("no prompt could be evaluated")

    best = max(scores, key=scores.get)
    selection = {
        "best_prompt": best,
        "best_ap_icartoon_val": scores[best],
        "weights": args.weights,
        "criterion": "AP@0.5, icartoon_val, single-scale 640",
        "candidates": scores,
    }
    with open(out_dir / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)
    print(f"\nBEST PROMPT: {best!r} (AP {scores[best]:.2f} on icartoon_val)")


if __name__ == "__main__":
    main()
