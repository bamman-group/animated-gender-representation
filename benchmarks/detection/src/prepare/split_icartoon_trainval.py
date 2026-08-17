"""Split the iCartoonFace training labels into a training part (used to fit
models) and a validation part (used for in-training monitoring and
best-checkpoint selection). The full detval release (10k images) is then a
pure held-out test set.

The split is a random shuffle with a fixed seed (default 42), so it is
reproducible but not correlated with filename order.

Outputs:
    --out-train  RetinaFace-format label file (boxes + landmarks) for training
    --out-val    detval-style CSV (filename,x1,y1,x2,y2,face) for evaluation;
                 the images stay in data/icartoonface/dettrain

Example (from the repo root):
    python -m src.prepare.split_icartoon_trainval \
        --label-file labels/icf_train.txt --val-n 5000 \
        --out-train labels/icf_train45.txt --out-val labels/icf_val.csv
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-file", required=True, help="full training label file")
    ap.add_argument("--val-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    args = ap.parse_args()

    blocks = {}
    current = None
    with open(args.label_file) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                current = []
                blocks[line[1:].strip()] = current
            else:
                current.append(line)

    names = sorted(blocks)
    if args.val_n >= len(names):
        raise SystemExit(f"val_n {args.val_n} >= dataset size {len(names)}")
    import random
    random.Random(args.seed).shuffle(names)
    val_names = sorted(names[:args.val_n])
    train_names = sorted(names[args.val_n:])

    n_train_faces = 0
    with open(args.out_train, "w") as f:
        for name in train_names:
            f.write(f"# {name}\n")
            for row in blocks[name]:
                f.write(row + "\n")
            n_train_faces += len(blocks[name])

    n_val_faces = 0
    with open(args.out_val, "w") as f:
        for name in val_names:
            for row in blocks[name]:
                x1, y1, x2, y2 = row.split()[:4]
                f.write(f"{name},{x1},{y1},{x2},{y2},face\n")
            n_val_faces += len(blocks[name])

    print(f"train: {len(train_names)} images, {n_train_faces} faces -> {args.out_train}")
    print(f"val:   {len(val_names)} images, {n_val_faces} faces -> {args.out_val}")


if __name__ == "__main__":
    main()
