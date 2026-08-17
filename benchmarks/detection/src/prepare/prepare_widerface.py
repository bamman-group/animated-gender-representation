"""Convert WIDER Face annotations to the unified label format (see
prepare_icartoonface.py).

Input is the official WIDER Face annotation file
(wider_face_split/wider_face_train_bbx_gt.txt):

    <relative image path>
    <num faces>
    x y w h blur expression illumination invalid occlusion pose

No facial keypoints are used anywhere in this benchmark, so boxes are the only
thing taken from it: every face is written with placeholder landmarks and flag
-1, which masks the landmark loss for it while keeping the box as a training
positive.

Output format, one block per image:
    # <filename>
    x1 y1 x2 y2 l0x l0y l1x l1y l2x l2y l3x l3y l4x l4y flag
"""
import argparse
import os

# placeholder landmarks + flag; see prepare_icartoonface.py for why flag is -1
NO_LANDMARKS = [-1.0] * 10
NO_LANDMARK_FLAG = -1


def parse_bbx_gt(path):
    """Yield (rel_image_path, [[x, y, w, h], ...])."""
    with open(path) as f:
        lines = iter(f.read().splitlines())
        for img in lines:
            img = img.strip()
            if not img:
                continue
            n = int(next(lines))
            rows = [next(lines) for _ in range(max(n, 1))]  # n=0 has a dummy row
            faces = []
            if n > 0:
                for row in rows:
                    faces.append([float(x) for x in row.split()[:4]])
            yield img, faces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbx-gt", required=True,
                    help="wider_face_split/wider_face_train_bbx_gt.txt")
    ap.add_argument("--images-dir", required=True, help="WIDER_train/images")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-size", type=float, default=8,
                    help="skip boxes with w or h below this (annotation noise)")
    args = ap.parse_args()

    n_images = n_faces = n_skipped = n_empty = n_missing = 0
    # read the annotations before opening the output, so a missing/unreadable
    # input does not leave an empty label file behind that setup.sh would then
    # treat as already-done
    annotations = list(parse_bbx_gt(args.bbx_gt))
    with open(args.output, "w") as out:
        for img, faces in annotations:
            img = img.lstrip("/")
            if not os.path.isfile(os.path.join(args.images_dir, img)):
                n_missing += 1
                continue
            rows = []
            for x, y, w, h in faces:
                if w < args.min_size or h < args.min_size:
                    n_skipped += 1
                    continue
                rows.append([x, y, x + w, y + h] + NO_LANDMARKS + [NO_LANDMARK_FLAG])
            if not rows:
                n_empty += 1
                continue
            out.write(f"# {img}\n")
            for r in rows:
                out.write(" ".join(f"{v:g}" for v in r) + "\n")
            n_images += 1
            n_faces += len(rows)

    print(f"images written:  {n_images}")
    print(f"faces written:   {n_faces} (landmark loss masked for all)")
    print(f"boxes skipped (< {args.min_size}px): {n_skipped}")
    print(f"images with no usable faces: {n_empty}")
    print(f"images missing on disk:      {n_missing}")


if __name__ == "__main__":
    main()
