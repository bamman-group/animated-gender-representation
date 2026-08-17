"""Convert the iCartoonFace bbox annotations into the unified label file used
for RetinaFace and YuNet training.

Ground-truth boxes come from the updated detection CSV
(personai_icartoonface_dettrain_anno_updatedv1.0.csv: filename,x1,y1,x2,y2).

No facial keypoints are used anywhere in this benchmark: every face is written
with placeholder landmarks and flag -1, which masks the landmark loss for it.
Flag -1 (not 0) matters - RetinaFace's MultiBoxLoss counts flag -1 faces as
box/classification positives (`conf_t != 0`) while excluding them from the
landmark term (`conf_t > 0`), so boxes still train normally. The 15-value row
layout is kept because the vendored RetinaFace dataset expects it.

Output format, one block per image:
    # <filename>
    x1 y1 x2 y2 l0x l0y l1x l1y l2x l2y l3x l3y l4x l4y flag
"""
import argparse
import os
from collections import defaultdict

# placeholder landmarks + flag; see the module docstring for why flag is -1
NO_LANDMARKS = [-1.0] * 10
NO_LANDMARK_FLAG = -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="training bbox CSV (filename,x1,y1,x2,y2)")
    ap.add_argument("--images-dir", required=True, help="training images directory (existence check)")
    ap.add_argument("--output", required=True, help="output label file")
    args = ap.parse_args()

    boxes_by_image = defaultdict(list)
    n_degenerate = 0
    with open(args.csv) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            fname = parts[0]
            x1, y1, x2, y2 = (float(v) for v in parts[1:5])
            if x2 <= x1 or y2 <= y1:
                n_degenerate += 1
                continue
            boxes_by_image[fname].append([x1, y1, x2, y2])

    n_images = n_faces = n_missing_img = 0
    with open(args.output, "w") as out:
        for fname in sorted(boxes_by_image):
            if not os.path.isfile(os.path.join(args.images_dir, fname)):
                n_missing_img += 1
                continue
            out.write(f"# {fname}\n")
            for box in boxes_by_image[fname]:
                n_faces += 1
                vals = box + NO_LANDMARKS + [NO_LANDMARK_FLAG]
                out.write(" ".join(f"{v:g}" for v in vals) + "\n")
            n_images += 1

    print(f"images written:        {n_images}")
    print(f"faces written:         {n_faces} (landmark loss masked for all)")
    print(f"degenerate boxes skipped: {n_degenerate}")
    print(f"images missing on disk:   {n_missing_img}")


if __name__ == "__main__":
    main()
