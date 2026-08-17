"""Convert the film dataset's annotations.json (frames from 100 animated
films) to the detval-style CSV consumed by the evaluators.

Example (from the repo root):
    python -m src.prepare.prepare_film \
        --annotations data/film/annotations.json \
        --images-root data/film/images \
        --output labels/film_val.csv

Input: a list of per-movie dicts: {"movie_id": ..., "<frame>.jpg": [ {x, y, w,
h, cluster_id, cluster_label}, ...], "saved_by": ..., "saved_at": ...}.
Output rows: <movie_id>/<frame>.jpg,x1,y1,x2,y2,face
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True, help="movie_faces annotations.json")
    ap.add_argument("--images-root", required=True, help="ctf_100 directory (existence check)")
    ap.add_argument("--output", required=True, help="output CSV")
    args = ap.parse_args()

    with open(args.annotations) as f:
        entries = json.load(f)

    n_images = n_faces = n_degenerate = n_missing = 0
    with open(args.output, "w") as out:
        for entry in entries:
            movie_id = entry["movie_id"]
            for key, faces in entry.items():
                if key == "movie_id" or not isinstance(faces, list):
                    continue  # skip saved_by / saved_at metadata
                rel = f"{movie_id}/{key}"
                if not os.path.isfile(os.path.join(args.images_root, rel)):
                    n_missing += 1
                    continue
                n_images += 1
                for face in faces:
                    x1, y1 = float(face["x"]), float(face["y"])
                    x2, y2 = x1 + float(face["w"]), y1 + float(face["h"])
                    if x2 <= x1 or y2 <= y1:
                        n_degenerate += 1
                        continue
                    out.write(f"{rel},{x1:g},{y1:g},{x2:g},{y2:g},face\n")
                    n_faces += 1

    print(f"images written:  {n_images}")
    print(f"faces written:   {n_faces}")
    print(f"degenerate boxes skipped: {n_degenerate}")
    print(f"images missing on disk:   {n_missing}")


if __name__ == "__main__":
    main()
