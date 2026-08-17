"""Shared library for the data/film in-the-wild evaluation set, using a
*per-movie* Rank@1 identification protocol (distinct from the official
iCartoonFace rectest protocol in src/evaluate.py - see below for how).

data/film/annotations.json is a list of per-movie entries:
    {
      "movie_id": "<movie_id>",
      "saved_by": "...", "saved_at": "...",   # metadata, not a frame
      "<frame_filename>.jpg": [
        {"x": .., "y": .., "w": .., "h": .., "cluster_id": .., "cluster_label": ..},
        ...
      ],
      ...
    }
Each face's bbox is given as (x, y, w, h) (top-left + width/height), unlike
icartoonface_rec*'s (x1, y1, x2, y2); converted here before reusing
src/evaluate.py's crop/embed helpers. cluster_label (a human-readable name,
sometimes a generic "Cluster <N>" placeholder) is not used for anything -
only cluster_id groups faces into the same identity.

Protocol, applied independently within each movie (never comparing faces
across movies):
  - Faces are grouped by cluster_id. A cluster_id with >= 2 faces is a probe
    candidate; cluster_ids with only 1 face, or a missing cluster_id, can
    never be a probe (nothing to leave one out against) but still take part
    as distractors below.
  - For a given probe cluster_id, every *other* face in the movie - i.e. every
    face whose cluster_id differs (whether that's another probe identity, a
    singleton, or a missing cluster_id) - is a distractor for that probe.
  - For each probe cluster_id's M (>=2) faces: each face in turn is added to
    that probe's distractor gallery as the one true match, and the other M-1
    faces of the same cluster_id are probed against it (same leave-one-out
    construction as src/evaluate.py's count_rank1_correct).
  - Rank@1 = fraction of probe trials, summed over all movies (and over all
    probe cluster_ids within each movie), where the true match is the closest
    gallery entry.

Every face in a movie is embedded exactly once and reused across all of that
movie's probe cluster_ids (the distractor gallery for probe A is just "every
embedding except A's own", not a separately-fetched set).

Not a standalone script - imported by src/evaluate_baseline.py,
src/train_dino.py, and src/train_buffalo.py for `rank1_identification_accuracy()`.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from src.evaluate import BBox, EmbeddingTimer, bootstrap_rank1_ci, count_rank1_correct, embed_paths
from src.datasets.icartoonface import DEFAULT_CROP_PADDING
from src.report import EvalResult


def to_xyxy(x: int, y: int, w: int, h: int) -> BBox:
    return x, y, x + w, y + h


def collect_movie_faces(movie: dict) -> list[tuple[str, str | None, BBox]]:
    """Returns [(frame_filename, cluster_id, bbox), ...] for every face in the movie."""
    faces = []
    for key, value in movie.items():
        if not isinstance(value, list):
            continue  # movie_id / saved_by / saved_at metadata, not a frame
        for face in value:
            bbox = to_xyxy(face["x"], face["y"], face["w"], face["h"])
            faces.append((key, face.get("cluster_id"), bbox))
    return faces


@torch.no_grad()
def rank1_identification_accuracy(
    model,
    device,
    images_dir: Path,
    annotations_file: Path,
    padding: float = DEFAULT_CROP_PADDING,
    n_boot: int = 1000,
    seed: int = 0,
    max_movies: int | None = None,
) -> EvalResult:
    with open(annotations_file, "r", encoding="utf-8") as f:
        movies = json.load(f)
    if max_movies:  # smoke-test cap (see --test); not a valid Rank@1 number
        movies = movies[:max_movies]

    total_correct, total_trials = 0, 0
    num_movies_used = 0
    cluster_counts: list[tuple[int, int]] = []  # one (correct, total) per movie
    timer = EmbeddingTimer()

    for movie in tqdm(movies, desc="Movies", unit="movie"):
        movie_id = movie["movie_id"]
        movie_dir = images_dir / movie_id
        faces = collect_movie_faces(movie)
        if not faces:
            continue

        indices_by_cluster: dict[str, list[int]] = defaultdict(list)
        for i, (_, cluster_id, _) in enumerate(faces):
            if cluster_id is not None:
                indices_by_cluster[cluster_id].append(i)

        probe_cluster_ids = [cid for cid, idxs in indices_by_cluster.items() if len(idxs) >= 2]
        if not probe_cluster_ids:
            continue
        num_movies_used += 1

        paths = [movie_dir / frame_filename for frame_filename, _, _ in faces]
        bboxes = [bbox for _, _, bbox in faces]
        with timer.track(len(paths)):
            embeddings = embed_paths(
                model, paths, bboxes, device, padding=padding, desc=f"  {movie_id}"
            )

        movie_correct, movie_total = 0, 0
        for cluster_id in probe_cluster_ids:
            probe_idxs = torch.tensor(indices_by_cluster[cluster_id])
            mask = torch.ones(len(faces), dtype=torch.bool)
            mask[probe_idxs] = False

            distractor_embeddings = embeddings[mask]
            identity_embeddings = embeddings[probe_idxs]
            correct, trials = count_rank1_correct(distractor_embeddings, identity_embeddings)
            movie_correct += correct
            movie_total += trials

        tqdm.write(
            f"{movie_id}: {len(probe_cluster_ids)} probe identities, "
            f"{len(faces)} total faces, {movie_correct}/{movie_total} correct"
        )
        total_correct += movie_correct
        total_trials += movie_total
        cluster_counts.append((movie_correct, movie_total))

    print(
        f"\nUsed {num_movies_used}/{len(movies)} movies "
        "(rest had no cluster_id with >= 2 faces)."
    )
    print(timer.report())
    if total_trials == 0:
        raise ValueError(
            f"No probe trials: no movie in {annotations_file} has a cluster_id "
            "with >= 2 faces (empty or malformed annotations?)"
        )
    accuracy = total_correct / total_trials
    ci = bootstrap_rank1_ci(cluster_counts, n_boot=n_boot, seed=seed)
    ci_str = f"  95% CI [{ci[0]:.4f}, {ci[1]:.4f}] (bootstrap over movies)" if ci else ""
    print(f"Rank@1 identification accuracy: {accuracy:.4f} ({total_correct}/{total_trials} probe trials){ci_str}")
    return EvalResult(
        accuracy, total_correct, total_trials, timer.elapsed, timer.count,
        ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None,
    )
