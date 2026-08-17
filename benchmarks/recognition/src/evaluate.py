"""Shared library for the official iCartoonFace rectest identification split,
matching the protocol described in Zheng et al., "Cartoon Face Recognition: A
Benchmark Dataset" (ACM MM'20), Section 3.3 "Face recognition":

  - icartoonface_rectest_info.txt has one row per test image:
        <filename> <x1> <y1> <x2> <y2> <label>
    Rows with label == -1 are the 2,500 fixed distractor images (identity
    doesn't matter - they should never be the correct match for any probe).
    Rows with label != -1 belong to one of the 2,000 probe identities
    (5-17 images each); the label groups images of the same identity.
  - For each probe identity's M images: each image in turn is added to the
    2,500-image distractor gallery (gallery size = 2,501, with exactly one
    true match), and each of the *other* M-1 images is used as a probe
    against that gallery. Rank@1 = fraction of probe trials where the true
    match is retrieved as the single nearest neighbor in the gallery.

This reads the real, officially released split directly rather than
constructing an equivalent one from scratch, so results are comparable to
the paper's Table 2/3 numbers. Each image is also cropped to its face bbox
(optionally padded - see `padding`/`DEFAULT_CROP_PADDING` in
src/datasets/icartoonface.py) before embedding.

Not a standalone script - imported by src/evaluate_baseline.py (InsightFace
buffalo_l baseline) for `rank1_identification_accuracy()` and the shared
`embed_paths()`/`EmbeddingTimer` machinery (also reused by
src/evaluate_film.py). Any `model` passed in just needs a `forward(batch)`
returning embeddings and an `embedding_dim` attribute - see
src/models/insightface_backbone.py, src/models/dino_backbone.py, and
src/models/buffalo_backbone.py for examples.
"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datasets.icartoonface import DEFAULT_CROP_PADDING, EVAL_TRANSFORM, crop_face
from src.report import EvalResult

EXPECTED_NUM_DISTRACTORS = 2500
EXPECTED_NUM_PROBE_IDENTITIES = 2000

BBox = tuple[int, int, int, int]


class EmbeddingTimer:
    """Accumulates wall-clock time spent inside embed_paths() across
    multiple calls, to report inference throughput (images/sec) separately
    from the rank-computation bookkeeping around it."""

    def __init__(self):
        self.elapsed = 0.0
        self.count = 0

    def track(self, num_images: int) -> "_EmbeddingTimerContext":
        return _EmbeddingTimerContext(self, num_images)

    @property
    def images_per_sec(self) -> float:
        return self.count / self.elapsed if self.elapsed > 0 else float("nan")

    def report(self) -> str:
        return (
            f"Inference: generated {self.count} embeddings in {self.elapsed:.1f}s "
            f"({self.images_per_sec:.1f} images/sec)"
        )


class _EmbeddingTimerContext:
    def __init__(self, timer: EmbeddingTimer, num_images: int):
        self.timer = timer
        self.num_images = num_images

    def __enter__(self) -> "_EmbeddingTimerContext":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.timer.elapsed += time.perf_counter() - self.start
        self.timer.count += self.num_images


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], bboxes: list[BBox | None], padding: float = DEFAULT_CROP_PADDING):
        self.paths = paths
        self.bboxes = bboxes
        self.padding = padding

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        bbox = self.bboxes[idx]
        if bbox is not None:
            image = crop_face(image, bbox, self.padding)
        return EVAL_TRANSFORM(image)


def parse_rectest_info(
    info_file: Path,
) -> tuple[list[str], dict[str, list[str]], dict[str, BBox]]:
    """Returns (distractor_filenames, {probe_identity_label: [filenames]}, {filename: bbox})."""
    distractors = []
    probes = defaultdict(list)
    bboxes: dict[str, BBox] = {}
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            filename, x1, y1, x2, y2, label = parts
            bboxes[filename] = (int(x1), int(y1), int(x2), int(y2))
            if label == "-1":
                distractors.append(filename)
            else:
                probes[label].append(filename)
    return distractors, probes, bboxes


@torch.no_grad()
def embed_paths(
    model,
    paths: list[Path],
    bboxes: list[BBox | None],
    device,
    batch_size: int = 128,
    num_workers: int = 8,
    padding: float = DEFAULT_CROP_PADDING,
    desc: str | None = None,
) -> torch.Tensor:
    if not paths:
        # e.g. a movie with zero distractor faces - keep the (0, embedding_dim)
        # shape so torch.cat([this, ...]) downstream still works.
        return torch.empty(0, model.embedding_dim)

    loader = DataLoader(
        ImagePathDataset(paths, bboxes, padding=padding),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    embeddings = []
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        batch = batch.to(device)
        embeddings.append(F.normalize(model(batch)).cpu())
    return torch.cat(embeddings)


@torch.no_grad()
def count_rank1_correct(
    distractor_embeddings: torch.Tensor, identity_embeddings: torch.Tensor
) -> tuple[int, int]:
    """Leave-one-out trials for a single identity's embeddings against a fixed
    distractor gallery: each of the M embeddings takes a turn as the one true
    gallery match, and the other M-1 are probed against
    concat(distractor_embeddings, that_one_embedding). Returns (correct, total)
    summed over all M trials.

    Exact similarity ties are scored worst-case: argmax returns the first
    maximal index, and the true match sits at the END of the gallery, so a
    probe tied between the true match and a distractor counts as wrong (the
    opposite convention from count_rank1_and_mrr's best-case tie rank, which
    is only used for mid-training dev selection). Exact float ties are rare
    enough that this does not measurably bias reported numbers."""
    correct, total = 0, 0
    m = identity_embeddings.shape[0]
    for gallery_idx in range(m):
        gallery = torch.cat(
            [distractor_embeddings, identity_embeddings[gallery_idx : gallery_idx + 1]]
        )
        probe_mask = torch.ones(m, dtype=torch.bool)
        probe_mask[gallery_idx] = False
        probes = identity_embeddings[probe_mask]

        sims = probes @ gallery.T
        top1 = sims.argmax(dim=1)
        correct += (top1 == gallery.shape[0] - 1).sum().item()
        total += probes.shape[0]
    return correct, total


@torch.no_grad()
def count_rank1_and_mrr(
    distractor_embeddings: torch.Tensor, identity_embeddings: torch.Tensor
) -> tuple[int, int, float]:
    """Like count_rank1_correct, but also returns the summed reciprocal rank of
    the true match across all leave-one-out trials, so callers can report
    MRR = mrr_sum / total alongside Rank@1 = correct / total. Used by the
    mid-training dev evals (a smoother checkpoint-selection signal than binary
    Rank@1); the fixed test protocols keep using count_rank1_correct. Ties in
    similarity are scored best-case (rank = 1 + #strictly-greater)."""
    correct, total, mrr_sum = 0, 0, 0.0
    m = identity_embeddings.shape[0]
    for gallery_idx in range(m):
        gallery = torch.cat(
            [distractor_embeddings, identity_embeddings[gallery_idx : gallery_idx + 1]]
        )
        probe_mask = torch.ones(m, dtype=torch.bool)
        probe_mask[gallery_idx] = False
        probes = identity_embeddings[probe_mask]

        sims = probes @ gallery.T  # (num_probes, gallery_size)
        true_col = gallery.shape[0] - 1
        correct += (sims.argmax(dim=1) == true_col).sum().item()
        ranks = 1 + (sims > sims[:, true_col : true_col + 1]).sum(dim=1)
        mrr_sum += (1.0 / ranks.float()).sum().item()
        total += probes.shape[0]
    return correct, total, mrr_sum


def bootstrap_rank1_ci(
    cluster_counts: list[tuple[int, int]], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    """95% bootstrap CI for Rank@1 accuracy, resampling *clusters* (probe
    identities here, movies in src/evaluate_film.py) with replacement rather
    than individual trials - trials within a cluster are correlated (they
    share a gallery / an identity), so a naive per-trial CI would be
    over-confident. `cluster_counts` is one (correct, total) pair per cluster.
    Returns (lo, hi) accuracy fractions, or None if there are <2 usable
    clusters or n_boot <= 0 (checkpoint selection skips the CI)."""
    clusters = [(c, t) for c, t in cluster_counts if t > 0]
    if n_boot <= 0 or len(clusters) < 2:
        return None
    correct = np.array([c for c, _ in clusters], dtype=np.float64)
    total = np.array([t for _, t in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(clusters)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t = total[idx].sum()
        if t > 0:
            accs.append(correct[idx].sum() / t)
    if not accs:
        return None
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


@torch.no_grad()
def rank1_identification_accuracy(
    model,
    device,
    test_dir: Path,
    info_file: Path,
    padding: float = DEFAULT_CROP_PADDING,
    n_boot: int = 1000,
    seed: int = 0,
    max_distractors: int | None = None,
    max_probe_identities: int | None = None,
) -> EvalResult:
    distractor_filenames, probe_identities, bboxes = parse_rectest_info(info_file)

    # Smoke-test caps (see --test): shrink the gallery / probe set so the whole
    # pipeline runs in seconds. Not a valid Rank@1 number - for verification only.
    if max_distractors:
        distractor_filenames = distractor_filenames[:max_distractors]
    if max_probe_identities:
        probe_identities = dict(list(probe_identities.items())[:max_probe_identities])

    if max_distractors is None and len(distractor_filenames) != EXPECTED_NUM_DISTRACTORS:
        print(
            f"Warning: expected {EXPECTED_NUM_DISTRACTORS} distractor images, "
            f"found {len(distractor_filenames)} in {info_file}."
        )
    if max_probe_identities is None and len(probe_identities) != EXPECTED_NUM_PROBE_IDENTITIES:
        print(
            f"Warning: expected {EXPECTED_NUM_PROBE_IDENTITIES} probe identities, "
            f"found {len(probe_identities)} in {info_file}."
        )
    print(
        f"Protocol: {len(distractor_filenames)} distractor images, "
        f"{len(probe_identities)} probe identities, "
        f"{sum(len(v) for v in probe_identities.values())} probe images total."
    )

    timer = EmbeddingTimer()

    distractor_paths = [test_dir / name for name in distractor_filenames]
    distractor_bboxes = [bboxes.get(name) for name in distractor_filenames]
    with timer.track(len(distractor_paths)):
        distractor_embeddings = embed_paths(
            model,
            distractor_paths,
            distractor_bboxes,
            device,
            padding=padding,
            desc="Embedding distractors",
        )

    correct, total = 0, 0
    cluster_counts: list[tuple[int, int]] = []  # one (correct, total) per probe identity
    for label, filenames in tqdm(probe_identities.items(), desc="Probe identities", unit="identity"):
        paths = [test_dir / name for name in filenames]
        probe_bboxes = [bboxes.get(name) for name in filenames]
        with timer.track(len(paths)):
            embeddings = embed_paths(model, paths, probe_bboxes, device, padding=padding)
        identity_correct, identity_total = count_rank1_correct(distractor_embeddings, embeddings)
        correct += identity_correct
        total += identity_total
        cluster_counts.append((identity_correct, identity_total))

    print(timer.report())
    accuracy = correct / total
    ci = bootstrap_rank1_ci(cluster_counts, n_boot=n_boot, seed=seed)
    ci_str = f"  95% CI [{ci[0]:.4f}, {ci[1]:.4f}] (bootstrap over probe identities)" if ci else ""
    print(f"Rank@1 identification accuracy: {accuracy:.4f} ({correct}/{total} probe trials){ci_str}")
    return EvalResult(
        accuracy, correct, total, timer.elapsed, timer.count,
        ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None,
    )
