# Animated Face Recognition

Training and evaluation pipeline for recognizing animated faces.
Two pretrained backbones are fine-tuned on the
[iCartoonFace](https://iqiyi.github.io/ICartoonFace/) dataset and evaluated on two datasets (iCartoonFace test data and annotated faces in 100 animated films) on a shared Rank@1 identification metric.

| Backbone | Type | Pretraining |
|---|---|---|
| InsightFace `buffalo_l` (`w600k_r50`) | CNN (ResNet-50) | ArcFace-trained for face recognition on WebFace-600K (real faces) |
| DINOv2 (ViT-S/B/L/G) | Vision Transformer (ViT), self-supervised | Self-supervised pretraining on the curated LVD-142M image dataset |

**Training.** Training uses two sources of information: with the iCartoonFace rectrain identity labels
(`--data-source identity`, the default), both backbones fine-tune with an
ArcFace margin-softmax classifier over the training identities
(`--loss arcface`, default).
A separate paired-**face-track** signal (`--data-source tracks`) is also
supported for video data that has no global identity labels, but where the temporal consistency of face tracks provide some form of information: there the
positive pair comes from within one track and the negative is a *co-occurring*
face from a different track in the same frame (using 
triplet loss, since two tracks in different frames may be the same, unlabeled
person — they can't be assumed to be negatives).

**Padding**.  All models are trained with a close crop of a detected face (i.e., 0% padding) and with 25% padding on each dimension (to explore how useful additional character information beyond the face is at recognition).

**Evaluation.** Models are evaluated on two held-out **test** sets, both scored by the same
`rank1_identification_accuracy()` with a 95% bootstrap CI: **iCartoonFace
rectest**, containing the official 2,500-distractor / 2,000-probe-identity split
(`src/evaluate.py`); **`data/film`**, containing 100 animated-film clips. Neither test set is
ever read during training: mid-training checkpoint selection uses a **dev
split carved out of the iCartoonFace training data itself** (`--dev-fraction`).

## Setup

```bash
conda env create -f environment.yml
conda activate animated-face-recognition

# download + prepare iCartoonFace and DINOv2:
bash scripts/setup.sh
```

`setup.sh` runs `scripts/download_data.sh` (iCartoonFace Google Drive folder +
evaluation code), `scripts/prepare_data.py` (unpacks archives, installs the
complete `icartoonface_rectest_info.txt`, verifies the layout), and
`scripts/download_dino.sh <arch>` (clones `dinov2` into `third_party/dinov2`
and downloads pretrained weights into `data/dinov2_weights/`) for both
`vitb14` and `vitl14`, the two DINOv2 architectures the batch scripts train by
default. 

`buffalo_l` itself downloads automatically on first use into
`~/.insightface/models/` (not into this repo).



## Train

### iCartoonFace (identity) training

Each run of `train_all.sh` covers all three models — `buffalo_l` once, plus one
DINOv2 fine-tune per architecture in `ARCHS` (default `"vitb14 vitl14"`).

```bash
# both crop modes for all three models, trained and evaluated side by side:
CROP_PCT=0  bash scripts/train_all.sh  && CROP_PCT=25 bash scripts/train_all.sh
CROP_PCT=0  bash scripts/evaluate_all.sh && CROP_PCT=25 bash scripts/evaluate_all.sh

# restrict to one DINOv2 arch: pass the SAME ARCHS to evaluate_all.sh
ARCHS=vitl14 EPOCHS=40 GPU=1 bash scripts/train_all.sh
ARCHS=vitl14 GPU=1 bash scripts/evaluate_all.sh
```

Checkpoints land in `outputs/<model>_identity_crop<pct>/backbone_best.pth`
(best dev MRR) plus periodic `backbone_epoch*.pth`, and each run writes a
`timing.json`. Individual commands:

```bash
python -m src.train_buffalo \
    --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx \
    --output-dir outputs/buffalo_identity_crop0 \
    --epochs 30 --batch-size 64 --gpu 0 --fp16 \
    --output results.md --timing-output timing.jsonl

python -m src.train_dino \
    --dinov2-dir third_party/dinov2 \
    --weights data/dinov2_weights/dinov2_vitb14_pretrain.pth --arch vitb14 --proj-dim 0 \
    --output-dir outputs/dino_vitb14_identity_crop0 \
    --epochs 30 --batch-size 64 --gpu 0 --fp16 \
    --output results.md --timing-output timing.jsonl
```


## Inference timing

```bash
bash scripts/benchmark_all.sh                                    # -> results/inference_timing.md
GPU=0 OUTPUT=results/inference_timing.md bash scripts/benchmark_all.sh
```


## Evaluate

```bash
bash scripts/evaluate_all.sh          # baselines (+ fine-tuned, if present) x both test sets
CROP_PCT=25 bash scripts/evaluate_all.sh   # the 25%-context crop mode (see Train)
```

Rows append to `results.md` with **Rank@1, its 95% bootstrap CI, and
embedding throughput (im/s)**.


## Tables

```bash
python -m src.collect_results --table variants      # -> results/table_variants.tex
```

The variants table compares the training variants — `baseline`,
`fine-tuned` (identity), `identity+tracks/train`, `identity+tracks` — across
both crop modes, taking its im/s column from the inference benchmark below
(`--inference-timing`, default `results/inference_timing.md`). Those four
strings are the variant field of each row's `--tag`, so a run only appears in
the table if its tag follows the `<model> (<variant>, crop N%)` convention.


## Licenses

The code in this repository is released under the MIT license (the components it builds on keep their own licenses).

