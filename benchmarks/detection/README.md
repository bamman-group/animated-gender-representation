# Animated Face Detection

Training and evaluation pipeline for detecting animated faces. Models are trained on [iCartoonFace](https://iqiyi.github.io/ICartoonFace/)/[WIDER Face](http://shuoyang1213.me/WIDERFACE/) data, and evaluated on iCartoonFace test data and annotated faces in 100 animated films.


| Model | Type | Pretraining |
|---|---|---|
| RetinaFace (ResNet-50) | CNN, anchor-based | Publicly released WIDER Face checkpoint |
| YOLO26-L | CNN, anchor-free, NMS-free | COCO pretrained |
| RT-DETR-L | Transformer (DETR family) | COCO pretrained |
| YuNet-N | Tiny CNN (~75k parameters) | Publicly released WIDER Face checkpoint |
| YOLOE-26L | Open-vocabulary, zero-shot | Open-vocabulary pretrained (zero-shot only) |

**Pretraining.** RetinaFace and YuNet initialize from publicly released checkpoints trained on WIDER Face. YOLO26-L and RT-DETR-L initialize from standard COCO-pretrained weights. For supervised experiments, each model is trained on either **WF** (WIDER Face), **ICF** (iCartoonFace), or **WF+ICF** (the union of both training sets, concatenated without
resampling or reweighting, so the two contribute in proportion to their
natural sizes). RetinaFace and YuNet use the released WIDER Face checkpoints directly for the WF condition and are fine-tuned from those checkpoints for the ICF and WF+ICF conditions. YOLOE-26L is evaluated only in its zero-shot configuration using a prompt (e.g., `"cartoon face"`) and is not fine-tuned.

Evaluation: the iCartoonFace training release (50,000 images) is split
into 45,000 training images and **icartoon_val** (5,000 images drawn by a
seeded random shuffle, used for in-training monitoring and checkpoint
selection); **icartoon_test** = the full detval release (10,000 images /
18,647 faces), held out entirely for final numbers; **film** = 2,458 frames /
3,700 faces from 100 animated films. For every training run, the checkpoint
with the best AP@0.5 on icartoon_val is selected (`src/eval/select_best.py`), and the YOLOE text prompt is selected the
same way from a candidate list (`src/eval/select_prompt.py`).  All models are then evaluated
**plain** (single-scale 640) and with test-time augmentation (**TTA**) (multi-scale 640/1100/1600 +
horizontal flip with merged NMS) on icartoon_test and film.

## Setup

```bash
git clone <this repo>
cd benchmarks/detection
conda env create -f environment.yml
conda activate animated-face

# download iCartoonFace (~4 GB) and WIDER Face (~1.5 GB) into place:
bash scripts/download_icartoonface.sh
bash scripts/download_widerface.sh
# film data (if available) should be placed under data/
# then:
bash scripts/setup.sh
```

`setup.sh` downloads the RetinaFace WIDER Face checkpoint (YuNet's released
checkpoints are version-controlled in `third_party/libfacedetection.train/weights/`),
builds the unified label files under `labels/`, and materializes the
per-setting training datasets under `datasets/`.

## Train

```bash
bash scripts/train_all.sh                 
```

`train_all.sh` just runs the four
per-model scripts in order, and each is standalone and produces exactly the
same runs on its own; to parallelize one per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_retinaface.sh &   # icf, wf_icf
CUDA_VISIBLE_DEVICES=1 bash scripts/train_yolo26.sh &       # wf, icf, wf_icf
CUDA_VISIBLE_DEVICES=2 bash scripts/train_rtdetr.sh &       # wf, icf, wf_icf
CUDA_VISIBLE_DEVICES=3 bash scripts/train_yunet.sh &        # icf, wf_icf
wait
```


## Evaluate

```bash
bash scripts/evaluate_all.sh    # all models x {plain,TTA} x {icartoon_test,film}
```

Rows append to
`results/results.md` — including an **im/s** inference-throughput column
(the detection loop including image read/decode; model load excluded), and
95% confidence intervals.

Latex tables can be generated from results/results.md:

```bash
python -m src.eval.collect_results --table film           # -> results/table_film.tex
python -m src.eval.collect_results --table icartoon_test  # -> results/table_icartoon_test.tex
```



## Licenses

The code in this repository is released under the MIT license (the components it builds on keep their own licenses).

