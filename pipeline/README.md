# Pipeline

This directory includes code and models to run the pipeline on one sample video. `setup.sh` downloads all models (the best detection model from the `benchmark/detection` process, and the best recognition model from `benchmark/recognition`), along with a sample clip from *Gulliver's Travels* (1939), which [Wikipedia](https://en.wikipedia.org/wiki/Gulliver%27s_Travels_(1939_film)) notes is in the public domain. `recognize_characters.py` runs the pipeline on this data, which includes shot segmentation (TransNetV2), face detection (RT-DETR), face tracking, and recognition (using DINOv2 for generating face embeddings).


## Setup

```bash
conda env create -f environment.yml
conda activate animated-characters-pipeline
bash scripts/setup.sh
```

| Data/Model | Location |
|---|---|
| TransNetV2 shot-detection weights | `models/transnetv2-weights/` |
| DINOv2 repo checkout (supplies the model definition) | `third_party/dinov2/` |
| RT-DETR animated-face detector | `models/rtdetr.icf.FINAL.pt` |
| Fine-tuned DINOv2 character embedder | `models/dino_vitl14_crop25.FINAL.pth` |
| Character embeddings for the sample | `character_embeddings/gullivers.trave_tt0031397.tsv` |
| Sample clip | `samples/gulliver_clip.mp4` |


## Run

Process the sample clip end to end, writing face bounding boxes plus matched character names to
an annotated mp4:

```bash
python recognize_characters.py samples/gulliver_clip.mp4 \
    --transnet models/transnetv2-weights \
    --detr models/rtdetr.icf.FINAL.pt \
    --dino models/dino_vitl14_crop25.FINAL.pth \
    --dinov2-dir third_party/dinov2 \
    --cast character_embeddings/gullivers.trave_tt0031397.tsv \
    --device cuda \
    --annotate --out out/
```


Outputs are written in`out/` (`shots/`, `faces_detected/`,
`tracks/`, `track_reps/`, `recog/`, `annotated_movies/`), each written as its
stage finishes. Existing stage outputs are reused on a rerun; drop `--annotate`
to skip the annotated video.

