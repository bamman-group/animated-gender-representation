# YuNet Training in PyTorch

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

This repository provides a lightweight PyTorch training pipeline for YuNet-based detection tasks.

It includes:

- YuNet face detection training on WIDER Face.
- YuNet face evaluation and export for ONNX, C++ weights, and TFLite.
- YuNet pose detection training for YOLO-pose style datasets.
- Pose validation, COCO keypoint AP evaluation, visualization, and ONNX export.

The code is implemented directly with PyTorch. It does not require MMDetection, MMCV, Ultralytics, or a large detection framework.

## Why Use This Repository

- Lightweight: small YuNet-style models and direct PyTorch training code.
- Practical exports: face models can be exported to ONNX, TFLite, and libfacedetection C++ weight data.
- Multi-task structure: face and pose are separate task implementations sharing the same YuNet backbone, neck, training utilities, assignment, priors, bbox losses, and NMS.
- Easy to inspect: datasets, transforms, models, losses, training loops, and evaluation code are plain Python modules.
- CI-friendly: most workflows have CPU smoke tests and small-sample modes.
- Visualization of the YuNet network architecture: [[netron]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_320_320.onnx)

## Repository Layout

```text
yunet_train/
  engine/        shared training and detection primitives
  models/        shared YuNet backbone, neck, layers, and initialization
  tasks/
    face/        WIDER Face detection task
    pose/        YOLO-pose / COCO-style keypoint detection task
  cli/           command-line entry points
  tools/         dataset checks, visualization, and debugging tools

doc/
  pose-adaptation-plan.md
  multitask-refactor-plan.md
```

## Installation

Create a Python environment and install PyTorch first. Choose the PyTorch build that matches your CUDA version from the official instructions: https://pytorch.org/.

```shell
conda create -n yunet python=3.11
conda activate yunet
# Install pytorch. This codebase has been tested with PyTorch 2.11.0 and CUDA 12.6.
```

Clone and install this repository:

```shell
git clone https://github.com/ShiqiYu/libfacedetection.train.git
cd libfacedetection.train
python -m pip install -e ".[dev]"
# or
python -m pip install -r requirements.txt
```

Optional dependencies:

```shell
python -m pip install -e ".[pose]"          # COCO keypoint evaluation
python -m pip install -r requirements-tflite.txt
```

`torch` is intentionally not pinned by this repository, so installing the package will not replace the PyTorch build in your environment.

## Face Detection

### Prepare WIDER Face

Download WIDER Face and place it under `data/widerface`:

```text
data/widerface
|-- WIDER_train
|   `-- images
|-- WIDER_val
|   `-- images
`-- labelv2
    |-- train
    |   `-- labelv2.txt
    `-- val
        |-- gt
        `-- labelv2.txt
```

The `labelv2` annotations come from SCRFD.

Check the dataset:

```shell
python -m yunet_train.tools.check_widerface --split train --check-images 10
python -m yunet_train.tools.check_widerface --split val --check-images 10
```

### Train Face Detector

CPU smoke test:

```shell
python -m yunet_train.cli.train --variant yunet_s --epochs 1 --batch-size 1 --workers 0 --device cpu --image-size 64 --limit-samples 1 --no-tensorboard
```

Train YuNet_n:

```shell
python -m yunet_train.cli.train --variant yunet_n --epochs 640 --batch-size 16 --workers 2 --device cuda --checkpoint-interval 80 --eval-interval 100 --work-dir work_dirs/yunet_n
```

Resume training:

```shell
python -m yunet_train.cli.train --variant yunet_n --epochs 640 --batch-size 16 --workers 2 --device cuda --resume work_dirs/yunet_n/latest.pth --work-dir work_dirs/yunet_n
```

Useful outputs:

```text
work_dirs/yunet_n/latest.pth
work_dirs/yunet_n/best_loss.pth
work_dirs/yunet_n/metrics.csv
work_dirs/yunet_n/train.log
work_dirs/yunet_n/tensorboard
```

### Evaluate on WIDER Face

```shell
python -m yunet_train.cli.eval_widerface work_dirs/yunet_n/best_loss.pth --variant yunet_n --device cuda --batch-size 1 --workers 4 --output-dir work_dirs/yunet_n_widerface_eval --save-preds
```

Evaluate the released YuNet_n checkpoint:

```shell
python -m yunet_train.cli.eval_widerface weights/yunet_n.pth --variant yunet_n --device cuda --batch-size 1 --workers 4 --mode origin --output-dir work_dirs/legacy_yunet_n_eval --save-preds
```

Released YuNet_n checkpoint on WIDER Face val:

```text
AP_easy=0.892, AP_medium=0.883, AP_hard=0.811
```

### Export Face Models

Export C++ weight data for libfacedetection:

```shell
python -m yunet_train.cli.export_cpp work_dirs/yunet_n/best_loss.pth --variant yunet_n --output-file work_dirs/export/facedetectcnn-data.cpp
```

Export ONNX:

```shell
python -m yunet_train.cli.export_onnx work_dirs/yunet_n/best_loss.pth --variant yunet_n --shape 640 640 --verify --output-file work_dirs/export/yunet_n_640_640.onnx
```

Export dynamic-shape ONNX:

```shell
python -m yunet_train.cli.export_onnx work_dirs/yunet_n/best_loss.pth --variant yunet_n --dynamic-export --output-file work_dirs/export/yunet_n_dynamic.onnx
```

Export TFLite:

```shell
python -m yunet_train.cli.export_tflite work_dirs/yunet_n/best_loss.pth --variant yunet_n --shape 640 640 --output-file work_dirs/export/yunet_n_640_640.tflite
```

The face ONNX output order follows the original YuNet convention:

```text
cls_8, cls_16, cls_32
obj_8, obj_16, obj_32
bbox_8, bbox_16, bbox_32
kps_8, kps_16, kps_32
```

### Compare ONNX Inference

Evaluate an exported ONNX model with ONNX Runtime:

```shell
python -m yunet_train.cli.compare_inference work_dirs/export/yunet_n_640_640.onnx --mode AUTO --eval --score-thresh 0.02 --nms-thresh 0.45 --out-dir work_dirs/compare_yunet_n
```

Run single-image inference:

```shell
python -m yunet_train.cli.compare_inference work_dirs/export/yunet_n_640_640.onnx --mode AUTO --image image.jpg --out-dir work_dirs/sample
```

Supported ONNX filename prefixes are `yunet`, `scrfd`, `yolo5face`, and `retinaface`.

## Pose Detection

The pose task trains a detection-style YuNet model that predicts person boxes and keypoints in one pass.

The initial target is YOLO-pose label format with COCO17 keypoints:

```text
class cx cy w h x1 y1 v1 x2 y2 v2 ... x17 y17 v17
```

Coordinates are normalized to image width and height.

### Prepare Pose Data

Default tiny dataset path:

```text
data/coco8-pose
|-- images
|   |-- train
|   `-- val
`-- labels
    |-- train
    `-- val
```

For full COCO keypoint AP, install the pose extra and provide a COCO keypoint annotation file:

```shell
python -m pip install -e ".[pose]"
```

### Train Pose Detector

CPU smoke test:

```shell
python -m yunet_train.cli.train_pose --data-root data/coco8-pose --variant yunet_n --epochs 1 --batch-size 1 --workers 0 --device cpu --image-size 160 --limit-samples 2
```

Train:

```shell
python -m yunet_train.cli.train_pose --data-root data/coco8-pose --variant yunet_n --epochs 100 --batch-size 16 --workers 4 --device cuda --work-dir work_dirs/yunet_pose_n
```

Run a tiny overfit check:

```shell
python -m yunet_train.tools.check_pose_overfit --data-root data/coco8-pose --samples 4 --epochs 120 --image-size 160 --device cpu
```

Visualize pose data and augmentations:

```shell
python -m yunet_train.tools.visualize_pose_dataset --data-root data/coco8-pose --split train --out-dir work_dirs/pose_vis
```

### Evaluate Pose Detector

Validation loss and optional visualizations:

```shell
python -m yunet_train.cli.eval_pose work_dirs/yunet_pose_n/best_loss.pth --data-root data/coco8-pose --device cuda --save-visualizations 16
```

COCO keypoint AP:

```shell
python -m yunet_train.cli.eval_pose_coco work_dirs/yunet_pose_n/best_loss.pth --ann-file data/coco/annotations/person_keypoints_val2017.json --image-dir data/coco/val2017 --device cuda
```

### Export Pose ONNX

```shell
python -m yunet_train.cli.export_pose_onnx work_dirs/yunet_pose_n/best_loss.pth --variant yunet_n --shape 640 640 --verify --output-file work_dirs/export/yunet_pose_n.onnx
```

## Console Commands

After editable installation, these command names are available:

```text
yunet-train
yunet-train-pose
yunet-eval-widerface
yunet-eval-pose
yunet-eval-pose-coco
yunet-export-onnx
yunet-export-pose-onnx
yunet-export-cpp
yunet-export-tflite
yunet-compare-inference
yunet-check-widerface
yunet-check-env
yunet-visualize-pose
yunet-check-pose-overfit
```

The examples above use `python -m ...` because it also works directly from a source checkout.

## Testing

Run the full test suite:

```shell
python -m pytest -q
python -m ruff check yunet_train tests
```

Some tests skip automatically when optional datasets, optional dependencies, or model weights are not available locally.

## Citation

YuNet:

```text
@article{yunet,
  title={YuNet: A Tiny Millisecond-level Face Detector},
  author={Wu, Wei and Peng, Hanyang and Yu, Shiqi},
  journal={Machine Intelligence Research},
  pages={1--10},
  year={2023},
  doi={10.1007/s11633-023-1423-y},
  publisher={Springer}
}
```

EIoU:

```text
@article{eiou,
 author={Peng, Hanyang and Yu, Shiqi},
 journal={IEEE Transactions on Image Processing},
 title={A Systematic IoU-Related Method: Beyond Simplified Regression for Better Localization},
 year={2021},
 volume={30},
 pages={5032-5044},
 doi={10.1109/TIP.2021.3077144}
}
```

Face detection survey:

```text
@article{facedetect-yu,
  author={Feng, Yuantao and Yu, Shiqi and Peng, Hanyang and Li, Yan-Ran and Zhang, Jianguo},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  title={Detect Faces Efficiently: A Survey and Evaluations},
  year={2022},
  volume={4},
  number={1},
  pages={1-18},
  doi={10.1109/TBIOM.2021.3120412}
}
```
