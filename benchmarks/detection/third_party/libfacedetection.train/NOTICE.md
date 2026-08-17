# Vendored copy of ShiqiYu/libfacedetection.train

Upstream: https://github.com/ShiqiYu/libfacedetection.train (Apache-2.0, see LICENSE)

Vendored as plain files (it was originally a git submodule; the history was
flattened when this benchmark was imported into the paper monorepo). The code
is unmodified from upstream — the mosaic augmentation this benchmark adds is
injected at runtime by `src/train/train_yunet.py` / `src/train/yunet_mosaic.py`
without touching anything in this directory.

`weights/yunet_n.pth` and `weights/yunet_s.pth` are upstream's released
WIDER Face-trained checkpoints, version-controlled here (see the exception at
the bottom of this repo's `.gitignore`) because they serve as both the
`yunet_wf` baseline and the initialization for YuNet fine-tuning.
