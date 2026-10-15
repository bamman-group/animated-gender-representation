# Vendored copy of biubug6/Pytorch_Retinaface

Upstream: https://github.com/biubug6/Pytorch_Retinaface (MIT license, see LICENSE.MIT)

Local modifications:
- `layers/modules/multibox_loss.py`: device-agnostic (removed hard-coded `.cuda()`).
- `models/retinaface.py`: torchvision >= 0.15 API (`weights=` instead of removed `pretrained=`).
- `data/extra_augment.py`: added zoom-out (expand) + low-probability
  photometric extras (blur/median-blur/grayscale/CLAHE) preproc wrapper.
- `data/mosaic.py`: added YOLO-style mosaic augmentation wrapper
  (landmark-aware; used via train_retinaface.py --mosaic).
- `data/icartoonface.py`: added dataset class for the label format produced by
  this repo's `src/prepare/` scripts (Nx15 targets: bbox corners, 10 landmark
  coords, landmark-validity flag).
