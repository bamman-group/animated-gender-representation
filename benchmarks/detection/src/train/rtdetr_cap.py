"""RT-DETR-only crowd-image filter for WIDER Face.

RT-DETR's contrastive-denoising query count is `num_dn // max_gt_in_batch`
(floored to a minimum group of 1, num_dn=100 by default), so once a training
sample's GT count exceeds num_dn, query count grows as `2 * max_gt` with no
further ceiling. A small tail of WIDER parade/crowd images have hundreds to
~1,400 faces; combined with mosaic, that explodes decoder self-attention
memory and caused a NaN-then-OOM training failure (see CLAUDE.md "Hard-won
fixes"). CappedRTDETRTrainer drops any training image with more than
MAX_FACES_PER_IMAGE faces from RT-DETR's training set entirely.

YOLO26 has no denoising-query mechanism (loss is linear in GT count) and
RetinaFace/YuNet are anchor-based with no CDN mechanism, so none of them
need or benefit from this filter. It is scoped to RT-DETR only via a trainer
subclass rather than editing the shared datasets/wf label files, which
YOLO26 also trains from.
"""
from __future__ import annotations

from ultralytics.models.rtdetr.train import RTDETRTrainer

MAX_FACES_PER_IMAGE = 500


class CappedRTDETRTrainer(RTDETRTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode != "train":
            return dataset
        keep = [len(label["bboxes"]) <= MAX_FACES_PER_IMAGE for label in dataset.labels]
        dropped = len(keep) - sum(keep)
        if dropped:
            # all of these are index-aligned with dataset.labels (see
            # ultralytics.data.base.BaseDataset.__init__ / load_image) and
            # must be filtered together or later epochs pair the wrong
            # cached image with a label at the same post-filter index.
            for attr in ("labels", "im_files", "npy_files", "ims", "im_hw0", "im_hw"):
                seq = getattr(dataset, attr)
                setattr(dataset, attr, [v for v, k in zip(seq, keep) if k])
            dataset.ni = len(dataset.labels)
            dataset.max_buffer_length = min(dataset.ni, dataset.batch_size * 8, 1000) if dataset.augment else 0
            print(f"[rtdetr-cap] dropped {dropped}/{len(keep)} training images "
                  f"with >{MAX_FACES_PER_IMAGE} faces")
        return dataset
