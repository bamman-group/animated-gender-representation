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
            for attr in ("labels", "im_files", "npy_files", "ims", "im_hw0", "im_hw"):
                seq = getattr(dataset, attr)
                setattr(dataset, attr, [v for v, k in zip(seq, keep) if k])
            dataset.ni = len(dataset.labels)
            dataset.max_buffer_length = min(dataset.ni, dataset.batch_size * 8, 1000) if dataset.augment else 0
            print(f"[rtdetr-cap] dropped {dropped}/{len(keep)} training images "
                  f"with >{MAX_FACES_PER_IMAGE} faces")
        return dataset
