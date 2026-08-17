import os

import cv2
import numpy as np
import torch
import torch.utils.data as data


class ICartoonFaceDetection(data.Dataset):
    """iCartoonFace training set in the label format written by
    prepare_icartoonface.py:

        # <filename>
        x1 y1 x2 y2 l0x l0y l1x l1y l2x l2y l3x l3y l4x l4y flag

    Yields (image, target) where target is Nx15: 4 bbox corners, 10 landmark
    coords, and a final label that is 1 (landmarks valid) or -1 (no landmarks),
    matching the WiderFaceDetection contract expected by preproc/MultiBoxLoss.
    """

    def __init__(self, label_file, images_root, preproc=None):
        self.preproc = preproc
        self.images_root = images_root
        self.imgs_path = []
        self.annotations = []

        current = None
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    current = []
                    self.annotations.append(current)
                    self.imgs_path.append(os.path.join(images_root, line[1:].strip()))
                else:
                    current.append([float(x) for x in line.split()])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        if img is None:
            raise FileNotFoundError(self.imgs_path[index])

        target = np.array(self.annotations[index], dtype=np.float64).reshape(-1, 15)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target
