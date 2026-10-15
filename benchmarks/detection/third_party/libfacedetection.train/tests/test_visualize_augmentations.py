from __future__ import annotations

import numpy as np
import torch

from yunet_train.tools.visualize_augmentations import draw_face_annotations, image_tensor_to_bgr_uint8


def test_image_tensor_to_bgr_uint8_restores_train_image_layout() -> None:
    image = torch.zeros((3, 4, 5), dtype=torch.float32)
    image[0] = 12.5
    image[1] = 300.0
    image[2] = -10.0

    restored = image_tensor_to_bgr_uint8(image)

    assert restored.shape == (4, 5, 3)
    assert restored.dtype == np.uint8
    assert restored[0, 0].tolist() == [12, 255, 0]


def test_draw_face_annotations_renders_boxes_and_keypoints() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    boxes = torch.tensor([[4.0, 5.0, 20.0, 22.0]])
    keypoints = torch.tensor([[[6.0, 7.0, 1.0], [18.0, 7.0, 1.0], [12.0, 12.0, 1.0], [8.0, 20.0, 1.0], [17.0, 20.0, 1.0]]])
    ignored_boxes = torch.tensor([[1.0, 1.0, 3.0, 3.0]])

    rendered = draw_face_annotations(
        image=image,
        boxes=boxes,
        keypoints=keypoints,
        ignored_boxes=ignored_boxes,
        title="debug",
    )

    assert rendered.shape == image.shape
    assert np.count_nonzero(rendered) > 0
    greenish = (rendered[:, :, 1] > rendered[:, :, 0]) & (rendered[:, :, 1] > rendered[:, :, 2])
    assert greenish.any()
