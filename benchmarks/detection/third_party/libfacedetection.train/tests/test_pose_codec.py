from __future__ import annotations

import torch

from yunet_train.tasks.pose.codec import pose_keypoints_decode, pose_keypoints_encode


def test_pose_keypoints_encode_decode_roundtrip_preserves_visibility_logits() -> None:
    priors = torch.tensor([[8.0, 16.0, 8.0, 8.0], [16.0, 16.0, 8.0, 8.0]])
    keypoints = torch.zeros(2, 17, 3)
    keypoints[..., 0] = 10.0
    keypoints[..., 1] = 20.0
    keypoints[..., 2] = torch.arange(17).float()

    encoded = pose_keypoints_encode(priors, keypoints, kpt_shape=(17, 3))
    decoded = pose_keypoints_decode(priors, encoded, kpt_shape=(17, 3))

    torch.testing.assert_close(decoded, keypoints)
