from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from yunet_train.tasks.face import FaceSample, ToTensor, collate_face_samples
from yunet_train.engine import LinearWarmupMultiStepLR
from yunet_train.tasks.face import YuNetCriterion, build_yunet, evaluate_loss, train_one_epoch


def _training_sample() -> FaceSample:
    image = np.ones((64, 64, 3), dtype=np.float32)
    return FaceSample(
        image=image,
        boxes=np.array([[0, 0, 8, 8]], dtype=np.float32),
        labels=np.array([0], dtype=np.int64),
        keypoints=np.array(
            [
                [
                    [0, 0, 1],
                    [8, 0, 1],
                    [4, 4, 1],
                    [0, 8, 1],
                    [8, 8, 1],
                ]
            ],
            dtype=np.float32,
        ),
        ignored_boxes=np.zeros((0, 4), dtype=np.float32),
        ignored_labels=np.zeros((0,), dtype=np.int64),
        filename="synthetic.jpg",
        original_shape=(64, 64, 3),
        image_shape=(64, 64, 3),
        pad_shape=(64, 64, 3),
    )


def test_train_one_epoch_smoke_updates_model() -> None:
    torch.manual_seed(20260503)
    sample = ToTensor()(_training_sample())
    data_loader = DataLoader(
        [sample],
        batch_size=1,
        collate_fn=collate_face_samples,
    )
    model = build_yunet("yunet_s")
    criterion = YuNetCriterion(strides=(8, 16, 32))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    stats = train_one_epoch(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        optimizer=optimizer,
        device="cpu",
    )

    assert stats.steps == 1
    assert np.isfinite(stats.loss)
    assert any(
        not torch.equal(before[name], param.detach())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def test_evaluate_loss_smoke_keeps_model_unchanged() -> None:
    torch.manual_seed(20260503)
    sample = ToTensor()(_training_sample())
    data_loader = DataLoader(
        [sample],
        batch_size=1,
        collate_fn=collate_face_samples,
    )
    model = build_yunet("yunet_s")
    criterion = YuNetCriterion(strides=(8, 16, 32))
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    stats = evaluate_loss(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        device="cpu",
    )

    assert stats.steps == 1
    assert np.isfinite(stats.loss)
    assert all(
        torch.equal(before[name], param.detach())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def test_train_one_epoch_steps_lr_scheduler() -> None:
    torch.manual_seed(20260503)
    sample = ToTensor()(_training_sample())
    data_loader = DataLoader(
        [sample],
        batch_size=1,
        collate_fn=collate_face_samples,
    )
    model = build_yunet("yunet_s")
    criterion = YuNetCriterion(strides=(8, 16, 32))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = LinearWarmupMultiStepLR(
        optimizer,
        milestones=(400, 544),
        warmup_iters=10,
        warmup_ratio=0.5,
    )

    train_one_epoch(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        optimizer=optimizer,
        device="cpu",
        epoch=1,
        lr_scheduler=scheduler,
    )

    assert scheduler.iter_count == 1
    assert optimizer.param_groups[0]["lr"] == 0.005
