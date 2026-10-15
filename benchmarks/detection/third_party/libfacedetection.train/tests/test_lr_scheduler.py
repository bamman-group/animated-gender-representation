from __future__ import annotations

import torch

from yunet_train.engine import LinearWarmupMultiStepLR


def test_linear_warmup_multi_step_lr_matches_yunet_schedule_shape() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.01)
    scheduler = LinearWarmupMultiStepLR(
        optimizer,
        milestones=(3,),
        gamma=0.1,
        warmup_iters=4,
        warmup_ratio=0.1,
    )

    assert scheduler.step(epoch=1) == [0.001]
    assert scheduler.step(epoch=1) == [0.0032500000000000003]
    assert scheduler.step(epoch=1) == [0.0055000000000000005]
    assert scheduler.step(epoch=1) == [0.007750000000000001]
    assert scheduler.step(epoch=1) == [0.01]
    assert scheduler.step(epoch=3) == [0.001]
    assert optimizer.param_groups[0]["lr"] == 0.001


def test_linear_warmup_multi_step_lr_restores_state() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.01)
    scheduler = LinearWarmupMultiStepLR(optimizer, milestones=(3,), warmup_iters=4)
    scheduler.step(epoch=1)
    scheduler.step(epoch=1)

    new_optimizer = torch.optim.SGD([param], lr=0.5)
    restored = LinearWarmupMultiStepLR(new_optimizer, milestones=(99,), warmup_iters=0)
    restored.load_state_dict(scheduler.state_dict())

    assert restored.iter_count == scheduler.iter_count
    assert restored.get_last_lr() == scheduler.get_last_lr()
    assert new_optimizer.param_groups[0]["lr"] == scheduler.get_last_lr()[0]
