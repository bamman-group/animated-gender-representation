from __future__ import annotations

import torch


class MlvlPointGenerator:
    def __init__(self, strides: tuple[int, ...], offset: float = 0.0):
        self.strides = tuple((stride, stride) for stride in strides)
        self.offset = offset

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def grid_priors(
        self,
        featmap_sizes: list[tuple[int, int]],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        with_stride: bool = True,
    ) -> list[torch.Tensor]:
        if len(featmap_sizes) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature maps, got {len(featmap_sizes)}")

        return [
            self.single_level_grid_priors(
                featmap_size,
                level_idx,
                dtype=dtype,
                device=device,
                with_stride=with_stride,
            )
            for level_idx, featmap_size in enumerate(featmap_sizes)
        ]

    def single_level_grid_priors(
        self,
        featmap_size: tuple[int, int],
        level_idx: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        with_stride: bool = True,
    ) -> torch.Tensor:
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        shift_x = shift_x.to(dtype)
        shift_y = shift_y.to(dtype)
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shift_xx = xx.reshape(-1)
        shift_yy = yy.reshape(-1)
        if not with_stride:
            return torch.stack([shift_xx, shift_yy], dim=-1)
        stride_w_tensor = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
        stride_h_tensor = shift_yy.new_full((shift_yy.shape[0],), stride_h).to(dtype)
        return torch.stack([shift_xx, shift_yy, stride_w_tensor, stride_h_tensor], dim=-1)

