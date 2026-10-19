"""Reconstruct the published MiewID-msv3 network without executing downloaded Python."""

from typing import TYPE_CHECKING, cast, override

import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path


def load_miewid(weights: Path) -> nn.Module:
    """Only the training classifier is omitted; all inference weights must match exactly."""
    from safetensors.torch import load_file  # ruff: ignore[import-outside-top-level] - optional encoder.

    model = _MiewID()
    model.load_state_dict({name: value for name, value in load_file(weights).items() if not name.startswith("final.")})
    return model.eval()


class _MiewID(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import timm  # ruff: ignore[import-outside-top-level] - optional encoder.

        self.backbone = timm.create_model("efficientnetv2_rw_m", pretrained=False, num_classes=0)
        self.backbone.global_pool = _GeM()
        self.bn = nn.BatchNorm1d(2152)

    @override
    def forward(  # ty: ignore[invalid-mutable-override] - torch declares forward as writable.
        self, pixels: torch.Tensor
    ) -> torch.Tensor:
        return cast("torch.Tensor", self.bn(self.backbone(pixels).flatten(1)))


class _GeM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * 3)

    @override
    def forward(  # ty: ignore[invalid-mutable-override] - torch declares forward as writable.
        self, pixels: torch.Tensor
    ) -> torch.Tensor:
        return pixels.clamp(min=1e-6).pow(self.p).mean(dim=(-2, -1), keepdim=True).pow(1 / self.p)


__all__ = ["load_miewid"]
