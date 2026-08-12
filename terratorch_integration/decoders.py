"""Custom TerraTorch decoders for lunar-fm downstream tasks."""

import torch
import torch.nn.functional as F
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY


@TERRATORCH_DECODER_REGISTRY.register
class SumFuseDeepGNDecoder(nn.Module):
    """DPT-style sum-fuse + deep GroupNorm head.

    Mirrors `RegressionFPN.forward` in the NASA-IMPACT lunarfm-downstream
    baseline: takes the list of FPN feature maps (all at the same channel dim,
    typically 256 from a torchvision `FeaturePyramidNetworkNeck`), bilinearly
    upsamples every level to the finest one, sums them, and then applies a
    4-stage GroupNorm+GELU 3x3 conv head. Output channels taper
    `c -> c -> c/2 -> c/2 -> c/4` so the terratorch `RegressionHead` finishes
    with a 1x1 to `out=1` at (near-)input resolution.

    The trailing bilinear upsample to input resolution is handled by
    `PixelWiseModel.forward` when `rescale=True` (default), so this decoder
    only needs to emit feature maps at the finest FPN level.
    """

    def __init__(
        self,
        embed_dim: list[int],
        upsample_output_size: list[int] | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if len(set(embed_dim)) != 1:
            raise ValueError(
                f"SumFuseDeepGNDecoder expects all input feature maps to share the "
                f"same channel count; got {embed_dim}. Pair with FeaturePyramidNetworkNeck "
                f"upstream to unify channels."
            )
        c = embed_dim[0]

        def blk(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, co), co),
                nn.GELU(),
            )

        self.head = nn.Sequential(
            blk(c, c),
            blk(c, c // 2),
            blk(c // 2, c // 2),
            blk(c // 2, c // 4),
        )
        self.out_channels = c // 4
        self.upsample_output_size = (
            tuple(upsample_output_size) if upsample_output_size is not None else None
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_hw = features[0].shape[-2:]
        merged = features[0]
        for f in features[1:]:
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            merged = merged + f
        # Run the deep GN head at the requested output resolution. Without this,
        # the head runs at the finest FPN scale.
        if self.upsample_output_size is not None and merged.shape[-2:] != self.upsample_output_size:
            merged = F.interpolate(
                merged, size=self.upsample_output_size,
                mode="bilinear", align_corners=False,
            )
        return self.head(merged)
