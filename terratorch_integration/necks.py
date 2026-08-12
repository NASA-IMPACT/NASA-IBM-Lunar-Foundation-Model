"""Custom neck modules for the lunar-fm TerraTorch integration."""

import torch
from torch import nn

from terratorch.models.necks import Neck
from terratorch.registry import TERRATORCH_NECK_REGISTRY


@TERRATORCH_NECK_REGISTRY.register
class LearnedTokenProjection(Neck):
    """Apply learned linear projection to transform the last dimension of features from C to D.

    Dynamically creates projection layers on first forward pass based on actual input dimensions.
    Each feature layer gets its own learned projection from its input channel dimension C_i to output dimension D.
    Works on any tensor shape by projecting only the last dimension.
    Useful for standardizing feature dimensions across layers or adapting to decoder requirements.

    Example usage in config:
        neck:
          - name: LearnedTokenProjection
            output_dim: 768
    """

    def __init__(self, channel_list: list[int], output_dim: int):
        """Initialize the neck. Projection layers are created on first forward pass.

        Args:
            channel_list: List of input channel dimensions (may be incorrect if previous neck doesn't update properly)
            output_dim: Target output dimension D for all layers
        """
        super().__init__(channel_list)
        self.output_dim = output_dim
        self.num_features = len(channel_list)
        # Initialize with empty ModuleList for checkpoint compatibility
        self.projections = nn.ModuleList()
        self._initialized = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Custom state dict loading to handle dynamically created projections."""
        # Check if projections exist in state_dict
        projection_keys = [k for k in state_dict.keys() if k.startswith(f"{prefix}projections.")]

        if projection_keys and len(self.projections) == 0:
            # Infer number of projections and their dimensions from state_dict
            num_projections = len([k for k in projection_keys if k.endswith('.weight')])

            # Create placeholder projections with correct dimensions
            for i in range(num_projections):
                weight_key = f"{prefix}projections.{i}.weight"
                if weight_key in state_dict:
                    in_features = state_dict[weight_key].shape[1]  # weight shape is (out_features, in_features)
                    self.projections.append(nn.Linear(in_features, self.output_dim))

            self._initialized = True
            print(f"LearnedTokenProjection: Created {num_projections} projections from checkpoint")

        # Call parent's load method
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, channel_list: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        """Apply learned projection to the last dimension of each feature.

        Args:
            channel_list (list[torch.Tensor]): List of feature tensors with shape (..., C_i)

        Returns:
            list[torch.Tensor]: List of projected features with shape (..., D)
        """
        # Initialize projections on first forward pass with actual input dimensions
        # Skip if already loaded from checkpoint (projections not empty)
        if not self._initialized and len(self.projections) == 0:
            actual_dims = [feat.shape[-1] for feat in channel_list]
            self.projections = nn.ModuleList([
                nn.Linear(in_dim, self.output_dim).to(channel_list[0].device)
                for in_dim in actual_dims
            ])
            self._initialized = True
            print(f"LearnedTokenProjection initialized with input dims: {actual_dims} -> output dim: {self.output_dim}")
        elif len(self.projections) > 0 and not self._initialized:
            # Loaded from checkpoint
            self._initialized = True
            print(f"LearnedTokenProjection loaded from checkpoint with {len(self.projections)} projections")

        projected_features = []
        for feat, projection in zip(channel_list, self.projections, strict=True):
            # Apply learned linear projection to the last dimension
            feat_projected = projection(feat)
            projected_features.append(feat_projected)

        return projected_features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        """Return the new channel list after projection.

        Args:
            channel_list (list[int]): Input channel dimensions

        Returns:
            list[int]: Output channel dimensions (all equal to output_dim)
        """
        return [self.output_dim] * len(channel_list)


class _LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) tensor.

    Equivalent to nn.LayerNorm(C) applied along channels at each spatial position — this is what
    ViTDet and detectron2 use inside the Simple Feature Pyramid. nn.LayerNorm expects channels-last; nn.GroupNorm(1, C)
    is subtly different because it also averages over (H, W), so we implement the channels-only variant explicitly.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


@TERRATORCH_NECK_REGISTRY.register
class SimpleFeaturePyramid(Neck):
    """Simple Feature Pyramid from ViTDet (Li et al., 2022).

    Builds a 4-level pyramid from a **single** input feature map (typically the last-block tokens of a plain ViT,
    reshaped to (B, C, H, W)) by applying parallel strided deconv / identity / maxpool branches — no
    cross-scale top-down fusion. Each level is then projected to a common `out_channels` via
    `conv1x1 -> LN -> conv3x3 -> LN`.

    Reference: "Exploring Plain Vision Transformer Backbones for Object Detection", Li et al. 2022, Sec. 3.

    Input: list with one `(B, C, H, W)` tensor. Pair upstream with `SelectIndices [-1]` + `ReshapeTokensToImage`.
    Output: list of 4 `(B, out_channels, H', W')` tensors at scales `{4H, 2H, H, H/2}` in that order.

    This neck replaces `LearnedInterpolateToPyramidal`. It also replaces
    `FeaturePyramidNetworkNeck` (the paper argues top-down lateral
    connections are unnecessary once every scale is derived from the same
    strong last-block feature), but chaining an FPN afterward is safe — all
    levels already share `out_channels` so the FPN's laterals become 1x1
    projections that the network can learn to leave near-identity.
    """

    _SCALES: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)

    def __init__(self, channel_list: list[int], out_channels: int = 256) -> None:
        super().__init__(channel_list)
        if len(channel_list) != 1:
            raise ValueError(
                f"SimpleFeaturePyramid expects exactly 1 input feature map, "
                f"got {len(channel_list)}. Pair with SelectIndices([-1]) upstream."
            )
        dim = channel_list[0]
        self.out_channels = out_channels
        self.stages = nn.ModuleList()
        for scale in self._SCALES:
            if scale == 4.0:
                resample: nn.Module = nn.Sequential(
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    _LayerNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                mid = dim // 4
            elif scale == 2.0:
                resample = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
                mid = dim // 2
            elif scale == 1.0:
                resample = nn.Identity()
                mid = dim
            else:  # 0.5
                resample = nn.MaxPool2d(kernel_size=2, stride=2)
                mid = dim
            self.stages.append(
                nn.Sequential(
                    resample,
                    nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                    _LayerNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    _LayerNorm2d(out_channels),
                )
            )

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                f"SimpleFeaturePyramid expects a single input feature map, got {len(features)}."
            )
        x = features[0]
        if x.dim() != 4:
            raise ValueError(
                f"SimpleFeaturePyramid expects (B, C, H, W); got shape {tuple(x.shape)}. "
                "Add ReshapeTokensToImage upstream."
            )
        return [stage(x) for stage in self.stages]

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [self.out_channels] * len(self._SCALES)


@TERRATORCH_NECK_REGISTRY.register
class MultilayerSimpleFeaturePyramid(Neck):
    """DPT-style multi-layer ViTDet Simple Feature Pyramid.

    Takes exactly 4 (B, C, H, W) feature maps -- one tap per pyramid scale --
    and feeds each into a distinct SFP head. Shallow taps target fine scales,
    deep taps target coarse scales, matching the layer ordering in
    `SelectIndices [2, 5, 8, 11]`:

        tap 0 (block 2)  -> scale 4.0  (2x ConvT -> LN -> GELU -> 2x ConvT)
        tap 1 (block 5)  -> scale 2.0  (2x ConvT)
        tap 2 (block 8)  -> scale 1.0  (identity)
        tap 3 (block 11) -> scale 0.5  (MaxPool)

    Each level is then projected to `out_channels` via
    `conv1x1 -> LN -> conv3x3 -> LN` (same as SFP).

    Pair upstream with:
        - SelectIndices [2, 5, 8, 11]
        - LearnedTokenProjection output_dim=<C>
        - ReshapeTokensToImage remove_cls_token=false h=<token_grid_h>
    """

    _SCALES: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)

    def __init__(self, channel_list: list[int], out_channels: int = 256) -> None:
        super().__init__(channel_list)
        if len(channel_list) != 4:
            raise ValueError(
                f"MultilayerSimpleFeaturePyramid expects exactly 4 input feature maps, "
                f"got {len(channel_list)}. Pair with SelectIndices([2,5,8,11]) upstream."
            )
        if len(set(channel_list)) != 1:
            raise ValueError(
                f"MultilayerSimpleFeaturePyramid expects all 4 inputs to share the same "
                f"channel dim; got {channel_list}. Insert LearnedTokenProjection upstream."
            )
        dim = channel_list[0]
        self.out_channels = out_channels
        self.stages = nn.ModuleList()
        for scale in self._SCALES:
            if scale == 4.0:
                resample: nn.Module = nn.Sequential(
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    _LayerNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                mid = dim // 4
            elif scale == 2.0:
                resample = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
                mid = dim // 2
            elif scale == 1.0:
                resample = nn.Identity()
                mid = dim
            else:  # 0.5
                resample = nn.MaxPool2d(kernel_size=2, stride=2)
                mid = dim
            self.stages.append(
                nn.Sequential(
                    resample,
                    nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
                    _LayerNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    _LayerNorm2d(out_channels),
                )
            )

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        if len(features) != 4:
            raise ValueError(
                f"MultilayerSimpleFeaturePyramid expects 4 input feature maps, "
                f"got {len(features)}."
            )
        outs = []
        for x, stage in zip(features, self.stages, strict=True):
            if x.dim() != 4:
                raise ValueError(
                    f"MultilayerSimpleFeaturePyramid expects (B, C, H, W); got shape "
                    f"{tuple(x.shape)}. Add ReshapeTokensToImage upstream."
                )
            outs.append(stage(x))
        return outs

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [self.out_channels] * len(self._SCALES)
