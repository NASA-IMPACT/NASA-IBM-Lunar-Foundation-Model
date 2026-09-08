"""Model registration with TerraTorch registry."""

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from .lunar_backbone import LunarBackbone


@TERRATORCH_BACKBONE_REGISTRY.register
def ni_lfm_v1_tiny(**kwargs):
    """NILunarFM v1 Tiny model.

    Architecture:
        - Encoder depth: 12 layers
        - Decoder depth: 4 layers
        - Model dimension: 192
        - Attention heads: 3
        - MLP ratio: 4.0
        - Parameters: ~8M

    Required kwargs:
        - cfg / backbone_cfg: path to the pretraining `config.yaml`
          (ships as `weights/backbone/config.yaml`). Carries both the
          model config and the per-modality info (`data.domains`).

    See `LunarBackbone` for all other kwargs.
    """
    return LunarBackbone(variant="tiny", **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def ni_lfm_v1_base(**kwargs):
    """NILunarFM v1 Base model.

    Architecture (from MODEL_CONFIGS["base"]):
        - Encoder depth: 12 layers
        - Decoder depth: 12 layers
        - Model dimension: 768
        - Attention heads: 12
        - MLP ratio: 4.0
        - Parameters: ~86M

    Same required kwargs as `ni_lfm_v1_tiny` (`cfg`). See
    `LunarBackbone` for full kwargs.
    """
    return LunarBackbone(variant="base", **kwargs)


@TERRATORCH_BACKBONE_REGISTRY.register
def ni_lfm_v1_large(**kwargs):
    """NILunarFM v1 Large model.

    Architecture (from MODEL_CONFIGS["large"]):
        - Encoder depth: 24 layers
        - Decoder depth: 24 layers
        - Model dimension: 1024
        - Attention heads: 16
        - MLP ratio: 4.0
        - Parameters: ~307M

    Same required kwargs as `ni_lfm_v1_tiny` (`cfg`). See
    `LunarBackbone` for full kwargs.
    """
    return LunarBackbone(variant="large", **kwargs)
