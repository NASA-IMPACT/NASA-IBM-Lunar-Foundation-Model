"""Pixel-wise regression task with the shared LLRD + split-group optimiser
recipe (:class:`_LunarLLRDMixin`).
"""

from __future__ import annotations

from terratorch.tasks import PixelwiseRegressionTask

from .lunar_llrd_mixin import _LunarLLRDMixin


class LunarPixelwiseRegressionTask(_LunarLLRDMixin, PixelwiseRegressionTask):
    """Pixel-wise regression with LLRD, split head/backbone LR, and a
    dedicated LR group for randomly-initialized new-modality embedders.

    All optimiser knobs (`backbone_lr`, `head_lr`, `layer_decay`,
    `weight_decay`, `head_weight_decay`, `warmup_steps`, `cosine_t_max`,
    `eta_min`, `betas`) come from :class:`_LunarLLRDMixin`. Any other
    keyword argument is forwarded to
    :class:`~terratorch.tasks.PixelwiseRegressionTask`.
    """
