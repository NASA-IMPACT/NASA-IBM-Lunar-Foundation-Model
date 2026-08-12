"""Classification and scalar-regression tasks with the same LLRD + split-LR
recipe as :class:`LunarPixelwiseRegressionTask` / :class:`LunarSegmentationTask`.

See :mod:`terratorch_integration.lunar_llrd_mixin` for the LR split rules.
"""

from __future__ import annotations

from typing import Any

import torch
from terratorch.tasks import ClassificationTask, ScalarRegressionTask

from .lunar_llrd_mixin import _LunarLLRDMixin


class LunarClassificationTask(_LunarLLRDMixin, ClassificationTask):
    """Whole-patch classification with LLRD + split head/backbone LR.

    Same knobs as :class:`LunarPixelwiseRegressionTask`
    (`backbone_lr`, `head_lr`, `layer_decay`, `weight_decay`,
    `head_weight_decay`, `warmup_steps`, `cosine_t_max`, `eta_min`,
    `betas`). Additionally supports `label_smoothing` which the stock
    TerraTorch `ClassificationTask` does not expose — useful for
    many-class problems where overconfident logits saturate CE early.

    All other kwargs pass through to :class:`~terratorch.tasks.ClassificationTask`.
    """

    def __init__(self, *args: Any, label_smoothing: float = 0.0, **kwargs: Any) -> None:
        # Store label_smoothing BEFORE calling super().__init__ — torchgeo's
        # BaseTask runs `self.configure_losses()` during construction, and
        # our override reads `self.label_smoothing` there.
        self.label_smoothing = float(label_smoothing)
        super().__init__(*args, **kwargs)

    def configure_losses(self) -> None:
        """Extend stock loss config with `label_smoothing` for the plain-CE case.

        The stock `init_loss(..., loss='ce')` builds `nn.CrossEntropyLoss`
        without a `label_smoothing` arg. Rather than override the whole loss
        table we call the parent first, then swap in a smoothed criterion when
        (a) the user asked for smoothing and (b) the parent chose plain CE.
        Any other loss (bce, focal, jaccard, custom module) is left untouched.
        """
        super().configure_losses()
        if self.label_smoothing <= 0.0:
            return
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            ignore_index = self.hparams.get("ignore_index", -100)
            weight = self.criterion.weight
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_index if ignore_index is not None else -100,
                label_smoothing=self.label_smoothing,
            )


class LunarScalarRegressionTask(_LunarLLRDMixin, ScalarRegressionTask):
    """Whole-patch scalar regression with LLRD + split head/backbone LR."""
