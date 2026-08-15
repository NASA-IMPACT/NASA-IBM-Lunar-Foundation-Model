"""Deterministic drop-in for ``nn.CrossEntropyLoss``.

``nn.CrossEntropyLoss`` dispatches to ``nll_loss2d_forward_out_cuda_template``,
which has no deterministic CUDA implementation. Under
``torch.use_deterministic_algorithms(True)`` (Lightning's
``trainer.deterministic: true``) it raises, so *every* config in this repo --
they all use ``ce`` as part of the composite loss -- fails to start.

This implementation is numerically identical to ``nn.CrossEntropyLoss`` with
``reduction="mean"`` but is built only from ops with deterministic CUDA
kernels:

* ``log_softmax``  -- deterministic
* one-hot multiply + ``sum`` over the channel dim -- deterministic

The obvious alternative, ``logp.gather(1, target)``, is NOT usable: gather's
backward is ``scatter_add``, which uses float atomics and is itself
nondeterministic. The one-hot formulation costs a (B, C, H, W) temporary,
which is negligible for the 2-class masks used here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicCrossEntropyLoss(nn.Module):
    """``nn.CrossEntropyLoss(reduction="mean")`` without nondeterministic ops.

    Args:
        ignore_index: Target value to exclude from the loss and from the
            normalising denominator.
        weight: Optional per-class weights, as in ``nn.CrossEntropyLoss``.
    """

    def __init__(self, ignore_index: int | None = -100, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.ignore_index = -100 if ignore_index is None else int(ignore_index)
        self.register_buffer("weight", weight if weight is None else weight.clone().float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        target = target.long()
        valid = target != self.ignore_index
        # Clamp ignored positions to a real class so one_hot stays in range;
        # they are masked out of both numerator and denominator below.
        safe = torch.where(valid, target, torch.zeros_like(target))

        logp = F.log_softmax(logits, dim=1)
        onehot = F.one_hot(safe, num_classes)
        # (B, H, W, C) -> (B, C, H, W); generalises to any spatial rank.
        dims = (0, safe.dim()) + tuple(range(1, safe.dim()))
        onehot = onehot.permute(*dims).to(logp.dtype)

        nll = -(logp * onehot).sum(dim=1)

        if self.weight is not None:
            w = self.weight.to(logits.dtype)[safe]
            nll = nll * w
            denom = (w * valid).sum()
        else:
            denom = valid.sum().to(nll.dtype)

        nll = nll * valid
        return nll.sum() / denom.clamp_min(1)


def make_deterministic(module: nn.Module) -> nn.Module:
    """Recursively swap every ``nn.CrossEntropyLoss`` for the deterministic one.

    Returns the replacement when *module* is itself a CrossEntropyLoss, so
    callers can use the return value rather than relying on mutation.
    """
    if isinstance(module, nn.CrossEntropyLoss):
        return DeterministicCrossEntropyLoss(
            ignore_index=module.ignore_index, weight=module.weight
        )
    for name, child in list(module.named_children()):
        setattr(module, name, make_deterministic(child))
    return module
