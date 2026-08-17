"""Make training runs reproducible.

For reproducibility, these have to hold:

1. `trainer.deterministic: true` — sets torch's global flag and moves cuDNN
   off its nondeterministic kernels.
   Issue - Some ops have no deterministic implementations:
   an error is thrown when trying to use those ops with "deterministic: true".
2. Seeded augmentation.
   Issue: `lightning.seed_everything` does **not** control albumentations >= 2.0.

This module creates:
- callback to seed albumentations pipelines
- deterministic implementation for nn.AdaptiveAvgPool2d
- callback to replace nn.CrossEntropyLoss with a deterministic implementation

"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info


def seed_albumentations(obj: object, seed: int, _depth: int = 0) -> int:
    """Recursively seed every albumentations transform reachable from *obj*.

    Returns the number of transform trees seeded. `A.Compose.set_random_seed`
    propagates to the transforms it holds, so seeding the Compose is enough.
    """
    if _depth > 3 or obj is None:
        return 0
    n = 0
    if hasattr(obj, "set_random_seed") and callable(obj.set_random_seed):
        obj.set_random_seed(seed)
        return 1
    values = (
        obj.values()
        if isinstance(obj, dict)
        else (obj if isinstance(obj, (list, tuple)) else getattr(obj, "__dict__", {}).values())
    )
    for v in values:
        if isinstance(v, (str, bytes, int, float, bool)):
            continue
        n += seed_albumentations(v, seed, _depth + 1)
    return n


class DeterministicAugmentation(Callback):
    """Seed the datamodule's albumentations pipelines so runs reproduce.

    Usage — add the callback to any config, alongside `deterministic: true`:

    trainer:
      deterministic: true
      benchmark: false
      callbacks:
        - class_path: terratorch_integration.DeterministicAugmentation

    Args:
        seed: Seed to apply. Defaults to the process-wide seed Lightning recorded in `PL_GLOBAL_SEED`
            (i.e. `--seed_everything`), so a run reproduces without repeating the seed in two places.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed

    def setup(self, trainer, pl_module, stage=None) -> None:
        seed = self.seed
        if seed is None:
            seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return
        n = seed_albumentations(dm, int(seed))
        trainer.print(f"DeterministicAugmentation: seeded {n} pipeline(s) with seed={seed}")


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Drop-in for `nn.AdaptiveAvgPool2d` built from deterministic ops.

    `adaptive_avg_pool2d_backward_cuda` has no deterministic implementation, so
    UperNetDecoder's pyramid-pooling module cannot run under `deterministic: true`.
    Adaptive average pooling is separable and linear, so it is exactly reproducible from slices and means,
    whose backward IS deterministic.
    """

    def __init__(self, output_size) -> None:
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)

    @staticmethod
    def _windows(in_size: int, out_size: int):
        """Adaptive pooling window bounds, matching PyTorch's definition."""
        return [
            ((i * in_size) // out_size, -(-((i + 1) * in_size) // out_size))
            for i in range(out_size)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2:]
        out_h = in_h if self.output_size[0] is None else self.output_size[0]
        out_w = in_w if self.output_size[1] is None else self.output_size[1]
        # Slice + mean + stack, applied separably.
        rows = [x[..., s:e, :].mean(dim=-2) for s, e in self._windows(in_h, out_h)]
        y = torch.stack(rows, dim=-2)
        cols = [y[..., s:e].mean(dim=-1) for s, e in self._windows(in_w, out_w)]
        return torch.stack(cols, dim=-1)


def replace_adaptive_pool(module: nn.Module) -> nn.Module:
    """Recursively swap every `nn.AdaptiveAvgPool2d` for the deterministic one.

    A no-op for models that contain no adaptive pooling.
    """
    n = sum(1 for m in module.modules() if isinstance(m, nn.AdaptiveAvgPool2d))
    module = _replace_adaptive_pool(module)
    if n:
        rank_zero_info(f"DeterministicAdaptiveAvgPool2d: replaced {n} pool(s)")
    return module


def _replace_adaptive_pool(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return DeterministicAdaptiveAvgPool2d(module.output_size)
    for name, child in list(module.named_children()):
        setattr(module, name, _replace_adaptive_pool(child))
    return module


class DeterministicLoss(Callback):
    """Swap `nn.CrossEntropyLoss` for a deterministic equivalent.

    `nll_loss2d_forward_out_cuda_template` has no deterministic kernel, so a
    loss containing a `ce` term raises under `deterministic: true`.
    A no-op unless torch determinism is on.
    """

    def setup(self, trainer, pl_module, stage=None) -> None:
        if not torch.are_deterministic_algorithms_enabled():
            return
        crit = getattr(pl_module, "criterion", None)
        if crit is None:
            return
        pl_module.criterion = make_deterministic(crit)
        trainer.print("DeterministicLoss: cross-entropy terms made deterministic")


class DeterministicCrossEntropyLoss(nn.Module):
    """`nn.CrossEntropyLoss(reduction="mean")` without nondeterministic ops.

    Args:
        ignore_index: Target value to exclude from the loss and from the
            normalising denominator.
        weight: Optional per-class weights, as in `nn.CrossEntropyLoss`.
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
    """Recursively swap every `nn.CrossEntropyLoss` for the deterministic one.

    Returns the replacement when *module* is itself a CrossEntropyLoss, so
    callers can use the return value rather than relying on mutation.
    """
    if isinstance(module, nn.CrossEntropyLoss):
        return DeterministicCrossEntropyLoss(ignore_index=module.ignore_index, weight=module.weight)
    for name, child in list(module.named_children()):
        setattr(module, name, make_deterministic(child))
    return module
