"""Make training runs bit-reproducible.

Two things have to hold, and neither implies the other:

1. ``trainer.deterministic: true`` — sets torch's global flag and moves cuDNN
   off its nondeterministic kernels.
2. Seeded augmentation — this module.

``lightning.seed_everything`` does **not** control albumentations >= 2.0.
``BasicTransform.__init__`` hardcodes ``self.seed = None`` and calls
``set_random_seed(None)``, which builds ``np.random.default_rng(None)`` — seeded
from OS entropy. Each transform therefore carries a private RNG that is
re-randomised on every process start, no matter what the global seed is.

TerraTorch compounds this: ``wrap_in_compose_is_list`` builds
``A.Compose(transforms, is_check_shapes=False)`` with no ``seed`` argument.

This is separate from, and larger than, GPU kernel nondeterminism — it changes
which augmentations each sample receives, so runs diverge even under
``deterministic: true``.

Usage — add the callback to any config, alongside `deterministic: true`::

    trainer:
      deterministic: true
      benchmark: false
      callbacks:
        - class_path: terratorch_integration.DeterministicAugmentation

Deterministic runs also need ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the
environment, or torch raises on the first cuBLAS GEMM.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from lightning.pytorch import Callback

from .deterministic_losses import make_deterministic


def seed_albumentations(obj: object, seed: int, _depth: int = 0) -> int:
    """Recursively seed every albumentations transform reachable from *obj*.

    Returns the number of transform trees seeded. ``A.Compose.set_random_seed``
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

    Args:
        seed: Seed to apply. Defaults to the process-wide seed Lightning
            recorded in ``PL_GLOBAL_SEED`` (i.e. ``--seed_everything``), so a
            run reproduces without repeating the seed in two places.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed

    def setup(self, trainer, pl_module, stage=None) -> None:  # noqa: D102
        seed = self.seed
        if seed is None:
            seed = int(os.environ.get("PL_GLOBAL_SEED", 0))
        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return
        n = seed_albumentations(dm, int(seed))
        trainer.print(f"DeterministicAugmentation: seeded {n} pipeline(s) with seed={seed}")


# ---------------------------------------------------------------------------
# Deterministic adaptive average pooling
# ---------------------------------------------------------------------------
# `adaptive_avg_pool2d_backward_cuda` has no deterministic implementation, so
# UperNetDecoder's pyramid-pooling module cannot run under `deterministic:
# true`. Adaptive average pooling is separable and linear, so it is exactly
# reproducible from slices and means, whose backward IS deterministic.


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """Drop-in for ``nn.AdaptiveAvgPool2d`` built from deterministic ops."""

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
        # Slice + mean + stack, applied separably. Deliberately NOT matmul or
        # einsum: a broadcasted matmul's backward dispatches to a Triton kernel
        # needing a JIT toolchain the compute nodes may not have. Slicing and
        # mean are eager, deterministic and dependency-free.
        rows = [x[..., s:e, :].mean(dim=-2) for s, e in self._windows(in_h, out_h)]
        y = torch.stack(rows, dim=-2)
        cols = [y[..., s:e].mean(dim=-1) for s, e in self._windows(in_w, out_w)]
        return torch.stack(cols, dim=-1)


def replace_adaptive_pool(module: nn.Module) -> nn.Module:
    """Recursively swap every ``nn.AdaptiveAvgPool2d`` for the deterministic one.

    A no-op for models that contain no adaptive pooling.
    """
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return DeterministicAdaptiveAvgPool2d(module.output_size)
    for name, child in list(module.named_children()):
        setattr(module, name, replace_adaptive_pool(child))
    return module


class DeterministicLoss(Callback):
    """Swap ``nn.CrossEntropyLoss`` for a deterministic equivalent.

    ``nll_loss2d_forward_out_cuda_template`` has no deterministic kernel, so a
    loss containing a ``ce`` term raises under ``deterministic: true``. `ce` is
    also the only term TerraTorch's ``init_loss`` feeds ``class_weights`` to, so
    dropping it silently drops class weighting -- which matters on imbalanced
    masks. This keeps both.

    A callback rather than a task override, so it also covers tasks built by
    ``SMPModelFactory``, which use TerraTorch's stock SemanticSegmentationTask.
    A no-op unless torch determinism is on.
    """

    def setup(self, trainer, pl_module, stage=None) -> None:  # noqa: D102
        if not torch.are_deterministic_algorithms_enabled():
            return
        crit = getattr(pl_module, "criterion", None)
        if crit is None:
            return
        pl_module.criterion = make_deterministic(crit)
        trainer.print("DeterministicLoss: cross-entropy terms made deterministic")
