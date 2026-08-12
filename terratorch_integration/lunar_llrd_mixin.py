"""LLRD + split-group optimiser mixin, reused by every Lunar-FM Lightning task
that fine-tunes a `LunarBackbone` encoder plus a fresh head
(segmentation, pixel regression, classification, scalar regression).

Layer-wise LR Decay (LLRD) + dual LR
------------------------------------
Each trainable parameter lands in one of three groups:

* new-modality embedder  →  `head_lr`, `head_weight_decay`
* pretrained encoder     →  `backbone_lr * layer_decay ** depth_from_top`
* head / decoder / neck  →  `head_lr`, `head_weight_decay`

Norms / biases / register tokens / positional embeddings always skip weight
decay regardless of which group they end up in.

`LunarObjectDetectionTask` uses a different backbone-vs-head classifier
(`torchvision_model.backbone.backbone.model.` prefix, because Faster R-CNN
nests the LunarBackbone one level deeper) and keeps its own inline copy of the
same recipe.

Usage
-----
Compose the mixin *before* the TerraTorch base task in the MRO so its
`__init__` / `configure_optimizers` / `configure_models` /
`on_load_checkpoint` overrides win:

    class LunarSegmentationTask(_LunarLLRDMixin, SemanticSegmentationTask):
        ...

Config note
-----------
Do **not** set top-level `optimizer:` / `lr_scheduler:` blocks in the YAML —
LightningCLI will monkey-patch `configure_optimizers` and silently drop the
LLRD param groups. Pass the LR knobs as task init args (`backbone_lr`,
`head_lr`, `layer_decay`, …).
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


_NO_DECAY_TOKENS = (
    "bias",
    "norm.weight",
    "layernorm.weight",
    "ln.weight",
    "register_tokens",
    "pos_emb",
    "pos_embed",
    "cls_token",
    "mod_emb",
)


def _no_decay(name: str) -> bool:
    lname = name.lower()
    return any(tok in lname for tok in _NO_DECAY_TOKENS)


def _encoder_block_index(name: str) -> int | None:
    """Return the integer block index if *name* sits inside an encoder block."""
    for key in (".model.encoder.", ".blocks.", ".layers."):
        idx = name.find(key)
        if idx < 0:
            continue
        tail = name[idx + len(key):]
        first = tail.split(".", 1)[0]
        if first.isdigit():
            return int(first)
    return None


def _is_encoder_param(name: str) -> bool:
    """True for parameters that belong to the backbone/encoder."""
    return name.startswith("encoder.")


def _is_new_modality_param(name: str, new_modality_names: list[str]) -> bool:
    """True for encoder-embedding params that belong to a *new* (randomly
    initialized) modality — one that was not present in the pretrained
    checkpoint.

    These live under one of the backbone-specific prefixes:
      - LunarBackbone / NILunarFM: `encoder.model.encoder_embeddings.<mod>.`

    They must be trained at `head_lr` rather than the tiny LLRD layer-0 LR,
    because they start from random init and need to converge at the same rate
    as the freshly initialized decoder/head.
    """
    if not new_modality_names:
        return False
    return any(
        f"encoder_embeddings.{mod}." in name or f"input_adapters.{mod}." in name
        for mod in new_modality_names
    )


def _find_lunar_backbone(encoder: Any) -> Any | None:
    """Return the underlying LunarBackbone if one is reachable.

    After `EncoderDecoderFactory` wraps a backbone with PEFT, the task sees
    `self.model.encoder` as a `PeftModel`; the original backbone lives at
    `encoder.base_model.model`. In the OD path we wrap the *inner*
    `LunarBackbone.model` (NILunarFM) instead, so the backbone still sits at
    the top and has a `PeftModel` under its `.model`. Handle both.
    """
    if hasattr(encoder, "_new_modalities"):
        return encoder
    inner = getattr(encoder, "base_model", None)
    if inner is not None:
        inner_model = getattr(inner, "model", None)
        if inner_model is not None and hasattr(inner_model, "_new_modalities"):
            return inner_model
    return None


def _has_active_peft_adapters(encoder: Any) -> bool:
    """True if `encoder` contains any trainable LoRA (or similar PEFT) param.

    Used to distinguish PEFT-induced freezing (bug we want to fix) from an
    intentional `freeze_backbone: true` (which we must leave alone).
    """
    for name, p in encoder.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if "lora_a" in lname or "lora_b" in lname or "adapter" in lname:
            return True
    return False


def _fix_frozen_new_modality_embedders_after_peft(encoder: Any) -> int:
    """Re-enable `requires_grad` on new-modality embedders that PEFT froze.

    `peft.get_peft_model` calls `mark_only_adapters_as_trainable` internally,
    which flips `requires_grad=False` on **every** non-adapter param —
    including freshly-initialized new-modality embedders
    (`encoder_embeddings.<mod>.*` on `LunarBackbone`). Those weights start
    from random init and MUST train, otherwise the new modality is dead weight.

    No-op when PEFT was not applied (nothing was frozen by us) or when the
    backbone declares no new modalities. Safe to call from any task's
    `configure_models` override — idempotent.

    Returns the number of params whose `requires_grad` was flipped back on.
    """
    bb = _find_lunar_backbone(encoder)
    if bb is None:
        return 0
    new_mods = getattr(bb, "_new_modalities", None) or {}
    if not new_mods:
        return 0
    if not _has_active_peft_adapters(encoder):
        # No PEFT wrap detected — the embedders are either already trainable
        # or intentionally frozen (e.g. freeze_backbone). Don't override.
        return 0
    new_mod_names = list(new_mods.keys())
    count = 0
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            continue
        for m in new_mod_names:
            if f"encoder_embeddings.{m}." in name or f"input_adapters.{m}." in name:
                p.requires_grad_(True)
                count += 1
                break
    if count > 0:
        print(
            f"[terratorch_integration] PEFT+new-modality fix: re-enabled "
            f"requires_grad on {count} new-modality embedder param(s) for "
            f"modalities {new_mod_names}"
        )
    return count


class _LunarLLRDMixin:
    """See module docstring for details."""

    # Filled in by __init__; declared here so type-checkers know they exist.
    backbone_lr: float
    head_lr: float
    layer_decay: float
    weight_decay: float
    head_weight_decay: float
    warmup_steps: int
    cosine_t_max: int | None
    eta_min: float
    betas: tuple[float, float]

    def __init__(
        self,
        *args: Any,
        backbone_lr: float = 5.0e-5,
        head_lr: float = 2.0e-4,
        layer_decay: float = 0.75,
        weight_decay: float = 0.05,
        head_weight_decay: float | None = None,
        warmup_steps: int = 500,
        cosine_t_max: int | None = None,
        eta_min: float = 1.0e-6,
        betas: tuple[float, float] = (0.9, 0.98),
        **kwargs: Any,
    ) -> None:
        # Disable base-class optimiser/scheduler wiring — our
        # configure_optimizers below runs the show. Without this, LightningCLI's
        # default optimiser path would silently win if a top-level
        # `optimizer:` block leaked into the YAML.
        kwargs.setdefault("optimizer", None)
        kwargs.setdefault("optimizer_hparams", None)
        kwargs.setdefault("scheduler", None)
        kwargs.setdefault("scheduler_hparams", None)
        super().__init__(*args, **kwargs)

        self.backbone_lr = float(backbone_lr)
        self.head_lr = float(head_lr)
        self.layer_decay = float(layer_decay)
        self.weight_decay = float(weight_decay)
        self.head_weight_decay = (
            float(head_weight_decay) if head_weight_decay is not None else float(weight_decay)
        )
        self.warmup_steps = int(warmup_steps)
        self.cosine_t_max = cosine_t_max
        self.eta_min = float(eta_min)
        self.betas = tuple(betas)

    # ------------------------------------------------------------------ utils

    def configure_models(self) -> None:
        super().configure_models()
        # Undo PEFT's freeze of new-modality embedders (see
        # `_fix_frozen_new_modality_embedders_after_peft` docstring).
        _fix_frozen_new_modality_embedders_after_peft(self.model.encoder)

    def _get_new_modality_names(self) -> list[str]:
        """Modality names that were NOT in the pretrained checkpoint."""
        enc = self.model.encoder
        new_mods = getattr(enc, "_new_modalities", None) or {}
        return list(new_mods.keys())

    def _num_encoder_blocks(self) -> int:
        """Number of transformer blocks in the encoder.

        Falls back to counting `model.encoder.blocks` (timm-style) then
        `model.encoder.model.encoder` (Lunar FM style), then a param-name scan,
        then 12.
        """
        enc = self.model.encoder
        if hasattr(enc, "blocks"):
            return len(enc.blocks)
        if hasattr(enc, "model") and hasattr(enc.model, "encoder"):
            return len(enc.model.encoder)
        max_idx = -1
        for name, _ in self.model.named_parameters():
            idx = _encoder_block_index(name)
            if idx is not None and idx > max_idx:
                max_idx = idx
        return max_idx + 1 if max_idx >= 0 else 12

    def _param_layer_id(self, name: str, num_blocks: int) -> int:
        """`0` = embeddings / register / patch; `1..N` = encoder blocks
        (1 = deepest, N = shallowest); `N+1` = encoder_norm."""
        blk = _encoder_block_index(name)
        if blk is not None:
            return blk + 1
        if "encoder_norm" in name:
            return num_blocks + 1
        return 0

    def _make_param_groups(self) -> list[dict[str, Any]]:
        num_blocks = self._num_encoder_blocks()
        top_layer = num_blocks + 1
        new_mod_names = self._get_new_modality_names()
        groups: dict[str, dict[str, Any]] = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            in_encoder = _is_encoder_param(name)
            nd = _no_decay(name)

            if in_encoder and _is_new_modality_param(name, new_mod_names):
                group_name = f"new_mod_emb_{'nd' if nd else 'wd'}"
                lr = self.head_lr
                wd = 0.0 if nd else self.head_weight_decay
            elif in_encoder:
                layer_id = self._param_layer_id(name, num_blocks)
                scale = self.layer_decay ** (top_layer - layer_id)
                lr = self.backbone_lr * scale
                wd = 0.0 if nd else self.weight_decay
                group_name = f"encoder_layer_{layer_id}_{'nd' if nd else 'wd'}"
            else:
                lr = self.head_lr
                wd = 0.0 if nd else self.head_weight_decay
                group_name = f"head_{'nd' if nd else 'wd'}"

            g = groups.setdefault(
                group_name,
                {"params": [], "lr": lr, "weight_decay": wd, "name": group_name},
            )
            g["params"].append(p)

        def _sort_key(g: dict) -> tuple:
            n = g["name"]
            if n.startswith("encoder_layer_"):
                try:
                    layer_num = int(n.split("_")[2])
                except (IndexError, ValueError):
                    layer_num = -1
                return (0, layer_num, n)
            return (1, 0, n)

        return sorted(groups.values(), key=_sort_key)

    # ---------------------------------------------------------- optim/sched

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Keep saved optimizer / scheduler state when this run rebuilds the same param groups;
        otherwise strip them and raise an explicit error.

        Compatible resume needs the optimizer / LR-scheduler state, or training effectively
        restarts from the loaded weights.

        Incompatible resume (new modality added, LoRA target modules changed, ...) triggers
        `ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group`
        inside `optimizer.load_state_dict`.
        """
        saved_pgs = (
            checkpoint.get("optimizer_states", [{}])[0].get("param_groups", [])
        )
        if not saved_pgs:
            return

        fresh_pgs = self._make_param_groups()

        def _sig(pg: dict) -> tuple:
            return (pg.get("name", "?"), len(pg.get("params", [])))

        saved_sig = [_sig(pg) for pg in saved_pgs]
        fresh_sig = [_sig(pg) for pg in fresh_pgs]

        if saved_sig != fresh_sig:
            checkpoint.pop("optimizer_states", None)
            checkpoint.pop("lr_schedulers", None)
            raise RuntimeError(
                "Cannot resume: param-group layout differs between this run "
                "and the checkpoint. "
                f"fresh={fresh_sig} checkpoint={saved_sig}. "
                "Restart training from scratch, or load only the weights via "
                "a separate init path instead of --ckpt_path."
            )

    def configure_optimizers(self):
        param_groups = self._make_param_groups()
        n_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
        cls_name = type(self).__name__
        print(
            f"[{cls_name}] {len(param_groups)} param groups, "
            f"{n_params / 1e6:.1f}M trainable params. "
            f"layer_decay={self.layer_decay} backbone_lr={self.backbone_lr} "
            f"head_lr={self.head_lr} "
            f"new_modalities={self._get_new_modality_names() or 'none'}"
        )
        for g in param_groups:
            print(
                f"  {g['name']:<36s} lr={g['lr']:.2e} wd={g['weight_decay']:.2e} "
                f"n={sum(p.numel() for p in g['params']):,}"
            )

        optimizer = torch.optim.AdamW(param_groups, betas=self.betas)

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 0
        t_max = self.cosine_t_max
        if t_max is None:
            t_max = max(1, total_steps - self.warmup_steps) if total_steps else 10_000

        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.eta_min)
        if self.warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1.0 / max(self.warmup_steps, 1),
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
