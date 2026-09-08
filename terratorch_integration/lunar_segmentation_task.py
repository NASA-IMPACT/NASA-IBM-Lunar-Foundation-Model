"""Custom semantic-segmentation tasks.

`LunarSegmentationTask` — LLRD + split encoder/decoder LR
------------------------------------------------------------
Inherits the shared LLRD + dual-LR + PEFT-fix recipe from :class:`_LunarLLRDMixin`:

* **backbone_lr** — base LR for the top encoder layer; deeper layers get
  `backbone_lr * layer_decay ** k` (LLRD).
* **head_lr** — LR for the decoder, segmentation head, and auxiliary heads.
* **layer_decay** — per-layer decay factor inside the encoder. 1.0 = flat.
* Warmup + cosine annealing schedule.
* Bias / norm / positional params always skip weight decay.

See :mod:`terratorch_integration.lunar_llrd_mixin` for the full recipe.

`LunarShapeSegmentationTask` — per-crater roundness regularizer
-----------------------------------------------------------------
Inherits from `LunarSegmentationTask` and adds an auxiliary shape-compactness
loss that penalises non-circular predicted crater blobs. The regularizer
operates per bounding-box ROI (bboxes passed through the batch under the
`"crater_boxes"` key by the `LunarNACDTMDataset` when constructed with
`keep_boxes=True`), so it is not confused by multiple craters in the same
image.

Shape term (per bbox ROI)::
    perimeter = Σ √(dx² + dy²)                (L2 gradient magnitude)
    compactness = 4π · area / perimeter²      ∈ (0, 1], =1 for a disk
    L_shape     = clamp(1 - compactness, min=0)

Turn the regularizer off by setting `shape_loss_weight: 0` — the class then
behaves exactly like `LunarSegmentationTask`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from terratorch.tasks import SemanticSegmentationTask
from terratorch.tasks.segmentation_tasks import to_segmentation_prediction

from .lunar_llrd_mixin import (
    _LunarLLRDMixin,
    _encoder_block_index,
    _find_lunar_backbone,
    _fix_frozen_new_modality_embedders_after_peft,
    _has_active_peft_adapters,
    _is_encoder_param,
    _is_new_modality_param,
    _no_decay,
)

# Backward-compatible re-exports.
__all__ = [
    "LunarSegmentationTask",
    "LunarShapeSegmentationTask",
    "crater_shape_loss",
    # Helpers (re-exported for back-compat)
    "_encoder_block_index",
    "_find_lunar_backbone",
    "_fix_frozen_new_modality_embedders_after_peft",
    "_has_active_peft_adapters",
    "_is_encoder_param",
    "_is_new_modality_param",
    "_no_decay",
]


class LunarSegmentationTask(_LunarLLRDMixin, SemanticSegmentationTask):
    """Segmentation task with LLRD and split encoder/decoder param groups.

    All optimiser knobs (`backbone_lr`, `head_lr`, `layer_decay`,
    `weight_decay`, `head_weight_decay`, `warmup_steps`, `cosine_t_max`,
    `eta_min`, `betas`) come from :class:`_LunarLLRDMixin`. Any other
    keyword argument is forwarded to
    :class:`~terratorch.tasks.SemanticSegmentationTask`.
    """


def crater_shape_loss(
    prob: Tensor,
    boxes_per_image: list[Tensor],
    *,
    pad_frac: float = 0.3,
    min_side: int = 4,
    eps: float = 1e-6,
) -> Tensor:
    """Per-crater compactness regularizer.

    Args:
        prob: `(B, H, W)` sigmoid/softmax probability of the crater class.
        boxes_per_image: list of `(N_i, 4)` tensors in `[x1, y1, x2, y2]` image-frame pixel
            coords (same frame as `prob`).
        pad_frac: How much to enlarge each bbox before cropping the ROI.
            Padding lets the perimeter term "see" the crater edge.
        min_side: Skip ROIs smaller than this in either dimension.
        eps: Small constant to avoid divide-by-zero when a crater is entirely absent from the prediction.

    Returns:
        Scalar tensor: mean per-crater `(1 - compactness)`, in `[0, 1)`.
        Zero when the prediction is a perfect disk; ~1 for very elongated
        blobs.  Returns `prob.new_tensor(0.0)` if no valid ROIs exist in the
        batch (e.g. every image has no craters).
    """
    if prob.dim() != 3:
        raise ValueError(f"prob must be (B, H, W), got {tuple(prob.shape)}")
    B, H, W = prob.shape
    if len(boxes_per_image) != B:
        raise ValueError(f"boxes_per_image has length {len(boxes_per_image)}, but batch dim is {B}")

    total = prob.new_tensor(0.0)
    count = 0
    for b, bxs in enumerate(boxes_per_image):
        if bxs is None or bxs.numel() == 0:
            continue
        for row in bxs.tolist():
            x1, y1, x2, y2 = row[:4]
            cw = max(x2 - x1, 0.0)
            ch = max(y2 - y1, 0.0)
            if cw < 1.0 or ch < 1.0:
                continue
            X1 = max(0, int(x1 - pad_frac * cw))
            Y1 = max(0, int(y1 - pad_frac * ch))
            X2 = min(W, int(x2 + pad_frac * cw))
            Y2 = min(H, int(y2 + pad_frac * ch))
            if (X2 - X1) < min_side or (Y2 - Y1) < min_side:
                continue
            patch = prob[b, Y1:Y2, X1:X2]
            area = patch.sum() + eps
            # Anti-alias the boundary with a light 2x2 mean before the gradient:
            # a plain discrete grid double-counts diagonal edges vs axis-aligned ones
            # (making squares score lower than disks otherwise), and this small blur equalises the two.
            p = patch.unsqueeze(0).unsqueeze(0)
            p = torch.nn.functional.avg_pool2d(p, kernel_size=2, stride=1, padding=0)
            p = torch.nn.functional.pad(p, (0, 1, 0, 1))
            p = p.squeeze(0).squeeze(0)
            dx = p[:-1, 1:] - p[:-1, :-1]
            dy = p[1:, :-1] - p[:-1, :-1]
            perim = torch.sqrt(dx * dx + dy * dy + eps).sum()
            compactness = 4.0 * math.pi * area / (perim * perim + eps)
            total = total + (1.0 - compactness).clamp(min=0.0)
            count += 1

    if count == 0:
        return total  # zero, same device/dtype as prob
    return total / count


class LunarShapeSegmentationTask(LunarSegmentationTask):
    """Segmentation task with LLRD **and** a per-crater roundness loss.

    Combines the encoder/decoder LR split from :class:`LunarSegmentationTask`
    with a shape-compactness regularizer.

    Extra init args:
        shape_loss_weight: Multiplier for the shape regularizer. Set to 0 to disable (recommended baseline: `0.05`;
            scan `{0.02, 0.05, 0.1, 0.2}` if needed).
        shape_loss_pad_frac: ROI padding around each bbox, as a fraction of the bbox side.
        crater_boxes_key: Batch key holding the per-image list of bboxes.
            Must match `LunarNACDTMDataset(keep_boxes=True)`.
        crater_class_index: Index of the crater class in the softmax output. Defaults to 1 (background=0, crater=1).
    """

    def __init__(
        self,
        *args: Any,
        shape_loss_weight: float = 0.05,
        shape_loss_pad_frac: float = 0.3,
        crater_boxes_key: str = "crater_boxes",
        crater_class_index: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shape_loss_weight = float(shape_loss_weight)
        self.shape_loss_pad_frac = float(shape_loss_pad_frac)
        self.crater_boxes_key = crater_boxes_key
        self.crater_class_index = int(crater_class_index)


    def _pop_crater_boxes(self, batch: dict[str, Any]) -> list[Tensor] | None:
        """Detach crater bboxes from the batch so they're not forwarded into the model as kwargs."""
        return batch.pop(self.crater_boxes_key, None)

    def _add_shape_loss(
        self,
        base_loss: Tensor,
        model_output: Any,
        crater_boxes: list[Tensor] | None,
        stage: str,
    ) -> Tensor:
        """Add the shape regularizer to `base_loss`.
        Logs the component under `{stage}/shape_loss`. No-op when the weight is 0 or when
        the batch contains no bboxes.
        """
        if self.shape_loss_weight == 0.0 or not crater_boxes:
            return base_loss
        logits = model_output.output  # (B, C, H, W)
        prob = logits.softmax(dim=1)[:, self.crater_class_index]  # (B, H, W)
        shape_l = crater_shape_loss(
            prob,
            crater_boxes,
            pad_frac=self.shape_loss_pad_frac,
        )
        self.log(f"{stage}/shape_loss", shape_l, prog_bar=(stage == "train"))
        return base_loss + self.shape_loss_weight * shape_l


    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        crater_boxes = self._pop_crater_boxes(batch)
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self(x, **rest)
        loss = self.train_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        loss["loss"] = self._add_shape_loss(loss["loss"], model_output, crater_boxes, "train")
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)
        return loss["loss"]

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        crater_boxes = self._pop_crater_boxes(batch)
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {k: batch[k] for k in other_keys}
        model_output = self.handle_full_or_tiled_inference(
            x, self.tiled_inference_on_validation, **rest
        )
        loss = self.val_loss_handler.compute_loss(
            model_output, y, self.criterion, self.aux_loss
        )
        loss["loss"] = self._add_shape_loss(loss["loss"], model_output, crater_boxes, "val")
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=y.shape[0])
        y_hat_hard = to_segmentation_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)
        if self._do_plot_samples(batch_idx):
            batch["prediction"] = y_hat_hard
            self.plot_sample(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # No shape loss at test time; just make sure the extra key doesn't
        # get forwarded into the model.
        self._pop_crater_boxes(batch)
        return super().test_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._pop_crater_boxes(batch)
        return super().predict_step(batch, batch_idx, dataloader_idx)
