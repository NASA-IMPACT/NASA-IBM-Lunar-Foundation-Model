"""ObjectDetectionTask with layer-wise LR decay (LLRD) and split param groups.

LLRD decays LR by depth inside the backbone, so bottom encoder layers (generic features) train slower
than top ones (task-specific features). Recipe follows ViTDet / BEiT / MAE-detection.

Config-level notes:
- Do NOT set top-level `optimizer:` / `lr_scheduler:` in the Lightning
  CLI config. Those monkey-patch `configure_optimizers` and would clobber
  the LLRD groups. Configure them via task init args instead.
- Params matching `no_decay_names` (biases, norms, register tokens,
  pos embeddings) skip weight decay regardless of their LR group.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torchgeo.datasets import RGBBandsMissingError, unbind_samples

from terratorch.tasks import ObjectDetectionTask
from terratorch.tasks.object_detection_task import get_batch_size
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

from .lunar_llrd_mixin import (
    _encoder_block_index as _mixin_encoder_block_index,
    _find_lunar_backbone,
    _is_new_modality_param,
)


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


# Block-index detection is shared with the non-OD tasks rather than duplicated:
# Layouts handled by the shared helper:
#   `...model.encoder.<i>.<rest>`         ni_lfm_v1_* (NILunarFM)
#   `...blocks.<i>.<rest>`                timm ViT, and FVQ's nested `model.encoder.blocks.<i>`
#   `...layers.<i>.<rest>`                generic
_encoder_block_index = _mixin_encoder_block_index


def _is_backbone_param(name: str) -> bool:
    # Module tree with Faster-R-CNN:
    #   task.model = ObjectDetectionModel
    #     .torchvision_model = FasterRCNN
    #       .backbone = BackboneWrapper
    #         .backbone = LunarBackbone
    #           .model = NILunarFM (encoder, embeddings, register_tokens, ...)
    # Everything under `torchvision_model.backbone.backbone.model.` is the
    # pretrained encoder + its embeddings. The FPN/neck sits at
    # `torchvision_model.backbone.necks.*` and is treated as a head param.
    return "torchvision_model.backbone.backbone.model." in name


class LunarObjectDetectionTask(ObjectDetectionTask):
    """Object detection task with LLRD and split backbone/head param groups.

    Args:
        backbone_lr: Base LR for the top backbone layer (encoder_norm + last block).
            Deeper blocks get `backbone_lr * layer_decay ** k`.
        head_lr: LR for necks, RPN, ROI heads, and anything not inside the backbone.
        layer_decay: Per-layer LR decay factor. 1.0 disables LLRD.
        weight_decay: Weight decay applied to matrix params (norms, biases, and positional/register tokens
            are always excluded).
        warmup_steps: Linear warmup length in optimizer steps. 0 disables.
        cosine_t_max: Total steps for the cosine phase. If `None`, falls back to
            `trainer.estimated_stepping_batches - warmup_steps`.
        eta_min: Final LR floor for cosine.
        betas: AdamW betas.

    Everything else is forwarded to :class:`ObjectDetectionTask`.
    """

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
        anchor_sizes: list[list[int]] | None = None,
        anchor_aspect_ratios: list[float] | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        # Eval-time FasterRCNN caps. These bypass model_args entirely because
        # nested dict entries added post-training aren't reliably surfaced by
        # LightningCLI + jsonargparse in the --ckpt_path path (framework_* keys
        # not present at training time silently vanish from self.model_args).
        # As top-level init args they always take effect, both on `fit` and on
        # `test`, and can be overridden from the CLI:
        #   --model.init_args.eval_box_detections_per_img 600
        eval_box_detections_per_img: int | None = None,
        eval_box_score_thresh: float | None = None,
        eval_box_nms_thresh: float | None = None,
        eval_rpn_pre_nms_top_n_train: int | None = None,
        eval_rpn_post_nms_top_n_train: int | None = None,
        eval_rpn_pre_nms_top_n_test: int | None = None,
        eval_rpn_post_nms_top_n_test: int | None = None,
        eval_rpn_nms_thresh: float | None = None,
        peft_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Force base-class optimizer settings off — we build the optimizer ourselves.
        kwargs.setdefault("optimizer", None)
        kwargs.setdefault("optimizer_hparams", None)
        kwargs.setdefault("scheduler", None)
        kwargs.setdefault("scheduler_hparams", None)
        # These must be set BEFORE super().__init__().
        self.anchor_sizes = anchor_sizes
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.metric_kwargs: dict[str, Any] = dict(metric_kwargs) if metric_kwargs else {}
        # Must be set BEFORE super().__init__() — BaseTask calls
        # self.configure_models() which reads self.peft_config.
        self.peft_config: dict[str, Any] | None = dict(peft_config) if peft_config else None
        self._eval_caps: dict[str, Any] = {
            "box_detections_per_img": eval_box_detections_per_img,
            "box_score_thresh": eval_box_score_thresh,
            "box_nms_thresh": eval_box_nms_thresh,
            "rpn_pre_nms_top_n_train": eval_rpn_pre_nms_top_n_train,
            "rpn_post_nms_top_n_train": eval_rpn_post_nms_top_n_train,
            "rpn_pre_nms_top_n_test": eval_rpn_pre_nms_top_n_test,
            "rpn_post_nms_top_n_test": eval_rpn_post_nms_top_n_test,
            "rpn_nms_thresh": eval_rpn_nms_thresh,
        }
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

    def reformat_batch(self, batch: Any, batch_size: int):
        """Same as :meth:`ObjectDetectionTask.reformat_batch` but tolerates a per-sample mask list of length zero."""
        has_masks = (
            "masks" in batch or "mask" in batch or self.masks_field in batch
        )
        if not has_masks:
            return [
                {
                    "boxes": batch[self.boxes_field][i],
                    "labels": batch[self.labels_field][i],
                }
                for i in range(batch_size)
            ]

        # Pull the spatial size (H, W) from the collated image tensor so we
        # can build a correctly-shaped empty mask tensor for negative samples.
        img = batch.get("image")
        if isinstance(img, torch.Tensor) and img.ndim >= 3:
            spatial_h, spatial_w = int(img.shape[-2]), int(img.shape[-1])
        else:
            spatial_h = spatial_w = 0

        y: list[dict[str, torch.Tensor]] = []
        for i in range(batch_size):
            per_masks = batch[self.masks_field][i]
            if isinstance(per_masks, list) and len(per_masks) == 0:
                masks_t = torch.zeros(
                    (0, spatial_h, spatial_w), dtype=torch.uint8,
                )
            else:
                masks_t = torch.cat(
                    [x[None].to(torch.uint8) for x in per_masks]
                )
            y.append({
                "boxes": batch[self.boxes_field][i],
                "labels": batch[self.labels_field][i],
                "masks": masks_t,
            })
        return y

    def configure_models(self) -> None:
        super().configure_models()
        # PEFT / LoRA wrapping. `ObjectDetectionModelFactory` has no
        # `peft_config` hook, so plug PEFT in here — after the FasterRCNN is built.
        # This keeps `LunarBackbone.forward`, `.out_channels`, modality unpacking,
        # and new-modality embedders untouched — only encoder-block Linears get LoRA'd.
        if self.peft_config is not None:
            from terratorch.models.peft_utils import get_peft_backbone
            lunar_bb = self.model.torchvision_model.backbone.backbone
            if not hasattr(lunar_bb, "model"):
                raise AttributeError(
                    "peft_config expects the OD backbone to be a LunarBackbone "
                    "with a .model attribute (NILunarFM); got "
                    f"{type(lunar_bb).__name__}"
                )
            lunar_bb.model = get_peft_backbone(self.peft_config, lunar_bb.model)
            # PEFT's mark_only_adapters_as_trainable froze every non-adapter
            # param — including any freshly-initialized new-modality
            # embedders. Re-enable them so they can actually train.
            from .lunar_llrd_mixin import _fix_frozen_new_modality_embedders_after_peft
            _fix_frozen_new_modality_embedders_after_peft(lunar_bb)
            n_trainable = sum(p.numel() for p in lunar_bb.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in lunar_bb.model.parameters())
            print(
                f"[LunarObjectDetectionTask] applied PEFT: "
                f"{n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M "
                f"({100 * n_trainable / max(n_total, 1):.2f}%) encoder params trainable"
            )
        # Force-apply framework caps from model_args EVERY time the model is built.
        # `terratorch test --ckpt_path` restores `self.hparams` from the checkpoint, so `model_args` is frozen.
        # This override reads the live model_args (via self.model_args) and get the values onto the
        # already-built FasterRCNN.
        tvm = self.model.torchvision_model
        rh = tvm.roi_heads
        rpn = tvm.rpn
        caps = dict(self._eval_caps)
        # Env-var overrides. Lightning restores hparams (including init-args like eval_box_nms_thresh)
        # from the checkpoint on `test` / `validate`, so YAML edits and --model.init_args.<...> CLI flags
        # are both silently overridden. Env vars are read at model-build time and bypass every hparams-restore path.
        #   LUNAR_EVAL_BOX_NMS_THRESH, LUNAR_EVAL_BOX_SCORE_THRESH,
        #   LUNAR_EVAL_BOX_DETECTIONS_PER_IMG,
        #   LUNAR_EVAL_RPN_NMS_THRESH,
        #   LUNAR_EVAL_RPN_{PRE,POST}_NMS_TOP_N_{TRAIN,TEST}
        env_map = {
            "box_detections_per_img":    "LUNAR_EVAL_BOX_DETECTIONS_PER_IMG",
            "box_score_thresh":          "LUNAR_EVAL_BOX_SCORE_THRESH",
            "box_nms_thresh":            "LUNAR_EVAL_BOX_NMS_THRESH",
            "rpn_pre_nms_top_n_train":   "LUNAR_EVAL_RPN_PRE_NMS_TOP_N_TRAIN",
            "rpn_post_nms_top_n_train":  "LUNAR_EVAL_RPN_POST_NMS_TOP_N_TRAIN",
            "rpn_pre_nms_top_n_test":    "LUNAR_EVAL_RPN_PRE_NMS_TOP_N_TEST",
            "rpn_post_nms_top_n_test":   "LUNAR_EVAL_RPN_POST_NMS_TOP_N_TEST",
            "rpn_nms_thresh":            "LUNAR_EVAL_RPN_NMS_THRESH",
        }
        import os
        for key, env_name in env_map.items():
            env_val = os.environ.get(env_name)
            if env_val is not None:
                caps[key] = env_val
                print(f"[LunarObjectDetectionTask]  env override {env_name}={env_val}")
        # Task-level thresholds are also stashed in hparams; expose env
        # overrides so a --ckpt_path eval can retune NMS/score without retraining.
        env_iou = os.environ.get("LUNAR_EVAL_IOU_THRESHOLD")
        if env_iou is not None:
            self.iou_threshold = float(env_iou)
            print(f"[LunarObjectDetectionTask]  env override LUNAR_EVAL_IOU_THRESHOLD={env_iou}")
        env_score = os.environ.get("LUNAR_EVAL_SCORE_THRESHOLD")
        if env_score is not None:
            self.score_threshold = float(env_score)
            print(f"[LunarObjectDetectionTask]  env override LUNAR_EVAL_SCORE_THRESHOLD={env_score}")
        setters = [
            ("box_detections_per_img", lambda v: setattr(rh, "detections_per_img", int(v))),
            ("box_score_thresh", lambda v: setattr(rh, "score_thresh", float(v))),
            ("box_nms_thresh", lambda v: setattr(rh, "nms_thresh", float(v))),
            ("rpn_pre_nms_top_n_train", lambda v: rpn._pre_nms_top_n.__setitem__("training", int(v))),
            ("rpn_post_nms_top_n_train", lambda v: rpn._post_nms_top_n.__setitem__("training", int(v))),
            ("rpn_pre_nms_top_n_test", lambda v: rpn._pre_nms_top_n.__setitem__("testing", int(v))),
            ("rpn_post_nms_top_n_test", lambda v: rpn._post_nms_top_n.__setitem__("testing", int(v))),
            ("rpn_nms_thresh", lambda v: setattr(rpn, "nms_thresh", float(v))),
        ]
        for key, setter in setters:
            if caps.get(key) is not None:
                print(f"[LunarObjectDetectionTask]  applying eval_{key}={caps[key]}")
                setter(caps[key])

        try:
            tvm = self.model.torchvision_model
            rh = tvm.roi_heads
            rpn = tvm.rpn
            print(
                "[LunarObjectDetectionTask] FasterRCNN caps: "
                f"box_detections_per_img={rh.detections_per_img} "
                f"box_score_thresh={rh.score_thresh} "
                f"box_nms_thresh={rh.nms_thresh} "
                f"rpn_pre_nms_top_n_train={rpn._pre_nms_top_n['training']} "
                f"rpn_post_nms_top_n_train={rpn._post_nms_top_n['training']} "
                f"rpn_pre_nms_top_n_test={rpn._pre_nms_top_n['testing']} "
                f"rpn_post_nms_top_n_test={rpn._post_nms_top_n['testing']}"
            )
        except AttributeError:
            pass
        # Override the factory's hardcoded anchor generator. The upstream
        # ObjectDetectionModelFactory uses (32, 64, 128, 256, 512) — those
        # are too big for lunar craters (min diameter 5 px).
        if self.anchor_sizes is None:
            return
        aspect = tuple(self.anchor_aspect_ratios or (0.5, 1.0, 2.0))
        sizes = tuple(tuple(int(s) for s in level) for level in self.anchor_sizes)
        aspect_ratios = (aspect,) * len(sizes)
        rpn = self.model.torchvision_model.rpn
        new_anchor_gen = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
        rpn.anchor_generator = new_anchor_gen
        # Rebuild the head to match the new anchor count. Reuse the head's existing in_channels so this
        # remains agnostic to backbone/neck dims.
        old_head = rpn.head
        try:
            first_conv = next(m for m in old_head.conv.modules()
                              if isinstance(m, torch.nn.Conv2d))
            in_channels = first_conv.in_channels
        except (AttributeError, StopIteration):
            # Fallback: pull from cls_logits.
            in_channels = old_head.cls_logits.in_channels
        num_anchors = new_anchor_gen.num_anchors_per_location()[0]
        new_head = RPNHead(in_channels, num_anchors)
        new_head.to(next(old_head.parameters()).device)
        rpn.head = new_head
        print(
            f"[LunarObjectDetectionTask] overrode RPN anchors: "
            f"sizes={sizes} aspect_ratios={aspect} "
            f"num_anchors_per_location={num_anchors} "
            f"(rebuilt RPN head, in_channels={in_channels})"
        )

    def configure_metrics(self) -> None:
        # Upstream hardcodes MeanAveragePrecision(iou_type=..., average="macro") with no way to pass
        # torchmetrics kwargs. We expose the constructor via self.metric_kwargs so the same
        # `max_detection_thresholds`, `class_metrics`, etc. can be shared across configs for fair comparison.
        kwargs = {"average": "macro", **self.metric_kwargs}
        _has_masks = self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False)
        iou_type = ("bbox", "segm") if _has_masks else ("bbox",)
        metrics = MetricCollection(
            {"mAP": MeanAveragePrecision(iou_type=iou_type, **kwargs)}
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Upstream calls `self.val_metrics(y_hat, y)` on every batch, which
        # runs `MeanAveragePrecision.compute()` per step over the full accumulated state.
        # That allocates giant pairwise IoU tensors — memory grows through the epoch and,
        # because the metric never gets `.reset()`.
        # We only `.update()` here and compute+reset in `on_validation_epoch_end`.
        x = batch["image"]
        batch_size = get_batch_size(x)
        batch = self.apply_ignore_index(batch, self.ignore_index)
        y = self.reformat_batch(batch, batch_size)
        y_hat = self(x)
        if not isinstance(y_hat, dict):
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)

        if self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False):
            for i in range(len(y_hat)):
                if "masks" in y_hat[i] and y_hat[i]["masks"].shape[0] > 0:
                    y_hat[i]["masks"] = (y_hat[i]["masks"] > 0.5).squeeze(1).to(torch.uint8)

        self.val_metrics.update(y_hat, y)

        # Accumulate score-distribution stats so we can log per-epoch what
        # the model's confidence profile actually looks like.
        if not hasattr(self, "_val_score_bucket"):
            self._val_score_bucket = []
            self._val_pred_count_bucket = []
            self._val_gt_count_bucket = []
        for i in range(len(y_hat)):
            self._val_score_bucket.append(y_hat[i]["scores"].detach().float().cpu())
            self._val_pred_count_bucket.append(int(y_hat[i]["scores"].numel()))
            self._val_gt_count_bucket.append(int(y[i]["boxes"].shape[0]))

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and (hasattr(self.logger.experiment, "add_figure") | hasattr(self.logger.experiment, "log_figure"))
        ):
            if "boxes" not in batch:
                batch["boxes"] = batch.pop(self.boxes_field)
            if "labels" not in batch:
                batch["labels"] = batch.pop(self.labels_field)
            if self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False):
                if "masks" not in batch.keys() and self.masks_field in batch.keys():
                    batch["masks"] = batch.pop(self.masks_field)

            batch["prediction_boxes"] = [b["boxes"].cpu() for b in y_hat]
            batch["prediction_labels"] = [b["labels"].cpu() for b in y_hat]
            batch["prediction_scores"] = [b["scores"].cpu() for b in y_hat]

            if "masks" in y_hat[0]:
                batch["prediction_masks"] = [b["masks"].cpu() for b in y_hat]
                if self.framework == "mask-rcnn":
                    batch["prediction_masks"] = [b.unsqueeze(1) for b in batch["prediction_masks"]]

            batch["image"] = batch["image"].cpu()
            sample = unbind_samples(batch)[0]
            fig: Figure | None = None
            # Plot uses the DEPLOYMENT threshold (0.5), not the metric's score_threshold (0.05).
            try:
                if hasattr(self.trainer.datamodule, "val_dataset") and hasattr(
                    self.trainer.datamodule.val_dataset, "plot"
                ):
                    fig = self.trainer.datamodule.val_dataset.plot(sample)
                elif hasattr(self.trainer.datamodule, "plot") and callable(self.trainer.datamodule.plot):
                    fig = self.trainer.datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                if hasattr(self.logger.experiment, "add_figure"):
                    summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                elif hasattr(self.logger.experiment, "log_figure"):
                    summary_writer.log_figure(
                        self.logger.run_id, fig, f"epoch_{self.current_epoch}_{batch_idx}.png"
                    )
                plt.close()

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        metrics.pop("val_classes", None)
        self.log_dict(metrics)
        self.val_metrics.reset()

        # Log the per-epoch score distribution so we can see whether the model is showing
        # few high-confidence predictions or many low-confidence ones.
        scores_bucket = getattr(self, "_val_score_bucket", None)
        if scores_bucket:
            scores = torch.cat(scores_bucket) if scores_bucket else torch.empty(0)
            preds_per_img = torch.tensor(self._val_pred_count_bucket, dtype=torch.float32)
            gt_per_img = torch.tensor(self._val_gt_count_bucket, dtype=torch.float32)
            if scores.numel() > 0:
                q = torch.quantile(scores, torch.tensor([0.5, 0.9, 0.99]))
                n_gt5 = (scores > 0.5).sum().item()
                n_gt3 = (scores > 0.3).sum().item()
                n_gt1 = (scores > 0.1).sum().item()
                print(
                    f"[val epoch {self.current_epoch}] preds={scores.numel()} "
                    f"across {len(preds_per_img)} imgs "
                    f"(preds/img mean={preds_per_img.mean():.1f} "
                    f"gt/img mean={gt_per_img.mean():.1f}) | "
                    f"score max={scores.max().item():.3f} "
                    f"p50={q[0].item():.3f} p90={q[1].item():.3f} p99={q[2].item():.3f} | "
                    f"n(>0.1)={n_gt1} n(>0.3)={n_gt3} n(>0.5)={n_gt5}"
                )
                self.log_dict({
                    "val_score_max": scores.max(),
                    "val_score_p50": q[0],
                    "val_score_p90": q[1],
                    "val_preds_per_img": preds_per_img.mean(),
                    "val_gt_per_img": gt_per_img.mean(),
                    "val_frac_score_gt_0.3": torch.tensor(n_gt3 / max(1, scores.numel())),
                    "val_frac_score_gt_0.5": torch.tensor(n_gt5 / max(1, scores.numel())),
                })
            self._val_score_bucket = []
            self._val_pred_count_bucket = []
            self._val_gt_count_bucket = []

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Same rationale as `validation_step`: Only `.update()` here; the
        # single compute+reset happens in `on_test_epoch_end`.
        x = batch["image"]
        batch_size = get_batch_size(x)
        batch = self.apply_ignore_index(batch, self.ignore_index)
        y = self.reformat_batch(batch, batch_size)
        y_hat = self(x)
        if not isinstance(y_hat, dict):
            y_hat = y_hat.output

        y_hat = self.apply_nms_batch(y_hat, batch_size)

        if self.framework == "mask-rcnn" or self.model_args.get("framework_masks", False):
            for i in range(len(y_hat)):
                if "masks" in y_hat[i] and y_hat[i]["masks"].shape[0] > 0:
                    y_hat[i]["masks"] = (y_hat[i]["masks"] > 0.5).squeeze(1).to(torch.uint8)

        self.test_metrics.update(y_hat, y)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        metrics.pop("test_classes", None)
        self.log_dict(metrics)
        self.test_metrics.reset()

    def _num_encoder_blocks(self) -> int:
        # Count blocks by scanning param names.
        max_blk = -1
        for name, _ in self.model.named_parameters():
            if not _is_backbone_param(name):
                continue
            blk = _encoder_block_index(name)
            if blk is not None and blk > max_blk:
                max_blk = blk
        if max_blk >= 0:
            return max_blk + 1
        # CNN backbones (ConvNeXt / ResNet) have no encoder-block layout —
        # fall back to the feature-map count.
        return len(self.model.torchvision_model.backbone.backbone.out_channels)

    def _get_new_modality_names(self) -> list[str]:
        """Modality names that were NOT in the pretrained checkpoint."""
        try:
            bb = self.model.torchvision_model.backbone.backbone
        except AttributeError:
            return []
        found = _find_lunar_backbone(bb)
        if found is None:
            return []
        return list(getattr(found, "_new_modalities", None) or {})

    def _param_layer_id(self, name: str, num_blocks: int) -> int:
        """Return an integer layer id in [0, num_blocks+1].

        0        = embeddings / register tokens
        1..N     = encoder blocks (1 = deepest / lowest LR, N = shallowest)
        N+1      = encoder_norm (top of backbone)
        """
        blk = _encoder_block_index(name)
        if blk is not None:
            # deeper block index i (larger i) is closer to the head; give it a
            # higher layer id so it gets a larger LR.
            return blk + 1
        if "encoder_norm" in name:
            return num_blocks + 1
        # embeddings, register tokens, anything else in the backbone
        return 0

    def _make_param_groups(self) -> list[dict[str, Any]]:
        num_blocks = self._num_encoder_blocks()
        top_layer = num_blocks + 1
        groups: dict[tuple[int, bool, bool], dict[str, Any]] = {}
        new_mods = self._get_new_modality_names()

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # New-modality embedders live *inside* the backbone subtree but are
            # randomly initialized, so they must train at head_lr like the RPN/ROI heads.
            if _is_new_modality_param(name, new_mods):
                in_backbone = False
            else:
                in_backbone = _is_backbone_param(name)
            nd = _no_decay(name)
            if in_backbone:
                layer_id = self._param_layer_id(name, num_blocks)
                # top_layer gets LR = backbone_lr; each step down multiplies by layer_decay.
                scale = self.layer_decay ** (top_layer - layer_id)
                lr = self.backbone_lr * scale
            else:
                layer_id = -1  # sentinel for "head"
                lr = self.head_lr
            if nd:
                wd = 0.0
            elif in_backbone:
                wd = self.weight_decay
            else:
                wd = self.head_weight_decay
            key = (layer_id, in_backbone, nd)
            g = groups.setdefault(
                key,
                {
                    "params": [],
                    "lr": lr,
                    "weight_decay": wd,
                    "name": (
                        f"backbone_layer_{layer_id}_{'nd' if nd else 'wd'}"
                        if in_backbone
                        else f"head_{'nd' if nd else 'wd'}"
                    ),
                },
            )
            g["params"].append(p)

        # Stable ordering: backbone first (deepest → shallowest), then head.
        ordered = sorted(
            groups.values(),
            key=lambda g: (0 if g["name"].startswith("backbone") else 1, g["name"]),
        )
        return ordered

    def configure_optimizers(self):
        param_groups = self._make_param_groups()
        n_params = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
        print(
            f"[LunarObjectDetectionTask] {len(param_groups)} param groups, "
            f"{n_params/1e6:.1f}M trainable params. "
            f"layer_decay={self.layer_decay} backbone_lr={self.backbone_lr} "
            f"head_lr={self.head_lr}"
        )
        for g in param_groups:
            print(
                f"  {g['name']:<32s} lr={g['lr']:.2e} wd={g['weight_decay']:.2e} "
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
