# TerraTorch Integration for Lunar-FM

Custom [TerraTorch](https://torchgeo.org/terratorch/) modules that let the NILunarFM backbones plug into TerraTorch's training and inference pipelines. Registers backbones, task subclasses, datamodules, necks, and decoders under `custom_modules_path: terratorch_integration`.

## What's here

- **Backbones** ([lunar_register.py](lunar_register.py)) — `ni_lfm_v1_{tiny,base,large}`.
- **Tasks** — subclasses of TerraTorch tasks that add LLRD + split backbone/head LR:
  - [`LunarObjectDetectionTask`](lunar_object_detection_task.py)
  - [`LunarSegmentationTask`](lunar_segmentation_task.py), `LunarShapeSegmentationTask`
  - [`LunarPixelwiseRegressionTask`](lunar_regression_task.py), `LunarClassificationTask`, `LunarScalarRegressionTask`
- **Datamodules** — [`LunarCraterDataModule`](data_adapter.py) (LRO grayscale), [`LunarNACDTMDataModule`](data_adapter.py) (NAC + DTM + metadata), [`LunarWACCraterDataModule`](data_adapter.py) (WAC multi-modal). IMP and ice-prospectivity configs use stock TerraTorch datamodules (`GenericNonGeoSegmentationDataModule`, `GenericMultiModalDataModule`).
- **Data utilities** — [data_utils.py](data_utils.py) (`D4DetectionTransform` and related augmentations).
- **Necks / decoders** — [necks.py](necks.py), [decoders.py](decoders.py).
- **Configs** ([configs/](configs/)) — one runnable `terratorch fit` YAML per experiment.

## Discovery

TerraTorch imports this package when a config sets:

```yaml
custom_modules_path: terratorch_integration
```

Importing triggers the `@register` decorators in `lunar_register.py`; backbones are then buildable by name from any downstream task.

## Using a Lunar-FM backbone

Every `ni_lfm_v1_*` backbone **requires** the pretraining `config.yaml` that ships with each checkpoint — the wrapper raises `ValueError` if it's missing. Repo convention:

```
backbone/
  checkpoint.pt
  config.yaml
```

Programmatic use:

```python
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import terratorch_integration  # noqa: F401  # registers backbones

model = TERRATORCH_BACKBONE_REGISTRY.build(
    "ni_lfm_v1_base",
    modalities=["vis"],
    cfg="backbone/config.yaml",
    checkpoint_path="backbone/checkpoint.pt",
)
```

YAML use (inside a task's `model_args`):

```yaml
model:
  class_path: terratorch_integration.LunarObjectDetectionTask
  init_args:
    model_factory: ObjectDetectionModelFactory
    model_args:
      framework: faster-rcnn
      backbone: ni_lfm_v1_base
      backbone_cfg: backbone/config.yaml
      backbone_checkpoint_path: backbone/checkpoint.pt
      backbone_modalities: [vis]
      backbone_merge_method: concat
      backbone_patch_size: 8
      # necks, num_classes, framework_min_size, ...
```

## Supported modalities

Modality metadata (channel counts, patch size, stats, tokenizer IDs) is loaded from the checkpoint's `config.yaml` under `data.domains`. The default bundle covers:

- **Raw imagery (low-res / WAC):** `vis` (5ch), `uv` (2ch), `dtm` (1ch), `slope` (1ch), `aspect` (2ch sin/cos).
- **Raw imagery (high-res / NAC):** `nac` (1ch), `dtm_3m` (1ch), `slope_3m` (1ch), `aspect_3m` (2ch).
- **Tokenized:** `tok_vis`, `tok_uv`, `tok_dtm`, `tok_slope`, `tok_aspect`, `tok_nac`, `tok_dtm_3m`, `tok_slope_3m`, `tok_aspect_3m`.
- **Sequences:** `metadata` (spatial), `static_maps`, `crater_bboxes`.

Register additional modalities at build time via `new_modalities={"my_mod": {"type": "image", "num_channels": 1}}`. See any `nac_craters/*.yaml` that adds a new tokenised modality for an end-to-end example.

## Multi-modal aggregation

`merge_method` (or YAML `backbone_merge_method:`) controls how per-modality tokens are combined after the encoder pass:

| `merge_method` | Output per encoder block |
|---|---|
| `None` (default) | `(B, N_total, D)` — all modality tokens concatenated |
| `"mean"` | `(B, N_common, D)` — averaged across modalities (requires matching token counts) |
| `"max"` | `(B, N_common, D)` — max-pooled across modalities |
| `"concat"` | `(B, N_common, D*M)` — concatenated along feature dim |
| `"masked_group_mean"` | `(B, N_common, D)` — masked group-mean across modalities |
| `"dict"` | `{modality: (B, N, D)}` — per-modality dict, useful for mixed resolutions |

## LLRD + split-LR training recipe

Every custom task (except `LunarObjectDetectionTask`, which keeps its own copy of the same recipe) mixes in [`_LunarLLRDMixin`](lunar_llrd_mixin.py). It builds AdamW param groups so that:

- The pretrained encoder gets a small `backbone_lr`, decayed per depth (`backbone_lr * layer_decay ** depth_from_top`).
- Freshly-initialized necks / heads / new-modality embedders get `head_lr` (typically 4-8x larger).
- Biases, norms, register tokens, positional embeddings skip weight decay regardless of group.
- Warmup → cosine annealing schedule.

**Do NOT set top-level `optimizer:` / `lr_scheduler:` blocks in the YAML.** Lightning CLI will monkey-patch `configure_optimizers` and silently drop the LLRD groups. Configure LR via the task's `init_args`:

```yaml
model:
  class_path: terratorch_integration.LunarSegmentationTask
  init_args:
    backbone_lr: 5.0e-5
    head_lr: 2.0e-4
    layer_decay: 0.75
    weight_decay: 0.05
    warmup_steps: 500
```

## Crater detection datamodules

Two flavours ship:

- [`LunarCraterDataModule`](data_adapter.py) — LRO grayscale JPGs, COCO annotations. Replicates the single-channel image to 5 channels (`output_mode: vis`) so the WAC-pretrained backbone's `vis` embedding applies unchanged; or emits a raw 1-channel tensor (`output_mode: image`) for baseline comparison methods. Also supports Mask R-CNN via `load_masks: true`.
- [`LunarNACDTMDataModule`](data_adapter.py) — NAC + DTM + metadata packed into a single tensor for the WAC-pretrained backbone's `packed` code path.

## LICENSE

Apache 2.0 (see the repo-root `LICENSE`).
