# Lunar-FM TerraTorch Integration Examples

This document provides configuration examples for using NILunarFM models with TerraTorch's task framework.

## Table of Contents

1. [Crater Detection (Object Detection)](#crater-detection-object-detection)
2. [Lunar Surface Segmentation](#lunar-surface-segmentation)
3. [Multi-Modal Examples](#multi-modal-examples)
4. [Transfer Learning](#transfer-learning)

---

## Crater Detection (Object Detection)

### Example 1: Basic Crater Detection with Faster R-CNN

```yaml
# crater_detection_tiny.yaml
# lightning.pytorch==2.1.1
seed_everything: 42

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 100
  check_val_every_n_epoch: 5
  log_every_n_steps: 10
  logger:
    class_path: TensorBoardLogger
    init_args:
      save_dir: ./logs
      name: lunar-crater-detection
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: ModelCheckpoint
      init_args:
        dirpath: ./checkpoints/crater_detection
        monitor: val_map
        mode: max
        filename: best-{epoch:02d}-{val_map:.3f}
        save_top_k: 3

data:
  class_path: terratorch_integration.data_adapter.LunarCraterDataModule
  init_args:
    root: <path_to_crater_data>
    batch_size: 4
    num_workers: 2
    image_size: 512
    apply_norm_in_datamodule: true
    norm_means: [0.5]  # Grayscale
    norm_stds: [0.25]

model:
  class_path: terratorch.tasks.ObjectDetectionTask
  init_args:
    model_factory: ObjectDetectionModelFactory
    model_args:
      framework: faster-rcnn
      backbone: ni_lfm_v1_tiny
      backbone_pretrained: false
      num_classes: 2  # background + crater
      framework_min_size: 512
      framework_max_size: 512
      backbone_modalities:
        - vis  # Single-channel grayscale
      in_channels: 1
      necks:
        - name: SelectIndices
          indices: [2, 5, 8, 11]  # Select encoder layers for tiny model
        - name: ReshapeTokensToImage
          remove_cls_token: false
        - name: LearnedInterpolateToPyramidal
        - name: FeaturePyramidNetworkNeck
    freeze_backbone: false
    freeze_decoder: false
    class_names:
      - Background
      - Crater

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-4
    weight_decay: 0.05

lr_scheduler:
  class_path: CosineAnnealingLR
  init_args:
    T_max: 20

# Specify custom module path
terratorch_integration_path: ./terratorch_integration
```

**Run training:**
```bash
terratorch fit --config crater_detection_tiny.yaml
```

---

## Lunar Surface Segmentation

### Example 2: Semantic Segmentation with UNet Decoder

```yaml
# lunar_segmentation_tiny.yaml
seed_everything: 42

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 200
  check_val_every_n_epoch: 10
  log_every_n_steps: 50
  logger: true
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: ModelCheckpoint
      init_args:
        dirpath: ./checkpoints/segmentation
        monitor: val_loss
        mode: min
        filename: best-{epoch:02d}
        save_top_k: 3

data:
  class_path: terratorch.datamodules.GenericMultiModalDataModule
  init_args:
    task: segmentation
    batch_size: 8
    num_workers: 4
    modalities:
      - vis
      - dtm
    rgb_modality: vis
    rgb_indices: [0, 1, 2]  # Use first 3 channels of vis
    train_data_root:
      vis: <path_to_data>/train/vis
      dtm: <path_to_data>/train/dtm
    train_label_data_root: <path_to_data>/train/labels
    val_data_root:
      vis: <path_to_data>/val/vis
      dtm: <path_to_data>/val/dtm
    val_label_data_root: <path_to_data>/val/labels
    no_label_replace: -1
    no_data_replace: 0
    num_classes: 5

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: ni_lfm_v1_tiny
      backbone_pretrained: true
      backbone_modalities:
        - vis
        - dtm
      backbone_merge_method: mean  # How to merge multi-modal features
      necks:
        - name: SelectIndices
          indices: [2, 5, 8, 11]  # tiny: [2,5,8,11], base: [2,5,8,11], large: [5,11,17,23]
        - name: ReshapeTokensToImage
          remove_cls_token: false
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [256, 128, 64, 32]
      head_dropout: 0.1
      num_classes: 5
    loss: dice
    ignore_index: -1
    freeze_backbone: false
    freeze_decoder: false
    class_names:
      - Regolith
      - Rock
      - Shadow
      - Crater_Floor
      - Crater_Rim

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-4
    weight_decay: 0.05

lr_scheduler:
  class_path: PolynomialLR
  init_args:
    total_iters: 200
    power: 0.9

terratorch_integration_path: ./terratorch_integration
```

**Run training:**
```bash
terratorch fit --config lunar_segmentation_tiny.yaml
```

---

## Multi-Modal Examples

### Example 3: Four Modalities (Visual + DTM + Slope + Aspect)

```yaml
# multimodal_segmentation_base.yaml
seed_everything: 42

trainer:
  accelerator: gpu
  devices: 2
  strategy: ddp
  precision: 16-mixed
  max_epochs: 300

data:
  class_path: terratorch.datamodules.GenericMultiModalDataModule
  init_args:
    task: segmentation
    batch_size: 4
    num_workers: 8
    modalities:
      - vis
      - dtm
      - slope
      - aspect
    rgb_modality: vis
    rgb_indices: [0, 1, 2]
    train_data_root:
      vis: <path_to_data>/train/vis
      dtm: <path_to_data>/train/dtm
      slope: <path_to_data>/train/slope
      aspect: <path_to_data>/train/aspect
    train_label_data_root: <path_to_data>/train/labels
    val_data_root:
      vis: <path_to_data>/val/vis
      dtm: <path_to_data>/val/dtm
      slope: <path_to_data>/val/slope
      aspect: <path_to_data>/val/aspect
    val_label_data_root: <path_to_data>/val/labels
    num_classes: 10

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: ni_lfm_v1_base
      backbone_pretrained: true
      backbone_checkpoint: <path_to_checkpoint>/checkpoint_weights_final.pt
      backbone_modalities:
        - vis
        - dtm
        - slope
        - aspect
      backbone_merge_method: mean
      necks:
        - name: SelectIndices
          indices: [2, 5, 8, 11]
        - name: ReshapeTokensToImage
          remove_cls_token: false
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [256, 128, 64, 32]
      num_classes: 10
    loss: dice
    freeze_backbone: false
    freeze_decoder: false

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 5e-5
    weight_decay: 0.05

terratorch_integration_path: ./terratorch_integration
```

### Example 4: Tokenized Modalities

```yaml
# tokenized_segmentation_base.yaml
seed_everything: 42

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 150

data:
  class_path: terratorch.datamodules.GenericMultiModalDataModule
  init_args:
    task: segmentation
    batch_size: 8
    num_workers: 4
    modalities:
      - tok_vis
      - tok_dtm
    # Note: Tokenized modalities don't use rgb_modality/rgb_indices
    train_data_root:
      tok_vis: <path_to_tokenized_data>/train/vis
      tok_dtm: <path_to_tokenized_data>/train/dtm
    train_label_data_root: <path_to_data>/train/labels
    val_data_root:
      tok_vis: <path_to_tokenized_data>/val/vis
      tok_dtm: <path_to_tokenized_data>/val/dtm
    val_label_data_root: <path_to_data>/val/labels
    num_classes: 5

model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: ni_lfm_v1_base
      backbone_pretrained: true
      backbone_checkpoint: <path_to_checkpoint>/checkpoint_weights_final.pt
      backbone_modalities:
        - tok_vis
        - tok_dtm
      backbone_merge_method: mean
      necks:
        - name: SelectIndices
          indices: [2, 5, 8, 11]
        - name: ReshapeTokensToImage
          remove_cls_token: false
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [256, 128, 64, 32]
      num_classes: 5
    loss: dice
    freeze_backbone: false
    
    # Codebook fusion configuration (if using multi-codebook tokenizers)
    encoder_embedding_params:
      tok_vis:
        fusion_type: attention  # or 'mlp', 'weighted', 'linear'
        fusion_hidden_ratio: 2.0
      tok_dtm:
        fusion_type: attention

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 5e-5
    weight_decay: 0.05

terratorch_integration_path: ./terratorch_integration
```

---

## Transfer Learning

### Example 5: Fine-tuning with Frozen Backbone

```yaml
# transfer_learning_crater_detection.yaml
seed_everything: 42

trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 50  # Shorter for transfer learning

data:
  class_path: terratorch_integration.data_adapter.LunarCraterDataModule
  init_args:
    root: <path_to_crater_data>
    batch_size: 8
    num_workers: 2

model:
  class_path: terratorch.tasks.ObjectDetectionTask
  init_args:
    model_factory: ObjectDetectionModelFactory
    model_args:
      framework: faster-rcnn
      backbone: ni_lfm_v1_tiny
      backbone_pretrained: true
      backbone_checkpoint: <path_to_checkpoint>/checkpoint_weights_final.pt
      num_classes: 2
      framework_min_size: 512
      framework_max_size: 512
      backbone_modalities:
        - vis
      in_channels: 1
      necks:
        - name: SelectIndices
          indices: [2, 5, 8, 11]
        - name: ReshapeTokensToImage
          remove_cls_token: false
        - name: LearnedInterpolateToPyramidal
        - name: FeaturePyramidNetworkNeck
    freeze_backbone: true  # Freeze pretrained backbone
    freeze_decoder: false  # Train detection head
    class_names:
      - Background
      - Crater

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-3  # Higher LR for head-only training
    weight_decay: 0.05

terratorch_integration_path: ./terratorch_integration
```

---

## Key Configuration Parameters

### Backbone Selection

- `ni_lfm_v1_base`: 12 encoder layers, 768 dim, ~86M params

### Neck Indices

Select encoder layers to use as features:
- **Tiny/Base**: `[2, 5, 8, 11]` (4 layers from 12)
- **Large**: `[5, 11, 17, 23]` (4 layers from 24)

### Backbone Merge Method

For multi-modal inputs:
- `mean`: Average features across modalities
- `concat`: Concatenate features (increases dimension)
- `attention`: Learned attention-based fusion

### Decoder Options

- `UNetDecoder`: Standard UNet decoder
- `FPNDecoder`: Feature Pyramid Network decoder
- `UperNetDecoder`: UperNet decoder for multi-scale

---

## Running Examples

### Training
```bash
# Basic training
terratorch fit --config <my_terratorch_config.yaml>

# Resume from checkpoint
terratorch fit --config <my_terratorch_config.yaml> \
  --ckpt_path checkpoints/segmentation/last.ckpt

# Override parameters
terratorch fit --config <my_terratorch_config.yaml> \
  --trainer.max_epochs 300 \
  --data.init_args.batch_size 16
```

### Validation
```bash
terratorch validate --config <my_terratorch_config.yaml> \
  --ckpt_path checkpoints/segmentation/best-epoch=199.ckpt
```

### Testing
```bash
terratorch test --config <my_terratorch_config.yaml> \
  --ckpt_path checkpoints/segmentation/best.ckpt
```

### Prediction
```bash
terratorch predict --config <my_terratorch_config.yaml> \
  --ckpt_path checkpoints/segmentation/best.ckpt
```

---

## Tips and Best Practices

1. **Modality Matching**: Ensure checkpoint modalities match config modalities
2. **Neck Indices**: Adjust based on model variant (tiny/base/large)
3. **Batch Size**: Reduce for larger models or higher resolution
4. **Precision**: Use `16-mixed` for faster training and lower memory
5. **Freeze Strategy**: Freeze backbone for small datasets, unfreeze for large datasets
6. **Multi-GPU**: Use `strategy: ddp` for multi-GPU training

## Next Steps

- See [`README.md`](README.md) for installation and API reference
- Check [TerraTorch documentation](https://terrastackai.github.io/terratorch/) for more task types