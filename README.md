# NASA-IBM Lunar-FM

Finetuning and inference release of the NASA-IBM Lunar Foundation Model foundation model. Two Python packages:

- [`ni_lfm/`](ni_lfm/) — the model package (backbone, tokenizers, data utilities). Vendored; not edited in day-to-day work.
- [`terratorch_integration/`](terratorch_integration/) — TerraTorch-compatible datamodules, tasks, backbone wrappers, and runnable configs for lunar downstream tasks (crater detection, IMP segmentation, ice prospectivity, etc.). This is the working surface.

Pretraining code is not included.

## Install

```bash
pyenv install -s 3.12.2
pyenv virtualenv 3.12.2 ni_lfm && pyenv activate ni_lfm
pip install -e .
```

(or `conda create -n ni_lfm python=3.12` if you prefer conda.)

## Weights and data

Configs use two relative roots, `data/` and `backbone/`, so no absolute paths are
baked into any YAML. Point them at the shared release bundle with two symlinks:

```bash
B=<path_to_your_dir_containing_data_and_weights>
ln -sfn "$B/downstream_dataset"   data
ln -sfn "$B/checkpoints/backbone" backbone
```

That gives every config the paths it expects:

```
backbone/checkpoint.pt                 # base backbone checkpoint
backbone/config.yaml                   # pretraining config + per-modality info (required)
data/prospectivity_dataset/            # ice_prosp/
data/imp_dataset/                      # imp/
data/nac_craters_dataset/              # nac_craters/  (COCO: images/*.npy + annotations_min5px.json)
data/wac_craters_dataset/              # wac_craters/  (images_tiff/, metadata.parquet, train|val|test.json)
```

**`backbone_cfg` is required** for `ni_lfm_v1_*` backbones — the wrapper raises
`ValueError` if missing.

To run against a different copy, you can either change the config path or re-point the symlinks.
Single-value overrides also work, e.g.
`--model.init_args.model_args.backbone_checkpoint_path /other/checkpoint.pt`.


## Fine-tuning

Every YAML under [`terratorch_integration/configs/`](terratorch_integration/configs/) is a runnable `terratorch fit` target:

```bash
PYTHONPATH=. terratorch fit -c terratorch_integration/configs/nac_craters/crater_detection_nac_dtm_meta.yaml
```

Common overrides:

```bash
# Short smoke test: one bounded epoch on CPU, no data workers
PYTHONPATH=. terratorch fit -c <config>.yaml -c examples/smoke_overlay.yaml --data.num_workers 0

# Point at a specific data root without editing the yaml
PYTHONPATH=. terratorch fit -c <config>.yaml \
  --data.data_dir /path/to/data \
  --data.metadata_file /path/to/metadata.parquet \
  --data.annotations_file /path/to/annotations.json
```

[`examples/smoke_overlay.yaml`](examples/smoke_overlay.yaml) bounds the run by epoch rather than with `--trainer.max_steps 1`; the latter truncates the epoch before validation, so any config with a val-monitored `EarlyStopping`/`ModelCheckpoint` aborts on a missing metric. The file explains that and the `limit_*_batches` int-vs-float trap.

Note: `terratorch fit` writes `config.yaml`/`config_deploy.yaml` to CWD by default — this is Lightning CLI's dumped merged config, not a project file. Delete after each run or configure `save_config_kwargs` to suppress.

## Test / evaluate

```bash
PYTHONPATH=. terratorch test --config <config_from_finetuning>.yaml --ckpt_path <finetuned_model>.ckpt
```

## Cluster / PBS

An example PBS wrapper for NASA-cluster batch submission lives at [`examples/pbs/run_finetuning.pbs`](examples/pbs/run_finetuning.pbs). Edit `CFG_PATH`, `#PBS -W group_list`, and the conda env activation to match your site.

## Repo layout

```
ni_lfm/
├── ni_lfm/                       # model package (backbone, tokenizers, data utils)
├── terratorch_integration/       # TerraTorch datamodules + tasks + configs
│   ├── configs/                  # runnable `terratorch fit` configs, grouped by task
│   │   ├── nac_craters/          # NAC crater detection
│   │   ├── wac_craters/          # WAC crater detection
│   │   │   ├── full_data/        #   100% of the train split
│   │   │   └── half_data/        #   50% of the train split (val/test still full)
│   │   ├── lro_craters/          # LRO crater detection (grayscale JPGs)
│   │   ├── imp/                  # Irregular Mare Patch (IMP) segmentation
│   │   └── ice_prosp/            # Ice prospectivity
│   ├── data_adapter.py           # LunarCraterDataModule, LunarNACDTMDataModule, LunarWACCraterDataModule
│   ├── data_utils.py             # D4DetectionTransform and related augmentations
│   ├── lunar_backbone.py         # TerraTorch backbone wrapper
│   ├── lunar_object_detection_task.py
│   ├── lunar_segmentation_task.py
│   ├── lunar_classification_task.py
│   ├── lunar_regression_task.py
│   ├── lunar_llrd_mixin.py       # layer-wise LR decay + split-group optimiser mixin
│   ├── lunar_register.py         # registers backbone variants with TerraTorch
│   ├── necks.py                  # LearnedTokenProjection, SimpleFeaturePyramid, MultilayerSimpleFeaturePyramid
│   └── decoders.py               # SumFuseDeepGNDecoder
├── examples/pbs/                 # cluster batch scripts
├── LICENSE                       # Apache-2.0
├── pyproject.toml
└── requirements.txt
```

Full documentation of TerraTorch is at <https://torchgeo.org/terratorch/quick_start/>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
