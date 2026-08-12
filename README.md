# NASA-IBM Lunar-FM

Finetuning and inference release of the Graha Lunar-FM foundation model. Two Python packages:

- [`ni_lfm/`](ni_lfm/) — the model package (backbone, tokenizers, data utilities). Vendored; not edited in day-to-day work.
- [`terratorch_integration/`](terratorch_integration/) — TerraTorch-compatible datamodules, tasks, backbone wrappers, and runnable configs for lunar downstream tasks (crater detection, IMP segmentation, ice prospectivity, etc.). This is the working surface.

Pretraining code is not included.

## Install

```bash
pyenv install -s 3.12.2
pyenv virtualenv 3.12.2 graha-lunar-fm && pyenv activate graha-lunar-fm
pip install -e .
```

(or `conda create -n graha-lunar-fm python=3.12` if you prefer conda.)

## Weights

Weights are distributed out-of-band as a tarball. Unpack it so that the layout is:

```
backbone/
  checkpoint.pt        # base backbone checkpoint
  config.yaml          # merged pretraining config + per-modality info (required)
```

Every `ni_lfm_v1_*` config references `backbone/checkpoint.pt` and `backbone/config.yaml`. If your bundle lives elsewhere, symlink `./backbone` to it.

**`backbone_cfg` is required** for `ni_lfm_v1_*` backbones — the wrapper raises `ValueError` if missing.

## Data

Downstream configs reference data under `./data/…` — e.g. `data/LRO_Craters/`, `data/NAC_handlabeled_1m_256/`, `data/IMP_dataset/`, `data/prospectivity_dataset/`. Symlink `./data` at your dataset root. Each config's `data.init_args` block documents the exact subpaths it expects.

## Finetuning

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
graha-lunar-fm/
├── ni_lfm/                       # model package (backbone, tokenizers, data utils)
├── terratorch_integration/       # TerraTorch datamodules + tasks + configs
│   ├── configs/                  # runnable `terratorch fit` configs, grouped by task
│   │   ├── nac_craters/          # NAC crater detection
│   │   ├── wac_craters/          # WAC crater detection
│   │   ├── lro_craters/          # LRO crater detection (grayscale JPGs)
│   │   ├── imp/                  # Impact Melt Pond segmentation
│   │   └── ice_prosp/             # Ice prospectivity
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
