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


Model weights and config can be downloaded from HuggingFace using Python. See examples below:

```bash
from huggingface_hub import snapshot_download

# Only download model weights and config
snapshot_download(repo_id="nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", allow_patterns="backbone/*", local_dir="./")

# Download entire model HuggingFace directory
snapshot_download(repo_id="nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", local_dir="./")

# Download ice-prospectivity data
snapshot_download(repo_id="nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression", local_dir="./")

```

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
PYTHONPATH=. terratorch fit -c terratorch_integration/configs/nac_craters/ni_lfm_ps8.yaml
```

Common overrides:

```bash

# Point at a specific data root without editing the yaml
PYTHONPATH=. terratorch fit -c <config>.yaml \
  --data.data_dir /path/to/data \
  --data.metadata_file /path/to/metadata.parquet \
  --data.annotations_file /path/to/annotations.json
```

Note: `terratorch fit` writes `config.yaml`/`config_deploy.yaml` to CWD by default — this is Lightning CLI's dumped merged config, not a project file. Delete after each run or configure `save_config_kwargs` to suppress.

## Test / evaluate

```bash
PYTHONPATH=. terratorch test --config <config_from_finetuning>.yaml --ckpt_path <finetuned_model>.ckpt
```

## Crater-head post-processing: shadow filter

The fine-tuned NAC crater head (`ni_lfm_v1_base` + Faster R-CNN,
`NAC_ni_lfm_ps8_s44.ckpt`) was evaluated on a south-pole NAC image at
0.13 deg sun elevation (213 tiles, 256x256 px @ 1 m/px, score >= 0.5,
Connecting Ridge, M164948614LE). Result: 65.0% of detections in tiles
that are > 90% below an Otsu shadow threshold are "implausible" (huge
circles, up to 246 m, on essentially featureless shadow), versus < 1.5%
in every brighter shadow-fraction bin (30-50%, 50-70%, 70-90%) — a
reliability collapse specific to deep grazing-illumination shadow, not a
general false-positive rate. South-pole landing/traverse imagery sits in
this illumination regime by default, so an unfiltered crater layer from
this head is not directly usable there.

`terratorch_integration/shadow_filter.py` is a no-retraining
post-processing gate for exactly this failure mode: it re-reads the tile
pixels behind each detection box and computes (a) the detection
footprint's own shadow fraction, (b) a local contrast ratio (mean DN
inside the detection vs. an annulus just outside it), (c) the tile's
overall shadow fraction, and (d) the detection's diameter relative to the
tile — then applies one of two tuned gates:

- **Variant A** (multi-feature, default): drops a detection if its
  diameter exceeds 60% of the tile edge, or if the tile is > 90% shadow
  AND the detection footprint is >= 95% shadow AND its contrast ratio is
  <= 1 (no local contrast against its surroundings). Tuned/held-out
  precision of the "removed" set = 100%, recall of implausible
  detections = 72.7%; keeps 99.4% of detections in tiles with 30-70%
  shadow (where the head's own output is presumably mostly correct).
- **Variant B** (simpler): the same diameter cap plus a single threshold
  on the detection's own footprint shadow fraction (>= 0.98). Precision =
  81.0%, recall = 98.1%, keeps 97.9% of the 30-70%-shadow-bin detections.

Both variants were tuned on the even-index half of the 213 evaluation
tiles against a heuristic "implausible" label and evaluated on the
odd-index half — the label is a documented heuristic (large diameter, or
a fully-shadowed footprint in an almost fully shadowed tile), not a
human-verified reference set. See the PR description and
`terratorch_integration/shadow_filter.py`'s module docstring for the full
numbers and caveats. Run `python terratorch_integration/shadow_filter_smoke_test.py`
for a dependency-free sanity check on synthetic tiles.

## Cluster (PBS / SLURM)

Example batch wrappers for cluster submission live at [`examples/pbs/run_finetuning.pbs`](examples/pbs/run_finetuning.pbs) (PBS) and [`examples/slurm/run_finetuning.sbatch`](examples/slurm/run_finetuning.sbatch) (SLURM). Edit `CFG_PATH`, the scheduler directives (`#PBS -W group_list` / `#SBATCH --account`, etc.), and the conda env activation to match your site.

## Repo layout

```
ni_lfm/
├── ni_lfm/                       # model package (backbone, tokenizers, data utils)
├── terratorch_integration/       # TerraTorch datamodules + tasks + configs
│   ├── README.md                 # package-level docs (backbones, tasks, determinism)
│   ├── configs/                  # runnable `terratorch fit` configs, grouped by task
│   │   ├── nac_craters/          # NAC crater detection
│   │   ├── wac_craters/          # WAC crater detection
│   │   │   ├── full_data/        #   100% of the train split
│   │   │   └── half_data/        #   50% of the train split (val/test still full)
│   │   ├── imp/                  # Irregular Mare Patch (IMP) segmentation
│   │   └── ice_prosp/            # Ice prospectivity
│   │       └── ablation/         #   modality-subset ablations (m2–m7)
│   ├── data_adapter.py           # LunarCraterDataModule, LunarNACDTMDataModule, LunarWACCraterDataModule
│   ├── data_utils.py             # D4DetectionTransform and related augmentations
│   ├── lunar_backbone.py         # TerraTorch backbone wrapper
│   ├── lunar_object_detection_task.py
│   ├── lunar_segmentation_task.py
│   ├── lunar_classification_task.py
│   ├── lunar_regression_task.py
│   ├── lunar_llrd_mixin.py       # layer-wise LR decay + split-group optimiser mixin
│   ├── lunar_register.py         # registers backbone variants with TerraTorch
│   ├── determinism.py            # deterministic drop-ins + Albumentations seeding callbacks
│   ├── necks.py                  # LearnedTokenProjection, SimpleFeaturePyramid, MultilayerSimpleFeaturePyramid
│   ├── decoders.py               # SumFuseDeepGNDecoder
│   ├── shadow_filter.py          # crater-head post-processing gate for grazing-illumination shadow
│   └── shadow_filter_smoke_test.py  # dependency-free sanity check on synthetic tiles
├── examples/
│   ├── pbs/                      # PBS batch scripts
│   ├── slurm/                    # SLURM batch scripts
├── README.md                     # this file
├── LICENSE                       # Apache-2.0
├── pyproject.toml
└── requirements.txt
```

Full documentation of TerraTorch is at <https://torchgeo.org/terratorch/quick_start/>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
