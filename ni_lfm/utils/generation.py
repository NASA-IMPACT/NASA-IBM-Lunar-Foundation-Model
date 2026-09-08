import copy

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omegaconf import OmegaConf
from timm.layers.helpers import to_2tuple
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from ni_lfm.data.io_utils import load_parquet_file
from ni_lfm.data.modality_info import MODALITY_INFO, setup_modality_transform
from ni_lfm.data.modality_transforms import AbstractTransform, get_transform, unscale_data
from ni_lfm.models.ni_lfm_generation import checkpoint_filter_fn_generate
from ni_lfm.utils.tokenizer import PAD_TOKEN, encode_sequence
from ni_lfm.vq.vqvae import DiVAE, build_divae


INDEX_FILE_KEYS = {
    "vis": "WAC_VIS_TILE",
    "uv": "WAC_UV_TILE",
    "dtm": "DTM_TILE",
    "slope": "SLOPE_TILE",
    "aspect": "ASPECT_TILE",
    "aspect_3m": "ASPECT_3M_TILE",
    "dtm_3m": "DTM_3M_TILE",
    "nac": "NAC_TILE",
    "slope_3m": "SLOPE_3M_TILE",
    "static_maps": "METADATA_AVG_TILE",
    "metadata": "METADATA_AVG_TILE",
}


MODALITY_TO_TOKENIZER_DIR = {
    "tok_vis": "WAC_vis",
    "tok_uv": "WAC_uv",
    "tok_dtm": "WAC_dtm",
    "tok_aspect": "WAC_aspect",
    "tok_slope": "WAC_slope",
    "tok_nac": "NAC_nac",
    "tok_aspect_3m": "NAC_aspect3m",
    "tok_slope_3m": "NAC_slope3m",
    "tok_dtm_3m": "NAC_dtm3m",
}

TOKENIZER_CONFIG_NAME = "config.yaml"
TOKENIZER_CKPT_NAME = "checkpoint.pt"


def collate_fn(batch: list[dict]):
    """Custom collate function that handles text modalities.

    Text modalities (metadata, static_maps) come as:
    - Lists of lists of strings: (e.g., [["var1", "value1"], ["var2", "value2"]])
    """
    def is_string_based(value):
        """Check if value is a string or nested list of strings."""
        if isinstance(value, str):
            return True
        if isinstance(value, list) and len(value) > 0:
            first = value[0]
            if isinstance(first, str):
                return True
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], str):
                return True
        return False

    if not batch:
        return {}

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [sample[key] for sample in batch]
        collated[key] = values if is_string_based(values[0]) else default_collate(values)

    return collated


class CenterCropImage(object):
    """Basic Center crop."""
    def __init__(self, target_size: int, main_domain: str = "vis"):
        self.target_size = to_2tuple(target_size)
        self.main_domain = main_domain

    def __call__(self, mod_dict):
        image = (
            mod_dict[self.main_domain]
            if self.main_domain is not None
            else mod_dict[list(mod_dict.keys())[0]]
        )
        orig_height, orig_width = image.shape[-2:]
        orig_size = (orig_height, orig_width)
        h, w = self.target_size

        # Image must be larger than target size
        if w <= orig_width and h <= orig_height:
            top = int(round((orig_height - h) / 2.0))
            left = int(round((orig_width - w) / 2.0))
        else:
            raise ValueError(f"Image size smaller than target size: {(orig_height, orig_width)} < {self.target_size}.")

        crop_coords = (top, left, h, w)

        return crop_coords, orig_size, self.target_size


class LunarGenerationDataset(Dataset):
    """A multimodal dataset where the samples are listed on a index file.

    Args:
        data_root (string): Root directory path.
        index_path (string): path to the index file pointing to sample paths.
        modalities (list): List of modalities as strings.
        modality_transforms (dict): Dict of transforms for each modality
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        max_samples (int, optional): Maximum number of samples to load. If None, all samples are loaded.
        max_samples_random: if max_samples is set, whether the selection is random or first max_samples.
        remove_key_mod_nan: if True (default) skip rows where key modalities have NaN values.
    """

    def __init__(
        self,
        data_root: str,
        index_path: str,
        modalities: list[str],
        modality_transforms: dict[str, AbstractTransform],
        center_crop: Callable | None,
        max_samples: int | None = None,
        max_samples_random: bool = True,
        remove_key_mod_nan: bool = True,
        return_file_name: bool = False,
        return_raw_modalities: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.modalities = modalities
        self.modality_transforms = modality_transforms
        self.return_file_name = return_file_name
        self.center_crop = center_crop
        self.return_raw_modalities = return_raw_modalities

        # Load parquet file and filter based on parameters
        index = load_parquet_file(
            index_path,
            remove_key_mod_nan=remove_key_mod_nan,
            filter_missing_paths=True,  # Always filter
            columns=[INDEX_FILE_KEYS[mod] for mod in modalities if mod in INDEX_FILE_KEYS],
        )

        # Select random subset of dataset if so specified
        if isinstance(max_samples, int):
            total_samples = len(index)
            if max_samples_random:
                permutation = np.random.permutation(total_samples)
                index = index.iloc[permutation[:max_samples]]
            else:
                index = index.iloc[:max_samples]

        self.index = index

    def unified_crop(self, mod_dict: dict[str, Any]):
        """Apply the image center crop to all modalities where it is applicable."""

        crop_coords, orig_size, target_size = self.center_crop(mod_dict)

        mod_dict = {
            k: get_transform(k, self.modality_transforms).image_augment(
                v,
                crop_coords=crop_coords,
                flip=False,
                orig_size=orig_size,
                target_size=target_size,
                rand_aug_idx=None,
            )
            for k, v in mod_dict.items()
        }

        return mod_dict

    def __getitem__(self, idx: int) -> dict[str, Any]:

        row = self.index.iloc[idx]

        sample_dict = {}
        stems = {}  # mod -> stem of the path used

        # Load modalities
        for mod in self.modalities:
            key = INDEX_FILE_KEYS[mod]

            path = Path(self.data_root) / row[key]
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            sample = self.modality_transforms[mod].load(path)
            sample_dict[mod] = sample
            stems[mod] = path.stem

        # Verify all modalities share the same stem
        unique_stems = set(stems.values())
        if len(unique_stems) != 1:
            raise ValueError(f"Sample {idx}: modality paths have inconsistent stems: {stems}")

        if self.return_raw_modalities is not None:
            sample_dict_raw = {
                f"{k}_raw": copy.deepcopy(v) for k, v in sample_dict.items() if k in self.return_raw_modalities
            }

        # Preprocess modalities
        sample_dict = {k: self.modality_transforms[k].preprocess(v) for k, v in sample_dict.items()}

        # Center crop image modalities
        if self.center_crop is not None:
            sample_dict = self.unified_crop(sample_dict)

        # Postprocess modalities
        sample_dict = {k: self.modality_transforms[k].postprocess(v) for k, v in sample_dict.items()}

        if self.return_file_name:
            sample_dict["file_name"] = unique_stems.pop()

        if self.return_raw_modalities is not None:
            return sample_dict | sample_dict_raw
        else:
            return sample_dict

    def __len__(self) -> int:
        return len(self.index)


def build_modality_info(cfg):
    """Build per-modality info from an already-loaded checkpoint config.

    Structural fields (type, embedding classes, data_range) come from
    `MODALITY_INFO` in the repo. Data-run-specific fields (num_channels,
    stats, ids, ...) come from `cfg.data.domains`. Both are merged; the
    yaml wins on conflicts.

    Args:
        cfg: DictConfig or dict — the merged checkpoint `config.yaml`.

    Returns:
        (modality_info dict, common_input_size int).
    """
    yaml_domains = OmegaConf.to_container(cfg.data.domains, resolve=True)

    modality_info = {}
    for mod, yaml_info in yaml_domains.items():
        if mod not in MODALITY_INFO:
            raise KeyError(f"Modality {mod} not in repo MODALITY_INFO registry")
        modality_info[mod] = {**MODALITY_INFO[mod], **yaml_info}

    print(f"Loaded modality info for: {list(modality_info.keys())}")

    common_input_size = next(
        (info["input_size"] for info in modality_info.values() if "input_size" in info),
        None,
    )
    if common_input_size is None:
        raise ValueError("Could not find input_size in loaded modality_info")
    print(f"Common input size: {common_input_size}")

    return modality_info, common_input_size


def denormalize_data(data: torch.Tensor, modality: str, modality_info: dict):
    """Denormalize data using modality_info stats.

    Args:
        data: Tensor to denormalize
        modality: Modality name (e.g., 'tok_vis' or 'vis')
        modality_info: Dictionary containing modality statistics

    Returns:
        Denormalized data in the same format as input
    """
    if modality not in modality_info:
        return data

    info = modality_info[modality]
    # Tokenized modalities redirect to their raw parent for stats/scaler.
    if info.get("pretokenized"):
        parent = info.get("parent_domain")
        if parent is None or parent not in modality_info:
            return data
        info = modality_info[parent]

    stats = info.get("stats")
    scaler = info.get("scaler")
    if not stats or scaler in (None, "None"):
        return data

    def _as_seq(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return [float(v)]
        return v

    return unscale_data(
        data=data,
        scaler=scaler,
        mean=_as_seq(stats.get("mean")),
        std=_as_seq(stats.get("std")),
        min_value=_as_seq(stats.get("min")),
        max_value=_as_seq(stats.get("max")),
    )


def build_tokenizers_from_modality_info(
    modality_info: dict,
    modalities: list[str],
    tokenizers_root: str | Path,
) -> torch.nn.ModuleDict:
    """Build tokenizers by auto-discovering per-modality subfolders under `tokenizers_root`.

    Args:
        modality_info: Dict containing modality information (used to check `pretokenized`).
        modalities: List of modalities to build tokenizers for.
        tokenizers_root: Root directory containing one subfolder per tokenizer, e.g.
            `.../tokenizers_final/{WAC_vis, NAC_nac, ...}`. Each subfolder must contain
            `config_reduced.yaml` and `checkpoint_weights_final.pt`.

    Returns:
        ModuleDict of tokenizers keyed by modality name.
    """
    tokenizers_root = Path(tokenizers_root)
    tokenizers = {}

    for modality in modalities:
        mod_info = modality_info.get(modality)
        if mod_info is None or not mod_info.get("pretokenized"):
            continue

        if modality not in MODALITY_TO_TOKENIZER_DIR:
            raise ValueError(f"No tokenizer folder mapping for modality {modality}.")

        tok_dir = tokenizers_root / MODALITY_TO_TOKENIZER_DIR[modality]
        cfg_path = tok_dir / TOKENIZER_CONFIG_NAME
        ckpt_path = tok_dir / TOKENIZER_CKPT_NAME
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Tokenizer files not found for {modality} at {tok_dir}: "
                f"expected {TOKENIZER_CONFIG_NAME} and {TOKENIZER_CKPT_NAME}.",
            )

        print(f"Loading tokenizer for {modality} from {tok_dir}")
        tok_cfg = OmegaConf.load(cfg_path)

        # Get image size used to initialize tokenizer - useful to guarantee positional embeddings are
        # interpolated correctly based on the original tokenizer input size
        input_size_tok = max(tok_cfg.data.resolutions)
        tokenizer = build_divae(cfg=tok_cfg, image_size=input_size_tok)

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[{modality}] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[{modality}] Unexpected keys: {len(unexpected)}")

        tokenizers[modality] = tokenizer

    return torch.nn.ModuleDict(tokenizers)


def filter_modality_info(all_domains: list[str], modality_info: dict):
    mod_info = copy.deepcopy(modality_info)
    mod_info = {mod: mod_info[mod] for mod in all_domains if mod in mod_info}
    return mod_info


def get_dataloader_for_generation(
    data_root: str,
    index_path: str,
    all_domains: list[str],
    modality_info: dict,
    input_size: int,
    num_samples: int,
    img_domains: list[str],
    return_file_name: bool = False,
    return_raw_modalities: list[str] | None = None,
):
    """Create a DataLoader over LunarDataset for generation (no masking, center crop)."""

    modality_transforms = setup_modality_transform(domains=all_domains, modality_info=modality_info)

    # Use nac or vis as main domain to determine crop dimensions
    if len(img_domains) == 0:
        center_crop = None
    else:
        if "nac" in img_domains:
            main_domain = "nac"
        elif "vis" in img_domains:
            main_domain = "vis"
        else:
            main_domain = img_domains[0]  # If neither wac/nac modalities are there, pick any one
        center_crop = CenterCropImage(target_size=input_size, main_domain=main_domain)

    dataset = LunarGenerationDataset(
        data_root=data_root,
        index_path=index_path,
        modalities=all_domains,
        modality_transforms=modality_transforms,
        center_crop=center_crop,
        max_samples=num_samples,
        max_samples_random=True,
        return_file_name=return_file_name,
        return_raw_modalities=return_raw_modalities,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return loader


def encode_sequence_batch(
    sequence: list[str] | list[list[str]],
    max_tokens: int,
    text_tokenizer: Tokenizer,
    pad_token: str = PAD_TOKEN,
) -> torch.Tensor:
    """Encode a batch of raw sequence samples to padded token tensors."""
    pad_id = text_tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError("Tokenizer does not contain [PAD] token")

    encoded_batch = []
    for value in sequence:
        if isinstance(value, list):
            seq_chunks: list[list[int]] = encode_sequence(value, text_tokenizer, max_tokens=max_tokens)
            seq_ids = [tok for chunk in seq_chunks for tok in chunk]
        else:
            seq_ids: list[int] = encode_sequence(value, text_tokenizer, max_tokens=max_tokens)
        encoded_batch.append(seq_ids)

    tensor = torch.full((len(encoded_batch), max_tokens), pad_id, dtype=torch.long)
    for idx, seq_ids in enumerate(encoded_batch):
        if seq_ids:
            tensor[idx, :len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.long)

    return tensor


def prepare_model_inputs(
    batch: dict,
    in_domains: list[str],
    modality_info: dict,
    text_tokenizer: Tokenizer | None,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Prepare model inputs from batch data."""
    model_inputs = {}
    for modality in in_domains:
        if modality not in batch:
            continue

        if modality_info[modality]["type"] == "seq":
            if text_tokenizer is None:
                raise ValueError(f"Text tokenizer required for sequence modality {modality}")
            tensor = encode_sequence_batch(sequence=batch[modality],
                                            max_tokens=modality_info[modality]["max_tokens"],
                                            text_tokenizer=text_tokenizer).to(device)
        else:
            tensor = batch[modality].to(device)

        model_inputs[modality] = tensor

    return model_inputs


def load_ckpt_for_generation(checkpoint: str, model):
    """"Load checkpoint tracking which keys came from file and which were filtered."""

    if checkpoint and Path(checkpoint).exists():
        print(f"\nLoading checkpoint from {checkpoint}")

        # Load checkpoint (weights_only=False to handle other objects)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if "model" in state_dict:
            state_dict = state_dict["model"]  # now get model weights only

        # checkpoint_filter_fn_generate remaps "key" -> "sampler.model.key" and fills any remaining
        # model keys from the current model state dict so strict=True works.
        raw_ckpt_keys = set(state_dict.keys())
        state_dict = checkpoint_filter_fn_generate(state_dict, model)

        # Keys that were mapped from the checkpoint (sampler.model.* prefix added)
        model_keys_loaded = {"sampler.model." + k for k in raw_ckpt_keys if "sampler.model." + k in state_dict}
        model_state = model.state_dict()
        backbone_key_count = 0
        backbone_params = 0
        backbone_params_loaded = 0
        tokenizer_params = 0

        for key, tensor in model_state.items():
            num_params = tensor.numel()
            if key.startswith("tokenizer"):
                tokenizer_params += num_params
            else:
                backbone_key_count += 1
                backbone_params += num_params
                if key in model_keys_loaded:
                    backbone_params_loaded += num_params

        missing, unexpected = model.load_state_dict(state_dict, strict=True)

        print("Checkpoint loaded successfully!")
        print(f"Backbone keys from checkpoint: {len(model_keys_loaded)} / {backbone_key_count} "
              f"({len(model_keys_loaded) / max(backbone_key_count, 1) * 100:.1f}%)")
        print(f"Backbone params from checkpoint: {backbone_params_loaded:,} / {backbone_params:,} "
              f"({backbone_params_loaded / max(backbone_params, 1) * 100:.1f}%)")
        print(f"Tokenizer params (pretrained, preserved): {tokenizer_params:,}")
        if missing:
            print(f"Missing keys after filter (should be 0 with strict=True): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    else:
        print("No checkpoint provided, using randomly initialized weights.")


def process_tok_target(
        raw_tok: torch.Tensor, tok_module: DiVAE, input_size: int, timesteps: int | None, verbose: bool = True,
    ) -> torch.Tensor:
    """Process target through tokenizer (autoencode or decode_tokens)."""
    patch_size = int(tok_module.patch_size)
    nh = input_size // patch_size
    nw = input_size // patch_size
    num_tokens = nh * nw

    # Case (a): raw pixel image (B, C, H, W)
    if raw_tok.ndim == 4:
        with torch.no_grad():
            result = tok_module.autoencode(raw_tok, timesteps=timesteps, verbose=verbose)
            return result if isinstance(result, torch.Tensor) else result[0]

    # Case (b): pre-tokenized indices
    elif raw_tok.ndim == 2 or (raw_tok.ndim == 3 and raw_tok.shape[1] == num_tokens):
        if raw_tok.ndim == 3:
            tok_grid = raw_tok.reshape(raw_tok.shape[0], nh, nw, raw_tok.shape[-1])
        else:
            tok_grid = raw_tok.reshape(raw_tok.shape[0], nh, nw)

        with torch.no_grad():
            return tok_module.decode_tokens(
                tok_grid, image_size=(input_size, input_size), timesteps=timesteps, verbose=verbose,
            )
    else:
        print(f"Warning: unexpected target shape: {raw_tok.shape}, using raw")
        return raw_tok


def sort_text(texts: list[str]):
    tokens_list = texts[0].split()  # Split by '' and sort the key-value pairs (each pair is like "EM_ANG=1.00-->1.50")
    tokens_sorted = sorted(tokens_list)
    text_sorted = " ".join(tokens_sorted)
    return text_sorted, tokens_sorted
