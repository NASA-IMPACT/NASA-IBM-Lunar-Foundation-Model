# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import math
import os
import random

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from scipy import ndimage

import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from einops import rearrange

from ni_lfm.data.io_utils import load_netcdf


DEFAULT_BINNING_FILE = "ni_lfm/utils/tokenizer/trained/metadata_binning_config.csv"


def get_transform(mod_name, transforms_dict):
    return transforms_dict.get(mod_name, IdentityTransform())


def get_resample_mode(resample_mode: str) -> TF.InterpolationMode:
    """Returns the torchvision resampling mode for the given resample mode string.

    Args:
        resample_mode: Resampling mode string
    """
    if resample_mode == "bilinear":
        return TF.InterpolationMode.BILINEAR
    elif resample_mode == "bicubic":
        return TF.InterpolationMode.BICUBIC
    elif resample_mode == "nearest":
        return TF.InterpolationMode.NEAREST
    else:
        raise ValueError(f"Resample mode {resample_mode} is not supported.")


def _check_tensor(tensor: torch.Tensor):
    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")
    if tensor.ndim < 3:
        raise ValueError(f"Expected tensor to be of size (..., C, H, W). Got tensor.size() = {tensor.size()}")


def minmax_op(tensor: torch.Tensor, min_value: Sequence[float], max_value: Sequence[float]):

    _check_tensor(tensor)

    min_tensor = torch.as_tensor(min_value, dtype=tensor.dtype, device=tensor.device)
    max_tensor = torch.as_tensor(max_value, dtype=tensor.dtype, device=tensor.device)

    if min_tensor.ndim == 1:
        min_tensor = min_tensor.view(-1, 1, 1)
    if max_tensor.ndim == 1:
        max_tensor = max_tensor.view(-1, 1, 1)

    tensor_minmax = (tensor - min_tensor) / (max_tensor - min_tensor)
    tensor_minmax = torch.nan_to_num(tensor_minmax, nan=0.0)

    return tensor_minmax


def minmax_op_reverse(tensor: torch.Tensor, min_value: Sequence[float], max_value: Sequence[float]):

    _check_tensor(tensor)

    min_tensor = torch.as_tensor(min_value, dtype=tensor.dtype, device=tensor.device)
    max_tensor = torch.as_tensor(max_value, dtype=tensor.dtype, device=tensor.device)

    if min_tensor.ndim == 1:
        min_tensor = min_tensor.view(-1, 1, 1)
    if max_tensor.ndim == 1:
        max_tensor = max_tensor.view(-1, 1, 1)

    tensor_rev = (tensor * (max_tensor - min_tensor)) + min_tensor
    tensor_rev = torch.nan_to_num(tensor_rev, nan=0.0)

    return tensor_rev


def scale_data(
        data: torch.Tensor,
        scaler: str | None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        min_value: Sequence[float] | None = None,
        max_value: Sequence[float] | None = None,
        inplace: bool = True,
):
    """Scale *data* according to the *scaler* defined.

    Args:
        data: torch.Tensor to scale.
        scaler: one of "std", "minmax", "local_mean_std".
        mean: (Optional) mean value - only used with std scaler.
        std: (Optional) std value  - only used with std, local_mean_std scalers.
        min_value: (Optional) min value - only used with minmax scaler.
        max_value: (Optional) max value - only used with minmax scaler.
        inplace: whether "std" and "local_mean_std" scaling operation should be done inplace.
    """
    if scaler == "std":
        if mean is None or std is None:
            raise ValueError("Std scaler selected but no mean/std provided.")
        return TF.normalize(data, mean=mean, std=std, inplace=inplace)
    elif scaler == "minmax":
        if min_value is None or max_value is None:
            raise ValueError("Minmax scaler selected but no min/max provided.")
        return minmax_op(tensor=data, min_value=min_value, max_value=max_value)
    elif scaler == "local_mean_std":
        if std is None:
            raise ValueError("local_mean_std scaler selected but no std provided.")
        return TF.normalize(data, mean=torch.mean(data, dim=(-1, -2)), std=std, inplace=inplace)
    elif scaler is None:
        return data
    else:
        raise NotImplementedError(f"Scaler {scaler} not implemented.")


def unscale_data(
        data: torch.Tensor,
        scaler: str | None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        min_value: Sequence[float] | None = None,
        max_value: Sequence[float] | None = None,
        inplace: bool = False,
):
    """Revert scaling done in *data* according to the *scaler* defined.

    Args:
        data: torch.Tensor to scale.
        scaler: one of "std", "minmax".
        mean: (Optional) mean value - only used with std scaler.
        std: (Optional) std value - only used with std scaler.
        min_value: (Optional) mean value - only used with minmax scaler.
        max_value: (Optional) std value - only used with minmax scaler.
        inplace: whether "std" scaling operation should be done inplace.
    """
    if scaler == "std":
        if mean is None or std is None:
            raise ValueError("Std scaler selected but no mean/std provided.")
        return TF.normalize(
            data.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std], inplace=inplace,
        )
    elif scaler == "minmax":
        if min_value is None or max_value is None:
            raise ValueError("Minmax scaler selected but no min/max provided.")
        return minmax_op_reverse(data.clone(), min_value=min_value, max_value=max_value)
    elif scaler is None:
        return data
    else:
        raise NotImplementedError(f"Scaler {scaler} not implemented.")


def domain_unscale(
        data: torch.Tensor, domain: str, scaler_dict: dict[str, str | None] | None, stats: dict,
    ) -> torch.Tensor:
    """Helper to unscale domain data based on selected scalers."""
    if domain in ["vis", "vis_604", "uv", "dtm", "slope", "aspect", "nac", "dtm_3m", "slope_3m", "aspect_3m", "psr", "wac_mosaic"]:
        x_unscaled = unscale_data(
            data,
            scaler=scaler_dict[domain] if scaler_dict is not None else None,
            mean=stats[domain]["mean"],
            std=stats[domain]["std"],
            min_value=stats[domain]["min"],
            max_value=stats[domain]["max"],
        )
    else:
        print(f"Domain {domain} not found. No unscaling performed.")
        x_unscaled = data

    return x_unscaled


def one_hot_encoder(sample: torch.Tensor, num_classes: int):
    """Convert sample with *num_classes* to one-hot encoding.

    Args:
        sample: torch.Tensor with shape (1, H, W)
        num_classes: number of classes to consider in the on-hot encodign scheme.

    Returns:
        one-hot encoded sample with shape (num_classes, H, W)
    """
    sample = sample.long()
    sample = torch.squeeze(sample)  # Remove the band dimension -> (H, W)
    one_hot = F.one_hot(sample, num_classes=num_classes).float()  # (H, W, num_classes)
    one_hot = rearrange(one_hot, "y x c -> c y x")

    return one_hot


def fill_nan_nearest_neighbor(img: np.ndarray) -> np.ndarray:
    """Fill NaNs using nearest neighbors (channel-first, pixel-wise).

    Args:
        img: (C, H, W) float array with NaNs

    Returns:
        filled array
    """
    if img.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got {img.shape}")

    # valid mask over spatial dimensions
    valid = np.isfinite(img).all(axis=0)   # (H, W)

    if valid.all():
        return img

    invalid_mask = ~valid

    idx = ndimage.distance_transform_edt(invalid_mask, return_indices=True, return_distances=False)
    rr, cc = idx

    filled = img.copy()
    filled[:, invalid_mask] = img[:, rr[invalid_mask], cc[invalid_mask]]

    return filled


class AbstractTransform(ABC):
    """Base Transform class."""
    @abstractmethod
    def load(self, path: str) -> np.ndarray | str:
        pass

    @abstractmethod
    def preprocess(self, sample: np.ndarray | str) -> torch.Tensor:
        pass

    @abstractmethod
    def image_augment(
        self,
        sample: torch.Tensor,
        crop_coords: tuple,
        flip: bool,
        orig_size: tuple,
        target_size: tuple,
        rand_aug_idx: int | None,
        resample_mode: str | None = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(self, sample: torch.Tensor) -> torch.Tensor:
        pass


class ImageTransform(AbstractTransform):
    """Image Transform class."""
    @staticmethod
    def numpy_loader(path: str) -> np.ndarray:
        img = np.load(path)
        return img

    @staticmethod
    def netcdf_loader(path: str, channels: Sequence[str] | None = None) -> np.ndarray:
        data = load_netcdf(path=path, channels=channels)

        if np.isnan(data).any():
            print(f"NaN detected in data from file '{path}'")
        return data

    @staticmethod
    def image_hflip(img: torch.Tensor, flip: bool):
        """Crop and resize an image.

        Args:
            img: Image to crop and resize
            flip: Whether to flip the image

        Returns:
            Flipped image (if flip = True)
        """
        if flip:
            img = TF.hflip(img)
        return img

    @staticmethod
    def image_crop_and_resize(img: torch.Tensor, crop_coords: tuple, target_size: tuple, resample_mode: str = "bilinear"):
        """Crop and resize an image.

        Args:
            img: Image to crop and resize
            crop_coords: Coordinates of the crop (top, left, h, w)
            target_size: Coordinates of the resize (height, width)
            resample_mode: resample mode string

        Returns:
            Cropped and resized image
        """

        top, left, h, w = crop_coords
        resize_height, resize_width = target_size
        img = TF.crop(img, top, left, h, w)
        mode = get_resample_mode(resample_mode)
        img = TF.resize(img, size=[resize_height, resize_width], interpolation=mode)
        return img

    @staticmethod
    def image_resize(img: torch.Tensor, target_size: Sequence[int], resample_mode: str = "bilinear"):
        """Resize an image.

        Args:
            img: Image to crop and resize
            target_size: Coordinates of the resize (height, width)
            resample_mode: resample mode string

        Returns:
            Resized image
        """
        resize_height, resize_width = target_size
        mode = get_resample_mode(resample_mode)
        img = TF.resize(img, size=[resize_height, resize_width], interpolation=mode)
        return img

    @staticmethod
    def image_crop(img: torch.Tensor, crop_coords: tuple, **kwargs):
        """Crop an image.

        Args:
            img: Image to crop and resize
            crop_coords: Coordinates of the crop (top, left, h, w)

        Returns:
            Cropped image
        """

        top, left, h, w = crop_coords
        img = TF.crop(img, top, left, h, w)
        return img


class LunarTransform(ImageTransform):
    """Lunar data Transform class."""
    def __init__(
        self,
        mean: list,
        std: list,
        max: list,
        min: list,
        channels: list,
        scaler: str | None,
        pre_resize: int | None = None,
        resample_mode: str = "bilinear",  # One out of ["bilinear", "bicubic", "nearest"]
        **kwargs,
    ):
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.channels = channels
        self.scaler = scaler
        self.pre_resize = pre_resize
        self.resample_mode = resample_mode

    def load(self, path) -> np.ndarray:
        sample = self.netcdf_loader(path, self.channels)
        return sample

    def preprocess(self, sample: np.ndarray):
        img = torch.Tensor(sample)
        img = torch.nan_to_num(img, nan=0.0)   # safety replacement to avoid issues
        if self.pre_resize is not None:
            img = self.image_resize(img, [self.pre_resize] * 2, resample_mode=self.resample_mode)
        img = scale_data(img, mean=self.mean, std=self.std, min_value=self.min, max_value=self.max, scaler=self.scaler)

        return img

    def image_augment(
        self,
        img: torch.Tensor,
        crop_coords: tuple,
        flip: bool,
        target_size: tuple,
        **kwargs,
    ):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=self.resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample: torch.Tensor):
        return sample


class UntokLunarTransform(LunarTransform):
    """Untokenized Lunar Transform class."""

    def image_augment(self, img, crop_coords: tuple, flip: bool, **kwargs):
        img = self.image_crop(img, crop_coords)
        img = self.image_hflip(img, flip)
        return img


class SingleValueTransform(AbstractTransform):
    """Base class for transforms that handle single-value parameters with binning."""

    def __init__(self, return_raw: bool, shuffle: bool, aggregation_method: str):
        self.return_raw = return_raw
        self.shuffle = shuffle
        self.aggregation_method = aggregation_method
        self.bin_sizes = None

    def load(self, path):
        return Path(path).read_text()

    def _load_binning_config(self, csv_path: str) -> dict[str, float]:
        """Load binning configuration from CSV file with format: parameter,min,max,step,bins_n."""

        if not os.path.exists(csv_path):
            raise ValueError(f"Binning config file not found: {csv_path}")

        bin_sizes = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                param_name = row["parameter"]
                step = float(row["step"])
                bin_sizes[param_name] = step
        return bin_sizes

    def _parse_items(self, sample: str) -> list[list[str]]:
        """Parse str sample into list of [key, value_str] lists."""

        items = []
        for raw_line in sample.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"Invalid line: {raw_line!r}")
            key, value = line.split("=", 1)
            items.append([key.strip(), value.strip()])

        return items

    @staticmethod
    def _bin_value(field_name: str, value: float, bin_sizes: dict[str, float], decimals: int | None = None) -> str:
        """Bin a value into the tokenizer's range-token format.

        Args:
            field_name: Field name
            value: Raw numeric value
            bin_sizes: Dictionary mapping field names to bin sizes

        Returns:
            Binned token string in format "FIELD=start-->end"
        """
        if field_name not in bin_sizes:
            raise KeyError(f"No bin size configured for field: {field_name}")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Value must be finite, got {value!r}")

        bin_size = bin_sizes[field_name]
        lower = math.floor(numeric_value / bin_size) * bin_size
        upper = lower + bin_size

        # Format with appropriate decimal places based on bin size
        if decimals is None:
            if bin_size >= 1.0:
                decimals = 0
            elif bin_size >= 0.1:
                decimals = 1
            else:
                decimals = 2

        return f"{field_name}={lower:.{decimals}f}-->{upper:.{decimals}f}"

    def preprocess(self, sample: str):
        return sample

    def image_augment(self, sample, **kwargs):
        return sample

    def _parse_value(self, value_str: str) -> float | list[float] | None:
        """Parse a value string, handling multiple values and ERROR cases."""
        value_str = value_str.strip()

        if value_str.upper() in ("ERROR", "MISSING", "NAN", ""):
            return None

        # Check if multiple values (comma-separated)
        if "," in value_str:
            values = []
            for v in value_str.split(","):
                v = v.strip()
                if v.lower() == "nan" or not v:
                    continue
                try:
                    values.append(float(v))
                except ValueError:
                    continue

            if len(values) == 0:
                return None

            return values
        else:
            return float(value_str)

    def parse_sample(self, sample: str, var_list: list | None = None) -> list[list]:
        """Parse and filter keys to specified vars only."""
        items = self._parse_items(sample)

        processed_items = []
        for key, value_str in items:
            if var_list is not None and key not in var_list:
                continue

            parsed_value = self._parse_value(value_str)
            processed_items.append([key, parsed_value])

        return processed_items

    def _aggregate_values(self, values: list[float]) -> float | list[float]:
        """Aggregate multiple values using the specified method. """
        if len(values) == 0:
            raise ValueError("Cannot aggregate empty list of values")

        if self.aggregation_method == "min":
            return min(values)
        elif self.aggregation_method == "max":
            return max(values)
        elif self.aggregation_method == "mean":
            return sum(values) / len(values)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def sample_to_str(self, items: list[list], decimals: int | None) -> list[str]:
        """Format items to binned text strings or MISSING markers."""
        result = []
        for key, value in items:
            if value is None:
                result.append(f"{key}=MISSING")
            elif isinstance(value, list):
                binned = self._bin_value(key, self._aggregate_values(value), self.bin_sizes, decimals)
                result.append(binned)
            else:
                result.append(self._bin_value(key, value, self.bin_sizes, decimals))

        return result


class MetadataTransform(SingleValueTransform):
    """Metadata transform for observation parameters."""

    METADATA_VARS = (
        "SS_GROUND_AZIMUTH",
        "SS_LAT",
        "SS_LON",
        "PHASE_ANG",
        "INC_ANG",
        "EM_ANG",
        "C_LON",
        "C_LAT",
    )

    def __init__(self, binning_config_path: str = DEFAULT_BINNING_FILE, return_raw: bool = False, shuffle: bool = True):
        """Initialize MetadataTransform.

        Args:
            binning_config_path: Path to CSV file with binning configuration.
            return_raw: whether to return raw list of lists or binned text strings.
            shuffle: whether to shuffle the order of metadata entries.
        """
        super().__init__(return_raw, shuffle, aggregation_method="mean")  # aggregation method not currently used
        all_bin_sizes = self._load_binning_config(binning_config_path)
        assert set(self.METADATA_VARS).issubset(set(all_bin_sizes.keys()))

        self.bin_sizes = {v: all_bin_sizes[v] for v in self.METADATA_VARS}
        self.decimals = 2  # Metadata vocabulary has 2 decimals
        self.selected_vars = list(self.METADATA_VARS)

    def postprocess(self, sample: str):

        items = self.parse_sample(sample, var_list=self.selected_vars)

        if self.return_raw:
            return items

        if self.shuffle:
            random.shuffle(items)

        return self.sample_to_str(items, decimals=self.decimals)


class StaticMapsTransform(SingleValueTransform):
    """Transform for static map variables."""

    def __init__(
        self,
        selected_vars: list[str] | None = None,
        aggregation_method: str = "mean",
        binning_config_path: str = DEFAULT_BINNING_FILE,
        return_raw: bool = False,
        shuffle: bool = True,
    ):
        """Initialize StaticMapsTransform.

        Args:
            selected_vars: List of variable names to tokenize. If None, uses all variables except 10 metadata variables
            aggregation_method: Method to aggregate multiple values: "min", "mean", or "max"
            binning_config_path: Path to CSV file with binning configuration.
            return_raw: whether to return raw list of lists or binned text strings.
            shuffle: whether to shuffle the order of entries.
        """
        super().__init__(return_raw, shuffle, aggregation_method)

        all_bin_sizes = self._load_binning_config(binning_config_path)
        exclude = MetadataTransform.METADATA_VARS  # Exclude metadata variables

        if selected_vars is None:
            self.selected_vars = [v for v in all_bin_sizes if v not in exclude]
            self.bin_sizes = {v: step for v, step in all_bin_sizes.items() if v not in exclude}
        else:
            self.selected_vars = selected_vars
            self.bin_sizes = {v: all_bin_sizes[v] for v in self.selected_vars if v in all_bin_sizes}

    def postprocess(self, sample: str):

        items = self.parse_sample(sample, var_list=self.selected_vars)

        if self.return_raw:
            return items

        if self.shuffle:
            random.shuffle(items)

        return self.sample_to_str(items, decimals=None)


class IdentityTransform(AbstractTransform):
    """Identity Transform class."""
    def load(self, path):
        raise NotImplementedError("IdentityTransform does not support loading")

    def preprocess(self, sample, **kwargs):
        return sample

    def image_augment(self, sample, **kwargs):
        return sample

    def postprocess(self, sample, **kwargs):
        return sample
