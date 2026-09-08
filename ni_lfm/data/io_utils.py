import io

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from omegaconf import DictConfig, ListConfig, OmegaConf


MISSING_FILE_STR = "__MISSING__"


def _fix_shape(data: np.ndarray, ndims=3):
    if data.ndim > ndims:
        data = data.squeeze(axis=-3)  # assuming (C, 1, H, W)
    return data


def load_netcdf(path: str | Path | io.BytesIO, channels: Sequence[str] | None) -> np.ndarray:
    """Load netcdf files from path or bytes."""
    with h5py.File(path, "r") as f:
        if channels is None:
            channels = [c for c in f.keys() if c not in ["y", "x"]]
        data = np.stack([f[ch][...] for ch in channels], axis=0)
        data = _fix_shape(data)  # Temporary hack to fix inconsistent shapes in .nc files

    return data


def nc_bytes_loader(value: bytes, channels: list[str] | None = None):
    with io.BytesIO(value) as bio:
        data = load_netcdf(path=bio, channels=channels)

    return data


def np_bytes_loader(value: bytes):
    return np.load(io.BytesIO(value), allow_pickle=True)


def load_parquet_file(
    file_path: str | Path,
    remove_key_mod_nan: bool = True,
    filter_missing_paths: bool = False,
    columns: list[str] | None = None,
):
    """Load parquet index file and filter the dataframe to exclude samples with missing data.

    Args:
        file_path: Path to the parquet file
        remove_key_mod_nan: If True, filter out samples where optical, dtm, slope or aspect has NaNs.\
        filter_missing_paths: If True, filter out samples where modality file paths are None/NaN
        columns: List of column names to check for None/NaN paths

    Returns:
        Filtered pandas DataFrame
    """
    df = pd.read_parquet(file_path)
    original_len = len(df)

    # Filter based on ALLOW_NANS_OPTICAL_DTM_SLOPE_ASPECT flag
    if remove_key_mod_nan and "ALLOW_NANS_OPTICAL_DTM_SLOPE_ASPECT" in df.columns:
        df = df[~df["ALLOW_NANS_OPTICAL_DTM_SLOPE_ASPECT"]].reset_index(drop=True)
        filtered_by_flag = original_len - len(df)
        if filtered_by_flag > 0:
            print(f"Filtered {filtered_by_flag} samples with ALLOW_NANS_OPTICAL_DTM_SLOPE_ASPECT=True "
                  f"({filtered_by_flag / original_len * 100:.1f}%)")

    # Filter out samples with None/NaN file paths
    if filter_missing_paths:
        assert columns is not None, "One should provide columns to filter if filter_missing_paths=True."

        before_filter = len(df)
        # Keep only rows where ALL specified modality columns are not None/NaN
        for col in columns:
            if col in df.columns:
                df = df[df[col].notna()].reset_index(drop=True)

        filtered_by_paths = before_filter - len(df)
        if filtered_by_paths > 0:
            print(f"Filtered {filtered_by_paths} samples with missing file paths in {columns} "
                  f"({filtered_by_paths / before_filter * 100:.1f}%)")

    final_len = len(df)
    total_filtered = original_len - final_len
    if total_filtered > 0:
        print(f"Total filtered: {original_len} -> {final_len} samples "
              f"(removed {total_filtered}, {total_filtered / original_len * 100:.1f}%)")

    return df
