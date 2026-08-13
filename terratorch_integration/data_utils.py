"""Shared helpers for the TerraTorch data adapters."""

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch


def load_nc_band(path: str | Path, band_name: str = "band_data") -> np.ndarray:
    """Load a single band from a NetCDF4/HDF5 `.nc` or NumPy `.npy` tile as a 2-D array.

    Args:
        path: Path to the `.nc` / `.npy` file (`band_name` is ignored for `.npy`).
        band_name: HDF5 dataset key to read (e.g. `"band_data"` for the
            NAC/DTM tiles, `"data"` for the IMP segmentation tiles).

    Returns:
        A float32 array of shape `(H, W)`.  If the stored band is
        `(1, H, W)` (single-band-first convention) the leading singleton
        is squeezed away.
    """
    if Path(path).suffix.lower() == ".npy":
        # Released NAC crater tiles ship as .npy; pretraining tiles are .nc.
        arr = np.asarray(np.load(path), dtype=np.float32)
    else:
        with h5py.File(path, "r") as f:
            arr = np.asarray(f[band_name], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


class D4Transform:
    """Random symmetry from the D4 group (8 elements).

    Applies a uniformly-random combination of {0°, 90°, 180°, 270°} rotation
    and an optional horizontal flip to both the image and its dense targets,
    keeping pixel-to-pixel alignment.  Meant to be used as the `transforms`
    argument of a dataset that returns per-sample dicts.

    Recognised keys:

    * `"image"` — `(C, H, W)` float tensor.
    * `mask_key` (default `"mask"`) — `(H, W)` or `(1, H, W)` int
      tensor.  Missing keys are ignored.

    Only sensible for square inputs (H == W) because rotations by 90° / 270°
    swap the last two dimensions.

    Args:
        mask_key: Sample-dict key that holds the segmentation mask (must
            match the `mask_output_tag` used by the dataset).
        p: Probability of applying a *non-identity* transform.  `1.0`
            means always sample one of the 8 group elements uniformly;
            `0.0` disables the transform entirely.
    """

    _NUM_ROTATIONS: int = 4  # 0°, 90°, 180°, 270°

    def __init__(self, mask_key: str = "mask", p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.mask_key = mask_key
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0.0 or torch.rand(()).item() >= self.p:
            return sample
        k = int(torch.randint(self._NUM_ROTATIONS, (1,)).item())
        flip = bool(torch.randint(2, (1,)).item())

        image = sample.get("image")
        if isinstance(image, torch.Tensor):
            if k:
                image = torch.rot90(image, k=k, dims=(-2, -1))
            if flip:
                image = torch.flip(image, dims=(-1,))
            sample["image"] = image

        mask = sample.get(self.mask_key)
        if isinstance(mask, torch.Tensor):
            if k:
                mask = torch.rot90(mask, k=k, dims=(-2, -1))
            if flip:
                mask = torch.flip(mask, dims=(-1,))
            sample[self.mask_key] = mask

        return sample


class D4DetectionTransform:
    """Random symmetry from the D4 group for detection samples.

    Applies a uniformly-random combination of {0°, 90°, 180°, 270°} rotation
    and an optional horizontal flip to the image *and* all bounding boxes
    in a per-sample dict, keeping image/box alignment.  Rotations swap the
    spatial axes, so this transform requires `H == W` (square inputs).

    Recognised keys:

    * `image_keys` — tensors with a spatial `(..., H, W)` layout (e.g.
      `"image"`, `"vis"`, `"nac"`, `"dtm_3m"`).  Missing keys are
      silently skipped, so the same transform instance can serve datasets
      that produce different subsets.
    * `boxes_key` (default `"boxes"`) — `(N, 4)` float tensor in
      `xyxy` format with pixel coordinates in `[0, S]`.
    * `masks_key` (default `"masks"`) — optional per-instance mask
      tensors.  Accepted either as a stacked `(N, H, W)` tensor or a
      list of `(H, W)` tensors (placeholder `(0, 0)` entries used by
      the crater datasets are passed through unchanged).

    Args:
        image_keys: Sample-dict keys whose values are spatial tensors that
            should be rotated / flipped alongside the boxes.  Defaults to
            `("image", "vis")`.
        boxes_key: Sample-dict key holding the `xyxy` boxes tensor.
        masks_key: Sample-dict key holding per-instance masks.  Set to
            `None` to skip mask handling entirely.
        p: Probability of applying a *non-identity* transform.  `1.0`
            (default) always samples one of the 8 group elements; `0.0`
            disables the transform.

    Notes:
        Craters are approximately isotropic and D4-invariant under lunar
        NAC/WAC imaging, so all 8 group elements produce plausible-looking
        training samples.  Lighting-direction cues (crater rim shadows) do
        get transformed, but Faster R-CNN / DETR don't rely on physically
        consistent shadow direction, so this is a safe augmentation.
    """

    _NUM_ROTATIONS: int = 4

    def __init__(
        self,
        image_keys: Iterable[str] = ("image", "vis"),
        boxes_key: str = "boxes",
        masks_key: str | None = "masks",
        p: float = 1.0,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.image_keys: tuple[str, ...] = tuple(image_keys)
        self.boxes_key = boxes_key
        self.masks_key = masks_key
        self.p = p

    @staticmethod
    def _rotate_boxes(boxes: torch.Tensor, k: int, size: int) -> torch.Tensor:
        """Rotate `xyxy` boxes by `k * 90°` counter-clockwise on an `SxS` canvas.

        Matches `torch.rot90(image, k=k, dims=(-2, -1))`, where continuous
        pixel coordinate `(x, y)` maps to:

        * k=0: identity
        * k=1: `(y, S - x)`          (CCW 90°)
        * k=2: `(S - x, S - y)`      (180°)
        * k=3: `(S - y, x)`          (CCW 270°)
        """
        if k == 0 or boxes.numel() == 0:
            return boxes
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        s = float(size)
        if k == 1:
            new = torch.stack([y1, s - x2, y2, s - x1], dim=-1)
        elif k == 2:
            new = torch.stack([s - x2, s - y2, s - x1, s - y1], dim=-1)
        else:  # k == 3
            new = torch.stack([s - y2, x1, s - y1, x2], dim=-1)
        return new

    @staticmethod
    def _flip_boxes(boxes: torch.Tensor, size: int) -> torch.Tensor:
        """Horizontal-flip `xyxy` boxes on an `S` -wide canvas."""
        if boxes.numel() == 0:
            return boxes
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        s = float(size)
        return torch.stack([s - x2, y1, s - x1, y2], dim=-1)

    def _apply_image(self, t: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
        if k:
            t = torch.rot90(t, k=k, dims=(-2, -1))
        if flip:
            t = torch.flip(t, dims=(-1,))
        return t

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.p <= 0.0 or torch.rand(()).item() >= self.p:
            return sample
        k = int(torch.randint(self._NUM_ROTATIONS, (1,)).item())
        flip = bool(torch.randint(2, (1,)).item())
        if k == 0 and not flip:
            return sample

        # Discover the canvas size from whichever image key is present.
        # Rotations swap H and W, so we require H == W.
        size: int | None = None
        for key in self.image_keys:
            t = sample.get(key)
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                h, w = int(t.shape[-2]), int(t.shape[-1])
                if h != w:
                    raise ValueError(
                        f"D4DetectionTransform requires square inputs, "
                        f"got '{key}' with shape {tuple(t.shape)}."
                    )
                if size is None:
                    size = h
                elif size != h:
                    raise ValueError(f"D4DetectionTransform inputs must share the same size, got {size} and{h}.")
        if size is None:
            return sample  # nothing spatial to transform

        # ------------------------------------------------------------------
        # Rotate / flip every present image tensor
        # ------------------------------------------------------------------
        for key in self.image_keys:
            t = sample.get(key)
            if isinstance(t, torch.Tensor):
                sample[key] = self._apply_image(t, k, flip)

        # Rotate / flip boxes
        boxes = sample.get(self.boxes_key)
        if isinstance(boxes, torch.Tensor):
            boxes = self._rotate_boxes(boxes, k, size)
            if flip:
                boxes = self._flip_boxes(boxes, size)
            sample[self.boxes_key] = boxes

        # Rotate / flip masks (per-instance, if present with real content)
        if self.masks_key is not None:
            masks = sample.get(self.masks_key)
            if isinstance(masks, torch.Tensor) and masks.ndim >= 2:
                sample[self.masks_key] = self._apply_image(masks, k, flip)
            elif isinstance(masks, list):
                new_masks = []
                for m in masks:
                    if isinstance(m, torch.Tensor) and m.ndim >= 2 and 0 not in m.shape:
                        new_masks.append(self._apply_image(m, k, flip))
                    else:
                        new_masks.append(m)
                sample[self.masks_key] = new_masks

        return sample
