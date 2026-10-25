"""Post-processing shadow filter for the NAC crater head under grazing
illumination.

Motivation: running the fine-tuned NAC crater head (`ni_lfm_v1_base` +
Faster R-CNN, `NAC_ni_lfm_ps8_s44.ckpt`) on a south-pole NAC image at
0.13 deg sun elevation (213 tiles, 256x256 px @ 1 m/px, score >= 0.5)
showed a sharp reliability collapse in deep self-shadow: 65.0% of
detections in tiles that are > 90% below an Otsu shadow threshold are
"implausible" (huge circles, up to 246 m, on essentially featureless
shadow) versus < 1.5% in every brighter shadow-fraction bin. See the
accompanying GitHub Discussion draft and the PR description for the full
write-up and numbers.

This module does NOT retrain or touch the model. It re-reads the tile
pixels the head already saw and applies a cheap geometric/photometric
gate to its output boxes: a detection whose footprint is (almost) fully
below the shadow threshold, sits in an already very dark tile, and shows
no local contrast against its surroundings, is dropped. A second, simpler
variant uses a single threshold on the detection's own footprint shadow
fraction.

Both variants were tuned on one half of the 213 evaluation tiles (even
tile index) against a heuristic "implausible" label (large diameter, or
fully-shadowed footprint in an almost fully shadowed tile -- itself not a
human-verified ground truth) and evaluated on the other half (odd tile
index). On the held-out half:

  - Variant "A" (multi-feature gate, default): precision of removed
    detections (fraction of removed detections that were actually
    heuristically implausible) = 100%, recall of removed detections
    (fraction of implausible detections actually removed) = 72.7%.
    False-removal rate on tiles with 30-70% shadow fraction (where the
    head's own detections are presumably mostly correct) = 0.62%.
  - Variant "B" (single threshold on footprint shadow fraction):
    precision = 81.0%, recall = 98.1%, false-removal rate on the
    30-70% bin = 2.10%.

Both keep >= 97% of detections in the 30-70% shadow bin. Variant A is
the safer default (near-zero risk of dropping a real detection); variant
B catches more of the implausible tail at a small extra risk. Neither
variant has been validated against human-labeled ground truth -- see the
module docstring caveat above and the PR description.

Usage on a single tile's detections (post score-threshold, pre any
further NMS)::

    from terratorch_integration.shadow_filter import (
        filter_detections, GateParams,
    )

    kept = filter_detections(detections, tile, shadow_thresh=54.3)

``detections`` is an iterable of dicts with at least ``x_m``/``y_m``
(box center in the tile's local pixel coordinates, 1 m/px) and
``diam_m`` (box diameter in meters/pixels). ``tile`` is the 2D grayscale
tile array (DN) the boxes were predicted on. ``shadow_thresh`` should be
computed once per scene (e.g. via Otsu over the whole scene, not per
tile -- per-tile Otsu makes "tile shadow fraction" tautologically ~50%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GateParams:
    """Parameters of one shadow-filter gate.

    ``variant="A"`` is the multi-feature gate (diameter + tile shadow
    fraction + footprint shadow fraction + contrast ratio). ``variant="B"``
    is the simpler gate: diameter cap plus a single threshold on the
    footprint shadow fraction alone.
    """

    variant: str
    diam_frac_thr: float = 0.6          # diameter > diam_frac_thr * tile edge -> always drop
    tile_shadow_thr: float = 0.9        # variant A only: tile shadow-fraction gate
    footprint_shadow_thr: float = 0.95  # variant A: detection footprint shadow-fraction gate
    contrast_thr: float = 1.0           # variant A: contrast-ratio gate (<=)
    single_shadow_thr: float = 0.98     # variant B: sole footprint shadow-fraction threshold
    annulus_factor: float = 1.8         # annulus outer radius = annulus_factor * detection radius


# Tuned on the even-index half of the 213-tile grazing-illumination
# evaluation set (see module docstring and the PR description for numbers).
DEFAULT_PARAMS_A = GateParams(variant="A")
DEFAULT_PARAMS_B = GateParams(variant="B")


def _circle_footprint(tile: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    """Pixel values inside the circle (cx, cy, radius), clipped to the tile."""
    h, w = tile.shape
    r = max(radius, 1.0)
    x0, x1 = int(max(cx - r, 0)), int(min(cx + r, w - 1))
    y0, y1 = int(max(cy - r, 0)), int(min(cy + r, h - 1))
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=tile.dtype)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return tile[y0:y1, x0:x1][mask]


def _annulus(tile: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    """Pixel values in the ring r_in < dist <= r_out around (cx, cy), clipped to the tile."""
    h, w = tile.shape
    x0, x1 = int(max(cx - r_out, 0)), int(min(cx + r_out, w - 1))
    y0, y1 = int(max(cy - r_out, 0)), int(min(cy + r_out, h - 1))
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=tile.dtype)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 <= r_out ** 2) & (dist2 > r_in ** 2)
    return tile[y0:y1, x0:x1][mask]


def shadow_fraction(tile: np.ndarray, cx: float, cy: float, diam_m: float,
                     shadow_thresh: float) -> float:
    """Fraction of pixels inside the detection's circumscribing circle that
    are below ``shadow_thresh``. Returns 1.0 (fully shadowed) if the circle
    falls entirely outside the tile -- a conservative default."""
    vals = _circle_footprint(tile, cx, cy, diam_m / 2.0)
    if vals.size == 0:
        return 1.0
    return float((vals < shadow_thresh).mean())


def tile_shadow_fraction(tile: np.ndarray, shadow_thresh: float) -> float:
    """Shadow fraction of the whole tile (the tile-context feature used by
    gate variant A)."""
    return float((tile.astype(np.float32) < shadow_thresh).mean())


def contrast_ratio(tile: np.ndarray, cx: float, cy: float, diam_m: float,
                    annulus_factor: float = 1.8) -> float:
    """Mean DN inside the detection divided by mean DN in an annulus just
    outside it (outer radius = ``annulus_factor`` * detection radius).
    A ratio <= 1 means the interior is not brighter than its surroundings --
    typical of a box that just circumscribes featureless shadow rather than
    a real crater's rim/floor contrast. Falls back to 1.0 when no annulus
    can be measured (detection at the tile edge)."""
    r = max(diam_m / 2.0, 1.0)
    inner = _circle_footprint(tile, cx, cy, r)
    if inner.size == 0:
        return 1.0
    outer = _annulus(tile, cx, cy, r, r * annulus_factor)
    if outer.size == 0:
        return 1.0
    mean_out = float(outer.mean())
    if mean_out <= 1e-6:
        return 1.0
    return float(inner.mean()) / mean_out


def _keep_single(cx: float, cy: float, diam_m: float, tile: np.ndarray,
                  shadow_thresh: float, tile_shadow_frac: float, tile_px: int,
                  params: GateParams) -> bool:
    if diam_m > params.diam_frac_thr * tile_px:
        return False

    if params.variant == "A":
        if tile_shadow_frac < params.tile_shadow_thr:
            return True
        sf = shadow_fraction(tile, cx, cy, diam_m, shadow_thresh)
        if sf < params.footprint_shadow_thr:
            return True
        cr = contrast_ratio(tile, cx, cy, diam_m, params.annulus_factor)
        return cr > params.contrast_thr

    if params.variant == "B":
        sf = shadow_fraction(tile, cx, cy, diam_m, shadow_thresh)
        return sf < params.single_shadow_thr

    raise ValueError(f"unknown gate variant: {params.variant!r}")


def filter_detections(detections: Iterable[dict], tile: np.ndarray,
                       shadow_thresh: float, params: GateParams = DEFAULT_PARAMS_A,
                       tile_px: int | None = None) -> list[dict]:
    """Apply the shadow-filter gate to one tile's detections.

    Each detection is a dict with at least ``x_m``/``y_m`` (box center in
    the tile's local pixel coordinates, 1 m/px) and ``diam_m`` (diameter in
    meters/pixels). Returns the kept subset, each entry a shallow copy of
    the input with ``kept=True`` added; dropped detections are simply
    omitted. To flag rather than drop, iterate and call ``_keep_single``
    directly instead.
    """
    tile_px = tile_px or tile.shape[-1]
    tsf = tile_shadow_fraction(tile, shadow_thresh)
    kept = []
    for det in detections:
        if _keep_single(det["x_m"], det["y_m"], det["diam_m"], tile,
                         shadow_thresh, tsf, tile_px, params):
            out = dict(det)
            out["kept"] = True
            kept.append(out)
    return kept


def params_for(variant: str) -> GateParams:
    if variant.upper() == "A":
        return DEFAULT_PARAMS_A
    if variant.upper() == "B":
        return DEFAULT_PARAMS_B
    raise ValueError(f"unknown variant: {variant!r}")
