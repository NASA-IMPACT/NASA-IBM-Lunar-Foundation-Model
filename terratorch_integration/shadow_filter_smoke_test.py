#!/usr/bin/env python
"""Smoke test for terratorch_integration/shadow_filter.py

No model, no weights, no data download -- three synthetic 64x64 tiles
built by hand so the expected outcome is known ahead of time:

1. A bright, uniform tile with a small detection that has real local
   contrast (a dark ring around a bright floor, like a real crater under
   oblique light) -- should be KEPT by both gate variants.
2. A fully dark (shadow) tile with one huge detection (> 60% of the tile
   edge) on otherwise flat noise -- should be DROPPED by both gate
   variants (the diameter cap alone catches this one).
3. A fully dark (shadow) tile with a small detection whose interior is
   flat and no brighter than its surroundings (a "circle drawn on
   featureless shadow") -- should be DROPPED by variant A (tile+footprint
   shadow fraction + contrast) and by variant B (footprint shadow
   fraction alone).

This checks the gate logic and I/O contract, not detection quality
end-to-end -- see worker/lfm_schattenfilter.md in the founder's repo for
the numbers on a real NAC scene.

Usage:
    python shadow_filter_smoke_test.py
"""
from __future__ import annotations

import sys

import numpy as np

from shadow_filter import (
    DEFAULT_PARAMS_A,
    DEFAULT_PARAMS_B,
    contrast_ratio,
    filter_detections,
    shadow_fraction,
    tile_shadow_fraction,
)

TILE_PX = 64
SHADOW_THRESH = 50.0
RNG = np.random.default_rng(0)


def make_bright_tile_with_real_crater() -> tuple[np.ndarray, dict]:
    # Mid-brightness background, distinctly brighter interior -- gives a
    # contrast_ratio clearly > 1, like a sunlit crater floor/wall standing
    # out against its surroundings.
    tile = np.full((TILE_PX, TILE_PX), 80.0) + RNG.normal(0, 2, (TILE_PX, TILE_PX))
    cx, cy, diam = 32.0, 32.0, 10.0
    yy, xx = np.mgrid[0:TILE_PX, 0:TILE_PX]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = diam / 2.0
    tile[dist <= r] = 220.0
    det = {"x_m": cx, "y_m": cy, "diam_m": diam, "label": "real_crater"}
    return tile, det


def make_shadow_tile_with_giant_detection() -> tuple[np.ndarray, dict]:
    tile = np.full((TILE_PX, TILE_PX), 20.0) + RNG.normal(0, 1, (TILE_PX, TILE_PX))
    det = {"x_m": 32.0, "y_m": 32.0, "diam_m": 0.7 * TILE_PX, "label": "giant_on_shadow"}
    return tile, det


def make_shadow_tile_with_featureless_detection() -> tuple[np.ndarray, dict]:
    tile = np.full((TILE_PX, TILE_PX), 20.0) + RNG.normal(0, 1, (TILE_PX, TILE_PX))
    det = {"x_m": 32.0, "y_m": 32.0, "diam_m": 10.0, "label": "featureless_in_shadow"}
    return tile, det


def check(label: str, cond: bool) -> None:
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise AssertionError(label)


def main() -> int:
    # --- feature functions on tile 1 (real crater) ---
    tile1, det1 = make_bright_tile_with_real_crater()
    sf1 = shadow_fraction(tile1, det1["x_m"], det1["y_m"], det1["diam_m"], SHADOW_THRESH)
    cr1 = contrast_ratio(tile1, det1["x_m"], det1["y_m"], det1["diam_m"])
    tsf1 = tile_shadow_fraction(tile1, SHADOW_THRESH)
    check("bright tile: not shadowed", tsf1 < 0.1)
    check("real crater: low footprint shadow fraction", sf1 < 0.5)
    check("real crater: contrast_ratio > 1 (bright floor)", cr1 > 1.0)

    kept_a1 = filter_detections([det1], tile1, SHADOW_THRESH, DEFAULT_PARAMS_A)
    kept_b1 = filter_detections([det1], tile1, SHADOW_THRESH, DEFAULT_PARAMS_B)
    check("variant A keeps the real crater", len(kept_a1) == 1)
    check("variant B keeps the real crater", len(kept_b1) == 1)

    # --- tile 2 (giant detection on shadow, dropped by diameter alone) ---
    tile2, det2 = make_shadow_tile_with_giant_detection()
    kept_a2 = filter_detections([det2], tile2, SHADOW_THRESH, DEFAULT_PARAMS_A)
    kept_b2 = filter_detections([det2], tile2, SHADOW_THRESH, DEFAULT_PARAMS_B)
    check("variant A drops the giant detection", len(kept_a2) == 0)
    check("variant B drops the giant detection", len(kept_b2) == 0)

    # --- tile 3 (small, featureless detection in deep shadow) ---
    tile3, det3 = make_shadow_tile_with_featureless_detection()
    tsf3 = tile_shadow_fraction(tile3, SHADOW_THRESH)
    sf3 = shadow_fraction(tile3, det3["x_m"], det3["y_m"], det3["diam_m"], SHADOW_THRESH)
    check("tile 3 is almost fully shadowed", tsf3 > 0.9)
    check("detection 3 footprint is almost fully shadowed", sf3 > 0.95)

    kept_a3 = filter_detections([det3], tile3, SHADOW_THRESH, DEFAULT_PARAMS_A)
    kept_b3 = filter_detections([det3], tile3, SHADOW_THRESH, DEFAULT_PARAMS_B)
    check("variant A drops the featureless shadow detection", len(kept_a3) == 0)
    check("variant B drops the featureless shadow detection", len(kept_b3) == 0)

    print("\nAll shadow_filter smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
