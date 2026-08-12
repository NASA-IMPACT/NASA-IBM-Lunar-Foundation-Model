# Copyright 2024 IBM and the Lunar-FM authors.
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

"""FlexiViT PI-resize for patch-embedding kernels.

Reference: Beyer et al., "FlexiViT: One Model for All Patch Sizes" (CVPR 2023).
JAX reference implementation:
https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py
"""

import torch
import torch.nn.functional as F


_PINV_CACHE: dict[tuple[int, int, int, int, torch.dtype, str], torch.Tensor] = {}


def _bilinear_resize(x: torch.Tensor, new_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize a 2D tensor to `new_hw`.

    Uses `align_corners=False` to match the half-pixel convention of
    `tf.image.resize` / `jax.image.resize` used in the FlexiViT reference.
    """
    return F.interpolate(
        x[None, None],
        size=new_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _build_resize_matrix(old_hw: tuple[int, int], new_hw: tuple[int, int]) -> torch.Tensor:
    """Construct the bilinear resize operator `B` from `old_hw` to `new_hw`.

    Each column of `B` is the flattened bilinear resize of a one-hot basis
    vector on the old grid. Shape `(prod(new_hw), prod(old_hw))`.

    Always built on CPU in float64 for numerical stability; this is a one-time
    O(n_old) cost and the result is cached. (MPS in particular does not support
    float64, so doing the construction on the device dtype directly is not
    portable.)
    """
    h_old, w_old = old_hw
    n_old = h_old * w_old
    n_new = new_hw[0] * new_hw[1]
    cols = torch.empty((n_new, n_old), device="cpu", dtype=torch.float64)
    basis = torch.zeros((h_old, w_old), device="cpu", dtype=torch.float64)
    for i in range(n_old):
        r, c = divmod(i, w_old)
        basis.zero_()
        basis[r, c] = 1.0
        cols[:, i] = _bilinear_resize(basis, new_hw).reshape(-1)
    return cols


def get_resize_pinv(
    old_hw: tuple[int, int],
    new_hw: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the cached pseudo-inverse used to PI-resize a patch kernel.

    Given old patch size `old_hw` and new patch size `new_hw`, returns a
    matrix `P` of shape `(prod(new_hw), prod(old_hw))` such that for any
    flattened kernel `k_old` of length `prod(old_hw)`, the resampled kernel
    is `k_new = P @ k_old` reshaped to `new_hw`. The matrix has no
    parameters and is non-learnable.

    Construction policy: `B` and `pinv(B.T)` are computed once on CPU in
    float64 (cached separately under a CPU key). Per-device copies are then
    materialized on demand by casting + `.to(device)` and cached under the
    requested `(device, dtype)` key. This makes the math portable across
    backends that don't support float64 (MPS in particular) while keeping the
    pinv numerically stable.
    """
    key = (old_hw[0], old_hw[1], new_hw[0], new_hw[1], dtype, str(device))
    cached = _PINV_CACHE.get(key)
    if cached is not None:
        return cached

    # CPU/float64 master copy, computed once per (old_hw, new_hw).
    cpu_key = (old_hw[0], old_hw[1], new_hw[0], new_hw[1], torch.float64, "cpu")
    P_cpu64 = _PINV_CACHE.get(cpu_key)
    if P_cpu64 is None:
        # `B` has shape (n_new, n_old). Match the reference which computes
        # `resize_mat_pinv = np.linalg.pinv(resize_mat.T)` so the result has
        # shape (n_new, n_old) and is applied as `k_new = P @ k_old`.
        B = _build_resize_matrix(old_hw, new_hw)
        P_cpu64 = torch.linalg.pinv(B.T)
        _PINV_CACHE[cpu_key] = P_cpu64

    P = P_cpu64.to(device=device, dtype=dtype)
    _PINV_CACHE[key] = P
    return P


def pi_resize_patch_embed(
    weight: torch.Tensor,
    old_patch_size: tuple[int, int],
    new_patch_size: tuple[int, int],
    num_channels: int,
) -> torch.Tensor:
    """PI-resize a patch-embedding `nn.Linear` weight to a new patch size.

    Matches the projection used in :class:`ImageEncoderEmbedding`:

        `proj = nn.Linear(C * PH * PW, D, bias=False)`

    and the einops rearrange `"b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)"`.
    In that pattern the einops name `d` is the input *channel* axis — so the
    flattened per-patch feature has order `(ph, pw, c)` with channels
    *innermost* (fastest-varying). PI-resize is applied along the spatial
    `(ph, pw)` axes only; the channel axis is untouched.

    Args:
        weight: `nn.Linear` weight of shape `(D, PH_old * PW_old * C)`.
        old_patch_size: `(PH_old, PW_old)` of the input `weight`.
        new_patch_size: `(PH_new, PW_new)` for the returned weight.
        num_channels: `C` — the number of input channels for this modality.

    Returns:
        Tensor of shape `(D, PH_new * PW_new * C)`, on the same device and
        dtype as `weight`.
    """
    if tuple(old_patch_size) == tuple(new_patch_size):
        return weight

    D, in_features = weight.shape
    expected_in = num_channels * old_patch_size[0] * old_patch_size[1]
    if in_features != expected_in:
        raise ValueError(
            f"weight.shape[1]={in_features} does not match "
            f"old_patch_size{tuple(old_patch_size)} * num_channels({num_channels})={expected_in}"
        )

    P = get_resize_pinv(
        old_hw=tuple(old_patch_size),
        new_hw=tuple(new_patch_size),
        device=weight.device,
        dtype=weight.dtype,
    )

    PH_old, PW_old = old_patch_size
    PH_new, PW_new = new_patch_size

    # The flat dim ordering is (ph, pw, c). Reshape to (D, PH_old, PW_old, C),
    # collapse spatial -> (D, PH_old*PW_old, C), apply P along the spatial axis,
    # then reshape back keeping channels innermost.
    w = weight.reshape(D, PH_old, PW_old, num_channels)
    w = w.reshape(D, PH_old * PW_old, num_channels)
    # w_new[d, j, c] = sum_i P[j, i] * w[d, i, c]
    w_new = torch.einsum("ji,dic->djc", P, w)
    w_new = w_new.reshape(D, PH_new, PW_new, num_channels)
    return w_new.reshape(D, PH_new * PW_new * num_channels)


def clear_pinv_cache() -> None:
    """Drop all cached pinv matrices. Useful after device changes or in tests."""
    _PINV_CACHE.clear()
