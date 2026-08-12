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
# --------------------------------------------------------
# Some functions are based on the timm code base
# https://github.com/huggingface/pytorch-image-models
# --------------------------------------------------------

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def softmax1(tensor):
    # See https://www.evanmiller.org/attention-is-off-by-one.html
    return F.pad(tensor, (0, 1)).softmax(dim=-1)[..., :-1]


def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d.

    Returns positional embedding of shape (1, N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32)  # Shape (N,)
    assert embed_dim % 2 == 0, "Embed dimension must be divisible by 2 for 1D sin-cos position embedding"
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  # Shape (D/2,)
    omega = 1.0 / (temperature**omega)
    out = torch.einsum("n,d->nd", [arange, omega])  # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0)  # Shape (1, N, D)
    return pos_emb


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings as used in MoCo-v3.

    Returns positional embedding of shape (1, N, D) where N = W*H
    """
    grid_w = torch.arange(w, dtype=torch.float32)  # Shape (W,)
    grid_h = torch.arange(h, dtype=torch.float32)  # Shape (H, )
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")  # Shapes (W, H)
    assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  # Shape (D/4,)
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("n,d->nd", [grid_w.reshape(-1), omega])  # Outer product, shape (W*H, D/4)
    out_h = torch.einsum("n,d->nd", [grid_h.reshape(-1), omega])  # Outer product, shape (W*H, D/4)
    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    ).unsqueeze(0)  # Shape (1, W*H, D)
    return pos_emb


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Implementation from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob if self.drop_prob is not None else 0.0, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term."""

    def __init__(self, normalized_shape: int, eps=1e-5, bias=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """Implement SwiGLU + other gated feed-forward layers from Noam Shazeer's paper https://arxiv.org/abs/2002.05202."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        # If gated, multiply hidden_dim by 2/3 to account for extra matmul
        hidden_features = int(2 * (hidden_features or in_features) / 3)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)) * self.fc3(x))
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        # SDPA expects attn_mask broadcastable to (B, num_heads, N, N).
        if mask is not None:
            # In NILunarFM: True = "do not attend" -> PyTorch: True = model *should* attend -> invert
            attn_mask = ~mask[:, None, :, :].to(dtype=torch.bool, device=x.device)
        else:
            attn_mask = None

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        x = y.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, H, N, Dh)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, H, M, Dh)

        if mask is not None:
            # In NILunarFM: True = "do not attend" -> PyTorch: True = model *should* attend -> invert
            if mask.dim() == 2:  # (B, M) key-padding mask
                attn_mask = ~mask[:, None, None, :].to(dtype=torch.bool, device=x.device)
            elif mask.dim() == 3:  # (B, N, M) attention mask
                attn_mask = ~mask[:, None, :, :].to(dtype=torch.bool, device=x.device)
            else:
                raise ValueError("mask must be (B, M) or (B, N, M) with True = 'do not attend'")
        else:
            attn_mask = None

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        x = y.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NormAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        norm_layer=nn.LayerNorm,
        attn_drop=0.0,
        proj_drop=0.0,
        allow_zero_attn=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.allow_zero_attn = allow_zero_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim)
        self.k_norm = norm_layer(head_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)  # Unsqueeze for multi-head
            attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

        if self.allow_zero_attn:
            attn = softmax1(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NormCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        norm_layer=nn.LayerNorm,
        attn_drop=0.0,
        proj_drop=0.0,
        allow_zero_attn=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.allow_zero_attn = allow_zero_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim)
        self.k_norm = norm_layer(head_dim)

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b n m -> b 1 n m")  # Unsqueeze / reshape for multi-head
            attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

        if self.allow_zero_attn:
            attn = softmax1(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        gated_mlp=False,
        qk_norm=False,
        allow_zero_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if not qk_norm:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.attn = NormAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
                allow_zero_attn=allow_zero_attn,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if not gated_mlp:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=mlp_bias,
                drop=drop,
            )
        else:
            self.mlp = GatedMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=mlp_bias,
            )

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        gated_mlp=False,
        qk_norm=False,
        allow_zero_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if not qk_norm:
            self.self_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.cross_attn = CrossAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.self_attn = NormAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
                allow_zero_attn=allow_zero_attn,
            )
            self.cross_attn = NormCrossAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
                allow_zero_attn=allow_zero_attn,
            )

        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if not gated_mlp:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=mlp_bias,
                drop=drop,
            )
        else:
            self.mlp = GatedMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                bias=mlp_bias,
            )

    def forward(self, x, context, sa_mask=None, xa_mask=None):
        x = x + self.drop_path(self.self_attn(self.norm1(x), sa_mask))
        x = x + self.drop_path(self.cross_attn(self.query_norm(x), self.context_norm(context), xa_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def init_enc_dec_embeddings(
    cfg,
    modality_info: dict[str, dict[str, Any]],
    in_domains: list[str],
    out_domains: list[str],
):
    """Initialize encoder and decoder embeddings using fixed, runtime and user-defined parameters.

    Args:
        cfg: Hydra configuration object.
        modality_info: Runtime modality info from setup_modality_info_tm()
        in_domains: List of input modality names
        out_domains: List of output modality names

    Returns:
        tuple of (encoder_embeddings, decoder_embeddings)

    Example:
        >>> # With config overrides (training)
        >>> enc_emb, dec_emb = init_enc_dec_embeddings(
        ...     cfg, modality_info, ["tok_vis"], ["tok_vis"]
        ... )
        >>>
        >>> # Without config overrides (generation)
        >>> enc_emb, dec_emb = init_enc_dec_embeddings(
        ...     None, modality_info, ["tok_vis"], ["tok_vis"]
        ... )
    """
    encoder_embeddings = {}
    decoder_embeddings = {}

    # Initialize encoder embeddings
    for mod_name in in_domains:
        info = modality_info[mod_name]

        if info.get("encoder_embedding") is not None:
            encoder_embeddings[mod_name] = _instantiate_embedding(
                modality_info=info,
                cfg=cfg,
                mod_name=mod_name,
                param_type="encoder",
            )

    # Initialize decoder embeddings
    for mod_name in out_domains:
        info = modality_info[mod_name]

        if info.get("decoder_embedding") is not None:
            decoder_embeddings[mod_name] = _instantiate_embedding(
                modality_info=info,
                cfg=cfg,
                mod_name=mod_name,
                param_type="decoder",
            )

    return encoder_embeddings, decoder_embeddings


def _instantiate_embedding(modality_info: dict, cfg, mod_name: str, param_type: str):
    """Instantiate an embedding using fixed, runtime and user-defined parameters.

    1. Starts with base kwargs from MODALITY_INFO
    2. Adds dataset + runtime params from modality_info for image modalities
    3. Applies user config overrides
    5. Instantiates the embedding class

    Args:
        modality_info: Modality info dict with dataset parameters
        cfg: Hydra config (can be None)
        mod_name: Modality name
        param_type: "encoder" or "decoder"

    Returns:
        Instantiated embedding module
    """

    embedding_cls = modality_info[f"{param_type}_embedding"]
    embedding_kwargs = modality_info[f"{param_type}_kwargs"]
    user_params = cfg.model.get(f"{param_type}_embedding_params", {})

    dataset_params = ["num_channels", "patch_size", "input_size"]
    tokenizer_params = ["codebook_size", "num_codebooks"]

    # These cannot be overriden by user
    restricted_params = dataset_params + tokenizer_params + ["vocab_size", "dim_tokens"]

    # Start with default kwargs from modality_info
    final_kwargs = deepcopy(embedding_kwargs)

    # Add dataset + runtime parameters from modality_info for image modalities
    if modality_info["type"] == "img":
        for key in dataset_params:
            final_kwargs[key] = modality_info[key]

        if modality_info["pretokenized"]:
            for key in tokenizer_params:
                if key == "codebook_size":
                    final_kwargs["vocab_size"] = modality_info[key]
                else:
                    final_kwargs[key] = modality_info[key]

    config_params = {}

    # Apply _global_ overrides
    if "_global_" in user_params:
        config_params.update(user_params["_global_"])

    # Apply modality-specific overrides
    if mod_name in user_params:
        config_params.update(user_params[mod_name])

    # Filter None values
    config_params = {k: v for k, v in config_params.items() if v is not None}

    restricted_found = set(config_params.keys()) & set(restricted_params)
    if restricted_found:
        raise ValueError(
            f"Cannot override parameters for {mod_name} {param_type} embedding: {sorted(restricted_found)}.\n"
            f"They are automatically set from dataset info.\nRestricted parameters: {sorted(restricted_params)}",
        )
    if config_params:
        final_kwargs.update(config_params)

    return embedding_cls(**final_kwargs)
