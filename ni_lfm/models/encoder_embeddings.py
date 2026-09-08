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

import torch
import torch.nn as nn

from einops import rearrange, repeat

from ni_lfm.models.codebook_fusion import UnifiedCodebookFusion
from ni_lfm.models.flexivit import pi_resize_patch_embed
from ni_lfm.models.lfm_utils import build_1d_sincos_posemb, build_2d_sincos_posemb, pair


class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
        padding_idx: Padding index for word embedding
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim_tokens: int | None = None,
        sincos_pos_emb: bool = True,
        max_sincos_pos_emb: int = 512,
        padding_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb

        if dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) > number of posembs ({self.max_sincos_pos_emb}")
            pos_emb = build_1d_sincos_posemb(
                max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens,
            )[:, :self.max_length, :]
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        # Task embedding identifying from which task a given token comes from
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        self.token_emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.dim_tokens,
            padding_idx=self.padding_idx,
        )

        # Scale initialization by 1/sqrt(vocab_size) for large vocabularies
        # to prevent gradient explosions with large vocabularies
        embedding_init_std = init_std * (1.0 / (self.vocab_size ** 0.5))
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=embedding_init_std)
        self.token_emb._fill_padding_idx_with_zero()

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - "tensor" (torch.Tensor): Input token sequence for each batch. Shape (B, L) where B is the batch size
                    and L is the sequence length.
                - "input_mask" (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens
                    and 1 otherwise). Shape (B, L).

        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - "x" (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - "emb" (torch.Tensor): Sum of positional and modality embeddings for input sequence. Shape (B, L, D).
        """
        if not isinstance(d, dict):
            d = {"tensor": d,
                 "input_mask": torch.zeros_like(d, dtype=torch.bool)}  # No masking

        ids = d["tensor"]
        B = ids.shape[0]
        assert self.dim_tokens is not None, "Need to call init(dim_tokens) function first"

        # Map to embedding
        x = self.token_emb(ids)

        pos_emb_buffer: torch.Tensor = self.pos_emb
        expanded_pos_emb = repeat(pos_emb_buffer, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d["input_mask"]
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(
            expanded_pos_emb,
            dim=1,
            index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
        )
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb
        return d


class ImageTokenEncoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size (int for homogeneous codebooks, list[int] for heterogeneous)
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        input_size: Input image size. Used to initialize size of positional embeddings.
        num_codebooks: Number of codebooks for multi-codebook token inputs
    """

    def __init__(
        self,
        vocab_size: int | list[int],
        patch_size: int | tuple[int, int] = 16,
        dim_tokens: int | None = None,
        sincos_pos_emb: bool = True,
        input_size: int | tuple[int, int] = 224,
        num_codebooks: int = 1,
        fusion_type: str = "mlp",
        fusion_num_heads: int = 8,
        fusion_hidden_ratio: int = 4,
        **kwargs,
    ):

        super().__init__()
        if isinstance(vocab_size, int):
            self.vocab_sizes = [vocab_size]
            self.vocab_size = vocab_size
            self.is_heterogeneous = False
        else:
            self.vocab_sizes = list(vocab_size)
            self.vocab_size = max(vocab_size)
            self.is_heterogeneous = len(set(vocab_size)) > 1

        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(input_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.num_codebooks = num_codebooks

        # Codebook fusion configuration (default: mlp)
        self.fusion_type = fusion_type
        self.fusion_num_heads = fusion_num_heads
        self.fusion_hidden_ratio = fusion_hidden_ratio

        if dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]

        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb)  # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        # Task embedding identifying from which task a given token comes from
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding with vocabulary-scaled initialization
        if self.is_heterogeneous:
            # Heterogeneous codebooks: create separate embedding table for each codebook
            self.token_emb = nn.ModuleList([
                nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.dim_tokens)
                for vocab_size in self.vocab_sizes
            ])
            # Initialize each embedding table with vocab-size-specific scaling
            for i, vocab_size in enumerate(self.vocab_sizes):
                embedding_init_std = init_std * (1.0 / (vocab_size ** 0.5))
                emb_module = self.token_emb[i]
                assert isinstance(emb_module, nn.Embedding), "Expected nn.Embedding module"
                nn.init.normal_(emb_module.weight, mean=0.0, std=embedding_init_std)
        else:
            # Homogeneous codebooks: single shared embedding table (backward compatible)
            self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

            # Scale initialization by 1/sqrt(vocab_size) for large vocabularies
            # This prevents gradient explosions with large vocabularies (e.g., 15k codes)
            # For small vocabularies, this still provides reasonable init
            embedding_init_std = init_std * (1.0 / (self.vocab_size ** 0.5))
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=embedding_init_std)

        # Codebook embeddings for multi-codebook case
        # Each codebook gets its own learnable embedding to differentiate tokens from different codebooks
        if self.num_codebooks > 1:
            self.codebook_emb = nn.Parameter(torch.zeros(self.num_codebooks, self.dim_tokens))
            nn.init.normal_(self.codebook_emb, std=init_std)

            self.codebook_fusion = UnifiedCodebookFusion(
                dim_tokens=self.dim_tokens,
                num_codebooks=self.num_codebooks,
                fusion_type=self.fusion_type,
                num_heads=self.fusion_num_heads,
                hidden_ratio=self.fusion_hidden_ratio,
            )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through embedding module, transforming image tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (torch.Tensor, dict[str, torch.Tensor]): Modality dict with at least the following key:
                - "tensor" (torch.Tensor): Input image tokens for each batch.
                   Shape (B, H, W) for single codebook or (B, H, W, num_codebooks) for multi-codebook.
                - "input_mask" (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens
                  and 1 otherwise). Shape (B, L).

        Returns:
            dict[str, torch.Tensor]: Modality dictionary with added keys:
                - "x" (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - "emb" (torch.Tensor): Sum of positional and modality embeddings for input sequence. Shape (B, H*W, D).
        """
        if not isinstance(d, dict):
            d = {"tensor": d}

        ids = d["tensor"]
        B = ids.shape[0]

        assert self.dim_tokens is not None, "Need to call init(dim_tokens) function first"

        if self.num_codebooks > 1:
            # Handle multi-codebook tokens: (B, H, W, num_codebooks) or (B, num_tokens, num_codebooks)
            if ids.ndim == 4:
                ids = ids.reshape(B, -1, ids.shape[-1])

            codebook_embeddings = []

            if self.is_heterogeneous:
                # Use separate embedding table for each codebook
                for c in range(self.num_codebooks):
                    emb_module = self.token_emb[c]
                    codebook_embeddings.append(emb_module(ids[:, :, c]))
            else:
                # Homogeneous: use shared embedding table for all codebooks
                for c in range(self.num_codebooks):
                    codebook_embeddings.append(self.token_emb(ids[:, :, c]) + self.codebook_emb[c])

            x = self.codebook_fusion(codebook_embeddings)   # (B, num_tokens, dim_tokens)

        else:
            # Single codebook case: (B, H, W) or (B, num_tokens)
            ids = ids.reshape(B, -1)
            x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        num_tokens = x.shape[1]

        # Use traditional sincos or learned positional embeddings
        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        # Ensure positional embeddings match the number of tokens
        if x_emb.shape[1] != num_tokens:
            x_emb = x_emb[:, :num_tokens, :]

        d["x"] = x
        d["emb"] = x_emb

        return d


class ImageEncoderEmbedding(nn.Module):
    """Embedding module for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    This adapter / embedding differs from the one of MultiMAE by taking as input a dict and
     separating positional embeddings and modality embeddings from the input projection
     Input projection is "x", posemb + modemb is "emb"

    Args:
        num_channels: Number of input channels of the image/feature map
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        input_size: Input image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        num_channels: int,
        patch_size: int | tuple[int, int],
        dim_tokens: int | None = None,
        sincos_pos_emb: bool = True,
        input_size: int | tuple[int, int] = 224,
        **kwargs,
    ):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(input_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        if dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]

        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb)  # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Image -> tokens projection. No bias term here, so modality embedding fully comes from self.mod_emb
        self.proj = nn.Linear(
            self.num_channels * self.patch_size[0] * self.patch_size[1],
            self.dim_tokens,
            bias=False,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through embedding module, transforming image to sequence of tokens.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (torch.Tensor, dict[str, torch.Tensor]): Modality dict with at least the following key:
                - "tensor" (torch.Tensor): Input image for each batch. Shape (B, C, H, W) where B is the batch size, C
                  is the number of channels, and H, W are height and width of the image.

                Optional FlexiViT key:
                - "runtime_patch_size" (int | tuple[int, int]): override patch size for this forward only.
                  When present and different from `self.patch_size`, the projection weight is PI-resized
                  on-the-fly from the canonical `self.patch_size` to the runtime size, and pos-emb /
                  spatial metadata are recomputed for the new grid. Gradients flow back to the canonical
                  `self.proj.weight` because PI-resize is a constant linear operator.

        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - "x" (torch.Tensor): Embedded token sequence. Shape (B, (H / PH) * (W / PW), D), where PH and PW are
                  the patch sizes (or the runtime patch sizes when overridden)
                - "emb" (torch.Tensor): Sum of positional and modality embeddings for the input sequence.
                  Shape (B, (H / PH) * (W / PW), D)
        """
        if not isinstance(d, dict):
            d = {"tensor": d}

        x = d["tensor"]
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, "Need to call init(dim_tokens) function first"

        # FlexiViT runtime patch size override. Accepts a Python int/tuple/list, or a 0-d/1-d torch.Tensor
        runtime_ps = d.get("runtime_patch_size")
        if runtime_ps is None:
            runtime_ps = self.patch_size
        elif isinstance(runtime_ps, torch.Tensor):
            if runtime_ps.numel() == 1:
                v = int(runtime_ps.item())
                runtime_ps = (v, v)
            else:
                rt = runtime_ps.flatten().tolist()
                runtime_ps = (int(rt[0]), int(rt[1]))
        else:
            runtime_ps = pair(runtime_ps)

        assert (H % runtime_ps[0] == 0) and (W % runtime_ps[1] == 0), (
            f"Image sizes {H}x{W} must be divisible by patch sizes {runtime_ps[0]}x{runtime_ps[1]}"
        )

        # PI-resize the projection weight when the runtime patch size differs from the canonical.
        if tuple(runtime_ps) != tuple(self.patch_size):
            proj_weight = pi_resize_patch_embed(
                self.proj.weight,
                old_patch_size=(self.patch_size[0], self.patch_size[1]),
                new_patch_size=(runtime_ps[0], runtime_ps[1]),
                num_channels=self.num_channels,
            )
        else:
            proj_weight = self.proj.weight

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = nn.functional.linear(
            rearrange(
                x,
                "b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)",
                ph=runtime_ps[0],
                pw=runtime_ps[1],
            ),
            proj_weight,
        )

        # Use traditional sincos or learned positional embeddings.
        # Pos-emb is stored at `self.image_size / self.patch_size`. We need to
        # interpolate it to the runtime grid `(H/runtime_ps, W/runtime_ps)`.
        new_h = H // runtime_ps[0]
        new_w = W // runtime_ps[1]
        if (new_h, new_w) != (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]):
            pos_emb = self._resize_pos_encoding(self.pos_emb.clone(), new_h, new_w)
        else:
            pos_emb = self.pos_emb

        # Create positional embedding + modality embedding
        x_emb = repeat(pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        d["x"] = x_patch
        d["emb"] = x_emb

        return d

    def _resize_pos_encoding(self, pos_embeddings: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
        """Bicubic resize of a stored 2D positional embedding to a new grid.

        Generalization of :meth:`interpolate_pos_encoding` that takes the new grid size
        directly (rather than image height/width and `self.patch_size`), so it works
        for FlexiViT runtime grids where the patch size differs from `self.patch_size`.
        """
        num_positions = pos_embeddings.shape[1]
        sqrt_num_positions = int(num_positions ** 0.5)
        assert self.dim_tokens is not None, "Need to call init(dim_tokens) function first"
        pos_embeddings = pos_embeddings.reshape(1, sqrt_num_positions, sqrt_num_positions, self.dim_tokens)
        pos_embeddings = pos_embeddings.permute(0, 3, 1, 2)
        pos_embeddings = nn.functional.interpolate(
            pos_embeddings,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )
        pos_embeddings = pos_embeddings.permute(0, 2, 3, 1).view(1, -1, self.dim_tokens)
        return pos_embeddings

    def interpolate_pos_encoding(self, pos_embeddings: torch.Tensor, height, width) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]
        return self._resize_pos_encoding(pos_embeddings, new_height, new_width)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """FlexiViT-aware state-dict loading.

        Two adaptations are applied transparently when the checkpoint was
        produced at a different patch size than this module is currently
        configured for:

        1. `proj.weight` is PI-resized along its spatial axes from the
           checkpoint's patch size to `self.patch_size`. The checkpoint's
           patch size is inferred from `weight.shape[1] // num_channels`
           (assumes square patches, matching the rest of this module).

        2. The 2D positional embedding (`pos_emb` — buffer when sincos,
           parameter when learnable) is bicubically interpolated to the new
           grid via the existing `interpolate_pos_encoding` helper.

        With matching patch size, behaviour is identical to the default
        `nn.Module._load_from_state_dict`.
        """
        proj_key = prefix + "proj.weight"
        if proj_key in state_dict and self.dim_tokens is not None:
            ckpt_w = state_dict[proj_key]
            cur_w = self.proj.weight
            if ckpt_w.shape != cur_w.shape:
                # Infer the checkpoint's patch size from its flat input dim.
                ckpt_in = ckpt_w.shape[1]
                if ckpt_in % self.num_channels != 0:
                    error_msgs.append(
                        f"{proj_key}: checkpoint input dim {ckpt_in} not divisible by "
                        f"num_channels={self.num_channels}; cannot infer patch size."
                    )
                else:
                    ckpt_n_pix = ckpt_in // self.num_channels
                    ckpt_ph = int(round(ckpt_n_pix ** 0.5))
                    if ckpt_ph * ckpt_ph != ckpt_n_pix:
                        error_msgs.append(
                            f"{proj_key}: checkpoint patch is non-square ({ckpt_n_pix} pixels); "
                            f"FlexiViT load only supports square patches."
                        )
                    else:
                        resized = pi_resize_patch_embed(
                            weight=ckpt_w,
                            old_patch_size=(ckpt_ph, ckpt_ph),
                            new_patch_size=(self.patch_size[0], self.patch_size[1]),
                            num_channels=self.num_channels,
                        )
                        if resized.shape != cur_w.shape:
                            error_msgs.append(
                                f"{proj_key}: PI-resize produced shape {tuple(resized.shape)}, "
                                f"expected {tuple(cur_w.shape)}."
                            )
                        else:
                            state_dict[proj_key] = resized

        # Pos-emb resize for sincos (buffer) and learnable (parameter) cases.
        # The RFF mode has no stored grid-shaped tensor, so nothing to do.
        pos_emb_key = prefix + "pos_emb"
        if pos_emb_key in state_dict and hasattr(self, "pos_emb") and self.dim_tokens is not None:
            ckpt_pe = state_dict[pos_emb_key]
            cur_pe = self.pos_emb
            if ckpt_pe.shape != cur_pe.shape:
                # Reuse the existing bicubic helper. It interpolates from the
                # checkpoint grid to `(image_size / patch_size)` — exactly the
                # current module's grid.
                resized_pe = self.interpolate_pos_encoding(ckpt_pe.clone(), self.image_size[0], self.image_size[1])
                if resized_pe.shape != cur_pe.shape:
                    error_msgs.append(
                        f"{pos_emb_key}: pos-emb interpolation produced shape "
                        f"{tuple(resized_pe.shape)}, expected {tuple(cur_pe.shape)}."
                    )
                else:
                    state_dict[pos_emb_key] = resized_pe

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
