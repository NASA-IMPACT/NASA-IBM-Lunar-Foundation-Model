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

from einops import repeat

from ni_lfm.models.codebook_fusion import UnifiedCodebookFusion, UnifiedProjectionFusion
from ni_lfm.models.lfm_utils import build_1d_sincos_posemb, build_2d_sincos_posemb, pair


class SequenceDecoderEmbedding(nn.Module):
    """Embedding module for sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        share_embedding: Set to True to share input and output embedding weights
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim_tokens: int | None = None,
        sincos_pos_emb: bool = True,
        max_sincos_pos_emb: int = 512,
        padding_idx: int = 0,
        share_embedding: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.share_embedding = share_embedding

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
            # Get all posembs, than truncate up to max length
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

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        if self.share_embedding:
            # Share input and output embedding weights
            # Use standard init_std (not vocabulary-scaled) to prevent logit collapse
            # The input embeddings can still use vocabulary-scaled init for gradient stability
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=init_std)
            self.token_emb._fill_padding_idx_with_zero()
            self.to_logits.weight = self.token_emb.weight
        else:
            # When not sharing, use vocabulary-scaled init for input embeddings only
            # to prevent gradient explosions with large vocabularies
            embedding_init_std = init_std * (1.0 / (self.vocab_size ** 0.5))
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=embedding_init_std)
            self.token_emb._fill_padding_idx_with_zero()
            # Output projection uses standard initialization
            nn.init.normal_(self.to_logits.weight, mean=0.0, std=init_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (dict[str, torch.Tensor]): Modality dict, with at least the following keys:
                - "tensor" (torch.Tensor): Token sequence for each batch. Shape (B, L) where B is the batch size
                    and L is the sequence length.
                - "target_mask" (torch.Tensor): Mask for valid tokens in the target sequence (set to 0 for valid
                    tokens and 1 otherwise). Shape (B, L).

        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - "x" (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - "emb" (torch.Tensor): Sum of positional and modality embeddings for target sequence. Shape (B, L, D).
                - "ids" (torch.Tensor): Original token sequence from input dict. Shape (B, L).
        """
        ids = d["tensor"]
        B = ids.shape[0]
        assert self.dim_tokens is not None, "Need to call init(dim_tokens) function first"

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)

        # Target pos encoding
        target_mask = d["target_mask"]
        target_pos_id = (~target_mask).int().cumsum(dim=1) - 1
        target_pos_id[target_mask] = 0
        # Sometimes target sequence is over max length, it will be truncated in decoder
        target_pos_id[target_pos_id >= self.max_length] = 0
        target_pos_emb = torch.gather(
            expanded_pos_emb,
            dim=1,
            index=repeat(target_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]),
        )
        target_pos_emb[target_mask] = 0

        x_emb = target_pos_emb + self.mod_emb

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = d["tensor"]

        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits


class ImageTokenDecoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size (int for homogeneous codebooks, list[int] for heterogeneous)
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        input_size: Input image size. Used to initialize size of positional embeddings.
        share_embedding: Set to True to share input and output embedding weights
        num_codebooks: Number of codebooks for multi-codebook tokens (default: 1)
    """

    def __init__(
        self,
        vocab_size: int | list[int],
        patch_size: int | tuple[int, int] = 16,
        dim_tokens: int | None = None,
        sincos_pos_emb: bool = True,
        input_size: int | tuple[int, int] = 224,
        share_embedding: bool = True,
        num_codebooks: int = 1,
        fusion_type: str = "mlp",
        fusion_num_heads: int = 8,
        fusion_hidden_ratio: int = 4,
        projection_type: str = "linear",
        projection_hidden_ratio: float = 2.0,
        projection_trunk_hidden_ratio: float = 2.0,
        projection_head_hidden_ratio: float = 0.5,
        projection_dropout: float = 0.1,
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
            # For heterogeneous codebooks, num_codebooks must match vocab_sizes length
            if num_codebooks != len(vocab_size):
                num_codebooks = len(vocab_size)

        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(input_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.share_embedding = share_embedding
        self.num_codebooks = num_codebooks

        # Codebook fusion configuration (embedding aggregation, default: mlp)
        self.fusion_type = fusion_type
        self.fusion_num_heads = fusion_num_heads
        self.fusion_hidden_ratio = fusion_hidden_ratio

        # Projection fusion configuration (logits generation, default: linear)
        self.projection_type = projection_type
        self.projection_hidden_ratio = projection_hidden_ratio
        self.projection_trunk_hidden_ratio = projection_trunk_hidden_ratio
        self.projection_head_hidden_ratio = projection_head_hidden_ratio
        self.projection_dropout = projection_dropout

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
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        # Task embedding identifying from which task a given token comes from
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding with vocabulary-scaled initialization
        # (not needed if only masked tokens are given as input, but can be useful to train Token Critic)
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

        # Output projection layers using UnifiedProjectionFusion
        # Supports:
        #   - Single codebook
        #   - Homogeneous multi-codebook: vocab_sizes=[8]*128
        #   - Heterogeneous multi-codebook: vocab_sizes=[1024, 8500]
        # Projection types: linear, mlp (per-head), trunk_mlp (shared trunk)
        self.to_logits = UnifiedProjectionFusion(
            dim_tokens=self.dim_tokens,
            vocab_sizes=self.vocab_sizes,
            projection_type=self.projection_type,
            share_weights=self.share_embedding and not self.is_heterogeneous,
            hidden_ratio=self.projection_hidden_ratio,
            trunk_hidden_ratio=self.projection_trunk_hidden_ratio,
            head_hidden_ratio=self.projection_head_hidden_ratio,
            dropout=self.projection_dropout,
        )

        if not self.is_heterogeneous:
            self.to_logits.set_embedding_weights(self.token_emb.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through the embedding module, transforming tokenized spatial inputs to embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (dict[str, torch.Tensor]): Modality dict, with at least the following key:
                - "tensor" (torch.Tensor): Modality tokens for each batch (e.g. from tokenized images). Shape (B, H, W)
                    where B is the batch size, H and W are height and width after tokenization.


        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - "x" (torch.Tensor): Embedded token sequence, which is replaced by mask tokens in the 4M decoder.
                    Shape (B, H*W, D) where D is the embedding dimension.
                - "emb" (torch.Tensor): Sum of positional and modality embeddings for the token sequence.
                    Shape (B, H*W, D).
                - "ids" (torch.Tensor): Reshaped token sequence from input dict, flattened in the spatial dimensions.
                    Shape (B, H*W).
        """
        ids = d["tensor"]
        B = ids.shape[0]

        # Handle multi-codebook tokens: (B, H, W, num_codebooks) or (B, num_tokens, num_codebooks)
        if self.num_codebooks > 1:
            # Multi-codebook case: embed each codebook and sum to create unified representation
            # This matches the encoder behavior and allows the decoder to predict all codebooks
            # from a single unified embedding
            # Reshape to (B, num_tokens, num_codebooks)
            if ids.ndim == 4:
                ids_reshaped = ids.reshape(B, -1, ids.shape[-1])
            elif ids.ndim == 3:
                ids_reshaped = ids
            elif ids.ndim == 2:
                # During generation, the tensor starts as (B, num_tokens) zeros (all masked).
                # Expand to (B, num_tokens, num_codebooks) by repeating the same ids across codebooks.
                ids_reshaped = ids.unsqueeze(-1).expand(-1, -1, self.num_codebooks).contiguous()
            else:
                raise ValueError(f"Expected 2, 3 or 4 dimensions for multi-codebook tokens, got {ids.ndim}")

            num_tokens = ids_reshaped.shape[1]

            codebook_embeddings = []

            if self.is_heterogeneous:
                # Use separate embedding table for each codebook
                for c in range(self.num_codebooks):
                    emb_module = self.token_emb[c]
                    codebook_embeddings.append(emb_module(ids_reshaped[:, :, c]))
            else:
                # Homogeneous: use shared embedding table for all codebooks
                for c in range(self.num_codebooks):
                    codebook_embeddings.append(self.token_emb(ids_reshaped[:, :, c]) + self.codebook_emb[c])

            x = self.codebook_fusion(codebook_embeddings)   # (B, num_tokens, dim_tokens)

            ids_output = ids_reshaped  # Shape: (B, num_tokens, num_codebooks)
        else:
            # Single codebook case: (B, H, W) or (B, num_tokens)
            ids_output = ids.reshape(B, -1)  # Shape: (B, num_tokens)
            x = self.token_emb(ids_output)

        # Create positional embedding + modality embedding
        num_tokens = x.shape[1]

        # Use traditional sincos or learned positional embeddings
        x_emb = repeat(self.pos_emb + self.mod_emb, "() n d -> b n d", b=B)

        # Ensure positional embeddings match the number of tokens
        if x_emb.shape[1] != num_tokens:
            x_emb = x_emb[:, :num_tokens, :]

        d["x"] = x
        d["emb"] = x_emb
        d["ids"] = ids_output
        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Uses UnifiedProjectionFusion which supports linear, mlp, and trunk_mlp projection strategies.
        For multi-codebook VQ-VAE, this predicts all codebooks from the unified embedding representation.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)

        Returns:
            torch.Tensor: Logits for each token in the sequence.
                - Single codebook: Shape (..., V)  where ... matches input x leading dims
                - Multi-codebook (homogeneous): Shape (..., V, num_codebooks) where all codebooks have same V
                - Multi-codebook (heterogeneous): Shape (..., max_V, num_codebooks) where max_V is max vocab size
                  Smaller vocabs are padded with -inf so they don't affect softmax/cross_entropy
                  e.g. if x is (B, M, D) → (B, M, max_V, num_codebooks)
                       if x is (L, D) after boolean masking → (L, max_V, num_codebooks)
        """
        logits_output = self.to_logits(x)

        if self.num_codebooks == 1:
            return logits_output
        else:
            # Multi-codebook: logits_output is a list of tensors
            assert isinstance(logits_output, list), "Expected list of logits for multi-codebook"

            if self.is_heterogeneous:
                # Heterogeneous: pad to max vocab size for stacking
                max_vocab_size = max(self.vocab_sizes)
                logits_list = []

                for c, logits_c in enumerate(logits_output):
                    if self.vocab_sizes[c] < max_vocab_size:
                        pad_size = max_vocab_size - self.vocab_sizes[c]
                        # Pad with -inf so padded positions don't affect softmax/cross_entropy
                        padding = torch.full(
                            (*logits_c.shape[:-1], pad_size),
                            fill_value=float("-inf"),
                            dtype=logits_c.dtype,
                            device=logits_c.device,
                        )
                        logits_c = torch.cat([logits_c, padding], dim=-1)

                    logits_list.append(logits_c)
            else:
                # Homogeneous: all codebooks have same vocab size, no padding needed
                logits_list = logits_output

            # Stack along last dimension: (..., V, num_codebooks) or (..., max_V, num_codebooks)
            logits = torch.stack(logits_list, dim=-1)
            return logits
