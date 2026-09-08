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

"""Codebook fusion modules for aggregating multiple VQ-VAE codebook embeddings
per patch into a single token representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookFusionAttention(nn.Module):
    """Aggregate multiple codebook embeddings per patch using attention-based pooling.

    This is the recommended approach as it provides maximum flexibility and allows
    the model to learn which codebooks are most important for each patch.

    Args:
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
        num_heads: Number of attention heads (default: 8)
    """

    def __init__(self, dim_tokens: int, num_codebooks: int, num_heads: int = 8):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens

        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, dim_tokens))
        nn.init.normal_(self.query, std=0.02)

        # Multi-head attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=dim_tokens,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(dim_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate codebook embeddings using attention pooling.

        Args:
            x: Codebook embeddings of shape (B, num_patches, num_codebooks, D)

        Returns:
            Aggregated embeddings of shape (B, num_patches, D)
        """
        B, N, C, D = x.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"

        # Reshape to process all patches together
        x_flat = x.reshape(B * N, C, D)

        query = self.query.expand(B * N, 1, D)

        # Attention pooling over codebooks; query: (B*N, 1, D), key/value: (B*N, C, D)
        aggregated, _ = self.attention(query, x_flat, x_flat)

        aggregated = aggregated.reshape(B, N, D)
        aggregated = self.norm(aggregated)

        return aggregated


class CodebookFusionWeighted(nn.Module):
    """Aggregate codebooks via learned weighted sum.

    This is a lightweight approach that learns a fixed weight for each codebook
    across all patches. More efficient than attention but less flexible.

    Args:
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
    """

    def __init__(self, dim_tokens: int, num_codebooks: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens

        # Learnable weights for each codebook
        self.weights = nn.Parameter(torch.ones(num_codebooks) / num_codebooks)
        # Layer norm for stability
        self.norm = nn.LayerNorm(dim_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate codebook embeddings using learned weights.

        Args:
            x: Codebook embeddings of shape (B, num_patches, num_codebooks, D)

        Returns:
            Aggregated embeddings of shape (B, num_patches, D)
        """
        B, N, C, D = x.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"

        # Softmax weights to ensure they sum to 1
        weights = F.softmax(self.weights, dim=0)

        # Weighted sum: (B, N, C, D) * (C,) -> (B, N, D)
        aggregated = torch.einsum("bncd,c->bnd", x, weights)

        aggregated = self.norm(aggregated)

        return aggregated


class CodebookFusionMLP(nn.Module):
    """Aggregate codebooks via MLP projection.

    This approach concatenates all codebook embeddings and projects them
    through an MLP. Provides good expressiveness but higher memory usage.

    Args:
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
        hidden_ratio: Ratio of hidden dimension to input dimension (default: 4)
    """

    def __init__(self, dim_tokens: int, num_codebooks: int, hidden_ratio: int = 4):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.dim_tokens = dim_tokens

        input_dim = dim_tokens * num_codebooks
        hidden_dim = dim_tokens * hidden_ratio

        # MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim_tokens),
            nn.LayerNorm(dim_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate codebook embeddings using MLP projection.

        Args:
            x: Codebook embeddings of shape (B, num_patches, num_codebooks, D)

        Returns:
            Aggregated embeddings of shape (B, num_patches, D)
        """
        B, N, C, D = x.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        assert D == self.dim_tokens, f"Expected dim {self.dim_tokens}, got {D}"

        x_flat = x.reshape(B, N, C * D)  # (B, N, C, D) -> (B, N, C*D)

        # Project through MLP
        aggregated = self.mlp(x_flat)

        return aggregated


def create_codebook_fusion(
    fusion_type: str,
    dim_tokens: int,
    num_codebooks: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create codebook fusion module.

    Args:
        fusion_type: Type of fusion ('attention', 'weighted', or 'mlp')
        dim_tokens: Dimension of token embeddings
        num_codebooks: Number of codebooks to aggregate
        **kwargs: Additional arguments for specific fusion types

    Returns:
        Codebook fusion module

    Raises:
        ValueError: If fusion_type is not recognized
    """
    if fusion_type == "attention":
        num_heads = kwargs.get("num_heads", 8)
        return CodebookFusionAttention(dim_tokens, num_codebooks, num_heads)
    elif fusion_type == "weighted":
        return CodebookFusionWeighted(dim_tokens, num_codebooks)
    elif fusion_type == "mlp":
        hidden_ratio = kwargs.get("hidden_ratio", 4)
        return CodebookFusionMLP(dim_tokens, num_codebooks, hidden_ratio)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}. Must be one of: 'attention', 'weighted', 'mlp'")


class UnifiedCodebookFusion(nn.Module):
    """Unified interface for codebook fusion supporting multiple strategies.

    Provides backward compatibility with existing linear projection while
    enabling advanced fusion mechanisms.

    Args:
        dim_tokens: Embedding dimension
        num_codebooks: Number of codebooks to fuse
        fusion_type: Type of fusion ('linear', 'attention', 'weighted', 'mlp')
        **fusion_kwargs: Additional arguments for specific fusion types
            - num_heads: Number of attention heads (for attention fusion)
            - hidden_ratio: Hidden dimension ratio (for MLP fusion)
    """

    def __init__(
        self,
        dim_tokens: int,
        num_codebooks: int,
        fusion_type: str = "linear",
        **fusion_kwargs,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.dim_tokens = dim_tokens
        self.num_codebooks = num_codebooks

        if fusion_type == "linear":
            self.fusion = nn.Linear(num_codebooks * dim_tokens, dim_tokens, bias=True)
            self._forward_fn = self._forward_linear

        elif fusion_type == "attention":
            num_heads = fusion_kwargs.get("num_heads", 8)
            self.fusion = CodebookFusionAttention(dim_tokens, num_codebooks, num_heads)
            self._forward_fn = self._forward_stacked

        elif fusion_type == "weighted":
            self.fusion = CodebookFusionWeighted(dim_tokens, num_codebooks)
            self._forward_fn = self._forward_stacked

        elif fusion_type == "mlp":
            hidden_ratio = fusion_kwargs.get("hidden_ratio", 4)
            self.fusion = CodebookFusionMLP(dim_tokens, num_codebooks, hidden_ratio)
            self._forward_fn = self._forward_stacked

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Must be: 'linear', 'attention', 'weighted' or 'mlp'")

    def _forward_linear(self, codebook_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Forward for linear fusion."""
        x_concat = torch.cat(codebook_embeddings, dim=-1)   # (B, N, C*D)
        return self.fusion(x_concat)

    def _forward_stacked(self, codebook_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Forward for attention/weighted/mlp fusion."""
        x_stacked = torch.stack(codebook_embeddings, dim=2)   # (B, N, C, D)
        return self.fusion(x_stacked)

    def forward(self, codebook_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple codebook embeddings (list of C tensors, each (B, N, D)) into single representation."""
        return self._forward_fn(codebook_embeddings)


class ProjectionFusionLinear(nn.Module):
    """Linear projection from embeddings to logits: Creates separate linear heads for each codebook.

    Args:
        dim_tokens: Embedding dimension
        vocab_sizes: List of vocabulary sizes for each codebook
        share_weights: Whether to share weights with embedding layer
    """

    def __init__(self, dim_tokens: int, vocab_sizes: list[int], share_weights: bool = False):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.vocab_sizes = vocab_sizes
        self.num_codebooks = len(vocab_sizes)
        self.share_weights = share_weights

        self.heads = nn.ModuleList([
            nn.Linear(dim_tokens, vocab_size, bias=False)
            for vocab_size in vocab_sizes
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Project embeddings to logits for each codebook.

        Args:
            x: Hidden states (B, M, D)

        Returns:
            List of logits tensors, one per codebook:
                - codebook i: (B, M, vocab_sizes[i])
        """
        logits_list = []
        for head in self.heads:
            logits_list.append(head(x))
        return logits_list


class ProjectionFusionMLP(nn.Module):
    """MLP-based projection from embeddings to logits with separate heads.

    Each codebook gets its own MLP projection head, allowing non-linear
    transformations before vocabulary prediction.

    Args:
        dim_tokens: Embedding dimension
        vocab_sizes: List of vocabulary sizes for each codebook
        hidden_ratio: Hidden layer expansion ratio (default: 2.0)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, dim_tokens: int, vocab_sizes: list[int], hidden_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.vocab_sizes = vocab_sizes
        self.num_codebooks = len(vocab_sizes)

        # Adjust hidden_ratio for many codebooks to avoid explosion
        if self.num_codebooks > 10:
            hidden_ratio = min(hidden_ratio, 0.5)

        # Create MLP head for each codebook
        self.heads = nn.ModuleList([
            self._create_mlp_head(dim_tokens, vocab_size, hidden_ratio, dropout)
            for vocab_size in vocab_sizes
        ])

    def _create_mlp_head(self, dim_tokens: int, vocab_size: int, hidden_ratio: float, dropout: float) -> nn.Module:
        """Create a single MLP projection head."""
        hidden_dim = int(dim_tokens * hidden_ratio)

        return nn.Sequential(
            nn.Linear(dim_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Project embeddings to logits via MLP for each codebook.

        Args:
            x: Hidden states (B, M, D)

        Returns:
            List of logits tensors, one per codebook:
                - codebook i: (B, M, vocab_sizes[i])
        """
        logits_list = []
        for head in self.heads:
            logits_list.append(head(x))
        return logits_list


class ProjectionFusionTrunkMLP(nn.Module):
    """Shared trunk MLP with separate lightweight heads for each codebook.

    Uses a shared MLP trunk to process all tokens, followed by lightweight
    per-codebook heads. More parameter-efficient than per-head MLPs for
    many codebooks.

    Args:
        dim_tokens: Embedding dimension
        vocab_sizes: List of vocabulary sizes for each codebook
        trunk_hidden_ratio: Trunk hidden layer expansion ratio (default: 2.0)
        head_hidden_ratio: Per-head hidden layer expansion ratio (default: 0.5)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        dim_tokens: int,
        vocab_sizes: list[int],
        trunk_hidden_ratio: float = 2.0,
        head_hidden_ratio: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.vocab_sizes = vocab_sizes
        self.num_codebooks = len(vocab_sizes)

        trunk_hidden = int(dim_tokens * trunk_hidden_ratio)
        head_hidden = int(dim_tokens * head_hidden_ratio)

        # Shared trunk processes all tokens
        self.trunk = nn.Sequential(
            nn.Linear(dim_tokens, trunk_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, dim_tokens),
            nn.LayerNorm(dim_tokens),
        )

        # Lightweight per-codebook heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_tokens, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, vocab_size, bias=False),
            )
            for vocab_size in vocab_sizes
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Project embeddings to logits via shared trunk + separate heads.

        Args:
            x: Hidden states (B, M, D)

        Returns:
            List of logits tensors, one per codebook:
                - codebook i: (B, M, vocab_sizes[i])
        """
        x_trunk = self.trunk(x)

        logits_list = []
        for head in self.heads:
            logits_list.append(head(x_trunk))

        return logits_list


class UnifiedProjectionFusion(nn.Module):
    """Unified interface for projection fusion supporting multiple strategies.

    Provides backward compatibility with existing linear projection while
    enabling advanced projection mechanisms (MLP, shared trunk).

    Supports three configurations:
        - Single codebook (15k vocab): Uses single projection head
        - Homogeneous multi-codebook (128c): All codebooks same vocab size
        - Heterogeneous multi-codebook: Different vocab sizes per codebook

    Args:
        dim_tokens: Embedding dimension
        vocab_sizes: List of vocabulary sizes (single element for single codebook)
        projection_type: Type of projection ('linear', 'mlp', 'trunk_mlp')
        share_weights: Whether to share weights with embedding layer (linear only)
        **projection_kwargs: Additional arguments for specific projection types
            - hidden_ratio: Hidden dimension ratio (for MLP)
            - trunk_hidden_ratio: Trunk hidden ratio (for trunk_mlp)
            - head_hidden_ratio: Head hidden ratio (for trunk_mlp)
            - dropout: Dropout probability
    """

    def __init__(
        self,
        dim_tokens: int,
        vocab_sizes: list[int],
        projection_type: str = "linear",
        share_weights: bool = False,
        **projection_kwargs,
    ):
        super().__init__()
        self.projection_type = projection_type
        self.dim_tokens = dim_tokens
        self.vocab_sizes = vocab_sizes
        self.num_codebooks = len(vocab_sizes)
        self.share_weights = share_weights

        # Single codebook case
        if self.num_codebooks == 1:
            vocab_size = vocab_sizes[0]

            if projection_type == "linear":
                self.projection = nn.Linear(dim_tokens, vocab_size, bias=False)
            else:  # mlp = trunk_mlp for single codebook; trunk_mlp not meaningful in this case
                hidden_ratio = projection_kwargs.get("hidden_ratio", 2.0)
                dropout = projection_kwargs.get("dropout", 0.1)
                hidden_dim = int(dim_tokens * hidden_ratio)

                self.projection = nn.Sequential(
                    nn.Linear(dim_tokens, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, vocab_size, bias=False),
                )
        # Multi-codebook case
        else:
            if projection_type == "linear":
                self.projection = ProjectionFusionLinear(dim_tokens, vocab_sizes, share_weights)

            elif projection_type == "mlp":
                hidden_ratio = projection_kwargs.get("hidden_ratio", 2.0)
                dropout = projection_kwargs.get("dropout", 0.1)
                self.projection = ProjectionFusionMLP(dim_tokens, vocab_sizes, hidden_ratio, dropout)

            elif projection_type == "trunk_mlp":
                trunk_hidden_ratio = projection_kwargs.get("trunk_hidden_ratio", 2.0)
                head_hidden_ratio = projection_kwargs.get("head_hidden_ratio", 0.5)
                dropout = projection_kwargs.get("dropout", 0.1)
                self.projection = ProjectionFusionTrunkMLP(
                    dim_tokens, vocab_sizes, trunk_hidden_ratio, head_hidden_ratio, dropout,
                )
            else:
                raise ValueError(f"Unknown projection_type: {projection_type}. Must be: 'linear', 'mlp' or 'trunk_mlp'")

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """Project hidden states to vocabulary logits.

        Args:
            x: Hidden states (B, M, D)

        Returns:
            Single codebook: Logits tensor (B, M, V)
            Multi-codebook: List of logits tensors, one per codebook
        """
        return self.projection(x)

    def set_embedding_weights(self, embedding_weights: torch.Tensor):
        """Set weights for weight tying with embedding layer. Only applicable for linear projection.

        Args:
            embedding_weights: Embedding weight tensor(s) to tie with
                - Single codebook: Single tensor (V, D)
                - Multi-codebook: List of tensors, one per codebook
        """
        if not self.share_weights or self.projection_type != "linear":
            return

        if self.num_codebooks == 1 and isinstance(self.projection, nn.Linear):
            self.projection.weight = embedding_weights
