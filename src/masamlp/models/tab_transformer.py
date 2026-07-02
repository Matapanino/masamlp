"""TabTransformer — Huang et al. 2020, "TabTransformer: Tabular Data
Modeling Using Contextual Embeddings" (arXiv:2012.06678).

Categorical features become tokens processed by a (post-norm) transformer
encoder into contextual embeddings; numeric features bypass the transformer
— layer-normalized and concatenated with the flattened contextual
embeddings — before the paper's ``(4l, 2l)`` MLP head. Extensions beyond the
paper: ``num_embedding`` optionally embeds the numeric block (PLR family)
before its LayerNorm, and ``num_scaling`` prepends the learnable scale.
With no categorical features the model degenerates to the MLP head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from masamlp.models.base import TokenEmbedding


class _PostNormBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ffn_d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn))
        return self.norm2(x + self.dropout(self.ffn(x)))


class TabTransformer(nn.Module):
    embedding_kind = "tokens"

    def __init__(
        self,
        embedding_config: dict,
        out_dim: int,
        d_token: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_d_hidden_multiplier: float = 4.0,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(
            d_token=d_token, tokenize_numeric=False, **embedding_config
        )
        self.blocks = nn.ModuleList(
            _PostNormBlock(d_token, n_heads, int(d_token * ffn_d_hidden_multiplier), dropout)
            for _ in range(n_layers)
        )
        self.num_norm = (
            nn.LayerNorm(self.embedding.d_num_flat) if self.embedding.d_num_flat else None
        )
        d_in = self.embedding.n_tokens * d_token + self.embedding.d_num_flat
        # The paper's MLP head: hidden sizes (4l, 2l) with ReLU.
        self.head = nn.Sequential(
            nn.Linear(d_in, 4 * d_in),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(4 * d_in, 2 * d_in),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        self.output_layer = nn.Linear(2 * d_in, out_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        tokens, num_flat = self.embedding(x_num, x_cat)
        parts = []
        if tokens.shape[1] > 0:
            for block in self.blocks:
                tokens = block(tokens)
            parts.append(tokens.flatten(1))
        if self.num_norm is not None:
            parts.append(self.num_norm(num_flat))
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.output_layer(self.head(x))
