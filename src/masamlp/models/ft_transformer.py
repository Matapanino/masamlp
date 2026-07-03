"""FT-Transformer — Gorishniy et al. 2021, "Revisiting Deep Learning Models
for Tabular Data" (arXiv:2106.11959).

Follows the MIT-licensed reference package (rtdl_revisiting_models): every
feature becomes a ``d_block``-dim token (linear tokens for numeric features
by default; the PLR family via ``num_embedding`` upgrades them), a [CLS]
token is prepended, and PreNorm transformer blocks with ReGLU feed-forwards
process the sequence. Two reference details are kept: the very first block
skips the attention normalization (crucial in the PreNorm setup), and the
last block computes attention only for the [CLS] query. Attention itself is
torch-native ``nn.MultiheadAttention``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from masamlp.models.base import TokenEmbedding


class _ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class _FTBlock(nn.Module):
    def __init__(
        self,
        d_block: int,
        n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        residual_dropout: float,
        prenorm_attention: bool,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_block) if prenorm_attention else None
        self.attention = nn.MultiheadAttention(
            d_block, n_heads, dropout=attention_dropout, batch_first=True
        )
        self.attention_residual_dropout = nn.Dropout(residual_dropout)
        self.ffn_norm = nn.LayerNorm(d_block)
        self.ffn = nn.Sequential(
            nn.Linear(d_block, 2 * ffn_d_hidden),  # ReGLU halves the width
            _ReGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_block),
        )
        self.ffn_residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: Tensor, query_only_cls: bool = False) -> Tensor:
        identity = x
        z = x if self.attention_norm is None else self.attention_norm(x)
        query = z[:, :1] if query_only_cls else z
        attn, _ = self.attention(query, z, z, need_weights=False)
        if query_only_cls:
            # Only the [CLS] representation is needed downstream.
            x = identity[:, :1] + self.attention_residual_dropout(attn)
        else:
            x = identity + self.attention_residual_dropout(attn)
        identity = x
        x = identity + self.ffn_residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class FTTransformer(nn.Module):
    """Defaults are the reference's ``get_default_kwargs(n_blocks=3)``."""

    embedding_kind = "tokens"
    # amp="auto" resolves to off: fp16 autocast measured slower AND less
    # accurate on T4 (2026-07-03 verdict: fit 19.4s->24.7s, rmse
    # 0.296->0.345 at 30k rows). Explicit amp=True still forces it.
    amp_auto = False

    def __init__(
        self,
        embedding_config: dict,
        out_dim: int,
        n_blocks: int = 3,
        d_block: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_d_hidden_multiplier: float = 4 / 3,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(
            d_token=d_block, tokenize_numeric=True, **embedding_config
        )
        self.cls_token = nn.Parameter(torch.empty(d_block))
        nn.init.uniform_(self.cls_token, -1.0 / math.sqrt(d_block), 1.0 / math.sqrt(d_block))
        ffn_d_hidden = int(d_block * ffn_d_hidden_multiplier)
        self.blocks = nn.ModuleList(
            _FTBlock(
                d_block,
                attention_n_heads,
                attention_dropout,
                ffn_d_hidden,
                ffn_dropout,
                residual_dropout,
                prenorm_attention=i > 0,  # the first normalization is skipped
            )
            for i in range(n_blocks)
        )
        self.head_norm = nn.LayerNorm(d_block)
        self.output_layer = nn.Linear(d_block, out_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        tokens, _ = self.embedding(x_num, x_cat)
        cls = self.cls_token.expand(tokens.shape[0], 1, -1)
        x = torch.cat([cls, tokens], dim=1)
        n_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            x = block(x, query_only_cls=i + 1 == n_blocks)
        return self.output_layer(torch.relu(self.head_norm(x[:, 0])))
