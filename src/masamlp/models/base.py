"""Feature embedding shared by every model.

Numeric features pass through optionally as periodic or PLR embeddings
(Gorishniy et al. 2022, "On Embeddings for Numerical Features in Tabular
Deep Learning"); categorical features get per-column ``nn.Embedding`` with
row 0 reserved for unknown/missing. The output is a flat ``(n, d_out)``
tensor the model trunks consume.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _auto_emb_dim(cardinality: int) -> int:
    return int(min(32, max(2, round(1.6 * cardinality**0.56))))


class PeriodicEmbedding(nn.Module):
    """Per-feature ``[cos(2*pi*c*x), sin(2*pi*c*x)]`` with learnable
    frequencies ``c ~ N(0, sigma^2)``; output ``(n, F, 2*n_frequencies)``."""

    def __init__(self, n_features: int, n_frequencies: int = 16, sigma: float = 0.1) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)

    def forward(self, x: Tensor) -> Tensor:
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class PLREmbedding(nn.Module):
    """Periodic -> per-feature Linear -> ReLU, flattened to ``(n, F * d)``."""

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 16,
        n_frequencies: int = 16,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.periodic = PeriodicEmbedding(n_features, n_frequencies, sigma)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_frequencies, d_embedding))
        self.bias = nn.Parameter(torch.zeros(n_features, d_embedding))
        nn.init.uniform_(
            self.weight, -1.0 / math.sqrt(2 * n_frequencies), 1.0 / math.sqrt(2 * n_frequencies)
        )

    def forward(self, x: Tensor) -> Tensor:
        p = self.periodic(x)
        out = torch.relu(torch.einsum("bfk,fkd->bfd", p, self.weight) + self.bias)
        return out.flatten(1)


class FeatureEmbedding(nn.Module):
    """Numeric (raw / periodic / PLR) + categorical embeddings, concatenated.

    ``cat_cardinalities`` already include the reserved unknown/missing slot
    (index 0), matching ``TabularPreprocessor.cat_cardinalities_``.
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        num_embedding: str | None = None,
        d_num_embedding: int = 16,
        n_frequencies: int = 16,
        sigma: float = 0.1,
        cat_emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        if n_num == 0 and not cat_cardinalities:
            raise ValueError("model needs at least one numeric or categorical feature")
        self.n_num = n_num
        self.num_embedding: nn.Module | None = None
        d_num = n_num
        if n_num > 0 and num_embedding is not None:
            if num_embedding == "periodic":
                self.num_embedding = PeriodicEmbedding(n_num, n_frequencies, sigma)
                d_num = n_num * 2 * n_frequencies
            elif num_embedding == "plr":
                self.num_embedding = PLREmbedding(n_num, d_num_embedding, n_frequencies, sigma)
                d_num = n_num * d_num_embedding
            else:
                raise ValueError(
                    f"Unknown num_embedding {num_embedding!r}. Expected 'plr', 'periodic', or None"
                )
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(card, cat_emb_dim if cat_emb_dim is not None else _auto_emb_dim(card))
                for card in cat_cardinalities
            ]
        )
        self.d_out = d_num + sum(emb.embedding_dim for emb in self.cat_embeddings)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        parts: list[Tensor] = []
        if self.n_num > 0:
            if self.num_embedding is not None:
                num = self.num_embedding(x_num)
                parts.append(num.flatten(1) if num.ndim == 3 else num)
            else:
                parts.append(x_num)
        for j, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, j]))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
