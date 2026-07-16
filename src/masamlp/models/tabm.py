"""TabM — Gorishniy et al. 2024, "TabM: Advancing Tabular Deep Learning
with Parameter-Efficient Ensembling" (arXiv:2410.24210).

An implicit deep ensemble of ``k`` members sharing one MLP backbone. Diversity
comes from a per-member multiplicative adapter on the (shared) feature
embedding; each member then has its own linear output head. The backbone is
shared and standard (no per-layer sign adapters — that "naive BatchEnsemble"
underperforms a single model), so it stays a strong feature extractor while the
k members diverge. Members train as independent predictors (the trainer
averages the per-member losses) and predictions are averaged on the
prediction scale — the combination that yields variance reduction.

This is the "TabM-mini" structure: parameter cost is ~1x a single MLP (the
backbone) plus ``k*(d_out + d*out_dim)`` for the adapter and per-member heads.

Contract notes (masaMLP, ADR 0005):
- ``forward`` always returns per-member outputs ``(n, k, out_dim)``. The
  trainer flattens members into rows for the loss (``weighted_loss``), so
  every objective — including customs — works unchanged; ``apply_transform``
  averages members on the prediction scale at eval/predict time.
- ``output_layer`` is the per-member head; its ``(k, out_dim)`` bias receives
  the ``(out_dim,)`` class-prior init by broadcast (``torch.Tensor.copy_``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class EnsembleHead(nn.Module):
    """k independent linear heads applied to shared per-member features.
    Input ``(n, k, d)`` -> ``(n, k, out)`` via ``weight`` ``(k, out, d)``."""

    def __init__(self, d: int, out_dim: int, k: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, out_dim, d))
        self.bias = nn.Parameter(torch.zeros(k, out_dim))
        bound = 1.0 / math.sqrt(d)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("nkd,kod->nko", x, self.weight) + self.bias


class TabM(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        out_dim: int,
        k: int = 32,
        d: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.1,
        adapter_std: float = 0.5,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.embedding = embedding
        self.k = k
        # Per-member multiplicative adapter on the shared embedding: the only
        # source of member diversity. Init ~ N(1, adapter_std) so members start
        # near the single model and diverge during training.
        self.adapter = nn.Parameter(torch.empty(k, embedding.d_out))
        with torch.no_grad():
            self.adapter.normal_(1.0, adapter_std)
        widths = [embedding.d_out] + [d] * n_blocks
        self.backbone = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(n_blocks)
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = EnsembleHead(d, out_dim, k)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = self.embedding(x_num, x_cat)               # (n, d_out) shared
        x = x.unsqueeze(1) * self.adapter              # (n, k, d_out) per-member
        for layer in self.backbone:
            x = self.dropout(torch.relu(layer(x)))     # (n, k, d) shared backbone
        return self.output_layer(x)                    # (n, k, out_dim) per-member heads
