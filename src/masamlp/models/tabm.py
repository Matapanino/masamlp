"""TabM — Gorishniy et al. 2024, "TabM: Advancing Tabular Deep Learning
with Parameter-Efficient Ensembling" (arXiv:2410.24210).

``variant="mini"`` preserves masaMLP's original TabM-mini byte for byte: one
member adapter before a shared plain MLP and independent heads.
``variant="full"`` is the paper's headline TabM: every backbone linear has a
shared weight plus member-specific ``r``, ``s``, and bias parameters, followed
by an independent head. Both variants keep a static ``(n, k, d)`` member axis.

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


def _init_chunked_(tensor: Tensor, chunks: list[int], distribution: str) -> None:
    if sum(chunks) != tensor.shape[-1]:
        raise ValueError(f"chunks sum to {sum(chunks)}, expected {tensor.shape[-1]}")
    start = 0
    with torch.no_grad():
        for size in chunks:
            value = torch.empty(*tensor.shape[:-1], 1)
            if distribution == "normal":
                value.normal_()
            elif distribution == "signs":
                value.bernoulli_(0.5).mul_(2.0).add_(-1.0)
            else:
                raise ValueError(f"Unknown adapter distribution {distribution!r}")
            tensor[..., start : start + size] = value
            start += size


class BatchEnsembleLinear(nn.Module):
    """Plain TabM linear: ``s * W(r * x) + b`` with one shared ``W``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        first_r_init: str = "ones",
        first_r_chunks: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.empty(k, in_features))
        self.s = nn.Parameter(torch.empty(k, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if first_r_init == "ones":
            nn.init.ones_(self.r)
        elif first_r_chunks is None:
            if first_r_init == "normal":
                nn.init.normal_(self.r)
            elif first_r_init == "signs":
                with torch.no_grad():
                    self.r.bernoulli_(0.5).mul_(2.0).add_(-1.0)
            else:
                raise ValueError(f"Unknown adapter initialization {first_r_init!r}")
        else:
            _init_chunked_(self.r, first_r_chunks, first_r_init)
        nn.init.ones_(self.s)
        initial_bias = torch.empty(out_features)
        nn.init.uniform_(initial_bias, -bound, bound)
        with torch.no_grad():
            self.bias.copy_(initial_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = (x * self.r) @ self.weight.T
        return x * self.s + self.bias


class FullEnsembleHead(nn.Module):
    """Independent full-TabM heads with official uniform weight and bias."""

    def __init__(self, d: int, out_dim: int, k: int) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(d)
        self.weight = nn.Parameter(torch.empty(k, out_dim, d))
        self.bias = nn.Parameter(torch.empty(k, out_dim))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("nkd,kod->nko", x, self.weight) + self.bias


class TabM(nn.Module):
    supports_member_batches = True

    def __init__(
        self,
        embedding: nn.Module,
        out_dim: int,
        variant: str = "mini",
        k: int = 32,
        d: int = 512,
        n_blocks: int | None = None,
        dropout: float = 0.1,
        adapter_std: float = 0.5,
    ) -> None:
        super().__init__()
        if variant not in ("mini", "full"):
            raise ValueError(f"variant must be 'mini' or 'full', got {variant!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n_blocks is None:
            n_blocks = 2 if variant == "full" and embedding.has_num_embedding else 3
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.embedding = embedding
        self.k = k
        self.variant = variant
        widths = [embedding.d_out] + [d] * n_blocks
        if variant == "mini":
            # Keep this construction order identical to masaMLP 0.9.2.
            self.adapter = nn.Parameter(torch.empty(k, embedding.d_out))
            with torch.no_grad():
                self.adapter.normal_(1.0, adapter_std)
            self.backbone = nn.ModuleList(
                nn.Linear(widths[i], widths[i + 1]) for i in range(n_blocks)
            )
            self.dropout = nn.Dropout(dropout)
            self.output_layer = EnsembleHead(d, out_dim, k)
        else:
            first_r_init = "normal" if embedding.has_num_embedding else "signs"
            self.backbone = nn.ModuleList(
                BatchEnsembleLinear(
                    widths[i],
                    widths[i + 1],
                    k,
                    first_r_init=first_r_init if i == 0 else "ones",
                    first_r_chunks=embedding.feature_chunk_sizes if i == 0 else None,
                )
                for i in range(n_blocks)
            )
            self.dropout = nn.Dropout(dropout)
            self.output_layer = FullEnsembleHead(d, out_dim, k)
            # The official head begins with independent uniform biases rather
            # than the estimator's optional target-prior bias initialization.
            self.preserve_output_bias_init = True

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        member_batches = x_num.ndim == 3
        if member_batches:
            n, k = x_num.shape[:2]
            if k != self.k or x_cat.ndim != 3 or x_cat.shape[:2] != (n, k):
                raise ValueError(
                    f"member batches must have leading shape (n, {self.k})"
                )
            x = self.embedding(x_num.flatten(0, 1), x_cat.flatten(0, 1)).unflatten(
                0, (n, k)
            )
        else:
            x = self.embedding(x_num, x_cat).unsqueeze(1)
        if self.variant == "mini":
            x = x * self.adapter
        for layer in self.backbone:
            x = self.dropout(torch.relu(layer(x)))
        return self.output_layer(x)
