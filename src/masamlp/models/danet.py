"""DANet — Deep Abstract Networks (Chen et al., AAAI 2022, arXiv:2112.02962).

Clean-room reimplementation following the MIT-licensed official repository
(WhatAShot/DANet). An Abstract Layer learns ``k`` sparse feature-group masks
(entmax15 over input features), applies a per-group linear map (grouped 1x1
convolution) with Ghost BatchNorm and a GLU-style gate, and sums the group
outputs. Basic blocks stack two Abstract Layers with a shortcut Abstract
Layer that always reads the raw (embedded) input features.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding
from masamlp.models.layers import GhostBatchNorm1d, entmax15


class AbstractLayer(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, k: int, virtual_batch_size: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.k = k
        self.d_out = d_out
        self.mask_weight = nn.Parameter(torch.rand(k, d_in))
        # Grouped 1x1 conv = one linear map per feature group; 2*d_out for the
        # sigmoid gate below.
        self.fc = nn.Conv1d(d_in * k, 2 * d_out * k, kernel_size=1, groups=k, bias=bias)
        gain = math.sqrt((d_in * k + 2 * d_out * k) / math.sqrt(d_in * k))
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        self.bn = GhostBatchNorm1d(2 * d_out * k, virtual_batch_size)

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        mask = entmax15(self.mask_weight, dim=-1)  # (k, d_in), rows on the simplex
        masked = mask.unsqueeze(0) * x.unsqueeze(1)  # (b, k, d_in)
        # The grouped 1x1 conv is per-group linear algebra; conv kernels take
        # a catastrophic slow path for kernel_size=1 / spatial=1 (KI-009:
        # ~76% of DANet's step time), so compute it as a batched matmul over
        # the same Conv1d parameters (state_dicts unchanged).
        w = self.fc.weight.squeeze(-1).reshape(self.k, 2 * self.d_out, -1)  # (k, 2*d_out, d_in)
        z = torch.einsum("bkd,kod->bko", masked, w)
        if self.fc.bias is not None:
            z = z + self.fc.bias.reshape(self.k, 2 * self.d_out)
        z = self.bn(z.reshape(b, -1))
        gate, value = z.reshape(b, self.k, 2, self.d_out).unbind(dim=2)
        return F.relu(torch.sigmoid(gate) * value).sum(dim=1)  # (b, d_out)


class _BasicBlock(nn.Module):
    def __init__(
        self, d_in: int, d_raw: int, base_outdim: int, k: int, virtual_batch_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = AbstractLayer(d_in, base_outdim // 2, k, virtual_batch_size)
        self.conv2 = AbstractLayer(base_outdim // 2, base_outdim, k, virtual_batch_size)
        self.shortcut_dropout = nn.Dropout(dropout)
        self.shortcut_layer = AbstractLayer(d_raw, base_outdim, k, virtual_batch_size)

    def forward(self, x_raw: Tensor, pre_out: Tensor | None = None) -> Tensor:
        out = self.conv2(self.conv1(x_raw if pre_out is None else pre_out))
        identity = self.shortcut_layer(self.shortcut_dropout(x_raw))
        return F.leaky_relu(out + identity, 0.01)


class DANet(nn.Module):
    """``n_layers`` counts Abstract Layers as in the paper (each block holds
    two on the main path), so the paper's DANet-20/24/32 correspond to
    ``n_layers=20/24/32``. The default is a lighter 8."""

    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        n_layers: int = 8,
        k: int = 5,
        base_outdim: int = 64,
        virtual_batch_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 2 or n_layers % 2 != 0:
            raise ValueError("n_layers must be an even number >= 2")
        self.embedding = embedding
        d_raw = embedding.d_out
        self.init_block = _BasicBlock(
            d_raw, d_raw, base_outdim, k, virtual_batch_size, dropout
        )
        self.blocks = nn.ModuleList(
            _BasicBlock(base_outdim, d_raw, base_outdim, k, virtual_batch_size, dropout)
            for _ in range(n_layers // 2 - 1)
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(base_outdim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Linear(512, out_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = self.embedding(x_num, x_cat)
        out = self.init_block(x)
        for block in self.blocks:
            out = block(x, out)
        return self.output_layer(self.head(self.head_dropout(out)))
