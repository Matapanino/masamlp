"""TabularResNet — the ResNet baseline of Gorishniy et al. 2021
("Revisiting Deep Learning Models for Tabular Data", arXiv:2106.11959).

Block: BN -> Linear(d -> d_hidden) -> ReLU -> Dropout -> Linear(-> d)
-> Dropout -> skip add. Head: BN -> ReLU -> Linear.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding


class _ResNetBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout1: float, dropout2: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.linear1 = nn.Linear(d, d_hidden)
        self.dropout1 = nn.Dropout(dropout1)
        self.linear2 = nn.Linear(d_hidden, d)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        z = self.dropout1(torch.relu(self.linear1(z)))
        z = self.dropout2(self.linear2(z))
        return x + z


class TabularResNet(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        d: int = 192,
        n_blocks: int = 3,
        d_hidden_factor: float = 2.0,
        dropout1: float = 0.25,
        dropout2: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.input_proj = nn.Linear(embedding.d_out, d)
        d_hidden = int(d * d_hidden_factor)
        self.blocks = nn.ModuleList(
            [_ResNetBlock(d, d_hidden, dropout1, dropout2) for _ in range(n_blocks)]
        )
        self.head_norm = nn.BatchNorm1d(d)
        self.output_layer = nn.Linear(d, out_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = self.input_proj(self.embedding(x_num, x_cat))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(torch.relu(self.head_norm(x)))
