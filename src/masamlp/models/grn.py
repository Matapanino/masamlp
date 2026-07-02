"""GRNNet — a stack of Gated Residual Networks.

The Gated Residual Network is the building block of the Temporal Fusion
Transformer (Lim et al. 2021, arXiv:1912.09363, section 4.1):
``LayerNorm(x + GLU(W1 * ELU(W2 x + b2) + b1))`` — a residual block whose
gate can suppress the whole nonlinear branch, letting the network modulate
its own depth. There is no canonical standalone "GRN tabular model" paper;
like ``lnn``, this composition (input projection -> GRN stack -> linear
head over embedded features) is masaMLP's own, clearly labeled as such.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding


class GatedResidualBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden = nn.Linear(d, d_hidden)
        self.out = nn.Linear(d_hidden, d)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d, 2 * d)  # GLU: sigmoid(a) * b
        self.norm = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        h = self.out(torch.nn.functional.elu(self.hidden(x)))
        gate, value = self.gate(self.dropout(h)).chunk(2, dim=-1)
        return self.norm(x + torch.sigmoid(gate) * value)


class GRNNet(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        d: int = 128,
        d_hidden: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.input_proj = nn.Linear(embedding.d_out, d)
        self.blocks = nn.ModuleList(
            GatedResidualBlock(d, d_hidden, dropout) for _ in range(n_blocks)
        )
        self.output_layer = nn.Linear(d, out_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = self.input_proj(self.embedding(x_num, x_cat))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)
