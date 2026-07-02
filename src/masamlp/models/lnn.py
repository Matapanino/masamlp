"""TabularLNN — experimental liquid-network model for static tabular data.

There is no established "tabular LNN" in the literature. This model adapts
the closed-form continuous-time (CfC) cell of Hasani et al. 2022
("Closed-form Continuous-time Neural Networks"; reference implementation:
the Apache-2.0 `ncps` package, reimplemented here in-house) to non-sequential
input: the embedded feature vector is fed as a constant input while the cell
state is unrolled for ``n_steps`` virtual time steps — a recurrent refinement
of a latent state rather than a sequence model. See docs/lnn.md.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding


def _lecun_tanh(x: Tensor) -> Tensor:
    return 1.7159 * torch.tanh(0.666 * x)


class CfCCell(nn.Module):
    """One CfC update: gates interpolate between two candidate states with a
    learned, input-conditioned notion of elapsed time."""

    def __init__(
        self, d_input: int, d_hidden: int, d_backbone: int = 128, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.backbone = nn.Linear(d_input + d_hidden, d_backbone)
        self.dropout = nn.Dropout(dropout)
        self.ff1 = nn.Linear(d_backbone, d_hidden)
        self.ff2 = nn.Linear(d_backbone, d_hidden)
        self.time_a = nn.Linear(d_backbone, d_hidden)
        self.time_b = nn.Linear(d_backbone, d_hidden)

    def forward(self, u: Tensor, h: Tensor, t: float) -> Tensor:
        z = self.dropout(_lecun_tanh(self.backbone(torch.cat([u, h], dim=1))))
        candidate1 = torch.tanh(self.ff1(z))
        candidate2 = torch.tanh(self.ff2(z))
        t_interp = torch.sigmoid(self.time_a(z) * t + self.time_b(z))
        return candidate1 * (1.0 - t_interp) + t_interp * candidate2


class TabularLNN(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        d_hidden: int = 128,
        n_steps: int = 6,
        d_backbone: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.embedding = embedding
        self.n_steps = n_steps
        self.input_proj = nn.Linear(embedding.d_out, d_hidden)
        self.cell = CfCCell(d_hidden, d_hidden, d_backbone, dropout)
        self.h0 = nn.Parameter(torch.zeros(d_hidden))
        self.head_norm = nn.LayerNorm(d_hidden)
        self.output_layer = nn.Linear(d_hidden, out_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        u = self.input_proj(self.embedding(x_num, x_cat))
        h = self.h0.expand(u.shape[0], -1)
        for _ in range(self.n_steps):
            h = self.cell(u, h, t=1.0)
        return self.output_layer(self.head_norm(h))
