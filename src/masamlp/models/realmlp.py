"""RealMLP — Holzmüller et al. 2024, "Better by Default: Strong Pre-Tuned
MLPs and Boosted Trees on Tabular Data" (arXiv:2407.04491).

This implements the RealMLP-TD-S architecture following the author's
MIT-licensed standalone reference (dholzmueller/realmlp-td-s_standalone):
a learnable per-feature scaling layer (via ``FeatureEmbedding``'s
``num_scaling``), neural-tangent-parametrized linear layers, SELU/Mish
activations, and a zero-initialized output layer. The matching training
recipe (coslog4 schedule, per-group learning rates, Adam betas, RSSC
preprocessing with one-hot categories) lives in :mod:`masamlp.presets`.
PBLD numeric embeddings (``num_embedding="pbld"``) upgrade it toward
RealMLP-TD.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding

_ACTIVATIONS = {"mish": nn.Mish, "selu": nn.SELU, "relu": nn.ReLU}


class NTPLinear(nn.Module):
    """Linear layer in the neural-tangent parametrization: weights are drawn
    from N(0, 1) and the forward pass scales by ``1/sqrt(in_features)``,
    which shifts the effective per-layer learning rates."""

    def __init__(self, in_features: int, out_features: int, zero_init: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return (1.0 / math.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class RealMLPNet(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        hidden_sizes: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "mish",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}. Expected {set(_ACTIVATIONS)}")
        self.embedding = embedding
        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        d_in = embedding.d_out
        for width in hidden_sizes:
            layers.append(NTPLinear(d_in, width))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.output_layer = NTPLinear(d_in, out_dim, zero_init=True)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.output_layer(self.trunk(self.embedding(x_num, x_cat)))

    def param_groups(self) -> list[dict]:
        """RealMLP's per-group learning rates: scaling layer 6x, NTP weights
        1x, NTP biases 0.1x; embedding parameters train at 1x."""
        ntp = [m for m in self.modules() if isinstance(m, NTPLinear)]
        weights = [m.weight for m in ntp]
        biases = [m.bias for m in ntp]
        scale = (
            list(self.embedding.scaling.parameters())
            if self.embedding.scaling is not None
            else []
        )
        assigned = {id(p) for p in weights + biases + scale}
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in assigned]
        groups = [
            {"params": scale, "lr_factor": 6.0},
            {"params": weights, "lr_factor": 1.0},
            {"params": biases, "lr_factor": 0.1},
            {"params": other, "lr_factor": 1.0},
        ]
        return [g for g in groups if g["params"]]
