"""RealMLP — Holzmüller et al. 2024, "Better by Default: Strong Pre-Tuned
MLPs and Boosted Trees on Tabular Data" (arXiv:2407.04491).

The architecture follows the author's MIT-licensed standalone TD-S reference
(dholzmueller/realmlp-td-s_standalone): a learnable per-feature scaling
layer (via ``FeatureEmbedding``'s ``num_scaling``), neural-tangent-
parametrized linear layers, SELU/Mish activations, and a zero-initialized
output layer. The RealMLP-TD extras (per pytabkit's Apache-2.0
implementation) are available as options: parametric activations
(``x + (act(x) - x) * alpha`` with per-unit learnable ``alpha``, trained at
``act_lr_factor``), flat_cos-scheduled dropout, PBLD numeric embeddings with
their own learning-rate factor, and — at the trainer level — flat_cos
weight decay with zero decay on biases.

The matching training recipes live in :mod:`masamlp.presets`
(``realmlp_params`` for TD-S, ``realmlp_td_params`` for TD). Not replicated
from TD: pytabkit's data-driven init modes (``he+5``/``std``); this keeps
the reference's N(0,1) NTP init.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from masamlp.core.trainer import flat_cos
from masamlp.models.base import FeatureEmbedding

_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "mish": torch.nn.functional.mish,
    "selu": torch.selu,
    "relu": torch.relu,
}


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


class ParametricActivation(nn.Module):
    """RealMLP-TD: ``x + (act(x) - x) * alpha`` with per-unit learnable
    ``alpha`` (init 1 — plain activation), so each unit can interpolate
    between identity and the nonlinearity."""

    def __init__(self, n_units: int, fn: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.ones(n_units))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.fn(x) - x) * self.alpha


class ScheduledDropout(nn.Module):
    """Dropout whose probability is scaled by a factor the trainer updates
    per step (RealMLP-TD's flat_cos dropout schedule)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.base_p = p
        self.p_factor = 1.0

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(
            x, self.base_p * self.p_factor, training=self.training
        )


class RealMLPNet(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        hidden_sizes: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "mish",
        dropout: float = 0.0,
        dropout_schedule: str = "none",
        use_parametric_act: bool = False,
        act_lr_factor: float = 0.1,
        plr_lr_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}. Expected {set(_ACTIVATIONS)}")
        if dropout_schedule not in ("none", "flat_cos"):
            raise ValueError(f"Unknown dropout_schedule {dropout_schedule!r}")
        self.embedding = embedding
        self.dropout_schedule = dropout_schedule
        self.act_lr_factor = act_lr_factor
        self.plr_lr_factor = plr_lr_factor
        fn = _ACTIVATIONS[activation]
        act_cls = {"mish": nn.Mish, "selu": nn.SELU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        d_in = embedding.d_out
        for width in hidden_sizes:
            layers.append(NTPLinear(d_in, width))
            layers.append(ParametricActivation(width, fn) if use_parametric_act else act_cls())
            if dropout > 0:
                layers.append(
                    ScheduledDropout(dropout)
                    if dropout_schedule == "flat_cos"
                    else nn.Dropout(dropout)
                )
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.output_layer = NTPLinear(d_in, out_dim, zero_init=True)

    def set_schedule_t(self, t: float) -> None:
        """Trainer hook: update scheduled dropout with training progress."""
        if self.dropout_schedule == "flat_cos":
            factor = flat_cos(t)
            for module in self.modules():
                if isinstance(module, ScheduledDropout):
                    module.p_factor = factor

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.output_layer(self.trunk(self.embedding(x_num, x_cat)))

    def param_groups(self) -> list[dict]:
        """RealMLP's per-group hyperparameters: scaling layer at 6x lr, NTP
        weights 1x, NTP biases 0.1x lr with no weight decay, parametric
        activations at ``act_lr_factor``, numeric embeddings at
        ``plr_lr_factor``."""
        ntp = [m for m in self.modules() if isinstance(m, NTPLinear)]
        weights = [m.weight for m in ntp]
        biases = [m.bias for m in ntp]
        scale = (
            list(self.embedding.scaling.parameters())
            if self.embedding.scaling is not None
            else []
        )
        act = [
            m.alpha for m in self.modules() if isinstance(m, ParametricActivation)
        ]
        plr = (
            list(self.embedding.num_embedding.parameters())
            if self.embedding.num_embedding is not None
            else []
        )
        assigned = {id(p) for p in weights + biases + scale + act + plr}
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in assigned]
        groups = [
            {"params": scale, "lr_factor": 6.0},
            {"params": weights, "lr_factor": 1.0},
            {"params": biases, "lr_factor": 0.1, "wd_factor": 0.0},
            {"params": act, "lr_factor": self.act_lr_factor},
            {"params": plr, "lr_factor": self.plr_lr_factor},
            {"params": other, "lr_factor": 1.0},
        ]
        return [g for g in groups if g["params"]]
