"""Convex, per-row arbitration of numeric estimates on a common scale."""
from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor, nn


def arbitration_layout(n_inputs, estimator_idx, reliability_idx, n_heads=2, hidden_size=32,
                       mode='conditional', keep_estimators=False):
    """Validate static input routing without creating parameters or drawing randomness."""
    for name, value in [('n_inputs', n_inputs), ('n_heads', n_heads), ('hidden_size', hidden_size)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if mode not in ('conditional', 'constant'):
        raise ValueError("mode must be 'conditional' or 'constant'")
    if not isinstance(keep_estimators, bool):
        raise ValueError('keep_estimators must be boolean')
    groups = []
    for name, values in [('estimator_idx', estimator_idx), ('reliability_idx', reliability_idx)]:
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f'{name} must be a nonempty list of numeric input indices')
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
            raise ValueError(f'{name} must contain integers')
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= n_inputs:
            raise ValueError(f'{name} must be unique and within [0, n_inputs)')
        groups.append(set(values))
    if groups[0] & groups[1]:
        raise ValueError('estimator_idx and reliability_idx must be disjoint')
    removed = groups[1] | (set() if keep_estimators else groups[0])
    kept = [i for i in range(n_inputs) if i not in removed]
    return kept, len(kept) + n_heads


class ReliabilityGate(nn.Module):
    """Replace estimates with convex mixtures; reliability is consumed only by the gate.

    Input indices address the preprocessed numeric block. Estimates must be on
    the same scale (e.g. logits); use ``numeric_passthrough_cols`` to preserve
    their units. Conditional weights see both the estimates and reliability.
    ``mode='constant'`` learns row-independent weights. ``keep_estimators=True``
    retains the original estimates and appends the mixtures. Other numeric
    coordinates keep their order, followed by the ``n_heads`` mixtures.

    ``weights(x)`` returns (rows, heads, estimates) without retaining mutable
    diagnostic state, so callers can inspect weights outside a training graph.
    """

    def __init__(self, n_inputs, estimator_idx, reliability_idx, n_heads=2, hidden_size=32,
                 mode='conditional', keep_estimators=False):
        super().__init__()
        kept, self.output_width = arbitration_layout(
            n_inputs, estimator_idx, reliability_idx, n_heads, hidden_size, mode, keep_estimators)
        self.n_inputs, self.n_heads = int(n_inputs), int(n_heads)
        self.n_estimators, self.mode = len(estimator_idx), mode
        self.register_buffer('estimator_idx', torch.tensor(estimator_idx, dtype=torch.long))
        self.register_buffer('reliability_idx', torch.tensor(reliability_idx, dtype=torch.long))
        self.register_buffer('kept_idx', torch.tensor(kept, dtype=torch.long))
        self.conditioner = None
        if mode == 'conditional':
            self.conditioner = nn.Sequential(
                nn.Linear(len(estimator_idx) + len(reliability_idx), hidden_size),
                nn.Tanh(), nn.Linear(hidden_size, n_heads * len(estimator_idx)))
            nn.init.normal_(self.conditioner[-1].weight, std=0.01)
            nn.init.zeros_(self.conditioner[-1].bias)
        else:
            self.logits = nn.Parameter(torch.zeros(n_heads, len(estimator_idx)))

    def weights(self, x: Tensor) -> Tensor:
        if self.conditioner is None:
            scores = self.logits.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            z = x.index_select(1, self.estimator_idx)
            r = x.index_select(1, self.reliability_idx)
            scores = self.conditioner(torch.cat([z, r], dim=1))
            scores = scores.reshape(-1, self.n_heads, self.n_estimators)
        return scores.softmax(dim=-1)

    def mix(self, x: Tensor) -> Tensor:
        z = x.index_select(1, self.estimator_idx)
        return (self.weights(x) * z.unsqueeze(1)).sum(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.index_select(1, self.kept_idx), self.mix(x)], dim=1)
