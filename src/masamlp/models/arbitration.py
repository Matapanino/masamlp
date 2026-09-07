"""Convex, per-row arbitration of numeric estimates on a common scale."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PostArbitrationFeature:
    """One semantic feature in an arbitration-reduced embedding frame.

    ``physical_slice`` addresses the flat vector emitted by
    :class:`FeatureEmbedding`; its order can differ from this semantic list
    because mixtures are numeric inputs while semantic mixtures come last.
    """

    name: str
    kind: str
    size: int
    physical_slice: slice


def post_arbitration_feature_layout(
    feature_chunk_sizes: list[int], n_numeric_chunks: int, n_heads: int
) -> list[PostArbitrationFeature]:
    """Describe post-gate semantic chunks and their embedding coordinates.

    ``feature_chunk_sizes`` is the physical output chunk layout of the reduced
    :class:`~masamlp.models.base.FeatureEmbedding`.  Its first
    ``n_numeric_chunks`` entries are numeric chunks, whose final ``n_heads``
    entries are the gate mixtures.  The returned semantic order is kept
    numeric chunks, categorical chunks, then mixtures.
    """
    if (
        isinstance(n_numeric_chunks, bool)
        or not isinstance(n_numeric_chunks, Integral)
        or n_numeric_chunks < 0
        or n_numeric_chunks > len(feature_chunk_sizes)
    ):
        raise ValueError("n_numeric_chunks must name a prefix of feature_chunk_sizes")
    if (
        isinstance(n_heads, bool)
        or not isinstance(n_heads, Integral)
        or n_heads < 1
        or n_heads > n_numeric_chunks
    ):
        raise ValueError("n_heads must be between one and n_numeric_chunks")
    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or size < 1
        for size in feature_chunk_sizes
    ):
        raise ValueError("feature_chunk_sizes must contain positive integers")

    offsets = [0]
    for size in feature_chunk_sizes:
        offsets.append(offsets[-1] + int(size))

    def feature(index: int, name: str, kind: str) -> PostArbitrationFeature:
        return PostArbitrationFeature(
            name, kind, int(feature_chunk_sizes[index]), slice(offsets[index], offsets[index + 1])
        )

    kept = n_numeric_chunks - n_heads
    return (
        [feature(index, f"numeric_{index}", "numeric") for index in range(kept)]
        + [
            feature(index, f"categorical_{index - n_numeric_chunks}", "categorical")
            for index in range(n_numeric_chunks, len(feature_chunk_sizes))
        ]
        + [
            feature(index, f"mixture_{index - kept}", "mixture")
            for index in range(kept, n_numeric_chunks)
        ]
    )


def arbitration_layout(
    n_inputs,
    estimator_idx,
    reliability_idx,
    n_heads=2,
    hidden_size=32,
    mode="conditional",
    keep_estimators=False,
):
    """Validate static input routing without creating parameters or drawing randomness."""
    for name, value in [("n_inputs", n_inputs), ("n_heads", n_heads), ("hidden_size", hidden_size)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if mode not in ("conditional", "constant"):
        raise ValueError("mode must be 'conditional' or 'constant'")
    if not isinstance(keep_estimators, bool):
        raise ValueError("keep_estimators must be boolean")
    groups = []
    for name, values in [("estimator_idx", estimator_idx), ("reliability_idx", reliability_idx)]:
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a nonempty list of numeric input indices")
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in values):
            raise ValueError(f"{name} must contain integers")
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= n_inputs:
            raise ValueError(f"{name} must be unique and within [0, n_inputs)")
        groups.append(set(values))
    if groups[0] & groups[1]:
        raise ValueError("estimator_idx and reliability_idx must be disjoint")
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

    def __init__(
        self,
        n_inputs,
        estimator_idx,
        reliability_idx,
        n_heads=2,
        hidden_size=32,
        mode="conditional",
        keep_estimators=False,
    ):
        super().__init__()
        kept, self.output_width = arbitration_layout(
            n_inputs, estimator_idx, reliability_idx, n_heads, hidden_size, mode, keep_estimators
        )
        self.n_inputs, self.n_heads = int(n_inputs), int(n_heads)
        self.n_estimators, self.mode = len(estimator_idx), mode
        self.register_buffer("estimator_idx", torch.tensor(estimator_idx, dtype=torch.long))
        self.register_buffer("reliability_idx", torch.tensor(reliability_idx, dtype=torch.long))
        self.register_buffer("kept_idx", torch.tensor(kept, dtype=torch.long))
        self.conditioner = None
        if mode == "conditional":
            self.conditioner = nn.Sequential(
                nn.Linear(len(estimator_idx) + len(reliability_idx), hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, n_heads * len(estimator_idx)),
            )
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
