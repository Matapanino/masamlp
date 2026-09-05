"""Opt-in model training terms (WP2, 2026-09-05).

A model may implement ``training_terms(batch, raw) -> TrainingTerms``.
``forward`` remains prediction-only. All trainable term state must belong
to the model as registered parameters/modules; persistent tensor state uses
registered buffers. See ``docs/training-terms.md`` for the complete contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class RowTerm:
    """Unreduced auxiliary losses shaped like ``raw.shape[:-1]``.

    The trainer applies sample weights and member normalization. A zero
    coefficient skips the term entirely, including its autograd graph.
    """

    values: Tensor
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("RowTerm coefficient must be finite and nonnegative")


@dataclass(frozen=True)
class ModelRegularizer:
    """A scalar penalty added as ``coefficient * value / normalizer``.

    ``normalizer`` is a fixed positive Python number (e.g. the number of
    table parameters), never a batch size, batch weight sum or step count.
    """

    value: Tensor
    normalizer: float
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.value.ndim != 0:
            raise ValueError("ModelRegularizer value must be a scalar tensor")
        if not math.isfinite(self.normalizer) or self.normalizer <= 0:
            raise ValueError("ModelRegularizer normalizer must be finite and positive")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("ModelRegularizer coefficient must be finite and nonnegative")


@dataclass(frozen=True)
class TrainingTerms:
    """Ephemeral tensors for one training step, never serialized themselves."""

    auxiliary: tuple[RowTerm, ...] = ()
    regularizers: tuple[ModelRegularizer, ...] = ()
