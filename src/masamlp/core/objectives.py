"""Training objectives.

An objective is a **per-sample** torch loss: :meth:`BaseObjective.per_sample_loss`
returns a ``(n,)`` tensor and the Trainer performs the weighted reduction
``(loss * w).sum() / w.sum()``. Keeping the reduction out of the objective is
what makes ``sample_weight`` (and ``class_weight``) work uniformly for every
objective, including user-supplied ones wrapped with :func:`make_objective`.

Raw model outputs always have shape ``(n, out_dim)``. ``transform`` maps them
to the prediction scale; its behaviour is identified by ``transform_name`` so
a saved model can predict without reconstructing the objective object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# exp() overflows float32 just past 88; clamping raw log-scale outputs keeps
# Poisson training recoverable instead of producing inf losses.
_MAX_LOG = 30.0


def apply_transform(raw: Tensor, transform_name: str) -> Tensor:
    if transform_name == "identity":
        return raw
    if transform_name == "sigmoid":
        return torch.sigmoid(raw)
    if transform_name == "softmax":
        return torch.softmax(raw, dim=1)
    if transform_name == "exp":
        return torch.exp(raw.clamp(max=_MAX_LOG))
    raise ValueError(f"Unknown transform {transform_name!r}")


def _weighted_mean(y: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
    if weight is None:
        return np.mean(y, axis=0)
    w = weight / weight.sum()
    return (y * w[:, None] if y.ndim == 2 else y * w).sum(axis=0)


def _weighted_quantile(y: np.ndarray, weight: np.ndarray | None, alpha: float) -> float:
    if weight is None:
        return float(np.quantile(y, alpha))
    order = np.argsort(y)
    cw = np.cumsum(weight[order])
    idx = int(np.searchsorted(cw, alpha * cw[-1]))
    return float(y[order][min(idx, len(y) - 1)])


class BaseObjective(ABC):
    """Abstract training objective (see module docstring for the contract)."""

    name: str = "base"
    transform_name: str = "identity"

    def out_dim(self, y: np.ndarray) -> int:
        """Number of raw model outputs for this target."""
        return 1 if y.ndim == 1 else y.shape[1]

    def prepare_target(self, y: np.ndarray) -> Tensor:
        """Target as the tensor ``per_sample_loss`` expects. Regression
        default: float32 ``(n, out_dim)``."""
        arr = np.asarray(y, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.from_numpy(arr)

    @abstractmethod
    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        """Per-row loss, shape ``(n,)``. Must not reduce over rows."""

    def transform(self, raw_pred: Tensor) -> Tensor:
        return apply_transform(raw_pred, self.transform_name)

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """Head-bias initialization: the weighted constant optimum, shape
        ``(out_dim,)``. Starting at the optimum removes the first epochs of
        merely learning the target's location (repleafgbm's init_score)."""
        return np.zeros(self.out_dim(y), dtype=np.float32)

    def torch_modules(self) -> list[nn.Module]:
        """Sub-modules with trainable parameters to add to the optimizer."""
        return []


class SquaredError(BaseObjective):
    name = "squared_error"

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        return ((raw_pred - y_true) ** 2).mean(dim=1)

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        return np.atleast_1d(_weighted_mean(y, weight)).astype(np.float32)


class MAEObjective(BaseObjective):
    name = "mae"

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        return (raw_pred - y_true).abs().mean(dim=1)

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        y2 = y[:, None] if y.ndim == 1 else y
        med = [_weighted_quantile(y2[:, j], weight, 0.5) for j in range(y2.shape[1])]
        return np.asarray(med, dtype=np.float32)


class Huber(BaseObjective):
    """Huber loss. With the regressor's default target standardization,
    ``delta=1.0`` sits at roughly one standard deviation."""

    name = "huber"

    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        err = (raw_pred - y_true).abs()
        quad = 0.5 * err**2
        lin = self.delta * (err - 0.5 * self.delta)
        return torch.where(err <= self.delta, quad, lin).mean(dim=1)

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        return np.atleast_1d(_weighted_mean(y, weight)).astype(np.float32)


class Quantile(BaseObjective):
    """Pinball loss for the ``alpha`` quantile."""

    name = "quantile"

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        err = y_true - raw_pred
        return torch.maximum(self.alpha * err, (self.alpha - 1.0) * err).mean(dim=1)

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        y2 = y[:, None] if y.ndim == 1 else y
        q = [_weighted_quantile(y2[:, j], weight, self.alpha) for j in range(y2.shape[1])]
        return np.asarray(q, dtype=np.float32)


class PoissonRegression(BaseObjective):
    """Poisson negative log-likelihood (up to the y! constant); raw output is
    the log rate, predictions are ``exp(raw)``."""

    name = "poisson"
    transform_name = "exp"

    def prepare_target(self, y: np.ndarray) -> Tensor:
        if np.any(np.asarray(y) < 0):
            raise ValueError("poisson objective requires non-negative targets")
        return super().prepare_target(y)

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        raw = raw_pred.clamp(max=_MAX_LOG)
        return (torch.exp(raw) - y_true * raw).mean(dim=1)

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        mean = float(np.atleast_1d(_weighted_mean(y, weight))[0])
        return np.asarray([np.log(max(mean, 1e-12))], dtype=np.float32)


class BinaryLogistic(BaseObjective):
    """Sigmoid cross-entropy on 0/1 labels; raw output is one logit."""

    name = "binary_logistic"
    transform_name = "sigmoid"

    def __init__(self, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = label_smoothing

    def out_dim(self, y: np.ndarray) -> int:
        return 1

    def prepare_target(self, y: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(y, dtype=np.float32))

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        s = self.label_smoothing
        target = y_true * (1.0 - s) + 0.5 * s
        return F.binary_cross_entropy_with_logits(raw_pred[:, 0], target, reduction="none")

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        s = self.label_smoothing
        p = float(np.atleast_1d(_weighted_mean(y.astype(np.float64), weight))[0])
        p = np.clip(p * (1.0 - s) + 0.5 * s, 1e-6, 1.0 - 1e-6)
        return np.asarray([np.log(p / (1.0 - p))], dtype=np.float32)


class MulticlassSoftmax(BaseObjective):
    """Softmax cross-entropy on integer labels ``0..K-1``."""

    name = "multiclass_softmax"
    transform_name = "softmax"

    def __init__(self, n_classes: int | None = None, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing

    def out_dim(self, y: np.ndarray) -> int:
        return self.n_classes if self.n_classes is not None else int(np.max(y)) + 1

    def prepare_target(self, y: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(y, dtype=np.int64))

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        if raw_pred.ndim == 3:
            # Weight-shared ensembles (TabM) emit per-member logits (n, k, K):
            # train each member independently and average the losses
            # (mean-of-per-member cross-entropy). Inert for every 2D model.
            n, k, _ = raw_pred.shape
            per_member = F.cross_entropy(
                raw_pred.transpose(1, 2),
                y_true.unsqueeze(1).expand(n, k),
                reduction="none",
                label_smoothing=self.label_smoothing,
            )  # (n, k)
            return per_member.mean(dim=1)
        return F.cross_entropy(
            raw_pred, y_true, reduction="none", label_smoothing=self.label_smoothing
        )

    def init_bias(self, y: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        k = self.out_dim(y)
        onehot = np.eye(k, dtype=np.float64)[np.asarray(y, dtype=np.int64)]
        priors = np.clip(_weighted_mean(onehot, weight), 1e-12, None)
        return np.log(priors).astype(np.float32)


class _CallableObjective(BaseObjective):
    """Adapter wrapping a user ``(y_true, raw_pred) -> per-sample Tensor``."""

    def __init__(
        self,
        fn: Callable[[Tensor, Tensor], Tensor],
        name: str,
        transform_name: str,
        out_dim: int | None,
        target_dtype: str,
    ) -> None:
        self._fn = fn
        self.name = name
        self.transform_name = transform_name
        self._out_dim = out_dim
        self._target_dtype = target_dtype

    def out_dim(self, y: np.ndarray) -> int:
        if self._out_dim is not None:
            return self._out_dim
        return super().out_dim(y)

    def prepare_target(self, y: np.ndarray) -> Tensor:
        if self._target_dtype == "int64":
            return torch.from_numpy(np.asarray(y, dtype=np.int64))
        return super().prepare_target(y)

    def per_sample_loss(self, y_true: Tensor, raw_pred: Tensor) -> Tensor:
        loss = self._fn(y_true, raw_pred)
        if loss.ndim != 1:
            raise ValueError(
                f"custom objective {self.name!r} must return a per-sample (n,) tensor, "
                f"got shape {tuple(loss.shape)}; do not reduce — the trainer applies "
                "the weighted reduction"
            )
        return loss

    def torch_modules(self) -> list[nn.Module]:
        return [self._fn] if isinstance(self._fn, nn.Module) else []


def make_objective(
    fn: Callable[[Tensor, Tensor], Tensor],
    name: str | None = None,
    transform: str = "identity",
    out_dim: int | None = None,
    target_dtype: str = "float32",
) -> BaseObjective:
    """Wrap a callable (or ``nn.Module``) as a training objective.

    Args:
        fn: ``(y_true, raw_pred) -> Tensor`` of shape ``(n,)`` — the
            **unreduced** per-sample loss. ``raw_pred`` is ``(n, out_dim)``;
            ``y_true`` is float32 ``(n, out_dim)`` (or int64 ``(n,)`` with
            ``target_dtype="int64"``). If ``fn`` is an ``nn.Module``, its
            parameters are trained jointly with the model.
        name: Identifier used in logs and serialization metadata.
        transform: How raw outputs map to predictions:
            ``"identity" | "sigmoid" | "softmax" | "exp"``.
        out_dim: Number of raw outputs; defaults to the target's width
            (softmax losses on integer labels must pass ``n_classes`` here).
        target_dtype: ``"float32"`` (default) or ``"int64"`` for class labels.
    """
    if not callable(fn):
        raise TypeError(f"make_objective expects a callable, got {type(fn).__name__}")
    if target_dtype not in ("float32", "int64"):
        raise ValueError("target_dtype must be 'float32' or 'int64'")
    apply_transform(torch.zeros(1, 1), transform)  # validate the name eagerly
    return _CallableObjective(
        fn,
        name or getattr(fn, "__name__", type(fn).__name__),
        transform,
        out_dim,
        target_dtype,
    )


_OBJECTIVE_REGISTRY: dict[str, type[BaseObjective]] = {
    SquaredError.name: SquaredError,
    "l2": SquaredError,
    "mse": SquaredError,
    MAEObjective.name: MAEObjective,
    "l1": MAEObjective,
    Huber.name: Huber,
    Quantile.name: Quantile,
    PoissonRegression.name: PoissonRegression,
    BinaryLogistic.name: BinaryLogistic,
    MulticlassSoftmax.name: MulticlassSoftmax,
}


def get_objective(name: str, **kwargs: object) -> BaseObjective:
    if name not in _OBJECTIVE_REGISTRY:
        raise ValueError(f"Unknown objective {name!r}. Available: {sorted(_OBJECTIVE_REGISTRY)}")
    return _OBJECTIVE_REGISTRY[name](**kwargs)  # type: ignore[arg-type]
