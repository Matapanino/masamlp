"""MasaRegressor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from masamlp.core.objectives import BaseObjective, get_objective
from masamlp.sklearn import BaseMasaModel, resolve_custom_objective


class MasaRegressor(RegressorMixin, BaseMasaModel):
    """Tabular deep learning regressor (single- or multi-output).

    Targets are standardized by default (``target_standardize=True``) — the
    scale NNs train well on — and predictions are mapped back, so metrics and
    ``predict`` always live on the original scale. Standardization is skipped
    for objectives whose raw output is not on the target scale (e.g.
    Poisson's log rate).
    """

    def __init__(
        self,
        *,
        model: str = "resnet",
        model_params: dict[str, Any] | None = None,
        objective: Any = None,
        eval_metric: Any = None,
        early_stopping_rounds: int | None = None,
        n_epochs: int = 256,
        batch_size: int | str | None = "auto",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
        lr_scheduler: str = "none",
        grad_clip: float | None = None,
        num_embedding: str | None = None,
        numeric_scaler: str = "quantile",
        categorical_features: Any = "auto",
        target_standardize: bool = True,
        device: str = "auto",
        amp: str | bool = "auto",
        compile: bool = False,
        n_threads: int | None = None,
        verbose: int = 0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_clip=grad_clip,
            num_embedding=num_embedding,
            numeric_scaler=numeric_scaler,
            categorical_features=categorical_features,
            device=device,
            amp=amp,
            compile=compile,
            n_threads=n_threads,
            verbose=verbose,
            random_state=random_state,
        )
        self.target_standardize = target_standardize

    def _setup_target(self, y: np.ndarray) -> tuple[BaseObjective, np.ndarray]:
        if y.dtype == object:
            raise ValueError("regression targets must be numeric")
        spec = self.objective
        if spec is None:
            objective = get_objective("squared_error")
        elif isinstance(spec, str):
            objective = get_objective(spec)
        else:
            objective = resolve_custom_objective(
                spec, transform="identity", out_dim=None, target_dtype="float32"
            )

        y_enc = np.asarray(y, dtype=np.float64)
        # Raw outputs must live on the target scale for standardization to be
        # invertible; skip it for log-link objectives like Poisson.
        standardize = self.target_standardize and objective.transform_name == "identity"
        if standardize:
            self.target_mean_ = np.atleast_1d(y_enc.mean(axis=0)).astype(np.float64)
            std = np.atleast_1d(y_enc.std(axis=0)).astype(np.float64)
            self.target_std_ = np.where(std > 0, std, 1.0)
            mean = self.target_mean_ if y_enc.ndim == 2 else self.target_mean_[0]
            scale = self.target_std_ if y_enc.ndim == 2 else self.target_std_[0]
            y_enc = (y_enc - mean) / scale
        else:
            self.target_mean_ = None
            self.target_std_ = None
        return objective, y_enc

    def _encode_eval_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def _default_metric_name(self) -> str:
        return "rmse"

    def _inverse_target(self) -> Callable[[np.ndarray], np.ndarray] | None:
        if self.target_mean_ is None:
            return None
        mean, std = self.target_mean_, self.target_std_
        if mean.shape[0] == 1:
            return lambda p: p * std[0] + mean[0]
        return lambda p: p * std + mean

    def predict(self, X: Any) -> np.ndarray:
        """Predict on the original target scale; ``(n,)`` for single-output,
        ``(n, k)`` for multi-output."""
        pred = self._predict_transformed(X)
        inverse = self._inverse_target()
        return inverse(pred) if inverse is not None else pred
