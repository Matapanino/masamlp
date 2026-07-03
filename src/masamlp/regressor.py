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
        eval_batch_size: int = 8192,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
        lr_scheduler: str = "none",
        grad_clip: float | None = None,
        num_embedding: str | None = None,
        numeric_scaler: str = "quantile",
        categorical_features: Any = "auto",
        cat_encoding: str = "embedding",
        optimizer_betas: tuple[float, float] | None = None,
        n_ens: int = 1,
        ens_mode: str = "loop",
        weight_decay_schedule: str = "none",
        ema_decay: float | None = None,
        candidate_budget: int | None = None,
        target_standardize: bool = True,
        clip_predictions: bool = False,
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
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_clip=grad_clip,
            num_embedding=num_embedding,
            numeric_scaler=numeric_scaler,
            categorical_features=categorical_features,
            cat_encoding=cat_encoding,
            optimizer_betas=optimizer_betas,
            n_ens=n_ens,
            ens_mode=ens_mode,
            weight_decay_schedule=weight_decay_schedule,
            ema_decay=ema_decay,
            candidate_budget=candidate_budget,
            device=device,
            amp=amp,
            compile=compile,
            n_threads=n_threads,
            verbose=verbose,
            random_state=random_state,
        )
        self.target_standardize = target_standardize
        self.clip_predictions = clip_predictions

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
        # RealMLP-style output clipping: predictions never leave the observed
        # target range (original scale).
        if self.clip_predictions:
            self.target_min_ = np.atleast_1d(y_enc.min(axis=0)).astype(np.float64)
            self.target_max_ = np.atleast_1d(y_enc.max(axis=0)).astype(np.float64)
        else:
            self.target_min_ = None
            self.target_max_ = None
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

    def _model_param_defaults(self) -> dict[str, Any]:
        if self.model == "realmlp":
            # RealMLP-TD-S uses Mish for regression.
            return {"num_scaling": True, "activation": "mish"}
        return super()._model_param_defaults()

    def _inverse_target(self) -> Callable[[np.ndarray], np.ndarray] | None:
        mean, std = self.target_mean_, self.target_std_
        tmin, tmax = self.target_min_, self.target_max_
        if mean is None and tmin is None:
            return None
        scalar = (mean if mean is not None else tmin).shape[0] == 1

        def inverse(p: np.ndarray) -> np.ndarray:
            if mean is not None:
                p = p * (std[0] if scalar else std) + (mean[0] if scalar else mean)
            if tmin is not None:
                p = np.clip(p, tmin[0] if scalar else tmin, tmax[0] if scalar else tmax)
            return p

        return inverse

    def predict(self, X: Any) -> np.ndarray:
        """Predict on the original target scale; ``(n,)`` for single-output,
        ``(n, k)`` for multi-output."""
        pred = self._predict_transformed(X)
        inverse = self._inverse_target()
        return inverse(pred) if inverse is not None else pred
