"""Shared estimator glue: the whole fit/predict flow lives here.

``MasaRegressor``/``MasaClassifier`` subclass :class:`BaseMasaModel` and only
own target handling (standardization / label encoding + class weights). The
LightGBM-style surface — ``fit(X, y, sample_weight=..., eval_set=...)``,
``evals_result_``, ``best_iteration_`` — matches repleafgbm.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from masamlp.core.device import resolve_device
from masamlp.core.metrics import BaseMetric, get_metric, make_metric
from masamlp.core.objectives import BaseObjective, apply_transform, make_objective
from masamlp.core.trainer import (
    EvalSet,
    Trainer,
    TrainerConfig,
    predict_transformed,
)
from masamlp.data.dataset import TabularData
from masamlp.data.preprocessing import TabularPreprocessor
from masamlp.models import build_model
from masamlp.utils.random import seed_everything
from masamlp.utils.validation import as_sample_weight, as_target, check_consistent_length


class BaseMasaModel(BaseEstimator):
    """Do not instantiate directly; use MasaRegressor or MasaClassifier."""

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
        cat_encoding: str = "embedding",
        optimizer_betas: tuple[float, float] | None = None,
        device: str = "auto",
        amp: str | bool = "auto",
        compile: bool = False,
        n_threads: int | None = None,
        verbose: int = 0,
        random_state: int | None = 42,
    ) -> None:
        self.model = model
        self.model_params = model_params
        self.objective = objective
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip
        self.num_embedding = num_embedding
        self.numeric_scaler = numeric_scaler
        self.categorical_features = categorical_features
        self.cat_encoding = cat_encoding
        self.optimizer_betas = optimizer_betas
        self.device = device
        self.amp = amp
        self.compile = compile
        self.n_threads = n_threads
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _setup_target(self, y: np.ndarray) -> tuple[BaseObjective, np.ndarray]:
        """Resolve the objective and encode/standardize the training target
        (sets fitted attrs like ``classes_`` / target statistics)."""
        raise NotImplementedError

    def _encode_eval_target(self, y: np.ndarray) -> np.ndarray:
        """Target as the metrics expect it (original scale / class indices)."""
        raise NotImplementedError

    def _default_metric_name(self) -> str:
        raise NotImplementedError

    def _adjust_weight(
        self, weight: np.ndarray | None, y_enc: np.ndarray
    ) -> np.ndarray | None:
        return weight

    def _inverse_target(self) -> Callable[[np.ndarray], np.ndarray] | None:
        return None

    def _model_param_defaults(self) -> dict[str, Any]:
        """Architecture defaults per model/task, overridable via
        ``model_params`` (e.g. RealMLP's SELU-for-classification)."""
        if self.model == "realmlp":
            return {"num_scaling": True}
        return {}

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
        eval_set: list[tuple[Any, Any]] | None = None,
    ) -> BaseMasaModel:
        """Fit on arrays/DataFrames.

        Args:
            X: Feature matrix (DataFrame or 2-D array). Categorical columns
                are detected from dtypes (or ``categorical_features``) and
                embedded; numeric columns are imputed and scaled.
            y: Target vector (regression also accepts an (n, k) matrix).
            sample_weight: Optional per-row weights. Non-negative and finite;
                every objective — including customs — sees the weighted
                reduction ``(loss * w).sum() / w.sum()``. The classifier
                multiplies these by ``class_weight``. None means uniform.
            eval_set: Optional list of ``(X, y)`` pairs evaluated after every
                epoch as ``valid_0``, ``valid_1``, ... in ``evals_result_``.
                The first metric on ``valid_0`` drives early stopping.
        """
        seed_everything(self.random_state)
        y_arr = as_target(y)

        pre = TabularPreprocessor(
            self.numeric_scaler, self.categorical_features, cat_encoding=self.cat_encoding
        )
        x_num, x_cat = pre.fit(X).transform(X)
        n_rows = x_num.shape[0]
        check_consistent_length(n_rows, y_arr)
        weight = as_sample_weight(sample_weight, n_rows)

        objective, y_enc = self._setup_target(y_arr)
        weight = self._adjust_weight(weight, y_enc)
        out_dim = objective.out_dim(y_enc)
        metrics = self._resolve_metrics()

        if self.early_stopping_rounds is not None and not eval_set:
            raise ValueError(
                "early_stopping_rounds requires eval_set; pass eval_set=[(X_val, y_val)]"
            )

        train = TabularData(
            x_num=torch.from_numpy(x_num),
            x_cat=torch.from_numpy(x_cat),
            y=objective.prepare_target(y_enc),
            weight=torch.from_numpy(weight) if weight is not None else None,
        )
        eval_sets: list[EvalSet] = []
        for i, pair in enumerate(eval_set or []):
            if len(pair) != 2:
                raise ValueError(
                    "eval_set entries must be (X, y) pairs; weighted eval sets are "
                    "not supported yet (metrics are unweighted)"
                )
            xe_num, xe_cat = pre.transform(pair[0])
            ye = as_target(pair[1])
            check_consistent_length(xe_num.shape[0], ye)
            eval_sets.append(
                EvalSet(
                    name=f"valid_{i}",
                    data=TabularData(torch.from_numpy(xe_num), torch.from_numpy(xe_cat)),
                    y_metric=self._encode_eval_target(ye),
                )
            )

        resolved_params = {**self._model_param_defaults(), **(self.model_params or {})}
        model = build_model(
            self.model,
            resolved_params,
            n_num=x_num.shape[1],
            cat_cardinalities=pre.cat_cardinalities_,
            out_dim=out_dim,
            num_embedding=self.num_embedding,
        )
        bias = np.asarray(objective.init_bias(y_enc, weight), dtype=np.float32)
        if hasattr(model, "output_layer") and bias.shape == (out_dim,):
            with torch.no_grad():
                model.output_layer.bias.copy_(torch.from_numpy(bias))
        if hasattr(model, "set_candidates"):
            # Retrieval models (TabR) keep the training set as their corpus.
            # Classification labels go in as class indices for the label
            # embedding; regression uses the (standardized) float targets.
            if hasattr(self, "classes_"):
                cand_y = torch.from_numpy(np.asarray(y_enc, dtype=np.int64))
            else:
                cand_y = objective.prepare_target(y_enc)
            model.set_candidates(train.x_num, train.x_cat, cand_y)

        config = TrainerConfig(
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer=self.optimizer,
            betas=self.optimizer_betas,
            lr_scheduler=self.lr_scheduler,
            grad_clip=self.grad_clip,
            device=self.device,
            amp=self.amp,
            compile=self.compile,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            verbose=self.verbose,
            n_threads=self.n_threads,
        )
        result = Trainer().fit(
            model, objective, train, eval_sets, metrics, config, self._inverse_target()
        )

        self.preprocessor_ = pre
        self.model_ = model
        self.resolved_model_params_ = resolved_params
        self.objective_ = objective
        self.transform_name_ = objective.transform_name
        self.out_dim_ = out_dim
        self.n_features_in_ = pre.n_features_in_
        self.feature_names_in_ = np.asarray(pre.feature_names_in_, dtype=object)
        self.evals_result_ = result.evals_result
        self.best_iteration_ = result.best_iteration
        self.best_score_ = result.best_score
        return self

    def _resolve_metrics(self) -> list[BaseMetric]:
        spec = self.eval_metric
        if spec is None:
            return [get_metric(self._default_metric_name())]
        items = spec if isinstance(spec, list | tuple) else [spec]
        metrics: list[BaseMetric] = []
        for item in items:
            if isinstance(item, str):
                metrics.append(get_metric(item))
            elif isinstance(item, BaseMetric):
                metrics.append(item)
            elif callable(item):
                metrics.append(make_metric(item))
            else:
                raise TypeError(
                    f"eval_metric entries must be str/BaseMetric/callable, got {item!r}"
                )
        return metrics

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call fit first"
            )

    def _predict_transformed(self, X: Any) -> np.ndarray:
        self._check_fitted()
        x_num, x_cat = self.preprocessor_.transform(X)
        device = resolve_device(self.device)
        self.model_.to(device)
        data = TabularData(torch.from_numpy(x_num), torch.from_numpy(x_cat)).to(device)
        transform = self.transform_name_
        return predict_transformed(self.model_, data, lambda raw: apply_transform(raw, transform))

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def save_model(self, path: str) -> None:
        """Save to a directory (manifest.json + preprocessor state +
        model_state.pt). Custom objective/metric objects are not serialized —
        prediction still works via the stored output transform."""
        from masamlp.core import serialization

        self._check_fitted()
        serialization.save_model_dir(self, path)

    @classmethod
    def load_model(cls, path: str) -> BaseMasaModel:
        from masamlp.core import serialization

        return serialization.load_model_dir(path, cls)


def resolve_custom_objective(
    fn: Any, transform: str, out_dim: int | None, target_dtype: str
) -> BaseObjective:
    """Wrap a user callable with task-appropriate defaults (used by the
    subclasses); pass-through for BaseObjective instances."""
    if isinstance(fn, BaseObjective):
        return fn
    return make_objective(fn, transform=transform, out_dim=out_dim, target_dtype=target_dtype)
