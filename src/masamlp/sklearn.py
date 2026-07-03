"""Shared estimator glue: the whole fit/predict flow lives here.

``MasaRegressor``/``MasaClassifier`` subclass :class:`BaseMasaModel` and only
own target handling (standardization / label encoding + class weights). The
LightGBM-style surface — ``fit(X, y, sample_weight=..., eval_set=...)``,
``evals_result_``, ``best_iteration_`` — matches repleafgbm.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from masamlp.core.device import resolve_device, resolve_device_plan
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
        self.eval_batch_size = eval_batch_size
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
        self.n_ens = n_ens
        self.ens_mode = ens_mode
        self.weight_decay_schedule = weight_decay_schedule
        self.ema_decay = ema_decay
        self.candidate_budget = candidate_budget
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

        With ``n_ens > 1``, members train with seeds ``random_state + i`` and
        each early-stops independently; ``evals_result_``/``best_iteration_``
        report the first member, and ``model_`` is ``models_[0]``.
        ``ens_mode="vectorized"`` trains all members in one vmapped pass for a
        ~k× speedup, but only for BatchNorm-free models (grn/realmlp/
        ft_transformer/gandalf/lnn); resnet/danet/modernnca keep the default
        ``ens_mode="loop"`` (a clear error is raised before training otherwise).
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

        if not isinstance(self.n_ens, int) or self.n_ens < 1:
            raise ValueError(f"n_ens must be a positive int, got {self.n_ens!r}")
        if self.ens_mode not in ("loop", "vectorized"):
            raise ValueError(f"ens_mode must be 'loop' or 'vectorized', got {self.ens_mode!r}")
        if self.ema_decay is not None and self.ens_mode == "vectorized" and self.n_ens > 1:
            raise ValueError(
                "ema_decay is not supported with ens_mode='vectorized'; use ens_mode='loop'"
            )
        resolved_params = {**self._model_param_defaults(), **(self.model_params or {})}
        bias = np.asarray(objective.init_bias(y_enc, weight), dtype=np.float32)

        # Ensemble members differ by their seed (init + shuffling; in
        # vectorized mode batches are shared), as in pytabkit's RealMLP
        # ensembling; predictions are averaged on the transformed scale
        # (probabilities for classification).
        members: list[torch.nn.Module] = []
        for member in range(self.n_ens):
            seed = None if self.random_state is None else self.random_state + member
            seed_everything(seed)
            model = build_model(
                self.model,
                resolved_params,
                n_num=x_num.shape[1],
                cat_cardinalities=pre.cat_cardinalities_,
                out_dim=out_dim,
                num_embedding=self.num_embedding,
            )
            if hasattr(model, "output_layer") and bias.shape == (out_dim,):
                with torch.no_grad():
                    model.output_layer.bias.copy_(torch.from_numpy(bias))
            members.append(model)

        if self.ens_mode == "vectorized" and self.n_ens > 1:
            # Fail fast, before any training or candidate setup, with a
            # model-aware message (BatchNorm / retrieval models are ineligible).
            from masamlp.core.ensemble import check_vectorizable

            check_vectorizable(members[0], self.model)

        if hasattr(members[0], "set_candidates"):
            # Retrieval models (TabR/ModernNCA) keep the training set as their
            # retrieval corpus. ``candidate_budget`` bounds that corpus with a
            # seeded, class-stratified subsample — and subsamples the aligned
            # training rows with it so each row's self-exclusion index stays
            # valid — to fix modernnca OOM / tabr superlinearity at scale.
            cand_idx = self._candidate_corpus_index(y_enc, n_rows)
            if cand_idx is not None:
                train = train.slice(torch.from_numpy(cand_idx))
                y_enc = y_enc[cand_idx]
            # Classification labels go in as class indices for the label
            # embedding; regression uses the (standardized) float targets.
            if hasattr(self, "classes_"):
                cand_y = torch.from_numpy(np.asarray(y_enc, dtype=np.int64))
            else:
                cand_y = objective.prepare_target(y_enc)
            for model in members:
                model.set_candidates(train.x_num, train.x_cat, cand_y)

        def member_config(
            seed: int | None,
            device: str | torch.device | None = None,
            seed_scope: str = "global",
        ) -> TrainerConfig:
            return TrainerConfig(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                eval_batch_size=self.eval_batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                optimizer=self.optimizer,
                betas=self.optimizer_betas,
                lr_scheduler=self.lr_scheduler,
                weight_decay_schedule=self.weight_decay_schedule,
                grad_clip=self.grad_clip,
                ema_decay=self.ema_decay,
                device=self.device if device is None else device,
                amp=self.amp,
                compile=self.compile,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=seed,
                seed_scope=seed_scope,
                verbose=self.verbose,
                n_threads=self.n_threads,
            )

        inverse = self._inverse_target()
        if self.ens_mode == "vectorized" and self.n_ens > 1:
            from masamlp.core.ensemble import fit_vectorized

            results = fit_vectorized(
                members, objective, train, eval_sets, metrics,
                member_config(self.random_state), inverse,
            )
            result = results[0]
        else:
            plan = resolve_device_plan(self.device, self.n_ens)
            if plan is not None and objective.torch_modules():
                warnings.warn(
                    "objectives with torch modules share state across members and "
                    "cannot train sharded; training sequentially on one device",
                    stacklevel=2,
                )
                plan = None
            if plan is not None:
                # Multiple GPUs detected: shard members across them, one
                # worker thread per device (see core/parallel.py).
                from masamlp.core.parallel import fit_members_sharded

                configs = [
                    member_config(
                        None if self.random_state is None else self.random_state + member,
                        device=plan[member],
                        seed_scope="device",
                    )
                    for member in range(self.n_ens)
                ]
                result = fit_members_sharded(
                    members, objective, train, eval_sets, metrics, configs, plan, inverse
                )[0]
            else:
                result = None
                for member, model in enumerate(members):
                    seed = None if self.random_state is None else self.random_state + member
                    member_result = Trainer().fit(
                        model, objective, train, eval_sets, metrics, member_config(seed), inverse
                    )
                    if result is None:
                        result = member_result

        self.preprocessor_ = pre
        self.models_ = members
        self.model_ = members[0]
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

    def _candidate_corpus_index(self, y_enc: np.ndarray, n_rows: int) -> np.ndarray | None:
        """Seeded corpus subsample (class-stratified for classification) for
        retrieval models, or ``None`` when ``candidate_budget`` imposes no
        bound. Only meaningful for ``tabr``/``modernnca``; ignored otherwise."""
        budget = self.candidate_budget
        if budget is None or n_rows <= budget:
            return None
        if not isinstance(budget, int) or budget < 1:
            raise ValueError(f"candidate_budget must be a positive int, got {budget!r}")
        from sklearn.model_selection import train_test_split

        stratify = y_enc if hasattr(self, "classes_") else None
        idx, _ = train_test_split(
            np.arange(n_rows),
            train_size=budget,
            random_state=self.random_state,
            stratify=stratify,
        )
        return np.sort(idx)

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
        transform = self.transform_name_
        members = getattr(self, "models_", None) or [self.model_]
        member_devices = {next(m.parameters()).device for m in members}
        if len({d for d in member_devices if d.type == "cuda"}) > 1:
            # Members are still sharded from a multi-GPU fit: predict on
            # their resident devices instead of dragging them onto one.
            from masamlp.core.parallel import predict_members_grouped

            data = TabularData(torch.from_numpy(x_num), torch.from_numpy(x_cat))
            preds = predict_members_grouped(
                members,
                data,
                lambda raw: apply_transform(raw, transform),
                self.eval_batch_size,
            )
            return preds[0] if len(preds) == 1 else np.mean(preds, axis=0)
        device = resolve_device(self.device)
        data = TabularData(torch.from_numpy(x_num), torch.from_numpy(x_cat)).to(device)
        preds = []
        for model in members:
            model.to(device)
            preds.append(
                predict_transformed(
                    model,
                    data,
                    lambda raw: apply_transform(raw, transform),
                    self.eval_batch_size,
                )
            )
        # Ensemble average on the transformed scale (probabilities for
        # classification), matching pytabkit's RealMLP ensembling.
        return preds[0] if len(preds) == 1 else np.mean(preds, axis=0)

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
