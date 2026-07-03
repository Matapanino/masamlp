"""MasaClassifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin

from masamlp.core.objectives import (
    BaseObjective,
    BinaryLogistic,
    MulticlassSoftmax,
    get_objective,
)
from masamlp.sklearn import BaseMasaModel, resolve_custom_objective


class MasaClassifier(ClassifierMixin, BaseMasaModel):
    """Tabular deep learning classifier (binary and multiclass).

    ``class_weight`` ("balanced" or a ``{label: weight}`` dict) multiplies
    into ``sample_weight``, so both flow through the same weighted-loss
    reduction. Custom objectives receive integer class labels and raw logits
    (one column for binary, ``n_classes`` for multiclass).
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
        class_weight: str | dict[Any, float] | None = None,
        label_smoothing: float = 0.0,
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
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def _setup_target(self, y: np.ndarray) -> tuple[BaseObjective, np.ndarray]:
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("y must contain at least two classes")

        spec = self.objective
        if spec is None:
            if n_classes == 2:
                objective: BaseObjective = BinaryLogistic(self.label_smoothing)
            else:
                objective = MulticlassSoftmax(n_classes, self.label_smoothing)
        elif isinstance(spec, str):
            if spec in ("binary_logistic", "multiclass_softmax"):
                objective = get_objective(spec, label_smoothing=self.label_smoothing)
            else:
                objective = get_objective(spec)
        else:
            objective = resolve_custom_objective(
                spec,
                transform="sigmoid" if n_classes == 2 else "softmax",
                out_dim=1 if n_classes == 2 else n_classes,
                target_dtype="float32" if n_classes == 2 else "int64",
            )
        if isinstance(objective, MulticlassSoftmax) and objective.n_classes is None:
            objective.n_classes = n_classes
        return objective, y_enc.astype(np.int64)

    def _encode_eval_target(self, y: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.classes_, y)
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        if not np.array_equal(self.classes_[idx], np.asarray(y)):
            raise ValueError("eval_set contains labels not present in the training data")
        return idx.astype(np.int64)

    def _adjust_weight(
        self, weight: np.ndarray | None, y_enc: np.ndarray
    ) -> np.ndarray | None:
        if self.class_weight is None:
            return weight
        n_classes = len(self.classes_)
        if self.class_weight == "balanced":
            counts = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
            per_class = len(y_enc) / (n_classes * np.maximum(counts, 1.0))
        elif isinstance(self.class_weight, dict):
            per_class = np.ones(n_classes, dtype=np.float64)
            for label, w in self.class_weight.items():
                matches = np.nonzero(self.classes_ == label)[0]
                if matches.size == 0:
                    raise ValueError(f"class_weight key {label!r} is not a training label")
                per_class[matches[0]] = float(w)
        else:
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        cw = per_class[y_enc].astype(np.float32)
        return cw if weight is None else weight * cw

    def _default_metric_name(self) -> str:
        return "logloss" if len(self.classes_) == 2 else "multi_logloss"

    def _model_param_defaults(self) -> dict[str, Any]:
        if self.model == "realmlp":
            # RealMLP-TD-S uses SELU for classification.
            return {"num_scaling": True, "activation": "selu"}
        if self.model in ("tabr", "modernnca"):
            # Retrieval models aggregate labels and need the class count.
            return {"n_label_classes": len(self.classes_)}
        return super()._model_param_defaults()

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities, shape ``(n, n_classes)`` (binary included)."""
        pred = self._predict_transformed(X)
        if pred.ndim == 1:
            return np.stack([1.0 - pred, pred], axis=1)
        return pred

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
