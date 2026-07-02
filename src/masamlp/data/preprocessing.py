"""DataFrame/ndarray -> tensor-ready arrays, fitted on training data only.

The preprocessor owns everything the models need to consume raw tabular
input: numeric scaling (quantile-normal by default), median imputation, and
categorical index encoding (index 0 is reserved for unknown/missing, so
embeddings use ``cardinality + 1`` rows). Its state is plain JSON + arrays —
no pickle — so saved models load with ``weights_only``-grade safety.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

_SCALERS = ("quantile", "standard", "robust", "none")
# Category values are keyed by str() at fit and transform time: consistent,
# JSON-serializable, and independent of the input container's dtype.
_MISSING = "__nan__"


def _cat_key(value: Any) -> str:
    # pd.isna covers None, float NaN, pd.NA, and pd.NaT (cell values are
    # scalars here); plain strings pass through.
    if value is None or (not isinstance(value, str | bytes) and pd.isna(value)):
        return _MISSING
    return str(value)


def _normal_ppf(u: np.ndarray) -> np.ndarray:
    # Inverse normal CDF via torch.erfinv — torch is a hard dependency, scipy
    # is not.
    t = torch.from_numpy(2.0 * u - 1.0)
    return (math.sqrt(2.0) * torch.erfinv(t)).numpy()


class TabularPreprocessor:
    """Fit on train data; transform any split to ``(x_num, x_cat)`` arrays."""

    def __init__(
        self,
        numeric_scaler: str = "quantile",
        categorical_features: str | list[int] | list[str] = "auto",
        max_quantiles: int = 1000,
    ) -> None:
        if numeric_scaler not in _SCALERS:
            raise ValueError(f"numeric_scaler must be one of {_SCALERS}")
        self.numeric_scaler = numeric_scaler
        self.categorical_features = categorical_features
        self.max_quantiles = max_quantiles

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, X: Any) -> TabularPreprocessor:
        df = self._as_frame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        self.n_features_in_ = df.shape[1]

        cat_idx = self._resolve_categorical(df)
        self.categorical_idx_ = sorted(cat_idx)
        self.numeric_idx_ = [i for i in range(df.shape[1]) if i not in cat_idx]

        num = self._numeric_block(df)
        self.medians_ = np.zeros(num.shape[1], dtype=np.float64)
        for j in range(num.shape[1]):
            col = num[:, j]
            finite = col[np.isfinite(col)]
            self.medians_[j] = np.median(finite) if finite.size else 0.0
        num = self._impute(num)

        if self.numeric_scaler == "quantile":
            m = int(min(self.max_quantiles, max(num.shape[0], 2)))
            probs = np.linspace(0.0, 1.0, m)
            self.quantiles_ = (
                np.quantile(num, probs, axis=0).T if num.shape[1] else np.zeros((0, m))
            )
            self.constant_ = (
                self.quantiles_[:, 0] == self.quantiles_[:, -1]
                if num.shape[1]
                else np.zeros(0, dtype=bool)
            )
        elif self.numeric_scaler == "standard":
            self.center_ = num.mean(axis=0)
            scale = num.std(axis=0)
            self.scale_ = np.where(scale > 0, scale, 1.0)
        elif self.numeric_scaler == "robust":
            self.center_ = np.median(num, axis=0)
            scale = np.quantile(num, 0.75, axis=0) - np.quantile(num, 0.25, axis=0)
            self.scale_ = np.where(scale > 0, scale, 1.0)

        self.categories_: list[list[str]] = []
        for i in self.categorical_idx_:
            col = df.iloc[:, i]
            keys = sorted({_cat_key(v) for v in col} - {_MISSING})
            self.categories_.append(keys)
        # +1 reserves index 0 for unknown/missing.
        self.cat_cardinalities_ = [len(c) + 1 for c in self.categories_]
        return self

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #
    def transform(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        df = self._as_frame(X)
        if df.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {df.shape[1]} features, expected {self.n_features_in_}"
            )
        num = self._impute(self._numeric_block(df))
        if self.numeric_scaler == "quantile":
            out = np.empty_like(num)
            m = self.quantiles_.shape[1] if num.shape[1] else 0
            eps = 1e-7
            for j in range(num.shape[1]):
                if self.constant_[j]:
                    out[:, j] = 0.0
                    continue
                u = np.interp(num[:, j], self.quantiles_[j], np.linspace(0.0, 1.0, m))
                out[:, j] = _normal_ppf(np.clip(u, eps, 1.0 - eps))
            num = out
        elif self.numeric_scaler in ("standard", "robust"):
            num = (num - self.center_) / self.scale_

        x_cat = np.zeros((df.shape[0], len(self.categorical_idx_)), dtype=np.int64)
        for pos, i in enumerate(self.categorical_idx_):
            mapping = {key: k + 1 for k, key in enumerate(self.categories_[pos])}
            col = df.iloc[:, i]
            x_cat[:, pos] = [mapping.get(_cat_key(v), 0) for v in col]
        return num.astype(np.float32), x_cat

    def fit_transform(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {arr.shape}")
        return pd.DataFrame(arr)

    def _resolve_categorical(self, df: pd.DataFrame) -> set[int]:
        spec = self.categorical_features
        if spec == "auto":
            # dtype-API detection, not string matching: pandas 3 renamed the
            # default string dtype to ``str``. Anything non-numeric (object,
            # str/string, category, datetime) plus bool is categorical.
            types = pd.api.types
            return {
                i
                for i, dtype in enumerate(df.dtypes)
                if types.is_bool_dtype(dtype) or not types.is_numeric_dtype(dtype)
            }
        if spec is None:
            return set()
        idx: set[int] = set()
        names = [str(c) for c in df.columns]
        for item in spec:
            if isinstance(item, str):
                if item not in names:
                    raise ValueError(f"categorical feature {item!r} not in columns")
                idx.add(names.index(item))
            else:
                idx.add(int(item))
        return idx

    def _numeric_block(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_idx_:
            return np.zeros((df.shape[0], 0), dtype=np.float64)
        return df.iloc[:, self.numeric_idx_].to_numpy(dtype=np.float64, copy=True)

    def _impute(self, num: np.ndarray) -> np.ndarray:
        mask = ~np.isfinite(num)
        if mask.any():
            num[mask] = np.broadcast_to(self.medians_, num.shape)[mask]
        return num

    # ------------------------------------------------------------------ #
    # Serialization (JSON + arrays; no pickle)
    # ------------------------------------------------------------------ #
    def get_state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        meta: dict[str, Any] = {
            "numeric_scaler": self.numeric_scaler,
            "max_quantiles": self.max_quantiles,
            "feature_names_in": self.feature_names_in_,
            "n_features_in": self.n_features_in_,
            "numeric_idx": self.numeric_idx_,
            "categorical_idx": self.categorical_idx_,
            "categories": self.categories_,
        }
        arrays: dict[str, np.ndarray] = {"medians": self.medians_}
        if self.numeric_scaler == "quantile":
            arrays["quantiles"] = self.quantiles_
            arrays["constant"] = self.constant_
        elif self.numeric_scaler in ("standard", "robust"):
            arrays["center"] = self.center_
            arrays["scale"] = self.scale_
        return meta, arrays

    @classmethod
    def from_state(
        cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> TabularPreprocessor:
        pre = cls(numeric_scaler=meta["numeric_scaler"], max_quantiles=meta["max_quantiles"])
        pre.feature_names_in_ = list(meta["feature_names_in"])
        pre.n_features_in_ = int(meta["n_features_in"])
        pre.numeric_idx_ = [int(i) for i in meta["numeric_idx"]]
        pre.categorical_idx_ = [int(i) for i in meta["categorical_idx"]]
        pre.categories_ = [list(c) for c in meta["categories"]]
        pre.cat_cardinalities_ = [len(c) + 1 for c in pre.categories_]
        pre.medians_ = np.asarray(arrays["medians"], dtype=np.float64)
        if pre.numeric_scaler == "quantile":
            pre.quantiles_ = np.asarray(arrays["quantiles"], dtype=np.float64)
            pre.constant_ = np.asarray(arrays["constant"], dtype=bool)
        elif pre.numeric_scaler in ("standard", "robust"):
            pre.center_ = np.asarray(arrays["center"], dtype=np.float64)
            pre.scale_ = np.asarray(arrays["scale"], dtype=np.float64)
        return pre
