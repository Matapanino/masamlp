"""DataFrame/ndarray -> tensor-ready arrays, fitted on training data only.

The preprocessor owns everything the models need to consume raw tabular
input: numeric scaling, median imputation, and one of two categorical
encodings — index encoding for embeddings (index 0 reserved for
unknown/missing) or RealMLP-style one-hot (binary columns become a single
±1 feature; unknown/missing rows are all-zeros) appended to the numeric
block *before* scaling, exactly like the RealMLP-TD pipeline.

Scalers: ``"quantile"`` (rank -> normal), ``"standard"``, ``"robust"``,
``"rssc"`` (RealMLP's robust-scale-smooth-clip: interquartile scaling with
min-max fallback followed by ``x / sqrt(1 + (x/3)^2)``), or ``"none"``.

State is plain JSON + arrays — no pickle — so saved models load with
``weights_only``-grade safety.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

_SCALERS = ("quantile", "standard", "robust", "rssc", "none")
# "hybrid" (RealMLP-TD): one-hot for small-cardinality columns
# (<= onehot_max_categories), index encoding + embeddings for the rest.
_CAT_ENCODINGS = ("embedding", "onehot", "hybrid")
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
        cat_encoding: str = "embedding",
        onehot_max_categories: int = 9,
    ) -> None:
        if numeric_scaler not in _SCALERS:
            raise ValueError(f"numeric_scaler must be one of {_SCALERS}")
        if cat_encoding not in _CAT_ENCODINGS:
            raise ValueError(f"cat_encoding must be one of {_CAT_ENCODINGS}")
        self.numeric_scaler = numeric_scaler
        self.categorical_features = categorical_features
        self.max_quantiles = max_quantiles
        self.cat_encoding = cat_encoding
        self.onehot_max_categories = onehot_max_categories

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

        self.categories_: list[list[str]] = []
        for i in self.categorical_idx_:
            col = df.iloc[:, i]
            keys = sorted({_cat_key(v) for v in col} - {_MISSING})
            self.categories_.append(keys)
        self._resolve_cat_split()

        num = self._matrix(df)
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
        elif self.numeric_scaler == "rssc":
            # RealMLP: interquartile range, min-max fallback where it is zero,
            # and factor 0 for train-constant features (stay constant later).
            self.center_ = np.median(num, axis=0)
            quant_diff = np.quantile(num, 0.75, axis=0) - np.quantile(num, 0.25, axis=0)
            zero_iqr = quant_diff == 0.0
            quant_diff[zero_iqr] = 0.5 * (num.max(axis=0) - num.min(axis=0))[zero_iqr]
            factors = 1.0 / (quant_diff + 1e-30)
            factors[quant_diff == 0.0] = 0.0
            self.factors_ = factors
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
        num = self._impute(self._matrix(df))
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
        elif self.numeric_scaler == "rssc":
            num = self.factors_ * (num - self.center_)
            num = num / np.sqrt(1.0 + (num / 3.0) ** 2)

        x_cat = np.zeros((df.shape[0], len(self.embed_pos_)), dtype=np.int64)
        for out_col, pos in enumerate(self.embed_pos_):
            mapping = {key: k + 1 for k, key in enumerate(self.categories_[pos])}
            col = df.iloc[:, self.categorical_idx_[pos]]
            x_cat[:, out_col] = [mapping.get(_cat_key(v), 0) for v in col]
        return num.astype(np.float32), x_cat

    def fit_transform(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(X).transform(X)

    def transform_width(self) -> tuple[int, int]:
        """Column counts of ``transform``'s ``(x_num, x_cat)`` outputs."""
        n_num = len(self.numeric_idx_) + sum(
            1 if len(self.categories_[pos]) == 2 else len(self.categories_[pos])
            for pos in self.onehot_pos_
        )
        return n_num, len(self.embed_pos_)

    def _resolve_cat_split(self) -> None:
        """Which categorical columns are one-hot vs embedding-encoded."""
        n_cats = len(self.categories_)
        if self.cat_encoding == "embedding":
            self.onehot_pos_: list[int] = []
        elif self.cat_encoding == "onehot":
            self.onehot_pos_ = list(range(n_cats))
        else:  # hybrid (RealMLP-TD): small cardinalities are one-hot
            self.onehot_pos_ = [
                pos
                for pos in range(n_cats)
                if len(self.categories_[pos]) <= self.onehot_max_categories
            ]
        onehot = set(self.onehot_pos_)
        self.embed_pos_ = [pos for pos in range(n_cats) if pos not in onehot]
        # +1 reserves index 0 for unknown/missing.
        self.cat_cardinalities_ = [
            len(self.categories_[pos]) + 1 for pos in self.embed_pos_
        ]

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

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Raw numeric columns, plus the one-hot-encoded categorical columns
        — appended before scaling so the scaler sees them, like RealMLP."""
        parts: list[np.ndarray] = []
        if self.numeric_idx_:
            parts.append(df.iloc[:, self.numeric_idx_].to_numpy(dtype=np.float64, copy=True))
        if self.onehot_pos_:
            parts.append(self._onehot_block(df))
        if not parts:
            return np.zeros((df.shape[0], 0), dtype=np.float64)
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def _onehot_block(self, df: pd.DataFrame) -> np.ndarray:
        n = df.shape[0]
        blocks: list[np.ndarray] = []
        for pos in self.onehot_pos_:
            i = self.categorical_idx_[pos]
            cats = self.categories_[pos]
            mapping = {key: k for k, key in enumerate(cats)}
            idx = np.array([mapping.get(_cat_key(v), -1) for v in df.iloc[:, i]])
            onehot = np.zeros((n, len(cats)), dtype=np.float64)
            known = idx >= 0  # unknown/missing stay all-zeros
            onehot[np.nonzero(known)[0], idx[known]] = 1.0
            if len(cats) == 2:
                # Binary: one ±1 feature; 0 keeps encoding missing/unknown.
                onehot = onehot[:, 0:1] - onehot[:, 1:2]
            blocks.append(onehot)
        return np.concatenate(blocks, axis=1)

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
            "cat_encoding": self.cat_encoding,
            "onehot_max_categories": self.onehot_max_categories,
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
        elif self.numeric_scaler == "rssc":
            arrays["center"] = self.center_
            arrays["factors"] = self.factors_
        return meta, arrays

    @classmethod
    def from_state(
        cls, meta: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> TabularPreprocessor:
        pre = cls(
            numeric_scaler=meta["numeric_scaler"],
            max_quantiles=meta["max_quantiles"],
            cat_encoding=meta.get("cat_encoding", "embedding"),
            onehot_max_categories=meta.get("onehot_max_categories", 9),
        )
        pre.feature_names_in_ = list(meta["feature_names_in"])
        pre.n_features_in_ = int(meta["n_features_in"])
        pre.numeric_idx_ = [int(i) for i in meta["numeric_idx"]]
        pre.categorical_idx_ = [int(i) for i in meta["categorical_idx"]]
        pre.categories_ = [list(c) for c in meta["categories"]]
        pre._resolve_cat_split()
        pre.medians_ = np.asarray(arrays["medians"], dtype=np.float64)
        if pre.numeric_scaler == "quantile":
            pre.quantiles_ = np.asarray(arrays["quantiles"], dtype=np.float64)
            pre.constant_ = np.asarray(arrays["constant"], dtype=bool)
        elif pre.numeric_scaler in ("standard", "robust"):
            pre.center_ = np.asarray(arrays["center"], dtype=np.float64)
            pre.scale_ = np.asarray(arrays["scale"], dtype=np.float64)
        elif pre.numeric_scaler == "rssc":
            pre.center_ = np.asarray(arrays["center"], dtype=np.float64)
            pre.factors_ = np.asarray(arrays["factors"], dtype=np.float64)
        return pre
