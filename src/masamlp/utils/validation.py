"""Input validation helpers shared by the estimators."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_sample_weight(weight: Any, n_rows: int) -> np.ndarray | None:
    """Validate per-row weights: 1-D, length ``n_rows``, finite, non-negative.

    Returns float32 (the training dtype) or None for uniform weights.
    """
    if weight is None:
        return None
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    if w.shape[0] != n_rows:
        raise ValueError(f"sample_weight has length {w.shape[0]}, expected {n_rows}")
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must be finite")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative")
    if not np.any(w > 0):
        raise ValueError("sample_weight must contain at least one positive weight")
    return w


def as_target(y: Any) -> np.ndarray:
    """Validate a target: numeric ndarray, 1-D or 2-D, no NaN."""
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D, got shape {arr.shape}")
    if arr.dtype == object:
        # Class labels may legitimately be strings; numeric checks happen later.
        return arr
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError("y contains NaN or infinite values")
    return arr


def check_consistent_length(n_rows: int, y: np.ndarray) -> None:
    if y.shape[0] != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {y.shape[0]}")
