"""Validation and fitted-coordinate conversion for explicit PLE state."""
from collections.abc import Mapping
from typing import Any

import numpy as np

from masamlp.data.preprocessing import TabularPreprocessor


def explicit_ple_bins(
    bins: Mapping[str, Any], pre: TabularPreprocessor, X: Any, indices: list[int],
) -> list[list[float]]:
    """Resolve raw named edges through the exact fitted preprocessing path.

    2026-09-05 WP5: explicit knots replace quantile fitting only when supplied.
    Require complete coverage: an accidental fallback must not change a caller's
    knot-selection budget. Reject float32/scaler collisions instead of silently
    changing the requested resolution. No targets are read here.
    """
    if not isinstance(bins, Mapping) or any(not isinstance(k, str) for k in bins):
        raise ValueError("ple_bins must be a mapping from column names to raw edges")
    names = pre.feature_names_in_
    if len(set(names)) != len(names):
        raise ValueError("ple_bins requires unique input column names")
    if any(i < 0 or i >= len(pre.numeric_idx_) for i in indices):
        raise ValueError("ple_bins can only address original numeric columns, not one-hot columns")
    expected = [names[pre.numeric_idx_[i]] for i in indices]
    if not expected or set(bins) != set(expected):
        raise ValueError(f"ple_bins must name exactly the embedded numeric columns: {expected}")
    frame = pre._as_frame(X)
    result = []
    for index, name in zip(indices, expected, strict=True):
        try:
            edges = np.asarray(bins[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ple_bins[{name!r}] must contain numeric edges") from exc
        if edges.ndim != 1:
            raise ValueError(f"ple_bins[{name!r}] must be 1-D")
        if len(edges) < 2:
            raise ValueError(f"ple_bins[{name!r}] needs at least two edges")
        if not np.isfinite(edges).all():
            raise ValueError(f"ple_bins[{name!r}] must be finite")
        if not (np.diff(edges) > 0).all():
            raise ValueError(f"ple_bins[{name!r}] must be strictly increasing")
        # Scalers are columnwise. A tiny probe reuses their exact conversion,
        # including one-hot offsets, nonlinear RSSC and float32 rounding.
        probe = frame.iloc[np.zeros(len(edges), dtype=int)].copy()
        probe[frame.columns[pre.numeric_idx_[index]]] = edges
        transformed = pre.transform(probe)[0][:, index]
        if not np.isfinite(transformed).all() or not (np.diff(transformed) > 0).all():
            raise ValueError(
                f"ple_bins[{name!r}] transformed edges must be finite and strictly increasing; "
                "the scaler or float32 precision collapsed the requested knots"
            )
        result.append(transformed.tolist())
    return result
