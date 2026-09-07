"""Standalone, global kernel solvers on caller-prepared NumPy matrices."""

from .landmarks import rpcholesky_landmarks
from .linear_residual import LinearResidualKRR
from .nystrom_krr import NystromKRR, rank_curve

__all__ = ["NystromKRR", "LinearResidualKRR", "rank_curve", "rpcholesky_landmarks"]
