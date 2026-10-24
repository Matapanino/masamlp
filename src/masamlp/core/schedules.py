"""Scalar training schedules with a separate execution limit and horizon.

The opt-in ``flat_anneal``, ``cos_log_15``, ``invsqrtp1e-3`` and
``sqrt_cos`` formulas follow pytabkit 1.7.3's scheduling.py. Existing
``coslog4`` and ``flat_cos`` retain their original arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral

import numpy as np


def coslog4(t: float) -> float:
    """Four cosine cycles in log time, starting at zero."""
    return float(0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t)))


def flat_cos(t: float) -> float:
    """Constant one for half the horizon, then cosine decay to zero."""
    if t < 0.5:
        return 1.0
    return float(0.5 * (1.0 + np.cos(np.pi * (t - 0.5) / 0.5)))


def flat_anneal(t: float) -> float:
    """pytabkit's 60% plateau, then cosine decay to a factor of 1e-5."""
    if t < 0.6:
        return 1.0
    progress = (t - 0.6) / 0.4
    # Keep ScaledSchedule(get_cos_sched(), 1.0, 1e-5)'s arithmetic.
    return 1.0 + (1e-5 - 1.0) * (0.5 * (1.0 - math.cos(math.pi * progress)))


def invsqrtp1e3(t: float) -> float:
    """pytabkit's ``sqrt(1e-3) / sqrt(t + 1e-3)`` factor."""
    return float(np.sqrt(1e-3) / np.sqrt(t + 1e-3))


def sqrt_cos(t: float) -> float:
    """pytabkit's decaying envelope with five cosine cycles."""
    return float(0.05 / (np.sqrt(t) + 0.05) *
                 (0.5 - 0.5 * np.cos(5 * 2 * np.pi * t)))


_SCHEDULES: dict[str, Callable[[float], float]] = {
    "coslog4": coslog4,
    "cos_log_15": coslog4,
    "flat_cos": flat_cos,
    "flat_anneal": flat_anneal,
    "invsqrtp1e-3": invsqrtp1e3,
    "sqrt_cos": sqrt_cos,
}
SCHEDULE_NAMES = tuple(_SCHEDULES)


def resolve_schedule(
    name: str, *, parameter: str = "schedule"
) -> Callable[[float], float] | None:
    """Resolve a scalar per-step schedule; ``none`` leaves a value constant."""
    if isinstance(name, str) and name == "none":
        return None
    if not isinstance(name, str) or name not in _SCHEDULES:
        raise ValueError(
            f"Unknown {parameter} {name!r}. Expected 'none' or one of {list(SCHEDULE_NAMES)}"
        )
    return _SCHEDULES[name]


def resolve_epoch_count(n_epochs: int, stop_epoch: int | None) -> int:
    """Validate an optional executed-epoch count without changing the horizon."""
    if isinstance(n_epochs, bool) or not isinstance(n_epochs, Integral) or n_epochs < 0:
        raise ValueError(f"n_epochs must be a nonnegative integer, got {n_epochs!r}")
    if stop_epoch is None:
        return int(n_epochs)
    if (isinstance(stop_epoch, bool) or not isinstance(stop_epoch, Integral)
            or not 1 <= stop_epoch <= n_epochs):
        raise ValueError(
            f"stop_epoch must be None or an integer in [1, n_epochs={n_epochs}], "
            f"got {stop_epoch!r}"
        )
    return int(stop_epoch)
