"""Validation and radial kernels; inputs are never standardized here."""

import numpy as np
import torch


def _matrix(X, *, allow_empty=False):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0 or (not allow_empty and len(X) == 0):
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if X.dtype.kind not in "fi" or not np.isfinite(X).all():
        raise ValueError("X must contain finite real numbers")
    return X


def _vector(values, n, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return values


def _weights(w, n):
    w = np.ones(n) if w is None else _vector(w, n, "sample_weight")
    if (w < 0).any() or not (w > 0).any():
        raise ValueError("sample_weight must be nonnegative with at least one positive weight")
    return w


def _positive(value, name):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _torch_options(device, dtype):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    return device, torch.float64 if dtype == "float64" else torch.float32


def _kernel(X, Z, bandwidth, kernel):
    # Direct distances avoid cancellation near the diagonal and TF32 matmul.
    distances = torch.cdist(X, Z, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    distances.div_(bandwidth)
    if kernel == "gaussian":
        distances.square_().mul_(-0.5)
    else:
        distances.neg_()
    return distances.exp_()


def _resolve_bandwidth(X, bandwidth, bandwidth_scale, rng):
    _positive(bandwidth_scale, "bandwidth_scale")
    if isinstance(bandwidth, str):
        if bandwidth != "median":
            raise ValueError("bandwidth must be a positive float or 'median'")
        indices = rng.choice(len(X), size=min(len(X), 4096), replace=False)
        sample = torch.as_tensor(np.ascontiguousarray(X[indices]), dtype=torch.float64)
        distances = torch.pdist(sample).numpy()
        bandwidth = float(np.median(distances)) if distances.size else 0.0
        # An all-identical/single-row sample has no positive distance scale.
        if bandwidth == 0:
            bandwidth = 1.0
    _positive(bandwidth, "bandwidth")
    result = float(bandwidth) * float(bandwidth_scale)
    _positive(result, "scaled bandwidth")
    return result


def _check_kernel(kernel):
    if kernel not in ("laplace", "gaussian"):
        raise ValueError("kernel must be laplace or gaussian")
