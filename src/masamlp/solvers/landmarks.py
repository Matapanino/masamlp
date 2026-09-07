"""Nested landmark sequences with an explicit on-device factor budget."""

import warnings

import numpy as np
import torch

from .kernels import (
    _check_kernel,
    _integer,
    _kernel,
    _matrix,
    _resolve_bandwidth,
    _torch_options,
)


def _rpcholesky_landmarks_reference(
    X, r_max, random_state=0, block=16384, *, kernel="laplace", bandwidth="median",
    bandwidth_scale=1.0, device="cpu",
):
    """Return ordered distinct row indices; every smaller rank uses a prefix.

    Pivots are sampled proportionally to the remaining kernel diagonal.
    Calculations use float64. Large problems recompute projection columns
    in row blocks instead of keeping an n-by-r Cholesky factor. This trades
    compute for memory; GPU runtime qualification is separate from this API.
    """
    X = _matrix(X)
    _integer(r_max, "r_max")
    _integer(block, "block", 5)
    _check_kernel(kernel)
    device, _ = _torch_options(device, "float64")
    rng = np.random.default_rng(random_state)
    bandwidth = _resolve_bandwidth(X, bandwidth, bandwidth_scale, rng)
    if r_max > len(X):
        warnings.warn("rank clipped to the number of rows", UserWarning, stacklevel=2)
    rank = min(r_max, len(X))
    indices = np.empty(rank, dtype=np.int64)
    diagonal = np.ones(len(X), dtype=np.float64)
    available = np.ones(len(X), dtype=bool)
    # Small-row fast path fits within the block workspace; never allocate an
    # n-by-r factor when n exceeds this fixed, user-controlled row budget.
    small = len(X) <= block // 4
    factor = torch.zeros((len(X) if small else rank, rank), dtype=torch.float64, device=device)
    centres = torch.empty((rank, X.shape[1]), dtype=torch.float64, device=device)
    rows = max(1, block // 5)
    for j in range(rank):
        total = diagonal.sum()
        if total <= np.finfo(np.float64).eps * len(X):
            # Exact/near rank exhaustion (e.g. duplicates): complete the ordered
            # sequence without replacing a pivot or dividing by a zero pivot.
            indices[j:] = rng.permutation(np.flatnonzero(available))[:rank - j]
            break
        pivot = int(rng.choice(len(X), p=diagonal / total))
        indices[j] = pivot
        available[pivot] = False
        centres[j].copy_(torch.as_tensor(X[pivot], dtype=torch.float64, device=device))
        root = float(np.sqrt(diagonal[pivot]))
        if small:
            previous = factor[pivot, :j].clone()
        elif j:
            kp = _kernel(centres[:j], centres[j:j + 1], bandwidth, kernel)
            previous = torch.linalg.solve_triangular(factor[:j, :j], kp, upper=False)[:, 0]
            factor[j, :j].copy_(previous)
            coefficients = torch.linalg.solve_triangular(
                factor[:j, :j].T, previous[:, None], upper=True
            )[:, 0]
        factor[pivot if small else j, j] = root
        for start in range(0, len(X), rows):
            stop = min(start + rows, len(X))
            xb = torch.as_tensor(np.ascontiguousarray(X[start:stop]),
                                 dtype=torch.float64, device=device)
            column = _kernel(xb, centres[j:j + 1], bandwidth, kernel)[:, 0]
            if j:
                projection = factor[start:stop, :j] if small else _kernel(
                    xb, centres[:j], bandwidth, kernel
                )
                column.sub_(projection @ (previous if small else coefficients))
            column.div_(root)
            if small:
                factor[start:stop, j].copy_(column)
            diagonal[start:stop] -= column.square().cpu().numpy()
        np.maximum(diagonal, 0, out=diagonal)
        diagonal[~available] = 0
    return indices


def _uniform_landmarks(X, r_max, random_state):
    """One O(n) seeded permutation; all rank choices are prefix slices."""
    _integer(r_max, "r_max")
    if r_max > len(X):
        warnings.warn("rank clipped to the number of rows", UserWarning, stacklevel=2)
    return np.random.default_rng(random_state).permutation(len(X))[:min(r_max, len(X))]


def rpcholesky_landmarks(
    X, r_max, random_state=0, block=16384, *, kernel="laplace", bandwidth="median",
    bandwidth_scale=1.0, device="cpu", max_factor_bytes=16 * 1024**3,
):
    """Sample distinct pivots from the residual diagonal, using prefix nesting.

    Standard single-pass Cholesky: each kernel column is evaluated once and
    projected against the stored n-by-r FP32 factor on ``device``. The residual
    diagonal stays on device in FP64, with one host copy per pivot for NumPy's
    seeded sampler. ``max_factor_bytes`` defaults to 16 GiB and bounds the
    factor alone, excluding features, diagonal and backend workspace. If it
    does not fit, use ``landmark_method="uniform"`` in the estimator/rank curve.
    """
    X = _matrix(X)
    _integer(r_max, "r_max")
    _integer(block, "block", 5)
    _integer(max_factor_bytes, "max_factor_bytes", 0)
    _check_kernel(kernel)
    device, _ = _torch_options(device, "float64")
    if r_max > len(X):
        warnings.warn("rank clipped to the number of rows", UserWarning, stacklevel=2)
    rank = min(r_max, len(X))
    required = len(X) * rank * 4
    if required > max_factor_bytes:
        raise ValueError(
            f"RPCholesky factor requires {required} bytes, exceeding "
            f"max_factor_bytes={max_factor_bytes}; use landmark_method='uniform'."
        )
    rng = np.random.default_rng(random_state)
    bandwidth = _resolve_bandwidth(X, bandwidth, bandwidth_scale, rng)
    features = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float64, device=device)
    factor = torch.empty((rank, len(X)), dtype=torch.float32, device=device).T
    diagonal = torch.ones(len(X), dtype=torch.float64, device=device)
    available = np.ones(len(X), dtype=bool)
    indices = np.empty(rank, dtype=np.int64)
    rows = max(1, block // 5)
    for j in range(rank):
        host_diagonal = diagonal.cpu().numpy()
        total = host_diagonal.sum()
        if total <= np.finfo(np.float64).eps * len(X):
            indices[j:] = rng.permutation(np.flatnonzero(available))[:rank - j]
            break
        pivot = int(rng.choice(len(X), p=host_diagonal / total))
        indices[j] = pivot
        available[pivot] = False
        root = float(np.sqrt(host_diagonal[pivot]))
        previous = factor[pivot, :j].clone()
        for start in range(0, len(X), rows):
            stop = min(start + rows, len(X))
            column = _kernel(features[start:stop], features[pivot:pivot + 1],
                             bandwidth, kernel)[:, 0]
            if j:
                column.sub_(factor[start:stop, :j] @ previous)
            column.div_(root)
            factor[start:stop, j].copy_(column)
            diagonal[start:stop].sub_(column.square())
        diagonal.clamp_(min=0)
        diagonal[pivot] = 0
    return indices
