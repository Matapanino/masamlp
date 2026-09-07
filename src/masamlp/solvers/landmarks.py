"""Randomly pivoted Cholesky, with bounded workspace for large row counts."""

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


def rpcholesky_landmarks(
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
