"""Blocked weighted Nyström KRR and its small-data dense reference."""

import json
import time
import warnings

import numpy as np
import torch

from .kernels import (
    _check_kernel,
    _integer,
    _kernel,
    _matrix,
    _positive,
    _resolve_bandwidth,
    _torch_options,
    _vector,
    _weights,
)
from .landmarks import _uniform_landmarks, rpcholesky_landmarks


class NystromKRR:
    """Global weighted kernel ridge regression, with NumPy input/output.

    Minimizes sum_i w_i (t_i - f_i)^2 + reg * alpha.T K_mm alpha.
    Scaling every weight by c therefore requires scaling reg by c.
    ``dtype`` selects kernel precision; accumulation and solve use float64.
    See docs/solvers.md for workspace accounting and all parameter defaults.
    """

    def __init__(
        self, r=1000, reg=1.0, kernel="laplace", bandwidth="median", bandwidth_scale=1.0,
        random_state=0, device="cpu", dtype="float32", block_rows=16384,
        predict_batch_rows=16384, dense=False, landmark_method="rpcholesky",
        max_factor_bytes=16 * 1024**3,
    ):
        _integer(r, "r")
        _integer(block_rows, "block_rows", 5)
        _integer(predict_batch_rows, "predict_batch_rows", 5)
        _integer(random_state, "random_state", 0)
        _positive(reg, "reg")
        _positive(bandwidth_scale, "bandwidth_scale")
        _check_kernel(kernel)
        _torch_options(device, dtype)
        if not (isinstance(bandwidth, str) and bandwidth == "median"):
            _positive(bandwidth, "bandwidth")
        if not isinstance(dense, bool):
            raise ValueError("dense must be bool")
        if landmark_method not in ("uniform", "rpcholesky"):
            raise ValueError("landmark_method must be uniform or rpcholesky")
        _integer(max_factor_bytes, "max_factor_bytes", 0)
        self.landmark_method = landmark_method
        self.max_factor_bytes = int(max_factor_bytes)
        self.r = int(r)
        self.reg = float(reg)
        self.kernel = kernel
        self.bandwidth = bandwidth if isinstance(bandwidth, str) else float(bandwidth)
        self.bandwidth_scale = float(bandwidth_scale)
        self.random_state = int(random_state)
        self.device = str(device)
        self.dtype = dtype
        self.block_rows = int(block_rows)
        self.predict_batch_rows = int(predict_batch_rows)
        self.dense = dense

    def _params(self):
        return {key: getattr(self, key) for key in (
            "r", "reg", "kernel", "bandwidth", "bandwidth_scale", "random_state", "device",
            "dtype", "block_rows", "predict_batch_rows", "dense",
            "landmark_method", "max_factor_bytes",
        )}

    def fit(self, X, t, sample_weight=None):
        """Fit on these rows only; zero-weight rows are removed before landmarks."""
        X = _matrix(X)
        if self.dense and len(X) > 5000:
            raise ValueError("dense=True is restricted to at most 5,000 rows")
        t = _vector(t, len(X), "t")
        w = _weights(sample_weight, len(X))
        active = np.flatnonzero(w > 0)
        X, t, w = X[active], t[active], w[active]
        self.__dict__.pop("alpha_", None)
        rng = np.random.default_rng(self.random_state)
        start = time.perf_counter()
        self._start_memory()
        bandwidth = _resolve_bandwidth(X, self.bandwidth, self.bandwidth_scale, rng)
        if self.dense:
            indices = np.arange(len(X))
        elif self.landmark_method == "uniform":
            indices = _uniform_landmarks(X, self.r, self.random_state)
        else:
            indices = rpcholesky_landmarks(
                X, self.r, rng, block=self.block_rows, kernel=self.kernel,
                bandwidth=bandwidth, device=self.device, max_factor_bytes=self.max_factor_bytes,
            )
        self._fit_centres(X, t, w, indices, bandwidth)
        self.landmark_indices_ = active[indices]
        self._finish_diagnostics(X, t, w, start)
        return self

    def _start_memory(self):
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def _fit_centres(self, X, t, w, indices, bandwidth):
        self.bandwidth_ = float(bandwidth)
        self.n_features_in_ = X.shape[1]
        self.landmark_indices_ = np.asarray(indices, dtype=np.int64).copy()
        self.landmarks_ = np.array(X[indices], copy=True)
        self.rank_ = len(indices)
        device, dtype = _torch_options(self.device, "float64" if self.dense else self.dtype)
        Z = torch.as_tensor(self.landmarks_, dtype=dtype, device=device)
        # Five microblocks reserve room for an fp32 kernel plus two fp64
        # operands. Thus 20*floor(block_rows/5)*r <= 4*block_rows*r bytes.
        rows = max(1, self.block_rows // 5)
        accumulation_dtype = torch.float64
        A = torch.empty((self.rank_, self.rank_), dtype=accumulation_dtype, device=device).T
        rhs = torch.zeros(self.rank_, dtype=accumulation_dtype, device=device)

        def rebuild():
            A.zero_()
            rhs.zero_()
            if self.dense:
                root_w = torch.as_tensor(np.sqrt(w), dtype=torch.float64, device=device)
                for start in range(0, len(X), rows):
                    stop = min(start + rows, len(X))
                    K = _kernel(Z[start:stop], Z, self.bandwidth_, self.kernel)
                    K.mul_(root_w[start:stop, None]).mul_(root_w[None, :])
                    A[start:stop].copy_(K)
                A.diagonal().add_(self.reg)
                rhs.copy_(root_w * torch.as_tensor(t, dtype=torch.float64, device=device))
                return
            for start in range(0, len(X), rows):
                stop = min(start + rows, len(X))
                xb = torch.as_tensor(np.ascontiguousarray(X[start:stop]),
                                     dtype=dtype, device=device)
                K = _kernel(xb, Z, self.bandwidth_, self.kernel)
                K64 = K.to(accumulation_dtype)
                wb = torch.as_tensor(w[start:stop], dtype=accumulation_dtype, device=device)
                tb = torch.as_tensor(t[start:stop], dtype=accumulation_dtype, device=device)
                weighted = K64 * wb[:, None]
                A.addmm_(K64.T, weighted)
                rhs.addmv_(K64.T, wb * tb)
                del K, K64, weighted
            for start in range(0, self.rank_, rows):
                stop = min(start + rows, self.rank_)
                K = _kernel(Z[start:stop], Z, self.bandwidth_, self.kernel)
                A[start:stop].add_(K.to(accumulation_dtype), alpha=self.reg)

        rebuild()
        self.alpha_, self.solver_path_ = self._solve(A, rhs, rebuild)
        if self.dense:
            self.alpha_ *= np.sqrt(w)

    @staticmethod
    def _solve(A, rhs, rebuild):
        # out= aliases the column-major system buffer. Failed factorizations
        # destroy it, so rebuild rather than retain a second r-by-r matrix.
        scale = max(float(A.diagonal().mean().item()), np.finfo(np.float64).tiny)
        jitter = 1e-10 * scale
        info = torch.empty((), dtype=torch.int32, device=A.device)
        for attempt in range(4):
            if attempt:
                rebuild()
                A.diagonal().add_(attempt * jitter)
            factor, info = torch.linalg.cholesky_ex(A, out=(A, info))
            if info.item() == 0:
                alpha = torch.cholesky_solve(rhs[:, None], factor)[:, 0]
                path = "cholesky" if attempt == 0 else f"cholesky+jitter{attempt}"
                break
        else:
            rebuild()
            eigenvalues = torch.empty(len(rhs), dtype=A.dtype, device=A.device)
            eigenvalues, eigenvectors = torch.linalg.eigh(A, out=(eigenvalues, A))
            eigenvalues.clamp_(min=jitter)
            alpha = eigenvectors @ ((eigenvectors.T @ rhs) / eigenvalues)
            path = "eigh"
        if not torch.isfinite(alpha).all():
            raise ValueError("nonfinite kernel solution")
        return alpha.cpu().numpy().astype(np.float64, copy=True), path

    def _finish_diagnostics(self, X, t, w, start):
        predictions = self.predict(X)
        self.train_residual_norm_ = float(np.sqrt(np.dot(w, (predictions - t) ** 2)))
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))
            self.peak_device_memory_ = torch.cuda.max_memory_allocated(torch.device(self.device))
        else:
            self.peak_device_memory_ = None
        self.wall_seconds_ = time.perf_counter() - start

    def predict(self, X):
        """Stream f(X) = K(X, landmarks) alpha; returns a float64 vector."""
        if not hasattr(self, "alpha_"):
            raise RuntimeError("fit must be called before predict")
        X = _matrix(X, allow_empty=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from fitted data")
        device, dtype = _torch_options(self.device, "float64" if self.dense else self.dtype)
        Z = torch.as_tensor(self.landmarks_, dtype=dtype, device=device)
        alpha = torch.as_tensor(self.alpha_, dtype=torch.float64, device=device)
        result = np.empty(len(X), dtype=np.float64)
        rows = max(1, self.predict_batch_rows // 3)
        for start in range(0, len(X), rows):
            stop = min(start + rows, len(X))
            xb = torch.as_tensor(np.ascontiguousarray(X[start:stop]), dtype=dtype, device=device)
            K = _kernel(xb, Z, self.bandwidth_, self.kernel)
            result[start:stop] = (K.to(torch.float64) @ alpha).cpu().numpy()
        return result

    def _state(self):
        if not hasattr(self, "alpha_"):
            raise RuntimeError("fit must be called before save")
        metadata = {"format": 1, "params": self._params(), "bandwidth": self.bandwidth_,
                    "solver_path": self.solver_path_, "wall_seconds": self.wall_seconds_,
                    "train_residual_norm": self.train_residual_norm_,
                    "peak_device_memory": self.peak_device_memory_}
        return {"metadata": np.array(json.dumps(metadata)), "landmarks": self.landmarks_,
                "alpha": self.alpha_, "indices": self.landmark_indices_}

    @classmethod
    def _from_state(cls, state, device=None):
        metadata = json.loads(str(state["metadata"]))
        if metadata["format"] != 1:
            raise ValueError("unsupported solver archive format")
        params = metadata["params"]
        if device is not None:
            params["device"] = device
        model = cls(**params)
        model.landmarks_ = _matrix(state["landmarks"]).copy()
        model.rank_, model.n_features_in_ = model.landmarks_.shape
        model.alpha_ = _vector(state["alpha"], model.rank_, "alpha").copy()
        model.landmark_indices_ = np.asarray(state["indices"], dtype=np.int64).copy()
        model.bandwidth_ = metadata["bandwidth"]
        _positive(model.bandwidth_, "stored bandwidth")
        model.solver_path_ = metadata["solver_path"]
        model.wall_seconds_ = metadata["wall_seconds"]
        model.train_residual_norm_ = metadata["train_residual_norm"]
        model.peak_device_memory_ = metadata["peak_device_memory"]
        return model

    def save(self, path):
        """Write a pickle-free np.savez archive at the exact supplied path."""
        with open(path, "wb") as file:
            np.savez(file, **self._state())

    @classmethod
    def load(cls, path, *, device=None):
        """Load an archive, optionally choosing a different cpu/cuda device."""
        with np.load(path, allow_pickle=False) as state:
            return cls._from_state(state, device=device)


def rank_curve(X, t, w=None, ranks=(1000, 2000, 4000, 8000), *, X_val, random_state=0,
               landmark_method="rpcholesky", max_factor_bytes=16 * 1024**3, **params):
    """One landmark run, then fresh prefix solves and caller-validation predictions.

    Returns one dict per requested rank with the model, predictions, rank,
    train_residual_norm, solver_path, wall_seconds and peak_device_memory.
    Neither validation labels nor any outer/held-out rows are accessed.
    """
    X = _matrix(X)
    X_val = _matrix(X_val, allow_empty=True)
    t, w = _vector(t, len(X), "t"), _weights(w, len(X))
    active = np.flatnonzero(w > 0)
    X, t, w = X[active], t[active], w[active]
    ranks = tuple(ranks)
    if not ranks:
        raise ValueError("ranks must not be empty")
    for rank in ranks:
        _integer(rank, "rank")
    if "r" in params or params.get("dense", False):
        raise ValueError("rank_curve requires prefix ranks and dense=False")
    params = dict(params, landmark_method=landmark_method, max_factor_bytes=max_factor_bytes)
    prototype = NystromKRR(r=max(ranks), random_state=random_state, **params)
    rng = np.random.default_rng(random_state)
    start = time.perf_counter()
    prototype._start_memory()
    bandwidth = _resolve_bandwidth(X, prototype.bandwidth, prototype.bandwidth_scale, rng)
    if landmark_method == "uniform":
        indices = _uniform_landmarks(X, max(ranks), random_state)
    else:
        indices = rpcholesky_landmarks(X, max(ranks), rng, block=prototype.block_rows,
                                      kernel=prototype.kernel, bandwidth=bandwidth,
                                      device=prototype.device, max_factor_bytes=max_factor_bytes)
    landmark_seconds = time.perf_counter() - start
    landmark_peak = None
    if torch.device(prototype.device).type == "cuda":
        torch.cuda.synchronize(torch.device(prototype.device))
        landmark_seconds = time.perf_counter() - start
        landmark_peak = torch.cuda.max_memory_allocated(torch.device(prototype.device))
    results = []
    for requested in ranks:
        rank = min(requested, len(X))
        if requested > len(X):
            warnings.warn("rank clipped to the number of positive-weight rows",
                          UserWarning, stacklevel=2)
        model = NystromKRR(r=rank, random_state=random_state, **params)
        start = time.perf_counter()
        model._start_memory()
        model._fit_centres(X, t, w, indices[:rank], bandwidth)
        model.landmark_indices_ = active[indices[:rank]]
        prediction = model.predict(X_val)
        model._finish_diagnostics(X, t, w, start)
        if landmark_peak is not None:
            model.peak_device_memory_ = max(landmark_peak, model.peak_device_memory_)
        results.append({"rank": rank, "requested_rank": requested, "model": model,
                        "predictions": prediction,
                        "train_residual_norm": model.train_residual_norm_,
                        "solver_path": model.solver_path_, "wall_seconds": model.wall_seconds_,
                        "landmark_seconds": landmark_seconds,
                        "peak_device_memory": model.peak_device_memory_})
    return results
