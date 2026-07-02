"""GPU speed / correctness check, meant for a CUDA machine (e.g. Colab).

Runs a synthetic regression workload per model on CPU and CUDA (AMP on and
off), plus loop-vs-vectorized ensembles, and reports fit seconds and test
RMSE so device parity is eyeballable.

Run:  PYTHONPATH=src python3 benchmarks/gpu_speed.py [--rows 50000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from masamlp import MasaRegressor, realmlp_params

CONFIGS: list[tuple[str, dict]] = [
    ("resnet", {"model": "resnet", "n_epochs": 40}),
    ("realmlp (TD-S recipe)", {**realmlp_params("regression"), "n_epochs": 40}),
    ("ft_transformer", {"model": "ft_transformer", "n_epochs": 20,
                        "model_params": {"n_blocks": 2, "d_block": 128}}),
    ("tabr", {"model": "tabr", "n_epochs": 20,
              "model_params": {"d_main": 96, "context_size": 64}}),
]


def make_data(n_rows: int, n_features: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    w = rng.normal(size=n_features)
    y = X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(0, 0.1, n_rows)
    n_train = int(0.8 * n_rows)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def run_one(name: str, kwargs: dict, device: str, amp, data) -> None:
    X, y, X_test, y_test = data
    est = MasaRegressor(random_state=0, verbose=0, device=device, amp=amp, **kwargs)
    start = time.perf_counter()
    est.fit(X, y)
    sec = time.perf_counter() - start
    rmse = float(np.sqrt(np.mean((est.predict(X_test) - y_test) ** 2)))
    print(f"{name:28s} device={device:5s} amp={str(amp):5s} "
          f"fit {sec:7.1f}s  rmse {rmse:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--skip-cpu", action="store_true",
                        help="skip CPU baselines (useful on weak-CPU GPU VMs)")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda={has_cuda}"
          + (f" ({torch.cuda.get_device_name(0)})" if has_cuda else ""))
    data = make_data(args.rows)

    for name, kwargs in CONFIGS:
        if not args.skip_cpu:
            run_one(name, dict(kwargs), "cpu", False, data)
        if has_cuda:
            run_one(name, dict(kwargs), "cuda", False, data)
            run_one(name, dict(kwargs), "cuda", "auto", data)

    print("\n-- n_ens=8 (lnn): loop vs vectorized --", flush=True)
    ens_kwargs = dict(
        model="lnn", model_params={"d_hidden": 64, "n_steps": 3, "d_backbone": 128},
        n_ens=8, n_epochs=25,
    )
    devices = ([] if args.skip_cpu else ["cpu"]) + (["cuda"] if has_cuda else [])
    for device in devices:
        for mode in ("loop", "vectorized"):
            run_one(f"lnn n_ens=8 [{mode}]", dict(ens_kwargs, ens_mode=mode),
                    device, False, data)


if __name__ == "__main__":
    main()
