"""GPU speed / correctness check, meant for a CUDA machine (e.g. Colab).

Runs a synthetic regression workload per model on CPU and CUDA (AMP on and
off), plus loop-vs-vectorized ensembles, and reports fit / predict seconds
and test RMSE so device parity is eyeballable. Numbers for the original
configs are comparable with docs/verdicts/2026-07-02-gpu-report.md (same
data, same hyperparameters).

Run:  PYTHONPATH=src python3 benchmarks/gpu_speed.py [--rows 50000]
      PYTHONPATH=src python3 benchmarks/gpu_speed.py --retrieval-scale
      PYTHONPATH=src python3 benchmarks/gpu_speed.py --multi-gpu   # >=2 GPUs
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from masamlp import MasaRegressor, realmlp_params

# (display name, estimator kwargs, needs categorical columns)
CONFIGS: list[tuple[str, dict, bool]] = [
    ("resnet", {"model": "resnet", "n_epochs": 40}, False),
    ("realmlp (TD-S recipe)", {**realmlp_params("regression"), "n_epochs": 40}, False),
    ("ft_transformer", {"model": "ft_transformer", "n_epochs": 20,
                        "model_params": {"n_blocks": 2, "d_block": 128}}, False),
    ("tabr", {"model": "tabr", "n_epochs": 20,
              "model_params": {"d_main": 96, "context_size": 64}}, False),
    ("tab_transformer", {"model": "tab_transformer", "n_epochs": 20,
                         "model_params": {"n_layers": 4, "d_token": 32}}, True),
    ("modernnca", {"model": "modernnca", "n_epochs": 20,
                   "model_params": {"dim": 128, "d_block": 256}}, False),
]

MULTI_GPU_MODELS = ("tabr", "modernnca", "ft_transformer", "tab_transformer")


def make_data(n_rows: int, n_features: int = 32, n_cats: int = 0,
              pred_rows: int = 0, seed: int = 0):
    """Synthetic regression split. With ``n_cats``, that many integer columns
    (cardinality 10) are appended and their indices returned as the
    categorical feature spec; ``pred_rows`` adds a large inference-only split."""
    rng = np.random.default_rng(seed)
    n_total = n_rows + pred_rows
    X = rng.normal(size=(n_total, n_features)).astype(np.float32)
    w = rng.normal(size=n_features)
    y = X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(0, 0.1, n_total)
    cat_idx: list[int] = []
    if n_cats:
        C = rng.integers(0, 10, size=(n_total, n_cats))
        y += (C[:, 0] == 3) * 1.5 - (C[:, 1] >= 7) * 0.5
        X = np.column_stack([X, C.astype(np.float32)])
        cat_idx = list(range(n_features, n_features + n_cats))
    n_train = int(0.8 * n_rows)
    train = slice(0, n_train)
    test = slice(n_train, n_rows)
    pred = slice(n_rows, n_total)
    return X[train], y[train], X[test], y[test], X[pred], cat_idx


def run_one(name: str, kwargs: dict, device: str, amp, data,
            label: str | None = None) -> dict:
    X, y, X_test, y_test, X_pred, cat_idx = data
    if cat_idx:
        kwargs = dict(kwargs, categorical_features=cat_idx)
    est = MasaRegressor(random_state=0, verbose=0, device=device, amp=amp, **kwargs)
    start = time.perf_counter()
    est.fit(X, y)
    fit_sec = time.perf_counter() - start
    X_eval = X_pred if len(X_pred) else X_test
    start = time.perf_counter()
    pred = est.predict(X_eval)
    pred_sec = time.perf_counter() - start
    rmse = float(np.sqrt(np.mean((est.predict(X_test) - y_test) ** 2)))
    print(f"{label or name:28s} device={device:7s} amp={str(amp):5s} "
          f"fit {fit_sec:7.1f}s  predict({len(X_eval)}) {pred_sec:6.2f}s  "
          f"rmse {rmse:.4f}", flush=True)
    return {"fit_sec": fit_sec, "pred_sec": pred_sec, "pred": pred}


def retrieval_scale(args) -> None:
    """Large-N stress for the retrieval models: the ModernNCA eval OOM and
    TabR's B x N eval matrix both reproduce here without the chunked paths."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for rows in args.scale_rows:
        data = make_data(rows, pred_rows=50_000)
        for model in ("tabr", "modernnca"):
            for budget in (None, 131_072):
                if budget is not None and budget >= rows:
                    continue
                label = f"{model} rows={rows} budget={budget}"
                try:
                    run_one(model, {"model": model, "n_epochs": 3,
                                    "candidate_budget": budget},
                            device, "auto", data, label=label)
                except RuntimeError as exc:
                    msg = str(exc).splitlines()[0][:80]
                    print(f"{label:28s} FAILED: {msg}", flush=True)


def multi_gpu(args) -> None:
    """Sharded (device='auto') vs single-GPU (device='cuda:0') ensembles.
    Meant for Kaggle 2xT4; no-op elsewhere."""
    if torch.cuda.device_count() < 2:
        print("multi-gpu section skipped: fewer than 2 CUDA devices", flush=True)
        return
    data = make_data(args.rows, n_cats=8, pred_rows=0)
    for model in MULTI_GPU_MODELS:
        kwargs = next(kw for name, kw, _ in CONFIGS if kw.get("model") == model)
        kwargs = dict(kwargs, n_ens=4)
        single = run_one(model, dict(kwargs), "cuda:0", "auto", data,
                         label=f"{model} n_ens=4 [cuda:0]")
        sharded = run_one(model, dict(kwargs), "auto", "auto", data,
                          label=f"{model} n_ens=4 [sharded]")
        diff = float(np.abs(single["pred"] - sharded["pred"]).max())
        speedup = single["fit_sec"] / sharded["fit_sec"]
        print(f"{model:28s} sharded speedup x{speedup:.2f}  "
              f"max|pred diff| {diff:.2e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--pred-rows", type=int, default=200_000,
                        help="size of the inference-only split used for predict timing")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="skip CPU baselines (useful on weak-CPU GPU VMs)")
    parser.add_argument("--retrieval-scale", action="store_true",
                        help="large-N tabr/modernnca stress (fit + predict, w/wo budget)")
    parser.add_argument("--scale-rows", type=int, nargs="+",
                        default=[200_000, 345_000])
    parser.add_argument("--multi-gpu", action="store_true",
                        help="sharded vs single-GPU n_ens=4 comparison (needs >=2 GPUs)")
    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda={has_cuda}"
          + (f" ({torch.cuda.get_device_name(0)} x{torch.cuda.device_count()})"
             if has_cuda else ""))

    if args.retrieval_scale:
        retrieval_scale(args)
        return
    if args.multi_gpu:
        multi_gpu(args)
        return

    numeric = make_data(args.rows, pred_rows=args.pred_rows)
    with_cats = make_data(args.rows, n_cats=8, pred_rows=args.pred_rows)
    for name, kwargs, needs_cats in CONFIGS:
        data = with_cats if needs_cats else numeric
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
    # "cuda:0", not "cuda": on multi-GPU hosts the index-less device would
    # shard the loop-mode ensemble while vectorized stays single-device,
    # invalidating the comparison (use --multi-gpu for the sharded numbers).
    devices = ([] if args.skip_cpu else ["cpu"]) + (["cuda:0"] if has_cuda else [])
    for device in devices:
        for mode in ("loop", "vectorized"):
            run_one(f"lnn n_ens=8 [{mode}]", dict(ens_kwargs, ens_mode=mode),
                    device, False, numeric)


if __name__ == "__main__":
    main()
