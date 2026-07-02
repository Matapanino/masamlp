"""Quick leaderboard: every masamlp model on two small real datasets.

This is a smoke-level comparison (single seed, capped epochs, subsampled
rows, no HPO) — it shows the zoo works end-to-end and gives rough relative
numbers, not paper-grade rankings.

Run:  PYTHONPATH=src python3 benchmarks/model_zoo.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split

from masamlp import MasaClassifier, MasaRegressor, realmlp_params

SEED = 0
N_ROWS = 8_000
N_EPOCHS = 60
PATIENCE = 10

# Per-model estimator settings: each model's recommended/known-good knobs.
MODEL_KWARGS: dict[str, dict] = {
    "resnet": {},
    "realmlp": {
        k: v for k, v in realmlp_params("regression").items() if k != "model"
    },
    "ft_transformer": {},
    "tab_transformer": {"num_embedding": "plr", "learning_rate": 3e-3},
    "danet": {},
    "tabr": {},
    "modernnca": {"num_embedding": "plr-lite", "learning_rate": 0.01},
    "gandalf": {"num_embedding": "plr"},
    "grn": {},
    "lnn": {},
}


def run(task: str):
    if task == "regression":
        data = fetch_california_housing(as_frame=True)
        X, y = data.data.iloc[:N_ROWS], data.target.iloc[:N_ROWS].to_numpy()
        estimator_cls, metric_name = MasaRegressor, "rmse"
    else:
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        X, y = data.data.iloc[:N_ROWS], data.target.iloc[:N_ROWS].to_numpy()
        estimator_cls, metric_name = MasaClassifier, "acc"

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=SEED
    )

    print(f"\n== {task} ({metric_name}) ==")
    for name, extra in MODEL_KWARGS.items():
        kwargs = dict(extra)
        if task == "classification" and name == "realmlp":
            kwargs = {k: v for k, v in realmlp_params("classification").items()
                      if k != "model"}
        kwargs.setdefault("n_epochs", N_EPOCHS)
        est = estimator_cls(
            model=name, early_stopping_rounds=PATIENCE, random_state=SEED,
            verbose=0, **kwargs,
        )
        start = time.perf_counter()
        try:
            est.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            pred = est.predict(X_test)
            if task == "regression":
                metric = float(np.sqrt(np.mean((pred - y_test) ** 2)))
            else:
                metric = float(np.mean(pred == y_test))
            sec = time.perf_counter() - start
            print(f"{name:16s} {metric_name}={metric:7.4f}  best_iter={est.best_iteration_}"
                  f"  fit {sec:6.1f}s")
        except Exception as exc:  # keep the leaderboard running
            print(f"{name:16s} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run("regression")
    run("classification")
