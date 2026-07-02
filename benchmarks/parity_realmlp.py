"""Parity benchmark: masamlp's RealMLP-TD-S recipe vs the author's
standalone reference implementation (vendored in benchmarks/vendor).

Both implement the same recipe (one-hot + robust-scale-smooth-clip, scaling
layer with 6x lr, NTP layers with 0.1x bias lr, coslog4 schedule, Adam
betas=(0.9, 0.95), 256 epochs at batch 256, best-epoch selection on the
validation split). Expected outcome: comparable test metrics — not bitwise
equality (different batch shuffling, no drop_last in masamlp).
sklearn's HistGradientBoosting is included as an anchor.

Run:  PYTHONPATH=src python3 benchmarks/parity_realmlp.py
      (set SSL_CERT_FILE=$(python -m certifi) if OpenML downloads fail)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
from realmlp_td_s_mlp import (  # noqa: E402
    Standalone_RealMLP_TD_S_Classifier,
    Standalone_RealMLP_TD_S_Regressor,
)

from masamlp import MasaClassifier, MasaRegressor, realmlp_params  # noqa: E402

SEED = 0
N_ROWS = 12_000  # subsample to keep the CPU run to a few minutes


def _splits(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _timed(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def bench_regression():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data.iloc[:N_ROWS], data.target.iloc[:N_ROWS].to_numpy()
    X_train, X_val, X_test, y_train, y_val, y_test = _splits(X, y)
    rmse = lambda p: float(np.sqrt(np.mean((p - y_test) ** 2)))  # noqa: E731
    rows = []

    params = realmlp_params("regression")
    m = MasaRegressor(**params, early_stopping_rounds=10_000, random_state=SEED)
    _, sec = _timed(lambda: m.fit(X_train, y_train, eval_set=[(X_val, y_val)]))
    rows.append(("masamlp realmlp (TD-S recipe)", rmse(m.predict(X_test)), sec))

    ref = Standalone_RealMLP_TD_S_Regressor()
    _, sec = _timed(
        lambda: ref.fit(X_train, y_train.astype(np.float32),
                        X_val, y_val.astype(np.float32))
    )
    rows.append(("reference TD-S standalone", rmse(ref.predict(X_test)), sec))

    gbm = HistGradientBoostingRegressor(random_state=SEED)
    _, sec = _timed(lambda: gbm.fit(X_train, y_train))
    rows.append(("sklearn HistGB (anchor)", rmse(gbm.predict(X_test)), sec))
    return "california housing (rmse, lower=better)", rows


def bench_classification():
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X = data.data.iloc[:N_ROWS]
    # Integer labels for all contenders: the reference standalone expects
    # y_val to be numeric already (it only encodes the training labels).
    y = (data.target.iloc[:N_ROWS] == ">50K").to_numpy().astype(np.int64)
    X_train, X_val, X_test, y_train, y_val, y_test = _splits(X, y)
    acc = lambda p: float(np.mean(p == y_test))  # noqa: E731
    rows = []

    params = realmlp_params("classification")
    m = MasaClassifier(**params, eval_metric="accuracy",
                       early_stopping_rounds=10_000, random_state=SEED)
    _, sec = _timed(lambda: m.fit(X_train, y_train, eval_set=[(X_val, y_val)]))
    rows.append(("masamlp realmlp (TD-S recipe)", acc(m.predict(X_test)), sec))

    ref = Standalone_RealMLP_TD_S_Classifier()
    _, sec = _timed(lambda: ref.fit(X_train, y_train, X_val, y_val))
    rows.append(("reference TD-S standalone", acc(ref.predict(X_test)), sec))

    gbm = HistGradientBoostingClassifier(random_state=SEED)
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for col in X_train_enc.select_dtypes("category"):
        X_test_enc[col] = X_test_enc[col].cat.codes
        X_train_enc[col] = X_train_enc[col].cat.codes
    _, sec = _timed(lambda: gbm.fit(X_train_enc, y_train))
    rows.append(("sklearn HistGB (anchor)", acc(gbm.predict(X_test_enc)), sec))
    return "adult (accuracy, higher=better)", rows


def main():
    for bench in (bench_regression, bench_classification):
        title, rows = bench()
        print(f"\n== {title} ==", flush=True)
        for name, metric, sec in rows:
            print(f"{name:35s} {metric:8.4f}   fit {sec:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
