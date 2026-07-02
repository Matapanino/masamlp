"""End-to-end quickstart: the differentiators in one script.

Covers: eval_set + custom metric + early stopping + sample_weight on a
classifier, save/load parity, a custom objective, and one regression fit per
model (resnet / danet / lnn).
"""

from __future__ import annotations

import tempfile

import numpy as np

from masamlp import MasaClassifier, MasaRegressor, make_metric

rng = np.random.default_rng(0)


# --------------------------------------------------------------------- #
# Classification: custom metric + sample_weight + early stopping
# --------------------------------------------------------------------- #
def f1_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    pred = y_proba >= 0.5
    tp = float(np.sum(pred & (y_true == 1)))
    if tp == 0:
        return 0.0
    precision = tp / max(pred.sum(), 1)
    recall = tp / max((y_true == 1).sum(), 1)
    return 2 * precision * recall / (precision + recall)


X = rng.normal(size=(2000, 8))
y = ((X[:, 0] + 0.5 * X[:, 1] + 0.2 * rng.normal(size=2000)) > 0.8).astype(int)
weight = np.where(y == 1, 3.0, 1.0)  # emphasize the rarer positive class
X_train, X_val, X_test = X[:1200], X[1200:1600], X[1600:]
y_train, y_val, y_test = y[:1200], y[1200:1600], y[1600:]

clf = MasaClassifier(
    model="resnet",
    eval_metric=[make_metric(f1_score, name="f1", minimize=False), "logloss"],
    early_stopping_rounds=15,
    n_epochs=200,
    random_state=0,
    verbose=0,
)
clf.fit(X_train, y_train, sample_weight=weight[:1200], eval_set=[(X_val, y_val)])
f1_test = f1_score(y_test, clf.predict_proba(X_test)[:, 1])
print(f"classifier: best_iteration={clf.best_iteration_} "
      f"best_f1={clf.best_score_:.3f} test_f1={f1_test:.3f}")
assert f1_test > 0.7

with tempfile.TemporaryDirectory() as tmp:
    clf.save_model(tmp)
    loaded = MasaClassifier.load_model(tmp)
    np.testing.assert_array_equal(clf.predict_proba(X_test), loaded.predict_proba(X_test))
print("save/load: predictions identical")


# --------------------------------------------------------------------- #
# Regression with a custom objective, one fit per model
# --------------------------------------------------------------------- #
import torch  # noqa: E402


def asymmetric_mse(y_true: torch.Tensor, raw_pred: torch.Tensor) -> torch.Tensor:
    err = raw_pred - y_true
    return torch.where(err < 0, 4.0 * err**2, err**2).mean(dim=1)  # under-prediction hurts 4x


Xr = rng.normal(size=(1500, 6))
yr = 2 * Xr[:, 0] - Xr[:, 1] + 0.5 * Xr[:, 2] * Xr[:, 3] + rng.normal(0, 0.2, 1500)
baseline = float(np.sqrt(np.mean((yr[1000:] - yr[:1000].mean()) ** 2)))

for name in ("resnet", "danet", "lnn"):
    reg = MasaRegressor(model=name, objective=asymmetric_mse, n_epochs=60, random_state=0)
    reg.fit(Xr[:1000], yr[:1000])
    rmse = float(np.sqrt(np.mean((reg.predict(Xr[1000:]) - yr[1000:]) ** 2)))
    print(f"regressor[{name}]: rmse={rmse:.3f} (mean-baseline {baseline:.3f})")
    assert rmse < baseline

print("quickstart: all green")
