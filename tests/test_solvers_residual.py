"""Synthetic T4–T7/T9 and falsifiers used for the T10 mutation experiment."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from masamlp.solvers import LinearResidualKRR, NystromKRR


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def example():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(100, 3))
    eta = X[:, 0] - X[:, 1]
    y = rng.binomial(1, sigmoid(eta))
    return X, eta, y


def assert_same_bits(actual, expected):
    assert actual is not expected
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=True)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("parent", ["logit", "proba"])
def test_t5_zero_gamma_bit_for_bit(tmp_path, monkeypatch, parent):
    X, eta, y = example()
    model = LinearResidualKRR(parent=parent, r=30, dtype="float64")
    model.fit(X, y, **({"eta0": eta} if parent == "logit" else {"p0": sigmoid(eta)}))
    model.gamma_ = 0.0
    path = tmp_path / "residual.npz"
    model.save(path)
    for current in (model, LinearResidualKRR.load(path)):
        assert current.parent == parent
        for dtype in (np.float32, np.float64):
            sentinel = eta.astype(dtype)
            sentinel[:5] = [0.0, -0.0, np.inf, -np.inf, np.nan]
            # inf/NaN corrections corrupt finite and infinite parent sentinels
            # under eta + 0*f. The shortcut must never evaluate the correction.
            monkeypatch.setattr(current, "correction", lambda X: np.full(len(X), np.inf))
            assert_same_bits(current.predict_logit(X, sentinel, gamma=0), sentinel)
            assert_same_bits(current.predict_logit(X, sentinel), sentinel)
            proba = sigmoid(eta).astype(dtype)
            proba[:5] = [0.0, -0.0, 1.0, np.nan, np.inf]
            assert_same_bits(current.predict_proba(X, p0=proba, gamma=0), proba)


def test_t4_weighted_working_residual_units():
    X, eta, y = example()
    sample_weight = np.linspace(0.1, 2, len(X))
    sample_weight[::7] = 0
    params = dict(r=40, reg=0.2, bandwidth=1.5, dtype="float64", random_state=8)
    head = LinearResidualKRR(**params).fit(X, y, eta0=eta, sample_weight=sample_weight)
    p = sigmoid(eta)
    w = np.maximum(p * (1 - p), 1e-4) * sample_weight
    z = np.divide(y - p, w, out=np.zeros_like(p), where=w > 0)
    expected = NystromKRR(**params).fit(X, z, w)
    np.testing.assert_allclose(head.solver_.predict(X), expected.predict(X), rtol=0, atol=1e-10)
    keep = sample_weight > 0
    dropped = LinearResidualKRR(**params).fit(X[keep], y[keep], eta0=eta[keep],
                                             sample_weight=sample_weight[keep])
    np.testing.assert_allclose(head.correction(X), dropped.correction(X), rtol=0, atol=1e-12)


def test_t6_selection_honesty_and_nonfinite_rejection(monkeypatch):
    X, eta, y = example()
    head = LinearResidualKRR(r=30, dtype="float64").fit(X[:60], y[:60], eta0=eta[:60])
    before = head.select_gamma(X[60:], eta[60:], y[60:])
    y[:60] = y[:60][::-1]
    assert head.select_gamma(X[60:], eta[60:], y[60:]) == before
    # Actual float64 overflow; +inf + -inf produces NaN for every nonzero gamma.
    head.clip_correction = np.inf

    def overflow(X):
        with np.errstate(over="ignore", invalid="ignore"):
            huge = np.full(len(X), np.finfo(np.float64).max) * 2
            return huge - huge

    monkeypatch.setattr(head.solver_, "predict", overflow)
    assert head.select_gamma(X[60:], eta[60:], y[60:]) == 0
    assert_same_bits(head.predict_logit(X[60:], eta[60:], gamma=0), eta[60:])


def test_t6_auc_ties_choose_smallest_gamma(monkeypatch):
    X, eta, y = example()
    head = LinearResidualKRR(gamma_grid=(1, 0.5, 0, 0.125), r=20).fit(X, y, eta0=eta)
    monkeypatch.setattr(head, "correction", lambda X: np.ones(len(X)))
    assert head.select_gamma(X, eta, y) == 0


def test_t7_missing_interaction_improves_logloss():
    # Predeclared before implementation: >= 10% relative validation logloss gain.
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(2400, 2))
    y = rng.binomial(1, sigmoid(0.8 * X[:, 0] - 0.5 * X[:, 1] + 3 * X[:, 0] * X[:, 1]))
    parent = LogisticRegression(C=100, max_iter=1000).fit(X[:1200], y[:1200])
    eta = parent.decision_function(X)
    head = LinearResidualKRR(r=160, reg=0.3, bandwidth=1.5, dtype="float64",
                             random_state=12).fit(X[:1200], y[:1200], eta0=eta[:1200])
    gamma = head.select_gamma(X[1200:], eta[1200:], y[1200:])
    assert gamma > 0
    baseline = log_loss(y[1200:], sigmoid(eta[1200:]))
    improved = log_loss(y[1200:], sigmoid(head.predict_logit(X[1200:], eta[1200:])))
    assert improved <= 0.9 * baseline, (baseline, improved, gamma)


def test_t7_true_parent_keeps_zero():
    # A monotone empirical validation law on repeated covariates makes the true
    # logistic parent AUC-optimal, without relying on a lucky validation seed.
    levels = np.linspace(-2.5, 2.5, 20)
    X = np.repeat(levels, 100).reshape(-1, 1)
    eta = X[:, 0]
    y = np.concatenate([np.arange(100) < round(100 * p) for p in sigmoid(levels)]).astype(int)
    head = LinearResidualKRR(r=20, bandwidth=1, dtype="float64").fit(X, y, eta0=eta)
    assert head.select_gamma(X.copy(), eta.copy(), y.copy()) == 0


def test_t9_nonzero_head_roundtrip(tmp_path):
    X, eta, y = example()
    head = LinearResidualKRR(r=35, dtype="float64").fit(X, y, eta0=eta)
    head.gamma_ = 0.25
    path = tmp_path / "head"
    head.save(path)
    loaded = LinearResidualKRR.load(path)
    assert loaded.gamma_ == 0.25
    np.testing.assert_allclose(head.predict_logit(X, eta), loaded.predict_logit(X, eta),
                               rtol=0, atol=1e-12)


def test_residual_validation():
    X, eta, y = example()
    for kwargs in ({"parent": "raw"}, {"gamma_grid": (0.5, 1)}, {"gamma_grid": (0, np.nan)},
                   {"w_min": 0}, {"clip_correction": -1}):
        with pytest.raises(ValueError):
            LinearResidualKRR(**kwargs).fit(X, y, eta0=eta)
    with pytest.raises(ValueError):
        LinearResidualKRR().fit(X, y + 0.1, eta0=eta)
    with pytest.raises(ValueError):
        LinearResidualKRR(parent="proba").fit(X, y, p0=np.full(len(X), 1.1))
