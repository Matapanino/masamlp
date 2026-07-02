"""The flagship differentiator tests: sample_weight semantics.

The exact-equality tests use the LNN model (LayerNorm only — no BatchNorm, so
rows cannot influence each other's forward pass), dropout 0, full-batch
training, no input scaling, and no target standardization, so the only
difference between the runs is the weighted loss reduction itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from masamlp.regressor import MasaRegressor

_ISOLATED = dict(
    model="lnn",
    model_params={"d_hidden": 16, "n_steps": 2, "d_backbone": 32, "dropout": 0.0},
    numeric_scaler="none",
    target_standardize=False,
    n_epochs=20,
    batch_size=None,  # full batch
    device="cpu",
    random_state=7,
)


@pytest.fixture
def data(rng):
    X = rng.normal(size=(120, 4))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.05, 120)
    X_extra = rng.normal(size=(40, 4)) + 3.0  # off-distribution rows to ignore
    y_extra = np.full(40, 25.0)
    return X, y, X_extra, y_extra


def test_zero_weight_rows_do_not_affect_fit(data, rng):
    X, y, X_extra, y_extra = data
    X_all = np.vstack([X, X_extra])
    y_all = np.concatenate([y, y_extra])
    w = np.concatenate([np.ones(len(y)), np.zeros(len(y_extra))])

    m_weighted = MasaRegressor(**_ISOLATED).fit(X_all, y_all, sample_weight=w)
    m_subset = MasaRegressor(**_ISOLATED).fit(X, y)

    X_test = rng.normal(size=(50, 4))
    np.testing.assert_allclose(
        m_weighted.predict(X_test), m_subset.predict(X_test), atol=1e-4
    )


def test_integer_weights_match_row_duplication(data, rng):
    X, y, _, _ = data
    w = np.ones(len(y))
    w[:20] = 3.0
    X_dup = np.vstack([X, np.repeat(X[:20], 2, axis=0)])
    y_dup = np.concatenate([y, np.repeat(y[:20], 2)])

    m_weighted = MasaRegressor(**_ISOLATED).fit(X, y, sample_weight=w)
    m_dup = MasaRegressor(**_ISOLATED).fit(X_dup, y_dup)

    X_test = rng.normal(size=(50, 4))
    np.testing.assert_allclose(m_weighted.predict(X_test), m_dup.predict(X_test), atol=1e-4)


def test_upweighting_shifts_predictions(rng):
    # Behavioral check for a BatchNorm model: two clusters with conflicting
    # targets; upweighting one pulls predictions toward its target.
    X = np.vstack([rng.normal(size=(100, 3)), rng.normal(size=(100, 3))])
    y = np.concatenate([np.full(100, 1.0), np.full(100, -1.0)])
    w_up_pos = np.concatenate([np.full(100, 10.0), np.full(100, 1.0)])
    kwargs = dict(model="resnet", model_params={"d": 32, "n_blocks": 1},
                  n_epochs=15, device="cpu", random_state=0)
    m_up = MasaRegressor(**kwargs).fit(X, y, sample_weight=w_up_pos)
    m_plain = MasaRegressor(**kwargs).fit(X, y)
    assert m_up.predict(X).mean() > m_plain.predict(X).mean()


def test_invalid_weights_raise(data):
    X, y, _, _ = data
    m = MasaRegressor(**_ISOLATED)
    with pytest.raises(ValueError, match="non-negative"):
        m.fit(X, y, sample_weight=-np.ones(len(y)))
    with pytest.raises(ValueError, match="length"):
        m.fit(X, y, sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        m.fit(X, y, sample_weight=np.full(len(y), np.nan))
