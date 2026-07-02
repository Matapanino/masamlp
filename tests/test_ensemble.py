import numpy as np
import pytest

from masamlp.classifier import MasaClassifier
from masamlp.regressor import MasaRegressor

_KW = dict(model="lnn",
           model_params={"d_hidden": 16, "n_steps": 2, "d_backbone": 32, "dropout": 0.0},
           n_epochs=15, device="cpu", random_state=0)


def _member_predictions(est, X):
    """Predict with each member alone through the public path."""
    members = est.models_
    preds = []
    for model in members:
        est.models_ = [model]
        preds.append(est.predict(X))
    est.models_ = members
    return preds


def test_prediction_is_member_average(reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(n_ens=3, **_KW).fit(X, y)
    assert len(m.models_) == 3 and m.model_ is m.models_[0]
    member_preds = _member_predictions(m, X_test)
    np.testing.assert_allclose(m.predict(X_test), np.mean(member_preds, axis=0), atol=1e-6)


def test_members_are_diverse_and_deterministic(reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(n_ens=2, **_KW).fit(X, y)
    p0, p1 = _member_predictions(m, X_test)
    assert not np.allclose(p0, p1), "members with different seeds must differ"
    m2 = MasaRegressor(n_ens=2, **_KW).fit(X, y)
    np.testing.assert_array_equal(m.predict(X_test), m2.predict(X_test))


def test_ensemble_mse_beats_member_average_mse(reg_data):
    # Jensen: MSE(mean prediction) <= mean member MSE, always.
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(n_ens=3, **_KW).fit(X, y)
    member_mses = [np.mean((p - y_test) ** 2) for p in _member_predictions(m, X_test)]
    ensemble_mse = np.mean((m.predict(X_test) - y_test) ** 2)
    assert ensemble_mse <= np.mean(member_mses) + 1e-9


def test_classifier_probability_averaging(clf_data):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(n_ens=3, model_params={"d": 32, "n_blocks": 1},
                       n_epochs=20, device="cpu", random_state=0).fit(X, y)
    proba = m.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert float((m.predict(X_test) == y_test).mean()) > 0.8


def test_ensemble_roundtrip(tmp_path, clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(n_ens=3, model_params={"d": 32, "n_blocks": 1},
                       n_epochs=5, device="cpu", random_state=0).fit(X, y)
    m.save_model(tmp_path / "m")
    loaded = MasaClassifier.load_model(tmp_path / "m")
    assert len(loaded.models_) == 3
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_ensemble_with_early_stopping_and_retrieval(reg_data):
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(model="tabr", model_params={"d_main": 16, "context_size": 8},
                      n_ens=2, n_epochs=15, early_stopping_rounds=5,
                      device="cpu", random_state=0)
    m.fit(X, y, eval_set=[(X_val, y_val)])
    assert len(m.models_) == 2
    assert m.best_iteration_ is not None  # first member's
    assert np.isfinite(m.predict(X_val)).all()


def test_invalid_n_ens_rejected(reg_data):
    X, y, _, _ = reg_data
    with pytest.raises(ValueError, match="n_ens"):
        MasaRegressor(n_ens=0, **_KW).fit(X, y)
