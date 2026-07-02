import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.models import build_model
from masamlp.regressor import MasaRegressor

_PARAMS = {"d_main": 16, "context_size": 8}
_KW = dict(model="tabr", model_params=_PARAMS, device="cpu", random_state=0)


def test_tabr_classification_learns(clf_data):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(n_epochs=30, **_KW).fit(X, y)
    assert float((m.predict(X_test) == y_test).mean()) > 0.75
    assert m.model_.current_batch_indices is None  # cleared after training


def test_tabr_regression_beats_baseline(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(n_epochs=40, **_KW).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    assert rmse < float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))


def test_tabr_roundtrip_keeps_candidates(tmp_path, clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(n_epochs=5, **_KW).fit(X, y)
    m.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    assert loaded.model_.has_candidates
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_tabr_multioutput_regression_rejected():
    with pytest.raises(ValueError, match="multi-output"):
        build_model("tabr", dict(_PARAMS), 4, [], 2, None)


def test_tabr_forward_without_candidates_raises():
    model = build_model("tabr", dict(_PARAMS), 4, [], 1, None)
    with pytest.raises(RuntimeError, match="candidates"):
        model(torch.randn(4, 4), torch.zeros(4, 0, dtype=torch.int64))


def test_tabr_candidates_in_state_dict(clf_data):
    X, y, _, _ = clf_data
    m = MasaClassifier(n_epochs=2, **_KW).fit(X, y)
    state = m.model_.state_dict()
    assert {"cand_x_num", "cand_x_cat", "cand_y"} <= set(state)
    assert state["cand_y"].shape[0] == len(y)


def test_tabr_self_exclusion_active_in_training(clf_data):
    # With context_size=1 and no self-exclusion, a memorizing model would
    # retrieve itself; the trainer protocol must prevent that during fit.
    X, y, _, _ = clf_data
    m = MasaClassifier(n_epochs=2, model="tabr", device="cpu", random_state=0,
                       model_params={"d_main": 8, "context_size": 1})
    m.fit(X, y)  # would leak (and often error via inf softmax) if broken
    proba = m.predict_proba(X)
    assert np.isfinite(proba).all()
