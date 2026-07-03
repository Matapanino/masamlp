import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.models import build_model
from masamlp.regressor import MasaRegressor

_PARAMS = {"dim": 32, "d_block": 64}
_KW = dict(model="modernnca", model_params=_PARAMS, device="cpu", random_state=0)


def test_classification_learns_and_probas_normalized(clf_data):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(n_epochs=30, **_KW).fit(X, y)
    assert float((m.predict(X_test) == y_test).mean()) > 0.75
    proba = m.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert m.model_.current_batch_indices is None


def test_multiclass(multiclass_data):
    X, y, X_test, y_test = multiclass_data
    m = MasaClassifier(n_epochs=30, **_KW).fit(X, y)
    assert m.predict_proba(X_test).shape == (len(y_test), 3)
    assert (m.predict(X_test) == y_test).mean() > 0.6


def test_regression_beats_baseline(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(n_epochs=40, **_KW).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    assert rmse < float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))


def test_plr_lite_embedding(reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(num_embedding="plr-lite", n_epochs=10, **_KW).fit(X, y)
    assert np.isfinite(m.predict(X_test)).all()
    assert m.model_.embedding.num_embedding.lite is True


def test_roundtrip_keeps_candidates(tmp_path, clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(n_epochs=5, **_KW).fit(X, y)
    m.save_model(tmp_path / "m")
    loaded = MasaClassifier.load_model(tmp_path / "m")
    assert loaded.model_.has_candidates
    # 1-ulp tolerance: the loaded corpus buffers live at a different memory
    # alignment, which can flip the last bit of BLAS results (platform-dependent).
    np.testing.assert_allclose(
        m.predict_proba(X_test), loaded.predict_proba(X_test), rtol=0, atol=1e-6
    )


def test_candidate_sampling_respects_rate(clf_data):
    X, y, _, _ = clf_data
    m = MasaClassifier(n_epochs=2, model="modernnca", device="cpu", random_state=0,
                       model_params={**_PARAMS, "sample_rate": 0.25})
    m.fit(X, y)  # runs with a quarter of the candidates per step
    model = m.model_
    model.train()
    model.current_batch_indices = torch.arange(10)
    idx = model._candidate_indices(torch.device("cpu"))
    expected = 10 + int((len(y) - 10) * 0.25)
    assert idx.shape[0] == expected
    assert torch.equal(idx[:10], torch.arange(10))  # batch always included first
    model.current_batch_indices = None


def test_invalid_sample_rate_rejected():
    with pytest.raises(ValueError, match="sample_rate"):
        build_model("modernnca", {"sample_rate": 0.0}, 4, [], 1, None)


def test_forward_without_candidates_raises():
    model = build_model("modernnca", dict(_PARAMS), 4, [], 1, None)
    with pytest.raises(RuntimeError, match="candidates"):
        model(torch.randn(4, 4), torch.zeros(4, 0, dtype=torch.int64))


def test_multioutput_regression_supported(rng):
    X = rng.normal(size=(300, 4))
    Y = np.stack([X[:, 0], -X[:, 1]], axis=1) + rng.normal(0, 0.1, (300, 2))
    m = MasaRegressor(n_epochs=20, **_KW).fit(X[:200], Y[:200])
    pred = m.predict(X[200:])
    assert pred.shape == (100, 2)
    assert np.isfinite(pred).all()
