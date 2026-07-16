"""predict_members / predict_proba_members (ADR 0005 §6): per-member
predictions on the prediction scale — probabilities for classification, the
original target scale for regression — shaped ``(n, m[, C])`` with
``m = n_ens * k`` members ordered outer-major. The frozen invariant: the mean
over the member axis reproduces ``predict`` / ``predict_proba`` (uniform
two-stage mean; affine inverse-standardization commutes). k-less
architectures expose their ``n_ens`` outer members; tabm contributes ``k``
inner members per outer member."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from masamlp.classifier import MasaClassifier
from masamlp.regressor import MasaRegressor

TABM = {"k": 4, "d": 16, "n_blocks": 1}
TABM_K2 = {"k": 2, "d": 16, "n_blocks": 1}
RESNET = {"d": 32, "n_blocks": 1}
TOL = dict(rtol=1e-5, atol=1e-6)


def test_regressor_invariant_and_shape(reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(
        model="tabm", model_params=TABM, n_ens=2,
        n_epochs=5, eval_batch_size=32,  # 100 test rows -> multi-chunk 3D path
        device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_members(X_test)
    assert members.shape == (len(X_test), 8)
    assert np.isfinite(members).all()
    # Members must differ (or the ensemble is a no-op) yet average back to
    # predict: transform-then-mean on both paths, and the original-scale
    # mapping is affine so it commutes with the mean.
    assert members.std(axis=1).max() > 1e-6
    np.testing.assert_allclose(members.mean(axis=1), m.predict(X_test), **TOL)


@pytest.mark.parametrize("ens_mode", ["loop", "vectorized"])
def test_multiclass_invariant_across_ens_modes(multiclass_data, ens_mode):
    X, y, X_test, _ = multiclass_data
    m = MasaClassifier(
        model="tabm", model_params=TABM, n_ens=2, ens_mode=ens_mode,
        n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_proba_members(X_test)
    assert members.shape == (len(X_test), 8, 3)
    # Every member is itself a distribution over the classes.
    np.testing.assert_allclose(members.sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(members.mean(axis=1), m.predict_proba(X_test), **TOL)


def test_binary_members_stack_both_classes(clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(
        model="tabm", model_params=TABM, n_epochs=5, device="cpu", random_state=0
    ).fit(X, y)
    members = m.predict_proba_members(X_test)
    assert members.shape == (len(X_test), 4, 2)
    np.testing.assert_allclose(members.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(members.mean(axis=1), m.predict_proba(X_test), **TOL)


@pytest.mark.parametrize("n_ens", [1, 2])
def test_k_less_arch_members_are_the_outer_members(reg_data, n_ens):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(
        model="resnet", model_params=RESNET, n_ens=n_ens,
        n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_members(X_test)
    assert members.shape == (len(X_test), n_ens)
    np.testing.assert_allclose(members.mean(axis=1), m.predict(X_test), **TOL)
    if n_ens == 1:
        # A single 2D member takes the exact same ops as predict.
        np.testing.assert_array_equal(members[:, 0], m.predict(X_test))


def test_member_order_is_outer_major(multiclass_data):
    """members[:, i*k:(i+1)*k] belong to models_[i]: restricting the
    estimator to one outer member must reproduce its block."""
    X, y, X_test, _ = multiclass_data
    m = MasaClassifier(
        model="tabm", model_params=TABM_K2, n_ens=2,
        n_epochs=4, device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_proba_members(X_test)
    assert members.shape == (len(X_test), 4, 3)
    all_models = m.models_
    for i, model in enumerate(all_models):
        m.models_ = [model]
        m.model_ = model
        np.testing.assert_array_equal(
            members[:, 2 * i : 2 * (i + 1)], m.predict_proba_members(X_test)
        )
    m.models_, m.model_ = all_models, all_models[0]


def test_members_survive_save_load(tmp_path, reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(
        model="tabm", model_params=TABM, n_ens=2,
        n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    before = m.predict_members(X_test)
    m.save_model(tmp_path / "m")
    loaded = MasaRegressor.load_model(tmp_path / "m")
    after = loaded.predict_members(X_test)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(after.mean(axis=1), loaded.predict(X_test), **TOL)


def test_multioutput_regression_members(rng):
    X = rng.normal(size=(200, 5))
    y = np.stack([X[:, 0] + 0.1 * rng.normal(size=200), X[:, 1] - X[:, 2]], axis=1)
    m = MasaRegressor(
        model="tabm", model_params=TABM_K2, n_ens=2,
        n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_members(X[:20])
    assert members.shape == (20, 4, 2)
    np.testing.assert_allclose(members.mean(axis=1), m.predict(X[:20]), **TOL)


def test_non_identity_transform_averages_on_prediction_scale(rng):
    """Poisson predicts exp(raw): members are exp-per-member (all positive)
    and their mean is predict — transform-then-mean, ADR 0005 decision 2."""
    X = rng.normal(size=(200, 4))
    y = rng.poisson(np.exp(0.5 * X[:, 0] + 0.2 * X[:, 1])).astype(np.float64)
    m = MasaRegressor(
        model="resnet", model_params=RESNET, objective="poisson", n_ens=2,
        n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_members(X[:30])
    assert members.shape == (30, 2)
    assert (members > 0).all()
    np.testing.assert_allclose(members.mean(axis=1), m.predict(X[:30]), **TOL)


def test_clip_predictions_clips_each_member(reg_data):
    """With clip_predictions, every member individually respects the observed
    target range (the mean-equality then holds only where no member clips)."""
    X, y, X_test, _ = reg_data
    m = MasaRegressor(
        model="resnet", model_params=RESNET, clip_predictions=True, n_ens=2,
        n_epochs=2, device="cpu", random_state=0,
    ).fit(X, y)
    members = m.predict_members(X_test)
    assert members.min() >= y.min()
    assert members.max() <= y.max()


def test_not_fitted_raises():
    with pytest.raises(NotFittedError):
        MasaRegressor().predict_members(np.zeros((3, 2)))
    with pytest.raises(NotFittedError):
        MasaClassifier().predict_proba_members(np.zeros((3, 2)))
