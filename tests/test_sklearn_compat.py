import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

from masamlp.classifier import MasaClassifier
from masamlp.regressor import MasaRegressor

_KW = dict(n_epochs=5, device="cpu", model_params={"d": 32, "n_blocks": 1})


def test_clone_and_params_roundtrip():
    m = MasaRegressor(learning_rate=0.01, model="lnn", **_KW)
    cloned = clone(m)
    assert cloned.get_params() == m.get_params()
    cloned.set_params(learning_rate=0.02)
    assert cloned.learning_rate == 0.02 and m.learning_rate == 0.01


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        MasaRegressor(**_KW).predict(np.zeros((2, 3)))
    with pytest.raises(NotFittedError):
        MasaClassifier(**_KW).predict_proba(np.zeros((2, 3)))


def test_fit_returns_self(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(**_KW)
    assert m.fit(X, y) is m


def test_score_methods(reg_data, clf_data):
    X, y, X_test, y_test = reg_data
    reg = MasaRegressor(n_epochs=30, device="cpu", random_state=0,
                        model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    assert reg.score(X_test, y_test) > 0.0  # R^2 from RegressorMixin
    Xc, yc, Xc_test, yc_test = clf_data
    clf = MasaClassifier(n_epochs=30, device="cpu", random_state=0,
                         model_params={"d": 32, "n_blocks": 1}).fit(Xc, yc)
    assert clf.score(Xc_test, yc_test) > 0.7  # accuracy from ClassifierMixin


def test_works_with_cross_val_score(clf_data):
    X, y, _, _ = clf_data
    scores = cross_val_score(MasaClassifier(**_KW), X, y, cv=2)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_fitted_attributes(mixed_df):
    df, y = mixed_df
    m = MasaRegressor(**_KW).fit(df, y)
    assert m.n_features_in_ == 3
    assert list(m.feature_names_in_) == ["num1", "num2", "cat"]
    assert m.best_iteration_ is None  # no early stopping requested
