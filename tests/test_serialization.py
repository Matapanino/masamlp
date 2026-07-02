import numpy as np
import pytest

from masamlp.classifier import MasaClassifier
from masamlp.regressor import MasaRegressor

_KW = dict(n_epochs=8, device="cpu", random_state=0, model_params={"d": 32, "n_blocks": 1})


def test_regressor_roundtrip(tmp_path, mixed_df):
    df, y = mixed_df
    m = MasaRegressor(**_KW).fit(df, y)
    m.save_model(tmp_path / "model")
    loaded = MasaRegressor.load_model(tmp_path / "model")
    np.testing.assert_array_equal(m.predict(df), loaded.predict(df))
    assert loaded.get_params()["model_params"] == {"d": 32, "n_blocks": 1}


def test_classifier_roundtrip_with_string_labels(tmp_path, clf_data):
    X, y, X_test, _ = clf_data
    labels = np.array(["neg", "pos"])[y]
    m = MasaClassifier(**_KW).fit(X, labels)
    m.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))
    np.testing.assert_array_equal(loaded.classes_, np.array(["neg", "pos"]))


def test_multiclass_roundtrip(tmp_path, multiclass_data):
    X, y, X_test, _ = multiclass_data
    m = MasaClassifier(**_KW).fit(X, y)
    m.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_custom_objective_warns_but_predicts(tmp_path, reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(objective=lambda t, r: ((r - t) ** 2).mean(dim=1), **_KW).fit(X, y)
    with pytest.warns(UserWarning, match="not serialized"):
        m.save_model(tmp_path / "model")
    loaded = MasaRegressor.load_model(tmp_path / "model")
    np.testing.assert_array_equal(m.predict(X_test), loaded.predict(X_test))


def test_wrong_class_raises(tmp_path, reg_data):
    X, y, _, _ = reg_data
    MasaRegressor(**_KW).fit(X, y).save_model(tmp_path / "model")
    with pytest.raises(ValueError, match="MasaRegressor"):
        MasaClassifier.load_model(tmp_path / "model")


def test_evals_result_survives(tmp_path, reg_data):
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(early_stopping_rounds=5, n_epochs=50, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1})
    m.fit(X, y, eval_set=[(X_val, y_val)])
    m.save_model(tmp_path / "model")
    loaded = MasaRegressor.load_model(tmp_path / "model")
    assert loaded.best_iteration_ == m.best_iteration_
    assert loaded.evals_result_ == m.evals_result_
