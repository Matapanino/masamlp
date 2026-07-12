import numpy as np
import pytest

from conftest import ALL_MODELS, TINY_PARAMS, TRAIN_KWARGS
from masamlp.classifier import MasaClassifier
from masamlp.regressor import MasaRegressor


@pytest.mark.parametrize("name", ALL_MODELS)
def test_regression_beats_mean_baseline(name, reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(
        model=name, model_params=dict(TINY_PARAMS[name]), n_epochs=40,
        device="cpu", random_state=0, **TRAIN_KWARGS.get(name, {}),
    ).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < 0.8 * baseline, f"{name}: rmse {rmse:.3f} vs baseline {baseline:.3f}"


@pytest.mark.parametrize("name", ALL_MODELS)
def test_binary_classification_beats_prior(name, clf_data):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(
        model=name, model_params=dict(TINY_PARAMS[name]), n_epochs=40,
        device="cpu", random_state=0, **TRAIN_KWARGS.get(name, {}),
    ).fit(X, y)
    acc = float((m.predict(X_test) == y_test).mean())
    assert acc > 0.8, f"{name}: accuracy {acc:.3f}"
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_multiclass(multiclass_data):
    X, y, X_test, y_test = multiclass_data
    m = MasaClassifier(n_epochs=40, device="cpu", random_state=0,
                       model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    assert m.predict_proba(X_test).shape == (len(y_test), 3)
    assert (m.predict(X_test) == y_test).mean() > 0.7
    assert m.evals_result_ == {}


def test_string_class_labels(clf_data):
    X, y, X_test, _ = clf_data
    labels = np.array(["neg", "pos"])[y]
    m = MasaClassifier(n_epochs=10, device="cpu", model_params={"d": 32, "n_blocks": 1})
    m.fit(X, labels)
    assert set(m.classes_) == {"neg", "pos"}
    assert set(m.predict(X_test)) <= {"neg", "pos"}


def test_multioutput_regression(rng):
    X = rng.normal(size=(400, 5))
    Y = np.stack([X[:, 0] * 2, -X[:, 1] + X[:, 2]], axis=1) + rng.normal(0, 0.1, (400, 2))
    m = MasaRegressor(n_epochs=40, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X[:300], Y[:300])
    pred = m.predict(X[300:])
    assert pred.shape == (100, 2)
    rmse = float(np.sqrt(np.mean((pred - Y[300:]) ** 2)))
    baseline = float(np.sqrt(np.mean((Y[300:] - Y[:300].mean(axis=0)) ** 2)))
    assert rmse < 0.8 * baseline


def test_mixed_dataframe_with_categoricals(mixed_df):
    df, y = mixed_df
    m = MasaRegressor(n_epochs=40, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(df, y)
    rmse = float(np.sqrt(np.mean((m.predict(df) - y) ** 2)))
    baseline = float(y.std())
    assert rmse < 0.5 * baseline  # the categorical carries most signal
    assert list(m.feature_names_in_) == ["num1", "num2", "cat"]


def test_class_weight_balanced_helps_minority(rng):
    X = rng.normal(size=(600, 4))
    margin = X[:, 0] + 0.2 * rng.normal(size=600)
    y = (margin > 1.1).astype(int)  # rare positive class
    kwargs = dict(n_epochs=30, device="cpu", random_state=0,
                  model_params={"d": 32, "n_blocks": 1})
    m_plain = MasaClassifier(**kwargs).fit(X, y)
    m_bal = MasaClassifier(class_weight="balanced", **kwargs).fit(X, y)
    recall_plain = (m_plain.predict(X)[y == 1] == 1).mean()
    recall_bal = (m_bal.predict(X)[y == 1] == 1).mean()
    assert recall_bal >= recall_plain


def test_class_weight_dict_and_validation(clf_data):
    X, y, _, _ = clf_data
    m = MasaClassifier(class_weight={0: 1.0, 1: 5.0}, n_epochs=5, device="cpu",
                       model_params={"d": 32, "n_blocks": 1})
    m.fit(X, y)
    bad = MasaClassifier(class_weight={99: 1.0}, n_epochs=2, device="cpu")
    with pytest.raises(ValueError, match="training label"):
        bad.fit(X, y)


def test_eval_set_with_unseen_label_raises(clf_data):
    X, y, X_val, y_val = clf_data
    m = MasaClassifier(n_epochs=2, device="cpu", model_params={"d": 32, "n_blocks": 1})
    with pytest.raises(ValueError, match="labels"):
        m.fit(X, y, eval_set=[(X_val, y_val + 5)])


def test_label_smoothing_runs(clf_data):
    X, y, _, _ = clf_data
    m = MasaClassifier(label_smoothing=0.1, n_epochs=5, device="cpu",
                       model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    assert np.isfinite(m.predict_proba(X)).all()


def test_quantile_objective_shifts_predictions(reg_data):
    from masamlp.core.objectives import Quantile

    X, y, X_test, _ = reg_data
    kwargs = dict(n_epochs=40, device="cpu", random_state=0,
                  model_params={"d": 32, "n_blocks": 1})
    m_hi = MasaRegressor(objective=Quantile(alpha=0.9), **kwargs).fit(X, y)
    m_lo = MasaRegressor(objective=Quantile(alpha=0.1), **kwargs).fit(X, y)
    assert m_hi.predict(X_test).mean() > m_lo.predict(X_test).mean()


def test_poisson_objective_skips_standardization_and_predicts_positive(rng):
    X = rng.normal(size=(300, 4))
    y = rng.poisson(np.exp(0.5 * X[:, 0] + 0.2))
    m = MasaRegressor(objective="poisson", n_epochs=20, device="cpu",
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    assert m.target_mean_ is None  # log-link: no target standardization
    assert (m.predict(X) >= 0).all()


def test_num_embedding_options(reg_data):
    X, y, X_test, y_test = reg_data
    for mode in ("plr", "periodic"):
        m = MasaRegressor(num_embedding=mode, n_epochs=20, device="cpu", random_state=0,
                          model_params={"d": 32, "n_blocks": 1}).fit(X, y)
        assert np.isfinite(m.predict(X_test)).all()


def test_eval_batch_size_reaches_training_eval_and_predict(monkeypatch, reg_data):
    import masamlp.core.trainer as trainer_mod
    import masamlp.sklearn as sklearn_mod

    X, y, X_val, y_val = reg_data
    seen: list[int] = []
    orig = trainer_mod.predict_transformed

    def spy(model, data, transform, batch_size=8192, autocast_dtype=None):
        seen.append(batch_size)
        return orig(model, data, transform, batch_size, autocast_dtype=autocast_dtype)

    monkeypatch.setattr(trainer_mod, "predict_transformed", spy)
    monkeypatch.setattr(sklearn_mod, "predict_transformed", spy)
    m = MasaRegressor(n_epochs=2, eval_batch_size=64, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1})
    m.fit(X, y, eval_set=[(X_val, y_val)])  # per-epoch eval batches
    m.predict(X_val)  # inference batches
    assert seen and set(seen) == {64}
