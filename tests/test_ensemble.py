import numpy as np
import pytest
import torch

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
    with pytest.raises(ValueError, match="ens_mode"):
        MasaRegressor(n_ens=2, ens_mode="parallel", **_KW).fit(X, y)


# ------------------------------------------------------------------ #
# Vectorized (torch.func) ensembles
# ------------------------------------------------------------------ #
def test_vectorized_matches_loop_quality_and_averages(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(n_ens=3, ens_mode="vectorized", **_KW).fit(X, y)
    assert len(m.models_) == 3
    member_preds = _member_predictions(m, X_test)
    np.testing.assert_allclose(m.predict(X_test), np.mean(member_preds, axis=0), atol=1e-6)
    assert not np.allclose(member_preds[0], member_preds[1]), "members must differ"
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < 0.8 * baseline


def test_vectorized_is_deterministic(reg_data):
    X, y, X_test, _ = reg_data
    p1 = MasaRegressor(n_ens=2, ens_mode="vectorized", **_KW).fit(X, y).predict(X_test)
    p2 = MasaRegressor(n_ens=2, ens_mode="vectorized", **_KW).fit(X, y).predict(X_test)
    np.testing.assert_array_equal(p1, p2)


def test_vectorized_per_member_early_stopping(reg_data):
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(n_ens=2, ens_mode="vectorized", model="lnn",
                      model_params={"d_hidden": 16, "n_steps": 2, "d_backbone": 32},
                      n_epochs=60, early_stopping_rounds=8, device="cpu", random_state=0)
    m.fit(X, y, eval_set=[(X_val, y_val)])
    assert m.best_iteration_ is not None
    history = m.evals_result_["valid_0"]["rmse"]
    assert m.best_iteration_ == int(np.argmin(history))
    # first member's restored weights reproduce its best validation score
    member_pred = _member_predictions(m, X_val)[0]
    rmse_now = float(np.sqrt(np.mean((member_pred - y_val) ** 2)))
    assert rmse_now == pytest.approx(min(history), abs=1e-5)


def test_vectorized_roundtrip(tmp_path, reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(n_ens=2, ens_mode="vectorized", **_KW).fit(X, y)
    m.save_model(tmp_path / "m")
    loaded = MasaRegressor.load_model(tmp_path / "m")
    np.testing.assert_array_equal(m.predict(X_test), loaded.predict(X_test))


def test_vectorized_rejects_ineligible_models(reg_data):
    X, y, _, _ = reg_data
    with pytest.raises(ValueError, match="BatchNorm"):
        MasaRegressor(model="resnet", n_ens=2, ens_mode="vectorized",
                      n_epochs=2, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="retrieval"):
        MasaRegressor(model="tabr", model_params={"d_main": 16, "context_size": 8},
                      n_ens=2, ens_mode="vectorized", n_epochs=2, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="grad_clip"):
        MasaRegressor(n_ens=2, ens_mode="vectorized", grad_clip=1.0, **_KW).fit(X, y)


def test_vectorized_bn_error_names_model_and_is_early(reg_data):
    # n_epochs is large but the raise must happen before any training, with a
    # message that names the offending model.
    X, y, _, _ = reg_data
    with pytest.raises(ValueError, match="resnet"):
        MasaRegressor(model="resnet", model_params={"d": 32, "n_blocks": 1},
                      n_ens=2, ens_mode="vectorized", n_epochs=5000,
                      device="cpu", random_state=0).fit(X, y)


# ------------------------------------------------------------------ #
# Vectorized parity with the loop path: the per-step schedules
# ------------------------------------------------------------------ #
def test_vectorized_dropout_schedule_follows_the_loop(reg_data):
    """RealMLP-TD's flat_cos dropout schedule must anneal in vectorized mode
    too. The keep probability the members apply is the *stacked* ``_keep``
    buffer, which is also what the unstack step writes back into each member —
    so the value left on a member is the one its last training step used."""
    from masamlp.core.trainer import flat_cos
    from masamlp.models.realmlp import ScheduledDropout

    X, y, _, _ = reg_data
    p, n_epochs = 0.4, 4
    kw = dict(
        model="realmlp",
        model_params={"hidden_sizes": [16], "dropout": p, "dropout_schedule": "flat_cos"},
        n_epochs=n_epochs, device="cpu", random_state=0,
        learning_rate=0.05, optimizer="adam", optimizer_betas=(0.9, 0.95),
        lr_scheduler="coslog4",
    )

    def keeps(est):
        return [float(m._keep) for model in est.models_
                for m in model.modules() if isinstance(m, ScheduledDropout)]

    loop = keeps(MasaRegressor(n_ens=2, ens_mode="loop", **kw).fit(X, y))
    vectorized = keeps(MasaRegressor(n_ens=2, ens_mode="vectorized", **kw).fit(X, y))
    # 300 rows train full-batch, so the last of the n_epochs steps sits at
    # t = (n_epochs - 1) / n_epochs — well past the frozen 1 - p.
    expected = 1.0 - p * flat_cos((n_epochs - 1) / n_epochs)
    assert loop == pytest.approx([expected] * 2)
    assert vectorized == pytest.approx(loop)


def test_vectorized_honours_weight_decay_mode(monkeypatch, reg_data):
    """``weight_decay_mode`` picks the optimizer class that implements the
    decay convention (AdamW decouples, Adam couples). The vectorized path must
    build the same one the loop path builds — and must actually train with it."""
    import masamlp.core.ensemble as ens_mod
    import masamlp.core.trainer as trainer_mod

    X, y, X_test, _ = reg_data
    base = dict(_KW, n_epochs=1, weight_decay=0.1)

    def spy(module) -> list[type]:
        seen: list[type] = []
        real = module._make_optimizer

        def wrapper(*args, **kwargs):
            optimizer = real(*args, **kwargs)
            seen.append(type(optimizer))
            return optimizer

        monkeypatch.setattr(module, "_make_optimizer", wrapper)
        return seen

    loop_seen, vectorized_seen = spy(trainer_mod), spy(ens_mod)
    for optimizer, mode in [("adam", "auto"), ("adamw", "coupled")]:
        kw = dict(base, optimizer=optimizer, weight_decay_mode=mode)
        MasaRegressor(n_ens=1, ens_mode="loop", **kw).fit(X, y)
        MasaRegressor(n_ens=2, ens_mode="vectorized", **kw).fit(X, y)
    assert vectorized_seen == loop_seen == [torch.optim.Adam, torch.optim.Adam]

    # ... and the choice reaches the weights: coupling the decay into the
    # gradient is not the same update as decoupling it.
    fit = dict(_KW, n_epochs=8, weight_decay=0.5, learning_rate=0.01, optimizer="adam")
    coupled = MasaRegressor(n_ens=2, ens_mode="vectorized",
                            weight_decay_mode="auto", **fit).fit(X, y).predict(X_test)
    decoupled = MasaRegressor(n_ens=2, ens_mode="vectorized",
                              weight_decay_mode="decoupled", **fit).fit(X, y).predict(X_test)
    assert not np.allclose(coupled, decoupled)
