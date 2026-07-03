import numpy as np
import pytest

from masamlp.regressor import MasaRegressor

_KW = dict(
    model="resnet",
    model_params={"d": 32, "n_blocks": 1},
    device="cpu",
    random_state=11,
)


def test_same_seed_same_predictions(reg_data):
    X, y, X_test, _ = reg_data
    p1 = MasaRegressor(n_epochs=10, **_KW).fit(X, y).predict(X_test)
    p2 = MasaRegressor(n_epochs=10, **_KW).fit(X, y).predict(X_test)
    np.testing.assert_array_equal(p1, p2)


def test_different_seeds_differ(reg_data):
    X, y, X_test, _ = reg_data
    kw = dict(_KW, random_state=1)
    p1 = MasaRegressor(n_epochs=10, **kw).fit(X, y).predict(X_test)
    kw = dict(_KW, random_state=2)
    p2 = MasaRegressor(n_epochs=10, **kw).fit(X, y).predict(X_test)
    assert not np.allclose(p1, p2)


def test_minibatch_and_fullbatch_both_learn(reg_data):
    X, y, X_test, y_test = reg_data
    base_rmse = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    for bs in (None, 64):
        m = MasaRegressor(n_epochs=30, batch_size=bs, **_KW).fit(X, y)
        rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
        assert rmse < base_rmse * 0.8, f"batch_size={bs} failed to learn"


def test_early_stopping_restores_best_epoch(reg_data):
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(
        n_epochs=200,
        learning_rate=5e-3,
        early_stopping_rounds=8,
        eval_metric="rmse",
        **_KW,
    )
    m.fit(X, y, eval_set=[(X_val, y_val)])
    history = m.evals_result_["valid_0"]["rmse"]
    assert len(history) < 200, "early stopping never triggered"
    assert m.best_iteration_ == int(np.argmin(history))
    assert m.best_score_ == pytest.approx(min(history))
    # Restored weights reproduce the best epoch's validation score.
    rmse_now = float(np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2)))
    assert rmse_now == pytest.approx(m.best_score_, abs=1e-5)


def test_early_stopping_requires_eval_set(reg_data):
    X, y, _, _ = reg_data
    with pytest.raises(ValueError, match="eval_set"):
        MasaRegressor(early_stopping_rounds=5, **_KW).fit(X, y)


def test_multiple_eval_sets_are_all_tracked(reg_data):
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(n_epochs=5, **_KW)
    m.fit(X, y, eval_set=[(X_val, y_val), (X, y)])
    assert set(m.evals_result_) == {"valid_0", "valid_1"}


def test_grad_clip_and_cosine_schedule_run(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(n_epochs=5, grad_clip=1.0, lr_scheduler="cosine", **_KW)
    m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_nonfinite_loss_raises(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(n_epochs=5, learning_rate=1e12, target_standardize=False, **_KW)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit(X, y * 1e6)


def test_weighted_eval_set_tuple_rejected(reg_data):
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(n_epochs=2, **_KW)
    with pytest.raises(ValueError, match="pairs"):
        m.fit(X, y, eval_set=[(X_val, y_val, np.ones(len(y_val)))])


def test_ema_changes_final_weights(reg_data):
    # The EMA weights lag the last optimizer step, so the fitted model (and
    # its predictions) must differ from an otherwise identical no-EMA run.
    X, y, X_test, _ = reg_data
    plain = MasaRegressor(n_epochs=40, learning_rate=5e-3, **_KW).fit(X, y).predict(X_test)
    ema = (
        MasaRegressor(n_epochs=40, learning_rate=5e-3, ema_decay=0.9, **_KW)
        .fit(X, y)
        .predict(X_test)
    )
    assert not np.allclose(plain, ema)


def test_ema_still_learns(reg_data):
    X, y, X_test, y_test = reg_data
    base_rmse = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    m = MasaRegressor(n_epochs=60, learning_rate=5e-3, ema_decay=0.9, **_KW).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    assert rmse < base_rmse * 0.8


def test_ema_early_stopping_restores_and_predicts_best(reg_data):
    # best_score is measured on the EMA weights; the restored model must
    # reproduce it at predict time (proves the EMA copy is what gets saved).
    X, y, X_val, y_val = reg_data
    m = MasaRegressor(
        n_epochs=200,
        learning_rate=5e-3,
        ema_decay=0.9,
        early_stopping_rounds=12,
        eval_metric="rmse",
        **_KW,
    )
    m.fit(X, y, eval_set=[(X_val, y_val)])
    rmse_now = float(np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2)))
    assert rmse_now == pytest.approx(m.best_score_, abs=1e-5)


def test_ema_decay_out_of_range_raises(reg_data):
    X, y, _, _ = reg_data
    with pytest.raises(ValueError, match="ema_decay"):
        MasaRegressor(n_epochs=2, ema_decay=1.5, **_KW).fit(X, y)


def test_ema_rejected_with_vectorized_ensemble(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(
        model="grn",
        model_params={"d": 32, "d_hidden": 32, "n_blocks": 2},
        n_ens=2,
        ens_mode="vectorized",
        ema_decay=0.9,
        device="cpu",
        random_state=11,
    )
    with pytest.raises(ValueError, match="ema_decay"):
        m.fit(X, y)
