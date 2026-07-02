import numpy as np

from masamlp.core.metrics import make_metric
from masamlp.regressor import MasaRegressor

_KW = dict(
    model="resnet",
    model_params={"d": 32, "n_blocks": 1},
    n_epochs=40,
    device="cpu",
    random_state=1,
)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def test_custom_metric_appears_in_evals_result(reg_data):
    X, y, X_val, y_val = reg_data
    metric = make_metric(_rmse, name="my_rmse")
    m = MasaRegressor(eval_metric=[metric, "mae"], **_KW)
    m.fit(X, y, eval_set=[(X_val, y_val)])
    assert set(m.evals_result_["valid_0"]) == {"my_rmse", "mae"}
    assert len(m.evals_result_["valid_0"]["my_rmse"]) == 40


def test_custom_metric_drives_early_stopping_both_directions(reg_data):
    X, y, X_val, y_val = reg_data
    kw = dict(_KW, early_stopping_rounds=5)

    min_metric = make_metric(_rmse, name="rmse_min", minimize=True)
    m_min = MasaRegressor(eval_metric=min_metric, **kw)
    m_min.fit(X, y, eval_set=[(X_val, y_val)])

    max_metric = make_metric(lambda t, p: -_rmse(t, p), name="neg_rmse", minimize=False)
    m_max = MasaRegressor(eval_metric=max_metric, **kw)
    m_max.fit(X, y, eval_set=[(X_val, y_val)])

    # Same training trajectory, mirrored metric: identical stopping decisions.
    assert m_min.best_iteration_ == m_max.best_iteration_
    assert m_min.best_score_ == -m_max.best_score_
    history = m_min.evals_result_["valid_0"]["rmse_min"]
    assert m_min.best_score_ == min(history)


def test_plain_callable_metric_is_wrapped(reg_data):
    X, y, X_val, y_val = reg_data

    def mean_abs(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    m = MasaRegressor(eval_metric=mean_abs, **_KW)
    m.fit(X, y, eval_set=[(X_val, y_val)])
    assert "mean_abs" in m.evals_result_["valid_0"]
