import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.core.objectives import make_objective
from masamlp.regressor import MasaRegressor

_KW = dict(
    model="lnn",
    model_params={"d_hidden": 16, "n_steps": 2, "d_backbone": 32, "dropout": 0.0},
    n_epochs=15,
    batch_size=None,
    device="cpu",
    random_state=3,
)


def test_custom_mse_matches_builtin(reg_data):
    X, y, X_test, _ = reg_data

    def my_mse(y_true, raw_pred):
        return ((raw_pred - y_true) ** 2).mean(dim=1)

    m_custom = MasaRegressor(objective=my_mse, **_KW).fit(X, y)
    m_builtin = MasaRegressor(objective="squared_error", **_KW).fit(X, y)
    np.testing.assert_allclose(m_custom.predict(X_test), m_builtin.predict(X_test), atol=1e-4)


def test_asymmetric_loss_shifts_predictions(reg_data):
    X, y, X_test, _ = reg_data

    def over_penalize_under(y_true, raw_pred):
        err = raw_pred - y_true
        return torch.where(err < 0, 10.0 * err**2, err**2).mean(dim=1)

    m_asym = MasaRegressor(objective=over_penalize_under, **_KW).fit(X, y)
    m_sym = MasaRegressor(objective="squared_error", **_KW).fit(X, y)
    assert m_asym.predict(X_test).mean() > m_sym.predict(X_test).mean()


def test_custom_objective_receives_sample_weight_reduction(reg_data):
    # A custom loss combined with weight=0 rows must ignore them, exactly
    # like built-ins: same trainer-side reduction.
    X, y, _, _ = reg_data
    y_bad = y.copy()
    y_bad[:50] = 100.0
    w = np.ones(len(y))
    w[:50] = 0.0

    def my_mae(y_true, raw_pred):
        return (raw_pred - y_true).abs().mean(dim=1)

    kw = dict(_KW, numeric_scaler="none", target_standardize=False)
    m = MasaRegressor(objective=my_mae, **kw).fit(X, y_bad, sample_weight=w)
    assert np.abs(m.predict(X[:50]) - 100.0).min() > 20.0  # nowhere near the poisoned target


def test_custom_binary_objective_gets_sigmoid_transform(clf_data):
    X, y, X_test, _ = clf_data

    def focal_ish(y_true, raw_pred):
        p = torch.sigmoid(raw_pred[:, 0])
        pt = torch.where(y_true[:, 0] > 0.5, p, 1 - p)  # y_true is (n, out_dim)
        return -((1 - pt) ** 2) * torch.log(pt.clamp(1e-8))

    m = MasaClassifier(objective=focal_ish, n_epochs=15, device="cpu", random_state=0,
                       model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert (m.predict(X_test) == ((X_test[:, 0] + X_test[:, 1]) > 0)).mean() > 0.8


def test_custom_multiclass_objective(multiclass_data):
    X, y, X_test, y_test = multiclass_data

    def my_ce(y_true, raw_pred):
        return torch.nn.functional.cross_entropy(raw_pred, y_true, reduction="none")

    m = MasaClassifier(objective=my_ce, n_epochs=15, device="cpu", random_state=0,
                       model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (m.predict(X_test) == y_test).mean() > 0.6


def test_nn_module_objective_parameters_are_trained(reg_data):
    X, y, _, _ = reg_data

    class LearnableScale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_scale = torch.nn.Parameter(torch.zeros(1))

        def forward(self, y_true, raw_pred):
            # Gaussian NLL with a learned homoscedastic variance.
            var = torch.exp(self.log_scale)
            return (((raw_pred - y_true) ** 2).mean(dim=1) / var + self.log_scale[0]) / 2

    loss_mod = LearnableScale()
    obj = make_objective(loss_mod, name="gauss_nll")
    MasaRegressor(objective=obj, **_KW).fit(X, y)
    assert loss_mod.log_scale.item() != 0.0  # the optimizer actually moved it


def test_reduced_custom_loss_raises(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(objective=lambda t, r: ((r - t) ** 2).mean(), **_KW)
    with pytest.raises(ValueError, match="per-sample"):
        m.fit(X, y)
