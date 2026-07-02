import numpy as np
import pytest
import torch

from masamlp.models import build_model
from masamlp.models.realmlp import NTPLinear
from masamlp.presets import realmlp_params
from masamlp.regressor import MasaRegressor


def test_param_groups_factors():
    model = build_model(
        "realmlp", {"hidden_sizes": [16], "num_scaling": True}, 4, [3], 1, None
    )
    groups = model.param_groups()
    factors = [g["lr_factor"] for g in groups]
    assert set(factors) == {6.0, 1.0, 0.1}
    scale_group = next(g for g in groups if g["lr_factor"] == 6.0)
    assert len(scale_group["params"]) == 1  # the scaling layer
    total = sum(len(g["params"]) for g in groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_zero_init_output_starts_at_bias():
    torch.manual_seed(0)
    layer = NTPLinear(8, 1, zero_init=True)
    x = torch.randn(5, 8)
    assert torch.allclose(layer(x), torch.zeros(5, 1))


def test_preset_contents():
    reg = realmlp_params("regression")
    clf = realmlp_params("classification")
    assert reg["model"] == "realmlp" and reg["numeric_scaler"] == "rssc"
    assert reg["lr_scheduler"] == "coslog4" and reg["cat_encoding"] == "onehot"
    assert clf["learning_rate"] == 0.04 and clf["label_smoothing"] == 0.1
    assert "label_smoothing" not in reg
    with pytest.raises(ValueError):
        realmlp_params("clustering")


def test_realmlp_recipe_learns(mixed_df):
    df, y = mixed_df
    params = realmlp_params("regression")
    params.update(n_epochs=30, random_state=0, device="cpu")
    m = MasaRegressor(**params).fit(df, y)
    rmse = float(np.sqrt(np.mean((m.predict(df) - y) ** 2)))
    assert rmse < 0.4 * float(y.std())
    # architecture defaults were applied and recorded
    assert m.resolved_model_params_["num_scaling"] is True
    assert m.resolved_model_params_["activation"] == "mish"


def test_model_params_override_defaults(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(model="realmlp", n_epochs=2, device="cpu",
                      model_params={"activation": "relu", "hidden_sizes": [16]})
    m.fit(X, y)
    assert m.resolved_model_params_["activation"] == "relu"


def test_clip_predictions_bounds_output(rng):
    X = rng.normal(size=(300, 3))
    y = 3.0 * X[:, 0]
    m = MasaRegressor(clip_predictions=True, n_epochs=20, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    X_far = rng.normal(size=(100, 3)) * 10  # force extrapolation
    pred = m.predict(X_far)
    assert pred.min() >= y.min() and pred.max() <= y.max()


def test_clip_predictions_survives_roundtrip(tmp_path, rng):
    X = rng.normal(size=(200, 3))
    y = 2.0 * X[:, 0]
    m = MasaRegressor(clip_predictions=True, n_epochs=5, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    m.save_model(tmp_path / "model")
    loaded = MasaRegressor.load_model(tmp_path / "model")
    X_far = rng.normal(size=(50, 3)) * 10
    np.testing.assert_array_equal(m.predict(X_far), loaded.predict(X_far))


def test_betas_validation(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(optimizer="sgd", optimizer_betas=(0.9, 0.95), n_epochs=2,
                      device="cpu", model_params={"d": 32, "n_blocks": 1})
    with pytest.raises(ValueError, match="betas"):
        m.fit(X, y)


def test_coslog4_with_minibatches(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(lr_scheduler="coslog4", batch_size=64, learning_rate=0.05,
                      optimizer="adam", n_epochs=30, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    assert rmse < 0.8 * float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
