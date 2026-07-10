import numpy as np
import pytest
import torch

from masamlp.core.trainer import flat_cos
from masamlp.models import build_model
from masamlp.models.realmlp import NTPLinear, ParametricActivation, ScheduledDropout
from masamlp.presets import realmlp_params, realmlp_td_params
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


def test_flat_cos_schedule_shape():
    assert flat_cos(0.0) == 1.0 and flat_cos(0.49) == 1.0
    assert flat_cos(0.75) == pytest.approx(0.5)
    assert flat_cos(1.0) == pytest.approx(0.0, abs=1e-12)


def test_parametric_activation_starts_as_plain_activation():
    torch.manual_seed(0)
    act = ParametricActivation(4, torch.selu)
    x = torch.randn(8, 4)
    assert torch.allclose(act(x), torch.selu(x))  # alpha init 1
    with torch.no_grad():
        act.alpha.zero_()
    assert torch.allclose(act(x), x)  # alpha 0 -> identity


def test_scheduled_dropout_reacts_to_schedule():
    drop = ScheduledDropout(0.5)
    drop.train()
    x = torch.ones(64, 32)
    drop.set_factor(0.0)  # end of flat_cos: dropout off, exact identity
    np.testing.assert_array_equal(drop(x).numpy(), x.numpy())
    drop.set_factor(1.0)
    out = drop(x)
    assert (out == 0).any()
    assert torch.allclose(out[out != 0], torch.full_like(out[out != 0], 2.0))
    drop.eval()
    np.testing.assert_array_equal(drop(x).numpy(), x.numpy())


def test_td_model_schedule_hook_and_param_groups():
    model = build_model(
        "realmlp",
        {"hidden_sizes": [16], "num_scaling": True, "use_parametric_act": True,
         "dropout": 0.15, "dropout_schedule": "flat_cos", "act_lr_factor": 0.1,
         "plr_lr_factor": 0.1},
        4, [3], 1, "pbld",
    )
    model.set_schedule_t(1.0)
    drops = [m for m in model.modules() if isinstance(m, ScheduledDropout)]
    # flat_cos(1.0) == 0 -> dropout off -> keep probability 1.
    assert drops and all(
        float(d._keep) == pytest.approx(1.0, abs=1e-12) for d in drops
    )
    groups = model.param_groups()
    bias_group = next(g for g in groups if g.get("wd_factor") == 0.0)
    assert bias_group["lr_factor"] == 0.1
    assert sum(len(g["params"]) for g in groups) == sum(
        1 for p in model.parameters() if p.requires_grad
    )
    factors = {g["lr_factor"] for g in groups}
    assert {6.0, 1.0, 0.1} <= factors


def test_td_preset_learns(mixed_df):
    df, y = mixed_df
    params = realmlp_td_params("regression")
    params.update(n_epochs=30, random_state=0, device="cpu")
    m = MasaRegressor(**params).fit(df, y)
    rmse = float(np.sqrt(np.mean((m.predict(df) - y) ** 2)))
    assert rmse < 0.4 * float(y.std())
    assert m.resolved_model_params_["use_parametric_act"] is True
    assert m.target_min_ is not None  # clip_predictions on for regression


def test_td_preset_contents():
    clf = realmlp_td_params("classification")
    assert clf["learning_rate"] == 0.04 and clf["label_smoothing"] == 0.1
    assert clf["weight_decay_schedule"] == "flat_cos" and clf["cat_encoding"] == "hybrid"
    reg = realmlp_td_params("regression")
    assert reg["learning_rate"] == 0.2 and reg["clip_predictions"] is True
    assert reg["num_embedding"] == "pbld"


def test_weight_decay_schedule_validation(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(weight_decay_schedule="cosine", n_epochs=2, device="cpu",
                      model_params={"d": 32, "n_blocks": 1})
    with pytest.raises(ValueError, match="weight_decay_schedule"):
        m.fit(X, y)


def test_coslog4_with_minibatches(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(lr_scheduler="coslog4", batch_size=64, learning_rate=0.05,
                      optimizer="adam", n_epochs=30, device="cpu", random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    assert rmse < 0.8 * float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
