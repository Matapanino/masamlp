"""Fixed-horizon refit controls.

Only tests whose names contain ``no_fit`` are suitable for local execution in
the competition workspace. All ``remote`` tests train and must run remotely.
"""

import math

import numpy as np
import pytest
import torch
from sklearn.base import clone
from torch import nn

from masamlp.classifier import MasaClassifier
from masamlp.core.ensemble import fit_vectorized
from masamlp.core.objectives import BinaryLogistic
from masamlp.core.schedules import (
    SCHEDULE_NAMES,
    coslog4,
    flat_anneal,
    flat_cos,
    invsqrtp1e3,
    resolve_epoch_count,
    resolve_schedule,
    sqrt_cos,
)
from masamlp.core.trainer import Trainer, TrainerConfig, _coslog4
from masamlp.core.trainer import flat_cos as legacy_flat_cos
from masamlp.data.dataset import TabularData
from masamlp.sklearn import BaseMasaModel


def test_no_fit_schedule_reference_points():
    assert resolve_schedule("none") is None
    assert resolve_schedule("cos_log_15") is resolve_schedule("coslog4")
    assert _coslog4 is coslog4
    assert legacy_flat_cos is flat_cos
    assert flat_anneal(0.0) == flat_anneal(0.6) == 1.0
    assert flat_anneal(0.8) == pytest.approx(0.500005, abs=1e-15)
    assert flat_anneal(1.0) == pytest.approx(1e-5, abs=1e-15)
    assert invsqrtp1e3(0.0) == 1.0
    assert invsqrtp1e3(0.003) == pytest.approx(0.5, abs=1e-15)
    assert sqrt_cos(0.0) == 0.0
    assert sqrt_cos(0.1) == pytest.approx(0.05 / (math.sqrt(0.1) + 0.05), abs=1e-15)
    assert sqrt_cos(0.2) == pytest.approx(0.0, abs=1e-15)
    # Each log-time midpoint lands at an independent known cosine peak.
    for cycle in range(4):
        t = (2 ** (cycle + 0.5) - 1) / 15
        assert coslog4(t) == pytest.approx(1.0, abs=1e-14)
    for name in SCHEDULE_NAMES:
        assert callable(resolve_schedule(name))
    with pytest.raises(ValueError, match="Unknown custom_knob"):
        resolve_schedule("unknown", parameter="custom_knob")


def test_no_fit_legacy_schedule_arithmetic_is_unchanged():
    for t in np.linspace(0, 1, 101):
        assert coslog4(t) == float(0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t)))
        old_flat = 1.0 if t < 0.5 else float(0.5 * (1 + np.cos(np.pi * (t - 0.5) / 0.5)))
        assert flat_cos(t) == old_flat


@pytest.mark.parametrize("name", [None, [], np.array(["none"])])
def test_no_fit_malformed_schedule_name(name):
    with pytest.raises(ValueError, match="Unknown schedule"):
        resolve_schedule(name)


@pytest.mark.parametrize("stop", [0, -1, True, False, 1.2, "2", 7, float("nan")])
def test_no_fit_invalid_stop_count(stop):
    with pytest.raises(ValueError, match="stop_epoch"):
        resolve_epoch_count(6, stop)


@pytest.mark.parametrize("horizon", [-1, True, 1.2, "6"])
def test_no_fit_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="n_epochs"):
        resolve_epoch_count(horizon, None)


def test_no_fit_epoch_count_and_estimator_parameter_roundtrip():
    assert resolve_epoch_count(0, None) == 0
    with pytest.raises(ValueError, match="stop_epoch"):
        resolve_epoch_count(0, 0)
    assert resolve_epoch_count(6, None) == 6
    assert resolve_epoch_count(6, 5) == 5
    assert resolve_epoch_count(6, 6) == 6
    assert resolve_epoch_count(np.int64(6), np.int64(5)) == 5
    assert TrainerConfig().stop_epoch is None
    model = BaseMasaModel(n_epochs=6, stop_epoch=5)
    assert clone(model).get_params()["stop_epoch"] == 5
    assert clone(model).get_params()["n_epochs"] == 6


class TraceLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = nn.Linear(2, 1)
        self.times = []

    def forward(self, x_num, x_cat):
        return self.output_layer(x_num)

    def set_schedule_t(self, t):
        self.times.append(t)

    def param_groups(self):
        return [
            {"params": [self.output_layer.weight], "lr_factor": 0.25, "wd_factor": 0.75},
            {"params": [self.output_layer.bias], "lr_factor": 0.1, "wd_factor": 0.0},
        ]


class TraceObjective(BinaryLogistic):
    def __init__(self):
        super().__init__(label_smoothing=0.02)
        self.smoothing = []

    def per_sample_loss(self, y_true, raw_pred):
        self.smoothing.append(self.label_smoothing)
        return super().per_sample_loss(y_true, raw_pred)


@pytest.mark.parametrize("mode", ["loop", "vectorized"])
@pytest.mark.parametrize("lr_name", ["flat_anneal", "cosine"])
def test_remote_fixed_horizon_schedule_trajectory(mode, lr_name, monkeypatch, capsys):
    # Three batches per epoch: stop at 15 steps of the original 18-step horizon.
    torch.manual_seed(17)
    members = [TraceLinear() for _ in range(2 if mode == "vectorized" else 1)]
    data = TabularData(
        x_num=torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24,
        x_cat=torch.empty(12, 0, dtype=torch.long),
        y=torch.arange(12).remainder(2).float(),
    )
    objective = TraceObjective()
    observed = []
    original_step = torch.optim.AdamW.step

    def observe_step(optimizer, *args, **kwargs):
        observed.append([(g["lr"], g["weight_decay"]) for g in optimizer.param_groups])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", observe_step)
    config = TrainerConfig(
        n_epochs=6, stop_epoch=5, batch_size=4, learning_rate=0.04, weight_decay=0.008,
        lr_scheduler=lr_name, weight_decay_schedule="cos_log_15",
        label_smoothing_schedule="sqrt_cos", device="cpu", amp=False, n_threads=1,
        random_state=17, verbose=999,
    )
    if mode == "loop":
        results = [Trainer().fit(members[0], objective, data, [], [], config)]
    else:
        results = fit_vectorized(members, objective, data, [], [], config)
    t = np.arange(15) / 18
    np.testing.assert_array_equal(members[0].times, t)
    assert len(observed) == 15
    for i, progress in enumerate(t):
        lr_factor = (1.0 if progress < 0.6 else 1.0 + (1e-5 - 1.0) *
                     0.5 * (1.0 - math.cos(math.pi * ((progress - 0.6) / 0.4))))
        if lr_name == "cosine":
            lr_factor = 0.5 * (1 + math.cos(math.pi * (i // 3) / 6))
        wd_factor = 0.5 * (1.0 - math.cos(2 * math.pi * math.log2(1 + 15 * progress)))
        np.testing.assert_allclose(
            observed[i], [[0.04 * 0.25 * lr_factor, 0.008 * 0.75 * wd_factor],
                          [0.04 * 0.1 * lr_factor, 0.0]], rtol=0, atol=1e-15,
        )
    expected_smoothing = [0.02 * 0.05 / (math.sqrt(v) + 0.05) *
                          (0.5 - 0.5 * math.cos(10 * math.pi * v)) for v in t]
    actual_smoothing = np.asarray(objective.smoothing).reshape(15, len(members))
    np.testing.assert_allclose(actual_smoothing, np.repeat(
        np.asarray(expected_smoothing)[:, None], len(members), axis=1), atol=1e-15, rtol=0)
    assert objective.label_smoothing == 0.02
    assert all(result.best_iteration == 4 for result in results)
    assert "[4]" in capsys.readouterr().out


def _tiny_classification():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(40, 3)).astype(np.float32)
    return x, (x[:, 0] > x[:, 1]).astype(np.int64)


@pytest.mark.parametrize("mode", ["loop", "vectorized"])
def test_remote_stop_count_save_load_and_history(mode, tmp_path):
    x, y = _tiny_classification()
    model = MasaClassifier(
        model="realmlp", model_params={"hidden_sizes": [8], "dropout": 0.0},
        numeric_scaler="standard", n_epochs=6, stop_epoch=3, batch_size=8,
        n_ens=2, ens_mode=mode, learning_rate=0.01, weight_decay=0.005,
        lr_scheduler="flat_anneal", weight_decay_schedule="cos_log_15",
        label_smoothing_schedule="sqrt_cos", label_smoothing=0.02,
        device="cpu", amp=False, n_threads=1, random_state=19,
    ).fit(x, y, eval_set=[(x, y)])
    assert model.best_iteration_ == 2
    assert all(len(values) == 3 for values in model.evals_result_["valid_0"].values())
    model.save_model(str(tmp_path / "model"))
    restored = MasaClassifier.load_model(str(tmp_path / "model"))
    assert restored.get_params()["n_epochs"] == 6
    assert restored.get_params()["stop_epoch"] == 3
    assert restored.best_iteration_ == 2
    assert restored.evals_result_ == model.evals_result_
    np.testing.assert_array_equal(model.predict_proba(x), restored.predict_proba(x))


@pytest.mark.parametrize("mode", ["loop", "vectorized"])
def test_remote_default_path_and_explicit_full_horizon_match(mode):
    x, y = _tiny_classification()
    kwargs = dict(
        model="realmlp", model_params={"hidden_sizes": [8], "dropout": 0.0},
        numeric_scaler="standard", n_epochs=3, n_ens=2, ens_mode=mode,
        lr_scheduler="coslog4", device="cpu", amp=False, n_threads=1, random_state=19,
    )
    default = MasaClassifier(**kwargs).fit(x, y)
    explicit_none = MasaClassifier(**kwargs, stop_epoch=None).fit(x, y)
    explicit_count = MasaClassifier(**kwargs, stop_epoch=3).fit(x, y)
    np.testing.assert_array_equal(default.predict_proba(x), explicit_none.predict_proba(x))
    np.testing.assert_array_equal(default.predict_proba(x), explicit_count.predict_proba(x))
    assert default.best_iteration_ is explicit_none.best_iteration_ is None
    assert explicit_count.best_iteration_ == 2


def test_remote_invalid_stop_is_rejected_before_preprocessing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid stop_epoch reached preprocessing")

    monkeypatch.setattr("masamlp.sklearn.TabularPreprocessor.fit", forbidden)
    x, y = _tiny_classification()
    with pytest.raises(ValueError, match="stop_epoch"):
        MasaClassifier(n_epochs=6, stop_epoch=7).fit(x, y)


@pytest.mark.parametrize("mode", ["loop", "vectorized"])
def test_remote_zero_epoch_initialization_remains_supported(mode, monkeypatch):
    x, y = _tiny_classification()

    def forbidden_step(*args, **kwargs):
        pytest.fail("Zero epochs must not take an optimizer step")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    model = MasaClassifier(
        model="realmlp", model_params={"hidden_sizes": [8]},
        numeric_scaler="standard", n_epochs=0, n_ens=2, ens_mode=mode,
        device="cpu", amp=False, n_threads=1, random_state=19,
    ).fit(x, y)
    assert model.best_iteration_ is None
    assert np.isfinite(model.predict_proba(x)).all()
