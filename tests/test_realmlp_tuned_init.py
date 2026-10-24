"""Opt-in tuned RealMLP controls, without claiming full pytabkit parity.

Tests named ``test_data_init_*`` execute data-driven initialization, and
``test_remote_*`` tests fit estimators; both must run on the remote CPU test
worker. The remaining tests only construct models or perform forward arithmetic.
"""

import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from masamlp.classifier import MasaClassifier
from masamlp.core import ensemble
from masamlp.core.trainer import Trainer
from masamlp.models import build_model
from masamlp.models.realmlp import (
    GroupedNTPLinear,
    NTPLinear,
    ParametricActivation,
    RealMLPNet,
    ScheduledDropout,
    _data_init_bias,
)

_LAYOUTS = {
    "dense": {},
    "grouped": {"first_layer_groups": [[0], [1, 2, 3]]},
    "towers": {"tower_groups": [[0], [1, 2, 3]]},
}


def _make(layout="dense", **kwargs):
    return build_model(
        "realmlp",
        {"hidden_sizes": [6, 4], "zero_init_output": False, **_LAYOUTS[layout], **kwargs},
        4, [], 2,
    )


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("parametric", [False, True])
def test_silu_forward_in_every_layout(layout, parametric):
    model = _make(layout, activation="silu", use_parametric_act=parametric)
    activations = [m for m in model.modules() if isinstance(m, (nn.SiLU, ParametricActivation))]
    assert len(activations) == (4 if layout == "towers" else 2)
    for act in activations:
        width = act.alpha.numel() if parametric else 4
        x = torch.linspace(-3, 3, 3 * width).reshape(3, width)
        if parametric:
            with torch.no_grad():
                act.alpha.copy_(torch.linspace(0, 2, width))
            expected = x + (x * torch.sigmoid(x) - x) * act.alpha
        else:
            expected = x * torch.sigmoid(x)
        torch.testing.assert_close(act(x), expected)
    model.eval()
    out = model(torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10,
                torch.empty(5, 0, dtype=torch.long))
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("mode", ["ntp", "std+he5", "std+negud2"])
@pytest.mark.parametrize("zero_output", [False, True])
def test_output_bias_policy_is_opt_in(layout, mode, zero_output):
    model = _make(layout, init_mode=mode, zero_init_output=zero_output)
    assert model.preserve_output_bias_init == (mode == "std+negud2" and not zero_output)
    assert model.needs_data_init == (mode != "ntp")
    if zero_output:
        assert torch.count_nonzero(model.output_layer.weight) == 0
        assert torch.count_nonzero(model.output_layer.bias) == 0


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_constructor_rng_and_state_do_not_depend_on_init_mode(layout):
    torch.manual_seed(14)
    old = _make(layout, init_mode="std+he5")
    old_rng = torch.random.get_rng_state()
    torch.manual_seed(14)
    tuned = _make(layout, init_mode="std+negud2")
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert old.state_dict().keys() == tuned.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, tuned.state_dict()[key]), key


_DROPOUT_CASES = [
    ("flat_cos", 0.8, 0.5 * (1 + math.cos(math.pi * 0.6))),
    ("flat_anneal", 0.8, 0.500005),
    ("coslog4", 0.2, 0.5 - 0.5 * math.cos(2 * math.pi * math.log2(4))),
    ("cos_log_15", 0.1, 0.5 - 0.5 * math.cos(2 * math.pi * math.log2(2.5))),
    ("invsqrtp1e-3", 0.008, 1 / 3),
    ("sqrt_cos", 0.5, 0.05 / (math.sqrt(0.5) + 0.05)),
]


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("name,t,factor", _DROPOUT_CASES)
def test_dropout_schedule_is_applied_in_every_layout(layout, name, t, factor):
    model = _make(layout, dropout=0.2, dropout_schedule=name)
    modules = [m for m in model.modules() if isinstance(m, ScheduledDropout)]
    assert len(modules) == (4 if layout == "towers" else 2)
    model.set_schedule_t(t)
    for module in modules:
        assert module._keep.item() == pytest.approx(1 - 0.2 * factor)
        module.eval()
        x = torch.ones(2, 3)
        assert torch.equal(module(x), x)
    assert all("_keep" not in key for key in model.state_dict())


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_unscheduled_dropout_keeps_legacy_module_and_rng(layout):
    torch.manual_seed(21)
    default = _make(layout, dropout=0.2)
    rng = torch.random.get_rng_state()
    torch.manual_seed(21)
    explicit = _make(layout, dropout=0.2, dropout_schedule="none")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not any(isinstance(m, ScheduledDropout) for m in default.modules())
    assert sum(isinstance(m, nn.Dropout) for m in default.modules()) == (
        4 if layout == "towers" else 2
    )
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key]), key
    default.set_schedule_t(0.8)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("name,value", [
    ("activation", "swish"), ("init_mode", "std+negud"),
    ("dropout_schedule", "flat-cos"), ("dropout_schedule", None),
])
def test_invalid_option_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        _make(**{name: value})


@pytest.mark.parametrize("parametric", [False, True])
@pytest.mark.parametrize("name,value", [
    ("activation", "silu"), ("init_mode", "std+negud2"),
    ("dropout_schedule", "invsqrtp1e-3"),
])
def test_realm_rejects_realmlp_only_options(name, value, parametric):
    with pytest.raises(ValueError, match=name):
        build_model(
            "realm", {"hidden_sizes": [6], "k": 2,
                      "use_parametric_act": parametric, name: value}, 4, [], 2,
        )


@pytest.mark.parametrize("zero_output", [False, True])
def test_auxiliary_rejects_unimplemented_backbone_init_mode(zero_output):
    with pytest.raises(ValueError, match="auxiliary_ordinal.*init_mode='std\\+negud2'"):
        build_model(
            "auxiliary_ordinal", {
                "parent_num_idx": [0], "excluded_num_idx": [1],
                "parent_cat_idx": [], "excluded_cat_idx": [0],
                "auxiliary_target_cat_idx": 0, "auxiliary_class_order": [1, 2, 3],
                "backbone_params": {"hidden_sizes": [6], "init_mode": "std+negud2",
                                    "zero_init_output": zero_output},
            }, 2, [4], 1,
        )


def test_data_init_bias_matches_dynamic_two_formula():
    # Deliberately nonzero mean and non-unit variance distinguish dynamic-2
    # from the older dynamic formula, which multiplies the mean by the std.
    pre = torch.tensor([[1., 2., 9.], [5., 2., 3.], [9., 2., 0.]])
    draws = torch.tensor([0.2, 0.7, 0.9])
    means = torch.tensor([5., 2., 4.])
    stds = torch.tensor([math.sqrt(32 / 3), 0., math.sqrt(14)])
    expected = -means - math.sqrt(3) * stds * draws
    with patch("torch.rand", return_value=draws) as rand:
        actual = _data_init_bias(pre, "neg-uniform-dynamic-2")
    rand.assert_called_once_with(3, device=pre.device)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("grouped", [False, True])
def test_data_init_standard_weights_and_dynamic_bias(grouped):
    layer = GroupedNTPLinear(4, 3, [[0], [1, 2, 3]]) if grouped else NTPLinear(4, 3)
    x = torch.tensor([[1., 2., -1., 0.], [4., -3., 2., 1.],
                      [2., 0., 3., -2.], [-1., 3., 0., 4.]])
    w = torch.tensor([[1., -2., 0.5], [2., 1., -1.], [-1., 0.5, 2.], [0.5, 1., 1.]])
    draws = torch.tensor([0.1, 0.5, 0.9])
    pre = (x @ (w * layer.group_mask)) * layer.group_scale if grouped else (x @ w) / 2
    std = pre.std(0, correction=0)
    standardized = pre / std
    expected_bias = -standardized.mean(0) - math.sqrt(3) * draws
    with patch("torch.randn_like", return_value=w), patch("torch.rand", return_value=draws):
        output = layer.data_init_(x, "neg-uniform-dynamic-2")
    torch.testing.assert_close(layer.weight, w / std)
    torch.testing.assert_close(layer.bias, expected_bias)
    torch.testing.assert_close(output, standardized + expected_bias)
    torch.testing.assert_close(layer(x), output)


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("zero_output", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_data_init_tuned_mode_reaches_hidden_and_output_layers(layout, zero_output, training):
    model = _make(layout, init_mode="std+negud2", zero_init_output=zero_output,
                  activation="silu", use_parametric_act=True,
                  dropout=0.2, dropout_schedule="invsqrtp1e-3")
    model.train(training)
    x = torch.linspace(-2, 3, 80).reshape(20, 4)
    cat = torch.empty(20, 0, dtype=torch.long)
    with patch("masamlp.models.realmlp._data_init_bias", wraps=_data_init_bias) as bias_init:
        model.data_init(x, cat)
    assert model.training == training
    hidden_count = 4 if layout == "towers" else 2
    assert bias_init.call_count == hidden_count + int(not zero_output)
    assert all(call.args[1] == "neg-uniform-dynamic-2" for call in bias_init.call_args_list)
    if zero_output:
        assert torch.count_nonzero(model.output_layer.weight) == 0
        assert torch.count_nonzero(model.output_layer.bias) == 0
    model.eval()
    assert torch.isfinite(model(x, cat)).all()


def _legacy_layer_init(layer, x):
    """Frozen pre-change arithmetic, including he+5's exact RNG draw order."""
    with torch.no_grad():
        w = torch.randn_like(layer.weight)
        if isinstance(layer, GroupedNTPLinear):
            pre = (x @ (w * layer.group_mask)) * layer.group_scale
        else:
            pre = (x @ w) / math.sqrt(layer.in_features)
        std = pre.std(dim=0, correction=0, keepdim=True).clamp_min(1e-30)
        layer.weight.copy_(w / std)
        pre = pre / std
        n_rows, n_units = pre.shape
        idx = torch.randint(0, n_rows, (n_units, 5), device=pre.device)
        weights = torch.distributions.Exponential(1.).sample((n_units, 5)).to(pre.device)
        weights = weights / weights.sum(dim=1, keepdim=True)
        cols = torch.arange(n_units, device=pre.device)
        picked = torch.stack([pre[idx[:, i], cols] for i in range(5)], dim=1)
        layer.bias.copy_(-(picked * weights.to(pre.dtype)).sum(dim=1))
        return pre + layer.bias


@pytest.mark.parametrize("grouped", [False, True])
def test_data_init_legacy_he5_preserves_draws_and_parameters(grouped):
    layer = GroupedNTPLinear(4, 5, [[0], [1, 2, 3]]) if grouped else NTPLinear(4, 5)
    reference = copy.deepcopy(layer)
    x = torch.linspace(-2, 4, 100).reshape(25, 4)
    torch.manual_seed(345)
    expected = _legacy_layer_init(reference, x)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(345)
    actual = layer.data_init_(x)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    assert torch.equal(layer.weight, reference.weight)
    assert torch.equal(layer.bias, reference.bias)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["loop", "vectorized"])
def test_remote_tuned_output_bias_reaches_trainer_and_roundtrips(mode, monkeypatch, tmp_path):
    expected = {}
    observed = []
    original_init = RealMLPNet.data_init

    def capture_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        expected[id(model)] = model.output_layer.bias.detach().clone()

    def check_members(models):
        for model in models:
            torch.testing.assert_close(model.output_layer.bias, expected[id(model)], rtol=0, atol=0)
            assert model.preserve_output_bias_init
            # A balanced target's prior logit is zero: distinguish preserving
            # the tuned data bias from overwriting it with the class prior.
            assert torch.count_nonzero(model.output_layer.bias) > 0
            observed.append(id(model))

    monkeypatch.setattr(RealMLPNet, "data_init", capture_init)
    if mode == "loop":
        original_fit = Trainer.fit

        def fit(trainer, model, *args, **kwargs):
            check_members([model])
            return original_fit(trainer, model, *args, **kwargs)

        monkeypatch.setattr(Trainer, "fit", fit)
    else:
        original_fit = ensemble.fit_vectorized

        def fit(models, *args, **kwargs):
            check_members(models)
            return original_fit(models, *args, **kwargs)

        monkeypatch.setattr(ensemble, "fit_vectorized", fit)

    rng = np.random.default_rng(41)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = np.arange(40) % 2
    model = MasaClassifier(
        model="realmlp", model_params={"hidden_sizes": [6, 4], "activation": "silu",
                                      "init_mode": "std+negud2", "zero_init_output": False},
        n_ens=2, ens_mode=mode, n_epochs=3, stop_epoch=1,
        numeric_scaler="standard", device="cpu", amp=False, n_threads=1,
        learning_rate=0.001, random_state=41,
    ).fit(x, y)
    assert len(expected) == 2
    assert sorted(observed) == sorted(expected)
    path = tmp_path / "tuned_model"
    model.save_model(path)
    restored = MasaClassifier.load_model(path)
    assert all(member.preserve_output_bias_init for member in restored.models_)
    np.testing.assert_array_equal(model.predict_proba(x), restored.predict_proba(x))
