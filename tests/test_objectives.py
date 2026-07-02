import numpy as np
import pytest
import torch

from masamlp.core.objectives import get_objective, make_objective


def test_squared_error_values_and_bias():
    obj = get_objective("squared_error")
    y = obj.prepare_target(np.array([1.0, 3.0]))
    raw = torch.tensor([[2.0], [3.0]])
    assert torch.allclose(obj.per_sample_loss(y, raw), torch.tensor([1.0, 0.0]))
    bias = obj.init_bias(np.array([1.0, 3.0]), np.array([3.0, 1.0]))
    assert bias == pytest.approx([1.5])


def test_mae_weighted_median_bias():
    obj = get_objective("mae")
    bias = obj.init_bias(np.array([0.0, 1.0, 10.0]), np.array([1.0, 1.0, 5.0]))
    assert bias == pytest.approx([10.0])


def test_huber_matches_quadratic_and_linear_regimes():
    obj = get_objective("huber", delta=1.0)
    y = obj.prepare_target(np.array([0.0, 0.0]))
    raw = torch.tensor([[0.5], [3.0]])
    expected = torch.tensor([0.5 * 0.25, 1.0 * (3.0 - 0.5)])
    assert torch.allclose(obj.per_sample_loss(y, raw), expected)


def test_quantile_pinball():
    obj = get_objective("quantile", alpha=0.9)
    y = obj.prepare_target(np.array([1.0, 1.0]))
    raw = torch.tensor([[0.0], [2.0]])  # under-, over-prediction
    assert torch.allclose(obj.per_sample_loss(y, raw), torch.tensor([0.9, 0.1]))


def test_poisson_rejects_negative_targets():
    with pytest.raises(ValueError, match="non-negative"):
        get_objective("poisson").prepare_target(np.array([-1.0]))


def test_binary_logistic_bias_is_logit_of_prior():
    obj = get_objective("binary_logistic")
    bias = obj.init_bias(np.array([1, 1, 1, 0]), None)
    p = 0.75
    assert bias == pytest.approx([np.log(p / (1 - p))])


def test_multiclass_softmax_loss_and_bias():
    obj = get_objective("multiclass_softmax")
    y = np.array([0, 1, 1, 2])
    raw = torch.zeros(4, 3)
    loss = obj.per_sample_loss(obj.prepare_target(y), raw)
    assert torch.allclose(loss, torch.full((4,), np.log(3.0)))
    bias = obj.init_bias(y, None)
    assert bias == pytest.approx(np.log([0.25, 0.5, 0.25]), abs=1e-6)


def test_multioutput_squared_error():
    obj = get_objective("squared_error")
    y2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    raw = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    assert obj.out_dim(y2) == 2
    assert torch.allclose(obj.per_sample_loss(obj.prepare_target(y2), raw),
                          torch.tensor([1.0, 0.0]))


def test_custom_objective_shape_check():
    obj = make_objective(lambda y, raw: ((raw - y) ** 2).mean(), name="bad")
    y = obj.prepare_target(np.array([1.0]))
    with pytest.raises(ValueError, match="per-sample"):
        obj.per_sample_loss(y, torch.zeros(1, 1))


def test_objective_aliases():
    assert get_objective("l2").name == "squared_error"
    assert get_objective("l1").name == "mae"
    with pytest.raises(ValueError, match="Unknown objective"):
        get_objective("nope")
