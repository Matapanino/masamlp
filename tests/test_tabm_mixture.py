"""Trainer-owned deployed-mean binary risk; tiny CPU contract checks."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from masamlp import MasaClassifier, MasaRegressor
from masamlp.core.objectives import BinaryLogistic, SquaredError, make_objective
from masamlp.core.trainer import weighted_loss
from masamlp.core.training_terms import ModelRegularizer, TrainingTerms
from masamlp.models.base import FeatureEmbedding
from masamlp.models.tabm import TabM


def make(**params):
    torch.manual_seed(17)
    return TabM(FeatureEmbedding(4, []), **(dict(
        out_dim=1, variant="full", k=3, d=8, n_blocks=1, dropout=0.2) | params))


def estimator(params=None, **kwargs):
    return MasaClassifier(**(dict(
        model="tabm", model_params=dict(
            variant="full", k=3, d=8, n_blocks=1, dropout=0.2) | (params or {}),
        numeric_scaler="none", n_epochs=3, batch_size=8, device="cpu",
        amp=False, random_state=17, n_threads=1) | kwargs))


def data():
    rng = np.random.default_rng(19)
    return rng.normal(size=(24, 4)), rng.uniform(size=24), rng.uniform(size=24)


def reference(raw, q, alpha):
    # Independent float64 probability-space oracle; random logits avoid saturation.
    p = raw.double().squeeze(-1).sigmoid()
    q = q.double()
    independent = -(q[:, None] * p.log() + (1 - q[:, None]) * (1 - p).log()).mean(1)
    mean = p.mean(1)
    mixture = -(q * mean.log() + (1 - q) * (1 - mean).log())
    return (1 - alpha) * independent + alpha * mixture


@pytest.mark.parametrize("shared", [True, False])
def test_alpha_zero_exact_legacy_steps_and_predictions(shared):
    X, y, w = data()
    a = estimator(share_training_batches=shared).fit(X, y, sample_weight=w)
    rng_a = torch.get_rng_state().clone()
    b = estimator({"mixture_alpha": 0.0}, share_training_batches=shared).fit(
        X, y, sample_weight=w)
    assert torch.equal(rng_a, torch.get_rng_state())
    assert a.model_.state_dict().keys() == b.model_.state_dict().keys()
    for key in a.model_.state_dict():
        assert torch.equal(a.model_.state_dict()[key], b.model_.state_dict()[key])
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    np.testing.assert_array_equal(a.predict_proba_members(X), b.predict_proba_members(X))
    assert not hasattr(b.model_, "training_terms")


def test_zero_loss_and_gradients_match_pre_option_flattened_formula():
    torch.manual_seed(29)
    raw = torch.randn(5, 3, 1, requires_grad=True)
    y, w = torch.rand(5), torch.rand(5)
    obj = BinaryLogistic()
    losses = obj.per_sample_loss(y.repeat_interleave(3), raw.reshape(-1, 1))
    weights = w.repeat_interleave(3)
    old = (losses * weights).sum() / weights.sum()
    new = weighted_loss(obj, y, raw, w, mixture_alpha=0)
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, raw)[0], torch.autograd.grad(new, raw)[0])


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_single_member_is_exact_noop(alpha):
    X, y, w = data()
    a = estimator({"k": 1}).fit(X, y, sample_weight=w)
    b = estimator({"k": 1, "mixture_alpha": alpha}).fit(X, y, sample_weight=w)
    for key in a.model_.state_dict():
        assert torch.equal(a.model_.state_dict()[key], b.model_.state_dict()[key])
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_loss_matches_probability_reference(alpha, soft, weighted):
    torch.manual_seed(21)
    raw = torch.randn(7, 4, 1)
    y = torch.rand(7) if soft else torch.randint(2, (7,)).float()
    w = torch.tensor([0., 1., 2., 0.5, 7., 0., 3.]) if weighted else None
    rows = reference(raw, y, alpha)
    expected = rows.mean() if w is None else (rows * w).sum() / w.sum()
    got = weighted_loss(BinaryLogistic(), y, raw, w, mixture_alpha=alpha)
    torch.testing.assert_close(got.double(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.4, 1.0])
def test_finite_difference_gradients(alpha):
    raw = torch.tensor([[[-1.2], [0.7]], [[2.1], [-0.3]]],
                       dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.2, 1.0], dtype=torch.float64)
    w = torch.tensor([3., 0.5], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda z: weighted_loss(BinaryLogistic(), q, z, w, mixture_alpha=alpha),
        (raw,), eps=1e-6, atol=1e-6, rtol=1e-4)
    rows = reference(raw, q, alpha)
    expected = (rows * w).sum() / w.sum()
    got = weighted_loss(BinaryLogistic(), q, raw, w, mixture_alpha=alpha)
    torch.testing.assert_close(torch.autograd.grad(got, raw)[0],
                               torch.autograd.grad(expected, raw)[0])


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_extreme_logits_finite_loss_and_gradients(alpha):
    raw = torch.tensor([[[1000.], [1001.]], [[-1000.], [-1001.]],
                        [[1000.], [-1000.]]], requires_grad=True)
    q = torch.tensor([0., 1., 0.3])
    loss = weighted_loss(BinaryLogistic(), q, raw, None, mixture_alpha=alpha)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[0].sum() > 0
    assert raw.grad[1].sum() < 0


def test_zero_weights_have_zero_risk_and_gradients():
    raw = torch.randn(3, 2, 1, requires_grad=True)
    q = torch.tensor([0., 0.3, 1.])
    loss = weighted_loss(BinaryLogistic(), q, raw, torch.zeros(3), mixture_alpha=0.5)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(raw.grad) == 0


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="mixture_alpha.*finite.*\\[0, 1\\]"):
        make(mixture_alpha=alpha)
    with pytest.raises(ValueError, match="mixture_alpha"):
        weighted_loss(BinaryLogistic(), torch.zeros(2), torch.zeros(2, 3, 1), None,
                      mixture_alpha=alpha)


def test_full_variant_and_binary_output_required():
    with pytest.raises(ValueError, match="mixture_alpha.*full"):
        make(variant="mini", mixture_alpha=0.5)
    with pytest.raises(ValueError, match="mixture_alpha.*binary"):
        make(out_dim=3, mixture_alpha=0.5)
    X, _, _ = data()
    with pytest.raises(ValueError, match="mixture_alpha.*binary"):
        estimator({"mixture_alpha": 0.5}).fit(X, np.arange(len(X)) % 3)


@pytest.mark.parametrize("k", [1, 3])
def test_independent_batches_rejected_before_training(k):
    X, y, _ = data()
    # n_epochs=0 ensures the guard is eager, including runs with no optimizer steps.
    with pytest.raises(ValueError, match="mixture_alpha.*share_training_batches=True"):
        estimator({"mixture_alpha": 0.5, "k": k}, n_epochs=0,
                  share_training_batches=False).fit(X, y)
    with pytest.raises(ValueError, match="mixture_alpha.*share_training_batches=True"):
        weighted_loss(BinaryLogistic(), torch.zeros(2, k), torch.zeros(2, k, 1),
                      None, member_batches=True, mixture_alpha=0.5)


@pytest.mark.parametrize("objective", [SquaredError(), make_objective(
    lambda y, raw: F.binary_cross_entropy_with_logits(raw[:, 0], y, reduction="none"),
    transform="sigmoid", out_dim=1)])
def test_non_builtin_binary_objectives_rejected(objective):
    X, y, _ = data()
    with pytest.raises(ValueError, match="mixture_alpha.*BinaryLogistic"):
        estimator({"mixture_alpha": 0.5}, objective=objective, n_epochs=0).fit(X, y)
    with pytest.raises(ValueError, match="mixture_alpha.*BinaryLogistic"):
        weighted_loss(objective, torch.zeros(2), torch.zeros(2, 3, 1), None,
                      mixture_alpha=0.5)


def test_regression_rejected():
    X, y, _ = data()
    with pytest.raises(ValueError, match="mixture_alpha.*BinaryLogistic"):
        MasaRegressor(model="tabm", model_params={"variant": "full", "mixture_alpha": 0.5,
                      "d": 8, "k": 2}, device="cpu", n_epochs=0).fit(X, y)


@pytest.mark.parametrize("shape", [(2, 1), (2, 3, 2)])
def test_reducer_requires_member_binary_logits(shape):
    with pytest.raises(ValueError, match="mixture_alpha.*\\(n, k, 1\\)"):
        weighted_loss(BinaryLogistic(), torch.zeros(2), torch.zeros(shape), None,
                      mixture_alpha=0.5)


def test_brier_and_regularizer_decomposition_with_smoothing():
    model = make(mixture_alpha=0.5, brier_coefficient=0.3)
    torch.manual_seed(25)
    raw = torch.randn(4, 3, 1, requires_grad=True)
    q, w = torch.tensor([0., 0.2, 0.9, 1.]), torch.tensor([1., 0., 3., 2.])
    obj = BinaryLogistic(label_smoothing=0.2)
    auxiliary = model.training_terms(SimpleNamespace(y=q), raw).auxiliary
    reg = raw.square().sum()
    terms = TrainingTerms(auxiliary, (ModelRegularizer(reg, normalizer=12, coefficient=0.1),))
    # Smoothing affects both BCE branches; Brier retains the original prepared q.
    rows = reference(raw, q * 0.8 + 0.1, 0.5)
    rows += 0.3 * (raw.double().squeeze(-1).sigmoid() - q[:, None]).square().mean(1)
    expected = (rows * w).sum() / w.sum() + 0.1 * reg.double() / 12
    got = weighted_loss(obj, q, raw, w, terms=terms, mixture_alpha=0.5)
    torch.testing.assert_close(got.double(), expected, atol=1e-6, rtol=0)
    got.backward()
    assert torch.isfinite(raw.grad).all()


def test_enabled_option_changes_training():
    X, y, w = data()
    a = estimator().fit(X, y, sample_weight=w)
    b = estimator({"mixture_alpha": 1.0}).fit(X, y, sample_weight=w)
    assert any(not torch.equal(a.model_.state_dict()[key], b.model_.state_dict()[key])
               for key in a.model_.state_dict())


@pytest.mark.parametrize("n_ens", [1, 2])
def test_structured_views_brier_roundtrip_and_inference(tmp_path, n_ens):
    X, y, w = data()
    params = dict(mixture_alpha=0.5, brier_coefficient=0.2, first_layer_groups=[0, 0, 1, -1],
                  member_feature_mask=[[True, True, True, True], [False, True, True, True],
                                       [True, False, True, True]])
    model = estimator(params, n_ens=n_ens).fit(X, y, sample_weight=w)
    model.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    assert loaded.model_params["mixture_alpha"] == 0.5
    assert loaded.resolved_model_params_["mixture_alpha"] == 0.5
    assert all(m.mixture_alpha == 0.5 for m in loaded.models_)
    np.testing.assert_array_equal(model.predict_proba(X), loaded.predict_proba(X))
    np.testing.assert_array_equal(model.predict_proba_members(X), loaded.predict_proba_members(X))
    members = model.predict_proba_members(X)
    np.testing.assert_allclose(model.predict_proba(X), members.mean(1), atol=1e-7)
    before = model.predict_proba(X)
    for m in model.models_:
        m.mixture_alpha = 0  # The option is never consulted by inference.
    np.testing.assert_array_equal(before, model.predict_proba(X))


@pytest.mark.parametrize("brier", [0., 0.2])
def test_outer_vectorized_rejected(brier):
    X, y, _ = data()
    with pytest.raises(ValueError, match="mixture_alpha.*vectorized.*loop"):
        estimator({"mixture_alpha": 0.5, "brier_coefficient": brier},
                  ens_mode="vectorized", n_ens=2).fit(X, y)


def test_zero_retains_vectorized_support_and_single_outer_member_is_allowed():
    X, y, _ = data()
    a = estimator({"mixture_alpha": 0}, ens_mode="vectorized", n_ens=2).fit(X, y)
    b = estimator({"mixture_alpha": 0.5}, ens_mode="vectorized", n_ens=1).fit(X, y)
    assert np.isfinite(a.predict_proba(X)).all()
    assert np.isfinite(b.predict_proba(X)).all()
