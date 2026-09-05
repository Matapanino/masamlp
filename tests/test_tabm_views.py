"""Opt-in TabM full source structure, member views, and weighted Brier terms."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.core.objectives import BinaryLogistic
from masamlp.core.trainer import weighted_loss
from masamlp.models.base import FeatureEmbedding
from masamlp.models.tabm import TabM


def make(**kwargs):
    torch.manual_seed(17)
    return TabM(FeatureEmbedding(4, []), 1, variant="full", k=2,
                d=8, n_blocks=1, dropout=0, **kwargs)


def test_defaults_preserve_exact_parameters_and_predictions():
    a = make()
    b = make(first_layer_groups=None, member_feature_mask=None, brier_coefficient=0)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    x = torch.randn(5, 4)
    cat = torch.empty(5, 0, dtype=torch.long)
    assert torch.equal(a(x, cat), b(x, cat))
    assert not hasattr(b, "training_terms")


def test_grouped_first_layer_prevents_cross_group_derivatives():
    m = make(first_layer_groups=[0, 0, 1, -1])
    layer = m.backbone[0]
    x = torch.randn(3, 2, 4, requires_grad=True)
    z = layer(x)
    # Row 0 belongs to group 0: group-1 coordinate is invisible, shared -1 is visible.
    grad = torch.autograd.grad(z[..., 0].sum(), x)[0]
    assert torch.count_nonzero(grad[..., 2]) == 0
    assert torch.count_nonzero(grad[..., 3]) > 0
    # Row 1 belongs to group 1: both group-0 coordinates are invisible.
    grad = torch.autograd.grad(layer(x)[..., 1].sum(), x)[0]
    assert torch.count_nonzero(grad[..., :2]) == 0


@pytest.mark.parametrize("member_batches", [False, True])
def test_member_views_block_only_selected_member_coordinates(member_batches):
    m = make(member_feature_mask=[[True, False, True, True], [False, True, True, True]])
    x = torch.randn(6, 2, 4) if member_batches else torch.randn(6, 4)
    cats = torch.empty(*x.shape[:-1], 0, dtype=torch.long)
    before = m(x, cats)
    after_x = x.clone()
    after_x[..., 0] += 9
    after = m(after_x, cats)
    assert torch.equal(before[:, 1], after[:, 1])
    assert not torch.equal(before[:, 0], after[:, 0])


@pytest.mark.parametrize("member_batches", [False, True])
def test_brier_is_unreduced_and_soft_target_aligned(member_batches):
    m = make(brier_coefficient=0.25)
    raw = torch.randn(5, 2, 1, requires_grad=True)
    y = torch.rand(5, 2) if member_batches else torch.rand(5)
    term = m.training_terms(SimpleNamespace(y=y), raw).auxiliary[0]
    expected = (raw.squeeze(-1).sigmoid() - (y if member_batches else y[:, None])).square()
    assert term.values.shape == (5, 2)
    assert term.coefficient == 0.25
    assert torch.equal(term.values, expected)


def test_brier_uses_member_normalized_sample_weights():
    m = make(brier_coefficient=0.25)
    raw = torch.tensor([[[0.2], [0.3]], [[-0.4], [0.8]]], requires_grad=True)
    y = torch.tensor([[0.1, 0.7], [0.9, 0.3]])
    weight = torch.tensor([[1., 0.], [3., 0.]])
    terms = m.training_terms(SimpleNamespace(y=y), raw)
    got = weighted_loss(BinaryLogistic(), y, raw, weight, member_batches=True, terms=terms)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(raw.squeeze(-1), y, reduction="none")
    losses = bce + 0.25 * (raw.squeeze(-1).sigmoid() - y).square()
    expected = (losses[0, 0] + 3 * losses[1, 0]) / 4 / 2
    torch.testing.assert_close(got, expected)
    got.backward()
    assert torch.count_nonzero(raw.grad[:, 1]) == 0


@pytest.mark.parametrize("kwargs", [
    {"first_layer_groups": [0]}, {"first_layer_groups": [-1]*4},
    {"first_layer_groups": [0, 1, 2, -2]},
    {"member_feature_mask": [[True]*4]},
    {"member_feature_mask": [[False]*4]*2},
    {"brier_coefficient": -1}, {"brier_coefficient": float("nan")},
])
def test_invalid_views_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_enabled_options_roundtrip_with_soft_member_weights(tmp_path):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(160, 4))
    y = rng.uniform(0.05, 0.95, 160)
    weight = rng.uniform(0.1, 3, 160)
    m = MasaClassifier(model="tabm", n_epochs=2, random_state=4, device="cpu",
        share_training_batches=False, model_params=dict(variant="full", k=2, d=8,
            n_blocks=1, first_layer_groups=[0, 0, 1, -1],
            member_feature_mask=[[True, False, True, True], [False, True, True, True]],
            brier_coefficient=0.25)).fit(x, y, sample_weight=weight)
    before = m.predict_proba(x)
    m.save_model(tmp_path / "model")
    restored = MasaClassifier.load_model(tmp_path / "model")
    np.testing.assert_array_equal(before, restored.predict_proba(x))


def test_structural_options_keep_outer_vectorized_support():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 4))
    y = (x[:, 0] > 0).astype(int)
    model = MasaClassifier(model="tabm", n_epochs=2, n_ens=2, ens_mode="vectorized",
        device="cpu", random_state=1, amp=False, grad_clip=None,
        model_params=dict(variant="full", k=2, d=8, n_blocks=1,
                          first_layer_groups=[0, 0, 1, -1],
                          member_feature_mask=[[True]*4, [False, True, True, True]]))
    model.fit(x, y)
    assert np.isfinite(model.predict_proba(x)).all()
