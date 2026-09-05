"""WP4: routing, proper ordinal likelihood, controls and ordinary persistence."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.core.objectives import BinaryLogistic
from masamlp.core.trainer import weighted_loss
from masamlp.data.dataset import TabularData
from masamlp.models import build_model


def model(weight=.2, **kwargs):
    return build_model('auxiliary_ordinal', dict(
        parent_num_idx=[0, 1], parent_cat_idx=[0],
        excluded_num_idx=[2, 3], excluded_cat_idx=[1, 2],
        auxiliary_target_cat_idx=2, auxiliary_class_order=[1, 2, 3],
        auxiliary_weight=weight, latent_dim=3, auxiliary_hidden_sizes=[8],
        backbone_params={'hidden_sizes': [8], 'zero_init_output': False}, **kwargs),
        n_num=4, cat_cardinalities=[3, 4, 4], out_dim=1)


def batch():
    torch.manual_seed(14)
    return TabularData(torch.randn(12, 4), torch.tensor(
        [[1, 1, 1], [2, 2, 2], [1, 3, 3]] * 4),
        torch.tensor([0., 1.] * 6), torch.arange(12).float())


def test_routing_masks_observed_target_and_all_declared_descendants():
    m, b = model().eval(), batch()
    altered = copy.deepcopy(b)
    altered.x_num[:, 2:] += 1000
    altered.x_cat[:, 1:] = altered.x_cat.flip(0)[:, 1:]
    torch.testing.assert_close(m.auxiliary_latent(b.x_num, b.x_cat),
                               m.auxiliary_latent(altered.x_num, altered.x_cat), rtol=0, atol=0)
    assert not torch.equal(m(b.x_num, b.x_cat), m(altered.x_num, altered.x_cat))
    b.x_num.requires_grad_()
    m.auxiliary_log_probs(b.x_num, b.x_cat).sum().backward()
    assert torch.count_nonzero(b.x_num.grad[:, 2:]) == 0
    assert torch.count_nonzero(b.x_num.grad[:, :2]) > 0


def test_overlap_and_incomplete_routing_fail_closed():
    from masamlp.models.auxiliary import AuxiliaryOrdinalNet
    base = dict(embedding_config={'n_num': 2, 'cat_cardinalities': [4]}, out_dim=1,
                parent_num_idx=[0], parent_cat_idx=[], excluded_num_idx=[1],
                excluded_cat_idx=[0], auxiliary_target_cat_idx=0,
                auxiliary_class_order=[1, 2, 3])
    for change in [dict(parent_num_idx=[0, 1]), dict(excluded_num_idx=[]),
                   dict(parent_cat_idx=[0]), dict(auxiliary_class_order=[1, 1, 3]),
                   dict(out_dim=2), dict(auxiliary_weight=float('nan'))]:
        with pytest.raises(ValueError):
            AuxiliaryOrdinalNet(**{**base, **change})


def test_ordinal_likelihood_is_normalized_monotone_and_stable():
    m, b = model(), batch()
    lp = m.auxiliary_log_probs(b.x_num, b.x_cat)
    torch.testing.assert_close(lp.exp().sum(1), torch.ones(12))
    assert torch.all(m.cutpoints().diff() > 0)
    loss = m.training_terms(b, m(b.x_num, b.x_cat)).auxiliary[0]
    torch.testing.assert_close(loss.values, -lp[torch.arange(12), b.x_cat[:, 2] - 1])
    assert loss.coefficient == .2
    with torch.no_grad():
        m.ordinal_head.weight.fill_(1e4)
    extreme = m.auxiliary_log_probs(b.x_num * 100, b.x_cat)
    assert torch.isfinite(extreme).all()
    (-extreme.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_zero_weight_matches_no_hook_optimizer_trajectory():
    m, b = model(0), batch()
    control = copy.deepcopy(m)
    control.training_terms = lambda batch, raw: None
    for net in (m, control):
        opt = torch.optim.Adam(net.parameters(), lr=.01)
        for _ in range(4):
            opt.zero_grad()
            raw = net(b.x_num, b.x_cat)
            weighted_loss(BinaryLogistic(), b.y, raw, b.weight,
                          terms=net.training_terms(b, raw)).backward()
            opt.step()
    for key, val in m.state_dict().items():
        torch.testing.assert_close(val, control.state_dict()[key], rtol=0, atol=0)
    assert m.ordinal_head.weight.grad is None


def test_shuffled_targets_change_only_auxiliary_loss_and_gradient():
    m, b = model(), batch()
    permuted = copy.deepcopy(b)
    permuted.x_cat[:, 2] = b.x_cat.flip(0)[:, 2]
    raw = m(b.x_num, b.x_cat)
    torch.testing.assert_close(raw, m(permuted.x_num, permuted.x_cat), rtol=0, atol=0)
    a = m.training_terms(b, raw).auxiliary[0].values
    s = m.training_terms(permuted, raw).auxiliary[0].values
    assert not torch.equal(a, s)
    ga = torch.autograd.grad(a.sum(), m.ordinal_head.weight, retain_graph=True)[0]
    gs = torch.autograd.grad(s.sum(), m.ordinal_head.weight)[0]
    assert not torch.equal(ga, gs)


def test_optimizer_groups_cover_each_parameter_once():
    m = model()
    params = [p for g in m.param_groups() for p in g['params']]
    assert len({id(p) for p in params}) == len(params)
    assert {id(p) for p in params} == {id(p) for p in m.parameters()}


def test_binary_estimator_save_load_without_auxiliary_labels(tmp_path):
    x = pd.DataFrame({'p0': np.arange(36) / 36, 'p1': np.sin(np.arange(36)),
                      'desc0': np.zeros(36), 'desc1': np.ones(36),
                      'parent': ['a', 'b', 'a'] * 12,
                      'observed': ['L', 'M', 'H'] * 12,
                      'target': ['0', '1', '2'] * 12})
    m = model()
    params = dict(parent_num_idx=[0, 1], parent_cat_idx=[0],
                  excluded_num_idx=[2, 3], excluded_cat_idx=[1, 2],
                  auxiliary_target_cat_idx=2, auxiliary_class_order=[1, 2, 3],
                  backbone_params={'hidden_sizes': [8]}, auxiliary_hidden_sizes=[8])
    est = MasaClassifier(model='auxiliary_ordinal', model_params=params,
                         cat_encoding='embedding', device='cpu', n_epochs=2,
                         amp=False, random_state=1).fit(x, np.arange(36) % 2)
    pred = est.predict_proba(x)
    assert pred.shape == (36, 2)
    np.testing.assert_allclose(pred.sum(1), 1)
    assert np.all((pred >= 0) & (pred <= 1))
    est.save_model(str(tmp_path))
    loaded = MasaClassifier.load_model(str(tmp_path))
    x['target'] = None
    np.testing.assert_array_equal(pred, loaded.predict_proba(x))
    assert loaded.objective_ is None
    del m


def test_auxiliary_weights_and_missing_targets_have_zero_direct_gradient():
    m, b = model(), batch()
    b.x_cat[2, 2] = 0
    raw = m(b.x_num, b.x_cat)
    terms = m.training_terms(b, raw)
    losses = terms.auxiliary[0].values
    losses.retain_grad()
    loss = weighted_loss(BinaryLogistic(), b.y, raw, b.weight, terms=terms)
    expected = ((BinaryLogistic().per_sample_loss(b.y, raw) + .2 * losses)
                * b.weight).sum() / b.weight.sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert losses[2] == 0
    assert losses.grad[0] == 0
    torch.testing.assert_close(losses.grad, .2 * b.weight / b.weight.sum())


def test_pretraining_updates_latent_without_updating_ev_branch():
    from masamlp.core.objectives import BaseObjective
    from masamlp.core.trainer import Trainer, TrainerConfig

    class AuxiliaryOnly(BaseObjective):
        transform_name = 'sigmoid'

        def per_sample_loss(self, y_true, raw_pred):
            return raw_pred[:, 0].detach() * 0

    m, b = model(), batch()
    before = copy.deepcopy(m.state_dict())
    Trainer().fit(m, AuxiliaryOnly(), b, [], [], TrainerConfig(
        device='cpu', amp=False, n_epochs=2, batch_size=6, random_state=3))
    for key in before:
        if key.startswith('ev.'):
            torch.testing.assert_close(before[key], m.state_dict()[key], rtol=0, atol=0)
    assert not torch.equal(before['auxiliary.output_layer.weight'],
                           m.auxiliary.output_layer.weight)


def test_data_driven_init_and_two_level_ordinal():
    m = build_model('auxiliary_ordinal', dict(
        parent_num_idx=[0], parent_cat_idx=[], excluded_num_idx=[1], excluded_cat_idx=[0],
        auxiliary_target_cat_idx=0, auxiliary_class_order=[2, 1], latent_dim=2,
        backbone_params={'hidden_sizes': [8], 'init_mode': 'std+he5',
                         'zero_init_output': False}), 2, [3], 1)
    x, c = torch.randn(16, 2), torch.ones(16, 1, dtype=torch.long)
    m.data_init(x, c)
    assert m.training
    assert torch.isfinite(m(x, c)).all()
    torch.testing.assert_close(m.auxiliary_log_probs(x, c).exp().sum(1), torch.ones(16))
