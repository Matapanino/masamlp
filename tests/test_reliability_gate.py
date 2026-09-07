"""Small synthetic contracts for RealMLP's optional estimator arbitration."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone

from masamlp import MasaClassifier, MasaRegressor
from masamlp.data.preprocessing import TabularPreprocessor
from masamlp.models import build_model
from masamlp.models.arbitration import ReliabilityGate


def config(**kw):
    return dict(estimator_idx=[0, 2, 3], reliability_idx=[4, 5], n_heads=2,
                hidden_size=8, **kw)


@pytest.mark.parametrize('keep', [False, True])
@pytest.mark.parametrize('mode', ['conditional', 'constant'])
def test_shapes_convexity_and_gradients(keep, mode):
    g = ReliabilityGate(7, **config(keep_estimators=keep, mode=mode))
    x = torch.randn(31, 7, requires_grad=True)
    w = g.weights(x)
    mixed = g.mix(x)
    assert w.shape == (31, 2, 3)
    assert torch.allclose(w.sum(-1), torch.ones(31, 2))
    z = x[:, [0, 2, 3]]
    assert (mixed >= z.min(1).values[:, None] - 1e-6).all()
    assert (mixed <= z.max(1).values[:, None] + 1e-6).all()
    assert g(x).shape == (31, 7 if keep else 4)
    g(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in g.parameters())


def test_constant_is_exact_linear_convex_map():
    g = ReliabilityGate(7, **config(mode='constant'))
    with torch.no_grad():
        g.logits.copy_(torch.tensor([[1., -2., 0.], [0., 1., 2.]]))
    x = torch.randn(43, 7)
    expected = x[:, [0, 2, 3]] @ g.logits.softmax(-1).T
    assert torch.allclose(g.mix(x), expected, atol=1e-7)
    y = x.clone()
    y[:, 4:6] = torch.randn(43, 2) * 100
    assert torch.equal(g(x), g(y))
    assert torch.equal(g.weights(x)[0], g.weights(x)[-1])


def test_joint_permutation_equivariance_and_conditioning():
    torch.manual_seed(8)
    g = ReliabilityGate(7, **config())
    x = torch.randn(43, 7)
    pi = torch.randperm(43)
    assert torch.allclose(g(x[pi]), g(x)[pi])
    y = x.clone()
    y[:, 4:6] = x[pi, 4:6]
    assert not torch.allclose(g.weights(x), g.weights(y))


@pytest.mark.parametrize('bad', [
    dict(estimator_idx=[]), dict(estimator_idx=[0, 0]), dict(estimator_idx=[9]),
    dict(reliability_idx=[0]), dict(reliability_idx=[]), dict(n_heads=0),
    dict(hidden_size=0), dict(mode='oops'), dict(estimator_idx=[0.5]),
])
def test_invalid_gate_config(bad):
    with pytest.raises((ValueError, TypeError)):
        ReliabilityGate(7, **{**config(), **bad})


@pytest.mark.parametrize('keep', [False, True])
def test_realmlp_build_data_init_param_groups_state(keep):
    torch.manual_seed(2)
    params = dict(hidden_sizes=[12], d_num_embedding=4, init_mode='std+he5',
                  arbitration=config(keep_estimators=keep))
    original = copy.deepcopy(params)
    m = build_model('realmlp', params, 7, [3], 1, 'pbld')
    assert params == original
    x, c = torch.randn(31, 7), torch.ones(31, 1, dtype=torch.long)
    m.data_init(x, c)
    m.eval()
    assert m(x, c).shape == (31, 1)
    ids = [id(p) for g in m.param_groups() for p in g['params']]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    replica = build_model('realmlp', params, 7, [3], 1, 'pbld')
    replica.load_state_dict(m.state_dict())
    replica.eval()
    assert torch.equal(m(x, c), replica(x, c))


def test_off_path_is_byte_identical():
    torch.manual_seed(7)
    a = build_model('realmlp', {'hidden_sizes': [12]}, 7, [], 1, 'pbld')
    torch.manual_seed(7)
    b = build_model('realmlp', {'hidden_sizes': [12], 'arbitration': None}, 7, [], 1, 'pbld')
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


@pytest.mark.parametrize('scaler', ['quantile', 'standard', 'robust', 'rssc', 'none'])
def test_passthrough_keeps_units_and_serializes(scaler):
    x = pd.DataFrame({'rest': [0., 2., 5., 8.], 'logit': [-4., -2., 0., 3.]})
    pre = TabularPreprocessor(scaler, numeric_passthrough_cols=['logit']).fit(x)
    a, _ = pre.transform(x)
    assert np.array_equal(a[:, 1], x.logit.to_numpy(dtype=np.float32))
    ref = TabularPreprocessor(scaler).fit_transform(x)[0]
    assert np.array_equal(a[:, 0], ref[:, 0])
    meta, arrays = pre.get_state()
    assert np.array_equal(a, TabularPreprocessor.from_state(meta, arrays).transform(x)[0])


@pytest.mark.parametrize('cols', [[], ['z'], ['x', 'x'], ['cat'], 'x'])
def test_passthrough_refuses_bad_names(cols):
    with pytest.raises(ValueError):
        TabularPreprocessor(numeric_passthrough_cols=cols).fit(
            pd.DataFrame({'x': [1., 2.], 'cat': ['a', 'b']}))


@pytest.mark.parametrize('cls', [MasaClassifier, MasaRegressor])
def test_estimator_exposes_passthrough(cls):
    m = cls(numeric_passthrough_cols=['a'])
    assert clone(m).numeric_passthrough_cols == ['a']


@pytest.mark.parametrize('mode', ['conditional', 'constant'])
def test_estimator_fit_ema_save_load(tmp_path, mode):
    rng = np.random.default_rng(12)
    x = pd.DataFrame(rng.normal(size=(64, 7)), columns=list('abcdefg'))
    y = (x.a + x.c > 0).astype(int)
    m = MasaClassifier(model='realmlp', numeric_scaler='rssc',
                       numeric_passthrough_cols=['a', 'c', 'd'],
                       model_params={'hidden_sizes': [12], 'arbitration': config(mode=mode)},
                       n_epochs=2, ema_decay=0.9, device='cpu', amp=False)
    m.fit(x, y)
    before = m.predict_proba(x)
    m.save_model(str(tmp_path / 'model'))
    loaded = MasaClassifier.load_model(str(tmp_path / 'model'))
    np.testing.assert_array_equal(before, loaded.predict_proba(x))
