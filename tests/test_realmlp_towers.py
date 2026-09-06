"""Full-depth additive tower groups: parallel trunks over disjoint embedding
chunks, separated through every hidden layer, summed by one packed head."""
import math

import numpy as np
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.models import build_model
from masamlp.models.base import FeatureEmbedding
from masamlp.models.realmlp import NTPLinear, RealMLPNet, ScheduledDropout


def make(towers=None, n_num=4, cats=(), num_embedding=None, **kwargs):
    params = {'hidden_sizes': [8, 6], 'zero_init_output': False, 'tower_groups': towers}
    params.update(kwargs)
    return build_model('realmlp', params, n_num, list(cats), 1, num_embedding)


def tower_logit(model, g, features):
    """The packed head's term for tower ``g``: ``h_g @ W_g / sqrt(width)``."""
    d_in = model.output_layer.in_features
    width = d_in // len(model.towers)
    weight = model.output_layer.weight[g * width:(g + 1) * width]
    return math.sqrt(len(model.towers)) * (features @ weight) / math.sqrt(d_in)


def group_signature(model):
    return [
        (g['lr_factor'], g.get('wd_factor'), [tuple(p.shape) for p in g['params']])
        for g in model.param_groups()
    ]


def record_layers(model):
    """Forward hooks on every module of every tower trunk."""
    seen, handles = {}, []
    for g, tower in enumerate(model.towers):
        for i, module in enumerate(tower.trunk):
            def hook(_m, _args, out, key=(g, i)):
                seen[key] = out.detach().clone()
            handles.append(module.register_forward_hook(hook))
    return seen, handles


# --- T1 -------------------------------------------------------------------

def test_tower_none_and_single_group_are_exact_dense():
    kw = dict(hidden_sizes=[8, 6], zero_init_output=False, dropout=0.25,
              dropout_schedule='flat_cos', use_parametric_act=True,
              scale_position='first_layer', num_scaling=True, init_mode='std+he5')
    x, cat = torch.randn(64, 4), torch.randint(0, 3, (64, 1))

    torch.manual_seed(17)
    dense = build_model('realmlp', dict(kw), 4, [3], 1, 'pbld')
    dense_rng = torch.random.get_rng_state()
    torch.manual_seed(5)
    dense.data_init(x, cat)
    dense_init_rng = torch.random.get_rng_state()
    torch.manual_seed(9)
    dense_train_out = dense(x, cat)      # train mode: seeded dropout draws
    dense.eval()
    dense_eval_out = dense(x, cat)

    # None, the full group in order, and the full group permuted.
    for groups in (None, [[0, 1, 2, 3, 4]], [[4, 2, 0, 3, 1]]):
        torch.manual_seed(17)
        model = build_model('realmlp', {**kw, 'tower_groups': groups}, 4, [3], 1, 'pbld')
        assert torch.equal(dense_rng, torch.random.get_rng_state())
        assert model.towers is None
        assert list(model.state_dict()) == list(dense.state_dict())
        for key, value in dense.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        assert group_signature(model) == group_signature(dense)
        torch.manual_seed(5)
        model.data_init(x, cat)
        assert torch.equal(dense_init_rng, torch.random.get_rng_state())
        for key, value in dense.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        torch.manual_seed(9)
        torch.testing.assert_close(model(x, cat), dense_train_out, atol=0, rtol=0)
        model.eval()
        torch.testing.assert_close(model(x, cat), dense_eval_out, atol=0, rtol=0)


# --- T2 -------------------------------------------------------------------

def test_each_hidden_layer_is_source_isolated():
    torch.manual_seed(0)
    model = make([[0, 1], [2, 3]]).eval()
    assert torch.count_nonzero(model.output_layer.weight) == model.output_layer.weight.numel()
    x = torch.randn(6, 4)
    cat = torch.empty(6, 0, dtype=torch.long)
    seen, handles = record_layers(model)

    def run(inputs):
        out = model(inputs, cat)
        activations = dict(seen)
        embedded = model.embedding(inputs, cat)
        logits = [tower_logit(model, g, tower(embedded)) for g, tower in enumerate(model.towers)]
        return out, activations, logits

    for source, other in ((1, 0), (0, 1)):
        base_out, base_act, base_logits = run(x)
        moved = x.clone()
        moved[:, 2 * source:2 * source + 2] += 3.0
        out, act, logits = run(moved)
        for (g, i), value in base_act.items():
            if g == other:
                torch.testing.assert_close(act[(g, i)], value, atol=0, rtol=0)
            else:
                assert not torch.equal(act[(g, i)], value)
        torch.testing.assert_close(logits[other], base_logits[other], atol=0, rtol=0)
        assert not torch.equal(logits[source], base_logits[source])
        assert not torch.equal(out, base_out)
        # the head is exactly the sum of the two tower terms plus one bias
        torch.testing.assert_close(
            out, logits[0] + logits[1] + model.output_layer.bias, atol=1e-5, rtol=1e-5
        )
    for handle in handles:
        handle.remove()


# --- T3 -------------------------------------------------------------------

def test_branch_jacobians_and_parameter_gradients_are_isolated():
    torch.manual_seed(1)
    model = make([[0, 1], [2, 3]], num_embedding='pbld', d_num_embedding=3,
                 num_scaling=True).eval()
    x = torch.randn(5, 4, requires_grad=True)
    cat = torch.empty(5, 0, dtype=torch.long)
    embedded = model.embedding(x, cat)
    tower_logit(model, 0, model.towers[0](embedded)).sum().backward()

    for param in model.towers[1].parameters():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0
    assert all(torch.count_nonzero(p.grad) > 0 for p in model.towers[0].parameters())
    width = model.output_layer.in_features // 2
    assert torch.count_nonzero(model.output_layer.weight.grad[width:]) == 0
    assert torch.count_nonzero(model.output_layer.weight.grad[:width]) > 0
    assert model.output_layer.bias.grad is None or torch.count_nonzero(
        model.output_layer.bias.grad) == 0
    for param in (model.embedding.num_embedding.frequencies,
                  model.embedding.num_embedding.cos_bias_param,
                  model.embedding.num_embedding.weight,
                  model.embedding.num_embedding.bias):
        assert torch.count_nonzero(param.grad[2:]) == 0
        assert torch.count_nonzero(param.grad[:2]) > 0
    assert torch.count_nonzero(model.embedding.scaling.scale.grad[2:]) == 0
    assert torch.count_nonzero(model.embedding.scaling.scale.grad[:2]) > 0
    assert torch.count_nonzero(x.grad[:, 2:]) == 0
    assert torch.count_nonzero(x.grad[:, :2]) > 0

    # ... and the mirror image: the second branch ignores the first's inputs.
    model.zero_grad(set_to_none=True)
    x = torch.randn(5, 4, requires_grad=True)
    tower_logit(model, 1, model.towers[1](model.embedding(x, cat))).sum().backward()
    assert torch.count_nonzero(x.grad[:, :2]) == 0
    assert torch.count_nonzero(x.grad[:, 2:]) > 0
    for param in model.towers[0].parameters():
        assert param.grad is None or torch.count_nonzero(param.grad) == 0


# --- T4 -------------------------------------------------------------------

def budget_model(hidden, **kwargs):
    embedding = FeatureEmbedding(n_num=50, cat_cardinalities=[], num_embedding='pbld',
                                 d_num_embedding=8, n_frequencies=16, num_scaling=True)
    return RealMLPNet(embedding, 1, hidden_sizes=hidden, activation='selu',
                      use_parametric_act=True, zero_init_output=True,
                      scale_position='input', **kwargs)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_budget_and_optimizer_partition():
    split = [list(range(8)), list(range(8, 50))]
    dense = budget_model((384, 384, 384))
    assert dense.embedding.d_out == 400
    assert n_params(dense.embedding) == 7600
    assert n_params(dense) == 458801

    towers = budget_model((288, 288, 288), tower_groups=split)
    assert n_params(towers) == 458609
    first = [tower.trunk[0].weight.numel() for tower in towers.towers]
    assert first == [64 * 288, 336 * 288] and sum(first) == 115200

    grouped = budget_model((384, 384, 384), first_layer_groups=split)
    assert n_params(grouped) == 458801

    priced = budget_model((16, 16), tower_groups=split, first_layer_lr_factor=4.0)
    groups = priced.param_groups()
    params = [p for g in groups for p in g['params']]
    assert len(params) == len({id(p) for p in params})
    assert len(params) == sum(1 for p in priced.parameters() if p.requires_grad)
    assert all(g['params'] for g in groups)
    first_group = next(g for g in groups if g['lr_factor'] == 4.0)
    assert {id(p) for p in first_group['params']} == {
        id(tower.trunk[0].weight) for tower in priced.towers
    }
    biases = {id(m.bias) for m in priced.modules() if isinstance(m, NTPLinear)}
    undecayed = set()
    for group in groups:
        if group.get('wd_factor') == 0.0:
            undecayed |= {id(p) for p in group['params']}
        else:
            assert not any(id(p) in biases for p in group['params'])
    assert biases <= undecayed
    assert priced.output_layer.bias.numel() == 1
    assert sum(isinstance(m, NTPLinear) for m in priced.modules()) == 5


# --- T5 -------------------------------------------------------------------

def test_data_init_walks_both_towers():
    model = make([[0, 1], [2, 3]], init_mode='std+he5', zero_init_output=True,
                 hidden_sizes=[8, 6])
    x = torch.randn(512, 4)
    cat = torch.empty(512, 0, dtype=torch.long)
    model.data_init(x, cat)
    assert torch.count_nonzero(model.output_layer.weight) == 0
    assert torch.count_nonzero(model.output_layer.bias) == 0

    seen, handles = {}, []
    for g, tower in enumerate(model.towers):
        for i, module in enumerate(tower.trunk):
            if isinstance(module, NTPLinear):
                def hook(_m, args, key=(g, i)):
                    seen[key] = args[0].detach().clone()
                handles.append(module.register_forward_pre_hook(hook))
    model.eval()
    model(x, cat)
    for handle in handles:
        handle.remove()
    assert len(seen) == 4
    for (g, i), inputs in seen.items():
        layer = model.towers[g].trunk[i]
        pre = (inputs @ layer.weight) / math.sqrt(layer.in_features)
        torch.testing.assert_close(pre.std(0, correction=0),
                                   torch.ones(layer.weight.shape[1]), atol=1e-5, rtol=1e-5)

    activations, handles = record_layers(model)
    model(x, cat)
    before = dict(activations)
    moved = x.clone()
    moved[:, 2:] += 4.0
    model(moved, cat)
    for (g, i), value in before.items():
        if g == 0:
            torch.testing.assert_close(activations[(g, i)], value, atol=0, rtol=0)
    for handle in handles:
        handle.remove()


# --- T6 -------------------------------------------------------------------

def test_towers_directory_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(256, 4)).astype('float32')
    y = (x[:, 0] + x[:, 2] > 0).astype(int)
    groups = [[0, 1], [2, 3]]
    est = MasaClassifier(model='realmlp', device='cpu', n_epochs=2, random_state=0,
                         model_params={'hidden_sizes': [8, 8], 'tower_groups': groups})
    est.fit(x, y)
    before = est.predict_proba(x)
    assert est.resolved_model_params_['tower_groups'] == groups
    assert est.resolved_model_params_['hidden_sizes'] == [8, 8]
    est.save_model(tmp_path / 'model')
    loaded = MasaClassifier.load_model(tmp_path / 'model')
    np.testing.assert_array_equal(before, loaded.predict_proba(x))
    assert loaded.resolved_model_params_['tower_groups'] == groups
    assert len(loaded.model_.towers) == 2
    assert [t.trunk[0].in_features for t in loaded.model_.towers] == [2, 2]
    assert np.isfinite(before).all()


# --- T7 -------------------------------------------------------------------

def test_ema_covers_towers_embedding_and_head(monkeypatch):
    from masamlp.core import trainer as trainer_module

    original = trainer_module._update_ema
    steps = []

    def spy(model, ema_params, decay):
        before = {k: v.detach().clone() for k, v in ema_params.items()}
        original(model, ema_params, decay)
        steps.append((before, {k: v.detach().clone() for k, v in model.named_parameters()},
                      {k: v.detach().clone() for k, v in ema_params.items()}))

    monkeypatch.setattr(trainer_module, '_update_ema', spy)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype('float32')
    y = (x[:, 1] - x[:, 3] > 0).astype(int)
    decay = 0.8
    est = MasaClassifier(
        model='realmlp', device='cpu', n_epochs=4, batch_size=None, random_state=0,
        ema_decay=decay, num_embedding='pbld',
        model_params={'hidden_sizes': [8, 6], 'tower_groups': [[0, 1], [2, 3]],
                      'num_scaling': True, 'd_num_embedding': 3},
    ).fit(x, y)

    assert len(steps) == 4
    names = set(steps[0][0])
    assert {'towers.0.trunk.0.weight', 'towers.1.trunk.0.weight',
            'output_layer.weight', 'output_layer.bias',
            'embedding.scaling.scale', 'embedding.num_embedding.frequencies'} <= names
    for before, live, after in steps:
        for name in names:
            expected = before[name] * decay + live[name] * (1.0 - decay)
            torch.testing.assert_close(after[name], expected, atol=1e-6, rtol=1e-6)
    final = dict(est.model_.named_parameters())
    assert set(final) == names
    for name, value in steps[-1][2].items():
        torch.testing.assert_close(final[name].detach(), value, atol=0, rtol=0)


# --- T8 -------------------------------------------------------------------

@pytest.mark.parametrize('groups', [
    [], [[]], [[0, 1], []], [[0, 1]], [[0, 1], [1, 2, 3]], [[0, 1], [2, 4]],
    [[0, 1], [2, -1]], [[0, 1], [2, True]], [[0, 1], [2, 3.0]], [[0, 1, 2], [3], [3]],
])
def test_bad_tower_partitions_raise(groups):
    with pytest.raises(ValueError, match='tower_groups must partition'):
        make(groups)


@pytest.mark.parametrize('hidden', [[], [0], [8, 0]])
def test_towers_need_positive_hidden_sizes(hidden):
    with pytest.raises(ValueError, match='tower_groups requires'):
        make([[0, 1], [2, 3]], hidden_sizes=hidden)


def test_towers_and_first_layer_groups_are_mutually_exclusive():
    with pytest.raises(ValueError, match='tower_groups and first_layer_groups'):
        make([[0, 1], [2, 3]], first_layer_groups=[[0, 1], [2, 3]])


def test_tower_validation_draws_no_random_numbers():
    torch.manual_seed(2)
    state = torch.random.get_rng_state()
    with pytest.raises(ValueError, match='tower_groups must partition'):
        make([[0, 1], [2, 4]])
    assert torch.equal(state, torch.random.get_rng_state())


# --- T10 ------------------------------------------------------------------

def test_chunk_integrity_and_schedule():
    model = make([[0, 2], [1]], n_num=2, cats=(4,), num_embedding='pbld',
                 d_num_embedding=3, cat_emb_dim=2, dropout=0.15,
                 dropout_schedule='flat_cos')
    assert model.embedding.feature_chunk_sizes == [3, 3, 2]
    assert model.towers[0].coordinates.tolist() == [0, 1, 2, 6, 7]
    assert model.towers[1].coordinates.tolist() == [3, 4, 5]
    assert model.towers[0].coordinates.dtype == torch.int64
    assert [t.trunk[0].in_features for t in model.towers] == [5, 3]
    assert 'towers.0.coordinates' in model.state_dict()

    calls = []
    model.embedding.register_forward_hook(lambda *_: calls.append(1))
    model.eval()
    model(torch.randn(4, 2), torch.randint(0, 4, (4, 1)))
    assert len(calls) == 1

    drops = [m for m in model.modules() if isinstance(m, ScheduledDropout)]
    assert len(drops) == 4
    assert all(any(isinstance(m, ScheduledDropout) for m in t.trunk) for t in model.towers)
    model.set_schedule_t(1.0)
    assert all(float(d._keep) == pytest.approx(1.0, abs=1e-12) for d in drops)
    model.set_schedule_t(0.0)
    assert all(float(d._keep) == pytest.approx(0.85, abs=1e-6) for d in drops)
