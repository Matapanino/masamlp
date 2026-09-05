"""Source-group first-layer invariants, without dataset-specific routing."""
import numpy as np
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.models import build_model


def make(groups=None, **kwargs):
    return build_model('realmlp', {'hidden_sizes': [8], 'zero_init_output': False,
        'first_layer_groups': groups, **kwargs}, 4, [], 1)


def test_none_and_single_full_group_preserve_dense_rng_and_output():
    torch.manual_seed(17)
    dense = make()
    after = torch.random.get_rng_state()
    torch.manual_seed(17)
    grouped = make([[0, 1, 2, 3]])
    assert torch.equal(after, torch.random.get_rng_state())
    x = torch.randn(20, 4)
    cat = torch.empty(20, 0, dtype=torch.long)
    torch.testing.assert_close(grouped(x, cat), dense(x, cat), atol=0, rtol=0)


@pytest.mark.parametrize('groups', [[], [[]], [[0, 1]], [[0, 1], [1, 2, 3]],
    [[0, 1], [2, 4]], [[0, 1], [2, True]], [[0, 1], [2, 3.0]]])
def test_invalid_groups_raise(groups):
    with pytest.raises(ValueError, match='first_layer_groups'):
        make(groups)


def test_units_read_only_their_group_and_forbidden_gradients_stay_zero():
    model = make([[0, 1], [2, 3]])
    layer = model.trunk[0]
    x = torch.randn(10, 4, requires_grad=True)
    before = layer(x)
    modified = x.detach().clone()
    modified[:, 2:] += 100
    after = layer(modified)
    torch.testing.assert_close(before[:, :4], after[:, :4], atol=0, rtol=0)
    assert not torch.equal(before[:, 4:], after[:, 4:])
    before.sum().backward()
    assert torch.count_nonzero(layer.weight.grad[layer.group_mask == 0]) == 0
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01, weight_decay=0.1)
    optimizer.step()
    torch.testing.assert_close(layer(x)[:, :4], layer(modified)[:, :4], atol=0, rtol=0)


def test_data_init_respects_groups_and_scales_each_active_fan_in():
    model = make([[0], [1, 2, 3]], init_mode='std+he5')
    x = torch.randn(128, 4)
    model.data_init(x, torch.empty(128, 0, dtype=torch.long))
    layer = model.trunk[0]
    pre = layer(x) - layer.bias
    torch.testing.assert_close(pre.std(0, correction=0), torch.ones(8), atol=1e-5, rtol=1e-5)
    changed = x.clone()
    changed[:, 1:] += 2
    torch.testing.assert_close(layer(x)[:, :4], layer(changed)[:, :4], atol=0, rtol=0)


def test_feature_chunk_expansion_keeps_whole_embeddings():
    model = build_model('realmlp', {'hidden_sizes': [6], 'd_num_embedding': 3,
        'cat_emb_dim': 2, 'first_layer_groups': [[0, 2], [1]]}, 2, [4], 1,
        num_embedding='pbld')
    mask = model.trunk[0].group_mask
    assert mask.shape == (8, 6)
    assert torch.all(mask[:3, :3] == 1) and torch.all(mask[6:, :3] == 1)
    assert torch.count_nonzero(mask[3:6, :3]) == 0


def test_too_few_hidden_units_rejected():
    with pytest.raises(ValueError, match='first_layer_groups'):
        make([[0], [1], [2], [3]], hidden_sizes=[2])


def test_grouped_estimator_save_load_and_weighted_fit(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(128, 4)).astype('float32')
    y = (x[:, 0] + x[:, 2] > 0).astype(int)
    model = MasaClassifier(model='realmlp', device='cpu', n_epochs=2,
        model_params={'hidden_sizes': [8], 'first_layer_groups': [[0, 1], [2, 3]]})
    model.fit(x, y, sample_weight=np.linspace(0.1, 2, len(y)))
    before = model.predict_proba(x)
    model.save_model(tmp_path / 'model')
    after = MasaClassifier.load_model(tmp_path / 'model').predict_proba(x)
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(before).all()
