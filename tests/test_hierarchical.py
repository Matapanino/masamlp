"""WP3 invariants: final-parameter shrinkage, routing and persistence."""
import copy

import numpy as np
import pytest
import torch

from masamlp.core.objectives import SquaredError
from masamlp.core.trainer import weighted_loss
from masamlp.models import build_model

HIERARCHIES = [{"feature": 0, "levels": [
    {"boundaries": [0.5], "counts": [4, 1]},
    {"values": [0., 1.], "counts": [4, 1]},
]}]


def make(shrinkage=2., hierarchies=None):
    return build_model("hierarchical_realmlp", {
        "hierarchies": copy.deepcopy(HIERARCHIES if hierarchies is None else hierarchies),
        "shrinkage": shrinkage, "support_power": 1., "d_num_embedding": 4,
        "num_scaling": True, "backbone_params": {"hidden_sizes": [8],
                                                   "zero_init_output": False},
    }, 2, [], 1, num_embedding="pbld")


def inputs():
    x = torch.tensor([[0., 2.], [1., 3.], [.25, 4.], [2., 5.]])
    return x, torch.empty(4, 0, dtype=torch.long)


def test_zero_deviation_is_byte_exact_bank_identity_and_no_rng_draw():
    torch.manual_seed(17)
    base = build_model("realmlp", {"hidden_sizes": [8], "zero_init_output": False,
                                   "d_num_embedding": 4, "num_scaling": True},
                       2, [], 1, num_embedding="pbld")
    after = torch.random.get_rng_state()
    torch.manual_seed(17)
    model = make()
    assert torch.equal(after, torch.random.get_rng_state())
    torch.testing.assert_close(model(*inputs()), base(*inputs()), rtol=0, atol=0)
    for table in model.embedding.residuals:
        assert torch.count_nonzero(table.weight) == 0


def test_unseen_leaf_uses_supported_parent_and_missing_uses_smooth_only():
    model = make()
    x, cat = inputs()
    with torch.no_grad():
        model.embedding.residuals[0].weight.copy_(torch.tensor([[1.]*4, [2.]*4]))
        model.embedding.residuals[1].weight.copy_(torch.tensor([[10.]*4, [20.]*4]))
    delta = model.embedding(x, cat) - model.embedding.base(x, cat)
    torch.testing.assert_close(delta[:, :4], torch.tensor([[11.]*4, [22.]*4, [1.]*4, [2.]*4]))
    assert torch.count_nonzero(delta[:, 4:]) == 0
    table = model.embedding.residuals[1]
    assert torch.count_nonzero(table(torch.tensor([float('nan'), float('inf'), -3.]))) == 0


def test_unobserved_parent_backoff_and_lookup_before_learned_scaling():
    h = copy.deepcopy(HIERARCHIES)
    h[0]["levels"][0]["counts"] = [0, 5]
    model = make(hierarchies=h)
    with torch.no_grad():
        for t in model.embedding.residuals:
            t.weight.fill_(3.)
        model.embedding.scaling.scale.fill_(7.)
    x, cat = inputs()
    delta = model.embedding(x, cat) - model.embedding.base(x, cat)
    torch.testing.assert_close(delta[:, :4], torch.tensor([[3.]*4, [6.]*4, [0.]*4, [3.]*4]))


@pytest.mark.parametrize("n, weight_scale", [(1, 1.), (7, 31.), (19, .2)])
def test_full_table_penalty_gradient_independent_of_batch_frequency(n, weight_scale):
    model = make()
    with torch.no_grad():
        for t in model.embedding.residuals:
            t.weight.fill_(2.)
    raw = torch.zeros(n, 1, requires_grad=True)
    terms = model.training_terms(None, raw)
    loss = weighted_loss(SquaredError(), raw.detach(), raw, torch.ones(n)*weight_scale, terms=terms)
    # Full table precision is 1/count; N = all 16 supported table coordinates.
    torch.testing.assert_close(loss, torch.tensor(5.))
    loss.backward()
    for t in model.embedding.residuals:
        torch.testing.assert_close(t.weight.grad, torch.tensor([[.125]*4, [.5]*4]))


def test_free_table_has_no_penalty_and_no_implicit_weight_decay():
    model = make(shrinkage=0.)
    assert not model.training_terms(None, torch.zeros(1, 1)).regularizers
    tables = {id(t.weight) for t in model.embedding.residuals}
    groups = model.param_groups()
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    for g in groups:
        if any(id(p) in tables for p in g["params"]):
            assert g["wd_factor"] == 0.


@pytest.mark.parametrize("bad", [
    [{"feature": 2, "levels": HIERARCHIES[0]["levels"]}],
    [{"feature": 0, "levels": [{"values": [1., 0.], "counts": [1, 1]}]}],
    [{"feature": 0, "levels": [{"values": [0.], "counts": [-1]}]}],
    [{"feature": 0, "levels": [{"values": [0.], "counts": [1]},
                                {"boundaries": [.5], "counts": [1, 1]}]}],
])
def test_invalid_hierarchy_rejected(bad):
    with pytest.raises(ValueError):
        make(hierarchies=bad)


def test_directory_save_load_restores_learned_tables_and_backoff(tmp_path):
    from masamlp import MasaClassifier

    x = np.tile([[0., 2.], [1., 3.]], (32, 1)).astype(np.float32)
    y = np.tile([0, 1], 32)
    model = MasaClassifier(model="hierarchical_realmlp", num_embedding="pbld",
                           numeric_scaler="none", n_epochs=2, device="cpu", random_state=5,
                           model_params={"hierarchies": HIERARCHIES, "shrinkage": .1,
                                         "backbone_params": {"hidden_sizes": [8]}}).fit(x, y)
    assert any(torch.count_nonzero(t.weight) for t in model.model_.embedding.residuals)
    query = np.array([[.25, 2.], [1., 3.], [8., 4.]], dtype=np.float32)
    model.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    np.testing.assert_array_equal(model.predict_proba(query), loaded.predict_proba(query))
    for k, v in model.model_.state_dict().items():
        torch.testing.assert_close(v, loaded.model_.state_dict()[k], rtol=0, atol=0)
