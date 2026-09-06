"""Full-depth source isolation and additive RealMLP readouts (no accuracy claim)."""

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from masamlp import MasaClassifier
from masamlp.core import trainer
from masamlp.models.base import FeatureEmbedding
from masamlp.models.realmlp import NTPLinear, ParametricActivation, RealMLPNet, ScheduledDropout
from masamlp.utils.random import seed_everything

PARTITION = [[0, 1], [2, 3]]


@pytest.fixture(scope="module")
def inputs():
    # Reuse 96 synthetic CPU rows across the unit and estimator tests.
    x = torch.from_numpy(np.random.default_rng(19).normal(size=(96, 4)).astype("float32"))
    return x, torch.empty(len(x), 0, dtype=torch.long)


def make(**kwargs):
    embedding = FeatureEmbedding(
        4, [], num_embedding="pbld", d_num_embedding=4, n_frequencies=3, num_scaling=True
    )
    return RealMLPNet(embedding, 1, **{
        "hidden_sizes": [7, 5, 4], "activation": "selu", "use_parametric_act": True,
        "zero_init_output": False, "tower_groups": PARTITION, **kwargs,
    })


def assert_state_equal(left, right):
    assert list(left) == list(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def group_signature(model):
    names = {id(p): name for name, p in model.named_parameters()}
    return [({k: v for k, v in group.items() if k != "params"},
             [names[id(p)] for p in group["params"]]) for group in model.param_groups()]


def capture(model, x, cat):
    """Observe the real forward path; derive each readout without its shared bias."""
    activations = {}
    handles = []
    for g, tower in enumerate(model.towers):
        for i, layer in enumerate(tower.trunk):
            def record(module, args, output, key=(g, i)):
                activations[key] = output
            handles.append(layer.register_forward_hook(record))
    try:
        raw = model(x, cat)
    finally:
        for handle in handles:
            handle.remove()
    width = model.output_layer.in_features // len(model.towers)
    contributions = [
        activations[g, len(tower.trunk) - 1]
        @ model.output_layer.weight[g * width:(g + 1) * width] / math.sqrt(width)
        for g, tower in enumerate(model.towers)
    ]
    return raw, activations, contributions


@pytest.mark.parametrize("init_mode,scale_position,schedule,ff", [
    ("ntp", "input", "none", 1.0),
    ("ntp", "first_layer", "flat_cos", 3.0),
    ("std+he5", "input", "flat_cos", 1.0),
    ("std+he5", "first_layer", "none", 3.0),
])
def test_tower_none_and_single_group_are_exact_dense(inputs, init_mode, scale_position,
                                                    schedule, ff):
    x, cat = (t[:16] for t in inputs)
    kwargs = dict(hidden_sizes=[7, 5], activation="selu", dropout=0.25,
                  dropout_schedule=schedule, use_parametric_act=True, init_mode=init_mode,
                  zero_init_output=False, scale_position=scale_position,
                  first_layer_lr_factor=ff, linear_skip_idx=[0, 3])

    def build(extra):
        seed_everything(31)
        embedding = FeatureEmbedding(4, [], num_embedding="pbld", d_num_embedding=4,
                                     n_frequencies=3, num_scaling=True)
        model = RealMLPNet(embedding, 1, **kwargs, **extra)
        return model, torch.random.get_rng_state().clone()

    dense, after = build({})
    before = {k: v.clone() for k, v in dense.state_dict().items()}
    seed_everything(32)
    dense.data_init(x, cat)
    after_init = torch.random.get_rng_state().clone()
    seed_everything(33)
    expected = dense(x, cat)
    after_forward = torch.random.get_rng_state().clone()
    for groups in (None, [[0, 1, 2, 3]], [[3, 1, 0, 2]]):
        candidate, candidate_rng = build({"tower_groups": groups})
        assert torch.equal(candidate_rng, after)
        assert_state_equal(before, candidate.state_dict())
        assert group_signature(candidate) == group_signature(dense)
        assert getattr(candidate, "towers", None) is None
        seed_everything(32)
        candidate.data_init(x, cat)
        assert torch.equal(torch.random.get_rng_state(), after_init)
        assert_state_equal(dense.state_dict(), candidate.state_dict())
        seed_everything(33)
        assert torch.equal(candidate(x, cat), expected)
        assert torch.equal(torch.random.get_rng_state(), after_forward)


@pytest.mark.parametrize("activation,parametric,schedule", [
    ("mish", False, "none"), ("selu", True, "flat_cos"), ("relu", False, "none"),
])
def test_each_hidden_layer_is_source_isolated(inputs, activation, parametric, schedule):
    seed_everything(37)
    model = make(activation=activation, use_parametric_act=parametric,
                 dropout=0.2, dropout_schedule=schedule, linear_skip_idx=[0, 3]).eval()
    with torch.no_grad():
        model.skip_weight.fill_(0.3)
        model.skip_bias.fill_(0.2)
    x, cat = (t[:16] for t in inputs)
    raw, before, parts = capture(model, x, cat)
    assert torch.count_nonzero(model.output_layer.weight) > 0
    torch.testing.assert_close(raw, sum(parts) + model.output_layer.bias
                               + x[:, [0, 3]] @ model.skip_weight + model.skip_bias)
    for g, tower in enumerate(model.towers):
        expected_act = ParametricActivation if parametric else {
            "mish": nn.Mish, "selu": nn.SELU, "relu": nn.ReLU,
        }[activation]
        expected_drop = ScheduledDropout if schedule == "flat_cos" else nn.Dropout
        assert [type(layer) for layer in tower.trunk] == [
            NTPLinear, expected_act, expected_drop,
        ] * 3
        changed = x.clone()
        changed[:, PARTITION[1 - g]] += 5
        _, after, changed_parts = capture(model, changed, cat)
        for i in range(len(tower.trunk)):
            assert torch.equal(before[g, i], after[g, i]), (g, i)
        assert torch.equal(parts[g], changed_parts[g])
        assert not torch.equal(parts[1 - g], changed_parts[1 - g])


@pytest.mark.parametrize("scale_position", ["input", "first_layer"])
def test_branch_jacobians_and_parameter_gradients_are_isolated(inputs, scale_position):
    seed_everything(41)
    model = make(scale_position=scale_position).eval()
    x, cat = inputs
    x, cat = x[:3].clone().requires_grad_(), cat[:3]
    optimizer = torch.optim.AdamW(model.param_groups(), lr=0.01, weight_decay=0.1)
    for g in range(2):
        model.zero_grad(set_to_none=True)
        _, _, parts = capture(model, x, cat)
        for row in range(len(x)):
            jacobian = torch.autograd.grad(parts[g][row, 0], x, retain_graph=True)[0]
            assert torch.count_nonzero(jacobian[:, PARTITION[1 - g]]) == 0
            assert torch.count_nonzero(jacobian[row, PARTITION[g]]) > 0
        parts[g].square().sum().backward()
        for p in model.towers[g].parameters():
            assert p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.towers[1 - g].parameters():
            assert p.grad is None or torch.count_nonzero(p.grad) == 0
        width = 4
        head_grad = model.output_layer.weight.grad
        assert torch.count_nonzero(head_grad[g * width:(g + 1) * width]) > 0
        assert torch.count_nonzero(head_grad[(1 - g) * width:(2 - g) * width]) == 0
        assert model.output_layer.bias.grad is None
        for p in model.embedding.num_embedding.parameters():
            assert torch.count_nonzero(p.grad[PARTITION[g]]) > 0
            assert torch.count_nonzero(p.grad[PARTITION[1 - g]]) == 0
        if scale_position == "input":
            grad = model.embedding.scaling.scale.grad
            own, other = PARTITION[g], PARTITION[1 - g]
        else:
            grad = model.front_scale.scale.grad
            own, other = model.towers[g].indices, model.towers[1 - g].indices
        assert torch.count_nonzero(grad[own]) > 0
        assert torch.count_nonzero(grad[other]) == 0
        optimizer.step()
        changed = x.detach().clone()
        changed[:, PARTITION[1 - g]] += 5
        assert torch.equal(capture(model, x, cat)[2][g], capture(model, changed, cat)[2][g])


def test_budget_and_optimizer_partition():
    partition = [list(range(8)), list(range(8, 50))]
    counts = []
    for widths, routing in [
        ([384] * 3, {}), ([288] * 3, {"tower_groups": partition}),
        ([384] * 3, {"first_layer_groups": partition}),
    ]:
        seed_everything(43)
        embedding = FeatureEmbedding(50, [], num_embedding="pbld", d_num_embedding=8,
                                     n_frequencies=16, num_scaling=True)
        assert embedding.d_out == 400
        assert sum(p.numel() for p in embedding.parameters()) == 7_600
        model = RealMLPNet(embedding, 1, hidden_sizes=widths, activation="selu",
                           use_parametric_act=True, zero_init_output=True,
                           scale_position="input", **routing)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if "tower_groups" in routing:
            assert [t.trunk[0].weight.shape for t in model.towers] == [(64, 288), (336, 288)]
            assert sum(t.trunk[0].weight.numel() for t in model.towers) == 115_200
            assert list(dict(model.output_layer.named_parameters())) == ["weight", "bias"]
            assert model.output_layer.bias.numel() == 1
    assert counts == [458_801, 458_609, 458_801]

    for ff in (1.0, 3.0):
        model = make(first_layer_lr_factor=ff, bias_lr_factor=0.2, act_lr_factor=0.4,
                     plr_lr_factor=0.5, scale_lr_factor=2.0, linear_skip_idx=[0],
                     linear_skip_lr_factor=0.6)
        groups = model.param_groups()
        assigned = [id(p) for group in groups for p in group["params"]]
        assert len(assigned) == len(set(assigned))
        assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
        assert all(group["params"] for group in groups)
        by_param = {id(p): group for group in groups for p in group["params"]}
        for tower in model.towers:
            assert by_param[id(tower.trunk[0].weight)]["lr_factor"] == ff
            assert by_param[id(tower.trunk[0].bias)]["lr_factor"] == ff * 0.2
            for layer in tower.trunk:
                if isinstance(layer, ParametricActivation):
                    assert by_param[id(layer.alpha)]["lr_factor"] == 0.4
        for layer in model.modules():
            if isinstance(layer, NTPLinear):
                assert by_param[id(layer.bias)]["wd_factor"] == 0.0
        for p in model.embedding.num_embedding.parameters():
            assert by_param[id(p)]["lr_factor"] == 0.5
        assert by_param[id(model.embedding.scaling.scale)]["lr_factor"] == 2.0
        for p in (model.skip_weight, model.skip_bias):
            assert by_param[id(p)]["lr_factor"] == 0.6
            assert by_param[id(p)]["wd_factor"] == 0.0
        assert by_param[id(model.output_layer.weight)]["lr_factor"] == 1.0


@pytest.mark.parametrize("zero_init_output", [False, True])
def test_data_init_walks_both_towers(inputs, zero_init_output):
    seed_everything(47)
    model = make(init_mode="std+he5", zero_init_output=zero_init_output, dropout=0.2,
                 dropout_schedule="flat_cos", scale_position="first_layer")
    assert model.needs_data_init
    x, cat = inputs
    model.data_init(x, cat)
    assert model.training
    model.eval()
    raw, activations, parts = capture(model, x, cat)
    for g, tower in enumerate(model.towers):
        for i, layer in enumerate(tower.trunk):
            if isinstance(layer, NTPLinear):
                pre = activations[g, i] - layer.bias
                torch.testing.assert_close(pre.std(0, correction=0), torch.ones(pre.shape[1]),
                                           atol=2e-5, rtol=2e-5)
        changed = x.clone()
        changed[:, PARTITION[1 - g]] += 5
        _, after, changed_parts = capture(model, changed, cat)
        for i in range(len(tower.trunk)):
            assert torch.equal(activations[g, i], after[g, i])
        assert torch.equal(parts[g], changed_parts[g])
    if zero_init_output:
        assert torch.count_nonzero(model.output_layer.weight) == 0
        assert torch.count_nonzero(model.output_layer.bias) == 0
        assert torch.count_nonzero(raw) == 0
    else:
        torch.testing.assert_close((raw - model.output_layer.bias).std(0, correction=0),
                                   torch.ones(1), atol=2e-5, rtol=2e-5)
    model = make(init_mode="ntp")
    assert not model.needs_data_init
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    model.data_init(x, cat)
    assert_state_equal(state, model.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_towers_directory_round_trip(inputs, tmp_path):
    x = inputs[0][:64].numpy()
    y = (x[:, 0] + x[:, 2] > 0).astype(int)
    params = {"tower_groups": [[1, 0], [3, 2]], "hidden_sizes": [7, 5, 4],
              "d_num_embedding": 4, "n_frequencies": 3, "num_scaling": True,
              "init_mode": "std+he5", "dropout": 0.1, "dropout_schedule": "flat_cos"}
    model = MasaClassifier(model="realmlp", model_params=params, num_embedding="pbld",
                           device="cpu", amp=False, n_epochs=2, random_state=53)
    model.fit(x, y, sample_weight=np.linspace(0.2, 2.0, len(y)))
    before = model.predict_proba(x)
    directory = tmp_path / "towers"
    model.save_model(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    resolved = manifest["fitted"]["resolved_model_params"]
    for key in ("tower_groups", "hidden_sizes"):
        assert resolved[key] == params[key]
    loaded = MasaClassifier.load_model(directory)
    assert loaded.resolved_model_params_ == model.resolved_model_params_
    np.testing.assert_array_equal(before, loaded.predict_proba(x))
    assert np.isfinite(before).all()
    assert_state_equal(model.model_.state_dict(), loaded.model_.state_dict())
    for g in range(2):
        key = f"towers.{g}.indices"
        assert key in loaded.model_.state_dict()
        assert loaded.model_.state_dict()[key].dtype == torch.int64


def test_ema_covers_towers_embedding_and_head(inputs, monkeypatch):
    expected = {}
    updates = []
    buffers = {}
    original = trainer._update_ema

    def record(model, ema_params, decay):
        params = dict(model.named_parameters())
        assert set(params) == set(ema_params)
        if not expected:
            expected.update({name: value.clone() for name, value in ema_params.items()})
            buffers.update({name: value.clone() for name, value in model.named_buffers()
                            if name.endswith("indices")})
            for prefix in ("embedding.", "towers.0.", "towers.1.", "output_layer."):
                assert any(name.startswith(prefix) for name in params)
        for name, p in params.items():
            expected[name] = decay * expected[name] + (1 - decay) * p.detach()
        original(model, ema_params, decay)
        for name in params:
            torch.testing.assert_close(ema_params[name], expected[name], atol=2e-7, rtol=2e-6)
        updates.append({name: value.clone() for name, value in ema_params.items()})

    monkeypatch.setattr(trainer, "_update_ema", record)
    x = inputs[0][:32].numpy()
    y = (x[:, 0] + x[:, 2] > 0).astype(int)
    model = MasaClassifier(model="realmlp", num_embedding="pbld", model_params={
        "tower_groups": PARTITION, "hidden_sizes": [7, 5], "num_scaling": True,
        "d_num_embedding": 4, "n_frequencies": 3, "use_parametric_act": True,
        "zero_init_output": False,
    }, n_epochs=3, batch_size=16, ema_decay=0.8, device="cpu", amp=False, random_state=59)
    model.fit(x, y)  # Fixed epochs: no eval_set or early stopping.
    assert len(updates) == 6
    assert_state_equal(updates[-1], dict(model.model_.named_parameters()))
    for name, value in buffers.items():
        assert torch.equal(value, model.model_.state_dict()[name])


@pytest.mark.parametrize("groups,widths,first", [
    ([], [4], None), ([[]], [4], None), ([[], [0, 1, 2, 3]], [4], None),
    ([[0, 1], [2]], [4], None), ([[0, 0], [1, 2, 3]], [4], None),
    ([[0, 1], [1, 2, 3]], [4], None), ([[-1, 0], [1, 2, 3]], [4], None),
    ([[0, 1], [2, 4]], [4], None), ([[0, 1], [2, True]], [4], None),
    ([[0, 1], [2, 3.0]], [4], None), ([[0, 1], [2, np.int64(3)]], [4], None),
    ([0, 1, 2, 3], [4], None), ("all", [4], None),
    (PARTITION, [], None), (PARTITION, [0], None), (PARTITION, [-1], None),
    (PARTITION, [4, 0], None), (PARTITION, [4, -1], None),
    ([[3, 1, 0, 2]], [], None), ([[3, 1, 0, 2]], [4, 0], None),
    (PARTITION, [4], PARTITION), ([[0, 1, 2, 3]], [4], [[0, 1, 2, 3]]),
])
def test_bad_tower_partitions_raise(groups, widths, first):
    embedding = FeatureEmbedding(4, [])
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="tower_groups"):
        RealMLPNet(embedding, 1, tower_groups=groups, hidden_sizes=widths,
                   first_layer_groups=first)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_first_layer_groups_regression():
    # Recorded from v0.13.0 / 572da21 before adding towers. Tolerance accommodates
    # platform-level randn/libm differences, as in test_realmlp_fidelity.py.
    seed_everything(17)
    model = RealMLPNet(FeatureEmbedding(4, [], num_scaling=True), 1, hidden_sizes=[3, 2],
                       activation="selu", use_parametric_act=True, zero_init_output=False,
                       first_layer_groups=[[0, 2], [1, 3]])
    expected = {
        "embedding.scaling.scale": [1., 1., 1., 1.],
        "trunk.0.weight": [[-1.41351318, .23363075, .03403318],
                           [.34991726, -.01452155, -.61236197],
                           [-1.18354678, -1.48305464, 1.80043614],
                           [.00957405, .15344739, -2.66308761]],
        "trunk.0.bias": [-1.43114412, -.54830325, .32318041],
        "trunk.0.group_mask": [[1., 1., 0.], [0., 0., 1.], [1., 1., 0.], [0., 0., 1.]],
        "trunk.0.group_scale": [.70710677, .70710677, .70710677],
        "trunk.1.alpha": [1., 1., 1.],
        "trunk.2.weight": [[-.47796881, 1.56182230], [-.12975445, -.13349894],
                           [1.27398169, -.12876794]],
        "trunk.2.bias": [-2.64808130, .61140561],
        "trunk.3.alpha": [1., 1.],
        "output_layer.weight": [[.47524163], [1.48780966]],
        "output_layer.bias": [.54644030],
    }
    assert list(model.state_dict()) == list(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, torch.tensor(expected[name]), atol=2e-6, rtol=2e-6)
    x = torch.tensor([[0., 1., 2., 3.], [-1., .5, 2., -.5]])
    torch.testing.assert_close(model(x, torch.empty(2, 0, dtype=torch.long)),
                               torch.tensor([[-.90444940], [-.84444076]]), atol=2e-6, rtol=2e-6)


def test_chunk_integrity_and_schedule(inputs):
    seed_everything(61)
    embedding = FeatureEmbedding(4, [4], num_embedding="pbld", d_num_embedding=3,
                                 n_frequencies=3, num_embedding_idx=[0, 2],
                                 cat_emb_dim=2, num_scaling=True)
    # Embedding order: numeric 0, numeric 2, bypass 1, bypass 3, categorical 0.
    assert embedding.feature_chunk_sizes == [3, 3, 1, 1, 2]
    model = RealMLPNet(embedding, 2, hidden_sizes=[1, 3], tower_groups=[[4, 0], [2, 1], [3]],
                       dropout=0.3, dropout_schedule="flat_cos", scale_position="first_layer",
                       zero_init_output=False).eval()
    assert [t.indices.tolist() for t in model.towers] == [[0, 1, 2, 8, 9], [3, 4, 5, 6], [7]]
    assert all(t.indices.dtype == torch.int64 for t in model.towers)
    calls, selected, packed = [], [], []
    handles = [embedding.register_forward_hook(lambda *args: calls.append(1)),
               model.output_layer.register_forward_pre_hook(lambda module, args:
                                                              packed.append(args[0]))]
    for tower in model.towers:
        handles.append(tower.trunk[0].register_forward_pre_hook(
            lambda module, args: selected.append(args[0])))
    with torch.no_grad():
        model.front_scale.scale.copy_(torch.arange(1., 11.))
    x = inputs[0][:8]
    cat = (torch.arange(8) % 4).reshape(-1, 1)
    try:
        raw, activations, parts = capture(model, x, cat)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 1
    h = model.front_scale(embedding(x, cat))
    for g, tower in enumerate(model.towers):
        assert torch.equal(selected[g], h[:, tower.indices])
    expected = math.sqrt(3) * torch.cat([
        activations[g, len(t.trunk) - 1] for g, t in enumerate(model.towers)
    ], dim=1)
    assert torch.equal(packed[0], expected)
    torch.testing.assert_close(raw, sum(parts) + model.output_layer.bias)
    for t in (0., 0.75, 1.):
        model.set_schedule_t(t)
        for tower in model.towers:
            drops = [m for m in tower.trunk if isinstance(m, ScheduledDropout)]
            assert len(drops) == 2
            for drop in drops:
                assert drop._keep.item() == pytest.approx(1 - 0.3 * trainer.flat_cos(t))
