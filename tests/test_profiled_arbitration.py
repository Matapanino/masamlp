"""Contracts for routing one reliability gate through profiled RealMLP."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.models import build_model
from masamlp.models.base import FeatureEmbedding
from masamlp.models.profiled_realmlp import ProfiledRealMLPNet

GATE = {
    "estimator_idx": [0, 1],
    "reliability_idx": [2],
    "n_heads": 2,
    "hidden_size": 4,
}

_GROUP_HASH = "b2663f442a5c57b840e21251bc39ffcbd333216330eba1f341714b4ebe905cc2"
_PROJECTED_STATE = "b219679e7b38812281af143921922cbc169f5c304a3a47157ec915eade6297ca"
_UNPROJECTED_STATE = "31db1066f75a3fb51f69a4706c52bbe9325629ae3c5f6817b1ad0cf14a700122"
_ACTIVE_FORWARD = "76979fab55a7c02fcedb8d17a124ebdde3b5cdab294a5b399413adcc6bf2050e"
_WARMUP_FORWARD = "0b3cd77fa9e2b45a2f1ceab12c5f98e76ac22a905fe691ba7abf3d1f300cee83"


def _digest(parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _state_hash(model):
    return _digest(
        name.encode() + str(tuple(value.shape)).encode() + value.detach().numpy().tobytes()
        for name, value in model.state_dict().items()
    )


def _group_hash(model):
    names = {id(param): name for name, param in model.named_parameters()}
    return _digest(
        [
            repr(
                [
                    (
                        {key: value for key, value in group.items() if key != "params"},
                        [names[id(param)] for param in group["params"]],
                    )
                    for group in model.param_groups()
                ]
            ).encode()
        ]
    )


def _parameter_inventory(model):
    return [(name, tuple(param.shape)) for name, param in model.named_parameters()]


def _optimizer_inventory(model):
    names = {id(param): name for name, param in model.named_parameters()}
    return [
        (
            group.get("lr_factor", 1.0),
            group.get("wd_factor", 1.0),
            [names[id(param)] for param in group["params"]],
        )
        for group in model.param_groups()
    ]


def _assert_equal_state_dicts(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name]), name


def _plain(mode="joint", warmup=0):
    torch.manual_seed(401)
    return ProfiledRealMLPNet(
        FeatureEmbedding(4, [], num_scaling=True),
        1,
        source_groups=[[0, 1], [2, 3]],
        hidden_sizes=(8, 6),
        source_hidden_sizes=(7, 3),
        profile_mode=mode,
        source_warmup_epochs=warmup,
        zero_init_output=False,
    )


def _combined(mode="joint", warmup=4):
    return build_model(
        "profiled_realmlp",
        {
            "arbitration": GATE,
            # The post-gate frame has five retained numeric chunks followed
            # by the two mixtures.  Sources split it 4/3; r receives all 7.
            "source_groups": [[0, 1, 2, 3], [4, 5, 6]],
            "hidden_sizes": [6, 5],
            "source_hidden_sizes": [5, 3],
            "profile_mode": mode,
            "source_warmup_epochs": warmup,
            "zero_init_output": False,
        },
        8,
        [],
        1,
    )


def _inputs():
    torch.manual_seed(403)
    return torch.randn(24, 8), torch.empty(24, 0, dtype=torch.long)


def test_arbitration_off_has_recorded_inventory_and_deterministic_legacy_path():
    """The no-gate profiled construction remains the original in-process path."""
    expected_parameters = [
        ("remainder.embedding.scaling.scale", (4,)),
        ("remainder.trunk.0.weight", (4, 8)),
        ("remainder.trunk.0.bias", (8,)),
        ("remainder.trunk.2.weight", (8, 6)),
        ("remainder.trunk.2.bias", (6,)),
        ("remainder.output_layer.weight", (6, 1)),
        ("remainder.output_layer.bias", (1,)),
        ("sources.embedding.scaling.scale", (4,)),
        ("sources.towers.0.trunk.0.weight", (2, 7)),
        ("sources.towers.0.trunk.0.bias", (7,)),
        ("sources.towers.0.trunk.2.weight", (7, 3)),
        ("sources.towers.0.trunk.2.bias", (3,)),
        ("sources.towers.1.trunk.0.weight", (2, 7)),
        ("sources.towers.1.trunk.0.bias", (7,)),
        ("sources.towers.1.trunk.2.weight", (7, 3)),
        ("sources.towers.1.trunk.2.bias", (3,)),
        ("sources.output_layer.weight", (6, 1)),
        ("sources.output_layer.bias", (1,)),
    ]
    expected_groups = [
        (6.0, 1.0, ["remainder.embedding.scaling.scale"]),
        (
        1.0,
        1.0,
        ["remainder.trunk.0.weight", "remainder.trunk.2.weight", "remainder.output_layer.weight"],
    ),
        (
            0.1,
            0.0,
            ["remainder.trunk.0.bias", "remainder.trunk.2.bias", "remainder.output_layer.bias"],
        ),
        (6.0, 1.0, ["sources.embedding.scaling.scale"]),
        (
            1.0,
            1.0,
            [
                "sources.towers.0.trunk.0.weight",
                "sources.towers.1.trunk.0.weight",
                "sources.towers.0.trunk.2.weight",
                "sources.towers.1.trunk.2.weight",
                "sources.output_layer.weight",
            ],
        ),
        (
            0.1,
            0.0,
            [
                "sources.towers.0.trunk.0.bias",
                "sources.towers.1.trunk.0.bias",
                "sources.towers.0.trunk.2.bias",
                "sources.towers.1.trunk.2.bias",
                "sources.output_layer.bias",
            ],
        ),
    ]
    first = _plain()
    assert first.arbitration is None
    assert _parameter_inventory(first) == expected_parameters
    assert sum(param.numel() for param in first.parameters()) == 206
    assert _optimizer_inventory(first) == expected_groups

    second = _plain()
    _assert_equal_state_dicts(first, second)
    torch.manual_seed(402)
    x = torch.randn(17, 4)
    c = torch.empty(17, 0, dtype=torch.long)
    first.set_training_epoch(0)
    second.set_training_epoch(0)
    first.eval()
    second.eval()
    assert torch.equal(first(x, c), second(x, c))


@pytest.mark.parametrize(
    "mode,warmup,state_hash,forward_hash",
    [
        ("joint", 0, _PROJECTED_STATE, _ACTIVE_FORWARD),
        ("joint", 4, _PROJECTED_STATE, _WARMUP_FORWARD),
        ("unprojected", 0, _UNPROJECTED_STATE, _ACTIVE_FORWARD),
        ("unprojected", 4, _UNPROJECTED_STATE, _WARMUP_FORWARD),
        ("frozen", 0, _PROJECTED_STATE, _ACTIVE_FORWARD),
        ("frozen", 4, _PROJECTED_STATE, _WARMUP_FORWARD),
    ],
)
@pytest.mark.skipif(
    not os.environ.get("MASAMLP_PIN_FINGERPRINTS"),
    reason=(
        "pin ee61e06 fingerprints came from the macOS read-only worktree; "
        "set MASAMLP_PIN_FINGERPRINTS=1 to verify them locally"
    ),
)
def test_no_arbitration_is_byte_identical_to_ee61e06(mode, warmup, state_hash, forward_hash):
    """Local-only macOS evidence generated in the read-only ee61e06 worktree."""
    net = _plain(mode, warmup)
    torch.manual_seed(402)
    x = torch.randn(17, 4)
    c = torch.empty(17, 0, dtype=torch.long)
    output = []
    for epoch in (0, 3, 4):
        net.set_training_epoch(epoch)
        net.eval()
        output.append(net(x, c).detach().numpy().tobytes())
    if net.needs_prediction_state():
        net.reset_prediction_state()
        net.update_prediction_state(x, c, None)
        net.finalize_prediction_state()
    assert _state_hash(net) == state_hash
    assert _digest(output) == forward_hash
    assert _group_hash(net) == _GROUP_HASH


def test_combined_gate_is_shared_once_and_routes_post_gate_source_groups():
    net = _combined()
    x, c = _inputs()
    assert net.arbitration is not None
    assert net.remainder.arbitration is None
    assert net.sources.arbitration is None
    assert [tower.trunk[0].in_features for tower in net.sources.towers] == [4, 3]
    assert net.remainder.trunk[0].in_features == 7
    calls = 0

    def counted(*_):
        nonlocal calls
        calls += 1

    hook = net.arbitration.register_forward_hook(counted)
    net.eval()
    net(x, c)
    assert calls == 1
    net.decompose(x, c)
    assert calls == 2
    net._source_basis(x, c)
    assert calls == 3
    net.update_prediction_state(x, c, None)
    assert calls == 4
    net.data_init(x, c)
    hook.remove()
    assert calls == 5
    net.train()
    net.set_training_epoch(0)
    net(x, c).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in net.arbitration.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() for p in net.sources.parameters())
    assert all(p.grad is None for p in net.remainder.parameters())
    net.zero_grad(set_to_none=True)
    net.set_training_epoch(4)
    net(x, c).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in net.sources.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() for p in net.remainder.parameters())
    ids = [id(param) for group in net.param_groups() for param in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(net.parameters()))
    gate_ids = {id(param) for param in net.arbitration.parameters()}
    gate_group = next(
        group for group in net.param_groups() if gate_ids & {id(p) for p in group["params"]}
    )
    assert gate_group["lr_factor"] == 1.0
    assert gate_group["wd_factor"] == 0.0


def test_source_groups_put_categorical_chunks_before_addressable_mixtures():
    net = build_model(
        "profiled_realmlp",
        {
            "arbitration": GATE,
            # Semantic index 5 is categorical, while 6 and 7 are mixtures.
            "source_groups": [[0, 5, 7], [1, 2, 3, 4, 6]],
            "hidden_sizes": [4],
            "source_hidden_sizes": [4, 2],
            "zero_init_output": False,
        },
        8,
        [5],
        1,
    )
    # Physical embedding order is retained numerics, mixtures, categorical;
    # the translated source has kept numeric 0, categorical 0, and mixture 1.
    assert [tower.indices.tolist() for tower in net.sources.towers] == [
        [0, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5],
    ]
    assert net.remainder.trunk[0].in_features == 11


def test_post_gate_six_by_thirtysix_source_partition_keeps_full_remainder():
    # 43 raw numeric coordinates lose one estimate and one reliability signal,
    # then gain one mixture: 42 post-gate semantic coordinates, split 6/36.
    net = build_model(
        "profiled_realmlp",
        {
            "arbitration": {
                "estimator_idx": [0],
                "reliability_idx": [1],
                "n_heads": 1,
                "mode": "constant",
            },
            "source_groups": [list(range(5)) + [41], list(range(5, 41))],
            "hidden_sizes": [4],
            "source_hidden_sizes": [4, 2],
        },
        43,
        [],
        1,
    )
    assert [tower.trunk[0].in_features for tower in net.sources.towers] == [6, 36]
    assert net.remainder.trunk[0].in_features == 42


def test_warmup_keeps_sources_and_gate_active_all_epochs():
    net = _combined(mode="unprojected", warmup=4)
    x, c = _inputs()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(16):
        net.train()
        net.set_training_epoch(epoch)
        source_before = [p.detach().clone() for p in net.sources.parameters()]
        remainder_before = [p.detach().clone() for p in net.remainder.parameters()]
        gate_before = [p.detach().clone() for p in net.arbitration.parameters()]
        net(x, c).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        source_changed = any(
            not torch.equal(a, b)
            for a, b in zip(source_before, net.sources.parameters(), strict=True)
        )
        gate_changed = any(
            not torch.equal(a, b)
            for a, b in zip(gate_before, net.arbitration.parameters(), strict=True)
        )
        remainder_changed = any(
            not torch.equal(a, b)
            for a, b in zip(remainder_before, net.remainder.parameters(), strict=True)
        )
        assert source_changed and gate_changed
        assert remainder_changed == (epoch >= 4)
    assert not net.needs_prediction_state()


def test_frozen_mode_leaves_gate_trainable():
    net = _combined(mode="frozen", warmup=4)
    x, c = _inputs()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(16):
        net.train()
        net.set_training_epoch(epoch)
        remainder_before = [p.detach().clone() for p in net.remainder.parameters()]
        gate_before = [p.detach().clone() for p in net.arbitration.parameters()]
        net(x, c).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gate_changed = any(
            not torch.equal(a, b)
            for a, b in zip(gate_before, net.arbitration.parameters(), strict=True)
        )
        remainder_changed = any(
            not torch.equal(a, b)
            for a, b in zip(remainder_before, net.remainder.parameters(), strict=True)
        )
        assert gate_changed
        assert remainder_changed == (epoch >= 4)
    assert net._source_frozen
    assert not net.sources.training
    assert all(not p.requires_grad for p in net.sources.parameters())
    assert all(p.requires_grad for p in net.arbitration.parameters())
    assert net.needs_prediction_state()


def test_combined_parameter_count_matches_hand_derived_formula():
    gate = {"estimator_idx": [0, 1], "reliability_idx": [2], "n_heads": 1, "mode": "constant"}
    net = build_model(
        "profiled_realmlp",
        {
            "arbitration": gate,
            "source_groups": [[0], [1]],
            "hidden_sizes": [4, 3],
            "source_hidden_sizes": [3, 2],
            "zero_init_output": False,
        },
        4,
        [],
        1,
    )
    # Gate: one head x two estimators. Remainder: (2x4+4)+(4x3+3)+(3x1+1).
    # Sources: two [(1x3+3)+(3x2+2)] towers plus a (4x1+1) packed head.
    expected = 2 + ((2 * 4 + 4) + (4 * 3 + 3) + (3 + 1)) + 2 * ((3 + 3) + (6 + 2)) + (4 + 1)
    assert sum(param.numel() for param in net.parameters()) == expected == 66


def test_combined_estimator_ema_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(51)
    x = rng.normal(size=(48, 8)).astype("float32")
    y = (x[:, 0] + x[:, 4] > 0).astype("int64")
    model = MasaClassifier(
        model="profiled_realmlp",
        model_params={
            "arbitration": GATE,
            "source_groups": [[0, 1, 2, 3], [4, 5, 6]],
            "hidden_sizes": [6, 5],
            "source_hidden_sizes": [5, 3],
            "source_warmup_epochs": 1,
            "profile_mode": "joint",
            "zero_init_output": False,
        },
        n_epochs=3,
        batch_size=16,
        ema_decay=0.8,
        device="cpu",
        amp=False,
        random_state=19,
    ).fit(x, y)
    before = model.predict_proba(x)
    model.save_model(tmp_path / "combined")
    loaded = MasaClassifier.load_model(tmp_path / "combined")
    np.testing.assert_array_equal(before, loaded.predict_proba(x))
