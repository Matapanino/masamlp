"""Contracts for semantic tower routing after RealMLP estimator arbitration."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.models import build_model
from masamlp.models.arbitration import post_arbitration_feature_layout

GATE = {
    "estimator_idx": [2, 3],
    "reliability_idx": [4],
    "n_heads": 2,
    "hidden_size": 4,
}


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


def _combined(groups, *, cat_cardinalities=None, num_embedding=None, chunks=None):
    params = {
        "arbitration": GATE,
        "tower_groups": groups,
        "hidden_sizes": [3, 2],
        "zero_init_output": False,
    }
    if chunks is not None:
        params["_num_input_chunks"] = chunks
    return build_model("realmlp", params, 5, cat_cardinalities or [], 1, num_embedding)


def test_post_arbitration_layout_keeps_semantic_order_and_physical_slices():
    # Physical embedding order is kept numerics, mixtures, categoricals; the
    # public semantic frame moves categoricals back before appended mixtures.
    layout = post_arbitration_feature_layout([3, 3, 3, 3, 2], 4, 2)
    assert [(item.name, item.kind, item.size, item.physical_slice) for item in layout] == [
        ("numeric_0", "numeric", 3, slice(0, 3)),
        ("numeric_1", "numeric", 3, slice(3, 6)),
        ("categorical_0", "categorical", 2, slice(12, 14)),
        ("mixture_0", "mixture", 3, slice(6, 9)),
        ("mixture_1", "mixture", 3, slice(9, 12)),
    ]


@pytest.mark.parametrize(
    "groups,index",
    [
        ([[0, 1], [1, 2, 3]], 1),
        ([[0, 1], [2]], 3),
        ([[0, 1], [2, 4]], 4),
    ],
)
def test_post_arbitration_tower_validation_names_bad_index(groups, index):
    with pytest.raises(ValueError, match=rf"{index}"):
        _combined(groups)


def test_post_arbitration_valid_map_and_removed_pre_gate_index():
    model = _combined([[0, 2, 3], [1, 4]], cat_cardinalities=[5])
    assert [tower.indices.tolist() for tower in model.towers] == [[0, 2, 4, 5, 6, 7], [1, 3]]
    # There are four post-gate semantic chunks.  Index four was a removed
    # pre-gate coordinate and must not silently address a tower.
    with pytest.raises(ValueError, match="4"):
        _combined([[0, 1, 2], [3, 4]])

    removed_estimator = {
        **GATE,
        "estimator_idx": [4],
        "reliability_idx": [3],
        "n_heads": 1,
    }
    with pytest.raises(ValueError, match="4"):
        build_model(
            "realmlp",
            {"arbitration": removed_estimator, "tower_groups": [[0, 1], [2, 3, 4]]},
            5,
            [],
            1,
        )
    with pytest.raises(ValueError, match="first_layer_groups"):
        build_model(
            "realmlp",
            {"arbitration": GATE, "first_layer_groups": [[0, 1], [2, 3]]},
            5,
            [],
            1,
        )


def test_post_arbitration_chunk_map_coalesces_embedded_numeric_coordinates():
    model = _combined([[0, 1], [2]], num_embedding="pbld", chunks=[2, 1, 1])
    assert model.embedding.feature_chunk_sizes == [32, 16, 16]
    assert [tower.trunk[0].in_features for tower in model.towers] == [48, 16]


def test_pin_parity_for_gate_only_and_towers_only():
    # These independent fingerprints were generated with the frozen
    # 6d652dc0 pin in ~/dev/masaMLP-wt-p1s2, not with this branch.
    x_gate = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10
    torch.manual_seed(712)
    gate = build_model(
        "realmlp",
        {"arbitration": GATE, "hidden_sizes": [5], "zero_init_output": False},
        5,
        [],
        1,
        "pbld",
    )
    assert _state_hash(gate) == "9cd6da5c31476b6a271b5e91ed2d781cd3f98297c2046500cb83616c110b703d"
    gate_output_hash = _digest(
        [gate(x_gate, torch.empty(4, 0, dtype=torch.long)).detach().numpy().tobytes()]
    )
    assert gate_output_hash == "7325af006aff9d71ad68d0cf181cc7cade18c9900b8eff0aba0b770bc890edb0"
    assert _group_hash(gate) == "8e5c0d2f286ca18c4488b784dfcddf1b5227f652ae2ed3eb166525c6ae51ba4c"

    x_towers = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
    torch.manual_seed(713)
    towers = build_model(
        "realmlp",
        {"tower_groups": [[0, 2], [1, 3]], "hidden_sizes": [5, 3], "zero_init_output": False},
        4,
        [],
        1,
        "pbld",
    )
    assert _state_hash(towers) == "c907a8926faeaf34e2e0cf937c5abc797ecdade4685fb1848d6028f0f3fd3d5f"
    towers_output_hash = _digest(
        [towers(x_towers, torch.empty(4, 0, dtype=torch.long)).detach().numpy().tobytes()]
    )
    assert towers_output_hash == "4a9474d4b5cf8af73f9ed110c8ea6ade96515a53c9fe02f82320fad1ee020a74"
    assert _group_hash(towers) == "141be65a8d6e847c3d2378b25bc2d173e7cd9275f1921c218ca1f2f780d3f2b9"


def test_combined_forward_backward_optimizer_and_tower_widths():
    model = _combined([[0, 2, 3], [1, 4]], cat_cardinalities=[5])
    x = torch.randn(8, 5)
    cat = torch.arange(8).remainder(5).reshape(-1, 1)
    model(x, cat).square().mean().backward()
    assert all(param.grad is not None for param in model.arbitration.conditioner.parameters())
    assert all(param.grad is not None for tower in model.towers for param in tower.parameters())
    ids = [id(param) for group in model.param_groups() for param in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    gate_ids = {id(param) for param in model.arbitration.parameters()}
    gate_group = next(
        group
        for group in model.param_groups()
        if gate_ids & {id(param) for param in group["params"]}
    )
    assert gate_group["wd_factor"] == 0.0
    # Semantic group 0 owns numeric 0, categorical 0, and mixture 0 only.
    assert [tower.trunk[0].in_features for tower in model.towers] == [6, 2]


def test_combined_parameter_count_matches_hand_derived_formula():
    model = _combined([[0, 2], [1, 3]])
    # Gate: (2 estimates + 1 reliability) * 4 + 4 + 4 * (2 heads * 2 estimates)
    # + 2 * 2.  Each tower: (2 * 3 + 3) + (3 * 2 + 2); head: (2 * 2) + 1.
    expected = (
        ((2 + 1) * 4 + 4 + 4 * (2 * 2) + 2 * 2) + 2 * ((2 * 3 + 3) + (3 * 2 + 2)) + (2 * 2 + 1)
    )
    assert sum(param.numel() for param in model.parameters()) == expected == 75


def test_combined_estimator_ema_and_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(33)
    x = rng.normal(size=(32, 5)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int64")
    estimator = MasaClassifier(
        model="realmlp",
        num_embedding="pbld",
        model_params={
            "arbitration": GATE,
            "tower_groups": [[0, 1], [2, 3]],
            "hidden_sizes": [5, 3],
            "zero_init_output": False,
        },
        n_epochs=2,
        batch_size=16,
        ema_decay=0.8,
        device="cpu",
        amp=False,
        random_state=17,
    ).fit(x, y)
    before = estimator.predict_proba(x)
    estimator.save_model(tmp_path / "combined")
    loaded = MasaClassifier.load_model(tmp_path / "combined")
    np.testing.assert_array_equal(before, loaded.predict_proba(x))
