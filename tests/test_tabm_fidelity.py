"""Paper-fidelity tests for full TabM, PLE-vB, and member batches."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest
import torch
from torch import nn

from masamlp.classifier import MasaClassifier
from masamlp.data.dataset import TabularData
from masamlp.data.preprocessing import TabularPreprocessor
from masamlp.models import build_model

_HAS_TABM_ORACLE = importlib.util.find_spec("tabm") is not None
_HAS_PLE_ORACLE = importlib.util.find_spec("rtdl_num_embeddings") is not None


def _empty_cat(n: int) -> torch.Tensor:
    return torch.zeros(n, 0, dtype=torch.long)


def _build_full(**params):
    return build_model(
        "tabm",
        {"variant": "full", "k": 3, "d": 8, "n_blocks": 2, "dropout": 0.0, **params},
        n_num=4,
        cat_cardinalities=[],
        out_dim=2,
    )


def test_mini_predictions_are_byte_identical_to_v092():
    """Pin captured from main@5ad623f before changing any source file."""
    X = np.array(
        [
            [-2.0, 0.5, 1.0],
            [-1.5, -0.25, 0.0],
            [-1.0, 1.5, -1.0],
            [-0.5, -1.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 1.0, -0.5],
            [1.0, -1.5, 1.0],
            [1.5, 0.25, -1.0],
            [2.0, -0.5, 0.5],
            [2.5, 1.5, -0.5],
            [3.0, -1.0, 1.5],
            [3.5, 0.75, -1.5],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    common = dict(
        model="tabm",
        numeric_scaler="none",
        categorical_features=None,
        n_epochs=3,
        batch_size=None,
        learning_rate=1e-3,
        device="cpu",
        amp=False,
        random_state=1729,
    )
    legacy = MasaClassifier(
        **common, model_params={"k": 4, "d": 8, "n_blocks": 2, "dropout": 0.0}
    ).fit(X, y)
    explicit = MasaClassifier(
        **common,
        model_params={
            "variant": "mini",
            "k": 4,
            "d": 8,
            "n_blocks": 2,
            "dropout": 0.0,
        },
    ).fit(X, y)
    probe = np.array(
        [[-1.25, 0.75, 0.25], [0.25, -0.5, 1.25], [2.25, 0.5, -0.75]],
        dtype=np.float64,
    )
    expected = np.frombuffer(
        bytes.fromhex("88edd83e3c89133fa089cf3e303b183f70b0ce3ec8a7183f"),
        dtype=np.float32,
    ).reshape(3, 2)
    np.testing.assert_array_equal(legacy.predict_proba(probe), expected)
    np.testing.assert_array_equal(explicit.predict_proba(probe), expected)


def test_full_k1_with_ones_equals_plain_mlp_after_weight_copy():
    torch.manual_seed(11)
    model = _build_full(k=1)
    plain_layers: list[nn.Module] = []
    with torch.no_grad():
        for layer in model.backbone:
            layer.r.fill_(1.0)
            layer.s.fill_(1.0)
            plain = nn.Linear(layer.in_features, layer.out_features)
            plain.weight.copy_(layer.weight)
            plain.bias.copy_(layer.bias[0])
            plain_layers.extend([plain, nn.ReLU()])
        head = nn.Linear(8, 2)
        head.weight.copy_(model.output_layer.weight[0])
        head.bias.copy_(model.output_layer.bias[0])
        plain_layers.append(head)
    plain = nn.Sequential(*plain_layers)
    x = torch.randn(17, 4)
    torch.testing.assert_close(
        model(x, _empty_cat(len(x)))[:, 0], plain(x), rtol=1e-6, atol=1e-7
    )


@pytest.mark.skipif(not _HAS_TABM_ORACLE, reason="tabm parity oracle is not installed")
def test_full_forward_matches_official_tabm_after_parameter_copy():
    import tabm as official_tabm

    torch.manual_seed(23)
    ours = _build_full()
    oracle = official_tabm.TabM.make(
        n_num_features=4,
        cat_cardinalities=None,
        d_out=2,
        arch_type="tabm",
        k=3,
        d_block=8,
        n_blocks=2,
        dropout=0.0,
    )

    def oracle_to_ours():
        with torch.no_grad():
            for ours_layer, oracle_block in zip(
                ours.backbone, oracle.backbone.blocks, strict=True
            ):
                oracle_layer = oracle_block[0]
                ours_layer.weight.copy_(oracle_layer.weight)
                ours_layer.r.copy_(oracle_layer.r)
                ours_layer.s.copy_(oracle_layer.s)
                ours_layer.bias.copy_(oracle_layer.bias)
            ours.output_layer.weight.copy_(oracle.output.weight.transpose(1, 2))
            ours.output_layer.bias.copy_(oracle.output.bias)

    def ours_to_oracle():
        with torch.no_grad():
            for ours_layer, oracle_block in zip(
                ours.backbone, oracle.backbone.blocks, strict=True
            ):
                oracle_layer = oracle_block[0]
                oracle_layer.weight.copy_(ours_layer.weight)
                oracle_layer.r.copy_(ours_layer.r)
                oracle_layer.s.copy_(ours_layer.s)
                oracle_layer.bias.copy_(ours_layer.bias)
            oracle.output.weight.copy_(ours.output_layer.weight.transpose(1, 2))
            oracle.output.bias.copy_(ours.output_layer.bias)

    x = torch.randn(19, 4)
    oracle_to_ours()
    torch.testing.assert_close(
        ours(x, _empty_cat(len(x))), oracle(x_num=x), rtol=1e-6, atol=1e-7
    )
    with torch.no_grad():
        ours.backbone[0].weight.add_(0.125)
        ours.backbone[1].bias.sub_(0.25)
        ours.output_layer.weight.mul_(0.75)
    ours_to_oracle()
    expected = oracle(x_num=x)
    actual = ours(x, _empty_cat(len(x)))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("embedded", [False, True])
def test_full_tabm_initialization_is_chunked_and_official(embedded):
    torch.manual_seed(5)
    params = {
        "variant": "full",
        "k": 64,
        "d": 16,
        "n_blocks": 2,
        "dropout": 0.0,
        "cat_emb_dim": 3,
    }
    model = build_model(
        "tabm",
        params,
        n_num=2,
        cat_cardinalities=[5],
        out_dim=2,
        num_embedding="pl" if embedded else None,
    )
    first, second = model.backbone
    starts = np.cumsum([0, *model.embedding.feature_chunk_sizes])
    for left, right in zip(starts[:-1], starts[1:], strict=True):
        torch.testing.assert_close(
            first.r[:, left:right], first.r[:, left : left + 1].expand(-1, right - left)
        )
    if embedded:
        assert abs(float(first.r.detach().mean())) < 0.25
        assert 0.6 < float(first.r.detach().std()) < 1.4
    else:
        assert set(first.r.unique().tolist()) <= {-1.0, 1.0}
    torch.testing.assert_close(first.s, torch.ones_like(first.s))
    torch.testing.assert_close(second.r, torch.ones_like(second.r))
    torch.testing.assert_close(second.s, torch.ones_like(second.s))
    torch.testing.assert_close(first.bias, first.bias[:1].expand_as(first.bias))
    torch.testing.assert_close(second.bias, second.bias[:1].expand_as(second.bias))
    head_bound = 1.0 / math.sqrt(16)
    assert float(model.output_layer.weight.detach().abs().max()) <= head_bound
    assert float(model.output_layer.bias.detach().abs().max()) <= head_bound
    assert torch.count_nonzero(model.output_layer.bias) > 0


def test_full_first_adapter_treats_onehot_block_as_one_feature():
    import pandas as pd

    frame = pd.DataFrame(
        {"number": np.arange(12, dtype=float), "color": ["r", "g", "b"] * 4}
    )
    pre = TabularPreprocessor(
        numeric_scaler="none", categorical_features=["color"], cat_encoding="onehot"
    ).fit(frame)
    x_num, _ = pre.transform(frame)
    model = build_model(
        "tabm",
        {
            "variant": "full",
            "k": 8,
            "d": 8,
            "n_blocks": 1,
            "_num_input_chunks": pre.numeric_chunk_sizes(),
        },
        n_num=x_num.shape[1],
        cat_cardinalities=[],
        out_dim=1,
    )
    assert model.embedding.feature_chunk_sizes == [1, 3]
    onehot_r = model.backbone[0].r[:, 1:]
    torch.testing.assert_close(onehot_r, onehot_r[:, :1].expand_as(onehot_r))


def test_full_default_depth_depends_on_numeric_embedding():
    raw = build_model("tabm", {"variant": "full", "d": 8}, 3, [], 1)
    embedded = build_model(
        "tabm", {"variant": "full", "d": 8}, 3, [], 1, num_embedding="pl"
    )
    mini = build_model("tabm", {"variant": "mini", "d": 8}, 3, [], 1)
    assert len(raw.backbone) == 3
    assert len(embedded.backbone) == 2
    assert len(mini.backbone) == 3


@pytest.mark.skipif(not _HAS_PLE_ORACLE, reason="PLE parity oracle is not installed")
def test_ple_quantile_bins_and_fixed_encoding_match_oracle():
    import rtdl_num_embeddings as oracle

    from masamlp.models.base import PiecewiseLinearEmbedding, compute_quantile_bins

    x = torch.tensor(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 3.0],
            [2.0, 4.0],
            [4.0, 9.0],
        ]
    )
    bins = compute_quantile_bins(x, n_bins=4)
    expected_bins = oracle.compute_bins(x, n_bins=4)
    for actual, expected in zip(bins, expected_bins, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    ours = PiecewiseLinearEmbedding(bins, d_embedding=3)
    theirs = oracle.PiecewiseLinearEmbeddings(
        expected_bins, d_embedding=3, activation=False, version="B"
    )
    probe = torch.tensor([[-20.0, -2.0], [-0.25, 0.5], [20.0, 20.0]])
    torch.testing.assert_close(ours.encode(probe), theirs.impl(probe), rtol=0, atol=0)


def test_ple_zero_projection_starts_as_linear_embedding():
    from masamlp.models.base import PiecewiseLinearEmbedding, compute_quantile_bins

    bins = compute_quantile_bins(torch.randn(64, 3), n_bins=8)
    embedding = PiecewiseLinearEmbedding(bins, d_embedding=5)
    x = torch.randn(11, 3)
    assert torch.count_nonzero(embedding.ple_weight) == 0
    torch.testing.assert_close(embedding(x), embedding.linear(x), rtol=0, atol=0)


def test_ple_estimator_serialization_roundtrip_and_fitting_rows_only(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(96, 3))
    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
    val = np.full((8, 3), 1e6)
    model = MasaClassifier(
        model="tabm",
        model_params={
            "variant": "full",
            "k": 2,
            "d": 12,
            "n_blocks": 1,
            "n_bins": 8,
            "d_num_embedding": 4,
        },
        num_embedding="ple",
        numeric_scaler="none",
        categorical_features=None,
        n_epochs=2,
        device="cpu",
        random_state=3,
    ).fit(X, y, eval_set=[(val, np.zeros(len(val), dtype=int))])
    bins = model.model_.embedding.num_embedding.bins
    for j, edges in enumerate(bins):
        assert float(edges[-1]) == pytest.approx(float(X[:, j].max()))
    expected = model.predict_proba(X[:12])
    model.save_model(tmp_path / "ple")
    loaded = MasaClassifier.load_model(tmp_path / "ple")
    np.testing.assert_array_equal(loaded.predict_proba(X[:12]), expected)
    for actual, wanted in zip(
        loaded.model_.embedding.num_embedding.bins, bins, strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_independent_member_batches_are_aligned_and_static():
    from masamlp.core.trainer import _make_epoch_batches

    gen = torch.Generator().manual_seed(41)
    batches = _make_epoch_batches(
        n_rows=24,
        batch_size=6,
        device=torch.device("cpu"),
        generator=gen,
        drop_last=True,
        n_members=4,
        share_training_batches=False,
    )
    assert [tuple(idx.shape) for idx in batches] == [(6, 4)] * 4
    all_idx = torch.cat(batches)
    for member in range(4):
        torch.testing.assert_close(torch.sort(all_idx[:, member]).values, torch.arange(24))
    assert not torch.equal(all_idx[:, 0], all_idx[:, 1])

    row_id = torch.arange(24, dtype=torch.float32)
    data = TabularData(
        x_num=row_id[:, None],
        x_cat=torch.zeros(24, 0, dtype=torch.long),
        y=2.0 * row_id + 1.0,
        weight=3.0 * row_id + 2.0,
    )
    gathered = data.slice(batches[0])
    torch.testing.assert_close(gathered.y, 2.0 * gathered.x_num[..., 0] + 1.0)
    torch.testing.assert_close(gathered.weight, 3.0 * gathered.x_num[..., 0] + 2.0)


def test_independent_member_batches_train_and_reduce_loss():
    from masamlp.core.trainer import TrainerConfig

    assert TrainerConfig().share_training_batches is True
    rng = np.random.default_rng(9)
    X = rng.normal(size=(160, 4))
    y = (1.5 * X[:, 0] - X[:, 1]).astype(np.float32)
    model = MasaClassifier(
        model="tabm",
        model_params={"variant": "full", "k": 4, "d": 16, "n_blocks": 1, "dropout": 0.0},
        share_training_batches=False,
        n_epochs=12,
        batch_size=32,
        learning_rate=5e-3,
        numeric_scaler="none",
        categorical_features=None,
        device="cpu",
        random_state=0,
    ).fit(X, y > np.median(y), eval_set=[(X, y > np.median(y))])
    losses = model.evals_result_["valid_0"]["logloss"]
    assert losses[-1] < losses[0]
