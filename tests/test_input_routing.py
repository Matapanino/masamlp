"""Opt-in input routing (0.9.1): selective numeric embedding and the additive
linear skip — and the guarantee that leaving both off changes nothing.

Two independent options, both defaulting to ``None`` (today's behaviour):

* ``num_embedding_cols`` — the numeric embedding (``"pbld"``/``"plr"``/...)
  applies only to the named numeric columns; every other numeric column
  bypasses it and enters the first layer linearly.
* ``linear_skip_cols`` (+ ``linear_skip_lr_factor``) — a linear map from the
  named (scaled) numeric columns straight onto the output logits, in its own
  optimizer param group with zero weight decay.

The no-change line is held the way ``test_realmlp_fidelity`` holds it: same
process, same seed, both code paths compared bit-for-bit — see
``test_defaults_are_the_unrouted_model`` (module level) and the two
``test_end_to_end_*_is_bit_identical`` estimator-level cases.  The pre-change
reference itself is ``tests/test_realmlp_fidelity.py``'s 0.8.0 fit fixture,
which this change must leave untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.models import build_model, model_accepts
from masamlp.models.base import FeatureEmbedding
from masamlp.regressor import MasaRegressor

_TINY = {"hidden_sizes": [16, 16], "num_scaling": True}
_EMB = {"d_num_embedding": 4, "n_frequencies": 8}


def _routing_frame(n: int = 240, seed: int = 0):
    """5 numeric columns + 1 categorical; the target uses both blocks."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "raw_a": rng.normal(size=n),
            "te_b": rng.normal(size=n),
            "raw_c": rng.normal(size=n),
            "te_d": rng.normal(size=n),
            "cat_e": rng.choice(["x", "y", "z"], n),
            "te_f": rng.normal(size=n),
        }
    )
    logit = (
        1.5 * df["raw_a"].to_numpy()
        + 1.0 * df["te_b"].to_numpy()
        - 0.8 * df["te_f"].to_numpy()
        + np.where(df["cat_e"].to_numpy() == "x", 1.0, -0.5)
    )
    y = (logit + rng.normal(0, 0.3, n) > 0).astype(int)
    return df, y


# ---------------------------------------------------------------------------
# The no-change guarantee
# ---------------------------------------------------------------------------
def test_defaults_are_the_unrouted_model():
    """Spelling every 0.9.1 routing argument out at its default must be
    bit-for-bit the same model as not passing them at all."""
    defaults = {
        "num_embedding_idx": None,
        "linear_skip_idx": None,
        "linear_skip_lr_factor": 1.0,
    }
    torch.manual_seed(7)
    plain = build_model("realmlp", {**_TINY, **_EMB}, 5, [3], 2, "pbld")
    torch.manual_seed(7)
    spelled = build_model("realmlp", {**_TINY, **_EMB, **defaults}, 5, [3], 2, "pbld")

    a, b = dict(plain.named_parameters()), dict(spelled.named_parameters())
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs"
    assert plain.state_dict().keys() == spelled.state_dict().keys()

    x_num, x_cat = torch.randn(8, 5), torch.randint(0, 3, (8, 1))
    plain.eval(), spelled.eval()
    with torch.no_grad():
        assert torch.equal(plain(x_num, x_cat), spelled(x_num, x_cat))

    def signature(model):
        return [
            (g["lr_factor"], g.get("wd_factor", 1.0), len(g["params"]))
            for g in model.param_groups()
        ]

    assert signature(plain) == signature(spelled)
    # No routing => no extra state and no extra buffers on the module tree.
    assert "skip_weight" not in dict(plain.named_parameters())
    assert not [n for n in dict(plain.named_buffers()) if "idx" in n]


def test_full_coverage_routing_is_the_unrouted_path():
    """Naming *every* numeric column reproduces the unrouted forward exactly:
    the routed path degenerates to today's path, so the option can only ever
    cost what it is asked to change."""
    torch.manual_seed(3)
    plain = FeatureEmbedding(5, [3], num_embedding="pbld", **_EMB)
    torch.manual_seed(3)
    routed = FeatureEmbedding(
        5, [3], num_embedding="pbld", num_embedding_idx=[0, 1, 2, 3, 4], **_EMB
    )
    assert routed.d_out == plain.d_out
    x_num, x_cat = torch.randn(16, 5), torch.randint(0, 3, (16, 1))
    with torch.no_grad():
        assert torch.equal(plain(x_num, x_cat), routed(x_num, x_cat))


def test_skip_does_not_shift_the_rng_stream():
    """The skip's tensors are zero-initialized, so adding it draws no random
    numbers: every other parameter is bit-identical to the model without it."""
    torch.manual_seed(11)
    plain = build_model("realmlp", {**_TINY, **_EMB}, 5, [3], 2, "pbld")
    torch.manual_seed(11)
    skipped = build_model(
        "realmlp", {**_TINY, **_EMB, "linear_skip_idx": [1, 3]}, 5, [3], 2, "pbld"
    )
    shared = dict(plain.named_parameters())
    got = dict(skipped.named_parameters())
    assert set(got) - set(shared) == {"skip_weight", "skip_bias"}
    for name, tensor in shared.items():
        assert torch.equal(tensor, got[name]), f"{name} differs"
    # ...and at init the skip contributes exactly nothing.
    x_num, x_cat = torch.randn(8, 5), torch.randint(0, 3, (8, 1))
    plain.eval(), skipped.eval()
    with torch.no_grad():
        assert torch.equal(plain(x_num, x_cat), skipped(x_num, x_cat))


# ---------------------------------------------------------------------------
# (A) Selective numeric embedding
# ---------------------------------------------------------------------------
def test_selective_embedding_widths_and_block_order():
    """Embedded columns first (``len(idx) * d_num_embedding`` wide), the
    bypassing numerics next in their original order, categoricals last."""
    emb = FeatureEmbedding(
        5, [4], num_embedding="pbld", num_embedding_idx=[0, 2], **_EMB
    )
    cat_dim = emb.cat_embeddings[0].embedding_dim
    d_num = 2 * _EMB["d_num_embedding"] + 3
    assert emb.d_out == d_num + cat_dim

    x_num = torch.randn(7, 5)
    x_cat = torch.randint(0, 4, (7, 1))
    out = emb(x_num, x_cat)
    assert out.shape == (7, emb.d_out)
    # The bypass block is the raw (scaled) input, untouched and in order.
    bypass = out[:, 2 * _EMB["d_num_embedding"] : d_num]
    assert torch.equal(bypass, x_num[:, [1, 3, 4]])


def test_selective_embedding_respects_the_scaling_layer():
    """``num_scaling`` still gains *all* numerics — the routing only decides
    what the embedding sees, not what the scale sees."""
    emb = FeatureEmbedding(
        4, [], num_embedding="plr", num_embedding_idx=[0], num_scaling=True, **_EMB
    )
    with torch.no_grad():
        emb.scaling.scale.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x_num = torch.randn(6, 4)
    out = emb(x_num, torch.zeros(6, 0, dtype=torch.int64))
    bypass = out[:, _EMB["d_num_embedding"] :]
    assert torch.allclose(bypass, x_num[:, 1:] * torch.tensor([2.0, 3.0, 4.0]))


def test_selective_embedding_works_with_periodic():
    emb = FeatureEmbedding(
        4, [], num_embedding="periodic", num_embedding_idx=[1, 2], n_frequencies=8
    )
    assert emb.d_out == 2 * 2 * 8 + 2
    out = emb(torch.randn(5, 4), torch.zeros(5, 0, dtype=torch.int64))
    assert out.shape == (5, emb.d_out)


def test_selective_embedding_index_validation():
    with pytest.raises(ValueError, match="num_embedding"):
        FeatureEmbedding(4, [], num_embedding_idx=[0])          # no num_embedding
    with pytest.raises(ValueError, match="out of range"):
        FeatureEmbedding(4, [], num_embedding="plr", num_embedding_idx=[4])
    with pytest.raises(ValueError, match="duplicate"):
        FeatureEmbedding(4, [], num_embedding="plr", num_embedding_idx=[1, 1])
    with pytest.raises(ValueError, match="at least one"):
        FeatureEmbedding(4, [], num_embedding="plr", num_embedding_idx=[])


def test_selective_embedding_rejected_for_token_models():
    with pytest.raises(ValueError, match="num_embedding_idx"):
        build_model(
            "ft_transformer",
            {"n_blocks": 1, "d_block": 32, "num_embedding_idx": [0]},
            4, [3], 1, "plr",
        )


# ---------------------------------------------------------------------------
# (B) Additive linear skip
# ---------------------------------------------------------------------------
def test_skip_is_a_linear_map_onto_the_output():
    model = build_model(
        "realmlp", {**_TINY, "linear_skip_idx": [0, 2]}, 4, [3], 3, None
    )
    model.eval()
    x_num, x_cat = torch.randn(9, 4), torch.randint(0, 3, (9, 1))
    with torch.no_grad():
        base = model(x_num, x_cat).clone()
        model.skip_weight.copy_(torch.randn(2, 3))
        model.skip_bias.copy_(torch.randn(3))
        moved = model(x_num, x_cat)
        expected = base + x_num[:, [0, 2]] @ model.skip_weight + model.skip_bias
    assert torch.allclose(moved, expected, atol=0, rtol=0)


def test_skip_gradient_reaches_the_weight():
    model = build_model(
        "realmlp", {**_TINY, "linear_skip_idx": [1, 3]}, 5, [3], 1, None
    )
    x_num, x_cat = torch.randn(12, 5), torch.randint(0, 3, (12, 1))
    out = model(x_num, x_cat)
    grad_out = torch.randn_like(out)
    (out * grad_out).sum().backward()
    assert model.skip_weight.grad is not None
    # d(sum g*out)/dW = x_skip^T g  — exactly, because the skip is linear.
    expected = x_num[:, [1, 3]].T @ grad_out
    assert torch.allclose(model.skip_weight.grad, expected, atol=1e-5)
    assert torch.allclose(model.skip_bias.grad, grad_out.sum(dim=0), atol=1e-5)


def test_skip_has_its_own_param_group():
    model = build_model(
        "realmlp",
        {**_TINY, "linear_skip_idx": [0], "linear_skip_lr_factor": 0.25},
        4, [3], 1, None,
    )
    groups = model.param_groups()
    skip = [
        g for g in groups
        if any(p is model.skip_weight for p in g["params"])
    ]
    assert len(skip) == 1
    assert skip[0]["lr_factor"] == 0.25
    assert skip[0]["wd_factor"] == 0.0
    assert {id(p) for p in skip[0]["params"]} == {
        id(model.skip_weight), id(model.skip_bias)
    }
    # every parameter is still accounted for exactly once
    assert sum(len(g["params"]) for g in groups) == sum(
        1 for p in model.parameters() if p.requires_grad
    )


def test_skip_index_validation():
    with pytest.raises(ValueError, match="out of range"):
        build_model("realmlp", {**_TINY, "linear_skip_idx": [5]}, 4, [3], 1, None)
    with pytest.raises(ValueError, match="duplicate"):
        build_model("realmlp", {**_TINY, "linear_skip_idx": [1, 1]}, 4, [3], 1, None)
    with pytest.raises(ValueError, match="at least one"):
        build_model("realmlp", {**_TINY, "linear_skip_idx": []}, 4, [3], 1, None)


# ---------------------------------------------------------------------------
# Name resolution at the estimator
# ---------------------------------------------------------------------------
def test_column_names_resolve_to_numeric_block_positions():
    """Names resolve against the *numeric block* the model sees. One-hot
    categorical columns are appended after it, so they never shift a
    numeric column's position."""
    df, y = _routing_frame(120)
    m = MasaClassifier(
        model="realmlp",
        model_params={**_TINY, **_EMB},
        num_embedding="pbld",
        cat_encoding="onehot",
        num_embedding_cols=["te_f", "raw_a"],
        linear_skip_cols=["te_b", "te_d"],
        linear_skip_lr_factor=0.5,
        n_epochs=2,
        device="cpu",
        random_state=0,
    ).fit(df, y)
    # numeric block order: raw_a, te_b, raw_c, te_d, te_f, then the one-hots
    assert m.resolved_model_params_["num_embedding_idx"] == [0, 4]
    assert m.resolved_model_params_["linear_skip_idx"] == [1, 3]
    assert m.resolved_model_params_["linear_skip_lr_factor"] == 0.5
    assert m.model_.skip_weight.shape == (2, 1)
    assert np.isfinite(m.predict_proba(df)).all()


def test_name_resolution_errors():
    df, y = _routing_frame(80)
    base = dict(
        model="realmlp", model_params=dict(_TINY), n_epochs=1, device="cpu",
        random_state=0,
    )
    with pytest.raises(ValueError, match="unknown column"):
        MasaClassifier(**base, num_embedding="plr",
                       num_embedding_cols=["nope"]).fit(df, y)
    with pytest.raises(ValueError, match="categorical"):
        MasaClassifier(**base, linear_skip_cols=["cat_e"]).fit(df, y)
    with pytest.raises(ValueError, match="duplicate"):
        MasaClassifier(**base, linear_skip_cols=["te_b", "te_b"]).fit(df, y)
    with pytest.raises(ValueError, match="at least one"):
        MasaClassifier(**base, linear_skip_cols=[]).fit(df, y)
    with pytest.raises(ValueError, match="list of column names"):
        MasaClassifier(**base, linear_skip_cols="te_b").fit(df, y)


def test_num_embedding_cols_needs_num_embedding():
    df, y = _routing_frame(80)
    with pytest.raises(ValueError, match="num_embedding"):
        MasaClassifier(
            model="realmlp", model_params=dict(_TINY), n_epochs=1, device="cpu",
            num_embedding_cols=["raw_a"],
        ).fit(df, y)


def test_linear_skip_cols_needs_a_supporting_model():
    df, y = _routing_frame(80)
    assert model_accepts("realmlp", "linear_skip_idx")
    assert not model_accepts("resnet", "linear_skip_idx")
    with pytest.raises(ValueError, match="linear_skip_cols"):
        MasaClassifier(
            model="resnet", model_params={"d": 16, "n_blocks": 1}, n_epochs=1,
            device="cpu", linear_skip_cols=["raw_a"],
        ).fit(df, y)


def test_routing_works_on_unnamed_arrays(reg_data):
    """A plain ndarray gets pandas' positional column names, so routing is
    still addressable without a DataFrame."""
    X, y, X_test, _ = reg_data
    m = MasaRegressor(
        model="realmlp", model_params={**_TINY, **_EMB}, num_embedding="plr",
        num_embedding_cols=["0", "1"], linear_skip_cols=["2"],
        n_epochs=3, device="cpu", random_state=0,
    ).fit(X, y)
    assert m.resolved_model_params_["num_embedding_idx"] == [0, 1]
    assert m.resolved_model_params_["linear_skip_idx"] == [2]
    assert np.isfinite(m.predict(X_test)).all()


# ---------------------------------------------------------------------------
# End-to-end: the routed estimator, and the two exact-identity controls
# ---------------------------------------------------------------------------
def test_end_to_end_full_coverage_is_bit_identical():
    """``num_embedding_cols`` naming every numeric column must reproduce the
    unrouted fit exactly — same process, same seed, same predictions.

    ``cat_encoding="embedding"`` here so the model's numeric block *is* the
    five named columns: under ``"onehot"``/``"hybrid"`` the one-hot columns
    join that block and naming only the real numerics is a genuine change
    (the one-hots then bypass the embedding)."""
    df, y = _routing_frame(200)
    numeric = ["raw_a", "te_b", "raw_c", "te_d", "te_f"]
    common = dict(
        model="realmlp", model_params={**_TINY, **_EMB}, num_embedding="pbld",
        n_epochs=4, batch_size=64, learning_rate=0.03,
        optimizer="adam", device="cpu", random_state=0,
    )
    plain = MasaClassifier(**common).fit(df, y)
    routed = MasaClassifier(**common, num_embedding_cols=numeric).fit(df, y)
    np.testing.assert_array_equal(plain.predict_proba(df), routed.predict_proba(df))


def test_end_to_end_frozen_skip_is_bit_identical():
    """A skip trained at ``linear_skip_lr_factor=0`` never leaves its zero
    init, so it adds exactly 0 to every logit: the fit must be bit-identical
    to the model without a skip. That isolates the skip's *only* effect to
    the parameters it is given a learning rate for."""
    df, y = _routing_frame(200)
    common = dict(
        model="realmlp", model_params={**_TINY, **_EMB}, num_embedding="pbld",
        cat_encoding="onehot", n_epochs=4, batch_size=64, learning_rate=0.03,
        optimizer="adam", weight_decay=1e-2, device="cpu", random_state=0,
    )
    plain = MasaClassifier(**common).fit(df, y)
    frozen = MasaClassifier(
        **common, linear_skip_cols=["te_b", "te_d"], linear_skip_lr_factor=0.0
    ).fit(df, y)
    np.testing.assert_array_equal(frozen.model_.skip_weight.detach().numpy(),
                                  np.zeros((2, 1), dtype=np.float32))
    np.testing.assert_array_equal(plain.predict_proba(df), frozen.predict_proba(df))


def test_trained_skip_moves_and_learns():
    df, y = _routing_frame(400)
    m = MasaClassifier(
        model="realmlp", model_params={**_TINY, **_EMB}, num_embedding="pbld",
        cat_encoding="onehot", linear_skip_cols=["raw_a", "te_b"],
        linear_skip_lr_factor=2.0, n_epochs=25, batch_size=64,
        learning_rate=0.03, optimizer="adam", device="cpu", random_state=0,
    ).fit(df, y)
    from sklearn.metrics import roc_auc_score

    assert float(np.abs(m.model_.skip_weight.detach().numpy()).max()) > 0.0
    assert roc_auc_score(y, m.predict_proba(df)[:, 1]) > 0.85


def test_routing_survives_save_load(tmp_path):
    df, y = _routing_frame(150)
    m = MasaClassifier(
        model="realmlp", model_params={**_TINY, **_EMB}, num_embedding="pbld",
        cat_encoding="onehot", num_embedding_cols=["raw_a", "raw_c"],
        linear_skip_cols=["te_b", "te_f"], linear_skip_lr_factor=0.5,
        n_epochs=5, batch_size=64, learning_rate=0.03, optimizer="adam",
        device="cpu", random_state=0,
    ).fit(df, y)
    m.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    assert loaded.resolved_model_params_["linear_skip_idx"] == [1, 4]
    np.testing.assert_array_equal(m.predict_proba(df), loaded.predict_proba(df))


def test_routing_trains_vectorized():
    """``ens_mode="vectorized"`` stacks the routing index buffers across
    members; each member must still select its own columns."""
    df, y = _routing_frame(200)
    m = MasaClassifier(
        model="realmlp", model_params={**_TINY, **_EMB}, num_embedding="pbld",
        cat_encoding="onehot", num_embedding_cols=["raw_a", "raw_c"],
        linear_skip_cols=["te_b"], n_ens=3, ens_mode="vectorized",
        n_epochs=4, batch_size=64, learning_rate=0.03, optimizer="adam",
        device="cpu", random_state=0,
    ).fit(df, y)
    proba = m.predict_proba(df)
    assert np.isfinite(proba).all()
    members = m.predict_proba_members(df)
    assert members.shape == (len(df), 3, 2)
    assert not np.allclose(members[:, 0, 1], members[:, 1, 1])


# ---------------------------------------------------------------------------
# XLA safety by construction
# ---------------------------------------------------------------------------
def test_index_selectors_are_device_buffers_with_static_shapes():
    """Column selection goes through registered int64 buffers — they move
    with ``.to(device)``, carry no autograd, and stay out of the state_dict
    (they are rebuilt from ``resolved_model_params_``). Nothing in the
    forward pass depends on the *values* of the data, so every batch
    compiles to one XLA graph."""
    model = build_model(
        "realmlp",
        {**_TINY, **_EMB, "num_embedding_idx": [0, 2], "linear_skip_idx": [1, 3]},
        5, [3], 2, "pbld",
    )
    buffers = dict(model.named_buffers())
    names = {"_skip_idx", "embedding._emb_idx", "embedding._rest_idx"}
    assert names <= set(buffers)
    for name in names:
        assert buffers[name].dtype == torch.int64
        assert not buffers[name].requires_grad
    assert not (names & set(model.state_dict()))  # non-persistent

    model.eval()
    with torch.no_grad():
        for n_rows in (1, 3, 17):
            out = model(torch.randn(n_rows, 5), torch.randint(0, 3, (n_rows, 1)))
            assert out.shape == (n_rows, 2)


def test_forward_has_no_host_synchronization():
    """A guard on the XLA rule the routing must not break: no ``.item()`` /
    ``.tolist()`` / boolean-mask indexing anywhere on the forward path — those
    force a host sync and a data-dependent shape."""
    import inspect

    from masamlp.models.base import FeatureEmbedding as FE
    from masamlp.models.realmlp import RealMLPNet

    for fn in (FE.forward, RealMLPNet.forward):
        src = inspect.getsource(fn)
        for banned in (".item()", ".tolist()", ".nonzero(", "masked_select"):
            assert banned not in src, f"{fn.__qualname__} uses {banned}"
