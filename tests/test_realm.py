"""RealM — the RealMLP backbone under TabM-style full BatchEnsemble.

What is pinned here is the *structure*, not the accuracy (ensembling quality
is measured at scale in a competition campaign, not on toy data):

* the k=1 reduction — bit-equality with ``realmlp`` at init, and a
  bit-identical training trajectory once the rank-1 adapters are frozen;
* the BatchEnsemble algebra, against hand-computed values;
* what is shared (one W, one PBLD, one scale, one alpha per layer) and what
  is per member (adapters, biases, heads), by parameter count;
* that members actually diverge, and that dropout decorrelates them;
* EMA + joint best-epoch restore reaching every k-shaped tensor;
* the (n, k, out) objective path and the estimator round-trips.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest
import torch
from sklearn.base import clone

from masamlp.classifier import MasaClassifier
from masamlp.core.objectives import BinaryLogistic, SquaredError, apply_transform
from masamlp.core.trainer import weighted_loss
from masamlp.models import _MODEL_REGISTRY, build_model
from masamlp.models.realm import BatchEnsembleLinear, EnsembleNTPHead, RealMNet
from masamlp.presets import realm_td_params, realmlp_td_params
from masamlp.regressor import MasaRegressor

# A small TD-flavoured architecture: PBLD embedding, parametric activations,
# scheduled dropout, the data-driven init and the relocated front scale — so
# the parity tests below exercise every RealMLP path, not a bare MLP.
_TD_ARCH = dict(
    hidden_sizes=[16, 12],
    num_scaling=True,
    use_parametric_act=True,
    init_mode="std+he5",
    zero_init_output=False,
    scale_position="first_layer",
    dropout=0.1,
    dropout_schedule="flat_cos",
    d_num_embedding=4,
    n_frequencies=8,
)


def _inputs(n=32, n_num=5, cards=(3, 4)):
    x_num = torch.randn(n, n_num)
    x_cat = torch.stack([torch.randint(0, c, (n,)) for c in cards], dim=1)
    return x_num, x_cat


def _build(name, seed=0, **extra):
    torch.manual_seed(seed)
    return build_model(
        name, {**_TD_ARCH, **extra}, n_num=5, cat_cardinalities=[3, 4],
        out_dim=2, num_embedding="pbld",
    )


def test_realm_registered():
    assert "realm" in _MODEL_REGISTRY


# --------------------------------------------------------------------- #
# (a) The k=1 reduction to RealMLP.
# --------------------------------------------------------------------- #

def test_k1_ones_is_bit_identical_to_realmlp_at_init():
    """With k=1 and the adapters at one, RealM *is* RealMLP: same draws in
    the same order (BatchEnsembleLinear mirrors NTPLinear's ``randn(d_in,
    d_out)`` then ``randn(d_out)``), the same data-driven ``std+he5`` walk,
    and multiplication by exactly 1.0 is exact in IEEE-754 — so the shared
    state and the forward output are **bit**-equal, not merely close."""
    ref = _build("realmlp")
    got = _build("realm", k=1, member_init="ones")
    x_num, x_cat = _inputs()

    torch.manual_seed(1)
    ref.data_init(x_num, x_cat)
    torch.manual_seed(1)
    got.data_init(x_num, x_cat)

    ref_state, got_state = ref.state_dict(), got.state_dict()
    # Only the ensembling parameters are new; nothing else changed name.
    assert set(got_state) - set(ref_state) == {
        "first_adapter", "trunk.0.r", "trunk.0.s", "trunk.3.r", "trunk.3.s",
    }
    assert not set(ref_state) - set(got_state)
    for key, value in ref_state.items():
        assert torch.equal(value, got_state[key].reshape(value.shape)), key

    ref.eval()
    got.eval()
    out_ref, out_got = ref(x_num, x_cat), got(x_num, x_cat)
    assert out_got.shape == (32, 1, 2)
    assert torch.equal(out_ref, out_got.squeeze(1))


def _fit_pair():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    def fit(preset, extra):
        params = dict(preset)
        params["model_params"] = {
            **params["model_params"], "hidden_sizes": [16, 16],
            "d_num_embedding": 4, "n_frequencies": 8, **extra,
        }
        params.update(
            n_epochs=6, batch_size=64, device="cpu", random_state=0, ema_decay=0.9
        )
        return MasaClassifier(**params).fit(X, y)

    ref = fit(realmlp_td_params("classification", pytabkit_fidelity=True), {})
    got = fit(
        realm_td_params("classification", pytabkit_fidelity=True),
        {"k": 1, "member_init": "ones"},
    )
    return ref, got, X


def test_k1_trains_identically_to_realmlp_when_adapters_are_frozen(monkeypatch):
    """The *only* thing k=1 RealM adds to RealMLP is the rank-1 adapters as
    trainable parameters (an overparametrization of the same function). Pin
    that literally: freeze ``first_adapter``/``r``/``s`` at one and six
    epochs of the full TD recipe — coslog4 lr, flat_cos weight decay,
    scheduled dropout, EMA — reproduce RealMLP bit for bit. Unfrozen they
    move, so the fits legitimately diverge; that is the architecture, not a
    bug."""
    import masamlp.sklearn as sklearn_module

    build = sklearn_module.build_model

    def freezing_build(*args, **kwargs):
        model = build(*args, **kwargs)
        if isinstance(model, RealMNet):
            model.first_adapter.requires_grad_(False)
            for module in model.modules():
                if isinstance(module, BatchEnsembleLinear):
                    module.r.requires_grad_(False)
                    module.s.requires_grad_(False)
        return model

    monkeypatch.setattr(sklearn_module, "build_model", freezing_build)
    ref, got, X = _fit_pair()
    np.testing.assert_array_equal(ref.predict_proba(X), got.predict_proba(X))
    ref_state, got_state = ref.model_.state_dict(), got.model_.state_dict()
    for key, value in ref_state.items():
        assert torch.equal(value, got_state[key].reshape(value.shape)), key


def test_trainable_adapters_move_the_fit():
    """The counterpart of the frozen test: left trainable, the adapters
    receive real gradients, so the same seed no longer reproduces RealMLP."""
    ref, got, X = _fit_pair()
    assert not np.array_equal(ref.predict_proba(X), got.predict_proba(X))
    assert not torch.allclose(
        got.model_.trunk[0].r, torch.ones_like(got.model_.trunk[0].r)
    )


# --------------------------------------------------------------------- #
# (b) The BatchEnsemble algebra, by hand.
# --------------------------------------------------------------------- #

def test_batch_ensemble_linear_matches_materialized_member_weights():
    """``((x*r) @ W)/sqrt(d_in) * s + b`` must equal what member i would
    compute with its own weight matrix ``W ⊙ (r_i s_iᵀ)`` — the identity the
    layer exists to avoid materializing."""
    torch.manual_seed(0)
    k, d_in, d_out, n = 3, 5, 4, 7
    layer = BatchEnsembleLinear(d_in, d_out, k)
    with torch.no_grad():
        layer.r.normal_()
        layer.s.normal_()
        layer.bias.normal_()
    x = torch.randn(n, k, d_in)
    got = layer(x)
    assert got.shape == (n, k, d_out)
    for i in range(k):
        w_i = layer.weight * torch.outer(layer.r[i], layer.s[i])
        expected = (x[:, i] @ w_i) / math.sqrt(d_in) + layer.bias[i]
        torch.testing.assert_close(got[:, i], expected)


def test_batch_ensemble_linear_hand_computed():
    torch.manual_seed(0)
    layer = BatchEnsembleLinear(2, 2, k=2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
        layer.r.copy_(torch.tensor([[1.0, 1.0], [2.0, 0.0]]))
        layer.s.copy_(torch.tensor([[1.0, 1.0], [1.0, 3.0]]))
        layer.bias.copy_(torch.tensor([[0.0, 0.0], [1.0, -1.0]]))
    x = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])          # (1, 2, 2)
    # member 0: ([1,1] @ W)/sqrt(2) = [1, 2]/sqrt(2)
    # member 1: ([2,0] @ W)/sqrt(2) * [1,3] + [1,-1] = [2/sqrt(2)+1, -1]
    scale = 1.0 / math.sqrt(2.0)
    expected = torch.tensor([[[scale, 2 * scale], [2 * scale + 1.0, -1.0]]])
    torch.testing.assert_close(layer(x), expected)


def test_ensemble_head_is_per_member_and_ntp_scaled():
    torch.manual_seed(0)
    head = EnsembleNTPHead(d=3, out_dim=2, k=4)
    x = torch.randn(6, 4, 3)
    got = head(x)
    assert got.shape == (6, 4, 2)
    assert head.weight.shape == (4, 3, 2) and head.bias.shape == (4, 2)
    for i in range(4):
        expected = (x[:, i] @ head.weight[i]) / math.sqrt(3) + head.bias[i]
        torch.testing.assert_close(got[:, i], expected)


def test_first_adapter_shapes_and_member_init():
    signs = _build("realm", k=4, member_init="signs")
    assert signs.first_adapter.shape == (4, signs.embedding.d_out)
    assert set(signs.first_adapter.unique().tolist()) <= {-1.0, 1.0}
    ones = _build("realm", k=4, member_init="ones")
    assert torch.equal(ones.first_adapter, torch.ones_like(ones.first_adapter))
    # "auto" follows TabM's rule: normal with a numeric embedding, signs without.
    with_pbld = _build("realm", k=4)
    assert with_pbld.member_init == "normal"
    torch.manual_seed(0)
    bare = build_model("realm", {"hidden_sizes": [8], "k": 4}, 5, [3], 2, None)
    assert bare.member_init == "signs"


@pytest.mark.parametrize(
    "params, match",
    [
        ({"k": 0}, "k must be a positive int"),
        ({"member_init": "bogus"}, "Unknown member_init"),
        ({"activation": "bogus"}, "Unknown activation"),
    ],
)
def test_realm_validates_its_knobs(params, match):
    with pytest.raises(ValueError, match=match):
        build_model("realm", {"hidden_sizes": [8], **params}, 4, [], 2, None)


# --------------------------------------------------------------------- #
# (c) Members receive gradients and diverge.
# --------------------------------------------------------------------- #

def test_member_parameters_get_gradients_and_diverge():
    model = _build("realm", k=4, init_mode="ntp", dropout=0.0)
    x_num, x_cat = _inputs()
    y = torch.randn(32, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    model.train()
    before = model(x_num, x_cat).detach()
    loss = weighted_loss(SquaredError(), y, model(x_num, x_cat), None)
    loss.backward()
    per_member = [model.first_adapter, model.output_layer.weight, model.output_layer.bias]
    for module in model.modules():
        if isinstance(module, BatchEnsembleLinear):
            per_member += [module.r, module.s, module.bias]
    for param in per_member:
        assert param.grad is not None
        # Every member slice must be driven, not just member 0.
        assert (param.grad.abs().flatten(1).sum(dim=1) > 0).all()
    optimizer.step()

    after = model(x_num, x_cat).detach()
    assert not torch.allclose(before, after)
    # The members are genuinely different predictors.
    spread = (after - after.mean(dim=1, keepdim=True)).abs().max()
    assert float(spread) > 1e-4


# --------------------------------------------------------------------- #
# (d) What is shared vs per member — by parameter count.
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [1, 4, 8])
def test_parameter_count_is_one_backbone_plus_k_times_the_member_parts(k):
    model = _build("realm", k=k)
    e = model.embedding.d_out                          # PBLD + one-hot width
    embedded = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("embedding.")
    )
    # One backbone: the embedding, the front scale, one W per layer, one
    # activation alpha per layer.
    shared = embedded + e + (e * 16 + 16 * 12) + (16 + 12)
    # Per member: the first adapter, each layer's r/s/bias, and one head.
    per_member = e + (e + 16 + 16) + (16 + 12 + 12) + (12 * 2 + 2)
    total = sum(p.numel() for p in model.parameters())
    assert total == shared + k * per_member

    assert model.trunk[0].weight.shape == (e, 16)
    assert model.trunk[1].alpha.shape == (16,)
    assert model.front_scale.scale.shape == (e,)


#: Exactly the parameters that carry a leading member axis.
_MEMBER_PARAMS = {
    "first_adapter",
    "trunk.0.r", "trunk.0.s", "trunk.0.bias",
    "trunk.3.r", "trunk.3.s", "trunk.3.bias",
    "output_layer.weight", "output_layer.bias",
}


def test_only_the_member_parameters_grow_with_k():
    """The shared half — one W per layer, the PBLD embedding, both scaling
    layers, the activation alphas — must be *single* tensors whose shape does
    not depend on k."""
    small, big = _build("realm", k=1), _build("realm", k=8)
    shapes_1 = dict(small.named_parameters())
    shapes_8 = dict(big.named_parameters())
    assert set(shapes_1) == set(shapes_8)
    for name, param in shapes_1.items():
        other = shapes_8[name]
        if name in _MEMBER_PARAMS:
            assert param.shape[0] == 1 and other.shape[0] == 8, name
            assert param.shape[1:] == other.shape[1:], name
        else:
            assert param.shape == other.shape, name


def test_param_groups_reduce_to_realmlp_factors_at_adapter_lr_factor_one():
    model = _build("realm", k=4)
    groups = model.param_groups()
    assert sum(len(g["params"]) for g in groups) == sum(
        1 for p in model.parameters() if p.requires_grad
    )
    factors = {g["lr_factor"] for g in groups}
    assert factors == {6.0, 1.0, 0.1}          # scale, weights+adapters, biases+act+plr
    for group in groups:
        # Only the bias group is exempt from weight decay, as in RealMLP.
        assert group.get("wd_factor", 1.0) in (0.0, 1.0)

    priced = _build("realm", k=4, adapter_lr_factor=4.0)
    groups = priced.param_groups()
    def group_of(param):
        return next(g for g in groups if any(p is param for p in g["params"]))

    assert group_of(priced.first_adapter)["lr_factor"] == 4.0
    assert group_of(priced.trunk[0].r)["lr_factor"] == 4.0
    assert group_of(priced.output_layer.weight)["lr_factor"] == 4.0
    biases = group_of(priced.output_layer.bias)
    assert biases["lr_factor"] == pytest.approx(0.1 * 4.0)
    assert biases["wd_factor"] == 0.0


# --------------------------------------------------------------------- #
# (e) Dropout is independent across members; eval is deterministic.
# --------------------------------------------------------------------- #

def test_dropout_masks_differ_across_members_and_eval_is_deterministic():
    model = _build("realm", k=8, init_mode="ntp", dropout=0.5)
    x_num, x_cat = _inputs()
    model.train()
    torch.manual_seed(0)
    hidden = model.embedding(x_num, x_cat)
    hidden = model.front_scale(hidden).unsqueeze(1) * model.first_adapter
    hidden = model.trunk[:3](hidden)          # linear -> activation -> dropout
    dropped = hidden == 0
    assert dropped.shape == (32, 8, 16)
    assert dropped.any(), "dropout did not fire"
    # Member slices must not share a mask, or dropout cannot decorrelate them.
    assert not torch.equal(dropped[:, 0], dropped[:, 1])

    model.eval()
    torch.manual_seed(0)
    first = model(x_num, x_cat)
    torch.manual_seed(1)
    torch.testing.assert_close(model(x_num, x_cat), first, rtol=0, atol=0)


# --------------------------------------------------------------------- #
# (f) EMA + the single joint best epoch reach every k-shaped tensor.
# --------------------------------------------------------------------- #

def test_ema_and_best_epoch_restore_preserve_every_member_tensor(rng):
    """The trainer's EMA shadow and its best-epoch snapshot both walk
    ``named_parameters`` / ``state_dict``, so nothing has to be taught about
    the member axis — but the restored model must actually carry k-shaped
    tensors, and they must equal the snapshot for the epoch the *ensemble
    mean* scored best (one joint epoch, not k cherry-picked ones)."""
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = MasaClassifier(
        model="realm",
        model_params={"hidden_sizes": [16, 16], "k": 4},
        n_epochs=12,
        batch_size=128,
        learning_rate=0.05,
        optimizer="adam",
        early_stopping_rounds=20,
        eval_metric="auc",
        ema_decay=0.9,
        device="cpu",
        random_state=0,
    ).fit(X[:300], y[:300], eval_set=[(X[300:], y[300:])])

    assert m.best_iteration_ is not None
    # One joint best epoch, chosen on the member-averaged prediction.
    scores = m.evals_result_["valid_0"]["auc"]
    assert m.best_iteration_ == int(np.argmax(scores))

    state = m.model_.state_dict()
    member_shaped = [k for k, v in state.items() if v.dim() and v.shape[0] == 4]
    assert {"first_adapter", "output_layer.weight", "output_layer.bias"} <= set(
        member_shaped
    )
    assert all(torch.isfinite(v).all() for v in state.values())
    # The restore is the EMA copy, not the last step: with ema_decay the
    # weights that scored are the ones that survive.
    assert float(m.model_.first_adapter.detach().std()) > 0.0


# --------------------------------------------------------------------- #
# (g) The estimator: it learns, and it round-trips.
# --------------------------------------------------------------------- #

def _binary_problem(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] * X[:, 3]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X[: n // 2], y[: n // 2], X[n // 2 :], y[n // 2 :]


def test_realm_fits_and_beats_chance():
    from sklearn.metrics import roc_auc_score

    X, y, X_test, y_test = _binary_problem()
    m = MasaClassifier(
        model="realm",
        model_params={"hidden_sizes": [32, 32], "k": 4, "dropout": 0.05},
        n_epochs=20,
        batch_size=256,
        learning_rate=0.04,
        optimizer="adam",
        optimizer_betas=(0.9, 0.95),
        lr_scheduler="coslog4",
        device="cpu",
        random_state=0,
    ).fit(X, y)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert roc_auc_score(y_test, proba[:, 1]) > 0.75

    # predict_proba is the member mean on the probability scale.
    members = m.predict_proba_members(X_test)
    assert members.shape == (len(y_test), 4, 2)
    np.testing.assert_allclose(members.mean(axis=1), proba, atol=1e-6)


def test_realm_estimator_round_trips(tmp_path):
    X, y, X_test, _ = _binary_problem(n=400)
    est = MasaClassifier(
        model="realm", model_params={"hidden_sizes": [16], "k": 4},
        n_epochs=5, device="cpu", random_state=0,
    )
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()

    m = est.fit(X, y)
    expected = m.predict_proba(X_test)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(m)).predict_proba(X_test), expected
    )
    m.save_model(tmp_path / "m")
    np.testing.assert_array_equal(
        MasaClassifier.load_model(tmp_path / "m").predict_proba(X_test), expected
    )


def test_realm_regression_and_determinism(reg_data):
    X, y, X_test, y_test = reg_data
    kwargs = dict(
        model="realm", model_params={"hidden_sizes": [32, 32], "k": 4},
        n_epochs=40, learning_rate=0.05, optimizer="adam",
        optimizer_betas=(0.9, 0.95), lr_scheduler="coslog4",
        device="cpu", random_state=0,
    )
    m1 = MasaRegressor(**kwargs).fit(X, y)
    m2 = MasaRegressor(**kwargs).fit(X, y)
    np.testing.assert_array_equal(m1.predict(X_test), m2.predict(X_test))
    rmse = float(np.sqrt(np.mean((m1.predict(X_test) - y_test) ** 2)))
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < 0.8 * baseline


def test_realm_rejects_vectorized_outer_ensemble(clf_data):
    """``realm`` already vectorizes its members; stacking whole models on top
    is the wrong axis, and the vmap path freezes the scheduled-dropout
    buffer. It must say so instead of silently training something else. The
    outer axis still composes in the default loop mode."""
    X, y, X_test, _ = clf_data
    kwargs = dict(
        model="realm", model_params={"hidden_sizes": [16], "k": 2},
        n_ens=2, n_epochs=3, device="cpu", random_state=0,
    )
    with pytest.raises(ValueError, match="vectorized"):
        MasaClassifier(**kwargs, ens_mode="vectorized").fit(X, y)

    m = MasaClassifier(**kwargs, ens_mode="loop").fit(X, y)
    assert m.predict_proba_members(X_test).shape == (len(X_test), 4, 2)


def test_realm_rejects_linear_skip_with_a_clear_message(clf_data):
    """The additive linear skip lives on ``realmlp``; the estimator's
    ``model_accepts`` check must name that rather than raise a TypeError."""
    X, y, _, _ = clf_data
    with pytest.raises(ValueError, match="linear_skip_cols is not supported"):
        MasaClassifier(
            model="realm", model_params={"hidden_sizes": [16], "k": 2},
            linear_skip_cols=["0"], n_epochs=1, device="cpu",
        ).fit(X, y)


# --------------------------------------------------------------------- #
# (h) The (n, k, out) objective path.
# --------------------------------------------------------------------- #

def test_loss_is_the_mean_of_the_per_member_losses():
    """RealM trains on the mean of the member BCEs, never the BCE of the mean
    probability — the former gives every member a proper supervised gradient,
    the latter couples them and invites collapse."""
    torch.manual_seed(0)
    n, k = 40, 5
    raw = torch.randn(n, k, 1)
    y = torch.randint(0, 2, (n,)).float()
    obj = BinaryLogistic()

    got = weighted_loss(obj, y, raw, None)
    members = torch.stack([obj.per_sample_loss(y, raw[:, j]).mean() for j in range(k)])
    torch.testing.assert_close(got, members.mean())

    mean_prob = torch.sigmoid(raw).mean(dim=1)
    bce_of_mean = torch.nn.functional.binary_cross_entropy(mean_prob[:, 0], y)
    assert not torch.allclose(got, bce_of_mean)

    # Evaluation, by contrast, is on the member-mean probability.
    torch.testing.assert_close(apply_transform(raw, "sigmoid"), mean_prob)


def test_preset_is_the_td_recipe_plus_the_ensembling_knobs():
    td = realmlp_td_params("classification", pytabkit_fidelity=True)
    realm = realm_td_params("classification", pytabkit_fidelity=True, k=16)
    assert realm["model"] == "realm"
    skip = ("model", "model_params")
    assert {k: v for k, v in realm.items() if k not in skip} == {
        k: v for k, v in td.items() if k not in skip
    }
    added = {"k": 16, "adapter_lr_factor": 1.0, "member_init": "auto"}
    assert realm["model_params"] == {**td["model_params"], **added}
    with pytest.raises(ValueError):
        realm_td_params("clustering")


def test_realm_module_has_no_vmap_path():
    """``k`` is a static shape, not a transform: nothing in the module
    imports or calls ``torch.vmap`` / ``torch.func``, and neither the layer's
    nor the net's forward loops over members."""
    import ast
    import inspect

    from masamlp.models import realm as realm_module

    tree = ast.parse(inspect.getsource(realm_module))
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("func" in name or "vmap" in name for name in imported), imported
    called = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for banned in ("vmap", "stack_module_state", "functional_call"):
        assert banned not in called

    for fn in (BatchEnsembleLinear.forward, EnsembleNTPHead.forward, RealMNet.forward):
        body = ast.parse(inspect.getsource(fn).lstrip())
        assert not any(
            isinstance(node, ast.For | ast.While) for node in ast.walk(body)
        ), fn.__qualname__
