"""TabM (BatchEnsemble MLP) and the (n, k, out) inner-ensemble contract
(ADR 0005). Quality-of-ensembling is measured at scale in the S6E7 campaign,
not asserted on toy data; here we pin the masaMLP contracts: registration,
shapes, k=1/k>1, determinism, head-bias broadcast across members,
serialization, adapter diversity — and the contract's reach: every task
(binary/multiclass/regression), custom objectives (which must never see a
member dim), exact sample_weight semantics, and composition with the outer
n_ens axis (loop and vectorized)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.core.objectives import MulticlassSoftmax, SquaredError, apply_transform
from masamlp.core.trainer import weighted_loss
from masamlp.models import _MODEL_REGISTRY, build_model
from masamlp.regressor import MasaRegressor


def test_tabm_registered():
    assert "tabm" in _MODEL_REGISTRY


def _fit(multiclass_data, **params):
    X, y, X_test, _ = multiclass_data
    m = MasaClassifier(
        model="tabm",
        model_params={"d": 32, "n_blocks": 2, **params},
        n_epochs=5,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    return m, X_test


@pytest.mark.parametrize("k", [1, 8, 32])
def test_tabm_fit_predict_shapes(multiclass_data, k):
    m, X_test = _fit(multiclass_data, k=k)
    proba = m.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_tabm_determinism(multiclass_data):
    m1, X_test = _fit(multiclass_data, k=8)
    m2, _ = _fit(multiclass_data, k=8)
    np.testing.assert_array_equal(m1.predict_proba(X_test), m2.predict_proba(X_test))


def test_tabm_members_are_distinct(multiclass_data):
    """The k members must not be identical, or the ensemble is a no-op. The
    per-member input adapter is what guarantees it."""
    m, _ = _fit(multiclass_data, k=8)
    a = m.model_.adapter.detach().cpu().numpy()  # (k, d_out)
    assert not np.allclose(a, a[0], atol=1e-6), "all members share one adapter -> no ensemble"


def test_tabm_head_bias_broadcasts_across_members(multiclass_data):
    """The estimator inits output_layer.bias from the (out_dim,) class priors;
    with a per-member (k, out_dim) bias this must broadcast, not raise."""
    m, X_test = _fit(multiclass_data, k=8)
    assert m.model_.output_layer.bias.shape[0] == 8
    assert np.isfinite(m.predict_proba(X_test)).all()


def test_tabm_serialization_roundtrip(tmp_path, multiclass_data):
    m, X_test = _fit(multiclass_data, k=8)
    m.save_model(tmp_path / "m")
    loaded = MasaClassifier.load_model(tmp_path / "m")
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_tabm_rejects_unknown_param(multiclass_data):
    X, y, _, _ = multiclass_data
    with pytest.raises(ValueError, match="Unknown model_params"):
        MasaClassifier(model="tabm", model_params={"bogus": 1}, device="cpu").fit(X, y)


def test_tabm_build_model_direct():
    """build_model wires FeatureEmbedding + params; forward always exposes
    per-member outputs (n, k, out) — averaging is apply_transform's job."""
    model = build_model("tabm", {"k": 4, "d": 16, "n_blocks": 1}, n_num=6,
                        cat_cardinalities=[], out_dim=3, num_embedding="plr-lite")
    xn, xc = torch.randn(10, 6), torch.zeros(10, 0, dtype=torch.long)
    model.train()
    assert model(xn, xc).shape == (10, 4, 3)
    model.eval()
    out = model(xn, xc)
    assert out.shape == (10, 4, 3)
    proba = apply_transform(out, "softmax")
    assert proba.shape == (10, 3)
    assert torch.isfinite(proba).all()
    torch.testing.assert_close(proba.sum(dim=-1), torch.ones(10))


# --------------------------------------------------------------------- #
# The (n, k, out) contract: every task, custom objectives, sample_weight,
# and composition with the outer n_ens axis (ADR 0005).
# --------------------------------------------------------------------- #

def test_tabm_binary_classification(clf_data):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(
        model="tabm", model_params={"k": 4, "d": 32, "n_blocks": 2},
        n_epochs=40, device="cpu", random_state=0,
    ).fit(X, y)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert float((m.predict(X_test) == y_test).mean()) > 0.8


def test_tabm_regression(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(
        model="tabm", model_params={"k": 4, "d": 32, "n_blocks": 2},
        n_epochs=40, learning_rate=3e-3, device="cpu", random_state=0,
    ).fit(X, y)
    pred = m.predict(X_test)
    assert pred.shape == y_test.shape
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < 0.8 * baseline


def test_tabm_custom_objective_sees_per_sample_contract(reg_data):
    """Inner ensembling must be invisible to custom objectives: the trainer
    flattens members into rows, so the custom fn keeps the standard
    ``(n, out) -> (n,)`` contract and never sees a member dim."""
    X, y, _, _ = reg_data
    seen_ndims: set[int] = set()

    def pseudo_huber(y_true: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        seen_ndims.add(raw.ndim)
        err = raw - y_true
        return (torch.sqrt(1.0 + err * err) - 1.0).mean(dim=1)

    m = MasaRegressor(
        model="tabm", model_params={"k": 4, "d": 16, "n_blocks": 1},
        objective=pseudo_huber, n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    assert seen_ndims == {2}
    assert np.isfinite(m.predict(X[:10])).all()


_TABM_ISOLATED = dict(
    model="tabm",
    model_params={"k": 4, "d": 16, "n_blocks": 1, "dropout": 0.0},
    numeric_scaler="none",
    target_standardize=False,
    n_epochs=20,
    batch_size=None,  # full batch
    device="cpu",
    random_state=7,
)


def test_tabm_integer_weights_match_row_duplication(rng):
    """The flagship sample_weight semantics must survive the member flatten:
    weight 3 on a row must equal training on that row duplicated 3x
    (dropout 0, full batch, no scaling — same isolation as
    test_sample_weight.py)."""
    X = rng.normal(size=(120, 4))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.05, 120)
    w = np.ones(len(y))
    w[:20] = 3.0
    X_dup = np.vstack([X, np.repeat(X[:20], 2, axis=0)])
    y_dup = np.concatenate([y, np.repeat(y[:20], 2)])

    m_weighted = MasaRegressor(**_TABM_ISOLATED).fit(X, y, sample_weight=w)
    m_dup = MasaRegressor(**_TABM_ISOLATED).fit(X_dup, y_dup)

    X_test = rng.normal(size=(50, 4))
    np.testing.assert_allclose(
        m_weighted.predict(X_test), m_dup.predict(X_test), atol=1e-4
    )


@pytest.mark.parametrize("ens_mode", ["loop", "vectorized"])
def test_tabm_composes_with_outer_ensemble(multiclass_data, ens_mode):
    """The two ensemble axes are orthogonal: n_ens outer members (loop or
    vmapped) each being a k-member weight-shared inner ensemble."""
    X, y, X_test, _ = multiclass_data
    m = MasaClassifier(
        model="tabm", model_params={"k": 4, "d": 16, "n_blocks": 1},
        n_ens=2, ens_mode=ens_mode, n_epochs=5, device="cpu", random_state=0,
    ).fit(X, y)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------- #
# Numeric equivalence pins for the contract itself.
# --------------------------------------------------------------------- #

def test_weighted_loss_flatten_equals_mean_of_member_losses():
    """weighted_loss on (n, k, out) must equal the mean over members of the
    per-member weighted losses — the definition the previous in-objective 3D
    branch implemented for softmax, now for every objective."""
    torch.manual_seed(0)
    n, k, n_classes = 50, 8, 3
    raw = torch.randn(n, k, n_classes)
    weight = torch.rand(n) + 0.1

    obj = MulticlassSoftmax(n_classes)
    y_int = torch.randint(0, n_classes, (n,))
    got = weighted_loss(obj, y_int, raw, weight)
    members = torch.stack([
        (obj.per_sample_loss(y_int, raw[:, j]) * weight).sum() / weight.sum()
        for j in range(k)
    ])
    torch.testing.assert_close(got, members.mean())

    obj_reg = SquaredError()
    y_reg = torch.randn(n, 1)
    got_reg = weighted_loss(obj_reg, y_reg, raw[..., :1], None)
    members_reg = torch.stack(
        [obj_reg.per_sample_loss(y_reg, raw[:, j, :1]).mean() for j in range(k)]
    )
    torch.testing.assert_close(got_reg, members_reg.mean())


def test_apply_transform_averages_on_the_prediction_scale():
    """3D softmax = member-wise softmax then mean — and it reproduces the
    pre-0.6.0 in-model logsumexp averaging exactly. 2D stays untouched."""
    torch.manual_seed(0)
    raw = torch.randn(20, 6, 4)
    got = apply_transform(raw, "softmax")
    torch.testing.assert_close(got, torch.softmax(raw, dim=-1).mean(dim=1))
    old_path = torch.softmax(
        torch.logsumexp(torch.log_softmax(raw, dim=-1), dim=1) - math.log(6), dim=-1
    )
    torch.testing.assert_close(got, old_path)

    raw_2d = torch.randn(20, 4)
    torch.testing.assert_close(
        apply_transform(raw_2d, "softmax"), torch.softmax(raw_2d, dim=1)
    )
    raw_reg = torch.randn(20, 6, 1)
    torch.testing.assert_close(
        apply_transform(raw_reg, "identity"), raw_reg.mean(dim=1)
    )
