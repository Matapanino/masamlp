"""TabM (BatchEnsemble MLP) — arch contract tests. Quality-of-ensembling is
measured at scale in the S6E7 campaign, not asserted on toy data; here we pin
the masaMLP contracts: registration, shapes, k=1/k>1, determinism, head-bias
broadcast across members, serialization, and adapter diversity."""

from __future__ import annotations

import numpy as np
import pytest

from masamlp.classifier import MasaClassifier
from masamlp.models import _MODEL_REGISTRY, build_model


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
    """build_model wires FeatureEmbedding + params; forward exposes per-member
    logits (n, k, K) in train mode and the mean-prob logit (n, K) in eval."""
    import torch

    model = build_model("tabm", {"k": 4, "d": 16, "n_blocks": 1}, n_num=6,
                        cat_cardinalities=[], out_dim=3, num_embedding="plr-lite")
    xn, xc = torch.randn(10, 6), torch.zeros(10, 0, dtype=torch.long)
    model.train()
    assert model(xn, xc).shape == (10, 4, 3)   # per-member logits for the loss
    model.eval()
    out = model(xn, xc)
    assert out.shape == (10, 3)                 # ensemble mean-prob logit
    assert torch.isfinite(out).all()
