"""XLA smoke tests: the differentiator gates, a tiny zoo, and save/load on
``device="xla"``.

In CI these run on the XLA:CPU backend (``PJRT_DEVICE=CPU`` with a
torch/torch_xla pair pinned in the workflow); the whole file skips where
torch_xla is not installed (macOS dev machines — no wheels exist). On a real
TPU VM the same tests run against the TPU.

Exact-equality expectations set ``amp=False``: ``amp="auto"`` means bf16 on
XLA, and bf16 noise would drown the semantics being asserted.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from conftest import ALL_MODELS, TINY_PARAMS

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch_xla") is None, reason="torch_xla not installed"
)

# Mirrors test_sample_weight._ISOLATED: LNN has no BatchNorm and dropout 0,
# full-batch fp32 training, so the weighted loss reduction is the only thing
# that can differ between the compared runs.
_ISOLATED = dict(
    model="lnn",
    model_params={"d_hidden": 16, "n_steps": 2, "d_backbone": 32, "dropout": 0.0},
    numeric_scaler="none",
    target_standardize=False,
    n_epochs=20,
    batch_size=None,
    device="xla",
    amp=False,
    random_state=7,
)


def _regressor(**overrides):
    from masamlp.regressor import MasaRegressor

    kw = dict(
        model="resnet",
        model_params=dict(TINY_PARAMS["resnet"]),
        n_epochs=10,
        device="xla",
        amp=False,
        random_state=0,
    )
    kw.update(overrides)
    return MasaRegressor(**kw)


def test_device_resolution_and_auto_guard():
    import torch

    from masamlp.core.device import resolve_device, xla_backend_type

    assert resolve_device("xla").type == "xla"
    assert resolve_device("xla:0") == torch.device("xla", 0)
    backend = xla_backend_type()
    if backend != "TPU":
        # "tpu" is a stricter claim than "xla": on XLA:CPU it must refuse.
        with pytest.raises(RuntimeError, match="backend"):
            resolve_device("tpu")
        # And "auto" must not fall into XLA outside a TPU environment.
        assert resolve_device("auto").type != "xla"


def test_zero_weight_rows_do_not_affect_fit_xla(rng):
    from masamlp.regressor import MasaRegressor

    X = rng.normal(size=(120, 4))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.05, 120)
    X_extra = rng.normal(size=(40, 4)) + 3.0
    y_extra = np.full(40, 25.0)
    w = np.concatenate([np.ones(len(y)), np.zeros(40)])

    m_weighted = MasaRegressor(**_ISOLATED).fit(
        np.vstack([X, X_extra]), np.concatenate([y, y_extra]), sample_weight=w
    )
    m_subset = MasaRegressor(**_ISOLATED).fit(X, y)

    X_test = rng.normal(size=(50, 4))
    np.testing.assert_allclose(
        m_weighted.predict(X_test), m_subset.predict(X_test), atol=1e-4
    )


def test_custom_objective_xla(reg_data):
    import torch

    X, y, X_test, y_test = reg_data

    def pseudo_huber(y_true: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        err = raw - y_true
        return (torch.sqrt(1.0 + err * err) - 1.0).mean(dim=1)

    m = _regressor(objective=pseudo_huber, n_epochs=30).fit(X, y)
    pred = m.predict(X_test)
    assert np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y_test)[0, 1] > 0.8


def test_custom_metric_early_stopping_xla(reg_data):
    from masamlp import make_metric

    X, y, X_test, y_test = reg_data

    def _neg_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -float(np.mean(np.abs(y_true - y_pred)))

    m = _regressor(
        n_epochs=200,
        eval_metric=make_metric(_neg_mae, name="neg_mae", minimize=False),
        early_stopping_rounds=10,
    ).fit(X, y, eval_set=[(X_test, y_test)])
    assert m.best_iteration_ is not None
    history = m.evals_result_["valid_0"]["neg_mae"]
    assert len(history) < 200  # actually stopped early
    assert np.all(np.isfinite(history))


def test_same_seed_same_result_xla(reg_data):
    # realmlp with flat_cos dropout exercises the tensor-p ScheduledDropout
    # and the XLA RNG seeding path end to end.
    X, y, X_test, _ = reg_data
    kw = dict(
        model="realmlp",
        model_params={"hidden_sizes": [32, 32], "dropout": 0.15,
                      "dropout_schedule": "flat_cos"},
        n_epochs=8,
        device="xla",
        amp=False,
        random_state=3,
    )
    from masamlp.regressor import MasaRegressor

    p1 = MasaRegressor(**kw).fit(X, y).predict(X_test)
    p2 = MasaRegressor(**kw).fit(X, y).predict(X_test)
    np.testing.assert_allclose(p1, p2, atol=1e-7)


def test_save_load_roundtrip_xla(tmp_path, reg_data):
    from masamlp.regressor import MasaRegressor

    X, y, X_test, _ = reg_data
    m = _regressor(n_epochs=15).fit(X, y)
    pred_xla = m.predict(X_test)
    m.save_model(str(tmp_path / "model"))
    loaded = MasaRegressor.load_model(str(tmp_path / "model"))
    np.testing.assert_allclose(loaded.predict(X_test), pred_xla, atol=1e-5)


def test_amp_auto_bf16_smoke_xla(reg_data):
    # amp="auto" -> bf16 autocast on XLA; loose sanity only.
    X, y, X_test, y_test = reg_data
    m = _regressor(amp="auto", n_epochs=15).fit(X, y)
    pred = m.predict(X_test)
    assert np.all(np.isfinite(pred))
    assert np.corrcoef(pred, y_test)[0, 1] > 0.8


@pytest.mark.parametrize("name", ALL_MODELS)
def test_zoo_fit_predict_xla(name, clf_data):
    from masamlp.classifier import MasaClassifier

    X, y, X_test, _ = clf_data
    m = MasaClassifier(
        model=name,
        model_params=dict(TINY_PARAMS[name]),
        n_epochs=3,
        device="xla",
        amp=False,
        random_state=0,
    ).fit(X, y)
    proba = m.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_retrieval_minibatch_xla(name, clf_data):
    # Regression: batch indices that were randperm().split() views crashed
    # torch_xla's index_fill lowering (SIGABRT on TPU v5e); on XLA the
    # trainer now feeds each batch as a clean per-chunk index tensor.
    from masamlp.classifier import MasaClassifier

    X, y, X_test, _ = clf_data
    m = MasaClassifier(
        model=name,
        model_params=dict(TINY_PARAMS[name]),
        n_epochs=2,
        batch_size=64,
        device="xla",
        amp=False,
        random_state=0,
    ).fit(X, y)
    assert np.all(np.isfinite(m.predict_proba(X_test)))


def test_vectorized_rejected_on_xla(reg_data):
    X, y, _, _ = reg_data
    m = _regressor(n_ens=2, ens_mode="vectorized", model="resnet")
    with pytest.raises(ValueError, match="vectorized"):
        m.fit(X, y)


def test_fuse_steps_invariant_without_device_rng(reg_data):
    # Without training-time device RNG (no dropout; shuffling permutations
    # are CPU-drawn) the traced ops are identical whatever the barrier
    # placement, so a fused fit must reproduce the per-step fit.
    # batch_size=64 over 300 rows gives 5 steps/epoch — the K=4 grouping
    # covers both the full-group and the partial-tail-group signatures.
    X, y, X_test, _ = reg_data
    kw = dict(_ISOLATED, n_epochs=6, batch_size=64)
    from masamlp.regressor import MasaRegressor

    p1 = MasaRegressor(**kw, xla_fuse_steps=1).fit(X, y).predict(X_test)
    p4 = MasaRegressor(**kw, xla_fuse_steps=4).fit(X, y).predict(X_test)
    np.testing.assert_allclose(p1, p4, atol=1e-6)


def test_fuse_steps_deterministic_per_k_with_dropout(reg_data):
    # Dropout masks come from the XLA device RNG, whose per-execution seed
    # advances at graph boundaries — so barrier placement selects a
    # *different but equally seeded-deterministic* mask stream. The contract
    # is: same seed + same K => identical; different K => different masks,
    # statistically equivalent training (measured on XLA:CPU: max|diff|
    # ~0.02 on this config).
    X, y, X_test, _ = reg_data
    kw = dict(
        model="realmlp",
        model_params={"hidden_sizes": [32, 32], "dropout": 0.15,
                      "dropout_schedule": "flat_cos"},
        n_epochs=6,
        batch_size=64,
        device="xla",
        amp=False,
        random_state=11,
    )
    from masamlp.regressor import MasaRegressor

    p4a = MasaRegressor(**kw, xla_fuse_steps=4).fit(X, y).predict(X_test)
    p4b = MasaRegressor(**kw, xla_fuse_steps=4).fit(X, y).predict(X_test)
    np.testing.assert_allclose(p4a, p4b, atol=1e-7)  # per-K determinism
    p1 = MasaRegressor(**kw, xla_fuse_steps=1).fit(X, y).predict(X_test)
    np.testing.assert_allclose(p1, p4a, atol=0.1)  # equivalent, not equal


def test_fuse_steps_validation(reg_data):
    X, y, _, _ = reg_data
    with pytest.raises(ValueError, match="xla_fuse_steps"):
        _regressor(xla_fuse_steps=0).fit(X, y)


def test_amp_predict_bf16_xla(reg_data):
    # bf16 prediction is opt-in and must stay close to the fp32 predictions
    # of the same fitted model (bf16 has ~3 significant decimal digits).
    X, y, X_test, _ = reg_data
    m = _regressor(n_epochs=15).fit(X, y)
    p32 = m.predict(X_test)
    m.set_params(amp_predict=True)
    p16 = m.predict(X_test)
    assert np.all(np.isfinite(p16))
    np.testing.assert_allclose(p16, p32, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_amp_predict_retrieval_cache_dtype_xla(name, clf_data):
    # Alternating fp32 / bf16 predicts on one fitted retrieval model must not
    # serve a stale-dtype eval cache (the cache is keyed by autocast state).
    from masamlp.classifier import MasaClassifier

    X, y, X_test, _ = clf_data
    m = MasaClassifier(
        model=name,
        model_params=dict(TINY_PARAMS[name]),
        n_epochs=2,
        device="xla",
        amp=False,
        random_state=0,
    ).fit(X, y)
    p32 = m.predict_proba(X_test)
    m.set_params(amp_predict=True)
    p16 = m.predict_proba(X_test)
    m.set_params(amp_predict=False)
    p32_again = m.predict_proba(X_test)
    np.testing.assert_allclose(p32, p32_again, atol=1e-6)
    np.testing.assert_allclose(p16, p32, atol=0.05)


def test_eval_sync_chunks_do_not_change_results_xla(clf_data):
    # TabR fuses several eval chunks per barrier (xla_eval_sync_chunks=8);
    # barrier placement must never change the predictions.
    from masamlp.classifier import MasaClassifier

    X, y, X_test, _ = clf_data
    kw = dict(
        model="tabr",
        model_params=dict(TINY_PARAMS["tabr"]),
        n_epochs=2,
        eval_batch_size=32,  # forces many chunks on the tiny test split
        device="xla",
        amp=False,
        random_state=0,
    )
    m = MasaClassifier(**kw).fit(X, y)
    assert m.model_.xla_eval_sync_chunks == 8
    p_fused = m.predict_proba(X_test)
    m.model_.xla_eval_sync_chunks = 1
    m.model_.invalidate_eval_cache()
    p_step = m.predict_proba(X_test)
    np.testing.assert_allclose(p_fused, p_step, atol=1e-6)
