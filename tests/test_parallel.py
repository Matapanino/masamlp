"""Multi-GPU ensemble-member sharding (core/parallel.py + resolve_device_plan).

Real 2-GPU behavior is covered by the CUDA-gated test at the bottom (run on
Kaggle 2xT4); everything else exercises the scheduler, the estimator branch,
and thread-safety contracts on CPU via an injected device plan.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from sklearn.exceptions import NotFittedError
from torch import nn

import masamlp.core.parallel as parallel_mod
import masamlp.core.trainer as trainer_mod
import masamlp.sklearn as sklearn_mod
from masamlp.classifier import MasaClassifier
from masamlp.core.device import resolve_device_plan
from masamlp.core.trainer import TrainerConfig, TrainResult
from masamlp.data.dataset import TabularData
from masamlp.regressor import MasaRegressor

# dropout=0 so nothing consumes global RNG inside worker threads: sharded
# and sequential fits must then be bit-identical.
_RESNET = {"d": 16, "n_blocks": 1, "dropout1": 0.0, "dropout2": 0.0}
_CPU2 = [torch.device("cpu"), torch.device("cpu")]


def _kwargs():
    return dict(
        model="resnet",
        model_params=dict(_RESNET),
        n_ens=2,
        device="cpu",
        n_epochs=8,
        random_state=7,
    )


def test_resolve_device_plan_detection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device_plan("auto", 4) == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]
    assert resolve_device_plan("cuda", 3) == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
        torch.device("cuda", 0),
    ]
    # Explicit devices, single member, or single GPU: no sharding.
    assert resolve_device_plan("cuda:0", 4) is None
    assert resolve_device_plan("cpu", 4) is None
    assert resolve_device_plan("mps", 4) is None
    assert resolve_device_plan(torch.device("cuda"), 4) is None
    assert resolve_device_plan("auto", 1) is None
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert resolve_device_plan("auto", 4) is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device_plan("auto", 4) is None


def test_sharded_cpu_matches_sequential(monkeypatch, reg_data):
    X, y, X_val, y_val = reg_data
    m_seq = MasaRegressor(**_kwargs()).fit(X, y, eval_set=[(X_val, y_val)])
    p_seq = m_seq.predict(X_val)

    monkeypatch.setattr(sklearn_mod, "resolve_device_plan", lambda device, n: list(_CPU2))
    m_shard = MasaRegressor(**_kwargs()).fit(X, y, eval_set=[(X_val, y_val)])
    p_shard = m_shard.predict(X_val)

    np.testing.assert_array_equal(p_seq, p_shard)
    assert m_seq.evals_result_ == m_shard.evals_result_


def test_sharded_exception_propagates(monkeypatch):
    class _BoomTrainer:
        def fit(self, model, objective, train, eval_sets, metrics, config, inverse_target=None):
            if config.random_state == 43:
                raise RuntimeError("boom-43")
            return TrainResult()

    monkeypatch.setattr(parallel_mod, "Trainer", _BoomTrainer)
    members = [nn.Linear(2, 1) for _ in range(3)]
    train = TabularData(torch.randn(8, 2), torch.zeros(8, 0, dtype=torch.int64))
    configs = [
        TrainerConfig(device="cpu", seed_scope="device", random_state=s) for s in (42, 43, 44)
    ]
    with pytest.raises(RuntimeError, match="boom-43"):
        parallel_mod.fit_members_sharded(
            members, None, train, [], [], configs, [torch.device("cpu")] * 3
        )


def test_sharded_failure_leaves_estimator_unfitted(monkeypatch, reg_data):
    X, y, _, _ = reg_data
    monkeypatch.setattr(sklearn_mod, "resolve_device_plan", lambda device, n: list(_CPU2))

    def boom(self, model, objective, train, eval_sets, metrics, config, inverse_target=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(parallel_mod.Trainer, "fit", boom)
    m = MasaRegressor(**_kwargs())
    with pytest.raises(RuntimeError, match="boom"):
        m.fit(X, y)
    with pytest.raises(NotFittedError):
        m.predict(X)


def test_sharded_candidate_storage_shared(monkeypatch, clf_data):
    X, y, X_val, y_val = clf_data
    monkeypatch.setattr(
        sklearn_mod, "resolve_device_plan", lambda device, n: [torch.device("cpu")] * 4
    )
    m = MasaClassifier(
        model="tabr",
        model_params={"d_main": 8, "context_size": 4},
        n_ens=4,
        n_epochs=6,
        early_stopping_rounds=2,
        eval_metric="logloss",
        device="cpu",
        random_state=0,
    )
    m.fit(X, y, eval_set=[(X_val, y_val)])
    # One corpus copy per device, shared by all members — and the
    # early-stopping restore (strict=False) must not have un-shared it.
    assert len({member.cand_x_num.data_ptr() for member in m.models_}) == 1
    assert np.isfinite(m.predict_proba(X_val)).all()


def test_sharded_compile_warns_and_trains(monkeypatch, reg_data):
    X, y, _, _ = reg_data
    monkeypatch.setattr(sklearn_mod, "resolve_device_plan", lambda device, n: list(_CPU2))
    m = MasaRegressor(**{**_kwargs(), "compile": True, "n_epochs": 2})
    with pytest.warns(UserWarning, match="sharded"):
        m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_seed_scope_device_skips_global_seeding(monkeypatch, reg_data):
    X, y, _, _ = reg_data
    calls: list[str] = []
    orig = trainer_mod.seed_everything
    monkeypatch.setattr(
        trainer_mod,
        "seed_everything",
        lambda seed: calls.append(threading.current_thread().name) or orig(seed),
    )
    monkeypatch.setattr(
        sklearn_mod,
        "resolve_device_plan",
        lambda device, n: [torch.device("cpu")] * n if n > 1 else None,
    )
    MasaRegressor(**{**_kwargs(), "n_epochs": 2}).fit(X, y)
    assert calls == []  # workers run with seed_scope="device"

    MasaRegressor(**{**_kwargs(), "n_epochs": 2, "n_ens": 1}).fit(X, y)
    assert calls and all(name == "MainThread" for name in calls)


def test_seed_scope_validation():
    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 1)

        def forward(self, x_num, x_cat):
            return self.lin(x_num)

    train = TabularData(
        torch.randn(8, 2), torch.zeros(8, 0, dtype=torch.int64), y=torch.randn(8, 1)
    )
    config = TrainerConfig(device="cpu", seed_scope="banana")
    with pytest.raises(ValueError, match="seed_scope"):
        trainer_mod.Trainer().fit(_Tiny(), None, train, [], [], config)


_n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.mark.skipif(_n_gpus < 2, reason="needs >= 2 CUDA devices")
def test_sharded_two_gpus_end_to_end(clf_data):
    X, y, X_test, _ = clf_data
    kw = dict(
        model="tabr",
        model_params={"d_main": 16, "context_size": 8},
        n_ens=2,
        n_epochs=10,
        random_state=0,
    )
    m = MasaClassifier(device="auto", **kw).fit(X, y)
    devices = {next(member.parameters()).device for member in m.models_}
    assert devices == {torch.device("cuda", 0), torch.device("cuda", 1)}
    proba = m.predict_proba(X_test)
    assert np.isfinite(proba).all()
    # Same seeds on one GPU: results agree up to cross-GPU float drift.
    m_ref = MasaClassifier(device="cuda:0", **kw).fit(X, y)
    np.testing.assert_allclose(proba, m_ref.predict_proba(X_test), atol=5e-3)
