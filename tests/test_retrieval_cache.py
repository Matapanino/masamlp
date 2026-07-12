"""Eval-time retrieval caching and chunked scoring (TabR / ModernNCA).

Covers the equivalence of the chunked paths with the unchunked math and
every cache-invalidation rule listed in models/retrieval.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.core.trainer import _swap_in_params
from masamlp.models import build_model
from masamlp.models.tabr import TabR
from masamlp.regressor import MasaRegressor

_N, _N_NUM = 53, 5


def _make_model(name: str, out_dim: int, n_label_classes: int | None, chunk: int = 7):
    torch.manual_seed(0)
    params: dict = {"candidate_chunk_size": chunk}
    if name == "tabr":
        params.update(d_main=16, context_size=8)
    else:
        params.update(dim=16, d_block=32, n_blocks=1)
    if n_label_classes is not None:
        params["n_label_classes"] = n_label_classes
    model = build_model(name, params, _N_NUM, [], out_dim, None)
    x_num = torch.randn(_N, _N_NUM)
    x_cat = torch.zeros(_N, 0, dtype=torch.int64)
    if n_label_classes is not None:
        y = torch.randint(0, n_label_classes, (_N,))
    else:
        y = torch.randn(_N, 1)
    model.set_candidates(x_num, x_cat, y)
    return model.eval()


def _query(n: int = 11):
    torch.manual_seed(1)
    return torch.randn(n, _N_NUM), torch.zeros(n, 0, dtype=torch.int64)


# --------------------------------------------------------------------- #
# Chunked scoring == unchunked math
# --------------------------------------------------------------------- #
def test_tabr_chunked_search_matches_full():
    model = _make_model("tabr", 3, 3)
    q_num, q_cat = _query()
    with torch.no_grad():
        cand_k = model._candidate_keys()
        k = model._encode(q_num, q_cat)[1]
    idx_chunked = model._search_topk(k, cand_k, 8, None)
    model.candidate_chunk_size = 10_000  # one covering chunk = plain cdist+topk
    idx_full = model._search_topk(k, cand_k, 8, None)
    torch.testing.assert_close(idx_chunked, idx_full)


def test_tabr_chunked_search_respects_exclusion():
    model = _make_model("tabr", 3, 3)
    q_num, q_cat = _query()
    with torch.no_grad():
        cand_k = model._candidate_keys()
        k = model._encode(q_num, q_cat)[1]
    exclude = torch.randint(0, _N, (q_num.shape[0],))
    idx_chunked = model._search_topk(k, cand_k, 8, exclude)
    assert not (idx_chunked == exclude[:, None]).any()
    model.candidate_chunk_size = 10_000
    idx_full = model._search_topk(k, cand_k, 8, exclude)
    torch.testing.assert_close(idx_chunked, idx_full)


def test_tabr_eval_forward_matches_across_chunk_sizes():
    model = _make_model("tabr", 3, 3)
    q_num, q_cat = _query()
    with torch.inference_mode():
        chunked = model(q_num, q_cat)
    model.candidate_chunk_size = 10_000
    model.invalidate_eval_cache()
    with torch.inference_mode():
        full = model(q_num, q_cat)
    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("out_dim", "n_label_classes"),
    [(3, 3), (1, 2), (1, None)],
    ids=["multiclass", "binary", "regression"],
)
def test_modernnca_streamed_eval_matches_legacy(out_dim, n_label_classes):
    model = _make_model("modernnca", out_dim, n_label_classes)
    q_num, q_cat = _query()
    with torch.inference_mode():
        streamed = model(q_num, q_cat)  # cached + chunked streaming softmax
    legacy = model(q_num, q_cat)  # grad-enabled eval keeps the single-cdist path
    torch.testing.assert_close(streamed, legacy.detach(), atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------- #
# Cache lifecycle
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_eval_cache_hit(name, monkeypatch):
    model = _make_model(name, 3, 3)
    q_num, q_cat = _query()
    encode = "_candidate_keys" if name == "tabr" else "_encoded_candidates"
    calls = []
    orig = getattr(model, encode)
    monkeypatch.setattr(model, encode, lambda: calls.append(1) or orig())
    with torch.inference_mode():
        model(q_num, q_cat)
        model(q_num, q_cat)
    assert model._eval_cache is not None
    if name == "tabr":
        assert len(calls) == 1  # second batch reused the cached keys
    else:
        cand_z, y_repr = model._eval_cache
        assert cand_z.shape[0] == _N and y_repr.shape == (_N, 3)


def _prime_cache(model):
    q_num, q_cat = _query()
    with torch.inference_mode():
        model(q_num, q_cat)
    assert model._eval_cache is not None


@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_eval_cache_invalidation_points(name):
    model = _make_model(name, 3, 3)

    _prime_cache(model)
    model.train(True)
    assert model._eval_cache is None
    model.eval()

    _prime_cache(model)
    model.set_candidates(model.cand_x_num, model.cand_x_cat, model.cand_y)
    assert model._eval_cache is None

    _prime_cache(model)
    model.to(torch.device("cpu"))  # _apply runs even for a no-op move
    assert model._eval_cache is None

    _prime_cache(model)
    model.load_state_dict(model.state_dict())
    assert model._eval_cache is None

    _prime_cache(model)
    _swap_in_params(model, {n: p.detach().clone() for n, p in model.named_parameters()})
    assert model._eval_cache is None


@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_grad_enabled_eval_does_not_touch_cache(name):
    model = _make_model(name, 3, 3)
    q_num, q_cat = _query()
    out = model(q_num, q_cat)
    assert out.requires_grad
    assert model._eval_cache is None


@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_eval_cache_keyed_by_autocast_dtype(name):
    # A cache built under bf16 prediction (amp_predict) must not serve a
    # later fp32 predict, and vice versa — the cache key is the encoder's
    # ambient autocast dtype.
    model = _make_model(name, 3, 3)
    _prime_cache(model)
    assert model._eval_cache_dtype == torch.float32
    q_num, q_cat = _query()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out_bf16 = model(q_num, q_cat)
    assert model._eval_cache_dtype == torch.bfloat16  # rebuilt, re-keyed
    with torch.no_grad():
        out_fp32 = model(q_num, q_cat)
    assert model._eval_cache_dtype == torch.float32
    assert torch.isfinite(out_bf16).all() and torch.isfinite(out_fp32).all()
    assert torch.allclose(out_bf16.float(), out_fp32, atol=0.1)


def test_cache_not_in_state_dict():
    model = _make_model("tabr", 3, 3)
    _prime_cache(model)
    assert "_eval_cache" not in model.state_dict()
    assert {"cand_x_num", "cand_x_cat", "cand_y"} <= set(model.state_dict())


# --------------------------------------------------------------------- #
# Trainer integration
# --------------------------------------------------------------------- #
_TABR_KW = dict(
    model="tabr",
    model_params={"d_main": 16, "context_size": 8},
    device="cpu",
    random_state=0,
)


def test_early_stopping_snapshot_excludes_candidates(monkeypatch, reg_data):
    X, y, X_val, y_val = reg_data
    captured: dict = {}
    orig = TabR.load_state_dict

    def spy(self, state_dict, strict=True, **kwargs):
        captured["keys"] = set(state_dict)
        captured["strict"] = strict
        return orig(self, state_dict, strict=strict, **kwargs)

    monkeypatch.setattr(TabR, "load_state_dict", spy)
    m = MasaRegressor(
        n_epochs=40, early_stopping_rounds=5, eval_metric="rmse", **_TABR_KW
    )
    m.fit(X, y, eval_set=[(X_val, y_val)])
    assert captured["strict"] is False
    assert not captured["keys"] & {"cand_x_num", "cand_x_cat", "cand_y"}
    # The filtered restore still reproduces the best epoch's score.
    history = m.evals_result_["valid_0"]["rmse"]
    assert m.best_iteration_ == int(np.argmin(history))
    rmse_now = float(np.sqrt(np.mean((m.predict(X_val) - y_val) ** 2)))
    assert rmse_now == pytest.approx(m.best_score_, abs=1e-5)


def test_eval_cache_survives_repeated_predict(clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(n_epochs=3, **_TABR_KW).fit(X, y)
    m.predict_proba(X_test)
    cache = m.model_._eval_cache
    assert cache is not None
    m.predict_proba(X_test)
    # The no-op device move in predict must not invalidate the cache.
    assert m.model_._eval_cache is cache


def test_tabr_ema_predicts_consistently(clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(n_epochs=10, ema_decay=0.9, **_TABR_KW).fit(X, y)
    p1 = m.predict_proba(X_test)
    p2 = m.predict_proba(X_test)
    assert np.isfinite(p1).all()
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.parametrize("name", ["tabr", "modernnca"])
def test_roundtrip_predictions_survive_cache(tmp_path, name, clf_data):
    X, y, X_test, _ = clf_data
    params = (
        {"d_main": 16, "context_size": 8}
        if name == "tabr"
        else {"dim": 16, "d_block": 32, "n_blocks": 1}
    )
    m = MasaClassifier(
        model=name, model_params=params, n_epochs=5, device="cpu", random_state=0
    ).fit(X, y)
    before = m.predict_proba(X_test)  # builds the eval cache
    m.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    # 1-ulp tolerance: the loaded corpus buffers live at a different memory
    # alignment, which can flip the last bit of BLAS results.
    np.testing.assert_allclose(before, loaded.predict_proba(X_test), rtol=0, atol=1e-6)
