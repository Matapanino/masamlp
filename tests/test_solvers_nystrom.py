"""Synthetic acceptance tests T1–T4, T8–T9 for standalone global solvers."""

import numpy as np
import pytest
import torch

from masamlp.solvers import NystromKRR, rank_curve, rpcholesky_landmarks


def kernel_reference(X, Z, bandwidth, kernel="laplace"):
    distance = np.sqrt(((X[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    return np.exp(-distance / bandwidth if kernel == "laplace" else
                  -(distance / bandwidth) ** 2 / 2)


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def data(n=180, d=4, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    t = np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
    return X, t, rng.uniform(0.1, 1, n)


@pytest.mark.parametrize("kernel", ["laplace", "gaussian"])
def test_t1_dense_parity(kernel):
    X, t, w = data(800, 12)
    params = dict(r=800, reg=0.3, bandwidth=2.5, kernel=kernel, random_state=42,
                  dtype="float64", block_rows=193)
    low = NystromKRR(**params).fit(X, t, sample_weight=w)
    dense = NystromKRR(**params, dense=True).fit(X, t, sample_weight=w)
    aligned = low.alpha_[np.argsort(low.landmark_indices_)]
    assert relative_error(aligned, dense.alpha_) <= 1e-6
    query = data(73, 12, 23)[0]
    assert relative_error(low.predict(query), dense.predict(query)) <= 1e-6
    K = kernel_reference(X, low.landmarks_, 2.5, kernel)
    Kmm = kernel_reference(low.landmarks_, low.landmarks_, 2.5, kernel)
    G = K.T @ (w[:, None] * K)
    b = K.T @ (w * t)
    assert np.linalg.norm((G + 0.3 * Kmm) @ low.alpha_ - b) / np.linalg.norm(b) <= 1e-6
    # Independent dense equation: ridge is on a sum of weighted losses.
    Knn = kernel_reference(X, X, 2.5, kernel)
    expected = np.linalg.solve(Knn + np.diag(0.3 / w), t)
    assert relative_error(dense.alpha_, expected) <= 1e-6
    fp32 = NystromKRR(**(params | {"dtype": "float32"})).fit(X, t, sample_weight=w)
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    assert np.max(np.abs(sigmoid(fp32.predict(query)) - sigmoid(low.predict(query)))) <= 1e-5


def test_t2_nested_rank_curve(monkeypatch):
    X, _, w = data(420)
    t = np.sin(X[:, 0]) + 0.3 * np.cos(X[:, 1])
    import masamlp.solvers.nystrom_krr as module

    original = module.rpcholesky_landmarks
    calls = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "rpcholesky_landmarks", observed)
    curve = rank_curve(X, t, w, ranks=(50, 100, 200, 400), X_val=X[:31],
                       random_state=9, reg=0.01, bandwidth=2, dtype="float64")
    assert len(calls) == 1
    for row in curve:
        np.testing.assert_array_equal(row["model"].landmark_indices_, calls[0][:row["rank"]])
        np.testing.assert_array_equal(row["predictions"], row["model"].predict(X[:31]))
        assert row["solver_path"].startswith(("cholesky", "eigh"))
        assert row["wall_seconds"] > 0
        assert row["peak_device_memory"] is None
    assert np.all(np.diff([row["train_residual_norm"] for row in curve]) <= 1e-9)
    with pytest.warns(UserWarning, match="clipp"):
        clipped = rank_curve(X[:10], t[:10], w[:10], ranks=(5, 20), X_val=X[:3],
                             random_state=9, bandwidth=2, dtype="float64")
    assert [row["rank"] for row in clipped] == [5, 10]


def test_t3_blocking_invariance():
    X, t, w = data(350)
    params = dict(r=95, reg=0.7, bandwidth=2, dtype="float64", random_state=3)
    a = NystromKRR(**params, block_rows=64, predict_batch_rows=17).fit(X, t, w)
    b = NystromKRR(**params, block_rows=10_000, predict_batch_rows=10_000).fit(X, t, w)
    np.testing.assert_allclose(a.alpha_, b.alpha_, rtol=0, atol=1e-10)
    np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=0, atol=1e-10)


def test_t4_zero_weights_and_weight_ridge_scaling():
    X, t, w = data()
    w[::5] = 0
    params = dict(r=65, reg=0.4, bandwidth="median", dtype="float64", random_state=7)
    weighted = NystromKRR(**params).fit(X, t, w)
    dropped = NystromKRR(**params).fit(X[w > 0], t[w > 0], w[w > 0])
    np.testing.assert_array_equal(weighted.landmarks_, dropped.landmarks_)
    np.testing.assert_allclose(weighted.predict(X), dropped.predict(X), rtol=0, atol=1e-12)
    scaled = NystromKRR(**(params | {"reg": 0.8})).fit(X, t, 2 * w)
    np.testing.assert_allclose(weighted.predict(X), scaled.predict(X), rtol=0, atol=1e-10)


def test_t8_determinism():
    X, t, w = data()
    params = dict(r=80, random_state=172, dtype="float64", bandwidth="median")
    a = NystromKRR(**params).fit(X, t, w)
    b = NystromKRR(**params).fit(X, t, w)
    np.testing.assert_array_equal(a.landmark_indices_, b.landmark_indices_)
    np.testing.assert_array_equal(a.alpha_, b.alpha_)
    first = rpcholesky_landmarks(X, 100, 17, block=23)
    second = rpcholesky_landmarks(X, 100, 17, block=23)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 100


@pytest.mark.parametrize("dense", [False, True])
def test_t9_serialization(tmp_path, dense):
    X, t, w = data(60)
    fitted = NystromKRR(r=30, dense=dense, dtype="float64").fit(X, t, w)
    path = tmp_path / "solver.model"
    fitted.save(path)
    assert path.is_file()
    loaded = NystromKRR.load(path)
    np.testing.assert_allclose(fitted.predict(X), loaded.predict(X), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fitted.landmark_indices_, loaded.landmark_indices_)
    assert fitted.bandwidth_ == loaded.bandwidth_
    assert fitted.solver_path_ == loaded.solver_path_


def test_kernel_and_input_contracts():
    X, t, w = data(16)
    for kwargs in ({"reg": -1}, {"r": 0}, {"block_rows": 0}, {"bandwidth": 0},
                   {"kernel": "tree"}, {"dtype": "float16"}, {"device": "mps"}):
        with pytest.raises(ValueError):
            NystromKRR(**kwargs).fit(X, t, w)
    for invalid in (-w, w * np.nan, np.zeros_like(w)):
        with pytest.raises(ValueError):
            NystromKRR(r=8).fit(X, t, invalid)
    with pytest.raises(ValueError, match="5,?000"):
        NystromKRR(dense=True).fit(np.zeros((5001, 1)), np.zeros(5001))
    with pytest.raises((ValueError, RuntimeError), match="fit"):
        NystromKRR().predict(X)
    fitted = NystromKRR(r=8, dtype="float64").fit(X, t)
    assert fitted.predict(X[:0]).shape == (0,)
    with pytest.raises(ValueError):
        fitted.predict(X[:, :2])


def test_degenerate_landmarks_and_solver_fallback(monkeypatch):
    X = np.ones((12, 3))
    with pytest.warns(UserWarning, match="clipp"):
        indices = rpcholesky_landmarks(X, 20, 0, bandwidth=1)
    assert len(np.unique(indices)) == 12
    fitted = NystromKRR(r=12, bandwidth=1, dtype="float64").fit(X, np.ones(12))
    assert np.isfinite(fitted.predict(X)).all()
    assert fitted.solver_path_.startswith(("cholesky", "eigh"))
    original = torch.linalg.cholesky_ex
    calls = []

    def not_pd(*args, **kwargs):
        factor, info = original(*args, **kwargs)
        calls.append(1)
        info.fill_(1)
        return factor, info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", not_pd)
    X, t, w = data(20)
    model = NystromKRR(r=10, bandwidth=2, dtype="float64").fit(X, t, w)
    assert model.solver_path_ == "eigh"
    assert len(calls) == 4  # initial attempt, at most three jitter retries
    assert np.isfinite(model.predict(X)).all()


def test_rpcholesky_matches_independent_dense_pivot_reference():
    X, _, _ = data(37)
    gram = kernel_reference(X, X, 2)
    rng = np.random.default_rng(19)
    factor = np.zeros((len(X), 21))
    diagonal = np.ones(len(X))
    expected = []
    for j in range(21):
        pivot = rng.choice(len(X), p=diagonal / diagonal.sum())
        expected.append(pivot)
        column = gram[:, pivot] - factor[:, :j] @ factor[pivot, :j]
        factor[:, j] = column / np.sqrt(diagonal[pivot])
        diagonal = np.maximum(diagonal - factor[:, j] ** 2, 0)
        diagonal[expected] = 0
    for block in (15, 16384):
        actual = rpcholesky_landmarks(X, 21, 19, block=block, bandwidth=2)
        np.testing.assert_array_equal(actual, expected)


def test_bandwidth_and_streamed_kernel_workspace(monkeypatch):
    import masamlp.solvers.nystrom_krr as module

    X, t, w = data(137)
    distance = np.sqrt(((X[:, None] - X[None, :]) ** 2).sum(axis=2))
    expected = np.median(distance[np.triu_indices(len(X), 1)]) * 1.25
    original = module._kernel
    shapes = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(module, "_kernel", observed)
    model = NystromKRR(r=17, block_rows=35, predict_batch_rows=30,
                       bandwidth_scale=1.25, dtype="float64").fit(X, t, w)
    assert model.bandwidth_ == pytest.approx(expected, rel=1e-14)
    assert shapes and all(rows <= 10 and columns == 17 for rows, columns in shapes)


@pytest.mark.parametrize("kernel,rank,block", [("laplace", 2000, 16384),
                                                ("gaussian", 256, 1000)])
def test_single_pass_matches_previous_pivots_on_2000_rows(kernel, rank, block):
    from masamlp.solvers.landmarks import _rpcholesky_landmarks_reference

    X, _, _ = data(2000, 12)
    expected = _rpcholesky_landmarks_reference(X, rank, 42, block=block,
                                              bandwidth=2.5, kernel=kernel)
    actual = rpcholesky_landmarks(X, rank, 42, block=block, bandwidth=2.5, kernel=kernel)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:128], rpcholesky_landmarks(
        X, 128, 42, block=1000, bandwidth=2.5, kernel=kernel))


def test_single_pass_kernel_entries_and_factor_budget(monkeypatch):
    import masamlp.solvers.landmarks as module

    X, t, _ = data(100)
    original = module._kernel
    entries = []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        entries.append(result.numel())
        assert result.shape[1] == 1  # no old projection kernel is recomputed
        return result

    copies = []
    allocations = []
    original_cpu = torch.Tensor.cpu
    original_empty = torch.empty

    def host_copy(tensor, *args, **kwargs):
        copies.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    def allocated(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append((tuple(result.shape), result.dtype, result.device.type))
        return result

    monkeypatch.setattr(module, "_kernel", observed)
    monkeypatch.setattr(torch.Tensor, "cpu", host_copy)
    monkeypatch.setattr(torch, "empty", allocated)
    rpcholesky_landmarks(X, 30, bandwidth=2, block=35, max_factor_bytes=100 * 30 * 4)
    assert sum(entries) == 100 * 30
    assert copies == [(100,)] * 30  # one diagonal host copy per pivot, not per block
    assert ((30, 100), torch.float32, "cpu") in allocations
    entries.clear()
    with pytest.raises(ValueError, match="landmark_method='uniform'"):
        rpcholesky_landmarks(X, 30, max_factor_bytes=100 * 30 * 4 - 1)
    assert not entries
    NystromKRR(r=30, landmark_method="uniform", max_factor_bytes=0).fit(X, t)


def test_uniform_prefix_determinism_and_roundtrip(tmp_path, monkeypatch):
    import masamlp.solvers.nystrom_krr as module

    X, t, w = data(180)
    w[::7] = 0
    active = np.flatnonzero(w > 0)
    expected = active[np.random.default_rng(42).permutation(len(active))]
    original = module._uniform_landmarks
    calls = []

    def observed(*args):
        result = original(*args)
        calls.append(result)
        return result

    monkeypatch.setattr(module, "_uniform_landmarks", observed)
    params = dict(landmark_method="uniform", max_factor_bytes=0, dtype="float64")
    curve = rank_curve(X, t, w, ranks=(20, 40, 80), X_val=X[:10], random_state=42, **params)
    assert len(calls) == 1
    for row in curve:
        model = row["model"]
        np.testing.assert_array_equal(model.landmark_indices_, expected[:row["rank"]])
        # A separate fit's bandwidth sampling must not perturb the permutation.
        repeated = NystromKRR(r=row["rank"], random_state=42, **params).fit(X, t, w)
        np.testing.assert_array_equal(repeated.landmark_indices_, model.landmark_indices_)
        np.testing.assert_array_equal(X[model.landmark_indices_], model.landmarks_)
    path = tmp_path / "uniform.npz"
    repeated.save(path)
    loaded = NystromKRR.load(path)
    assert loaded.landmark_method == "uniform"
    assert loaded.max_factor_bytes == 0
    np.testing.assert_array_equal(loaded.predict(X), repeated.predict(X))
    with pytest.raises(ValueError, match="landmark_method"):
        NystromKRR(landmark_method="unknown")


def test_solver_parameters_documented():
    import inspect
    from pathlib import Path

    from masamlp.solvers import LinearResidualKRR

    doc = (Path(__file__).resolve().parents[1] / "docs" / "parameters.md").read_text()
    for obj in (NystromKRR, LinearResidualKRR, rank_curve, rpcholesky_landmarks):
        for name, param in inspect.signature(obj).parameters.items():
            if param.default is inspect.Parameter.empty or param.kind == param.VAR_KEYWORD:
                continue
            assert f"`{name}`" in doc, (obj.__name__, name)
