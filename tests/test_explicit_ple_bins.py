"""WP5: explicit training-local knots, routing, coordinates and persistence."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.models.base import compute_quantile_bins


def data():
    rng = np.random.default_rng(4)
    x = pd.DataFrame({"b": rng.uniform(10, 30, 96), "cat": ["a", "b"] * 48,
                      "a": rng.uniform(-3, 3, 96), "linear": rng.normal(size=96)})
    return x, (np.arange(96) % 3 == 0).astype(int)


def estimator(**kwargs):
    return MasaClassifier(model="realmlp", model_params={"hidden_sizes": [8], "n_bins": 4},
                          num_embedding="ple", num_embedding_cols=["a", "b"],
                          cat_encoding="onehot", n_epochs=1, device="cpu", **kwargs)


@pytest.mark.parametrize("scaler", ["none", "standard", "robust", "rssc", "quantile"])
def test_explicit_bins_coordinates_routing_and_load(scaler, tmp_path):
    x, y = data()
    bins = {"a": np.array([-2., 0., 2.]), "b": [11., 16., 20., 29.]}
    original = copy.deepcopy(bins)
    m = estimator(numeric_scaler=scaler).fit(x, y, ple_bins=bins)
    assert m.resolved_model_params_["num_embedding_idx"] == [0, 1]
    probe = x.iloc[[0] * 4].copy()
    probe["b"] = bins["b"]
    probe["a"] = [-2., 0., 2., 2.]
    expected = m.preprocessor_.transform(probe)[0]
    for actual, wanted in zip(m.resolved_model_params_["_ple_bins"],
                              [expected[:, 0], expected[:3, 1]], strict=True):
        np.testing.assert_array_equal(actual, wanted)
    for key in bins:
        np.testing.assert_array_equal(bins[key], original[key])
    m.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    assert loaded.resolved_model_params_ == m.resolved_model_params_
    np.testing.assert_array_equal(m.predict_proba(x), loaded.predict_proba(x))


def test_default_quantiles_are_unchanged():
    x, y = data()
    m = estimator().fit(x, y)
    z = m.preprocessor_.transform(x)[0][:, [0, 1]]
    expected = compute_quantile_bins(torch.from_numpy(z), 4)
    for a, b in zip(m.resolved_model_params_["_ple_bins"], expected, strict=True):
        np.testing.assert_array_equal(a, b.numpy())
    n = estimator().fit(x, y, ple_bins=None)
    np.testing.assert_array_equal(m.predict_proba(x), n.predict_proba(x))


def test_eval_labels_cannot_change_knots():
    x, y = data()
    bins = {"a": [-2., 0., 2.], "b": [11., 20., 29.]}
    a = estimator().fit(x, y, ple_bins=bins, eval_set=[(x, y)])
    b = estimator().fit(x, y, ple_bins=bins, eval_set=[(x, 1-y)])
    assert a.resolved_model_params_["_ple_bins"] == b.resolved_model_params_["_ple_bins"]


@pytest.mark.parametrize("bins,match", [
    ({}, "exactly"), ({"a": [-2, 0, 2]}, "exactly"),
    ({"a": [-2, 0, 2], "cat": [0, 1]}, "exactly"),
    ({"a": [-2, 0, 2], "b": [11, 20, 29], "linear": [0, 1]}, "exactly"),
    ({"a": [-2, -2, 2], "b": [11, 29]}, "increasing"),
    ({"a": [2, -2], "b": [11, 29]}, "increasing"),
    ({"a": [-2, np.nan], "b": [11, 29]}, "finite"),
    ({"a": [-2], "b": [11, 29]}, "at least two"),
    ({"a": [[-2, 2]], "b": [11, 29]}, "1-D"),
    ([-2, 0, 2], "mapping"),
])
def test_reject_invalid_bins(bins, match):
    x, y = data()
    with pytest.raises(ValueError, match=match):
        estimator().fit(x, y, ple_bins=bins)


def test_reject_transformed_collisions():
    x, y = data()
    with pytest.raises(ValueError, match="transformed"):
        estimator(numeric_scaler="quantile").fit(
            x, y, ple_bins={"a": [-100, -99, 0, 2], "b": [11, 29]})


def test_requires_ple():
    x, y = data()
    with pytest.raises(ValueError, match="num_embedding='ple'"):
        MasaClassifier(n_epochs=1).fit(x, y, ple_bins={"a": [-2, 2]})
