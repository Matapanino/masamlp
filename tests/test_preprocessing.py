import numpy as np
import pandas as pd
import pytest

from masamlp.data.preprocessing import TabularPreprocessor


def test_auto_categorical_detection_and_encoding():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pre = TabularPreprocessor(numeric_scaler="none").fit(df)
    x_num, x_cat = pre.transform(df)
    assert x_num.shape == (3, 1) and x_cat.shape == (3, 1)
    assert pre.cat_cardinalities_ == [3]  # a, b + reserved unknown slot
    assert x_cat[:, 0].tolist() == [1, 2, 1]


def test_unseen_and_missing_categories_map_to_zero():
    df = pd.DataFrame({"cat": ["a", "b", "a"]})
    pre = TabularPreprocessor().fit(df)
    _, x_cat = pre.transform(pd.DataFrame({"cat": ["c", None, "b"]}))
    assert x_cat[:, 0].tolist() == [0, 0, 2]


def test_numeric_nan_median_impute():
    X = np.array([[1.0], [np.nan], [3.0]])
    pre = TabularPreprocessor(numeric_scaler="none").fit(X)
    x_num, _ = pre.transform(X)
    assert x_num[1, 0] == pytest.approx(2.0)


def test_quantile_scaling_is_bounded_and_monotone():
    rng = np.random.default_rng(0)
    X = rng.lognormal(size=(500, 1))  # heavily skewed
    pre = TabularPreprocessor(numeric_scaler="quantile").fit(X)
    z, _ = pre.transform(X)
    assert np.all(np.isfinite(z))
    assert np.abs(z).max() < 6.0
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(z[order, 0]) >= 0)


def test_constant_column_maps_to_zero():
    X = np.ones((50, 1))
    pre = TabularPreprocessor(numeric_scaler="quantile").fit(X)
    z, _ = pre.transform(X)
    assert np.allclose(z, 0.0)


def test_standard_and_robust_scaling():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    z_std, _ = TabularPreprocessor(numeric_scaler="standard").fit(X).transform(X)
    assert z_std.mean() == pytest.approx(0.0, abs=1e-6)
    z_rob, _ = TabularPreprocessor(numeric_scaler="robust").fit(X).transform(X)
    assert z_rob[1, 0] < 0 < z_rob[2, 0]


def test_ndarray_input_gets_generated_names():
    X = np.zeros((5, 3))
    pre = TabularPreprocessor().fit(X)
    assert pre.feature_names_in_ == ["0", "1", "2"]
    with pytest.raises(ValueError, match="features"):
        pre.transform(np.zeros((5, 2)))


def test_explicit_categorical_by_name_and_index():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    pre = TabularPreprocessor(categorical_features=["a"]).fit(df)
    assert pre.categorical_idx_ == [0]
    pre2 = TabularPreprocessor(categorical_features=[1]).fit(df)
    assert pre2.categorical_idx_ == [1]


def test_rssc_scaling_bounded_and_constant_safe():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.lognormal(size=(200, 1)), np.ones((200, 1))], axis=1)
    pre = TabularPreprocessor(numeric_scaler="rssc").fit(X)
    z, _ = pre.transform(X)
    # smooth clip keeps everything in (-3, 3); constant columns stay 0.
    assert np.all(np.abs(z[:, 0]) < 3.0)
    assert np.allclose(z[:, 1], 0.0)
    order = np.argsort(X[:, 0])
    assert np.all(np.diff(z[order, 0]) >= 0)


def test_onehot_encoding_realmlp_style():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "tri": ["a", "b", "c"], "bin": ["p", "q", "p"]})
    pre = TabularPreprocessor(numeric_scaler="none", cat_encoding="onehot").fit(df)
    x_num, x_cat = pre.transform(df)
    assert x_cat.shape == (3, 0)
    assert pre.cat_cardinalities_ == []
    assert pre.transform_width() == (1 + 3 + 1, 0)  # num + 3-way one-hot + binary +-1
    np.testing.assert_array_equal(x_num[:, 1:4], np.eye(3, dtype=np.float32))
    np.testing.assert_array_equal(x_num[:, 4], np.array([1.0, -1.0, 1.0], dtype=np.float32))
    # unknown / missing become all-zeros (0 for the binary column)
    x_new, _ = pre.transform(pd.DataFrame({"num": [1.0], "tri": ["zzz"], "bin": [None]}))
    np.testing.assert_array_equal(x_new[0, 1:], np.zeros(4, dtype=np.float32))


def test_hybrid_encoding_splits_by_cardinality():
    df = pd.DataFrame({
        "num": np.arange(20.0),
        "small": ["a", "b", "c", "d"] * 5,             # 4 categories -> one-hot
        "big": [f"c{i}" for i in range(20)],           # 20 categories -> embedding
    })
    pre = TabularPreprocessor(numeric_scaler="none", cat_encoding="hybrid",
                              onehot_max_categories=9).fit(df)
    x_num, x_cat = pre.transform(df)
    assert x_num.shape == (20, 1 + 4)  # numeric + 4-way one-hot
    assert x_cat.shape == (20, 1) and pre.cat_cardinalities_ == [21]
    assert pre.transform_width() == (5, 1)
    restored = TabularPreprocessor.from_state(*pre.get_state())
    b_num, b_cat = restored.transform(df)
    np.testing.assert_array_equal(x_num, b_num)
    np.testing.assert_array_equal(x_cat, b_cat)


def test_onehot_state_roundtrip():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "c", "a"]})
    pre = TabularPreprocessor(numeric_scaler="rssc", cat_encoding="onehot").fit(df)
    restored = TabularPreprocessor.from_state(*pre.get_state())
    a_num, _ = pre.transform(df)
    b_num, _ = restored.transform(df)
    np.testing.assert_array_equal(a_num, b_num)
    assert restored.transform_width() == pre.transform_width()


def test_state_roundtrip():
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", "b", "a"]})
    pre = TabularPreprocessor().fit(df)
    meta, arrays = pre.get_state()
    restored = TabularPreprocessor.from_state(meta, arrays)
    a_num, a_cat = pre.transform(df)
    b_num, b_cat = restored.transform(df)
    np.testing.assert_array_equal(a_num, b_num)
    np.testing.assert_array_equal(a_cat, b_cat)
