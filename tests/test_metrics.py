import numpy as np
import pytest

from masamlp.core.metrics import get_metric, make_metric


def test_rmse_mae():
    y = np.array([0.0, 2.0])
    p = np.array([1.0, 0.0])
    assert get_metric("rmse")(y, p) == pytest.approx(np.sqrt(2.5))
    assert get_metric("mae")(y, p) == pytest.approx(1.5)


def test_logloss():
    y = np.array([1, 0])
    p = np.array([0.9, 0.1])
    assert get_metric("logloss")(y, p) == pytest.approx(-np.log(0.9))


def test_logloss_float32_saturation():
    # float32 predictions equal to exactly 1.0 must not produce log(0).
    y = np.array([1, 0])
    p = np.array([1.0, 0.0], dtype=np.float32)
    assert np.isfinite(get_metric("logloss")(y, p))
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.isfinite(get_metric("multi_logloss")(np.array([0, 0]), y_pred))


def test_multi_logloss_and_accuracy():
    y = np.array([0, 2])
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    expected = -(np.log(0.7) + np.log(0.8)) / 2
    assert get_metric("multi_logloss")(y, p) == pytest.approx(expected)
    assert get_metric("accuracy")(y, p) == 1.0


def test_auc_with_ties():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.5, 0.5, 0.1])
    # One clear win + one tie (0.5 credit) per positive-negative pair.
    assert get_metric("auc")(y, p) == pytest.approx((1.0 + 1.0 + 0.5 + 1.0) / 4)


def test_auc_single_class_raises():
    with pytest.raises(ValueError):
        get_metric("auc")(np.ones(3), np.array([0.1, 0.2, 0.3]))


def test_balanced_accuracy():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.1, 0.9, 0.9])
    assert get_metric("balanced_accuracy")(y, p) == pytest.approx((2 / 3 + 1.0) / 2)


def test_make_metric_name_and_direction():
    metric = make_metric(lambda t, p: float(np.mean(p)), name="mean_pred", minimize=False)
    assert metric.name == "mean_pred"
    assert metric.minimize is False
    assert metric(np.zeros(2), np.array([1.0, 3.0])) == 2.0


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric("nope")
