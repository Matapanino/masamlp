"""Soft binary targets and exact uniform-weight compatibility."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from masamlp import MasaClassifier
from masamlp.core.objectives import BinaryLogistic, make_objective
from masamlp.core.trainer import weighted_loss

_FAMILY_PARAMS = {
    "realmlp": {"hidden_sizes": [16], "dropout": 0.0},
    "tabm": {"k": 2, "d": 16, "n_blocks": 1, "dropout": 0.0},
    "ft_transformer": {
        "n_blocks": 1,
        "d_block": 16,
        "attention_n_heads": 4,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.0,
        "k": 2,
    },
}


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(20260904)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    hard = (X[:, 0] - 0.4 * X[:, 1] > 0).astype(np.int64)
    soft = hard.astype(np.float32) * 0.8 + 0.1
    return X[:48], hard[:48], soft[:48], X[48:], hard[48:]


def _estimator(model: str, *, n_epochs: int = 2, early_stopping_rounds=None):
    return MasaClassifier(
        model=model,
        model_params=dict(_FAMILY_PARAMS[model]),
        numeric_scaler="none",
        n_epochs=n_epochs,
        batch_size=16,
        n_ens=2 if model == "realmlp" else 1,
        early_stopping_rounds=early_stopping_rounds,
        device="cpu",
        n_threads=1,
        random_state=7,
    )


def _assert_same_fit(left: MasaClassifier, right: MasaClassifier, X: np.ndarray) -> None:
    assert len(left.models_) == len(right.models_)
    for member, (left_model, right_model) in enumerate(
        zip(left.models_, right.models_, strict=True)
    ):
        left_state = left_model.state_dict()
        right_state = right_model.state_dict()
        assert left_state.keys() == right_state.keys()
        for name in left_state:
            assert torch.equal(left_state[name], right_state[name]), f"member {member}: {name}"
    np.testing.assert_array_equal(left.predict_proba(X), right.predict_proba(X))


def test_soft_binary_target_and_weight_reach_bce_unchanged():
    target = np.array([0.05, 0.25, 0.7, 0.95], dtype=np.float32)
    objective, encoded = MasaClassifier()._setup_target(target)
    np.testing.assert_array_equal(encoded, target)

    raw = torch.tensor([[-1.2], [0.4], [1.1], [-0.3]])
    weight = torch.tensor([0.5, 1.0, 3.0, 2.0])
    got = weighted_loss(objective, objective.prepare_target(encoded), raw, weight)
    per_row = F.binary_cross_entropy_with_logits(
        raw[:, 0], torch.from_numpy(target), reduction="none"
    )
    expected = (per_row * weight).sum() / weight.sum()

    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("model", _FAMILY_PARAMS)
def test_model_family_fits_soft_binary_targets(model, binary_data):
    X, _, y_soft, X_valid, _ = binary_data
    weight = np.linspace(0.5, 2.0, len(y_soft), dtype=np.float32)
    fitted = _estimator(model).fit(X, y_soft, sample_weight=weight)

    assert isinstance(fitted.objective_, BinaryLogistic)
    np.testing.assert_array_equal(fitted.classes_, np.array([0, 1]))
    assert np.issubdtype(fitted.classes_.dtype, np.integer)
    assert np.issubdtype(fitted.predict(X_valid).dtype, np.integer)
    proba = fitted.predict_proba(X_valid)
    assert proba.shape == (len(X_valid), 2)
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("model", _FAMILY_PARAMS)
def test_model_family_hard_float_labels_are_byte_identical(model, binary_data):
    X, y_hard, _, X_valid, _ = binary_data
    integer_fit = _estimator(model).fit(X, y_hard)
    float_fit = _estimator(model).fit(X, y_hard.astype(np.float32))

    _assert_same_fit(integer_fit, float_fit, X_valid)


@pytest.mark.parametrize("model", _FAMILY_PARAMS)
def test_model_family_constant_sample_weight_is_byte_identical(model, binary_data):
    X, y_hard, _, X_valid, _ = binary_data
    unweighted = _estimator(model).fit(X, y_hard)
    weighted = _estimator(model).fit(
        X, y_hard, sample_weight=np.full(len(y_hard), 7.0, dtype=np.float32)
    )

    _assert_same_fit(unweighted, weighted, X_valid)


@pytest.mark.parametrize("model", _FAMILY_PARAMS)
def test_model_family_rejects_out_of_range_soft_targets(model, binary_data):
    X, _, y_soft, _, _ = binary_data
    y_bad = y_soft.copy()
    y_bad[0] = 1.01

    with pytest.raises(ValueError, match=r"soft binary targets.*\[0, 1\]"):
        _estimator(model).fit(X, y_bad)


def test_soft_target_early_stopping_uses_hard_eval_labels(binary_data):
    X, _, y_soft, X_valid, y_valid = binary_data
    fitted = (
        _estimator("realmlp", n_epochs=8, early_stopping_rounds=3)
        .set_params(eval_metric="auc")
        .fit(X, y_soft, eval_set=[(X_valid, y_valid)])
    )

    history = fitted.evals_result_["valid_0"]["auc"]
    assert fitted.best_iteration_ == int(np.argmax(history))
    assert fitted.best_score_ == pytest.approx(max(history))

    with pytest.raises(ValueError, match="eval_set labels must be hard"):
        _estimator("realmlp", n_epochs=1).fit(
            X, y_soft, eval_set=[(X_valid, y_valid.astype(np.float32) * 0.8 + 0.1)]
        )


def test_multiclass_soft_targets_raise_clear_error(binary_data):
    X, y_hard, _, _, _ = binary_data
    distribution = np.eye(3, dtype=np.float32)[y_hard]

    with pytest.raises(ValueError, match="multiclass soft targets are not supported"):
        _estimator("realmlp").fit(X, distribution)


def test_two_dimensional_hard_targets_raise_hard_label_shape_error(binary_data):
    X, y_hard, _, _, _ = binary_data
    y_2d = np.column_stack([y_hard, y_hard])

    with pytest.raises(ValueError, match="hard class labels must be a 1-D vector"):
        _estimator("realmlp").fit(X, y_2d)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_float_boundary_targets_explain_hard_label_interpretation(value, binary_data):
    X, y_hard, _, _, _ = binary_data
    y_constant = np.full(y_hard.shape, value, dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"constant floating targets.*0\.0 or 1\.0.*hard labels",
    ):
        _estimator("realmlp").fit(X, y_constant)


def test_int64_custom_objective_rejects_fractional_targets():
    objective = make_objective(
        lambda y, raw: (raw[:, 0] - y.float()).abs(),
        target_dtype="int64",
    )

    with pytest.raises(ValueError, match="target_dtype='int64'.*fractional targets"):
        objective.prepare_target(np.array([0.1, 0.9], dtype=np.float32))


def test_v094_hard_label_fit_fixture_is_byte_identical():
    """Pin the complete model state and predictions produced by 0.9.4."""
    rng = np.random.default_rng(20260904)
    X = rng.normal(size=(48, 4)).astype(np.float32)
    y = (X[:, 0] - 0.4 * X[:, 1] > 0).astype(np.int64)
    fitted = MasaClassifier(
        model="resnet",
        model_params={"d": 8, "n_blocks": 1, "dropout1": 0.0, "dropout2": 0.0},
        numeric_scaler="none",
        n_epochs=3,
        batch_size=None,
        device="cpu",
        n_threads=1,
        random_state=19,
    ).fit(X, y)

    state_hash = hashlib.sha256()
    for name, tensor in fitted.model_.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(str(tensor.dtype).encode())
        state_hash.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    pred_hash = hashlib.sha256(fitted.predict_proba(X[:13]).tobytes()).hexdigest()

    assert state_hash.hexdigest() == (
        "6e045f6191e0c90dfda8ed7584bcff85288809b12a237403075b8c86d2ea9dd9"
    )
    assert pred_hash == "f84a8df249b0c285a54e8660799de8c1f2dd15e9183a1b55c2c7aeebb64d68d3"
