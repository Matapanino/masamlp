"""``candidate_budget`` bounds the retrieval corpus (and the aligned training
rows) of tabr/modernnca with a seeded, class-stratified subsample — the S6E7
field report P2 fix for modernnca OOM / tabr superlinearity at scale."""

from __future__ import annotations

import numpy as np
import pytest

from masamlp.classifier import MasaClassifier
from masamlp.regressor import MasaRegressor

RETRIEVAL = {
    "tabr": {"d_main": 16, "context_size": 8},
    "modernnca": {"dim": 32, "d_block": 64},
}


@pytest.mark.parametrize("model", sorted(RETRIEVAL))
def test_candidate_budget_bounds_corpus(model, clf_data):
    X, y, X_test, _ = clf_data  # 300 train rows
    m = MasaClassifier(
        model=model,
        model_params=RETRIEVAL[model],
        candidate_budget=100,
        n_epochs=3,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    assert m.model_.cand_y.shape[0] == 100
    assert np.isfinite(m.predict_proba(X_test)).all()


@pytest.mark.parametrize("model", sorted(RETRIEVAL))
def test_candidate_budget_noop_when_not_smaller(model, clf_data):
    X, y, _, _ = clf_data
    m = MasaClassifier(
        model=model,
        model_params=RETRIEVAL[model],
        candidate_budget=10_000,  # >= n_rows -> full corpus
        n_epochs=2,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    assert m.model_.cand_y.shape[0] == len(y)


def test_candidate_budget_stratified_keeps_all_classes(multiclass_data):
    X, y, _, _ = multiclass_data  # 350 train rows, 3 classes
    m = MasaClassifier(
        model="tabr",
        model_params=RETRIEVAL["tabr"],
        candidate_budget=30,
        n_epochs=2,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    assert set(np.unique(m.model_.cand_y.numpy())) == set(range(len(m.classes_)))


def test_candidate_budget_seeded_reproducible(clf_data):
    X, y, X_test, _ = clf_data
    kw = dict(
        model="tabr",
        model_params=RETRIEVAL["tabr"],
        candidate_budget=80,
        n_epochs=3,
        device="cpu",
        random_state=0,
    )
    m1 = MasaClassifier(**kw).fit(X, y)
    m2 = MasaClassifier(**kw).fit(X, y)
    np.testing.assert_array_equal(m1.model_.cand_y.numpy(), m2.model_.cand_y.numpy())
    np.testing.assert_array_equal(m1.predict_proba(X_test), m2.predict_proba(X_test))


def test_candidate_budget_regression_bounds_corpus(reg_data):
    X, y, X_test, _ = reg_data  # 300 train rows
    m = MasaRegressor(
        model="tabr",
        model_params=RETRIEVAL["tabr"],
        candidate_budget=120,
        n_epochs=3,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    assert m.model_.cand_y.shape[0] == 120
    assert np.isfinite(m.predict(X_test)).all()


def test_candidate_budget_ignored_for_non_retrieval(clf_data):
    # resnet has no candidate corpus; the budget is a documented no-op.
    X, y, X_test, _ = clf_data
    m = MasaClassifier(
        model="resnet",
        model_params={"d": 32, "n_blocks": 1},
        candidate_budget=10,
        n_epochs=2,
        device="cpu",
        random_state=0,
    ).fit(X, y)
    assert not hasattr(m.model_, "cand_y")
    assert np.isfinite(m.predict_proba(X_test)).all()
