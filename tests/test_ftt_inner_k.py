"""FT-Transformer inner-`k` — TabM-style ensembling on the attention backbone
(ADR 0005 §4). `k=1` must stay the exact legacy module tree; `k>1` returns
`(n, k, out)`, trains through the shared contract, and its members average to
`predict_proba`. A lenient naive-BatchEnsemble guard (ADR-0001 §6) catches a
collapsing inner ensemble before any GPU/TPU spend.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from masamlp import MasaClassifier
from masamlp.models import build_model
from masamlp.models.tabm import EnsembleHead

_CFG = dict(n_num=8, cat_cardinalities=[], out_dim=3, num_embedding="plr-lite")


def _toy(n=400, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, n_redundant=1,
        n_classes=3, random_state=seed,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y


def test_k1_is_legacy_tree():
    """k=1 == the plain FT-Transformer: nn.Linear head, no adapter, and the same
    parameter-key set as the default build (state_dict compat for old ckpts)."""
    m1 = build_model("ft_transformer", {"k": 1}, **_CFG)
    m0 = build_model("ft_transformer", {}, **_CFG)  # default k
    assert isinstance(m1.output_layer, torch.nn.Linear)
    assert not any("adapter" in n for n, _ in m1.named_parameters())
    assert {n for n, _ in m1.named_parameters()} == {n for n, _ in m0.named_parameters()}


def test_k_gt1_structure_and_shapes():
    m = build_model("ft_transformer", {"k": 4}, **_CFG)
    assert isinstance(m.output_layer, EnsembleHead)
    assert m.adapter.shape == (4, 1, 192)
    x_num, x_cat = torch.randn(16, 8), torch.zeros(16, 0, dtype=torch.long)
    assert m(x_num, x_cat).shape == (16, 4, 3)                 # (n, k, out)
    m1 = build_model("ft_transformer", {"k": 1}, **_CFG)
    assert m1(x_num, x_cat).shape == (16, 3)                   # legacy 2D


def test_k_gt1_fits_and_members_average():
    X, y = _toy()
    clf = MasaClassifier(
        model="ft_transformer", n_epochs=8, random_state=0, num_embedding="plr-lite",
        model_params={"k": 4, "d_block": 64, "n_blocks": 2},
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 3) and np.allclose(p.sum(1), 1, atol=1e-4)
    pm = clf.predict_proba_members(X)
    assert pm.shape == (len(X), 4, 3)                          # m = n_ens(1) * k(4)
    assert np.allclose(pm.mean(1), p, atol=1e-5)               # mean over members == proba


def test_inner_k_composes_with_n_ens():
    X, y = _toy()
    clf = MasaClassifier(
        model="ft_transformer", n_epochs=6, random_state=0, num_embedding="plr-lite",
        n_ens=2, model_params={"k": 3, "d_block": 64, "n_blocks": 2},
    )
    clf.fit(X, y)
    pm = clf.predict_proba_members(X)
    assert pm.shape == (len(X), 6, 3)                          # m = n_ens(2) * k(3)


def test_naive_be_guard_k_not_collapsing():
    """Inner-k must NOT collapse vs k=1 (the naive-BatchEnsemble failure mode,
    ~+0.05 nat monotone worse). Lenient — the real signal is the fold-0 probe."""
    X, y = _toy(n=600, seed=1)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    def ll(k):
        clf = MasaClassifier(
            model="ft_transformer", n_epochs=20, random_state=0, num_embedding="plr-lite",
            eval_metric="multi_logloss", model_params={"k": k, "d_block": 64, "n_blocks": 2},
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        return log_loss(yva, clf.predict_proba(Xva), labels=list(clf.classes_))

    l1, l8 = ll(1), ll(8)
    print(f"[naive-BE guard] k=1 logloss={l1:.4f}  k=8 logloss={l8:.4f}  delta={l8 - l1:+.4f}")
    assert l8 <= l1 + 0.10, f"k=8 logloss {l8:.4f} collapsed vs k=1 {l1:.4f}"
