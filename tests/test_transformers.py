import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.models import build_model
from masamlp.models.base import TokenEmbedding
from masamlp.regressor import MasaRegressor

_FTT = {"n_blocks": 1, "d_block": 64, "attention_dropout": 0.1, "ffn_dropout": 0.0}
_TABT = {"n_layers": 2, "d_token": 16}


def test_token_embedding_shapes():
    emb = TokenEmbedding(3, [4, 5], d_token=8, tokenize_numeric=True)
    tokens, num_flat = emb(torch.randn(6, 3), torch.zeros(6, 2, dtype=torch.int64))
    assert tokens.shape == (6, 5, 8) and num_flat.shape == (6, 0)
    assert emb.n_tokens == 5

    emb = TokenEmbedding(3, [4], d_token=8, tokenize_numeric=False)
    tokens, num_flat = emb(torch.randn(6, 3), torch.zeros(6, 1, dtype=torch.int64))
    assert tokens.shape == (6, 1, 8) and num_flat.shape == (6, 3)


def test_token_embedding_plr_tokens():
    emb = TokenEmbedding(2, [], d_token=8, num_embedding="pbld", n_frequencies=4)
    tokens, _ = emb(torch.randn(6, 2), torch.zeros(6, 0, dtype=torch.int64))
    assert tokens.shape == (6, 2, 8)
    with pytest.raises(ValueError, match="token-based"):
        TokenEmbedding(2, [], d_token=8, num_embedding="periodic")


def test_ftt_cls_token_drives_output():
    model = build_model("ft_transformer", dict(_FTT), 3, [4], 1, None)
    model.eval()
    x_num = torch.randn(5, 3)
    x_cat = torch.zeros(5, 1, dtype=torch.int64)
    out1 = model(x_num, x_cat)
    with torch.no_grad():
        model.cls_token += 1.0
    assert not torch.allclose(out1, model(x_num, x_cat))


def test_ftt_learns_and_roundtrips(tmp_path, clf_data):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(model="ft_transformer", model_params=dict(_FTT), n_epochs=25,
                       device="cpu", random_state=0).fit(X, y)
    assert float((m.predict(X_test) == y_test).mean()) > 0.8
    m.save_model(tmp_path / "m")
    loaded = MasaClassifier.load_model(tmp_path / "m")
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_tab_transformer_learns_with_categoricals(mixed_df):
    df, y = mixed_df
    m = MasaRegressor(model="tab_transformer", model_params=dict(_TABT), n_epochs=30,
                      device="cpu", random_state=0).fit(df, y)
    rmse = float(np.sqrt(np.mean((m.predict(df) - y) ** 2)))
    assert rmse < 0.6 * float(y.std())


def test_tab_transformer_numeric_only_degenerates_to_mlp(reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(model="tab_transformer", model_params=dict(_TABT), n_epochs=10,
                      device="cpu", random_state=0).fit(X, y)
    assert np.isfinite(m.predict(X_test)).all()


def test_tab_transformer_numeric_embedding_extension(mixed_df):
    df, y = mixed_df
    m = MasaRegressor(model="tab_transformer", num_embedding="plr",
                      model_params=dict(_TABT), n_epochs=5, device="cpu", random_state=0)
    m.fit(df, y)
    assert m.model_.embedding.d_num_flat == 2 * 16  # 2 numeric cols x d_num_embedding


# ---------------------------------------------------------------------------
# num_embedding_idx / num_embedding_cols routing on ft_transformer (0.9.4)
#
# tokenize_numeric=True (ft_transformer): the listed numeric columns keep
# going through num_embedding (PLR family) as tokens; every other numeric
# column becomes a plain linear token instead of bypassing tokenization
# (unlike FeatureEmbedding's flat bypass) -- every numeric column still
# produces exactly one token either way. tab_transformer never tokenizes
# numerics (tokenize_numeric=False) so it keeps rejecting the option
# (test_input_routing.py::test_selective_embedding_rejected_for_non_tokenizing_token_model).
# ---------------------------------------------------------------------------
def test_ft_transformer_num_embedding_idx_none_matches_main_pin():
    """Pinned reference recorded from `main` (before 0.9.4's routing change)
    with `python -c` at the bottom of this test file's PR: build_model +
    forward at a fixed seed, `num_embedding_idx` left at its default `None`.
    Locks the "routing off changes nothing" guarantee across the change that
    added the option to ft_transformer, the same way test_realmlp_fidelity.py
    locks RealMLP's 0.8.0 fixture. A small tolerance, not bit-identity: CI
    measured ~3e-8 absolute / ~7e-7 relative drift on this exact tensor
    between the macOS/arm64 machine the pin was recorded on and the CI
    runners (macOS-14/ubuntu, py3.10/3.12) -- the same cross-machine fp32
    non-reproducibility test_realmlp_fidelity's `_INIT_ATOL` documents, not a
    behavioural difference (same seed, same ops, different BLAS/SIMD
    codepath)."""
    torch.manual_seed(5)
    model = build_model(
        "ft_transformer",
        {
            "n_blocks": 1, "d_block": 4, "attention_n_heads": 2,
            "attention_dropout": 0.0, "ffn_dropout": 0.0,
        },
        3, [2], 1, "plr-lite",
    )
    model.eval()
    torch.manual_seed(99)
    x_num = torch.randn(4, 3)
    x_cat = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    with torch.no_grad():
        out = model(x_num, x_cat)
    expected = torch.tensor(
        [[-0.03631585091352463], [-0.032606251537799835],
         [-0.03982421010732651], [-0.03340648114681244]]
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_token_embedding_full_coverage_routing_is_the_unrouted_path():
    """Naming *every* numeric column reproduces the unrouted forward exactly
    (mirrors FeatureEmbedding's own version in test_input_routing.py): the
    routed path degenerates to today's path."""
    torch.manual_seed(3)
    plain = TokenEmbedding(4, [3], d_token=8, num_embedding="pbld", n_frequencies=8)
    torch.manual_seed(3)
    routed = TokenEmbedding(
        4, [3], d_token=8, num_embedding="pbld", n_frequencies=8,
        num_embedding_idx=[0, 1, 2, 3],
    )
    x_num, x_cat = torch.randn(6, 4), torch.randint(0, 3, (6, 1))
    with torch.no_grad():
        tokens1, flat1 = plain(x_num, x_cat)
        tokens2, flat2 = routed(x_num, x_cat)
    assert torch.equal(tokens1, tokens2) and torch.equal(flat1, flat2)
    assert routed.num_bypass_tokens is None


def test_token_embedding_partial_routing_bypass_is_linear_tokens():
    """Columns not named in num_embedding_idx become plain linear tokens
    (x_j*w_j+b_j, the FT-Transformer paper's own numeric tokenizer) -- every
    numeric column still becomes exactly one token, embedded columns first
    (in idx order) then bypassing ones (in original order)."""
    torch.manual_seed(4)
    emb = TokenEmbedding(
        4, [], d_token=6, num_embedding="pbld", num_embedding_idx=[0, 2], n_frequencies=8
    )
    assert emb.n_tokens == 4
    x_num = torch.randn(5, 4)
    tokens, num_flat = emb(x_num, torch.zeros(5, 0, dtype=torch.int64))
    assert tokens.shape == (5, 4, 6) and num_flat.shape == (5, 0)
    # Bypass columns (1, 3), in original order, are exactly the standalone
    # linear-token formula on the bypass module's own weights.
    bypass_cols = x_num[:, [1, 3]]
    expected_bypass = (
        bypass_cols.unsqueeze(-1) * emb.num_bypass_tokens.weight + emb.num_bypass_tokens.bias
    )
    torch.testing.assert_close(tokens[:, 2:4], expected_bypass)
    # Embedded columns (0, 2), in idx order, match calling the PLR tokenizer
    # directly on the same two columns.
    with torch.no_grad():
        expected_embedded = emb.num_tokens(x_num[:, [0, 2]])
    torch.testing.assert_close(tokens[:, :2], expected_embedded)


def test_token_embedding_index_validation():
    with pytest.raises(ValueError, match="num_embedding"):
        TokenEmbedding(4, [], d_token=8, num_embedding_idx=[0])           # no num_embedding
    with pytest.raises(ValueError, match="out of range"):
        TokenEmbedding(4, [], d_token=8, num_embedding="plr", num_embedding_idx=[4])
    with pytest.raises(ValueError, match="duplicate"):
        TokenEmbedding(4, [], d_token=8, num_embedding="plr", num_embedding_idx=[1, 1])
    with pytest.raises(ValueError, match="at least one"):
        TokenEmbedding(4, [], d_token=8, num_embedding="plr", num_embedding_idx=[])
    with pytest.raises(ValueError, match="num_embedding_idx"):
        TokenEmbedding(
            4, [], d_token=8, tokenize_numeric=False,
            num_embedding="plr", num_embedding_idx=[0],
        )


def test_ft_transformer_num_embedding_idx_composes_with_inner_k():
    """Routing + the inner-k adapter (ADR 0005 §4) are orthogonal axes: k
    folds members into the batch dim after tokenization, so it never sees
    which columns were routed."""
    torch.manual_seed(0)
    m = build_model(
        "ft_transformer",
        {
            "n_blocks": 1, "d_block": 32, "attention_n_heads": 4,
            "k": 3, "num_embedding_idx": [0, 2],
        },
        4, [3], 2, "plr-lite",
    )
    assert m.embedding.n_tokens == 4 + 1  # 4 numeric tokens + 1 categorical token
    x_num, x_cat = torch.randn(5, 4), torch.zeros(5, 1, dtype=torch.long)
    out = m(x_num, x_cat)
    assert out.shape == (5, 3, 2)  # (n, k, out)


def test_ft_transformer_num_embedding_cols_end_to_end(clf_data):
    """The estimator-level option (`MasaClassifier(num_embedding_cols=...)`)
    resolves names to positions and reaches a real fit -- no estimator-side
    change was needed, only build_model / TokenEmbedding accepting it."""
    X, y, X_test, _ = clf_data
    m = MasaClassifier(
        model="ft_transformer", model_params=dict(_FTT), num_embedding="plr-lite",
        num_embedding_cols=["0", "2"], n_epochs=3, device="cpu", random_state=0,
    ).fit(X, y)
    assert m.resolved_model_params_["num_embedding_idx"] == [0, 2]
    assert m.model_.embedding.num_bypass_tokens is not None
    assert np.isfinite(m.predict_proba(X_test)).all()
