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
