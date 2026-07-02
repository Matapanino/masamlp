import numpy as np
import pytest
import torch

from masamlp.models.base import FeatureEmbedding, PLREmbedding


@pytest.mark.parametrize("variant", ["pl", "plr", "pbld"])
def test_plr_family_shapes_and_grads(variant):
    torch.manual_seed(0)
    emb = FeatureEmbedding(4, [], num_embedding=variant, d_num_embedding=6, n_frequencies=8)
    x = torch.randn(16, 4, requires_grad=True)
    out = emb(x, torch.zeros(16, 0, dtype=torch.int64))
    assert out.shape == (16, 4 * 6)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_pl_is_linear_plr_is_rectified():
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    pl = PLREmbedding(2, d_embedding=4, n_frequencies=8, activation="linear")
    plr = PLREmbedding(2, d_embedding=4, n_frequencies=8, activation="relu")
    assert (pl(x) < 0).any(), "PL output should be unrectified"
    assert (plr(x) >= 0).all(), "PLR output must be non-negative"


def test_pbld_densenet_carries_raw_feature():
    torch.manual_seed(0)
    emb = PLREmbedding(
        3, d_embedding=5, n_frequencies=8, activation="linear", cos_bias=True, densenet=True
    )
    x = torch.randn(10, 3)
    out = emb(x).reshape(10, 3, 5)
    # densenet: the last slot of each feature's embedding is the raw value.
    assert torch.allclose(out[:, :, -1], x)
    assert emb.cos_bias_param is not None


def test_num_scaling_is_identity_at_init():
    torch.manual_seed(0)
    plain = FeatureEmbedding(4, [3])
    torch.manual_seed(0)
    scaled = FeatureEmbedding(4, [3], num_scaling=True)
    x_num = torch.randn(8, 4)
    x_cat = torch.randint(0, 3, (8, 1))
    # scale initializes to 1, so outputs match until training moves it.
    assert torch.allclose(plain(x_num, x_cat), scaled(x_num, x_cat))
    assert scaled.scaling is not None and plain.scaling is None


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="num_embedding"):
        FeatureEmbedding(2, [], num_embedding="wavelet")


def test_estimator_accepts_all_embedding_variants(reg_data):
    from masamlp.regressor import MasaRegressor

    X, y, X_test, _ = reg_data
    for mode in ("pl", "pbld"):
        m = MasaRegressor(num_embedding=mode, n_epochs=5, device="cpu", random_state=0,
                          model_params={"d": 32, "n_blocks": 1}).fit(X, y)
        assert np.isfinite(m.predict(X_test)).all()
