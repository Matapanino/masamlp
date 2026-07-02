import numpy as np
import pytest
import torch

from masamlp.classifier import MasaClassifier
from masamlp.models import build_model
from masamlp.models.gandalf import GatedFeatureLearningUnit
from masamlp.models.grn import GatedResidualBlock
from masamlp.models.layers import t_softmax, t_softmax_initial_t
from masamlp.regressor import MasaRegressor


def test_t_softmax_is_a_sparse_distribution():
    torch.manual_seed(0)
    x = torch.randn(4, 10) * 3
    out = t_softmax(x, torch.tensor(0.5))
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.all(out >= 0)
    # entries more than t below the max carry only the 1e-8 floor mass
    shifted = x - x.max(dim=-1, keepdim=True).values
    assert out[shifted + 0.5 < 0].max() < 1e-6


def test_t_softmax_initial_t_hits_target_sparsity():
    torch.manual_seed(0)
    masks = torch.rand(3, 100)
    t = t_softmax_initial_t(masks, sparsity=0.3)
    out = t_softmax(masks, torch.relu(t), dim=-1)
    near_zero = (out < 1e-6).float().mean(dim=-1)
    assert torch.all((near_zero > 0.15) & (near_zero < 0.45))


def test_gflu_shapes_and_gradients():
    torch.manual_seed(0)
    gflu = GatedFeatureLearningUnit(n_features=7, n_stages=3)
    x = torch.randn(16, 7, requires_grad=True)
    out = gflu(x)
    assert out.shape == (16, 7)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert gflu.t is not None and gflu.t.requires_grad
    masks = gflu.masks()
    assert torch.allclose(masks.sum(dim=-1), torch.ones(3), atol=1e-5)


@pytest.mark.parametrize("mask_function", ["entmax15", "sparsemax"])
def test_gflu_alternative_masks(mask_function):
    torch.manual_seed(0)
    gflu = GatedFeatureLearningUnit(7, 2, mask_function=mask_function)
    assert gflu.t is None
    out = gflu(torch.randn(8, 7))
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="mask_function"):
        GatedFeatureLearningUnit(7, 2, mask_function="softmax")


def test_gandalf_feature_importances(mixed_df):
    df, y = mixed_df
    m = MasaRegressor(model="gandalf", model_params={"n_stages": 2}, n_epochs=10,
                      device="cpu", random_state=0).fit(df, y)
    imp = m.model_.feature_importances()
    assert imp.shape == (m.model_.embedding.d_out,)
    assert float(imp.sum()) == pytest.approx(2.0, abs=1e-4)  # one simplex per stage


def test_gandalf_input_batch_norm_option(reg_data):
    X, y, X_test, _ = reg_data
    m = MasaRegressor(model="gandalf", model_params={"n_stages": 2, "input_batch_norm": True},
                      n_epochs=5, device="cpu", random_state=0).fit(X, y)
    assert np.isfinite(m.predict(X_test)).all()


def test_grn_block_is_gated_residual():
    torch.manual_seed(0)
    block = GatedResidualBlock(8, 16, dropout=0.0)
    x = torch.randn(4, 8)
    with torch.no_grad():
        # Force the gate shut: the block reduces to LayerNorm(x).
        block.gate.weight.zero_()
        block.gate.bias[:8] = -100.0
        block.gate.bias[8:] = 0.0
    assert torch.allclose(block(x), block.norm(x), atol=1e-5)


def test_grn_with_num_embedding(clf_data, tmp_path):
    X, y, X_test, y_test = clf_data
    m = MasaClassifier(model="grn", num_embedding="pbld",
                       model_params={"d": 32, "d_hidden": 32, "n_blocks": 2},
                       n_epochs=25, device="cpu", random_state=0).fit(X, y)
    assert float((m.predict(X_test) == y_test).mean()) > 0.8
    m.save_model(tmp_path / "m")
    loaded = MasaClassifier.load_model(tmp_path / "m")
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_gandalf_roundtrip(tmp_path, clf_data):
    X, y, X_test, _ = clf_data
    m = MasaClassifier(model="gandalf", model_params={"n_stages": 2}, n_epochs=5,
                       device="cpu", random_state=0).fit(X, y)
    m.save_model(tmp_path / "m")
    loaded = MasaClassifier.load_model(tmp_path / "m")
    np.testing.assert_array_equal(m.predict_proba(X_test), loaded.predict_proba(X_test))


def test_grn_default_build():
    model = build_model("grn", None, 4, [3], 1, None)
    out = model(torch.randn(6, 4), torch.zeros(6, 1, dtype=torch.int64))
    assert out.shape == (6, 1)
