import numpy as np
import pytest
import torch

from conftest import ALL_MODELS, TINY_PARAMS, TOKEN_MODELS
from masamlp.models import FeatureEmbedding, build_model, register_model
from masamlp.models.layers import entmax15, sparsemax


def _forward(name, n_num=4, cards=(3, 5), out_dim=2, num_embedding=None, n=32):
    torch.manual_seed(0)
    params = dict(TINY_PARAMS[name])
    if name in ("tabr", "modernnca") and out_dim > 1:
        params["n_label_classes"] = out_dim  # retrieval models aggregate class labels
    model = build_model(name, params, n_num, list(cards), out_dim, num_embedding)
    x_num = torch.randn(n, n_num)
    if cards:
        x_cat = torch.stack([torch.randint(0, c, (n,)) for c in cards], dim=1)
    else:
        x_cat = torch.zeros(n, 0, dtype=torch.int64)
    if hasattr(model, "set_candidates"):
        y = torch.randint(0, out_dim, (n,)) if out_dim > 1 else torch.randn(n, 1)
        model.set_candidates(x_num, x_cat, y)
    return model, model(x_num, x_cat)


@pytest.mark.parametrize("name", ALL_MODELS)
@pytest.mark.parametrize("num_embedding", [None, "plr", "periodic"])
def test_forward_shapes_and_grads(name, num_embedding):
    if num_embedding == "periodic" and name in TOKEN_MODELS:
        pytest.skip("periodic embeddings have no fixed token width")
    model, out = _forward(name, num_embedding=num_embedding)
    assert out.shape == (32, 2)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_numeric_only_and_categorical_only(name):
    _, out = _forward(name, n_num=4, cards=())
    assert out.shape == (32, 2)
    _, out = _forward(name, n_num=0, cards=(4,))
    assert out.shape == (32, 2)


def test_no_features_raises():
    with pytest.raises(ValueError, match="at least one"):
        FeatureEmbedding(0, [])


def test_sparsemax_known_value():
    p = sparsemax(torch.tensor([[2.0, 1.0, -1.0]]))
    assert torch.allclose(p, torch.tensor([[1.0, 0.0, 0.0]]))
    p2 = sparsemax(torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(p2, torch.tensor([[0.5, 0.5]]))


def test_entmax15_simplex_and_sparsity():
    torch.manual_seed(0)
    x = torch.randn(8, 10) * 3
    p = entmax15(x, dim=-1)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert (p == 0).any(), "entmax15 should zero out low-scoring entries"


def test_entmax15_gradients_flow():
    x = torch.randn(4, 6, requires_grad=True)
    entmax15(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_danet_masks_live_on_simplex():
    model, _ = _forward("danet")
    mask = entmax15(model.init_block.conv1.mask_weight, dim=-1)
    assert torch.allclose(mask.sum(dim=-1), torch.ones(mask.shape[0]), atol=1e-5)


def test_lnn_steps_change_output_and_eval_is_deterministic():
    torch.manual_seed(0)
    x_num = torch.randn(16, 4)
    x_cat = torch.zeros(16, 0, dtype=torch.int64)
    outs = []
    for steps in (1, 3):
        torch.manual_seed(0)
        params = dict(TINY_PARAMS["lnn"], n_steps=steps)
        model = build_model("lnn", params, 4, [], 1, None)
        model.eval()
        outs.append(model(x_num, x_cat))
    assert not torch.allclose(outs[0], outs[1])
    model.eval()
    assert torch.allclose(model(x_num, x_cat), model(x_num, x_cat))


def test_register_model_contract():
    class Custom(torch.nn.Module):
        def __init__(self, embedding, out_dim):
            super().__init__()
            self.embedding = embedding
            self.output_layer = torch.nn.Linear(embedding.d_out, out_dim)

        def forward(self, x_num, x_cat):
            return self.output_layer(self.embedding(x_num, x_cat))

    register_model("custom_linear")(Custom)
    model = build_model("custom_linear", None, 3, [], 1, None)
    assert model(torch.randn(4, 3), torch.zeros(4, 0, dtype=torch.int64)).shape == (4, 1)
    with pytest.raises(ValueError, match="already registered"):
        register_model("custom_linear")(Custom)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("missing", None, 3, [], 1, None)


def test_model_without_output_layer_is_allowed():
    # output_layer is optional (ModernNCA has none); bias init is skipped.
    class NoHead(torch.nn.Module):
        def __init__(self, embedding, out_dim):
            super().__init__()
            self.embedding = embedding
            self.linear = torch.nn.Linear(embedding.d_out, out_dim)

        def forward(self, x_num, x_cat):
            return self.linear(self.embedding(x_num, x_cat))

    register_model("no_head")(NoHead)
    from masamlp.regressor import MasaRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    m = MasaRegressor(model="no_head", n_epochs=2, device="cpu")
    m.fit(X, X[:, 0])
    assert np.isfinite(m.predict(X)).all()
