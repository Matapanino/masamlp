"""Synthetic contracts for alternating source profiling and fixed-source residual fitting."""
import numpy as np
import pytest
import torch

from masamlp import MasaClassifier
from masamlp.models.base import FeatureEmbedding
from masamlp.models.profiled_realmlp import ProfiledRealMLPNet


def make(mode="joint", warmup=0):
    torch.manual_seed(7)
    emb = FeatureEmbedding(4, [], num_embedding=None, num_scaling=True)
    return ProfiledRealMLPNet(emb, 1, source_groups=[[0, 1], [2, 3]],
                             hidden_sizes=(8, 6), source_hidden_sizes=(7, 3),
                             profile_mode=mode, source_warmup_epochs=warmup,
                             zero_init_output=False)


def inputs():
    torch.manual_seed(9)
    return torch.randn(96, 4), torch.empty(96, 0, dtype=torch.long)


def refresh(net, x, c, w=None, batch=19):
    net.eval()
    net.reset_prediction_state()
    for start in range(0, len(x), batch):
        sl = slice(start, start + batch)
        net.update_prediction_state(x[sl], c[sl], None if w is None else w[sl])
    net.finalize_prediction_state()


def test_projection_is_orthogonal_and_predictions_are_batch_invariant():
    net = make()
    x, c = inputs()
    refresh(net, x, c)
    u, r, b = net.decompose(x, c)
    assert u.shape == (96, 2, 1)
    assert r.shape == (96, 1)
    assert (b.double().T @ r.double()).norm() / len(x) < 1e-5
    expected = net(x, c)
    assert torch.allclose(expected, u.sum(1) + r)
    actual = torch.cat([net(x[i:i+7], c[i:i+7]) for i in range(0, len(x), 7)])
    assert torch.allclose(actual, expected, atol=2e-6)


def test_source_isolation_and_joint_gradients():
    net = make()
    x, c = inputs()
    refresh(net, x, c)
    u, _, _ = net.decompose(x, c)
    changed = x.clone()
    changed[:, 2:] += 30
    v, _, _ = net.decompose(changed, c)
    assert torch.equal(u[:, 0], v[:, 0])
    net.train()
    net(x, c).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.sources.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.remainder.parameters())


def test_frozen_sources_stay_identical_after_warmup():
    net = make("frozen", warmup=1)
    x, c = inputs()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01)
    net.set_training_epoch(0)
    net(x, c).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    net.set_training_epoch(1)
    before = {k: v.clone() for k, v in net.sources.state_dict().items()}
    refresh(net, x, c)
    net.train()
    net(x, c).square().mean().backward()
    optimizer.step()
    assert not net.sources.training
    for k, v in net.sources.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(p.grad is None for p in net.sources.parameters())


def test_unprojected_is_exact_sum_and_warmup_has_no_remainder_gradient():
    net = make("unprojected", warmup=1)
    x, c = inputs()
    net.set_training_epoch(0)
    net(x, c).sum().backward()
    assert all(p.grad is None for p in net.remainder.parameters())
    net.set_training_epoch(1)
    net.eval()
    assert torch.allclose(net(x, c), net.sources(x, c) + net.remainder(x, c), atol=2e-6)


def test_weighted_projection_ignores_zero_weight_rows_and_rank_deficiency():
    net = make()
    x, c = inputs()
    w = torch.ones(96)
    w[48:] = 0
    refresh(net, x, c, w)
    expected = net.projection.clone()
    x[48:] *= 100
    refresh(net, x, c, w)
    assert torch.allclose(net.projection, expected, atol=1e-7)
    refresh(net, torch.ones_like(x), c)
    assert torch.isfinite(net.projection).all()


def test_groups_cover_once_and_optimizer_groups_cover_once():
    net = make()
    params = [p for g in net.param_groups() for p in g["params"]]
    assert len(params) == len(set(map(id, params)))
    assert set(map(id, params)) == set(map(id, net.parameters()))
    with pytest.raises(ValueError):
        ProfiledRealMLPNet(FeatureEmbedding(4, []), 1, source_groups=[[0, 1], [1, 2, 3]])


@pytest.mark.parametrize("mode", ["joint", "unprojected", "frozen"])
def test_estimator_ema_save_load_and_final_projection(tmp_path, mode):
    x, _ = inputs()
    x = x.numpy()
    y = (x[:, 0] + x[:, 2] > 0).astype(int)
    model = MasaClassifier(model="profiled_realmlp", n_epochs=3, batch_size=32,
                           device="cpu", amp=False, ema_decay=.8, random_state=2,
                           model_params=dict(source_groups=[[0, 1], [2, 3]],
                           hidden_sizes=(8, 6), source_hidden_sizes=(7, 3),
                           source_warmup_epochs=1, profile_mode=mode))
    model.fit(x, y)
    net = model.models_[0]
    xn, xc = model.preprocessor_.transform(x)
    xn, xc = torch.from_numpy(xn), torch.from_numpy(xc)
    _, r, b = net.decompose(xn, xc)
    if mode != "unprojected":
        assert (b.double().T @ r.double()).norm() / len(x) < 1e-5
        assert int(net.profile_refreshes) >= 3
    p = model.predict_proba(x)
    model.save_model(tmp_path / "model")
    loaded = MasaClassifier.load_model(tmp_path / "model")
    np.testing.assert_allclose(loaded.predict_proba(x), p, atol=1e-7)
