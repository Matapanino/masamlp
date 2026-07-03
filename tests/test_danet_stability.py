"""DANet's ``entmax15`` mask path must not produce a non-finite training
trajectory or crash (S6E7 field report P0).

Two independent guards are exercised:

1. ``entmax15``'s ``sqrt`` is gradient-bounded, so feeding it DANet's raw
   ``mask_weight`` parameter never poisons that parameter with NaN/Inf.
2. Even on an already non-finite input, ``k_star`` is clamped to >= 1 so the
   final ``gather`` cannot index ``-1`` (the reported CPU IndexError / CUDA
   device-side assert at ``layers.py:150``).
"""

from __future__ import annotations

import numpy as np
import torch

from masamlp.models.danet import AbstractLayer
from masamlp.models.layers import entmax15


def _is_simplex(p: torch.Tensor, dim: int = -1) -> bool:
    return bool(
        (p >= 0).all()
        and torch.allclose(p.sum(dim=dim), torch.ones_like(p.sum(dim=dim)), atol=1e-5)
    )


def test_entmax15_backward_finite_on_large_magnitude_inputs():
    # Large magnitudes drive ``ss`` past 1 so ``delta`` hits the clamp
    # boundary where an unguarded ``sqrt`` has an infinite gradient.
    torch.manual_seed(0)
    for scale in (1.0, 10.0, 100.0, 1000.0):
        x = (torch.randn(64, 17) * scale).requires_grad_(True)
        out = entmax15(x)
        assert torch.isfinite(out).all()
        assert _is_simplex(out)
        out.pow(2).sum().backward()
        assert torch.isfinite(x.grad).all()


def test_entmax15_survives_non_finite_rows_without_crashing():
    # A non-finite row would make every ``tau <= x_sorted`` comparison False,
    # giving ``k_star == 0`` -> ``gather(-1)`` crash without the clamp.
    x = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [float("nan"), 1.0, 2.0, 3.0], [float("inf"), 0.0, 0.0, 0.0]]
    )
    out = entmax15(x)  # must not raise
    assert _is_simplex(out[:1])  # the finite row is still a valid simplex


def test_danet_abstract_layer_grad_finite_with_extreme_mask_weight():
    # The exact failure surface: entmax over a drifted raw ``mask_weight``.
    layer = AbstractLayer(d_in=8, d_out=4, k=3, virtual_batch_size=64)
    with torch.no_grad():
        layer.mask_weight.mul_(50.0)  # simulate an unbounded trajectory
    layer.train()
    x = torch.randn(128, 8)
    out = layer(x)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    assert torch.isfinite(layer.mask_weight.grad).all()


def test_danet_trains_with_finite_loss():
    from masamlp import MasaClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = MasaClassifier(
        model="danet",
        model_params={"n_layers": 2, "base_outdim": 16, "k": 2, "virtual_batch_size": 64},
        n_epochs=5,
        learning_rate=1e-2,
        device="cpu",
        random_state=0,
    )
    clf.fit(X, y)  # must complete without a non-finite-loss ValueError
    proba = clf.predict_proba(X)
    assert np.isfinite(proba).all()
