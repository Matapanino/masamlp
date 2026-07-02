"""The fused GhostBatchNorm training path must be exactly equivalent to the
reference chunk-loop implementation (per-chunk normalization AND the
sequential running-statistics EMA), for even and uneven chunking."""

import math

import pytest
import torch
from torch import nn

from masamlp.models.layers import GhostBatchNorm1d


class _LoopGBN(nn.Module):
    """The pre-KI-009 reference: sequential BatchNorm over chunks."""

    def __init__(self, num_features: int, virtual_batch_size: int) -> None:
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.shape[0] <= self.virtual_batch_size:
            return self.bn(x)
        chunks = x.chunk(int(math.ceil(x.shape[0] / self.virtual_batch_size)), dim=0)
        return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)


def _pair(num_features=6, vbs=32, seed=0):
    torch.manual_seed(seed)
    fused = GhostBatchNorm1d(num_features, vbs)
    loop = _LoopGBN(num_features, vbs)
    loop.load_state_dict({f"bn.{k}": v.clone() for k, v in fused.bn.state_dict().items()})
    # Non-trivial affine params so the affine path is exercised.
    with torch.no_grad():
        for gbn in (fused, loop):
            gbn.bn.weight.uniform_(0.5, 1.5)
            gbn.bn.bias.uniform_(-0.5, 0.5)
        loop.bn.weight.copy_(fused.bn.weight)
        loop.bn.bias.copy_(fused.bn.bias)
    return fused, loop


@pytest.mark.parametrize("n_rows", [128, 100, 200 + 17])  # even, sub-vbs pass-through, tail
def test_fused_matches_loop_output_and_running_stats(n_rows):
    fused, loop = _pair(vbs=64)
    fused.train(), loop.train()
    for step in range(4):
        x = torch.randn(n_rows, 6) * (1 + step) + step
        out_f = fused(x)
        out_l = loop(x)
        torch.testing.assert_close(out_f, out_l, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fused.bn.running_mean, loop.bn.running_mean,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fused.bn.running_var, loop.bn.running_var,
                               atol=1e-6, rtol=1e-6)
    assert fused.bn.num_batches_tracked == loop.bn.num_batches_tracked
    # eval path (running stats) agrees too
    fused.eval(), loop.eval()
    x = torch.randn(50, 6)
    torch.testing.assert_close(fused(x), loop(x), atol=1e-6, rtol=1e-6)


def test_gradients_flow_through_fused_path():
    fused = GhostBatchNorm1d(4, virtual_batch_size=16)
    fused.train()
    x = torch.randn(70, 4, requires_grad=True)  # 4 full chunks + tail of 6
    fused(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert fused.bn.weight.grad is not None


def test_state_dict_layout_unchanged():
    # Saved DANet models predate the fused path; keys must not change.
    keys = set(GhostBatchNorm1d(3).state_dict())
    assert keys == {"bn.weight", "bn.bias", "bn.running_mean", "bn.running_var",
                    "bn.num_batches_tracked"}


def test_small_batch_uses_plain_bn():
    fused, loop = _pair(vbs=256)
    fused.train(), loop.train()
    x = torch.randn(100, 6)  # <= vbs: single BatchNorm call in both
    torch.testing.assert_close(fused(x), loop(x), atol=1e-6, rtol=1e-6)
