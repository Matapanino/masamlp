"""Shared building blocks: Ghost BatchNorm and sparse mappings.

``sparsemax`` (Martins & Astudillo 2016) and ``entmax15`` (Peters et al.
2019) are in-house implementations of the exact sorting-based algorithms —
no dependency on the ``entmax`` package. Gradients flow through the forward
computation, which matches the analytic Jacobian almost everywhere.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class GhostBatchNorm1d(nn.Module):
    """BatchNorm over virtual sub-batches (Hoffer et al. 2017), as used by
    DANet/TabNet to keep normalization statistics healthy at large batch
    sizes. Falls back to plain BatchNorm in eval mode."""

    def __init__(self, num_features: int, virtual_batch_size: int = 256) -> None:
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or x.shape[0] <= self.virtual_batch_size:
            return self.bn(x)
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), dim=0)
        return torch.cat([self.bn(chunk) for chunk in chunks], dim=0)


def sparsemax(x: Tensor, dim: int = -1) -> Tensor:
    """Euclidean projection onto the simplex: sparse alternative to softmax."""
    x = x - x.max(dim=dim, keepdim=True).values
    x_sorted = torch.sort(x, dim=dim, descending=True).values
    k = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[dim] = -1
    k = k.view(shape)
    cumsum = x_sorted.cumsum(dim)
    support = 1.0 + k * x_sorted > cumsum
    k_star = support.sum(dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, k_star - 1) - 1.0) / k_star.to(x.dtype)
    return torch.clamp(x - tau, min=0.0)


def entmax15(x: Tensor, dim: int = -1) -> Tensor:
    """1.5-entmax: sparse simplex mapping between softmax and sparsemax."""
    x = (x - x.max(dim=dim, keepdim=True).values) / 2.0
    x_sorted = torch.sort(x, dim=dim, descending=True).values
    k = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
    shape = [1] * x.ndim
    shape[dim] = -1
    k = k.view(shape)
    mean = x_sorted.cumsum(dim) / k
    mean_sq = (x_sorted**2).cumsum(dim) / k
    ss = k * (mean_sq - mean**2)
    delta = torch.clamp((1.0 - ss) / k, min=0.0)
    tau = mean - torch.sqrt(delta)
    k_star = (tau <= x_sorted).sum(dim=dim, keepdim=True)
    tau_star = tau.gather(dim, k_star - 1)
    return torch.clamp(x - tau_star, min=0.0) ** 2
