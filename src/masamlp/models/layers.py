"""Shared building blocks: Ghost BatchNorm and sparse mappings.

``sparsemax`` (Martins & Astudillo 2016) and ``entmax15`` (Peters et al.
2019) are in-house implementations of the exact sorting-based algorithms —
no dependency on the ``entmax`` package. Gradients flow through the forward
computation, which matches the analytic Jacobian almost everywhere.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ScalingLayer(nn.Module):
    """Learnable per-feature scale (init 1) — RealMLP's first layer. Trained
    with a higher learning rate (see ``param_groups`` on RealMLPNet)."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class GhostBatchNorm1d(nn.Module):
    """BatchNorm over virtual sub-batches (Hoffer et al. 2017), as used by
    DANet/TabNet to keep normalization statistics healthy at large batch
    sizes. Falls back to plain BatchNorm in eval mode.

    The training path is **fused**: instead of looping ``nn.BatchNorm1d``
    over chunks (one tiny kernel per virtual batch — the KI-009 GPU
    bottleneck), per-chunk statistics are computed in one reshape, and the
    running statistics replicate the sequential per-chunk EMA exactly in
    closed form: ``r <- (1-m)^n r + sum_i m (1-m)^(n-1-i) mu_i``. The inner
    ``nn.BatchNorm1d`` still owns all parameters and buffers, so state_dicts
    are unchanged.
    """

    def __init__(self, num_features: int, virtual_batch_size: int = 256) -> None:
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor) -> Tensor:
        # momentum=None (cumulative averaging) would need different bookkeeping.
        if (
            not self.training
            or x.shape[0] <= self.virtual_batch_size
            or self.bn.momentum is None
        ):
            return self.bn(x)
        return self._fused_train(x)

    def _fused_train(self, x: Tensor) -> Tensor:
        bn = self.bn
        n_rows, n_features = x.shape
        # torch.chunk semantics (as the loop reference / official DANet GBN):
        # n_chunks near-equal parts of size ceil(n_rows / n_chunks), with a
        # possibly smaller final chunk — NOT vbs-sized chunks plus remainder.
        n_chunks = -(-n_rows // self.virtual_batch_size)
        base = -(-n_rows // n_chunks)
        n_full = n_rows // base
        tail = n_rows - n_full * base

        head = x[: n_full * base].reshape(n_full, base, n_features)
        mean = head.mean(dim=1)  # (n_full, C)
        var = head.var(dim=1, unbiased=False)  # biased, as BN normalizes with
        out = (head - mean[:, None, :]) * torch.rsqrt(var[:, None, :] + bn.eps)
        outs = [out.reshape(-1, n_features)]
        chunk_means = [mean]
        # Running stats use the unbiased variance, matching nn.BatchNorm1d.
        chunk_vars = [var * (base / (base - 1))]
        if tail:
            t = x[n_full * base :]
            t_mean = t.mean(dim=0, keepdim=True)
            t_var = t.var(dim=0, unbiased=False, keepdim=True)
            outs.append((t - t_mean) * torch.rsqrt(t_var + bn.eps))
            chunk_means.append(t_mean)
            chunk_vars.append(t_var * (tail / max(tail - 1, 1)))
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        if bn.affine:
            out = out * bn.weight + bn.bias

        if bn.track_running_stats:
            with torch.no_grad():
                means = torch.cat(chunk_means, dim=0)  # (n_chunks, C)
                variances = torch.cat(chunk_vars, dim=0)
                n_chunks = means.shape[0]
                m = bn.momentum
                # Exact closed form of applying r <- (1-m) r + m stat_i for
                # chunks i = 0..n-1 in order.
                weights = m * (1.0 - m) ** torch.arange(
                    n_chunks - 1, -1, -1, device=x.device, dtype=means.dtype
                )
                decay = (1.0 - m) ** n_chunks
                bn.running_mean.mul_(decay).add_(weights @ means)
                bn.running_var.mul_(decay).add_(weights @ variances)
                bn.num_batches_tracked += n_chunks
        return out


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


def t_softmax(x: Tensor, t: Tensor, dim: int = -1) -> Tensor:
    """Temperature-controlled sparse softmax (Joseph & Raj, GANDALF): entries
    further than ``t`` below the max get (near-)zero weight; ``t`` can be a
    learnable per-row tensor."""
    shifted = x - x.max(dim=dim, keepdim=True).values
    w = torch.relu(shifted + t) + 1e-8
    return torch.softmax(shifted + torch.log(w), dim=dim)


def t_softmax_initial_t(masks: Tensor, sparsity: float, dim: int = -1) -> Tensor:
    """Per-row ``t`` so that roughly a ``sparsity`` fraction of entries start
    (near-)zero — the R-softmax initialization from the GANDALF reference."""
    shifted = masks - masks.max(dim=dim, keepdim=True).values
    q = torch.tensor(float(sparsity))
    return (-torch.quantile(shifted.detach(), q, dim=dim, keepdim=True)) + 1e-8


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
