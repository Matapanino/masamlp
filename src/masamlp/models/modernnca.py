"""ModernNCA — Ye, Liu, Zhan 2024, "Modern Neighborhood Component Analysis:
A Deep Tabular Baseline Two Decades Later" (arXiv:2407.03257, ICLR 2025).

Clean-room reimplementation following the MIT-licensed official code in
LAMDA-Tabular/TALENT: rows are encoded (linear projection plus optional
non-residual MLP blocks with BatchNorm), and predictions are a
soft-nearest-neighbor aggregation over the training set — softmax of
negative Euclidean distances (scaled by ``temperature``) times the candidate
labels (one-hot for classification, raw targets for regression). During
training a random ``sample_rate`` fraction of non-batch candidates is used,
the batch itself is always included, and each row is excluded from its own
context (the official diagonal-inf trick), via the trainer's batch-index
protocol.

The paper's strongest configuration uses PLR-lite numeric embeddings —
``num_embedding="plr-lite"`` here — and trains with lr 0.01, weight decay
2e-4. Classification outputs are log-probabilities, which plug into the
standard cross-entropy objectives unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding

_EPS = 1e-7


class _MLPBlock(nn.Module):
    """The official (non-residual) block: BN -> Linear -> ReLU -> Dropout ->
    Linear back to width ``dim``."""

    def __init__(self, dim: int, d_block: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, d_block),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_block, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ModernNCA(nn.Module):
    wants_batch_indices = True

    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        n_label_classes: int | None = None,
        dim: int = 128,
        d_block: int = 512,
        n_blocks: int = 0,
        dropout: float = 0.1,
        temperature: float = 1.0,
        sample_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1]")
        self.embedding = embedding
        self.n_label_classes = n_label_classes
        self.out_dim = out_dim
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.current_batch_indices: Tensor | None = None
        self.encoder = nn.Linear(embedding.d_out, dim)
        self.post_encoder = (
            nn.Sequential(
                *[_MLPBlock(dim, d_block, dropout) for _ in range(n_blocks)],
                nn.BatchNorm1d(dim),
            )
            if n_blocks > 0
            else None
        )

    # Same candidate API as TabR, so the estimator and serialization treat
    # both retrieval models uniformly.
    def set_candidates(self, x_num: Tensor, x_cat: Tensor, y: Tensor) -> None:
        for name, tensor in (("cand_x_num", x_num), ("cand_x_cat", x_cat), ("cand_y", y)):
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor)

    @property
    def has_candidates(self) -> bool:
        return "cand_y" in self._buffers

    def _encode(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        z = self.encoder(self.embedding(x_num, x_cat))
        return self.post_encoder(z) if self.post_encoder is not None else z

    def _candidate_indices(self, device: torch.device) -> Tensor:
        """Training: the batch itself plus a sample_rate fraction of the
        remaining rows (batch first, so self-pairs sit on the diagonal)."""
        n = self.cand_y.shape[0]
        batch_idx = self.current_batch_indices
        if not self.training or batch_idx is None:
            return torch.arange(n, device=device)
        pool_mask = torch.ones(n, dtype=torch.bool, device=device)
        pool_mask[batch_idx] = False
        pool = pool_mask.nonzero(as_tuple=True)[0]
        k = int(pool.shape[0] * self.sample_rate)
        sampled = pool[torch.randperm(pool.shape[0], device=device)[:k]]
        return torch.cat([batch_idx, sampled])

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        if not self.has_candidates:
            raise RuntimeError("ModernNCA candidates are not set; call set_candidates first")
        z = self._encode(x_num, x_cat)
        idx = self._candidate_indices(z.device)
        cand_z = self._encode(self.cand_x_num[idx], self.cand_x_cat[idx])

        distances = torch.cdist(z, cand_z, p=2) / self.temperature
        if self.training and self.current_batch_indices is not None:
            rows = torch.arange(z.shape[0], device=z.device)
            distances[rows, rows] = torch.inf  # batch rows lead `idx`
        weights = F.softmax(-distances, dim=-1)

        y = self.cand_y[idx]
        if self.n_label_classes is not None:
            y_repr = F.one_hot(y.long(), self.n_label_classes).to(z.dtype)
        else:
            y_repr = y.to(z.dtype)
            if y_repr.ndim == 1:
                y_repr = y_repr[:, None]
        agg = weights @ y_repr  # (B, K) class probabilities / (B, k) targets

        if self.n_label_classes is None:
            return agg
        log_p = torch.log(agg + _EPS)
        if self.out_dim == 1:
            # Binary heads are a single logit column.
            return (log_p[:, 1] - log_p[:, 0]).unsqueeze(1)
        return log_p
