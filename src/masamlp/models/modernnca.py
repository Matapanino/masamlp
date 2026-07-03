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

At inference the soft-nearest-neighbor aggregation runs against the whole
corpus, streamed over ``candidate_chunk_size`` blocks with a numerically
stable running softmax, so peak memory is B x chunk instead of B x N; the
encoded corpus is cached across query batches (see models/retrieval.py for
the invalidation rules).

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
from masamlp.models.retrieval import RetrievalBase

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


class ModernNCA(RetrievalBase):
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
        candidate_chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError("sample_rate must be in (0, 1]")
        self.embedding = embedding
        self.n_label_classes = n_label_classes
        self.out_dim = out_dim
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.candidate_chunk_size = candidate_chunk_size
        self.encoder = nn.Linear(embedding.d_out, dim)
        self.post_encoder = (
            nn.Sequential(
                *[_MLPBlock(dim, d_block, dropout) for _ in range(n_blocks)],
                nn.BatchNorm1d(dim),
            )
            if n_blocks > 0
            else None
        )

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

    def _label_repr(self, y: Tensor, dtype: torch.dtype) -> Tensor:
        if self.n_label_classes is not None:
            return F.one_hot(y.long(), self.n_label_classes).to(dtype)
        y_repr = y.to(dtype)
        return y_repr[:, None] if y_repr.ndim == 1 else y_repr

    def _encoded_candidates(self) -> Tensor:
        """Whole-corpus encodings, chunked (eval BatchNorm uses running stats,
        so chunked encoding is exact) and cached across query batches."""
        if self._eval_cache is None:
            n = self.cand_y.shape[0]
            self._eval_cache = torch.cat(
                [
                    self._encode(
                        self.cand_x_num[start : start + self.candidate_chunk_size],
                        self.cand_x_cat[start : start + self.candidate_chunk_size],
                    )
                    for start in range(0, n, self.candidate_chunk_size)
                ]
            )
        return self._eval_cache

    def _aggregate_streamed(self, z: Tensor) -> Tensor:
        """softmax(-cdist/T) @ y_repr over the whole corpus, streamed in
        ``candidate_chunk_size`` blocks with a running max so peak memory is
        B x chunk instead of the B x N matrix that OOMs at scale."""
        cand_z = self._encoded_candidates()
        n = cand_z.shape[0]
        if self.n_label_classes is not None:
            k = self.n_label_classes
        else:
            k = 1 if self.cand_y.ndim == 1 else self.cand_y.shape[1]
        running_max = torch.full((z.shape[0],), -torch.inf, device=z.device, dtype=z.dtype)
        num = torch.zeros(z.shape[0], k, device=z.device, dtype=z.dtype)
        den = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for start in range(0, n, self.candidate_chunk_size):
            stop = min(start + self.candidate_chunk_size, n)
            scores = -torch.cdist(z, cand_z[start:stop], p=2) / self.temperature
            y_repr = self._label_repr(self.cand_y[start:stop], z.dtype)
            new_max = torch.maximum(running_max, scores.max(dim=1).values)
            weights = torch.exp(scores - new_max[:, None])
            rescale = torch.exp(running_max - new_max)  # 0 on the first chunk
            num = num * rescale[:, None] + weights @ y_repr
            den = den * rescale + weights.sum(dim=1)
            running_max = new_max
        return num / den[:, None]

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        if not self.has_candidates:
            raise RuntimeError("ModernNCA candidates are not set; call set_candidates first")
        z = self._encode(x_num, x_cat)

        if not self.training and self._eval_cache_usable():
            agg = self._aggregate_streamed(z)
        else:
            # Training (batch + sample_rate subsample) and the rare
            # grad-enabled eval forward keep the exact single-cdist path.
            idx = self._candidate_indices(z.device)
            cand_z = self._encode(self.cand_x_num[idx], self.cand_x_cat[idx])
            distances = torch.cdist(z, cand_z, p=2) / self.temperature
            if self.training and self.current_batch_indices is not None:
                rows = torch.arange(z.shape[0], device=z.device)
                distances[rows, rows] = torch.inf  # batch rows lead `idx`
            weights = F.softmax(-distances, dim=-1)
            agg = weights @ self._label_repr(self.cand_y[idx], z.dtype)

        if self.n_label_classes is None:
            return agg  # (B, k) regression targets
        log_p = torch.log(agg + _EPS)  # (B, K) class probabilities
        if self.out_dim == 1:
            # Binary heads are a single logit column.
            return (log_p[:, 1] - log_p[:, 0]).unsqueeze(1)
        return log_p
