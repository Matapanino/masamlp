"""TabR — Gorishniy et al. 2023, "TabR: Unlocking the Power of
Retrieval-Augmented Tabular Deep Learning" (arXiv:2307.14338).

Clean-room reimplementation following the MIT-licensed official repository
(yandex-research/tabular-dl-tabr). A query row is encoded, its nearest
training rows (by L2 in key space) are retrieved, and their label embeddings
plus a learned key-difference transform are aggregated with softmax
similarity weights before the predictor head.

Implementation notes vs the original:
- The candidate set (the training data) lives in registered buffers, so it
  moves with ``.to(device)`` and is saved/loaded through ``state_dict``.
- Retrieval always uses the original repo's ``memory_efficient`` strategy:
  candidate keys are computed without gradients for the search, and only the
  selected context rows are re-encoded with gradients.
- Search is a plain ``torch.cdist`` + ``topk`` (no faiss dependency) —
  appropriate for the small/medium datasets this library targets.
- During training the trainer exposes the batch's candidate indices
  (``wants_batch_indices``) so each row can exclude itself from its context.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding


def _block(
    d_main: int, d_block: int, dropout0: float, dropout1: float, prenorm: bool
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(d_main)] if prenorm else []
    layers += [
        nn.Linear(d_main, d_block),
        nn.ReLU(),
        nn.Dropout(dropout0),
        nn.Linear(d_block, d_main),
        nn.Dropout(dropout1),
    ]
    return nn.Sequential(*layers)


class TabR(nn.Module):
    wants_batch_indices = True

    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        n_label_classes: int | None = None,
        d_main: int = 96,
        d_multiplier: float = 2.0,
        encoder_n_blocks: int = 0,
        predictor_n_blocks: int = 1,
        context_size: int = 96,
        dropout0: float = 0.1,
        dropout1: float = 0.0,
        context_dropout: float = 0.2,
        candidate_chunk_size: int = 8192,
    ) -> None:
        super().__init__()
        if out_dim > 1 and n_label_classes is None:
            raise ValueError("tabr does not support multi-output regression")
        self.embedding = embedding
        self.context_size = context_size
        self.candidate_chunk_size = candidate_chunk_size
        self.current_batch_indices: Tensor | None = None
        d_block = int(d_main * d_multiplier)

        # >>> encoder
        self.linear = nn.Linear(embedding.d_out, d_main)
        self.encoder_blocks = nn.ModuleList(
            _block(d_main, d_block, dropout0, dropout1, prenorm=i > 0)
            for i in range(encoder_n_blocks)
        )
        # 'auto' mixer normalization from the original: only with encoder blocks.
        self.mixer_norm = nn.LayerNorm(d_main) if encoder_n_blocks > 0 else None

        # >>> retrieval
        if n_label_classes is None:
            self.label_encoder: nn.Module = nn.Linear(1, d_main)
            bound = 1.0 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            self.label_encoder = nn.Embedding(n_label_classes, d_main)
            nn.init.uniform_(self.label_encoder.weight, -1.0, 1.0)
        self.key_proj = nn.Linear(d_main, d_main)
        self.value_transform = nn.Sequential(
            nn.Linear(d_main, d_block),
            nn.ReLU(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.context_dropout = nn.Dropout(context_dropout)

        # >>> predictor
        self.predictor_blocks = nn.ModuleList(
            _block(d_main, d_block, dropout0, dropout1, prenorm=True)
            for _ in range(predictor_n_blocks)
        )
        self.head_norm = nn.LayerNorm(d_main)
        self.output_layer = nn.Linear(d_main, out_dim)

    # ------------------------------------------------------------------ #
    # Candidates (the training set)
    # ------------------------------------------------------------------ #
    def set_candidates(self, x_num: Tensor, x_cat: Tensor, y: Tensor) -> None:
        """Store the retrieval corpus. ``y`` is int64 class indices for
        classification, or float ``(n, 1)`` (training-scale) for regression."""
        for name, tensor in (("cand_x_num", x_num), ("cand_x_cat", x_cat), ("cand_y", y)):
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor)

    @property
    def has_candidates(self) -> bool:
        return "cand_y" in self._buffers

    # ------------------------------------------------------------------ #
    def _encode(self, x_num: Tensor, x_cat: Tensor) -> tuple[Tensor, Tensor]:
        x = self.linear(self.embedding(x_num, x_cat))
        for block in self.encoder_blocks:
            x = x + block(x)
        k = self.key_proj(x if self.mixer_norm is None else self.mixer_norm(x))
        return x, k

    def _candidate_keys(self) -> Tensor:
        n = self.cand_y.shape[0]
        keys = []
        for start in range(0, n, self.candidate_chunk_size):
            stop = min(start + self.candidate_chunk_size, n)
            keys.append(self._encode(self.cand_x_num[start:stop], self.cand_x_cat[start:stop])[1])
        return torch.cat(keys)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        if not self.has_candidates:
            raise RuntimeError("TabR candidates are not set; call set_candidates first")
        x, k = self._encode(x_num, x_cat)
        n_candidates = self.cand_y.shape[0]
        excluding_self = self.training and self.current_batch_indices is not None
        m = min(self.context_size, n_candidates - int(excluding_self))
        if m < 1:
            raise ValueError("TabR needs at least 2 training rows for retrieval")

        # Search without gradients; only the selected context is re-encoded
        # with gradients below (the original repo's memory_efficient path).
        with torch.no_grad():
            cand_k = self._candidate_keys()
            dists = torch.cdist(k, cand_k)
            if excluding_self:
                rows = torch.arange(k.shape[0], device=k.device)
                dists[rows, self.current_batch_indices] = torch.inf
            context_idx = dists.topk(m, dim=1, largest=False).indices  # (B, m)

        batch_size = k.shape[0]
        flat = context_idx.reshape(-1)
        ctx_k = self._encode(self.cand_x_num[flat], self.cand_x_cat[flat])[1]
        ctx_k = ctx_k.reshape(batch_size, m, -1)

        similarities = -((k[:, None, :] - ctx_k) ** 2).sum(dim=-1)  # (B, m)
        probs = self.context_dropout(torch.softmax(similarities, dim=-1))

        y_ctx = self.cand_y[flat]
        if isinstance(self.label_encoder, nn.Embedding):
            label_emb = self.label_encoder(y_ctx.long())
        else:
            label_emb = self.label_encoder(y_ctx.float().reshape(-1, 1))
        values = label_emb.reshape(batch_size, m, -1) + self.value_transform(
            k[:, None, :] - ctx_k
        )
        x = x + (probs.unsqueeze(1) @ values).squeeze(1)

        for block in self.predictor_blocks:
            x = x + block(x)
        return self.output_layer(torch.relu(self.head_norm(x)))
