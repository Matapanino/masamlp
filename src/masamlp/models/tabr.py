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
- Search is a plain ``torch.cdist`` + ``topk`` (no faiss dependency),
  streamed over ``candidate_chunk_size`` blocks so peak memory is
  B x chunk instead of B x N.
- In eval mode the candidate keys are cached across query batches
  (see models/retrieval.py for the invalidation rules).
- During training the trainer exposes the batch's candidate indices
  (``wants_batch_indices``) so each row can exclude itself from its context.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding
from masamlp.models.retrieval import RetrievalBase


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


class TabR(RetrievalBase):
    # Above this corpus size, several eval chunks fuse into one XLA program
    # (barrier every 8 chunks) to win back the cross-chunk optimization the
    # 0.4.0 per-chunk barrier cost: measured on TPU v5e at a 276k corpus,
    # predict 86.1s -> 48.4s, identical values, no OOM (the graphs are
    # HBM-light). Small corpora keep per-chunk barriers: there each chunk's
    # search graph is cheap and the fused mega-graph executes *slower*
    # (40k corpus: 5s -> 14s, measured) — the same compile/size trade that
    # sank xla_fuse_steps. ModernNCA is pinned at 1 (HBM-heavy graphs).
    _EVAL_FUSION_MIN_CANDIDATES = 100_000

    @property
    def xla_eval_sync_chunks(self) -> int:
        if self._eval_sync_override is not None:
            return self._eval_sync_override
        if not self.has_candidates:
            return 1
        return 8 if self.cand_y.shape[0] >= self._EVAL_FUSION_MIN_CANDIDATES else 1

    @xla_eval_sync_chunks.setter
    def xla_eval_sync_chunks(self, value: int) -> None:
        # Escape hatch (also used by benchmarks to measure both regimes).
        self._eval_sync_override = value

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
        self._eval_sync_override: int | None = None
        self.embedding = embedding
        self.context_size = context_size
        self.candidate_chunk_size = candidate_chunk_size
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
    def _encode(self, x_num: Tensor, x_cat: Tensor) -> tuple[Tensor, Tensor]:
        x = self.linear(self.embedding(x_num, x_cat))
        for block in self.encoder_blocks:
            x = x + block(x)
        k = self.key_proj(x if self.mixer_norm is None else self.mixer_norm(x))
        return x, k

    def _candidate_keys(self) -> Tensor:
        return torch.cat(
            [
                self._encode(self.cand_x_num[start:stop], self.cand_x_cat[start:stop])[1]
                for start, stop in self._chunk_bounds(self.cand_y.shape[0])
            ]
        )

    def _search_topk(self, k: Tensor, cand_k: Tensor, m: int, exclude: Tensor | None) -> Tensor:
        """Nearest-candidate indices ``(B, m)`` by L2, streamed over
        ``candidate_chunk_size`` blocks so peak memory is B x chunk. Row ``i``
        excludes global candidate index ``exclude[i]`` when given. Matches an
        unchunked ``cdist(...).topk(...)`` up to ties in the distances."""
        best_vals: Tensor | None = None
        best_idx: Tensor | None = None
        for start, stop in self._chunk_bounds(cand_k.shape[0]):
            dists = torch.cdist(k, cand_k[start:stop])
            if exclude is not None:
                # Scatter +inf into each row's excluded column with static
                # shapes; a nonzero() row gather has a data-dependent size
                # (recompile per batch on XLA, host sync for the size on
                # CUDA). Rows whose exclusion lies outside this chunk write
                # their current value back.
                local = exclude - start
                in_chunk = (local >= 0) & (local < stop - start)
                col = local.clamp(0, stop - start - 1).unsqueeze(1)
                cur = dists.gather(1, col)
                dists.scatter_(
                    1, col, torch.where(in_chunk.unsqueeze(1), torch.inf, cur)
                )
            vals, idx = dists.topk(min(m, stop - start), dim=1, largest=False)
            idx = idx + start
            if best_vals is not None:
                vals = torch.cat([best_vals, vals], dim=1)
                idx = torch.cat([best_idx, idx], dim=1)
            keep = min(m, vals.shape[1])
            best_vals, sel = vals.topk(keep, dim=1, largest=False)
            best_idx = idx.gather(1, sel)
        return best_idx

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
        # The cache gate is evaluated outside no_grad so a grad-enabled eval
        # forward never touches (or creates) the cache.
        cache_usable = self._eval_cache_usable()
        with torch.no_grad():
            if cache_usable:
                cand_k = self._eval_cache_get()
                if cand_k is None:
                    cand_k = self._candidate_keys()
                    self._eval_cache_set(cand_k)
            else:
                cand_k = self._candidate_keys()
            exclude = self.current_batch_indices if excluding_self else None
            context_idx = self._search_topk(k, cand_k, m, exclude)  # (B, m)

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
