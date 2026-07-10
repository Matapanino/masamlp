"""Shared machinery for retrieval-based models (TabR, ModernNCA).

Both models keep the training set as a retrieval corpus in registered
buffers (``cand_x_num``/``cand_x_cat``/``cand_y``), so it moves with
``.to(device)`` and is saved/loaded through ``state_dict``, and both use the
trainer's batch-index protocol (``wants_batch_indices``) for self-exclusion.

``RetrievalBase`` also owns the eval-time encoding cache: in eval mode the
corpus encoding depends only on the parameters and the corpus, so it is
computed once and reused across query batches instead of being recomputed
for every batch (the KI-008 inference cost). The cache is a plain attribute
— never a buffer — so it stays out of ``state_dict``. It must be dropped
whenever parameters, corpus, or device can have changed:

1. ``train(True)`` — the optimizer is about to update the encoder;
2. ``set_candidates`` — the corpus changed;
3. ``_apply`` (``.to()``/``.cpu()``/``.half()``) — the cache tensor is not
   moved along with the module;
4. ``load_state_dict`` (post-hook) — covers the trainer's best-epoch restore
   and deserialization;
5. the trainer's EMA parameter swap (``_swap_in_params`` calls
   ``invalidate_eval_cache`` duck-typed) — parameters change in-place with
   no mode transition.

The cache is only built and used in no-autograd contexts (inference_mode or
grad disabled); a grad-enabled eval forward computes everything fresh so
autograd semantics are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn


class RetrievalBase(nn.Module):
    wants_batch_indices = True
    # KI-010 is a CUDA finding: autocast around the cdist/topk search is
    # slower there and fp16 distances are accuracy-risky. On XLA the MXU is
    # bf16-native — measured on TPU v5e at 345k rows, bf16 matches fp32
    # predictions exactly and fits 1.5x faster (predicts 7x faster).
    amp_auto = {"cuda": False}
    # Buffers that never change during fit; the trainer skips them in
    # early-stopping snapshots and restores with strict=False.
    static_state_keys = ("cand_x_num", "cand_x_cat", "cand_y")

    def __init__(self) -> None:
        super().__init__()
        self.current_batch_indices: Tensor | None = None
        # Payload is subclass-defined (TabR: candidate keys; ModernNCA:
        # (encoded corpus, label representation)).
        self._eval_cache: Any = None
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: module.invalidate_eval_cache()
        )

    def _chunk_bounds(self, n: int) -> Iterator[tuple[int, int]]:
        """(start, stop) pairs covering ``range(n)`` in
        ``candidate_chunk_size`` steps (subclasses set the attribute)."""
        for start in range(0, n, self.candidate_chunk_size):
            yield start, min(start + self.candidate_chunk_size, n)

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
        self.invalidate_eval_cache()

    @property
    def has_candidates(self) -> bool:
        return "cand_y" in self._buffers

    # ------------------------------------------------------------------ #
    # Eval-time corpus-encoding cache
    # ------------------------------------------------------------------ #
    def invalidate_eval_cache(self) -> None:
        self._eval_cache = None

    def _eval_cache_usable(self) -> bool:
        """Cache only in eval mode and only where autograd cannot observe it
        (a cached tensor may be an inference tensor from a prior
        ``inference_mode`` pass; grad-enabled eval computes fresh instead)."""
        return not self.training and (
            torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
        )

    def train(self, mode: bool = True) -> RetrievalBase:
        if mode:
            self.invalidate_eval_cache()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        self.invalidate_eval_cache()
        return super()._apply(fn, recurse)
