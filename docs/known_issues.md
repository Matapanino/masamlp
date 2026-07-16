# Known issues

- **KI-001 — eval_set does not accept weights.** `eval_set=[(X, y, w)]`
  raises; metrics are unweighted in 0.1.x. Roadmap: weighted metric
  contract.
- **KI-002 — torch.compile on hosts without a working C++ toolchain.**
  Inductor fails lazily at the first training step; masaMLP catches this and
  falls back to eager with a warning. Nothing to fix on our side, but the
  warning can surprise users (macOS without full Xcode CLT, minimal Linux
  images).
- **KI-003 — MPS trains in float32 only.** AMP and `torch.compile` are
  gated off on MPS; requesting `amp=True` on MPS warns and runs float32.
- **KI-004 — cross-device reproducibility.** Same seed on the same device
  is reproducible; CPU vs CUDA vs MPS results differ by accumulated float
  error (documented, not fixable). The same applies to multi-GPU sharded
  ensembles (0.3.0): reproducible for a fixed GPU topology, expected — but
  not guaranteed — to match a single-device run on identical GPU models.
- **KI-005 — categorical values are keyed by `str(value)`.** `1` and `"1"`
  in the same column collide into one category. Consistent across fit and
  transform, but a semantic surprise for mixed-type columns.
- **KI-007 — virtualized macOS reports MPS but cannot allocate.** GitHub
  Actions macos-14 runners return `mps.is_available() == True` yet fail on
  the first allocation. `resolve_device` probes with a real allocation
  (`mps_functional()`); `device="auto"` falls back to CPU there, and explicit
  `device="mps"` raises.
- **KI-009 — DANet slow on GPU (RESOLVED 2026-07-02).** Root cause was the
  grouped 1x1 Conv1d slow path (76% of step time), not primarily
  GhostBatchNorm. Fixed by computing the grouped projection as a batched
  einsum over the same parameters and fusing GBN's chunk loop: T4 171.5s ->
  3.4s, CPU 251.8s -> 17.6s on the smoke workload, identical math.
- **KI-010 — TabR gains nothing from AMP (RESOLVED 0.3.0).** autocast
  around cdist/topk roughly doubled fit time on T4 (10.7s -> 21.3s). Fixed
  by the per-model AMP policy: ``amp="auto"`` now resolves to off for the
  retrieval models (``amp_auto = False`` on ``tabr``/``modernnca``); an
  explicit ``amp=True`` still forces it.
- **KI-008 — TabR re-encodes all candidates every training step
  (inference side RESOLVED 0.3.0).** The retrieval search runs over the
  whole training set per batch (the original design); at *training* time
  this remains O(N) per step by design — bound the corpus with
  ``candidate_budget`` (0.2.0). The *inference* side is fixed in 0.3.0:
  candidate keys (TabR) / encodings (ModernNCA) are computed once per
  predict pass and cached, and scoring is streamed in
  ``candidate_chunk_size`` blocks, so eval peak memory is B x chunk instead
  of the B x N matrix that OOM'd ModernNCA at 345k rows.
- **KI-006 — custom objective/metric objects are not serialized.**
  `save_model` warns and stores everything needed for prediction; refitting
  a loaded estimator requires re-setting the custom objects.
- **KI-011 — DANet non-finite trajectory / entmax crash (RESOLVED 0.2.0).**
  At real-data scale `danet` could diverge and crash inside `entmax15`
  (`index -1 is out of bounds` on CPU / CUDA device-side assert). The genesis
  was entmax's `sqrt` at the support boundary (infinite gradient) poisoning
  DANet's raw `mask_weight` to NaN. Fixed with a gradient-bounded `sqrt` plus a
  `k_star >= 1` clamp so any residual non-finite input degrades to a clean
  non-finite-loss error instead of a hard crash.
- **KI-012 — early stopping on a discrete task metric is noisy.** On
  imbalanced tasks, stopping on `balanced_accuracy` (or accuracy) was
  measurably worse than stopping on a probability-quality metric. Prefer
  `eval_metric="logloss"`/`"multi_logloss"` for the stop signal and convert to
  your task metric post-hoc; a discrete metric is a jumpy early-stopping
  signal.
- **KI-013 — tab_transformer trains slowly on TPU: the attention backward
  is the tax.** Profiled on v5e (batch 1024, bf16): the full train step is
  ~100 ms/iter while the forward is ~8 ms — `nn.MultiheadAttention` at
  `d_token=32`/`n_heads=8` (head_dim 4) lowers to MXU-hostile small ops in
  reverse mode (no aten fallbacks, no recompiles — pure lowering quality).
  Mitigations: larger `d_token`/fewer heads help the shapes; an SDPA-based
  attention block is on the roadmap. GPU is unaffected.
- **KI-014 — tabm measurement coverage (0.6.0).** Multiclass ensembling
  quality was measured in the S6E7 campaign (and per-layer naive
  BatchEnsemble rejected there); **binary and regression are
  contract-tested only** — exact sample_weight semantics, custom-objective
  transparency, and equivalence pins in `tests/test_tabm.py`, but no
  at-scale quality verdict yet (next campaign). TabM also has **no TPU
  verdict**: it runs in the XLA CI smoke (`PJRT_DEVICE=CPU`), and the
  README scopes the TPU-verified claim to the ten 0.5.0 models.
