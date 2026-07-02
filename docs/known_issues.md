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
  error (documented, not fixable).
- **KI-005 — categorical values are keyed by `str(value)`.** `1` and `"1"`
  in the same column collide into one category. Consistent across fit and
  transform, but a semantic surprise for mixed-type columns.
- **KI-007 — virtualized macOS reports MPS but cannot allocate.** GitHub
  Actions macos-14 runners return `mps.is_available() == True` yet fail on
  the first allocation. `resolve_device` probes with a real allocation
  (`mps_functional()`); `device="auto"` falls back to CPU there, and explicit
  `device="mps"` raises.
- **KI-009 — DANet is slow on GPU.** T4 verification (2026-07-02): 171s
  where peers took 0.2–4s. GhostBatchNorm's per-virtual-batch chunk loop
  plus per-step entmax sorting issue many tiny kernels; needs a fused GBN
  (reshape + one BatchNorm) before DANet is GPU-practical.
- **KI-010 — TabR gains nothing from AMP.** autocast around cdist/topk
  roughly doubled fit time on T4 (10.7s -> 21.3s); use ``amp=False`` for
  ``tabr``.
- **KI-008 — TabR re-encodes all candidates every training step.** The
  retrieval search runs over the whole training set per batch (the original
  design); fine for the small/medium datasets this library targets, O(N)
  per step otherwise. Candidate subsampling and inference-time key caching
  are on the roadmap.
- **KI-006 — custom objective/metric objects are not serialized.**
  `save_model` warns and stores everything needed for prediction; refitting
  a loaded estimator requires re-setting the custom objects.
