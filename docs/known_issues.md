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
- **KI-006 — custom objective/metric objects are not serialized.**
  `save_model` warns and stores everything needed for prediction; refitting
  a loaded estimator requires re-setting the custom objects.
