# Roadmap

## Near-term

- **Weighted eval sets** — accept `(X, y, w)` in `eval_set` and a weighted
  metric contract (repleafgbm-compatible extension of `BaseMetric`).
- **Generic schedule/param-group levers** — promote label-smoothing and
  dropout schedules (currently RealMLP-only) to estimator options, and add a
  `param_group_overrides` escape hatch for per-group lr/wd multipliers.
  (EMA weight averaging shipped as `ema_decay` in 0.2.0.)
- **RealMLP-TD remaining bits** — pytabkit's data-driven init modes
  (`he+5`/`std`) and coupled-Adam weight decay; `drop_last`-style batching
  option. (Parametric activations, wd/dropout schedules, PBLD factors, and
  hybrid categorical encoding shipped in `realmlp_td_params`.)
- **HPO presets** — `model_params` presets per model ("fast" / "accurate";
  DANet paper depths 20/24/32 as named presets).
- **Inner-ensemble follow-ups (ADR 0005; `tabm` 0.6.0, `predict_members` 0.7.0,
  `ft_transformer` inner `k` 0.8.0):**
  (a) `ft_transformer` inner `k` via a per-member token adapter — **shipped in
  0.8.0** (ADR 0005 §4). The pre-registered S6E7 gate passed: inner-`k=4`
  improved a single FT-Transformer by −0.00038 nat (6/7 folds, 690k, 7-fold
  prior-free balanced logloss) — a real gain, though an independent seed
  ensemble (`n_ens`) remained the stronger variance reducer, so it is a
  single-fit efficiency feature. `danet` already uses `k` for feature groups —
  the parameter docs disambiguate. (b) `resnet`/`realmlp` inner `k` —
  UNFROZEN now (a) has a positive verdict; a candidate for a future round,
  though the S6E7 read (inner-`k` < a seed ensemble) lowers its priority.
- **Vectorized `n_ens` extensions** — shipped for BatchNorm-free models
  (`ens_mode="vectorized"`); remaining: AMP and grad-clip support inside the
  vmapped path, GhostBatchNorm-aware variant for DANet.
- **Per-model estimator aliases** — `ResNetRegressor` etc., thin subclasses.

## Performance

- `torch.compile` tuning (mode/dynamic settings) and CUDA graphs for the
  full-batch path.
- Multi-GPU: per-GPU vmapped subgroups for `ens_mode="vectorized"`
  (loop-mode member sharding shipped in 0.3.0), AMP/grad-clip inside the
  vmapped path.
- Dedupe retrieval candidate buffers in multi-member checkpoints (the corpus
  is currently serialized once per member in `model_state.pt`).
- DANet inference-time mask fusion (structure re-parameterization from the
  paper) to speed up prediction.
- Single-table categorical embedding (offset trick) when many categorical
  columns are present.
- (Shipped in 0.3.0: TabR inference-time key caching, ModernNCA chunked
  eval scoring, per-model AMP auto-policy, multi-GPU member sharding.)
- **TPU follow-ups** (0.4.0 shipped single-device TPU/XLA; 0.5.0 addressed
  the list — see CHANGELOG and verdicts/2026-07-12): still open after the
  0.5.0 measurements: (a) in-library TPU member sharding — decided against
  for now in ADR 0004; re-evaluate when TorchTPU is public or torch_xla's
  executor becomes thread-safe across devices; (b) step-loop-in-graph via
  `scan` — blocked on torch_xla 2.8 (`torch.func.grad` fails in the scan
  body), the only remaining route to the small-batch dispatch floor since
  unrolled K-step fusion measured as a compile-cost loss; (c) an SDPA-based
  attention block for `tab_transformer` (KI-013 — the TPU attention
  backward is ~92% of its step); (d) a self-contained repro of the openxla
  dynamo miscompile (minimal archs did not reproduce it) before filing
  upstream.

## Models

- TabularLNN feature-token sequence mode; revisit against any published
  tabular liquid-network work.
- RealTabR-style hybrid (TabR + RealMLP tricks); FT-Transformer-lite —
  considered only if they add coverage, not to compete with pytabkit's zoo.

## API

- Public callbacks (LightGBM-style) once the surface stabilizes.
- Optional integration with catstat target encoding (same author) as a
  preprocessing option.
