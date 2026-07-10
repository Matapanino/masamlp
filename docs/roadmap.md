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
- **TabM-style ensembling** — parameter-efficient ensemble flag for
  `resnet`/`realmlp` (Gorishniy et al. 2024); distinct from the existing
  `n_ens` seed ensembling.
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
- **TPU follow-ups** (0.4.0 shipped single-device TPU/XLA; verdict
  2026-07-10, research/tpu-xla.md §7–9): fuse small-batch training steps
  (`torch_xla.experimental.scan` / multi-step barriers) to cut the
  per-step dispatch floor that makes resnet/realmlp fits dispatch-bound;
  TabR's TPU eval search (topk over barrier-separated chunks, 85.6s vs
  L4 3.1s at 345k); profile tab_transformer on TPU (5x slower than L4);
  optional bf16 prediction (autocast currently wraps training only);
  in-library TPU member sharding (blocked: torch_xla's graph executor is
  not thread-safe across devices — revisit on TorchTPU or via opt-in
  multiprocessing); diagnose the openxla-backend training inaccuracy
  upstream; verify Colab v5e-1/v6e-1 (docs guidance exists, unverified).

## Models

- TabularLNN feature-token sequence mode; revisit against any published
  tabular liquid-network work.
- RealTabR-style hybrid (TabR + RealMLP tricks); FT-Transformer-lite, TabM —
  considered only if they add coverage, not to compete with pytabkit's zoo.

## API

- Public callbacks (LightGBM-style) once the surface stabilizes.
- Optional integration with catstat target encoding (same author) as a
  preprocessing option.
