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
- TabR: cached candidate keys at inference. (Training-time corpus bounding
  shipped as `candidate_budget`.)
- ModernNCA: chunked candidate scoring in the eval path to bound peak memory
  independently of `candidate_budget`.
- DANet inference-time mask fusion (structure re-parameterization from the
  paper) to speed up prediction.
- Single-table categorical embedding (offset trick) when many categorical
  columns are present.

## Models

- TabularLNN feature-token sequence mode; revisit against any published
  tabular liquid-network work.
- RealTabR-style hybrid (TabR + RealMLP tricks); FT-Transformer-lite, TabM —
  considered only if they add coverage, not to compete with pytabkit's zoo.

## API

- Public callbacks (LightGBM-style) once the surface stabilizes.
- Optional integration with catstat target encoding (same author) as a
  preprocessing option.
