# Roadmap

## Near-term

- **Weighted eval sets** — accept `(X, y, w)` in `eval_set` and a weighted
  metric contract (repleafgbm-compatible extension of `BaseMetric`).
- **RealMLP-TD completion** — parametric activations, weight-decay schedule,
  categorical embeddings for high-cardinality columns under `onehot` mode,
  `drop_last`-style batching option.
- **HPO presets** — `model_params` presets per model ("fast" / "accurate";
  DANet paper depths 20/24/32 as named presets).
- **TabM-style ensembling** — parameter-efficient ensemble flag for
  `resnet`/`realmlp` (Gorishniy et al. 2024); distinct from the existing
  `n_ens` seed ensembling.
- **Vectorized `n_ens`** — train members jointly via `torch.func`
  (stacked parameters + vmap, pytabkit's speed trick) for BatchNorm-free
  models; the current implementation loops.
- **Per-model estimator aliases** — `ResNetRegressor` etc., thin subclasses.

## Performance

- `torch.compile` tuning (mode/dynamic settings) and CUDA graphs for the
  full-batch path.
- TabR: cached candidate keys at inference; optional candidate subsampling
  during training for larger datasets.
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
