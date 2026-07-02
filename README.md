# masaMLP

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**Extensible tabular deep learning** — TabularResNet, DANet, and TabularLNN
behind sklearn-compatible estimators with first-class **sample_weight**,
**custom objectives**, **custom metrics**, and **early stopping on any
metric**. The sibling library of
[repleafgbm](https://github.com/Matapanino/repleafgbm) (same author, same API
philosophy), for the neural side of tabular ML.

> **Status: alpha (0.1.x).** Built with heavy use of
> [Claude Code](https://claude.com/claude-code) (coding and architecture
> design).

## Why masaMLP

Excellent tabular DL libraries exist — [pytabkit](https://github.com/dholzmueller/pytabkit)
ships state-of-the-art models like RealMLP and TabM, and
[rtdl](https://github.com/yandex-research/rtdl) provides reference modules.
What they don't make easy is *extension*: `sample_weight` in `fit`, custom
training losses, custom evaluation metrics, and early stopping driven by
them. masaMLP is built around exactly those hooks:

- **`fit(X, y, sample_weight=..., eval_set=...)`** — LightGBM-style, sklearn
  compatible. Weights flow through a single reduction
  `(loss * w).sum() / w.sum()` that every objective shares.
- **Custom objectives** are per-sample torch losses — a plain function (or
  `nn.Module` with trainable parameters). Because the trainer owns the
  weighted reduction, your loss gets correct `sample_weight` and
  `class_weight` handling for free.
- **Custom metrics** are plain NumPy callables via `make_metric`, and any of
  them (minimize or maximize) can drive early stopping with best-epoch weight
  restoration.
- **Multiclass, multioutput regression, class_weight, label smoothing**
  supported natively; built-in preprocessing (quantile scaling, missing
  values, categorical embeddings) so DataFrames go straight into `fit`.
- **CPU / CUDA / MPS** behind `device="auto"`: device-resident tensors with
  no DataLoader overhead, automatic full-batch mode for small data, bf16 AMP
  on CUDA, opt-in `torch.compile` with eager fallback.

masaMLP deliberately does *not* try to re-benchmark the field — see
[docs/attribution.md](docs/attribution.md) for the research and libraries it
builds on.

## Models

| name | source | notes |
|---|---|---|
| `resnet` | Gorishniy et al. 2021 (arXiv:2106.11959) | default; strong baseline |
| `realmlp` | Holzmüller et al. 2024 (arXiv:2407.04491) | RealMLP-TD-S architecture (scaling layer, NTP linear layers, SELU/Mish); pair with `masamlp.realmlp_params(task)` for the full training recipe |
| `ft_transformer` | Gorishniy et al. 2021 (arXiv:2106.11959) | feature tokens + [CLS] + PreNorm/ReGLU transformer, per the rtdl reference |
| `tab_transformer` | Huang et al. 2020 (arXiv:2012.06678) | transformer over categorical tokens; numerics bypass (or embed via `num_embedding`) |
| `danet` | Chen et al. AAAI 2022 (arXiv:2112.02962) | Abstract Layers with learnable sparse feature groups (in-house entmax15) |
| `tabr` | Gorishniy et al. 2023 (arXiv:2307.14338) | retrieval-augmented: nearest training rows are aggregated into each prediction |
| `modernnca` | Ye et al. 2024 (arXiv:2407.03257) | soft-nearest-neighbor aggregation with stochastic candidate sampling; pairs well with `num_embedding="plr-lite"` |
| `gandalf` | Joseph & Raj 2022 (arXiv:2207.08548) | GFLU stages: learnable sparse feature masks (t-softmax) with GRU-style gating; exposes `feature_importances()` |
| `grn` | GRN blocks from TFT, Lim et al. 2021 (arXiv:1912.09363) | stack of Gated Residual Networks over embedded features (masaMLP's own composition) |
| `lnn` | CfC cells, Hasani et al. 2022 | **experimental** liquid-network adaptation for static tabular data — see [docs/lnn.md](docs/lnn.md) |

Third-party architectures plug in with `register_model` and get the whole
estimator surface (weights, objectives, metrics, early stopping) for free.

### RealMLP insights are composable options

The tricks from the RealMLP paper are estimator-level options usable with
*any* model (`lnn` included), not baked into one architecture:

- `numeric_scaler="rssc"` — robust scale + smooth clip preprocessing
- `cat_encoding="onehot"` — RealMLP-style one-hot (binary → ±1, missing → 0)
- `num_embedding="pbld" | "plr" | "plr-lite" | "pl" | "periodic"` — the
  numeric embedding zoo (arXiv:2203.05556 + PBLD); token models
  (`ft_transformer`, `tab_transformer`) use the same options as feature
  tokenizers
- `model_params={"num_scaling": True}` — learnable per-feature input scale
- `lr_scheduler="coslog4"`, `optimizer_betas=(0.9, 0.95)` — the training
  schedule
- `clip_predictions=True` (regressor) — clip to the observed target range
- `n_ens=k` — seed ensembling as in pytabkit's RealMLP: k members trained
  with seeds `random_state + i`, predictions averaged on the probability /
  value scale; works with every model including the retrieval ones.
  `ens_mode="vectorized"` trains all members in one vmapped forward/backward
  (`torch.func`) for BatchNorm-free models — pytabkit's speed trick
- `weight_decay_schedule="flat_cos"` — RealMLP-TD's scheduled weight decay
  (param groups can opt out, e.g. biases)
- `masamlp.realmlp_td_params(task)` — the **full RealMLP-TD recipe**:
  parametric activations, flat_cos-scheduled dropout and weight decay, PBLD
  embeddings with their own lr factor, and hybrid categorical encoding
  (one-hot ≤ 9 categories, embeddings above)

```python
from masamlp import MasaClassifier, realmlp_params

clf = MasaClassifier(**realmlp_params("classification"))    # the TD-S recipe
clf = MasaClassifier(**{**realmlp_params("classification"),
                        "num_embedding": "pbld"})           # toward RealMLP-TD
```

## Install

```bash
pip install masamlp        # torch, numpy, pandas, scikit-learn
```

## Quickstart

```python
import numpy as np
from masamlp import MasaClassifier, make_metric

def f1(y_true, y_proba):
    pred = y_proba >= 0.5
    tp = np.sum(pred & (y_true == 1))
    return 2 * tp / max(pred.sum() + (y_true == 1).sum(), 1)

clf = MasaClassifier(
    model="resnet",
    eval_metric=make_metric(f1, name="f1", minimize=False),
    early_stopping_rounds=15,
    class_weight="balanced",
)
clf.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)])
proba = clf.predict_proba(X_test)
print(clf.best_iteration_, clf.best_score_, clf.evals_result_["valid_0"]["f1"][:3])
```

Custom objective (regression, asymmetric loss):

```python
import torch
from masamlp import MasaRegressor

def asymmetric_mse(y_true, raw_pred):          # -> per-sample (n,) tensor
    err = raw_pred - y_true                    # raw_pred: (n, out_dim)
    return torch.where(err < 0, 4.0 * err**2, err**2).mean(dim=1)

reg = MasaRegressor(model="danet", objective=asymmetric_mse)
reg.fit(X, y, sample_weight=w)                 # weights just work
```

Save/load is a plain directory (`manifest.json` + tensors, loaded with
`weights_only=True` — no pickle execution):

```python
reg.save_model("model_dir")
reg2 = MasaRegressor.load_model("model_dir")
```

## Devices

`device="auto"` resolves cuda > mps > cpu. CUDA gets bf16 AMP by default and
optional `compile=True`; MPS and CPU train in float32. Details and caveats:
[docs/devices.md](docs/devices.md).

## Development

```bash
pip install -e ".[dev]"
bash scripts/check.sh      # ruff + pytest + examples/quickstart.py
```

Development rules live in [CLAUDE.md](CLAUDE.md); roadmap in
[docs/roadmap.md](docs/roadmap.md).

## License

MIT. Architecture attributions: [docs/attribution.md](docs/attribution.md).
