# Parameter reference

Every knob in masaMLP lives in one of three places:

- **Estimator parameters** — constructor arguments of `MasaRegressor` /
  `MasaClassifier` (training loop, preprocessing, hardware). Listed first.
- **`model_params`** — a dict forwarded to the selected architecture's
  constructor. Depth, width, dropout, and every other architecture knob is
  free per model; the per-architecture tables below are the complete list.
- **Shared embedding keys** — five keys accepted *inside* `model_params` for
  every model; they configure the feature embedding rather than the trunk.

```python
from masamlp import MasaClassifier

clf = MasaClassifier(
    model="ft_transformer",
    model_params={"n_blocks": 4, "d_block": 256,     # architecture knobs
                  "cat_emb_dim": 12},                # shared embedding key
    learning_rate=3e-4,                              # estimator parameter
)
```

Unknown `model_params` keys raise a `ValueError` listing the valid keys for
the selected model.

This file is checked against the code by `tests/test_docs_parameters.py`:
every constructor parameter must appear here, so the tables cannot silently
go stale.

## Estimator parameters

Shared by `MasaRegressor` and `MasaClassifier` unless marked otherwise.

### Model and objective

| Parameter | Default | Meaning |
|---|---|---|
| `model` | `"resnet"` | Architecture name (see [per-architecture tables](#per-architecture-parameters)). Third-party models plug in via `register_model`. |
| `model_params` | `None` | Dict of architecture kwargs + shared embedding keys; merged over per-model defaults (e.g. the classifier sets RealMLP's `activation="selu"`, and retrieval models get `n_label_classes` injected automatically). The resolved dict is stored as `resolved_model_params_` after `fit`. |
| `objective` | `None` | Training loss. `None` picks the task default; a string picks a built-in (table below); a callable / `BaseObjective` is a custom per-sample torch loss (returns an `(n,)` tensor — the trainer owns the weighted reduction, so `sample_weight` works for free). |
| `eval_metric` | `None` | Metric(s) computed on each `eval_set` after every epoch: a string (table below), a `make_metric(...)` result, a plain NumPy callable, or a list of these. The first metric on `valid_0` drives early stopping. `None` = task default (`rmse` / `logloss` / `multi_logloss`). |

Built-in objectives (string values; regression unless noted):

| Value | Loss |
|---|---|
| `"squared_error"` (aliases `"l2"`, `"mse"`) | Squared error — regressor default. |
| `"mae"` (alias `"l1"`) | Absolute error. |
| `"huber"` | Huber loss; for a custom `delta` pass an instance: `masamlp.core.objectives.Huber(delta=2.0)`. |
| `"quantile"` | Pinball loss; pass `masamlp.core.objectives.Quantile(alpha=0.9)` for other quantiles. |
| `"poisson"` | Poisson deviance with log link (predictions come out through `exp`; target standardization is skipped). |
| `"binary_logistic"` | Logistic loss — classifier default for 2 classes. |
| `"multiclass_softmax"` | Softmax cross-entropy — classifier default for 3+ classes. |

Built-in metrics (string values):

| Value | Task | Notes |
|---|---|---|
| `"rmse"`, `"mae"` | regression | Computed on the original (un-standardized) target scale. |
| `"logloss"`, `"multi_logloss"` | classification | Task defaults (binary / multiclass). |
| `"accuracy"`, `"balanced_accuracy"` | classification | Thresholded / argmax predictions. |
| `"auc"` | binary classification | Rank-based (Mann–Whitney U), tie-aware. |

### Training loop

| Parameter | Default | Meaning |
|---|---|---|
| `early_stopping_rounds` | `None` | Patience in epochs on the first metric of `valid_0`; restores the best epoch's weights. Requires `eval_set`. |
| `n_epochs` | `256` | Maximum epochs (early stopping may end training sooner). |
| `batch_size` | `"auto"` | `"auto"`: full-batch when the training set has ≤ 4096 rows, else minibatches of 1024. `None`: always full-batch. An int is used as-is (capped at the row count). |
| `eval_batch_size` | `8192` | Forward-pass batch for evaluation and `predict`; a pure memory knob with no effect on results. |
| `learning_rate` | `1e-3` | Optimizer learning rate. Models may scale it per parameter group (RealMLP does). |
| `weight_decay` | `0.0` | Decoupled weight decay (AdamW-style). |
| `optimizer` | `"adamw"` | `"adamw"`, `"adam"`, or `"sgd"`. |
| `optimizer_betas` | `None` | `(beta1, beta2)` for adam/adamw (`None` = torch defaults). The RealMLP recipe uses `(0.9, 0.95)`. |
| `lr_scheduler` | `"none"` | `"none"`, `"cosine"` (per-epoch cosine annealing), or `"coslog4"` (RealMLP's per-step schedule). |
| `weight_decay_schedule` | `"none"` | `"none"` or `"flat_cos"` (RealMLP-TD: constant for the first half of training, cosine decay after; param groups can opt out, e.g. biases). |
| `grad_clip` | `None` | Global gradient-norm clip; `None` disables. |
| `ema_decay` | `None` | Exponential moving average (Polyak averaging) of the weights, e.g. `0.999`; evaluation, early stopping, and the final model use the averaged parameters. Not supported with `ens_mode="vectorized"`. |

### Preprocessing and numeric embeddings

| Parameter | Default | Meaning |
|---|---|---|
| `numeric_scaler` | `"quantile"` | `"quantile"` (rank → normal), `"standard"`, `"robust"`, `"rssc"` (RealMLP's robust-scale-smooth-clip), or `"none"`. Fitted on the training set only. |
| `categorical_features` | `"auto"` | `"auto"` detects categorical columns from DataFrame dtypes; or pass a list of column names/indices. |
| `cat_encoding` | `"embedding"` | `"embedding"` (per-column `nn.Embedding`, index 0 reserved for unknown/missing), `"onehot"` (RealMLP-style: binary → ±1, missing → 0), or `"hybrid"` (one-hot up to 9 categories, embeddings above — RealMLP-TD). |
| `num_embedding` | `None` | Numeric-feature embedding: `None` (raw), `"periodic"`, or the PLR family `"pl"` / `"plr"` / `"plr-lite"` / `"pbld"` (arXiv:2203.05556 + PBLD per pytabkit). Token models (`ft_transformer`, `tab_transformer`) accept the PLR family (not `"periodic"`) as their feature tokenizer. Tune the embedding itself via the [shared embedding keys](#shared-embedding-keys-inside-model_params). |

### Ensembling

| Parameter | Default | Meaning |
|---|---|---|
| `n_ens` | `1` | Seed-ensemble size: members train with seeds `random_state + i` and predictions are averaged on the probability/value scale. |
| `ens_mode` | `"loop"` | `"loop"` trains members sequentially (sharded across GPUs when several are visible). `"vectorized"` trains all members in one vmapped forward/backward (`torch.func`) for a ~k× speedup — BatchNorm-free models only (`grn`, `realmlp`, `ft_transformer`, `gandalf`, `lnn`). |
| `candidate_budget` | `None` | For `tabr` / `modernnca`: bounds the retrieval corpus with a seeded, class-stratified subsample of at most this many training rows (memory/compute control on large data). No-op for other models. |

`model="tabm"` adds a second, **inner** ensemble axis: `k` weight-shared
members inside one model (see the [`tabm` section](#tabm--tabm-parameter-efficient-deep-ensemble)).
It is orthogonal to `n_ens` and the two compose — `n_ens=m` trains `m`
independent TabM models for `m·k` members in total, in any `ens_mode`.

### Hardware and execution

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `"auto"` | `"auto"` resolves tpu > cuda > mps > cpu; or an explicit `"cpu"` / `"cuda"` / `"cuda:0"` / `"mps"` / `"xla"` / `"tpu"` (experimental — `"tpu"` requires the XLA backend to really be a TPU, `"xla"` accepts any, e.g. `PJRT_DEVICE=CPU`; needs `torch_xla`). With multiple GPUs and `n_ens > 1`, members train concurrently, one worker per GPU (see [devices.md](devices.md)). |
| `amp` | `"auto"` | Mixed precision on CUDA and XLA: `"auto"` follows each model's policy (bf16 by default; retrieval models opt out on CUDA only — on TPUs bf16 measured moderately faster at equivalent rmse; `ft_transformer` accepts bf16 but not fp16), `True`/`"on"` forces AMP, `False`/`"off"` disables it. TPUs always use bf16 (never fp16) and need no gradient scaling. Wraps the training step only — see `amp_predict` for inference. |
| `amp_predict` | `False` | Opt-in bf16 autocast for evaluation and `predict` (training `amp` never covers them). `True`/`"on"` casts on XLA/TPU, bf16-capable CUDA, and CPU; warns and stays fp32 elsewhere. fp16 is never used. Expect prediction-scale differences at bf16 precision (~3 significant decimal digits); metrics and early stopping see the bf16 predictions when enabled. Measured on TPU v5e: rmse-equivalent, speed neutral (fp32 prediction is already fast there) — a memory/marginal knob. |
| `xla_fuse_steps` | `1` | XLA/TPU only (no effect on other devices): optimizer steps per XLA graph barrier. `K > 1` fuses `K` steps into one XLA program. **Measured on TPU v5e: keep 1** — compile time grows super-linearly with the fused graph and outweighs the ~20% steady-state dispatch saving at realistic fit lengths (see devices.md); an escape hatch for very long fits only. Deterministic for a fixed `K` (same seed + same `K` ⇒ same model); models whose training step draws device RNG (dropout, retrieval sampling) draw a different — equally random — stream under a different `K`, like changing `batch_size` does. RNG-free training is `K`-invariant. |
| `compile` | `False` | Opt-in `torch.compile` with eager fallback. Refused (warning) on XLA — the openxla backend trained inaccurately in TPU verification; lazy-tensor mode is the XLA path. |
| `n_threads` | `None` | Caps torch CPU threads (`None` = torch default). |
| `verbose` | `0` | `0` is silent; `k > 0` logs the metrics every `k` epochs. |
| `random_state` | `42` | Seed for init, shuffling, and subsampling; same seed ⇒ same model. `None` leaves the RNGs unseeded. |

### MasaClassifier only

| Parameter | Default | Meaning |
|---|---|---|
| `class_weight` | `None` | `"balanced"` or a `{label: weight}` dict; multiplied into `sample_weight` before the shared weighted reduction, so it composes with custom objectives. |
| `label_smoothing` | `0.0` | Label smoothing for the built-in logistic/softmax objectives. |

### MasaRegressor only

| Parameter | Default | Meaning |
|---|---|---|
| `target_standardize` | `True` | Standardize the target for training; predictions are transformed back, and metrics are computed on the original scale. Skipped for log-link objectives (`"poisson"`). |
| `clip_predictions` | `False` | Clip predictions to the observed target range (RealMLP-style). |

## Shared embedding keys (inside `model_params`)

Accepted in `model_params` for **every** model; they configure the
`FeatureEmbedding` / token embedding, not the trunk.

| Key | Default | Meaning |
|---|---|---|
| `d_num_embedding` | `16` | Output dim per numeric feature for the PLR-family `num_embedding`. Token models ignore it in favor of the token width. |
| `n_frequencies` | `16` | Number of random frequencies for `"periodic"`/PLR embeddings. |
| `sigma` | `0.1` | Scale of the initial frequencies (`c ~ N(0, sigma²)`); the most sensitive periodic-embedding knob in arXiv:2203.05556. |
| `cat_emb_dim` | `None` | Per-column categorical embedding dim; `None` = auto (`min(32, max(2, round(1.6 · cardinality^0.56)))`). Token models ignore it (tokens are `d_token`/`d_block` wide). |
| `num_scaling` | `False` | Learnable per-feature scale on numeric inputs before embedding (RealMLP's scaling layer, trained at 6× lr there). Defaults to `True` when `model="realmlp"`. |

## Per-architecture parameters

Everything below goes in `model_params`. Defaults follow each paper's
reference configuration (see [attribution.md](attribution.md)); ranges cited
in the sizing notes come from the named papers or reference packages — where
a model has no published tuning space, the note only says what each knob
does.

### `resnet` — TabularResNet

| Parameter | Default | Meaning |
|---|---|---|
| `n_blocks` | `3` | Number of residual blocks (depth). |
| `d` | `192` | Main width: block input/output dim, projected from the embedding. |
| `d_hidden_factor` | `2.0` | In-block expansion: hidden dim = `d × d_hidden_factor`. |
| `dropout1` | `0.25` | Dropout after the in-block ReLU — the main regularizer. |
| `dropout2` | `0.0` | Dropout after the second in-block linear (before the skip add). |

**Sizing notes.** Depth and width are `n_blocks` and `d`; defaults are the
paper's baseline configuration (Gorishniy et al. 2021, arXiv:2106.11959).
Regularize with `dropout1` first; `dropout2` stays 0 in the reference
defaults.

### `tabm` — TabM (parameter-efficient deep ensemble)

| Parameter | Default | Meaning |
|---|---|---|
| `k` | `32` | Ensemble members. All members share the embedding and the MLP backbone; each gets its own multiplicative adapter on the embedding and its own output head, so the parameter cost stays ~1× a single MLP. `k=1` is a plain MLP. |
| `d` | `512` | Backbone width. |
| `n_blocks` | `3` | Backbone depth (Linear → ReLU → Dropout blocks). |
| `dropout` | `0.1` | Dropout after every backbone activation. |
| `adapter_std` | `0.5` | Std of the per-member adapter init `N(1, adapter_std)`: members start near the single model and diverge during training (masaMLP's init — the paper's per-layer ±1 sign adapters measured *worse* than a single model here and were rejected). |

**Sizing notes.** Defaults are the paper's reference configuration
(Gorishniy et al. 2024, arXiv:2410.24210; the TabM-mini structure). The `k`
members train as independent predictors on every row — every objective
(including custom ones) and `sample_weight` work unchanged — and
predictions are averaged on the probability/value scale. Early stopping
monitors the ensemble-average metric (per-member stopping is undefined on
shared weights). `k` is an inner, weight-shared axis and composes with the
outer seed ensemble `n_ens`. Pairs well with `num_embedding="plr-lite"`
(the paper's TabM†).

### `realmlp` — RealMLP-TD-S

| Parameter | Default | Meaning |
|---|---|---|
| `hidden_sizes` | `(256, 256, 256)` | Width of every hidden layer, fully free — e.g. `(512, 256, 128)` gives a tapered 3-layer MLP. |
| `activation` | `"mish"` | `"mish"`, `"selu"`, or `"relu"`. The classifier switches the default to `"selu"` (RealMLP-TD-S uses SELU for classification). |
| `dropout` | `0.0` | Dropout after each activation. The TD recipe uses 0.15, scheduled. |
| `dropout_schedule` | `"none"` | `"none"` or `"flat_cos"` (dropout decays over training, RealMLP-TD). |
| `use_parametric_act` | `False` | Learnable per-unit activation scale (RealMLP-TD), trained at `act_lr_factor`. |
| `act_lr_factor` | `0.1` | Learning-rate factor for the parametric activations. |
| `plr_lr_factor` | `1.0` | Learning-rate factor for the numeric-embedding parameters (RealMLP-TD uses 0.1 with PBLD). |

**Sizing notes.** `hidden_sizes` is the whole architecture: length = depth,
entries = per-layer width. The `(256, 256, 256)` default is the RealMLP-TD(-S)
architecture of Holzmüller et al. 2024 (arXiv:2407.04491). This model is
about its *training recipe* as much as its shape — pair it with
`masamlp.realmlp_params(task)` or `realmlp_td_params(task)` rather than
tuning in isolation.

### `ft_transformer` — FT-Transformer

| Parameter | Default | Meaning |
|---|---|---|
| `n_blocks` | `3` | Transformer blocks (depth). |
| `d_block` | `192` | Token width; must be divisible by `attention_n_heads`. |
| `attention_n_heads` | `8` | Attention heads. |
| `attention_dropout` | `0.2` | Dropout inside multi-head attention. |
| `ffn_d_hidden_multiplier` | `4/3` | FFN hidden dim = `d_block × multiplier` (ReGLU feed-forward). |
| `ffn_dropout` | `0.1` | Dropout inside the FFN. |
| `residual_dropout` | `0.0` | Dropout on both residual branches. |

**Sizing notes.** Defaults are the reference package's
`get_default_kwargs(n_blocks=3)` (rtdl_revisiting_models, arXiv:2106.11959);
the reference scales `d_block` and the dropouts together with `n_blocks` in
its presets, so when you deepen the model, widen it too. Compute grows
quadratically with the number of features (attention over one token per
feature).

### `tab_transformer` — TabTransformer

| Parameter | Default | Meaning |
|---|---|---|
| `d_token` | `32` | Width of the categorical tokens; must be divisible by `n_heads`. |
| `n_layers` | `6` | Transformer blocks over the categorical tokens (depth). |
| `n_heads` | `8` | Attention heads. |
| `ffn_d_hidden_multiplier` | `4.0` | FFN hidden dim = `d_token × multiplier`. |
| `dropout` | `0.1` | Dropout in attention and FFN. |
| `head_dropout` | `0.0` | Dropout in the final MLP head. |

**Sizing notes.** Defaults follow Huang et al. 2020 (arXiv:2012.06678). Only
categorical features go through the transformer; numerics bypass it into the
`(4×, 2×)` MLP head, whose width scales automatically with the token count.
With few categorical columns the transformer part is nearly idle — prefer
`ft_transformer` there.

### `danet` — Deep Abstract Network

| Parameter | Default | Meaning |
|---|---|---|
| `n_layers` | `8` | Abstract Layers (depth); must be even and ≥ 2 (each block holds two on the main path). |
| `k` | `5` | Sparse feature-group masks per Abstract Layer. |
| `base_outdim` | `64` | Per-block output width. |
| `virtual_batch_size` | `256` | Ghost BatchNorm chunk size; keep it ≤ the effective `batch_size`. |
| `dropout` | `0.1` | Dropout on the block shortcut and before the head. |

**Sizing notes.** `n_layers` counts Abstract Layers as in the paper (Chen et
al. AAAI 2022, arXiv:2112.02962): the published DANet-20/24/32 variants are
`n_layers=20/24/32`; the default 8 is a deliberately lighter net. Capacity
also grows with `k` (more feature groups) and `base_outdim` (wider groups).

### `tabr` — TabR (retrieval-augmented)

| Parameter | Default | Meaning |
|---|---|---|
| `d_main` | `96` | Main width of encoder, retrieval module, and predictor. |
| `d_multiplier` | `2.0` | Hidden dim of the internal blocks = `d_main × d_multiplier`. |
| `encoder_n_blocks` | `0` | Residual blocks before retrieval. |
| `predictor_n_blocks` | `1` | Residual blocks after retrieval. |
| `context_size` | `96` | Nearest training rows retrieved per prediction. |
| `dropout0` | `0.1` | Dropout inside blocks and the value transform. |
| `dropout1` | `0.0` | Dropout on block outputs. |
| `context_dropout` | `0.2` | Dropout on the retrieved-context attention weights. |
| `candidate_chunk_size` | `8192` | Streaming chunk for the candidate search — a pure memory knob (peak memory is batch × chunk instead of batch × corpus). |

**Sizing notes.** Defaults are the reference configuration of Gorishniy et
al. 2023 (arXiv:2307.14338). `context_size` is the retrieval knob; `d_main`
the width knob. Training cost grows superlinearly with the corpus — on large
data bound it with the estimator's `candidate_budget`. No multi-output
regression.

### `modernnca` — ModernNCA (retrieval-augmented)

| Parameter | Default | Meaning |
|---|---|---|
| `dim` | `128` | Width of the learned metric space (encoder output). |
| `d_block` | `512` | Hidden width of the optional encoder MLP blocks. |
| `n_blocks` | `0` | Encoder MLP blocks; `0` = a plain linear encoder (the paper's default). |
| `dropout` | `0.1` | Dropout inside the encoder blocks. |
| `temperature` | `1.0` | Scales the negative distances before the softmax over neighbors. |
| `sample_rate` | `0.5` | Fraction of non-batch training rows used as candidates each step (in `(0, 1]`). |
| `candidate_chunk_size` | `8192` | Streaming chunk for inference over the whole corpus (memory knob, as in `tabr`). |

**Sizing notes.** Defaults follow Ye et al. 2024 (arXiv:2407.03257). The
paper's strongest configuration adds PLR-lite numeric embeddings
(`num_embedding="plr-lite"`) and trains with lr 0.01, weight decay 2e-4.
`sample_rate` trades step cost against gradient quality; inference always
uses the full corpus (bound it with `candidate_budget`).

### `gandalf` — GANDALF

| Parameter | Default | Meaning |
|---|---|---|
| `n_stages` | `6` | GFLU stages (depth) — the main capacity knob. |
| `mask_function` | `"t_softmax"` | Sparse mask: `"t_softmax"`, `"entmax15"`, or `"sparsemax"`. |
| `feature_sparsity` | `0.3` | Initial fraction of near-zero mask mass (t-softmax only). |
| `learnable_sparsity` | `True` | Whether the t-softmax temperature is trained (t-softmax only). |
| `dropout` | `0.0` | Dropout on the GFLU hidden state. |
| `input_batch_norm` | `False` | BatchNorm on the embedded input, for strict parity with the reference (masaMLP's preprocessing already scales inputs). |

**Sizing notes.** Defaults follow Joseph & Raj 2022 (arXiv:2207.08548).
There is no width knob: the GFLU operates in the embedded feature space, so
capacity scales with `n_stages` and the embedding width (e.g.
`d_num_embedding` via a PLR `num_embedding`). Exposes
`feature_importances()` after fitting.

### `grn` — Gated Residual Network stack

| Parameter | Default | Meaning |
|---|---|---|
| `d` | `128` | Main width (block input/output dim). |
| `d_hidden` | `128` | Hidden width of the ELU branch inside each block. |
| `n_blocks` | `2` | GRN blocks (depth). |
| `dropout` | `0.1` | Dropout before each block's GLU gate. |

**Sizing notes.** The GRN block is from the Temporal Fusion Transformer (Lim
et al. 2021, arXiv:1912.09363); the standalone stack is masaMLP's own
composition with no published tuning space. Its gates can suppress whole
blocks, so it tolerates extra depth; `d` and `n_blocks` are the knobs.

### `lnn` — TabularLNN (experimental)

| Parameter | Default | Meaning |
|---|---|---|
| `d_hidden` | `128` | Width of the recurrent cell state. |
| `n_steps` | `6` | Virtual time steps the CfC cell is unrolled (depth-like; parameter count stays constant, compute grows linearly). |
| `d_backbone` | `128` | Width of the cell's shared backbone layer. |
| `dropout` | `0.1` | Dropout inside the cell backbone. |

**Sizing notes.** An experimental in-house adaptation of the CfC cell
(Hasani et al. 2022) to static tabular data — there is no published tuning
space; see [lnn.md](lnn.md) for what is and isn't established. Because the
cell is shared across steps, `n_steps` adds depth without adding parameters.

## Presets

`masamlp.realmlp_params(task)` and `masamlp.realmlp_td_params(task)` return
plain kwargs dicts for the RealMLP-TD-S / RealMLP-TD recipes — spread and
override them freely:

```python
from masamlp import MasaClassifier, realmlp_params

clf = MasaClassifier(**{**realmlp_params("classification"), "n_epochs": 128})
```
