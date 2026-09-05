# Parameter reference

Every knob in masaMLP lives in one of three places:

- **Estimator parameters** — constructor arguments of `MasaRegressor` /
  `MasaClassifier` (training loop, preprocessing, hardware). Listed first.
- **`model_params`** — a dict forwarded to the selected architecture's
  constructor. Depth, width, dropout, and every other architecture knob is
  free per model; the per-architecture tables below are the complete list.
- **Shared embedding keys** — seven keys accepted *inside* `model_params` for
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

### Target and fit-data contract

`MasaClassifier.fit(X, y, sample_weight=None, eval_set=None)` accepts the
existing 1-D hard-label target for binary or multiclass classification. A
floating 1-D `y` containing fractional values is instead a **soft binary
target**: every value must be finite and in `[0, 1]`, `classes_` is `[0, 1]`,
and the default objective is BCE-with-logits against those probabilities.
Hard `{0, 1}` targets keep the pre-0.10.0 label-encoding and training path
byte-for-byte. Multiclass probability matrices are not supported, and
`class_weight` is not defined for soft targets; use `sample_weight` for
per-row weighting.

Every `eval_set` remains an unweighted `(X, y)` pair. In particular, a soft
binary training target still requires **hard 0/1 validation labels**. Thus
the configured validation metric (`auc`, `logloss`, and so on), best-epoch
selection, and early stopping have exactly their hard-label semantics.

`sample_weight` is a finite, non-negative 1-D vector aligned with the
training rows. It multiplies each per-row loss and the trainer normalizes by
the batch's sum of weights. Weights stay aligned through shuffled batches and
inner/outer ensembles. Any positive constant vector is canonicalized to the
unweighted path, giving byte-identical seeded fits rather than merely
scale-equivalent gradients. A zero weight means that row contributes no
training loss. If every row in a minibatch has zero weight, that minibatch is
skipped; the full vector must still contain at least one positive weight.

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
| `eval_batch_size` | `8192` | Forward-pass batch for evaluation and `predict`; a pure memory knob with no effect on results. Lower it on models whose per-call memory scales with a model-internal axis (e.g. `ft_transformer`'s inner `k`, which folds members into the batch dim — see its sizing notes). |
| `learning_rate` | `1e-3` | Optimizer learning rate. Models may scale it per parameter group (RealMLP does). |
| `weight_decay` | `0.0` | Decoupled weight decay (AdamW-style). |
| `optimizer` | `"adamw"` | `"adamw"`, `"adam"`, or `"sgd"`. |
| `optimizer_betas` | `None` | `(beta1, beta2)` for adam/adamw (`None` = torch defaults). The RealMLP recipe uses `(0.9, 0.95)`. |
| `lr_scheduler` | `"none"` | `"none"`, `"cosine"` (per-epoch cosine annealing), or `"coslog4"` (RealMLP's per-step schedule). |
| `weight_decay_schedule` | `"none"` | `"none"` or `"flat_cos"` (RealMLP-TD: constant for the first half of training, cosine decay after; param groups can opt out, e.g. biases). |
| `weight_decay_mode` | `"auto"` | How weight decay enters the update. `"auto"` keeps each optimizer's own convention (`adamw` decoupled, `adam`/`sgd` coupled) — the pre-0.9.0 behaviour. `"decoupled"` is AdamW's `p *= 1 - lr·wd`; `"coupled"` adds `wd·p` to the gradient so it passes through Adam's second moment. pytabkit's RealMLP-TD is **decoupled**. |
| `label_smoothing_schedule` | `"none"` | Per-step schedule on the objective's `label_smoothing` (pytabkit's `ls_eps_sched`): `"none"`, `"coslog4"` (RealMLP-TD — 0 at both ends of training, four cosine cycles between) or `"flat_cos"`. Needs an objective with a `label_smoothing` attribute. |
| `drop_last` | `False` | Drop each epoch's final short batch, so every optimizer step sees exactly `batch_size` rows (pytabkit's dataloader default). Also shortens the per-step schedule denominator to `floor(n/batch_size)` steps per epoch. |
| `share_training_batches` | `True` | Inner ensembles (`tabm`, `realm`): `True` broadcasts the same shuffled row batch to every member (the historical code default); `False` gives every member an independently shuffled row stream, as in the TabM paper's headline experiments. Labels and `sample_weight` follow the member-indexed rows. |
| `grad_clip` | `None` | Global gradient-norm clip; `None` disables. |
| `ema_decay` | `None` | Exponential moving average (Polyak averaging) of the weights, e.g. `0.999`; evaluation, early stopping, and the final model use the averaged parameters. Not supported with `ens_mode="vectorized"`. |

### Preprocessing and numeric embeddings

| Parameter | Default | Meaning |
|---|---|---|
| `numeric_scaler` | `"quantile"` | `"quantile"` (rank → normal), `"standard"`, `"robust"`, `"rssc"` (RealMLP's robust-scale-smooth-clip), or `"none"`. Fitted on the training set only. |
| `categorical_features` | `"auto"` | `"auto"` detects categorical columns from DataFrame dtypes; or pass a list of column names/indices. |
| `cat_encoding` | `"embedding"` | `"embedding"` (per-column `nn.Embedding`, index 0 reserved for unknown/missing), `"onehot"` (RealMLP-style: binary → ±1, missing → 0), or `"hybrid"` (one-hot up to 9 categories, embeddings above — RealMLP-TD). |
| `onehot_max_categories` | `9` | Cardinality threshold for `cat_encoding="hybrid"`: columns with at most this many observed categories are one-hot encoded (and then scaled with the numerics), the rest get embeddings. pytabkit's `max_one_hot_cat_size` counts its reserved missing slot as well, so its 18 corresponds to `17` here. |
| `num_embedding` | `None` | Numeric-feature embedding: `None` (raw), `"ple"` (quantile-bin PLE-vB from the TabM paper), `"periodic"`, or the Fourier/periodic PLR family `"pl"` / `"plr"` / `"plr-lite"` / `"pbld"` (arXiv:2203.05556 + PBLD per pytabkit). Token models accept the PLR family, but not `"ple"`/`"periodic"`, as their tokenizer. Tune the embedding via the [shared embedding keys](#shared-embedding-keys-inside-model_params). |
| `num_embedding_cols` | `None` | Input routing: apply `num_embedding` to **these numeric columns only** (a list of column names). Every other numeric column bypasses the embedding and enters the first layer linearly, still under `num_scaling`. `None` embeds all of them. Requires `num_embedding`. On `ft_transformer` (0.9.4) the unlisted columns become plain linear tokens instead — every numeric column still gets exactly one token — since there is no flat first layer to bypass into; still rejected on `tab_transformer`, whose numerics never become tokens at all. |
| `linear_skip_cols` | `None` | Input routing: add a linear map from these (scaled) numeric columns straight onto the output — `raw = trunk(x) + x_skip @ W_skip + b_skip`. `W_skip`/`b_skip` are zero-initialized, so the fit starts exactly where it would without the skip, and they train in their own param group with **zero weight decay**. `model="realmlp"` only. |
| `linear_skip_lr_factor` | `1.0` | Learning-rate factor for the `linear_skip_cols` parameters. `0.0` freezes them at zero (an exact no-op control). |

**Input routing.** Both options take column **names**, resolved at `fit`
against the numeric block the model sees: the non-categorical columns in
their original order, followed by the one-hot block `cat_encoding` produces
(so a one-hot column never shifts a named column's position, and is not
itself addressable). Naming a categorical or unknown column raises. A plain
ndarray works too — pandas' positional names `"0"`, `"1"`, … Note that with
`cat_encoding="onehot"`/`"hybrid"` the one-hot columns *are* part of the
numeric block, so naming only the real numerics in `num_embedding_cols`
deliberately keeps the one-hots out of the embedding. Positions can be given
directly instead via `num_embedding_idx` / `linear_skip_idx` in
`model_params`. Typical use: a target-encoded log-odds column is a monotone
score whose linear use is the point, while a raw measurement benefits from
the periodic embedding's flexible 1-D response.

`ft_transformer`'s tokenizer accepts `num_embedding_idx` on the same terms —
listed columns keep going through `num_embedding` as PLR tokens, the rest
become linear tokens (`x_j * w_j + b_j`, the FT-Transformer paper's own
numeric tokenizer) — so every numeric column still becomes exactly one
token, just via a different tokenizer. `tab_transformer` still rejects it:
its numerics bypass the transformer as a flat vector and never become
tokens, so there is nothing to route between.

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

Member-level predictions are exposed on both axes:
`MasaRegressor.predict_members(X)` returns `(n, n_ens · k)`
(`(n, n_ens · k, n_targets)` for multi-output) and
`MasaClassifier.predict_proba_members(X)` returns `(n, n_ens · k,
n_classes)`, ordered outer-major (the `k` inner members of `models_[0]`
first), on the prediction scale — probabilities for classification, the
original target scale for regression. The mean over the member axis
reproduces `predict` / `predict_proba`; the per-member spread is the usual
deep-ensemble uncertainty signal. With `clip_predictions=True` members are
clipped individually, so the mean-equality holds only where clipping does
not bind.

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
| `class_weight` | `None` | `"balanced"` or a `{label: weight}` dict for hard-label classification; multiplied into `sample_weight` before the shared weighted reduction, so it composes with custom objectives. Not supported for soft binary targets. |
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
| `d_num_embedding` | `16` | Output dim per numeric feature for `"ple"` and the PLR-family embeddings. This is the shared embedding-width key used by TabM†. Token models ignore it in favor of the token width. |
| `n_frequencies` | `16` | Number of random frequencies for `"periodic"`/PLR embeddings. |
| `n_bins` | `48` | Quantile-bin count for `num_embedding="ple"`. Boundaries are fitted on training rows only at `linspace(0, 1, n_bins + 1)`, duplicate quantiles are removed, and the fitted boundaries are serialized with the estimator. |
| `sigma` | `0.1` | Scale of the initial frequencies (`c ~ N(0, sigma²)`); the most sensitive periodic-embedding knob in arXiv:2203.05556. |
| `cat_emb_dim` | `None` | Per-column categorical embedding dim; `None` = auto (`min(32, max(2, round(1.6 · cardinality^0.56)))`). Token models ignore it (tokens are `d_token`/`d_block` wide). |
| `num_scaling` | `False` | Learnable per-feature scale on numeric inputs before embedding (RealMLP's scaling layer, trained at 6× lr there). Defaults to `True` when `model="realmlp"`. |
| `num_embedding_idx` | `None` | Positional form of the estimator's `num_embedding_cols`: a list of positions in the numeric block the embedding applies to; the rest bypass it. Output layout becomes `[embedded | bypassing | categorical]`, each block in ascending column order. Rejected for `tab_transformer` (numerics never become tokens there); accepted on `ft_transformer` (0.9.4), where the rest become linear tokens instead of bypassing — see [Input routing](#preprocessing-and-numeric-embeddings). |

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
| `variant` | `"mini"` | `"mini"` preserves masaMLP 0.9.2 byte for byte: one input adapter, shared plain MLP, independent heads. `"full"` is headline TabM: every backbone linear has shared `W` plus member-specific `r`, `s`, and bias, with an independent head. |
| `k` | `32` | Inner-ensemble members, vectorized on a static `(n, k, d)` axis. |
| `d` | `512` | Backbone width. |
| `n_blocks` | automatic | Plain `Linear → ReLU → Dropout` depth. Mini stays at 3. Full defaults to 3 with raw numerics and 2 when a numeric embedding is configured, matching `TabM.make`. |
| `dropout` | `0.1` | Dropout after every backbone activation. |
| `adapter_std` | `0.5` | Mini-only std of its legacy input adapter `N(1, adapter_std)`. Full uses the official initialization: chunk-shared first `r` is random ±1 for raw input or `N(0,1)` with a numeric embedding; every later `r` and every `s` starts at one. |
| `first_layer_groups` | `None` | Full-only integer group ID per `embedding.feature_chunk_sizes` entry; `-1` is shared across groups. Hidden units cycle through sorted nonnegative groups, with all other first-layer connections fixed to zero. Initial active weights are rescaled for their effective fan-in. |
| `member_feature_mask` | `None` | Full-only `k` lists of booleans, one per embedding feature chunk. False removes that entire chunk from the member in training and prediction. Each member must retain at least one chunk. Fixed masks are saved with the model. |
| `brier_coefficient` | `0.0` | Full-only nonnegative auxiliary squared probability error for **binary classification**, added to the primary objective through trainer-weighted per-row terms. Requires one logit; use only with binary targets in `[0,1]`. Soft targets and independent member weights are supported. Zero creates no term tensors. |

These three options are experimental departures from the paper defaults. They
retain the full model's shared backbone weights, member adapters, independent
heads, and probability-mean prediction. Default values preserve initialization
and the original state-dict layout. Chunk indices describe the **embedding
output**, not input dataframe order: selectively embedded numeric chunks come
first, then bypass numeric/one-hot chunks, then embedded categorical chunks.
Use the fitted preprocessor and embedding routing to derive this ordering.
Connectivity restricts only the first layer; later dense layers can mix groups.
Member masks do not add inverse-retention scaling.
The Brier option follows the training-terms interface's existing execution
limits (outer vectorized ensembling rejects enabled hooks). With coefficient
zero, no hook is attached and the original outer ensembling support is intact.

Full layers initialize shared weights and initially equal member biases from
`U(-1/sqrt(fan_in), 1/sqrt(fan_in))`; the independent head initializes both
member weights and biases from the same fan-in bounds. PLE-vB uses the fixed
`[1, …, 1, fraction-in-current-bin, 0, …, 0]` encoding, a zero-initialized
per-feature PLE projection, and a linear residual embedding with no
activation. Thus `model="tabm"`, `model_params={"variant": "full"}` plus
`num_embedding="ple"` is TabM†; `plr-lite` is a Fourier/periodic embedding
and is not TabM†.

For a faithful paper-scale run, set the training protocol explicitly:

| Aspect | Paper/official value | masaMLP setting |
|---|---:|---|
| Variant / members | full / 32 | `model_params={"variant": "full", "k": 32}` |
| Width / depth / dropout | 512 / raw 3 or PLE 2 / 0.1 | `d=512`; automatic `n_blocks`; `dropout=0.1` |
| PLE | quantile PLE-vB, 48 bins, width 16 | `num_embedding="ple"`; `n_bins=48`; `d_num_embedding=16` |
| Optimizer | AdamW, lr 0.002, wd 0.0003 | `optimizer="adamw"`; `learning_rate=0.002`; `weight_decay=0.0003` |
| Schedule / stop / clip | none / patience 16 / global 1.0 | `lr_scheduler="none"`; `early_stopping_rounds=16`; `grad_clip=1.0` |
| Batch protocol | 1024 at this scale; independent members | `batch_size=1024`; `share_training_batches=False` |

Training remains the mean of per-member losses; classification prediction
is the mean of per-member probabilities, and regression prediction is the
mean value. Early stopping monitors the one joint ensemble-average metric.
The inner axis composes with outer `n_ens` in loop or vectorized mode.

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
| `init_mode` | `"ntp"` | `"ntp"` — weights and biases both `N(0, 1)`, the standalone TD-S reference's init. `"std+he5"` — pytabkit's RealMLP-**TD** init, applied layer by layer on a batch of training rows: each weight is rescaled so its pre-activations have unit sample std, and each bias is minus a random Exp(1)-simplex combination of five sampled pre-activations of that unit. Requires the estimator (which supplies the batch); harmless on a model built directly. |
| `zero_init_output` | `True` | Zero-initialize the output layer (RealMLP-TD-S). RealMLP-TD does **not** — set `False` to init it like any other layer. |
| `scale_position` | `"input"` | Where the learnable diagonal gain sits. `"input"` scales the raw numeric features before any embedding (TD-S, and what `num_scaling` builds). `"first_layer"` moves it to the first hidden layer's input — after the numeric embedding and beside the one-hots and categorical embeddings — which is where RealMLP-TD's `add_front_scale` puts it. |
| `scale_lr_factor` | `6.0` | Learning-rate factor for the scaling layer. RealMLP-TD's tuned configurations use values around 2–10. |
| `first_layer_lr_factor` | `1.0` | Extra learning-rate factor on the first hidden layer's weight **and** bias (pytabkit applies it to both, and to neither the scaling layer nor the numeric embedding). |
| `bias_lr_factor` | `0.1` | Learning-rate factor for every NTP bias (biases also never receive weight decay). |
| `linear_skip_idx` | `None` | Positional form of the estimator's `linear_skip_cols`: positions in the numeric block that also feed a zero-initialized linear map onto the output (`raw = trunk(x) + x_skip @ W_skip + b_skip`), read *before* the scaling layer and any numeric embedding. Its parameters form their own optimizer group at `linear_skip_lr_factor` with zero weight decay. |

**Sizing notes.** `hidden_sizes` is the whole architecture: length = depth,
entries = per-layer width. The `(256, 256, 256)` default is the RealMLP-TD(-S)
architecture of Holzmüller et al. 2024 (arXiv:2407.04491). This model is
about its *training recipe* as much as its shape — pair it with
`masamlp.realmlp_params(task)` or `realmlp_td_params(task)` rather than
tuning in isolation.

### `realm` — RealM (RealMLP + full BatchEnsemble)

The `realmlp` architecture with `k` weight-shared members, TabM-style. Every
`realmlp` knob above carries over unchanged (`hidden_sizes`, `activation`,
`dropout`, `dropout_schedule`, `use_parametric_act`, `act_lr_factor`,
`plr_lr_factor`, `init_mode`, `zero_init_output`, `scale_position`,
`scale_lr_factor`, `first_layer_lr_factor`, `bias_lr_factor`) with the same
meaning; the table below is only the ensembling half. See
[realm.md](realm.md).

| Parameter | Default | Meaning |
|---|---|---|
| `k` | `8` | Ensemble members. Each layer keeps **one shared weight matrix** and gives every member a rank-1 input adapter `r`, output adapter `s`, its own bias and its own output head, so parameter cost stays near one backbone — but **compute is ~k member-forwards** and one `(n, k, d)` activation grows linearly in `k`. `k=1` is RealMLP plus an (unused) rank-1 reparametrization. Start at 8/16; `k=32` is a scale-up confirmation. |
| `member_init` | `"auto"` | Init of the **first** adapter — the one that turns the shared embedding into k different member representations. `"auto"` follows TabM's reference rule: `"normal"` (`N(1, adapter_std)`) when a numeric embedding is configured, `"signs"` (random ±1) when the numeric features enter raw. `"ones"` is the degenerate parity mode — every member starts *and stays* identical — kept as a control and for the k=1 equivalence test. |
| `adapter_std` | `0.5` | Std of the `"normal"` first-adapter init `N(1, adapter_std)` (masaMLP's `tabm` convention). |
| `adapter_lr_factor` | `1.0` | Learning-rate factor for every **per-member** parameter: the first adapter, each layer's `r`/`s`, and the head weight. The per-member biases keep RealMLP's `bias_lr_factor` and zero weight decay, with this factor applied on top — so at `1.0` every group's effective (lr, wd) factor equals RealMLP's. TabM's tuning space is {1, 2, 4}. |

**Sizing notes.** RealM's members train as independent predictors on the
**mean of their per-member losses** (never the loss of the mean prediction),
and the per-epoch metric — hence the single, **joint** best epoch — is
computed on the member-averaged prediction, which is also what
`predict_proba` returns. `linear_skip_idx` is *not* implemented here (it
stays a `realmlp` option), and `ens_mode="vectorized"` is rejected: `realm`
already vectorizes its inner ensemble, and the outer `n_ens` axis composes
with it in the default loop mode. Pair with
`masamlp.realm_td_params(task, k=...)`.

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
| `k` | `1` | TabM-style inner ensemble members (ADR 0005 §4): a per-member multiplicative adapter on the feature tokens, a shared attention backbone + `[CLS]` query, and a per-member output head; predictions averaged on the probability scale. `k=1` is the plain FT-Transformer (byte-identical module tree — pre-0.6.0 checkpoints load). Composes with the outer seed ensemble `n_ens`. |
| `adapter_std` | `0.5` | Std of the per-member adapter init `N(1, adapter_std)` (used only when `k>1`); members start near the single model and diverge during training. |

**Sizing notes.** Defaults are the reference package's
`get_default_kwargs(n_blocks=3)` (rtdl_revisiting_models, arXiv:2106.11959);
the reference scales `d_block` and the dropouts together with `n_blocks` in
its presets, so when you deepen the model, widen it too. Compute grows
quadratically with the number of features (attention over one token per
feature). Inner ensembling (`k>1`) recommends `k=4–8` here: the members fold
into the batch dim so the shared attention runs ~k× the compute (unlike the
MLP-backbone `tabm`, where `k=32` is cheap) — and ~k× the *memory* per
forward call, since every chunk `predict_transformed` runs (per-epoch eval,
`predict_proba`/`predict_members`) effectively becomes `chunk_size * k` rows
through attention at once. `eval_batch_size` is the mitigation: it is a
pure memory knob (identical predictions at any value, tests/test_ftt_inner_k.py
`test_chunked_predict_matches_unchunked_at_k8` / `test_chunked_per_epoch_eval_matches_unchunked_at_k8`)
already applied to every prediction path, so lower it when `k` is large and
the device is memory-constrained (e.g. a TPU v5e-8 chip's 15.75 GB HBM) —
confirmed as the fix for a `k=8` HBM OOM measured at the default `8192`
(s6e9/ftt `findings/FINDINGS.md` "TPU smoke", 2026-09-04).

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

`masamlp.realmlp_params(task)`, `masamlp.realmlp_td_params(task)` and
`masamlp.realm_td_params(task, k=...)` return plain kwargs dicts for the
RealMLP-TD-S / RealMLP-TD / RealM-TD recipes — spread and override them
freely:

```python
from masamlp import MasaClassifier, realmlp_params

clf = MasaClassifier(**{**realmlp_params("classification"), "n_epochs": 128})
```

### Hierarchical RealMLP (`model="hierarchical_realmlp"`, WP3 2026-09-05)

The smooth root is the existing numeric embedding (`num_embedding="pbld"`,
`"plr"`, etc.). Zero-initialized residual tables add coarse-to-fine deviations
inside the same numeric token; trunk width and initialization remain unchanged.
Shared embedding options stay at the top level of `model_params`.

- `hierarchies` (required): JSON list of `{"feature": numeric_index, "levels": [...]}`.
  Feature indices address the preprocessed numeric block **before learnable scaling**
  and must select embedded columns. Each coarse level is
  `{"boundaries": [ascending split points], "counts": [count per interval]}`;
  intervals are left-closed at a boundary (`searchsorted(..., right=True)`).
  Finer boundaries must include every coarser boundary. The final level is
  `{"values": [ascending exact values], "counts": [count per value]}`.
  Counts must be finite and nonnegative, with positive support in every level.
  Coordinates must remain distinct after float32 conversion.
- `shrinkage` (default `1.0`): nonnegative coefficient of the full-table penalty.
  `0.0` disables the training term and defines the unregularized control.
- `support_power` (default `1.0`): nonnegative exponent for precision
  `max(count, 1) ** (-support_power)` at supported entries. At zero, the prior
  precision is constant; positive values further shrink rare effects relative
  to common effects. Counts are fixed training support, never minibatch counts.
- `backbone_params` (default `None`): dictionary of the ordinary `RealMLPNet`
  constructor options, e.g. `{"hidden_sizes": [384, 384, 384], "activation": "selu"}`.
  Set shared `num_scaling=True` explicitly to match the estimator's `realmlp`
  defaults; the new registration does not change estimator defaults.

Fit all coordinates and counts within the model's fitting boundary, excluding
scoring and early-stopping rows. The caller supplies coordinates transformed by
the same fitted `TabularPreprocessor` used by the estimator (or chooses
`numeric_scaler="none"` for already-transformed inputs). Do not recompute keys
from values after the model's learnable numeric scaling. No targets are needed
to construct the hierarchy. Save/load stores constructor configuration and
registered tables, coordinates, counts, support masks and precision buffers.

An unseen exact value contributes no leaf deviation and retains any supported
parent deviations; an unsupported parent contributes zero. The smooth root is
always retained. Nonfinite keys contribute no residual (ordinary preprocessor
imputation still applies before the model sees inputs).

The objective adds `shrinkage * sum(precision * residual**2) / N`, where `N`
is the fixed number of supported residual coordinates across every level.
It is computed in FP32 through `ModelRegularizer`, once per optimizer step,
independent of batch size, sampled value frequency, sample-weight scale or
steps per epoch. Different numbers of steps can still change the optimization
trajectory. Residual tables use zero optimizer weight decay, so this penalty
is their only explicit regularization; backbone optimizer groups are preserved.
See [training terms](training-terms.md) for loop/EMA/compile support and the
explicit rejection of vectorized outer ensembles.
### Explicit fitted PLE knots (2026-09-05 WP5)

`fit(X, y, ..., ple_bins={"income": [0., 10., 25., 100.]})` accepts
raw-coordinate **edges including endpoints**, with `num_embedding="ple"`.
Use `num_embedding_cols=["income"]` to restrict the embedded block. The mapping
must cover exactly the embedded original numeric columns; categorical/one-hot
columns and duplicate input names are rejected. Arrays and lists are accepted.
Each column may have a different edge count; `model_params["n_bins"]` controls
only the default quantile path, not explicit resolution.

The fitted preprocessor converts edges through the same scaling path as inputs.
Edges must remain finite and strictly increasing after float32 conversion;
clipping/collisions fail instead of silently changing resolution. Converted
edges live in `resolved_model_params_["_ple_bins"]` and saved model buffers,
so loaded predictions need no knot builder. A subsequent `fit` must receive
`ple_bins` again; omitting it recomputes quantiles. Supervised edge selection
is the caller's responsibility and must exclude scoring and validation labels.

## Masked ordinal auxiliary model (`auxiliary_ordinal`, WP4, 2026-09-05)

The primary output is a single binary logit; ordinary `MasaClassifier.predict_proba`
and directory save/load keep their usual two-column probability contract. Set
`cat_encoding="embedding"` to keep the auxiliary target carrier categorical.
All routing indices below address the **preprocessed** arrays, not raw DataFrame
positions. The caller owns causal provenance and must exclude every target-derived
feature. Numeric preprocessing and auxiliary-label vocabularies must be fitted
inside the training boundary.

- `parent_num_idx`, `parent_cat_idx`: explicit allowlists for auxiliary predictors.
- `excluded_num_idx`, `excluded_cat_idx`: explicit complements; overlaps, duplicate
  indices, missing declarations and out-of-range indices raise. This prevents
  an unreviewed new input from silently becoming an auxiliary predictor.
- `auxiliary_target_cat_idx`: separate categorical target carrier, excluded from
  **both** prediction branches. The observed response may remain as a separate
  primary input. At inference supply a missing carrier column; its value is ignored.
- `auxiliary_class_order`: all nonmissing category codes `1..K` in ordinal order.
  Resolve against the fitted preprocessor vocabulary. Code 0 is unknown/missing
  and has zero auxiliary row loss, with the ordinary primary-weight denominator.
- `auxiliary_weight` (0.1): nonnegative multiplier of the proper cumulative-logit
  category negative log likelihood. Zero returns empty WP2 terms before auxiliary
  likelihood computation and reproduces the same architecture without the hook.
- `latent_dim` (8), `auxiliary_hidden_sizes` ((32, 32)): auxiliary RealMLP dimensions.
  The deterministic auxiliary branch uses SELU, learnable scaling and a nonzero
  NTP latent head. The primary binary head reads its latent alongside the primary
  RealMLP hidden representation; both losses update the shared latent branch.
- `backbone_params` (None): JSON-compatible RealMLP constructor options for the
  primary branch, e.g. `hidden_sizes`, dropout, initialization and scheduling.
  `linear_skip_idx` is currently unsupported and raises. Shared numeric embedding
  options configure the primary branch; the auxiliary branch embeds only parents.

The ordinal head uses increasing softplus-spaced cutpoints and FP32 stable log
probabilities. It returns one unreduced row loss via WP2, so sample weighting
remains the Trainer's responsibility. This is an opt-in research architecture;
no causal claim or competition gain is implied. Vectorized outer ensembles are
unsupported under the WP2 hook contract. Auxiliary-only pretraining can use the
ordinary Trainer with a zero primary objective, then a fresh binary Trainer over
the same module with `auxiliary_weight=0`; optimizer state is reset between stages.
