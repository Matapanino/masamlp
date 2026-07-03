# Changelog

## 0.3.0 (2026-07-03)

Speed release: retrieval-model inference cost, per-model AMP policy, and
multi-GPU ensemble training.

- **Multi-GPU ensemble-member sharding.** With `device="auto"` (or the
  index-less `"cuda"`), more than one CUDA device, and `n_ens > 1`, ensemble
  members are distributed round-robin across all GPUs and trained
  concurrently (one worker thread per device). Prediction after a sharded
  fit runs on the members' resident devices. Opt out with an explicit
  device (`device="cuda:0"`). Loop-mode only; `ens_mode="vectorized"`
  stays single-device. Reproducible for a fixed GPU topology
  (`docs/devices.md`).
- **TabR inference-time key caching + chunked retrieval (KI-008).** In eval
  mode candidate keys are computed once per predict pass and reused across
  query batches, and the top-k search is streamed over
  `candidate_chunk_size` blocks — peak memory B x chunk instead of the
  B x N distance matrix (~11 GB at 345k rows before).
- **ModernNCA chunked eval scoring.** The soft-nearest-neighbor aggregation
  streams over the corpus with a numerically stable running softmax and a
  cached encoded corpus, fixing the 8.4 GiB eval OOM at S6E7 scale
  (`candidate_chunk_size` constructor kwarg, default 8192).
- **Per-model `amp="auto"` policy (KI-010).** Models can qualify the auto
  policy via a class attribute: `amp_auto = False` opts out entirely — the
  retrieval models do (autocast made TabR ~2x slower on T4 and fp16
  distances lose accuracy) — and `amp_auto = "bf16"` accepts bf16 but not
  fp16, as `ft_transformer` does (fp16 measured slower and less accurate on
  T4: fit 19.4s -> 24.7s, rmse 0.296 -> 0.345 at 30k rows; bf16 GPUs keep
  AMP). Explicit `amp=True` still forces AMP.
- **`eval_batch_size`** is now an estimator parameter (was a fixed 8192),
  used by both the per-epoch eval loop and `predict`.
- Early-stopping snapshots no longer CPU-copy static retrieval corpus
  buffers on every improvement (hundreds of MB at scale).

## 0.2.0 (2026-07-03)

Field-report follow-up from running the 0.1.0 model zoo in production on a
large, imbalanced Kaggle task (`docs/s6e7-field-report.md`).

- **DANet stability fix.** `danet` could diverge to a non-finite training
  trajectory and crash inside `entmax15` (`gather(-1)` IndexError / CUDA
  device-side assert) at real-data scale. Root cause: feeding DANet's raw
  `mask_weight` parameter through entmax's `sqrt`, whose gradient is infinite
  at the support boundary and poisoned the parameter to NaN. `entmax15` now
  uses a gradient-bounded `sqrt` (no more NaN genesis) and clamps `k_star >= 1`
  (a non-finite input degrades to a clean non-finite-loss error instead of a
  hard crash). Also hardens `gandalf`, which shares `entmax15`.
- **`ema_decay`** — exponential moving average (Polyak averaging) of the model
  parameters on both estimators. When set (e.g. `0.999`), per-epoch
  evaluation, early-stopping best-epoch selection, and the final fitted
  weights all use the EMA copy. Not supported with `ens_mode="vectorized"`.
- **`candidate_budget`** — bounds the retrieval corpus of `tabr`/`modernnca`
  (and the aligned training rows, keeping per-row self-exclusion valid) with a
  seeded, class-stratified subsample. Fixes `modernnca` OOM and `tabr`
  superlinear scaling on large data; a no-op for non-retrieval models.
- **Vectorized-ensemble guardrail.** `ens_mode="vectorized"` with a BatchNorm
  or retrieval model now raises a model-named error *before* training starts
  (previously only deep in the fit), and the limitation is documented on the
  estimator.
- Docs: note that early stopping should monitor a probability-quality metric
  (`logloss`/`multi_logloss`) rather than a discrete task metric on imbalanced
  data (`docs/known_issues.md`).

## 0.1.0 (2026-07-02)

Initial release.

- `MasaRegressor` / `MasaClassifier` sklearn-compatible estimators with
  `fit(X, y, sample_weight=..., eval_set=...)`, early stopping on any metric,
  and directory-format save/load.
- Models: `resnet` and `ft_transformer` (Gorishniy et al. 2021), `realmlp`
  (Holzmüller et al. 2024, TD-S architecture with the full training recipe
  in `masamlp.realmlp_params`), `tab_transformer` (Huang et al. 2020),
  `danet` (Chen et al. AAAI 2022), `tabr` (retrieval-augmented, Gorishniy
  et al. 2023), `modernnca` (Ye et al. 2024, soft-nearest-neighbor),
  `gandalf` (Joseph & Raj 2022, GFLU with t-softmax feature masks),
  `grn` (stacked TFT Gated Residual Networks), and `lnn` (experimental
  CfC-based liquid network for static tabular data), plus a
  `register_model` hook for custom architectures (token-based models via
  `embedding_kind = "tokens"`).
- `n_ens` seed ensembling on both estimators (pytabkit semantics: members
  seeded `random_state + i`, predictions averaged on the transformed scale);
  save/load stores all members. `ens_mode="vectorized"` trains all members
  in one vmapped forward/backward (torch.func) for BatchNorm-free models,
  with per-member best-epoch tracking.
- Full RealMLP-TD recipe via `masamlp.realmlp_td_params(task)`: parametric
  activations (`act_lr_factor`), flat_cos-scheduled dropout and weight decay
  (`weight_decay_schedule`, zero decay on biases), PBLD embedding lr factor,
  and `cat_encoding="hybrid"` (one-hot up to 9 categories, embeddings of
  size 8 above).
- RealMLP insights as composable estimator options: `numeric_scaler="rssc"`,
  `cat_encoding="onehot"`, numeric embedding zoo
  (`num_embedding="pbld"/"plr"/"pl"/"periodic"`), learnable input scaling
  (`num_scaling`), `lr_scheduler="coslog4"` with per-group learning-rate
  factors, `optimizer_betas`, and regression `clip_predictions`.
- Objective plugin system: per-sample torch losses with a uniform
  sample-weight contract; built-ins for regression (squared error, MAE,
  Huber, quantile, Poisson) and classification (binary logistic, multiclass
  softmax, both with label smoothing).
- Metric plugin system ported from repleafgbm (`get_metric` / `make_metric`).
- Built-in preprocessing: quantile/standard/robust numeric scaling, median
  imputation, categorical index encoding with embeddings.
- Device support: CPU, CUDA (bf16 AMP, optional `torch.compile`), and MPS,
  behind `device="auto"`. Verified on Colab T4 (docs/verdicts/).
- DANet made GPU-practical (KI-009): the grouped 1x1 conv is computed as a
  batched einsum over the same parameters and GhostBatchNorm's training
  path is fused — 50x on T4, 14x on CPU, bit-for-bit state_dict compatible.
