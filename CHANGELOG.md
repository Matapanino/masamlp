# Changelog

## 0.8.0 (2026-07-20)

- **`ft_transformer` inner-`k` ensembling** — TabM-style inner ensembling
  (ADR 0005 §4) on the FT-Transformer: a per-member multiplicative token
  adapter `(k, 1, d)` after `TokenEmbedding`, the members folded into the
  batch dim through the *shared* attention (a batched forward, not a
  `torch.func.vmap` over `k` modules — so it works where
  `ens_mode="vectorized"` cannot, on attention archs), then a per-member
  `EnsembleHead`. `k=1` (the default) builds the exact legacy module tree, so
  0.5.0+ FT-Transformer checkpoints keep loading and every existing result is
  byte-unchanged; `k>1` returns the `(n, k, out)` inner-ensemble contract
  (0.6.0) and composes with the outer seed ensemble `n_ens`. Compute is
  `~k×` (attention included); docs recommend `k=4–8`. Pairs with
  `num_embedding="plr-lite"` (the paper's TabM†). Measurement-gated per
  ADR 0005 §7: on a large imbalanced tabular task (690k rows, 7-fold,
  prior-free balanced logloss) inner-`k=4` improved a single FT-Transformer
  by **−0.00038 nat (6/7 folds)** — a real gain, though an independent seed
  ensemble (`n_ens`) remained the stronger variance reducer there.

## 0.7.0 (2026-07-16)

- **`predict_members` / `predict_proba_members`** — per-member predictions
  for ensembles (the API pre-decided in ADR 0005 §6, on the existing
  `transform_members` hook; the averaged `predict` path is unchanged).
  `MasaRegressor.predict_members(X)` returns `(n, m)` (`(n, m, n_targets)`
  for multi-output) and `MasaClassifier.predict_proba_members(X)` returns
  `(n, m, n_classes)`, where `m = n_ens · k` counts outer seed-ensemble
  members × weight-shared inner members (`tabm`; `k = 1` elsewhere),
  ordered outer-major. Members are on the prediction scale — probabilities
  for classification, the original target scale for regression — and the
  mean over the member axis reproduces `predict` / `predict_proba` (with
  `clip_predictions=True` each member is clipped individually, so the
  equality holds where clipping does not bind). Works after
  `save_model`/`load_model`, in every `ens_mode`, and with non-identity
  prediction transforms (e.g. Poisson's `exp`).

## 0.6.0 (2026-07-16)

TabM, and inner ensembling as a model contract (design: ADR 0005).

- **New architecture: `tabm`** — TabM (Gorishniy et al. 2024,
  arXiv:2410.24210): a parameter-efficient deep ensemble where `k` members
  share one MLP backbone and diverge via per-member embedding adapters and
  output heads (the TabM-mini structure; naive per-layer BatchEnsemble
  measured worse than a single model on synthetic and was rejected).
  Defaults follow the paper (`k=32`, `d=512`, `n_blocks=3`); pairs with
  `num_embedding="plr-lite"` (the paper's TabM†).
- **The `(n, k, out)` inner-ensemble contract.** Any model — including
  third-party ones registered via `register_model` — may return per-member
  raw outputs; the trainer flattens members into rows for the loss
  (`core.trainer.weighted_loss`) and `apply_transform` averages members on
  the prediction scale (probability averaging). Consequences: every
  objective — binary, multiclass, regression, Poisson, quantile, **and
  customs** — trains inner ensembles unchanged (objectives never see a
  member dim); `sample_weight` semantics stay exact (weight 3 ≡ the row
  duplicated 3×, tested); prediction after `load_model` still needs only
  the stored transform name. Multiclass TabM predictions are unchanged by
  the refactor (member-wise softmax then mean ≡ the previous in-model
  logsumexp averaging).
- `n_ens` composes with the inner axis — `n_ens=m` × `k` gives `m·k`
  members with a two-stage mean, in every `ens_mode` including
  `"vectorized"`. Inner-`k` early stopping monitors the ensemble-average
  metric: per-member best-epoch restoration is ill-defined on shared
  weights (a documented asymmetry with `n_ens`).

## 0.5.0 (2026-07-13)

TPU optimization round 2 (follow-ups from the 0.4.0 verification; design:
ADR 0003/0004, measurements: docs/verdicts/).

- **`xla_fuse_steps`** — fuse K optimizer steps into one XLA program
  (default 1 = the 0.4.0 one-barrier-per-step behavior). Measured verdict
  (TPU v5e): the default stays 1 — fusing buys ~20% steady-state per-step
  overhead but XLA compile time grows super-linearly with the fused graph
  and dominates at realistic fit lengths; shipped as a documented escape
  hatch for very long fits. `torch_xla.experimental.scan` (the in-graph
  loop that would amortize compilation) is blocked on torch_xla 2.8 —
  `torch.func.grad` fails inside the scan body. Deterministic per K; a
  different K gives training-time device RNG (dropout, retrieval sampling)
  a different — equally random — stream, like changing `batch_size` does.
  RNG-free training is K-invariant. No effect on non-XLA devices.
- **`amp_predict`** — opt-in bf16 autocast for evaluation and prediction
  (training `amp` has never covered them). XLA/TPU, bf16-capable CUDA, and
  CPU; fp16 never. Retrieval models key their eval-encoding cache by the
  ambient autocast dtype so alternating fp32/bf16 predicts stay correct,
  and ModernNCA's streamed eval softmax now accumulates in fp32 (a no-op
  for the default fp32 path). Verified on TPU v5e: rmse-equivalent on all
  six benchmark models; speed-neutral (fp32 prediction is already fast) —
  a memory/marginal knob, not a speedup.
- **TabR TPU eval search: cross-chunk fusion restored on large corpora.**
  Models now declare `xla_eval_sync_chunks`: TabR fuses 8 eval chunks per
  XLA graph barrier when its corpus holds ≥100k candidates — measured on
  TPU v5e at 345k rows: predict 86.1s → **48.4s (−44%)**, identical
  predictions, no OOM (recovering the fusion the 0.4.0 per-chunk barrier
  — added for ModernNCA's HBM safety — had cost it). Small corpora keep
  per-chunk barriers (the fused mega-graph measured slower there);
  ModernNCA is pinned at 1.
- **ADR 0004:** no in-library TPU multi-device path (xmp.spawn rejected);
  the `TPU_VISIBLE_CHIPS` one-process-per-chip recipe remains the
  full-board story until TorchTPU is public.
- Benchmarks: step-fusion sweep/parity modes, bf16-predict matrix,
  tab_transformer TPU profile, a torch_xla `scan` step-loop prototype, and
  a masamlp-free openxla-backend inaccuracy repro for the upstream report.
- **Docs:** Colab TPU (v5e-1) verified end to end — the runtime preinstalls
  torch/torch_xla, so `pip install masamlp` suffices there. devices.md
  documents that torch_xla 2.9 dropped TPU fp32 matmuls to one-pass bf16 by
  default (0.4.0's bitwise TPU↔CPU prediction parity was a 2.8 artifact)
  and the `torch_xla.backends.set_mat_mul_precision` recipe — which must
  run before the first fit in the process — plus a TPU batch/lr tuning
  note.

## 0.4.0 (2026-07-11)

- **TPU / XLA support (experimental).** `device="tpu"` / `"xla"` /
  `"xla:N"` via `torch_xla` (lazily imported; not a dependency —
  torch↔torch_xla versions are strictly paired, see docs/devices.md).
  `device="auto"` prefers a detected TPU. bf16 autocast under
  `amp="auto"`; training runs in lazy-tensor mode with one graph barrier
  per optimizer step; all ten models train, predict, and round-trip
  through `save_model`/`load_model` (bitwise CPU-load parity measured on
  a Kaggle TPU VM v5e-8). `ens_mode="vectorized"` and `compile=True` are
  refused on XLA (the openxla backend trained inaccurately in
  verification). CI runs the XLA suite on the PJRT CPU backend. Design:
  ADR 0002/0003; measurements: docs/verdicts/2026-07-10-tpu-report.md.
- **Per-model AMP policies are now device-aware.** `amp_auto` may map
  device types to policies; the retrieval models' KI-010 fp32 opt-out is
  CUDA-only — on TPUs bf16 trained them moderately faster (ModernNCA −28%,
  TabR@345k −9%) at equivalent rmse. Prediction is amp-independent.
- **Cross-version note:** the tabr exclusion mask and ModernNCA candidate
  sampling were rewritten to static-shape forms (required for XLA, also
  removes a host sync on CUDA), and `save_model` now normalizes state
  dicts to CPU tensors. Same-seed results may differ at ulp level from
  0.3.x; ModernNCA's candidate-sampling RNG stream changed.
- **Parameter reference (`docs/parameters.md`).** Complete documentation of
  every estimator constructor parameter and every architecture's
  `model_params` (depth/width/dropout knobs, defaults, sizing notes with
  sources), plus the shared embedding keys. The README gains a
  "Key parameters" summary table. Kept in sync with the code by
  `tests/test_docs_parameters.py` (signature inspection).
- **Friendlier `model_params` errors.** Unknown keys now raise a
  `ValueError` listing the model's valid keys and the shared embedding keys
  (previously a bare `TypeError` from the constructor). Builders accepting
  `**kwargs` are exempt.

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
