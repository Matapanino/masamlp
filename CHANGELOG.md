# Changelog

## Unreleased

- Add opt-in TabM full first-layer feature groups, fixed member feature views,
  and an auxiliary binary Brier term through the existing weighted training-term
  interface. Defaults preserve original initialization, predictions and state
  layout; enabled masks and terms support soft targets and save/load.

## 0.11.0 (2026-09-05)

- WP3 (2026-09-05): register `hierarchical_realmlp`, adding coarse-to-fine,
  zero-initialized numeric residual tables to the existing smooth RealMLP
  embeddings without widening the trunk. Unseen values back off to supported
  parents. Full-table support-dependent shrinkage uses WP2 training terms,
  independently of minibatch frequency; tables and support state survive save/load.
- Add opt-in `auxiliary_ordinal`: two RealMLP branches with explicit parent/excluded
  routing, a shared latent and stable ordinal per-row likelihood via WP2. Separate
  auxiliary target carriers support shuffled controls; binary prediction and
  save/load remain ordinary. Includes routing, zero-weight, falsifier and persistence tests.
- WP2 (2026-09-05): opt-in model `training_terms(batch, raw)` hook with
  trainer-weighted per-row auxiliary losses and separately normalized scalar
  model penalties; zero coefficients preserve the existing training path.
  Registered term state uses existing checkpoint and save/load handling.
- Correct independent TabM member weighting to the mean of separately
  normalized member risks when weight totals differ. Zero-total members
  contribute zero to the fixed-member mean; all-zero batches remain skipped.
- Reject training-term models in vectorized outer ensembles; document the
  research interface, scaling contract, and example in `docs/training-terms.md`.
- **Explicit PLE knots (2026-09-05 WP5)** — `fit(..., ple_bins={name: edges})`
  accepts raw-coordinate edges for every embedded numeric column. Names follow
  numeric routing through preprocessing, converted edges are validated after
  scaling/float32 conversion and persist in the existing saved embedding state.
  Default quantile fitting is unchanged. Supervised knot fitting remains the
  caller's training-boundary responsibility; refits require the mapping again.

## 0.10.0 (2026-09-04)

- **Soft binary targets** — `MasaClassifier.fit` accepts a floating 1-D `y`
  with values in `[0, 1]` and trains the built-in binary objective with
  BCE-with-logits against the probabilities themselves. Hard `{0, 1}` labels
  retain the 0.9.4 path byte-for-byte. Multiclass soft targets remain out of
  scope and raise a direct error.
- **Validation stays hard** — `eval_set` requires hard labels even when the
  training target is soft, so AUC/logloss evaluation, best-epoch selection,
  and early stopping keep their existing semantics.
- **Exact uniform-weight identity** — any positive constant `sample_weight`
  vector is canonicalized to the unweighted path. Non-uniform weights retain
  the normalized `(loss * w).sum() / w.sum()` reduction and stay aligned
  through minibatches and inner/outer ensembles.
- **Constant-weight regression bias fix** — canonicalizing constant weights
  also gives `MasaRegressor` MAE and quantile objectives their unweighted
  NumPy quantiles for `init_bias` (for `y=[1,2,3,4]`, median `2.5` and the
  0.9 quantile `3.7`) instead of the weighted step quantiles `2.0` / `4.0`.
- **Constant-weight compatibility note** — constant `sample_weight` now
  canonicalizes to unweighted; fits with a constant weight vector differ from
  0.9.4 at the 1e-4 level on resnet.
- **Zero-weight minibatches** — a row with weight zero contributes no training
  loss, and an all-zero-weight minibatch is skipped instead of producing a
  divide-by-zero loss.

## 0.9.4 (2026-09-04)

- **`num_embedding_cols` / `num_embedding_idx` on `ft_transformer`** —
  token-model input routing: the listed numeric columns still go through
  `num_embedding` (PLR family) as tokens; every other numeric column becomes
  a plain linear token (`x_j * w_j + b_j`, the FT-Transformer paper's own
  numeric tokenizer) instead of the previous hard `ValueError`. Every
  numeric column still becomes exactly one token either way — no linear
  bypass, since token models have no flat first layer to bypass into.
  `None` (the default) is byte-identical to before
  (`tests/test_transformers.py::test_ft_transformer_num_embedding_idx_none_matches_main_pin`,
  pinned from `main`); naming every numeric column reproduces the unrouted
  PLR tokenizer exactly
  (`test_token_embedding_full_coverage_routing_is_the_unrouted_path`).
  Composes unchanged with the inner-`k` adapter
  (`test_ft_transformer_num_embedding_idx_composes_with_inner_k`).
  `tab_transformer` still rejects the option: its numerics bypass the
  transformer as a flat vector and never become tokens, so there is nothing
  to route between (moved to
  `test_input_routing.py::test_selective_embedding_rejected_for_non_tokenizing_token_model`).
- **Documentation: `eval_batch_size` is the memory-safe mitigation for large
  `k` on `ft_transformer`.** No code change — `predict_transformed`'s
  existing chunking (driven by `eval_batch_size`, default `8192`) already
  reaches every prediction path (the per-epoch eval used by early stopping,
  `predict_proba`/`predict_members`, the vectorized-ensemble eval loop, and
  the multi-GPU sharded predict path) and is a pure memory knob — chunked
  and unchunked predictions are numerically identical to 1e-6
  (`tests/test_ftt_inner_k.py::test_chunked_predict_matches_unchunked_at_k8`
  / `test_chunked_per_epoch_eval_matches_unchunked_at_k8`, plus the XLA:CPU
  CI smoke in `tests/test_xla.py`). This is confirmed as the fix for the
  `mp_k=8` TPU v5e-8 HBM OOM measured in s6e9/ftt's TPU smoke
  (`s6e9/ftt/findings/FINDINGS.md` "TPU smoke", 2026-09-04): the per-epoch
  eval split (4,800 rows) was smaller than the default `eval_batch_size`
  (8,192), so no chunking occurred and the whole split folded `×8` into the
  attention batch dim in one call — lower `eval_batch_size` below the eval
  split size when `k` is large and the device is memory-constrained. See
  `docs/parameters.md`'s `ft_transformer` sizing notes.

## 0.9.3 (2026-09-03)

- **Faithful full TabM** — `model="tabm"` now accepts
  `model_params={"variant": "mini" | "full"}`. Mini preserves the 0.9.2
  predictions byte for byte; full puts official BatchEnsemble `r`/`s`/bias
  parameters on every plain ReLU-MLP linear and uses independent uniformly
  initialized member heads.
- **TabM† PLE-vB** — `num_embedding="ple"` fits quantile boundaries on the
  training rows, stores them with the estimator, and implements the fixed
  piecewise-linear encoding plus zero-initialized projection and linear
  residual. The shared width key is `model_params["d_num_embedding"]`; bin
  count is `model_params["n_bins"]` (default 48).
- **Independent member batches** — `share_training_batches=False` gives each
  inner-ensemble member its own shuffled row stream while keeping labels and
  sample weights aligned. The default remains `True` for compatibility.
- **Documentation correction** — `plr-lite` is a Fourier/periodic embedding,
  not the TabM paper's PLE-based TabM†.

## 0.9.2 (2026-09-03)

- **`model="realm"` — RealM: the RealMLP backbone under TabM-style *full*
  BatchEnsemble.** `k` members share one weight matrix per layer and differ
  only through rank-1 adapters (`r` on the input, `s` on the output), a
  per-member bias and a per-member output head, so diversity is *learned
  jointly* on a shared feature extractor instead of bought with `n_ens`
  independent fits. New module `masamlp/models/realm.py`
  (`BatchEnsembleLinear`, `EnsembleNTPHead`, `RealMNet`), preset
  `masamlp.realm_td_params(task, k=8)`, docs in
  [docs/realm.md](docs/realm.md).
  - Member *i*'s effective weight is `W ⊙ (r_i s_iᵀ)` but is **never
    materialized**: the forward broadcasts the adapters around one shared
    matmul. `k` is a fixed config value, every shape is static, and there is
    no `torch.vmap` and no Python loop over members — one XLA graph per
    batch shape (`tests/test_xla.py::test_realm_traces_one_static_graph_per_step_xla`).
  - Ensembling knobs: `k` (default 8), `member_init` (default `"auto"` —
    TabM's rule: `"normal"` with a numeric embedding, `"signs"` without),
    `adapter_std` (0.5) and `adapter_lr_factor` (1.0, one optimizer group
    for every per-member parameter). Every `realmlp` knob carries over
    unchanged with the same meaning.
  - Members train on the **mean of the per-member losses** through the
    existing `(n, k, out)` flatten path (so every objective, `sample_weight`
    and custom losses work untouched), while the per-epoch metric,
    `predict_proba` and the **single joint** best epoch are all the member
    mean on the prediction scale. EMA and the best-epoch snapshot reach every
    k-shaped tensor with no special casing.
  - `init_mode="std+he5"` walks the trunk on the members flattened into rows
    `(n*k, d)` — one shared weight must be fitted against all member
    representations — and copies the resulting bias to every member.
  - **`realmlp` is untouched.** At `k=1` with `member_init="ones"` RealM is
    bit-identical to `realmlp` at init (same draws, same order, `std+he5`
    included) and, with the adapters frozen, trains bit-identically through
    the full TD recipe (`tests/test_realm.py`).
  - Not carried over, deliberately: `linear_skip_idx` stays a `realmlp`
    option, and `ens_mode="vectorized"` is rejected for `realm` — it already
    vectorizes its inner ensemble, so stacking whole models on top is the
    wrong axis. The outer `n_ens` axis composes in the default loop mode.
- Models may now set `supports_vectorized = False` to opt out of
  `ens_mode="vectorized"` with a message naming the model, instead of
  silently training something else.
- **Fixed** — three trainer settings `ens_mode="vectorized"` silently dropped,
  so a vectorized ensemble trained a different recipe from the loop path it is
  supposed to be a faster copy of (`tests/test_ensemble.py`). The loop path is
  unchanged:
  - **Scheduled dropout was frozen.** `stack_module_state` stacks
    `ScheduledDropout`'s non-persistent `_keep` buffer, and the vmapped forward
    reads only the stacked copies — so `set_schedule_t`, called on the
    meta-device template whose buffers nothing reads, moved nothing and every
    member trained at the initial `1 - p` for the whole run. The hook now runs
    on a live member and its buffers are broadcast into the stacked slices
    (measured: 4 epochs of `dropout=0.4` + `dropout_schedule="flat_cos"` ended
    at keep 0.600 vectorized vs 0.800 in the loop).
  - **`weight_decay_mode` was ignored**, falling through to `_make_optimizer`'s
    own `"decoupled"` default: every vectorized fit built an AdamW. The default
    was affected too — `optimizer="adam"` couples the decay in the loop path and
    decoupled it here.
  - **`label_smoothing_schedule` and `drop_last` were ignored** — the smoothing
    epsilon stayed constant (and an unknown schedule name was not rejected), and
    the short tail batch of each epoch was still trained on.

## 0.9.1 (2026-09-02)

- **Input routing** — two opt-in options on both estimators that say *how*
  individual numeric columns reach the network. Both take column **names**,
  resolved at `fit` against the numeric block the model sees, and both
  default to `None` = the 0.9.0 behaviour, byte for byte
  (`tests/test_input_routing.py`):
  - `num_embedding_cols` — apply `num_embedding` (`"pbld"`, `"plr"`, …) to
    **these numeric columns only**; every other numeric column bypasses the
    embedding and enters the first layer linearly, still under
    `num_scaling`. The flat embedding's layout becomes
    `[embedded | bypassing | categorical]`, each block in ascending column
    order. The motivating case: a numeric block that mixes raw measurements
    (where a periodic embedding buys a flexible 1-D response) with
    target-encoded log-odds columns (already monotone scores, whose linear
    use is the point). Positional form: `num_embedding_idx` in
    `model_params`, a new [shared embedding key](docs/parameters.md).
    Rejected for token models, which have no linear bypass.
  - `linear_skip_cols` + `linear_skip_lr_factor` (`realmlp`) — a linear map
    from those (scaled) numeric columns straight onto the output,
    `raw = trunk(x) + x_skip @ W_skip + b_skip`, in its **own optimizer param
    group** at `linear_skip_lr_factor` (default 1.0) with **zero weight
    decay**. `W_skip`/`b_skip` are zero-initialized, so switching the skip on
    draws no random numbers: every other parameter stays bit-identical and
    the fit starts exactly where it would have. The skip reads the numeric
    input *before* the learnable scaling layer and any numeric embedding, so
    its weights are plain per-unit output coefficients. Positional form:
    `linear_skip_idx` in `model_params`.
  - Column selection runs through registered non-persistent **int64 buffers**
    and `index_select`, so there is no data-dependent shape or host sync on
    the forward path: CPU, CUDA and XLA/TPU all trace one graph per batch
    shape (`tests/test_xla.py::test_input_routing_xla`), and
    `ens_mode="vectorized"` works unchanged.
- **Fix** — `ens_mode="vectorized"` unstacked *every* stacked buffer through
  a strict `load_state_dict` at the end of training, which raises on any
  **non-persistent** buffer (`ScheduledDropout`'s keep probability, and now
  the routing index selectors). Those are copied in place instead.
- `masamlp.models.model_accepts(name, param)` — whether a registered model
  takes `param` in `model_params`; used to reject an estimator-level option
  a model does not implement while naming the option the user actually typed.

## 0.9.0 (2026-09-02)

- **RealMLP-TD fidelity** — the last structural gaps between masaMLP's
  `realmlp` and pytabkit's `RealMLP_TD_*` are now options, every one of them
  defaulting to the 0.8.0 behaviour, so existing results reproduce
  bit-for-bit (`tests/test_realmlp_fidelity.py`):
  - `init_mode="std+he5"` — pytabkit's **data-driven layerwise init**: each
    `NTPLinear`'s weight is redrawn and rescaled so its pre-activations have
    unit sample std on a batch of training rows, and its bias is set to minus
    a random Exp(1)-simplex combination of five sampled pre-activations of
    that unit (`he+5`). The trunk is walked in order, so every layer sees the
    activations the re-initialized layers before it actually produce. Models
    declare `needs_data_init` and the estimator supplies a capped random
    subsample (`masamlp.sklearn.DATA_INIT_ROWS`).
  - `zero_init_output=False` — RealMLP-**TD** does not zero its output layer
    (only TD-S does).
  - `scale_position="first_layer"` — TD's `add_front_scale` diagonal gain
    sits on the first hidden layer's *input* (after the numeric embedding),
    not on the raw numerics.
  - `scale_lr_factor` / `first_layer_lr_factor` / `bias_lr_factor` are now
    parameters (were hard-coded 6.0 / 1.0 / 0.1). `first_layer_lr_factor`
    multiplies the first layer's weight **and** bias, as pytabkit does.
  - `realmlp_td_params(task, pytabkit_fidelity=True)` bundles the above with
    `onehot_max_categories=8` and `drop_last=True`.
- **Trainer levers** — `weight_decay_mode` (`"auto"` / `"decoupled"` /
  `"coupled"`), `label_smoothing_schedule` (`"none"` / `"coslog4"` /
  `"flat_cos"`, pytabkit's `ls_eps_sched`; the objective is restored
  afterwards) and `drop_last` (drops each epoch's short tail batch and
  shortens the per-step schedule denominator accordingly).
- **`onehot_max_categories`** is exposed on both estimators — the
  `cat_encoding="hybrid"` threshold was fixed at 9.
- **Correction.** The preset docs claimed pytabkit couples weight decay into
  Adam. It does not: pytabkit's decay is decoupled and lr-scaled
  (`p *= 1 - wd*lr`), which is what `optimizer="adamw"` already did. The
  `weight_decay_mode` lever ships anyway, but the fidelity gap was the init,
  the output layer and the scale position — not the decay.

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
