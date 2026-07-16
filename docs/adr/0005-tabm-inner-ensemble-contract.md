# ADR 0005 — TabM-style inner ensembling: the (n, k, out) contract and per-arch injection

Status: accepted (2026-07-16)
Related: [[ADR-0001]] (the docs gate this lands under), `docs/roadmap.md`
(the ftt-`k` and `predict_members` follow-ups), KI-014 (measurement
coverage).

## Context

TabM (Gorishniy et al. 2024, arXiv:2410.24210) landed on `feat/tabm-arch` as
`models/tabm.py` — the TabM-mini structure: shared FeatureEmbedding →
per-member multiplicative adapter → shared MLP backbone → per-member
`EnsembleHead`. As committed, the ensemble contract is smeared across three
places: only `MulticlassSoftmax` understands 3D raw outputs, and
`TabM.forward`'s eval branch averages members with a softmax-specific
logsumexp inside the model. Measured consequences: **binary classification
and regression fits crash** (shape errors in `BinaryLogistic` /
`SquaredError`), custom objectives would receive 3D raw logits (violating
the custom-objective differentiator), and the docs gate is red
(`tabm.adapter_std` undocumented).

The feature ask (design interview 2026-07-16): make TabM-style ensembling
available to other architectures (ft_transformer first) as a parameter,
without selecting `tabm`. Prior measurement constraint: naive full
BatchEnsemble (per-layer ±1 sign adapters) measured worse than a single
model on synthetic; the mini structure did not.

## Decisions

1. **The contract is the feature — `(n, k, out)` becomes a first-class
   model output.** `forward` may return `(n, k, out)` in any mode; dim 1 is
   ensemble members. Loss side: a shared helper
   `weighted_loss(objective, y, raw, weight)` in `core/trainer.py` flattens
   3D to `(n·k, out)` and repeats `y`/`weight` with
   `repeat_interleave(k)`, so objectives keep the per-sample `(n,)`
   contract untouched (custom objectives see standard shapes). The flat
   weighted reduction equals the mean over members of per-member weighted
   means — identical to the previous 3D softmax math. Both
   `Trainer.train_step` and `ensemble.py::fit_vectorized` (the one call
   site that bypassed the trainer) use the helper. The
   `MulticlassSoftmax` 3D branch is deleted (never released, no compat).
   *Rejected:* 3D handling inside each objective (every custom objective
   would need it too); keeping tabm multiclass-only.

2. **Prediction: transform per member, then mean — centralized in
   `apply_transform`.** Internally two composable steps:
   `transform_members(raw, name)` (softmax on `dim=-1` — identical for 2D;
   elementwise otherwise) then `mean(dim=1)` when 3D. All four prediction
   consumers (trainer eval, `sklearn._predict_transformed`, `parallel.py`,
   `ensemble.py`) already funnel through
   `apply_transform`/`objective.transform`, so no call-site changes.
   `TabM.forward` loses its eval branch and logsumexp hack — always
   `(n, k, out)`. Prediction-identical for multiclass
   (`softmax(log p̄) = p̄` ≡ mean of member softmaxes). Serialization still
   needs only `transform_name`. Probability averaging matches the paper and
   the existing `n_ens` convention ("average on the transformed scale").
   *Rejected:* model-internal averaging (the model would own objective
   knowledge); logit-mean (≠ probability mean).

3. **Injection = per-arch explicit `k` (uniform name, default 1), shared
   parts in `layers.py`** (`EnsembleHead` moves there, plus the member
   adapter). No universal wrapper: embeddings live inside models (flat
   models receive `FeatureEmbedding`, token models build `TokenEmbedding`
   themselves), so an external wrapper cannot reach the mini adapter
   point. Third-party models get inner ensembling by just returning
   `(n, k, out)` — documented in the `register_model` docstring as public
   contract. Non-supporting archs keep the existing `_check_model_params`
   error. Scope this round: `ft_transformer` only; retrieval archs excluded
   (corpus×k memory, batch-index protocol), BatchNorm archs deferred
   (flattened members would share BN statistics).

4. **FTT structure (PR2): token-adapter mini mirror.** After
   `TokenEmbedding`: tokens `(n, s, d)` → per-member multiplicative adapter
   `(k, 1, d)`, init `N(1, adapter_std)` (the house init measured for
   tabm) → flatten `(n·k, s, d)` → existing blocks unchanged
   (`nn.MultiheadAttention` is 3D-only, so the flatten is forced anyway) →
   CLS `(n·k, d)` → unflatten `(n, k, d)` → shared `head_norm` →
   `EnsembleHead`. **`k=1` must build the exact legacy module tree** so
   0.5.0 checkpoints keep loading (state_dict key compat). Compute is k×
   (attention included); docs recommend k = 4–8 for ftt.
   *Rejected:* per-member CLS only (no paper backing, weaker diversity);
   BatchEnsemble on the FFN linears (contradicts the measured naive-BE
   rejection).

5. **The two ensemble axes are orthogonal and composable.** `n_ens`
   (outer, independent models; loop/vectorized/sharded) × `k` (inner,
   weight-shared): total `n_ens·k` members, two-stage mean ≡ flat mean.
   `ens_mode="vectorized"` supports inner-k via the shared `weighted_loss`
   (shape-driven — no model-detection machinery). Inner-k early stopping is
   necessarily on the ensemble-average metric: per-member best-epoch
   restoration is ill-defined on shared weights (unlike `n_ens`, whose
   members restore independently) — the asymmetry is documented.

6. **Member-level prediction API deferred but pre-decided.**
   `predict_members` / `predict_proba_members` → `(n, m[, C])` with
   `m = n_ens·k`, prediction scale (regression on the original target
   scale), invariant `members.mean(axis=1) == predict/predict_proba`
   (exact: uniform two-stage mean, affine inverse-standardization
   commutes). `transform_members` exists from PR1, so the future change is
   only exposing the path that skips the mean. No estimator surface change
   in 0.6.0 (repleafgbm-mirror surface).

7. **Landing: two PRs, measurement-gated.** PR1 (`feat/tabm-arch` →
   0.6.0): decisions 1–2, tabm always-3D, tests (binary / regression /
   custom-objective / vectorized×tabm / multiclass loss+eval equivalence
   pinned numerically), `docs/parameters.md` tabm section (fixes the red
   gate), CHANGELOG, this ADR, `register_model` docstring. PR2
   (`feat/ftt-k` → 0.7.0): decision 4, merged only if an S6E7-style Colab
   measurement shows ftt `k>1` is not worse than single FTT (precedent:
   naive BE was rejected on measurement).

## Verification (this session, 2026-07-16, CPU)

- **Pre-fix crashes reproduced** before the design was accepted: tabm +
  binary → `ValueError: Target size (torch.Size([200])) must be the same as
  input size (torch.Size([200, 1]))`; tabm + regression → `RuntimeError:
  The size of tensor a (4) must match the size of tensor b (200)`. Both fit
  and predict cleanly after the contract landed.
- **Equivalence pinned numerically in CI** (`tests/test_tabm.py`):
  `weighted_loss` on `(n, k, out)` ≡ mean over members of per-member
  weighted losses (softmax and squared-error); `apply_transform` 3D softmax
  ≡ the pre-0.6.0 in-model `logsumexp` averaging (`torch.testing
  .assert_close`); 2D paths bit-unchanged.
- **sample_weight exactness holds through the flatten**: weight 3 ≡ row
  duplicated 3× at `atol=1e-4` (tabm, dropout 0, full batch — the
  `test_sample_weight.py` isolation).
- **Both ensemble axes compose**: `n_ens=2 × k=4` fits and predicts in
  `ens_mode="loop"` and `"vectorized"`.
- Test-fixture learning-rate note: 40 full-batch steps at lr `1e-3` left
  the plain-MLP backbone above the generic quality bar (rmse 2.247, bar
  2.180); `3e-3` → rmse **0.803**. Recorded in `conftest.TRAIN_KWARGS`
  (a test-speed setting, not a model default).
- Full suite green including the previously red docs gate
  (`tabm.adapter_std`).

## Consequences

- Objectives stay 2D-only forever; inner ensembling is invisible to
  objectives, metrics, and their custom variants.
- Any registered model can opt into inner ensembling by shape alone.
- Eval/predict memory and FLOPs scale ×k for inner-k models (batched
  kernels; `eval_batch_size` chunking unchanged).
- tabm binary/regression become supported for the first time — new
  capability; at-scale quality verdicts pending the next campaign
  (**KI-014** tracks the measurement gap, including the missing TPU
  verdict).
- Documents updated with this ADR: `docs/parameters.md` (tabm section +
  inner-axis note under Ensembling), `docs/glossary.md` (inner ensemble,
  outer-vs-inner member), `docs/known_issues.md` (KI-014),
  `docs/roadmap.md` (ftt-`k` conditional, `predict_members`,
  resnet/realmlp-`k` frozen until the ftt verdict), `docs/attribution.md`
  (TabM entry), README model table + TPU scope, CLAUDE.md model contract,
  CHANGELOG 0.6.0.
- **Retreat path**: the contract is additive — reverting means restoring
  the in-model eval averaging in `tabm.forward`, the 3D branch in
  `MulticlassSoftmax`, and inlining the reduction back into
  `Trainer.train_step`/`fit_vectorized`. Serialized model directories are
  unaffected either way (`manifest.json` and `model_state.pt` formats
  unchanged; only forward-shape semantics moved).
