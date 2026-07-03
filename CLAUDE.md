# CLAUDE.md — masaMLP development rules

## Project summary

masaMLP is a PyTorch tabular deep learning library and the sibling of
repleafgbm (same author, same API philosophy). It ships ten architectures
(resnet, realmlp, ft_transformer, tab_transformer, danet, tabr, modernnca,
gandalf, grn, lnn) behind two sklearn-compatible estimators, and
differentiates from existing libraries (pytabkit, pytorch_tabular) by making
**sample_weight, custom objectives, custom metrics, and early stopping on
any metric** first-class.

## Core architectural rules

- **Objectives are per-sample torch losses.** `BaseObjective.per_sample_loss`
  returns a `(n,)` tensor; the Trainer owns the weighted reduction
  `(loss * w).sum() / w.sum()`. Never reduce inside an objective — that is
  what guarantees sample_weight works for every objective, including customs.
- **Models are pure `nn.Module`s** with `forward(x_num, x_cat) -> raw` and an
  `output_layer` attribute (final `nn.Linear`) for bias initialization. All
  device/AMP/compile/batching/early-stopping logic lives in `core/`
  (trainer.py, device.py, parallel.py). Models may *declare* policy via
  class attributes (`amp_auto`, `static_state_keys`, `wants_batch_indices`)
  but never implement device logic.
- **No DataLoader.** Tensors are moved to the device once; minibatches are
  index slices. Small data (<= ~4096 rows) trains full-batch.
- **Metrics are NumPy** (`(y_true, y_pred) -> float`), computed on
  prediction-scale outputs, same contract as repleafgbm. Regression metrics
  are computed on the original (un-standardized) target scale.
- **Serialization is a directory** (manifest.json + preprocessor state +
  `model_state.pt` loaded with `weights_only=True`). Prediction after load
  must not require the objective object — only its stored `transform_name`.

## Avoid

- No DataLoader / multiprocessing workers.
- No hidden global state; seed via `utils/random.seed_everything`.
- No dependency on entmax/ncps packages — sparsemax/entmax15/CfC are in-house.
- No dataset-specific hacks; no notebook-only code.
- No CUDA-only code paths without a CPU fallback.
- Do not reduce losses inside objectives (see core rule above).

## Code map

- `src/masamlp/sklearn.py` — `BaseMasaModel`: the whole fit flow (validation →
  preprocessing → objective/metric resolution → build_model → Trainer).
- `src/masamlp/regressor.py`, `classifier.py` — task-specific target handling
  (standardization / label encoding + class_weight).
- `src/masamlp/core/` — objectives, metrics, trainer, device, serialization.
- `src/masamlp/models/` — registry (`register_model`/`build_model`),
  `base.py` FeatureEmbedding (categorical embeddings + PLR/periodic numeric),
  `layers.py` (GhostBatchNorm1d, ScalingLayer, sparsemax/entmax15/t_softmax),
  `retrieval.py` (candidate corpus + eval cache for tabr/modernnca), and one
  file per architecture.
- `src/masamlp/data/` — `TabularPreprocessor` (fitted on train only, state is
  json/npz-serializable) and the `TabularData` tensor bundle.
- `tests/` — pytest; small seeded synthetic datasets (hundreds of rows).

## Attribution

Architectures follow published work — see `docs/attribution.md`. DANet is a
clean-room reimplementation of the MIT-licensed official repo; the CfC cell
follows Hasani et al.'s closed-form continuous-time equations.

## Verification

`bash scripts/check.sh` runs ruff + pytest + examples/quickstart.py. The
differentiator gate is `tests/test_sample_weight.py`,
`tests/test_custom_objective.py`, `tests/test_custom_metric.py`.
`tests/test_docs_parameters.py` enforces that every estimator/model
constructor parameter is documented in `docs/parameters.md` — new parameters
need a doc entry in the same change.
