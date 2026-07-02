# Design

## The one core idea

Everything extensible is a plugin with a minimal contract, and the trainer
owns the mechanics:

| plugin | contract | registry / wrapper |
|---|---|---|
| objective | `per_sample_loss(y_true, raw_pred) -> (n,) Tensor` (+ `transform`, `init_bias`, `prepare_target`, `out_dim`) | `get_objective` / `make_objective` |
| metric | `(y_true: np.ndarray, y_pred: np.ndarray) -> float` with `name`, `minimize` | `get_metric` / `make_metric` |
| model | `nn.Module(embedding, out_dim, **model_params)` with `forward(x_num, x_cat) -> (n, out_dim)` and an `output_layer` | `register_model` / `build_model` |

Because objectives never reduce, the trainer's single weighted reduction
`(loss * w).sum() / w.sum()` gives every objective — built-in or custom —
correct `sample_weight` and `class_weight` semantics. This is the load-bearing
design decision; do not move the reduction into objectives.

## Layering

```
regressor.py / classifier.py     target handling only
        |
sklearn.py (BaseMasaModel)       fit flow: validate -> preprocess -> resolve
        |                        objective/metrics -> build_model -> Trainer
core/trainer.py                  devices, AMP, compile, batching, early stop
models/*                         pure nn.Modules
data/preprocessing.py            DataFrame -> float32/int64 arrays, stateful
core/serialization.py            directory format, weights_only load
```

## Prediction scale conventions

- Raw model outputs are `(n, out_dim)`; `objective.transform` maps to the
  prediction scale (identity/sigmoid/softmax/exp), identified by a string so
  loaded models predict without the objective object.
- Regression targets are standardized inside the estimator by default;
  metrics and `predict` always operate on the original scale
  (`_inverse_target` is applied to eval predictions before metrics).
- Classification metrics receive encoded integer labels and probability
  outputs — identical to repleafgbm.

## Why no DataLoader

Small/medium tabular tensors fit in device memory; worker processes,
collation, and per-batch host->device copies dominate the step time
otherwise. Index-slicing device-resident tensors makes CPU epochs and GPU
epochs alike allocation-light, and full-batch mode removes even the
slicing when the data is small.
