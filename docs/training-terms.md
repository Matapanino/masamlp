# Model training terms — WP2 interface, 2026-09-05

This opt-in research interface supports joint auxiliary tasks (WP4 / Card 2)
and final-parameter shrinkage (WP3 / Card 1). Existing models need no changes.
`forward(x_num, x_cat)` still returns ordinary raw predictions. A model may add:

```python
def training_terms(self, batch: TabularData, raw: Tensor) -> TrainingTerms | None:
    ...
```

Import `TrainingTerms`, `RowTerm`, and `ModelRegularizer` from
`masamlp.core.training_terms`. The Trainer calls the hook once after each
training forward, under the training autocast context. It never calls it for
validation, early stopping, or prediction. Returning `None` or empty terms
preserves the objective path. Zero coefficients are skipped in the reduction,
so their tensors do not create optimizer gradients. To avoid auxiliary compute
and RNG consumption too, return empty terms before computing a disabled branch.

The hook receives the current, device-resident `TabularData` batch (`x_num`,
`x_cat`, `y`, `weight`) and its raw predictions with their autograd graph intact.
Targets in `batch.y` are objective-prepared (including regression scaling),
not original-scale targets. Extra targets can come from explicitly routed
input columns, or model-owned buffers indexed by the existing
`wants_batch_indices = True` / `current_batch_indices` protocol. Full shared
batches use `arange(n)`; independent member batches use indices shaped `(b, k)`.
Keep auxiliary labels inside the fitting boundary and prevent their identity
or descendant features from entering a branch that predicts them. WP4 owns
that routing; this interface does not infer masks or fit auxiliary targets.

## Reduction and scaling

`RowTerm(values, coefficient=1.0)` requires exactly `raw.shape[:-1]`:
`(b,)` for ordinary models, `(b, k)` for inner ensembles. It contains **unreduced**
per-row losses. Reduce output/task dimensions within the hook; do not reduce
rows, apply sample weights, or broadcast a scalar regularizer over rows.
Multiple auxiliary terms are added to the primary per-row loss with their
coefficients, then reduced together by the Trainer. Coefficients must be
finite, nonnegative Python numbers. Zero-weight rows have zero direct loss
contribution; this does not remove their possible influence through BatchNorm
or other cross-row model operations.

`ModelRegularizer(value, normalizer, coefficient=1.0)` requires a scalar tensor,
a fixed positive finite Python normalizer, and a finite nonnegative coefficient.
It adds `coefficient * value / normalizer` **once per optimizer step**, after
reducing the data terms. Choose a fixed normalizer such as the number of
penalized parameters or fitting rows; encode any support-dependent factors
inside `value`. Never use the current batch size, weight sum, member count,
or steps per epoch as an automatic divisor. The objective is
`weighted_data_risk + lambda * normalized_parameter_penalty` at every step.
Changing batch size or globally scaling sample weights therefore does not
change the penalty coefficient. Changing the number of optimization steps can
still change the fitted parameters; this is not a promise of identical
optimizer trajectories or an epoch-level penalty budget.

Compute numerically sensitive penalties in FP32 inside the hook, for example
`self.table.weight.float().square().sum()`. Casting the resulting scalar to
FP32 in the reducer cannot repair earlier FP16 overflow. Term tensors must
stay on the same device as `raw`. All trainable state belongs to registered
model parameters/submodules; fixed persistent tensor state belongs to buffers.
The normalizer may be reconstructed from JSON constructor configuration or
parameter shapes. Do not extract a device tensor into Python each step.

## TabM independent-member weights: intentional contract correction

For independent member batches, let `L_ij` include primary and auxiliary
losses and `W_j = sum_i w_ij`. WP2 implements:

```
member_loss_j = sum_i(w_ij * L_ij) / W_j  if W_j > 0, else 0
loss = sum_j(member_loss_j) / k + sum(model_regularizers)
```

The mean always divides by `k`, including zero-total members. Their direct data
gradients are zero; shared parameters, regularizers, optimizer momentum and
weight decay may still update them. An entirely zero-weight batch is skipped
by the Trainer, including regularizers, optimizer and EMA updates, preserving
the pre-WP2 skip policy. Calling `weighted_loss` directly returns zero data
risk for all-zero weights; supplied regularizers still contribute there.

For losses `[[1, 4], [9, 16]]` and weights `[[1, 1], [1, 3]]`, the member
risks are `5` and `13`, giving **9**. The old pooled denominator gave
`62 / 6 = 10.333333`. If the second member's weights are both zero, the new
risk is `5 / 2 = 2.5`. This deliberately repairs the documented contract
mismatch identified in the review §1. Unit-weight behavior is unchanged;
shared batches keep the original flattened weighted reduction. Objectives,
including customs, still receive flattened `(b*k, out)` predictions and
aligned targets. Prediction remains the member mean on the transformed scale.

## Example: auxiliary head and final-parameter shrinkage

This minimal registered model uses numeric column 0 as an auxiliary target
and columns 1 onward as its predictors. It illustrates the interface, not the
WP4 architecture or its required full descendant mask. The primary branch
may use all input columns.

```python
import torch
from torch import nn

from masamlp import MasaRegressor
from masamlp.core.training_terms import ModelRegularizer, RowTerm, TrainingTerms
from masamlp.models import register_model


@register_model("joint_example")
class JointExample(nn.Module):
    def __init__(self, embedding, out_dim, auxiliary_weight=0.1, shrinkage=0.01):
        super().__init__()
        self.embedding = embedding
        self.output_layer = nn.Linear(embedding.d_out, out_dim)
        self.aux_head = nn.Linear(embedding.n_num - 1, 1)
        self.auxiliary_weight = auxiliary_weight
        self.shrinkage = shrinkage

    def forward(self, x_num, x_cat):
        return self.output_layer(self.embedding(x_num, x_cat))

    def training_terms(self, batch, raw):
        auxiliary = ()
        if self.auxiliary_weight:
            pred = self.aux_head(batch.x_num[:, 1:]).squeeze(-1)
            auxiliary = (RowTerm(
                (pred - batch.x_num[:, 0]).square(), self.auxiliary_weight
            ),)
        penalty = self.output_layer.weight.float().square().sum()
        return TrainingTerms(auxiliary=auxiliary, regularizers=(
            ModelRegularizer(penalty, self.output_layer.weight.numel(), self.shrinkage),
        ))


# X has at least two numeric columns; preprocessing is fitted on training only.
# est = MasaRegressor(model="joint_example", device="cpu", n_epochs=3)
# est.fit(X, y)
# est.save_model("joint-model")
# loaded = MasaRegressor.load_model("joint-model")
```

Registered auxiliary heads and penalty-related buffers follow existing best
checkpoint / EMA / directory save-load behavior. The ephemeral `TrainingTerms`
objects and their autograd graphs are not stored. Import the custom model's
registration before loading in a new process. Constructor options must remain
JSON-serializable. Loaded prediction does not need the objective or a separate
training-term callback. This interface does not add optimizer-resume support.

Loop ensembles (including device-sharded loops) use the hook on each member.
`ens_mode="vectorized"` with multiple outer members rejects models exposing
`training_terms`, even if coefficients are zero; choose `ens_mode="loop"`.
Inner ensembles are supported with the shapes above. Compile applies to
`forward`; the hook executes outside that compiled forward on the original
model with the same parameters. Avoid storing forward graphs on the model.
XLA hooks must use tensor operations with static shapes and avoid device-to-host
conditionals. No new estimator constructor argument or serialization version
is needed.
