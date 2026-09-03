# RealM — RealMLP under full BatchEnsemble

`model="realm"` is the [RealMLP](https://arxiv.org/abs/2407.04491) backbone
ensembled the [TabM](https://arxiv.org/abs/2410.24210) way: `k` members share
one weight matrix per layer and differ only through rank-1 adapters, a
per-member bias and a per-member output head.

masaMLP already had the two halves separately, and neither is this:

| | backbone | ensembling |
|---|---|---|
| `realmlp` | full RealMLP-TD | none — diversity only from the **outer** `n_ens` axis (k independent fits, k× the compute) |
| `tabm` | plain ReLU MLP | TabM-**mini**: one multiplicative adapter on the shared embedding |
| `realm` | full RealMLP-TD | TabM **full BatchEnsemble**: `r`/`s` adapters on *every* layer |

The point is that diversity is **learned jointly** on a shared feature
extractor. `n_ens` buys diversity by paying for k separate models; `realm`
trains one model whose members are regularized by each other and by the
shared representation they must all use.

## Shapes

With embedding width `e`, hidden widths `d_1 … d_L` and `out` outputs:

```text
z = embedding(x_num, x_cat)          # (n, e)      shared (PBLD / one-hot / routing)
z = front_scale(z)                   # (n, e)      shared, if scale_position="first_layer"
h = z[:, None, :] * r0               # (n, k, e)   <- one representation becomes k

for each layer l:                    # BatchEnsembleLinear
    h = (h * r_l)                    # (n, k, d_in)   broadcast over rows
    h = (h @ W_l) / sqrt(d_in)       # (n, k, d_out)  ONE matmul for every member
    h = h * s_l + b_l                # (n, k, d_out)
    h = ParametricActivation_l(h)    # alpha is (d_out,)  — shared
    h = ScheduledDropout_l(h)        # masks independent over (n, k, d_out)

raw = einsum("nkd,kdo->nko", h, head_w) / sqrt(d_L) + head_b   # (n, k, out)
```

Member *i*'s effective weight is `W ⊙ (r_i s_iᵀ)`, but it is **never
materialized** — that is the whole trick. `k` is a fixed config value, every
shape is static, and there is no `torch.vmap` and no Python loop over members
(`tests/test_realm.py::test_realm_module_has_no_vmap_path`).

Per-member tensors: `first_adapter (k, e)`, per layer `r (k, d_in)`,
`s (k, d_out)`, `bias (k, d_out)`, and the head `(k, d_L, out)` + `(k, out)`.
Everything else — the embedding, both scaling layers, every `W`, every
activation `alpha` — is a single tensor.

## Training and prediction

- **Loss** = the mean of the per-member losses. The trainer flattens the
  member axis into rows before the objective runs (`weighted_loss`), so every
  objective, `sample_weight` and custom losses work unchanged and never see a
  member dim. This is *not* the loss of the mean prediction: the mean loss
  gives each member a proper supervised gradient, where the loss of the mean
  couples members and invites collapse.
- **Evaluation** = the member mean on the prediction scale (`apply_transform`
  averages dim 1), which is also what `predict_proba` / `predict` return, and
  what the per-epoch metric is computed on. Early stopping therefore restores
  **one joint best epoch** for the whole ensemble — per-member checkpoint
  selection would be a selection leak against the deployed object.
  `predict_proba_members` still exposes the k members individually.
- **EMA and the best-epoch snapshot** walk `named_parameters` /
  `state_dict`, so every `k`-shaped tensor is tracked and restored with no
  special casing.
- **Optimizer groups** are RealMLP's, plus one: every per-member parameter
  sits in its own group at `adapter_lr_factor` (per-member biases keep
  RealMLP's `bias_lr_factor` and zero weight decay, with the adapter factor
  on top). At `adapter_lr_factor=1.0` every group's effective (lr, wd) factor
  equals RealMLP's.
- **`init_mode="std+he5"`** walks the trunk on the members **flattened into
  rows** `(n*k, d)`: one shared weight has to be fitted against all member
  representations, not one of them. The resulting bias is copied to every
  member, so at init the members differ only through the first adapter —
  exactly as TabM's reference does.

## Cost — read this before raising `k`

Parameter memory stays near one backbone, but **compute is ~k
member-forwards** and every hidden activation carries the member axis.

Measured at the s6e9 shape (41 numeric inputs, PBLD `d_num_embedding=8`,
`hidden_sizes=(384, 384, 384)`, one output):

| | parameters | fwd+bwd @ B=2048, 1 CPU thread |
|---|---|---|
| `realmlp` | 430,072 | 146 ms/step |
| `realm`, k=1 | 432,648 | 182 ms/step |
| `realm`, k=8 | 461,439 | 1,565 ms/step |
| `realm`, k=32 | 560,151 | — |

Parameters grow by only 4,113 per member (+0.96 % each), but the step time
grows ~linearly in `k`. Activation memory does too: at `B=2048, k=32, d=384`
one `(n, k, d)` tensor is 25.2M values — ~96 MiB fp32, ~48 MiB bf16 — and a
three-layer trunk keeps several of those alive for the backward pass. That is
plausible on a TPU in bf16 and risky on a 16 GB T4.

**Start at `k=8` or `k=16`.** Promote to `k=32` only after a static-shape XLA
run and a real memory check on the target device. If a batch has to be split,
split it along `B` with a **fixed** chunk size — never loop over members.

## Relationship to `realmlp`

`k=1` with `member_init="ones"` is RealMLP: the draws happen in the same
order with the same shapes, the adapters are exactly `1.0`, and the forward
output plus every shared tensor are **bit**-identical, `std+he5` init
included. Freeze the adapters and six epochs of the full TD recipe (coslog4
lr, flat_cos weight decay, scheduled dropout, EMA) reproduce RealMLP bit for
bit. Left trainable — the normal case — the rank-1 adapters move, so a `k=1`
`realm` is a mild overparametrization of RealMLP rather than a copy of it.

Two `realmlp` behaviours deliberately do **not** carry over:

- `linear_skip_idx` / `linear_skip_cols` (the additive linear skip) is not
  implemented for `realm`; the estimator says so by name.
- `ens_mode="vectorized"` is rejected. `realm` already vectorizes its inner
  ensemble; stacking whole models on top of it with `torch.func` is the wrong
  axis (k × `n_ens` member-forwards in one vmap) and the stacked
  non-persistent buffers freeze the scheduled-dropout probability. The outer
  `n_ens` axis composes fine in the default loop mode.

## Usage

```python
from masamlp import MasaClassifier, realm_td_params

# The RealMLP-TD recipe, ensembled: same training knobs, k members.
clf = MasaClassifier(**realm_td_params("classification", k=8))

# Or point at the architecture directly.
clf = MasaClassifier(
    model="realm",
    model_params={
        "hidden_sizes": (384, 384, 384),
        "k": 8,
        "adapter_lr_factor": 2.0,
        "member_init": "auto",      # "normal" here, because PBLD is on
        "use_parametric_act": True,
        "dropout": 0.03,
        "dropout_schedule": "flat_cos",
        "d_num_embedding": 8,
    },
    num_embedding="pbld",
    ema_decay=0.995,
)
```

Suggested first search: `k ∈ {8, 16}`, `adapter_lr_factor ∈ {1, 2, 4}`,
`dropout ∈ {0.01, 0.03, 0.06}`, `member_init ∈ {normal, signs}` — with the
rest of the recipe held at whatever the `realmlp` flagship already uses.
`k=32` is a scale-up confirmation, not a search axis.
