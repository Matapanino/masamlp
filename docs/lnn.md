# TabularLNN — design notes and status

**Status: experimental.** Unlike `resnet` and `danet`, this model does not
implement a published tabular architecture.

## What "LNN" means here

Liquid Neural Networks (LTC/CfC; Hasani et al.) are continuous-time
recurrent models built for sequences. There is no established "tabular LNN"
in the literature (as of mid-2026, applying liquid networks to static,
non-sequential data is an open research direction — e.g. generalized-LNN
proposals). masaMLP's `lnn` is therefore an *adaptation*, clearly labeled as
such:

1. Features are embedded to a flat vector (same `FeatureEmbedding` as the
   other models) and projected to the hidden width.
2. An in-house **CfC cell** (closed-form continuous-time update: two tanh
   candidate states blended by a learned, input-conditioned time gate) is
   unrolled for `n_steps` **virtual time steps with the embedded vector as a
   constant input**.
3. The final hidden state goes through LayerNorm into the output head.

Intuition: instead of one deep feed-forward stack, a small recurrent cell
refines a latent state a few times while re-reading the same input — a
weight-tied deep network with a gated, continuous-time-flavored update rule.

## Practical notes

- `model_params`: `d_hidden` (128), `n_steps` (6), `d_backbone` (128),
  `dropout` (0.1).
- Because it uses LayerNorm (no BatchNorm), rows are fully independent in
  the forward pass; the exact sample-weight equality tests use this model.
- Compute grows linearly with `n_steps`; `n_steps=1` degenerates to a gated
  two-layer MLP.

## Alternatives considered

- Depending on the `ncps` package (Apache-2.0) instead of the in-house cell:
  rejected to keep zero extra dependencies and MIT purity.
- Feature-token sequence mode (feed features one per step): roadmap.
- If "LNN" was intended to mean something else (e.g. IBM's Logical Neural
  Networks), this module is the one to replace — the estimator surface would
  not change.
