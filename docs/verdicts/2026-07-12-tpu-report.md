# TPU 0.5.0 verification — Kaggle v5e-8, torch 2.8.0 / torch_xla 2.8.0

Branch `v0.5.0-tpu` (PR #6), 2026-07-12. Wave E1 (50k-scale: step fusion,
scan prototype, bf16 predict, tab_transformer profile, openxla repro) +
wave E2 (345k retrieval eval). Raw log: `2026-07-12-tpu-run.log`; narrative:
`research/tpu-xla.md` §10–11; decisions: ADR 0003/0004.

**Method note.** The TPU pool was congested (Saturday, right after the
weekly quota reset; the interactive queue showed "#31") and Kaggle's batch
save pipeline was rolling completed runs back unsaved and requeuing them
forever. Results were recovered by polling every 30 s and downloading inside
the short COMPLETE window; unsaved runs did not consume quota. All timed
rows are cold-cache, one measurement per process.

## Step fusion (`xla_fuse_steps`) — measured verdict: default stays 1

Verdict configs, amp=auto (bf16), batch 1024, cold fit seconds:

| model | K=1 | K=8 | K=32 |
|---|---|---|---|
| resnet (40 ep) | 48.5 | 102.0 | 161.8 |
| realmlp TD-S (40 ep) | 71.0 | 99.4 | 150.5 |
| tab_transformer (20 ep) | 76.5 | 417.5 | 704.5 |

Fusion loses everywhere: XLA compile time grows super-linearly with the
unrolled K-step graph and dwarfs the real ~20% steady-state per-step saving
(prototype MLP, steady epochs: 0.165s at K=1 vs 0.13s at K=8, **bitwise
param parity** without dropout). Break-even is roughly ≥256 epochs at
resnet scale — shipped as a documented escape hatch, not a default.

- Parity at verdict scale (same seed, K=1 vs K=8): rmse-equivalent but not
  value-equal for dropout models (resnet 0.2347 vs 0.2481, realmlp 0.1592
  vs 0.1588; max|pred diff| 0.42 / 0.92) — the XLA RNG seed advances per
  graph execution, so K selects a different mask stream (contract
  documented; RNG-free training is K-invariant, bitwise-verified by the
  prototype and CI).
- `torch_xla.experimental.scan` over training steps — the in-graph While
  loop that would amortize compilation — **fails on torch_xla 2.8**:
  `torch.func.grad` inside the scan body ("element 19 of tensors does not
  require grad"). Roadmap: revisit on TorchTPU.

## Honest cold baseline correction (0.4.0 erratum)

Cold K=1 rows (compiles=33) show the 0.4.0 small-model rows (compiles=4,
e.g. resnet bf16 fit 34.8s, predict 0.80s) rode the same-process compile
cache — the wave-C ordering correction had only been applied to retrieval
rows. Honest cold numbers on the same image: resnet fit 48.5s
(first-predict 19.3s); steady-state predict matches the old "bf16" column
(prediction always was fp32 and amp-independent).

## bf16 prediction (`amp_predict`) — accuracy-safe, speed-neutral

Steady-state 200k-row predicts (fp32 → bf16): resnet 0.81→0.67s, realmlp
0.18→0.17s, ft_transformer 1.38→1.38s, tabr 14.4→14.8s, tab_transformer
2.38→2.03s, modernnca 2.37→2.68s. Δrmse ≤ 0.003 on all six; max|pred diff|
0.06–0.29 (bf16 scale). Ships as a correctness-verified opt-in for
memory/marginal gains; TPU fp32 prediction is already fast.

## tab_transformer — cause identified (KI-013)

Per-section profile (batch 1024, bf16, barrier per iter): full train step
**100.3 ms/iter** vs forward-only 8.1 ms (transformer blocks 7.1, embedding
0.7, head 0.5). The backward+optimizer is ~92% of the step — ~11× the
forward, against the usual 2–3× — because `nn.MultiheadAttention` at
d_token 32 / head_dim 4 lowers to MXU-hostile small ops in reverse mode.
Zero aten fallbacks, zero recompiles: pure lowering quality. Not the
categorical-embedding gathers, not GhostBatchNorm. Fix candidate on the
roadmap: SDPA-based attention block.

## openxla dynamo backend — miscompile not reproduced in minimal form

{mlp, residual_ln, tiny-attention} × {fp32, bf16} × {lazy, openxla}: all 12
configs train to equal rmse (openxla up to ~1.6× faster). Wave B's
ft_transformer collapse (rmse 0.20 → 3.18) needs something
masamlp-specific; the upstream issue is deferred until a self-contained
repro exists, and `compile=True` stays refused on XLA.

## Retrieval eval @345k (wave E2)

*(pending — TabR eval-chunk fusion sync 1 vs 8 × predict fp32/bf16;
ModernNCA OOM watch)*

## Quota spent

Wave E1 ≈ 1.1 TPU-h; ~5 h of additional run-time was burned by the
rollback-and-requeue cycles but did not count against quota.
