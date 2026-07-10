# TPU/XLA verification — Kaggle TPU VM (v5e-8), torch 2.8.0 / torch_xla 2.8.0

Branch `v0.4.0-tpu` (PR #5), 2026-07-10/11. Four private Kaggle kernels
(`masamlp-tpu-wave-{a,b,c,d}`), git-pin installs, weekly TPU quota. Full
narrative and corrections: `docs/research/tpu-xla.md` §7–§9; decisions:
ADR 0002/0003. Kaggle's `Tpu1VmV38` accelerator provisions a **v5e-8**
(8 × single-core v5e chips, 16 GiB HBM each, 224-vCPU host); masaMLP uses
one device (`xla:0`).

## Correctness (wave A + C)

- All ten models fit/predict on `device="xla"`, fp32: **save →
  CPU-load prediction parity bitwise exact** for all ten.
- 3–10 XLA compiles per fit, **independent of epoch count** — including
  coslog4 lr + flat_cos weight-decay + scheduled dropout (per-step Python
  scalars are lifted to graph parameters; the tensor-p ScheduledDropout
  covers the one true op-attribute constant).
- Only aten fallback on the hot path: `aten::_local_scalar_dense` × n_epochs
  = the designed one-host-sync-per-epoch loss check.
- Differentiator gates (sample_weight / custom objective / custom metric),
  early stopping, and ulp-level load parity run per-PR on XLA:CPU in CI.

## Speed — 50k rows, verdict configs (cold cache; L4 = 2026-07-03 report)

| model | TPU fp32 fit | TPU bf16 fit | TPU bf16 predict(200k) | L4 fit (amp) | L4 predict |
|---|---|---|---|---|---|
| resnet | 46.7s | 34.8s | 0.80s | 5.0s | 0.93s |
| realmlp (TD-S) | 56.8s | 60.3s | 0.17s | 12.3s | 0.21s |
| ft_transformer | 35.2s | 38.4s | 1.92s | 24.7s | 3.18s |
| tabr | 40.9s | 41.0s | 28.3s | 12.3s | 3.77s |
| tab_transformer | 64.4s | 69.0s | 2.62s | 12.3s | 2.78s |
| modernnca | 49.1s | 35.5s | 1.16s | 5.4s | 2.33s |

- **Warm-cache repeats fit 2–3x faster** (in-process XLA compile cache;
  e.g. tabr 40.9s → 17.1s). First-fit numbers above are the honest cold
  case; HPO-style repeated fits in one process ride the cache.
- `batch_size=8192` roughly halves TPU fit times (resnet 21.8s, realmlp
  25.8s, ft 23.3s) **but changes convergence at fixed epochs** (rmse
  column degraded in every case) — hence the device-independent
  `batch_size="auto"` default and explicit-batch guidance in devices.md.
- bf16 helps the matmul-heavy fits moderately (modernnca −28%, resnet
  −25%) and is neutral for tabr/realmlp at this scale; prediction is
  amp-independent (autocast wraps training only).

## Speed — 345k rows retrieval (cold cache, bf16, shipping code = wave D)

| model | TPU fit (3 ep) | TPU predict(50k) | L4 fit | L4 predict |
|---|---|---|---|---|
| tabr | 152.0s | 85.6s | 45.9s | 3.09s |
| modernnca | **71.3s** | 20.4s | 81s | 4.1s |

ModernNCA's 345k fit **beats the L4** — the softmax-over-corpus training
step is exactly MXU-shaped. The per-chunk eval barrier (bug 3 below) costs
TabR's chunked search some cross-chunk fusion (predict 47.8s → 85.6s), the
price of ModernNCA's eval fitting in HBM at all; TabR eval on TPU stays the
documented slow path either way.

## Verdict against the ADR 0002 §1 success bar (≥ T4 parity)

Mixed, model-shaped, and now documented rather than promised:

- **At or better than T4-class** (T4@30k×5/3 extrapolation):
  ft_transformer, tabr (fit), and everything's *prediction* path — TPU
  bf16 predict beats even the L4 on 4 of 6 models.
- **Below T4 at the defaults**: the small MLPs (resnet, realmlp) and
  tab_transformer — per-step dispatch overhead dominates at batch 1024;
  torch_xla's floor, not a masaMLP knob. Large batches close most of the
  gap at a convergence cost the user must opt into.
- TabR eval search (cdist/topk at 345k) is the one clear regression vs
  CUDA (85.6s vs 3.1s) — XLA's topk over streamed, barrier-separated
  chunks; documented as a known slow path. ModernNCA at the same scale
  *out-fits* the L4 (71.3s vs 81s).
- The feature still ships on the quota argument: Kaggle TPU hours are a
  separate free budget from GPU hours, verified end-to-end here.

## Bugs found by this verification (all fixed on the branch)

1. `torch.inference_mode` breaks XLA lazy tracing → prediction uses
   `no_grad` on XLA (caught by the XLA:CPU CI before any TPU time).
2. `randperm().split()` batch-index views SIGABRT torch_xla's `index_fill`
   lowering (ModernNCA minibatch fits) → per-chunk index transfers on XLA
   + CI regression test.
3. Unbarriered eval chunks fuse into one XLA program → 50.6 GiB HBM demand
   at a 276k-candidate ModernNCA eval → per-chunk graph barrier, transfer
   still batched.
4. `torch.compile(backend="openxla")` trained inaccurately (ft rmse 0.20 →
   3.18) → refused with a warning on XLA.
5. Thread-per-device concurrent fits crash torch_xla's graph executor →
   in-library TPU sharding rejected for 0.4.0; `TPU_VISIBLE_CHIPS`
   one-process-per-device recipe documented instead.

## Quota spent

Four batch kernels ≈ 3.6 TPU-hours total (wave B's thread probe alone
consumed ~65 min by serializing — the price of a definitive sharding no).
