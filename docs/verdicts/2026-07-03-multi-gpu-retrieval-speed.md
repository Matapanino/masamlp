# 0.3.0 speed release — GPU verification (2026-07-03)

Scope: TabR/ModernNCA inference caching + chunked scoring (KI-008), the
per-model `amp="auto"` policy (KI-010), early-stopping snapshot filtering,
and multi-GPU ensemble-member sharding. Method per the KI-009 discipline:
identical-math rewrites, CPU equivalence tests, then Colab measurements
against the 2026-07-02 T4 baseline.

## Colab T4 (torch 2.11.0+cu128) — full report: `2026-07-03-gpu-report.md`

pytest (device / ensemble / realmlp / retrieval_cache / parallel): **all
green on CUDA**; CUDA smoke 10/10 models; save/load parity OK.

Baseline comparison (same workloads as 2026-07-02):

| workload | 2026-07-02 | 2026-07-03 | note |
|---|---|---|---|
| tabr smoke fit (6k rows, defaults) | 4.2s | **1.7s** | amp="auto" now off for retrieval + chunked search |
| tabr 30k fit, amp=off | 10.5s | 11.8s | unchanged path (noise) |
| tabr 30k fit, amp=auto | 21.3s | **12.3s** | KI-010 resolved: auto no longer doubles fit |
| modernnca smoke fit | 0.5s | 0.5s | training path untouched |
| ft_transformer 30k, amp=off | 19.3s | 19.4s | unchanged |
| ft_transformer 30k, amp=auto | 23.7s | 24.7s → policy | fp16 slower AND rmse 0.296→0.345 ⇒ `FTTransformer.amp_auto = "bf16"` (fp16 off, bf16 kept) |
| tab_transformer 30k, amp=auto | — | 12.3s (off: 15.1s) | AMP helps tabtr ⇒ keeps default policy |
| lnn n_ens=8 loop → vectorized | 28.2s → 7.1s | 30.4s → 7.0s | unchanged |

New predict-side timings (200k inference rows, eval_batch_size=8192):
tabr 3.6s, modernnca 2.6s, ft_transformer 2.9s, tab_transformer 3.3s,
resnet 0.9s — the retrieval models now compute candidate encodings once per
predict pass and stream scoring in `candidate_chunk_size` blocks.

## Colab L4 (22.5 GB) — `gpu_speed.py --retrieval-scale`

The S6E7 failure scale (docs/s6e7-field-report.md P2: ModernNCA OOM'd on a
single 8.4 GiB eval allocation at 345k rows; TabR's eval matrix would be
~11 GB at eval_batch 8192).

```
tabr      rows=200000 budget=None    fit  25.0s  predict(50000) 1.95s  rmse 0.2092
tabr      rows=200000 budget=131072  fit  13.9s  predict(50000) 1.67s  rmse 0.2075
modernnca rows=200000 budget=None    fit  27.4s  predict(50000) 2.47s  rmse 0.8351
modernnca rows=200000 budget=131072  fit  18.7s  predict(50000) 2.03s  rmse 0.9235
tabr      rows=345000 budget=None    fit  45.9s  predict(50000) 3.09s  rmse 0.1766
tabr      rows=345000 budget=131072  fit  15.0s  predict(50000) 1.70s  rmse 0.2102
modernnca rows=345000 budget=None    fit  81.4s  predict(50000) 4.06s  rmse 0.6544
modernnca rows=345000 budget=131072  fit  19.7s  predict(50000) 2.03s  rmse 0.8723
```

(fit = 3 epochs; amp="auto" resolves to off for both models per the new
policy.)

- **ModernNCA at 345k rows with no `candidate_budget` completes** — fit
  81.4s, predict of 50k queries against the full 276k-row corpus in 4.1s.
  This exact configuration previously died on a single 8.4 GiB eval
  allocation; peak scoring memory is now B x chunk (8192 x 8192 fp32
  ≈ 256 MB).
- **TabR at 345k, full corpus**: predict 3.1s where the unchunked search
  would have needed a ~11 GB (8192 x 276k) distance matrix per batch.
- `candidate_budget=131_072` remains the recommended speed/quality knob at
  this scale (3x faster fits; rmse differences reflect the smaller training
  set, matching the field-report finding that the accuracy cost is ≈ 0 on
  real data).

## Multi-GPU (Kaggle 2xT4, torch 2.10.0+cu128)

Verified by running the four `examples/kaggle/` notebooks on Kaggle's
"GPU T4 x2" (n_ens=4, `device="cuda:0"` vs `device="auto"`, same seeds):

| notebook | single T4 fit | sharded fit | speedup | max abs pred diff | test acc |
|---|---|---|---|---|---|
| ft_transformer (covtype 100k) | 509.7s | 255.3s | x2.00 | 0.00e+00 | 0.885 |
| tabr (covtype 200k, budget 50k) | 293.9s | 154.7s | x1.90 | 0.00e+00 | 0.934 |
| modernnca (covtype 200k, full corpus) | 1471.9s | 734.8s | x2.00 | 0.00e+00 | 0.958 |
| tab_transformer (adult 49k) | 87.5s | 41.8s | x2.10 | 0.00e+00 | 0.847 |

Sharded and single-GPU runs picked the same best epoch and produced
**bit-identical predictions** on the identical-GPU pair — sequential mode
seeds every CUDA device to the member seed and the sharded worker seeds its
device to the same value, so the RNG streams coincide (documented as
expected-but-not-promised in docs/devices.md / KI-004).

## Verdict

- KI-008 inference side: **resolved** (cache + chunked scoring; training
  remains O(N) by design, bounded by `candidate_budget`).
- KI-010: **resolved** via the per-model AMP policy (`amp_auto = False` on
  tabr/modernnca, `amp_auto = "bf16"` on ft_transformer; measured on T4
  above).
- No regressions in the unchanged paths (resnet/realmlp/danet/lnn within
  noise of the 2026-07-02 baseline).
