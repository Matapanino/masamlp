# TPU / torch_xla research notes (2026-07)

Survey backing ADR 0002 (TPU support) and ADR 0003 (XLA execution
strategy). Conclusions only in `docs/devices.md`; sources at the bottom.
Items still marked *(wave A)* are confirmed empirically in the first
Kaggle TPU session and this file is updated with the findings.

## 1. torch_xla state of the world

- Current stable is **torch_xla 2.8.0** (nightlies 2.9), wheels for
  Python 3.10–3.13, strictly minor-version-paired with torch. [1][2]
- **TorchTPU** (announced April 2026, in preview) is Google's successor
  to PyTorch/XLA — "once public it will replace PyTorch/XLA", with a
  public repo planned through 2026. [3][4] Consequence for masaMLP: ship
  0.4.0 on torch_xla (the only GA path), and keep the integration surface
  small — a handful of `device.type == "xla"` gates in `core/` — so a
  later TorchTPU migration is contained. Its stated focus (fewer
  recompilations from dynamic shapes, precompiled kernels) validates the
  static-shape work in ADR 0003 rather than replacing it.
- Execution modes: **lazy tensor tracing is still the default**; eager
  mode (`torch_xla.experimental.eager_mode(True)`) plus
  `torch_xla.compile` is usability-focused and experimental; long term,
  `torch.compile` is intended as the single compile API. [5][6]
  masaMLP v1 therefore uses **lazy mode + one `torch_xla.sync()` per
  optimizer step**, and maps the existing `compile=True` flag to
  `torch.compile(backend="openxla")` as the experimental alternative.
  `torch_xla.sync()` is the modern name for the step barrier
  (`xm.mark_step` is the legacy spelling — keep a shim if Kaggle's image
  ships an older torch_xla, see §6). [7]

## 2. Recompilation: the failure mode that matters

XLA compiles one executable per (graph, input shapes) signature;"graph
compilations in XLA are pretty expensive". Documented recompile sources
[8] map 1:1 onto the masaMLP audit in ADR 0003:

| Source (docs) | masaMLP instance | Fix |
|---|---|---|
| Input shape variation | last train/eval batch (`n % batch_size`) | none needed — a finite 2-shape set, compiled once each |
| Data-dependent output shapes (`nonzero` etc.) | `tabr.py:136`, `modernnca.py:103` | static-shape mask rewrites (ADR 0003 §2) |
| Python scalars baked as graph constants | per-step lr (`coslog4`), flat_cos weight decay, `ScheduledDropout` p | tensor-valued schedules; wrapping lr in a Tensor is the same fix the official `torch.compile` optimizer recipe prescribes [9] |
| Host round-trips (`.item()`, `.size()` on dynamic tensors) | per-chunk `.cpu()` in eval; epoch-loss `float()` | keep the single per-epoch sync; batch eval transfers |

Debugging workflow: `torch_xla.debug.metrics` —
`met.metrics_report()` / counters (CompileTime, TransferFromDeviceTime,
`aten::*` fallback counters). Every TPU verdict run records this per
model; acceptance is *zero unexpected recompiles and zero aten fallbacks
on the hot path*, not just wall-clock. [10]

## 3. PJRT topology on TPU v3-8: sharding verdict

From the PJRT runtime docs [11]:

- v2/v3: "distributed workloads always run multithreaded … only one
  process may open a TPU chip at a time"; default topology on a v3-8
  host is **4 processes × 2 threads** (one thread per core).
- **A single process cannot address all 8 cores** — one process opens at
  most one chip (2 cores). `TPU_PROCESS_BOUNDS` / `TPU_VISIBLE_CHIPS`
  select chips per process.
- The global torch RNG is not thread-safe across replica threads — the
  same constraint `core/parallel.py` already handles with
  `seed_scope="device"`.

Consequences (resolves ADR 0002 §1's condition):

- **Full 8-core member sharding inside one `fit()` is impossible without
  multiprocessing** → deferred to 0.5.0, as pre-agreed.
- Thread-per-device *within* one chip (2 cores) is PJRT's native mode, so
  a 2-way in-process shard is technically open — but using 2 of 8 cores
  is not a compelling ship. *(wave A: probe what a single Kaggle process
  actually enumerates.)*
- The honest full-machine story for 0.4.0 is a **documented recipe**:
  run one Python process per chip with `TPU_VISIBLE_CHIPS=<i>`
  (user-orchestrated HPO/ensembling, e.g. 4 concurrent fits), which uses
  the whole board without any library multiprocessing.

## 4. Mixed precision on TPU

- `torch.autocast("xla", dtype=torch.bfloat16)` is the sanctioned form —
  the trainer's existing `torch.autocast(device.type, ...)` is already
  correct on an `xla` device. [12]
- "Since TPUs use bfloat16 mixed precision, gradient scaling is not
  necessary" — no GradScaler on XLA, ever (fp16 is not offered). [12]
- `torch_xla.amp.syncfree` optimizers exist to remove device–host syncs
  in scaler-style flows; with bf16 (no scaler) they are an optimization
  to *measure*, not a requirement. *(wave B: AdamW vs syncfree.AdamW.)*
- `XLA_USE_BF16=1` (global fp32→bf16 rewrite) is legacy/deprecated in
  ecosystem guidance and would corrupt loss/metric precision — rejected
  in ADR 0002 §5. [13]

## 5. Hardware background (why the T4 baseline is the right bar)

TPU v3-8 = 4 chips × 2 cores; ~420 bf16 TFLOPS per 4-chip board
(~50–60 TFLOPS per core), 32 GiB HBM per chip, one 128×128 bf16 MXU per
core (16K MACs/cycle). [14][15] A single v3 core is therefore
**T4-class** (T4 ≈ 65 fp16 TFLOPS) and well under an L4 — hence the
success criterion "≥ T4 parity per core on the matmul-heavy six", and
hence why bf16 (MXU-native) is the default and fp32-only TPU support
would be pointless. Background papers: the original TPU analysis
(Jouppi et al. 2017 [16]), the v2/v3 design retrospective (Norrie et
al. 2021 [17]), TPU v4 (Jouppi et al. 2023 [18]), and the bf16 training
study (Kalamkar et al. 2019 [19]).

## 6. Kaggle TPU environment (experiment vehicle)

- Kernel accelerator `Tpu1VmV38` = TPU VM v3-8, free weekly quota
  separate from the 30 h/week GPU quota; batch sessions cap at ~9 h.
- pytorch/xla's own Kaggle notebooks historically note "preinstalled
  Python 3.10 and PT/XLA 2.1" [20] — the image may lag far behind 2.8.
  *(wave A: probe `torch.__version__`/`torch_xla.__version__` first;
  then either (a) run against the preinstalled pair with an
  `xm.mark_step` shim, or (b) pip-install a matched modern
  torch/torch_xla pair in the kernel — decide on the probe results,
  prefer (b) only if (a)'s version is too old for `torch.autocast("xla")`.)*
- No tabular DL library advertises first-class TPU support today
  (pytorch_tabular inherits at most nominal Lightning TPU strategies;
  pytabkit is CPU/CUDA) — the feature is a real differentiator.

## 7. Wave A findings (Kaggle, 2026-07-10 — answers the probe list)

1. **Kaggle's `Tpu1VmV38` accelerator now provisions a TPU v5e-8**
   (`TPU_ACCELERATOR_TYPE=v5litepod-8`): 8 single-core v5e chips
   (~197 bf16 TFLOPS each — L4-class+, not the v3 core assumed in §5),
   224-vCPU host, python 3.12.13, torch 2.8.0+cpu, torch_xla 2.8.0
   (`torch_xla.sync`/`manual_seed` present — the legacy shim is a dormant
   safety net).
2. **One process addresses all 8 devices** (`xla:0..7`,
   `addressable_device_count=8`) — the §3 one-process-per-chip constraint
   is v2/v3-specific and moot on v5e. `TPU_VISIBLE_CHIPS=0` correctly
   restricts to one device. In-library thread-per-device member sharding
   is therefore *re-opened*; wave B carries the thread-concurrency probe
   that decides it.
3. Zoo (all 10 models, fp32): trained and predicted correctly; save →
   CPU-load prediction parity **bitwise exact** for all ten; **3 compiles
   per model**; the only aten fallback is `aten::_local_scalar_dense` ×
   n_epochs — the designed one-host-sync-per-epoch. Zero unexpected
   recompiles or fallbacks: the static-shape rewrites (ADR 0003 §2) hold
   on hardware.
4. **Scalar-lifting confirmed on TPU**: coslog4 lr + flat_cos wd +
   scheduled dropout, 2 epochs → 3 compiles vs 8 epochs → 2 compiles.
   Per-step Python-float schedules do NOT recompile in lazy mode; no
   tensor-lr machinery needed (dropout's op-attribute `p` was the one
   real constant-bake, fixed by the tensor-p ScheduledDropout).
5. `torch.inference_mode` breaks XLA lazy tracing ("Cannot set
   version_counter for inference tensor") — caught by the XLA:CPU CI
   before any TPU time was spent; prediction uses `no_grad` on XLA.
6. bf16 at toy scale (resnet 50k, 3 epochs incl. compile) was not faster
   than fp32 (22.0s vs 17.3s) — expected at compile-dominated sizes;
   wave B measures the real matrix (KI-010 re-measure included).

## 8. Wave B findings (Kaggle v5e-8, 2026-07-10 — the decisions they forced)

1. **torch_xla `index_fill` crashes (SIGABRT) on split-view indices**: batch
   indices produced as `randperm(n).to(device).split(batch_size)` views
   carry a tuple-typed XLA IR shape that `IndexFillOp`'s `EnsureRank1`
   cannot read (`xla::Shape::array_state` check failure). Hit by
   ModernNCA's key sampling in every minibatch fit; invisible to wave A
   and the first CI suite because both were full-batch. Fix: on XLA the
   trainer transfers each permutation chunk separately (clean device-data
   nodes); a minibatch retrieval test now guards it in CI.
2. **bf16 default, and KI-010 is CUDA-scoped** — *numbers corrected by
   wave C*: wave B's "amp=auto" retrieval rows predated the device-keyed
   policy and were actually fp32 reruns riding a warm compile cache (see
   §9.1 — the benchmark-ordering trap). True cold-cache bf16 (wave C):
   ModernNCA fit −28%, TabR@345k fit −9%, rmse equivalent; prediction is
   amp-independent (autocast wraps the training step only). `amp_auto`
   became device-keyed (`{"cuda": False}` on retrieval models) — still
   the right call, on the corrected evidence.
3. **openxla dynamo backend miscompiles training**: ft_transformer lazy
   49.2s / rmse 0.2012 vs `compile=True` 29.1s / **rmse 3.18**.
   `compile=True` on XLA now warns and stays lazy.
4. **Thread-per-device sharding is dead**: 4 concurrent single-device fits
   from threads → 2 crashed inside `XLAGraphExecutor::CollectSyncTensors`
   / `XLATensor::shape` ("Check failed: tensor_data"), 1 degraded from
   ~15s to 3785s. torch_xla's lazy executor is process-global, not
   per-device-thread-safe. In-library TPU sharding is off the table for
   0.4.0 (CUDA sharding unaffected); the full-board story is the
   one-process-per-device `TPU_VISIBLE_CHIPS` recipe (validated in wave A).
5. **Speed vs the L4 verdict (50k rows, verdict configs, bf16)**: resnet
   34.8s (21.8s at batch 8192), realmlp-TD 60.3s, ft_transformer 38.4s
   (L4: 24.7s), tabr 17.1s (L4: 12.3s), tab_transformer 69.0s (L4: 12.3s —
   the one clearly dispatch-bound loser at batch 1024; 34.0s at 8192).
   Large batches reclaim throughput but change convergence (rmse column) —
   exactly why batch defaults stay device-independent and the big-batch
   story is documentation. Host-CPU context: the 224-vCPU host beats the
   single chip on realmlp at batch 1024 (34.1s) — per-step dispatch, not
   FLOPs, is the tax at tabular batch sizes.
6. In-process compile caching is real: repeat fits of the same config
   compile 0 times and run 2-3x faster than the first fit.

## 9. Wave C findings (Kaggle v5e-8, 2026-07-10 — post-fix re-measurement)

1. **The benchmark-ordering trap.** Same-process sequential runs share the
   XLA compile cache, and fp32 *prediction* graphs are identical whatever
   `amp` is — so any "amp=auto" row run after an fp32 row inherits warm
   predict graphs (and, before the device-keyed policy landed, warm fit
   graphs too, because retrieval "auto" still meant fp32). Wave B's
   headline retrieval bf16 wins were this artifact; §8.2 is corrected.
   Rule for future TPU verdicts: report cold-cache numbers, or randomize/
   isolate run order per cell.
2. **True bf16 (cold cache)**: modernnca 50k fit 35.5s vs fp32 49.1s
   (−28%), rmse 0.6830 vs 0.6833; tabr 50k fit 41.0s vs 40.9s (nil),
   345k fit 139.9s vs 153.1s (−9%), rmse equivalent. Policy unchanged
   (bf16 on XLA), justification corrected.
3. **The minibatch index fix holds on TPU**: modernnca trains through the
   former SIGABRT path (§8.1) at 50k and 345k.
4. **Batched eval transfers need per-chunk graph barriers on XLA**:
   accumulating chunk outputs as pending lazy IR fused every eval chunk
   into one program — ModernNCA's streamed eval at a 276k-candidate
   corpus demanded 50.6 GiB of 15.75 GiB HBM (`RESOURCE_EXHAUSTED`,
   measured). `predict_transformed` now cuts the graph per chunk and
   still transfers once. (Wave D re-verifies 345k eval end to end.)
5. batch 8192 rows keep showing the honest convergence cost (modernnca
   rmse 0.68 → 2.05 at fixed epochs) — the device-independent batch
   default (ADR 0002 §4) keeps protecting result quality.
6. **Wave D (per-cell cold processes) confirmed the barrier fix**:
   modernnca 345k fit 71.3s / predict 20.4s with no OOM — the fit now
   *beats* the L4's 81s. The barrier costs TabR's chunked eval its
   cross-chunk fusion (predict 47.8s → 85.6s), accepted: correctness of
   ModernNCA eval over speed of an already-documented slow path.

## 10. Colab v5e-1 verification + the torch_xla 2.9 matmul-precision change
## (2026-07-12, 0.5.0 cycle)

1. **Colab TPU v5e-1 works end to end** (billable CLI pool; ~20 min
   session): the runtime ships python 3.12.13 + torch 2.9.0+cpu +
   torch_xla 2.9.0 preinstalled (no install line needed, unlike the docs'
   Cloud-VM guidance), 24-vCPU host, `TPU_ACCELERATOR_TYPE=v5e-1`.
   `resolve_device("tpu"/"xla"/"auto")` → xla:0; resnet fit/predict,
   save→CPU-load roundtrip, `xla_fuse_steps=8` fit and `amp_predict=True`
   all ran (branch v0.5.0-tpu).
2. **torch_xla 2.9 changed the TPU fp32 matmul default** from full
   precision to one-pass bf16. Measured on v5e-1 (512×512, N(0,1),
   CPU-vs-TPU max|diff|): as-shipped 2.6e-1; after
   `torch_xla.backends.set_mat_mul_precision("high")` 1.4e-3 (bf16
   3-pass); `"highest"` 4.6e-5 (fp32). Model-level: fp32-trained resnet
   predictions deviated from their CPU-loaded copy by ~3e-2 — versus the
   **bitwise** parity 0.4.0 measured on torch_xla 2.8 (wave A §7.3), which
   was therefore a 2.8-default artifact, not a durable contract.
3. **The precision knob must be set before the first compilation in the
   process**: later calls are silently ignored for already-compiled graph
   shapes (the compile cache keys on the graph, not the precision setting
   — torch_xla itself warns about this). An in-session probe that flipped
   the setting after a warm-up matmul measured no change; per-fresh-process
   probes show the real effect. Same family as the §9.1 benchmark-ordering
   trap: on XLA, *process state at first compile wins*.
4. `torch.set_float32_matmul_precision` is not wired to the XLA backend
   (reports "highest" while the TPU runs bf16 matmuls). Docs now carry the
   `torch_xla.backends` recipe; masaMLP does not override the platform
   default (no hidden global state).
5. RNG placement: the XLA device RNG seed advances per graph *execution*,
   so `xla_fuse_steps` (barrier placement) selects a different — equally
   seeded-deterministic — stream for training-time device RNG (dropout,
   retrieval sampling). Found by the XLA:CPU CI parity test (K=4 vs K=1
   diverged ~2% on realmlp with scheduled dropout; RNG-free lnn is
   K-invariant at 1e-6). Contract documented as batch_size-like: same
   seed + same K ⇒ same model.

## Sources

1. https://github.com/pytorch/xla/releases
2. https://pypi.org/project/torch-xla/
3. https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/
4. https://techinformed.com/google-moves-to-make-tpus-feel-native-to-pytorch-as-it-targets-nvidias-cuda-advantage/
5. https://docs.pytorch.org/xla/master/learn/eager.html
6. https://github.com/pytorch/xla/issues/7253
7. https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html
8. https://docs.pytorch.org/xla/release/r2.7/perf/recompilation.html
9. https://docs.pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html
10. https://docs.pytorch.org/xla/master/learn/troubleshoot.html
11. https://docs.pytorch.org/xla/master/runtime.html
12. https://docs.pytorch.org/xla/master/perf/amp.html
13. https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/appnotes/torch-neuronx/migration-from-xla-downcast-bf16.html
14. https://docs.cloud.google.com/tpu/docs/v3
15. https://jax-ml.github.io/scaling-book/tpus/
16. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit", ISCA 2017. https://arxiv.org/abs/1704.04760
17. Norrie et al., "The Design Process for Google's Training Chips: TPUv2 and TPUv3", IEEE Micro 2021.
18. Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning", ISCA 2023. https://arxiv.org/abs/2304.01433
19. Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", 2019. https://arxiv.org/abs/1905.12322
20. https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb
