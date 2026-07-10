# ADR 0002 — TPU support via torch_xla: scope, API, and delivery

Status: accepted (2026-07-11)

## Context

masaMLP runs on CPU, CUDA (single and multi-GPU), and MPS. TPUs are the one
mainstream accelerator missing, and they matter for exactly this library's
audience: Kaggle grants a **separate** weekly TPU quota (TPU VM v3-8) on top
of the 30 h/week GPU quota, so TPU support is free extra accelerator time
for competition users even where it only matches a T4. The architecture is
already unusually XLA-friendly — no DataLoader, tensors are moved to the
device once, minibatches are index slices, one host sync per epoch — so the
gap is device plumbing plus XLA-specific hazards, not a redesign.

Decisions from the 2026-07-10/11 design interview (grilling session):

## Decisions

1. **Goal — single-device speed first, sharding conditional.** v1 optimizes
   the single-XLA-device path. Success criterion: the matmul-heavy models
   (`resnet`, `realmlp`, `ft_transformer`, `tab_transformer`, `tabr`,
   `modernnca`) at ~345k rows train at least as fast on one TPU device as
   the measured T4 baselines (docs/verdicts/2026-07-02). Ensemble-member
   sharding across a multi-device TPU (the TPU analog of the 0.3.0
   multi-GPU sharding) is conditional on PJRT letting one process drive
   all devices from threads. The docs' one-process-per-chip rule turned
   out to be v2/v3-specific — **Kaggle's TPU runtime is now a v5e-8,
   where one process addresses all 8 devices** (wave A, research §7) —
   so the condition is decided empirically by wave B's thread-per-device
   concurrency probe: works-and-scales ⇒ sharding ships as a
   `core/parallel.py` extension; otherwise it defers to 0.5.0 and 0.4.0
   documents the one-process-per-device `TPU_VISIBLE_CHIPS` recipe.
   *Rejected:* multi-core as the v1 headline (the single-device path must
   land first regardless); inference-only TPU support (forfeits the
   training quota value).

2. **Backend — torch_xla, nothing else.** Models stay pure PyTorch
   `nn.Module`s; torch_xla is the only backend that makes TPU a *device*
   rather than a rewrite. The lazy-tensor vs `torch.compile(backend=
   "openxla")` execution mode is decided by experiment, not doctrine; the
   trainer's `compile=True` maps to the openxla backend on XLA devices.
   *Rejected:* torchax / JAX bridges (experimental, second dependency
   stack); a JAX port (a fork, not a feature).

3. **API — `"xla"` canonical, `"tpu"` alias, `"auto"` prefers TPU.**
   `resolve_device` accepts `"xla"` (any PJRT backend, including XLA:CPU —
   what CI uses) and `"tpu"` (asserts the PJRT device really is a TPU;
   fail-fast instead of silently training on CPU). `"auto"` resolves
   tpu > cuda > mps > cpu; TPU detection is guarded (torch_xla importable
   AND a TPU environment marker) so non-TPU environments pay nothing.
   *Rejected:* `"tpu"`-only vocabulary (would make the CI device a hidden
   feature); keeping `"auto"` TPU-blind (silent CPU training on a TPU VM is
   the worst outcome).

4. **Semantics are device-independent; numerics are not.** `batch_size=
   "auto"` resolves identically on every device (full-batch ≤ 4096, else
   1024) — no TPU-specific convergence behavior. The TPU large-batch story
   is documentation plus benchmark tables (auto vs tuned), never a silent
   default switch. The existing promise stands: same seed + same device =
   same result; across devices results are close, not bitwise.
   *Rejected:* device-aware auto batch sizes (same code + same seed would
   change model quality across devices).

5. **AMP — bf16 autocast by default, per-model policy respected.** On XLA,
   `amp="auto"` enables bf16 autocast exactly like CUDA, honoring
   `amp_auto` (`tabr`/`modernnca` start fp32; their `amp_auto=False` encodes
   T4 measurements, so TPU experiments re-measure and `amp_auto` may become
   device-type-aware if the numbers disagree). GradScaler is bypassed on XLA
   (bf16 needs no loss scaling). *Rejected:* fp32 default (forfeits the
   MXU); `XLA_USE_BF16`-style global bf16 (corrupts loss/metric precision,
   against the metric-fidelity contract).

6. **Coverage — all ten models must work; six get speed targets.** Everything
   trains and predicts correctly on XLA (enforced by CI and a TPU zoo run);
   speed targets apply to the matmul-heavy six. The entmax/sort-dominated
   models (`danet`, `gandalf`, `grn`) and `lnn` are documented best-effort.
   `ens_mode="vectorized"` + XLA raises (torch.func vmap over XLA is
   unvalidated; silently falling back to loop mode would misreport what ran).
   *Rejected:* gating `device="xla"` to the fast six (splits the API); speed
   targets for all ten (sort performance on TPU is not worth the tuning
   budget).

7. **Testing — PR-level XLA:CPU smoke.** A dedicated CI job installs a
   pinned torch/torch_xla pair (pinned in the workflow, not pyproject),
   sets `PJRT_DEVICE=CPU`, and runs the differentiator gates
   (sample_weight / custom objective / custom metric), a tiny model zoo,
   and a save/load roundtrip on `device="xla"`. XLA tests auto-skip where
   torch_xla is absent (macOS dev machines). *Rejected:* nightly-only
   (broken PRs could merge); real-TPU-only verification (no regression
   safety).

8. **Packaging — 0.4.0, no `[tpu]` extra.** torch_xla is lazily imported;
   a missing install raises with the exact install line. No pip extra:
   torch↔torch_xla minor versions are strictly coupled and an extra that
   drags torch around would break Kaggle/Colab preinstalled images. Docs
   carry per-environment install guidance; TPU support is labeled
   experimental in docs, not gated in code. *Rejected:* `masamlp[tpu]`
   extra (footgun), rc prerelease (overweight for an experimental-labeled
   feature).

9. **Experiments — Kaggle weekly TPU quota, private kernels.** Verification
   and benchmarking run as private Kaggle kernels (`Tpu1VmV38`) installing
   the pushed branch by git pin, one self-contained batch kernel per
   experiment wave, results retrieved via `kaggle kernels output` and
   recorded under `docs/verdicts/`. Baselines: the existing T4/L4 verdicts
   plus the TPU VM's own CPU. *Rejected:* billable Colab TPU sessions (the
   user directed experiments to the free Kaggle quota; Colab's CLI pool is
   v5e-1/v6e-1 single-chip anyway, which cannot exercise the sharding
   stretch goal).

10. **Research is a committed artifact.** The survey backing these decisions
    (torch_xla API state, PJRT threading model, XLA performance practice,
    TPU architecture background) lives at `docs/research/tpu-xla.md` with
    sources; ADRs cite it and `docs/devices.md` gets conclusions only.

## Consequences

- All TPU work lands in `core/` (device.py, trainer.py, serialization.py,
  parallel.py if sharding confirms) — models stay device-free, per the core
  architectural rule. The two data-dependent-shape ops in model code
  (`tabr`/`modernnca` `nonzero()`) are rewritten to static-shape equivalents
  on **all** devices (see ADR 0003), not forked per device.
- Even at T4 parity the feature ships: separate Kaggle TPU quota means more
  free accelerator hours for the target audience.
- `docs/parameters.md` device/amp rows change (CI-enforced), and
  reproducibility docs gain the XLA paragraph.
