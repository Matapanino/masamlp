# ADR 0004 — TPU multi-device: no in-library multiprocessing; wait for TorchTPU

Status: accepted (2026-07-12)

## Context

0.4.0 shipped single-device TPU support. The natural next step — the TPU
analog of 0.3.0's multi-GPU ensemble-member sharding — is blocked by a
measured fact (research/tpu-xla.md §8.4): torch_xla's lazy graph executor is
process-global and not thread-safe across devices. Four concurrent
single-device fits from worker threads on a v5e-8 crashed inside
`XLAGraphExecutor::CollectSyncTensors` or degraded ~250x. The CUDA sharding
architecture (one worker thread per device, `core/parallel.py`) therefore
cannot be ported as-is.

The remaining in-library option is multiprocessing (`xmp.spawn` or
spawn-per-chip subprocesses with `TPU_VISIBLE_CHIPS`), which collides with a
core design rule: masaMLP has no DataLoader workers and no multiprocessing,
deliberately. Meanwhile Google's **TorchTPU** (announced April 2026, preview;
public repo planned within 2026) is the declared successor to PyTorch/XLA —
"once public it will replace PyTorch/XLA" — with a new runtime whose
threading model is unknown but freshly designed.

## Decision

**No in-library TPU multi-device path in 0.5.0 — neither `xmp.spawn` nor
managed subprocesses. The full-board story remains the documented
one-process-per-chip `TPU_VISIBLE_CHIPS` recipe. Revisit when TorchTPU is
public (or torch_xla's executor becomes thread-safe), with thread-based
member sharding — the proven CUDA architecture — as the preferred shape.**

Reasons, in order of weight:

1. **The audience runs notebooks.** Kaggle/Colab notebooks are the primary
   TPU environment for this library. `xmp.spawn` requires a spawn-safe
   `__main__` and re-imports the calling module in children — hostile to
   notebook cells and to sklearn-style interactive use. The failure modes
   (silent hangs, pickling errors deep in a fit) would land on exactly the
   users least equipped to debug them.
2. **Multiprocessing breaks the differentiators.** Custom objectives and
   metrics are arbitrary user callables (lambdas, closures over local
   state); they do not reliably pickle. A sharding mode that rejects — or
   worse, sometimes rejects — the library's headline features is a bad
   trade for a speedup on one accelerator family.
3. **Poor amortization at tabular scale.** Per-process runtime init + XLA
   recompilation (the in-process compile cache does not cross processes)
   costs tens of seconds against member fits measured at 35–70s. The
   recipe's manual orchestration pays the same tax but makes the cost
   visible and opt-in, instead of a library promise that under-delivers.
4. **The foundation is being replaced.** torch_xla is in maintenance
   descent; TorchTPU is expected public within 2026. Building a
   multiprocessing subsystem on the outgoing runtime, against its known
   executor bug, is effort spent where the successor may simply not have
   the constraint. Keeping the XLA surface small (ADR 0002 §2's containment
   argument) is what makes the eventual migration cheap.

*Rejected:* `xmp.spawn` opt-in (reasons 1–2); library-managed
`TPU_VISIBLE_CHIPS` subprocess pool (reason 2–3 — it is the recipe with the
costs hidden); documenting nothing (the recipe demonstrably uses all 8 chips
today, validated in wave A).

## Consequences

- `docs/devices.md` keeps the recipe as the supported multi-chip pattern;
  `resolve_device_plan` stays CUDA-only.
- Roadmap: "in-library TPU member sharding" carries an explicit re-evaluate
  trigger — TorchTPU public release or a torch_xla release note fixing
  cross-device executor thread safety — instead of a version target.
- The `xla:N` device vocabulary (accepted since 0.4.0) already lets a user
  pin a fit to a specific visible device, which the recipe composes with.
