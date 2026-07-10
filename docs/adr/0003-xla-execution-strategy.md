# ADR 0003 — XLA execution strategy: static shapes, tensor schedules, sync points

Status: accepted (2026-07-11); items marked *(confirm)* are validated by the
research spike / first TPU session before implementation is considered done.

## Context

XLA compiles one graph per tensor-shape signature and replays it; anything
that changes shapes or bakes fresh Python scalars into the graph forces a
recompilation, and anything that moves data device→host forces a blocking
sync. The audit of the current trainer/models found exactly four hazards —
everything else in the no-DataLoader design is already XLA-shaped:

1. `tabr._search_topk` builds exclusion rows with
   `(...).nonzero(as_tuple=True)[0]` — output shape is data-dependent
   (tabr.py:136). On XLA this is a per-batch recompile and/or host fallback;
   on CUDA `nonzero` already forces a host sync for the output size.
2. `modernnca` samples its candidate pool via `pool_mask.nonzero(...)`
   (modernnca.py:103) — same class of hazard, per training step.
3. Per-step **Python-scalar schedules** rewrite graph constants every step:
   the trainer's `coslog4`/`cosine`-per-step lr writes `group["lr"]`, the
   flat_cos weight-decay schedule writes `group["weight_decay"]`, and
   RealMLP's `ScheduledDropout` mutates the dropout probability. In lazy
   XLA each becomes a fresh graph per optimizer step — the recompile worst
   case.
4. Host syncs inside loops: `predict_transformed` calls `.cpu()` per eval
   chunk; the epoch loop's `float(epoch_loss)` is fine (one sync per epoch,
   by design) but in lazy mode there must be a step boundary or the whole
   epoch fuses into one unbounded graph.

Batch shapes themselves are already static: `randperm(n).split(batch_size)`
yields exactly `{batch_size, n % batch_size}` — two shapes, two compiles,
cached for every subsequent epoch. Same for eval chunks. `lnn`'s
`for _ in range(n_steps)` is a static unroll; GhostBatchNorm chunk counts
are functions of batch shape only; sparsemax/entmax sort compiles fine
(speed is the models' problem, per ADR 0002 §6).

## Decisions

1. **No padding, no drop_last.** The two-shape batch set is left as is;
   two compilations are cheaper than any semantics-preserving padding
   scheme, and dropping the tail batch would change training semantics.
   (If a future need for one-shape batching appears, the zero-weight
   padding trick — the trainer's weighted reduction `(loss*w).sum()/w.sum()`
   makes padded rows exactly inert — is the sanctioned mechanism; noted,
   not implemented.)

2. **Static-shape rewrites for the two `nonzero()` sites, on all devices.**
   Both rewrite to mask arithmetic with data-independent shapes (e.g.
   scatter `+inf` into excluded distance columns instead of gathering
   exclusion row indices; sample the ModernNCA pool by weighted choice on a
   masked distribution instead of materializing indices). One code path —
   no `if device.type == "xla"` forks in model code (core rule: models are
   device-free). Prediction-level parity with 0.3.x is expected at the
   couple-of-ulp level, release-noted like the 0.3.0 retrieval change.
   *Rejected:* XLA-only forks (double maintenance, models learn about
   devices).

3. **Only op-attribute scalars need tensorizing; arithmetic scalars are
   lifted.** torch_xla's lazy mode lifts arithmetic scalars (optimizer lr,
   weight decay, EMA decay, bias corrections) into graph *parameters*, so
   per-step Python-float schedules do not recompile — **measured on TPU:
   compile count is epoch-independent under coslog4 + flat_cos wd
   (wave A, research §7.4)**. The one true constant-bake was
   `F.dropout`'s `p` (an op attribute, not an operand); scheduled dropout
   gets a tensor-probability implementation (bernoulli mask from
   `torch.rand_like` compared to a device tensor) selected transparently —
   same math, same seeds discipline, no per-step graph constants.
   *Rejected:* quantizing schedules to per-epoch (changes RealMLP-TD
   semantics); leaving them and eating the recompiles (kills the target
   models' TPU speed).

4. **Sync points.** Lazy mode: one `torch_xla.sync()` (mark_step) per
   optimizer step, placed in the trainer's step loop under an XLA gate —
   mirrors the existing "one sync per epoch" comment culture; the per-epoch
   `float(epoch_loss)` stays the only host round-trip. Eval:
   `predict_transformed` accumulates per-chunk outputs on-device and
   transfers once per eval set *(confirm marginal value on TPU; keep the
   current per-chunk `.cpu()` if measurement says it does not matter)*.
   Confirmed by research: lazy tracing is still torch_xla's default mode,
   `torch_xla.sync()` is the current step-barrier name (`xm.mark_step`
   legacy shim kept in case Kaggle's image lags — research §1/§6), eager
   mode stays experimental, and `torch.compile(backend="openxla")` is the
   forward-looking alternative behind the existing `compile=True` flag.

5. **GradScaler bypassed; autocast targets `"xla"`.** `resolve_amp` returns
   bf16 on XLA under `amp="auto"` (ADR 0002 §5); the trainer constructs no
   GradScaler on XLA (bf16 needs none; fp16 is not offered on TPU).

6. **Seeding and RNG.** `seed_everything` gains the XLA generator
   (torch_xla RNG seed) under the same `seed_scope` contract; shuffling
   permutations remain CPU-drawn from the seeded generator (already
   device-independent by design). Same seed + same device ⇒ same result
   extends to XLA.

7. **Serialization normalizes to CPU at save.** `save()` moves state_dict
   tensors to CPU before `torch.save` on every device (today's save writes
   device-resident tensors; XLA tensors must not be pickled raw, and
   CPU-normalized archives are healthier for CUDA too). Loading is already
   `map_location="cpu"`; prediction after load continues to work with the
   stored `transform_name` only.

8. **Sharding: deferred (resolved per ADR 0002 §1).** PJRT on v2/v3 caps
   a process at one chip (2 cores), so in-library member sharding cannot
   reach the full board without multiprocessing — deferred to 0.5.0.
   0.4.0 documents the `TPU_VISIBLE_CHIPS` one-process-per-chip recipe
   instead; wave A still records what a single Kaggle process enumerates.

## Consequences

- The trainer gains a small number of `device.type == "xla"` gates (sync
  placement, scaler bypass, seeding) — same pattern as the existing
  cpu/mps gates; models gain none.
- The tensor-schedule work incidentally removes per-step Python-float
  churn for CUDA too (harmless there, measured in the benchmark).
- `benchmarks/` and the Kaggle kernels report `met.metrics_report()`
  recompile/fallback counters alongside wall-clock, so "zero unexpected
  recompiles, zero aten fallbacks on the hot path" is a testable
  acceptance criterion per model, not a vibe.
