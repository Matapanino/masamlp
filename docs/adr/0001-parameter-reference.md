# ADR 0001 — Parameter reference: placement, depth, and freshness

Status: accepted (2026-07-03)

## Context

masaMLP exposes ~35 estimator constructor parameters and ~80 architecture
parameters across ten models via `model_params` — including the
depth/width knobs (`n_blocks`, `d`, `hidden_sizes`, ...) users need most —
but until now none of this was documented: the README mentioned
`model_params` once in passing, and the only way to discover a model's knobs
was reading its source. Unknown `model_params` keys surfaced as a bare
`TypeError` from the model constructor.

This ADR records the decisions from the design session that introduced the
parameter documentation. (First ADR in this repo; the convention is imported
from repleafgbm's `docs/adr/`.)

## Decisions

1. **Placement — split.** The complete reference lives in
   `docs/parameters.md`; the README gets a 12-row "Key parameters" table
   (fit essentials + the differentiators: `objective`, `eval_metric`,
   `n_ens`, ...) and links out for the rest.
   *Rejected:* everything in the README (repleafgbm's style — fine for its
   ~10 params, would blow this README past 400 lines); docstrings + a
   generated API-reference site (new infra, off the house style of plain
   markdown docs).

2. **Depth — reference plus cited sizing notes.** Each model gets a
   `Parameter | Default | Meaning` table, enum-valued estimator params get
   value sub-tables, and each model gets a short "Sizing notes" paragraph
   naming the depth/width/regularization/memory knobs. Defaults and variants
   are cited to the source papers / reference packages; where no published
   tuning space exists (grn, lnn), the notes stay qualitative rather than
   inventing recommendations (consistent with the "no re-benchmarking"
   stance). *Rejected:* meaning-only tables (doesn't answer "which knob do I
   touch first"); a full tuning guide with search spaces (large maintenance
   surface, drifts into unbacked claims).

3. **Freshness — a signature-inspection test.** `tests/test_docs_parameters.py`
   walks the model registry and the estimator signatures and asserts every
   parameter appears backticked in `docs/parameters.md`, so adding a
   parameter without documenting it fails CI. Third-party registrations
   (module outside `masamlp.`) are exempt. *Rejected:* review discipline
   only (single-maintainer project — it will be forgotten); generating the
   markdown from code (moves the staleness problem into a descriptions dict
   and can't generate the hand-written sizing notes).

4. **Runtime discoverability.** `build_model` validates `model_params`
   against the builder's signature up front and raises a `ValueError`
   listing the model's valid keys, the shared embedding keys, and a pointer
   to `docs/parameters.md` (builders accepting `**kwargs` are not
   validated). This replaces the bare `TypeError`.

## Consequences

- `docs/parameters.md` is a public-API commitment: parameter renames now
  show up as doc edits and are visible in review.
- Every new model or constructor parameter requires a doc entry in the same
  PR — enforced by CI, not convention.
- The `_NON_PARAM_KEYS` set (`embedding`, `embedding_config`, `out_dim`,
  `n_label_classes`) is the single definition of "plumbing, not a knob",
  shared by the validator and the docs test.
