# Field report: masamlp 0.1.0 in production (Kaggle Playground S6E7) — bugs & upgrade requests

**Audience: the agent doing the next masamlp version-up.** This is a from-the-field report from
shipping all 10 architectures on S6E7 (690,088 rows, 3-class extreme-imbalance, balanced
accuracy; 7 numerics + 6 low-card string cats, injected NaN everywhere). Evidence, repro
commands and the full experiment ledger live in the competition workspace:
`~/competition/s6e7/mlp/findings/FINDINGS.md` (+ per-arch `~/competition/s6e7/<arch>/findings/`).
Everything below was observed with masamlp **0.1.0** on torch 2.9.1 (macOS CPU/MPS) and torch
2.11.0+cu128 (Colab L4).

Overall verdict first, for morale: **9 of 10 architectures shipped, top score 0.94915 balanced
accuracy (GRN) ≈ GBDT parity** — the estimator surface (`fit(X, y, sample_weight=, eval_set=)`,
`classes_`, `best_iteration_`, DataFrame-in, native cats, device="auto") plugged into an
sklearn-style CV harness with zero adapters. The issues below are ranked by severity.

---

## P0 — `entmax15` crashes when mask weights go non-finite → **DANet is unusable at scale**

**Symptom.** `model="danet"` training crashes stochastically after ~tens of epochs at real data
scale (56k+ rows/fold). The error MOVES depending on device/timing — all of these are the SAME
underlying failure:
- CUDA: `AcceleratorError: CUDA error: device-side assert triggered` (mid-`backward`), or
  `CUBLAS_STATUS_EXECUTION_FAILED` on `cublasSgemm`/`cublasSgemmStridedBatched` (async late
  report of the assert);
- CPU: `RuntimeError: index -1 is out of bounds for dimension 1 with size 354` at
  **`src/masamlp/models/layers.py:150`**, i.e. `tau_star = tau.gather(dim, k_star - 1)`.

**Mechanism (pinned).** In `entmax15`, `k_star = (tau <= x_sorted).sum(dim, keepdim=True)`.
For FINITE inputs `k_star >= 1` is guaranteed (after the max-shift, `x_sorted[0] = 0` and
`tau_1 = mean_1 - sqrt(clamp((1-ss_1)/1)) = -1 <= 0`). `k_star == 0` therefore requires
**NaN/Inf in the mask logits** (`AbstractLayer.mask_weight`) — every comparison goes False →
`gather(-1)` → CPU index error / CUDA device assert. So the crash is a *symptom*; the disease
is a **non-finite training trajectory** in DANet.

**Probe matrix (all on S6E7, 70k subsample, 5-fold; fold-of-crash varies run-to-run):**
| config | outcome |
|---|---|
| defaults (`amp="auto"` → bf16 on L4) | crash (device assert) |
| `amp=False` (fp32) | crash on 1/5 folds (varies) |
| `amp=False, grad_clip=1.0` | crash on 1/5 folds (different fold than before) |
| `amp=False, grad_clip=1.0, lr=3e-4` | crash on **3/5** folds |
| CPU fp32 (same config) | same crash, clean Python traceback at layers.py:150 |
| tiny runs (8k rows × ≤3 epochs), any device | always pass |

`grad_clip` NOT preventing it is diagnostic: clipping rescales finite grads but propagates NaN
— so the NaN is born in the **forward/loss**, not by a finite-gradient blowup. Suspects to
check, in order: (1) the fused GhostBatchNorm stats path (KI-009 rewrite) — a degenerate
virtual chunk / eps handling; (2) the entmax input magnitudes growing unbounded (no
normalization on `mask_weight`); (3) the sqrt in entmax (`delta` clamp is min=0, but
`mean_sq - mean^2` cancellation in fp32/bf16 at large magnitudes); (4) AMP interactions.

**Requested fix (two layers):**
1. **Guard (one line, ship regardless):** `k_star = k_star.clamp(min=1)` after the sum in
   `entmax15` — converts a hard crash into a degraded step and makes every downstream user
   safe. Optionally also `x = torch.nan_to_num(x)` at entry, or a debug-mode assert with a
   clear message ("non-finite entmax input — reduce lr / check normalization").
2. **Root cause:** find the NaN genesis (train DANet on S6E7 70k with
   `torch.autograd.set_detect_anomaly(True)` on CPU — the clean CPU repro makes this cheap),
   then fix properly (e.g. bound/normalize mask logits, guard the GBN fused path, or
   skip-batch-on-nonfinite-loss in the Trainer like LightGBM does).

**CPU repro (crashes within the first folds, ~10-20 min):**
```python
# in ~/competition (needs the s6e7 plugin + kagglehub-cached comp data)
import sys; [sys.path.insert(0, p) for p in ('s6e7/mlp', 's6e7', '.')]
import config as C, common as Cm
from mlp_core import config_for, make_model      # thin wrapper; adds missing-flags + plr-lite
from sklearn.model_selection import StratifiedKFold
tr = Cm.load_train(70000, 0); X, y = tr[C.FEAT], tr[C.TARGET]
cfg = {**config_for('danet'), 'device': 'cpu'}    # amp=False + grad_clip=1.0 already in ARCHS
for tr_idx, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    make_model(cfg, 0).fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[va], y.iloc[va])])
```
**Acceptance test:** the loop above completes 5/5 folds on CPU AND on a CUDA T4/L4, twice in a
row. (Then S6E7's danet ships via `python3 s6e7/mlp/experiments/run_stage2.py danet`.)

**Ops note for the fixing agent:** a CUDA device-side assert **poisons the whole process's CUDA
context** — every later CUDA call fails (including `torch.manual_seed`). Test danet in a
subprocess per run, or restart the kernel between attempts.

---

## P1 — `ens_mode="vectorized"` rejects BatchNorm architectures

`MasaClassifier(model="resnet", n_ens=k, ens_mode="vectorized")` raises
`ValueError: models with BatchNorm running statistics cannot train vectorized; use ens_mode='loop'`.
That is a correct guard, but it silently costs resnet/danet the ~k× speedup that the LayerNorm
archs (grn/realmlp/ftt/...) get for free (verified: grn n_ens=8 vectorized trains fine, ~2.5×
single-member wall time for 8 members on L4).

**Request:** support vectorized BN by batching the running-stat buffers per member (shape
`(n_ens, C)` buffers + a member-aware functional BN), or at minimum document the limitation in
the estimator docstring / raise early at construction (it currently raises at fit time).

---

## P1 — Feature request: **EMA weight averaging** (the strongest missing training lever)

The top public S6E7 NN (kaggle.com/code/yekenot/ps-s6-e7-realmlp-pytorch, OOF BA **0.95048**,
above our whole family) uses a pytabkit-style RealMLP-TD recipe whose components we could NOT
reproduce through masamlp's public API. We decomposed it on paired CV; its *feature* tricks
(numeric-as-categorical embedding copies, per-value numeric TE) were **null at full scale** on
our resnet (0.94896 vs 0.94904 base) — the remaining edge is in the **training loop**:
1. **EMA of model weights** (decay ≈0.998), validated/early-stopped on the EMA copy — not
   available in masamlp. This is the highest-value request: `ema_decay: float | None = None`
   on the estimator; when set, keep an EMA state dict updated per optimizer step, run eval /
   best-state selection on the EMA weights.
2. Scheduled label smoothing (`ls_eps` cos→0) and scheduled dropout (`expm4t`) — masamlp has
   `dropout_schedule` for realmlp only; consider promoting LS/dropout schedules to generic
   estimator options.
3. Per-parameter-group lr/wd multipliers (scale layer ×10, biases ×0.1, first layer wd ×0.1,
   PBLD ×0.093...) — partially covered by the realmlp preset internals; consider exposing a
   `param_group_overrides` escape hatch.

---

## P2 — Retrieval models (tabr / modernnca) need a corpus budget

Both re-encode the full training corpus per batch; there is no API to bound it. Measured on L4
(22 GB) at 345k rows (276k train/fold): **modernnca OOMs** (single 8.4 GiB candidate alloc);
**tabr runs but superlinear** (~960 s/fold vs 83 s at 56k — ~11.6× for ~5× rows). We worked
around it OUTSIDE the estimator (stratified subsample of the fit corpus to 131072 rows;
accuracy cost ≈0: tabr still hit 0.94799, modernnca 0.94796 at full-data evaluation).

**Request:** a first-class `candidate_budget: int | None` (or `corpus_subsample`) kwarg for
tabr/modernnca that bounds the retrieval corpus (stratified, seeded) — plus chunked candidate
scoring in modernnca's eval path to bound peak memory.

---

## P3 — Notes, no action required

- `eval_metric="balanced_accuracy"` as the early-stopping signal was **much worse** than
  `multi_logloss` on this imbalanced task (−0.012 BA): the metric is a noisy discrete stop
  signal. Maybe worth a docs note ("stop on a proba-quality metric; convert to your task metric
  post-hoc").
- `classes_ = np.unique(y)`, string-label eval_set handling, sample_weight, and
  `preprocessor_.categorical_idx_` introspection all behaved exactly as documented — these are
  load-bearing for harness integration; please keep them stable.
- Full-scale S6E7 reference numbers (5-fold, beta-tuned OOF BA) for regression-testing a new
  version: grn 0.94915 / resnet 0.94904 / gandalf 0.94899 / ftt 0.94892 / lnn 0.94887 /
  tabtr 0.94851 / tabr 0.94799* / modernnca 0.94796* / realmlp(TD-S) 0.94532 (*with the 131k
  corpus workaround). A version-up should reproduce these ±0.001 with the same recipe
  (plr-lite numeric embeddings + missingness flags; see the s6e7 lab).
