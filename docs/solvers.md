# Standalone global kernel solvers

`from masamlp.solvers import NystromKRR, LinearResidualKRR, rank_curve,
rpcholesky_landmarks` exposes NumPy-in/NumPy-out estimators using Torch on
`cpu` or `cuda`. Pass an already prepared finite matrix: standardized numeric
features and unscaled one-hot categories. There is no preprocessing, tree,
AGOP, feature learning, EigenPro, model registration or Trainer integration.
Sklearn dispatch and shared serialization are a separate follow-up.

## Parameters and equations

`NystromKRR` takes these constructor parameters:

| Parameter | Default | Contract |
|---|---|---|
| `r` | `1000` | Landmark count; clipped to positive-weight row count with a warning. |
| `reg` | `1.0` | Positive ridge on the **sum** of weighted squared errors. |
| `kernel` | `"laplace"` | `exp(-distance/L)`; `"gaussian"` uses `exp(-distance²/(2L²))`. Distance is Euclidean. |
| `bandwidth` | `"median"` | Positive float or median unordered pairwise distance on a seeded sample of up to 4,096 fitting rows. A zero median uses 1.0. |
| `bandwidth_scale` | `1.0` | Positive multiplier applied to either bandwidth choice. |
| `random_state` | `0` | Nonnegative seed; local NumPy generators leave global RNG state unchanged. Uniform uses a fresh seeded permutation independent of bandwidth sampling. |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` (also an explicit CUDA index); MPS/XLA are unsupported. |
| `dtype` | `"float32"` | Kernel precision; `"float64"` enables parity. Accumulation and solve stay FP64. |
| `block_rows` | `16384` | Training workspace row budget, integer ≥5; internal microblocks reserve FP64 operand space. |
| `predict_batch_rows` | `16384` | Inference workspace row budget, integer ≥5. |
| `dense` | `False` | Exact FP64 reference, restricted to at most 5,000 input rows; ignores `r`. |
| `landmark_method` | `"rpcholesky"` | `"rpcholesky"` single-pass projection or `"uniform"` seeded permutation; both support nested prefixes. |
| `max_factor_bytes` | `16 * 1024**3` (16 GiB) | Nonnegative integer budget for the RPCholesky FP32 factor alone. Oversized factors raise an error recommending `"uniform"`. |

`fit(X, t, sample_weight=None)` removes zero-weight rows before bandwidth
estimation and landmark selection; weights must be finite, nonnegative and
not all zero. With landmark rows Z, it solves
`(K_nm.T W K_nm + reg K_mm) alpha = K_nm.T W t` and predicts `K_xm alpha`.
Doubling weights **and reg** preserves predictions. The dense reference
solves the equivalent symmetric system
`(sqrt(W) K_nn sqrt(W) + reg I) beta = sqrt(W) t`, `alpha = sqrt(W) beta`.
Compare full-rank Nyström coefficients in input-row order using
`model.alpha_[np.argsort(model.landmark_indices_)]`; dense centres use input order.

Cholesky runs in place. A failed factorization rebuilds the streamed system,
then retries at most three times with diagonal jitter on a freshly rebuilt system:
`attempt * 1e-10 * trace(system)/r`. Final fallback is `eigh` with eigenvalues
clipped below `1e-10 * trace(system)/r`. `solver_path_` records `cholesky`,
`cholesky+jitter1/2/3`, or `eigh`.

## Nested landmarks and rank curves

`rpcholesky_landmarks(X, r_max, random_state=0, block=16384, *,
kernel="laplace", bandwidth="median", bandwidth_scale=1.0, device="cpu",
max_factor_bytes=16 * 1024**3)` returns distinct ordered row indices sampled
proportionally to the residual kernel diagonal. The single-pass algorithm
computes each new column once as `K[:, pivot] - F[:, :k] @ F[pivot, :k].T`,
then normalizes it and stores it in the on-device FP32 factor F. Kernels and
the residual diagonal use FP64; projection uses FP32. The diagonal stays on
device with one host copy per pivot for the seeded NumPy sampler. Exhausted
numerical rank completes the sequence with a seeded permutation of remaining
indices. Run once at the largest rank and use prefixes.

`landmark_method="uniform"` selects the first r indices of ONE permutation,
`np.random.default_rng(random_state).permutation(n)`, of the positive-weight
rows. It is independent of bandwidth sampling, deterministic and nested by
prefix, with no kernel/projection work. Creating the permutation is O(n) time
and memory; taking a prefix view is O(1). It has no n-by-r factor allocation.

`rank_curve(X, t, w=None, ranks=(1000, 2000, 4000, 8000), *, X_val,
random_state=0, landmark_method="rpcholesky", max_factor_bytes=16 * 1024**3,
**params)` does that single landmark run and a fresh solve for each
prefix. It returns dictionaries containing `rank`, `requested_rank`, `model`,
`predictions`, `train_residual_norm`, `solver_path`, `wall_seconds`,
`landmark_seconds` and `peak_device_memory`. The norm is `sqrt(sum(w*(f-t)²))`.
Per-rank wall time includes its solve and both predictions; the shared
landmark cost is reported separately. CUDA peak allocation includes the
shared landmark phase and is synchronized; CPU reports `None`.
The caller must supply training-internal validation rows. No labels are
read for rank selection. Residual monotonicity is tested on a smooth fixture;
it is not a theorem for every regularized target.

## Weighted-logit residual and exact fallback

`LinearResidualKRR(parent="logit", gamma_grid=(0, 0.125, 0.25, 0.5, 1.0),
w_min=1e-4, clip_correction=4.0, landmark_method="rpcholesky",
max_factor_bytes=16 * 1024**3, **params)` forwards solver options to `NystromKRR`.
Supply `fit(X, y, eta0=parent_logits)` for `parent="logit"`, or
`fit(X, y, p0=parent_probabilities)` for `parent="proba"`, with optional
`sample_weight`. Labels must be binary. It fits the literal working target
`w = max(p0*(1-p0), w_min) * sample_weight`, `z = (y-p0)/w`; zero-weight
rows have unused `z=0`. Sample weights enter **both** w and z's denominator.
With `sample_weight=None` (frozen for stage 2 and stage 3), this is the
**one-step Newton / IRLS working-response objective**
`sum(w_i * (z_i - f(x_i))²) + reg * ||f||_K²` for an additive logistic
correction at the supplied parent. The closed form is
`(K_nm.T W K_nm + reg K_mm) alpha = K_nm.T (y-p0)`.
It equals the Newton step of summed logistic loss plus `reg/2 * ||f||_K²`
when the curvature floor is inactive; `w_min` stabilizes the Hessian otherwise.
High-curvature rows have smaller working targets/steps by design.

`sample_weight` multiplies w as the external row-importance multiplier.
Under the preserved convention it also divides z, so positive multipliers
change the curvature without scaling the logistic gradient. Thus nonuniform
weights do **not** give the usual importance-weighted logistic Newton step.
Consequently the weight/ridge scaling identity above applies to KRR with
fixed targets, not to refitting this head after changing its working targets.

`correction(X)` clips the fit to `[-clip_correction, clip_correction]`.
Positive infinity disables clipping. `predict_logit(X, eta0, gamma=None)`
adds `gamma*correction` for nonzero gamma. Zero returns a copy of eta0 with
no arithmetic or correction evaluation, preserving dtype, signed zeros,
infinities and NaN payloads. `predict_proba(X, p0=..., gamma=0)` likewise
copies the actual probability vector; a sigmoid of logits cannot reproduce
the original probability bits. Nonzero probabilities are one-dimensional.

`select_gamma(X_val, eta0_val, y_val)` maximizes AUC on only those supplied
rows; ties choose smaller gamma, and nonfinite predictions/metrics are
discarded. Zero must belong to the finite nonnegative grid and is retained
if no admissible candidate improves it. Invalid parent scores or single-class
validation retain zero. `gamma_` starts at zero and is the prediction default.

Both estimators use `save(path)` / `load(path, device=None)` and pickle-free
`np.savez` archives at the exact path, including centres, indices, coefficients,
resolved bandwidth and parameters; the head also stores parent and gamma.

## Memory, precision and qualification limits

The production solve's dominant explicit matrix storage is bounded by
**`block_rows * r * 4 B + r² * 8 B`**. In detail, `b=floor(block_rows/5)`
reserves `20*b*r` bytes for an FP32 kernel and two FP64 operands; the single
FP64 system is reused as its factor. Predictions use at most
`12*floor(predict_batch_rows/3)*r` bytes of matrix workspace. These bounds
exclude the caller's input/output arrays, O(n) weight/target/residual vectors,
O((b+r)*d) feature tensors, and backend BLAS/eigensolver workspace/allocator
overhead. The FP64 kernel path uses the same training budget and an
`8*predict_batch_rows*r` inference bound. No full cross-kernel is stored.

The separate RPCholesky landmark phase stores an **n-by-r FP32 factor**:
`n * r * 4 B <= max_factor_bytes` is checked before bandwidth or allocation.
The default is 16 GiB, excluding O(n) diagonal/host sampling vectors, on-device
FP64 features, per-block columns and backend workspace. A factor that exceeds
the budget raises a clear error recommending `landmark_method="uniform"`;
there is no recomputing fallback. The solve above starts after that factor is
released, so peak memory is the maximum of the two phases plus their respective
overhead. RPCholesky now evaluates O(n*r) kernel entries and performs O(n*r²)
projection arithmetic. Uniform uses O(n) permutation storage instead.
Median bandwidth uses at most 4,096 rows and 8,386,560 FP64 pair distances in
its separate phase.

CPU FP64 same-input/same-seed indices and coefficients are bitwise repeatable
within the same Torch/BLAS environment. The CUDA FP32 acceptance tolerance is
≤1e-6 relative for repeatability; cross-hardware/library bitwise identity is
not promised. CPU tests check dense solution/prediction and system residual
errors ≤1e-6, and FP32 sigmoid discrepancy ≤1e-5. CUDA tolerance, actual peak
VRAM, the 1k/2k/4k/8k real-data curve and ≤600-second fit runtime still require
the separately authorized stage-2 qualification. No real-data fit is part of
this change.
