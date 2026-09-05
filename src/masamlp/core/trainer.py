"""The shared training engine.

Models are pure ``nn.Module``s; everything operational lives here: device
resolution, AMP, optional ``torch.compile``, batching (device-resident
tensors with index slicing — no DataLoader; small data trains full-batch),
gradient clipping, schedulers, per-epoch evaluation, and early stopping with
best-epoch weight restoration.

The loss reduction contract: objectives return per-sample losses and the
trainer computes ``(loss * w).sum() / w.sum()`` (:func:`weighted_loss`).
Models may emit per-member outputs ``(n, k, out)`` — a weight-shared inner
ensemble (TabM); the members flatten into rows before the objective runs,
so objectives never see a member dim — see core/objectives.py.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from masamlp.core.device import (
    maybe_compile,
    resolve_amp,
    resolve_device,
    resolve_predict_amp,
    set_threads,
    xla_seed,
    xla_sync_fn,
)
from masamlp.core.metrics import BaseMetric
from masamlp.core.objectives import BaseObjective
from masamlp.core.training_terms import TrainingTerms
from masamlp.data.dataset import TabularData
from masamlp.utils.random import seed_everything


@dataclass
class TrainerConfig:
    n_epochs: int = 256
    batch_size: int | str | None = "auto"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    betas: tuple[float, float] | None = None
    lr_scheduler: str = "none"
    # "flat_cos" scales weight decay per step (RealMLP-TD); param groups may
    # carry a "wd_factor" (e.g. 0.0 for biases).
    weight_decay_schedule: str = "none"
    # "auto" (default) keeps each named optimizer's own convention — adamw
    # decouples (`p *= 1 - lr*wd`), adam couples (`wd*p` into the gradient),
    # sgd couples. "decoupled" / "coupled" force one regardless of the name.
    # (pytabkit's RealMLP-TD is DECOUPLED, and additionally scales the decay
    # by lr, which `adamw` already does.)
    weight_decay_mode: str = "auto"
    # "coslog4" / "flat_cos" scale the objective's label smoothing per step
    # (pytabkit's `ls_eps_sched`); "none" keeps it constant.
    label_smoothing_schedule: str = "none"
    # Drop the final short batch of every epoch (pytabkit's `drop_last=True`),
    # so every optimizer step sees exactly `batch_size` rows.
    drop_last: bool = False
    # True broadcasts one row batch to all members. False follows the TabM
    # paper's headline protocol with an independent row permutation per member.
    share_training_batches: bool = True
    grad_clip: float | None = None
    # Exponential moving average of the model parameters (Polyak averaging).
    # When set (typically ~0.99-0.999), eval / early stopping / the final
    # weights use the EMA copy instead of the last optimizer step.
    ema_decay: float | None = None
    device: str | torch.device = "auto"
    amp: str | bool = "auto"
    # bf16 autocast for evaluation and prediction (training AMP never covers
    # them). Opt-in: False keeps the fp32 predict contract; True/"on" forces
    # bf16 where the device supports it.
    amp_predict: str | bool = False
    # XLA only: optimizer steps per graph barrier. 1 = one graph per step
    # (the 0.4.0 behavior); K > 1 fuses K steps into one XLA program, which
    # amortizes the per-step dispatch floor that dominates small-model fits
    # on TPU at tabular batch sizes. No effect on other devices.
    xla_fuse_steps: int = 1
    compile: bool = False
    early_stopping_rounds: int | None = None
    random_state: int | None = None
    # "global" seeds python/NumPy/torch process-wide (the default, and the
    # only mode that may run on the main thread). "device" seeds only this
    # fit's CUDA generator — required inside ensemble-sharding worker
    # threads, where touching process-global RNG state would race.
    seed_scope: str = "global"
    verbose: int = 0
    n_threads: int | None = None
    eval_batch_size: int = 8192
    # batch_size="auto": full-batch at or below the threshold, else minibatch.
    full_batch_threshold: int = 4096
    default_batch_size: int = 1024


@dataclass
class EvalSet:
    """A named eval split. ``y_metric`` holds the original-scale targets the
    metrics see (regression targets may be standardized inside ``data.y``)."""

    name: str
    data: TabularData
    y_metric: np.ndarray

    def to(self, device: torch.device) -> EvalSet:
        return EvalSet(self.name, self.data.to(device), self.y_metric)


@dataclass
class TrainResult:
    evals_result: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    best_iteration: int | None = None
    best_score: float | None = None


class EarlyStopper:
    def __init__(self, patience: int, minimize: bool) -> None:
        self.patience = patience
        self.minimize = minimize
        self.best_value = np.inf if minimize else -np.inf
        self.best_epoch = -1
        self._bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        improved = value < self.best_value if self.minimize else value > self.best_value
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self._bad_epochs = 0
        else:
            self._bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self._bad_epochs >= self.patience


def weighted_loss(
    objective: BaseObjective,
    y: Tensor,
    raw: Tensor,
    weight: Tensor | None,
    member_batches: bool = False,
    *,
    terms: TrainingTerms | None = None,
) -> Tensor:
    """Trainer-owned weighted data risk plus optional model training terms.

    WP2, 2026-09-05: independent ``(n, k, out)`` member batches now use
    ``mean_j(sum_i(w_ij * loss_ij) / sum_i(w_ij))``, correcting the former
    pooled-denominator implementation. A zero-total member contributes zero
    to that fixed-k mean (all-zero weights give zero data risk). Shared
    batches retain their original flattened reduction. Objectives always
    receive flattened rows, including custom objectives.

    Auxiliary losses follow the same reduction; model regularizers use their
    own fixed normalizers and are added once. The Trainer skips an entirely
    zero-weight batch, including regularizers and optimizer/EMA updates.
    """
    row_shape = raw.shape[:-1]
    independent = raw.ndim == 3 and member_batches
    if raw.ndim == 3:
        n, k, out_dim = raw.shape
        raw = raw.reshape(n * k, out_dim)
        if member_batches:
            y = y.reshape(n * k, *y.shape[2:])
            weight = None if weight is None else weight.reshape(n * k)
        else:
            y = y.repeat_interleave(k, dim=0)
            weight = None if weight is None else weight.repeat_interleave(k, dim=0)
    loss_i = objective.per_sample_loss(y, raw)
    if terms is not None:
        for term in terms.auxiliary:
            if term.values.shape != row_shape:
                raise ValueError(
                    f"RowTerm shape must be {tuple(row_shape)}, got {tuple(term.values.shape)}"
                )
            if term.coefficient != 0:
                loss_i = loss_i + term.coefficient * term.values.float().reshape(-1)
    if weight is None:
        loss = loss_i.mean()
    elif independent:
        w = weight.reshape(n, k)
        totals = w.sum(dim=0)
        # Tensor-only guard: safe gradients and static XLA graphs even when
        # different members have zero total weight on successive steps.
        denominator = torch.where(totals > 0, totals, torch.ones_like(totals))
        loss = ((loss_i.reshape(n, k) * w).sum(dim=0) / denominator).mean()
    else:
        total = weight.sum()
        denominator = torch.where(total > 0, total, torch.ones_like(total))
        loss = (loss_i * weight).sum() / denominator
    if terms is not None:
        for term in terms.regularizers:
            if term.coefficient != 0:
                loss = loss + term.coefficient * term.value.float() / term.normalizer
    return loss


def predict_transformed(
    model: nn.Module,
    data: TabularData,
    transform: Callable[[Tensor], Tensor],
    batch_size: int = 8192,
    autocast_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Batched inference -> prediction-scale NumPy array; ``(n,)`` when the
    output has a single column. Chunk outputs stay on the device and move to
    the host once at the end. On XLA a graph barrier runs every
    ``xla_eval_sync_chunks`` chunks (a model policy attribute, default 1):
    without any barrier every chunk's forward fuses into one program whose
    intermediates all coexist — ModernNCA's streamed eval at 345k rows
    demanded 50.6 GiB of 16 GiB HBM (measured), which is why its policy stays
    at 1 — while models whose eval graphs are HBM-light (TabR) fuse several
    chunks to win back cross-chunk optimization. The transfer stays batched
    either way. ``autocast_dtype`` wraps the forward in autocast (bf16
    prediction, opt-in via ``amp_predict``); outputs are always fp32."""
    model.eval()
    outs: list[Tensor] = []
    device = data.x_num.device
    barrier = xla_sync_fn() if device.type == "xla" else None
    sync_chunks = max(1, int(getattr(model, "xla_eval_sync_chunks", 1)))
    # inference_mode tensors break XLA's lazy tracing ("Cannot set
    # version_counter for inference tensor"); no_grad is the XLA-safe
    # equivalent and passes the retrieval models' eval-cache gate too.
    no_autograd = torch.no_grad() if device.type == "xla" else torch.inference_mode()
    with no_autograd:
        for chunk, start in enumerate(range(0, len(data), batch_size), start=1):
            idx = torch.arange(start, min(start + batch_size, len(data)), device=device)
            batch = data.slice(idx)
            with torch.autocast(
                device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
            ):
                out = transform(model(batch.x_num, batch.x_cat))
            outs.append(out.float())
            if barrier is not None and chunk % sync_chunks == 0:
                barrier()
    pred = torch.cat(outs).cpu().numpy()
    return pred[:, 0] if pred.ndim == 2 and pred.shape[1] == 1 else pred


def _build_param_groups(
    model: nn.Module, extra_modules: list[nn.Module], lr: float, weight_decay: float = 0.0
) -> list[dict]:
    """Optimizer param groups. Models may expose ``param_groups()`` returning
    ``[{"params": [...], "lr_factor": f}, ...]`` (RealMLP trains its scaling
    layer at 6x and biases at 0.1x); schedulers preserve the factors."""
    if hasattr(model, "param_groups"):
        groups = [dict(g) for g in model.param_groups()]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    for module in extra_modules:
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params})
    for group in groups:
        group.setdefault("lr_factor", 1.0)
        group.setdefault("wd_factor", 1.0)
        group["lr"] = lr * group["lr_factor"]
        group["weight_decay"] = weight_decay * group["wd_factor"]
    return [g for g in groups if g["params"]]


def _make_optimizer(
    name: str,
    groups: list[dict],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] | None,
    weight_decay_mode: str = "decoupled",
) -> torch.optim.Optimizer:
    if weight_decay_mode not in ("auto", "decoupled", "coupled"):
        raise ValueError(
            f"Unknown weight_decay_mode {weight_decay_mode!r}. "
            "Expected 'auto', 'decoupled' or 'coupled'"
        )
    # AdamW decouples, Adam couples: the mode picks the torch class that
    # already implements it, so the per-group `weight_decay` the scheduler
    # writes keeps meaning the same thing in both. "auto" keeps each named
    # optimizer's own convention, which is what pre-0.9.0 did.
    if name in ("adamw", "adam"):
        mode = (
            ("decoupled" if name == "adamw" else "coupled")
            if (weight_decay_mode == "auto")
            else weight_decay_mode
        )
        cls = torch.optim.AdamW if mode == "decoupled" else torch.optim.Adam
        return cls(groups, lr=lr, weight_decay=weight_decay, betas=betas or (0.9, 0.999))
    if name == "sgd":
        if betas is not None:
            raise ValueError("betas is only supported for adam/adamw")
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}. Expected 'adamw', 'adam', or 'sgd'")


def _coslog4(t: float) -> float:
    """RealMLP's lr factor: ``0.5 - 0.5*cos(2*pi*log2(1 + 15*t))`` for
    ``t`` in [0, 1] — warmup, one full oscillation, and decay to ~0."""
    return float(0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t)))


def flat_cos(t: float) -> float:
    """pytabkit's ``flat_cos``: constant 1 for the first half of training,
    cosine decay to 0 over the second half. Used by RealMLP-TD for the
    weight-decay and dropout schedules."""
    if t < 0.5:
        return 1.0
    return float(0.5 * (1.0 + np.cos(np.pi * (t - 0.5) / 0.5)))


def _update_ema(model: nn.Module, ema_params: dict[str, Tensor], decay: float) -> None:
    """In-place EMA update of the shadow parameters after an optimizer step."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


def _swap_in_params(model: nn.Module, new_params: dict[str, Tensor]) -> dict[str, Tensor]:
    """Copy ``new_params`` into the model, returning the replaced values so the
    caller can restore them (used to evaluate on EMA weights without disturbing
    the live training parameters). Buffers (BN running stats, retrieval
    candidate corpus) are left untouched."""
    saved: dict[str, Tensor] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            saved[name] = param.detach().clone()
            param.copy_(new_params[name])
    if hasattr(model, "invalidate_eval_cache"):
        # Retrieval models cache eval-time corpus encodings; an in-place
        # parameter swap changes them without any mode transition.
        model.invalidate_eval_cache()
    return saved


def _resolve_batch_size(config: TrainerConfig, n_rows: int) -> int:
    bs = config.batch_size
    if bs is None:
        return n_rows
    if bs == "auto":
        return n_rows if n_rows <= config.full_batch_threshold else config.default_batch_size
    if isinstance(bs, int) and bs > 0:
        return min(bs, n_rows)
    raise ValueError(f"Invalid batch_size {bs!r}. Expected 'auto', None, or a positive int")


def _make_epoch_batches(
    *,
    n_rows: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    drop_last: bool,
    n_members: int,
    share_training_batches: bool,
) -> list[Tensor | None]:
    """Build one epoch's shared or member-indexed static row batches."""
    full_batch = batch_size >= n_rows
    if share_training_batches and full_batch:
        return [None]
    if share_training_batches:
        order = torch.randperm(n_rows, generator=generator)
    else:
        order = torch.stack(
            [torch.randperm(n_rows, generator=generator) for _ in range(n_members)],
            dim=1,
        )
    cpu_chunks = list(order.split(batch_size, dim=0))
    if drop_last and len(cpu_chunks[-1]) < batch_size:
        cpu_chunks = cpu_chunks[:-1]
    if device.type == "xla":
        # Separate transfers avoid tuple-typed split-view IR on XLA.
        return [chunk.to(device) for chunk in cpu_chunks]
    resident = order.to(device)
    chunks = list(resident.split(batch_size, dim=0))
    if drop_last and len(chunks[-1]) < batch_size:
        chunks = chunks[:-1]
    return chunks


class Trainer:
    def fit(
        self,
        model: nn.Module,
        objective: BaseObjective,
        train: TabularData,
        eval_sets: list[EvalSet],
        metrics: list[BaseMetric],
        config: TrainerConfig,
        inverse_target: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> TrainResult:
        device = resolve_device(config.device)
        if config.seed_scope == "global":
            seed_everything(config.random_state)
        elif config.seed_scope == "device":
            if config.random_state is not None and device.type == "cuda":
                torch.cuda.init()
                index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.default_generators[index].manual_seed(config.random_state)
        else:
            raise ValueError(
                f"Unknown seed_scope {config.seed_scope!r}. Expected 'global' or 'device'"
            )
        if device.type == "xla" and config.random_state is not None:
            # torch.manual_seed does not reach the XLA device generator
            # (dropout, rand_like); seed it under both seed scopes.
            xla_seed(config.random_state)
        if device.type == "cpu":
            set_threads(config.n_threads)

        model.to(device)
        extra_modules = objective.torch_modules()
        for module in extra_modules:
            module.to(device)
        train = train.to(device)
        eval_sets = [es.to(device) for es in eval_sets]

        run_model = maybe_compile(model, config.compile, device)
        groups = _build_param_groups(
            model, extra_modules, config.learning_rate, config.weight_decay
        )
        params = [p for g in groups for p in g["params"]]
        optimizer = _make_optimizer(
            config.optimizer,
            groups,
            config.learning_rate,
            config.weight_decay,
            config.betas,
            config.weight_decay_mode,
        )
        scheduler = None
        per_step_schedule = None
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
        elif config.lr_scheduler == "coslog4":
            # RealMLP's schedule, applied per optimizer step over the whole run.
            per_step_schedule = _coslog4
        elif config.lr_scheduler != "none":
            raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}")
        if config.weight_decay_schedule not in ("none", "flat_cos"):
            raise ValueError(f"Unknown weight_decay_schedule {config.weight_decay_schedule!r}")
        wd_scheduled = config.weight_decay_schedule == "flat_cos"
        model_has_schedule = hasattr(model, "set_schedule_t")
        # pytabkit's `ls_eps_sched`: the smoothing epsilon is scaled per step
        # and enters the LABELS, so the objective's attribute is what moves.
        _LS_SCHEDULES = {"coslog4": _coslog4, "flat_cos": flat_cos}
        if config.label_smoothing_schedule not in ("none", *_LS_SCHEDULES):
            raise ValueError(
                f"Unknown label_smoothing_schedule {config.label_smoothing_schedule!r}. "
                f"Expected 'none' or one of {sorted(_LS_SCHEDULES)}"
            )
        ls_schedule = _LS_SCHEDULES.get(config.label_smoothing_schedule)
        ls_base = getattr(objective, "label_smoothing", None)
        if ls_schedule is not None and ls_base is None:
            raise ValueError(
                "label_smoothing_schedule needs an objective with a "
                "`label_smoothing` attribute (binary_logistic / multiclass_softmax)"
            )

        amp_enabled, amp_dtype = resolve_amp(config.amp, device, model)
        predict_dtype = resolve_predict_amp(config.amp_predict, device)
        # GradScaler only exists for cuda/cpu; fp16 (pre-bf16 GPUs) needs it.
        # XLA is always bf16 or fp32, so the scaler stays a disabled
        # passthrough there.
        scaler = torch.amp.GradScaler(
            "cuda" if device.type == "cuda" else "cpu",
            enabled=amp_enabled and amp_dtype == torch.float16,
        )
        # Lazy XLA queues ops until a barrier; one barrier per optimizer step
        # is the safe granularity (without any the whole epoch fuses into one
        # unboundedly large graph). xla_fuse_steps > 1 barriers every K steps
        # instead, fusing K steps into one XLA program to amortize the
        # per-step dispatch floor; epochs always end on a barrier so graph
        # signatures stay stable across epochs.
        fuse_steps = config.xla_fuse_steps
        if not isinstance(fuse_steps, int) or isinstance(fuse_steps, bool) or fuse_steps < 1:
            raise ValueError(
                f"xla_fuse_steps must be a positive int, got {config.xla_fuse_steps!r}"
            )
        step_barrier = xla_sync_fn() if device.type == "xla" else None

        n = len(train)
        batch_size = _resolve_batch_size(config, n)
        full_batch = batch_size >= n
        supports_member_batches = getattr(model, "supports_member_batches", False)
        if not config.share_training_batches and not supports_member_batches:
            raise ValueError(
                "share_training_batches=False needs a model with an inner ensemble "
                "that supports member-indexed inputs"
            )
        n_batch_members = int(getattr(model, "k", 1))
        # Permutations are drawn on CPU so runs are reproducible across devices.
        gen = torch.Generator()
        if config.random_state is not None:
            gen.manual_seed(config.random_state)

        result = TrainResult(
            evals_result={es.name: {m.name: [] for m in metrics} for es in eval_sets}
        )
        tracking = config.early_stopping_rounds is not None and bool(eval_sets)
        stopper = (
            EarlyStopper(config.early_stopping_rounds, metrics[0].minimize) if tracking else None
        )
        best_state: dict[str, Tensor] | None = None

        ema_decay = config.ema_decay
        ema_params: dict[str, Tensor] | None = None
        if ema_decay is not None:
            if not 0.0 < ema_decay < 1.0:
                raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay!r}")
            ema_params = {name: p.detach().clone() for name, p in model.named_parameters()}

        training_terms = getattr(model, "training_terms", None)

        def train_step(step_model: nn.Module, batch: TabularData) -> Tensor:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                raw = step_model(batch.x_num, batch.x_cat)
                terms = training_terms(batch, raw) if training_terms is not None else None
                if terms is not None and not isinstance(terms, TrainingTerms):
                    raise TypeError("model.training_terms must return TrainingTerms or None")
                loss = weighted_loss(
                    objective,
                    batch.y,
                    raw.float(),
                    batch.weight,
                    member_batches=not config.share_training_batches,
                    terms=terms,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            return loss.detach()

        # drop_last (pytabkit's default) makes every step a full batch; with
        # fewer rows than one batch there is nothing to drop.
        drop_last = bool(config.drop_last) and not full_batch and n >= batch_size
        steps_per_epoch = (
            1 if full_batch else int(n // batch_size if drop_last else np.ceil(n / batch_size))
        )
        total_steps = max(1, config.n_epochs * steps_per_epoch)
        global_step = 0
        # Retrieval models (TabR) need to know which candidate rows are in the
        # current training batch to exclude themselves from their context.
        wants_batch_indices = getattr(model, "wants_batch_indices", False)
        full_idx = torch.arange(n, device=device) if wants_batch_indices else None

        first_step = True
        for epoch in range(config.n_epochs):
            run_model.train()
            epoch_loss = torch.zeros((), device=device)
            unflushed_steps = 0
            batches = _make_epoch_batches(
                n_rows=n,
                batch_size=batch_size,
                device=device,
                generator=gen,
                drop_last=drop_last,
                n_members=n_batch_members,
                share_training_batches=config.share_training_batches,
            )
            for idx in batches:
                batch = train if idx is None else train.slice(idx)
                # A zero weight removes a row's training contribution. If an
                # entire shuffled minibatch has zero weight, there is no
                # objective to optimize and the normalized reduction would
                # divide by zero, so skip that optimizer step altogether.
                if batch.weight is not None and not torch.any(batch.weight > 0):
                    continue
                t = global_step / total_steps
                if per_step_schedule is not None:
                    factor = per_step_schedule(t)
                    for group in optimizer.param_groups:
                        group["lr"] = config.learning_rate * group["lr_factor"] * factor
                if wd_scheduled:
                    wd_now = config.weight_decay * flat_cos(t)
                    for group in optimizer.param_groups:
                        group["weight_decay"] = wd_now * group["wd_factor"]
                if model_has_schedule:
                    # RealMLP-TD schedules its dropout probability over training.
                    model.set_schedule_t(t)
                if ls_schedule is not None:
                    objective.label_smoothing = ls_base * ls_schedule(t)
                if wants_batch_indices:
                    model.current_batch_indices = idx if idx is not None else full_idx
                global_step += 1
                if first_step and run_model is not model:
                    # torch.compile backends fail lazily, on the first real
                    # step — recover by dropping to eager (same parameters).
                    try:
                        loss = train_step(run_model, batch)
                    except Exception as exc:
                        warnings.warn(
                            f"torch.compile failed at first step ({type(exc).__name__}); "
                            "running eager",
                            stacklevel=2,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        run_model = model
                        loss = train_step(run_model, batch)
                else:
                    loss = train_step(run_model, batch)
                first_step = False
                if ema_params is not None:
                    _update_ema(model, ema_params, ema_decay)
                epoch_loss += loss * (len(batch) / n)
                unflushed_steps += 1
                if step_barrier is not None and unflushed_steps >= fuse_steps:
                    step_barrier()
                    unflushed_steps = 0
            if step_barrier is not None and unflushed_steps:
                step_barrier()
            # One host sync per epoch keeps the GPU pipeline full.
            epoch_loss_val = float(epoch_loss)
            if not np.isfinite(epoch_loss_val):
                raise ValueError(
                    f"Training loss became non-finite at epoch {epoch}; "
                    "try a lower learning_rate or grad_clip"
                )
            if scheduler is not None:
                scheduler.step()

            # Evaluate (and pick the best checkpoint) on the EMA parameters
            # when enabled, then restore the live weights for the next epoch.
            saved_params = _swap_in_params(model, ema_params) if ema_params is not None else None

            for es in eval_sets:
                pred = predict_transformed(
                    run_model,
                    es.data,
                    objective.transform,
                    config.eval_batch_size,
                    autocast_dtype=predict_dtype,
                )
                if inverse_target is not None:
                    pred = inverse_target(pred)
                for metric in metrics:
                    result.evals_result[es.name][metric.name].append(metric(es.y_metric, pred))

            if config.verbose > 0 and (epoch % config.verbose == 0 or epoch == config.n_epochs - 1):
                parts = [f"[{epoch}] train_loss: {epoch_loss_val:.5f}"]
                for es in eval_sets:
                    for metric in metrics:
                        parts.append(
                            f"{es.name}-{metric.name}: "
                            f"{result.evals_result[es.name][metric.name][-1]:.5f}"
                        )
                print("  ".join(parts))

            should_stop = False
            if stopper is not None:
                monitor = result.evals_result[eval_sets[0].name][metrics[0].name][-1]
                if stopper.update(monitor, epoch):
                    # Static buffers (retrieval corpora, possibly hundreds of
                    # MB) never change during fit — keep them out of the
                    # per-improvement CPU snapshot.
                    static_keys = getattr(model, "static_state_keys", ())
                    best_state = {
                        k: v.detach().to("cpu", copy=True)
                        for k, v in model.state_dict().items()
                        if k not in static_keys
                    }
                should_stop = stopper.should_stop

            if saved_params is not None:
                _swap_in_params(model, saved_params)
            if should_stop:
                break

        if wants_batch_indices:
            model.current_batch_indices = None
        if ls_schedule is not None:
            # The schedule mutates the objective in place; hand it back
            # unchanged so a reused objective (n_ens members, refits) is not
            # left at whatever epsilon the last step happened to set.
            objective.label_smoothing = ls_base
        if stopper is not None and best_state is not None:
            # strict=False tolerates exactly the static buffers the snapshot
            # deliberately omits — anything else missing is a real bug.
            static_keys = getattr(model, "static_state_keys", ())
            incompatible = model.load_state_dict(best_state, strict=False)
            stray = [k for k in incompatible.missing_keys if k not in static_keys]
            if stray or incompatible.unexpected_keys:
                raise RuntimeError(
                    "best-epoch restore mismatch: "
                    f"missing={stray} unexpected={list(incompatible.unexpected_keys)}"
                )
            model.to(device)
            result.best_iteration = stopper.best_epoch
            result.best_score = float(stopper.best_value)
        elif ema_params is not None:
            # No early stopping: the final weights are the EMA parameters.
            _swap_in_params(model, ema_params)
        return result
