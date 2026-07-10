"""The shared training engine.

Models are pure ``nn.Module``s; everything operational lives here: device
resolution, AMP, optional ``torch.compile``, batching (device-resident
tensors with index slicing — no DataLoader; small data trains full-batch),
gradient clipping, schedulers, per-epoch evaluation, and early stopping with
best-epoch weight restoration.

The loss reduction contract: objectives return per-sample losses and the
trainer computes ``(loss * w).sum() / w.sum()`` — see core/objectives.py.
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
    set_threads,
    xla_seed,
    xla_sync_fn,
)
from masamlp.core.metrics import BaseMetric
from masamlp.core.objectives import BaseObjective
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
    grad_clip: float | None = None
    # Exponential moving average of the model parameters (Polyak averaging).
    # When set (typically ~0.99-0.999), eval / early stopping / the final
    # weights use the EMA copy instead of the last optimizer step.
    ema_decay: float | None = None
    device: str | torch.device = "auto"
    amp: str | bool = "auto"
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


def predict_transformed(
    model: nn.Module,
    data: TabularData,
    transform: Callable[[Tensor], Tensor],
    batch_size: int = 8192,
) -> np.ndarray:
    """Batched inference -> prediction-scale NumPy array; ``(n,)`` when the
    output has a single column. Chunk outputs stay on the device and move to
    the host once — one transfer (and on XLA one graph execution) per call
    instead of one per chunk."""
    model.eval()
    outs: list[Tensor] = []
    device = data.x_num.device
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(data)), device=device)
            batch = data.slice(idx)
            outs.append(transform(model(batch.x_num, batch.x_cat)).float())
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
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(
            groups, lr=lr, weight_decay=weight_decay, betas=betas or (0.9, 0.999)
        )
    if name == "adam":
        return torch.optim.Adam(
            groups, lr=lr, weight_decay=weight_decay, betas=betas or (0.9, 0.999)
        )
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
            config.optimizer, groups, config.learning_rate, config.weight_decay, config.betas
        )
        scheduler = None
        per_step_schedule = None
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.n_epochs
            )
        elif config.lr_scheduler == "coslog4":
            # RealMLP's schedule, applied per optimizer step over the whole run.
            per_step_schedule = _coslog4
        elif config.lr_scheduler != "none":
            raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}")
        if config.weight_decay_schedule not in ("none", "flat_cos"):
            raise ValueError(
                f"Unknown weight_decay_schedule {config.weight_decay_schedule!r}"
            )
        wd_scheduled = config.weight_decay_schedule == "flat_cos"
        model_has_schedule = hasattr(model, "set_schedule_t")

        amp_enabled, amp_dtype = resolve_amp(config.amp, device, model)
        # GradScaler only exists for cuda/cpu; fp16 (pre-bf16 GPUs) needs it.
        # XLA is always bf16 or fp32, so the scaler stays a disabled
        # passthrough there.
        scaler = torch.amp.GradScaler(
            "cuda" if device.type == "cuda" else "cpu",
            enabled=amp_enabled and amp_dtype == torch.float16,
        )
        # Lazy XLA queues ops until a barrier; one barrier per optimizer step
        # is the canonical granularity (without it the whole epoch fuses into
        # one unboundedly large graph).
        step_barrier = xla_sync_fn() if device.type == "xla" else None

        n = len(train)
        batch_size = _resolve_batch_size(config, n)
        full_batch = batch_size >= n
        # Permutations are drawn on CPU so runs are reproducible across devices.
        gen = torch.Generator()
        if config.random_state is not None:
            gen.manual_seed(config.random_state)

        result = TrainResult(
            evals_result={es.name: {m.name: [] for m in metrics} for es in eval_sets}
        )
        tracking = config.early_stopping_rounds is not None and bool(eval_sets)
        stopper = (
            EarlyStopper(config.early_stopping_rounds, metrics[0].minimize)
            if tracking
            else None
        )
        best_state: dict[str, Tensor] | None = None

        ema_decay = config.ema_decay
        ema_params: dict[str, Tensor] | None = None
        if ema_decay is not None:
            if not 0.0 < ema_decay < 1.0:
                raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay!r}")
            ema_params = {
                name: p.detach().clone() for name, p in model.named_parameters()
            }

        def train_step(step_model: nn.Module, batch: TabularData) -> Tensor:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                raw = step_model(batch.x_num, batch.x_cat)
                loss_i = objective.per_sample_loss(batch.y, raw.float())
                if batch.weight is not None:
                    loss = (loss_i * batch.weight).sum() / batch.weight.sum()
                else:
                    loss = loss_i.mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            return loss.detach()

        steps_per_epoch = 1 if full_batch else int(np.ceil(n / batch_size))
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
            batches = (
                [None]
                if full_batch
                else torch.randperm(n, generator=gen).to(device).split(batch_size)
            )
            for idx in batches:
                batch = train if idx is None else train.slice(idx)
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
                if step_barrier is not None:
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
            saved_params = (
                _swap_in_params(model, ema_params) if ema_params is not None else None
            )

            for es in eval_sets:
                pred = predict_transformed(
                    run_model, es.data, objective.transform, config.eval_batch_size
                )
                if inverse_target is not None:
                    pred = inverse_target(pred)
                for metric in metrics:
                    result.evals_result[es.name][metric.name].append(
                        metric(es.y_metric, pred)
                    )

            if config.verbose > 0 and (
                epoch % config.verbose == 0 or epoch == config.n_epochs - 1
            ):
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
