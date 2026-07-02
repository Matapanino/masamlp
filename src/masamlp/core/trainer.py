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

from masamlp.core.device import maybe_compile, resolve_amp, resolve_device, set_threads
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
    lr_scheduler: str = "none"
    grad_clip: float | None = None
    device: str = "auto"
    amp: str | bool = "auto"
    compile: bool = False
    early_stopping_rounds: int | None = None
    random_state: int | None = None
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
    output has a single column."""
    model.eval()
    outs: list[Tensor] = []
    device = data.x_num.device
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(data)), device=device)
            batch = data.slice(idx)
            outs.append(transform(model(batch.x_num, batch.x_cat)).float().cpu())
    pred = torch.cat(outs).numpy()
    return pred[:, 0] if pred.ndim == 2 and pred.shape[1] == 1 else pred


def _make_optimizer(
    name: str, params: list[Tensor], lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}. Expected 'adamw', 'adam', or 'sgd'")


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
        seed_everything(config.random_state)
        device = resolve_device(config.device)
        if device.type == "cpu":
            set_threads(config.n_threads)

        model.to(device)
        extra_modules = objective.torch_modules()
        for module in extra_modules:
            module.to(device)
        train = train.to(device)
        eval_sets = [EvalSet(es.name, es.data.to(device), es.y_metric) for es in eval_sets]

        run_model = maybe_compile(model, config.compile, device)
        params = [p for p in model.parameters() if p.requires_grad]
        for module in extra_modules:
            params += [p for p in module.parameters() if p.requires_grad]
        optimizer = _make_optimizer(
            config.optimizer, params, config.learning_rate, config.weight_decay
        )
        scheduler = None
        if config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.n_epochs
            )
        elif config.lr_scheduler != "none":
            raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}")

        amp_enabled, amp_dtype = resolve_amp(config.amp, device)
        # GradScaler only exists for cuda/cpu; fp16 (pre-bf16 GPUs) needs it.
        scaler = torch.amp.GradScaler(
            "cuda" if device.type == "cuda" else "cpu",
            enabled=amp_enabled and amp_dtype == torch.float16,
        )

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
                epoch_loss += loss * (len(batch) / n)
            # One host sync per epoch keeps the GPU pipeline full.
            epoch_loss_val = float(epoch_loss)
            if not np.isfinite(epoch_loss_val):
                raise ValueError(
                    f"Training loss became non-finite at epoch {epoch}; "
                    "try a lower learning_rate or grad_clip"
                )
            if scheduler is not None:
                scheduler.step()

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

            if stopper is not None:
                monitor = result.evals_result[eval_sets[0].name][metrics[0].name][-1]
                if stopper.update(monitor, epoch):
                    best_state = {
                        k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                    }
                if stopper.should_stop:
                    break

        if stopper is not None and best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
            result.best_iteration = stopper.best_epoch
            result.best_score = float(stopper.best_value)
        return result
