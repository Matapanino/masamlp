"""Vectorized ensemble training via ``torch.func`` — pytabkit's speed trick.

All members' parameters are stacked into single tensors and every training
step runs one vmapped forward/backward for the whole ensemble, sharing the
batch pipeline. Semantics vs ``ens_mode="loop"``: members still differ by
initialization (seeded per member) and dropout masks (``randomness=
"different"``), but they see the *same* batch order; per-member best-epoch
tracking and restoration are preserved. Total loss is the sum of member
losses, so each member's gradients match independent training.

Not supported here (use the loop): models with BatchNorm running statistics,
retrieval models (candidate buffers / batch-index protocol), custom
``nn.Module`` objectives, ``grad_clip`` (a global clip would couple
members), AMP, and ``torch.compile``.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state
from torch.nn.modules.batchnorm import _BatchNorm

from masamlp.core.device import resolve_amp, resolve_device, set_threads
from masamlp.core.metrics import BaseMetric
from masamlp.core.objectives import BaseObjective
from masamlp.core.trainer import (
    EarlyStopper,
    EvalSet,
    TrainerConfig,
    TrainResult,
    _coslog4,
    _make_optimizer,
    _resolve_batch_size,
    flat_cos,
)
from masamlp.data.dataset import TabularData
from masamlp.utils.random import seed_everything


def check_vectorizable(model: nn.Module, model_name: str | None = None) -> None:
    where = f" {model_name!r}" if model_name else ""
    if getattr(model, "wants_batch_indices", False) or hasattr(model, "set_candidates"):
        raise ValueError(
            f"retrieval models{where} cannot train vectorized; use ens_mode='loop'"
        )
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            raise ValueError(
                f"model{where} has BatchNorm running statistics and cannot train "
                "vectorized; use ens_mode='loop' (BatchNorm-free models such as "
                "grn/realmlp/ft_transformer/gandalf/lnn support ens_mode='vectorized')"
            )


def _name_factors(model: nn.Module) -> dict[str, tuple[float, float]]:
    """(lr_factor, wd_factor) per parameter name, from the member-0 grouping."""
    by_id: dict[int, tuple[float, float]] = {}
    if hasattr(model, "param_groups"):
        for group in model.param_groups():
            for p in group["params"]:
                by_id[id(p)] = (group.get("lr_factor", 1.0), group.get("wd_factor", 1.0))
    return {
        name: by_id.get(id(p), (1.0, 1.0)) for name, p in model.named_parameters()
    }


def fit_vectorized(
    models: list[nn.Module],
    objective: BaseObjective,
    train: TabularData,
    eval_sets: list[EvalSet],
    metrics: list[BaseMetric],
    config: TrainerConfig,
    inverse_target: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[TrainResult]:
    check_vectorizable(models[0])
    if objective.torch_modules():
        raise ValueError("nn.Module objectives are not supported with ens_mode='vectorized'")
    if config.grad_clip is not None:
        raise ValueError("grad_clip is not supported with ens_mode='vectorized'")
    if config.compile:
        warnings.warn("torch.compile is ignored for vectorized ensembles", stacklevel=2)
    device = resolve_device(config.device)
    if device.type == "xla":
        # torch.func vmap over XLA lazy tensors is unvalidated; failing loud
        # beats silently falling back to the loop mode the user didn't ask for.
        raise ValueError(
            "ens_mode='vectorized' is not supported on XLA devices; "
            "use the default loop mode"
        )
    if device.type == "cpu":
        set_threads(config.n_threads)
    if resolve_amp(config.amp, device, models[0])[0]:
        warnings.warn(
            "AMP is disabled for vectorized ensembles; training in float32", stacklevel=2
        )
    if config.amp_predict not in (False, "off"):
        warnings.warn(
            "amp_predict is ignored for vectorized ensembles; predicting in float32",
            stacklevel=2,
        )

    seed_everything(config.random_state)
    k = len(models)
    for model in models:
        model.to(device)
    train = train.to(device)
    eval_sets = [es.to(device) for es in eval_sets]

    stacked_params, stacked_buffers = stack_module_state(models)
    params = {n: p.detach().clone().requires_grad_(True) for n, p in stacked_params.items()}
    buffers = {n: b.detach().clone() for n, b in stacked_buffers.items()}
    base = copy.deepcopy(models[0]).to("meta")

    def fmodel(p: dict, b: dict, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        return functional_call(base, (p, b), (x_num, x_cat))

    vmodel = torch.vmap(fmodel, in_dims=(0, 0, None, None), randomness="different")

    factors = _name_factors(models[0])
    group_map: dict[tuple[float, float], list[torch.Tensor]] = {}
    for name, p in params.items():
        group_map.setdefault(factors.get(name, (1.0, 1.0)), []).append(p)
    groups = [
        {
            "params": ps,
            "lr_factor": lf,
            "wd_factor": wf,
            "lr": config.learning_rate * lf,
            "weight_decay": config.weight_decay * wf,
        }
        for (lf, wf), ps in group_map.items()
    ]
    optimizer = _make_optimizer(
        config.optimizer, groups, config.learning_rate, config.weight_decay, config.betas
    )
    scheduler = None
    per_step_schedule = None
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    elif config.lr_scheduler == "coslog4":
        per_step_schedule = _coslog4
    elif config.lr_scheduler != "none":
        raise ValueError(f"Unknown lr_scheduler {config.lr_scheduler!r}")
    wd_scheduled = config.weight_decay_schedule == "flat_cos"
    model_has_schedule = hasattr(base, "set_schedule_t")

    n = len(train)
    batch_size = _resolve_batch_size(config, n)
    full_batch = batch_size >= n
    gen = torch.Generator()
    if config.random_state is not None:
        gen.manual_seed(config.random_state)
    steps_per_epoch = 1 if full_batch else int(np.ceil(n / batch_size))
    total_steps = max(1, config.n_epochs * steps_per_epoch)
    global_step = 0

    results = [
        TrainResult(evals_result={es.name: {m.name: [] for m in metrics} for es in eval_sets})
        for _ in range(k)
    ]
    tracking = config.early_stopping_rounds is not None and bool(eval_sets)
    stoppers = (
        [EarlyStopper(config.early_stopping_rounds, metrics[0].minimize) for _ in range(k)]
        if tracking
        else None
    )
    best_slices: list[dict[str, torch.Tensor] | None] = [None] * k

    def predict_members(data: TabularData) -> list[np.ndarray]:
        base.eval()
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(data), config.eval_batch_size):
                idx = torch.arange(
                    start, min(start + config.eval_batch_size, len(data)), device=device
                )
                batch = data.slice(idx)
                chunks.append(vmodel(params, buffers, batch.x_num, batch.x_cat))
        raw = torch.cat(chunks, dim=1)  # (k, n, out)
        preds = []
        for j in range(k):
            pred = objective.transform(raw[j]).float().cpu().numpy()
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred[:, 0]
            if inverse_target is not None:
                pred = inverse_target(pred)
            preds.append(pred)
        return preds

    for epoch in range(config.n_epochs):
        base.train()
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
                base.set_schedule_t(t)
            global_step += 1

            raw = vmodel(params, buffers, batch.x_num, batch.x_cat)  # (k, n, out)
            member_losses = []
            for j in range(k):
                loss_i = objective.per_sample_loss(batch.y, raw[j].float())
                if batch.weight is not None:
                    member_losses.append((loss_i * batch.weight).sum() / batch.weight.sum())
                else:
                    member_losses.append(loss_i.mean())
            # Sum keeps each member's gradients identical to training alone.
            loss = torch.stack(member_losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() / k * (len(batch) / n)

        if not np.isfinite(float(epoch_loss)):
            raise ValueError(
                f"Training loss became non-finite at epoch {epoch}; "
                "try a lower learning_rate"
            )
        if scheduler is not None:
            scheduler.step()

        for es in eval_sets:
            for j, pred in enumerate(predict_members(es.data)):
                for metric in metrics:
                    results[j].evals_result[es.name][metric.name].append(
                        metric(es.y_metric, pred)
                    )

        if config.verbose > 0 and (epoch % config.verbose == 0 or epoch == config.n_epochs - 1):
            parts = [f"[{epoch}] mean_train_loss: {float(epoch_loss):.5f}"]
            if eval_sets:
                name0, metric0 = eval_sets[0].name, metrics[0].name
                values = [results[j].evals_result[name0][metric0][-1] for j in range(k)]
                parts.append(f"{name0}-{metric0} (mean over {k}): {np.mean(values):.5f}")
            print("  ".join(parts))

        if stoppers is not None:
            for j, stopper in enumerate(stoppers):
                monitor = results[j].evals_result[eval_sets[0].name][metrics[0].name][-1]
                if stopper.update(monitor, epoch):
                    best_slices[j] = {
                        name: params[name][j].detach().to("cpu", copy=True)
                        for name in params
                    }
            if all(stopper.should_stop for stopper in stoppers):
                break

    if stoppers is not None:
        with torch.no_grad():
            for j, best in enumerate(best_slices):
                if best is not None:
                    for name in params:
                        params[name][j].copy_(best[name].to(device))
            for j, stopper in enumerate(stoppers):
                results[j].best_iteration = stopper.best_epoch
                results[j].best_score = float(stopper.best_value)

    # Unstack back into the individual member modules.
    with torch.no_grad():
        for j, model in enumerate(models):
            state = {name: params[name][j] for name in params}
            state.update({name: buffers[name][j] for name in buffers})
            model.load_state_dict(state)
            model.eval()
    return results
