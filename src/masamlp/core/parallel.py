"""Multi-GPU ensemble-member sharding.

One worker thread per device; each worker trains its members sequentially,
so a device's RNG stream is never interleaved and runs are reproducible for
a fixed GPU topology. CUDA kernels release the GIL, so distinct devices
train concurrently. Threads only — no multiprocessing: tensors stay
in-process and each split is copied to a device exactly once, shared by all
of that device's members (including the retrieval corpus buffers).

Worker fits use ``seed_scope="device"`` configs; process-global seeding
(``seed_everything``) must never run inside a worker thread. Shared
``objective.torch_modules()`` cannot shard — the caller falls back to the
sequential loop. Built-in objectives/metrics are stateless; custom callables
must be thread-safe to train sharded.
"""

from __future__ import annotations

import contextlib
import threading
import warnings
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor, nn

from masamlp.core.metrics import BaseMetric
from masamlp.core.objectives import BaseObjective
from masamlp.core.trainer import (
    EvalSet,
    Trainer,
    TrainerConfig,
    TrainResult,
    predict_transformed,
)
from masamlp.data.dataset import TabularData


def _run_per_device(
    devices: list[torch.device],
    worker: Callable[[torch.device, list[int], threading.Event], None],
    label: str,
) -> None:
    """One thread per unique device, each handling its member ids in order.
    ``worker`` must raise on failure; the lowest-member-index exception is
    re-raised after every thread has joined."""
    groups: dict[torch.device, list[int]] = {}
    for i, device in enumerate(devices):
        groups.setdefault(device, []).append(i)

    errors: list[tuple[int, BaseException]] = []
    errors_lock = threading.Lock()
    stop = threading.Event()
    if any(device.type == "cuda" for device in groups):
        # Initialize CUDA once on the main thread; concurrent first-time
        # lazy init from worker threads is historically fragile.
        torch.cuda.init()

    def run(device: torch.device, member_ids: list[int]) -> None:
        # Pin the thread's current CUDA device so device-implicit
        # allocations in user code (objectives, custom layers) land on the
        # worker's GPU rather than cuda:0.
        ctx = (
            torch.cuda.device(device.index or 0)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        try:
            with ctx:
                worker(device, member_ids, stop)
        except _MemberError as exc:
            with errors_lock:
                errors.append((exc.member, exc.cause))
            stop.set()

    threads = [
        threading.Thread(target=run, args=(device, ids), name=f"masamlp-{label}-{device}")
        for device, ids in groups.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise min(errors, key=lambda item: item[0])[1]


class _MemberError(Exception):
    def __init__(self, member: int, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.member = member
        self.cause = cause


def fit_members_sharded(
    members: list[nn.Module],
    objective: BaseObjective,
    train: TabularData,
    eval_sets: list[EvalSet],
    metrics: list[BaseMetric],
    configs: list[TrainerConfig],
    devices: list[torch.device],
    inverse_target: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[TrainResult]:
    """Train ensemble members concurrently, one worker thread per device.

    ``configs[i]``/``devices[i]`` belong to ``members[i]``; ``train`` and
    ``eval_sets`` may live on any device (each worker copies them once).
    On any member failure the remaining queues stop and the lowest-index
    exception is re-raised.
    """
    if not (len(members) == len(configs) == len(devices)):
        raise ValueError("members, configs, and devices must have equal lengths")
    if any(config.compile for config in configs):
        warnings.warn("torch.compile is ignored for sharded ensembles", stacklevel=2)
        configs = [replace(config, compile=False) for config in configs]

    results: list[TrainResult | None] = [None] * len(members)

    def worker(device: torch.device, member_ids: list[int], stop: threading.Event) -> None:
        try:
            train_d = train.to(device)
            eval_d = [es.to(device) for es in eval_sets]
            first = members[member_ids[0]]
            candidates = None
            if getattr(first, "has_candidates", False):
                # One corpus copy per device, shared by all its members.
                candidates = (
                    first.cand_x_num.to(device),
                    first.cand_x_cat.to(device),
                    first.cand_y.to(device),
                )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            raise _MemberError(member_ids[0], exc) from exc
        for i in member_ids:
            if stop.is_set():
                return
            try:
                if candidates is not None:
                    members[i].set_candidates(*candidates)
                results[i] = Trainer().fit(
                    members[i], objective, train_d, eval_d, metrics, configs[i], inverse_target
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                raise _MemberError(i, exc) from exc

    _run_per_device(devices, worker, "fit")
    return results  # type: ignore[return-value]  # all filled when no errors


def predict_members_grouped(
    members: list[nn.Module],
    data: TabularData,
    transform: Callable[[Tensor], Tensor],
    batch_size: int = 8192,
) -> list[np.ndarray]:
    """Batched inference for members resident on several devices: one worker
    thread and one data copy per device, member order preserved."""
    from masamlp.core.device import module_device

    devices = [module_device(m) for m in members]
    preds: list[np.ndarray | None] = [None] * len(members)

    def worker(device: torch.device, member_ids: list[int], stop: threading.Event) -> None:
        try:
            data_d = data.to(device)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            raise _MemberError(member_ids[0], exc) from exc
        for i in member_ids:
            if stop.is_set():
                return
            try:
                preds[i] = predict_transformed(members[i], data_d, transform, batch_size)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                raise _MemberError(i, exc) from exc

    _run_per_device(devices, worker, "predict")
    return preds  # type: ignore[return-value]  # all filled when no errors
