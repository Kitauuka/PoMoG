from __future__ import annotations

import os
from typing import Any, Dict

import torch


def _loader_worker_init(_worker_id: int) -> None:
    torch.set_num_threads(int(os.environ.get("POMOG_WORKER_TORCH_THREADS", "1")))


def build_dataloader_kwargs(
    config: Dict[str, Any],
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
) -> Dict[str, Any]:
    num_workers = int(config.get("num_workers", 0))
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size if batch_size is not None else config.get("batch_size", 1)),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
    }
    if bool(config.get("pin_memory", True)) and torch.cuda.is_available():
        kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        kwargs["worker_init_fn"] = _loader_worker_init
    return kwargs
