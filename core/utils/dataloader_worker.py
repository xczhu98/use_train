"""Shared DataLoader worker_init_fn (SE / AEC training)."""

from __future__ import annotations

import importlib
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _clear_worker_cache(
    worker_id: int,
    module_path: str,
    attribute: str,
    *,
    absent_module_prefix: str | None = None,
) -> None:
    """清空 worker 继承来的模块级缓存，且不静默吞掉导入/清理失败。"""
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if absent_module_prefix is not None and (exc.name or "").startswith(
            absent_module_prefix
        ):
            logger.debug("cache module %s is unavailable: %s", module_path, exc)
        else:
            logger.warning(
                "Failed to import %s in worker %d: %s",
                module_path,
                worker_id,
                exc,
                exc_info=True,
            )
        return
    cache = getattr(module, attribute, None)
    if cache is None:
        logger.debug("%s has no %s in worker %d", module_path, attribute, worker_id)
        return
    try:
        cache.clear()
    except Exception as exc:
        logger.warning(
            "Failed to clear %s.%s in worker %d: %s",
            module_path,
            attribute,
            worker_id,
            exc,
            exc_info=True,
        )


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Seed per-worker `random` / `numpy` so workers are independent.

    Mixes PyTorch's per-worker `initial_seed`, `worker_id`, and process id.
    ``torch.initial_seed()`` is derived from the DataLoader ``generator``,
    which advances each epoch, so worker-level randomness changes across
    epochs even when ``persistent_workers=True``.

    Also clears the byte-budget LRU audio caches.  Under ``fork``, workers
    inherit parent-populated cache entries whose numpy arrays may be CoW-
    shared with the parent process.  A fresh clear here ensures every worker
    starts clean; the epoch-change detection in ``_worker_maybe_clear_caches``
    handles subsequent epoch boundaries.
    """
    t = torch.initial_seed() % 2**32
    s = (t ^ ((worker_id + 1) * 0x9E3779B9) ^ ((os.getpid() & 0xFFFF) << 16)) & 0xFFFFFFFF
    np.random.seed(s)
    random.seed(s)

    # Clear module-level audio caches inherited from parent under fork.
    # The byte-budget LRU (_ByteLRUCache) is process-shared under fork;
    # clearing here prevents workers from serving stale parent-loaded entries.
    _clear_worker_cache(worker_id, "core.data.simulation_audio_utils", "_AUDIO_CACHE")
    # AEC 立体声缓存位于任务包：这是 core→tasks 的 lazy-import 兼容 seam，
    # 只有任务子树整体缺失才允许跳过，任务包内部的导入失败必须可见。
    _clear_worker_cache(
        worker_id,
        "tasks.aec.data.dynamic_dataset",
        "_AEC_STEREO_CACHE",
        absent_module_prefix="tasks.aec",
    )
