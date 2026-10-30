"""RIR task worker 与 CPU/GPU 执行器。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

from tools.prepare_data_tools.generators.se.rir_backends import get_backend
from tools.prepare_data_tools.generators.se.rir_config import RIRConfig
from tools.prepare_data_tools.generators.se.rir_output import save_result
from tools.prepare_data_tools.generators.se.rir_plan import (
    effective_mode,
    iter_gpu_tasks,
    iter_tasks,
)


logger = logging.getLogger("generate_simulated_rirs")


def _cpu_worker_generate(task: dict[str, Any]) -> tuple[str, int, int, str] | None:
    """CPU backend 单任务：生成并保存。"""
    try:
        backend = get_backend(task["backend"])
        cfg = RIRConfig(**task["cfg"])
        result = backend.generate_one(task["idx"], cfg, gpu_id=0)
        return save_result(result, task)
    except Exception as exc:
        logger.error("Failed to generate %s #%d: %s", task["backend"], task["idx"], exc)
        return None


def _gpu_worker_init(gpu_id: int | str) -> None:
    """gpuRIR worker 初始化：绑定 CUDA 可见设备。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _gpu_worker_generate(task: dict[str, Any]) -> tuple[str, int, int, str] | None:
    """gpuRIR backend 单任务：生成并保存。"""
    try:
        backend = get_backend("gpurir")
        cfg = RIRConfig(**task["cfg"])
        result = backend.generate_one(task["idx"], cfg, gpu_id=0)
        return save_result(result, task)
    except Exception as exc:
        logger.error(
            "Failed to generate gpurir #%d on GPU %s: %s",
            task["idx"],
            task["gpu_id"],
            exc,
        )
        return None


def run_cpu_backend(
    backend: str,
    field: str,
    n_rirs: int,
    cfg: RIRConfig,
    output_dir: Path,
    output_format: str,
    write_metadata_json: bool,
    workers: int,
    shards: int,
    start_index: int = 0,
    *,
    mode: str | None = None,
) -> list[tuple[str, int, int, str]]:
    """运行 rirgen 或 FRAM CPU backend。"""
    mode = effective_mode(cfg, mode)
    tasks = iter_tasks(
        backend,
        field,
        n_rirs,
        cfg,
        output_dir,
        output_format,
        write_metadata_json,
        shards,
        start_index,
        mode=mode,
    )
    entries: list[tuple[str, int, int, str] | None] = []
    workers = min(max(1, workers), n_rirs) if n_rirs > 0 else 0
    logger.info(
        "[%s/%s/%s] Generating %d RIRs with %d workers",
        backend,
        mode,
        field,
        n_rirs,
        workers,
    )
    if n_rirs == 0:
        return []
    with mp.Pool(workers) as pool:
        entries.extend(pool.imap_unordered(_cpu_worker_generate, tasks))
    valid = [entry for entry in entries if entry is not None]
    logger.info(
        "[%s/%s/%s] Done: %d/%d succeeded",
        backend,
        mode,
        field,
        len(valid),
        n_rirs,
    )
    return valid


def run_gpu_backend(
    field: str,
    n_rirs: int,
    cfg: RIRConfig,
    output_dir: Path,
    output_format: str,
    write_metadata_json: bool,
    gpu_ids: list[int | str],
    shards: int,
    start_index: int = 0,
    *,
    mode: str | None = None,
) -> list[tuple[str, int, int, str]]:
    """运行 gpuRIR，每个 GPU 使用一个 spawn 进程。"""
    mode = effective_mode(cfg, mode)
    if n_rirs == 0:
        return []
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")

    entries: list[tuple[str, int, int, str] | None] = []
    logger.info(
        "[gpurir/%s/%s] Generating %d RIRs on GPUs %s",
        mode,
        field,
        n_rirs,
        gpu_ids,
    )
    ctx = mp.get_context("spawn")
    pools = []
    pool_results = []
    try:
        for gpu_slot, gpu_id in enumerate(gpu_ids):
            if gpu_slot >= n_rirs:
                break
            tasks = iter_gpu_tasks(
                field,
                n_rirs,
                cfg,
                output_dir,
                output_format,
                write_metadata_json,
                gpu_ids,
                gpu_slot,
                shards,
                start_index,
                mode=mode,
            )
            pool = ctx.Pool(1, initializer=_gpu_worker_init, initargs=(gpu_id,))
            pools.append(pool)
            pool_results.append(pool.imap_unordered(_gpu_worker_generate, tasks))
        pending = list(pool_results)
        while pending:
            for results in list(pending):
                try:
                    entries.append(next(results))
                except StopIteration:
                    pending.remove(results)
    except BaseException:
        for pool in pools:
            pool.terminate()
        raise
    else:
        for pool in pools:
            pool.close()
    finally:
        for pool in pools:
            pool.join()

    valid = [entry for entry in entries if entry is not None]
    logger.info(
        "[gpurir/%s/%s] Done: %d/%d succeeded",
        mode,
        field,
        len(valid),
        n_rirs,
    )
    return valid


__all__ = ["run_cpu_backend", "run_gpu_backend"]
