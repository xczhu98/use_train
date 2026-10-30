"""客观指标进程池的真实 spawn 生命周期测试。"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

import core.metrics.batch as batch_mod

pytestmark = [pytest.mark.integration, pytest.mark.subprocess]


def _hang_until_terminated():
    """真实 spawn worker：等待测试中的有界 shutdown 回收。"""
    while True:
        time.sleep(1.0)


def _complete_with_slow_normal_teardown(delay: float) -> int:
    """任务完成后模拟 native runtime 的正常、有界 worker teardown。"""
    import multiprocessing.util

    multiprocessing.util.Finalize(
        None,
        time.sleep,
        args=(delay,),
        exitpriority=0,
    )
    return 7


def test_completed_real_executor_gets_bounded_normal_teardown_grace(
    monkeypatch, caplog
):
    """future 全部完成后应允许正常退出，不得误报 survived 或强杀 worker。"""
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    future = pool.submit(_complete_with_slow_normal_teardown, 1.2)
    assert future.result(timeout=10.0) == 7
    process = next(iter(pool._processes.values()))
    monkeypatch.setattr(
        batch_mod,
        "_INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC",
        5.0,
        raising=False,
    )

    with caplog.at_level(logging.ERROR, logger="core.metrics.batch"):
        batch_mod._shutdown_intrusive_pool(pool, terminate=False)

    assert process.exitcode == 0
    assert "survived terminate/kill" not in caplog.text


def test_shutdown_pool_bounded_reaps_real_spawn_worker(monkeypatch):
    """正常 shutdown 也必须有界；卡死 worker 应在 grace 后被实际终止。"""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_hang_until_terminated)
    process.start()

    class _Pool:
        def __init__(self):
            self._processes = {process.pid: process}
            self.shutdown_calls = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    pool = _Pool()
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC", 0.1)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC", 0.1)
    try:
        batch_mod._shutdown_intrusive_pool(pool, terminate=False)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)

    # shutdown 返回本身即证明 grace 生效（无界 join 会在此挂死）；不再叠加墙钟
    # 断言，避免高负载 runner 上把调度抖动误判为回归。
    assert pool.shutdown_calls == [(False, False)]
    assert not process.is_alive()
