"""子进程父死看门狗。

mp spawn / ProcessPoolExecutor 的 worker 阻塞在 ``queue.get()`` 时无法自查
父进程状态（spawn 会把 queue 双端 fd 都复制进子进程，父进程死亡不会产生
EOF），父进程被 SIGKILL 等方式杀死后 worker 会变孤儿永久滞留。
本模块提供 daemon 看门狗线程：定期比对 ppid，父进程死亡即自杀。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


def is_parent_alive(parent_pid: int) -> bool:
    """父进程是否仍在。父进程死亡后子进程会被 reparent，ppid 发生变化。"""
    return os.getppid() == parent_pid


def start_parent_watchdog(
    parent_pid: int,
    interval_s: float = 30.0,
    exit_fn: Callable[[int], None] = os._exit,
    name: str = "parent-watchdog",
) -> threading.Thread:
    """启动 daemon 看门狗线程：父进程死亡时调用 ``exit_fn(0)`` 让本进程退出。

    用于阻塞在 queue.get() 等无法自查父进程状态的 worker 子进程。
    daemon 线程不阻止进程正常退出；``exit_fn`` 可注入以便测试。

    Returns:
        已启动的看门狗线程。
    """

    def _watch() -> None:
        while True:
            time.sleep(interval_s)
            if not is_parent_alive(parent_pid):
                # print 兜底：子进程可能未配置 logging
                msg = f"父进程 {parent_pid} 已退出，子进程自行退出避免成为孤儿"
                logger.warning(msg)
                print(msg, flush=True)
                exit_fn(0)
                return

    t = threading.Thread(target=_watch, daemon=True, name=name)
    t.start()
    return t
