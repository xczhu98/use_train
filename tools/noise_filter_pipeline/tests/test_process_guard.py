"""core.process_guard 父死看门狗测试。"""
import os
import threading

import pytest

from noise_filter_pipeline.core.process_guard import is_parent_alive, start_parent_watchdog

pytestmark = pytest.mark.pipeline


def test_is_parent_alive():
    # 当前进程的父进程必然存活
    assert is_parent_alive(os.getppid())
    # 一个不等于当前 ppid 的 pid（如 1，容器/系统环境下 pytest 的父进程不是 init）
    if os.getppid() != 1:
        assert not is_parent_alive(1)


def test_watchdog_calls_exit_on_parent_death():
    # parent_pid 传 1（与真实 ppid 不同），模拟父进程已死；
    # exit_fn 注入 mock，避免真的退出 pytest 进程
    called = threading.Event()

    def fake_exit(code):
        assert code == 0
        called.set()

    if os.getppid() == 1:
        # 极端环境下父进程恰为 init，改用另一个必然不等的 pid
        fake_parent = os.getppid() + 1
    else:
        fake_parent = 1
    t = start_parent_watchdog(fake_parent, interval_s=0.05, exit_fn=fake_exit)
    assert t.daemon
    assert called.wait(timeout=5.0), "看门狗未在父进程死亡时触发退出"
    t.join(timeout=1.0)
    assert not t.is_alive(), "exit_fn 返回后看门狗线程必须终止"
