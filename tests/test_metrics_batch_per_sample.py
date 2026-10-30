"""JAE-M06 回归测试：compute_metrics_batch 逐样本 AQ 返回。

- 默认 return_per_sample=False 时返回类型与聚合值完全不变（Trainer 等旧调用方兼容）；
- return_per_sample=True 时返回 (results, per_sample)，per_sample 与裁剪后 pairs
  索引对齐，仅含有效值，且与聚合 sum/count 严格一致；
- 覆盖侵入式顺序/并行两条路径与 MOS（dict 子分数 / 标量 / 失败 None）路径。
"""

from __future__ import annotations

import numpy as np
import pytest

import concurrent.futures
import sys
import time
import types

import core.metrics.batch as batch_mod
from core.metrics.batch import compute_metrics_batch
from core.metrics.intrusive import _INTRUSIVE_FNS, compute_pesq

SR = 16000


def _make_pairs(n: int, seed: int = 0, sec: float = 1.0):
    """构造 n 对 (ref, enh, fs)：enh 为 ref 的衰减加噪版本，保证指标有限。"""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        ref = rng.standard_normal(int(SR * sec)).astype(np.float32)
        enh = (0.9 * ref + 0.05 * rng.standard_normal(ref.size)).astype(np.float32)
        pairs.append((ref, enh, SR))
    return pairs


def _expected_sisdr(pairs):
    fn, _ = _INTRUSIVE_FNS["SI-SDR"]
    return [fn(r, e, fs) for r, e, fs in pairs]


def test_compute_pesq_rejects_return_value_error_codes(monkeypatch):
    """pesq RETURN_VALUES 的负错误码不是有效分数，不得计入 coverage。"""
    fake_error = types.SimpleNamespace(
        RETURN_VALUES=1,
        SUCCESS=0,
        UNKNOWN=-1,
        INVALID_SAMPLE_RATE=-2,
        OUT_OF_MEMORY_REF=-3,
        OUT_OF_MEMORY_DEG=-4,
        OUT_OF_MEMORY_TMP=-5,
        BUFFER_TOO_SHORT=-6,
        NO_UTTERANCES_DETECTED=-7,
    )
    monkeypatch.setitem(
        sys.modules,
        "pesq",
        types.SimpleNamespace(PesqError=fake_error, pesq=lambda *_args, **_kwargs: -6),
    )
    audio = np.zeros(SR, dtype=np.float32)

    assert compute_pesq(audio, audio, SR) is None


def test_default_return_type_unchanged():
    """默认返回仍为聚合 dict（无 per_sample），键为 sum + _count_。"""
    pairs = _make_pairs(3)
    raw = compute_metrics_batch(pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=False)
    assert isinstance(raw, dict)
    assert set(raw.keys()) == {"SI-SDR", "_count_SI-SDR"}
    assert raw["_count_SI-SDR"] == 3.0


@pytest.mark.parametrize("n_pairs", [1, 2, 3])
def test_public_small_intrusive_batch_stays_sequential(monkeypatch, n_pairs):
    """公共 API 默认对 1–3 条顺序计算，避免 stdin/notebook 强制 spawn。"""
    monkeypatch.setattr(
        batch_mod,
        "ProcessPoolExecutor",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("small public batch must not construct a process pool")
        ),
    )
    pairs = _make_pairs(n_pairs)
    expected = _expected_sisdr(pairs)

    raw = compute_metrics_batch(pairs=pairs, metrics=["SI-SDR"])

    assert raw["SI-SDR"] == pytest.approx(sum(expected))
    assert raw["_count_SI-SDR"] == float(n_pairs)


def test_per_sample_sequential_intrusive():
    """顺序路径：per_sample 与输入对齐，逐样本值与直接计算一致，sum 等于聚合值。"""
    pairs = _make_pairs(3)
    raw, per_sample = compute_metrics_batch(
        pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=False,
        return_per_sample=True,
    )
    expected = _expected_sisdr(pairs)
    assert len(per_sample) == len(pairs)
    for rec, exp in zip(per_sample, expected):
        assert rec["SI-SDR"] == pytest.approx(exp)
    assert raw["SI-SDR"] == pytest.approx(sum(expected))


def test_per_sample_parallel_intrusive_matches_sequential():
    """并行路径（>=4 样本走 ProcessPool）：per_sample 顺序保序，结果与顺序路径一致。"""
    pairs = _make_pairs(5)
    raw_s, ps_s = compute_metrics_batch(
        pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=False,
        return_per_sample=True,
    )
    raw_p, ps_p = compute_metrics_batch(
        pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=True,
        return_per_sample=True,
    )
    assert len(ps_p) == len(pairs)
    for rec_s, rec_p in zip(ps_s, ps_p):
        assert rec_p["SI-SDR"] == pytest.approx(rec_s["SI-SDR"])
    assert raw_p["SI-SDR"] == pytest.approx(raw_s["SI-SDR"])


def test_per_sample_zero_length_pairs_trimmed():
    """零长 pair 在裁剪时被丢弃：per_sample 与裁剪后 pairs 对齐（长度减一）。"""
    pairs = _make_pairs(3)
    pairs.insert(1, (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), SR))
    raw, per_sample = compute_metrics_batch(
        pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=False,
        return_per_sample=True,
    )
    assert len(per_sample) == 3
    assert raw["_count_SI-SDR"] == 3.0
    expected = _expected_sisdr([p for p in pairs if len(p[0]) > 0])
    for rec, exp in zip(per_sample, expected):
        assert rec["SI-SDR"] == pytest.approx(exp)


def test_per_sample_mos_dict_scalar_and_failure(monkeypatch):
    """MOS 路径：dict 子分数 / 标量结果逐样本记录；None 与 NaN 失败样本缺键。"""
    n = 4

    def fake_mos_batch(model, name, items, batch_size=16, device="cpu", **kw):
        assert len(items) == n
        if name == "DNSMOS":
            return [
                {"OVRL": 3.0, "SIG": 4.0, "BAK": 3.5},
                {"OVRL": 3.1, "SIG": 4.1, "BAK": None},  # BAK 无效 → 该键缺失
                None,                                     # 整条失败 → 三个键全缺
                {"OVRL": 3.3, "SIG": 4.3, "BAK": 3.8},
            ]
        if name == "MYMOS":
            return [1.0, float("nan"), 3.0, 4.0]  # 标量；nan → 失败
        raise AssertionError(name)

    monkeypatch.setattr(batch_mod, "compute_mos_batch", fake_mos_batch)
    pairs = _make_pairs(n)
    raw, per_sample = compute_metrics_batch(
        pairs=pairs, metrics=[],
        mos_models={"DNSMOS": object(), "MYMOS": object()},
        return_per_sample=True,
    )
    assert len(per_sample) == n
    # 子分数键
    assert per_sample[0]["DNSMOS_OVRL"] == pytest.approx(3.0)
    assert "DNSMOS_BAK" not in per_sample[1]
    assert not {k for k in per_sample[2] if k.startswith("DNSMOS")}  # 该样本 DNSMOS 整条失败 → 三个键全缺
    assert per_sample[3]["DNSMOS_BAK"] == pytest.approx(3.8)
    # 标量模型
    assert per_sample[0]["MYMOS"] == pytest.approx(1.0)
    assert "MYMOS" not in per_sample[1]
    # 聚合与逐样本一致
    assert raw["DNSMOS_OVRL"] == pytest.approx(3.0 + 3.1 + 3.3)
    assert raw["_count_DNSMOS_OVRL"] == 3.0
    assert raw["_count_DNSMOS_BAK"] == 2.0
    assert raw["MYMOS"] == pytest.approx(1.0 + 3.0 + 4.0)
    assert raw["_count_MYMOS"] == 3.0


def test_custom_scalar_mos_uses_canonical_key_in_strict_mode(monkeypatch):
    monkeypatch.setattr(
        batch_mod,
        "compute_mos_batch",
        lambda model, name, items, **kwargs: [3.25] * len(items),
    )

    raw = compute_metrics_batch(
        pairs=_make_pairs(2),
        metrics=[],
        mos_models={"MYMOS": object()},
        strict=True,
    )

    assert raw == {"MYMOS": 6.5, "_count_MYMOS": 2.0}


def test_custom_scalar_mos_strict_failure_reports_canonical_key(monkeypatch):
    monkeypatch.setattr(
        batch_mod,
        "compute_mos_batch",
        lambda model, name, items, **kwargs: [None] * len(items),
    )

    with pytest.raises(RuntimeError) as exc_info:
        compute_metrics_batch(
            pairs=_make_pairs(2),
            metrics=[],
            mos_models={"MYMOS": object()},
            strict=True,
        )

    message = str(exc_info.value)
    assert "MYMOS(fail=2)" in message
    assert "MYMOS_MYMOS" not in message


def test_parallel_intrusive_uses_spawn_context(monkeypatch):
    """CUDA/DDP 初始化后禁止 fork 指标 worker：pool 必须显式使用 spawn。"""
    captured = {}

    class _Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _Pool:
        def __init__(self, *, max_workers, mp_context):
            captured["max_workers"] = max_workers
            captured["mp_context"] = mp_context

        def submit(self, fn, task):
            assert fn is batch_mod._compute_intrusive_one
            return _Future(fn(task))

        def shutdown(self, wait=True, cancel_futures=False):
            captured["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(batch_mod.multiprocessing, "get_context", lambda name: f"ctx:{name}")
    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(
        batch_mod,
        "wait",
        lambda pending, timeout=None, return_when=None: (set(pending), set()),
    )
    pairs = _make_pairs(4)
    raw = compute_metrics_batch(pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=True)

    assert captured["mp_context"] == "ctx:spawn"
    assert captured["shutdown"] == (False, False)
    assert raw["_count_SI-SDR"] == 4.0


def test_parallel_intrusive_timeout_terminates_pool_and_skips_batch(monkeypatch):
    """单样本也必须进 pool；无进展时有界终止，禁止主进程顺序重算。"""
    class _PendingFuture:
        def result(self):
            raise AssertionError("pending future must not be consumed")

    class _Pool:
        def __init__(self, *, max_workers, mp_context):
            self.shutdown_calls = []

        def submit(self, fn, task):
            return _PendingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    pools = []

    def _make_pool(**kwargs):
        pool = _Pool(**kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", _make_pool)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_TASK_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_WAIT_POLL_SEC", 0.001)
    monkeypatch.setattr(
        batch_mod,
        "wait",
        lambda pending, timeout=None, return_when=None: (set(), pending),
    )

    pairs = _make_pairs(1)
    raw = compute_metrics_batch(
        pairs=pairs,
        metrics=["SI-SDR"],
        parallel_intrusive=True,
        should_abort=lambda: False,
    )

    assert len(pools) == 1
    assert pools[0].shutdown_calls == [(False, True)]
    assert raw == {}


def test_parallel_intrusive_abort_request_terminates_pool_and_skips_batch(monkeypatch):
    """shutdown flag 必须在等待中可观察；触发后立即终止 pool 并返回空结果。"""
    class _PendingFuture:
        def result(self):
            raise AssertionError("pending future must not be consumed")

    class _Pool:
        def __init__(self, *, max_workers, mp_context):
            self.shutdown_calls = []

        def submit(self, fn, task):
            return _PendingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    pools = []

    def _make_pool(**kwargs):
        pool = _Pool(**kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", _make_pool)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_TASK_TIMEOUT_SEC", 60.0)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_WAIT_POLL_SEC", 0.001)
    monkeypatch.setattr(
        batch_mod,
        "wait",
        lambda pending, timeout=None, return_when=None: (set(), pending),
    )

    pairs = _make_pairs(4)
    raw = compute_metrics_batch(
        pairs=pairs,
        metrics=["SI-SDR"],
        parallel_intrusive=True,
        should_abort=lambda: True,
    )

    # flag 在 pool 创建前即已置位，快速路径不应启动任何 worker。
    assert pools == []
    assert raw == {}


def test_parallel_intrusive_abort_becomes_visible_while_waiting(monkeypatch):
    """pool 创建后才观察到 shutdown 时，必须终止 worker 且不顺序重放。"""
    class _PendingFuture:
        pass

    class _Pool:
        def __init__(self, *, max_workers, mp_context):
            self.shutdown_calls = []

        def submit(self, fn, task):
            return _PendingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    pools = []

    def _make_pool(**kwargs):
        pool = _Pool(**kwargs)
        pools.append(pool)
        return pool

    abort_checks = 0

    def should_abort():
        nonlocal abort_checks
        abort_checks += 1
        return abort_checks >= 2

    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", _make_pool)
    monkeypatch.setattr(batch_mod, "wait", lambda pending, **kwargs: (set(), pending))

    raw = compute_metrics_batch(
        pairs=_make_pairs(4),
        metrics=["SI-SDR"],
        parallel_intrusive=True,
        should_abort=should_abort,
    )

    assert pools[0].shutdown_calls == [(False, True)]
    assert raw == {}


def test_parallel_intrusive_timeout_keeps_mos_and_strict_contract(monkeypatch):
    """侵入式 timeout 只跳侵入式项，MOS 继续；strict 仍报告缺失指标。"""
    class _PendingFuture:
        pass

    class _Pool:
        def __init__(self, *, max_workers, mp_context):
            pass

        def submit(self, fn, task):
            return _PendingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_TASK_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(batch_mod, "_INTRUSIVE_WAIT_POLL_SEC", 0.001)
    monkeypatch.setattr(batch_mod, "wait", lambda pending, **kwargs: (set(), pending))
    monkeypatch.setattr(
        batch_mod,
        "compute_mos_batch",
        lambda model, name, items, **kwargs: [3.0] * len(items),
    )
    pairs = _make_pairs(4)

    raw = compute_metrics_batch(
        pairs=pairs,
        metrics=["SI-SDR"],
        mos_models={"MYMOS": object()},
        parallel_intrusive=True,
    )
    assert raw["MYMOS"] == pytest.approx(12.0)
    assert raw["_count_MYMOS"] == 4.0
    assert "SI-SDR" not in raw

    with pytest.raises(RuntimeError, match="SI-SDR"):
        compute_metrics_batch(
            pairs=pairs,
            metrics=["SI-SDR"],
            parallel_intrusive=True,
            strict=True,
        )


def test_parallel_intrusive_broken_pool_terminates_and_skips_batch(monkeypatch):
    """worker 崩溃时终止 pool 并跳过，禁止在训练主进程重放故障指标。"""
    class _FailingFuture:
        def result(self):
            raise concurrent.futures.process.BrokenProcessPool("synthetic crash")

    class _Pool:
        def __init__(self, *, max_workers, mp_context):
            self.shutdown_calls = []

        def submit(self, fn, task):
            return _FailingFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append((wait, cancel_futures))

    pools = []

    def _make_pool(**kwargs):
        pool = _Pool(**kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", _make_pool)
    monkeypatch.setattr(
        batch_mod,
        "wait",
        lambda pending, timeout=None, return_when=None: (set(pending), set()),
    )

    pairs = _make_pairs(4)
    raw = compute_metrics_batch(pairs=pairs, metrics=["SI-SDR"], parallel_intrusive=True)

    assert pools[0].shutdown_calls == [(False, True)]
    assert raw == {}


def test_shutdown_pool_cleanup_api_failures_are_noexcept(monkeypatch, caplog):
    """executor/process 生命周期 API 全部异常时，清理仍必须 noexcept。"""
    calls = []

    class _Process:
        @property
        def pid(self):
            calls.append("pid")
            raise RuntimeError("pid failed")

        def is_alive(self):
            calls.append("is_alive")
            raise RuntimeError("is_alive failed")

        def join(self, timeout=None):
            calls.append("join")
            raise RuntimeError("join failed")

        def terminate(self):
            calls.append("terminate")
            raise RuntimeError("terminate failed")

        def kill(self):
            calls.append("kill")
            raise RuntimeError("kill failed")

    process = _Process()

    class _Pool:
        _processes = {1: process}

        def shutdown(self, wait=True, cancel_futures=False):
            calls.append("shutdown")
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(batch_mod, "_INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC", 0.001)
    monkeypatch.setattr(
        batch_mod, "_INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC", 0.001
    )
    with caplog.at_level("ERROR", logger="core.metrics.batch"):
        batch_mod._shutdown_intrusive_pool(_Pool(), terminate=False)

    assert {"shutdown", "is_alive", "join", "terminate", "kill", "pid"} <= set(calls)
    assert "survived terminate/kill" in caplog.text


def test_shutdown_pool_process_lookup_failure_is_noexcept(caplog):
    """读取 executor 私有 process 集合失败也不得传播到训练主流程。"""
    class _Pool:
        @property
        def _processes(self):
            raise RuntimeError("process lookup failed")

        def shutdown(self, wait=True, cancel_futures=False):
            raise RuntimeError("shutdown failed")

    with caplog.at_level("ERROR", logger="core.metrics.batch"):
        batch_mod._shutdown_intrusive_pool(_Pool(), terminate=True)

    assert "process handles" in caplog.text
    assert "shut down intrusive metric pool" in caplog.text


def test_shutdown_pool_join_budget_is_fair_across_workers(monkeypatch, caplog):
    """慢首 worker 不得耗尽共享 join 预算并饿死后续 worker。"""
    class _Process:
        def __init__(self, pid):
            self.pid = pid
            self.alive = True
            self.phase = "grace"

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.phase = "terminate"

        def kill(self):
            self.phase = "kill"

        def join(self, timeout=None):
            timeout = float(timeout or 0.0)
            if timeout > 0:
                time.sleep(timeout)
            # 模拟 native teardown：kill 后仍需获得正 join 预算才能完成回收。
            if self.phase == "kill" and timeout > 0:
                self.alive = False

    processes = [_Process(1001), _Process(1002)]

    class _Pool:
        def __init__(self):
            self._processes = {process.pid: process for process in processes}

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    monkeypatch.setattr(batch_mod, "_INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(
        batch_mod, "_INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC", 0.01
    )
    with caplog.at_level("ERROR", logger="core.metrics.batch"):
        batch_mod._shutdown_intrusive_pool(_Pool(), terminate=False)

    assert all(not process.is_alive() for process in processes)
    assert "survived terminate/kill" not in caplog.text


pytestmark = [pytest.mark.unit]
