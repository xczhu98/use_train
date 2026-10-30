"""批量客观指标编排入口。"""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Optional

import numpy as np

from core.metrics.intrusive import _INTRUSIVE_FNS, _compute_intrusive_one
from core.metrics.mos import _MOS_SUB_SCORES, compute_mos_batch

_log = logging.getLogger(__name__)

# 整个侵入式 chunk 的绝对上界：验证 chunk 正常为秒级，超过即认为 worker
# 或 executor 无法继续。使用绝对 deadline，不能因其他 future 完成而无限续期。
_INTRUSIVE_TASK_TIMEOUT_SEC = 120.0
# 指标等待需要让训练主流程的 shutdown flag 可观察，不能把整个 timeout 一次性睡完。
_INTRUSIVE_WAIT_POLL_SEC = 0.5
# 异常路径每个 terminate/kill 阶段的短回收上限。
_INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC = 1.0
# future 已全部消费后的正常 executor teardown 可比异常强杀稍慢，但仍须有界。
_INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC = 5.0


class _IntrusiveMetricsAborted(RuntimeError):
    """训练主流程已请求停机，当前指标批必须立即放弃。"""


class _IntrusiveMetricsTimeout(TimeoutError):
    """指标 worker 长时间无进展，当前指标批必须立即放弃。"""


def _intrusive_worker_is_alive(process: object) -> bool:
    """Best-effort 查询 worker；无法判断时保守视为仍存活。"""
    try:
        return bool(process.is_alive())
    except Exception:
        _log.exception("Failed to query intrusive metric worker state")
        return True


def _join_intrusive_workers(
    processes: tuple[object, ...] | list[object],
    *,
    failure_message: str,
    timeout_sec: float = _INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC,
) -> list[object]:
    """在统一墙钟上限内公平等待全部 worker，返回仍存活的进程。

    每次等待按当前剩余 worker 数均分剩余预算，避免首个慢 worker 独占整个
    deadline，导致后续 worker 只得到 ``join(timeout=0)`` 而无法完成回收。
    """
    pending = [process for process in processes if _intrusive_worker_is_alive(process)]
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    for index, process in enumerate(pending):
        remaining = max(0.0, deadline - time.monotonic())
        timeout = remaining / max(1, len(pending) - index)
        try:
            process.join(timeout=timeout)
        except Exception:
            _log.exception(failure_message)
    return [process for process in pending if _intrusive_worker_is_alive(process)]


def _shutdown_intrusive_pool(pool: ProcessPoolExecutor, *, terminate: bool) -> None:
    """有界且 noexcept 地关闭指标池，必要时 terminate/kill 残留 worker。

    ``shutdown(wait=False)`` 本身不会终止正在执行或卡在 native teardown 的
    worker；``shutdown(wait=True)`` 又没有 timeout。因此正常和异常路径都先
    非等待关闭 executor，再分阶段公平回收 worker：正常完成先给较宽但有界的
    teardown grace，异常路径直接 terminate；残留进程仍按短 deadline terminate
    后 kill。任何 executor/process 生命周期异常都只记录，不得阻断调用方后续的
    分布式 collective。
    """
    try:
        processes = tuple((getattr(pool, "_processes", None) or {}).values())
    except Exception:
        _log.exception("Failed to get intrusive metric worker process handles")
        processes = ()
    try:
        manager_thread = getattr(pool, "_executor_manager_thread", None)
    except Exception:
        _log.exception("Failed to get intrusive metric executor manager thread")
        manager_thread = None
    try:
        pool.shutdown(wait=False, cancel_futures=terminate)
    except Exception:
        _log.exception("Failed to shut down intrusive metric pool")

    if not terminate and manager_thread is not None:
        # ProcessPoolExecutor 由 manager thread 下发 worker sentinel；只 join worker
        # 会等不到正常退出并误判为卡死。先给 manager 一个统一有界 grace，让它
        # 完成 sentinel、worker join 与 queue teardown，再复核残留进程。
        try:
            manager_thread.join(
                timeout=max(0.0, _INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC)
            )
        except Exception:
            _log.exception("Failed to join intrusive metric executor manager thread")
        survivors = [
            process for process in processes if _intrusive_worker_is_alive(process)
        ]
    elif not terminate:
        # 测试 stub 或非标准 executor 没有 manager thread，直接有界等待 worker。
        survivors = _join_intrusive_workers(
            processes,
            failure_message="Failed to join intrusive metric worker",
            timeout_sec=_INTRUSIVE_NORMAL_WORKER_JOIN_TIMEOUT_SEC,
        )
    else:
        survivors = [
            process for process in processes if _intrusive_worker_is_alive(process)
        ]

    for process in survivors:
        try:
            process.terminate()
        except Exception:
            _log.exception("Failed to terminate intrusive metric worker")
    survivors = _join_intrusive_workers(
        survivors,
        failure_message="Failed to join terminated intrusive metric worker",
        timeout_sec=_INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC,
    )

    killable = []
    for process in survivors:
        try:
            kill = getattr(process, "kill")
        except Exception:
            _log.exception("Failed to get intrusive metric worker kill method")
            continue
        killable.append(process)
        try:
            kill()
        except Exception:
            _log.exception("Failed to kill intrusive metric worker")
    survivors = _join_intrusive_workers(
        killable,
        failure_message="Failed to join killed intrusive metric worker",
        timeout_sec=_INTRUSIVE_WORKER_JOIN_TIMEOUT_SEC,
    )
    for process in survivors:
        try:
            pid = process.pid
        except Exception:
            _log.exception("Failed to get intrusive metric worker pid")
            pid = "unknown"
        _log.error(
            "Intrusive metric worker pid=%s survived terminate/kill",
            pid,
        )


def _mos_metric_keys(name: str) -> list[str]:
    """返回 MOS 模型在聚合、失败统计与 strict 检查中的 canonical 键。"""
    sub_scores = _MOS_SUB_SCORES.get(name)
    if sub_scores is None:
        return [name]
    return [f"{name}_{sub}" for sub in sub_scores]


def compute_metrics_batch(
    pairs: list[tuple[np.ndarray, np.ndarray, int]],
    metrics: list[str],
    mos_models: Optional[dict[str, object]] = None,
    strict: bool = False,
    parallel_intrusive: bool = True,
    mos_batch_size: int = 16,
    max_workers_intrusive: Optional[int] = None,
    device: str = "cpu",
    return_per_sample: bool = False,
    should_abort: Optional[Callable[[], bool]] = None,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, float]]]:
    """计算一批 (ref, enh, fs) 音频对的客观指标。

    Args:
        pairs:      [(ref_np, enh_np, fs), ...]
        metrics:    侵入式指标名列表，如 ["SI-SDR", "PESQ", "STOI", "ESTOI"]
        mos_models: 已加载的 MOS 模型字典，例如 {"DNSMOS": scorer, "NISQA": scorer}
        strict:     没有任何有效结果时是否报错
        parallel_intrusive: 是否允许对 PESQ/STOI 等使用 ProcessPoolExecutor（默认 True）；
            公共调用的 1–3 条小批量保持顺序计算，传入 should_abort 的 Trainer 调用
            则任意批量都使用隔离进程。
        mos_batch_size: MOS 推理 batch size（UTMOS GPU batch；默认 16）
        max_workers_intrusive: ProcessPoolExecutor 最大 worker 数（None/0=自动，上限 8）
        device:     MOS batch 使用的设备（如 "cuda:0"），仅当模型支持 batch 时生效
        return_per_sample: 是否同时返回逐样本值（JAE-M06）。默认 False 保持原返回类型，
            Trainer / evaluate.common 等聚合调用方无需改动。
        should_abort: 可选的中断谓词（如训练 shutdown flag）；并行等待期间周期性检查，
            触发后终止当前 pool 并跳过本批侵入式指标，使主流程尽快返回安全边界。

    Returns:
        默认返回 {metric_name: sum_value, "_count_"+metric_name: count}，用于分布式
        all_reduce，Trainer 通过 sum / count 计算均值。

        return_per_sample=True 时返回 (results, per_sample)：per_sample 与裁剪后的
        pairs 索引对齐（零长样本在裁剪时被丢弃），每个元素为 {metric_name: value}，
        仅记录有效（有限）值；某样本某指标失败时对应键缺失。

        对含子分数的 MOS 模型，输出键为 "{MODEL}_{SUB}"，例如 "DNSMOS_OVRL"。
    """
    if mos_models is None:
        mos_models = {}

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    failures: dict[str, int] = {}

    unknown = [m for m in metrics if m not in _INTRUSIVE_FNS]
    if unknown:
        msg = f"Unknown intrusive metrics: {unknown}. Supported: {list(_INTRUSIVE_FNS.keys())}"
        if strict:
            raise KeyError(msg)
        _log.warning(msg)

    active_intrusive = [m for m in metrics if m in _INTRUSIVE_FNS]
    for m in active_intrusive:
        sums[m] = 0.0
        counts[m] = 0
        failures[m] = 0

    active_mos: dict[str, object] = {}
    for name, model in mos_models.items():
        if model is None:
            continue
        active_mos[name] = model
        for key in _mos_metric_keys(name):
            sums[key] = 0.0
            counts[key] = 0
            failures[key] = 0

    trimmed_pairs = []
    for ref_np, enh_np, fs in pairs:
        minlen = min(len(ref_np), len(enh_np))
        if minlen == 0:
            continue
        trimmed_pairs.append((ref_np[:minlen], enh_np[:minlen], fs))

    # 逐样本容器（JAE-M06）：与 trimmed_pairs 索引对齐，仅记录有效（有限）值。
    # 调用方若保证输入 pairs 无零长样本，per_sample 即与其输入顺序一一对应。
    per_sample: list[dict[str, float]] = [dict() for _ in trimmed_pairs]

    # 公共 API 的 1–3 条小批量保持顺序语义，兼容 stdin/notebook 等无法 spawn
    # ``__main__`` 的调用环境。Trainer 会传 should_abort，此时无论样本数都必须
    # 隔离到 spawn worker，才能观察 shutdown/timeout 并避免 native 指标卡死主进程。
    _parallel_metrics = {"SI-SDR", "PESQ", "STOI", "ESTOI"}
    can_parallel = (
        parallel_intrusive
        and bool(trimmed_pairs)
        and (should_abort is not None or len(trimmed_pairs) >= 4)
        and any(m in _parallel_metrics for m in active_intrusive)
    )

    skip_intrusive = False
    if can_parallel and active_intrusive:
        if should_abort is not None and should_abort():
            empty = {}
            if return_per_sample:
                return empty, [dict() for _ in trimmed_pairs]
            return empty

        import os as _os

        _cpu_count = _os.cpu_count() or 1
        auto_cap = max(1, min(8, _cpu_count))
        cap = max_workers_intrusive if max_workers_intrusive and max_workers_intrusive > 0 else auto_cap
        n_workers = min(max(1, cap), len(trimmed_pairs), _cpu_count)
        tasks = []
        for ref_np, enh_np, fs in trimmed_pairs:
            for m in active_intrusive:
                tasks.append((ref_np, enh_np, fs, m))

        pool = None
        try:
            # CUDA/NCCL/DataLoader 初始化后禁止默认 fork 指标 worker；spawn 只导入
            # 顶层可 pickle 的 _compute_intrusive_one，避免继承父进程锁/运行时状态。
            ctx = multiprocessing.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
            future_to_task = {
                pool.submit(_compute_intrusive_one, task): task_i
                for task_i, task in enumerate(tasks)
            }
            pending = set(future_to_task)
            deadline = time.monotonic() + _INTRUSIVE_TASK_TIMEOUT_SEC
            while pending:
                if should_abort is not None and should_abort():
                    raise _IntrusiveMetricsAborted(
                        "metrics computation aborted by shutdown request"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _IntrusiveMetricsTimeout(
                        "intrusive metrics exceeded chunk deadline of "
                        f"{_INTRUSIVE_TASK_TIMEOUT_SEC:.0f}s"
                    )
                done, pending = wait(
                    pending,
                    timeout=min(_INTRUSIVE_WAIT_POLL_SEC, remaining),
                    return_when="FIRST_COMPLETED",
                )
                for fut in done:
                    task_i = future_to_task[fut]
                    metric_name, value, ok = fut.result()
                    if ok:
                        sums[metric_name] += value
                        counts[metric_name] += 1
                        per_sample[task_i // len(active_intrusive)][metric_name] = value
                    else:
                        failures[metric_name] += 1
        except _IntrusiveMetricsAborted as e:
            _log.warning("Parallel intrusive metrics stopped (%s); skipping current batch", e)
            if pool is not None:
                _shutdown_intrusive_pool(pool, terminate=True)
                pool = None
            # 停机请求优先：连 MOS 也不再启动，立即把控制权还给训练安全边界。
            empty = {}
            if return_per_sample:
                return empty, [dict() for _ in trimmed_pairs]
            return empty
        except Exception as e:
            _log.warning(
                "Parallel intrusive metrics failed (%s); skipping intrusive metrics for current batch",
                e,
            )
            if pool is not None:
                _shutdown_intrusive_pool(pool, terminate=True)
                pool = None
            # pool 已创建/提交后任何异常都禁止在训练主进程顺序重放；native
            # PESQ/STOI 可能再次卡死或崩溃。仅清空侵入式结果，后续 MOS 与
            # strict missing 检查仍按公共 API 合同执行。
            for m in active_intrusive:
                sums[m] = 0.0
                counts[m] = 0
                failures[m] = 0
            per_sample = [dict() for _ in trimmed_pairs]
            skip_intrusive = True
        finally:
            if pool is not None:
                _shutdown_intrusive_pool(pool, terminate=False)

    if not can_parallel and not skip_intrusive and active_intrusive:
        for i, (ref_np, enh_np, fs) in enumerate(trimmed_pairs):
            for m in active_intrusive:
                fn, _ = _INTRUSIVE_FNS[m]
                v = fn(ref_np, enh_np, fs)
                if v is not None and np.isfinite(v):
                    sums[m] += v
                    counts[m] += 1
                    per_sample[i][m] = v
                else:
                    failures[m] += 1

    # 非侵入式 MOS：优先使用原生 batch（如 UTMOS），否则逐条
    enh_fs_list = [(enh_np, fs) for _, enh_np, fs in trimmed_pairs]
    for name, model in active_mos.items():
        results_list = compute_mos_batch(
            model, name, enh_fs_list,
            batch_size=mos_batch_size, device=device,
        )
        metric_keys = _mos_metric_keys(name)
        if len(results_list) != len(enh_fs_list):
            _log.warning(
                "%s MOS batch output size mismatch: expected %d, got %d; "
                "marking missing items failed",
                name, len(enh_fs_list), len(results_list),
            )
        for i, result in enumerate(results_list[:len(enh_fs_list)]):
            if result is None:
                for key in metric_keys:
                    failures[key] += 1
                continue
            if isinstance(result, dict):
                for sub, key in zip(_MOS_SUB_SCORES.get(name, ()), metric_keys):
                    v = result.get(sub)
                    if v is not None and np.isfinite(v):
                        sums[key] += v
                        counts[key] += 1
                        per_sample[i][key] = v
                    else:
                        failures[key] += 1
            elif isinstance(result, (int, float)) and np.isfinite(result):
                key = metric_keys[0]
                sums[key] += result
                counts[key] += 1
                per_sample[i][key] = float(result)
            else:
                for key in metric_keys:
                    failures[key] += 1
        for _ in range(len(results_list), len(enh_fs_list)):
            for key in metric_keys:
                failures[key] += 1

    if strict:
        all_keys = list(active_intrusive) + [
            key for name in active_mos for key in _mos_metric_keys(name)
        ]
        missing = [k for k in all_keys if counts.get(k, 0) == 0]
        if missing:
            detail = ", ".join(f"{m}(fail={failures.get(m, 0)})" for m in missing)
            raise RuntimeError(f"Strict metrics mode: no valid values for {detail}")

    results = {}
    for m in list(sums.keys()):
        if counts.get(m, 0) > 0:
            results[m] = sums[m]
            results[f"_count_{m}"] = float(counts[m])
    if return_per_sample:
        return results, per_sample
    return results
