"""
LLM batch processor 抽象基类。

统一 Qwen-Omni 系列 Processor 的公共能力：
  - 磁盘缓存（namespace 绑定已验证的 qwen_model_sha256）
  - max_retries 重试
  - CUDA OOM / 一般异常时 batch 折半重试（单条毒记录 None 占位隔离）
  - 可选 checkpoint 标记
  - 解析失败用 None 占位，保持输出长度与输入一致

子类只需实现：
  - build_prompts(records) -> List[str]
  - parse_responses(texts) -> List[Any]
  - _load_model() / _unload_model() 模型加载钩子
  - （可选）_dry_run_results(records) -> List[Any]
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.llm_cache import LLMResultCache
from ..core.processor_base import Processor

logger = logging.getLogger(__name__)


class WorkerPoolInfraError(RuntimeError):
    """worker pool 基础设施故障（如 worker 进程死亡），区别于内容级推理失败。

    该异常不应被当作普通推理失败无限重试并降级为 ``[None] * n`` 拒绝；
    连续多批出现时必须中止 stage。
    """


def extract_assistant_reply(text: str) -> Optional[str]:
    """从 Qwen apply_chat_template 输出中提取 assistant 字段内容。

    匹配前为行首/字符串开头、后为空白符的 ``assistant`` role marker，
    避免把生成正文中的普通单词 "assistant" 误判为边界。
    """
    if not text:
        return None

    # 匹配 role marker：assistant 前为行首/换行/Qwen 特殊 token，后是空白；
    # 避免把正文中的普通单词 "assistant" 误判为边界。
    matches = list(re.finditer(r"(?:^|\n|<\|im_start\|>)\s*assistant(\s+)", text, flags=re.IGNORECASE))
    if matches:
        reply = text[matches[-1].end():].strip()
        return reply or None

    return None


def _infer_with_halving(
    infer_fn: Callable[[List[Dict[str, Any]]], Optional[List[str]]],
    items: List[Dict[str, Any]],
    context: str,
) -> List[Optional[str]]:
    """执行 batch 推理，返回长度不匹配、CUDA OOM 或一般异常时递归折半重试。

    一般异常常由单条毒记录（损坏音频等）触发，折半可隔离故障记录，
    保证同批其余记录不受影响；单条记录仍失败时返回 None 占位交给上层
    按失败语义处理，避免一条坏数据卡死整批。
    """
    n = len(items)
    try:
        texts = infer_fn(items)
    except Exception as e:
        if "CUDA out of memory" in str(e):
            if n <= 1:
                raise
            logger.warning("%s CUDA OOM，折半重试: %d", context, n)
        elif n <= 1:
            # 单条毒记录持续失败：None 占位，不再上抛异常中断整个 stage
            logger.error("%s 单条记录推理异常，按失败处理: %s", context, e)
            return [None]
        else:
            # 非 OOM 异常也可能是毒记录触发：折半隔离，定位故障记录
            logger.warning("%s batch 推理异常，折半隔离重试: %d (%s)", context, n, e)
        texts = None

    if texts is None or len(texts) != n:
        logger.warning(
            "%s batch 返回长度不匹配 (%d vs %d)，尝试折半重试",
            context,
            len(texts) if texts else 0,
            n,
        )
        if n <= 1:
            return [None]
        mid = n // 2
        first = _infer_with_halving(infer_fn, items[:mid], context)
        second = _infer_with_halving(infer_fn, items[mid:], context)
        return (
            (first if first is not None else [None] * mid)
            + (second if second is not None else [None] * (n - mid))
        )
    return texts


def _qwen_worker_main(
    gpu_id: int,
    model_path: str,
    torch_dtype: str,
    cache_dir: Optional[str],
    qwen_model_sha256: str,
    max_new_tokens: int,
    input_queue: "mp.queues.Queue",
    output_queue: "mp.queues.Queue",
) -> None:
    """Qwen worker 进程入口：绑定到指定 GPU 并持续服务推理请求。"""
    # spawn 启动的新进程需重新设置可见 GPU，避免继承父进程 CUDA 上下文
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 限制 worker 的 CPU 线程数：默认 PyTorch 按物理核数开 OMP 线程池，
    # 8 个 worker 各自开满会严重 oversubscription（实测每 worker ~950% CPU
    # 且大量为线程空转）；音频预处理并行度有限，4 线程足够，省下的核
    # 留给其他 worker 与主进程
    try:
        import torch

        torch.set_num_threads(int(os.environ.get("USE_TRAIN_QWEN_WORKER_OMP_THREADS", "4")))
    except Exception:
        pass
    # 父死看门狗：worker 阻塞在 input_queue.get() 时无法自查父进程状态，
    # 父进程被 SIGKILL 杀死后 worker 会变孤儿永久滞留（实测滞留数天）
    from ..core.process_guard import start_parent_watchdog

    start_parent_watchdog(os.getppid(), name=f"qwen-worker-gpu{gpu_id}-watchdog")
    # 延迟导入，避免父进程与 worker 进程出现循环 import
    from .qwen_single_event import get_qwen_model

    try:
        model = get_qwen_model(
            model_path=model_path,
            device_map="cuda",
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            qwen_model_sha256=qwen_model_sha256,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        logger.exception("Qwen worker (gpu=%s) 模型加载失败", gpu_id)
        output_queue.put({"task_id": "__startup__", "status": "error", "gpu_id": gpu_id, "error": str(e)})
        return

    # 通知父进程模型加载成功
    output_queue.put({"task_id": "__startup__", "status": "ok", "gpu_id": gpu_id})

    while True:
        task = input_queue.get()
        if task is None:
            break
        items: List[Dict[str, Any]] = task["items"]
        task_id: int = task["task_id"]
        try:
            texts = _infer_with_halving(
                lambda batch: model.batch_query_raw(batch),
                items,
                f"Qwen worker (gpu={gpu_id})",
            )
        except Exception as e:
            logger.error("Qwen worker (gpu=%s) 推理失败: %s", gpu_id, e)
            texts = [None] * len(items)
        output_queue.put({"task_id": task_id, "texts": texts})


class _QwenWorkerPool:
    """多 GPU Qwen 推理 worker pool（每个 GPU 一个独立进程）。"""

    def __init__(
        self,
        num_workers: int,
        gpu_ids: List[int],
        model_path: str,
        torch_dtype: str,
        cache_dir: Optional[str],
        timeout: int = 3600,
        qwen_model_sha256: str = "unset:qwen_model_sha256",
        max_new_tokens: int = 8,
    ):
        self.num_workers = max(1, num_workers)
        self.gpu_ids = gpu_ids if gpu_ids else list(range(self.num_workers))
        if len(self.gpu_ids) < self.num_workers:
            self.num_workers = len(self.gpu_ids)
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.qwen_model_sha256 = qwen_model_sha256
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

        self._ctx = mp.get_context("spawn")
        self._in_queues: List[mp.queues.Queue] = []
        self._out_queue: Optional[mp.queues.Queue] = None
        self._procs: List[mp.Process] = []
        self._started = False
        # infer 互斥锁：并发 infer 会让 round-robin task_id 与响应错位（整批 None）
        self._infer_lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._started:
            return
        self._out_queue = self._ctx.Queue()
        for gpu_id in self.gpu_ids[: self.num_workers]:
            in_q = self._ctx.Queue()
            p = self._ctx.Process(
                target=_qwen_worker_main,
                args=(
                    gpu_id,
                    self.model_path,
                    self.torch_dtype,
                    self.cache_dir,
                    self.qwen_model_sha256,
                    self.max_new_tokens,
                    in_q,
                    self._out_queue,
                ),
            )
            p.start()
            self._in_queues.append(in_q)
            self._procs.append(p)

        # 等待所有 worker 加载就绪，任一失败立即停止并释放资源
        startup_timeout = max(60, min(600, self.timeout))
        startup_errors = []
        gpu_ids_used = self.gpu_ids[: self.num_workers]
        for _ in range(len(self._procs)):
            resp = None
            deadline = time.monotonic() + startup_timeout
            while resp is None:
                # worker 启动阶段早死（模型加载崩溃/OOM killer）永远等不到响应，
                # 必须立即报错释放资源，而不是白等整个 startup_timeout
                dead = [g for g, p in zip(gpu_ids_used, self._procs) if not p.is_alive()]
                if dead:
                    startup_errors.append(f"worker (gpu={dead}) 启动阶段提前退出")
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    startup_errors.append(f"读取 worker 启动响应超时（>{startup_timeout}s）")
                    break
                try:
                    resp = self._out_queue.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    # 短轮询：留出检查 worker 存活状态的机会
                    continue
                except Exception as e:
                    startup_errors.append(f"读取 worker 启动响应失败: {e}")
                    break
            if resp is None:
                continue
            if resp.get("task_id") != "__startup__":
                startup_errors.append(f"worker 启动响应格式异常: {resp}")
                continue
            if resp.get("status") != "ok":
                startup_errors.append(
                    f"worker (gpu={resp.get('gpu_id')}) 启动失败: {resp.get('error')}"
                )
        if startup_errors:
            self.stop()
            # 启动失败属基础设施失败：抛 WorkerPoolInfraError 进入熔断计数
            # （普通 RuntimeError 会被 generic except 吞掉走逐批重试，
            # 永久性启动失败时每批都烧一次 8 GPU 模型重载）
            raise WorkerPoolInfraError(
                "Qwen worker pool 启动失败:\n" + "\n".join(startup_errors)
            )

        self._started = True
        logger.info("Qwen worker pool 启动完成：%d 个 worker，GPU %s", self.num_workers, self.gpu_ids[: self.num_workers])

    def infer(self, items: List[Dict[str, Any]]) -> List[Optional[str]]:
        """将 items 分片到各 worker，返回与输入等长的原始文本列表（保持顺序）。"""
        if not items:
            return []
        # 互斥：并发 infer 会让 round-robin task_id 与 out_queue 响应错位
        # （迟到的响应被下一个 batch 消费，整批 None）
        with self._infer_lock:
            return self._infer_locked(items)

    def _infer_locked(self, items: List[Dict[str, Any]]) -> List[Optional[str]]:
        """infer 的实际实现，调用方必须持有 self._infer_lock。"""
        self._ensure_started()
        # worker 进程中途死亡（OOM killer/segfault）时立即失败，
        # 避免在 out_queue 上白等 timeout（默认 3600s）
        dead = [p for p in self._procs if not p.is_alive()]
        if dead:
            raise WorkerPoolInfraError(
                f"Qwen worker 进程已退出（共 {len(dead)}/{len(self._procs)} 个），"
                "可能因 OOM 或崩溃被杀"
            )
        n_workers = len(self._in_queues)
        # 按 round-robin 分片，保持记录相对顺序
        chunks: List[List[Dict[str, Any]]] = [[] for _ in range(n_workers)]
        for i, item in enumerate(items):
            chunks[i % n_workers].append(item)

        active_workers = [i for i, chunk in enumerate(chunks) if chunk]
        for i in active_workers:
            self._in_queues[i].put({"task_id": i, "items": chunks[i]})

        results: List[Optional[List[str]]] = [None] * n_workers
        # 总等待预算 = timeout × 参与 worker 数（保持旧"每次 get 独立计时"的
        # 语义：8 个 worker 各需 ~500s 的批不应被单次 3600s 预算误杀）；
        # 轮询短超时 + 每轮检测 worker 存活——worker 中途死亡（OOM
        # killer/segfault）时立即失败，避免在 out_queue 上白等满整个 timeout
        deadline = time.monotonic() + self.timeout * len(active_workers)
        for _ in active_workers:
            while True:
                dead_workers = [i for i in active_workers if not self._procs[i].is_alive()]
                if dead_workers:
                    logger.error(
                        "Qwen worker 进程已退出（worker_idx=%s），终止并重建 pool",
                        dead_workers,
                    )
                    self.stop()
                    raise WorkerPoolInfraError(
                        f"Qwen worker 进程已退出（共 {len(dead_workers)}/{len(active_workers)} 个），"
                        "可能因 OOM 或崩溃被杀"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 超时后旧任务可能仍在 worker 侧运行，其迟到响应会被后续 batch
                    # 的 get 消费导致 task_id 错位（整批 None）；必须杀掉旧 worker、
                    # 丢弃残留队列，让上层走基础设施失败路径，下次 infer 重建 pool。
                    logger.error(
                        "Qwen worker pool 推理超时（%ds），终止旧 worker 并重建 pool",
                        self.timeout,
                    )
                    self.stop()
                    raise WorkerPoolInfraError(
                        f"Qwen worker pool 推理超时（{self.timeout}s），已终止旧 worker"
                    )
                try:
                    resp = self._out_queue.get(timeout=min(1.0, remaining))
                    break
                except queue.Empty:
                    continue
            task_id = resp["task_id"]
            # 防御：异常 task_id（协议损坏/串批）会 IndexError 裸崩且不计数
            # 熔断——统一走基础设施失败路径
            if not isinstance(task_id, int) or task_id < 0 or task_id >= n_workers:
                logger.error("Qwen worker 返回异常 task_id=%r，终止并重建 pool", task_id)
                self.stop()
                raise WorkerPoolInfraError(
                    f"Qwen worker 返回异常 task_id={task_id!r}，已终止旧 worker"
                )
            texts = resp.get("texts")
            expected_len = len(chunks[task_id])
            if not isinstance(texts, list) or len(texts) != expected_len:
                logger.warning(
                    "Qwen worker task_id=%d 返回长度不匹配，使用 None 填充", task_id
                )
                texts = [None] * expected_len
            results[task_id] = texts

        # 按 round-robin 顺序还原
        out: List[Optional[str]] = []
        max_len = max(len(c) for c in chunks) if chunks else 0
        for pos in range(max_len):
            for i in range(n_workers):
                if pos < len(chunks[i]):
                    out.append(results[i][pos])
        return out

    def stop(self) -> None:
        for in_q in self._in_queues:
            try:
                in_q.put(None, timeout=5)
            except Exception:
                pass
        for p in self._procs:
            try:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        # terminate 后仍不退出（如卡在 CUDA 同步）：必须强杀，
                        # 否则残留进程持续占 GPU，污染后续重建的 pool
                        logger.warning("Qwen worker terminate 后仍存活，强制 kill")
                        p.kill()
                        p.join(timeout=5)
            except Exception as e:
                logger.warning("Qwen worker 进程退出异常: %s", e)
        # 清空队列与进程引用：残留的迟到响应/死亡进程会污染重启后的 pool
        # （旧 out_queue 的迟到响应串批、dead 检查误判），_ensure_started 据此重建
        self._in_queues = []
        self._procs = []
        self._out_queue = None
        self._started = False


class LLMBatchProcessor(Processor, ABC):
    """LLM batch 处理器抽象基类。"""

    def __init__(
        self,
        config: Dict[str, Any],
        stage_name: str,
        cache_subdir: Optional[str] = None,
        qwen_model_sha256: Optional[str] = None,
        cache_key_salt: str = "",
    ):
        super().__init__(config, stage_name=stage_name)
        self.model_path = config.get("model_path") or config.get("qwen_model")
        identity = config.get("artifact_fingerprints") or {}
        self.qwen_model_sha256 = (
            qwen_model_sha256
            if qwen_model_sha256 is not None
            else identity.get("qwen_model_sha256", "unset:qwen_model_sha256")
        )
        self.batch_size = config.get("batch_size", 8)
        self.max_retries = config.get("max_retries", 3)
        self.device_map = config.get("device_map", "auto")
        self.torch_dtype = config.get("torch_dtype", "auto")
        self.cache_dir = config.get("cache_dir") or config.get("qwen_cache_dir")
        self.dry_run = config.get("dry_run", False)
        self.enable_checkpoint = config.get("enable_checkpoint", True)
        # 多 GPU 数据并行：默认单卡，配置 num_workers/gpu_ids 后启用子进程 worker pool
        self.num_workers = int(config.get("num_workers", 1))
        self.gpu_ids = config.get("gpu_ids", [])

        self._cache_value_key = "result"
        self._model: Optional[Any] = None
        self._model_lock = threading.Lock()
        self._worker_pool: Optional["_QwenWorkerPool"] = None
        self._worker_pool_lock = threading.Lock()
        # worker pool 连续基础设施失败计数：达到上限即中止 stage，
        # 避免 worker 死亡后所有 batch 被无限降级为 [None]*n 拒绝
        self._consecutive_worker_infra_failures = 0
        self._max_consecutive_worker_infra_failures = int(
            config.get("max_consecutive_worker_infra_failures", 3)
        )

        full_cache_dir: Optional[str] = None
        if self.cache_dir and cache_subdir:
            full_cache_dir = os.path.join(self.cache_dir, cache_subdir)
        elif cache_subdir:
            full_cache_dir = cache_subdir

        self.cache = LLMResultCache(
            cache_dir=full_cache_dir,
            model_name=self.model_path or f"{stage_name.lower()}-dryrun",
            namespace=self.qwen_model_sha256,
            key_salt=cache_key_salt,
        )

    @abstractmethod
    def build_prompts(self, records: List[Any]) -> List[str]:
        """为输入记录构造 prompt 列表，长度与 records 一致。"""
        raise NotImplementedError

    @abstractmethod
    def parse_responses(self, texts: List[str]) -> List[Any]:
        """解析模型原始输出；无法解析的条目用 None 占位。"""
        raise NotImplementedError

    def _load_model(self) -> None:
        """子类覆盖：将加载好的模型赋值给 self._model。"""
        raise NotImplementedError

    def _unload_model(self) -> None:
        """释放模型引用；子类可按需覆盖。"""
        self._model = None

    def _get_model(self) -> Optional[Any]:
        """延迟加载模型，dry_run 下返回 None。"""
        if self.dry_run:
            return None
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _extract_audio_path(self, record: Any) -> str:
        """默认从 AudioRecord 中提取音频路径；子类可按需覆盖。"""
        return record.audio["path"]

    def _dry_run_results(self, records: List[Any]) -> List[Any]:
        """dry_run 模式下的伪结果；子类必须覆盖才能使用 dry_run。"""
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 _dry_run_results，无法使用 dry_run"
        )

    def _infer_raw(self, items: List[Dict[str, Any]]) -> Optional[List[str]]:
        """调用已加载模型做原始 batch 推理（单卡模式）。

        内部通过 ``_infer_with_halving`` 处理 CUDA OOM 与返回长度不匹配，
        保证输出长度与输入一致。
        """
        model = self._get_model()
        if model is None:
            raise RuntimeError(
                f"{self.stage_name} 模型未加载 (dry_run={self.dry_run}); "
                f"请检查 _load_model 是否正确设置 self._model 或启用 dry_run"
            )
        return _infer_with_halving(
            lambda batch: model.batch_query_raw(batch),
            items,
            self.stage_name,
        )

    def _get_worker_pool(self) -> _QwenWorkerPool:
        """延迟初始化多 GPU worker pool。"""
        if self._worker_pool is None:
            with self._worker_pool_lock:
                if self._worker_pool is None:
                    gpu_ids = self.gpu_ids if self.gpu_ids else list(range(self.num_workers))
                    self._worker_pool = _QwenWorkerPool(
                        num_workers=self.num_workers,
                        gpu_ids=gpu_ids,
                        model_path=self.model_path,
                        torch_dtype=self.torch_dtype,
                        cache_dir=self.cache_dir,
                        timeout=int(os.environ.get("USE_TRAIN_QWEN_WORKER_TIMEOUT", "3600")),
                        qwen_model_sha256=self.qwen_model_sha256,
                        max_new_tokens=self.max_new_tokens,
                    )
        return self._worker_pool

    def stop_worker_pool(self) -> None:
        """关闭 worker pool，释放 GPU 资源。应由子类在 process finally 中调用。"""
        if self._worker_pool is not None:
            self._worker_pool.stop()
            self._worker_pool = None

    def close(self) -> None:
        """统一资源收口（Processor.close 契约）：关闭 worker pool。"""
        self.stop_worker_pool()

    def _query_with_retry_and_oom_halving(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Any]:
        """带重试、指数退避的内部推理；OOM 折半由 _infer_raw 统一处理。"""
        n = len(items)
        base_delay = 0.5
        for attempt in range(self.max_retries):
            try:
                raw_texts = self._infer_raw(items)
                if raw_texts is None or len(raw_texts) != n:
                    logger.warning(
                        "%s batch 推理返回长度不匹配 (%d vs %d)",
                        self.stage_name,
                        len(raw_texts) if raw_texts else 0,
                        n,
                    )
                    # 非致命错误：继续重试前也做退避
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                parsed = self.parse_responses(raw_texts)
                if len(parsed) != n:
                    logger.warning(
                        "%s parse_responses 返回长度不匹配 (%d vs %d)",
                        self.stage_name,
                        len(parsed),
                        n,
                    )
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return parsed
            except Exception as e:
                logger.warning(
                    "%s batch 推理失败 (attempt %d/%d): %s",
                    self.stage_name,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                # CUDA OOM 已在 _infer_raw 内部通过折半处理；
                # 此处只处理非 OOM 异常，按指数退避重试。
                if attempt < self.max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    logger.error(
                        "%s batch 达到最大重试次数，跳过本 batch",
                        self.stage_name,
                    )
        return [None] * n

    def _worker_infer_with_retry(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Any]:
        """worker pool 推理 + 重试，与单卡路径 _query_with_retry_and_oom_halving 语义对齐。"""
        n = len(items)
        base_delay = 0.5
        for attempt in range(self.max_retries):
            try:
                raw_texts = self._get_worker_pool().infer(items)
                parsed = self.parse_responses(raw_texts)
                if len(parsed) != n:
                    logger.warning(
                        "%s worker pool parse 返回长度不匹配 (%d vs %d)",
                        self.stage_name, len(parsed), n,
                    )
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                self._consecutive_worker_infra_failures = 0
                return parsed
            except WorkerPoolInfraError as e:
                # 基础设施故障（worker 进程死亡）：重试无意义，直接按整批失败处理；
                # 连续多批基础设施失败说明 pool 无法自愈，必须中止 stage，
                # 而不是把后续所有 batch 静默降级为 [None]*n 拒绝。
                self._consecutive_worker_infra_failures += 1
                logger.error(
                    "%s worker pool 基础设施故障（连续第 %d 批）: %s",
                    self.stage_name, self._consecutive_worker_infra_failures, e,
                )
                if self._consecutive_worker_infra_failures >= self._max_consecutive_worker_infra_failures:
                    raise RuntimeError(
                        f"{self.stage_name} worker pool 连续 "
                        f"{self._consecutive_worker_infra_failures} 批基础设施失败，中止 stage"
                    ) from e
                break
            except Exception as e:
                logger.warning(
                    "%s worker pool 推理失败 (attempt %d/%d): %s",
                    self.stage_name, attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    logger.error("%s worker pool 达到最大重试次数，跳过本 batch", self.stage_name)
        return [None] * n

    def process_records(
        self,
        records: List[Any],
    ) -> List[Any]:
        """
        统一 batch 处理入口。

        返回与 records 长度一致的结果列表，缓存命中或推理失败的条目用 None 占位。
        注意：本方法不操作 checkpoint。调用方应在把结果写入 manifest 后再自行
        mark_done，以保证 checkpoint 与输出文件的一致性。
        """
        if not records:
            return []

        n = len(records)
        prompts = self.build_prompts(records)
        audio_paths = [self._extract_audio_path(r) for r in records]

        if self.dry_run:
            results = self._dry_run_results(records)
        else:
            results: List[Any] = [None] * n
            missing_indices: List[int] = []
            for idx, (path, prompt) in enumerate(zip(audio_paths, prompts)):
                cached = self.cache.get(path, prompt) if self.cache else None
                if cached is not None and self._cache_value_key in cached:
                    results[idx] = cached[self._cache_value_key]
                else:
                    missing_indices.append(idx)

            if missing_indices:
                missing_items = [
                    {"audio_path": audio_paths[i], "prompt": prompts[i]}
                    for i in missing_indices
                ]
                if self.num_workers > 1 and not self.dry_run:
                    # 多 GPU 数据并行：worker 返回原始文本，父进程统一解析（带重试）
                    inferred = self._worker_infer_with_retry(missing_items)
                else:
                    inferred = self._query_with_retry_and_oom_halving(missing_items)
                for rel_idx, abs_idx in enumerate(missing_indices):
                    value = inferred[rel_idx]
                    results[abs_idx] = value
                    if value is not None and self.cache:
                        self.cache.set(
                            audio_paths[abs_idx],
                            prompts[abs_idx],
                            {self._cache_value_key: value},
                        )

        return results
