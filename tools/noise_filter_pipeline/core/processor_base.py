"""
Processor 抽象基类。

所有 stage 内的具体处理单元（VAD、分类、Qwen、超分等）都应继承 Processor，
实现统一的 process(input_manifest, output_dir) -> output_manifest 接口。
"""
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, Optional, TypeVar

from .manifest_io import AudioRecord

T = TypeVar("T")


def _process_in_threads(
    items: Iterable[T],
    fn: Callable[[T], Any],
    num_workers: int,
) -> Iterator[Any]:
    """在 ThreadPoolExecutor 中并发处理 items，返回结果保持原序。

    当 num_workers <= 1 时直接顺序执行，避免线程池开销。
    num_workers > 1 时使用有界提交窗口：任意时刻最多 num_workers 个任务
    在飞行中，不会把 items 全量物化到内存，适合流式 manifest。
    """
    if num_workers <= 1:
        for item in items:
            yield fn(item)
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending: Deque[Future] = deque()
        item_iter = iter(items)
        # 先填满窗口
        for _ in range(num_workers):
            try:
                pending.append(executor.submit(fn, next(item_iter)))
            except StopIteration:
                break
        # 每产出一个结果再补交一个任务，保持窗口大小
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(fn, next(item_iter)))
            except StopIteration:
                continue


class Processor(ABC):
    """统一处理器抽象基类。"""

    def __init__(
        self,
        config: Dict[str, Any],
        stage_name: str,
        progress_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            config: 当前 processor 的配置字典（通常来自 YAML 的对应段）。
            stage_name: 所属阶段名，如 "B1"、"C1"，用于增量跳过。
            progress_callback: 可选的全局进度回调，每完成一个文件/记录调用一次。
        """
        self.config = config or {}
        self.stage_name = stage_name
        self.progress_callback = progress_callback

    @abstractmethod
    def process(self, input_manifest: str, output_dir: str) -> str:
        """
        处理输入 manifest，返回输出 manifest 路径。

        Args:
            input_manifest: 输入 JSONL manifest 路径。
            output_dir: 本 processor 的输出目录（产物音频、中间文件、输出 manifest）。

        Returns:
            输出 manifest 的绝对路径。
        """
        raise NotImplementedError

    def close(self) -> None:
        """释放长驻资源（daemon 进程池、VAD 进程池、LLM worker pool 等）。

        默认无操作；持有长驻资源的子类应重写。orchestrator / process 的
        finally 块统一调用本方法收尾，替代各处理器各自命名的
        shutdown/stop_worker_pool 等收口方法（保留为兼容别名）。
        """

    def should_skip(self, record: AudioRecord) -> bool:
        """默认按 record.stages[stage_name].status == 'passed' 判断是否跳过。"""
        return record.is_stage_passed(self.stage_name)
