"""Tests for noise_filter_pipeline.processors.llm_base helpers."""
import queue as std_queue
import threading
import time

import pytest

from noise_filter_pipeline.processors.llm_base import (
    WorkerPoolInfraError,
    _QwenWorkerPool,
    _infer_with_halving,
    extract_assistant_reply,
)

pytestmark = pytest.mark.unit


def test_extract_assistant_reply_standard():
    text = "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\nTrue"
    assert extract_assistant_reply(text) == "True"


def test_extract_assistant_reply_no_newline():
    # 正文中的普通 "assistant" 单词不应被误判为 role marker
    text = "user says hi assistant False"
    assert extract_assistant_reply(text) is None


def test_extract_assistant_reply_multiple_markers():
    # 只有符合 role marker 边界的 assistant 才参与提取；
    # 第二个 "assistant False" 前是空格而非行首/换行，应被忽略。
    text = "assistant\nTrue assistant False"
    assert extract_assistant_reply(text) == "True assistant False"


def test_extract_assistant_reply_empty():
    assert extract_assistant_reply("") is None
    assert extract_assistant_reply("hello world") is None
    assert extract_assistant_reply("assistant") is None
    assert extract_assistant_reply("assistant   ") is None


# ---------------------------------------------------------------------------
# _infer_with_halving：CUDA OOM 折半 + 一般异常折半隔离毒记录
# ---------------------------------------------------------------------------


def test_infer_with_halving_cuda_oom_halves():
    """CUDA OOM 时折半重试，单条粒度可正常返回。"""

    def infer_fn(items):
        if len(items) > 1:
            raise RuntimeError("CUDA out of memory")
        return ["ok"] * len(items)

    result = _infer_with_halving(infer_fn, [{"a": 1}, {"a": 2}], "test")
    assert result == ["ok", "ok"]


def test_infer_with_halving_cuda_oom_single_raises():
    """单条仍 CUDA OOM：保持原有行为上抛，由外层重试/降级逻辑处理。"""

    def infer_fn(items):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _infer_with_halving(infer_fn, [{"a": 1}], "test")


def test_infer_with_halving_isolates_poison_record():
    """一般异常（毒记录）同样折半隔离：毒记录 None 占位，同批其他正常返回。"""

    def infer_fn(items):
        texts = []
        for it in items:
            if it["audio_path"] == "poison.wav":
                raise RuntimeError("corrupt audio frame")
            texts.append("ok:" + it["audio_path"])
        return texts

    items = [{"audio_path": f"{i}.wav"} for i in range(3)] + [{"audio_path": "poison.wav"}]
    result = _infer_with_halving(infer_fn, items, "test")
    assert result == ["ok:0.wav", "ok:1.wav", "ok:2.wav", None]


def test_infer_with_halving_single_poison_returns_none():
    """单条毒记录持续抛非 OOM 异常：返回 [None] 占位而非上抛卡死整批。"""

    def infer_fn(items):
        raise ValueError("bad audio")

    assert _infer_with_halving(infer_fn, [{"audio_path": "poison.wav"}], "test") == [None]


class _FakeProc:
    """模拟 worker 进程：join 后视为已退出。"""

    def __init__(self, alive: bool = True):
        self._alive = alive
        self.terminated = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False


class _FakeInQueue:
    def __init__(self):
        self.items = []

    def put(self, item, timeout=None) -> None:
        self.items.append(item)


def _make_started_pool(timeout: float = 0.05) -> _QwenWorkerPool:
    """构造一个 fake 的已启动 pool：空 out_queue 保证 get 必然超时。"""
    pool = _QwenWorkerPool(
        num_workers=1,
        gpu_ids=[0],
        model_path="fake-model",
        torch_dtype="auto",
        cache_dir=None,
        timeout=timeout,
    )
    pool._started = True
    pool._out_queue = std_queue.Queue()
    pool._in_queues = [_FakeInQueue()]
    pool._procs = [_FakeProc()]
    return pool


def test_infer_timeout_raises_infra_error_and_stops_pool():
    # 超时必须抛 WorkerPoolInfraError（基础设施失败路径），而不是静默串批
    pool = _make_started_pool()
    with pytest.raises(WorkerPoolInfraError):
        pool.infer([{"audio_path": "a.wav", "prompt": "p"}])


def test_infer_timeout_rebuild_state():
    # 超时后 pool 状态必须被清空，下次 infer 时 _ensure_started 能干净重建，
    # 旧 worker 的迟到响应不会串到后续 batch
    pool = _make_started_pool()
    with pytest.raises(WorkerPoolInfraError):
        pool.infer([{"audio_path": "a.wav", "prompt": "p"}])
    assert pool._started is False
    assert pool._procs == []
    assert pool._in_queues == []
    assert pool._out_queue is None


def test_infer_fails_fast_when_worker_dies_after_dispatch():
    """dispatch 后 worker 中途死亡（OOM killer/segfault）：立即失败，不白等 timeout。

    回归：此前只在 dispatch 前检查 worker 存活，dispatch 后死亡会在空队列上
    白等满整个推理超时（默认 3600s）。等待循环必须轮询存活状态快速失败。
    """

    class LaterDyingProc(_FakeProc):
        def __init__(self, min_alive_calls=2):
            super().__init__(alive=True)
            self._min_alive_calls = min_alive_calls
            self._alive_calls = 0

        def is_alive(self):
            self._alive_calls += 1
            # 宽松约束：只要求"前若干次调用存活、之后死亡"。min_alive_calls=2
            # 只是下限，保证死亡发生在 dispatch 的存活检查之后；生产等待循环
            # 对 is_alive 的实际调用次数是内部实现细节，测试不得与精确次数
            # 耦合——循环持续轮询时，worker 死亡在下一轮即可被发现。
            return self._alive_calls <= self._min_alive_calls

    pool = _make_started_pool(timeout=3600)
    pool._procs = [LaterDyingProc()]
    t0 = time.monotonic()
    with pytest.raises(WorkerPoolInfraError, match="进程已退出"):
        pool.infer([{"audio_path": "a.wav", "prompt": "p"}])
    # 远小于 3600s 超时，证明走的是死检测而非超时路径
    assert time.monotonic() - t0 < 10
    assert pool._started is False
    assert pool._procs == []


def test_stop_clears_state_for_restart():
    pool = _make_started_pool()
    pool.stop()
    assert pool._started is False
    assert pool._procs == []
    assert pool._in_queues == []
    assert pool._out_queue is None


def test_infer_is_serialized_by_lock():
    """并发 infer 必须互斥：持锁期间第二个 infer 阻塞，避免 task_id 错位。"""

    class FakeOutQueue:
        def get(self, timeout=None):
            return {"task_id": 0, "texts": ["ok"]}

    pool = _make_started_pool()
    pool._out_queue = FakeOutQueue()

    entered = threading.Event()
    done = []

    def run():
        entered.set()
        done.append(pool.infer([{"audio_path": "a.wav", "prompt": "p"}]))

    pool._infer_lock.acquire()
    t = threading.Thread(target=run)
    t.start()
    try:
        assert entered.wait(2)
        # 轮询等待替代固定 sleep：给 infer 线程足够时间尝试抢锁，期间不得返回。
        # 固定 sleep 的时长是魔法数字，机器负载高时可能提前/延后，易 flaky。
        deadline = time.monotonic() + 2.0
        while not done and time.monotonic() < deadline:
            time.sleep(0.01)
        # 锁被主线程持有期间，infer 线程不得进入
        assert done == []
    finally:
        pool._infer_lock.release()
    t.join(5)
    assert done == [["ok"]]


def test_stop_kills_worker_surviving_terminate():
    """terminate 后仍存活的 worker 必须 kill，避免残留进程持续占 GPU。"""

    class StubbornProc(_FakeProc):
        def __init__(self):
            super().__init__(alive=True)
            self.killed = False

        def join(self, timeout=None):
            pass  # join 不会让其退出

        def terminate(self):
            self.terminated = True  # terminate 无效

        def kill(self):
            self.killed = True
            self._alive = False

    pool = _make_started_pool()
    stubborn = StubbornProc()
    pool._procs = [stubborn]
    pool.stop()
    assert stubborn.terminated
    assert stubborn.killed
    assert pool._procs == []
    assert pool._started is False


def test_ensure_started_fails_fast_when_worker_dies_early():
    """worker 启动阶段早死：立即报错释放资源，不白等 startup_timeout。"""

    class FakeDyingProc:
        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

        def kill(self):
            pass

    class FakeCtx:
        def Queue(self):
            return std_queue.Queue()

        def Process(self, target=None, args=()):
            return FakeDyingProc()

    pool = _QwenWorkerPool(
        num_workers=1,
        gpu_ids=[0],
        model_path="fake-model",
        torch_dtype="auto",
        cache_dir=None,
        timeout=3600,
    )
    pool._ctx = FakeCtx()
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="启动失败"):
        pool._ensure_started()
    # startup_timeout 下限为 60s；早死检查必须在数秒内报错
    assert time.monotonic() - t0 < 10
    assert pool._started is False
    assert pool._procs == []


def test_llm_cache_key_includes_file_fingerprint(tmp_path):
    """同一路径文件内容变更（size/mtime 变化）后，旧缓存结论不得命中。"""
    import os

    from noise_filter_pipeline.core.llm_cache import LLMResultCache

    audio = os.path.join(str(tmp_path), "a.wav")
    with open(audio, "wb") as f:
        f.write(b"1234")
    cache = LLMResultCache(str(tmp_path / "cache"), model_name="m")
    cache.set(audio, "p", {"result": "noise"})
    assert cache.get(audio, "p") == {"result": "noise"}

    # 重写文件（size 变化）→ 旧结论不得命中
    with open(audio, "wb") as f:
        f.write(b"12345678")
    assert cache.get(audio, "p") is None


def test_llm_cache_namespace_isolates_qwen_model_identity(tmp_path):
    """持久缓存 namespace 变化后不得复用另一 Qwen 模型的结论。"""
    import os

    from noise_filter_pipeline.core.llm_cache import LLMResultCache

    audio = os.path.join(str(tmp_path), "a.wav")
    with open(audio, "wb") as f:
        f.write(b"audio")
    cache_a = LLMResultCache(
        str(tmp_path / "cache"), model_name="m", namespace="a" * 64
    )
    cache_b = LLMResultCache(
        str(tmp_path / "cache"), model_name="m", namespace="b" * 64
    )
    cache_a.set(audio, "p", {"result": "noise"})
    assert cache_a.get(audio, "p") == {"result": "noise"}
    assert cache_b.get(audio, "p") is None


def test_llm_cache_rejects_namespace_escape(tmp_path):
    """恶意 namespace 不得逃逸配置的 cache_dir。"""
    from noise_filter_pipeline.core.llm_cache import LLMResultCache

    for namespace in ("../escape", "/absolute", "a/b"):
        with pytest.raises(ValueError, match="namespace"):
            LLMResultCache(str(tmp_path / "cache"), namespace=namespace)
    assert not (tmp_path / "escape").exists()


def test_llm_cache_rejects_symlink_bucket(tmp_path):
    """预置 symlink bucket 时缓存不得读取或写入外部 target。"""
    import os

    from noise_filter_pipeline.core.llm_cache import LLMResultCache

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")
    cache = LLMResultCache(str(tmp_path / "cache"), model_name="m")
    key = cache._key(str(audio), "p", "m")
    outside = tmp_path / "outside"
    outside.mkdir()
    bucket = cache.cache_dir / key[:2]
    bucket.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="bucket"):
        cache.set(str(audio), "p", {"result": "noise"})
    assert not any(outside.iterdir())


def test_llm_cache_key_salt_isolates_generation_contract(tmp_path):
    """生成参数 salt 变化时不得命中旧缓存。"""
    from noise_filter_pipeline.core.llm_cache import LLMResultCache

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")
    first = LLMResultCache(
        str(tmp_path / "cache"), model_name="m", key_salt="max_new_tokens=8"
    )
    second = LLMResultCache(
        str(tmp_path / "cache"), model_name="m", key_salt="max_new_tokens=3"
    )
    first.set(str(audio), "p", {"result": "noise"})
    assert second.get(str(audio), "p") is None


def test_qwen_model_cache_isolates_verified_model_identity():
    """同路径但 Qwen SHA 变化时不得复用进程内模型实例。"""
    import noise_filter_pipeline.processors.qwen_single_event as qwen_module

    qwen_module._MODEL_CACHE.clear()
    try:
        model_a = qwen_module.get_qwen_model(
            "same-model-path", qwen_model_sha256="a" * 64
        )
        model_b = qwen_module.get_qwen_model(
            "same-model-path", qwen_model_sha256="b" * 64
        )
        assert model_a is not model_b
        assert (
            qwen_module.get_qwen_model(
                "same-model-path", qwen_model_sha256="a" * 64
            )
            is model_a
        )
    finally:
        qwen_module._MODEL_CACHE.clear()


def test_qwen_processor_binds_verified_identity_to_cache_namespace(tmp_path):
    """C1 处理器必须把已验证 Qwen SHA 传给持久缓存 namespace。"""
    import hashlib

    from noise_filter_pipeline.processors.qwen_single_event import (
        DEFAULT_PROMPT,
        QwenSingleEventProcessor,
    )

    model_sha256 = "c" * 64
    prompt_sha256 = hashlib.sha256(DEFAULT_PROMPT.encode("utf-8")).hexdigest()
    proc = QwenSingleEventProcessor(
        {
            "dry_run": True,
            "model_path": "same-model-path",
            "cache_dir": str(tmp_path / "cache"),
            "artifact_fingerprints": {
                "qwen_model_sha256": model_sha256,
                "c1_prompt_sha256": prompt_sha256,
            },
        }
    )
    assert proc.qwen_model_sha256 == model_sha256
    assert proc.cache.namespace == model_sha256
    assert proc.cache.cache_dir == tmp_path / "cache" / "c1_qwen" / model_sha256
