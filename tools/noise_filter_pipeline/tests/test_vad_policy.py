"""
VAD 与策略引擎回归测试。

覆盖：
  - Silero VAD 时间轴（stride 而非 window_size）
  - WebRTC VAD 残余尾帧跳过
  - create_vad_detector 参数归一
  - MultiVAD 失败检测器后的 results/vad_types 对齐
  - B4 尊重 B3 的 strict_vad_models / reject_on_any_speech 硬判定
"""
import os
import sys
import tempfile
import types

import numpy as np
import pytest
import soundfile as sf

from noise_filter_pipeline.processors.policy_engine import PolicyEngine
from noise_filter_pipeline.processors.vad import (
    MultiVADDetector,
    VADResult,
    WebRTCVADWrapper,
    create_vad_detector,
)

pytestmark = pytest.mark.pipeline


def _make_wav(path, duration=2.0, sr=16000):
    wav = np.random.randn(int(duration * sr)) * 0.01
    sf.write(path, wav, sr)


class _ConstantSileroModel:
    """恒定输出的 Silero 替身：prob_fn(window_index) -> prob。

    生产代码为 reset_states + 逐窗（batch=1）顺序前向（有状态模型的正确
    用法），替身用累计调用序号模拟窗口序号，reset_states 时归零。"""

    def __init__(self, prob_fn):
        self._prob_fn = prob_fn
        self._idx = 0
        self.reset_count = 0

    def reset_states(self):
        self._idx = 0
        self.reset_count += 1

    def __call__(self, batch, sr):
        import torch

        n = batch.shape[0]
        out = torch.tensor([[self._prob_fn(self._idx + i)] for i in range(n)])
        self._idx += n
        return out


def test_silero_vad_resets_state_between_calls():
    """有状态模型必须在每次 detect 前 reset：同一段音频连续两次检测，
    结果必须完全一致（旧实现跨调用不 reset，第二次会被第一次的尾状态污染）。"""
    from noise_filter_pipeline.processors.vad import SileroVADWrapper

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_path, duration=1.0)

        model = _ConstantSileroModel(lambda i: 0.9 if i < 2 else 0.1)
        wrapper = SileroVADWrapper.__new__(SileroVADWrapper)  # 跳过模型加载
        wrapper.threshold = 0.5
        wrapper.use_gpu = False
        wrapper.model = model

        r1 = wrapper.detect(wav_path)
        r2 = wrapper.detect(wav_path)
        assert r1.speech_ratio == r2.speech_ratio
        assert r1.speech_timestamps == r2.speech_timestamps
        # 每次 detect 都应触发一次 reset
        assert model.reset_count == 2


def test_silero_vad_time_axis_uses_stride():
    """窗口按 50% overlap 步进，时间轴必须用 stride：0/1 帧为语音、其余非语音时，
    语音段应为 [0, 2*stride/sr] ≈ 0.064s，而不是 2 倍。"""
    from noise_filter_pipeline.processors.vad import SileroVADWrapper

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_path, duration=1.0)

        wrapper = SileroVADWrapper.__new__(SileroVADWrapper)  # 跳过模型加载
        wrapper.threshold = 0.5
        wrapper.use_gpu = False
        wrapper.model = _ConstantSileroModel(lambda i: 0.9 if i < 2 else 0.1)

        result = wrapper.detect(wav_path)

        assert result.has_speech
        assert len(result.speech_timestamps) == 1
        start, end = result.speech_timestamps[0]
        stride_sec = 256 / 16000  # window 512，50% overlap
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(2 * stride_sec, rel=1e-6)
        # 语音占比应在量级上正确（远小于 2 倍高估的值）
        assert result.speech_ratio == pytest.approx((2 * stride_sec) / 1.0, rel=1e-6)


def test_webrtc_vad_skips_residual_tail_frame():
    """残余尾帧（长度 != frame_samples）必须跳过，不得抛异常导致整文件 fallback。"""
    calls = []

    class FakeVad:
        def __init__(self, mode=3):
            self.mode = mode

        def is_speech(self, frame_bytes, sample_rate):
            calls.append(len(frame_bytes))
            # 模拟真实 webrtcvad：长度非法直接抛错
            if len(frame_bytes) != int(sample_rate * 30 / 1000) * 2:
                raise ValueError("invalid frame length")
            return True

    fake_module = types.SimpleNamespace(Vad=FakeVad)
    orig = sys.modules.get("webrtcvad")
    sys.modules["webrtcvad"] = fake_module
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "a.wav")
            # 1.005s：最后 0.005s 是残余尾帧
            _make_wav(wav_path, duration=1.005)
            result = WebRTCVADWrapper(mode=3).detect(wav_path)
    finally:
        if orig is None:
            sys.modules.pop("webrtcvad", None)
        else:
            sys.modules["webrtcvad"] = orig

    assert result.vad_type == "webrtc"
    assert result.has_speech
    # 所有送入 is_speech 的帧都是合法长度
    assert all(n == 16000 * 30 // 1000 * 2 for n in calls)
    assert len(calls) == 33  # 1.005s -> 33 个完整 30ms 帧 + 1 个被跳过的尾帧


def test_create_vad_detector_normalizes_kwargs():
    """单 VAD 模式的 canonical 参数名必须正确映射到各 wrapper。"""
    # webrtc：不接受任何额外 kwargs，旧实现直接 TypeError
    det = create_vad_detector("webrtc", model_dir="/x", onnx_path="/y", use_gpu=True, threshold=0.3)
    assert isinstance(det, WebRTCVADWrapper)

    # nemo_onnx：onnx_path / threshold 必须生效（旧实现读 nemo_onnx_path/speech_threshold 导致丢失）
    from noise_filter_pipeline.processors.vad import NeMoONNXVADWrapper

    det = create_vad_detector("nemo_onnx", onnx_path="/tmp/model.onnx", threshold=0.3, use_gpu=True)
    assert isinstance(det, NeMoONNXVADWrapper)
    assert det.threshold == pytest.approx(0.3)

    # firered：threshold 必须映射到 speech_threshold（旧实现被 **kwargs 吞掉，静默用默认值）
    from unittest.mock import patch
    from noise_filter_pipeline.processors.vad import FireRedVADWrapper

    # 避免真实加载 FireRedVAD 模型（测试环境可能无模型文件）
    with patch.object(FireRedVADWrapper, "_load_model", lambda self, **kwargs: None):
        det = create_vad_detector("firered", model_dir="/tmp/firered", threshold=0.3)
        assert isinstance(det, FireRedVADWrapper)
        assert det.speech_threshold == pytest.approx(0.3)


def test_vad_pool_worker_binds_physical_gpu_id_when_cvd_restricted(monkeypatch):
    """父进程 CUDA_VISIBLE_DEVICES 受限时，worker 必须绑定物理 id 而非序号。

    回归：CVD='2,3' 时 device_count()=2，旧实现把 worker 序号 0/1 当作物理
    卡号写回 CVD，worker 实际绑到物理 0/1（可能是别的任务的卡）。
    """
    import contextlib

    import torch

    from noise_filter_pipeline.processors.vad import multi as vad_multi

    det = MultiVADDetector.__new__(MultiVADDetector)  # 跳过模型加载
    det.num_workers = 2
    det._ctor_kwargs = {"use_gpu": True}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    captured = {}

    class _FakePool:
        def __init__(self, max_workers, mp_context, initializer, initargs):
            captured["initializer"] = initializer
            captured["initargs"] = initargs

    monkeypatch.setattr(vad_multi, "ProcessPoolExecutor", _FakePool)
    det._create_pool()
    _, _, gpu_ids = captured["initargs"]
    assert gpu_ids == ["2", "3"]

    # worker 序号 0/1 必须 round-robin 绑定物理卡 2/3
    class _Counter:
        def __init__(self):
            self.value = 0

        @contextlib.contextmanager
        def get_lock(self):
            yield

    monkeypatch.setattr(vad_multi, "MultiVADDetector", lambda **kw: object())
    monkeypatch.setattr(
        "noise_filter_pipeline.core.process_guard.start_parent_watchdog",
        lambda *a, **k: None,
    )

    counter = _Counter()
    vad_multi._vad_worker_init({}, counter, ["2", "3"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    vad_multi._vad_worker_init({}, counter, ["2", "3"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    # 序号超出卡数时取模回绕
    vad_multi._vad_worker_init({}, counter, ["2", "3"])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    # 生产 _vad_worker_init 内部直接写 CUDA_VISIBLE_DEVICES（测试侧无法避免），
    # 这里显式清理残余值；即使断言中途失败，monkeypatch.setenv 的 teardown
    # 也会把该变量恢复为测试前的未设置状态，环境不会泄漏到其他用例。
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def test_vad_pool_gpu_ids_default_to_range_when_cvd_unset(monkeypatch):
    """未设置 CUDA_VISIBLE_DEVICES 时，物理 id 列表退化为 range(device_count)。"""
    import torch

    from noise_filter_pipeline.processors.vad import multi as vad_multi

    det = MultiVADDetector.__new__(MultiVADDetector)
    det.num_workers = 2
    det._ctor_kwargs = {"use_gpu": True}

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    captured = {}

    class _FakePool:
        def __init__(self, max_workers, mp_context, initializer, initargs):
            captured["initargs"] = initargs

    monkeypatch.setattr(vad_multi, "ProcessPoolExecutor", _FakePool)
    det._create_pool()
    _, _, gpu_ids = captured["initargs"]
    assert gpu_ids == ["0", "1", "2"]


def test_multivad_pooled_mode_parent_detectors_lazy(monkeypatch):
    """num_workers>1 时父进程 detectors 必须懒加载。

    回归：pooled 模式下父进程从不使用本地 detectors（worker 按
    _ctor_kwargs 各自构造），旧实现构造时即加载整套模型，白占内存/显存。
    修复后：构造与 _create_pool 均不加载，仅非池化 detect_array 首次
    使用时才加载。
    """
    from noise_filter_pipeline.processors.vad import multi as vad_multi

    builds = []

    class _FakeWrapper:
        def __init__(self, **kwargs):
            builds.append(kwargs)

        def detect_array(self, audio, sr):
            return VADResult(False, [], 0.0, 0.0, vad_type="fake")

    monkeypatch.setattr(vad_multi, "FallbackVAD", _FakeWrapper)

    det = MultiVADDetector(vad_types=["energy"], num_workers=2)
    assert det.detectors == {}
    assert builds == []

    # _create_pool 不触发父进程加载（worker 内才按 _ctor_kwargs 构造）
    class _FakePool:
        def __init__(self, max_workers, mp_context, initializer, initargs):
            pass

    monkeypatch.setattr(vad_multi, "ProcessPoolExecutor", _FakePool)
    det._create_pool()
    assert det.detectors == {}
    assert builds == []

    # 非池化 detect_array 首次使用时才加载
    audio = np.zeros(16000, dtype=np.float32)
    result = det.detect_array(audio, 16000)
    assert set(det.detectors) == {"energy"}
    assert len(builds) == 1
    assert result.final_category == "noise"


def test_multivad_single_process_eagerly_loads_detectors(monkeypatch):
    """num_workers<=1 时保持即时加载（构造即预检模型可用性，行为不变）。"""
    from noise_filter_pipeline.processors.vad import multi as vad_multi

    monkeypatch.setattr(
        vad_multi, "FallbackVAD", lambda **kw: types.SimpleNamespace()
    )
    det = MultiVADDetector(vad_types=["energy"], num_workers=1)
    assert set(det.detectors) == {"energy"}


def test_multivad_rejects_unknown_detector_type():
    """VAD 配置拼写错误不得静默得到空检测器并错误放行。"""
    with pytest.raises(ValueError, match="不支持"):
        MultiVADDetector(vad_types=["not-a-vad"])


def test_multivad_failed_detector_keeps_names_aligned():
    """某个 VAD 失败后，vad_results 的键必须与成功的检测器一一对应。"""

    class _OK:
        def detect(self, path):
            return VADResult(True, [(0.0, 1.0)], 0.5, 0.9, vad_type="ok")

    class _Boom:
        def detect(self, path):
            raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_path)

        det = MultiVADDetector.__new__(MultiVADDetector)
        det.vad_types = ["first", "second", "third"]
        det.detectors = {"first": _OK(), "second": _Boom(), "third": _OK()}
        det.speech_ratio_threshold = 0.05
        det.strict_vad_models = False
        det.reject_on_any_speech = False

        result = det.detect(wav_path)

    vad_results = result.get_result_dict()["vad_results"]
    assert set(vad_results.keys()) == {"first", "third"}
    assert "second" not in vad_results


def test_create_vad_detector_rejects_unknown_type():
    """未知 vad_type 不得静默退化为 FallbackVAD（对齐 MultiVADDetector 的 ValueError）。

    配置拼写错误必须尽早暴露；静默 fallback 会让 strict 部署在模型名写错时
    全程用能量 VAD 跑完整个 stage。
    """
    with pytest.raises(ValueError, match="不支持"):
        create_vad_detector("not-a-vad")


def test_vad_single_strict_rejects_energy_fallback_result():
    """单 VAD + strict：结果为 energy fallback 时必须判 B3 rejected。

    回归测试：单 VAD 分支此前不消费 strict_vad_models——模型加载失败退化
    为 energy fallback 后仍按正常结果 passed，与 MultiVAD 的 strict 口径
    （任一指定 VAD 退化 fallback 即 rejected）不一致。
    """
    from noise_filter_pipeline.core.manifest_io import AudioRecord
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    class _EnergyDetector:
        def detect_batch_arrays(self, audio_arrays):
            return [VADResult(False, [], 0.0, 0.0, vad_type="energy") for _ in audio_arrays]

    def _run(strict):
        processor = VADProcessor(
            {
                "enable": True,
                "use_multi_vad": False,
                "vad_type": "energy",
                "strict_vad_models": strict,
            }
        )
        processor._detector = _EnergyDetector()
        rec = AudioRecord(
            record_id="ds__src__chunk_0001",
            dataset_id="ds",
            source_id="src",
            chunk_id="chunk_0001",
            audio={"path": "/not/used", "sr": 16000, "duration": 2.0, "channels": 1},
        )
        arr = np.zeros(32000, dtype=np.float32)
        return processor.process_records_with_arrays([rec], [(arr, 16000)])[0]

    rec_strict = _run(True)
    b3 = rec_strict.stages.get("B3", {})
    assert b3.get("status") == "rejected"
    assert b3.get("reason") == "vad_model_fallback"
    assert rec_strict.rejected_reason == "b3_vad_model_fallback"

    # 非 strict 对照：同样的 energy 结果按正常 VAD 结果放行
    rec_lax = _run(False)
    assert rec_lax.stages.get("B3", {}).get("status") == "passed"


def test_vad_processor_rejects_unknown_vad_type_fail_fast():
    """vad_types 非法值必须在 __init__ fail-fast，不得被预加载容错吞掉。

    回归测试：此前预加载 blanket except 会吞掉 MultiVADDetector 的
    ValueError，推理时再抛被 batch 级 except 兜底成整批 vad_batch_failed，
    stage "成功"但 0 产出。
    """
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    with pytest.raises(ValueError, match="不支持的 VAD 类型"):
        VADProcessor({"enable": False, "vad_types": ["silerro"]})


def test_vad_processor_normalizes_string_vad_types():
    """vad_types 误写为字符串时归一化为单元素 list。

    回归测试：字符串会被 set() 按字符拆散（"silero" -> {s,i,l,e,r,o}），
    全部判为未知类型后静默降级为 vad_batch_failed。
    """
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor({"enable": False, "vad_types": "silero"})
    assert processor.vad_types == ["silero"]


def test_firered_load_runtime_error_falls_back(monkeypatch, tmp_path):
    """FireRedVAD from_pretrained 抛非 ImportError（权重损坏/OOM）时必须置 None 走 fallback。

    回归：旧实现只 catch ImportError，其他异常传播导致每批重试加载。
    """
    from noise_filter_pipeline.processors.vad import FireRedVADWrapper

    class _FakeFireRedVad:
        @staticmethod
        def from_pretrained(model_dir, vad_config):
            raise RuntimeError("corrupted weights")

    fake_module = types.SimpleNamespace(
        FireRedVad=_FakeFireRedVad,
        FireRedVadConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "fireredvad", fake_module)

    wrapper = FireRedVADWrapper(model_dir="/tmp/firered", use_gpu=False)
    assert wrapper.vad is None

    wav_path = os.path.join(tmp_path, "a.wav")
    _make_wav(wav_path)
    result = wrapper.detect(wav_path)
    assert result.vad_type == "energy"


def test_funasr_explicit_model_dir_runtime_error_falls_back(monkeypatch, tmp_path):
    """FunASR 显式 model_dir 分支 AutoModel 抛非 ImportError 时必须置 None 走 fallback。

    回归：旧实现该分支无 try 保护（仅 hub 默认分支有），异常传播导致每批重试加载。
    """
    from noise_filter_pipeline.processors.vad import FunASRVADWrapper

    class _FakeAutoModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("corrupted weights")

    fake_module = types.SimpleNamespace(AutoModel=_FakeAutoModel)
    monkeypatch.setitem(sys.modules, "funasr", fake_module)

    model_dir = tmp_path / "funasr_model"
    model_dir.mkdir()
    wrapper = FunASRVADWrapper(model_dir=str(model_dir), use_gpu=False)
    assert wrapper.model is None

    wav_path = os.path.join(tmp_path, "a.wav")
    _make_wav(wav_path)
    result = wrapper.detect(wav_path)
    assert result.vad_type == "energy"


def test_vad_file_mode_batch_failure_marks_vad_batch_failed_non_strict(tmp_path):
    """file-mode 非 strict：整批 infra 失败必须标 vad_batch_failed（rejected）。

    与 array-mode 口径一致；此前标 suspected 放行给 B4，infra 失败无法被
    checkpoint 排除逻辑覆盖（应可重试而非按内容级拒绝永久沉淀）。
    """
    from noise_filter_pipeline.core.manifest_io import (
        AudioRecord,
        read_manifest,
        write_manifest,
    )
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    wav_path = tmp_path / "a.wav"
    _make_wav(str(wav_path))
    manifest = tmp_path / "in.jsonl"
    rec = AudioRecord(
        record_id="ds__src__chunk_0001",
        dataset_id="ds",
        source_id="src",
        chunk_id="chunk_0001",
        audio={"path": str(wav_path), "sr": 16000, "duration": 2.0, "channels": 1},
    )
    write_manifest(iter([rec]), str(manifest))

    processor = VADProcessor(
        {
            "enable": True,
            "use_multi_vad": False,
            "vad_type": "energy",
            "strict_vad_models": False,
        }
    )

    class _BoomDetector:
        def detect_batch(self, paths):
            raise RuntimeError("simulated infra failure")

    processor._detector = _BoomDetector()
    out = processor.process(str(manifest), str(tmp_path / "out"))
    recs = list(read_manifest(out))
    assert len(recs) == 1
    b3 = recs[0].stages["B3"]
    assert b3["status"] == "rejected"
    assert b3["reason"] == "vad_batch_failed"
    assert recs[0].rejected_reason == "b3_vad_batch_failed"


def test_nemo_wrapper_moves_model_to_cuda_when_use_gpu(monkeypatch):
    """NeMoVADWrapper 必须消费 use_gpu：use_gpu 且 cuda 可用时模型 .to('cuda')。"""
    import torch

    from noise_filter_pipeline.processors.vad import NeMoVADWrapper

    class _FakeModel:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    models = []

    def _from_pretrained(model_name):
        m = _FakeModel()
        models.append(m)
        return m

    fake_models_mod = types.SimpleNamespace(
        EncDecClassificationModel=types.SimpleNamespace(from_pretrained=_from_pretrained)
    )
    # from-import 按全名查 sys.modules；父链一并占位，避免触发真实 nemo 导入
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", fake_models_mod)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    wrapper = NeMoVADWrapper(use_gpu=True)
    assert wrapper.model is models[0]
    assert models[0].moved_to == "cuda"

    NeMoVADWrapper(use_gpu=False)
    assert models[1].moved_to is None


def test_ensure_fireredvad_path_rolls_back_sys_path_on_import_failure(monkeypatch, tmp_path):
    """候选路径 import 失败后必须从 sys.path 移除，不得残留无效路径污染解析。"""
    from noise_filter_pipeline.processors.vad import bootstrap

    (tmp_path / "FireRedVAD").mkdir()
    monkeypatch.setenv("USE_TRAIN_THIRD_PARTY_DIR", str(tmp_path))
    # 强制 fireredvad 不可 import（sys.modules 置 None 触发 ImportError）
    monkeypatch.setitem(sys.modules, "fireredvad", None)

    before = list(sys.path)
    bootstrap._ensure_fireredvad_path()
    assert sys.path == before


def test_fireredvad_download_lock_serializes_concurrent_download(monkeypatch, tmp_path):
    """并发首次下载必须互斥：只有一个进程实际下载，其余等锁后复用结果。

    回归：VAD worker 等多进程同时首次进入下载会互相写出半成品权重。
    """
    import threading
    import time

    from noise_filter_pipeline.processors.vad import bootstrap

    monkeypatch.setenv("USE_TRAIN_THIRD_PARTY_DIR", str(tmp_path / "third_party"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(bootstrap, "_fireredvad_root_cache", None)
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK_TIMEOUT_SEC", 10.0)
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK_POLL_SEC", 0.02)
    monkeypatch.setitem(sys.modules, "modelscope", None)  # 强制 huggingface 后端

    download_calls = []

    def _fake_snapshot_download(repo_id, local_dir):
        download_calls.append(repo_id)
        time.sleep(0.2)  # 模拟下载耗时，制造并发窗口
        os.makedirs(os.path.join(local_dir, "VAD"), exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_fake_snapshot_download),
    )

    results = []

    def _worker():
        results.append(bootstrap._get_or_download_fireredvad_root())

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(download_calls) == 1
    assert len(results) == 4
    assert all(r is not None and r == results[0] for r in results)


def test_fireredvad_download_lock_timeout_returns_none(monkeypatch, tmp_path):
    """锁文件残留（持有进程崩溃）时不得死锁：超时返回 None 走 fallback。"""
    from noise_filter_pipeline.processors.vad import bootstrap

    monkeypatch.setenv("USE_TRAIN_THIRD_PARTY_DIR", str(tmp_path / "third_party"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(bootstrap, "_fireredvad_root_cache", None)
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK_TIMEOUT_SEC", 0.2)
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK_POLL_SEC", 0.05)
    monkeypatch.setitem(sys.modules, "modelscope", None)

    cache_parent = tmp_path / "cache" / "use_train"
    cache_parent.mkdir(parents=True)
    (cache_parent / "FireRedVAD").mkdir()
    # 模拟崩溃进程残留的锁文件
    (cache_parent / "FireRedVAD.download.lock").touch()

    download_calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **kwargs: download_calls.append(kwargs)
        ),
    )

    assert bootstrap._get_or_download_fireredvad_root() is None
    assert download_calls == []


def test_fireredvad_download_lock_stale_holder_cleared(monkeypatch, tmp_path):
    """锁持有进程已死（SIGKILL/OOM）时残留锁必须被清理重抢。

    回归：此前锁只由持锁进程 unlink，崩溃后锁文件永久残留，后续每次
    运行都干等满 timeout（1800s），自动下载永久失效。
    """
    import time

    from noise_filter_pipeline.processors.vad import bootstrap

    monkeypatch.setenv("USE_TRAIN_THIRD_PARTY_DIR", str(tmp_path / "third_party"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(bootstrap, "_fireredvad_root_cache", None)
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK_TIMEOUT_SEC", 30.0)
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK_POLL_SEC", 0.05)
    monkeypatch.setitem(sys.modules, "modelscope", None)

    cache_parent = tmp_path / "cache" / "use_train"
    cache_parent.mkdir(parents=True)
    (cache_parent / "FireRedVAD").mkdir()
    # 模拟崩溃进程残留的锁：内容为已不存在的 pid
    (cache_parent / "FireRedVAD.download.lock").write_text("999999999\n")

    download_calls = []

    def _fake_snapshot_download(repo_id, local_dir):
        download_calls.append(repo_id)
        os.makedirs(os.path.join(local_dir, "VAD"), exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=_fake_snapshot_download),
    )

    t0 = time.monotonic()
    result = bootstrap._get_or_download_fireredvad_root()
    # 死锁被清理并完成下载（远小于 30s 超时，证明走的是自愈而非超时路径）
    assert result is not None
    assert time.monotonic() - t0 < 10
    assert len(download_calls) == 1
    # 残留锁已被替换为本次持锁（清理后重抢成功）
    lock_path = cache_parent / "FireRedVAD.download.lock"
    assert not lock_path.exists()


def test_policy_engine_respects_strict_vad_models_fallback():
    """strict_vad_models 下有 VAD 退化为 fallback 时，B4 不得把 B3 的 rejected 翻成 noise。"""
    engine = PolicyEngine({"keep_noise": True, "keep_suspected": True, "keep_rejected": False})
    chunk = {
        "vad_result": {
            "strict_vad_models": True,
            "individual_results": [
                {"vad_type": "silero", "speech_ratio": 0.0, "is_real_model": True},
                {"vad_type": "energy", "speech_ratio": 0.0, "is_real_model": False},
            ],
        }
    }
    pr = engine.apply_to_chunk(chunk)
    assert pr.final_category == "rejected"
    assert pr.kept is False


def test_policy_engine_respects_reject_on_any_speech():
    """reject_on_any_speech=True 时，fail_count>0 必须 rejected 而非 suspected。"""
    engine = PolicyEngine({"keep_noise": True, "keep_suspected": True, "keep_rejected": False})
    chunk = {
        "vad_result": {
            "reject_on_any_speech": True,
            "individual_results": [
                {"vad_type": "silero", "speech_ratio": 0.2},
                {"vad_type": "webrtc", "speech_ratio": 0.0},
            ],
        }
    }
    pr = engine.apply_to_chunk(chunk)
    assert pr.final_category == "rejected"
    assert pr.kept is False


def test_vad_result_serializes_for_b4_recompute():
    """单 VAD 模式下 VADResult 必须能序列化出 speech_ratio，供 B4 重算。"""
    r = VADResult(True, [(0.0, 1.0)], 0.3, 0.9, vad_type="silero", frame_probs=[0.1, 0.9])
    d = r.get_result_dict()
    assert d["speech_ratio"] == pytest.approx(0.3)
    assert d["vad_type"] == "silero"
    assert d["frame_probs"] == [0.1, 0.9]


def test_vad_process_records_with_arrays_no_output_manifest_non_strict():
    """process_records_with_arrays 在结果缺失且 output_manifest=None 时不应崩溃。

    Stage B unified 调用本方法时只要求内存中标记 B3 状态，不写 manifest；
    当 detector 返回结果少于输入记录时，非 strict 模式应标记为 suspected
    （由 B4 策略拒绝），而不是 passed。
    """
    from noise_filter_pipeline.core.manifest_io import AudioRecord
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor(
        {
            "enable": True,
            "use_multi_vad": False,
            "vad_type": "energy",
            "strict_vad_models": False,
        }
    )

    class _ShortDetector:
        def detect_batch_arrays(self, audio_arrays):
            return []

    processor._detector = _ShortDetector()

    rec = AudioRecord(
        record_id="ds__src__chunk_0001",
        dataset_id="ds",
        source_id="src",
        chunk_id="chunk_0001",
        audio={"path": "/not/used", "sr": 16000, "duration": 2.0, "channels": 1},
    )
    arr = np.zeros(32000, dtype=np.float32)
    results = processor.process_records_with_arrays([rec], [(arr, 16000)])
    assert len(results) == 1
    b3 = results[0].stages.get("B3", {})
    assert b3.get("status") == "suspected"
    assert b3.get("reason") == "vad_failed"


def test_vad_process_records_with_arrays_strict_rejects_on_missing_results():
    """strict 模式下 VAD 结果缺失应直接拒绝，并设置 rejected_reason。"""
    from noise_filter_pipeline.core.manifest_io import AudioRecord
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor(
        {
            "enable": True,
            "use_multi_vad": False,
            "vad_type": "energy",
            "strict_vad_models": True,
        }
    )

    class _ShortDetector:
        def detect_batch_arrays(self, audio_arrays):
            return []

    processor._detector = _ShortDetector()

    rec = AudioRecord(
        record_id="ds__src__chunk_0001",
        dataset_id="ds",
        source_id="src",
        chunk_id="chunk_0001",
        audio={"path": "/not/used", "sr": 16000, "duration": 2.0, "channels": 1},
    )
    arr = np.zeros(32000, dtype=np.float32)
    results = processor.process_records_with_arrays([rec], [(arr, 16000)])
    assert len(results) == 1
    b3 = results[0].stages.get("B3", {})
    assert b3.get("status") == "rejected"
    assert b3.get("reason") == "vad_failed"
    assert results[0].rejected_reason == "b3_vad_failed"


def test_vad_process_records_with_arrays_batch_exception_non_strict():
    """detector 整批抛异常时，非 strict 模式也必须标记 vad_batch_failed（rejected），

    与 strict 路径一致：infra 失败不写 checkpoint，下次运行可重试；
    此前标 suspected/vad_failed 会被 B4 拒绝并永久 checkpoint，无法再重试。
    """
    from noise_filter_pipeline.core.manifest_io import AudioRecord
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor(
        {
            "enable": True,
            "use_multi_vad": False,
            "vad_type": "energy",
            "strict_vad_models": False,
        }
    )

    class _BoomDetector:
        def detect_batch_arrays(self, audio_arrays):
            raise RuntimeError("simulated VAD failure")

    processor._detector = _BoomDetector()

    rec = AudioRecord(
        record_id="ds__src__chunk_0001",
        dataset_id="ds",
        source_id="src",
        chunk_id="chunk_0001",
        audio={"path": "/not/used", "sr": 16000, "duration": 2.0, "channels": 1},
    )
    arr = np.zeros(32000, dtype=np.float32)
    results = processor.process_records_with_arrays([rec], [(arr, 16000)])
    assert len(results) == 1
    b3 = results[0].stages.get("B3", {})
    assert b3.get("status") == "rejected"
    assert b3.get("reason") == "vad_batch_failed"
    assert results[0].rejected_reason == "b3_vad_batch_failed"


def test_policy_engine_honors_b3_vad_category_rejected():
    """B3 已按自身严格阈值判 rejected 时，B4 不得用宽松阈值重算翻案。"""
    engine = PolicyEngine({
        "vad_speech_ratio_threshold": 0.9,  # 故意偏宽，模拟 B3/B4 阈值不一致
        "keep_noise": True,
        "keep_suspected": True,
        "keep_rejected": False,
    })
    chunk = {
        "vad_category": "rejected",
        "vad_result": {
            "individual_results": [
                {"vad_type": "silero", "speech_ratio": 0.02},
            ],
        },
    }
    pr = engine.apply_to_chunk(chunk)
    assert pr.final_category == "rejected"
    assert pr.kept is False


def test_policy_engine_strict_threshold_rejects_low_speech_ratio():
    """live check：silero speech_ratio=0.02 在严格阈值 0.005 + reject_on_any_speech 下必须 rejected。"""
    engine = PolicyEngine({
        "vad_speech_ratio_threshold": 0.005,
        "keep_noise": True,
        "keep_suspected": False,
        "keep_rejected": False,
    })
    chunk = {
        "vad_category": "rejected",
        "vad_result": {
            "reject_on_any_speech": True,
            "individual_results": [
                {"vad_type": "silero", "speech_ratio": 0.02},
                {"vad_type": "funasr", "speech_ratio": 0.0},
                {"vad_type": "firered", "speech_ratio": 0.0},
            ],
        },
    }
    pr = engine.apply_to_chunk(chunk)
    assert pr.final_category == "rejected"
    assert pr.kept is False


def test_multivad_strict_treats_raising_detector_as_failure():
    """strict 模式下指定 VAD 推理抛异常（非 energy fallback）也必须判 rejected。"""
    from noise_filter_pipeline.processors.vad import MultiVADResult

    class _OK:
        def detect(self, path):
            return VADResult(True, [(0.0, 1.0)], 0.0, 0.9, vad_type="silero")

    class _Boom:
        def detect(self, path):
            raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "a.wav")
        _make_wav(wav_path)

        det = MultiVADDetector.__new__(MultiVADDetector)
        det.vad_types = ["silero", "funasr"]
        det.detectors = {"silero": _OK(), "funasr": _Boom()}
        det.speech_ratio_threshold = 0.005
        det.strict_vad_models = True
        det.reject_on_any_speech = True

        result = det.detect(wav_path)

    assert result.final_category == "rejected"
    d = result.get_result_dict()
    assert d["failed_vad_types"] == ["funasr"]
    assert d["has_fallback"] is True
    # 非 strict 对照：同样的抛异常不应触发强制拒绝
    non_strict = MultiVADResult(
        result.results,
        vad_types=["silero"],
        speech_ratio_threshold=0.005,
        strict_vad_models=False,
        failed_vad_types=["funasr"],
    )
    assert non_strict.final_category == "noise"


def test_policy_processor_delete_rejected_audio_only_under_output_root():
    """delete_rejected_audio 只允许删除输出根目录内的文件，不得误删用户源音频。"""
    from noise_filter_pipeline.core.manifest_io import AudioRecord
    from noise_filter_pipeline.processors.policy_processor import PolicyProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        out_root = os.path.join(tmpdir, "out")
        os.makedirs(out_root)
        # 输出目录内的 rejected chunk：应被删除
        inside_path = os.path.join(out_root, "chunk.wav")
        _make_wav(inside_path)
        # 输出目录外的用户源音频：必须保留
        outside_path = os.path.join(tmpdir, "source.wav")
        _make_wav(outside_path)

        processor = PolicyProcessor({
            "delete_rejected_audio": True,
            "vad_speech_ratio_threshold": 0.05,
        })
        processor.allowed_delete_root = out_root

        def _rejected_record(path, rid):
            rec = AudioRecord(
                record_id=rid,
                dataset_id="ds",
                source_id=rid,
                chunk_id="source",
                audio={"path": path, "sr": 16000, "duration": 2.0, "channels": 1},
            )
            rec.mark_stage("B3", "passed", extra={"vad_result": {"speech_ratio": 1.0}})
            return rec

        rec_in = processor._process_record(_rejected_record(inside_path, "ds__in__source"))
        assert not rec_in.is_stage_passed("B4")
        assert not os.path.exists(inside_path), "输出根目录内的 rejected 音频应被删除"

        rec_out = processor._process_record(_rejected_record(outside_path, "ds__out__source"))
        assert not rec_out.is_stage_passed("B4")
        assert os.path.exists(outside_path), "输出根目录外的用户源音频不得被删除"


def test_policy_engine_explicit_none_vad_result_is_rejected():
    """缺失 VAD 证据时不得信任输入携带的 final_category。"""
    engine = PolicyEngine({"vad_speech_ratio_threshold": 0.05})
    chunk = {"vad_result": None, "final_category": "noise"}
    pr = engine.apply(chunk)
    assert pr.final_category == "rejected"
    assert pr.fail_count == 2

    chunk_rej = {"vad_result": None, "final_category": "rejected"}
    pr_rej = engine.apply(chunk_rej)
    assert pr_rej.final_category == "rejected"
    assert pr_rej.fail_count == 2


@pytest.mark.parametrize(
    "vad_result",
    [
        {"speech_ratio": np.nan},
        {"speech_ratio": np.inf},
        {"speech_ratio": -np.inf},
        {"individual_results": "not-a-list"},
        {"individual_results": []},
        {"individual_results": ["not-a-dict"]},
        {"individual_results": [{"speech_ratio": np.nan}]},
        {"individual_results": [{"speech_ratio": 1.1}]},
        {"vad_results": []},
        {"vad_results": {}},
        {"vad_results": {"silero": np.nan}},
        {"vad_results": {"silero": np.inf}},
        {"vad_results": {"silero": -0.1}},
        {"vad_results": {"silero": 1.1}},
    ],
    ids=[
        "speech-ratio-nan",
        "speech-ratio-inf",
        "speech-ratio-negative-inf",
        "individual-not-list",
        "individual-empty",
        "individual-not-dict",
        "individual-ratio-nan",
        "individual-ratio-out-of-range",
        "vad-results-not-dict",
        "vad-results-empty",
        "vad-results-nan",
        "vad-results-inf",
        "vad-results-negative",
        "vad-results-over-one",
    ],
)
def test_policy_engine_rejects_malformed_vad_evidence(vad_result):
    """VAD 证据结构、有限性或概率范围非法时必须 fail-closed。"""
    engine = PolicyEngine({"vad_speech_ratio_threshold": 0.05})
    pr = engine.apply({"vad_result": vad_result, "final_category": "noise"})
    assert pr.final_category == "rejected"
    assert pr.kept is False
    assert pr.fail_count == 2


class _SeqStateModel:
    """模拟 silero 有状态模型：每次前向对 batch 内所有序列返回 fn(step)。

    逐窗顺序（batch=1）与 lockstep（batch=K）两种喂法下，第 t 个窗口
    都应得到 fn(t)，可用于验证两种路径结果一致。"""

    def __init__(self, fn):
        self.fn = fn
        self.step = 0
        self.reset_count = 0

    def reset_states(self):
        self.step = 0
        self.reset_count += 1

    def __call__(self, batch, sr):
        import torch

        n = batch.shape[0]
        out = torch.tensor([[self.fn(self.step)] for _ in range(n)])
        self.step += 1
        return out


def _make_silero_wrapper(model):
    from noise_filter_pipeline.processors.vad import SileroVADWrapper

    wrapper = SileroVADWrapper.__new__(SileroVADWrapper)  # 跳过模型加载
    wrapper.threshold = 0.5
    wrapper.use_gpu = False
    wrapper.model = model
    return wrapper


def test_silero_detect_arrays_batch_matches_sequential():
    """lockstep 批量检测与逐条顺序检测结果必须完全一致（含不等长与超短输入）。"""
    wrapper = _make_silero_wrapper(_SeqStateModel(lambda t: 0.9 if t % 3 == 0 else 0.1))

    rng = np.random.RandomState(0)
    audios = [
        rng.randn(16000).astype(np.float32),          # 1s，61 窗
        rng.randn(8000).astype(np.float32) * 0.01,    # 0.5s，29 窗（不等长）
        np.zeros(100, dtype=np.float32),              # 不足一窗
    ]
    batch_results = wrapper.detect_arrays_batch(audios, 16000)
    seq_results = [wrapper.detect_array(a, 16000) for a in audios]

    assert len(batch_results) == len(audios)
    for rb, rs in zip(batch_results, seq_results):
        assert rb.speech_ratio == rs.speech_ratio
        assert rb.speech_timestamps == rs.speech_timestamps
        assert rb.frame_probs == rs.frame_probs


def test_multivad_detect_arrays_group_matches_per_chunk(tmp_path):
    """MultiVAD 分组路径与逐条 detect_array 的结果（含投票/失败标记）必须一致。"""
    silero = _make_silero_wrapper(_SeqStateModel(lambda t: 0.9 if t < 2 else 0.1))

    class _FakeVAD:
        def detect_array(self, audio, sr):
            return VADResult(True, [(0, 1.0)], 0.5, 0.9, "fake", frame_probs=[0.9])

    det = MultiVADDetector.__new__(MultiVADDetector)  # 跳过模型加载
    det.speech_ratio_threshold = 0.05
    det.strict_vad_models = False
    det.reject_on_any_speech = False
    det.temp_dir = str(tmp_path)
    det.detectors = {"silero": silero, "fake": _FakeVAD()}

    rng = np.random.RandomState(1)
    arrays = [
        (rng.randn(16000).astype(np.float32), 16000),
        (rng.randn(8000).astype(np.float32), 16000),
    ]
    group = det.detect_arrays_group(arrays)
    per_chunk = [det.detect_array(a, sr) for a, sr in arrays]

    assert len(group) == len(arrays)
    for g, p in zip(group, per_chunk):
        assert g.vad_types == p.vad_types
        assert g.failed_vad_types == p.failed_vad_types
        assert g.fail_count == p.fail_count
        assert g.final_category == p.final_category
        for rg, rp in zip(g.results, p.results):
            assert rg.vad_type == rp.vad_type
            assert rg.speech_ratio == rp.speech_ratio


def test_vad_result_extra_strips_frame_probs_by_default():
    """B3 extra 落盘默认剥离 frame_probs（占单条记录体积 ~95%）。

    顶层与 individual_results 内每项的 frame_probs 都必须被移除，
    其余字段原样保留；输入 dict 不得被原地修改。
    """
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor({"enable": False, "vad_types": "silero"})
    assert processor.keep_frame_probs is False

    payload = {
        "speech_ratio": 0.5,
        "frame_probs": [0.1, 0.9],
        "individual_results": [
            {"vad_type": "silero", "speech_ratio": 0.5, "frame_probs": [0.1, 0.9]},
            {"vad_type": "energy", "speech_ratio": 0.4},
            "not-a-dict",
        ],
    }

    class _FakeResult:
        def get_result_dict(self):
            return payload

    out = processor._vad_result_extra(_FakeResult())
    assert "frame_probs" not in out
    assert out["speech_ratio"] == 0.5
    ind = out["individual_results"]
    assert "frame_probs" not in ind[0]
    assert ind[0]["vad_type"] == "silero"
    assert ind[1] == {"vad_type": "energy", "speech_ratio": 0.4}
    assert ind[2] == "not-a-dict"
    # 输入不得被原地修改
    assert "frame_probs" in payload
    assert "frame_probs" in payload["individual_results"][0]


def test_vad_result_extra_keeps_frame_probs_when_enabled():
    """keep_frame_probs: true 时完整保留 frame_probs（调试用开关）。"""
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor(
        {"enable": False, "vad_types": "silero", "keep_frame_probs": True}
    )
    assert processor.keep_frame_probs is True

    payload = {
        "frame_probs": [0.1, 0.9],
        "individual_results": [{"vad_type": "silero", "frame_probs": [0.1, 0.9]}],
    }

    class _FakeResult:
        def get_result_dict(self):
            return payload

    out = processor._vad_result_extra(_FakeResult())
    assert out["frame_probs"] == [0.1, 0.9]
    assert out["individual_results"][0]["frame_probs"] == [0.1, 0.9]


def test_vad_result_extra_passthrough_non_dict():
    """get_result_dict 返回非 dict（含缺失方法）时原样透传，不得抛错。"""
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    processor = VADProcessor({"enable": False, "vad_types": "silero"})

    class _NonDictResult:
        def get_result_dict(self):
            return ["not", "a", "dict"]

    assert processor._vad_result_extra(_NonDictResult()) == ["not", "a", "dict"]
    # 无 get_result_dict 方法的对象回退为 {}
    assert processor._vad_result_extra(object()) == {}


# ---------------------------------------------------------------------------
# B4 内容型延期：defer_speech_to_stage_c（purity-first 前置条件——
# 任何真实 VAD 语音票都应进入 Stage C 由 C1/C2/C3 裁决，而不是在 B4 硬拒）
# ---------------------------------------------------------------------------
import pytest as _pytest

from noise_filter_pipeline.processors.policy_engine import PolicyEngine


def _defer_engine(**overrides):
    cfg = {
        "vad_speech_ratio_threshold": 0.005,
        "speech_threshold": 0.6,
        "keep_noise": True,
        "keep_suspected": True,
        "keep_rejected": False,
        "defer_speech_to_stage_c": True,
    }
    cfg.update(overrides)
    return PolicyEngine(cfg)


@_pytest.mark.parametrize("fail_count", [1, 3])
def test_b4_defer_normalizes_any_real_voice_vote_to_speech_suspected(fail_count):
    """单票/多票真实语音票均归一为 speech_suspected 并放行进 Stage C。"""
    chunk = {
        "vad_result": {
            "individual_results": [
                {"speech_ratio": 0.9, "is_real_model": True}
            ] * fail_count,
            "strict_vad_models": True,
            "reject_on_any_speech": True,
        },
    }
    pr = _defer_engine().apply(chunk)
    assert pr.final_category == "speech_suspected"
    assert pr.kept is True
    assert pr.fail_count == fail_count


def test_b4_defer_keeps_strict_technical_failures_rejected():
    """strict fallback / failed detector / 无有效结果不参与延期，仍 fail-closed。"""
    fallback = {
        "vad_result": {
            "individual_results": [{"speech_ratio": 0.9, "is_real_model": False}],
            "strict_vad_models": True,
        },
    }
    pr = _defer_engine().apply(fallback)
    assert (pr.final_category, pr.kept) == ("rejected", False)

    failed = {
        "vad_result": {
            "individual_results": [{"speech_ratio": 0.9, "is_real_model": True}],
            "strict_vad_models": True,
            "failed_vad_types": ["funasr"],
        },
    }
    pr = _defer_engine().apply(failed)
    assert (pr.final_category, pr.kept) == ("rejected", False)

    pr = _defer_engine().apply({})
    assert (pr.final_category, pr.kept) == ("rejected", False)


def test_b4_defer_keeps_clean_noise_and_defaults_off():
    """无语音票仍为 noise；默认配置（未显式开启）保持旧行为。"""
    clean = {
        "vad_result": {
            "individual_results": [{"speech_ratio": 0.0, "is_real_model": True}],
            "strict_vad_models": True,
        },
    }
    pr = _defer_engine().apply(clean)
    assert (pr.final_category, pr.kept) == ("noise", True)

    legacy_hit = {
        "vad_result": {
            "individual_results": [{"speech_ratio": 0.9, "is_real_model": True}],
            "reject_on_any_speech": True,
        },
    }
    off = PolicyEngine({
        "vad_speech_ratio_threshold": 0.005,
        "keep_noise": True,
        "keep_suspected": True,
        "keep_rejected": False,
        "defer_speech_to_stage_c": False,
    }).apply(legacy_hit)
    assert (off.final_category, off.kept) == ("rejected", False)


def test_b4_defer_rejects_legacy_suspected_without_vad_evidence():
    """旧 manifest 仅有 suspected 类别时不得绕过缺失的 VAD 证据。"""
    pr = _defer_engine().apply({"final_category": "suspected"})
    assert pr.final_category == "rejected"
    assert pr.kept is False


@pytest.mark.unit
def test_funasr_build_result_frame_probs_matches_interval_scan():
    """FunASR 帧概率优化必须保持旧实现的闭区间语义和逐元素结果。"""
    from noise_filter_pipeline.processors.vad import FunASRVADWrapper

    wrapper = FunASRVADWrapper.__new__(FunASRVADWrapper)
    audio = np.zeros(2000, dtype=np.float32)  # 0.125 秒，6 帧
    generate_result = {
        "value": [
            [70, 90],   # 乱序且不命中前几帧
            [0, 10],    # t_center=0.01 命中 end 边界
            [20, 60],
            [50, 70],   # 与上一段重叠
        ]
    }

    result = wrapper._build_result(audio, 16000, generate_result)
    timestamps = [(0.07, 0.09), (0.0, 0.01), (0.02, 0.06), (0.05, 0.07)]
    expected_probs = [
        1.0 if any(start <= (i + 0.5) * 0.02 <= end for start, end in timestamps)
        else 0.0
        for i in range(max(1, int((len(audio) / 16000) / 0.02)))
    ]

    assert result.speech_timestamps == timestamps
    assert result.frame_probs == expected_probs
    assert result.speech_ratio == pytest.approx(sum(end - start for start, end in timestamps) / 0.125)
    assert result.confidence == 1.0


@pytest.mark.unit
def test_multivad_group_reuses_one_shared_wav_per_audio(tmp_path):
    """分组路径中同一音频的文件型 detector 必须共享临时 WAV。"""
    seen = []

    class _FileVAD:
        needs_wav_file = True

        def __init__(self, name):
            self.name = name

        def detect_array(self, audio, sr, shared_wav_path=None):
            seen.append((self.name, shared_wav_path, os.path.exists(shared_wav_path)))
            return VADResult(False, [], 0.0, 0.0, self.name)

    det = MultiVADDetector.__new__(MultiVADDetector)
    det.temp_dir = str(tmp_path)
    det.detectors = {"funasr": _FileVAD("funasr"), "firered": _FileVAD("firered")}
    det.speech_ratio_threshold = 0.05
    det.strict_vad_models = False
    det.reject_on_any_speech = False

    arrays = [
        (np.zeros(1600, dtype=np.float32), 16000),
        (np.zeros(3200, dtype=np.float32), 16000),
    ]
    results = det.detect_arrays_group(arrays)

    assert len(results) == 2
    assert len(seen) == 4
    first_detector_paths = [path for _, path, exists in seen[:2]]
    second_detector_paths = [path for _, path, exists in seen[2:]]
    assert first_detector_paths[0] != first_detector_paths[1]
    assert second_detector_paths[0] != second_detector_paths[1]
    assert first_detector_paths == second_detector_paths
    assert all(exists for _, _, exists in seen)
    assert not os.listdir(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["detect", "detect_array", "detect_arrays_group"])
def test_multivad_shared_wav_write_failure_aborts_infrastructure(
    tmp_path, monkeypatch, method
):
    """临时 WAV 写失败必须中止，不得回退原路径或继续调用 detector。"""
    from noise_filter_pipeline.core.exceptions import InfraAbortError
    from noise_filter_pipeline.processors.vad import multi as vad_multi

    calls = []

    class _FileVAD:
        needs_wav_file = True

        def detect(self, path):
            calls.append(path)

        def detect_array(self, audio, sr, shared_wav_path=None):
            calls.append(shared_wav_path)

    det = MultiVADDetector.__new__(MultiVADDetector)
    det.temp_dir = str(tmp_path)
    det.detectors = {"file": _FileVAD()}
    det.speech_ratio_threshold = 0.05
    det.strict_vad_models = False
    det.reject_on_any_speech = False
    if method == "detect":
        audio_path = tmp_path / "input.wav"
        _make_wav(str(audio_path), duration=0.1)
        invoke = lambda: det.detect(str(audio_path))
    elif method == "detect_array":
        invoke = lambda: det.detect_array(np.zeros(1600, dtype=np.float32), 16000)
    else:
        invoke = lambda: det.detect_arrays_group([
            (np.zeros(1600, dtype=np.float32), 16000),
            (np.zeros(3200, dtype=np.float32), 16000),
        ])
    monkeypatch.setattr(
        vad_multi.sf,
        "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(InfraAbortError, match="临时 WAV 写入失败"):
        invoke()
    assert calls == []
    assert sorted(os.listdir(tmp_path)) == (["input.wav"] if method == "detect" else [])


@pytest.mark.parametrize("value", [0, -1, 1.5, "bad", True, None])
def test_vad_rejects_invalid_silero_group_size(value):
    """silero_group_size 必须是正整数，禁止静默钳制或截断。"""
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    with pytest.raises(ValueError, match="silero_group_size"):
        VADProcessor({
            "enable": False,
            "vad_types": ["silero"],
            "silero_group_size": value,
        })


def test_vad_propagates_silero_group_size_to_spawn_ctor(monkeypatch):
    """VADProcessor 的 group 配置必须进入 spawn worker 构造参数。"""
    from noise_filter_pipeline.processors import vad_processor as vad_processor_module
    from noise_filter_pipeline.processors.vad_processor import VADProcessor

    seen = {}

    class _Detector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(vad_processor_module, "MultiVADDetector", _Detector)
    processor = VADProcessor({
        "enable": True,
        "vad_types": ["silero"],
        "silero_group_size": 3,
        "num_workers": 2,
    })

    assert processor.silero_group_size == 3
    assert seen["silero_group_size"] == 3


def _make_group_test_detector(group_size, num_workers):
    detector = MultiVADDetector.__new__(MultiVADDetector)
    detector.silero_group_size = group_size
    detector.num_workers = num_workers
    detector.keep_frame_probs = False
    return detector


def test_multivad_single_process_uses_configured_groups(monkeypatch):
    """单进程路径也必须按 silero_group_size 分组且保持输入顺序。"""
    detector = _make_group_test_detector(group_size=2, num_workers=1)
    seen = []

    def fake_group(group):
        seen.append([int(audio[0]) for audio, _ in group])
        return [int(audio[0]) for audio, _ in group]

    monkeypatch.setattr(detector, "detect_arrays_group", fake_group)
    arrays = [(np.array([i], dtype=np.float64), 16000) for i in range(5)]

    assert detector.detect_batch_arrays(arrays) == [0, 1, 2, 3, 4]
    assert seen == [[0, 1], [2, 3], [4]]


def test_multivad_pool_uses_configured_groups_and_prepares_ipc(monkeypatch):
    """进入 pool 前应按配置分组，并统一为 mono 16k C-contiguous float32。"""
    detector = _make_group_test_detector(group_size=2, num_workers=2)
    detector._pool = None
    detector._consecutive_pool_failures = 0
    detector._max_consecutive_pool_failures = 3
    seen = []

    class _Pool:
        def map(self, fn, groups, chunksize):
            groups = list(groups)
            for group in groups:
                seen.append(group)
            return [[int(audio[0]) for audio, _ in group] for group in groups]

    monkeypatch.setattr(detector, "_create_pool", lambda: _Pool())
    base = np.arange(24, dtype=np.float64).reshape(4, 6)
    arrays = [
        (base[:, ::2], 8000),
        (np.array([2.0], dtype=np.float64), 16000),
        (np.array([3.0], dtype=np.float64), 16000),
    ]

    out = detector.detect_batch_arrays(arrays)

    assert len(out) == 3
    assert [len(group) for group in seen] == [2, 1]
    for group in seen:
        for audio, sr in group:
            assert sr == 16000
            assert audio.ndim == 1
            assert audio.dtype == np.float32
            assert audio.flags.c_contiguous


def test_vad_worker_strips_frame_probs_before_ipc_when_disabled(monkeypatch):
    """worker 应在 pickle 返回前剥离帧概率，但保留投票策略所需字段。"""
    from noise_filter_pipeline.processors.vad import multi as vad_multi

    result = VADResult(
        True, [(0.0, 0.5)], 0.5, 0.9, "silero", frame_probs=[0.1, 0.9]
    )
    grouped = vad_multi.MultiVADResult(
        [result], vad_types=["silero"], speech_ratio_threshold=0.05
    )

    class _WorkerDetector:
        keep_frame_probs = False

        def detect_arrays_group(self, payload):
            return [grouped]

    monkeypatch.setattr(vad_multi, "_VAD_WORKER_DETECTOR", _WorkerDetector())
    returned = vad_multi._vad_worker_detect_arrays([
        (np.zeros(1600, dtype=np.float32), 16000)
    ])

    assert returned[0].results[0].frame_probs is None
    assert returned[0].results[0].speech_ratio == 0.5
    assert returned[0].fail_count == 1
    assert returned[0].final_category == "suspected"


def test_pcm16_array_matches_soundfile_wav_roundtrip():
    """direct-array 量化必须与旧 PCM16 WAV 写读口径逐样本一致。"""
    import io

    from noise_filter_pipeline.processors.vad import wrappers

    audio = np.array(
        [-1.2, -1.0, -0.99999, -0.5, -1 / 32768, 0.0, 1 / 32768,
         0.5, 0.99999, 1.0, 1.2],
        dtype=np.float32,
    )
    wav = io.BytesIO()
    sf.write(wav, audio, 16000, format="WAV", subtype="PCM_16")
    wav.seek(0)
    expected_int16, _ = sf.read(wav, dtype="int16")
    wav.seek(0)
    expected_float, _ = sf.read(wav, dtype="float32")

    pcm16, quantized = wrappers._pcm16_roundtrip_arrays(audio)

    np.testing.assert_array_equal(pcm16, expected_int16)
    np.testing.assert_array_equal(quantized, expected_float)


def test_firered_direct_array_uses_pcm16_tuple_without_wav(monkeypatch):
    """FireRed 新版 API 应直接接收 (PCM16, 16000)，不得创建 WAV。"""
    from noise_filter_pipeline.processors.vad import wrappers

    seen = []

    class _Model:
        def detect(self, value):
            seen.append(value)
            return {"dur": 0.25, "timestamps": []}, np.array([0.1])

    wrapper = wrappers.FireRedVADWrapper.__new__(wrappers.FireRedVADWrapper)
    wrapper.vad = _Model()
    wrapper.temp_dir = None
    wrapper._array_input_supported = None
    monkeypatch.setattr(
        wrappers.sf, "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected WAV")),
    )

    result = wrapper.detect_array(np.array([0.5, -0.5], dtype=np.float32), 16000)

    assert result.vad_type == "firered_vad"
    pcm16, sr = seen[0]
    assert sr == 16000
    np.testing.assert_array_equal(pcm16, np.array([16384, -16384], dtype=np.int16))


def test_funasr_direct_array_uses_quantized_memory_without_wav(monkeypatch):
    """FunASR 应逐条接收 PCM16 roundtrip 等价浮点数组，不伪造列表 batch。"""
    from noise_filter_pipeline.processors.vad import wrappers

    seen = []

    class _Model:
        def generate(self, input, **kwargs):
            seen.append((input, kwargs))
            return [{"value": []}]

    wrapper = wrappers.FunASRVADWrapper.__new__(wrappers.FunASRVADWrapper)
    wrapper.model = _Model()
    wrapper.temp_dir = None
    wrapper._array_input_supported = None
    monkeypatch.setattr(
        wrappers.sf, "write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected WAV")),
    )
    audio = np.array([0.99999, -0.99999], dtype=np.float32)

    result = wrapper.detect_array(audio, 16000)

    assert result.vad_type == "funasr"
    assert isinstance(seen[0][0], np.ndarray)
    assert seen[0][0].ndim == 1
    np.testing.assert_array_equal(
        seen[0][0], wrappers._pcm16_roundtrip_arrays(audio)[1]
    )
    assert seen[0][1]["batch_size_s"] == 300


@pytest.mark.parametrize("wrapper_name", ["FireRedVADWrapper", "FunASRVADWrapper"])
def test_direct_array_capability_falls_back_to_pcm16_wav(
    tmp_path, wrapper_name
):
    """仅旧 API 的 capability TypeError 才落临时 PCM16 WAV。"""
    from noise_filter_pipeline.processors.vad import wrappers

    seen_paths = []
    if wrapper_name == "FireRedVADWrapper":
        class _Model:
            def detect(self, value):
                if not isinstance(value, str):
                    raise TypeError("array input unsupported")
                seen_paths.append(value)
                data, sr = sf.read(value, dtype="int16")
                assert sr == 16000
                np.testing.assert_array_equal(
                    data, np.array([16384, -16384], dtype=np.int16)
                )
                return {"dur": 2 / 16000, "timestamps": []}, []

        wrapper = wrappers.FireRedVADWrapper.__new__(wrappers.FireRedVADWrapper)
        wrapper.vad = _Model()
    else:
        class _Model:
            def generate(self, input, **kwargs):
                if not isinstance(input, str):
                    raise TypeError("array input unsupported")
                seen_paths.append(input)
                data, sr = sf.read(input, dtype="int16")
                assert sr == 16000
                np.testing.assert_array_equal(
                    data, np.array([16384, -16384], dtype=np.int16)
                )
                return [{"value": []}]

        wrapper = wrappers.FunASRVADWrapper.__new__(wrappers.FunASRVADWrapper)
        wrapper.model = _Model()
    wrapper.temp_dir = str(tmp_path)
    wrapper._array_input_supported = None

    result = wrapper.detect_array(
        np.array([0.5, -0.5], dtype=np.float32), 16000
    )

    assert result.vad_type != "energy"
    assert wrapper._array_input_supported is False
    assert len(seen_paths) == 1
    assert not os.listdir(tmp_path)


def test_multivad_direct_array_failure_isolated_per_audio():
    """一个 direct-array detector 的单条失败不得影响组内其他音频。"""
    class _FlakyVAD:
        needs_wav_file = False

        def detect_array(self, audio, sr):
            if audio[0] < 0:
                raise RuntimeError("one bad item")
            return VADResult(False, [], 0.0, 0.0, "flaky")

    detector = MultiVADDetector.__new__(MultiVADDetector)
    detector.temp_dir = None
    detector.detectors = {"flaky": _FlakyVAD()}
    detector.speech_ratio_threshold = 0.05
    detector.strict_vad_models = False
    detector.reject_on_any_speech = False
    detector.silero_group_size = 16

    results = detector.detect_arrays_group([
        (np.array([1.0], dtype=np.float32), 16000),
        (np.array([-1.0], dtype=np.float32), 16000),
        (np.array([2.0], dtype=np.float32), 16000),
    ])

    assert results[0].vad_types == ["flaky"]
    assert results[1].vad_types == []
    assert results[1].failed_vad_types == ["flaky"]
    assert results[2].vad_types == ["flaky"]
