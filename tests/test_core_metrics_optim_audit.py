"""core metrics 与 scheduler 审计回归测试。"""

from __future__ import annotations

import contextlib
import logging
import sys
import types

import numpy as np
import pytest
import torch

import core.metrics.batch as batch_mod
from core.metrics.aec import compute_erle_energy_sums
from core.metrics.intrusive import compute_pesq, compute_sisdr, compute_stoi
from core.metrics.mos_models import _SCOREQScorer, _UTMOSScorer, _scoreq_mono_audio
from core.optim.scheduler import (
    CosineAnnealingWarmupRestarts,
    LinearWarmupCosineAnnealing,
    resolve_scheduler_auto_args,
)


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-3)


@pytest.mark.unit
def test_erle_fst_gate_extends_last_frame_to_non_hop_tail() -> None:
    """最后完整帧的 VAD 标签必须覆盖 frame/hop 余量及音频末尾。"""
    far = np.zeros(11, dtype=np.float32)
    far[6] = 1.0
    near = np.zeros_like(far)
    mic = np.ones_like(far)
    enh = np.zeros_like(far)

    stats = compute_erle_energy_sums(
        mic,
        near,
        enh,
        fs=1000,
        far=far,
        gate_frame_ms=4.0,
        gate_hop_ms=2.0,
        far_delay_samples=0,
    )

    assert stats == pytest.approx((7.0, 0.0, 7))


@pytest.mark.unit
@pytest.mark.parametrize(
    "ref, enh",
    [
        (np.array([]), np.array([])),
        (np.ones(3), np.ones(2)),
        (np.ones((1, 3)), np.ones((1, 3))),
        (np.array([1.0, np.nan]), np.ones(2)),
    ],
)
def test_sisdr_rejects_invalid_waveform_contract(ref: np.ndarray, enh: np.ndarray) -> None:
    assert compute_sisdr(ref, enh) is None


@pytest.mark.unit
def test_pesq_and_stoi_resample_failures_are_optional_failures(monkeypatch) -> None:
    pesq_error = types.SimpleNamespace(RETURN_VALUES=1)
    monkeypatch.setitem(
        sys.modules,
        "pesq",
        types.SimpleNamespace(PesqError=pesq_error, pesq=lambda *_args, **_kwargs: 2.0),
    )
    monkeypatch.setitem(
        sys.modules,
        "pystoi",
        types.SimpleNamespace(stoi=lambda *_args, **_kwargs: 0.9),
    )
    monkeypatch.setattr(
        "core.metrics.intrusive._resample_np",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("resample failed")),
    )

    audio = np.ones(16000, dtype=np.float32)
    assert compute_pesq(audio, audio, 12000) is None
    assert compute_stoi(audio, audio, 24000) is None


@pytest.mark.unit
def test_nonfinite_mos_result_is_counted_as_failure_not_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        batch_mod,
        "compute_mos_batch",
        lambda *_args, **_kwargs: [{"MOS": float("nan")}],
    )

    results = batch_mod.compute_metrics_batch(
        [(np.ones(4), np.ones(4), 16000)],
        metrics=[],
        mos_models={"SCOREQ": object()},
    )

    assert results == {}
    with pytest.raises(RuntimeError, match="SCOREQ_MOS\\(fail=1\\)"):
        batch_mod.compute_metrics_batch(
            [(np.ones(4), np.ones(4), 16000)],
            metrics=[],
            mos_models={"SCOREQ": object()},
            strict=True,
        )


@pytest.mark.unit
def test_truncated_mos_batch_output_counts_missing_samples_as_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        batch_mod,
        "compute_mos_batch",
        lambda *_args, **_kwargs: [{"MOS": 3.0}],
    )
    pairs = [(np.ones(4), np.ones(4), 16000) for _ in range(2)]

    results = batch_mod.compute_metrics_batch(
        pairs, metrics=[], mos_models={"SCOREQ": object()}
    )

    assert results == {"SCOREQ_MOS": 3.0, "_count_SCOREQ_MOS": 1.0}


@pytest.mark.unit
def test_utmos_batches_exact_resampled_lengths_and_restores_order() -> None:
    scorer = _UTMOSScorer.__new__(_UTMOSScorer)
    scorer._device = "cpu"
    scorer._MAX_PADDED_SAMPLES_PER_RUN = 640
    seen_shapes = []

    class _Model:
        def __call__(self, waveforms, _sr):
            seen_shapes.append(tuple(waveforms.shape))
            return waveforms.sum(dim=1)

    scorer._model = _Model()
    short = np.ones(160, dtype=np.float32)
    long = np.ones(480, dtype=np.float32)
    scores = scorer.batch([(long, 16_000), (short, 16_000), (long, 16_000)], 16)

    assert seen_shapes == [(1, 480), (1, 480), (1, 160)]
    assert [score["MOS"] for score in scores] == pytest.approx([480.0, 160.0, 480.0])


@pytest.mark.unit
def test_scoreq_uses_time_first_downmix_and_rejects_output_mismatch() -> None:
    stereo = np.array(
        [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(
        _scoreq_mono_audio(stereo), np.array([2.0, 6.0, 10.0])
    )

    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._resamplers = {}

    class _Session:
        @staticmethod
        def get_inputs():
            return [types.SimpleNamespace(name="audio")]

        @staticmethod
        def run(*_args, **_kwargs):
            return [np.array([[3.0]], dtype=np.float32)]

    scorer._sess = _Session()
    results = scorer.run_batch([(stereo, 16000), (stereo, 16000)])
    assert len(results) == 2
    assert all(np.isnan(result["MOS"]) for result in results)


@pytest.mark.unit
def test_scoreq_batches_only_identical_post_cap_padded_lengths() -> None:
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._resamplers = {}
    scorer._use_gpu = False
    scorer._broken_error = None
    seen_shapes = []

    class _Session:
        @staticmethod
        def get_inputs():
            return [types.SimpleNamespace(name="audio")]

        @staticmethod
        def run(_outputs, feeds):
            batch = feeds["audio"]
            seen_shapes.append(batch.shape)
            return [batch.sum(axis=1, keepdims=True)]

    scorer._sess = _Session()
    scorer._MAX_SAMPLES_PER_RUN = 1_280
    short = np.ones(160, dtype=np.float32)
    medium = np.ones(321, dtype=np.float32)
    long = np.ones(1_024, dtype=np.float32)

    scores = scorer.run_batch([
        (medium, 16_000), (short, 16_000), (long, 16_000), (short, 16_000),
    ])

    # _MIN_PADDED_SAMPLES=640（SSL conv 前端要求）后 short/medium 落在同一 640 桶：
    # 两条 640 可同批，1_280 的 long 必须单独成批。
    assert seen_shapes == [(2, 640), (1, 640), (1, 1_280)]
    assert [score["MOS"] for score in scores] == pytest.approx(
        [321.0, 160.0, 1_024.0, 160.0]
    )


@pytest.mark.unit
def test_scoreq_single_and_batch_share_torchaudio_resampling(monkeypatch) -> None:
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._resamplers = {}
    scorer._use_gpu = False
    scorer._broken_error = None
    constructor_calls = []

    class _Resample:
        def __call__(self, waveform):
            return torch.repeat_interleave(waveform, 2, dim=1)

    def resample_factory(source_sr, target_sr):
        constructor_calls.append((source_sr, target_sr))
        return _Resample()

    class _Session:
        @staticmethod
        def get_inputs():
            return [types.SimpleNamespace(name="audio")]

        @staticmethod
        def run(_outputs, feeds):
            return [feeds["audio"].sum(axis=1, keepdims=True)]

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(transforms=types.SimpleNamespace(Resample=resample_factory)),
    )
    scorer._sess = _Session()
    source = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    single = scorer(source, 8_000)
    batched = scorer.run_batch([(source, 8_000)])[0]

    assert single == batched == {"MOS": 12.0}
    assert constructor_calls == [(8_000, 16_000)]


@pytest.mark.unit
def test_scoreq_gpu_resets_before_eleventh_real_onnx_run() -> None:
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._resamplers = {}
    scorer._use_gpu = True
    scorer._runs_since_reset = 0
    scorer._broken_error = None
    reset_calls = []

    class _Session:
        @staticmethod
        def get_inputs():
            return [types.SimpleNamespace(name="audio")]

        @staticmethod
        def run(_outputs, feeds):
            return [np.full((feeds["audio"].shape[0], 1), 4.0, dtype=np.float32)]

    scorer._sess = _Session()
    scorer._MAX_SAMPLES_PER_RUN = 320

    def reset():
        reset_calls.append(scorer._runs_since_reset)
        scorer._sess = _Session()
        scorer._runs_since_reset = 0

    scorer._reset_gpu_session = reset
    scores = scorer.run_batch([
        (np.ones(160, dtype=np.float32), 16_000) for _ in range(11)
    ])

    assert scores == [{"MOS": 4.0}] * 11
    assert reset_calls == [10]
    assert scorer._runs_since_reset == 1


@pytest.mark.unit
def test_scoreq_cpu_never_resets_and_empty_batch_never_runs() -> None:
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._resamplers = {}
    scorer._use_gpu = False
    scorer._broken_error = None
    calls = []

    class _Session:
        @staticmethod
        def get_inputs():
            return [types.SimpleNamespace(name="audio")]

        @staticmethod
        def run(_outputs, feeds):
            calls.append(feeds["audio"].shape)
            return [np.full((feeds["audio"].shape[0], 1), 4.0, dtype=np.float32)]

    scorer._sess = _Session()
    scorer._reset_gpu_session = lambda: pytest.fail("CPU must not reset")

    assert scorer.run_batch([]) == []
    for _ in range(11):
        scorer.run_batch([(np.ones(160, dtype=np.float32), 16_000)])
    # 160 样本按 _MIN_PADDED_SAMPLES=640（SSL conv 前端要求）补齐后逐条 run
    assert calls == [(1, 640)] * 11


@pytest.mark.unit
def test_scoreq_gpu_reset_failure_is_fail_closed_and_keeps_snapshot_lease() -> None:
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._resamplers = {}
    scorer._use_gpu = True
    scorer._runs_since_reset = 10
    scorer._broken_error = None
    lease_entries = []

    @contextlib.contextmanager
    def lease():
        lease_entries.append("entered")
        yield "/proc/self/fd/123"

    scorer._session_path_lease = lease
    scorer._onnx_filename = "adapt_nr_telephone.onnx"
    scorer._build_session = lambda _path: (_ for _ in ()).throw(
        RuntimeError("provider lost"),
    )

    with pytest.raises(RuntimeError, match="reset failed; scorer is broken"):
        scorer._run(np.ones((1, 320), dtype=np.float32))
    with pytest.raises(RuntimeError, match="broken"):
        scorer._run(np.ones((1, 320), dtype=np.float32))
    assert lease_entries == ["entered"]


@pytest.mark.unit
def test_scoreq_gpu_session_uses_bounded_cuda_primary_hybrid_ep(monkeypatch) -> None:
    calls = []

    class _Options:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class _Session:
        def __init__(self, path, options, providers):
            calls.append((path, options.entries, providers))

        @staticmethod
        def get_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=_Options, InferenceSession=_Session,
    ))
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._use_gpu = True
    session = scorer._build_session("model.onnx")

    assert session.get_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert scorer._device_str == "GPU"
    assert calls == [(
        "model.onnx",
        {
            "gpu.mem_limit": str(16 * 1024 * 1024 * 1024),
            "gpu.arena_extend_strategy": "kSameAsRequested",
        },
        [
            ("CUDAExecutionProvider", {
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 16 * 1024 * 1024 * 1024,
            }),
            "CPUExecutionProvider",
        ],
    )]


@pytest.mark.unit
def test_scoreq_gpu_session_requires_cuda_primary_provider(monkeypatch) -> None:
    class _Options:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class _Session:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=_Options, InferenceSession=_Session,
    ))
    scorer = _SCOREQScorer.__new__(_SCOREQScorer)
    scorer._use_gpu = True

    with pytest.raises(RuntimeError, match="CUDA-primary"):
        scorer._build_session("model.onnx")


@pytest.mark.unit
def test_scheduler_rejects_invalid_parameter_domains() -> None:
    optimizer = _optimizer()
    with pytest.raises(ValueError, match="warmup_steps"):
        LinearWarmupCosineAnnealing(optimizer, warmup_steps=-1)
    with pytest.raises(ValueError, match="must not exceed"):
        LinearWarmupCosineAnnealing(optimizer, warmup_steps=8, decay_until_step=4)
    with pytest.raises(ValueError, match="min_lr"):
        LinearWarmupCosineAnnealing(optimizer, max_lr=1e-5, min_lr=1e-3)
    with pytest.raises(ValueError, match="first_cycle_steps"):
        CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=0)
    with pytest.raises(ValueError, match="cycle_mult"):
        CosineAnnealingWarmupRestarts(optimizer, cycle_mult=0.5)
    with pytest.raises(ValueError, match="warmup_steps"):
        CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=4, warmup_steps=4)


@pytest.mark.unit
def test_aecmos_model_name_cannot_escape_model_directory(tmp_path, monkeypatch) -> None:
    """model_name 只接受纯文件名，路径穿越不得越出 AECMOS 目录。"""
    import core.metrics.aecmos as aecmos

    (tmp_path / "good.onnx").write_bytes(b"placeholder")
    monkeypatch.setattr(aecmos, "_AECMOS_DIR", tmp_path)
    captured: dict[str, object] = {}

    class _Estimator:
        def __init__(self, path):
            captured["path"] = str(path)

    monkeypatch.setattr(aecmos, "AECMOSEstimator", _Estimator)

    aecmos.load_aecmos("good")
    assert captured["path"] == str(tmp_path / "good.onnx")

    # '' / '.' / '..' 解析后没有文件名，必须显式拒绝而不是拼出隐藏路径。
    with pytest.raises(ValueError, match="invalid AECMOS model name"):
        aecmos.load_aecmos("..")

    with pytest.raises(FileNotFoundError) as excinfo:
        aecmos.load_aecmos("../../../../etc/passwd")
    assert str(tmp_path / "passwd.onnx") in str(excinfo.value)


@pytest.mark.unit
def test_vendored_package_import_does_not_inject_sys_path(tmp_path) -> None:
    """vendored 包按显式路径加载，目录不得进入 sys.path（避免任意模块注入面）。"""
    import sys

    from core.metrics.mos_models import _import_vendored_package

    root = tmp_path / "SpeechMOS" / "vendored_probe"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("MARKER = 'vendored'\n", encoding="utf-8")
    namespace_root = tmp_path / "ns" / "probe_ns"
    namespace_root.mkdir(parents=True)
    (namespace_root / "child.py").write_text("VALUE = 3\n", encoding="utf-8")
    sys_path_before = list(sys.path)
    try:
        module = _import_vendored_package("vendored_probe", root)
        assert module.MARKER == "vendored"
        assert str(root.parent) not in sys.path

        # PEP 420 命名空间包分支（vendored NISQA 无 __init__.py）。
        _import_vendored_package("probe_ns", namespace_root)
        from probe_ns import child

        assert child.VALUE == 3
        assert str(namespace_root.parent) not in sys.path
        assert sys.path == sys_path_before

        with pytest.raises(ImportError, match="vendored package directory not found"):
            _import_vendored_package("vendored_missing", tmp_path / "nope")
    finally:
        for name in ("vendored_probe", "probe_ns", "probe_ns.child"):
            sys.modules.pop(name, None)


@pytest.mark.unit
def test_scheduler_auto_warmup_horizon_warning_describes_actual_behavior(caplog) -> None:
    cfg = {
        "type": "LinearWarmupCosineAnnealing",
        "args": {"warmup_steps": 10, "decay_until_step": 10},
    }
    with caplog.at_level(logging.WARNING, logger="scheduler"):
        resolve_scheduler_auto_args(cfg, steps_per_epoch=2, epochs=5)

    assert "remains in linear warmup" in caplog.text
    with pytest.raises(ValueError, match="warmup_ratio"):
        resolve_scheduler_auto_args(
            {"type": "LinearWarmupCosineAnnealing", "auto": {"warmup_ratio": 1.1}},
            steps_per_epoch=1,
            epochs=1,
        )


pytestmark = [pytest.mark.unit]
