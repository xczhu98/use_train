"""MOS 独立模型批处理隔离回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from audio_filter_pipeline.processors import mos


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("use_gpu", (False, True))
def test_utmos_batch_uses_exact_preprocessed_lengths_and_restores_order(use_gpu):
    class PaddingSensitiveModel:
        def __call__(self, waveforms, _sr):
            lengths = (waveforms != 0).sum(dim=1).to(torch.float32)
            return lengths + waveforms.shape[1] / 10_000.0

    estimator = mos.UTMOSEstimator.__new__(mos.UTMOSEstimator)
    estimator.model = PaddingSensitiveModel()
    estimator._device = torch.device("cpu")
    estimator.use_gpu = use_gpu
    estimator._load_waveform = lambda *, audio_path=None, audio_data=None: (
        torch.from_numpy(np.asarray(audio_data[0], dtype=np.float32)).unsqueeze(0),
        int(audio_data[1]),
    )

    short = np.ones(320, dtype=np.float32)
    long = np.ones(1_600, dtype=np.float32)
    scores = estimator.compute_batch([(long, 16_000), (short, 16_000), (long, 16_000)])

    assert [score["utmos_score"] for score in scores] == pytest.approx(
        [1_600.16, 320.032, 1_600.16]
    )


def test_utmos_load_waveform_reads_file_with_soundfile(tmp_path):
    """文件路径读取不依赖 torchaudio/torchcodec，并统一取首通道与目标采样率。"""
    import soundfile as sf

    wav_path = tmp_path / "stereo_8k.wav"
    left = np.full(8000, 0.5, dtype=np.float32)
    sf.write(wav_path, np.stack([left, np.zeros_like(left)], axis=1), 8000)

    estimator = mos.UTMOSEstimator.__new__(mos.UTMOSEstimator)
    waveform, sr = estimator._load_waveform(audio_path=str(wav_path))

    assert sr == 16_000
    assert waveform.shape == (1, 16_000)
    assert torch.isfinite(waveform).all()
    # 取首通道（0.5）而非第二通道（0），重采样后均值仍接近 0.5
    assert waveform.mean().item() == pytest.approx(0.5, abs=0.02)


def test_scoreq_estimator_delegates_reset_to_real_onnx_runs_not_outer_batches():
    calls = []

    class Scoreq:
        def run_batch(self, items):
            calls.append(list(items))
            return [{"MOS": 4.0} for _item in items]

    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = Scoreq()
    estimator._allow_fallback = False
    estimator.use_gpu = True
    estimator.reset = lambda: pytest.fail("outer compute_batch must not reset SCOREQ")

    assert estimator.compute_batch([(np.zeros(160, dtype=np.float32), 16_000)]) == [
        {"scoreq_mos": 4.0}
    ]
    assert estimator.compute_batch([(np.zeros(320, dtype=np.float32), 16_000)]) == [
        {"scoreq_mos": 4.0}
    ]
    assert len(calls) == 2


def test_scoreq_estimator_reset_failure_marks_estimator_broken(monkeypatch):
    estimator = mos.SCOREQEstimator.__new__(mos.SCOREQEstimator)
    estimator._scoreq = object()
    estimator.use_gpu = True
    estimator._allow_fallback = False
    estimator._closed = False
    estimator._new_scoreq = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("reset failed"))

    with pytest.raises(RuntimeError, match="SCOREQ session reset failed"):
        estimator.reset()

    assert estimator._scoreq is None
    with pytest.raises(RuntimeError, match="broken"):
        estimator.compute_batch([(np.zeros(160, dtype=np.float32), 16_000)])
