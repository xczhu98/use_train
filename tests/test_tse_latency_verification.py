"""evaluate.latency_verification 单元测试（官方扰动法移植版）。

用合成模型验证 future-dependency 估计语义，无需真实模型/音频：
- 恒等模型（严格因果、零延迟）→ future dependency ≈ 0
- 移位模型（y[t] = x[t+k]，等效 k 样本 look-ahead）→ future dependency ≈ k/fs
"""

from __future__ import annotations
import json
import sys

import pytest
import torch

import evaluate.latency_verification as latency_verification
from evaluate.latency_verification import (
    build_aec_model_fn,
    build_se_model_fn,
    build_tse_model_fn,
    estimate_latency,
    summarize_future_dependency,
)

FS = 16000


def _identity_model(x, enroll):
    return (x,)


def _shift_model(k: int):
    """y[t] = x[t+k]（输出提前 k 样本看到未来输入；尾部补零）。"""

    def model_fn(x, enroll):
        y = torch.cat([x[:, k:], torch.zeros_like(x[:, :k])], dim=1)
        return (y,)

    return model_fn


class TestFutureDependencySummary:
    def test_summary_converts_to_ms_and_preserves_negative_values(self):
        summary = summarize_future_dependency(
            [-0.01, 0.0, 0.02],
            fs=16000,
            threshold=1e-5,
            perturb_starts=(1.0, 2.0, 3.0),
        )

        assert set(summary) == {
            "metric_kind",
            "unit",
            "min_future_dependency_ms",
            "mean_future_dependency_ms",
            "max_future_dependency_ms",
            "values_ms",
            "sample_rate",
            "threshold",
            "perturb_starts_s",
            "n_perturbations",
        }
        assert summary["metric_kind"] == "future_dependency"
        assert summary["unit"] == "ms"
        assert summary["min_future_dependency_ms"] == -10.0
        assert summary["mean_future_dependency_ms"] == pytest.approx(3.3333, abs=1e-4)
        assert summary["max_future_dependency_ms"] == 20.0
        assert summary["values_ms"] == [-10.0, 0.0, 20.0]
        assert summary["sample_rate"] == 16000
        assert summary["threshold"] == 1e-5
        assert summary["perturb_starts_s"] == [1.0, 2.0, 3.0]
        assert summary["n_perturbations"] == 3


class TestEstimateLatency:
    def test_identity_model_zero_future_dependency(self):
        results = estimate_latency(
            _identity_model, fs=FS, signal_len=4, enroll_len=2, seed=0
        )
        for fd in results:
            assert abs(fd) < 1e-6, f"恒等模型 future dependency 应为 0，得 {fd}"

    def test_shift_model_recovers_lookahead(self):
        k = 128  # 8ms @16k
        results = estimate_latency(
            _shift_model(k), fs=FS, signal_len=4, enroll_len=2, seed=0
        )
        for fd in results:
            assert abs(fd - k / FS) < 1e-6, (
                f"移位模型 future dependency 应≈{k / FS}s，得 {fd}"
            )

    def test_seed_makes_results_reproducible(self):
        r1 = estimate_latency(_identity_model, fs=FS, signal_len=4, enroll_len=2, seed=42)
        r2 = estimate_latency(_identity_model, fs=FS, signal_len=4, enroll_len=2, seed=42)
        assert r1 == r2

    def test_seed_does_not_leak_global_rng_state(self):
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()

        estimate_latency(_identity_model, fs=FS, signal_len=4, enroll_len=2, seed=7)

        assert torch.equal(torch.random.get_rng_state(), before)


class TestJsonOutput:
    def test_main_writes_structured_json_output(self, monkeypatch, tmp_path):
        output_path = tmp_path / "nested" / "future_dependency.json"

        monkeypatch.setattr(
            latency_verification,
            "estimate_latency",
            lambda *_args, **_kwargs: [-0.01, 0.0, 0.02],
        )
        monkeypatch.setattr(
            latency_verification,
            "load_se_aec_model",
            lambda *_args, **_kwargs: (object(), None, FS),
        )
        monkeypatch.setattr(
            latency_verification,
            "build_se_model_fn",
            lambda *_args, **_kwargs: _identity_model,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "latency_verification.py",
                "--task", "se",
                "--model", "fake_model.tar",
                "--config", "fake_config.yaml",
                "--device", "cpu",
                "--json-output", str(output_path),
            ],
        )

        assert latency_verification.main() == 0

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["metric_kind"] == "future_dependency"
        assert payload["values_ms"] == [-10.0, 0.0, 20.0]
        assert payload["task"] == "se"
        assert payload["model"] == "fake_model.tar"
        assert payload["config"] == "fake_config.yaml"
        assert payload["device"] == "cpu"


class _FakeExtractor:
    def __init__(self, output_peak):
        self.output_peak = output_peak
        self.sample_rate = FS

    def extract_speech_from_pcm(self, pcm_mix, sr_mix, pcm_enroll, sr_enroll):
        import numpy as np

        return np.zeros(pcm_mix.shape[-1], dtype=np.float32)


class TestBuildTSEModelFn:
    def test_rejects_peak_normalization(self):
        import pytest

        with pytest.raises(ValueError, match="output_peak"):
            build_tse_model_fn(_FakeExtractor(output_peak=0.9))

    def test_official_signature_roundtrip(self):
        model_fn = build_tse_model_fn(_FakeExtractor(output_peak=None))
        x = torch.randn(1, FS)
        enroll = torch.randn(1, FS // 2)
        y = model_fn(x, enroll)[0]
        assert isinstance(y, torch.Tensor) and y.shape == x.shape


# ---------------------------------------------------------------------------
# SE / AEC 适配器
# ---------------------------------------------------------------------------
class _IdentitySEWaveformModel(torch.nn.Module):
    """波形端到端恒等 SE 模型（无 stft 块路径）。"""

    def forward(self, wav, lengths=None):
        return wav


class _ShiftSEWaveformModel(torch.nn.Module):
    """y[t] = x[t+k]，等效 k 样本 look-ahead；声明 fs 参数以覆盖 fs 注入分支。"""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, wav, lengths=None, fs=None):
        return torch.cat([wav[:, self.k:], torch.zeros_like(wav[:, : self.k])], dim=1)


class _IdentitySESpecModel(torch.nn.Module):
    """谱进谱出恒等 SE 模型（stft 块路径）：返回输入谱。"""

    def forward(self, spec, lengths=None):
        return spec


class _IdentityAECModel(torch.nn.Module):
    """AEC dict 恒等模型：nearend_estimate_spectra = mic_spectra。"""

    def forward(self, batch: dict) -> dict:
        batch["nearend_estimate_spectra"] = batch["mic_spectra"]
        return batch


def _small_stft():
    from core.stft import build_stft_istft

    return build_stft_istft(
        {"n_fft": 64, "hop_length": 16, "win_length": 64, "window": "hann"}
    )


class TestSEAdapter:
    def test_waveform_identity_zero_fd(self):
        model_fn = build_se_model_fn(
            _IdentitySEWaveformModel(), None, torch.device("cpu"), FS
        )
        results = estimate_latency(model_fn, fs=FS, signal_len=4, enroll_len=2, seed=0)
        for fd in results:
            assert abs(fd) < 1e-6

    def test_waveform_shift_recovers_lookahead_with_fs_kwarg(self):
        k = 160  # 10ms @16k
        model_fn = build_se_model_fn(
            _ShiftSEWaveformModel(k), None, torch.device("cpu"), FS
        )
        results = estimate_latency(model_fn, fs=FS, signal_len=4, enroll_len=2, seed=0)
        for fd in results:
            assert abs(fd - k / FS) < 1e-6

    def test_spectrum_identity_zero_fd(self):
        model_fn = build_se_model_fn(
            _IdentitySESpecModel(), _small_stft(), torch.device("cpu"), FS
        )
        results = estimate_latency(model_fn, fs=FS, signal_len=4, enroll_len=2, seed=0)
        # STFT/iSTFT 重建误差远小于阈值 1e-5，扰动区域差异 >> 阈值
        for fd in results:
            assert abs(fd) < 1e-3


class TestAECAdapter:
    def test_identity_zero_fd_and_aux_ignored(self):
        model_fn = build_aec_model_fn(
            _IdentityAECModel(), _small_stft(), torch.device("cpu")
        )
        results = estimate_latency(model_fn, fs=FS, signal_len=4, enroll_len=2, seed=0)
        for fd in results:
            assert abs(fd) < 1e-3

    def test_requires_stft(self):
        import pytest

        with pytest.raises(ValueError, match="stft"):
            build_aec_model_fn(_IdentityAECModel(), None, torch.device("cpu"))

pytestmark = [pytest.mark.unit]
