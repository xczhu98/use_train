"""AEC canonical adapter 部署语义回归测试。

覆盖从旧批量路径迁移进 ``AECInferenceAdapter`` 的两项部署语义：
- 长音频按 mic 原始帧数/原始采样率（重采样前）切换到 CPU，处理完切回请求设备；
- 审听多通道写出 ``[mic..., reference?, enhanced]``，使用 PCM_F 32-bit。

torchaudio.load/save 全部 monkeypatch，模型使用恒等/缩放 dummy，测试不依赖真实
音频文件与音频后端；真实 GPU→CPU 切换由 ``gpu`` marker 的用例显式选择。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchaudio

import tasks.aec.inference.adapter as adapter_module
from core.inference import FileInferenceItem
from tasks.aec.inference.adapter import AECInferenceAdapter

N_FFT = 320
HOP = 160
STFT_CFG = {"n_fft": N_FFT, "hop_length": HOP, "win_length": N_FFT}
SR = 16000

pytestmark = [pytest.mark.unit]


class _IdentityAECModel(torch.nn.Module):
    """恒等 AEC 模型：near-end 估计直接回传 mic 谱。"""

    def forward(self, model_input):
        return {"nearend_estimate_spectra": model_input["mic_spectra"]}


class _ScaledAECModel(torch.nn.Module):
    """缩放 AEC 模型：让 enhanced 与审听 mic 通道可明确区分。"""

    def forward(self, model_input):
        return {"nearend_estimate_spectra": model_input["mic_spectra"] * 0.5}


class _DeviceRecordingModel(torch.nn.Module):
    """记录 forward 时模型实际所在设备的恒等 AEC 模型。"""

    def __init__(self):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.seen_devices: list[torch.device] = []

    def forward(self, model_input):
        self.seen_devices.append(self.probe.device)
        return {"nearend_estimate_spectra": model_input["mic_spectra"]}


def _fake_save(captured: dict):
    """伪造 torchaudio.save：不写盘，只记录待保存张量、采样率与编码参数。"""

    def fake_save(path, tensor, sr, **kwargs):
        captured["path"] = path
        captured["tensor"] = tensor.clone()
        captured["sr"] = sr
        captured["save_kwargs"] = kwargs

    return fake_save


def _fake_load(mic: torch.Tensor, far: torch.Tensor, *, mic_sr: int = SR, far_sr: int = SR):
    """伪造 torchaudio.load：按路径名返回预置 mic/farend 张量与采样率。"""
    wavs = {"mic": (mic, mic_sr), "far": (far, far_sr)}

    def load(path):
        tensor, rate = wavs["mic" if "mic" in str(path) else "far"]
        return tensor.clone(), rate

    return load


def _build_adapter(model=None, *, device: str = "cpu", **kwargs) -> AECInferenceAdapter:
    return AECInferenceAdapter(
        model or _IdentityAECModel(),
        device=device,
        stft=STFT_CFG,
        model_fixed_sr=SR,
        **kwargs,
    )


def _infer(adapter: AECInferenceAdapter, tmp_path, uid: str = "u1") -> None:
    item = FileInferenceItem(
        uid=uid,
        inputs={"mic": "mic.wav", "farend": "far.wav"},
        output=tmp_path / f"{uid}.wav",
    )
    adapter.infer(item).writer(tmp_path / f"{uid}.wav")


def test_adapter_accepts_deploy_flags_without_not_implemented():
    adapter = _build_adapter(cpu_long_audio_seconds=30.0, review_multichannel=True)

    assert adapter.cpu_long_audio_seconds == 30.0
    assert adapter.review_multichannel is True


@pytest.mark.parametrize("threshold", [-1.0, float("nan")])
def test_adapter_rejects_invalid_cpu_threshold(threshold):
    with pytest.raises(ValueError, match="cpu_long_audio_seconds"):
        _build_adapter(cpu_long_audio_seconds=threshold)


def test_adapter_policy_uses_raw_mic_frames_and_rate(monkeypatch, tmp_path):
    """设备判定必须使用重采样前的 mic 原始帧数与原始采样率。"""
    seen: dict = {}
    moves: list = []

    def fake_policy(requested, num_samples, sample_rate, threshold):
        seen.update(
            requested=torch.device(requested),
            num_samples=num_samples,
            sample_rate=sample_rate,
            threshold=threshold,
        )
        return torch.device(requested)

    monkeypatch.setattr(adapter_module, "resolve_aec_file_device", fake_policy)
    monkeypatch.setattr(
        adapter_module,
        "move_aec_runtime",
        lambda model, stft, target, rollback: moves.append(target),
    )
    monkeypatch.setattr(
        torchaudio,
        "load",
        _fake_load(torch.zeros(1, 8000), torch.zeros(1, 8000), mic_sr=8000, far_sr=8000),
    )
    captured: dict = {}
    monkeypatch.setattr(torchaudio, "save", _fake_save(captured))

    _infer(_build_adapter(cpu_long_audio_seconds=30.0), tmp_path)

    assert seen == {
        "requested": torch.device("cpu"),
        "num_samples": 8000,
        "sample_rate": 8000,
        "threshold": 30.0,
    }
    # 8 kHz 1 s 输入重采样到模型 SR 后输出 16000 帧单通道；同设备命中时不迁移。
    assert captured["tensor"].shape == (1, SR)
    assert moves == []


def test_adapter_activate_device_switches_and_restores(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        adapter_module,
        "move_aec_runtime",
        lambda model, stft, target, rollback: calls.append(
            (torch.device(target), torch.device(rollback))
        ),
    )
    adapter = _build_adapter()

    adapter._activate_device(torch.device("cuda:0"))
    assert adapter.device == torch.device("cuda:0")

    adapter._activate_device(torch.device("cpu"))
    assert adapter.device == torch.device("cpu")

    # 已处于目标设备时不重复迁移。
    adapter._activate_device(torch.device("cpu"))

    assert calls == [
        (torch.device("cuda:0"), torch.device("cpu")),
        (torch.device("cpu"), torch.device("cuda:0")),
    ]


def test_adapter_default_mode_writes_mono_enhanced(monkeypatch, tmp_path):
    mic = np.linspace(-0.5, 0.5, HOP + 1, dtype=np.float32)
    far = np.linspace(0.25, -0.25, HOP + 1, dtype=np.float32)
    monkeypatch.setattr(
        torchaudio,
        "load",
        _fake_load(torch.from_numpy(mic).unsqueeze(0), torch.from_numpy(far).unsqueeze(0)),
    )
    captured: dict = {}
    monkeypatch.setattr(torchaudio, "save", _fake_save(captured))

    _infer(_build_adapter(), tmp_path)

    assert captured["tensor"].shape == (1, mic.size)
    assert captured["save_kwargs"] == {}
    assert np.allclose(captured["tensor"][0].numpy(), mic, atol=1e-4)


def test_adapter_review_zero_reference_writes_mic_then_enhanced(monkeypatch, tmp_path):
    mic = np.linspace(-0.5, 0.5, HOP + 1, dtype=np.float32)
    far = np.zeros_like(mic)
    monkeypatch.setattr(
        torchaudio,
        "load",
        _fake_load(torch.from_numpy(mic).unsqueeze(0), torch.from_numpy(far).unsqueeze(0)),
    )
    captured: dict = {}
    monkeypatch.setattr(torchaudio, "save", _fake_save(captured))

    _infer(_build_adapter(_ScaledAECModel(), review_multichannel=True), tmp_path)

    assert captured["tensor"].shape == (2, mic.size)
    assert captured["save_kwargs"] == {"encoding": "PCM_F", "bits_per_sample": 32}
    assert np.allclose(captured["tensor"][0].numpy(), mic, atol=1e-4)
    assert np.allclose(captured["tensor"][1].numpy(), mic * 0.5, atol=1e-4)


def test_adapter_review_nonzero_reference_writes_mic_ref_enhanced(monkeypatch, tmp_path):
    mic = np.linspace(-0.5, 0.5, HOP + 1, dtype=np.float32)
    far = np.linspace(0.25, -0.25, HOP + 1, dtype=np.float32)
    monkeypatch.setattr(
        torchaudio,
        "load",
        _fake_load(torch.from_numpy(mic).unsqueeze(0), torch.from_numpy(far).unsqueeze(0)),
    )
    captured: dict = {}
    monkeypatch.setattr(torchaudio, "save", _fake_save(captured))

    _infer(_build_adapter(_ScaledAECModel(), review_multichannel=True), tmp_path)

    assert captured["tensor"].shape == (3, mic.size)
    assert np.allclose(captured["tensor"][0].numpy(), mic, atol=1e-4)
    assert np.array_equal(captured["tensor"][1].numpy(), far)
    assert np.allclose(captured["tensor"][2].numpy(), mic * 0.5, atol=1e-4)


@pytest.mark.gpu
def test_adapter_long_file_runs_on_cpu_and_restores_cuda(monkeypatch, tmp_path):
    """31 秒 mic 在 cuda 请求下必须切到 CPU 处理，并在文件结束后切回。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = _DeviceRecordingModel().to("cuda:0")
    adapter = _build_adapter(model, device="cuda:0", cpu_long_audio_seconds=30.0)
    assert adapter.device == torch.device("cuda:0")
    long_mic = torch.zeros(1, 31 * SR)
    monkeypatch.setattr(torchaudio, "load", _fake_load(long_mic, long_mic))
    captured: dict = {}
    monkeypatch.setattr(torchaudio, "save", _fake_save(captured))

    _infer(adapter, tmp_path)

    assert model.seen_devices == [torch.device("cpu")]
    assert adapter.device == torch.device("cuda:0")
    assert captured["tensor"].shape == (1, 31 * SR)
