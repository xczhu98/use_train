"""SE 任务推理 adapter 的调用风格与文件推理合同测试。

原文件位于 ``tasks/se/inference/test_adapter.py``，不在 ``pyproject.toml`` 的
``testpaths = ["tests"]`` 内，因此 ``@pytest.mark.unit`` 断言从未被执行；
这里迁入 ``tests/``，并补上模型调用风格探测（不再静默吞 TypeError）与
``chunk_seconds`` 构造期校验的回归覆盖。
"""

import pytest
import torch

from core.inference import FileInferenceItem
from tasks.se.inference import SEInferenceAdapter


class _IdentitySE(torch.nn.Module):
    sample_rate = 16000

    def forward(self, spec, lengths=None, *, fs=None):
        del lengths, fs
        return spec


def _stft():
    return {
        "n_fft": 32,
        "hop_length": 16,
        "win_length": 32,
        "window": "hamming",
        "center": True,
    }


@pytest.mark.unit
def test_se_adapter_round_trips_waveform_and_returns_canonical_spectra():
    waveform = torch.randn(2, 96)
    output = SEInferenceAdapter(_IdentitySE(), stft=_stft(), sample_rate=16000)(waveform)
    assert output["estimate"].shape == waveform.shape
    assert output["estimate_spectra"].shape[:2] == (2, 17)
    assert torch.allclose(output["estimate"], waveform, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_se_adapter_waveform_path_enables_model_output_peak_normalization():
    class _NormalizeModel(torch.nn.Module):
        sample_rate = 16000

        def __init__(self):
            super().__init__()
            self.seen_normalize = None

        def forward(self, wav, lengths=None, *, fs=None, normalize_output=False):
            del lengths, fs
            self.seen_normalize = normalize_output
            return wav

    model = _NormalizeModel()
    adapter = SEInferenceAdapter(model, sample_rate=16000)
    output = adapter._waveform_output(
        torch.randn(1, 64), torch.tensor([64]), 16000
    )

    assert output.shape == (64,)
    assert model.seen_normalize is True


@pytest.mark.unit
def test_se_adapter_rejects_sample_rate_mismatch():
    adapter = SEInferenceAdapter(_IdentitySE(), stft=_stft(), sample_rate=16000)
    with pytest.raises(ValueError, match="sample_rate"):
        adapter(torch.randn(1, 64), sample_rate=8000)


@pytest.mark.unit
def test_se_adapter_file_infer_uses_public_item_output_contract(tmp_path, monkeypatch):
    import torchaudio

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    waveform = torch.randn(1, 64)
    monkeypatch.setattr(torchaudio, "load", lambda path: (waveform, 16000))
    saved = {}
    monkeypatch.setattr(
        torchaudio,
        "save",
        lambda path, wav, sr: saved.update(path=path, wav=wav, sr=sr),
    )
    adapter = SEInferenceAdapter(_IdentitySE(), stft=_stft(), sample_rate=16000)
    output = adapter.infer(
        FileInferenceItem("utt", {"noisy": input_path}, output_path)
    )
    output.writer(output_path)
    assert saved["sr"] == 16000
    assert saved["wav"].shape == waveform.shape


@pytest.mark.unit
def test_se_adapter_propagates_model_type_error_without_silent_fallback():
    class _BrokenModel(torch.nn.Module):
        sample_rate = 16000

        def forward(self, spec, lengths=None, *, fs=None):
            del spec, lengths, fs
            raise TypeError("internal model bug")

    adapter = SEInferenceAdapter(_BrokenModel(), stft=_stft(), sample_rate=16000)
    with pytest.raises(TypeError, match="internal model bug"):
        adapter(torch.randn(1, 64))


@pytest.mark.unit
def test_se_adapter_calls_spectrum_only_model_without_lengths_keyword():
    calls: list[tuple[int, ...]] = []

    class _SpectrumOnlyModel(torch.nn.Module):
        sample_rate = 16000

        def forward(self, spec):
            calls.append(tuple(spec.shape))
            return spec

    adapter = SEInferenceAdapter(
        _SpectrumOnlyModel(), stft=_stft(), sample_rate=16000
    )
    output = adapter(torch.randn(1, 64))
    assert calls == [(1, 17, 5, 2)]
    assert output["estimate"].shape == (1, 64)


@pytest.mark.unit
def test_se_adapter_multi_sr_passes_fs_keyword_once():
    seen: dict[str, int | None] = {}

    class _MultiSrModel(torch.nn.Module):
        sample_rate = 16000

        def forward(self, spec, lengths=None, *, fs=None):
            del lengths
            seen["fs"] = fs
            return spec

    adapter = SEInferenceAdapter(
        _MultiSrModel(), stft=_stft(), sample_rate=16000, multi_sr=True
    )
    adapter(torch.randn(1, 64))
    assert seen["fs"] == 16000


@pytest.mark.unit
@pytest.mark.parametrize("payload", ["mapping", "tuple"])
def test_se_adapter_rejects_non_tensor_spectrum_output(payload):
    """谱进谱出契约只接受 tensor；mapping/元组形态必须显式报错而非被拆包。"""

    class _NonTensorOutput(torch.nn.Module):
        sample_rate = 16000

        def forward(self, spec, lengths=None, *, fs=None):
            del lengths, fs
            if payload == "mapping":
                return {"enhanced_spectra": spec}
            return (spec, None)

    adapter = SEInferenceAdapter(
        _NonTensorOutput(), stft=_stft(), sample_rate=16000
    )
    with pytest.raises(TypeError, match="must return a spectrum torch.Tensor"):
        adapter(torch.randn(1, 64))


@pytest.mark.unit
def test_se_adapter_rejects_multi_sr_without_fs_keyword():
    class _NoFsModel(torch.nn.Module):
        sample_rate = 16000

        def forward(self, spec, lengths=None):
            del lengths
            return spec

    with pytest.raises(TypeError, match="multi_sr=True requires"):
        SEInferenceAdapter(
            _NoFsModel(), stft=_stft(), sample_rate=16000, multi_sr=True
        )


@pytest.mark.unit
def test_se_adapter_file_infer_chunks_waveform_native_model(tmp_path, monkeypatch):
    """分块推理（stft=None + chunk_seconds>0）必须发布 mono 波形。"""
    import torchaudio

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    waveform = torch.randn(1, 96)
    monkeypatch.setattr(torchaudio, "load", lambda path: (waveform, 16000))
    saved = {}
    monkeypatch.setattr(
        torchaudio,
        "save",
        lambda path, wav, sr: saved.update(path=path, wav=wav, sr=sr),
    )

    class _IdentityWaveform(torch.nn.Module):
        sample_rate = 16000

        def forward(self, wav, lengths=None):
            del lengths
            return wav

    adapter = SEInferenceAdapter(
        _IdentityWaveform(), sample_rate=16000, chunk_seconds=16 / 16000
    )
    output = adapter.infer(
        FileInferenceItem("utt", {"noisy": input_path}, output_path)
    )
    output.writer(output_path)

    assert saved["sr"] == 16000
    assert saved["wav"].shape == waveform.shape


@pytest.mark.unit
def test_se_adapter_file_infer_rejects_tuple_waveform_output(tmp_path, monkeypatch):
    """waveform-native 契约只接受 tensor；元组返回必须显式报错。"""
    import torchaudio

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    monkeypatch.setattr(torchaudio, "load", lambda path: (torch.randn(1, 96), 16000))

    class _TupleWaveform(torch.nn.Module):
        sample_rate = 16000

        def forward(self, wav, lengths=None):
            del lengths
            return wav, None

    adapter = SEInferenceAdapter(_TupleWaveform(), sample_rate=16000)
    with pytest.raises(TypeError, match="must return a torch.Tensor"):
        adapter.infer(FileInferenceItem("utt", {"noisy": input_path}, output_path))


@pytest.mark.unit
@pytest.mark.parametrize("chunk_seconds", [-1.0, float("nan"), float("inf")])
def test_se_adapter_rejects_invalid_chunk_seconds(chunk_seconds):
    with pytest.raises(ValueError, match="chunk_seconds"):
        SEInferenceAdapter(
            _IdentitySE(),
            stft=_stft(),
            sample_rate=16000,
            chunk_seconds=chunk_seconds,
        )


@pytest.mark.unit
def test_se_adapter_restores_non_16k_input_rate_on_output(tmp_path, monkeypatch):
    """固定 16k 模型对 48k 输入应写回 48k 文件，且时长不变。"""
    import torchaudio

    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    waveform = torch.randn(1, 480)
    monkeypatch.setattr(torchaudio, "load", lambda path: (waveform, 48000))
    saved = {}
    monkeypatch.setattr(
        torchaudio,
        "save",
        lambda path, wav, sr: saved.update(path=path, wav=wav, sr=sr),
    )
    adapter = SEInferenceAdapter(_IdentitySE(), sample_rate=16000)
    output = adapter.infer(
        FileInferenceItem("utt", {"noisy": input_path}, output_path)
    )
    output.writer(output_path)
    assert saved["sr"] == 48000
    assert saved["wav"].shape[-1] == 480
