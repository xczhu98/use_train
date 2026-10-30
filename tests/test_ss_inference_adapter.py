"""SS 文件推理 adapter 的 waveform/spectrum 双域合同。

原位于 ``tasks/ss/inference/test_adapter.py``，不在 ``testpaths`` 收集域内；
迁移到 ``tests/`` 后由默认 unit marker 执行，任务包内重复副本已删除。
"""

import pytest
import torch

from core.stft import build_stft_istft
from core.inference import FileInferenceItem
from core.models.io_contract import ModelIOContract, TensorField
from tasks.ss.inference import SSInferenceAdapter
from tasks.ss.model_io import resolve_ss_execution_plan


class _WaveformSeparator(torch.nn.Module):
    input_domain = "waveform"
    sample_rate = 16000

    def forward(self, mixture, lengths, *, fs=None):
        del lengths, fs
        return {"est_sources": torch.stack((mixture, -mixture), dim=1)}


class _SpectrumSeparator(torch.nn.Module):
    input_domain = "spectrum"
    sample_rate = 16000
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(
            TensorField(
                "mixture_spectra",
                "spectrum",
                ("batch", "microphone", "frequency", "frame", "complex"),
            ),
        ),
        outputs=(
            TensorField(
                "est_sources_spectra",
                "spectrum",
                ("batch", "source", "frequency", "frame", "complex"),
                multiple=True,
            ),
        ),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def forward(self, spectra, lengths, *, fs=None):
        del lengths, fs
        return {"est_sources_spectra": torch.stack((spectra, -spectra), dim=1)}


def _config(stft=None):
    result = {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}}
    if stft:
        result["stft"] = stft
    return result


def _stft_cfg():
    return {"n_fft": 32, "hop_length": 16, "win_length": 32, "window": "hamming", "center": True}


@pytest.mark.unit
def test_ss_adapter_returns_waveform_canonical_output_for_tdanet_domain():
    model = _WaveformSeparator()
    adapter = SSInferenceAdapter(model, plan=resolve_ss_execution_plan(_config(), model))
    output = adapter(torch.randn(1, 96), sample_rate=16000)
    assert output["est_sources"].shape == (1, 2, 96)
    assert "est_sources_spectra" not in output


@pytest.mark.unit
def test_ss_adapter_decodes_tiger_spectra_and_keeps_canonical_spectra():
    model = _SpectrumSeparator()
    stft = build_stft_istft(_stft_cfg())
    adapter = SSInferenceAdapter(
        model,
        plan=resolve_ss_execution_plan(_config(_stft_cfg()), model),
        stft_module=stft,
    )
    output = adapter(torch.randn(1, 96), sample_rate=16000)
    assert output["est_sources"].shape == (1, 2, 96)
    assert output["est_sources_spectra"].shape[:2] == (1, 2)


@pytest.mark.unit
def test_ss_adapter_file_infer_writes_selected_source_multichannel(tmp_path, monkeypatch):
    import torchaudio

    input_path = tmp_path / "mixture.wav"
    output_path = tmp_path / "source.wav"
    mixture = torch.randn(1, 64)
    monkeypatch.setattr(torchaudio, "load", lambda path: (mixture, 16000))
    saved = {}
    monkeypatch.setattr(
        torchaudio,
        "save",
        lambda path, wav, sr: saved.update(path=path, wav=wav, sr=sr),
    )
    model = _WaveformSeparator()
    adapter = SSInferenceAdapter(model, plan=resolve_ss_execution_plan(_config(), model))
    output = adapter.infer(
        FileInferenceItem(
            "utt",
            {"mixture": input_path},
            output_path,
            attributes={"source_index": 1},
        )
    )
    output.writer(output_path)
    assert saved["sr"] == 16000
    assert torch.allclose(saved["wav"], -mixture)


@pytest.mark.unit
def test_ss_adapter_file_infer_accepts_multichannel_spatialnet(tmp_path, monkeypatch):
    import torchaudio

    input_path = tmp_path / "multichannel.wav"
    output_path = tmp_path / "source.wav"
    mixture = torch.randn(2, 96)
    monkeypatch.setattr(torchaudio, "load", lambda path: (mixture, 16000))
    saved = {}
    monkeypatch.setattr(
        torchaudio,
        "save",
        lambda path, wav, sr: saved.update(path=path, wav=wav, sr=sr),
    )

    class _RecordingSpectrumSeparator(_SpectrumSeparator):
        def forward(self, spectra, lengths, *, fs=None):
            del lengths, fs
            self.seen_shape = tuple(spectra.shape)
            reference = spectra[:, 0]
            return {"est_sources_spectra": torch.stack((reference, -reference), dim=1)}

    model = _RecordingSpectrumSeparator()
    stft = build_stft_istft(_stft_cfg())
    adapter = SSInferenceAdapter(
        model,
        plan=resolve_ss_execution_plan(_config(_stft_cfg()), model),
        stft_module=stft,
    )
    result = adapter.infer(
        FileInferenceItem("spatial", {"mixture": input_path}, output_path, attributes={"source_index": 1})
    )
    result.writer(output_path)
    assert model.seen_shape[0:2] == (1, 2)
    assert saved["wav"].shape == (1, 96)
    assert saved["sr"] == 16000
