"""AEC 推理 adapter 的谱域、采样率与文件输入合同测试。"""

import pytest
import torch

from core.inference import FileInferenceItem
from tasks.aec.inference import AECInferenceAdapter, build_inference_adapter
from tasks.aec.models.discriminative.deepvqe import DeepVQE


class _IdentityAEC(torch.nn.Module):
    def forward(self, batch):
        return {"nearend_estimate_spectra": batch["mic_spectra"]}


@pytest.mark.unit
def test_aec_adapter_normalizes_dict_spectrum_contract():
    adapter = AECInferenceAdapter(
        _IdentityAEC(),
        stft={"n_fft": 32, "hop_length": 16, "win_length": 32, "window": "hamming", "center": True},
        sample_rate=16000,
    )
    mic = torch.randn(1, 96)
    output = adapter.forward(mic, torch.randn_like(mic), sample_rate=16000)
    assert output["estimate"].shape == mic.shape
    assert output["nearend_estimate_spectra"].shape[:2] == (1, 17)


@pytest.mark.unit
def test_aec_factory_maps_root_stft_cfg_to_task_stft_contract():
    adapter = build_inference_adapter(
        _IdentityAEC(),
        stft_cfg={
            "n_fft": 32,
            "hop_length": 16,
            "win_length": 32,
            "window": "hamming",
            "center": True,
        },
        model_fixed_sr=16000,
    )
    assert isinstance(adapter, AECInferenceAdapter)
    assert adapter.sample_rate == 16000


@pytest.mark.unit
def test_aec_adapter_rejects_multichannel_mic_for_mono_model():
    adapter = AECInferenceAdapter(
        _IdentityAEC(),
        stft={"n_fft": 32, "hop_length": 16, "win_length": 32, "window": "hamming", "center": True},
        sample_rate=16000,
    )
    mic = torch.randn(1, 2, 96)
    farend = torch.randn(1, 96)
    with pytest.raises(ValueError, match="native model IO contract"):
        adapter.forward(mic, farend, sample_rate=16000)


@pytest.mark.unit
def test_aec_adapter_file_contract_requires_both_inputs(tmp_path):
    adapter = AECInferenceAdapter(
        _IdentityAEC(),
        stft={"n_fft": 32, "hop_length": 16, "win_length": 32, "window": "hamming", "center": True},
        sample_rate=16000,
    )
    with pytest.raises(ValueError, match="mic and farend"):
        adapter.infer(FileInferenceItem("utt", {"mic": tmp_path / "mic.wav"}, tmp_path / "out.wav"))


@pytest.mark.unit
def test_aec_adapter_rejects_deepvqe_with_spatial_observation():
    model = DeepVQE(
        variant="small",
        n_fft=32,
        hop_length=16,
        align_delay=1,
    ).eval()
    adapter = AECInferenceAdapter(
        model,
        stft={
            "n_fft": 32,
            "hop_length": 16,
            "win_length": 32,
            "window": "hann",
            "center": True,
        },
        sample_rate=16000,
    )
    mic = torch.randn(1, 2, 256)
    farend = torch.randn(1, 256)
    with pytest.raises(ValueError, match="native model IO contract"):
        adapter.forward(mic, farend, sample_rate=16000)
