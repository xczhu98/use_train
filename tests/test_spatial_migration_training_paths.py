"""真实 gpuRIR 数据链：SS 原生模型训练，TSE 多通道输入与 mono 模型拒绝。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from core.data.simulation.rir_service import close_online_rir_services
from core.models.io_contract import ModelIOContract, TensorField
from core.stft import build_stft_istft
from test_gpurir_moving_end_to_end import _config_dict

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _assets(root):
    rate, length = 8000, 512
    rows, records, enrollment = [], [], {}
    t = np.arange(length) / rate
    for i in range(2):
        uid = f"speaker_{i}"
        path, enroll = root / f"{uid}.wav", root / f"{uid}_enroll.wav"
        wave = (.04 * np.sin(2 * np.pi * (250 + i * 120) * t)).astype(np.float32)
        sf.write(path, wave, rate, subtype="FLOAT")
        sf.write(enroll, wave, rate, subtype="FLOAT")
        rows.append(f"{uid} {rate} {length} {path}\n")
        records.append({"uid": uid, "fs": rate, "length": length, "path": str(path),
                        "language": "en", "dataset_id": "fixture", "speaker_id": uid})
        enrollment[uid] = [str(enroll)]
    (root / "speech.scp").write_text("".join(rows))
    (root / "speech.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (root / "enroll.json").write_text(json.dumps({"fixture": enrollment}))
    noise = np.random.default_rng(42).normal(0, .01, (length, 2)).astype(np.float32)
    sf.write(root / "noise.wav", noise, rate, subtype="FLOAT")
    (root / "noise.scp").write_text(f"noise {rate} {length} {root / 'noise.wav'}\n")
    return rate, length


def _simulation(moving):
    return {**_config_dict(), "prob_reverberation": 1., "prob_moving_rir": float(moving),
            "prob_channel_enabled": False, "noisy_post_mix_degradations_enabled": False,
            "peak_normalization_mode": "off"}


def _multichannel_tse_probe(rate):
    """只声明公共多通道谱输入与 cue 合同的 probe，不是可训练的原生空间 TSE 模型。"""
    return SimpleNamespace(
        sample_rate=rate,
        supports_multichannel_input=True,
        ref_channel=0,
        IO_CONTRACT=ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=(
                TensorField("mix_spectra", "spectrum",
                            ("batch", "microphone", "frequency", "frame", "complex")),
                TensorField("speaker_waveform", "waveform", ("batch", "time"), required=False),
                TensorField("speaker_waveform_present", "mask", ("batch",), required=False),
                TensorField("speaker_waveform_lengths", "label", ("batch",), required=False),
                TensorField("spatial", "embedding", ("batch", "coordinate", "point"), required=False),
                TensorField("spatial_present", "mask", ("batch",), required=False),
                TensorField("spatial_lengths", "label", ("batch",), required=False),
                TensorField("spatial_timebase", "label", ("batch", "point"), required=False),
            ),
            outputs=(
                TensorField("target_estimate_spectra", "spectrum",
                            ("batch", "frequency", "frame", "complex")),
            ),
            forward_kind="estimate",
            frontend_owner="runtime",
        ),
    )


@pytest.mark.parametrize("moving", [False, True])
def test_real_ss_scene_to_native_model_loss(tmp_path, moving):
    from tasks.ss.data.dynamic_dataset import SSDynamicMixingDataset
    from tasks.ss.data.collate import ss_collate_fn
    from tasks.ss.model_io import SSModelRuntime, resolve_ss_execution_plan
    from tasks.ss.trainer.entry import SSTrainingEntry
    rate, length = _assets(tmp_path)
    try:
        dataset = SSDynamicMixingDataset(
            str(tmp_path / "speech.scp"), noise_scp=str(tmp_path / "noise.scp"),
            num_sources=2, sample_rate=rate, segment_seconds=length/rate,
            simulation_cfg={**_simulation(moving), "prob_target_only": 0., "prob_target_noise": 1.},
        )
        batch = ss_collate_fn([dataset[0]])
        cfg = {"data": {"mode": "dynamic_ss", "sample_rate": rate}, "ss": {"num_sources": 2},
               "stft": {"n_fft": 64, "win_length": 64, "hop_length": 32, "window": "hamming", "center": True},
               "model": {"type": "SpatialNet", "args": {
                   "num_freqs": 33, "num_layers": 1, "dim_hidden": 8, "dim_ffn": 16,
                   "dim_squeeze": 4, "num_heads": 2, "conv_groups": [1, 1],
               }}}
        model = object.__new__(SSTrainingEntry).build_model_for_training(cfg, 1, train_dataset=dataset)
        assert model.n_channel == 2 and model.ref_channel == 0
        runtime = SSModelRuntime(model, plan=resolve_ss_execution_plan(cfg, model),
                                 stft_module=build_stft_istft(cfg["stft"]))
        output = runtime.separate(batch.mixture, batch.lengths, rate)
        assert output.sources.shape == batch.sources.shape == (1, 2, length)
        loss = (output.sources - batch.sources).abs().mean()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    finally:
        close_online_rir_services()


@pytest.mark.parametrize("moving", [False, True])
def test_real_tse_scene_to_collate_and_mono_model_gate(tmp_path, moving):
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset
    from tasks.tse.data.collate import tse_collate_fn
    from tasks.tse.model_io import build_tse_model_inputs
    from tasks.tse.trainer.entry import TSETrainingEntry
    rate, length = _assets(tmp_path)
    probabilities = {f"prob_{name}": float(name == "target_interference_noise") for name in (
        "target_only", "interference_only", "noise_only", "interference_noise_only",
        "target_noise", "target_interference", "target_interference_noise",
    )}
    try:
        dataset = TSEDynamicMixingDataset(
            str(tmp_path / "speech.jsonl"), noise_scp=str(tmp_path / "noise.scp"),
            spk2enroll_json=str(tmp_path / "enroll.json"), target_sr=rate,
            max_duration_seconds=length/rate, enrollment_duration=length/rate,
            filter_by_duration=False, enroll_policy="fixed",
            simulation_cfg={**_simulation(moving), **probabilities, "num_interferers": {1: 1.0},
                            "sir_range": [-5.0, 5.0]},
        )
        batch = tse_collate_fn([dataset[0]], cue_specs=dataset.cue_specs)
        stft = build_stft_istft({"n_fft": 64, "win_length": 64, "hop_length": 32, "window": "hamming", "center": True})
        # 这里只验证公共谱输入构造，不把 probe 称为可训练的原生空间 TSE 模型。
        inputs = build_tse_model_inputs(
            model=_multichannel_tse_probe(rate), stft=stft,
            mixture=batch.mixture, target=batch.target, cues=batch.cues, sample_rate=rate,
            lengths=batch.lengths, num_targets=batch.num_targets,
        )
        assert inputs["mix_spectra"].shape[:3] == (1, 2, 33)
        assert inputs["target"].shape == inputs["speaker_waveform"].shape == (1, length)
        with pytest.raises(ValueError, match="native microphone-axis"):
            object.__new__(TSETrainingEntry).build_model_for_training(
                {"data": {"mode": "dynamic_tse"}, "model": {"type": "TSE_BSRNN_SPK", "args": {}}},
                1, train_dataset=dataset,
            )
    finally:
        close_online_rir_services()
