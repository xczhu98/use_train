"""IPDNetFixed offline 分块前向与真实评估时间轴合同。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.unit
_DEVICE_CASES = (
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.gpu),
)


def _select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device(name)


def _fixed_model(*, causal: bool = False, n_seg: int = 24, input_size: int = 6):
    from tasks.ssl.models.ipdnet_official import IPDNetFixed

    return IPDNetFixed(
        input_size=input_size,
        hidden_size=4,
        max_track=2,
        n_fft=16,
        causal=causal,
        dropout=0.0,
        cnn_hidden_dim=4,
        n_seg=n_seg,
    )


def _features(frames: int, *, batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8102 + frames)
    return torch.randn(batch, frames, 9, 6, generator=generator)


def _segmented_forward_oracle(model, features: torch.Tensor) -> torch.Tensor:
    """逐块调用默认整段路径，独立构造官方的补齐、拼接和裁剪结果。"""
    batch, frames, frequencies, channels = features.shape
    segments = math.ceil(frames / model.n_seg)
    padded_frames = segments * model.n_seg
    if padded_frames > frames:
        features = torch.cat(
            (
                features,
                features.new_zeros(batch, padded_frames - frames, frequencies, channels),
            ),
            dim=1,
        )
    outputs = []
    for index in range(segments):
        start = index * model.n_seg
        outputs.append(model(features[:, start : start + model.n_seg]))
    return torch.cat(outputs, dim=1)[:, : frames // model.stride]


@pytest.mark.parametrize(
    ("frames", "expected_output_frames"),
    [(48, 4), (41, 3)],
    ids=("whole-segments", "right-padded-tail"),
)
@pytest.mark.parametrize("device_name", _DEVICE_CASES)
def test_ipdnet_fixed_offline_chunks_match_independent_segment_oracle(
    frames: int, expected_output_frames: int, device_name: str
):
    device = _select_device(device_name)
    model = _fixed_model().to(device).eval()
    features = _features(frames).to(device)

    actual = model(features, offline_inference=True)
    expected = _segmented_forward_oracle(model, features)

    assert tuple(actual.shape) == (2, expected_output_frames, 9, 2, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_ipdnet_fixed_offline_shorter_than_stride_returns_empty_canonical_axis():
    model = _fixed_model().eval()

    output = model(_features(11), offline_inference=True)

    assert tuple(output.shape) == (2, 0, 9, 2, 2, 2)


def test_ipdnet_fixed_offline_gates_and_default_path_semantics():
    features = _features(24)

    online_model = _fixed_model(causal=True).eval()
    with pytest.raises(ValueError, match="causal=false"):
        online_model(features, offline_inference=True)

    training_model = _fixed_model()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        training_model(features, offline_inference=True)

    invalid_segment_model = _fixed_model(n_seg=25).eval()
    with pytest.raises(ValueError, match="multiple of the official time stride"):
        invalid_segment_model(features, offline_inference=True)

    # n_seg 的整除限制只适用于显式分块；默认整段前向不受它影响。
    default_output = invalid_segment_model(features)
    explicit_default_output = invalid_segment_model(features, offline_inference=False)
    assert tuple(default_output.shape) == (2, 2, 9, 2, 2, 2)
    torch.testing.assert_close(default_output, explicit_default_output, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", _DEVICE_CASES)
def test_ipdnet_fixed_global_mean_is_computed_over_the_complete_observation(device_name: str):
    from tasks.ssl.data.projection import SSLProjectionConfig, _stft

    device = _select_device(device_name)
    torch.manual_seed(522)
    waveform = torch.randn(3, 208, device=device)
    projection = SSLProjectionConfig(
        n_fft=16,
        hop_length=4,
        win_length=16,
        window="hamming",
        center=False,
        normalization="global_mean",
        output_stride=12,
    )
    spectra = _stft(waveform, projection)
    features = torch.view_as_real(spectra.permute(2, 1, 0).contiguous()).reshape(
        1, spectra.shape[-1], spectra.shape[-2], 6
    )
    assert spectra.abs().mean().item() == pytest.approx(1.0, abs=1e-6)

    model = _fixed_model().to(device).eval()
    captured: list[torch.Tensor] = []
    original_core = model._forward_core

    def capture_segment(segment: torch.Tensor) -> torch.Tensor:
        captured.append(segment.detach().clone())
        return original_core(segment)

    model._forward_core = capture_segment
    model(features, offline_inference=True)

    segment_count = math.ceil(features.shape[1] / 24)
    assert len(captured) == math.ceil(segment_count / model.offline_chunk_batch_size)
    segment_view = torch.cat(captured, dim=0).reshape(1, segment_count, 24, 9, 6)
    canonical = features.reshape(1, features.shape[1], 9, 3, 2)
    official_layout = torch.cat((canonical[..., 0], canonical[..., 1]), dim=-1)
    padded_features = torch.cat(
        (
            official_layout,
            official_layout.new_zeros(
                1, segment_count * 24 - features.shape[1], 9, 6
            ),
        ),
        dim=1,
    ).reshape(1, segment_count, 24, 9, 6)
    torch.testing.assert_close(segment_view, padded_features)
    # 补零只作用于全录音归一化后的尾部，不再按 chunk 重新估计均值。
    padding_frames = segment_count * 24 - features.shape[1]
    if padding_frames:
        assert torch.count_nonzero(segment_view[:, -1, -padding_frames:]) == 0


def _projection_and_spec(model, *, output_stride: int = 12, normalization: str = "global_mean"):
    from tools.verify.ssl.real_eval import _projection_from_config
    from tasks.ssl.model_io import resolve_ssl_model_spec

    spec = resolve_ssl_model_spec(model)
    config = {
        "stft": {
            "n_fft": 16,
            "hop_length": 4,
            "win_length": 16,
            "window": "hamming",
            "center": False,
        },
        "ssl": {
            "time_stride": output_stride,
            "normalization": normalization,
            "reference_channel": 0,
            "sound_speed": 343.0,
            "vad_mode": "direct_path",
        },
    }
    projection = _projection_from_config(config, spec, sample_rate=16000, model=model)
    return config, spec, projection


def test_real_eval_rejects_retired_or_conflicting_output_stride():
    from tools.verify.ssl.real_eval import (
        _projection_from_config,
        _validate_model_time_stride,
    )

    model = _fixed_model().eval()
    _, spec, projection = _projection_and_spec(model, output_stride=1)
    with pytest.raises(ValueError, match="conflicts with ssl.time_stride"):
        _validate_model_time_stride(model, spec, projection)

    config, _, _ = _projection_and_spec(model)
    config["ssl"]["output_stride"] = 12
    with pytest.raises(ValueError, match="ssl.output_stride is retired"):
        _projection_from_config(config, spec, sample_rate=16000, model=model)


def test_real_eval_offline_cli_is_opt_in_and_unused_n_seg_stays_unconstrained():
    from tools.verify.ssl.real_eval import build_parser, evaluate_records

    base_args = [
        "--manifest", "records.jsonl",
        "--config", "config.yaml",
        "--checkpoint", "model.pt",
        "--output", "result.json",
    ]
    assert build_parser().parse_args(base_args).offline_inference is False
    assert build_parser().parse_args([*base_args, "--offline-inference"]).offline_inference is True

    model = _fixed_model(n_seg=25).eval()
    _, spec, projection = _projection_and_spec(model)
    with pytest.raises(ValueError, match="received no records"):
        evaluate_records(
            [],
            model=model,
            spec=spec,
            projection=projection,
            device=torch.device("cpu"),
            sample_rate=16000,
            azimuth_domain=(-math.pi / 2, math.pi / 2),
            azimuth_step_deg=90,
            vad_threshold=0.5,
            accuracy_threshold_deg=10.0,
        )


def test_real_eval_label_axis_uses_exact_left_aligned_stride_groups(tmp_path: Path):
    from tools.verify.ssl.real_eval import EvalRecord, _align_labels

    frame_count = 25
    sample_rate = 16000
    hop_length = 4
    frame_times = (np.arange(frame_count) * hop_length + 8) / sample_rate
    angles = np.linspace(-0.4, 0.4, frame_count)
    vad = np.linspace(0.0, 1.0, frame_count)
    record = EvalRecord(
        key="labels",
        audio=tmp_path / "unused.wav",
        sample_rate=sample_rate,
        mic_positions=np.asarray(((-0.04, 0, 0), (0.04, 0, 0)), dtype=np.float32),
        label_times_s=frame_times,
        azimuth_rad=angles,
        vad=vad,
    )
    projection = _projection_for_test(output_stride=12)

    aligned_angle, aligned_vad, inside = _align_labels(
        record,
        frame_count=frame_count,
        projection=projection,
        output_count=2,
        sample_rate=sample_rate,
    )

    expected_angles = np.angle(np.mean(np.exp(1j * angles[:24].reshape(2, 12)), axis=1))
    np.testing.assert_allclose(aligned_angle, expected_angles)
    np.testing.assert_allclose(aligned_vad, vad[:24].reshape(2, 12).mean(axis=1))
    assert inside.tolist() == [True, True]
    with pytest.raises(ValueError, match="actual_output_frames=1"):
        _align_labels(
            record,
            frame_count=frame_count,
            projection=projection,
            output_count=1,
            sample_rate=sample_rate,
        )


def _projection_for_test(*, output_stride: int):
    from tasks.ssl.data.projection import SSLProjectionConfig

    return SSLProjectionConfig(
        n_fft=16,
        hop_length=4,
        win_length=16,
        window="hamming",
        center=False,
        output_stride=output_stride,
    )


@pytest.mark.parametrize("device_name", _DEVICE_CASES)
def test_real_eval_offline_inference_end_to_end_and_metadata(
    tmp_path: Path, monkeypatch, device_name: str
):
    sf = pytest.importorskip("soundfile")
    from tools.verify.ssl.real_eval import EvalRecord, evaluate_records

    device = _select_device(device_name)
    model = _fixed_model(input_size=4).to(device).eval()
    _, spec, projection = _projection_and_spec(model)
    samples = np.arange(208, dtype=np.float32)
    left = np.sin(2 * np.pi * samples / 31).astype(np.float32)
    right = (0.7 * np.sin(2 * np.pi * samples / 27 + 0.3)).astype(np.float32)
    audio_path = tmp_path / "record.wav"
    sf.write(audio_path, np.stack((left, right), axis=1), 16000)
    record = EvalRecord(
        key="offline-record",
        audio=audio_path,
        sample_rate=16000,
        mic_positions=np.asarray(((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0)), dtype=np.float32),
        label_times_s=np.asarray((0.0,), dtype=np.float64),
        azimuth_rad=np.asarray((0.2,), dtype=np.float64),
        vad=np.asarray((1.0,), dtype=np.float64),
    )
    captured_features: list[torch.Tensor] = []
    original_forward = model.forward

    def capture_forward(
        multichannel_stft_real_imag=None,
        *,
        features=None,
        offline_inference: bool = False,
    ):
        value = multichannel_stft_real_imag if multichannel_stft_real_imag is not None else features
        captured_features.append(value.detach().clone())
        if multichannel_stft_real_imag is not None:
            return original_forward(
                multichannel_stft_real_imag,
                offline_inference=offline_inference,
            )
        return original_forward(features=features, offline_inference=offline_inference)

    monkeypatch.setattr(model, "forward", capture_forward)
    result = evaluate_records(
        [record],
        model=model,
        spec=spec,
        projection=projection,
        device=device,
        sample_rate=16000,
        azimuth_domain=(-math.pi / 2, math.pi / 2),
        azimuth_step_deg=90,
        vad_threshold=0.5,
        accuracy_threshold_deg=10.0,
        offline_inference=True,
    )

    assert result["inference"] == {
        "offline_inference": True,
        "n_seg": 24,
        "offline_chunk_batch_size": 1,
    }
    assert result["projection"]["normalization"] == "global_mean"
    assert len(captured_features) == 1
    feature = captured_features[0]
    complex_features = torch.view_as_complex(
        feature.reshape(1, feature.shape[1], feature.shape[2], 2, 2).contiguous()
    )
    assert complex_features.abs().mean().item() == pytest.approx(1.0, abs=1e-5)
    item = result["items"][0]
    assert item["output_frames"] == item["input_frames"] // 12
    assert item["alignment"]["time_stride"] == 12
    assert item["alignment"]["output_group"] == "left_aligned_stride_groups"


def test_real_eval_offline_rejects_online_and_non_global_normalization():
    from tools.verify.ssl.real_eval import evaluate_records

    online_model = _fixed_model(causal=True).eval()
    _, online_spec, online_projection = _projection_and_spec(online_model)
    args = dict(
        records=[],
        model=online_model,
        spec=online_spec,
        projection=online_projection,
        device=torch.device("cpu"),
        sample_rate=16000,
        azimuth_domain=(-math.pi / 2, math.pi / 2),
        azimuth_step_deg=90,
        vad_threshold=0.5,
        accuracy_threshold_deg=10.0,
        offline_inference=True,
    )
    with pytest.raises(ValueError, match="causal=false"):
        evaluate_records(**args)

    model = _fixed_model().eval()
    _, spec, projection = _projection_and_spec(model, normalization="none")
    args.update(model=model, spec=spec, projection=projection)
    with pytest.raises(ValueError, match="global_mean"):
        evaluate_records(**args)
