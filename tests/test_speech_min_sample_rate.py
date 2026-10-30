"""Source speech 最低采样率过滤（官方 UniPASE/GAP ``fs >= default_fs``）测试。

官方 dataloader 读取 speech SCP 时执行 ``if int(fs) >= self.default_fs``，
低于阈值的源语音不会进入采样池；本文件验证本地 Dataset 的等价语义。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile

from tasks.se.data.dynamic_dataset import DynamicMixingDataset


def _make_wav(path: Path, duration: float, sr: int, *, seed: int = 0) -> None:
    """写一段合成单声道 WAV（正弦 + 少量噪声）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.sin(2.0 * np.pi * 300.0 * t).astype(np.float32)
    signal += 0.01 * rng.standard_normal(signal.shape).astype(np.float32)
    signal = np.clip(signal, -0.99, 0.99)
    soundfile.write(path, signal, sr)


def _make_impulse_rir(path: Path, sr: int) -> None:
    """写单位冲激 RIR，使混响近似恒等。"""
    rir = np.zeros(sr, dtype=np.float32)
    rir[0] = 1.0
    soundfile.write(path, rir, sr)


def _write_scp(path: Path, entries: dict[str, Path]) -> None:
    """按 ``uid fs num_samples path`` 四列写 source SCP。"""
    with path.open("w", encoding="utf-8") as handle:
        for uid, audio_path in entries.items():
            info = soundfile.info(str(audio_path))
            handle.write(f"{uid} {info.samplerate} {info.frames} {audio_path}\n")


def _write_assets(tmp_path: Path) -> dict[str, str]:
    """生成 8k/16k/48k speech 与统一的 16k noise/rir/wind 清单。"""
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    for directory in (speech_dir, noise_dir, rir_dir, wind_dir):
        directory.mkdir(exist_ok=True)

    speech_8k = speech_dir / "spk8.wav"
    speech_16k = speech_dir / "spk16.wav"
    speech_48k = speech_dir / "spk48.wav"
    noise_wav = noise_dir / "noise.wav"
    rir_wav = rir_dir / "rir.wav"
    wind_wav = wind_dir / "wind.wav"

    _make_wav(speech_8k, 2.0, 8000, seed=1)
    _make_wav(speech_16k, 2.0, 16000, seed=2)
    _make_wav(speech_48k, 2.0, 48000, seed=3)
    _make_wav(noise_wav, 2.0, 16000, seed=4)
    _make_impulse_rir(rir_wav, 16000)
    _make_wav(wind_wav, 2.0, 16000, seed=5)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, {"spk8": speech_8k, "spk16": speech_16k, "spk48": speech_48k})
    _write_scp(noise_scp, {"noise": noise_wav})
    _write_scp(rir_scp, {"rir": rir_wav})
    _write_scp(wind_scp, {"wind": wind_wav})
    return {
        "speech_scp": str(speech_scp),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "windnoise_scp": str(wind_scp),
    }


def _build_dataset(tmp_path: Path, *, speech_min_sample_rate: int) -> DynamicMixingDataset:
    """构造 target_sr=None 的 SE 动态数据集，保留源自身采样率。"""
    paths = _write_assets(tmp_path)
    dataset = DynamicMixingDataset(
        simulation_cfg={
            "prob_target_only": 1.0,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
        },
        max_duration_seconds=1.0,
        repeat_to_max_duration=True,
        target_sr=None,
        speech_min_sample_rate=speech_min_sample_rate,
        filter_by_duration=True,
        strict_load=True,
        **paths,
    )
    dataset.set_epoch(0, base_seed=123)
    return dataset


@pytest.mark.unit
def test_sidecar_with_filtered_out_uid_is_allowed(tmp_path: Path) -> None:
    """source 过滤后，覆盖全量 speech SCP 的 sidecar 额外 UID 应被忽略。"""
    paths = _write_assets(tmp_path)
    sidecar = tmp_path / "speech_lengths.txt"
    sidecar.write_text(
        "spk8 16000\nspk16 32000\nspk48 96000\n", encoding="utf-8"
    )
    dataset = DynamicMixingDataset(
        simulation_cfg={
            "prob_target_only": 1.0,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
        },
        max_duration_seconds=1.0,
        repeat_to_max_duration=True,
        target_sr=None,
        speech_min_sample_rate=16000,
        speech_length_file=str(sidecar),
        filter_by_duration=True,
        strict_load=True,
        **paths,
    )
    dataset.set_epoch(0, base_seed=123)

    assert set(dataset.source_length) == {"spk16", "spk48"}
    assert dataset.samplerates == [16000, 48000]


@pytest.mark.unit
def test_filters_sources_below_min_sample_rate(tmp_path: Path) -> None:
    """阈值 16k 时只保留 16k/48k 源，样本与 get_srs 都不再出现 8k。"""
    dataset = _build_dataset(tmp_path, speech_min_sample_rate=16000)

    assert dataset.samplerates == [16000, 48000]
    assert set(dataset.speech_flt) == {"spk16", "spk48"}
    assert set(dataset.source_length) == {"spk16", "spk48"}
    assert dataset.speech_min_sample_rate == 16000

    assert 8000 not in dataset.get_srs()
    assert set(dataset.get_srs()) == {16000, 48000}
    assert len(dataset) == 2
    for index in range(len(dataset)):
        _clean, _noisy, fs_out, length, *_ = dataset[index]
        assert fs_out >= 16000
        assert length > 0


@pytest.mark.unit
def test_threshold_48000_keeps_only_48k(tmp_path: Path) -> None:
    """阈值 48k 时只保留恰好等于阈值的 48k 源。"""
    dataset = _build_dataset(tmp_path, speech_min_sample_rate=48000)

    assert dataset.samplerates == [48000]
    assert set(dataset.speech_flt) == {"spk48"}
    assert dataset.get_srs() == [48000]
    assert len(dataset) == 1


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "16000"])
def test_rejects_invalid_speech_min_sample_rate(tmp_path: Path, value) -> None:
    """bool/0/负值/float/str 必须在构造边界 fail-fast。"""
    with pytest.raises(ValueError, match="speech_min_sample_rate"):
        DynamicMixingDataset(
            speech_scp=str(tmp_path / "missing.scp"),
            noise_scp=str(tmp_path / "missing.scp"),
            rir_scp=str(tmp_path / "missing.scp"),
            speech_min_sample_rate=value,
        )


@pytest.mark.unit
def test_rejects_threshold_above_every_source(tmp_path: Path) -> None:
    """过滤后没有任何语音时抛错，并报告阈值与原 samplerates。"""
    with pytest.raises(ValueError, match="filtered out every speech source") as excinfo:
        _build_dataset(tmp_path, speech_min_sample_rate=96000)

    message = str(excinfo.value)
    assert "96000" in message
    assert "[8000, 16000, 48000]" in message


@pytest.mark.unit
def test_filter_helper_syncs_lengths_and_metadata() -> None:
    """过滤 helper 同步裁剪长度映射与 metadata，且不修改输入容器。"""
    from core.data.dynamic_dataset_base import _filter_speech_by_min_sample_rate

    speech_source = {8000: {"a": "a.wav"}, 16000: {"b": "b.wav"}}
    speech_uids = {8000: ["a"], 16000: ["b"]}
    speech_flt = {"a": "a.wav", "b": "b.wav"}
    sp_lens = {"a": 8000, "b": 16000}
    metadata = {"a": {"speaker_id": "s0"}, "b": {"speaker_id": "s1"}}

    (
        filtered_source,
        filtered_uids,
        filtered_flt,
        filtered_lens,
        filtered_metadata,
    ) = _filter_speech_by_min_sample_rate(
        speech_source, speech_uids, speech_flt, sp_lens, metadata, 16000
    )

    assert dict(filtered_source) == {16000: {"b": "b.wav"}}
    assert filtered_uids == {16000: ["b"]}
    assert filtered_flt == {"b": "b.wav"}
    assert filtered_lens == {"b": 16000}
    assert filtered_metadata == {"b": {"speaker_id": "s1"}}
    # 输入容器保持完整，避免污染 read_source_scp 的缓存对象。
    assert set(speech_source) == {8000, 16000}
    assert set(speech_flt) == {"a", "b"}
    assert set(sp_lens) == {"a", "b"}
    assert set(metadata) == {"a", "b"}


@pytest.mark.unit
def test_validation_can_override_speech_min_sample_rate() -> None:
    """inherit 合并时 validation 的阈值覆盖 train，而不是被忽略。"""
    from core.data.dataset_build import merge_validation_section

    merged = merge_validation_section(
        {"speech_scp": "train.scp", "speech_min_sample_rate": 16000},
        {"inherit_simulation": True, "speech_min_sample_rate": 48000},
    )
    assert merged["speech_min_sample_rate"] == 48000


@pytest.mark.unit
def test_builder_forwards_and_validates_min_sample_rate() -> None:
    """train 阈值进入 Dataset kwargs；val inherit 时可用 validation 值覆盖。"""
    from tasks.se.data import build_dataset

    cfg = {
        "data": {
            "mode": "dynamic",
            "sample_rate": 16000,
            "train": {
                "speech_scp": "dummy.scp",
                "noise_scp": "dummy.scp",
                "speech_min_sample_rate": 16000,
                "simulation_cfg": {},
            },
            "validation": {
                "mode": "dynamic",
                "inherit_simulation": True,
                "speech_min_sample_rate": 48000,
            },
        },
        "meta": {"seed": 0},
    }
    with patch("tasks.se.data.DynamicMixingDataset") as mock_ds:
        build_dataset(cfg["data"], mode="train")
        assert mock_ds.call_args.kwargs["speech_min_sample_rate"] == 16000
        build_dataset(cfg["data"], mode="val", meta=cfg["meta"])
        assert mock_ds.call_args.kwargs["speech_min_sample_rate"] == 48000


pytestmark = [pytest.mark.unit]
