"""Source speech 语料子集过滤（官方 UniPASE/GAP ``selected_set``）测试。

官方 ``loaders/dataloader.py`` 先按 ``selected_set`` 对 ``audio_path`` 做子串匹配，
再执行 ``fs >= default_fs``；本文件验证本地 ``speech_selected_sets`` 的等价语义。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile

from core.data.dynamic_dataset_base import (
    _filter_speech_by_selected_sets,
    _normalize_speech_selected_sets,
)
from tasks.se.data.dynamic_dataset import DynamicMixingDataset

pytestmark = [pytest.mark.unit]

# 语料标签直接写进路径，模拟官方 audio_path 中的语料目录名。
_SPEECH_SPECS = {
    "spk_dns_16k": ("dns5_fullband", 16000),
    "spk_dns_8k": ("dns5_fullband", 8000),
    "spk_lib_48k": ("libritts", 48000),
    "spk_vctk_16k": ("vctk", 16000),
    "spk_other_16k": ("other", 16000),
}


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
    """生成带语料目录标签的 speech 与统一 16k noise/rir/wind 清单。"""
    speech_dir = tmp_path / "speech"
    noise_dir = tmp_path / "noise"
    rir_dir = tmp_path / "rir"
    wind_dir = tmp_path / "wind"
    for directory in (speech_dir, noise_dir, rir_dir, wind_dir):
        directory.mkdir(exist_ok=True)

    speech_entries: dict[str, Path] = {}
    for seed, (uid, (corpus, sr)) in enumerate(_SPEECH_SPECS.items(), start=1):
        corpus_dir = speech_dir / corpus
        corpus_dir.mkdir(exist_ok=True)
        audio_path = corpus_dir / f"{uid}.wav"
        _make_wav(audio_path, 2.0, sr, seed=seed)
        speech_entries[uid] = audio_path

    noise_wav = noise_dir / "noise.wav"
    rir_wav = rir_dir / "rir.wav"
    wind_wav = wind_dir / "wind.wav"
    _make_wav(noise_wav, 2.0, 16000, seed=10)
    _make_impulse_rir(rir_wav, 16000)
    _make_wav(wind_wav, 2.0, 16000, seed=11)

    speech_scp = tmp_path / "speech.scp"
    noise_scp = tmp_path / "noise.scp"
    rir_scp = tmp_path / "rir.scp"
    wind_scp = tmp_path / "wind.scp"
    _write_scp(speech_scp, speech_entries)
    _write_scp(noise_scp, {"noise": noise_wav})
    _write_scp(rir_scp, {"rir": rir_wav})
    _write_scp(wind_scp, {"wind": wind_wav})
    return {
        "speech_scp": str(speech_scp),
        "noise_scp": str(noise_scp),
        "rir_scp": str(rir_scp),
        "windnoise_scp": str(wind_scp),
    }


def _build_dataset(
    tmp_path: Path,
    *,
    speech_selected_sets=None,
    speech_min_sample_rate=None,
    speech_length_file=None,
) -> DynamicMixingDataset:
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
        speech_selected_sets=speech_selected_sets,
        speech_length_file=speech_length_file,
        filter_by_duration=True,
        strict_load=True,
        **paths,
    )
    dataset.set_epoch(0, base_seed=123)
    return dataset


def test_selected_sets_keep_matching_sources(tmp_path: Path) -> None:
    """命中任一标签的 UID 保留，未命中的语料被剔除。"""
    dataset = _build_dataset(
        tmp_path, speech_selected_sets=["dns5_fullband", "libritts"]
    )

    assert set(dataset.speech_flt) == {"spk_dns_16k", "spk_dns_8k", "spk_lib_48k"}
    assert set(dataset.samplerates) == {8000, 16000, 48000}
    assert set(dataset.get_srs()) == {8000, 16000, 48000}
    assert dataset.speech_selected_sets == ("dns5_fullband", "libritts")
    assert len(dataset) == 3
    _clean, _noisy, fs_out, length, *_ = dataset[0]
    assert fs_out in {8000, 16000, 48000}
    assert length > 0


@pytest.mark.parametrize("value", [None, []])
def test_none_or_empty_selected_sets_disables_filter(tmp_path: Path, value) -> None:
    """None 与空 list 都表示不过滤，全部源语音进入采样池。"""
    dataset = _build_dataset(tmp_path, speech_selected_sets=value)

    assert set(dataset.speech_flt) == set(_SPEECH_SPECS)
    assert set(dataset.samplerates) == {8000, 16000, 48000}
    assert dataset.speech_selected_sets is None
    assert len(dataset) == len(_SPEECH_SPECS)


def test_unmatched_selected_sets_fail_fast(tmp_path: Path) -> None:
    """没有任何 UID 命中时抛错，并报告配置与原始规模。"""
    with pytest.raises(ValueError, match="filtered out every speech source") as excinfo:
        _build_dataset(tmp_path, speech_selected_sets=["missing_corpus"])

    message = str(excinfo.value)
    assert "missing_corpus" in message
    assert "5 UID(s)" in message


def test_selected_sets_match_is_case_sensitive(tmp_path: Path) -> None:
    """官方 ``s in audio_path`` 是大小写敏感匹配；大写标签不能命中小写路径。"""
    dataset = _build_dataset(
        tmp_path, speech_selected_sets=["DNS5_FULLBAND", "libritts"]
    )

    assert set(dataset.speech_flt) == {"spk_lib_48k"}


def test_selected_sets_combine_with_min_sample_rate(tmp_path: Path) -> None:
    """语料过滤与采样率过滤可组合：命中标签但采样率不足的 UID 仍被剔除。"""
    dataset = _build_dataset(
        tmp_path,
        speech_selected_sets=["dns5_fullband", "libritts"],
        speech_min_sample_rate=16000,
    )

    assert set(dataset.speech_flt) == {"spk_dns_16k", "spk_lib_48k"}
    assert dataset.samplerates == [16000, 48000]
    assert 8000 not in dataset.get_srs()
    assert dataset.speech_min_sample_rate == 16000
    assert dataset.speech_selected_sets == ("dns5_fullband", "libritts")


def test_sidecar_with_filtered_out_uid_is_allowed(tmp_path: Path) -> None:
    """过滤后覆盖全量 speech SCP 的 sidecar 额外 UID 应被忽略。"""
    sidecar = tmp_path / "speech_lengths.txt"
    sidecar.write_text(
        "spk_dns_16k 32000\n"
        "spk_dns_8k 16000\n"
        "spk_lib_48k 96000\n"
        "spk_vctk_16k 32000\n"
        "spk_other_16k 32000\n",
        encoding="utf-8",
    )
    dataset = _build_dataset(
        tmp_path,
        speech_selected_sets=["libritts"],
        speech_length_file=str(sidecar),
    )

    assert set(dataset.source_length) == {"spk_lib_48k"}
    assert set(dataset.speech_flt) == {"spk_lib_48k"}
    assert dataset.samplerates == [48000]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (("dns5_fullband",), TypeError),
        ({"dns5_fullband": 1}, TypeError),
        ([123], TypeError),
        ([""], ValueError),
        ([" dns5_fullband"], ValueError),
        (["dns5_fullband", "dns5_fullband"], ValueError),
    ],
)
def test_rejects_invalid_speech_selected_sets(tmp_path: Path, value, error) -> None:
    """非 list、非 str、空白串与重复项必须在构造边界 fail-fast。"""
    with pytest.raises(error, match="speech_selected_sets"):
        DynamicMixingDataset(
            speech_scp=str(tmp_path / "missing.scp"),
            noise_scp=str(tmp_path / "missing.scp"),
            rir_scp=str(tmp_path / "missing.scp"),
            target_sr=None,
            speech_selected_sets=value,
        )


def test_normalize_selected_sets_returns_frozen_tuple() -> None:
    """归一化保持顺序并返回 tuple；None/[] 归一为 None。"""
    assert _normalize_speech_selected_sets(None) is None
    assert _normalize_speech_selected_sets([]) is None
    assert _normalize_speech_selected_sets(["libritts", "dns5_fullband"]) == (
        "libritts",
        "dns5_fullband",
    )


def test_normalize_all_expands_official_default_sets() -> None:
    """``selected_set: all`` 与官方一致展开为六个默认语料集合。"""
    assert _normalize_speech_selected_sets("all") == (
        "dns5_fullband",
        "vctk",
        "libritts",
        "commonvoice",
        "mls_segments",
        "ears",
    )


def test_filter_helper_syncs_lengths_and_metadata() -> None:
    """过滤 helper 同步裁剪长度映射与 metadata，且不修改输入容器。"""
    speech_source = {
        8000: {"a": "dns5_fullband/a.wav"},
        16000: {"b": "libritts/b.wav", "c": "other/c.wav"},
    }
    speech_uids = {8000: ["a"], 16000: ["b", "c"]}
    speech_flt = {
        "a": "dns5_fullband/a.wav",
        "b": "libritts/b.wav",
        "c": "other/c.wav",
    }
    sp_lens = {"a": 8000, "b": 16000, "c": 16000}
    metadata = {"a": {"speaker_id": "s0"}, "b": {"speaker_id": "s1"}}

    (
        filtered_source,
        filtered_uids,
        filtered_flt,
        filtered_lens,
        filtered_metadata,
    ) = _filter_speech_by_selected_sets(
        speech_source,
        speech_uids,
        speech_flt,
        sp_lens,
        metadata,
        ("dns5_fullband", "libritts"),
    )

    assert dict(filtered_source) == {
        8000: {"a": "dns5_fullband/a.wav"},
        16000: {"b": "libritts/b.wav"},
    }
    assert filtered_uids == {8000: ["a"], 16000: ["b"]}
    assert filtered_flt == {"a": "dns5_fullband/a.wav", "b": "libritts/b.wav"}
    assert filtered_lens == {"a": 8000, "b": 16000}
    assert filtered_metadata == {
        "a": {"speaker_id": "s0"},
        "b": {"speaker_id": "s1"},
    }
    # 输入容器保持完整，避免污染 read_source_scp 的缓存对象。
    assert set(speech_source) == {8000, 16000}
    assert speech_uids == {8000: ["a"], 16000: ["b", "c"]}
    assert set(speech_flt) == {"a", "b", "c"}
    assert set(sp_lens) == {"a", "b", "c"}
    assert set(metadata) == {"a", "b"}


def test_filter_helper_returns_none_lengths() -> None:
    """sp_lens=None（无长度 sidecar）时过滤结果保持 None。"""
    filtered = _filter_speech_by_selected_sets(
        {16000: {"a": "libritts/a.wav"}},
        {16000: ["a"]},
        {"a": "libritts/a.wav"},
        None,
        {},
        ("libritts",),
    )
    assert filtered[3] is None


def test_validation_can_override_speech_selected_sets() -> None:
    """inherit 合并时 validation 的标签列表覆盖 train。"""
    from core.data.dataset_build import merge_validation_section

    merged = merge_validation_section(
        {"speech_scp": "train.scp", "speech_selected_sets": ["dns5_fullband"]},
        {"inherit_simulation": True, "speech_selected_sets": ["libritts"]},
    )
    assert merged["speech_selected_sets"] == ["libritts"]


def test_builder_forwards_selected_sets() -> None:
    """train 标签列表进入 Dataset kwargs；val inherit 时可用 validation 值覆盖。"""
    from tasks.se.data import build_dataset

    cfg = {
        "data": {
            "mode": "dynamic",
            "sample_rate": 16000,
            "train": {
                "speech_scp": "dummy.scp",
                "noise_scp": "dummy.scp",
                "speech_selected_sets": ["dns5_fullband", "libritts"],
                "simulation_cfg": {},
            },
            "validation": {
                "mode": "dynamic",
                "inherit_simulation": True,
                "speech_selected_sets": ["vctk"],
            },
        },
        "meta": {"seed": 0},
    }
    with patch("tasks.se.data.DynamicMixingDataset") as mock_ds:
        build_dataset(cfg["data"], mode="train")
        assert mock_ds.call_args.kwargs["speech_selected_sets"] == [
            "dns5_fullband",
            "libritts",
        ]
        build_dataset(cfg["data"], mode="val", meta=cfg["meta"])
        assert mock_ds.call_args.kwargs["speech_selected_sets"] == ["vctk"]
