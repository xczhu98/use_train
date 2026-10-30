"""Tests for tasks.aec.data.offline_dataset (offline_aec mode)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.aec.data.batch import AECBatch  # noqa: E402
from tasks.aec.data.collate import aec_collate_fn  # noqa: E402
from tasks.aec.data.offline_dataset import (  # noqa: E402
    AECOfflineDataset,
    build_offline_aec_dataset,
)
from core.trainer.entry import detect_task  # noqa: E402


def _write_wav(path: Path, n: int, fs: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (np.random.randn(n) * 0.1).astype(np.float32), fs)


def _make_triplets(root: Path, uids: list[str], n: int = 8000,
                   with_error: bool = False) -> dict[str, Path]:
    """生成 mic/farend/nearend（可选 error）四列 scp，返回 scp 路径。"""
    tracks = ["mic", "farend", "nearend"] + (["error"] if with_error else [])
    lines: dict[str, list[str]] = {t: [] for t in tracks}
    for uid in uids:
        for t in tracks:
            # farend 故意短 100 采样点，验证对齐补零逻辑
            length = n - 100 if t in ("farend", "error") else n
            p = root / f"{uid}_{t}.wav"
            _write_wav(p, length)
            lines[t].append(f"{uid} 16000 {length} {p}")
    scps = {}
    for t in tracks:
        scp = root / f"{t}.scp"
        scp.write_text("\n".join(lines[t]) + "\n")
        scps[t] = scp
    return scps


def test_pairing_mismatch_raises(tmp_path):
    scps = _make_triplets(tmp_path, ["a", "b"])
    # farend 缺 uid=b
    (tmp_path / "farend.scp").write_text(
        "\n".join(l for l in scps["farend"].read_text().splitlines() if not l.startswith("b ")) + "\n"
    )
    with pytest.raises(ValueError, match="pairing check failed"):
        AECOfflineDataset(
            mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
            nearend_scp=str(scps["nearend"]),
        )


@pytest.mark.parametrize("track", ["mic", "farend", "nearend", "error"])
def test_each_aec_track_requires_four_column_scp(tmp_path, track):
    scps = _make_triplets(tmp_path, ["a"], with_error=track == "error")
    audio_path = tmp_path / f"a_{track}.wav"
    scps[track].write_text(f"a {audio_path}\n", encoding="utf-8")
    kwargs = {
        "mic_scp": str(scps["mic"]),
        "farend_scp": str(scps["farend"]),
        "nearend_scp": str(scps["nearend"]),
    }
    if track == "error":
        kwargs["error_scp"] = str(scps["error"])

    with pytest.raises(ValueError, match="非法列数 2"):
        AECOfflineDataset(**kwargs)


@pytest.mark.parametrize("track", ["mic", "farend", "nearend", "error"])
def test_aec_loaded_payload_must_match_each_track_metadata(tmp_path, track):
    scps = _make_triplets(tmp_path, ["a"], with_error=track == "error")
    fields = scps[track].read_text(encoding="utf-8").split()
    fields[2] = str(int(fields[2]) + 1)
    scps[track].write_text(" ".join(fields) + "\n", encoding="utf-8")
    kwargs = {
        "mic_scp": str(scps["mic"]),
        "farend_scp": str(scps["farend"]),
        "nearend_scp": str(scps["nearend"]),
    }
    if track == "error":
        kwargs["error_scp"] = str(scps["error"])

    dataset = AECOfflineDataset(**kwargs)
    with pytest.raises(RuntimeError, match=rf"track={track}.*metadata mismatch"):
        dataset[0]


def test_aec_source_length_uses_manifest_metadata_without_audio_probe(
    tmp_path, monkeypatch
):
    scps = _make_triplets(tmp_path, ["a"], n=8000)
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
    )

    def reject_audio_probe(*_args, **_kwargs):
        raise AssertionError("四列 SCP 长度查询不得探测音频 payload")

    monkeypatch.setattr(sf, "info", reject_audio_probe)
    assert ds.get_source_length() == [8000]


def test_aec_source_length_is_cached(tmp_path, monkeypatch):
    scps = _make_triplets(tmp_path, ["a"], n=8000)
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
    )
    expected = ds.get_source_length()
    monkeypatch.setattr(
        ds,
        "_source_length",
        lambda _index: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    assert ds.get_source_length() == expected


def test_aec_mixed_sr_contract_is_strict_or_resamples_to_mic(tmp_path):
    scps = _make_triplets(tmp_path, ["a"], n=8000)
    _write_wav(tmp_path / "a_nearend.wav", 4000, fs=8000)
    scps["nearend"].write_text(
        f"a 8000 4000 {tmp_path / 'a_nearend.wav'}\n", encoding="utf-8"
    )

    strict = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), strict_load=True,
    )
    with pytest.raises(RuntimeError, match="sample rate mismatch"):
        strict[0]

    lenient = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), strict_load=False,
    )
    item = lenient[0]
    assert item[3:5] == (16000, 8000)
    assert item[0].shape == item[1].shape == (1, 8000)


def test_seven_tuple_contract_and_collate(tmp_path):
    scps = _make_triplets(tmp_path, ["a", "b"], n=8000)
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
    )
    assert len(ds) == 2
    item = ds[0]
    assert len(item) == 7
    nearend, mic, farend, fs, length, error, error_mask = item
    # 四轨等长（farend 短 100 点被补零），length 为有效长度
    assert mic.shape == nearend.shape == farend.shape == error.shape == (1, 8000)
    assert fs == 16000 and length == 8000
    assert error_mask is False
    assert float(error.abs().sum()) == 0.0
    assert ds.get_srs() == [16000, 16000]
    assert ds.get_source_length() == [8000, 8000]

    batch = aec_collate_fn([ds[0], ds[1]])
    b = batch
    assert isinstance(b, AECBatch)
    assert b.mic.shape[0] == 2
    assert int(b.fs) == 16000
    assert b.error_mask.tolist() == [False, False]


def test_error_scp_sets_error_mask(tmp_path):
    scps = _make_triplets(tmp_path, ["a"], with_error=True)
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), error_scp=str(scps["error"]),
    )
    item = ds[0]
    assert item[6] is True
    assert float(item[5].abs().sum()) > 0.0  # error 为真实残差而非全零
    assert item[5].shape == item[1].shape    # error 与 mic 等长（补零对齐）


@pytest.mark.parametrize("strict_load", [True, False])
def test_optional_error_track_checks_nonfinite_values(
    tmp_path, monkeypatch, strict_load
):
    scps = _make_triplets(tmp_path, ["a"], with_error=True)
    from tasks.aec.data import offline_dataset as offline_module

    real_load_audio = offline_module.load_audio

    def load_with_nan(path):
        wav, sr = real_load_audio(path)
        if str(path).endswith("_error.wav"):
            wav[:, 0] = float("nan")
        return wav, sr

    monkeypatch.setattr(offline_module, "load_audio", load_with_nan)
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), error_scp=str(scps["error"]),
        strict_load=strict_load,
    )

    if strict_load:
        with pytest.raises(RuntimeError, match=r"NaN/Inf in error waveform"):
            ds[0]
    else:
        item = ds[0]
        assert item[6] is True
        assert item[5].isfinite().all()
        assert item[5][0, 0] == 0


def test_lenient_single_sample_metadata_failure_returns_bucket_placeholder(tmp_path):
    scps = _make_triplets(tmp_path, ["a"])
    fields = scps["mic"].read_text(encoding="utf-8").split()
    fields[2] = str(int(fields[2]) + 1)
    scps["mic"].write_text(" ".join(fields) + "\n", encoding="utf-8")
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), strict_load=False,
        placeholder_num_samples=32,
    )

    item = ds[0]

    assert item[3:5] == (16000, 32)
    assert all(float(track.abs().sum()) == 0.0 for track in (*item[:3], item[5]))
    assert item[6] is False


def test_lenient_fallback_only_uses_same_output_sr(tmp_path, monkeypatch):
    scps = _make_triplets(tmp_path, ["failed", "same", "other"])
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), strict_load=False,
    )
    ds.utt2fs.update({"failed": 8000, "same": 8000, "other": 16000})
    visited = []

    def fake_getitem(real_index):
        visited.append(real_index)
        if real_index == 0:
            raise OSError("failed sample")
        sr = ds.utt2fs[ds.uids[real_index]]
        wav = torch.full((1, 4), float(real_index))
        return wav, wav, wav, sr, 4, wav, False

    monkeypatch.setattr(ds, "_getitem_impl", fake_getitem)

    item = ds[0]

    assert item[3] == 8000
    assert visited == [0, 1]


def test_lenient_fallback_without_same_output_sr_returns_placeholder(
    tmp_path, monkeypatch
):
    scps = _make_triplets(tmp_path, ["failed", "other"])
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), strict_load=False,
        placeholder_num_samples=24,
    )
    ds.utt2fs.update({"failed": 8000, "other": 16000})
    visited = []

    def fake_getitem(real_index):
        visited.append(real_index)
        raise OSError("failed sample")

    monkeypatch.setattr(ds, "_getitem_impl", fake_getitem)

    item = ds[0]

    assert item[3:5] == (8000, 24)
    assert visited == [0]


def test_duration_filter(tmp_path):
    # short=0.5s 被过滤，long=2s 保留
    scps = _make_triplets(tmp_path, ["long"], n=32000)
    _write_wav(tmp_path / "short_mic.wav", 8000)
    _write_wav(tmp_path / "short_farend.wav", 8000)
    _write_wav(tmp_path / "short_nearend.wav", 8000)
    for t in ("mic", "farend", "nearend"):
        with open(scps[t], "a") as f:
            f.write(f"short 16000 8000 {tmp_path / ('short_' + t + '.wav')}\n")
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
        filter_by_duration=True, filter_min_duration=1.0, filter_max_duration=5.0,
    )
    assert len(ds) == 1
    assert ds.uids[ds._filtered_indices[0]] == "long"


def test_max_duration_seconds_crops_all_tracks(tmp_path):
    scps = _make_triplets(tmp_path, ["a"], n=32000)  # 2s
    ds = AECOfflineDataset(
        mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]), max_duration_seconds=1.0,
    )
    item = ds[0]
    assert item[1].shape[1] == 16000
    assert item[4] == 16000
    assert ds.get_source_length() == [16000]


def test_aec_same_sr_crop_uses_equivalent_synchronized_window_reads(
    tmp_path, monkeypatch
):
    from tasks.aec.data import offline_dataset as offline_module
    from core.data.offline_multitrack import load_audio_window as real_window_read

    scps = _make_triplets(tmp_path, ["a"], n=32000, with_error=True)
    baseline = AECOfflineDataset(
        mic_scp=str(scps["mic"]),
        farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
        error_scp=str(scps["error"]),
        max_duration=16000,
        seed=5,
    )
    baseline.set_epoch(3, base_seed=17)
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            offline_module,
            "validate_audio_file_metadata",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("强制参照路径整段读取")
            ),
        )
        expected = baseline[0]
    expected_start = baseline.sample_rng(0).randint(0, 16000)
    calls = []

    def tracked_window_read(path, start, n_frames):
        calls.append((str(path), start, n_frames))
        return real_window_read(path, start, n_frames)

    monkeypatch.setattr(offline_module, "load_audio_window", tracked_window_read)
    monkeypatch.setattr(
        offline_module,
        "load_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("可证明等价的同 SR 裁切不得整段读取")
        ),
    )
    windowed = AECOfflineDataset(
        mic_scp=str(scps["mic"]),
        farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
        error_scp=str(scps["error"]),
        max_duration=16000,
        seed=5,
    )
    windowed.set_epoch(3, base_seed=17)

    actual = windowed[0]

    assert calls == [
        (str(tmp_path / f"a_{track}.wav"), expected_start, 16000)
        for track in ("mic", "nearend", "farend", "error")
    ]
    for actual_track, expected_track in zip(
        (*actual[:3], actual[5]), (*expected[:3], expected[5])
    ):
        assert torch.equal(actual_track, expected_track)
    assert actual[3:5] == expected[3:5] == (16000, 16000)
    assert actual[6] is expected[6] is True


def test_aec_resampling_path_falls_back_to_full_read(tmp_path, monkeypatch):
    from tasks.aec.data import offline_dataset as offline_module

    scps = _make_triplets(tmp_path, ["a"], n=8000)
    _write_wav(tmp_path / "a_nearend.wav", 4000, fs=8000)
    scps["nearend"].write_text(
        f"a 8000 4000 {tmp_path / 'a_nearend.wav'}\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        offline_module,
        "load_audio_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("异 SR 路径不得启用 window read")
        ),
        raising=False,
    )
    dataset = AECOfflineDataset(
        mic_scp=str(scps["mic"]),
        farend_scp=str(scps["farend"]),
        nearend_scp=str(scps["nearend"]),
        strict_load=False,
        max_duration=2000,
    )

    assert dataset[0][4] == 2000


@pytest.mark.parametrize("value", [0, -1, True, False, 16000.5, "16000"])
def test_aec_offline_constructor_requires_exact_positive_integer_target_sr(
    tmp_path, value
):
    scps = _make_triplets(tmp_path, ["a"])

    with pytest.raises(ValueError, match="target_sr must be a positive integer"):
        AECOfflineDataset(
            mic_scp=str(scps["mic"]),
            farend_scp=str(scps["farend"]),
            nearend_scp=str(scps["nearend"]),
            target_sr=value,
        )


def test_unexpected_kwargs_rejected(tmp_path):
    scps = _make_triplets(tmp_path, ["a"])
    with pytest.raises(TypeError, match="unexpected kwargs"):
        AECOfflineDataset(
            mic_scp=str(scps["mic"]), farend_scp=str(scps["farend"]),
            nearend_scp=str(scps["nearend"]), typo_key=1,
        )


def test_build_offline_aec_dataset_from_cfg(tmp_path):
    train = _make_triplets(tmp_path / "train", ["a", "b"])
    val = _make_triplets(tmp_path / "val", ["c"])
    cfg = {
        "data": {
            "mode": "offline_aec",
            "sample_rate": 16000,
            "train": {
                "mic_scp": str(train["mic"]), "farend_scp": str(train["farend"]),
                "nearend_scp": str(train["nearend"]),
            },
            "validation": {
                "mic_scp": str(val["mic"]), "farend_scp": str(val["farend"]),
                "nearend_scp": str(val["nearend"]),
            },
        }
    }
    train_ds = build_offline_aec_dataset(cfg, "train")
    val_ds = build_offline_aec_dataset(cfg, "val")
    assert len(train_ds) == 2 and len(val_ds) == 1


def test_detect_task_routes_offline_aec():
    cfg = {
        "data": {"mode": "offline_aec", "task": "joint"},
        "model": {"type": "DeepVQE"},
        "trainer": {"type": "AECTrainer"},
    }
    info = detect_task(cfg)
    assert info.task == "aec"


pytestmark = [pytest.mark.unit]
