"""OfflineDataset target_sr 输出长度回归测试。"""

import numpy as np
import pytest
import soundfile as sf

from core.data.offline_dataset import OfflineDataset, _read_scp_four_col

pytestmark = [pytest.mark.unit]


def _build_dataset(
    tmp_path, mode, source_sr, frames, noisy_frames=None, noisy_sr=None, **kwargs
):
    assert mode == "scp"
    paths = [tmp_path / f"{kind}.wav" for kind in ("clean", "noisy")]
    frame_counts = [frames, frames if noisy_frames is None else noisy_frames]
    sample_rates = [source_sr, source_sr if noisy_sr is None else noisy_sr]
    manifests = [tmp_path / f"{kind}.scp" for kind in ("clean", "noisy")]
    for manifest, path, frame_count, sample_rate in zip(
        manifests, paths, frame_counts, sample_rates
    ):
        sf.write(path, np.zeros(frame_count, dtype=np.float32), sample_rate, subtype="FLOAT")
        manifest.write_text(
            f"utt {sample_rate} {frame_count} {path}\n", encoding="utf-8"
        )
    target_sr = kwargs.pop("target_sr", 16000)
    return OfflineDataset(
        clean_scp=str(manifests[0]),
        noisy_scp=str(manifests[1]),
        target_sr=target_sr,
        **kwargs,
    )


def _assert_length(dataset, expected):
    clean, noisy, sample_rate, item_length = dataset[0]
    assert dataset.get_source_length() == [expected]
    assert sample_rate == 16000
    assert item_length == clean.shape[-1] == noisy.shape[-1] == expected


@pytest.mark.parametrize(
    ("source_sr", "frames", "expected"),
    [(48000, 48001, 16001), (8000, 8001, 16002)],
)
def test_source_length_uses_target_sr_domain(tmp_path, source_sr, frames, expected):
    _assert_length(_build_dataset(tmp_path, "scp", source_sr, frames), expected)


def test_target_sr_length_applies_cap_after_upsampling(tmp_path):
    dataset = _build_dataset(
        tmp_path, "scp", 8000, 4001, noisy_frames=2000, max_duration=6000
    )
    _assert_length(dataset, 6000)


def test_noisy_shorter_controls_paired_output_length(tmp_path):
    dataset = _build_dataset(tmp_path, "scp", 48000, 48001, noisy_frames=24001)
    _assert_length(dataset, 8001)


def test_noisy_length_follows_clean_sr_then_target_sr_resampling(tmp_path):
    dataset = _build_dataset(
        tmp_path, "scp", 11025, 100,
        noisy_frames=1, noisy_sr=8000, strict_load=False,
    )
    _assert_length(dataset, 3)


def test_target_sr_length_applies_repeat_after_conversion(tmp_path):
    dataset = _build_dataset(
        tmp_path, "scp", 48000, 6001,
        max_duration=4000, repeat_to_max_duration=True,
    )
    _assert_length(dataset, 4000)


def test_scp_filter_uses_manifest_samplerate(tmp_path):
    dataset = _build_dataset(
        tmp_path, "scp", 48000, 24000,
        filter_by_duration=True,
        filter_min_duration=0.4,
        filter_max_duration=0.6,
    )
    assert len(dataset) == 1
    _assert_length(dataset, 8000)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("utt audio.wav", "非法列数 2"),
        ("utt 16000 audio.wav", "非法列数 3"),
        ("utt 16000 10 audio.wav extra", "非法列数 5"),
        ("utt 0 10 audio.wav", "fs 必须是正整数"),
        ("utt 16000 0 audio.wav", "length 必须是正整数"),
    ],
)
def test_four_column_scp_rejects_malformed_records(tmp_path, line, message):
    manifest = tmp_path / "invalid.scp"
    manifest.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _read_scp_four_col(manifest)


def test_four_column_scp_rejects_duplicate_uid(tmp_path):
    manifest = tmp_path / "duplicate.scp"
    manifest.write_text(
        "utt 16000 10 first.wav\nutt 16000 20 second.wav\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复 uid"):
        _read_scp_four_col(manifest)


def test_four_column_scp_accepts_quoted_path_with_spaces(tmp_path):
    manifest = tmp_path / "quoted.scp"
    manifest.write_text("utt 16000 10 'audio with spaces.wav'\n", encoding="utf-8")

    paths, sample_rates, lengths = _read_scp_four_col(manifest)

    assert paths == {"utt": str(tmp_path / "audio with spaces.wav")}
    assert sample_rates == {"utt": 16000}
    assert lengths == {"utt": 10}


@pytest.mark.parametrize("track", ["clean", "noisy"])
def test_loaded_payload_must_match_each_track_metadata(tmp_path, track):
    dataset = _build_dataset(tmp_path, "scp", 16000, 100)
    metadata = dataset.lengths if track == "clean" else dataset.noisy_lengths
    metadata["utt"] += 1

    with pytest.raises(RuntimeError, match=rf"track={track}.*metadata mismatch"):
        dataset[0]


def test_scp_source_length_uses_both_track_metadata_without_audio_probe(
    tmp_path, monkeypatch
):
    dataset = _build_dataset(
        tmp_path, "scp", 16000, 100, noisy_frames=80, noisy_sr=8000
    )

    def reject_audio_probe(*_args, **_kwargs):
        raise AssertionError("四列 SCP 长度查询不得探测音频 payload")

    monkeypatch.setattr("core.data.offline_dataset.sf.info", reject_audio_probe)
    assert dataset.get_source_length() == [100]


@pytest.mark.parametrize("retired_key", ["clean_list", "noisy_list", "clean_json", "noisy_json", "list_file_sample_rate"])
def test_offline_constructor_rejects_retired_source_keywords(retired_key):
    with pytest.raises(TypeError, match=retired_key):
        OfflineDataset(**{retired_key: "retired"})


@pytest.mark.parametrize("value", [0, -1, True, False, 16000.5, "16000"])
def test_offline_constructor_requires_exact_positive_integer_target_sr(
    tmp_path, value
):
    with pytest.raises(ValueError, match="target_sr must be a positive integer"):
        _build_dataset(tmp_path, "scp", 16000, 100, target_sr=value)


def test_se_source_length_and_srs_are_cached(tmp_path, monkeypatch):
    dataset = _build_dataset(tmp_path, "scp", 16000, 100)
    assert dataset.get_source_length() == [100]
    assert dataset.get_srs() == [16000]
    monkeypatch.setattr(
        dataset,
        "_source_length",
        lambda _index: (_ for _ in ()).throw(AssertionError("length cache miss")),
    )
    monkeypatch.setattr(
        dataset,
        "_output_sr",
        lambda _index: (_ for _ in ()).throw(AssertionError("SR cache miss")),
    )
    assert dataset.get_source_length() == [100]
    assert dataset.get_srs() == [16000]


def test_se_set_epoch_makes_random_crop_reproducible(tmp_path):
    ramp = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    for kind in ("clean", "noisy"):
        sf.write(tmp_path / f"{kind}.wav", ramp, 16000, subtype="FLOAT")
        (tmp_path / f"{kind}.scp").write_text(
            f"utt 16000 1000 {tmp_path / f'{kind}.wav'}\n", encoding="utf-8"
        )
    dataset = OfflineDataset(
        clean_scp=tmp_path / "clean.scp",
        noisy_scp=tmp_path / "noisy.scp",
        max_duration=200,
        seed=5,
    )

    dataset.set_epoch(3, base_seed=17)
    first = dataset[0][0]
    dataset.set_epoch(3, base_seed=17)
    second = dataset[0][0]

    assert pytest.importorskip("torch").equal(first, second)


def test_se_same_sr_crop_uses_equivalent_synchronized_window_reads(
    tmp_path, monkeypatch
):
    from core.data import offline_dataset as offline_module
    from core.data.offline_multitrack import load_audio_window as real_window_read

    ramp = np.linspace(-0.9, 0.9, 1000, dtype=np.float32)
    for kind in ("clean", "noisy"):
        sf.write(tmp_path / f"{kind}.wav", ramp, 16000, subtype="PCM_16")
        (tmp_path / f"{kind}.scp").write_text(
            f"utt 16000 1000 {tmp_path / f'{kind}.wav'}\n", encoding="utf-8"
        )
    decoded, _ = sf.read(tmp_path / "clean.wav", dtype="float32")
    dataset = OfflineDataset(
        clean_scp=tmp_path / "clean.scp",
        noisy_scp=tmp_path / "noisy.scp",
        max_duration=200,
        seed=5,
    )
    dataset.set_epoch(3, base_seed=17)
    expected_start = dataset.sample_rng(0).randint(0, 800)
    calls = []

    def tracked_window_read(path, start, n_frames):
        calls.append((str(path), start, n_frames))
        return real_window_read(path, start, n_frames)

    monkeypatch.setattr(offline_module, "load_audio_window", tracked_window_read)
    monkeypatch.setattr(
        offline_module,
        "_load_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("可证明等价的同 SR 裁切不得整段读取")
        ),
    )

    clean, noisy, sr, length = dataset[0]

    assert calls == [
        (str(tmp_path / "clean.wav"), expected_start, 200),
        (str(tmp_path / "noisy.wav"), expected_start, 200),
    ]
    expected = pytest.importorskip("torch").from_numpy(
        decoded[expected_start : expected_start + 200]
    ).unsqueeze(0)
    assert pytest.importorskip("torch").equal(clean, expected)
    assert pytest.importorskip("torch").equal(noisy, expected)
    assert (sr, length) == (16000, 200)


def test_se_resampling_path_falls_back_to_full_read(tmp_path, monkeypatch):
    from core.data import offline_dataset as offline_module

    dataset = _build_dataset(
        tmp_path,
        "scp",
        8000,
        1000,
        target_sr=16000,
        max_duration=200,
    )
    monkeypatch.setattr(
        offline_module,
        "load_audio_window",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("重采样路径不得启用 source-domain window read")
        ),
    )

    _assert_length(dataset, 200)


def test_offline_scp_rejects_zero_length_metadata(tmp_path):
    manifest = tmp_path / "invalid.scp"
    manifest.write_text("utt 16000 0 noisy.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="length 必须是正整数"):
        _read_scp_four_col(manifest)


def test_lenient_single_sample_metadata_failure_returns_bucket_placeholder(tmp_path):
    dataset = _build_dataset(
        tmp_path, "scp", 8000, 100, strict_load=False,
        target_sr=None, placeholder_num_samples=32,
    )
    dataset.lengths["utt"] += 1

    clean, noisy, sr, length = dataset[0]

    assert (sr, length) == (8000, 32)
    assert clean.shape == noisy.shape == (1, 32)
    assert float(clean.abs().sum() + noisy.abs().sum()) == 0.0


def test_lenient_fallback_only_uses_same_output_sr(tmp_path, monkeypatch):
    dataset = _build_dataset(
        tmp_path, "scp", 16000, 100, strict_load=False, target_sr=None
    )
    dataset.uids = ["failed", "same", "other"]
    dataset.clean = dataset.noisy = {uid: "unused.wav" for uid in dataset.uids}
    dataset.utt2fs = {"failed": 8000, "same": 8000, "other": 16000}
    dataset.lengths = dataset.noisy_lengths = {uid: 4 for uid in dataset.uids}
    dataset.noisy_utt2fs = dict(dataset.utt2fs)
    dataset._full_len = 3
    dataset._filtered_indices = [0, 1, 2]
    visited = []

    def fake_getitem(real_index):
        visited.append(real_index)
        if real_index == 0:
            raise OSError("failed sample")
        sr = dataset.utt2fs[dataset.uids[real_index]]
        wav = pytest.importorskip("torch").full((1, 4), float(real_index))
        return wav, wav, sr, 4

    monkeypatch.setattr(dataset, "_getitem_impl", fake_getitem)

    item = dataset[0]

    assert item[2] == 8000
    assert visited == [0, 1]


def test_lenient_fallback_without_same_output_sr_returns_placeholder(
    tmp_path, monkeypatch
):
    dataset = _build_dataset(
        tmp_path, "scp", 16000, 100, strict_load=False,
        target_sr=None, placeholder_num_samples=24,
    )
    dataset.uids = ["failed", "other"]
    dataset.clean = dataset.noisy = {uid: "unused.wav" for uid in dataset.uids}
    dataset.utt2fs = {"failed": 8000, "other": 16000}
    dataset.lengths = dataset.noisy_lengths = {uid: 4 for uid in dataset.uids}
    dataset.noisy_utt2fs = dict(dataset.utt2fs)
    dataset._full_len = 2
    dataset._filtered_indices = [0, 1]
    visited = []

    def fake_getitem(real_index):
        visited.append(real_index)
        raise OSError("failed sample")

    monkeypatch.setattr(dataset, "_getitem_impl", fake_getitem)

    clean, noisy, sr, length = dataset[0]

    assert (sr, length) == (8000, 24)
    assert clean.shape == noisy.shape == (1, 24)
    assert visited == [0]
