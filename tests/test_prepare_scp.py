"""Tests for tools/prepare_data_tools/scp/prepare_scp.py."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.resource_paths import resolve_manifest_resource_path  # noqa: E402
from tools.prepare_data_tools.scp.prepare_scp import (  # noqa: E402
    PrepareScpError,
    build_entries,
    main,
    split_entries_random,
)
from tools.prepare_data_tools.scp.prepare_testset_scps import (  # noqa: E402
    main as prepare_testset_main,
)
from tools.shared.scp_io import (  # noqa: E402
    parse_scp_fields,
    read_scp_dict,
    serialize_manifest_relative_path,
    write_scp_entries,
)


def _write_wavs(dir_path: Path, prefix: str, n: int, fs: int = 16000) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sf.write(
            dir_path / f"{prefix}_{i}.wav",
            np.random.randn(fs // 2).astype(np.float32),
            fs,
        )


def _write_rir_pkl(dir_path: Path, *, fs: int | None = 16000) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "rir.pkl"
    payload = {"rir": np.ones(32, dtype=np.float32)}
    if fs is not None:
        payload["fs"] = fs
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_manifest_relative_path_serialization_keeps_absolute_compatibility(tmp_path):
    manifest = tmp_path / "indexes" / "train.scp"
    audio = tmp_path / "audio" / "clip with space.wav"
    audio.parent.mkdir()
    audio.touch()

    relative = serialize_manifest_relative_path(str(audio), manifest)

    assert relative == "../audio/clip with space.wav"
    assert Path(resolve_manifest_resource_path(relative, manifest)) == audio.resolve()
    assert serialize_manifest_relative_path(str(audio), manifest) == relative
    assert serialize_manifest_relative_path("waveforms:train/clip.wav", manifest) == (
        "waveforms:train/clip.wav"
    )


def _run(argv: list[str]) -> int:
    old = sys.argv
    sys.argv = ["prepare_scp.py", *argv]
    try:
        return main()
    finally:
        sys.argv = old


def _run_expect_error(argv: list[str]) -> str:
    with pytest.raises(SystemExit) as ei:
        _run(argv)
    assert ei.value.code != 0
    return str(ei.value.code)


def test_default_split_all_resources(tmp_path):
    sp, nz = tmp_path / "sp", tmp_path / "nz"
    _write_wavs(sp, "s", 10)
    _write_wavs(nz, "n", 10)
    out = tmp_path / "out"
    assert _run(["--speech", str(sp), "--noise", str(nz), "-o", str(out)]) == 0

    # 默认 val_ratio=0.05：所有资源都产出 train_/val_ 两份
    assert sorted(p.name for p in out.iterdir()) == [
        "train_noise_sources.scp",
        "train_speech_sources.scp",
        "val_noise_sources.scp",
        "val_speech_sources.scp",
    ]
    assert len(read_scp_dict(out / "train_speech_sources.scp")) == 9
    assert len(read_scp_dict(out / "val_speech_sources.scp")) == 1
    assert len(read_scp_dict(out / "train_noise_sources.scp")) == 9
    assert len(read_scp_dict(out / "val_noise_sources.scp")) == 1
    for manifest in out.glob("*.scp"):
        for path in read_scp_dict(manifest).values():
            assert not Path(path).is_absolute()
            assert Path(resolve_manifest_resource_path(path, manifest)).is_file()


def test_val_ratio_zero_keeps_legacy_names(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 5)
    out = tmp_path / "out"
    assert _run(["--speech", str(sp), "-o", str(out), "--val_ratio", "0"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["speech_sources.scp"]
    assert len(read_scp_dict(out / "speech_sources.scp")) == 5


def test_split_reproducible(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 10)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    argv = ["--speech", str(sp), "-o", str(out1), "--val_ratio", "0.3"]
    assert _run(argv) == 0
    assert _run([a.replace(str(out1), str(out2)) for a in argv]) == 0
    a = (out1 / "train_speech_sources.scp").read_text()
    b = (out2 / "train_speech_sources.scp").read_text()
    assert a == b  # 相同 seed 完全一致


def test_speech_explicit_split_excludes_random(tmp_path):
    tr, va = tmp_path / "tr", tmp_path / "va"
    _write_wavs(tr, "t", 4)
    _write_wavs(va, "v", 2)
    out = tmp_path / "out"
    assert _run(["--speech_train", str(tr), "--speech_val", str(va), "-o", str(out)]) == 0
    assert len(read_scp_dict(out / "train_speech_sources.scp")) == 4
    assert len(read_scp_dict(out / "val_speech_sources.scp")) == 2


def test_speech_and_speech_val_mutex(tmp_path):
    sp, va = tmp_path / "sp", tmp_path / "va"
    _write_wavs(sp, "s", 3)
    _write_wavs(va, "v", 2)
    out = tmp_path / "out"
    assert _run_expect_error(["--speech", str(sp), "--speech_val", str(va), "-o", str(out)]) != "0"


def test_val_ratio_out_of_range(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 3)
    out = tmp_path / "out"
    assert _run_expect_error(["--speech", str(sp), "-o", str(out), "--val_ratio", "1.0"]) != "0"
    assert _run_expect_error(["--speech", str(sp), "-o", str(out), "--val_ratio", "-0.1"]) != "0"
    assert _run_expect_error(["--speech", str(sp), "-o", str(out), "--val_ratio", "nan"]) != "0"


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--workers", "0"),
        ("--sample_rate", "0"),
        ("--validate_max_peak", "inf"),
        ("--validate_min_mean_abs", "-0.1"),
        ("--validate_dc_offset_threshold", "nan"),
    ],
)
def test_invalid_numeric_options_are_rejected(tmp_path, flag, value):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 1)
    out = tmp_path / "out"
    assert _run_expect_error(["--speech", str(sp), "-o", str(out), flag, value]) != "0"
    assert not out.exists()


def test_batch_failure_publishes_no_partial_indexes(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 3)
    missing_noise = tmp_path / "missing_noise"
    out = tmp_path / "out"

    assert (
        _run(
            [
                "--speech",
                str(sp),
                "--noise",
                str(missing_noise),
                "-o",
                str(out),
                "--val_ratio",
                "0",
            ]
        )
        == 1
    )
    assert not out.exists()


def test_uid_collision_raises(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 1)
    # 同目录同名不同扩展名 -> uid 碰撞必须报错而不是静默覆盖
    sf.write(sp / "s_0.flac", np.random.randn(8000).astype(np.float32), 16000)
    with pytest.raises(PrepareScpError, match="Duplicate uids"):
        build_entries([sp], workers=2)


def test_empty_after_validation_raises(tmp_path):
    # 全部音频为静音 -> 校验后零条目必须报错，不写空 scp
    sp = tmp_path / "sp"
    sp.mkdir()
    sf.write(sp / "silent.wav", np.zeros(8000, dtype=np.float32), 16000)
    out = tmp_path / "out"
    assert _run(["--speech", str(sp), "-o", str(out), "--val_ratio", "0", "--validate_audio"]) == 1
    assert not (out / "speech_sources.scp").exists()


def test_validation_disabled_by_default_keeps_silent_audio(tmp_path):
    # 默认不做全解码校验：静音文件也保留（读取失败仍按条目级跳过/报错路径处理）
    sp = tmp_path / "sp"
    sp.mkdir()
    sf.write(sp / "silent.wav", np.zeros(8000, dtype=np.float32), 16000)
    out = tmp_path / "out"
    assert _run(["--speech", str(sp), "-o", str(out), "--val_ratio", "0"]) == 0
    assert (out / "speech_sources.scp").exists()


def test_split_entries_random_small_set():
    entries = [(f"u{i}", 16000, 16000, f"/p/{i}.wav") for i in range(30)]
    tr, va = split_entries_random(entries, 0.05, 42)
    assert len(va) == 1  # max(1, int(30*0.05)) = 1
    assert len(tr) == 29
    assert not set(tr) & set(va)


def test_extra_name_conflict_rejected(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 3)
    out = tmp_path / "out"
    assert (
        _run_expect_error(["--extra", "noise_sources=/tmp/x", "--speech", str(sp), "-o", str(out)])
        != "0"
    )
    # train_/val_ 前缀保留给划分输出，--extra 不得占用
    assert (
        _run_expect_error(["--extra", "train_myres=/tmp/x", "--speech", str(sp), "-o", str(out)])
        != "0"
    )
    # --music 成内置资源后，--extra 不得再用 music_sources 名
    assert (
        _run_expect_error(["--extra", "music_sources=/tmp/x", "--speech", str(sp), "-o", str(out)])
        != "0"
    )


@pytest.mark.parametrize(
    "extra",
    [
        "missing_equals",
        "=/tmp/source",
        "name=",
        "name=/tmp/a,,/tmp/b",
        "../escape=/tmp/source",
        "sub/name=/tmp/source",
        r"sub\name=/tmp/source",
        ".=/tmp/source",
        "..=/tmp/source",
        "bad\nname=/tmp/source",
    ],
)
def test_extra_requires_safe_name_and_nonempty_paths(tmp_path, extra):
    out = tmp_path / "out"
    _run_expect_error(["--extra", extra, "-o", str(out), "--val_ratio", "0"])
    assert not out.exists()
    assert not (tmp_path / "escape.scp").exists()


def test_extra_duplicate_name_rejected(tmp_path):
    source = tmp_path / "source"
    _write_wavs(source, "s", 1)
    out = tmp_path / "out"
    _run_expect_error(
        [
            "--extra",
            f"custom={source}",
            "--extra",
            f"custom={source}",
            "-o",
            str(out),
            "--val_ratio",
            "0",
        ]
    )
    assert not out.exists()


def test_validation_summary_uses_shared_real_stats(tmp_path, caplog):
    source = tmp_path / "source"
    _write_wavs(source, "valid", 1)
    sf.write(source / "silent.wav", np.zeros(8000, dtype=np.float32), 16000)
    out = tmp_path / "out"

    with caplog.at_level("INFO", logger="prepare_scp"):
        assert _run(["--speech", str(source), "-o", str(out), "--val_ratio", "0", "--validate_audio"]) == 0

    summaries = [record.message for record in caplog.records if "Validation summary" in record.message]
    assert summaries == [
        "Validation summary: total=2, kept=1, skipped=1; too_silent=1"
    ]


@pytest.mark.parametrize("bad", ["bad\rvalue", "bad\nvalue", "bad\0value"])
@pytest.mark.parametrize("field", ["uid", "path"])
def test_scp_io_rejects_record_control_characters(tmp_path, bad, field):
    target = tmp_path / "target.scp"
    target.write_text("old\n", encoding="utf-8")
    uid = bad if field == "uid" else "uid"
    path = bad if field == "path" else "/audio.wav"

    with pytest.raises(ValueError, match="CR|LF|NUL"):
        write_scp_entries([(uid, path)], target)
    assert target.read_text(encoding="utf-8") == "old\n"

    with pytest.raises(ValueError, match="CR|LF|NUL"):
        parse_scp_fields(f"{uid} {path}", "input.scp", 1, allowed_columns=(2,))


def test_scp_io_rejects_controls_in_four_col_and_list(tmp_path):
    with pytest.raises(ValueError, match="CR|LF|NUL"):
        write_scp_entries(
            [("uid", 16000, 10, "bad\npath.wav")],
            tmp_path / "four.scp",
            four_col=True,
        )
    with pytest.raises(ValueError, match="CR|LF|NUL"):
        write_scp_entries(
            [("uid", "bad\rpath.wav")],
            tmp_path / "paths.list",
            fmt="list",
        )
    assert not (tmp_path / "four.scp").exists()
    assert not (tmp_path / "paths.list").exists()


def test_explicit_aec_sim_missing_root_fails(tmp_path):
    out = tmp_path / "out"
    assert prepare_testset_main(
        [
            "--datasets",
            "aec_sim",
            "--aec_sim_testset_root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(out),
        ]
    ) == 1
    assert not list(out.rglob("*.scp"))


def test_explicit_aec_sim_zero_valid_entries_fails_without_publish(tmp_path):
    source = tmp_path / "aec_sim"
    (source / "prob_target_only" / "sample_000").mkdir(parents=True)
    out = tmp_path / "out"

    assert prepare_testset_main(
        [
            "--datasets",
            "aec_sim",
            "--aec_sim_testset_root",
            str(source),
            "--output-dir",
            str(out),
        ]
    ) == 1
    assert not list(out.rglob("*.scp"))


def test_explicit_aec_sim_publishes_aligned_complete_samples(tmp_path):
    source = tmp_path / "aec_sim"
    sample = source / "prob_target_only" / "sample_000"
    sample.mkdir(parents=True)
    for wav_name in ("mic", "nearend", "farend", "noisy_se"):
        (sample / f"{wav_name}.wav").touch()
    out = tmp_path / "out"

    assert prepare_testset_main(
        [
            "--datasets",
            "aec_sim",
            "--aec_sim_testset_root",
            str(source),
            "--output-dir",
            str(out),
        ]
    ) == 0
    scps = sorted(out.rglob("*.scp"))
    assert [path.name for path in scps] == ["farend.scp", "mic.scp", "nearend.scp", "noisy_se.scp"]
    assert {path.read_text(encoding="utf-8").split()[0] for path in scps} == {"sample_000"}


def test_music_first_class_resource(tmp_path):
    sp, mu = tmp_path / "sp", tmp_path / "mu"
    _write_wavs(sp, "s", 6)
    _write_wavs(mu, "m", 8)
    out = tmp_path / "out"
    assert _run(["--speech", str(sp), "--music", str(mu), "-o", str(out)]) == 0

    assert sorted(p.name for p in out.iterdir()) == [
        "train_music_sources.scp",
        "train_speech_sources.scp",
        "val_music_sources.scp",
        "val_speech_sources.scp",
    ]
    assert len(read_scp_dict(out / "train_music_sources.scp")) == 7
    assert len(read_scp_dict(out / "val_music_sources.scp")) == 1


def test_music_val_ratio_zero_keeps_canonical_name(tmp_path):
    mu = tmp_path / "mu"
    _write_wavs(mu, "m", 5)
    out = tmp_path / "out"
    assert _run(["--music", str(mu), "-o", str(out), "--val_ratio", "0"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["music_sources.scp"]


def test_strict_sample_rate_fails_on_mismatch(tmp_path):
    """--strict_sample_rate：声明采样率与真实不一致时直接失败，不写残缺索引。"""
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 2, fs=8000)
    out = tmp_path / "out"
    assert _run([
        "--speech", str(sp), "-o", str(out), "--val_ratio", "0",
        "--sample_rate", "16000", "--strict_sample_rate",
    ]) == 1
    assert not list(out.glob("*.scp"))


def test_sample_rate_mismatch_default_counts_without_failing(tmp_path, caplog):
    """默认（非 strict）：声明与真实不一致仅计入 fs_mismatch，索引照常生成。"""
    import logging

    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 3, fs=8000)
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO):
        assert _run([
            "--speech", str(sp), "-o", str(out), "--val_ratio", "0",
            "--sample_rate", "16000",
        ]) == 0
    lines = (out / "speech_sources.scp").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # 声明列仍为 --sample_rate 覆盖值（16000），真实 8000 仅计入 fs_mismatch
    assert all(line.split()[1] == "16000" for line in lines)
    assert "fs_mismatch=3" in caplog.text


def test_rir_pkl_strict_sample_rate_fails_on_metadata_mismatch(tmp_path):
    rir = tmp_path / "rir"
    _write_rir_pkl(rir, fs=8000)
    out = tmp_path / "out"

    assert _run([
        "--rir", str(rir), "-o", str(out), "--val_ratio", "0",
        "--sample_rate", "16000", "--strict_sample_rate",
    ]) == 1
    assert not list(out.glob("*.scp"))


def test_rir_pkl_sample_rate_mismatch_counts_and_uses_declared_rate(tmp_path, caplog):
    import logging

    rir = tmp_path / "rir"
    _write_rir_pkl(rir, fs=8000)
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO):
        assert _run([
            "--rir", str(rir), "-o", str(out), "--val_ratio", "0",
            "--sample_rate", "16000",
        ]) == 0

    line = (out / "rirs.scp").read_text(encoding="utf-8").strip()
    assert line.split()[1] == "16000"
    assert "fs_mismatch=1" in caplog.text


def test_rir_pkl_without_declared_rate_uses_metadata_rate(tmp_path):
    rir = tmp_path / "rir"
    _write_rir_pkl(rir, fs=8000)
    out = tmp_path / "out"

    assert _run([
        "--rir", str(rir), "-o", str(out), "--val_ratio", "0",
    ]) == 0
    assert (out / "rirs.scp").read_text(encoding="utf-8").split()[1] == "8000"


def test_rir_pkl_without_fs_metadata_uses_declared_or_legacy_fallback_rate(tmp_path, caplog):
    import logging

    rir = tmp_path / "rir"
    _write_rir_pkl(rir, fs=None)
    declared_out = tmp_path / "declared_out"
    fallback_out = tmp_path / "fallback_out"

    with caplog.at_level(logging.INFO):
        assert _run([
            "--rir", str(rir), "-o", str(declared_out), "--val_ratio", "0",
            "--sample_rate", "8000", "--strict_sample_rate",
        ]) == 0
    assert (declared_out / "rirs.scp").read_text(encoding="utf-8").split()[1] == "8000"
    assert "fs_mismatch" not in caplog.text

    assert _run([
        "--rir", str(rir), "-o", str(fallback_out), "--val_ratio", "0",
    ]) == 0
    assert (fallback_out / "rirs.scp").read_text(encoding="utf-8").split()[1] == "16000"


def test_strict_sample_rate_requires_sample_rate(tmp_path):
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 1)
    out = tmp_path / "out"
    _run_expect_error(["--speech", str(sp), "-o", str(out), "--strict_sample_rate"])


def test_removed_no_validate_audio_flag_is_rejected(tmp_path):
    """已移除的 --no_validate_audio 必须在解析期显式报错，而不是静默 no-op。"""
    sp = tmp_path / "sp"
    _write_wavs(sp, "s", 1)
    out = tmp_path / "out"
    _run_expect_error([
        "--speech", str(sp), "-o", str(out), "--val_ratio", "0", "--no_validate_audio",
    ])
    assert not out.exists()


pytestmark = [pytest.mark.unit]
