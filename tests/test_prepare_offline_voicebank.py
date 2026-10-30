"""Tests for tools/prepare_data_tools/scp/prepare_offline.py and prepare_voicebank.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.resource_paths import resolve_manifest_resource_path  # noqa: E402
from core.utils import file_transaction  # noqa: E402
from tools.prepare_data_tools.scp.prepare_offline import (  # noqa: E402
    main as offline_main,
    scan_dir,
)
from tools.prepare_data_tools.scp.prepare_voicebank import (  # noqa: E402
    main as voicebank_main,
    scan_wavs,
)


def _write_wav(path: Path, fs: int = 16000, frames: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.random.randn(frames).astype(np.float32), fs)


def _run(fn, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = [fn.__module__, *argv]
    try:
        return fn()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# prepare_offline.py
# ---------------------------------------------------------------------------

def test_offline_default_95_5_split(tmp_path):
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    for i in range(10):
        _write_wav(clean / f"utt_{i}.wav")
        _write_wav(noisy / f"utt_{i}.wav")
    out = tmp_path / "out"
    assert _run(
        offline_main,
        ["--clean_dir", str(clean), "--noisy_dir", str(noisy), "--output-dir", str(out)],
    ) == 0
    # 默认 val_ratio=0.05, seed=42：n_val = max(1, int(10*0.05)) = 1
    train_clean = (out / "train" / "spk1.scp").read_text().strip().split("\n")
    train_noisy = (out / "train" / "wav.scp").read_text().strip().split("\n")
    val_clean = (out / "validation" / "spk1.scp").read_text().strip().split("\n")
    val_noisy = (out / "validation" / "wav.scp").read_text().strip().split("\n")
    assert len(train_clean) == len(train_noisy) == 9
    assert len(val_clean) == len(val_noisy) == 1
    for clean_line, noisy_line in zip(train_clean + val_clean, train_noisy + val_noisy):
        clean_fields = clean_line.split()
        noisy_fields = noisy_line.split()
        clean_manifest = out / ("train" if clean_line in train_clean else "validation") / "spk1.scp"
        noisy_manifest = out / ("train" if noisy_line in train_noisy else "validation") / "wav.scp"
        assert len(clean_fields) == len(noisy_fields) == 4
        assert clean_fields[:3] == noisy_fields[:3]
        assert clean_fields[1:3] == ["16000", "8000"]
        assert not Path(clean_fields[3]).is_absolute()
        assert not Path(noisy_fields[3]).is_absolute()
        assert Path(resolve_manifest_resource_path(clean_fields[3], clean_manifest)).is_file()
        assert Path(resolve_manifest_resource_path(noisy_fields[3], noisy_manifest)).is_file()
    assert not (out / "train" / "utt2fs").exists()
    assert not (out / "train" / "speech_length.scp").exists()


def test_offline_four_column_tracks_keep_independent_metadata(tmp_path):
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    for uid in ("a", "b"):
        _write_wav(clean / f"{uid}.wav", fs=16000, frames=8000)
        _write_wav(noisy / f"{uid}.wav", fs=8000, frames=3000)
    out = tmp_path / "out"

    assert _run(
        offline_main,
        ["--clean_dir", str(clean), "--noisy_dir", str(noisy), "--output-dir", str(out)],
    ) == 0
    lines = []
    for split in ("train", "validation"):
        clean_line = (out / split / "spk1.scp").read_text().strip().split()
        noisy_line = (out / split / "wav.scp").read_text().strip().split()
        lines.append((clean_line, noisy_line))
    assert all(clean_line[1:3] == ["16000", "8000"] for clean_line, _ in lines)
    assert all(noisy_line[1:3] == ["8000", "3000"] for _, noisy_line in lines)


def test_offline_val_ratio_out_of_range(tmp_path):
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    _write_wav(clean / "a.wav")
    _write_wav(noisy / "a.wav")
    out = tmp_path / "out"
    for bad in ("0", "1", "-0.1", "1.5"):
        with pytest.raises(SystemExit):
            _run(
                offline_main,
                [
                    "--clean_dir",
                    str(clean),
                    "--noisy_dir",
                    str(noisy),
                    "--output-dir",
                    str(out),
                    "--val_ratio",
                    bad,
                ],
            )


def test_offline_scan_dir_rejects_duplicate_uids(tmp_path):
    """递归目录下同名 stem 会让 SCP 出现重复 uid，必须显式报错而非静默保留首个。"""
    clean = tmp_path / "clean"
    _write_wav(clean / "a.wav")
    _write_wav(clean / "sub" / "a.wav")
    with pytest.raises(ValueError, match="duplicate uid"):
        scan_dir(clean)


@pytest.mark.parametrize("fmt", ["scp", "list"])
def test_offline_cli_rejects_removed_format_option(capsys, fmt):
    with pytest.raises(SystemExit):
        offline_main(
            [
                "--clean_dir",
                "clean",
                "--noisy_dir",
                "noisy",
                "--output-dir",
                "out",
                "--format",
                fmt,
            ]
        )

    assert "unrecognized arguments: --format" in capsys.readouterr().err


def test_offline_cli_rejects_removed_legacy_names_option(capsys):
    with pytest.raises(SystemExit):
        offline_main(
            [
                "--clean_dir",
                "clean",
                "--noisy_dir",
                "noisy",
                "--output-dir",
                "out",
                "--legacy_names",
            ]
        )

    assert "unrecognized arguments: --legacy_names" in capsys.readouterr().err


@pytest.mark.parametrize(
    "relative_conflict",
    [
        "train/utt2fs",
        "validation/speech_length.scp",
        "clean.scp",
        "noisy.scp",
        "train/clean.list",
        "validation/noisy.list",
    ],
)
def test_offline_publish_rejects_retired_or_cross_format_artifacts(
    tmp_path, caplog, relative_conflict
):
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    for i in range(4):
        _write_wav(clean / f"utt_{i}.wav")
        _write_wav(noisy / f"utt_{i}.wav")
    out = tmp_path / "out"
    conflict = out / relative_conflict
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_text("keep me\n", encoding="utf-8")
    argv = [
        "--clean_dir", str(clean), "--noisy_dir", str(noisy),
        "--output-dir", str(out),
    ]

    assert _run(offline_main, argv) == 1
    assert conflict.read_text(encoding="utf-8") == "keep me\n"
    assert "手工移走或删除后重试" in caplog.text


def test_offline_same_format_rerun_remains_supported(tmp_path):
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    for i in range(4):
        _write_wav(clean / f"utt_{i}.wav")
        _write_wav(noisy / f"utt_{i}.wav")
    out = tmp_path / "out"
    argv = [
        "--clean_dir", str(clean), "--noisy_dir", str(noisy),
        "--output-dir", str(out),
    ]

    assert _run(offline_main, argv) == 0
    before = (out / "train" / "spk1.scp").read_text(encoding="utf-8")
    assert _run(offline_main, argv) == 0
    assert (out / "train" / "spk1.scp").read_text(encoding="utf-8") == before


def test_offline_whole_fileset_publish_rolls_back_on_failure(tmp_path, monkeypatch):
    clean, noisy = tmp_path / "clean", tmp_path / "noisy"
    for i in range(4):
        _write_wav(clean / f"utt_{i}.wav")
        _write_wav(noisy / f"utt_{i}.wav")
    out = tmp_path / "out"
    argv = [
        "--clean_dir",
        str(clean),
        "--noisy_dir",
        str(noisy),
        "--output-dir",
        str(out),
    ]
    assert _run(offline_main, argv) == 0
    before = {
        path.relative_to(out): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }

    _write_wav(clean / "utt_new.wav")
    _write_wav(noisy / "utt_new.wav")
    real_replace = file_transaction._replace_at
    publish_calls = 0

    def fail_third_publish(source, target, directory_fds):
        nonlocal publish_calls
        if source.suffix == ".tmp":
            publish_calls += 1
            if publish_calls == 3:
                raise OSError("injected publish failure")
        return real_replace(source, target, directory_fds)

    monkeypatch.setattr(file_transaction, "_replace_at", fail_third_publish)
    assert _run(offline_main, argv) == 1
    after = {
        path.relative_to(out): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert publish_calls >= 3
    assert after == before


# ---------------------------------------------------------------------------
# prepare_voicebank.py
# ---------------------------------------------------------------------------

def test_voicebank_resample_and_layout(tmp_path):
    data_root = tmp_path / "vb"
    # VoiceBank 官方目录布局；train noisy 为 8k（需重采样到 16k）
    _write_wav(data_root / "clean_trainset_wav" / "p226_001.wav", fs=16000)
    _write_wav(data_root / "noisy_trainset_wav" / "p226_001.wav", fs=8000)
    _write_wav(data_root / "clean_testset_wav" / "p226_002.wav", fs=16000)
    _write_wav(data_root / "noisy_testset_wav" / "p226_002.wav", fs=16000)
    out = tmp_path / "out"
    assert _run(
        voicebank_main,
        ["--data_root", str(data_root), "--output-dir", str(out), "--target_sr", "16000"],
    ) == 0
    for split in ("train", "test"):
        assert (out / split / "clean.scp").exists()
        assert (out / split / "noisy.scp").exists()
    # 8k noisy 重采样后应为 16k 的 4 列条目（8000 样本 → 16000 样本）
    noisy_manifest = out / "train" / "noisy.scp"
    noisy_line = noisy_manifest.read_text().strip().split()
    assert noisy_line[1] == "16000"
    assert noisy_line[2] == "16000"
    assert not Path(noisy_line[3]).is_absolute()
    resolved_noisy = Path(
        resolve_manifest_resource_path(noisy_line[3], noisy_manifest)
    )
    assert resolved_noisy.is_file()
    assert sf.info(resolved_noisy).samplerate == 16000


def test_voicebank_val_speakers_holdout(tmp_path):
    data_root = tmp_path / "vb"
    # 3 个说话人，无 testset：--val-speakers 1 应划出字典序最后的 p232
    for spk, n in (("p226", 3), ("p228", 2), ("p232", 2)):
        for i in range(n):
            _write_wav(data_root / "clean_trainset_wav" / f"{spk}_{i:03d}.wav")
            _write_wav(data_root / "noisy_trainset_wav" / f"{spk}_{i:03d}.wav")
    out = tmp_path / "out"
    assert _run(
        voicebank_main,
        ["--data_root", str(data_root), "--output-dir", str(out), "--val-speakers", "1"],
    ) == 0
    train_uids = {
        line.split()[0]
        for line in (out / "train" / "clean.scp").read_text().strip().split("\n")
    }
    val_uids = {
        line.split()[0]
        for line in (out / "val" / "clean.scp").read_text().strip().split("\n")
    }
    assert len(train_uids) == 5 and len(val_uids) == 2
    assert all(u.startswith("p232_") for u in val_uids)
    assert not (train_uids & val_uids)  # 无 train/val 泄漏
    assert (out / "val" / "noisy.scp").exists()


def test_voicebank_val_speakers_out_of_range(tmp_path):
    data_root = tmp_path / "vb"
    _write_wav(data_root / "clean_trainset_wav" / "p226_001.wav")
    _write_wav(data_root / "noisy_trainset_wav" / "p226_001.wav")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="val-speakers"):
        _run(
            voicebank_main,
            ["--data_root", str(data_root), "--output-dir", str(out), "--val-speakers", "1"],
        )


def test_voicebank_scan_wavs_rejects_duplicate_uids(tmp_path):
    """同名 stem 会被下游按 uid 建 dict 时静默丢弃，扫描阶段直接 fail-fast。"""
    data_root = tmp_path / "vb"
    _write_wav(data_root / "p226_001.wav")
    _write_wav(data_root / "sub" / "p226_001.wav")
    with pytest.raises(ValueError, match="duplicate uid"):
        scan_wavs(str(data_root))


pytestmark = [pytest.mark.unit]
