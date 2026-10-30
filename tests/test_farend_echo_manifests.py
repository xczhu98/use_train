"""Tests for echo-pair manifest scripts (merge / build)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.prepare_data_tools.generators.echo.build_real_farend_echo_pair_manifest import (  # noqa: E402
    main as build_main,
)
from tools.prepare_data_tools.generators.echo.merge_farend_echo_pair_manifests import (  # noqa: E402
    main as merge_main,
)


def _write_stereo_wav(path: Path, fs: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.random.randn(fs // 2).astype(np.float32)
    sf.write(path, np.stack([x, x], axis=1), fs)


def _run(fn, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = [fn.__module__, *argv]
    try:
        return fn()
    finally:
        sys.argv = old


def _write_manifest(path: Path, pairs: list) -> None:
    path.write_text(json.dumps({"pairs": pairs}, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# merge_farend_echo_pair_manifests.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "payload",
    [
        ["stereo.wav"],
        {"pairs": [["farend.wav", "echo.wav"]]},
        {"pairs": [{"farend": "farend.wav", "echo": "echo.wav"}]},
        {"pairs": [" "]},
        {"pairs": ["stereo.wav"], "legacy": True},
        {"pairs": ["stereo.wav"], "meta": []},
    ],
)
def test_merge_rejects_noncanonical_manifests(tmp_path, payload):
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        _run(merge_main, ["--inputs", str(manifest), "-o", str(tmp_path / "out.json")])


def test_merge_dedup_resolves_relative_to_manifest_dir(tmp_path):
    # 相对路径按 manifest 所在目录解析：不同目录下同名相对路径不应被误判重
    d1, d2 = tmp_path / "d1", tmp_path / "d2"
    _write_stereo_wav(d1 / "a.wav")
    _write_stereo_wav(d2 / "a.wav")
    m1 = d1 / "m1.json"
    m2 = d2 / "m2.json"
    _write_manifest(m1, ["a.wav"])
    _write_manifest(m2, ["a.wav"])
    out = tmp_path / "merged.json"
    assert _run(merge_main, ["--inputs", str(m1), str(m2), "-o", str(out)]) == 0
    merged = json.loads(out.read_text())
    assert len(merged["pairs"]) == 2  # 两个不同目录的文件都保留
    # 输出条目必须是按各自源 manifest 解析后的绝对路径（合并后 base_dir 不再可用）
    assert sorted(merged["pairs"]) == sorted([
        str((d1 / "a.wav").resolve()),
        str((d2 / "a.wav").resolve()),
    ])


def test_merge_dedup_same_file_absolute_and_relative(tmp_path):
    d = tmp_path / "d"
    _write_stereo_wav(d / "a.wav")
    m = d / "m.json"
    _write_manifest(m, ["a.wav", str((d / "a.wav").resolve())])
    out = tmp_path / "merged.json"
    assert _run(merge_main, ["--inputs", str(m), "-o", str(out)]) == 0
    merged = json.loads(out.read_text())
    assert len(merged["pairs"]) == 1  # 相对 + 绝对同文件判重


# ---------------------------------------------------------------------------
# build_real_farend_echo_pair_manifest.py
# ---------------------------------------------------------------------------

def test_build_default_split(tmp_path):
    root = tmp_path / "corpus"
    for i in range(4):
        _write_stereo_wav(root / f"seg_{i}.wav")
    out = tmp_path / "out.json"
    assert _run(
        build_main,
        ["--no_validate", "--out", str(out), str(root)],
    ) == 0
    assert out.with_name("out_train.json").exists()
    assert out.with_name("out_val.json").exists()
    train = json.loads(out.with_name("out_train.json").read_text())
    val = json.loads(out.with_name("out_val.json").read_text())
    assert len(train["pairs"]) == 3
    assert len(val["pairs"]) == 1


def test_build_val_ratio_zero_single_file(tmp_path):
    root = tmp_path / "corpus"
    _write_stereo_wav(root / "a.wav")
    out = tmp_path / "out.json"
    assert _run(
        build_main,
        ["--no_validate", "--val_ratio", "0", "--out", str(out), str(root)],
    ) == 0
    assert out.exists()
    assert not out.with_name("out_train.json").exists()


def test_build_val_ratio_out_of_range(tmp_path):
    root = tmp_path / "corpus"
    _write_stereo_wav(root / "a.wav")
    with pytest.raises(SystemExit):
        _run(build_main, ["--no_validate", "--val_ratio", "1.0", str(root)])

pytestmark = [pytest.mark.unit]
