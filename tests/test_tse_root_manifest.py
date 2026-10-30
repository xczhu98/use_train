"""TSE 根文件推理 JSONL manifest 的路径与字段合同测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.tse.inference.root_manifest import read_tse_root_manifest


pytestmark = pytest.mark.unit


def _write_manifest(tmp_path: Path, rows: list[str]) -> Path:
    manifest = tmp_path / "infer.jsonl"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def test_root_manifest_resolves_canonical_cues_and_inline_attributes(tmp_path: Path):
    mix = tmp_path / "mix.wav"
    enroll = tmp_path / "enroll.wav"
    mix.write_bytes(b"mix")
    enroll.write_bytes(b"enroll")
    manifest = _write_manifest(
        tmp_path,
        [
            json.dumps(
                {
                    "uid": "utt-1",
                    "mixture": "mix.wav",
                    "speaker_waveform": "enroll.wav",
                    "attributes": {
                        "cues": {"spatial": {"azimuth": 0.0, "elevation": 0.1}}
                    },
                }
            )
        ],
    )

    item = read_tse_root_manifest(manifest)[0]

    assert item.uid == "utt-1"
    assert item.inputs == {"mixture": mix, "speaker_waveform": enroll}
    assert item.attributes["cues"]["spatial"]["elevation"] == 0.1


def test_root_manifest_rejects_unknown_fields_and_duplicate_uids(tmp_path: Path):
    audio = tmp_path / "mix.wav"
    audio.write_bytes(b"mix")
    unknown = _write_manifest(
        tmp_path,
        [json.dumps({"uid": "utt", "mixture": "mix.wav", "enrollment": "mix.wav"})],
    )
    with pytest.raises(ValueError, match="unknown fields"):
        read_tse_root_manifest(unknown)

    duplicate = _write_manifest(
        tmp_path,
        [
            json.dumps({"uid": "utt", "mixture": "mix.wav"}),
            json.dumps({"uid": "utt", "mixture": "mix.wav"}),
        ],
    )
    with pytest.raises(ValueError, match="repeats uid"):
        read_tse_root_manifest(duplicate)


def test_root_manifest_enforces_root_dir_and_rejects_symlink(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    manifest = _write_manifest(
        root,
        [json.dumps({"uid": "utt", "mixture": "../outside.wav"})],
    )
    with pytest.raises(ValueError, match="escapes --scp_root_dir"):
        read_tse_root_manifest(manifest, root_dir=root)

    target = root / "target.wav"
    target.write_bytes(b"target")
    link = root / "link.wav"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    linked = _write_manifest(
        root,
        [json.dumps({"uid": "utt", "mixture": "link.wav"})],
    )
    with pytest.raises(ValueError, match="symbolic link"):
        read_tse_root_manifest(linked, root_dir=root)
