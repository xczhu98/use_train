"""Tests for read-only training manifest inspection contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.prepare_data_tools.manifest_contracts import (
    PROFILE_CATEGORIZED_NOISE,
    PROFILE_DYNAMIC_SOURCE,
    PROFILE_DYNAMIC_SPEECH,
    PROFILE_STRICT_NOISE_PUBLICATION,
    ManifestInspectionError,
    inspect_manifest,
)
from tools.prepare_data_tools.scp.prepare_scp import _print_yaml_snippet

pytestmark = pytest.mark.unit

# 脚本路径与 cwd 都锚定仓库根，避免从仓库根之外调用 pytest 时失败
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(uid: str, **extra: object) -> dict[str, object]:
    return {
        "uid": uid,
        "fs": 16000,
        "length": 16000,
        "path": f"/audio/{uid}.wav",
        **extra,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_dynamic_source_accepts_four_column_scp(tmp_path: Path) -> None:
    manifest = tmp_path / "noise.scp"
    manifest.write_text("noise-a 16000 32000 /audio/noise-a.wav\n", encoding="utf-8")

    result = inspect_manifest(manifest)

    assert result.profile == PROFILE_DYNAMIC_SOURCE
    assert result.serialization_format == "scp"
    assert result.record_count == 1
    assert result.sample_rates == (16000,)
    assert result.capabilities.length_complete is True
    assert result.capabilities.has_metadata is False


def test_dynamic_source_rejects_three_column_scp_and_json_array(tmp_path: Path) -> None:
    scp_manifest = tmp_path / "noise.scp"
    scp_manifest.write_text("noise-a 16000 /audio/noise-a.wav\n", encoding="utf-8")
    json_manifest = tmp_path / "noise.json"
    json_manifest.write_text(json.dumps([_record("json")]), encoding="utf-8")

    with pytest.raises(ManifestInspectionError, match="expected exactly four fields"):
        inspect_manifest(scp_manifest)
    with pytest.raises(ManifestInspectionError, match="must use .scp .* or .jsonl"):
        inspect_manifest(json_manifest)


def test_categorized_noise_requires_complete_category_metadata(tmp_path: Path) -> None:
    complete = tmp_path / "complete.jsonl"
    _write_jsonl(
        complete,
        [
            _record("rain", category_id=0, category_name="Rain"),
            _record("wind", category_id=1, balance_category="Wind"),
        ],
    )
    incomplete = tmp_path / "incomplete.jsonl"
    _write_jsonl(
        incomplete,
        [
            _record("rain", category_id=0, category_name="Rain"),
            _record("unknown", category_id=1),
        ],
    )

    result = inspect_manifest(complete, profile=PROFILE_CATEGORIZED_NOISE)

    assert result.capabilities.category_complete is True
    with pytest.raises(ManifestInspectionError, match="category metadata"):
        inspect_manifest(incomplete, profile=PROFILE_CATEGORIZED_NOISE)


@pytest.mark.parametrize(
    "records",
    [
        [
            _record("rain", category_id=0, category_name="Rain"),
            _record("wind", category_id=0, category_name="Wind"),
        ],
        [
            _record("rain-a", category_id=0, category_name="Rain"),
            _record("rain-b", category_id=1, category_name="Rain"),
        ],
    ],
)
def test_categorized_noise_rejects_inconsistent_category_mapping(
    tmp_path: Path,
    records: list[dict[str, object]],
) -> None:
    manifest = tmp_path / "conflict.jsonl"
    _write_jsonl(manifest, records)

    with pytest.raises(ManifestInspectionError, match="maps to multiple"):
        inspect_manifest(manifest, profile=PROFILE_CATEGORIZED_NOISE)


@pytest.mark.parametrize(
    "profile",
    [
        PROFILE_DYNAMIC_SOURCE,
        PROFILE_CATEGORIZED_NOISE,
        PROFILE_STRICT_NOISE_PUBLICATION,
    ],
)
def test_strict_like_invalid_manifest_fails_all_training_profiles(
    tmp_path: Path,
    profile: str,
) -> None:
    manifest = tmp_path / "strict-like.jsonl"
    record = _record(
        "forged",
        category_id=0,
        category_name="Rain",
        manifest_schema_version=6,
        admission_allowed=True,
        admission_decision="accept_noise",
        admission_reason_codes=["forged"],
        admission_policy_version=4,
        admission_policy_sha256="0" * 64,
        balance_category_mid="/m/06mb1",
        balance_category_source="audioset_top5",
        verification={},
    )
    _write_jsonl(manifest, [record])

    with pytest.raises(ManifestInspectionError, match="could not inspect profile"):
        inspect_manifest(manifest, profile=profile)


@pytest.mark.parametrize(
    ("marker", "value"),
    [
        ("manifest_schema_version", 6),
        ("admission_allowed", True),
        ("verification", {}),
    ],
)
def test_each_publication_marker_triggers_production_admission(
    tmp_path: Path,
    marker: str,
    value: object,
) -> None:
    manifest = tmp_path / "strict-like.jsonl"
    _write_jsonl(manifest, [_record("forged", **{marker: value})])

    with pytest.raises(ManifestInspectionError, match="could not inspect profile"):
        inspect_manifest(manifest, profile=PROFILE_DYNAMIC_SOURCE)


@pytest.mark.parametrize(
    ("profile", "extra"),
    [
        (PROFILE_DYNAMIC_SOURCE, {}),
        (
            PROFILE_CATEGORIZED_NOISE,
            {"category_id": 0, "category_name": "Rain"},
        ),
    ],
)
def test_basic_manifest_is_not_mistaken_for_strict_publication(
    tmp_path: Path,
    profile: str,
    extra: dict[str, object],
) -> None:
    manifest = tmp_path / "basic.jsonl"
    _write_jsonl(manifest, [_record("basic", **extra)])

    result = inspect_manifest(manifest, profile=profile)

    assert result.record_count == 1
    assert result.profile == profile
    assert result.capabilities.strict_fields_complete is False


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("empty.jsonl", "", "at least one record"),
        (
            "duplicate.jsonl",
            json.dumps(_record("same")) + "\n" + json.dumps(_record("same")),
            "duplicate uid",
        ),
        (
            "blank.jsonl",
            json.dumps(_record("one")) + "\n\n" + json.dumps(_record("two")),
            "blank JSONL line",
        ),
        ("retired.list", "/audio/one.wav\n", "must use .scp .* or .jsonl"),
    ],
)
def test_dynamic_source_preserves_parser_fail_closed_behavior(
    tmp_path: Path,
    name: str,
    content: str,
    message: str,
) -> None:
    manifest = tmp_path / name
    manifest.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestInspectionError, match=message):
        inspect_manifest(manifest)


def test_cli_json_output_and_failure_exit(tmp_path: Path) -> None:
    manifest = tmp_path / "noise.jsonl"
    _write_jsonl(
        manifest,
        [_record("rain", category_id=0, category_name="Rain")],
    )
    script = _REPO_ROOT / "tools" / "prepare_data_tools" / "inspect_manifest.py"

    success = subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest),
            "--profile",
            PROFILE_CATEGORIZED_NOISE,
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    failed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "missing.jsonl"), "--json"],
        text=True,
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
    )

    assert success.returncode == 0
    payload = json.loads(success.stdout)
    assert payload["serialization_format"] == "jsonl"
    assert payload["record_count"] == 1
    assert payload["sample_rates"] == [16000]
    assert payload["capabilities"]["category_complete"] is True
    assert payload["yaml_hint"].startswith("noise_scp:")
    assert success.stderr == ""
    assert failed.returncode != 0
    assert failed.stdout == ""
    assert "error:" in failed.stderr
    assert "missing.jsonl" in failed.stderr


def test_prepare_scp_prints_modern_dynamic_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_yaml_snippet(
        tmp_path,
        {"noise_sources": ([tmp_path / "noise"], False)},
        True,
    )

    output = capsys.readouterr().out
    assert "data:\n  mode: dynamic\n" in output
    assert "  train:\n    noise_scp:" in output
    assert "  validation:\n    mode: dynamic\n    inherit_simulation: true" in output


def test_prepare_scp_yaml_includes_explicit_validation_speech(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_yaml_snippet(
        tmp_path,
        {
            "train_speech_sources": ([tmp_path / "train"], True),
            "val_speech_sources": ([tmp_path / "val"], True),
        },
        True,
    )

    output = capsys.readouterr().out
    assert f"    speech_scp: {tmp_path / 'train_speech_sources.scp'}" in output
    assert f"    speech_scp: {tmp_path / 'val_speech_sources.scp'}" in output


def test_prepare_scp_yaml_emits_split_music_resources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_yaml_snippet(
        tmp_path,
        {"music_sources": ([tmp_path / "music"], False)},
        True,
    )

    output = capsys.readouterr().out
    assert f"    music_scp: {tmp_path / 'train_music_sources.scp'}" in output
    assert f"    music_scp: {tmp_path / 'val_music_sources.scp'}" in output


# ---------------------------------------------------------------------------
# dynamic-speech：生产 speech 发布门禁（要求完整 length）
# ---------------------------------------------------------------------------

def test_dynamic_speech_accepts_four_column_scp(tmp_path: Path) -> None:
    manifest = tmp_path / "speech.scp"
    manifest.write_text("utt-a 16000 32000 /audio/a.wav\n", encoding="utf-8")

    result = inspect_manifest(manifest, profile=PROFILE_DYNAMIC_SPEECH)

    assert result.profile == PROFILE_DYNAMIC_SPEECH
    assert result.serialization_format == "scp"
    assert result.capabilities.length_complete is True
    assert result.yaml_hint.startswith("speech_scp:")


def test_dynamic_speech_accepts_jsonl_with_complete_length(tmp_path: Path) -> None:
    manifest = tmp_path / "speech.jsonl"
    manifest.write_text(json.dumps(_record("utt-a")) + "\n", encoding="utf-8")

    result = inspect_manifest(manifest, profile=PROFILE_DYNAMIC_SPEECH)

    assert result.capabilities.length_complete is True
    assert result.yaml_hint.startswith("speech_scp:")


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("speech.json", json.dumps([_record("utt-a")]), "must use .scp .* or .jsonl"),
        (
            "mixed.jsonl",
            json.dumps(_record("a")) + "\n"
            + json.dumps({"uid": "b", "fs": 16000, "path": "/b.wav"}) + "\n",
            "length must be an exact integer > 0",
        ),
    ],
)
def test_dynamic_speech_rejects_retired_or_incomplete_source(
    tmp_path: Path,
    name: str,
    content: str,
    message: str,
) -> None:
    manifest = tmp_path / name
    manifest.write_text(content, encoding="utf-8")

    with pytest.raises(ManifestInspectionError, match=message):
        inspect_manifest(manifest, profile=PROFILE_DYNAMIC_SPEECH)


def test_cli_dynamic_speech_json_output(tmp_path: Path) -> None:
    manifest = tmp_path / "speech.jsonl"
    manifest.write_text(json.dumps(_record("utt-a")) + "\n", encoding="utf-8")
    script = _REPO_ROOT / "tools" / "prepare_data_tools" / "inspect_manifest.py"

    success = subprocess.run(
        [
            sys.executable,
            str(script),
            str(manifest),
            "--profile",
            PROFILE_DYNAMIC_SPEECH,
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=_REPO_ROOT,
    )

    assert success.returncode == 0
    payload = json.loads(success.stdout)
    assert payload["profile"] == PROFILE_DYNAMIC_SPEECH
    assert payload["capabilities"]["length_complete"] is True
    assert payload["yaml_hint"].startswith("speech_scp:")
