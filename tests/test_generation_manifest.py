"""GenerationManifest schema v2-only 合同。"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tools.prepare_data_tools.generation_manifest import (
    GenerationManifest,
    GenerationManifestError,
    load_generation_manifest,
    sha256_file,
    write_generation_root_marker,
)
from tools.prepare_data_tools.offline_publication import write_ownership_marker

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _entry(alias: str, root: Path, relative: str, *, kind: str = "metadata") -> dict[str, object]:
    path = root / relative
    return {
        "root_alias": alias,
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": kind,
    }


def _payload(indexes: Path, waveforms: Path) -> dict[str, object]:
    (indexes / "clean.scp").write_text("utt 16000 4 waveforms:clean.wav\n", encoding="utf-8")
    (indexes / "noisy.scp").write_text("utt 16000 4 waveforms:noisy.wav\n", encoding="utf-8")
    (waveforms / "clean.wav").write_bytes(b"clean")
    (waveforms / "noisy.wav").write_bytes(b"noisy")
    return {
        "schema_version": 2,
        "generation_id": "tests-generation-2",
        "status": "complete",
        "artifact_kind": "offline-dataset",
        "task": "se",
        "generator": {"id": "tests/fake-generator", "version": "2.0"},
        "root_bindings": [
            {"alias": "indexes", "role": "manifest"},
            {"alias": "waveforms", "role": "resource"},
        ],
        "splits": ["train"],
        "parameters": {"seed": 7, "nested": {"enabled": True, "ratios": [0.1]}},
        "source_identities": [],
        "inventory": [
            _entry("indexes", indexes, "clean.scp"),
            _entry("indexes", indexes, "noisy.scp"),
            _entry("waveforms", waveforms, "clean.wav", kind="waveform"),
            _entry("waveforms", waveforms, "noisy.wav", kind="waveform"),
        ],
        "counts": {"requested": 1, "published": 1, "failed": 0, "dropped": 0},
        "required_gates": [
            {
                "name": "train-reload",
                "adapter": "se-offline-scp-v1",
                "split": "train",
                "inputs": {
                    "clean_scp": {"root_alias": "indexes", "path": "clean.scp"},
                    "noisy_scp": {"root_alias": "indexes", "path": "noisy.scp"},
                },
                "options": {"target_sr": 16000},
            }
        ],
    }


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    (root / "generation_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def _write_collection(indexes: Path, waveforms: Path) -> dict[str, object]:
    for root in (indexes, waveforms):
        write_ownership_marker(root, owner_id="tests/multi-root")
    payload = _payload(indexes, waveforms)
    manifest = GenerationManifest.from_dict(payload)
    _write_manifest(indexes, payload)
    for alias, root in (("indexes", indexes), ("waveforms", waveforms)):
        write_generation_root_marker(
            root, manifest=manifest, root_alias=alias, owner_id="tests/multi-root"
        )
    return payload


def test_v2_round_trip_is_frozen_and_alias_qualified(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    indexes.mkdir()
    waveforms.mkdir()
    payload = _payload(indexes, waveforms)

    manifest = GenerationManifest.from_dict(payload)

    assert manifest.to_dict() == payload
    assert set(manifest.inventory_by_resource()) == {
        ("indexes", "clean.scp"),
        ("indexes", "noisy.scp"),
        ("waveforms", "clean.wav"),
        ("waveforms", "noisy.wav"),
    }
    json.loads(manifest.to_json())
    with pytest.raises(FrozenInstanceError):
        manifest.status = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.parameters["seed"] = 9  # type: ignore[index]


@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "1", None])
def test_schema_version_requires_exact_current_v2(tmp_path: Path, value: object) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    indexes.mkdir()
    waveforms.mkdir()
    payload = _payload(indexes, waveforms)
    payload["schema_version"] = value

    with pytest.raises(GenerationManifestError, match="exact int|当前 v2"):
        GenerationManifest.from_dict(payload)


def test_schema_v1_is_rejected_without_fallback(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    indexes.mkdir()
    waveforms.mkdir()
    payload = _payload(indexes, waveforms)
    payload["schema_version"] = 1
    payload.pop("generation_id")
    payload.pop("root_bindings")

    with pytest.raises(GenerationManifestError, match="旧 v1 publication 必须重新生成"):
        GenerationManifest.from_dict(payload)


def test_v2_requires_aliases_for_inventory_and_gate_inputs(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    indexes.mkdir()
    waveforms.mkdir()
    payload = _payload(indexes, waveforms)
    payload["inventory"][0].pop("root_alias")
    with pytest.raises(GenerationManifestError, match="root_alias"):
        GenerationManifest.from_dict(payload)

    payload = _payload(indexes, waveforms)
    payload["required_gates"][0]["inputs"]["clean_scp"] = "clean.scp"
    with pytest.raises(GenerationManifestError, match="JSON object"):
        GenerationManifest.from_dict(payload)


def test_v2_load_requires_exact_bindings_and_valid_markers(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    indexes.mkdir()
    waveforms.mkdir()
    _write_collection(indexes, waveforms)

    loaded = load_generation_manifest(indexes, root_bindings={"waveforms": waveforms})
    assert loaded.generation_id == "tests-generation-2"

    with pytest.raises(GenerationManifestError, match="缺少.*waveforms"):
        load_generation_manifest(indexes)
    with pytest.raises(GenerationManifestError, match="未知.*other"):
        load_generation_manifest(
            indexes, root_bindings={"waveforms": waveforms, "other": tmp_path}
        )

    marker = waveforms / ".generation-root.json"
    forged = json.loads(marker.read_text(encoding="utf-8"))
    forged["generation_id"] = "forged"
    marker.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(GenerationManifestError, match="marker 不匹配"):
        load_generation_manifest(indexes, root_bindings={"waveforms": waveforms})


def test_v2_rejects_unknown_alias_path_escape_and_legacy_adapter(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    indexes.mkdir()
    waveforms.mkdir()
    payload = _payload(indexes, waveforms)
    payload["inventory"][0]["root_alias"] = "unknown"
    with pytest.raises(GenerationManifestError, match="未知 root_alias"):
        GenerationManifest.from_dict(payload)

    payload = _payload(indexes, waveforms)
    payload["inventory"][2]["path"] = "../outside.wav"
    with pytest.raises(GenerationManifestError, match="不允许"):
        GenerationManifest.from_dict(payload)

    payload = _payload(indexes, waveforms)
    payload["task"] = "aec"
    payload["required_gates"][0]["adapter"] = "aec-offline-scp-v1"
    with pytest.raises(GenerationManifestError, match="不匹配"):
        GenerationManifest.from_dict(payload)
