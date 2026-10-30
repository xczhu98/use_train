"""schema v2 SE/TSE reload gate 与旧 publication adapter 下线合同。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_data_tools.generation_manifest import (
    GenerationManifest,
    sha256_file,
    write_generation_root_marker,
)
from tools.prepare_data_tools import offline_publication
from tools.prepare_data_tools.offline_publication import write_ownership_marker
from tools.prepare_data_tools.offline_reload_gate import (
    OfflineReloadError,
    require_publish_eligible,
    run_offline_reload_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.linspace(-0.2, 0.2, 64, dtype=np.float32), 16000, subtype="FLOAT")


def _entry(alias: str, root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "root_alias": alias,
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": "waveform" if path.suffix == ".wav" else "metadata",
    }


def _make_v2_se(indexes: Path, waveforms: Path, *, reference: str = "waveforms:clean.wav") -> GenerationManifest:
    indexes.mkdir()
    waveforms.mkdir()
    for root in (indexes, waveforms):
        write_ownership_marker(root, owner_id="tests/v2-reload")
    _write_wav(waveforms / "clean.wav")
    _write_wav(waveforms / "noisy.wav")
    (indexes / "clean.scp").write_text(f"utt 16000 64 {reference}\n", encoding="utf-8")
    (indexes / "noisy.scp").write_text("utt 16000 64 waveforms:noisy.wav\n", encoding="utf-8")
    payload = {
        "schema_version": 2,
        "generation_id": "tests-v2-reload",
        "status": "complete",
        "artifact_kind": "se-multi-root-offline",
        "task": "se",
        "generator": {"id": "tests/v2-reload", "version": "1.0"},
        "root_bindings": [
            {"alias": "indexes", "role": "manifest"},
            {"alias": "waveforms", "role": "resource"},
        ],
        "splits": ["train"],
        "parameters": {},
        "source_identities": [],
        "inventory": [
            _entry("indexes", indexes, "clean.scp"),
            _entry("indexes", indexes, "noisy.scp"),
            _entry("waveforms", waveforms, "clean.wav"),
            _entry("waveforms", waveforms, "noisy.wav"),
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
                "options": {"strict_load": True},
            }
        ],
    }
    manifest = GenerationManifest.from_dict(payload)
    (indexes / "generation_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    for alias, root in (("indexes", indexes), ("waveforms", waveforms)):
        write_generation_root_marker(
            root, manifest=manifest, root_alias=alias, owner_id="tests/v2-reload"
        )
    return manifest


def test_v2_se_full_reload_and_probe_share_explicit_roots(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    manifest = _make_v2_se(indexes, waveforms)

    full = run_offline_reload_gates(
        indexes, manifest=manifest, root_bindings={"waveforms": waveforms}
    )
    probe = run_offline_reload_gates(
        indexes, manifest=manifest, root_bindings={"waveforms": waveforms}, probe=1
    )

    assert full.publish_eligible is True
    assert full.gates[0].dataset_size == full.gates[0].scanned == 1
    assert probe.publish_eligible is False
    with pytest.raises(OfflineReloadError, match="full scan"):
        require_publish_eligible(probe)


@pytest.mark.parametrize(
    "reference",
    ["/absolute.wav", "../outside.wav", "unknown:clean.wav", "waveforms:../clean.wav"],
)
def test_v2_se_rejects_scp_escape_and_unknown_alias(tmp_path: Path, reference: str) -> None:
    indexes = tmp_path / "indexes"
    waveforms = tmp_path / "waveforms"
    manifest = _make_v2_se(indexes, waveforms, reference=reference)

    with pytest.raises(OfflineReloadError, match="绝对|不允许|未知 root alias|缺少受控 root alias"):
        run_offline_reload_gates(
            indexes, manifest=manifest, root_bindings={"waveforms": waveforms}
        )


def test_schema_v1_manifest_is_rejected_before_reload(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "generation_manifest.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    with pytest.raises(Exception, match="旧 v1 publication 必须重新生成|当前 v2"):
        run_offline_reload_gates(root)


def test_single_root_publication_api_is_removed() -> None:
    assert not hasattr(offline_publication, "publish_owned_offline_dataset")
