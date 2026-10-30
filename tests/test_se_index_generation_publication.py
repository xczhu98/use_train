"""SE 通用 indexer 与 DNS3 schema v2 publication 接入测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.data.resource_paths import resolve_manifest_resource_path
from tools.prepare_data_tools.generation_manifest import (
    load_generation_manifest,
    write_generation_root_marker,
)
from tools.prepare_data_tools.offline_publication import (
    RECEIPT_FILE,
    write_ownership_marker,
)
from tools.prepare_data_tools.offline_reload_gate import OfflineReloadError
from tools.prepare_data_tools.scp import prepare_offline
from tools.prepare_data_tools.generators.se import generate_dns3_offline_dataset as dns3
from tools.prepare_data_tools.generators.se._dns3_dataset_generation import (
    DNS3GenerationConfig,
)
from tools.prepare_data_tools.se_index_publication import (
    build_se_index_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_generator_versions_increment_once_and_record_history() -> None:
    assert prepare_offline.GENERATOR_VERSION_HISTORY == ("1.0", "2.0", "2.1")
    assert prepare_offline.GENERATOR_VERSION == "2.1"
    assert dns3.GENERATOR_VERSION_HISTORY[-2:] == ("2.0", "2.1")
    assert dns3.GENERATOR_VERSION == "2.1"


def _write_wav(path: Path, *, frames: int = 64, fs: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.linspace(-0.2, 0.2, frames, dtype=np.float32),
        fs,
        subtype="FLOAT",
    )


def _prepare_verified_waveform_root(root: Path, draft_indexes: Path) -> None:
    owner_id = "tests/prepare-offline-existing-waveforms"
    root.mkdir()
    write_ownership_marker(root, owner_id=owner_id)
    for uid in ("a", "b"):
        _write_wav(root / "clean" / f"{uid}.wav")
        _write_wav(root / "noisy" / f"{uid}.wav")

    clean = prepare_offline.scan_dir(root / "clean")
    noisy = prepare_offline.scan_dir(root / "noisy")
    outputs = prepare_offline.plan_split_outputs(
        ["a"],
        clean,
        noisy,
        draft_indexes / "train",
        waveform_root=root,
    )
    outputs.extend(
        prepare_offline.plan_split_outputs(
            ["b"],
            clean,
            noisy,
            draft_indexes / "validation",
            waveform_root=root,
        )
    )
    prepare_offline.publish_outputs(outputs)
    manifest = build_se_index_manifest(
        index_root=draft_indexes,
        waveform_root=root,
        generation_id="tests-index-generation",
        generator_id=prepare_offline.GENERATOR_ID,
        generator_version=prepare_offline.GENERATOR_VERSION,
        split_scps={
            "train": ("train/spk1.scp", "train/wav.scp"),
            "validation": (
                "validation/spk1.scp",
                "validation/wav.scp",
            ),
        },
        waveform_paths=sorted(root.rglob("*.wav")),
        parameters={"seed": 42, "sample_rate": None},
    )
    write_generation_root_marker(
        root,
        manifest=manifest,
        root_alias="waveforms",
        owner_id=owner_id,
    )


def test_prepare_offline_legacy_mode_has_no_verified_receipt(tmp_path: Path) -> None:
    clean = tmp_path / "legacy-clean"
    noisy = tmp_path / "legacy-noisy"
    for uid in ("a", "b"):
        _write_wav(clean / f"{uid}.wav")
        _write_wav(noisy / f"{uid}.wav")
    output = tmp_path / "legacy-indexes"

    assert prepare_offline.main(
        [
            "--clean_dir",
            str(clean),
            "--noisy_dir",
            str(noisy),
            "--output-dir",
            str(output),
        ]
    ) == 0

    assert not (output / "generation_manifest.json").exists()
    assert not (output / RECEIPT_FILE).exists()
    manifests = (
        output / "train" / "spk1.scp",
        output / "validation" / "spk1.scp",
    )
    assert any(
        Path(
            resolve_manifest_resource_path(line.split()[-1], manifest)
        ) == (clean / "a.wav").resolve()
        for manifest in manifests
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    )


def test_prepare_offline_defaults_to_unmatched_failure(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    _write_wav(clean / "paired.wav")
    _write_wav(noisy / "paired.wav")
    _write_wav(clean / "orphan.wav")
    output = tmp_path / "indexes"

    assert prepare_offline.main(
        [
            "--clean_dir",
            str(clean),
            "--noisy_dir",
            str(noisy),
            "--output-dir",
            str(output),
        ]
    ) == 1
    assert not output.exists()


def test_prepare_offline_verified_publication_full_reloads_existing_root(
    tmp_path: Path,
) -> None:
    waveform_root = tmp_path / "waveforms"
    draft_indexes = tmp_path / "draft-indexes"
    _prepare_verified_waveform_root(waveform_root, draft_indexes)
    output = tmp_path / "published-indexes"

    assert prepare_offline.main(
        [
            "--clean_dir",
            str(waveform_root / "clean"),
            "--noisy_dir",
            str(waveform_root / "noisy"),
            "--output-dir",
            str(output),
            "--verified-publication",
            "--waveform-root",
            str(waveform_root),
        ]
    ) == 0

    manifest = load_generation_manifest(
        output,
        root_bindings={"waveforms": waveform_root},
    )
    receipt = json.loads((output / RECEIPT_FILE).read_text(encoding="utf-8"))
    assert manifest.schema_version == 2
    assert manifest.generator.version == "2.1"
    assert receipt["generation_id"] == "tests-index-generation"
    assert receipt["reload_report"]["scan_mode"] == "full"
    assert receipt["reload_report"]["publish_eligible"] is True
    scp_text = (output / "train" / "spk1.scp").read_text(encoding="utf-8")
    assert "waveforms:" in scp_text
    assert str(waveform_root) not in scp_text


def test_prepare_offline_verified_rejects_unowned_waveforms(tmp_path: Path) -> None:
    waveform_root = tmp_path / "waveforms"
    _write_wav(waveform_root / "clean" / "a.wav")
    _write_wav(waveform_root / "noisy" / "a.wav")
    _write_wav(waveform_root / "clean" / "b.wav")
    _write_wav(waveform_root / "noisy" / "b.wav")
    output = tmp_path / "indexes"

    assert prepare_offline.main(
        [
            "--clean_dir",
            str(waveform_root / "clean"),
            "--noisy_dir",
            str(waveform_root / "noisy"),
            "--output-dir",
            str(output),
            "--verified-publication",
            "--waveform-root",
            str(waveform_root),
        ]
    ) == 1
    assert not output.exists()


def _dns3_generation_stub(config: DNS3GenerationConfig) -> None:
    for split, count in (
        ("train", config.num_train),
        ("val", config.num_val),
        ("test", config.num_test),
    ):
        for index in range(count):
            uid = str(index + 1)
            _write_wav(config.output_dir / f"{split}_clean" / f"{uid}.wav")
            _write_wav(config.output_dir / f"{split}_noisy" / f"{uid}.wav")


def test_dns3_full_generation_jointly_publishes_and_scp_only_reuses_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dns3.generation, "run_generation", _dns3_generation_stub)
    waveform_target = tmp_path / "dns3-waveforms"
    index_target = tmp_path / "dns3-indexes"
    argv = [
        "--data-root",
        str(tmp_path / "unused-raw"),
        "--output-dir",
        str(waveform_target),
        "--scp-dir",
        str(index_target),
        "--num-train",
        "1",
        "--num-val",
        "1",
        "--num-test",
        "0",
        "--num-workers",
        "1",
    ]

    assert dns3.main(argv) == 0
    manifest = load_generation_manifest(
        index_target,
        root_bindings={"waveforms": waveform_target},
    )
    receipt = json.loads((index_target / RECEIPT_FILE).read_text(encoding="utf-8"))
    assert dns3.GENERATOR_VERSION == "2.1"
    assert manifest.schema_version == 2
    assert manifest.generator.version == "2.1"
    assert receipt["reload_report"]["publish_eligible"] is True
    assert "waveforms:train_clean/1.wav" in (
        index_target / "train" / "clean.scp"
    ).read_text(encoding="utf-8")

    second_indexes = tmp_path / "dns3-indexes-scp-only"
    assert dns3.main(
        [
            "--data-root",
            str(tmp_path / "unused-raw"),
            "--output-dir",
            str(waveform_target),
            "--scp-dir",
            str(second_indexes),
            "--scp-only",
            "--num-train",
            "1",
            "--num-val",
            "1",
            "--num-test",
            "0",
            "--num-workers",
            "1",
        ]
    ) == 0
    second_receipt = json.loads(
        (second_indexes / RECEIPT_FILE).read_text(encoding="utf-8")
    )
    assert second_receipt["generation_id"] == receipt["generation_id"]
    assert second_receipt["reload_report"]["scan_mode"] == "full"


def test_dns3_final_reload_failure_exposes_no_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dns3.generation, "run_generation", _dns3_generation_stub)
    monkeypatch.setattr(
        "tools.prepare_data_tools.offline_publication.run_offline_reload_gates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OfflineReloadError("injected final reload failure")
        ),
    )
    waveform_target = tmp_path / "waveforms"
    index_target = tmp_path / "indexes"

    assert dns3.main(
        [
            "--data-root",
            str(tmp_path / "unused"),
            "--output-dir",
            str(waveform_target),
            "--scp-dir",
            str(index_target),
            "--num-train",
            "1",
            "--num-val",
            "0",
            "--num-test",
            "0",
            "--num-workers",
            "1",
        ]
    ) == 1
    assert not waveform_target.exists()
    assert not index_target.exists()
    assert not list(tmp_path.glob("*.backup-*"))


def test_dns3_generate_scp_defaults_to_unmatched_failure(tmp_path: Path) -> None:
    waveform_root = tmp_path / "waveforms"
    _write_wav(waveform_root / "train_clean" / "paired.wav")
    _write_wav(waveform_root / "train_noisy" / "paired.wav")
    _write_wav(waveform_root / "train_clean" / "orphan.wav")

    with pytest.raises(ValueError, match="match 1:1"):
        dns3.generation.generate_scp(
            str(waveform_root), str(tmp_path / "indexes"), waveform_root_alias="waveforms"
        )


def test_dns3_scp_only_rejects_root_without_generation_identity(tmp_path: Path) -> None:
    waveform_root = tmp_path / "legacy-waveforms"
    _write_wav(waveform_root / "train_clean" / "1.wav")
    _write_wav(waveform_root / "train_noisy" / "1.wav")
    index_target = tmp_path / "indexes"

    assert dns3.main(
        [
            "--data-root",
            str(tmp_path / "unused"),
            "--output-dir",
            str(waveform_root),
            "--scp-dir",
            str(index_target),
            "--scp-only",
        ]
    ) == 1
    assert not index_target.exists()
