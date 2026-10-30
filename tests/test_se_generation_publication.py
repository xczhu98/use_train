"""SE offline generator 的 GenerationManifest 与 owned publication 合同。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_data_tools.generation_manifest import load_generation_manifest
from tools.prepare_data_tools.generators.se import generate_offline_dataset as generator
from tools.prepare_data_tools.offline_publication import RECEIPT_FILE
from tools.prepare_data_tools.offline_reload_gate import (
    OfflineReloadError,
    require_publish_eligible,
    run_offline_reload_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _FixtureDataset:
    def __init__(self, values: tuple[float, ...] = (0.1, 0.2)) -> None:
        self.values = values
        self.epoch_calls: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self.values)

    def set_epoch(self, epoch: int, *, base_seed: int) -> None:
        self.epoch_calls.append((epoch, base_seed))

    def __getitem__(self, index: int):
        waveform = np.full(80, self.values[index], dtype=np.float32)
        return waveform, waveform * 0.5, 16000, 80, False, False, False


def _install_fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> _FixtureDataset:
    dataset = _FixtureDataset()
    source_config = {
        "data": {
            "mode": "dynamic",
            "sample_rate": 16000,
            "train": {"simulation_cfg": {}},
        }
    }
    monkeypatch.setattr(generator.OmegaConf, "load", lambda _path: source_config)
    monkeypatch.setattr(
        generator.OmegaConf,
        "to_container",
        lambda config, resolve: config,
    )
    import tasks.se.data

    monkeypatch.setattr(
        tasks.se.data, "build_dataset", lambda _config, mode: dataset
    )
    return dataset


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: Path | None = None
) -> tuple[int, Path, _FixtureDataset]:
    dataset = _install_fixture_dataset(monkeypatch)
    published = output or tmp_path / "published"
    result = generator.main(
        [
            "-C",
            str(tmp_path / "dynamic.yaml"),
            "--output-dir",
            str(published),
            "--num-samples",
            "2",
            "--seed",
            "73",
        ]
    )
    return result, published, dataset


def test_cli_contract_and_version_increment_once() -> None:
    canonical = generator.parse_args(
        ["-C", "config.yaml", "--output-dir", "out", "--num-samples", "2"]
    )
    assert canonical.num_samples == 2
    with pytest.raises(SystemExit):
        generator.parse_args(
            ["-C", "config.yaml", "--output-dir", "out", "--num_samples", "2"]
        )
    history = generator.GENERATOR_VERSION_HISTORY
    versions = [tuple(map(int, item.split("."))) for item in history]
    assert versions == sorted(set(versions))
    assert generator.GENERATOR_VERSION == history[-1]
    assert versions[-1] >= (2, 11)  # 显式场景概率参与仿真的最低版本


def test_owned_publication_binds_inventory_and_full_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, published, dataset = _run(tmp_path, monkeypatch)

    assert result == 0
    assert dataset.epoch_calls == [(0, 73)]
    manifest = load_generation_manifest(
        published, generator.ENVELOPE_MANIFEST_NAME
    )
    assert manifest.schema_version == 2
    assert manifest.generation_id
    assert manifest.generator.id == generator.GENERATOR_ID
    assert manifest.generator.version == generator.GENERATOR_VERSION
    assert [(binding.alias, binding.role) for binding in manifest.root_bindings] == [
        (generator._ROOT_ALIAS, "manifest")
    ]
    assert manifest.counts.requested == manifest.counts.published == 2
    business = json.loads((published / generator.MANIFEST_NAME).read_text())
    assert business == {
        "generator_id": generator.GENERATOR_ID,
        "generator_version": generator.GENERATOR_VERSION,
        "counts": {"requested": 2, "success": 2, "failed": 0},
        "uid_semantics": "successful_samples_contiguous_zero_based",
        "seed": 73,
        "debug_files": 0,
    }
    inventory = set(manifest.inventory_by_resource())
    assert {
        (generator._ROOT_ALIAS, "clean.scp"),
        (generator._ROOT_ALIAS, "noisy.scp"),
        (generator._ROOT_ALIAS, generator.RELOAD_CLEAN_SCP),
        (generator._ROOT_ALIAS, generator.RELOAD_NOISY_SCP),
        (generator._ROOT_ALIAS, generator.MANIFEST_NAME),
        (generator._ROOT_ALIAS, "clean/utt_00000000.wav"),
        (generator._ROOT_ALIAS, "noisy/utt_00000000.wav"),
    } <= inventory
    assert (published / "clean.scp").read_text(encoding="utf-8").split()[3] == str(
        published / "clean" / "utt_00000000.wav"
    )
    assert (published / generator.RELOAD_CLEAN_SCP).read_text(
        encoding="utf-8"
    ).split()[3] == f"{generator._ROOT_ALIAS}:clean/utt_00000000.wav"

    report = run_offline_reload_gates(
        published, manifest=manifest
    )
    assert report.scan_mode == "full"
    assert report.publish_eligible is True
    assert report.gates[0].dataset_size == report.gates[0].scanned == 2
    receipt = json.loads((published / RECEIPT_FILE).read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["generation_id"] == manifest.generation_id
    assert receipt["root_digests"].keys() == {generator._ROOT_ALIAS}
    assert receipt["reload_report"]["scan_mode"] == "full"
    assert receipt["reload_report"]["publish_eligible"] is True


def test_probe_cannot_publish_se_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, published, _dataset = _run(tmp_path, monkeypatch)
    assert result == 0

    manifest = load_generation_manifest(
        published, generator.ENVELOPE_MANIFEST_NAME
    )
    probe = run_offline_reload_gates(published, manifest=manifest, probe=1)

    assert probe.publish_eligible is False
    with pytest.raises(OfflineReloadError, match="full scan"):
        require_publish_eligible(probe)


def test_final_reload_failure_restores_old_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, published, _dataset = _run(tmp_path, monkeypatch)
    assert result == 0
    sentinel = published / "previous-success.txt"
    sentinel.write_text("keep", encoding="utf-8")
    old_manifest = (published / "generation_manifest.json").read_bytes()
    old_receipt = (published / RECEIPT_FILE).read_bytes()

    monkeypatch.setattr(
        "tools.prepare_data_tools.offline_publication.run_offline_reload_gates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OfflineReloadError("injected final reload failure")
        ),
    )
    result, _, _dataset = _run(tmp_path, monkeypatch, published)

    assert result == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (published / "generation_manifest.json").read_bytes() == old_manifest
    assert (published / RECEIPT_FILE).read_bytes() == old_receipt
    assert not list(tmp_path.glob(f".{published.name}.backup-*"))
    assert not list(tmp_path.glob(f".{published.name}.owned-staging-*"))


def test_final_reload_failure_without_old_target_leaves_no_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fixture_dataset(monkeypatch)
    published = tmp_path / "published"
    monkeypatch.setattr(
        "tools.prepare_data_tools.offline_publication.run_offline_reload_gates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OfflineReloadError("injected final reload failure")
        ),
    )

    result = generator.main(
        ["-C", "config.yaml", "--output-dir", str(published), "--num-samples", "2"]
    )

    assert result == 1
    assert not published.exists()
    assert not list(tmp_path.glob(f".{published.name}.owned-staging-*"))
