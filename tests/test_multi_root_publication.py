"""GenerationManifest v2 multi-target owned publication transaction 合同。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_data_tools.generation_manifest import (
    GenerationManifest,
    sha256_file,
    write_generation_root_marker,
)
from tools.prepare_data_tools.offline_publication import (
    OWNER_FILE,
    RECEIPT_FILE,
    ExistingOwnedRoot,
    MultiRootPublicationTarget,
    OfflinePublicationError,
    OwnedPublicationPaths,
    PublicationCommittedError,
    publish_owned_offline_dataset_roots,
    write_ownership_marker,
)
from tools.prepare_data_tools.offline_reload_gate import OfflineReloadError

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_OWNER = "tests/multi-root-publication"


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.linspace(-0.2, 0.2, 64, dtype=np.float32), 16000, subtype="FLOAT")


def _inventory_entry(alias: str, root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "root_alias": alias,
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "kind": "waveform" if path.suffix == ".wav" else "metadata",
    }


def _make_v2_staging(
    indexes: Path,
    waveforms: Path,
    *,
    generation_id: str = "tests-generation-new",
    existing_waveforms: bool = False,
) -> GenerationManifest:
    if not existing_waveforms:
        waveforms.mkdir()
        write_ownership_marker(waveforms, owner_id=_OWNER)
        _write_wav(waveforms / "clean.wav")
        _write_wav(waveforms / "noisy.wav")
    indexes.mkdir()
    write_ownership_marker(indexes, owner_id=_OWNER)
    (indexes / "clean.scp").write_text(
        "utt 16000 64 waveforms:clean.wav\n", encoding="utf-8"
    )
    (indexes / "noisy.scp").write_text(
        "utt 16000 64 waveforms:noisy.wav\n", encoding="utf-8"
    )
    manifest = GenerationManifest.from_dict(
        {
            "schema_version": 2,
            "generation_id": generation_id,
            "status": "complete",
            "artifact_kind": "se-multi-root-offline",
            "task": "se",
            "generator": {"id": "tests/multi-root-publication", "version": "1.0"},
            "root_bindings": [
                {"alias": "indexes", "role": "manifest"},
                {"alias": "waveforms", "role": "resource"},
            ],
            "splits": ["train"],
            "parameters": {},
            "source_identities": [],
            "inventory": [
                _inventory_entry("indexes", indexes, "clean.scp"),
                _inventory_entry("indexes", indexes, "noisy.scp"),
                _inventory_entry("waveforms", waveforms, "clean.wav"),
                _inventory_entry("waveforms", waveforms, "noisy.wav"),
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
    )
    (indexes / "generation_manifest.json").write_text(
        manifest.to_json(), encoding="utf-8"
    )
    write_generation_root_marker(
        indexes, manifest=manifest, root_alias="indexes", owner_id=_OWNER
    )
    write_generation_root_marker(
        waveforms, manifest=manifest, root_alias="waveforms", owner_id=_OWNER
    )
    return manifest


def _paths(parent: Path, prefix: str) -> OwnedPublicationPaths:
    parent.mkdir()
    return OwnedPublicationPaths(
        allowed_parent=parent,
        staging=parent / f"{prefix}-staging",
        target=parent / f"{prefix}-target",
        backup=parent / f"{prefix}-backup",
        owner_id=_OWNER,
    )


def _transaction(tmp_path: Path, *, old_targets: bool = True):
    index_paths = _paths(tmp_path / "index-parent", "indexes")
    wave_paths = _paths(tmp_path / "wave-parent", "waveforms")
    manifest = _make_v2_staging(index_paths.staging, wave_paths.staging)
    if old_targets:
        _make_old_target(index_paths.target, "indexes", "old-indexes")
        _make_old_target(wave_paths.target, "waveforms", "old-waveforms")
    targets = (
        MultiRootPublicationTarget(alias="waveforms", paths=wave_paths),
        MultiRootPublicationTarget(alias="indexes", paths=index_paths),
    )
    return index_paths, wave_paths, targets, manifest


def _make_old_target(root: Path, alias: str, sentinel: str) -> None:
    root.mkdir()
    write_ownership_marker(root, owner_id=_OWNER)
    (root / ".generation-root.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner_id": _OWNER,
                "generation_id": "tests-generation-old",
                "manifest_digest": "0" * 64,
                "root_alias": alias,
                "resource_digest": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "old.txt").write_text(sentinel, encoding="utf-8")


def test_multi_target_success_full_reloads_receipts_and_cleans_backups(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)

    receipt = publish_owned_offline_dataset_roots(
        targets, manifest_root_alias="indexes"
    )

    assert receipt.generation_id == "tests-generation-new"
    assert receipt.reload_report.publish_eligible is True
    assert receipt.reload_report.scan_mode == "full"
    assert set(receipt.root_digests) == {"indexes", "waveforms"}
    payload = json.loads((index_paths.target / RECEIPT_FILE).read_text(encoding="utf-8"))
    assert payload["generation_id"] == receipt.generation_id
    assert not wave_paths.staging.exists()
    assert not index_paths.staging.exists()
    assert not wave_paths.backup.exists()
    assert not index_paths.backup.exists()
    assert not (wave_paths.target / "old.txt").exists()


def test_multi_target_success_without_old_targets(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(
        tmp_path, old_targets=False
    )

    publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")

    assert index_paths.target.is_dir()
    assert wave_paths.target.is_dir()
    assert not index_paths.backup.exists()
    assert not wave_paths.backup.exists()


def test_preexisting_backup_rejected_before_any_target_changes(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    wave_paths.backup.mkdir()
    sentinel = wave_paths.backup / "foreign.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(OfflinePublicationError, match="backup 必须"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")

    assert (index_paths.target / "old.txt").read_text() == "old-indexes"
    assert (wave_paths.target / "old.txt").read_text() == "old-waveforms"
    assert sentinel.read_text() == "keep"


def test_partial_rename_failure_rolls_back_every_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    real_replace = os.replace

    def fail_wave_publish(source, destination):
        if Path(source) == wave_paths.staging and Path(destination) == wave_paths.target:
            raise OSError("injected waveform rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr("tools.prepare_data_tools.offline_publication.os.replace", fail_wave_publish)
    with pytest.raises(OfflinePublicationError, match="全部 roots 已回滚"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")

    assert (index_paths.target / "old.txt").read_text() == "old-indexes"
    assert (wave_paths.target / "old.txt").read_text() == "old-waveforms"
    assert index_paths.staging.is_dir()
    assert wave_paths.staging.is_dir()
    assert not index_paths.backup.exists()
    assert not wave_paths.backup.exists()


@pytest.mark.parametrize("failure", ["reload", "receipt", "cleanup"])
def test_final_phase_failure_removes_receipt_and_rolls_back_all_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    module = "tools.prepare_data_tools.offline_publication"
    if failure == "reload":
        monkeypatch.setattr(
            f"{module}.run_offline_reload_gates",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OfflineReloadError("injected reload failure")
            ),
        )
    elif failure == "receipt":
        monkeypatch.setattr(
            f"{module}._write_receipt_atomically",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected receipt failure")
            ),
        )
    else:
        monkeypatch.setattr(
            f"{module}._remove_declared_owned_tree",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("injected cleanup failure before deletion")
            ),
        )

    with pytest.raises(OfflinePublicationError, match="全部 roots 已回滚"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")

    assert (index_paths.target / "old.txt").read_text() == "old-indexes"
    assert (wave_paths.target / "old.txt").read_text() == "old-waveforms"
    assert index_paths.staging.is_dir()
    assert wave_paths.staging.is_dir()
    assert not (index_paths.target / RECEIPT_FILE).exists()


def test_partial_cleanup_failure_is_reported_as_committed_not_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    from tools.prepare_data_tools import offline_publication as publication

    real_remove = publication._remove_declared_owned_tree
    calls = 0

    def fail_second_cleanup(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second cleanup failure")
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(publication, "_remove_declared_owned_tree", fail_second_cleanup)
    with pytest.raises(PublicationCommittedError, match="已提交"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")

    assert (index_paths.target / RECEIPT_FILE).is_file()
    assert not (index_paths.target / "old.txt").exists()
    assert not index_paths.backup.exists()
    assert wave_paths.backup.is_dir()
    assert not index_paths.staging.exists()
    assert not wave_paths.staging.exists()


def test_symlink_old_target_and_cleanup_symlink_fail_before_rename(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(
        tmp_path, old_targets=False
    )
    referent = tmp_path / "referent"
    referent.mkdir()
    sentinel = referent / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    wave_paths.target.symlink_to(referent, target_is_directory=True)

    with pytest.raises(OfflinePublicationError, match="symlink"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")
    assert sentinel.read_text() == "keep"

    wave_paths.target.unlink()
    _make_old_target(index_paths.target, "indexes", "old-indexes")
    external = tmp_path / "external"
    external.write_text("external", encoding="utf-8")
    (index_paths.target / "linked").symlink_to(external)
    with pytest.raises(OfflinePublicationError, match="cleanup 拒绝 symlink"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")
    assert (index_paths.target / "old.txt").read_text() == "old-indexes"


def test_root_swap_and_forged_generation_marker_fail_before_rename(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    swapped = (
        MultiRootPublicationTarget(alias="indexes", paths=wave_paths),
        MultiRootPublicationTarget(alias="waveforms", paths=index_paths),
    )
    with pytest.raises(
        OfflinePublicationError, match="静态校验|identity|root_alias 不匹配"
    ):
        publish_owned_offline_dataset_roots(swapped, manifest_root_alias="indexes")
    assert (index_paths.target / "old.txt").read_text() == "old-indexes"

    marker = wave_paths.staging / ".generation-root.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["generation_id"] = "forged"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfflinePublicationError, match="静态校验|identity"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")
    assert (wave_paths.target / "old.txt").read_text() == "old-waveforms"


def test_existing_waveform_root_same_identity_supports_scp_only(tmp_path: Path) -> None:
    index_paths = _paths(tmp_path / "index-parent", "indexes")
    wave_parent = tmp_path / "wave-parent"
    wave_parent.mkdir()
    waveform_root = wave_parent / "waveforms-existing"
    waveform_root.mkdir()
    write_ownership_marker(waveform_root, owner_id=_OWNER)
    _write_wav(waveform_root / "clean.wav")
    _write_wav(waveform_root / "noisy.wav")
    manifest = _make_v2_staging(
        index_paths.staging, waveform_root, existing_waveforms=True
    )
    assert manifest.generation_id == "tests-generation-new"

    receipt = publish_owned_offline_dataset_roots(
        (MultiRootPublicationTarget(alias="indexes", paths=index_paths),),
        manifest_root_alias="indexes",
        existing_roots=(
            ExistingOwnedRoot(
                alias="waveforms",
                allowed_parent=wave_parent,
                root=waveform_root,
                owner_id=_OWNER,
            ),
        ),
    )

    assert receipt.reload_report.publish_eligible is True
    assert waveform_root.is_dir()
    assert (index_paths.target / RECEIPT_FILE).is_file()


def test_existing_waveform_without_generation_marker_fails_closed(tmp_path: Path) -> None:
    index_paths = _paths(tmp_path / "index-parent", "indexes")
    wave_parent = tmp_path / "wave-parent"
    wave_parent.mkdir()
    waveform_root = wave_parent / "legacy-waveforms"
    waveform_root.mkdir()
    write_ownership_marker(waveform_root, owner_id=_OWNER)
    _write_wav(waveform_root / "clean.wav")
    _write_wav(waveform_root / "noisy.wav")
    _make_v2_staging(index_paths.staging, waveform_root, existing_waveforms=True)
    (waveform_root / ".generation-root.json").unlink()

    with pytest.raises(OfflinePublicationError, match="generation marker|静态校验"):
        publish_owned_offline_dataset_roots(
            (MultiRootPublicationTarget(alias="indexes", paths=index_paths),),
            manifest_root_alias="indexes",
            existing_roots=(
                ExistingOwnedRoot(
                    alias="waveforms",
                    allowed_parent=wave_parent,
                    root=waveform_root,
                    owner_id=_OWNER,
                ),
            ),
        )
    assert not index_paths.target.exists()
    assert index_paths.staging.exists()


def test_existing_allowed_parent_symlink_fails_before_rename(tmp_path: Path) -> None:
    index_paths = _paths(tmp_path / "index-parent", "indexes")
    wave_parent = tmp_path / "wave-parent"
    wave_parent.mkdir()
    waveform_root = wave_parent / "waveforms-existing"
    waveform_root.mkdir()
    write_ownership_marker(waveform_root, owner_id=_OWNER)
    _write_wav(waveform_root / "clean.wav")
    _write_wav(waveform_root / "noisy.wav")
    _make_v2_staging(index_paths.staging, waveform_root, existing_waveforms=True)
    linked_parent = tmp_path / "wave-parent-link"
    linked_parent.symlink_to(wave_parent, target_is_directory=True)

    with pytest.raises(OfflinePublicationError, match="existing parent.*symlink"):
        publish_owned_offline_dataset_roots(
            (MultiRootPublicationTarget(alias="indexes", paths=index_paths),),
            manifest_root_alias="indexes",
            existing_roots=(
                ExistingOwnedRoot(
                    alias="waveforms",
                    allowed_parent=linked_parent,
                    root=linked_parent / waveform_root.name,
                    owner_id=_OWNER,
                ),
            ),
        )
    assert not index_paths.target.exists()
    assert index_paths.staging.exists()


def test_hardlink_and_special_node_fail_before_any_rename(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    hardlink = wave_paths.staging / "hardlink.wav"
    hardlink.hardlink_to(wave_paths.staging / "clean.wav")

    with pytest.raises(OfflinePublicationError, match="hardlink/special"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")
    assert (index_paths.target / "old.txt").read_text() == "old-indexes"
    assert (wave_paths.target / "old.txt").read_text() == "old-waveforms"

    hardlink.unlink()
    fifo = wave_paths.staging / "special.fifo"
    os.mkfifo(fifo)
    with pytest.raises(OfflinePublicationError, match="hardlink/special"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")
    assert (index_paths.target / "old.txt").read_text() == "old-indexes"


def test_old_targets_from_different_generations_fail_before_rename(tmp_path: Path) -> None:
    index_paths, wave_paths, targets, _manifest = _transaction(tmp_path)
    marker = wave_paths.target / ".generation-root.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["generation_id"] = "different-old-generation"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OfflinePublicationError, match="旧 targets generation identity 不一致"):
        publish_owned_offline_dataset_roots(targets, manifest_root_alias="indexes")
    assert (index_paths.target / "old.txt").read_text() == "old-indexes"
    assert (wave_paths.target / "old.txt").read_text() == "old-waveforms"


def test_all_paths_must_be_direct_children_of_their_allowed_parent(tmp_path: Path) -> None:
    index_paths, wave_paths, _targets, _manifest = _transaction(tmp_path)
    invalid = OwnedPublicationPaths(
        allowed_parent=index_paths.allowed_parent,
        staging=index_paths.staging,
        target=index_paths.allowed_parent / "nested" / "target",
        backup=index_paths.backup,
        owner_id=_OWNER,
    )
    with pytest.raises(OfflinePublicationError, match="直接子路径"):
        publish_owned_offline_dataset_roots(
            (
                MultiRootPublicationTarget(alias="indexes", paths=invalid),
                MultiRootPublicationTarget(alias="waveforms", paths=wave_paths),
            ),
            manifest_root_alias="indexes",
        )
