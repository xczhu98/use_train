"""共享文件事务 helper 的 commit、权限与 symlink 契约测试。"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import core.utils.file_transaction as file_transaction
from core.utils.file_transaction import (
    publish_staged_files,
    write_text_files_transactionally,
)


def test_preserves_existing_mode_and_uses_0644_for_new_files(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    created = tmp_path / "created.txt"
    existing.write_text("old", encoding="utf-8")
    existing.chmod(0o640)

    write_text_files_transactionally(
        [
            (existing, lambda output: output.write("new")),
            (created, lambda output: output.write("created")),
        ],
        overwrite=True,
    )

    assert existing.read_text(encoding="utf-8") == "new"
    assert created.read_text(encoding="utf-8") == "created"
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640
    assert stat.S_IMODE(created.stat().st_mode) == 0o644


def test_rejects_leaf_symlink_without_touching_referent(tmp_path: Path) -> None:
    referent = tmp_path / "referent.txt"
    target = tmp_path / "target.txt"
    referent.write_text("old", encoding="utf-8")
    target.symlink_to(referent)

    with pytest.raises(RuntimeError, match="symbolic links"):
        write_text_files_transactionally(
            [(target, lambda output: output.write("new"))], overwrite=True
        )

    assert target.is_symlink()
    assert referent.read_text(encoding="utf-8") == "old"


def test_rejects_symlink_ancestor_without_touching_referent(tmp_path: Path) -> None:
    referent = tmp_path / "referent"
    referent.mkdir()
    target = referent / "target.txt"
    link = tmp_path / "link"
    link.symlink_to(referent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="ancestors"):
        write_text_files_transactionally(
            [(link / "target.txt", lambda output: output.write("new"))],
            overwrite=True,
        )

    assert not target.exists()
    assert not list(referent.iterdir())


def test_rejects_non_directory_ancestor(tmp_path: Path) -> None:
    ancestor = tmp_path / "not-a-directory"
    ancestor.write_text("old", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="ancestor"):
        write_text_files_transactionally(
            [(ancestor / "target.txt", lambda output: output.write("new"))],
            overwrite=True,
        )

    assert ancestor.read_text(encoding="utf-8") == "old"


def test_publishes_multiple_staged_files_normally(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    staged_first = tmp_path / "first.staged"
    staged_second = tmp_path / "second.staged"
    staged_first.write_text("one", encoding="utf-8")
    staged_second.write_text("two", encoding="utf-8")

    publish_staged_files([staged_first, staged_second], [first, second])

    assert first.read_text(encoding="utf-8") == "one"
    assert second.read_text(encoding="utf-8") == "two"
    assert not staged_first.exists()
    assert not staged_second.exists()


def test_uses_original_parent_fd_after_parent_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "outputs"
    parent.mkdir()
    target = parent / "target.txt"
    staged = parent / "staged.tmp"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    moved_parent = tmp_path / "outputs-moved"
    real_backup = file_transaction._backup_existing_target_at

    def replace_parent(target_path: Path, directory_fds: dict[Path, int]) -> Path | None:
        backup = real_backup(target_path, directory_fds)
        os.rename(parent, moved_parent)
        parent.mkdir()
        return backup

    monkeypatch.setattr(file_transaction, "_backup_existing_target_at", replace_parent)
    publish_staged_files([staged], [target])

    assert (moved_parent / "target.txt").read_text(encoding="utf-8") == "new"
    assert not target.exists()
    assert parent.is_dir()


def test_rolls_back_all_published_targets_when_a_later_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    staged_first = tmp_path / "first.staged"
    staged_second = tmp_path / "second.staged"
    first.write_text("old first", encoding="utf-8")
    second.write_text("old second", encoding="utf-8")
    staged_first.write_text("new first", encoding="utf-8")
    staged_second.write_text("new second", encoding="utf-8")
    calls = 0
    real_replace = file_transaction._replace_at

    def replace_with_failure(
        source: Path, target: Path, directory_fds: dict[Path, int]
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        real_replace(source, target, directory_fds)

    monkeypatch.setattr(file_transaction, "_replace_at", replace_with_failure)
    with pytest.raises(OSError, match="injected publish failure"):
        publish_staged_files([staged_first, staged_second], [first, second])

    assert first.read_text(encoding="utf-8") == "old first"
    assert second.read_text(encoding="utf-8") == "old second"
    assert not list(tmp_path.glob(".*.bak"))


def test_reports_cleanup_failure_as_committed(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.tmp"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")

    def fail_backup_cleanup(path: Path) -> None:
        if path.suffix == ".bak":
            raise OSError("injected cleanup failure")
        path.unlink(missing_ok=True)

    with pytest.raises(RuntimeError, match="publish committed.*backup cleanup failed"):
        publish_staged_files(
            [staged],
            [target],
            unlink_if_exists=fail_backup_cleanup,
        )

    assert target.read_text(encoding="utf-8") == "new"
    backups = list(tmp_path.glob(".*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink", "hardlink"])
def test_rejects_invalid_staged_leaf_before_callbacks(
    tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.tmp"
    referent = tmp_path / "referent.txt"
    target.write_text("old", encoding="utf-8")
    if kind == "fifo":
        os.mkfifo(staged)
    elif kind == "directory":
        staged.mkdir()
    elif kind == "symlink":
        referent.write_text("new", encoding="utf-8")
        staged.symlink_to(referent)
    else:
        referent.write_text("new", encoding="utf-8")
        os.link(referent, staged)

    def unexpected_callback(*args) -> None:
        raise AssertionError(f"callback must not run for invalid staged leaf: {args!r}")

    with pytest.raises(
        RuntimeError,
        match="staging files must be regular files with exactly one hard link",
    ):
        publish_staged_files(
            [staged],
            [target],
            replace=unexpected_callback,
            backup_existing=unexpected_callback,
            unlink_if_exists=unexpected_callback,
            fsync_parent_directories=unexpected_callback,
        )

    assert target.read_text(encoding="utf-8") == "old"
    assert staged.exists() or staged.is_symlink()


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_rejects_special_target_leaf_before_callbacks(
    tmp_path: Path, kind: str
) -> None:
    staged = tmp_path / "staged.tmp"
    target = tmp_path / "target.txt"
    staged.write_text("new", encoding="utf-8")
    if kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()

    def unexpected_callback(*args) -> None:
        raise AssertionError(f"callback must not run for special target: {args!r}")

    with pytest.raises(RuntimeError, match="transaction target is not a regular file"):
        publish_staged_files(
            [staged],
            [target],
            replace=unexpected_callback,
            backup_existing=unexpected_callback,
        )

    assert staged.read_text(encoding="utf-8") == "new"
    assert target.exists()


def test_rejects_same_staged_and_target_path_before_publish(tmp_path: Path) -> None:
    shared = tmp_path / "shared.txt"
    shared.write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="staging files and targets must be different"):
        publish_staged_files([shared], [shared])

    assert shared.read_text(encoding="utf-8") == "old"


def test_rejects_staged_target_physical_alias_before_publish(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.tmp"
    target.write_text("old", encoding="utf-8")
    os.link(target, staged)

    with pytest.raises(
        ValueError,
        match="staging files and targets must refer to different physical files",
    ):
        publish_staged_files([staged], [target])

    assert target.read_text(encoding="utf-8") == "old"
    assert staged.read_text(encoding="utf-8") == "old"


def test_rejects_staging_path_replaced_after_backup_callback(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.tmp"
    original_staged = tmp_path / "original-staged.tmp"
    attacker = tmp_path / "attacker.tmp"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    attacker.write_text("attacker", encoding="utf-8")

    def replace_staging(_target: Path) -> None:
        staged.replace(original_staged)
        attacker.replace(staged)
        return None

    with pytest.raises(RuntimeError, match="staging file identity changed"):
        publish_staged_files(
            [staged],
            [target],
            backup_existing=replace_staging,
        )

    assert target.read_text(encoding="utf-8") == "old"
    assert staged.read_text(encoding="utf-8") == "attacker"
    assert original_staged.read_text(encoding="utf-8") == "new"


def test_rolls_back_if_replace_publishes_wrong_staging_identity(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    staged = tmp_path / "staged.tmp"
    expected_published = tmp_path / "expected-published.tmp"
    attacker = tmp_path / "attacker.tmp"
    target.write_text("old", encoding="utf-8")
    staged.write_text("new", encoding="utf-8")
    attacker.write_text("attacker", encoding="utf-8")
    calls = 0

    def swap_after_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        source.replace(destination)
        if calls == 1:
            destination.replace(expected_published)
            attacker.replace(destination)

    with pytest.raises(RuntimeError, match="bound staging identity"):
        publish_staged_files([staged], [target], replace=swap_after_replace)

    assert target.read_text(encoding="utf-8") == "old"
    assert expected_published.read_text(encoding="utf-8") == "new"


pytestmark = [pytest.mark.unit]
