"""descriptor-bound 输出路径与临时清理安全契约测试。"""

import os
import stat
import time

import pytest

import audio_filter_pipeline.utils.path_utils as path_utils
from audio_filter_pipeline.utils.path_utils import (
    cleanup_expired_temp,
    open_safe_directory,
    open_safe_output_path,
)


pytestmark = pytest.mark.unit


def test_open_safe_output_path_exposes_bound_directory_contract(tmp_path):
    root = tmp_path / "output"
    target = root / "nested" / "bundle.jsonl"

    safe = open_safe_output_path(str(root), str(target), create=True)
    try:
        assert stat.S_ISDIR(os.fstat(safe.root_fd).st_mode)
        assert stat.S_ISDIR(os.fstat(safe.parent_fd).st_mode)
        assert safe.target_name == "bundle.jsonl"
        assert safe.canonical_target == str(target)
        assert safe.target_stat() is None

        descriptor, temporary_name = safe.create_temp_file(".probe.")
        os.close(descriptor)
        os.unlink(temporary_name, dir_fd=safe.parent_fd)
    finally:
        safe.close()


def test_safe_output_path_rejects_access_after_close(tmp_path):
    safe = open_safe_output_path(
        str(tmp_path / "output"),
        str(tmp_path / "output" / "bundle.jsonl"),
        create=True,
    )
    safe.close()

    with pytest.raises(ValueError, match="closed"):
        _ = safe.root_fd
    with pytest.raises(ValueError, match="closed"):
        _ = safe.parent_fd
    with pytest.raises(ValueError, match="closed"):
        safe.verify()
    with pytest.raises(ValueError, match="closed"):
        safe.target_stat()


def test_safe_output_path_verify_fails_after_parent_replacement(tmp_path):
    root = tmp_path / "output"
    parent = root / "nested"
    target = parent / "bundle.jsonl"
    safe = open_safe_output_path(str(root), str(target), create=True)
    try:
        parent.rename(root / "old-nested")
        parent.mkdir()
        with pytest.raises(ValueError, match="changed"):
            safe.verify()
    finally:
        safe.close()


def test_open_safe_directory_rejects_root_and_deep_symlinks(tmp_path):
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        open_safe_directory(str(linked_root))

    output_root = tmp_path / "output"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = output_root / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        open_safe_output_path(
            str(output_root),
            str(linked_parent / "bundle.jsonl"),
            create=True,
        )

    missing_under_link = linked_parent / "missing" / "temp"
    with pytest.raises(ValueError, match="symlink"):
        cleanup_expired_temp(str(missing_under_link))


@pytest.mark.parametrize("entry_kind", ["file", "fifo"])
def test_open_safe_directory_rejects_non_directory_components(tmp_path, entry_kind):
    root = tmp_path / "output"
    root.mkdir()
    entry = root / "not-directory"
    if entry_kind == "file":
        entry.write_text("not a directory", encoding="utf-8")
    else:
        os.mkfifo(entry)

    with pytest.raises(ValueError, match="directory"):
        open_safe_directory(str(entry))


def test_open_safe_output_path_rejects_symlink_directory_and_fifo_targets(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    symlink_target = root / "symlink.jsonl"
    symlink_target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        open_safe_output_path(str(root), str(symlink_target), create=True)

    directory_target = root / "directory.jsonl"
    directory_target.mkdir()
    with pytest.raises(ValueError, match="unsupported type"):
        open_safe_output_path(str(root), str(directory_target), create=True)

    fifo_target = root / "fifo.jsonl"
    os.mkfifo(fifo_target)
    with pytest.raises(ValueError, match="unsupported type"):
        open_safe_output_path(str(root), str(fifo_target), create=True)


def test_safe_output_path_link_from_rejects_path_components(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "source.jsonl").write_text("source", encoding="utf-8")
    source_fd = open_safe_directory(str(source_root))
    try:
        output_root = tmp_path / "output"
        target = output_root / "bundle.jsonl"
        with open_safe_output_path(str(output_root), str(target), create=True) as safe:
            for unsafe_name in ("../source.jsonl", "nested/source.jsonl", ".."):
                with pytest.raises(ValueError, match="basename"):
                    safe.link_from(source_fd, unsafe_name)
    finally:
        os.close(source_fd)


def test_safe_output_path_rename_from_noreplace_preserves_source_on_collision(tmp_path):
    root = tmp_path / "output"
    source_path = root / "staging" / "audio.tmp"
    destination_path = root / "published.flac"
    source_safe = open_safe_output_path(str(root), str(source_path), create=True)
    destination_safe = open_safe_output_path(str(root), str(destination_path), create=True)
    try:
        descriptor = source_safe.open_target(os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, b"owned audio")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        source_identity = (source_path.stat().st_dev, source_path.stat().st_ino)

        destination_safe.rename_from_noreplace(source_safe)

        assert not source_path.exists()
        assert destination_path.read_bytes() == b"owned audio"
        published = destination_path.stat()
        assert (published.st_dev, published.st_ino) == source_identity
        assert published.st_nlink == 1

        descriptor = source_safe.open_target(os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        with pytest.raises(FileExistsError):
            destination_safe.rename_from_noreplace(source_safe)
        assert source_path.is_file()
        assert destination_path.read_bytes() == b"owned audio"
    finally:
        source_safe.close()
        destination_safe.close()


def test_safe_output_path_unlink_keeps_competing_replacement(tmp_path, monkeypatch):
    root = tmp_path / "output"
    target = root / "bundle.jsonl"
    target.parent.mkdir()
    target.write_text("owned", encoding="utf-8")

    safe = open_safe_output_path(str(root), str(target), create=True)
    try:
        identity = (target.stat().st_dev, target.stat().st_ino)
        real_rename = path_utils._rename_noreplace
        injected = False

        def rename_then_replace(source_fd, source_name, target_fd, target_name):
            nonlocal injected
            real_rename(source_fd, source_name, target_fd, target_name)
            if not injected and source_name == target.name and target_name == "entry":
                injected = True
                descriptor = os.open(
                    source_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=source_fd,
                )
                try:
                    os.write(descriptor, b"competitor")
                finally:
                    os.close(descriptor)

        monkeypatch.setattr(path_utils, "_rename_noreplace", rename_then_replace)
        with pytest.raises(ValueError, match="replaced|changed"):
            safe.unlink_target(expected_identity=identity)
        assert target.read_text(encoding="utf-8") == "competitor"
        assert not list(root.glob(f".{target.name}.cleanup-*"))
    finally:
        safe.close()


def test_cleanup_keeps_competing_replacement_after_quarantine_move(tmp_path, monkeypatch):
    root = tmp_path / "temp"
    root.mkdir()
    candidate = root / "audio-filter-race"
    candidate.write_text("owned", encoding="utf-8")
    old_timestamp = time.time() - 48 * 3600
    os.utime(candidate, (old_timestamp, old_timestamp))

    real_rename = path_utils._rename_noreplace
    injected = False

    def rename_then_replace(source_fd, source_name, target_fd, target_name):
        nonlocal injected
        real_rename(source_fd, source_name, target_fd, target_name)
        if not injected and source_name == candidate.name:
            injected = True
            descriptor = os.open(
                source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=source_fd,
            )
            try:
                os.write(descriptor, b"competitor")
            finally:
                os.close(descriptor)

    monkeypatch.setattr(path_utils, "_rename_noreplace", rename_then_replace)

    assert cleanup_expired_temp(str(root), max_age_hours=24) == 1
    assert candidate.read_text(encoding="utf-8") == "competitor"
    assert not list(root.glob(f".{candidate.name}.cleanup-*"))


def test_quarantine_entry_replacement_is_not_deleted(tmp_path, monkeypatch):
    root = tmp_path / "temp"
    root.mkdir()
    candidate = root / "audio-filter-entry-race"
    candidate.write_text("owned", encoding="utf-8")
    parent_fd = open_safe_directory(str(root))
    try:
        real_open = path_utils._open_cleanup_entry_fd
        injected = False

        def open_then_replace(parent, name, label):
            nonlocal injected
            descriptor = real_open(parent, name, label)
            if name == "entry" and not injected:
                injected = True
                os.unlink(name, dir_fd=parent)
                replacement_fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent,
                )
                try:
                    os.write(replacement_fd, b"competitor")
                finally:
                    os.close(replacement_fd)
            return descriptor

        monkeypatch.setattr(path_utils, "_open_cleanup_entry_fd", open_then_replace)
        assert not path_utils._quarantine_remove_entry(
            parent_fd,
            candidate.name,
            str(candidate),
        )
        quarantines = list(root.glob(f".{candidate.name}.cleanup-*"))
        assert quarantines
        assert (quarantines[0] / "entry").read_text(encoding="utf-8") == "competitor"
        assert not candidate.exists()
    finally:
        os.close(parent_fd)


def test_quarantine_claimed_tomb_is_reopened_and_foreign_replacement_is_preserved(
    tmp_path, monkeypatch,
):
    root = tmp_path / "temp"
    root.mkdir()
    candidate = root / "audio-filter-tomb-race"
    candidate.write_text("owned", encoding="utf-8")
    parent_fd = open_safe_directory(str(root))
    real_open = path_utils._open_cleanup_entry_fd
    reopened_tombs = []

    def open_then_replace(parent, name, label):
        descriptor = real_open(parent, name, label)
        if name.startswith(".entry.remove-"):
            reopened_tombs.append(name)
            os.unlink(name, dir_fd=parent)
            replacement_fd = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=parent,
            )
            try:
                os.write(replacement_fd, b"competitor")
            finally:
                os.close(replacement_fd)
        return descriptor

    try:
        monkeypatch.setattr(path_utils, "_open_cleanup_entry_fd", open_then_replace)
        assert not path_utils._quarantine_remove_entry(
            parent_fd, candidate.name, str(candidate),
        )
        assert reopened_tombs
        quarantines = list(root.glob(f".{candidate.name}.cleanup-*"))
        assert len(quarantines) == 1
        tombs = list(quarantines[0].glob(".entry.remove-*"))
        assert len(tombs) == 1
        assert tombs[0].read_bytes() == b"competitor"
    finally:
        os.close(parent_fd)


def test_quarantine_directory_replacement_is_not_removed(tmp_path, monkeypatch):
    root = tmp_path / "temp"
    root.mkdir()
    quarantine = root / ".quarantine"
    quarantine.mkdir()
    parent_fd = open_safe_directory(str(root))
    quarantine_fd = os.open(
        quarantine.name,
        path_utils._TEMP_DIRECTORY_FLAGS,
        dir_fd=parent_fd,
    )
    identity_stat = os.fstat(quarantine_fd)
    identity = (identity_stat.st_dev, identity_stat.st_ino)
    real_stat = path_utils._cleanup_stat_at
    injected = False
    try:
        def stat_then_replace(parent, name):
            nonlocal injected
            value = real_stat(parent, name)
            if not injected:
                injected = True
                quarantine.rename(root / ".quarantine-held")
                (root / name).mkdir()
            return value

        monkeypatch.setattr(path_utils, "_cleanup_stat_at", stat_then_replace)
        assert not path_utils._cleanup_quarantine_directory(
            parent_fd,
            quarantine_fd,
            quarantine.name,
            identity,
        )
        assert (root / ".quarantine").is_dir()
        assert (root / ".quarantine-held").is_dir()
    finally:
        os.close(quarantine_fd)
        os.close(parent_fd)


def test_safe_output_path_close_attempts_all_descriptors_after_base_exception(
    tmp_path, monkeypatch,
):
    safe = open_safe_output_path(
        str(tmp_path / "output"),
        str(tmp_path / "output" / "nested" / "bundle.jsonl"),
        create=True,
    )
    descriptors = tuple(safe._descriptors)
    real_close = path_utils.os.close
    closed = []

    def close_then_raise(fd):
        closed.append(fd)
        real_close(fd)
        if fd == descriptors[-1]:
            raise KeyboardInterrupt("injected close failure")

    monkeypatch.setattr(path_utils.os, "close", close_then_raise)
    with pytest.raises(KeyboardInterrupt, match="close failure"):
        safe.close()

    assert closed == list(reversed(descriptors))
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
