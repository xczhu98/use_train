from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from tools.ops.data import reorganize_noise_datasets as reorganize

main = reorganize.main

pytestmark = pytest.mark.unit


def _write_source(
    source: Path,
    sample_rate: str,
    noise_type: str,
    dataset: str,
    relative: str,
    content: bytes,
) -> Path:
    path = source / sample_rate / noise_type / dataset / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _run(source: Path, output: Path, *extra: str) -> int:
    return main(
        [
            "--source",
            str(source),
            "--output",
            str(output),
            "--workers",
            "2",
            *extra,
        ]
    )


def _assert_no_output_files(output: Path) -> None:
    if output.exists():
        assert not [path for path in output.rglob("*") if path.is_file() or path.is_symlink()]


def test_preserves_unicode_dataset_noise_type_and_filename(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "48k", "风声", "数据集甲", "分片/音频一.wav", b"audio")

    assert _run(source, output) == 0

    assert (output / "数据集甲/风声/音频一.wav").read_bytes() == b"audio"


def test_flattens_hash_shards_below_noise_type(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(
        source,
        "48k",
        "stationary",
        "dataset_a",
        "hash_a/hash_b/clip.wav",
        b"deep audio",
    )

    assert _run(source, output) == 0

    copied = output / "dataset_a/stationary/clip.wav"
    assert copied.read_bytes() == b"deep audio"
    assert copied.stat().st_mtime_ns == original.stat().st_mtime_ns
    assert not (output / "dataset_a/stationary/hash_a").exists()


def test_merges_noise_types_under_same_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "traffic", "dataset_a", "a.wav", b"traffic")
    _write_source(source, "16k", "wind", "dataset_a", "b.wav", b"wind")

    assert _run(source, output) == 0

    assert (output / "dataset_a/traffic/a.wav").read_bytes() == b"traffic"
    assert (output / "dataset_a/wind/b.wav").read_bytes() == b"wind"


def test_merges_sample_rates_under_same_noise_type(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "wind", "dataset_a", "aa/a.wav", b"16k")
    _write_source(source, "48k", "wind", "dataset_a", "bb/b.wav", b"48k")

    assert _run(source, output) == 0

    assert (output / "dataset_a/wind/a.wav").read_bytes() == b"16k"
    assert (output / "dataset_a/wind/b.wav").read_bytes() == b"48k"
    assert not (output / "dataset_a/16k").exists()
    assert not (output / "dataset_a/48k").exists()


def test_rejects_duplicate_names_after_flattening(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "wind", "dataset_a", "aa/clip.wav", b"first")
    _write_source(source, "48k", "wind", "dataset_a", "bb/clip.wav", b"second")

    assert _run(source, output) == 1

    captured = capsys.readouterr()
    assert "移除采样率和哈希分片后出现同名文件" in captured.err


def test_second_run_skips_matching_target(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(source, "16k", "wind", "dataset_a", "clip.wav", b"same")

    assert _run(source, output) == 0
    target = output / "dataset_a/wind/clip.wav"
    target.touch()
    assert target.stat().st_mtime_ns != original.stat().st_mtime_ns
    capsys.readouterr()
    assert _run(source, output) == 0

    captured = capsys.readouterr()
    assert "files=1" in captured.out
    assert "skipped=1" in captured.out


def test_same_size_and_mtime_but_different_content_is_not_skipped(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(source, "16k", "wind", "dataset_a", "clip.wav", b"source")
    target = output / "dataset_a/wind/clip.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target")
    os.utime(target, ns=(original.stat().st_atime_ns, original.stat().st_mtime_ns))

    assert _run(source, output) == 1
    assert target.read_bytes() == b"target"


def test_conflicting_target_fails_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "wind", "dataset_a", "clip.wav", b"source")
    target = output / "dataset_a/wind/clip.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"conflict")

    assert _run(source, output) == 1

    captured = capsys.readouterr()
    assert "目标文件冲突" in captured.err
    assert target.read_bytes() == b"conflict"


def test_overwrite_replaces_conflicting_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(
        source, "16k", "wind", "dataset_a", "clip.wav", b"source"
    )
    target = output / "dataset_a/wind/clip.wav"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"conflict")

    assert _run(source, output, "--overwrite") == 0

    assert target.read_bytes() == b"source"
    assert target.stat().st_mtime_ns == original.stat().st_mtime_ns


def test_dry_run_does_not_create_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "wind", "dataset_a", "clip.wav", b"audio")

    assert _run(source, output, "--dry-run") == 0

    captured = capsys.readouterr()
    assert "planned=1" in captured.out
    assert not output.exists()


@pytest.mark.parametrize("nested_direction", ["output_in_source", "source_in_output"])
def test_rejects_nested_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    nested_direction: str,
) -> None:
    if nested_direction == "output_in_source":
        source = tmp_path / "source"
        source.mkdir()
        output = source / "output"
    else:
        output = tmp_path / "output"
        source = output / "source"
        source.mkdir(parents=True)

    assert _run(source, output) == 1

    captured = capsys.readouterr()
    assert "不能互相嵌套" in captured.err


def test_rejects_symlink_inside_dataset_without_partial_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    dataset_root = source / "16k/wind/dataset_a"
    _write_source(source, "16k", "wind", "dataset_a", "a-valid.wav", b"valid")
    external = tmp_path / "external.wav"
    external.write_bytes(b"audio")
    link = dataset_root / "z-linked.wav"
    try:
        os.symlink(external, link)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持符号链接")

    assert _run(source, output) == 1
    assert not output.exists()


def test_overwrite_rejects_target_symlink_without_touching_referent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "wind", "dataset_a", "clip.wav", b"source")
    referent = tmp_path / "referent.wav"
    referent.write_bytes(b"external")
    target = output / "dataset_a/wind/clip.wav"
    target.parent.mkdir(parents=True)
    try:
        target.symlink_to(referent)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持符号链接")

    assert _run(source, output, "--overwrite") == 1
    assert target.is_symlink()
    assert referent.read_bytes() == b"external"


def test_publish_failure_rolls_back_all_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "wind", "dataset_a", "a.wav", b"a")
    _write_source(source, "16k", "wind", "dataset_a", "b.wav", b"b")

    def fail_publish(staged, targets):
        raise OSError("injected publish failure")

    monkeypatch.setattr(reorganize, "publish_staged_files", fail_publish)
    assert _run(source, output) == 1
    _assert_no_output_files(output)


@pytest.mark.parametrize("link_position", ["leaf", "ancestor"])
def test_rejects_output_path_symlink_without_writing_referent(
    tmp_path: Path,
    link_position: str,
) -> None:
    source = tmp_path / "source"
    _write_source(source, "16k", "wind", "数据集", "音频.wav", b"audio")
    referent = tmp_path / "referent"
    referent.mkdir()
    if link_position == "leaf":
        output = tmp_path / "output"
        link = output
    else:
        link = tmp_path / "linked-parent"
        output = link / "output"
    try:
        link.symlink_to(referent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持符号链接")

    assert _run(source, output) == 1
    assert not list(referent.rglob("*.wav"))


def test_empty_source_fails_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    assert _run(source, tmp_path / "output") == 1

    captured = capsys.readouterr()
    assert "源目录中没有可复制的文件" in captured.err


def test_rejects_special_source_file_without_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    fifo = source / "16k/wind/dataset_a/audio.wav"
    fifo.parent.mkdir(parents=True)
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("当前平台不支持 FIFO")

    assert _run(source, output) == 1
    assert not output.exists()


def test_source_identity_change_after_preflight_aborts_without_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(
        source, "16k", "wind", "数据集", "音频.wav", b"same-content"
    )
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(b"same-content")
    original_preflight = reorganize._preflight_task
    replaced = False

    def replace_after_preflight(task, *, dry_run, overwrite):
        nonlocal replaced
        result = original_preflight(task, dry_run=dry_run, overwrite=overwrite)
        if not replaced:
            os.replace(replacement, original)
            replaced = True
        return result

    monkeypatch.setattr(reorganize, "_preflight_task", replace_after_preflight)

    assert _run(source, output) == 1
    _assert_no_output_files(output)


def test_source_content_change_after_preflight_aborts_without_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(
        source, "16k", "wind", "数据集", "音频.wav", b"before"
    )
    original_preflight = reorganize._preflight_task
    changed = False

    def mutate_after_preflight(task, *, dry_run, overwrite):
        nonlocal changed
        result = original_preflight(task, dry_run=dry_run, overwrite=overwrite)
        if not changed:
            original.write_bytes(b"after!")
            changed = True
        return result

    monkeypatch.setattr(reorganize, "_preflight_task", mutate_after_preflight)

    assert _run(source, output) == 1
    _assert_no_output_files(output)


def test_source_hash_mismatch_after_preflight_aborts_without_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source, "16k", "风声", "数据集", "音频.wav", b"audio")
    original_preflight = reorganize._preflight_task

    def corrupt_expected_hash(task, *, dry_run, overwrite):
        result = original_preflight(task, dry_run=dry_run, overwrite=overwrite)
        assert result.source_snapshot is not None
        return replace(
            result,
            source_snapshot=replace(result.source_snapshot, sha256="0" * 64),
        )

    monkeypatch.setattr(reorganize, "_preflight_task", corrupt_expected_hash)

    assert _run(source, output) == 1
    _assert_no_output_files(output)


def test_source_symlink_swap_after_preflight_aborts_without_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _write_source(
        source, "16k", "wind", "数据集", "音频.wav", b"original"
    )
    referent = tmp_path / "referent.wav"
    referent.write_bytes(b"external")
    replacement = tmp_path / "replacement-link.wav"
    try:
        replacement.symlink_to(referent)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持符号链接")
    original_preflight = reorganize._preflight_task
    replaced = False

    def replace_after_preflight(task, *, dry_run, overwrite):
        nonlocal replaced
        result = original_preflight(task, dry_run=dry_run, overwrite=overwrite)
        if not replaced:
            os.replace(replacement, original)
            replaced = True
        return result

    monkeypatch.setattr(reorganize, "_preflight_task", replace_after_preflight)

    assert _run(source, output) == 1
    _assert_no_output_files(output)
    assert referent.read_bytes() == b"external"
