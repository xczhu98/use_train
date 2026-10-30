"""SCOREQ/DNSMOS owned model-tree snapshot 安全契约测试。"""

from __future__ import annotations

import builtins
import contextlib
import gc
import multiprocessing
import os
import pickle
import shutil
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from audio_filter_pipeline.processors import mos, optional_runtime
from audio_filter_pipeline.processors.optional_runtime import (
    AssetScanLimits,
    LocalAssetError,
    OwnedModelTreeSnapshot,
    local_model_tree_sha256,
    open_owned_model_tree_snapshot,
)


pytestmark = pytest.mark.unit


def _write_model_tree(root: Path, *, payload: bytes = b"model-v1") -> None:
    (root / "weights" / "nested").mkdir(parents=True)
    (root / "config.json").write_bytes(b'{"version":1}')
    (root / "weights" / "nested" / "model.onnx").write_bytes(payload)
    os.chmod(root, 0o755)
    os.chmod(root / "weights", 0o775)
    os.chmod(root / "weights" / "nested", 0o711)
    os.chmod(root / "config.json", 0o666)
    os.chmod(root / "weights" / "nested" / "model.onnx", 0o777)


def _create_snapshot(
    root: Path,
    *,
    limits: AssetScanLimits = AssetScanLimits(),
) -> OwnedModelTreeSnapshot:
    return open_owned_model_tree_snapshot(str(root), limits=limits)


def _snapshot_path(snapshot) -> Path:
    return Path(snapshot.path)


def _assert_snapshot_removed(snapshot_path: Path) -> None:
    assert not snapshot_path.exists()


def _remove_test_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run_fork_probe(probe, *, timeout: float = 10.0):
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child_main():
        receiver.close()
        try:
            payload = {"result": probe(), "error": None}
        except BaseException as exc:
            payload = {
                "result": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            sender.send(payload)
        finally:
            sender.close()

    process = context.Process(target=child_main)
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            process.terminate()
            process.join(timeout=5)
            pytest.fail(f"fork child did not report within {timeout} seconds")
        try:
            payload = receiver.recv()
        except EOFError:
            payload = {"result": None, "error": "child pipe closed without a result"}
    finally:
        receiver.close()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        pytest.fail("fork child reported but did not exit")
    assert process.exitcode == 0
    assert payload["error"] is None, payload["error"]
    return payload["result"]


@pytest.mark.skipif(
    os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods(),
    reason="POSIX fork is unavailable",
)
def test_snapshot_child_verify_and_close_reject_parent_owned_handle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"parent-owned")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)

    def child_probe():
        errors = {}
        for operation in ("verify", "close"):
            try:
                getattr(snapshot, operation)()
            except BaseException as exc:
                errors[operation] = (type(exc).__name__, str(exc))
            else:
                errors[operation] = None
        return {
            "errors": errors,
            "snapshot_exists": snapshot_path.exists(),
        }

    try:
        child = _run_fork_probe(child_probe)
        assert child["errors"] == {
            "verify": (
                "LocalAssetError",
                "owned model tree snapshot belongs to another process",
            ),
            "close": (
                "LocalAssetError",
                "owned model tree snapshot belongs to another process",
            ),
        }
        assert child["snapshot_exists"] is True
        assert snapshot_path.exists()
        assert snapshot.verify() == snapshot.sha256
    finally:
        snapshot.close()

    _assert_snapshot_removed(snapshot_path)


def test_nested_tree_copy_preserves_digest_and_enforces_owned_modes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_model_tree(source)
    expected = local_model_tree_sha256(str(source))

    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    try:
        assert snapshot_path != source
        assert snapshot.sha256 == expected
        assert local_model_tree_sha256(str(snapshot_path)) == expected
        assert (snapshot_path / "config.json").read_bytes() == b'{"version":1}'
        assert (
            snapshot_path / "weights" / "nested" / "model.onnx"
        ).read_bytes() == b"model-v1"

        directories = (
            snapshot_path,
            snapshot_path / "weights",
            snapshot_path / "weights" / "nested",
        )
        files = (
            snapshot_path / "config.json",
            snapshot_path / "weights" / "nested" / "model.onnx",
        )
        for directory in directories:
            info = directory.stat()
            assert stat.S_IMODE(info.st_mode) == 0o500
            assert (info.st_uid, info.st_gid) == (os.geteuid(), os.getegid())
        for file_path in files:
            info = file_path.stat()
            assert stat.S_IMODE(info.st_mode) == 0o400
            assert (info.st_uid, info.st_gid) == (os.geteuid(), os.getegid())
    finally:
        snapshot.close()

    _assert_snapshot_removed(snapshot_path)


def test_snapshot_reads_never_return_to_origin_after_creation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _write_model_tree(source)
    expected = local_model_tree_sha256(str(source))
    held = tmp_path / "held"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _write_model_tree(replacement, payload=b"replacement")

    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    source.rename(held)
    replacement.rename(source)
    try:
        assert snapshot.verify() == expected
        assert local_model_tree_sha256(snapshot.path) == expected
        assert (
            snapshot_path / "weights" / "nested" / "model.onnx"
        ).read_bytes() == b"model-v1"
    finally:
        snapshot.close()

    _assert_snapshot_removed(snapshot_path)


@pytest.mark.parametrize("model", ("SCOREQ", "DNSMOS"))
def test_estimator_loader_and_reset_read_only_lifetime_snapshot(
    tmp_path, monkeypatch, model,
):
    source = tmp_path / model.lower()
    source.mkdir()
    if model == "SCOREQ":
        (source / "adapt_nr_telephone.onnx").write_bytes(b"scoreq-original")
    else:
        (source / "sig_bak_ovr.onnx").write_bytes(b"dnsmos-primary-original")
        (source / "model_v8.onnx").write_bytes(b"dnsmos-p808-original")
        (source / "dnsmos_local.py").write_text(
            "import builtins\n"
            "class ComputeScore:\n"
            "    def __init__(self, primary, p808, use_gpu=False):\n"
            "        builtins._record_dnsmos_load(primary, p808, use_gpu)\n",
            encoding="utf-8",
        )
    expected = local_model_tree_sha256(str(source))
    held = tmp_path / f"{model.lower()}-held"
    replacement = tmp_path / f"{model.lower()}-replacement"
    replacement.mkdir()
    loaded_paths: list[str] = []
    loaded_payloads: list[bytes] = []
    swapped = False

    def swap_origin() -> None:
        nonlocal swapped
        if swapped:
            return
        source.rename(held)
        replacement.rename(source)
        swapped = True

    if model == "SCOREQ":
        from core.metrics import mos_models

        class FakeScoreq:
            pass

        def load_scoreq(*, model_dir, use_gpu):
            assert use_gpu is False
            loaded_paths.append(model_dir)
            swap_origin()
            loaded_payloads.append(
                (Path(model_dir) / "adapt_nr_telephone.onnx").read_bytes()
            )
            return FakeScoreq()

        monkeypatch.setattr(mos_models, "load_scoreq", load_scoreq)
        estimator = mos.SCOREQEstimator(
            use_gpu=False,
            model_dir=str(source),
            verified_model_tree_sha256=expected,
            allow_fallback=False,
        )
    else:
        def record_dnsmos_load(primary, p808, use_gpu):
            assert use_gpu is False
            loaded_paths.extend((primary, p808))
            swap_origin()
            loaded_payloads.extend(
                (Path(primary).read_bytes(), Path(p808).read_bytes())
            )

        real_import = builtins.__import__

        def reject_bare_dnsmos_import(name, *args, **kwargs):
            if name == "dnsmos_local":
                pytest.fail("DNSMOS 不得通过普通 import 加载 bare 模块")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            builtins, "_record_dnsmos_load", record_dnsmos_load, raising=False,
        )
        monkeypatch.setattr(builtins, "__import__", reject_bare_dnsmos_import)
        estimator = mos.DNSMOSEstimator(
            use_gpu=False,
            model_dir=str(source),
            verified_model_tree_sha256=expected,
            allow_fallback=False,
        )

    estimator.reset()
    assert swapped
    if model == "SCOREQ":
        assert loaded_payloads == [b"scoreq-original"] * 2
        snapshot_roots = [Path(path) for path in loaded_paths]
    else:
        assert loaded_payloads == [
            b"dnsmos-primary-original",
            b"dnsmos-p808-original",
        ] * 2
        snapshot_roots = [Path(path).parent for path in loaded_paths]
    assert loaded_paths
    assert len(set(snapshot_roots)) == 1
    snapshot_root = snapshot_roots[0]
    assert snapshot_root != source
    assert snapshot_root != held
    assert snapshot_root.exists()
    assert all(Path(path).exists() for path in loaded_paths)

    estimator.close()

    assert not snapshot_root.exists()


def test_source_file_change_during_descriptor_copy_fails_closed(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    model = source / "model.onnx"
    model.write_bytes(b"A" * 64)
    real_read = os.read
    changed = False

    def read_then_change(fd, size):
        nonlocal changed
        chunk = real_read(fd, size)
        if chunk and not changed and stat.S_ISREG(os.fstat(fd).st_mode):
            changed = True
            replacement = model.with_suffix(".replacement")
            replacement.write_bytes(b"B" * 64)
            replacement.replace(model)
        return chunk

    monkeypatch.setattr(optional_runtime.os, "read", read_then_change)

    with pytest.raises(LocalAssetError, match="changed|binding|snapshot|copy"):
        _create_snapshot(
            source,
            limits=AssetScanLimits(read_chunk_bytes=8),
        )

    assert changed


def test_source_root_replacement_during_copy_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"original")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "model.onnx").write_bytes(b"replacement")
    held = tmp_path / "held"
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    real_listdir = os.listdir
    swapped = False

    def listdir_then_replace(path):
        nonlocal swapped
        names = real_listdir(path)
        if (
            not swapped
            and isinstance(path, int)
            and (os.fstat(path).st_dev, os.fstat(path).st_ino) == source_identity
        ):
            source.rename(held)
            replacement.rename(source)
            swapped = True
        return names

    monkeypatch.setattr(optional_runtime.os, "listdir", listdir_then_replace)

    with pytest.raises(LocalAssetError, match="changed|binding|root|snapshot"):
        _create_snapshot(source)

    assert swapped


def test_source_tree_rejects_symlink_without_reading_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"must-not-be-copied")
    (source / "escape.onnx").symlink_to(outside)

    with pytest.raises(LocalAssetError, match="symlink"):
        _create_snapshot(source)


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo"),
    reason="FIFO resources are unavailable",
)
def test_source_tree_rejects_fifo_without_blocking(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "model.pipe")

    with pytest.raises(LocalAssetError, match="regular|special|FIFO|fifo"):
        _create_snapshot(source)


@pytest.mark.parametrize(
    ("builder", "limits", "error"),
    (
        (
            lambda root: (
                (root / "one").write_bytes(b"1"),
                (root / "two").write_bytes(b"2"),
            ),
            AssetScanLimits(max_entries=1),
            "entry limit",
        ),
        (
            lambda root: (root / "model").write_bytes(b"12345"),
            AssetScanLimits(max_total_bytes=4),
            "byte limit|total bytes",
        ),
        (
            lambda root: (root / "long-name").write_bytes(b"x"),
            AssetScanLimits(max_path_bytes=4),
            "path|depth limit",
        ),
        (
            lambda root: (
                (root / "a" / "b").mkdir(parents=True),
                (root / "a" / "b" / "model").write_bytes(b"x"),
            ),
            AssetScanLimits(max_depth=1),
            "path|depth limit",
        ),
    ),
    ids=("entries", "bytes", "path", "depth"),
)
def test_snapshot_creation_enforces_scan_limits(
    tmp_path, builder, limits, error,
):
    source = tmp_path / "source"
    source.mkdir()
    builder(source)

    with pytest.raises(LocalAssetError, match=error):
        _create_snapshot(source, limits=limits)


def test_snapshot_handle_is_factory_only_immutable_and_nonserializable(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"original")
    snapshot = _create_snapshot(source)
    try:
        with pytest.raises(TypeError, match="created by"):
            OwnedModelTreeSnapshot()
        with pytest.raises(AttributeError, match="immutable"):
            snapshot.path = str(source)
        with pytest.raises(TypeError, match="serialized"):
            pickle.dumps(snapshot)
    finally:
        snapshot.close()


def test_snapshot_verify_rejects_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"original")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    copied_model = snapshot_path / "model.onnx"
    os.chmod(copied_model, 0o600)
    copied_model.write_bytes(b"tampered")

    try:
        with pytest.raises(LocalAssetError, match="digest|changed|tamper|permissions|manifest"):
            snapshot.verify()
    finally:
        snapshot.close()

    _assert_snapshot_removed(snapshot_path)


def test_snapshot_manifest_is_immutable_and_verify_does_not_read_file_content(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    _write_model_tree(source)
    snapshot = _create_snapshot(source)
    manifest = snapshot.manifest
    try:
        assert isinstance(manifest, tuple)
        assert {entry.path for entry in manifest} == {
            b"", b"config.json", b"weights", b"weights/nested",
            b"weights/nested/model.onnx",
        }
        with pytest.raises(AttributeError):
            manifest[0].size = 0
        with pytest.raises(AttributeError, match="immutable"):
            snapshot._manifest = ()
        monkeypatch.setattr(
            optional_runtime.os,
            "read",
            lambda *_args, **_kwargs: pytest.fail(
                "verify_manifest must not read regular file content"
            ),
        )
        assert snapshot.verify_manifest() == snapshot.sha256
    finally:
        snapshot.close()


@pytest.mark.parametrize(
    "tamper",
    ("add", "delete", "replace", "chmod", "size", "time"),
)
def test_snapshot_manifest_rejects_metadata_tamper_matrix(tmp_path, tamper):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"original")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    copied_model = snapshot_path / "model.onnx"

    added = snapshot_path / "added.onnx"
    held = snapshot_path / "held.onnx"
    if tamper in {"add", "delete", "replace"}:
        # 非 root 测试环境也需临时取得目录写权限；恢复 mode 后 ctime 仍会证明篡改。
        os.chmod(snapshot_path, 0o700)
        if tamper == "add":
            added.write_bytes(b"added")
        elif tamper == "delete":
            copied_model.rename(held)
        else:
            copied_model.rename(held)
            copied_model.write_bytes(b"replacement")
        os.chmod(snapshot_path, 0o500)
    elif tamper == "chmod":
        os.chmod(copied_model, 0o600)
    elif tamper == "size":
        os.chmod(copied_model, 0o600)
        with copied_model.open("ab") as stream:
            stream.write(b"-larger")
        os.chmod(copied_model, 0o400)
    else:
        info = copied_model.stat()
        os.utime(
            copied_model,
            ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000),
        )

    try:
        with pytest.raises(LocalAssetError, match="manifest"):
            snapshot.verify_manifest()
    finally:
        if tamper in {"add", "delete", "replace"}:
            os.chmod(snapshot_path, 0o700)
            if added.exists():
                added.unlink()
            if copied_model.exists() and held.exists():
                copied_model.unlink()
            if held.exists():
                held.rename(copied_model)
            os.chmod(snapshot_path, 0o500)
        elif tamper == "chmod":
            os.chmod(copied_model, 0o400)
        snapshot.close()

    _assert_snapshot_removed(snapshot_path)


def test_mos_module_does_not_expose_test_only_verified_tree_loader():
    assert not hasattr(mos, "_verified_model_tree_load")


def test_scoreq_loader_uses_owned_snapshot_and_cleans_after_close(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapt_nr_telephone.onnx").write_bytes(b"original")
    expected = local_model_tree_sha256(str(source))
    loaded_paths = []

    from core.metrics import mos_models

    def load_scoreq(*, model_dir, use_gpu):
        assert use_gpu is False
        loaded_paths.append(Path(model_dir))
        assert (Path(model_dir) / "adapt_nr_telephone.onnx").read_bytes() == b"original"
        return object()

    monkeypatch.setattr(mos_models, "load_scoreq", load_scoreq)
    estimator = mos.SCOREQEstimator(
        use_gpu=False,
        model_dir=str(source),
        verified_model_tree_sha256=expected,
        allow_fallback=False,
    )
    owned_path = loaded_paths[0]

    assert owned_path != source
    assert owned_path.exists()
    estimator.close()
    _assert_snapshot_removed(owned_path)


def test_scoreq_loader_preserves_primary_exception_and_cleans_snapshot(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapt_nr_telephone.onnx").write_bytes(b"original")
    expected = local_model_tree_sha256(str(source))
    primary = RuntimeError("loader failed")
    loaded_paths = []

    from core.metrics import mos_models

    def load_scoreq(*, model_dir, use_gpu):
        assert use_gpu is False
        loaded_paths.append(Path(model_dir))
        raise primary

    monkeypatch.setattr(mos_models, "load_scoreq", load_scoreq)
    with pytest.raises(RuntimeError, match="loader failed") as raised:
        mos.SCOREQEstimator(
            use_gpu=False,
            model_dir=str(source),
            verified_model_tree_sha256=expected,
            allow_fallback=False,
        )

    assert raised.value is primary
    assert loaded_paths
    _assert_snapshot_removed(loaded_paths[0])


class _CleanupFailureSnapshot:
    def __init__(self, path: Path):
        self.path = str(path)
        self.sha256 = "1" * 64
        self.close_calls = 0

    @contextlib.contextmanager
    def loader_path_lease(self):
        yield self.path

    def verify(self):
        return self.sha256

    def verify_manifest(self):
        return self.sha256

    def close(self):
        self.close_calls += 1
        raise RuntimeError("snapshot cleanup failed")


def test_scoreq_loader_surfaces_cleanup_failure_without_primary(
    tmp_path, monkeypatch,
):
    owned = tmp_path / "owned"
    owned.mkdir()
    fake = _CleanupFailureSnapshot(owned)
    monkeypatch.setattr(
        mos, "open_owned_model_tree_snapshot", lambda *_args, **_kwargs: fake,
    )
    from core.metrics import mos_models
    monkeypatch.setattr(mos_models, "load_scoreq", lambda **_kwargs: object())
    estimator = mos.SCOREQEstimator(
        use_gpu=False,
        model_dir="/origin",
        verified_model_tree_sha256="1" * 64,
        allow_fallback=False,
    )

    with pytest.raises(RuntimeError, match="snapshot cleanup failed"):
        estimator.close()

    assert fake.close_calls == 1


def test_scoreq_loader_keeps_primary_when_cleanup_also_fails(
    tmp_path, monkeypatch,
):
    owned = tmp_path / "owned"
    owned.mkdir()
    fake = _CleanupFailureSnapshot(owned)
    monkeypatch.setattr(
        mos, "open_owned_model_tree_snapshot", lambda *_args, **_kwargs: fake,
    )
    primary = RuntimeError("loader failed")
    from core.metrics import mos_models

    def load_scoreq(**_kwargs):
        raise primary

    monkeypatch.setattr(mos_models, "load_scoreq", load_scoreq)
    with pytest.raises(RuntimeError, match="loader failed") as raised:
        mos.SCOREQEstimator(
            use_gpu=False,
            model_dir="/origin",
            verified_model_tree_sha256="1" * 64,
            allow_fallback=False,
        )

    assert raised.value is primary
    assert fake.close_calls == 1


def test_concurrent_snapshots_are_isolated(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"shared-model")
    controlled_tmp = tmp_path / "controlled-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    barrier = threading.Barrier(2)

    def consume_snapshot(_index):
        snapshot = _create_snapshot(source)
        snapshot_path = _snapshot_path(snapshot)
        try:
            barrier.wait(timeout=5)
            return snapshot_path, (snapshot_path / "model.onnx").read_bytes()
        finally:
            snapshot.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume_snapshot, range(2)))

    paths = [path for path, _payload in results]
    assert paths[0] != paths[1]
    assert [payload for _path, payload in results] == [b"shared-model"] * 2
    assert all(not path.exists() for path in paths)
    assert list(controlled_tmp.iterdir()) == []


def test_snapshot_close_is_idempotent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"model")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)

    snapshot.close()
    snapshot.close()

    _assert_snapshot_removed(snapshot_path)


def test_snapshot_cleanup_rejects_file_replaced_at_final_unlink(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    competitor = tmp_path / "competitor.onnx"
    competitor.write_bytes(b"competitor")
    competitor_identity = (competitor.stat().st_dev, competitor.stat().st_ino)
    held = tmp_path / "held-owned.onnx"
    snapshot = _create_snapshot(source)
    owned_fd = os.open("model.onnx", os.O_RDONLY, dir_fd=snapshot._root_fd)
    owned_identity = (os.fstat(owned_fd).st_dev, os.fstat(owned_fd).st_ino)
    real_unlink = optional_runtime.os.unlink
    attack_parent_fd = -1
    tomb_name = None

    def replace_tomb_at_unlink(path, *, dir_fd=None):
        nonlocal attack_parent_fd, tomb_name
        current = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        if tomb_name is None and (current.st_dev, current.st_ino) == owned_identity:
            attack_parent_fd = os.dup(dir_fd)
            tomb_name = path
            os.rename(path, held, src_dir_fd=dir_fd)
            os.link(competitor, path, dst_dir_fd=dir_fd)
        return real_unlink(path, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "unlink", replace_tomb_at_unlink)
            with pytest.raises(LocalAssetError, match="binding|removal|cleanup|snapshot"):
                snapshot.close()

        assert tomb_name is not None
        assert not os.path.exists(
            os.path.join(f"/proc/self/fd/{attack_parent_fd}", tomb_name)
        )
        assert (competitor.stat().st_dev, competitor.stat().st_ino) == competitor_identity
        assert competitor.read_bytes() == b"competitor"
        assert (held.stat().st_dev, held.stat().st_ino) == owned_identity
        assert snapshot._closed is False

        os.rename(held, tomb_name, dst_dir_fd=attack_parent_fd)
        snapshot.close()

        assert snapshot._closed is True
        assert os.fstat(owned_fd).st_nlink == 0
        assert (competitor.stat().st_dev, competitor.stat().st_ino) == competitor_identity
        assert competitor.read_bytes() == b"competitor"
        _remove_test_path(competitor)
        assert list(controlled_tmp.iterdir()) == []
    finally:
        try:
            snapshot.close()
        except BaseException:
            pass
        if attack_parent_fd >= 0:
            os.close(attack_parent_fd)
        os.close(owned_fd)
        for path in (competitor, held):
            _remove_test_path(path)
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_snapshot_cleanup_rejects_nested_directory_replaced_at_final_rmdir(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    _write_model_tree(source)
    competitor = tmp_path / "competitor-directory"
    competitor.mkdir()
    (competitor / "sentinel").write_bytes(b"competitor")
    competitor_identity = (competitor.stat().st_dev, competitor.stat().st_ino)
    held = tmp_path / "held-owned-directory"
    snapshot = _create_snapshot(source)
    nested_fd = os.open(
        "weights/nested", os.O_RDONLY | os.O_DIRECTORY, dir_fd=snapshot._root_fd,
    )
    owned_identity = (os.fstat(nested_fd).st_dev, os.fstat(nested_fd).st_ino)
    real_rmdir = optional_runtime.os.rmdir
    attack_parent_fd = -1
    tomb_name = None

    def replace_tomb_at_rmdir(path, *, dir_fd=None):
        nonlocal attack_parent_fd, tomb_name
        current = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        if tomb_name is None and (current.st_dev, current.st_ino) == owned_identity:
            attack_parent_fd = os.dup(dir_fd)
            tomb_name = path
            os.rename(path, held, src_dir_fd=dir_fd)
            os.rename(competitor, path, dst_dir_fd=dir_fd)
        return real_rmdir(path, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "rmdir", replace_tomb_at_rmdir)
            with pytest.raises(OSError, match="not empty|Directory not empty"):
                snapshot.close()

        assert tomb_name is not None
        current = os.stat(tomb_name, dir_fd=attack_parent_fd, follow_symlinks=False)
        assert (current.st_dev, current.st_ino) == competitor_identity
        competitor_fd = os.open(
            tomb_name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=attack_parent_fd,
        )
        try:
            assert os.listdir(competitor_fd) == ["sentinel"]
            sentinel_fd = os.open("sentinel", os.O_RDONLY, dir_fd=competitor_fd)
            try:
                assert os.read(sentinel_fd, 32) == b"competitor"
            finally:
                os.close(sentinel_fd)
        finally:
            os.close(competitor_fd)
        assert (held.stat().st_dev, held.stat().st_ino) == owned_identity
        assert snapshot._closed is False

        os.rename(tomb_name, competitor, src_dir_fd=attack_parent_fd)
        os.rename(held, "nested", dst_dir_fd=attack_parent_fd)
        snapshot.close()

        assert snapshot._closed is True
        assert os.fstat(nested_fd).st_nlink == 0
        assert (competitor.stat().st_dev, competitor.stat().st_ino) == competitor_identity
        assert (competitor / "sentinel").read_bytes() == b"competitor"
        _remove_test_path(competitor)
        assert list(controlled_tmp.iterdir()) == []
    finally:
        try:
            snapshot.close()
        except BaseException:
            pass
        if attack_parent_fd >= 0:
            os.close(attack_parent_fd)
        os.close(nested_fd)
        for path in (competitor, held):
            _remove_test_path(path)
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_snapshot_cleanup_rejects_top_level_replacement_at_final_rmdir(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    competitor = tmp_path / "competitor-snapshot"
    competitor.mkdir()
    (competitor / "sentinel").write_bytes(b"competitor")
    competitor_identity = (competitor.stat().st_dev, competitor.stat().st_ino)
    held = tmp_path / "held-owned-snapshot"
    snapshot = _create_snapshot(source)
    owned_fd = os.dup(snapshot._root_fd)
    owned_identity = (os.fstat(owned_fd).st_dev, os.fstat(owned_fd).st_ino)
    real_rmdir = optional_runtime.os.rmdir
    attack_parent_fd = -1
    tomb_name = None

    def replace_tomb_at_rmdir(path, *, dir_fd=None):
        nonlocal attack_parent_fd, tomb_name
        current = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        if tomb_name is None and (current.st_dev, current.st_ino) == owned_identity:
            attack_parent_fd = os.dup(dir_fd)
            tomb_name = path
            os.rename(path, held, src_dir_fd=dir_fd)
            os.rename(competitor, path, dst_dir_fd=dir_fd)
        return real_rmdir(path, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "rmdir", replace_tomb_at_rmdir)
            with pytest.raises(OSError, match="not empty|Directory not empty"):
                snapshot.close()

        assert tomb_name is not None
        current = os.stat(tomb_name, dir_fd=attack_parent_fd, follow_symlinks=False)
        assert (current.st_dev, current.st_ino) == competitor_identity
        competitor_fd = os.open(
            tomb_name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=attack_parent_fd,
        )
        try:
            assert os.listdir(competitor_fd) == ["sentinel"]
            sentinel_fd = os.open("sentinel", os.O_RDONLY, dir_fd=competitor_fd)
            try:
                assert os.read(sentinel_fd, 32) == b"competitor"
            finally:
                os.close(sentinel_fd)
        finally:
            os.close(competitor_fd)
        assert (held.stat().st_dev, held.stat().st_ino) == owned_identity
        assert snapshot._closed is False

        os.rename(tomb_name, competitor, src_dir_fd=attack_parent_fd)
        os.rename(held, tomb_name, dst_dir_fd=attack_parent_fd)
        snapshot.close()

        assert snapshot._closed is True
        assert os.fstat(owned_fd).st_nlink == 0
        assert (competitor.stat().st_dev, competitor.stat().st_ino) == competitor_identity
        assert (competitor / "sentinel").read_bytes() == b"competitor"
        _remove_test_path(competitor)
        assert list(controlled_tmp.iterdir()) == []
    finally:
        try:
            snapshot.close()
        except BaseException:
            pass
        if attack_parent_fd >= 0:
            os.close(attack_parent_fd)
        os.close(owned_fd)
        for path in (competitor, held):
            _remove_test_path(path)
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_snapshot_close_retries_after_path_cleanup_fails_once(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    owned_fd = os.open("model.onnx", os.O_RDONLY, dir_fd=snapshot._root_fd)
    root_fd = os.dup(snapshot._root_fd)
    real_unlink = optional_runtime.os.unlink
    failed = False

    def fail_once(path, *, dir_fd=None):
        nonlocal failed
        current = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        opened = os.fstat(owned_fd)
        if (
            not failed
            and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
        ):
            failed = True
            raise OSError("injected snapshot unlink failure")
        return real_unlink(path, dir_fd=dir_fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "unlink", fail_once)
            with pytest.raises(OSError, match="injected snapshot unlink failure"):
                snapshot.close()

        assert failed
        assert snapshot._closed is False
        assert snapshot._cleanup_state["path_removed"] is False
        assert os.fstat(owned_fd).st_nlink == 1

        snapshot.close()

        assert snapshot._closed is True
        assert snapshot._cleanup_state["path_removed"] is True
        assert os.fstat(owned_fd).st_nlink == 0
        assert os.fstat(root_fd).st_nlink == 0
        assert list(controlled_tmp.iterdir()) == []
    finally:
        try:
            snapshot.close()
        except BaseException:
            pass
        os.close(owned_fd)
        os.close(root_fd)
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_controlled_tmpdir_has_no_residue_after_success_and_failure(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "controlled-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"model")

    snapshot = _create_snapshot(source)
    snapshot.close()
    (source / "escape.onnx").symlink_to(source / "model.onnx")
    with pytest.raises(LocalAssetError, match="symlink"):
        _create_snapshot(source)

    assert list(controlled_tmp.iterdir()) == []


@pytest.mark.parametrize("phase", ("before", "midway"))
def test_quarantine_creation_failure_releases_factory_resources(
    tmp_path, monkeypatch, phase,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    real_create = optional_runtime._create_snapshot_quarantine
    real_open = optional_runtime.os.open
    held_descriptors = []

    def remember_factory_descriptors(parent_fd):
        for entry in controlled_tmp.iterdir():
            target = str(entry.resolve())
            for descriptor_name in os.listdir("/proc/self/fd"):
                descriptor = int(descriptor_name)
                try:
                    if os.path.realpath(f"/proc/self/fd/{descriptor}") == target:
                        opened = os.fstat(descriptor)
                        held_descriptors.append(
                            (descriptor, (opened.st_dev, opened.st_ino))
                        )
                except OSError:
                    pass
        opened = os.fstat(parent_fd)
        held_descriptors.append(
            (parent_fd, (opened.st_dev, opened.st_ino))
        )

    def fail_before(parent_fd):
        remember_factory_descriptors(parent_fd)
        raise OSError("injected pre-quarantine creation failure")

    def fail_quarantine_open(path, flags, *args, **kwargs):
        if os.fsdecode(path).startswith("quarantine-"):
            raise OSError("injected midway quarantine creation failure")
        descriptor = real_open(path, flags, *args, **kwargs)
        if os.fsdecode(path).startswith(".audio-filter-model-cleanup-"):
            opened = os.fstat(descriptor)
            held_descriptors.append(
                (descriptor, (opened.st_dev, opened.st_ino))
            )
        return descriptor

    def fail_midway(parent_fd):
        remember_factory_descriptors(parent_fd)
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "open", fail_quarantine_open)
            return real_create(parent_fd)

    monkeypatch.setattr(
        optional_runtime,
        "_create_snapshot_quarantine",
        fail_before if phase == "before" else fail_midway,
    )
    try:
        with pytest.raises(LocalAssetError, match="snapshot failed") as raised:
            _create_snapshot(source)
        residue = list(controlled_tmp.iterdir())
        for descriptor, identity in held_descriptors:
            with pytest.raises(OSError):
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != identity:
                    raise OSError("descriptor number was reused")
        if residue:
            assert all(
                path.name.startswith("audio-filter-model-tree-")
                for path in residue
            )
            details = " ".join(
                [str(raised.value), str(raised.value.__cause__)]
                + list(getattr(raised.value, "__notes__", ()))
                + list(getattr(raised.value.__cause__, "__notes__", ()))
            )
            assert "cleanup" in details and (
                "ownership" in details or "evidence" in details
            )
    finally:
        monkeypatch.undo()
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_quarantine_creation_rollback_failure_is_reported(tmp_path, monkeypatch):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    real_create = optional_runtime._create_snapshot_quarantine
    real_open = optional_runtime.os.open
    real_rmdir = optional_runtime.os.rmdir
    rollback_attempted = False

    def fail_quarantine_open(path, flags, *args, **kwargs):
        if os.fsdecode(path).startswith("quarantine-"):
            raise OSError("injected quarantine setup failure")
        return real_open(path, flags, *args, **kwargs)

    def fail_container_rollback(path, *args, **kwargs):
        nonlocal rollback_attempted
        if os.fsdecode(path).startswith(".audio-filter-model-cleanup-"):
            rollback_attempted = True
            raise OSError("injected cleanup container rollback failure")
        return real_rmdir(path, *args, **kwargs)

    def create_with_failing_rollback(parent_fd):
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "open", fail_quarantine_open)
            patch.setattr(optional_runtime.os, "rmdir", fail_container_rollback)
            return real_create(parent_fd)

    monkeypatch.setattr(
        optional_runtime,
        "_create_snapshot_quarantine",
        create_with_failing_rollback,
    )
    try:
        with pytest.raises(LocalAssetError) as raised:
            _create_snapshot(source)
        cause = raised.value.__cause__
        details = " ".join(
            [str(raised.value), str(cause)]
            + list(getattr(raised.value, "__notes__", ()))
            + list(getattr(cause, "__notes__", ()))
        )
        assert rollback_attempted
        assert "injected cleanup container rollback failure" in details
    finally:
        monkeypatch.undo()
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_discard_after_retryable_path_cleanup_failure_runs_finalizer(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    descriptors = tuple(
        (fd, (os.fstat(fd).st_dev, os.fstat(fd).st_ino))
        for fd in (
            snapshot._root_fd,
            snapshot._parent_fd,
            snapshot._cleanup_state["quarantine_fd"],
            snapshot._cleanup_state["container_fd"],
        )
    )
    real_unlink = optional_runtime.os.unlink
    attempts = 0

    def fail_once(path, *, dir_fd=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected retryable public path cleanup failure")
        return real_unlink(path, dir_fd=dir_fd)

    try:
        monkeypatch.setattr(optional_runtime.os, "unlink", fail_once)
        with pytest.raises(OSError, match="injected retryable"):
            snapshot.close()
        assert snapshot_path.exists()
        del snapshot
        gc.collect()
        observed = (attempts, snapshot_path.exists(), list(controlled_tmp.iterdir()))
    finally:
        monkeypatch.undo()
        for fd, identity in descriptors:
            try:
                opened = os.fstat(fd)
                if (opened.st_dev, opened.st_ino) == identity:
                    os.close(fd)
            except OSError:
                pass
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)

    assert observed == (2, False, [])


@pytest.mark.parametrize(
    ("stage", "state_key"),
    (("root", "path_removed"),
     ("quarantine", "quarantine_removed"),
     ("container", "container_removed")),
)
def test_close_reconciles_rmdir_success_before_post_fstat_commit(
    tmp_path, monkeypatch, stage, state_key,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    state = snapshot._cleanup_state
    targets = {
        "root": (state["claim_name"], state["quarantine_fd"], snapshot._root_fd),
        "quarantine": (
            state["quarantine_name"], state["container_fd"], state["quarantine_fd"],
        ),
        "container": (state["container_name"], snapshot._parent_fd, state["container_fd"]),
    }
    target_name, target_parent_fd, target_fd = targets[stage]
    real_rmdir = optional_runtime.os.rmdir
    real_fstat = optional_runtime.os.fstat
    armed = failed = False

    def rmdir_then_arm(path, *, dir_fd=None):
        nonlocal armed
        result = real_rmdir(path, dir_fd=dir_fd)
        if path == target_name and dir_fd == target_parent_fd:
            armed = True
        return result

    def fail_post_rmdir_fstat(fd):
        nonlocal failed
        if armed and not failed and fd == target_fd:
            failed = True
            raise OSError(f"injected {stage} post-rmdir fstat failure")
        return real_fstat(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "rmdir", rmdir_then_arm)
            patch.setattr(optional_runtime.os, "fstat", fail_post_rmdir_fstat)
            with pytest.raises(OSError, match=f"injected {stage} post-rmdir"):
                snapshot.close()
        assert failed
        assert state[state_key] is False

        snapshot.close()

        assert snapshot._closed is True
        assert all(state[key] for key in (
            "path_removed", "quarantine_removed", "container_removed",
        ))
        assert list(controlled_tmp.iterdir()) == []
    finally:
        try:
            snapshot.close()
        except BaseException:
            pass
        for fd in (
            snapshot._root_fd,
            snapshot._parent_fd,
            state["quarantine_fd"],
            state["container_fd"],
        ):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)


def test_fd_close_error_is_terminal_uncertainty_and_fd_is_not_reused(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    failed_fd = snapshot._root_fd
    real_close = optional_runtime.os.close
    injected = False

    def close_then_report_error(fd):
        nonlocal injected
        if fd == failed_fd and not injected:
            injected = True
            real_close(fd)
            raise OSError("injected uncertain descriptor close")
        return real_close(fd)

    with monkeypatch.context() as patch:
        patch.setattr(optional_runtime.os, "close", close_then_report_error)
        with pytest.raises(LocalAssetError, match="failed to close snapshot root"):
            snapshot.close()

    anchor = os.open("/dev/null", os.O_RDONLY)
    replacement_fd = failed_fd
    reused_close_calls = []
    try:
        if anchor != replacement_fd:
            os.dup2(anchor, replacement_fd)
            os.close(anchor)
            anchor = -1

        def record_reused_close(fd):
            if fd == replacement_fd:
                reused_close_calls.append(fd)
            return real_close(fd)

        errors = []
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "close", record_reused_close)
            for operation in (
                snapshot.verify,
                snapshot.__enter__,
                snapshot.close,
                snapshot.verify,
                snapshot.__enter__,
                snapshot.close,
            ):
                with pytest.raises(LocalAssetError, match="terminal|uncertain|closed") as raised:
                    operation()
                errors.append((type(raised.value), str(raised.value)))

        assert len(set(errors)) == 1
        assert snapshot._cleanup_state["fd_close_failed"] is True
        assert snapshot._root_fd == -1
        assert reused_close_calls == []
        assert os.fstat(replacement_fd).st_mode
    finally:
        if anchor >= 0 and anchor != replacement_fd:
            os.close(anchor)
        os.close(replacement_fd)


def test_descriptor_loader_path_survives_public_snapshot_path_replacement(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned-original")
    snapshot = _create_snapshot(source)
    public_path = _snapshot_path(snapshot)
    held_path = tmp_path / "held-owned-snapshot"
    replacement = tmp_path / "replacement-snapshot"
    replacement.mkdir()
    (replacement / "model.onnx").write_bytes(b"public-replacement")

    public_path.rename(held_path)
    replacement.rename(public_path)
    try:
        loader_path = Path(snapshot.loader_path)
        assert (public_path / "model.onnx").read_bytes() == b"public-replacement"
        assert (loader_path / "model.onnx").read_bytes() == b"owned-original"
    finally:
        public_path.rename(replacement)
        held_path.rename(public_path)
        snapshot.close()

    _assert_snapshot_removed(public_path)
    assert (replacement / "model.onnx").read_bytes() == b"public-replacement"


@pytest.mark.parametrize("replacement", (False, True), ids=("missing", "replaced"))
def test_verify_normalizes_public_snapshot_path_loss_to_local_asset_error(
    tmp_path, replacement,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    public_path = _snapshot_path(snapshot)
    held_path = tmp_path / "held-owned-snapshot"
    replacement_path = tmp_path / "replacement-snapshot"
    public_path.rename(held_path)
    if replacement:
        replacement_path.mkdir()
        replacement_path.rename(public_path)

    try:
        with pytest.raises(LocalAssetError, match="path binding") as raised:
            snapshot.verify()
        assert type(raised.value) is LocalAssetError
    finally:
        if replacement:
            public_path.rename(replacement_path)
        held_path.rename(public_path)
        snapshot.close()

    _assert_snapshot_removed(public_path)


def test_renameat2_noreplace_capability_is_checked_before_root_creation(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    root_creation_calls = []

    def unsupported_renameat2(*_args):
        optional_runtime.ctypes.set_errno(optional_runtime.errno.ENOSYS)
        return -1

    def record_root_creation():
        root_creation_calls.append(True)
        pytest.fail("能力门禁失败后不得创建 snapshot root")

    monkeypatch.setattr(
        optional_runtime, "_snapshot_renameat2", lambda: unsupported_renameat2,
    )
    monkeypatch.setattr(optional_runtime, "_create_snapshot_root", record_root_creation)

    with pytest.raises(LocalAssetError, match="RENAME_NOREPLACE.*unavailable"):
        _create_snapshot(source)

    assert root_creation_calls == []


@pytest.mark.skipif(
    os.name != "posix" or "fork" not in multiprocessing.get_all_start_methods(),
    reason="POSIX fork is unavailable",
)
def test_fork_child_consumes_inherited_snapshot_fds_without_affecting_parent(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    inherited = (
        snapshot._root_fd,
        snapshot._parent_fd,
        snapshot._cleanup_state["quarantine_fd"],
        snapshot._cleanup_state["container_fd"],
    )
    identities = tuple(
        (os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in inherited
    )

    def child_probe():
        try:
            snapshot.verify()
        except LocalAssetError as exc:
            error = str(exc)
        else:
            error = None
        still_open = []
        for fd in inherited:
            try:
                os.fstat(fd)
            except OSError:
                still_open.append(False)
            else:
                still_open.append(True)
        return {"error": error, "still_open": still_open}

    try:
        child = _run_fork_probe(child_probe)
        assert child == {
            "error": "owned model tree snapshot belongs to another process",
            "still_open": [False, False, False, False],
        }
        assert snapshot_path.exists()
        assert tuple(
            (os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in inherited
        ) == identities
        assert snapshot.verify() == snapshot.sha256
    finally:
        snapshot.close()

    _assert_snapshot_removed(snapshot_path)


def test_failed_claim_verification_and_restore_enters_terminal_provenance(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    state = snapshot._cleanup_state
    real_open_entry = optional_runtime._open_snapshot_cleanup_entry
    real_rename = optional_runtime._rename_snapshot_noreplace
    failed_verification = False
    rename_calls = 0

    def fail_claim_verification(parent_fd, name, expected):
        nonlocal failed_verification
        if parent_fd == state["quarantine_fd"] and name == state["claim_name"]:
            failed_verification = True
            raise OSError("injected claim verification failure")
        return real_open_entry(parent_fd, name, expected)

    def record_rename(*args):
        nonlocal rename_calls
        rename_calls += 1
        return real_rename(*args)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                optional_runtime, "_open_snapshot_cleanup_entry",
                fail_claim_verification,
            )
            patch.setattr(optional_runtime, "_restore_snapshot_root", lambda *_args: False)
            patch.setattr(optional_runtime, "_rename_snapshot_noreplace", record_rename)
            with pytest.raises(OSError, match="claim verification failure"):
                snapshot.close()

        assert failed_verification
        assert state["claimed"] is True
        assert "restoration failed" in state["terminal_provenance"]
        assert rename_calls == 1
        for operation in (snapshot.close, snapshot.verify, snapshot.close):
            with pytest.raises(LocalAssetError, match="provenance is terminal"):
                operation()
        assert rename_calls == 1
    finally:
        # 测试解除故障后按已记录的 claimed provenance 完成真实清理。
        state["terminal_provenance"] = None
        snapshot.close()


def test_loader_path_lease_blocks_concurrent_close_until_context_exit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    snapshot = _create_snapshot(source)
    snapshot_path = _snapshot_path(snapshot)
    close_started = threading.Event()
    close_finished = threading.Event()
    close_errors = []

    def close_snapshot():
        close_started.set()
        try:
            snapshot.close()
        except BaseException as exc:
            close_errors.append(exc)
        finally:
            close_finished.set()

    with snapshot.loader_path_lease() as loader_path:
        with pytest.raises(LocalAssetError, match="active loader path lease"):
            snapshot.close()
        closer = threading.Thread(target=close_snapshot)
        closer.start()
        assert close_started.wait(timeout=5)
        assert not close_finished.wait(timeout=0.05)
        assert (Path(loader_path) / "model.onnx").read_bytes() == b"owned"

    closer.join(timeout=5)
    assert not closer.is_alive()
    assert close_errors == []
    assert close_finished.is_set()
    assert snapshot._closed is True
    _assert_snapshot_removed(snapshot_path)


def test_factory_root_creation_is_descriptor_first(tmp_path, monkeypatch):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    real_mkdir = optional_runtime.os.mkdir
    root_mkdir_dir_fds = []

    def record_mkdir(path, mode=0o777, *, dir_fd=None):
        if os.fsdecode(path).startswith("audio-filter-model-tree-"):
            root_mkdir_dir_fds.append(dir_fd)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        optional_runtime.os,
        "supports_dir_fd",
        optional_runtime.os.supports_dir_fd | {record_mkdir},
    )
    monkeypatch.setattr(optional_runtime.os, "mkdir", record_mkdir)
    monkeypatch.setattr(
        optional_runtime.tempfile, "mkdtemp",
        lambda *_args, **_kwargs: pytest.fail("不得使用 path-first mkdtemp"),
    )

    snapshot = _create_snapshot(source)
    try:
        assert len(root_mkdir_dir_fds) == 1
        assert root_mkdir_dir_fds[0] is not None
        assert os.path.realpath(
            f"/proc/self/fd/{root_mkdir_dir_fds[0]}"
        ) == str(controlled_tmp)
    finally:
        snapshot.close()

    assert list(controlled_tmp.iterdir()) == []


def test_factory_root_identity_failure_reports_unsafe_cleanup_skip(
    tmp_path, monkeypatch,
):
    controlled_tmp = tmp_path / "snapshot-tmp"
    controlled_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(controlled_tmp))
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"owned")
    real_fstat = optional_runtime.os.fstat
    failed = False

    def fail_root_identity(fd):
        nonlocal failed
        if not failed:
            target = os.path.realpath(f"/proc/self/fd/{fd}")
            if Path(target).name.startswith("audio-filter-model-tree-"):
                failed = True
                raise OSError("injected root identity failure")
        return real_fstat(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(optional_runtime.os, "fstat", fail_root_identity)
            with pytest.raises(LocalAssetError, match="snapshot failed") as raised:
                _create_snapshot(source)
        details = " ".join(
            [str(raised.value), str(raised.value.__cause__)]
            + list(getattr(raised.value, "__notes__", ()))
            + list(getattr(raised.value.__cause__, "__notes__", ()))
        )
        assert failed
        assert "identity evidence is incomplete" in details
        residue = list(controlled_tmp.iterdir())
        assert len(residue) == 1
        assert residue[0].name.startswith("audio-filter-model-tree-")
    finally:
        monkeypatch.undo()
        for path in list(controlled_tmp.iterdir()):
            _remove_test_path(path)
