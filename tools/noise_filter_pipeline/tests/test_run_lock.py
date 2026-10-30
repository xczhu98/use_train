"""Pipeline 锁文件安全契约。"""
from __future__ import annotations

import os

import pytest

from noise_filter_pipeline.core import run_lock

from .module_patch_test_helpers import patched_os


pytestmark = pytest.mark.unit


def test_open_lock_file_rejects_hardlink(tmp_path):
    """攻击者预置 hardlink 时，锁必须 fail closed。"""
    target = tmp_path / "target.lock"
    target.write_text("", encoding="utf-8")
    linked = tmp_path / "linked.lock"
    os.link(target, linked)

    with pytest.raises(RuntimeError, match="不得是 hardlink"):
        run_lock._open_lock_file(str(linked), purpose="测试锁")


def test_writer_lock_path_uses_output_physical_identity(tmp_path):
    """输出目录 symlink alias 必须映射到同一把物理 identity 锁。"""
    output = tmp_path / "output"
    output.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)

    assert run_lock.writer_lock_path(str(output)) == run_lock.writer_lock_path(str(alias))


def test_substage_writer_lock_creates_missing_output_before_identity_lock(
    tmp_path, monkeypatch
):
    """缺失 output 必须先创建实体目录，再按该目录的物理 identity 加锁。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_d" / "D2"
    original_acquire = run_lock._acquire_lock_with_timeout

    def assert_directory_then_acquire(lock_path, *, purpose, timeout):
        assert output.is_dir()
        assert not output.is_symlink()
        assert lock_path == run_lock.writer_lock_path(str(output))
        return original_acquire(lock_path, purpose=purpose, timeout=timeout)

    monkeypatch.setattr(
        run_lock, "_acquire_lock_with_timeout", assert_directory_then_acquire
    )
    with run_lock.substage_writer_lock(str(output), timeout=0):
        pass


def test_substage_writer_lock_rejects_symlink_injected_during_leaf_creation(
    tmp_path, monkeypatch
):
    """缺失 leaf 创建竞态中出现 symlink 时不得跟随到攻击者 target。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_d" / "D2"
    target = tmp_path / "target"
    target.mkdir()
    original_mkdir = run_lock.os.mkdir

    def inject_symlink(path, mode=0o777, *, dir_fd=None):
        if path == "D2":
            os.symlink(str(target), path, dir_fd=dir_fd)
            return
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(run_lock, "os", patched_os(mkdir=inject_symlink))
    with pytest.raises(RuntimeError, match="无法安全打开输出目录"):
        with run_lock.substage_writer_lock(str(output), timeout=0):
            pass
    assert output.is_symlink()
    assert not any(target.iterdir())


def test_substage_writer_lock_rejects_output_symlink_replacement_after_acquire(
    tmp_path, monkeypatch
):
    """锁获取后 output leaf 被替换为 symlink 时必须在 yield 前拒绝。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_d" / "D2"
    output.mkdir(parents=True)
    target = tmp_path / "target"
    target.mkdir()
    displaced = tmp_path / "displaced"
    original_acquire = run_lock._acquire_lock_with_timeout
    replaced = False

    def replace_output_after_acquire(lock_path, *, purpose, timeout):
        nonlocal replaced
        fd = original_acquire(lock_path, purpose=purpose, timeout=timeout)
        if not replaced:
            replaced = True
            output.rename(displaced)
            output.symlink_to(target, target_is_directory=True)
        return fd

    monkeypatch.setattr(
        run_lock, "_acquire_lock_with_timeout", replace_output_after_acquire
    )
    with pytest.raises(RuntimeError, match="输出目录.*被替换"):
        with run_lock.substage_writer_lock(str(output), timeout=0):
            pass


def test_substage_writer_lock_rejects_output_symlink_replacement_while_held(
    tmp_path, monkeypatch
):
    """writer 区间内 output leaf 被替换时必须在释放锁前 fail closed。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    displaced = tmp_path / "displaced"

    with pytest.raises(RuntimeError, match="输出目录.*被替换"):
        with run_lock.substage_writer_lock(str(output), timeout=0):
            output.rename(displaced)
            output.symlink_to(target, target_is_directory=True)


def test_substage_writer_lock_rejects_dangling_symlink_output(tmp_path, monkeypatch):
    """dangling symlink leaf 不得触发在 symlink target 下创建业务目录。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    target = tmp_path / "missing-target"
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="带 symlink 的输出路径必须已指向现有目录"):
        with run_lock.substage_writer_lock(str(output), timeout=0):
            pass
    assert not target.exists()


def test_substage_writer_lock_rejects_lock_file_symlink_replacement_after_acquire(
    tmp_path, monkeypatch
):
    """flock 后 registry 锁路径被替换为 symlink 时必须在返回前拒绝。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_d" / "D2"
    output.mkdir(parents=True)
    lock_path = run_lock.writer_lock_path(str(output))
    target = tmp_path / "target"
    target.write_text("not a lock", encoding="utf-8")
    displaced = tmp_path / "displaced.lock"
    original_flock = run_lock.fcntl.flock
    replaced = False

    def replace_after_acquire(fd, operation):
        nonlocal replaced
        original_flock(fd, operation)
        if operation & run_lock.fcntl.LOCK_EX and not replaced:
            replaced = True
            os.rename(lock_path, displaced)
            os.symlink(target, lock_path)

    monkeypatch.setattr(run_lock.fcntl, "flock", replace_after_acquire)
    with pytest.raises(RuntimeError, match="路径类型已变化"):
        with run_lock.substage_writer_lock(str(output), timeout=0):
            pass


def test_substage_writer_lock_serializes_symlink_alias(tmp_path, monkeypatch):
    """真实目录与既有 symlink alias 并发进入时必须争用同一把锁。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "output"
    output.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(output, target_is_directory=True)

    with run_lock.substage_writer_lock(str(alias), timeout=1):
        with pytest.raises(RuntimeError, match="获取子阶段 writer 锁.*超时"):
            with run_lock.substage_writer_lock(str(output), timeout=0):
                pass


def test_lock_registry_is_private_and_lock_file_persists(tmp_path, monkeypatch):
    """registry 与锁文件必须是当前用户私有且释放后仍保留。"""
    registry = tmp_path / "registry"
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(registry))
    stage_d = tmp_path / "run" / "stage_d"
    stage_d.mkdir(parents=True)

    fd = run_lock.acquire_run_lock(str(stage_d))
    lock_path = run_lock.run_lock_path(str(stage_d))
    try:
        assert registry.stat().st_mode & 0o777 == 0o700
        assert os.stat(lock_path).st_mode & 0o777 == 0o600
        assert os.stat(lock_path).st_uid == os.getuid()
    finally:
        run_lock.release_run_lock(fd)
    assert os.path.isfile(lock_path)


def test_lock_path_replacement_after_open_fails_closed(tmp_path, monkeypatch):
    """flock 前后锁路径被替换时不得返回已持有的 fd。"""
    registry = tmp_path / "registry"
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(registry))
    stage_d = tmp_path / "run" / "stage_d"
    stage_d.mkdir(parents=True)
    lock_path = run_lock.run_lock_path(str(stage_d))
    original_open = run_lock.os.open
    replaced = False

    def replace_after_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if (path == lock_path or path == os.path.basename(lock_path)) and not replaced:
            replaced = True
            os.rename(lock_path, str(lock_path) + ".old")
            open(lock_path, "wb").close()
            os.chmod(lock_path, 0o600)
        return fd

    monkeypatch.setattr(run_lock, "os", patched_os(open=replace_after_open))
    with pytest.raises(RuntimeError, match="被替换"):
        run_lock.acquire_run_lock(str(stage_d))


def test_lock_file_open_stays_bound_to_validated_registry_dirfd(tmp_path, monkeypatch):
    """registry 路径换绑后，锁文件不得跟随到攻击者目录。"""
    registry = tmp_path / "registry"
    attacker = tmp_path / "attacker"
    displaced = tmp_path / "registry-displaced"
    attacker.mkdir(mode=0o700)
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(registry))
    stage_d = tmp_path / "run" / "stage_d"
    stage_d.mkdir(parents=True)
    lock_path = run_lock.run_lock_path(str(stage_d))
    lock_name = os.path.basename(lock_path)
    original_open = run_lock.os.open
    replaced = False

    def replace_registry_before_lock_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if (path == lock_path or path == lock_name) and not replaced:
            replaced = True
            registry.rename(displaced)
            registry.symlink_to(attacker, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        run_lock, "os", patched_os(open=replace_registry_before_lock_open)
    )
    fd = run_lock._open_lock_file(lock_path, purpose="测试锁")
    run_lock.release_lock(fd)

    assert replaced
    assert (displaced / lock_name).is_file()
    assert list(attacker.iterdir()) == []


def test_substage_writer_lock_times_out_with_observable_context(tmp_path, monkeypatch):
    """writer 冲突必须有限等待，并报告 lock/output identity 与等待时长。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_d" / "D2"
    output.mkdir(parents=True)

    with run_lock.substage_writer_lock(str(output), timeout=1):
        with pytest.raises(RuntimeError) as raised:
            with run_lock.substage_writer_lock(str(output), timeout=0.01):
                pass

    message = str(raised.value)
    assert "lock_path=" in message
    assert f"output={output.resolve()}" in message
    assert "waited=" in message


@pytest.mark.parametrize("timeout", [True, -1, float("inf"), float("nan")])
def test_substage_writer_lock_rejects_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="非负有限数值"):
        with run_lock.substage_writer_lock(str(tmp_path / "output"), timeout=timeout):
            pass


def test_stage_c_run_lock_serializes_physical_output_alias(tmp_path, monkeypatch):
    """Stage C 输出根的真实路径与 symlink alias 必须争用同一把 run 锁。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_c"
    output.mkdir()
    alias = tmp_path / "stage_c_alias"
    alias.symlink_to(output, target_is_directory=True)

    with run_lock.stage_c_run_lock(str(alias), timeout=1):
        with pytest.raises(RuntimeError, match="获取Stage C run namespace 锁.*超时"):
            with run_lock.stage_c_run_lock(str(output), timeout=0):
                pass


def test_stage_c_run_lock_acquires_namespace_before_inode(tmp_path, monkeypatch):
    """所有 Stage C 事务必须按 namespace→inode 的固定顺序获取双锁。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_c"
    acquired = []
    original_acquire = run_lock._acquire_lock_with_timeout

    def record_acquire(lock_path, *, purpose, timeout):
        acquired.append(purpose)
        return original_acquire(lock_path, purpose=purpose, timeout=timeout)

    monkeypatch.setattr(run_lock, "_acquire_lock_with_timeout", record_acquire)
    with run_lock.stage_c_run_lock(str(output), timeout=0):
        pass

    assert "namespace" in acquired[0]
    assert "inode" in acquired[1]


def test_stage_c_run_lock_blocks_same_namespace_after_inode_replacement(
    tmp_path, monkeypatch
):
    """锁持有期间路径换 inode，第二事务仍须被 canonical namespace 锁阻塞。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_c"
    displaced = tmp_path / "stage_c-displaced"
    output.mkdir()

    entered_second_transaction = False
    second_error = None
    with pytest.raises(RuntimeError, match="输出目录.*被替换"):
        with run_lock.stage_c_run_lock(str(output), timeout=1):
            output.rename(displaced)
            output.mkdir()
            try:
                with run_lock.stage_c_run_lock(str(output), timeout=0):
                    entered_second_transaction = True
            except RuntimeError as exc:
                second_error = exc

    assert not entered_second_transaction
    assert second_error is not None
    assert "namespace" in str(second_error)
    assert "超时" in str(second_error)


def test_stage_c_and_substage_locks_use_distinct_namespaces(tmp_path, monkeypatch):
    """Stage C run 锁不得与同一目录的子阶段 writer 锁自死锁。"""
    monkeypatch.setenv("USE_TRAIN_LOCK_REGISTRY", str(tmp_path / "registry"))
    output = tmp_path / "stage_c"

    with run_lock.stage_c_run_lock(str(output), timeout=0):
        with run_lock.substage_writer_lock(str(output), timeout=0):
            pass
