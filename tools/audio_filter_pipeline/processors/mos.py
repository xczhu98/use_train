import numpy as np
import soundfile as sf
from collections import OrderedDict
from numbers import Real
import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Optional, Mapping
import os
import sys
import io
import atexit
import contextlib
import contextvars
import functools
import logging
import tempfile
import warnings
import threading
import multiprocessing
from multiprocessing.util import Finalize
import shutil
import stat
import math
import hashlib
import importlib.util
import subprocess
import types
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from core.utils.safe_load import safe_torch_load

from .batch_splitter import WeightedBatchSplitter
from .mos_contract import MOS_THRESHOLD_DIMENSIONS
from .optional_runtime import (
    AssetDigestMismatchError,
    AssetScanLimits,
    LocalAssetError,
    LocalModelAsset,
    OwnedModelTreeSnapshot,
    local_model_tree_sha256,
    open_owned_model_tree_snapshot,
    verify_local_model_asset,
)

logger = logging.getLogger(__name__)


MOS_AUDIO_CHANNEL_POLICY = "first_channel_index_0"


def _first_channel_audio(audio: Any) -> np.ndarray:
    """选择 Python 索引 0 通道，返回连续 float32 mono waveform。"""
    array = np.asarray(audio)
    if array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError("audio must contain at least one channel")
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError("audio must be one- or two-dimensional")
    return np.ascontiguousarray(array, dtype=np.float32)


class MosModelAssetBindingError(RuntimeError):
    """本地 MOS 模型树与预验证内容身份不一致。"""


class MosResultContractError(ValueError):
    """模型输出违反不可重试 schema/count/value 契约。"""


def _add_exception_note(exc: BaseException, note: str) -> None:
    """兼容 Python 3.10/3.11，为主异常记录次要失败。"""
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = getattr(exc, "__notes__", None)
    if notes is None:
        notes = []
        exc.__notes__ = notes
    notes.append(note)


@dataclass(frozen=True)
class _LocalCheckpointSnapshot:
    """由原 checkpoint descriptor 复制出的私有、只读加载快照。"""

    path: str
    descriptor_path: str
    sha256: str
    size_bytes: int


def _local_checkpoint_fd_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    """绑定文件内容身份；最终 pathname 复核忽略纯 rename 导致的 ctime 变化。"""
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


_PENDING_CHECKPOINT_SNAPSHOT_CLEANUP: Dict[str, None] = {}
_PENDING_CHECKPOINT_SNAPSHOT_CLEANUP_LOCK = threading.RLock()


def _cleanup_checkpoint_snapshot_dir(snapshot_dir: str) -> None:
    """清理 owned checkpoint snapshot；失败前先登记路径所有权。"""
    with _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP_LOCK:
        _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP[snapshot_dir] = None
    try:
        shutil.rmtree(snapshot_dir)
    except FileNotFoundError:
        pass
    except BaseException:
        raise
    else:
        pass
    with _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP_LOCK:
        _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP.pop(snapshot_dir, None)


def _retry_pending_checkpoint_snapshot_cleanup() -> None:
    """重试 checkpoint snapshot 清理，成功前始终保留路径强引用。"""
    with _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP_LOCK:
        pending = tuple(_PENDING_CHECKPOINT_SNAPSHOT_CLEANUP)
    first_error = None
    for snapshot_dir in pending:
        try:
            _cleanup_checkpoint_snapshot_dir(snapshot_dir)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                _add_exception_note(
                    first_error, f"additional checkpoint snapshot cleanup failed: {exc}",
                )
    if first_error is not None:
        raise first_error


@contextlib.contextmanager
def _open_local_checkpoint_snapshot(path: str):
    """从 O_NOFOLLOW descriptor 创建 0700/0400 owned snapshot，并复核原路径。"""
    _retry_pending_checkpoint_snapshot_cleanup()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    source_fd = os.open(path, flags)
    snapshot_fd = None
    snapshot_dir = None
    pathname_identity = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("local MOS checkpoint must be a regular file")
        pathname_identity = _local_checkpoint_fd_identity(before)
        snapshot_dir = tempfile.mkdtemp(prefix="audio-filter-mos-checkpoint-")
        os.chmod(snapshot_dir, 0o700)
        snapshot_path = os.path.join(snapshot_dir, "checkpoint.snapshot")
        output_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        )
        output_fd = os.open(snapshot_path, output_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied += len(chunk)
                offset = 0
                while offset < len(chunk):
                    offset += os.write(output_fd, chunk[offset:])
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        after = os.fstat(source_fd)
        if (
            _local_checkpoint_fd_identity(after) != pathname_identity
            or after.st_ctime_ns != before.st_ctime_ns
            or copied != before.st_size
        ):
            raise RuntimeError("local MOS checkpoint changed while snapshotting")
        os.chmod(snapshot_path, 0o400)
        snapshot_fd = os.open(snapshot_path, flags)
        yield _LocalCheckpointSnapshot(
            path=snapshot_path,
            descriptor_path=f"/proc/self/fd/{snapshot_fd}",
            sha256=digest.hexdigest(),
            size_bytes=copied,
        )
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        try:
            os.close(source_fd)
        except BaseException as exc:
            cleanup_errors.append(("source descriptor cleanup failed", exc))
        if snapshot_fd is not None:
            try:
                os.close(snapshot_fd)
            except BaseException as exc:
                cleanup_errors.append(("snapshot descriptor cleanup failed", exc))
        pathname_changed = False
        if pathname_identity is not None:
            try:
                current = os.stat(path, follow_symlinks=False)
                pathname_changed = (
                    _local_checkpoint_fd_identity(current) != pathname_identity
                )
            except OSError:
                pathname_changed = True
        if snapshot_dir is not None:
            try:
                _cleanup_checkpoint_snapshot_dir(snapshot_dir)
            except BaseException as exc:
                cleanup_errors.append(("checkpoint snapshot cleanup failed", exc))
        if pathname_changed:
            cleanup_errors.insert(0, (
                "checkpoint pathname verification failed",
                RuntimeError("local MOS checkpoint changed while loading"),
            ))
        if primary_error is not None:
            for label, cleanup_error in cleanup_errors:
                _add_exception_note(primary_error, f"{label}: {cleanup_error}")
        elif cleanup_errors:
            label, cleanup_error = cleanup_errors[0]
            for extra_label, extra_error in cleanup_errors[1:]:
                _add_exception_note(
                    cleanup_error, f"{extra_label}: {extra_error}",
                )
            raise cleanup_error


def snapshot_local_checkpoint_identity(path: str) -> dict[str, Any]:
    """返回 descriptor-bound snapshot 的内容身份，并立即清理 owned snapshot。"""
    with _open_local_checkpoint_snapshot(path) as snapshot:
        return {
            "checkpoint_sha256": snapshot.sha256,
            "checkpoint_size_bytes": snapshot.size_bytes,
        }


def _verify_expected_checkpoint_digest(
    model: str, actual: str, expected: Optional[str],
) -> None:
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"{model} checkpoint digest mismatch: expected={expected}, actual={actual}"
        )


def _verified_model_tree_digest(model: str, value: Any) -> Optional[str]:
    """校验预验证模型树 digest；旧调用方可省略。"""
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            f"{model} verified_model_tree_sha256 must be lowercase SHA-256 hex"
        )
    return value


def _canonical_model_dir(model_dir: Any, default: str) -> str:
    """把模型目录固定为真实 loader 与 worker 共用的 canonical path。"""
    selected = default if model_dir is None else os.fspath(model_dir)
    return os.path.realpath(os.path.abspath(selected))


def _create_owned_model_tree_snapshot(
    model_dir: str,
    expected: Optional[str],
    *,
    limits: AssetScanLimits = AssetScanLimits(),
):
    """创建 descriptor-bound owned snapshot，并绑定预验证 digest。"""
    snapshot = open_owned_model_tree_snapshot(model_dir, limits=limits)
    try:
        if expected is not None and snapshot.sha256 != expected:
            raise AssetDigestMismatchError(
                f"owned snapshot digest mismatch: expected {expected}, "
                f"actual {snapshot.sha256}"
            )
        return snapshot
    except BaseException as exc:
        try:
            snapshot.close()
        except BaseException as cleanup:
            if hasattr(exc, "add_note"):
                exc.add_note(f"snapshot cleanup failed: {cleanup}")
        raise


def _open_verified_model_tree_snapshot(
    model: str, model_dir: str, expected: Optional[str],
):
    """创建供 estimator 整个生命周期持有的 verified owned snapshot。"""
    try:
        return _create_owned_model_tree_snapshot(model_dir, expected)
    except LocalAssetError as exc:
        if expected is None:
            raise
        raise MosModelAssetBindingError(
            f"{model} model tree snapshot verification failed: {exc}"
        ) from exc


def _verify_owned_model_tree_snapshot(model: str, snapshot) -> None:
    """完整复核 owned snapshot 内容；任何篡改均按绑定错误 fail-closed。"""
    try:
        snapshot.verify()
    except LocalAssetError as exc:
        raise MosModelAssetBindingError(
            f"{model} owned model tree snapshot verification failed: {exc}"
        ) from exc


def _verify_owned_model_tree_manifest(model: str, snapshot) -> None:
    """快速复核 owned snapshot metadata manifest，不读取普通文件内容。"""
    try:
        snapshot.verify_manifest()
    except LocalAssetError as exc:
        raise MosModelAssetBindingError(
            f"{model} owned model tree snapshot manifest verification failed: {exc}"
        ) from exc


def _owned_model_tree_root_descriptor_path(model: str, snapshot) -> str:
    """从仍持有的 snapshot root fd 派生稳定 descriptor path。"""
    root_fd = getattr(snapshot, "_root_fd", None)
    if type(root_fd) is not int or root_fd < 0:
        # 兼容不属于现实威胁模型的旧测试替身；真实 OwnedModelTreeSnapshot
        # 必须持有 root fd，关闭后也不会走到这个分支继续加载。
        if not isinstance(snapshot, OwnedModelTreeSnapshot):
            return snapshot.path
        raise MosModelAssetBindingError(
            f"{model} owned model tree snapshot descriptor is unavailable"
        )
    descriptor_path = f"/proc/self/fd/{root_fd}"
    try:
        held = os.fstat(root_fd)
        resolved = os.stat(descriptor_path)
    except OSError as exc:
        raise MosModelAssetBindingError(
            f"{model} owned model tree snapshot descriptor is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(held.st_mode)
        or (held.st_dev, held.st_ino) != (resolved.st_dev, resolved.st_ino)
    ):
        raise MosModelAssetBindingError(
            f"{model} owned model tree snapshot descriptor binding changed"
        )
    return descriptor_path


def _owned_model_tree_descriptor_path(model: str, snapshot) -> str:
    """验证 snapshot 后返回仅在其持有期间有效的 descriptor root path。"""
    _verify_owned_model_tree_snapshot(model, snapshot)
    return _owned_model_tree_root_descriptor_path(model, snapshot)


# GPU 推理失败（显存 OOM 等）时是否允许降级 CPU 真实模型继续推理。
# ``_ALLOW_CPU_DOWNGRADE`` 保留为旧测试/调用方可见镜像；实际读取 ContextVar，
# 避免并发 Runner 相互覆盖。ContextVar 未在当前线程显式设置时回退镜像值：
# 生产拓扑每进程单配置（MOSStage/configure_mos_runtime 在主线程设置），
# 推理工作线程无需感知；显式 set 的线程则获得独立值（并发隔离）。
_ALLOW_CPU_DOWNGRADE = False
_UNSET = object()
_ALLOW_CPU_DOWNGRADE_CONTEXT = contextvars.ContextVar(
    "audio_filter_allow_cpu_downgrade", default=_UNSET,
)
_URGENT_PK_OPTIONS: Dict[str, Any] = {}
_SCOREQ_OPTIONS: Dict[str, Any] = {}
_DNSMOS_OPTIONS: Dict[str, Any] = {}
_NISQA_OPTIONS: Dict[str, Any] = {}
_UTMOS_OPTIONS: Dict[str, Any] = {}
_URGENT_PK_RUNTIME_LOCK = threading.RLock()
_URGENT_PK_PRIVATE_MODULES: Dict[str, types.ModuleType] = {}
_URGENT_PK_PRIVATE_SNAPSHOT: Optional[OwnedModelTreeSnapshot] = None
_NISQA_ATEXIT_CALLBACKS = []


def _unload_urgent_pk_private_modules() -> Dict[str, types.ModuleType]:
    """卸载仍由本模块拥有的 URGENT-PK namespace，并返回旧绑定。"""
    global _URGENT_PK_PRIVATE_MODULES
    owned = _URGENT_PK_PRIVATE_MODULES
    _URGENT_PK_PRIVATE_MODULES = {}
    for name, module in owned.items():
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
    return owned


def _detach_urgent_pk_private_snapshot() -> Optional[OwnedModelTreeSnapshot]:
    """从当前 private namespace 拆离其唯一持有的 package snapshot。"""
    global _URGENT_PK_PRIVATE_SNAPSHOT
    snapshot, _URGENT_PK_PRIVATE_SNAPSHOT = _URGENT_PK_PRIVATE_SNAPSHOT, None
    return snapshot


def _restore_urgent_pk_private_modules(
    owned: Mapping[str, types.ModuleType],
) -> None:
    """新身份加载失败时恢复未被外部占用的旧 namespace。"""
    global _URGENT_PK_PRIVATE_MODULES
    restored = {}
    for name, module in owned.items():
        current = sys.modules.get(name)
        if current is None or current is module:
            sys.modules[name] = module
            restored[name] = module
    _URGENT_PK_PRIVATE_MODULES = restored


def set_urgent_pk_options(**options: Any) -> None:
    """设置 URGENT-PK 显式资产配置，供旧调用方与缓存创建器使用。"""
    global _URGENT_PK_OPTIONS
    with _URGENT_PK_RUNTIME_LOCK:
        _URGENT_PK_OPTIONS = dict(options)


@contextlib.contextmanager
def bind_urgent_pk_options(options: Optional[Mapping[str, Any]]):
    """原子绑定任务级 URGENT-PK options，覆盖完整 estimator 推理调用。"""
    global _URGENT_PK_OPTIONS
    if options is None:
        yield
        return
    with _URGENT_PK_RUNTIME_LOCK:
        previous = _URGENT_PK_OPTIONS
        _URGENT_PK_OPTIONS = dict(options)
        try:
            yield
        finally:
            _URGENT_PK_OPTIONS = previous


def set_scoreq_options(**options: Any) -> None:
    """设置 SCOREQ ``model_dir`` 与预验证模型树 digest。"""
    global _SCOREQ_OPTIONS
    with _URGENT_PK_RUNTIME_LOCK:
        _SCOREQ_OPTIONS = dict(options)


@contextlib.contextmanager
def bind_scoreq_options(options: Optional[Mapping[str, Any]]):
    """原子绑定任务级 SCOREQ 目录与预验证模型树 digest。"""
    global _SCOREQ_OPTIONS
    if options is None:
        yield
        return
    with _URGENT_PK_RUNTIME_LOCK:
        previous = _SCOREQ_OPTIONS
        _SCOREQ_OPTIONS = dict(options)
        try:
            yield
        finally:
            _SCOREQ_OPTIONS = previous


def set_dnsmos_options(**options: Any) -> None:
    """设置 DNSMOS ``model_dir`` 与预验证模型树 digest。"""
    global _DNSMOS_OPTIONS
    with _URGENT_PK_RUNTIME_LOCK:
        _DNSMOS_OPTIONS = dict(options)


@contextlib.contextmanager
def bind_dnsmos_options(options: Optional[Mapping[str, Any]]):
    """原子绑定任务级 DNSMOS 目录与预验证模型树 digest。"""
    global _DNSMOS_OPTIONS
    if options is None:
        yield
        return
    with _URGENT_PK_RUNTIME_LOCK:
        previous = _DNSMOS_OPTIONS
        _DNSMOS_OPTIONS = dict(options)
        try:
            yield
        finally:
            _DNSMOS_OPTIONS = previous


@contextlib.contextmanager
def bind_nisqa_options(options: Optional[Mapping[str, Any]]):
    """原子绑定任务级 NISQA 本地资产配置。"""
    global _NISQA_OPTIONS
    if options is None:
        yield
        return
    with _URGENT_PK_RUNTIME_LOCK:
        previous = _NISQA_OPTIONS
        _NISQA_OPTIONS = dict(options)
        try:
            yield
        finally:
            _NISQA_OPTIONS = previous


@contextlib.contextmanager
def bind_utmos_options(options: Optional[Mapping[str, Any]]):
    """原子绑定任务级 UTMOS 本地资产配置。"""
    global _UTMOS_OPTIONS
    if options is None:
        yield
        return
    with _URGENT_PK_RUNTIME_LOCK:
        previous = _UTMOS_OPTIONS
        _UTMOS_OPTIONS = dict(options)
        try:
            yield
        finally:
            _UTMOS_OPTIONS = previous


def normalize_urgent_pk_score_range(value: Any) -> Optional[Tuple[float, float]]:
    """规范化显式或 checkpoint 声明的 URGENT-PK 分数范围。"""
    if value is None:
        return None
    if value == "mos_1_to_5":
        return (1.0, 5.0)
    try:
        if isinstance(value, Mapping):
            result = (float(value["min"]), float(value["max"]))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            result = (float(value[0]), float(value[1]))
        else:
            raise TypeError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "URGENT-PK score range 必须是 'mos_1_to_5'、[min, max] 或 {'min': ..., 'max': ...}"
        ) from exc
    if not all(math.isfinite(v) for v in result) or result[0] >= result[1]:
        raise ValueError("URGENT-PK score range 必须为有限且 min < max")
    return result


_URGENT_PK_METADATA_CACHE_MAX_SIZE = 16
_URGENT_PK_METADATA_MAX_STRING_LENGTH = 1024
_URGENT_PK_METADATA_KEYS = (
    "urgent_pk_score_range", "score_range", "encoder", "backbone",
    "model_version", "version",
)
_URGENT_PK_METADATA_RANGE_KEYS = {"urgent_pk_score_range", "score_range"}
_MISSING = object()
_URGENT_PK_METADATA_CACHE = OrderedDict()
_URGENT_PK_METADATA_CACHE_LOCK = threading.RLock()


def clear_urgent_pk_checkpoint_metadata_cache() -> None:
    """清空 URGENT-PK checkpoint metadata/digest cache 与已验证记录
    （测试与显式资源回收 hook）。"""
    with _URGENT_PK_METADATA_CACHE_LOCK:
        _URGENT_PK_METADATA_CACHE.clear()
    with _URGENT_PK_DIGEST_CACHE_LOCK:
        _URGENT_PK_DIGEST_CACHE.clear()
        _URGENT_PK_VERIFIED_ASSETS.clear()
    with _URGENT_PK_COMMIT_CACHE_LOCK:
        _URGENT_PK_COMMIT_CACHE.clear()


def _urgent_checkpoint_cache_identity(path: str) -> Tuple[str, int, int, int, int, int]:
    """返回 URGENT metadata/digest cache 使用的 pathname stat identity。"""
    realpath = os.path.realpath(path)
    stat_result = os.stat(realpath)
    return (
        realpath, stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
        stat_result.st_mtime_ns, stat_result.st_ctime_ns,
    )


class UrgentPKTrustError(ValueError):
    """URGENT-PK checkpoint/source package 必要门禁校验失败。

    门禁把可信 pickle 的 checkpoint SHA256、source git HEAD 及实际将 import 的
    ``urgentpk`` package 全量 Git blob 内容同时绑定到批准身份；安全 scanner 仍
    拒绝 symlink、特殊文件和扫描期间的路径/内容变化。
    """


_URGENT_PK_DIGEST_CACHE_MAX_SIZE = 16
_URGENT_PK_DIGEST_CACHE = OrderedDict()
_URGENT_PK_DIGEST_CACHE_LOCK = threading.RLock()
# 本进程已验证通过的 (checkpoint realpath, source realpath) → 实际 digest/commit，
# 供 metadata/provenance 记录真实资产身份。
_URGENT_PK_VERIFIED_ASSETS: Dict[Tuple[str, str], Dict[str, str]] = {}

# source git HEAD commit 缓存：key 为 git HEAD 状态的 stat identity
# （.git/HEAD + 其指向的 ref 文件 + packed-refs），commit/checkout/分支切换必然
# 改写其中某个文件 → identity 变化 → 自动失效重新 exec。仓库未变化时零 exec。
_URGENT_PK_COMMIT_CACHE_MAX_SIZE = 16
_URGENT_PK_COMMIT_CACHE = OrderedDict()
_URGENT_PK_COMMIT_CACHE_LOCK = threading.RLock()


def _sha256_file_streaming(path: str) -> str:
    """流式分块计算文件 SHA256（1MB 块），避免整文件载入内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256_digest(path: str) -> str:
    """计算 checkpoint 文件 SHA256，stat identity 缓存避免重复全读。

    缓存 key 绑定 ``realpath/dev/ino/size/mtime_ns/ctime_ns``（同 metadata
    cache），文件被替换/重写必然改变 identity → 自动失效重新计算；
    计算期间文件变化时重试一次，仍变化则明确失败。
    """
    with _URGENT_PK_DIGEST_CACHE_LOCK:
        for attempt in range(2):
            identity = _urgent_checkpoint_cache_identity(path)
            cached = _URGENT_PK_DIGEST_CACHE.get(identity)
            if cached is not None:
                _URGENT_PK_DIGEST_CACHE.move_to_end(identity)
                return cached
            digest = _sha256_file_streaming(identity[0])
            if _urgent_checkpoint_cache_identity(path) != identity:
                if attempt == 0:
                    continue
                raise RuntimeError("URGENT-PK checkpoint changed while computing digest")
            for key in tuple(_URGENT_PK_DIGEST_CACHE):
                if key[0] == identity[0] and key != identity:
                    del _URGENT_PK_DIGEST_CACHE[key]
            _URGENT_PK_DIGEST_CACHE[identity] = digest
            _URGENT_PK_DIGEST_CACHE.move_to_end(identity)
            while len(_URGENT_PK_DIGEST_CACHE) > _URGENT_PK_DIGEST_CACHE_MAX_SIZE:
                _URGENT_PK_DIGEST_CACHE.popitem(last=False)
            return digest
    raise RuntimeError("unreachable")


def _checkpoint_sha256_digest_if_exists(path: Optional[str]) -> Optional[str]:
    """缓存 key 用 lenient 版本：文件不存在（含占位测试路径）返回 None。"""
    if not path or not os.path.isfile(path):
        return None
    return _checkpoint_sha256_digest(path)


def _git_head_identity(source_dir: str) -> Optional[Tuple[Tuple, ...]]:
    """source_dir 的 git HEAD 状态 stat identity（供 commit 缓存判失效）。

    取 ``.git/HEAD`` 及其指向的 ref 文件（``ref: refs/...``）、``packed-refs``
    （ref 被打包时 loose 文件不存在，packed-refs 变化也要使缓存失效）的
    ``dev/ino/size/mtime_ns/ctime_ns``。支持 worktree/submodule 的 gitfile
    （``.git`` 为文本文件，内容 ``gitdir: <path>``）。非 git 仓库/读取失败
    返回 None（不缓存，保持原 fail-closed 行为）。
    """
    git_dir = os.path.join(source_dir, ".git")
    if os.path.isfile(git_dir):
        # worktree/submodule：.git 是 gitfile，指向真实 git dir（绝对或相对路径）
        try:
            with open(git_dir, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            return None
        if not content.startswith("gitdir:"):
            return None
        git_dir = content[len("gitdir:"):].strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.join(source_dir, git_dir)
    if not os.path.isdir(git_dir):
        return None
    head_path = os.path.realpath(os.path.join(git_dir, "HEAD"))
    try:
        head_stat = os.stat(head_path)
        with open(head_path, "r", encoding="utf-8") as f:
            head_content = f.read().strip()
    except OSError:
        return None
    identity = [
        (head_path, head_stat.st_dev, head_stat.st_ino, head_stat.st_size,
         head_stat.st_mtime_ns, head_stat.st_ctime_ns),
    ]
    if head_content.startswith("ref:"):
        ref_path = os.path.realpath(os.path.join(git_dir, head_content[4:].strip()))
        try:
            ref_stat = os.stat(ref_path)
            identity.append(
                (ref_path, ref_stat.st_dev, ref_stat.st_ino, ref_stat.st_size,
                 ref_stat.st_mtime_ns, ref_stat.st_ctime_ns)
            )
        except OSError:
            pass  # loose ref 不存在：可能已打包，packed-refs 的 stat 一并兜底
        packed_path = os.path.realpath(os.path.join(git_dir, "packed-refs"))
        try:
            packed_stat = os.stat(packed_path)
            identity.append(
                (packed_path, packed_stat.st_dev, packed_stat.st_ino,
                 packed_stat.st_size, packed_stat.st_mtime_ns, packed_stat.st_ctime_ns)
            )
        except OSError:
            pass
    return tuple(identity)


def _source_git_head(source_dir: str) -> str:
    """读取 source_dir 当前 git HEAD commit。

    用 subprocess argv ``git -C <dir> rev-parse HEAD``（禁 shell=True）；
    git 缺失 / 非 git 仓库 / 输出非法一律 fail-closed 抛错。

    commit 按 ``_git_head_identity`` 的 stat identity 缓存：仓库未变化时
    （HEAD/ref 文件无改动）直接返回缓存，不再重复 exec 子进程；commit/
    checkout/分支切换必然改写相关文件 → identity 变化 → 自动失效重新 exec。
    失败（非 git 等）不缓存，保持 fail-closed。
    """
    if not source_dir:
        raise ValueError("URGENT-PK source_dir 未配置，无法校验 git HEAD")
    identity = _git_head_identity(source_dir)
    if identity is not None:
        with _URGENT_PK_COMMIT_CACHE_LOCK:
            cached = _URGENT_PK_COMMIT_CACHE.get(identity)
            if cached is not None:
                _URGENT_PK_COMMIT_CACHE.move_to_end(identity)
                return cached
    try:
        proc = subprocess.run(
            ["git", "-C", source_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:
        raise RuntimeError(
            f"无法执行 git rev-parse（source_dir={source_dir!r}）: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"source_dir 不是有效 git 仓库或 rev-parse 失败（source_dir={source_dir!r}）: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    commit = proc.stdout.strip()
    if len(commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in commit):
        raise RuntimeError(f"git rev-parse 输出非法 commit: {commit!r}")
    if identity is not None:
        with _URGENT_PK_COMMIT_CACHE_LOCK:
            # 同 HEAD 文件的历史 identity 条目作废（同一仓库只保留当前状态一条），
            # 避免反复 checkout 把 16 条缓存挤满
            for key in tuple(_URGENT_PK_COMMIT_CACHE):
                if key[0][0] == identity[0][0] and key != identity:
                    del _URGENT_PK_COMMIT_CACHE[key]
            _URGENT_PK_COMMIT_CACHE[identity] = commit
            _URGENT_PK_COMMIT_CACHE.move_to_end(identity)
            while len(_URGENT_PK_COMMIT_CACHE) > _URGENT_PK_COMMIT_CACHE_MAX_SIZE:
                _URGENT_PK_COMMIT_CACHE.popitem(last=False)
    return commit


def _source_git_head_if_available(source_dir: Optional[str]) -> Optional[str]:
    """缓存 key 用 lenient 版本：非 git 目录/读取失败返回 None。"""
    if not source_dir or not os.path.isdir(source_dir):
        return None
    try:
        return _source_git_head(source_dir)
    except Exception:
        return None


def verify_urgent_pk_trusted_assets(
    checkpoint: str,
    source_dir: str,
    expected_sha256: Optional[str],
    expected_commit: Optional[str],
) -> Dict[str, str]:
    """weights_only=False（可信 pickle）前置强校验。

    期望 digest/commit 必须由配置显式提供且与实际一致，否则拒绝——绝不默认
    信任任意本地文件。该身份校验只是可信加载的必要门禁；真实 loader 仍会
    通过安全 tree scanner 要求整个 source tree 仅含无 symlink 的普通文件。
    返回实际 (checkpoint_sha256, source_commit) 供 metadata/provenance 记录；
    校验通过的结果记入 ``_URGENT_PK_VERIFIED_ASSETS``。
    """
    if not expected_sha256 or not expected_commit:
        raise UrgentPKTrustError(
            "URGENT-PK 需要可信 pickle 加载（weights_only=False），必须显式配置 "
            "urgent_pk_expected_sha256 与 urgent_pk_expected_commit 才能信任本地资产"
        )
    actual_sha256 = _checkpoint_sha256_digest(checkpoint)
    if str(expected_sha256).strip().lower() != actual_sha256:
        raise UrgentPKTrustError(
            f"URGENT-PK checkpoint SHA256 不匹配: 实际 {actual_sha256} != 期望 {expected_sha256}"
        )
    actual_commit = _source_git_head(source_dir)
    expected_commit = str(expected_commit).strip().lower()
    if expected_commit != actual_commit:
        raise UrgentPKTrustError(
            f"URGENT-PK source git HEAD 不匹配: 实际 {actual_commit} != 期望 {expected_commit}"
        )
    snapshot = _open_urgent_pk_pinned_package_snapshot(source_dir, expected_commit)
    try:
        package_sha256 = _verify_urgent_pk_pinned_package_snapshot(
            snapshot, source_dir, expected_commit,
        )
    finally:
        snapshot.close()
    verified = {
        "checkpoint_sha256": actual_sha256,
        "source_commit": actual_commit,
        "source_package_sha256": package_sha256,
    }
    key = (os.path.realpath(checkpoint), os.path.realpath(source_dir))
    _URGENT_PK_VERIFIED_ASSETS[key] = verified
    return verified


def urgent_pk_verified_assets(checkpoint: Optional[str], source_dir: Optional[str]) -> Dict[str, str]:
    """返回已强校验通过的资产身份记录（无则空 dict），供 status/provenance 使用。"""
    if not checkpoint or not source_dir:
        return {}
    return dict(
        _URGENT_PK_VERIFIED_ASSETS.get(
            (os.path.realpath(checkpoint), os.path.realpath(source_dir)), {}
        )
    )


def _pinned_urgent_pk_package_blobs(
    source_dir: str, expected_commit: str,
) -> dict[bytes, str]:
    """读取批准 commit 中实际 ``urgentpk`` package 的 blob 清单。"""
    try:
        proc = subprocess.run(
            ["git", "-C", source_dir, "ls-tree", "-r", "-z", expected_commit, "--", "urgentpk"],
            capture_output=True,
            timeout=60,
        )
    except Exception as exc:
        raise UrgentPKTrustError(
            f"无法读取 URGENT-PK 批准源码树（source_dir={source_dir!r}）: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise UrgentPKTrustError(
            "无法读取 URGENT-PK 批准源码树: "
            + (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip()
        )
    result: dict[bytes, str] = {}
    for record in proc.stdout.split(b"\0"):
        if not record:
            continue
        try:
            header, path = record.split(b"\t", 1)
            mode, object_type, object_id = header.split(b" ", 2)
        except ValueError as exc:
            raise UrgentPKTrustError("URGENT-PK 批准源码树条目格式非法") from exc
        if (
            mode not in {b"100644", b"100755"}
            or object_type != b"blob"
            or not path.startswith(b"urgentpk/")
            or path == b"urgentpk/"
            or b"\0" in path
            or any(component in {b"", b".", b".."} for component in path.split(b"/"))
        ):
            raise UrgentPKTrustError("URGENT-PK 批准源码树包含不支持的 package 条目")
        try:
            digest = object_id.decode("ascii")
        except UnicodeDecodeError as exc:
            raise UrgentPKTrustError("URGENT-PK 批准源码 blob identity 非 ASCII") from exc
        if len(digest) not in {40, 64} or any(char not in "0123456789abcdef" for char in digest):
            raise UrgentPKTrustError("URGENT-PK 批准源码 blob identity 非法")
        relative = path[len(b"urgentpk/"):]
        if relative in result:
            raise UrgentPKTrustError("URGENT-PK 批准源码树包含重复 package 路径")
        result[relative] = digest
    if not result:
        raise UrgentPKTrustError("URGENT-PK 批准源码树不包含 urgentpk package 文件")
    return result


def _snapshot_urgent_pk_file_blob(
    snapshot: OwnedModelTreeSnapshot, relative_path: bytes, source_dir: str,
) -> str:
    """以 Git blob 算法计算 owned snapshot 文件身份，不读取 live source。"""
    try:
        relative = os.fsdecode(relative_path)
    except UnicodeDecodeError as exc:
        raise UrgentPKTrustError("URGENT-PK package 路径不是有效 UTF-8") from exc
    with snapshot.loader_path_lease() as snapshot_root:
        path = os.path.join(snapshot_root, relative)
        try:
            with open(path, "rb", buffering=0) as source:
                proc = subprocess.Popen(
                    ["git", "-C", source_dir, "hash-object", "--stdin"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                assert proc.stdin is not None
                try:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        proc.stdin.write(chunk)
                    proc.stdin.close()
                    proc.stdin = None
                    stdout, stderr = proc.communicate(timeout=60)
                except BaseException:
                    proc.kill()
                    proc.communicate()
                    raise
        except OSError as exc:
            raise UrgentPKTrustError(
                f"无法读取 URGENT-PK owned snapshot 文件: {relative!r}"
            ) from exc
    if proc.returncode != 0:
        raise UrgentPKTrustError(
            "无法计算 URGENT-PK snapshot blob identity: "
            + stderr.decode("utf-8", "replace").strip()
        )
    try:
        digest = stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise UrgentPKTrustError("URGENT-PK snapshot blob identity 非 ASCII") from exc
    if len(digest) not in {40, 64} or any(char not in "0123456789abcdef" for char in digest):
        raise UrgentPKTrustError("URGENT-PK snapshot blob identity 非法")
    return digest


def _verify_urgent_pk_pinned_package_snapshot(
    snapshot: OwnedModelTreeSnapshot, source_dir: str, expected_commit: str,
) -> str:
    """把真实将 import 的完整 package snapshot 逐 blob 绑定到批准 commit。"""
    try:
        snapshot.verify()
    except LocalAssetError as exc:
        raise UrgentPKTrustError(
            f"URGENT-PK package snapshot 安全扫描失败: {exc}"
        ) from exc
    expected = _pinned_urgent_pk_package_blobs(source_dir, expected_commit)
    actual = {
        entry.path
        for entry in snapshot.manifest
        if stat.S_ISREG(entry.file_type)
    }
    expected_directories = {b""}
    for relative_path in expected:
        parent = relative_path.rpartition(b"/")[0]
        while parent:
            expected_directories.add(parent)
            parent = parent.rpartition(b"/")[0]
    actual_directories = {
        entry.path
        for entry in snapshot.manifest
        if stat.S_ISDIR(entry.file_type)
    }
    if actual != set(expected) or actual_directories != expected_directories:
        missing = len(set(expected) - actual)
        unexpected = len(actual - set(expected))
        unexpected_directories = len(actual_directories - expected_directories)
        raise UrgentPKTrustError(
            "URGENT-PK package 内容与批准 commit 不一致: "
            f"missing={missing}, unexpected={unexpected}, "
            f"unexpected_directories={unexpected_directories}"
        )
    for relative_path, expected_blob in expected.items():
        actual_blob = _snapshot_urgent_pk_file_blob(snapshot, relative_path, source_dir)
        if actual_blob != expected_blob:
            raise UrgentPKTrustError(
                "URGENT-PK package 内容与批准 blob 不一致: "
                + os.fsdecode(relative_path)
            )
    try:
        snapshot.verify_manifest()
    except LocalAssetError as exc:
        raise UrgentPKTrustError(
            f"URGENT-PK package snapshot manifest 校验失败: {exc}"
        ) from exc
    return snapshot.sha256


def _open_urgent_pk_pinned_package_snapshot(
    source_dir: str, expected_commit: str,
) -> OwnedModelTreeSnapshot:
    """创建仅含批准 blob 内容的 owned package snapshot，失败必清理。"""
    source_root = os.path.realpath(source_dir)
    if _source_git_head(source_root) != expected_commit:
        raise UrgentPKTrustError("URGENT-PK source git HEAD 在 snapshot 前发生变化")
    package_dir = os.path.join(source_root, "urgentpk")
    snapshot = open_owned_model_tree_snapshot(package_dir)
    try:
        _verify_urgent_pk_pinned_package_snapshot(snapshot, source_root, expected_commit)
        if _source_git_head(source_root) != expected_commit:
            raise UrgentPKTrustError("URGENT-PK source git HEAD 在 snapshot 后发生变化")
        return snapshot
    except BaseException:
        snapshot.close()
        raise


def _metadata_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and value.bit_length() <= 64:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= _URGENT_PK_METADATA_MAX_STRING_LENGTH:
        return value
    return _MISSING


def _metadata_range(value: Any) -> Any:
    if value is None:
        return None
    if value == "mos_1_to_5":
        return value
    if isinstance(value, Mapping):
        if set(map(str, value)) != {"min", "max"}:
            return _MISSING
        result = {str(key): item for key, item in value.items()}
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        result = list(value)
    else:
        return _MISSING
    try:
        normalize_urgent_pk_score_range(result)
    except ValueError:
        return _MISSING
    return result


def _urgent_pk_trusted_load_gate(checkpoint: str) -> None:
    """weights_only=False（可信 pickle）前置强校验门禁。

    期望 digest/commit 来自当前绑定的 URGENT-PK options，实际校验对象是本次
    即将加载的 checkpoint 文件本身（与 options.checkpoint 可能不同，如
    metadata 读取显式传入路径）；缺失或不匹配抛 ``UrgentPKTrustError``
    fail-closed，绝不静默信任任意本地文件。
    """
    options = _URGENT_PK_OPTIONS
    verify_urgent_pk_trusted_assets(
        checkpoint,
        options.get("source_dir"),
        options.get("expected_sha256"),
        options.get("expected_commit"),
    )


def _load_urgent_pk_checkpoint(checkpoint: str) -> Any:
    """先走中央安全加载器；仅在可信资产门禁后显式启用 pickle。"""
    try:
        return safe_torch_load(
            checkpoint,
            map_location="cpu",
            purpose="read URGENT-PK checkpoint metadata safely",
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        requires_unsafe = (
            "requires non-numpy pickled global" in message
            or "does not support weights_only" in message
        )
        if not requires_unsafe:
            raise

        # 官方 checkpoint 含 model_mod.Config；旧 PyTorch 又无法提供安全模式。
        # 两种情况都必须先固定校验 checkpoint digest 与 source commit；这只是
        # 必要门禁，真实 loader 的 symlink-free regular-file tree 扫描仍独立执行。
        # 随后通过中央入口显式声明 unsafe 目的，绝不在本模块直接调用 torch.load。
        _urgent_pk_trusted_load_gate(checkpoint)
        try:
            return safe_torch_load(
                checkpoint,
                map_location="cpu",
                allow_unsafe=True,
                purpose="load digest/commit-verified URGENT-PK checkpoint",
            )
        except ModuleNotFoundError as fallback_exc:
            if getattr(fallback_exc, "name", None) not in {None, "model_mod"}:
                raise
            # 保留原始安全加载失败，供调用方注册可信 source 中的 model_mod 后重试。
            raise exc from fallback_exc


def _extract_urgent_pk_metadata(data: Any) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}

    metadata: Dict[str, Any] = {}
    sources = [data]
    for source_name in ("hyper_parameters", "cfg", "config"):
        source = data.get(source_name)
        if source is not None:
            sources.append(source)

    for source in sources:
        for key in _URGENT_PK_METADATA_KEYS:
            if isinstance(source, Mapping):
                value = source.get(key, _MISSING)
            else:
                value = getattr(source, key, _MISSING)
            if value is _MISSING:
                continue
            pure = _metadata_range(value) if key in _URGENT_PK_METADATA_RANGE_KEYS else _metadata_scalar(value)
            if pure is not _MISSING:
                metadata[key] = pure
    return metadata


def read_urgent_pk_checkpoint_metadata(checkpoint: str) -> Mapping[str, Any]:
    """读取并缓存本地可信 checkpoint 的轻量 metadata。

    cache key 绑定 ``realpath/dev/ino/size/mtime_ns/ctime_ns``，锁内加载保证同 identity
    并发 single-flight；只缓存白名单字段及有界的纯数据副本。
    文件在读取期间变化时重试一次，仍变化则明确失败。
    """
    with _URGENT_PK_METADATA_CACHE_LOCK:
        for attempt in range(2):
            identity = _urgent_checkpoint_cache_identity(checkpoint)
            cached = _URGENT_PK_METADATA_CACHE.get(identity)
            if cached is not None:
                _URGENT_PK_METADATA_CACHE.move_to_end(identity)
                return copy.deepcopy(cached)

            data = _load_urgent_pk_checkpoint(identity[0])
            after = _urgent_checkpoint_cache_identity(checkpoint)
            if after != identity:
                if attempt == 0:
                    continue
                raise RuntimeError("URGENT-PK checkpoint changed while reading metadata")

            metadata = _extract_urgent_pk_metadata(data)
            for key in tuple(_URGENT_PK_METADATA_CACHE):
                if key[0] == identity[0] and key != identity:
                    del _URGENT_PK_METADATA_CACHE[key]
            _URGENT_PK_METADATA_CACHE[identity] = copy.deepcopy(metadata)
            _URGENT_PK_METADATA_CACHE.move_to_end(identity)
            while len(_URGENT_PK_METADATA_CACHE) > _URGENT_PK_METADATA_CACHE_MAX_SIZE:
                _URGENT_PK_METADATA_CACHE.popitem(last=False)
            return copy.deepcopy(metadata)
    raise RuntimeError("unreachable")


def load_urgent_pk_model_module(source_dir: str, expected_commit: Optional[str] = None):
    """串行加载批准 Git blob 绑定的 URGENT-PK 私有 namespace。"""
    if expected_commit is None:
        expected_commit = _URGENT_PK_OPTIONS.get("expected_commit")
    if (
        type(expected_commit) is not str
        or len(expected_commit) != 40
        or any(char not in "0123456789abcdef" for char in expected_commit)
    ):
        raise ValueError("URGENT-PK expected commit must be lowercase 40-character git commit")
    with _URGENT_PK_RUNTIME_LOCK:
        # import 生成的 ``__pycache__`` 不属于 source identity 变化；加载窗口禁写
        # bytecode，避免前后完整树摘要被解释器自身的缓存写入扰动。
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        snapshot = _open_urgent_pk_pinned_package_snapshot(source_dir, expected_commit)
        try:
            module = _load_urgent_pk_model_module_from_snapshot(
                source_dir, expected_commit, snapshot,
            )
            if _URGENT_PK_PRIVATE_SNAPSHOT is not snapshot:
                snapshot.close()
            return module
        except BaseException:
            if _URGENT_PK_PRIVATE_SNAPSHOT is not snapshot:
                snapshot.close()
            raise
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode


def _load_urgent_pk_model_module_from_snapshot(
    source_dir: str, expected_commit: str, snapshot: OwnedModelTreeSnapshot,
):
    """从显式 source_dir 以独立 namespace 加载 ``urgentpk.urgentpk_model``。

    标准 ``import urgentpk`` 会受全局 ``sys.modules`` 污染，多个 checkpoint/source
    在同进程共存时可错误复用先加载的模块。本 loader 以 canonical source path 的
    hash 构造私有 package namespace，并保留所有既有 namespace，故不会删除或污染
    其他资产正在使用的模块；package-relative import 仍能在私有 package 内解析。

    真实的 ``urgentpk_model.py``/``urgentpk_encoder.py`` 等使用绝对自引用 import
    （``from urgentpk.resnet import ...``），因此加载前把私有 package 别名注册到
    顶层 ``urgentpk`` 名（其 ``__path__`` 指向经批准 blob 验证的 owned snapshot，
    子模块只能从该 snapshot 加载）。官方 checkpoint 还以顶层模块名 ``model_mod`` pickle 了 ``Config``
    （训练时 model_mod.py 在 sys.path 上），source 内存在 model_mod.py 时同样
    注册顶层别名以供 torch.load 反序列化。两个顶层名已被其他资产占用时均
    fail-closed 抛 ImportError，绝不静默混用两个 source——同一进程只支持一个
    使用绝对 import 的 URGENT-PK 资产。
    """
    global _URGENT_PK_PRIVATE_SNAPSHOT
    configured_source_root = os.path.realpath(source_dir)
    _verify_urgent_pk_pinned_package_snapshot(
        snapshot, configured_source_root, expected_commit,
    )
    package_dir = _owned_model_tree_descriptor_path("URGENT-PK", snapshot)
    package_file = os.path.join(package_dir, "__init__.py")
    model_file = os.path.join(package_dir, "urgentpk_model.py")
    if not os.path.isfile(package_file) or not os.path.isfile(model_file):
        raise MosModelAssetBindingError(
            "URGENT-PK approved package snapshot is missing required modules"
        )
    package_realpath = os.path.realpath(package_file)
    model_realpath = os.path.realpath(model_file)
    snapshot_root = os.path.realpath(package_dir)
    if os.path.commonpath([snapshot_root, package_realpath]) != snapshot_root:
        raise MosModelAssetBindingError(
            f"URGENT-PK package module escaped owned snapshot: {package_file!r}"
        )
    if os.path.commonpath([snapshot_root, model_realpath]) != snapshot_root:
        raise MosModelAssetBindingError(
            f"URGENT-PK model module escaped owned snapshot: {model_file!r}"
        )
    digest = hashlib.sha256(
        f"{configured_source_root}\0{expected_commit}\0{snapshot.sha256}".encode("utf-8")
    ).hexdigest()[:20]
    package_name = f"_urgent_pk_{digest}"
    module_name = f"{package_name}.urgentpk_model"
    module = sys.modules.get(module_name)
    if module is not None:
        if _URGENT_PK_PRIVATE_MODULES.get(module_name) is not module:
            raise RuntimeError(
                f"URGENT-PK isolated namespace ownership mismatch: {module_name!r}"
            )
        return module
    previous_owned = _unload_urgent_pk_private_modules()
    previous_snapshot = _detach_urgent_pk_private_snapshot()
    package_spec = importlib.util.spec_from_file_location(
        package_name, package_file, submodule_search_locations=[package_dir],
    )
    if package_spec is None or package_spec.loader is None:
        _restore_urgent_pk_private_modules(previous_owned)
        _URGENT_PK_PRIVATE_SNAPSHOT = previous_snapshot
        raise ImportError(f"无法为 URGENT-PK package 创建 loader: {package_file!r}")
    package = importlib.util.module_from_spec(package_spec)
    existing_private = {name for name in sys.modules if name == package_name or name.startswith(f"{package_name}.")}
    existing_alias = {name for name in sys.modules if name == "urgentpk" or name.startswith("urgentpk.")}
    model_mod_created = False
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[package_name] = package
    try:
        # 绝对自引用 import（``from urgentpk.resnet import ...``）要求顶层
        # ``urgentpk`` 名可解析：把私有 package 别名过去，子模块经其 __path__
        # 仍从 owned snapshot 加载。已被其他 source 占用时 fail-closed。
        alias = sys.modules.get("urgentpk")
        if alias is None:
            sys.modules["urgentpk"] = package
        else:
            alias_file = getattr(alias, "__file__", None)
            if not alias_file or os.path.realpath(alias_file) != package_realpath:
                raise ImportError(
                    f"URGENT-PK 顶层 'urgentpk' 已被其他 source 占用: {alias_file!r} != {package_file!r}"
                )
        package_spec.loader.exec_module(package)
        # 真实 package 的 __init__ 已通过相对 import 加载 model 模块（``from
        # .urgentpk_model import AbTestModel``），直接复用同一模块对象，避免重复执行。
        module = sys.modules.get(module_name)
        if module is None:
            model_spec = importlib.util.spec_from_file_location(module_name, model_file)
            if model_spec is None or model_spec.loader is None:
                raise ImportError(f"无法为 URGENT-PK model 创建 loader: {model_file!r}")
            module = importlib.util.module_from_spec(model_spec)
            sys.modules[module_name] = module
            model_spec.loader.exec_module(module)
        elif os.path.realpath(getattr(module, "__file__", "")) != model_realpath:
            raise ImportError(
                f"URGENT-PK model module path mismatch: {getattr(module, '__file__', None)!r} != {model_file!r}"
            )
        # 官方 checkpoint 以顶层模块名 ``model_mod`` pickle 了 ``Config``，torch.load
        # 反序列化要求顶层 ``model_mod`` 可解析；source 内存在该文件时经私有 package
        # 加载后别名到顶层，与 ``urgentpk`` 别名同规约（fail-closed、单资产）。
        model_mod_file = os.path.join(package_dir, "model_mod.py")
        if os.path.isfile(model_mod_file):
            model_mod_alias = sys.modules.get("model_mod")
            if model_mod_alias is None:
                sys.modules["model_mod"] = importlib.import_module(f"{package_name}.model_mod")
                model_mod_created = True
            else:
                model_mod_alias_file = getattr(model_mod_alias, "__file__", None)
                if not model_mod_alias_file or os.path.realpath(model_mod_alias_file) != os.path.realpath(model_mod_file):
                    raise ImportError(
                        f"URGENT-PK 顶层 'model_mod' 已被其他模块占用: {model_mod_alias_file!r} != {model_mod_file!r}"
                    )
        # package-relative import 产生的每个私有子模块与顶层别名子模块同样属于
        # 资产边界；逐个校验 __file__ 的 canonical path，禁止 helper 等借 symlink
        # 越出 source。
        for name, imported in tuple(sys.modules.items()):
            if (
                name == package_name or name.startswith(f"{package_name}.")
                or name == "urgentpk" or name.startswith("urgentpk.")
                or name == "model_mod"
            ):
                imported_file = getattr(imported, "__file__", None)
                if imported_file and os.path.commonpath([
                    snapshot_root, os.path.realpath(imported_file),
                ]) != snapshot_root:
                    raise ImportError(
                        f"URGENT-PK private module loaded outside owned snapshot: {imported_file!r}"
                    )
        _verify_owned_model_tree_snapshot("URGENT-PK", snapshot)
        post_import_snapshot = _open_urgent_pk_pinned_package_snapshot(
            configured_source_root, expected_commit,
        )
        try:
            _verify_urgent_pk_pinned_package_snapshot(
                post_import_snapshot, configured_source_root, expected_commit,
            )
        finally:
            post_import_snapshot.close()
        created_names = {
            name for name in sys.modules
            if (
                (name == package_name or name.startswith(f"{package_name}."))
                and name not in existing_private
            ) or (
                (name == "urgentpk" or name.startswith("urgentpk."))
                and name not in existing_alias
            ) or (name == "model_mod" and model_mod_created)
        }
        _URGENT_PK_PRIVATE_MODULES.update({
            name: sys.modules[name] for name in created_names
        })
        _URGENT_PK_PRIVATE_SNAPSHOT = snapshot
        if previous_snapshot is not None:
            previous_snapshot.close()
    except Exception:
        # package-relative import 可能创建 helper 子模块。仅撤销本次新增的当前
        # 私有 namespace 条目与顶层 ``urgentpk``/``model_mod`` 别名条目，绝不触碰
        # 预先存在的同 namespace 条目、其他 source 的模块。
        for name in tuple(sys.modules):
            if (name == package_name or name.startswith(f"{package_name}.")) and name not in existing_private:
                sys.modules.pop(name, None)
            elif (name == "urgentpk" or name.startswith("urgentpk.")) and name not in existing_alias:
                sys.modules.pop(name, None)
        if model_mod_created:
            sys.modules.pop("model_mod", None)
        _restore_urgent_pk_private_modules(previous_owned)
        _URGENT_PK_PRIVATE_SNAPSHOT = previous_snapshot
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    loaded_path = os.path.realpath(getattr(module, "__file__", ""))
    if (
        loaded_path != os.path.realpath(model_file)
        or os.path.commonpath([snapshot_root, loaded_path]) != snapshot_root
    ):
        raise ImportError(f"URGENT-PK loader loaded outside owned snapshot: {loaded_path!r}")
    return module


def set_allow_cpu_downgrade(allow: bool) -> None:
    """设置是否允许 GPU 推理失败时降级 CPU（进程级开关，幂等）。

    同时写入全局镜像（旧调用方/工作线程回退读取）与当前线程的 ContextVar
    （并发隔离：显式 set 的线程读到独立值）。
    """
    global _ALLOW_CPU_DOWNGRADE
    _ALLOW_CPU_DOWNGRADE = bool(allow)
    _ALLOW_CPU_DOWNGRADE_CONTEXT.set(bool(allow))


def cpu_downgrade_allowed() -> bool:
    """当前线程是否允许 GPU→CPU 降级。

    线程内显式 ``set_allow_cpu_downgrade`` 过则以线程值为准；否则回退进程级
    镜像（保持 MOSStage 主线程设置、推理工作线程继承的既有语义）。
    """
    value = _ALLOW_CPU_DOWNGRADE_CONTEXT.get(_UNSET)
    if value is not _UNSET:
        return bool(value)
    return bool(_ALLOW_CPU_DOWNGRADE)


def _cpu_downgrade_disallowed_message(model_name: str, reason: str) -> str:
    """禁止 CPU 降级时的报错信息（fail-closed）。"""
    return (
        f"{model_name} GPU 推理失败且已禁用 CPU 降级"
        f"（mos_allow_cpu_downgrade=false）: {reason or 'unknown'}"
    )

_USE_TRAIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
THIRD_PARTY_DIR = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(_USE_TRAIN_ROOT, "third_party_scripts")

_CWD_LOCK = threading.Lock()

# ── MOS registry ──
@dataclass(frozen=True)
class MOSSpec:
    """MOS 模型的稳定元数据、raw 输出与 artifact 分数约定。"""

    name: str
    target_sr: int
    score_kind: str
    primary_score_key: str
    safe_total_seconds: float
    speed_per_second: float
    selectable: bool
    languages: Tuple[str, ...]
    validate_threshold: Optional[Tuple[str, ...]] = None
    artifact_score_keys: Tuple[Tuple[str, str], ...] = ()
    version: str = "builtin"


_MOS_RESULT_MARKERS = frozenset({"error", "fallback", "skipped"})


def _finite_mos_number(value: Any, label: str) -> float:
    """将 JSON-safe 有限实数规范为 float；bool 明确拒绝。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _artifact_mos_number(value: Any, label: str) -> float:
    """校验写入 artifact 的 MOS 分数为有限实数；不修改模型原始输出。

    官方 DNSMOS 的多项式拟合外推对极端音频会系统性输出 [1.0, 5.0] 量表外的
    分数（实测 sig/ovrl/bak 低至 0.82）；artifact 记录真实模型输出，由下游
    选择阈值决定 keep/reject，本层不做钳位也不做量表拦截。
    """
    return _finite_mos_number(value, label)


def _bounded_error_detail(exc: BaseException, limit: int = 384) -> str:
    """保留底层异常类型与有界消息，避免严格 error artifact 无界膨胀。"""
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) <= limit:
        return detail
    return detail[:limit - 3] + "..."


def _model_parameters_device(model: Any, label: str) -> str:
    """返回模型参数的唯一 device；缺失或混用 device 时 fail-closed。"""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise ValueError(f"Step3 {label} model parameters are unavailable")
    devices = {str(parameter.device).lower() for parameter in parameters()}
    if len(devices) != 1:
        raise ValueError(f"Step3 {label} model parameters do not share one device")
    return next(iter(devices))


def _attested_checkpoint_sha256(estimator: Any, label: str) -> Optional[str]:
    """核对单文件模型已加载 checkpoint 与构造期望 digest。"""
    actual = getattr(estimator, "_checkpoint_sha256", None)
    expected = getattr(estimator, "_expected_checkpoint_sha256", None)
    if expected is not None and actual != expected:
        raise ValueError(
            f"Step3 {label} checkpoint digest mismatch: "
            f"expected={expected}, actual={actual}"
        )
    return actual


_MOS_REGISTRY: Dict[str, type] = {}
_MOS_SPECS: Dict[str, MOSSpec] = {}
_ESTIMATOR_LIFECYCLE_INIT_LOCK = threading.Lock()


class _EstimatorLifecycleState:
    """协调 estimator 推理借用、缓存租约与独占生命周期。"""

    def __init__(self):
        self.condition = threading.Condition()
        self.active_calls = 0
        self.active_by_thread: Dict[int, int] = {}
        self.mutation_owner: Optional[int] = None
        self.mutation_depth = 0
        self.phase = "open"
        self.retired = False
        self.cache_leases = 0
        self.retirement_cleanup_started = False
        self.retirement_cleanup_finished = False


def _estimator_lifecycle_state(estimator) -> _EstimatorLifecycleState:
    state = getattr(estimator, "_mos_lifecycle_state", None)
    if state is not None:
        return state
    with _ESTIMATOR_LIFECYCLE_INIT_LOCK:
        state = getattr(estimator, "_mos_lifecycle_state", None)
        if state is None:
            state = _EstimatorLifecycleState()
            estimator._mos_lifecycle_state = state
        return state


def _lifecycle_unavailable_message(phase: str) -> str:
    if phase == "broken":
        return "MOS estimator is broken; only close retry is allowed"
    return f"MOS estimator is {phase}"


def _wrap_estimator_lifecycle(cls):
    """给 estimator 的推理与独占生命周期操作增加并发门禁。"""
    for method_name in ("compute", "compute_batch"):
        method = getattr(cls, method_name, None)
        if not callable(method):
            continue

        @functools.wraps(method)
        def borrowed(self, *args, __method=method, **kwargs):
            state = _estimator_lifecycle_state(self)
            thread_id = threading.get_ident()
            with state.condition:
                while (
                    state.mutation_owner is not None
                    and state.mutation_owner != thread_id
                ):
                    state.condition.wait()
                if state.phase != "open":
                    raise RuntimeError(_lifecycle_unavailable_message(state.phase))
                state.active_calls += 1
                state.active_by_thread[thread_id] = (
                    state.active_by_thread.get(thread_id, 0) + 1
                )
            try:
                return __method(self, *args, **kwargs)
            finally:
                with state.condition:
                    state.active_calls -= 1
                    remaining = state.active_by_thread[thread_id] - 1
                    if remaining:
                        state.active_by_thread[thread_id] = remaining
                    else:
                        del state.active_by_thread[thread_id]
                    state.condition.notify_all()

        setattr(cls, method_name, borrowed)

    for method_name in ("runtime_attestation", "reset", "close"):
        method = getattr(cls, method_name, None)
        if not callable(method):
            continue
        is_close = method_name == "close"

        @functools.wraps(method)
        def exclusive(self, *args, __method=method, __is_close=is_close, **kwargs):
            state = _estimator_lifecycle_state(self)
            thread_id = threading.get_ident()
            with state.condition:
                while (
                    state.mutation_owner is not None
                    and state.mutation_owner != thread_id
                ):
                    state.condition.wait()
                if __is_close:
                    if state.phase == "closed":
                        return None
                    if state.phase == "closing" and state.mutation_owner != thread_id:
                        raise RuntimeError("MOS estimator is closing")
                elif state.phase != "open":
                    raise RuntimeError(_lifecycle_unavailable_message(state.phase))
                if state.mutation_owner is None:
                    state.mutation_owner = thread_id
                state.mutation_depth += 1
                if __is_close:
                    state.phase = "closing"
                own_calls = state.active_by_thread.get(thread_id, 0)
                while state.active_calls > own_calls:
                    state.condition.wait()
            succeeded = False
            try:
                result = __method(self, *args, **kwargs)
                succeeded = True
                return result
            finally:
                with state.condition:
                    if __is_close:
                        state.phase = "closed" if succeeded else "broken"
                    state.mutation_depth -= 1
                    if state.mutation_depth == 0:
                        state.mutation_owner = None
                    state.condition.notify_all()

        setattr(cls, method_name, exclusive)
    return cls


def register_mos(name: str, spec: MOSSpec):
    """注册 MOS estimator 与其 spec；重复名称在导入期立即拒绝。"""
    normalized = name.strip().lower()
    if normalized != spec.name:
        raise ValueError(f"MOS registry name mismatch: {name!r} != {spec.name!r}")

    def decorator(cls):
        if normalized in _MOS_REGISTRY or normalized in _MOS_SPECS:
            raise ValueError(f"duplicate MOS spec: {normalized}")
        _MOS_REGISTRY[normalized] = _wrap_estimator_lifecycle(cls)
        _MOS_SPECS[normalized] = spec
        return cls
    return decorator


def get_mos_spec(name: str) -> MOSSpec:
    """返回名称对应的 MOSSpec；未知模型显式失败。"""
    normalized = str(name).strip().lower()
    try:
        return _MOS_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown MOS type: {name}. Available: {sorted(_MOS_SPECS)}") from exc


def list_mos_specs() -> Mapping[str, MOSSpec]:
    """返回只读的已注册 MOSSpec 映射。"""
    return MappingProxyType(_MOS_SPECS)


def normalize_selectable_mos_models(models: Any) -> Tuple[str, ...]:
    """规范化非空、无重复且可由 selection policy 选择的模型 tuple。"""
    if isinstance(models, (str, bytes)) or not isinstance(models, (list, tuple)):
        raise ValueError("MOS models must be a nonempty list or tuple")
    normalized: List[str] = []
    seen = set()
    for model in models:
        if type(model) is not str or not model.strip():
            raise ValueError("MOS model names must be nonblank strings")
        name = model.strip().lower()
        if name in seen:
            raise ValueError(f"duplicate MOS model: {name}")
        spec = get_mos_spec(name)
        if not spec.selectable:
            raise ValueError(f"MOS type is not selectable: {name}")
        seen.add(name)
        normalized.append(name)
    if not normalized:
        raise ValueError("MOS models must be nonempty")
    return tuple(normalized)


def mos_artifact_score_schema(model: str) -> Mapping[str, str]:
    """返回 artifact 维度到 estimator raw key 的稳定映射。"""
    spec = get_mos_spec(model)
    pairs = spec.artifact_score_keys or (("score", spec.primary_score_key),)
    return MappingProxyType(dict(pairs))


def map_mos_raw_result(raw: Any, models: Any) -> Dict[str, Any]:
    """把 estimator raw result 严格映射为 Step3 artifact 分数。"""
    selected = normalize_selectable_mos_models(models)
    if not isinstance(raw, Mapping) or set(raw) - {"audio_path"} != set(selected):
        raise ValueError("MOS runtime result must exactly match selected models")
    mapped: Dict[str, Any] = {}
    for model in selected:
        result = raw[model]
        if not isinstance(result, Mapping):
            raise ValueError(f"{model} result must be a mapping")
        for marker in _MOS_RESULT_MARKERS:
            if result.get(marker):
                raise ValueError(f"{model} result is {marker}")
        schema = mos_artifact_score_schema(model)
        if set(result) != set(schema.values()):
            raise ValueError(f"{model} result keys must exactly match registered score schema")
        values = {
            dimension: _artifact_mos_number(
                result[raw_key], f"{model} {dimension} score",
            )
            for dimension, raw_key in schema.items()
        }
        mapped[model] = values["score"] if tuple(schema) == ("score",) else values
    return mapped


def validate_mos_artifact_scores(scores: Any, models: Any) -> Dict[str, Any]:
    """严格校验 artifact score，拒绝缺失、额外维度与非有限值。"""
    selected = normalize_selectable_mos_models(models)
    if not isinstance(scores, Mapping) or set(scores) != set(selected):
        raise ValueError("MOS scores must exactly match selected models")
    validated: Dict[str, Any] = {}
    for model in selected:
        dimensions = tuple(mos_artifact_score_schema(model))
        value = scores[model]
        if dimensions == ("score",):
            validated[model] = _artifact_mos_number(value, f"{model} score")
            continue
        if not isinstance(value, Mapping) or set(value) != set(dimensions):
            raise ValueError(f"{model} scores must exactly contain {'/'.join(dimensions)}")
        validated[model] = {
            dimension: _artifact_mos_number(
                value[dimension], f"{model} {dimension} score",
            )
            for dimension in dimensions
        }
    return validated


def validate_mos_threshold(model: str, threshold: Any) -> Any:
    """按 registry 声明严格校验单模型 threshold。"""
    dimensions = tuple(mos_artifact_score_schema(model))
    if dimensions == ("score",):
        if isinstance(threshold, Mapping):
            raise ValueError(f"{model} threshold must be scalar")
        value = _finite_mos_number(threshold, f"{model} threshold")
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"{model} threshold must be in [1.0, 5.0]")
        return value
    if not isinstance(threshold, Mapping) or set(threshold) != set(dimensions):
        raise ValueError(f"{model} threshold must exactly contain {'/'.join(dimensions)}")
    values = {
        dimension: _finite_mos_number(
            threshold[dimension], f"{model} {dimension} threshold",
        )
        for dimension in dimensions
    }
    if any(not 1.0 <= value <= 5.0 for value in values.values()):
        raise ValueError(f"{model} thresholds must be in [1.0, 5.0]")
    return values


def mos_model_passed(model: str, score: Any, threshold: Any) -> bool:
    """按 registry 约定计算单模型通过状态；多维模型使用逐维 AND。"""
    validated_score = validate_mos_artifact_scores({model: score}, (model,))[model]
    validated_threshold = validate_mos_threshold(model, threshold)
    if isinstance(validated_threshold, Mapping):
        return all(
            validated_score[dimension] >= validated_threshold[dimension]
            for dimension in validated_threshold
        )
    return validated_score >= validated_threshold


def get_model_target_sr(mos_type: str) -> int:
    """返回 MOS 模型声明的目标采样率。"""
    return get_mos_spec(mos_type).target_sr


def _mos_speed_per_second(mos_type: str) -> float:
    """排序辅助：直接 API 的未知模型置末，由调用方写 error 保持兼容。"""
    try:
        return get_mos_spec(mos_type).speed_per_second
    except ValueError:
        return 0.0


def resample_to_target_sr(audio: np.ndarray, sr: int, target_sr: int, method: str = "auto") -> Tuple[np.ndarray, int]:
    """Resample audio to target_sr. Returns (audio, target_sr). No-op if sr == target_sr."""
    if sr == target_sr:
        return audio, sr
    try:
        from ..utils.resample import resample_audio
        out = resample_audio(audio, sr, target_sr, method=method)
        return out, target_sr
    except Exception as e:
        import logging
        log = logging.getLogger(__name__)
        # 运行期重采样失败不应静默，warning 级别记录后走 scipy 兜底
        log.warning("resample_audio failed, using scipy fallback: %s", e)
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        out = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        return out, target_sr


_WORKER_DNSMOS = None
_WORKER_DNSMOS_ASSET_KEY = None
_WORKER_DNSMOS_ATEXIT_REGISTERED = False
_WORKER_DNSMOS_FINALIZER = None
_DNSMOS_MODULE_COUNTER = 0
_DNSMOS_MODULE_LOCK = threading.Lock()
_DNSMOS_PRIVATE_MODULES: Dict[str, types.ModuleType] = {}


def _remove_private_dnsmos_module(name, module) -> None:
    """只移除本 estimator 私有 package 中仍由本模块拥有的条目。"""
    package_name = name.rpartition(".")[0] if name else ""
    for owned_name, owned_module in tuple(_DNSMOS_PRIVATE_MODULES.items()):
        if owned_name == package_name or owned_name.startswith(f"{package_name}."):
            if sys.modules.get(owned_name) is owned_module:
                sys.modules.pop(owned_name, None)
            _DNSMOS_PRIVATE_MODULES.pop(owned_name, None)
    if name and sys.modules.get(name) is module:
        sys.modules.pop(name, None)


def _load_private_dnsmos_module(
    snapshot, descriptor_root: Optional[str] = None,
) -> tuple[str, types.ModuleType]:
    """从 owned snapshot descriptor 编译脚本到唯一私有 namespace。"""
    if descriptor_root is None:
        descriptor_root = _owned_model_tree_descriptor_path("DNSMOS", snapshot)
    else:
        descriptor_root = _owned_model_tree_root_descriptor_path(
            "DNSMOS", snapshot,
        )
    module_path = os.path.join(descriptor_root, "dnsmos_local.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"DNSMOS snapshot missing dnsmos_local.py: {module_path!r}"
        )
    with open(module_path, "rb") as source:
        code = compile(source.read(), module_path, "exec")
    # 防止脚本内容在 descriptor read 期间被篡改后进入 exec。
    _verify_owned_model_tree_snapshot("DNSMOS", snapshot)
    global _DNSMOS_MODULE_COUNTER
    with _DNSMOS_MODULE_LOCK:
        _DNSMOS_MODULE_COUNTER += 1
        serial = _DNSMOS_MODULE_COUNTER
    namespace_digest = hashlib.sha256(
        f"{os.getpid()}:{snapshot.sha256}:{descriptor_root}:{serial}".encode("utf-8")
    ).hexdigest()[:20]
    package_name = f"_audio_filter_dnsmos_{namespace_digest}"
    module_name = f"{package_name}.dnsmos_local"
    package = types.ModuleType(package_name)
    package.__file__ = None
    package.__package__ = package_name
    package.__path__ = [descriptor_root]
    package.__loader__ = None
    package.__spec__ = importlib.util.spec_from_loader(
        package_name, loader=None, is_package=True,
    )
    module = types.ModuleType(module_name)
    module.__file__ = module_path
    module.__package__ = package_name
    module.__loader__ = None
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
    before = set(sys.modules)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[package_name] = package
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__, module.__dict__)
        _verify_owned_model_tree_snapshot("DNSMOS", snapshot)
    except BaseException:
        for imported_name in tuple(sys.modules):
            if imported_name not in before and (
                imported_name == package_name
                or imported_name.startswith(f"{package_name}.")
            ):
                sys.modules.pop(imported_name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    for imported_name in tuple(sys.modules):
        if imported_name == package_name or imported_name.startswith(
            f"{package_name}."
        ):
            _DNSMOS_PRIVATE_MODULES[imported_name] = sys.modules[imported_name]
    return module_name, module


def _cleanup_dnsmos_worker_estimator() -> None:
    """子进程退出时清理独占 estimator，并重试此前失败的清理。"""
    global _WORKER_DNSMOS, _WORKER_DNSMOS_ASSET_KEY
    estimator = _WORKER_DNSMOS
    _WORKER_DNSMOS = None
    _WORKER_DNSMOS_ASSET_KEY = None
    with _PENDING_ESTIMATOR_CLEANUP_LOCK:
        pending = tuple(_PENDING_ESTIMATOR_CLEANUP.values())
    _close_estimators((*pending, estimator))


def _dnsmos_worker_compute(
    idx: int,
    item,
    model_dir=None,
    verified_model_tree_sha256=None,
    allow_fallback: bool = True,
):
    """在 spawn 子进程按 PID、origin、digest 与 fallback 隔离 estimator。"""
    global _WORKER_DNSMOS, _WORKER_DNSMOS_ASSET_KEY
    global _WORKER_DNSMOS_ATEXIT_REGISTERED, _WORKER_DNSMOS_FINALIZER
    canonical_dir = _canonical_model_dir(
        model_dir, os.path.join(THIRD_PARTY_DIR, "DNSMOS"),
    )
    digest = _verified_model_tree_digest(
        "DNSMOS", verified_model_tree_sha256,
    )
    asset_key = (os.getpid(), canonical_dir, digest, bool(allow_fallback))
    _retry_pending_estimator_cleanup()
    if _WORKER_DNSMOS is None or _WORKER_DNSMOS_ASSET_KEY != asset_key:
        replacement = DNSMOSEstimator(
            use_gpu=False,
            model_dir=canonical_dir,
            verified_model_tree_sha256=digest,
            allow_fallback=allow_fallback,
        )
        previous = _WORKER_DNSMOS
        _WORKER_DNSMOS = replacement
        _WORKER_DNSMOS_ASSET_KEY = asset_key
        if not _WORKER_DNSMOS_ATEXIT_REGISTERED:
            atexit.register(_cleanup_dnsmos_worker_estimator)
            # multiprocessing worker 不保证执行普通 atexit handler；Finalize
            # 覆盖其受控退出路径，普通进程则由幂等 close 再兜底。
            _WORKER_DNSMOS_FINALIZER = Finalize(
                None, _cleanup_dnsmos_worker_estimator, exitpriority=10,
            )
            _WORKER_DNSMOS_ATEXIT_REGISTERED = True
        if previous is not None:
            _close_estimators((previous,))
    est = _WORKER_DNSMOS
    if isinstance(item, str):
        return idx, est.compute(audio_path=item, audio_data=None)
    return idx, est.compute(audio_data=item)


# safe_total_seconds=64.0 实测依据：DNSMOS 为 SSL 模型（wav2vec 特征）显存峰值
# 最高，其 run_batch_audio 内部按 9.01s 窗/1s hop 滑窗切段、显存 ∝ segment 数；
# 显存紧张环境（多进程挤卡）实测 64s≈7 段才安全（1.7GB BFCArena 事故值）。
# 放宽前须实测显存峰值，勿凭显存余量目测放大
@register_mos("dnsmos", MOSSpec(
    name="dnsmos", target_sr=16000, score_kind="p835",
    primary_score_key="dnsmos_ovr_mos", safe_total_seconds=64.0,
    speed_per_second=105.0, selectable=True, languages=("multilingual",),
    validate_threshold=MOS_THRESHOLD_DIMENSIONS["dnsmos"],
    artifact_score_keys=(
        ("ovrl", "dnsmos_ovr_mos"), ("sig", "dnsmos_sig_mos"),
        ("bak", "dnsmos_bak_mos"), ("p808", "dnsmos_p808_mos"),
    ),
    version="DNSMOS P.835",
))
class DNSMOSEstimator:
    """DNSMOS P.835 scorer using local ONNX models.

    Supports both file-path and in-memory (audio_array, sr) input to avoid
    temporary file I/O when the caller already has data in memory.
    """
    name = "dnsmos"

    def __init__(
        self,
        model_path: str = None,
        use_gpu: bool = False,
        model_dir=None,
        verified_model_tree_sha256: str = None,
        allow_fallback: bool = True,
    ):
        self.use_gpu = use_gpu
        self.compute_score = None
        self._is_fallback = True
        self._allow_fallback = bool(allow_fallback)
        self._model_dir = _canonical_model_dir(
            model_dir, os.path.join(THIRD_PARTY_DIR, "DNSMOS"),
        )
        self._verified_model_tree_sha256 = _verified_model_tree_digest(
            "DNSMOS", verified_model_tree_sha256,
        )
        self._snapshot = None
        self._dnsmos_module = None
        self._dnsmos_module_name = None
        self._primary_model_path = None
        self._p808_model_path = None
        # CPU 批推理用的常驻 spawn 进程池：每个 child 自建并持有 owned snapshot。
        self._process_pool = None
        self._pool_lock = threading.Lock()
        self._closed = False
        self._load_model()

    def _new_compute_score(self, *, use_gpu: bool):
        if self._snapshot is None or self._dnsmos_module is None:
            raise RuntimeError("DNSMOS owned snapshot/module is unavailable")
        descriptor_root = _owned_model_tree_descriptor_path(
            "DNSMOS", self._snapshot,
        )
        primary_model_path = os.path.join(
            descriptor_root, self._primary_model_path,
        )
        p808_model_path = os.path.join(
            descriptor_root, self._p808_model_path,
        )
        replacement = self._dnsmos_module.ComputeScore(
            primary_model_path,
            p808_model_path,
            use_gpu=use_gpu,
        )
        _verify_owned_model_tree_snapshot("DNSMOS", self._snapshot)
        return replacement

    def _load_model(self):
        snapshot = module = None
        module_name = None
        try:
            snapshot = _open_verified_model_tree_snapshot(
                "DNSMOS", self._model_dir, self._verified_model_tree_sha256,
            )
            module_name, module = _load_private_dnsmos_module(snapshot)
            descriptor_root = _owned_model_tree_root_descriptor_path(
                "DNSMOS", snapshot,
            )
            # 固化显存修复（third_party 不随仓库提交，此处注入保证任何环境生效）。
            _apply_dnsmos_vram_fix(module)
            primary_model_path = p808_model_path = None
            for sub in ("", "DNSMOS"):
                root = (
                    os.path.join(descriptor_root, sub)
                    if sub else descriptor_root
                )
                primary = os.path.join(root, "sig_bak_ovr.onnx")
                p808 = os.path.join(root, "model_v8.onnx")
                if os.path.isfile(primary) and os.path.isfile(p808):
                    primary_model_path = os.path.relpath(primary, descriptor_root)
                    p808_model_path = os.path.relpath(p808, descriptor_root)
                    break
            if not primary_model_path or not p808_model_path:
                raise FileNotFoundError(
                    "DNSMOS model files not found in owned snapshot descriptor"
                )

            self._snapshot = snapshot
            self._dnsmos_module = module
            self._dnsmos_module_name = module_name
            self._primary_model_path = primary_model_path
            self._p808_model_path = p808_model_path
            replacement = self._new_compute_score(use_gpu=self.use_gpu)
        except BaseException as exc:
            try:
                if module is not None:
                    _remove_private_dnsmos_module(module_name, module)
                if snapshot is not None:
                    snapshot.close()
            except BaseException as cleanup:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"DNSMOS load rollback failed: {cleanup}")
            self._snapshot = None
            self._dnsmos_module = None
            self._dnsmos_module_name = None
            self._primary_model_path = None
            self._p808_model_path = None
            if (
                isinstance(exc, MosModelAssetBindingError)
                or not isinstance(exc, Exception)
                or not self._allow_fallback
            ):
                raise
            logger.warning("Failed to load DNSMOS model: %s, using fallback", exc)
            return

        self.compute_score = replacement
        self._is_fallback = False
        logger.debug(
            "DNSMOS: Loaded ONNX models (%s) from %s",
            "GPU" if self.use_gpu else "CPU",
            os.path.dirname(primary_model_path),
        )

    def _downgrade_to_cpu(self, reason: str = "") -> None:
        """从同一 owned snapshot 与私有 module 事务式重建 CPU session。"""
        if not cpu_downgrade_allowed():
            raise RuntimeError(_cpu_downgrade_disallowed_message(self.name, reason))
        if not self.use_gpu or self._is_fallback:
            return
        try:
            # _new_compute_score 会在返回前完成 replacement 构造与 snapshot 复核；
            # 两步全部成功后才原子发布，失败时保留原 GPU session 与状态。
            replacement = self._new_compute_score(use_gpu=False)
        except Exception as exc:
            raise RuntimeError(
                f"DNSMOS CPU downgrade failed; GPU session preserved: {exc}"
            ) from exc
        self.compute_score = replacement
        self.use_gpu = False
        logger.warning("DNSMOS 已降级 CPU 推理%s", f"（{reason}）" if reason else "")

    def runtime_attestation(self) -> Dict[str, Any]:
        """校验两个 ONNX session 的真实 provider 并返回既有 wire payload。"""
        if self._is_fallback or self.compute_score is None:
            raise ValueError("Step3 dnsmos estimator is a fallback")
        required = "CUDAExecutionProvider" if self.use_gpu else "CPUExecutionProvider"
        requested = "cuda" if self.use_gpu else "cpu"
        fallback_disabled = bool(getattr(
            self.compute_score, "_cpu_ep_fallback_disabled", False,
        ))
        providers = {}
        for name in ("onnx_sess", "p808_onnx_sess"):
            session = getattr(self.compute_score, name, None)
            actual = tuple(session.get_providers()) if session is not None else ()
            if self.use_gpu:
                cuda_only = actual == (required,)
                strict_cuda = actual[:1] == (required,) and fallback_disabled
                if not (cuda_only or strict_cuda):
                    raise ValueError(
                        f"Step3 dnsmos {name} provider is not {requested}"
                    )
            elif required not in actual:
                raise ValueError(
                    f"Step3 dnsmos {name} provider is not {requested}"
                )
            providers[name] = list(actual)
        if self.use_gpu:
            providers["cpu_ep_fallback_disabled"] = fallback_disabled
        return providers

    def compute(self, audio_path: str = None, *, audio_data: tuple = None) -> Dict:
        """Score audio quality.

        Args:
            audio_path: Path to WAV file.
            audio_data: ``(audio_array, sr)`` tuple — when provided, avoids
                reading from disk.  Multi-channel input uses channel 1 (Python index 0)
                and is converted to contiguous float32 mono before scoring.
        """
        if self.compute_score is not None:
            try:
                if audio_data is not None and audio_path is None:
                    audio, sr = audio_data
                    audio = _first_channel_audio(audio)
                    # sampling_rate 是模型内部工作采样率；run_batch_audio 会按每个
                    # item 自带的 sr 重采样，因此混合采样率输入无需在此分组
                    result = self.compute_score.run_batch_audio(
                        [(audio, sr)], sampling_rate=16000, is_personalized_MOS=False
                    )[0]
                else:
                    audio, sr = sf.read(audio_path, dtype="float32")
                    audio = _first_channel_audio(audio)
                    # 通过内存接口传入 mono，避免 provider 自行读取并混合原始多通道。
                    result = self.compute_score.run_batch_audio(
                        [(audio, sr)], sampling_rate=16000, is_personalized_MOS=False
                    )[0]
                return {
                    'dnsmos_p808_mos': float(result['P808_MOS']),
                    'dnsmos_sig_mos': float(result['SIG']),
                    'dnsmos_bak_mos': float(result['BAK']),
                    'dnsmos_ovr_mos': float(result['OVRL'])
                }
            except Exception as e:
                if self.use_gpu:
                    if not cpu_downgrade_allowed():
                        # fail-closed：禁止降级时直接抛明确错误，不再打印
                        # "降级 CPU 重试"误导日志（实际并未降级，只是尝试被拒绝）
                        raise RuntimeError(
                            _cpu_downgrade_disallowed_message(self.name, f"GPU inference error: {e}")
                        ) from e
                    # 允许降级：CPU replacement 构造与复核成功后才发布，并用
                    # 真实模型重试本条；降级失败会保留原 GPU session 并明确抛错，
                    # 不进入 fallback 或混合状态。
                    logger.warning("DNSMOS GPU inference failed (%s)，降级 CPU 重试", e)
                    self._downgrade_to_cpu(reason="GPU inference error")
                    return self.compute(audio_path=audio_path, audio_data=audio_data)
                if not self._allow_fallback:
                    raise RuntimeError(
                        f"DNSMOS inference failed and energy fallback is forbidden: {e}"
                    ) from e
                logger.debug("DNSMOS computation failed: %s, using fallback", e)

        if not self._allow_fallback:
            raise RuntimeError("DNSMOS model is unavailable and energy fallback is forbidden")
        if audio_data is not None:
            audio, sr = audio_data
        else:
            audio, sr = sf.read(audio_path)
        audio = _first_channel_audio(audio)
        energy = np.sqrt(np.mean(audio ** 2))
        mos = 3.0 + energy * 0.5
        return {
            'dnsmos_p808_mos': float(np.clip(mos, 1, 5)),
            'dnsmos_sig_mos': float(np.clip(mos + 0.2, 1, 5)),
            'dnsmos_bak_mos': float(np.clip(mos - 0.1, 1, 5)),
            'dnsmos_ovr_mos': float(np.clip(mos, 1, 5)),
            # 标记为能量伪分数，过滤投票与分数融合时不计入有效模型（P2 修复）
            'fallback': True,
        }

    def _items_to_audio_sr(self, items: list) -> list:
        """把 items（str 或 (audio, sr)）统一转成 (audio, sr) 列表；字符串会先读盘。"""
        out = []
        for it in items:
            if isinstance(it, str):
                audio, sr = sf.read(it, dtype="float32")
                out.append((_first_channel_audio(audio), sr))
            else:
                audio, sr = it
                out.append((_first_channel_audio(audio), sr))
        return out

    def compute_batch(self, items: list) -> list:
        """批打分：优先用 ComputeScore.run_batch_audio 做真正 ONNX batch。

        GPU 模式下直接在主进程做 batch，避免 ProcessPool 退化为 CPU；
        CPU 大容量时仍用 ProcessPoolExecutor 子进程推理绕过 GIL。
        """
        if not items:
            return []
        if self.use_gpu and self.compute_score is not None:
            try:
                audio_sr_list = self._items_to_audio_sr(items)
                # run_batch_audio 内部按每个 item 的 sr 重采样到 sampling_rate，
                # 混合采样率的 batch 在此已被正确处理
                batch_results = self.compute_score.run_batch_audio(
                    audio_sr_list, sampling_rate=16000, is_personalized_MOS=False
                )
                return [
                    {
                        'dnsmos_p808_mos': float(r['P808_MOS']),
                        'dnsmos_sig_mos': float(r['SIG']),
                        'dnsmos_bak_mos': float(r['BAK']),
                        'dnsmos_ovr_mos': float(r['OVRL']),
                    }
                    for r in batch_results
                ]
            except Exception as e:
                if not cpu_downgrade_allowed():
                    # fail-closed：禁止降级时直接抛明确错误，不再打印
                    # "降级 CPU 推理"误导日志（实际并未降级）
                    raise RuntimeError(
                        _cpu_downgrade_disallowed_message(self.name, f"GPU batch inference error: {e}")
                    ) from e
                # GPU OOM（SSL 模型显存峰值高）：重建 CPU session 降级，
                # 避免后续批次反复 OOM 或整体退化为能量伪分数
                logger.warning("DNSMOS GPU batch failed (%s)，降级 CPU 推理", e)
                self._downgrade_to_cpu(reason="GPU batch OOM")
        if len(items) <= 4:
            return [
                self.compute(audio_path=it, audio_data=None)
                if isinstance(it, str)
                else self.compute(audio_data=it)
                for it in items
            ]
        max_workers = min(8, len(items), multiprocessing.cpu_count() or 4)
        results = [None] * len(items)
        # 复用常驻进程池，整个 subset 的多个 chunk 共享同一批 worker
        ex = self._get_process_pool(max_workers)
        futs = [
            ex.submit(
                _dnsmos_worker_compute,
                i,
                it,
                self._model_dir,
                self._verified_model_tree_sha256,
                self._allow_fallback,
            )
            for i, it in enumerate(items)
        ]
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
        return results

    def _get_process_pool(self, max_workers: int) -> ProcessPoolExecutor:
        """惰性创建并复用 spawn ProcessPoolExecutor。"""
        with self._pool_lock:
            if self._process_pool is None:
                self._process_pool = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            return self._process_pool

    def close(self):
        """幂等清理进程池、私有 module 与 owned snapshot。"""
        with self._pool_lock:
            if self._closed:
                return
            process_pool = self._process_pool

        # shutdown 可能等待 worker 退出；生命周期门禁已阻止新 compute，不能再
        # 持有 pool 锁等待，避免 worker/回调需要该锁时形成死锁。
        first_error = None
        if process_pool is not None:
            try:
                process_pool.shutdown(wait=True)
            except BaseException as exc:
                first_error = exc
            else:
                with self._pool_lock:
                    if self._process_pool is process_pool:
                        self._process_pool = None

        module = self._dnsmos_module
        module_name = self._dnsmos_module_name
        if module is not None or module_name is not None:
            try:
                _remove_private_dnsmos_module(module_name, module)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                elif hasattr(first_error, "add_note"):
                    first_error.add_note(f"DNSMOS module cleanup failed: {exc}")
            else:
                self._dnsmos_module = None
                self._dnsmos_module_name = None

        # ComputeScore 持有 descriptor model paths；必须先撤销引用，再关闭 fd。
        self.compute_score = None
        snapshot = self._snapshot
        if snapshot is not None:
            try:
                snapshot.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                elif hasattr(first_error, "add_note"):
                    first_error.add_note(f"DNSMOS snapshot cleanup failed: {exc}")
            else:
                self._snapshot = None

        if first_error is not None:
            raise first_error
        self._primary_model_path = None
        self._p808_model_path = None
        self._closed = True

    def reset(self) -> None:
        """从同一 owned snapshot 重建 onnx session，清空 arena。"""
        if self.compute_score is not None and not self._is_fallback:
            try:
                replacement = self._new_compute_score(use_gpu=self.use_gpu)
                self.compute_score = replacement
                logger.info("DNSMOS session 已重置（清空 arena）")
            except Exception as exc:
                if not self._allow_fallback:
                    raise
                logger.warning("DNSMOS session 重置失败: %s", exc)

    def batch_weight(self, item, fallback_sec: float) -> float:
        """DNSMOS 批权重：run_batch_audio 的 segment 数（显存 ∝ segment）。"""
        d = _item_duration_sec(item)
        return _dnsmos_segment_estimate(d if d is not None else fallback_sec)

    def batch_budget(self, budget_sec: float, safe_total_sec: Dict[str, float]) -> float:
        """DNSMOS 批预算（权重单位=段）：显存安全秒数 / 9.01（≈7 段）。

        默认值取自 MOSSpec（避免与内置表双源漂移）。
        """
        default_cap = get_mos_spec(self.name).safe_total_seconds
        return max(1.0, min(budget_sec, safe_total_sec.get(self.name, default_cap) / 9.01))


@register_mos("nisqa", MOSSpec(
    name="nisqa", target_sr=48000, score_kind="mos",
    primary_score_key="nisqa_mos", safe_total_seconds=300.0,
    speed_per_second=25.5, selectable=True, languages=("multilingual",),
    version="NISQA",
))
class NISQAEstimator:
    """复用已加载模型与临时输出目录的 NISQA scorer。

    模型通常在 ``__init__`` 加载并跨调用复用；允许 GPU→CPU 降级时会基于同一
    显式 checkpoint 重新加载。实例的 atexit callback 只作进程退出兜底，正常
    生命周期由 ``close()`` 注销 callback 并收敛临时路径、dataset 与模型引用。

    进程级副作用（仅限 NISQA 官方 constructor 的作用域）：``__init__`` 会在
    ``_CWD_LOCK`` 保护下临时 ``os.chdir(self.nisqa_dir)`` 并把 ``torch.load``
    换成只接受同一已验证 snapshot descriptor 的 wrapper（NISQA 官方代码内部
    调用裸 ``torch.load``），退出时恢复；``get_mos_estimator`` 的缓存路径还
    额外持有 ``_MODEL_CACHE_LOCK``。两把锁只串行化 estimator 构造本身：构造
    期间同进程其它线程的相对路径 I/O 不受保护，其 ``torch.load`` 会被 wrapper
    fail-closed 拒绝。因此调用方必须保证 NISQA 构造与其它线程的相对路径 I/O
    串行化；需要并发构造时，应先把 NISQA 构造移入子进程，而不是放宽这里的
    全局替换。
    """
    name = "nisqa"

    def __init__(self, model_path: str = None, use_gpu: bool = False,
                 checkpoint_sha256: str = None,
                 allow_fallback: bool = True):
        self.use_gpu = use_gpu
        self._allow_fallback = bool(allow_fallback)
        self._expected_checkpoint_sha256 = checkpoint_sha256
        self._nisqa_model = None
        self._is_fallback = True
        self._target_sr = 48000
        self.nisqa_dir = os.path.abspath(os.path.join(THIRD_PARTY_DIR, 'NISQA'))
        self.model_path = None
        # 复用临时输出目录以降低频繁创建/销毁开销；优先用 /dev/shm（内存文件系统）
        self._tmpdir_base = "/dev/shm/nisqa_tmp" if os.path.exists("/dev/shm") else None
        if self._tmpdir_base:
            # mkdtemp/NamedTemporaryFile 的 dir 参数要求父目录已存在，惰性建一次
            try:
                os.makedirs(self._tmpdir_base, exist_ok=True)
            except OSError:
                self._tmpdir_base = None
        self._nisqa_tmpdir = None
        self._pending_tmp_paths: Dict[str, bool] = {}
        self._nisqa_cleanup_lock = threading.RLock()
        self._closed = False
        self._load_model(model_path)
        # atexit 兜底：进程退出时清理复用的临时输出目录（/dev/shm 下不会随进程自动回收）；
        # 保留 callback 引用，fork child 可注销继承 handler 而不触碰父进程对象。
        callback = self.close
        self._atexit_callback = callback
        atexit.register(callback)
        _NISQA_ATEXIT_CALLBACKS.append(callback)

    def _load_model(self, model_path: str = None):
        explicit_model_path = model_path is not None
        try:
            if self.nisqa_dir not in sys.path:
                sys.path.insert(0, self.nisqa_dir)

            if model_path is None:
                model_path = os.path.join(self.nisqa_dir, 'weights', 'nisqa.tar')
            model_path = os.path.realpath(os.path.abspath(model_path))

            if not os.path.exists(model_path):
                error = FileNotFoundError(f"NISQA model not found at {model_path}")
                if not self._allow_fallback:
                    raise error
                logger.warning("%s, using fallback", error)
                return

            self.model_path = model_path

            with _open_local_checkpoint_snapshot(model_path) as checkpoint_snapshot:
                self._checkpoint_sha256 = checkpoint_snapshot.sha256
                _verify_expected_checkpoint_digest(
                    "NISQA", self._checkpoint_sha256,
                    self._expected_checkpoint_sha256,
                )
                with tempfile.TemporaryDirectory(prefix="nisqa_init_") as tmpdir:
                    dummy_path = os.path.join(tmpdir, "dummy.wav")
                    sf.write(dummy_path, np.zeros(16000, dtype="float32"), 16000)

                    args = {
                        'mode': 'predict_file',
                        'pretrained_model': checkpoint_snapshot.descriptor_path,
                        'deg': dummy_path,
                        'output_dir': tmpdir,
                        'num_workers': 0,
                        'bs': 1,
                        'tr_bs_val': 1,
                        'tr_num_workers': 0,
                        'ms_channel': None,
                        'tr_device': 'cuda' if self.use_gpu else 'cpu',
                    }

                    with _CWD_LOCK:
                        # NISQA 官方 constructor 内部调用裸 torch.load。仅在全局锁保护的
                        # constructor 作用域内替换，且只允许同一 snapshot descriptor。
                        import torch
                        original_cwd = os.getcwd()
                        original_torch_load = torch.load

                        def safe_nisqa_load(requested_path, *load_args, **load_kwargs):
                            if load_args:
                                raise RuntimeError("NISQA torch.load positional options are not allowed")
                            requested = os.fspath(requested_path)
                            if requested != checkpoint_snapshot.descriptor_path:
                                raise RuntimeError(
                                    f"NISQA attempted to load an unverified checkpoint: {requested!r}"
                                )
                            allowed = {"map_location"}
                            unknown = set(load_kwargs) - allowed
                            if unknown:
                                raise RuntimeError(
                                    f"NISQA torch.load options are not allowed: {sorted(unknown)!r}"
                                )
                            # safe_torch_load 内部仍调用 torch.load；临时恢复真实入口，
                            # 避免 wrapper 递归拦截其 weights_only=True 安全参数。
                            torch.load = original_torch_load
                            try:
                                return safe_torch_load(
                                    checkpoint_snapshot.descriptor_path,
                                    map_location=load_kwargs.get("map_location"),
                                    purpose="load verified NISQA weights safely",
                                )
                            finally:
                                torch.load = safe_nisqa_load

                        os.chdir(self.nisqa_dir)
                        torch.load = safe_nisqa_load
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                _orig_stdout = sys.stdout
                                sys.stdout = io.StringIO()
                                try:
                                    from nisqa.NISQA_model import nisqaModel
                                    self._nisqa_model = nisqaModel(args)
                                finally:
                                    sys.stdout = _orig_stdout
                        finally:
                            torch.load = original_torch_load
                            os.chdir(original_cwd)
            if self._nisqa_model is not None:
                self._nisqa_model.args["tr_parallel"] = False
                model = getattr(self._nisqa_model, "model", None)
                move_model = getattr(model, "to", None)
                if not callable(move_model):
                    raise RuntimeError("NISQA model does not expose device migration")
                move_model(self._nisqa_model.dev)
                try:
                    if hasattr(self._nisqa_model.args, 'get'):
                        ms_sr = self._nisqa_model.args.get("ms_sr", 48000)
                    else:
                        ms_sr = getattr(self._nisqa_model.args, 'ms_sr', 48000)
                    self._target_sr = int(ms_sr) if ms_sr is not None else 48000
                except Exception:
                    self._target_sr = 48000
                logger.debug("NISQA: Model loaded and cached from %s", model_path)
                self._is_fallback = False

        except Exception as e:
            self._nisqa_model = None
            self._is_fallback = True
            if explicit_model_path or not self._allow_fallback:
                raise
            logger.warning("Failed to setup NISQA: %s, using fallback", e)

    def _nisqa_output_dir(self) -> str:
        """single-flight 分配本实例唯一持有的 NISQA 输出目录。"""
        state = _estimator_lifecycle_state(self)
        with state.condition:
            tmpdir = getattr(self, "_nisqa_tmpdir", None)
            if tmpdir is None:
                tmpdir = tempfile.mkdtemp(
                    prefix="nisqa_out_", dir=self._tmpdir_base,
                )
                self._nisqa_tmpdir = tmpdir
            return tmpdir

    def _run_nisqa(self, audio_path: str) -> Optional[Dict]:
        """以已加载模型评分单文件，并复用实例级临时输出目录。

        模型不可用、结果为空，或允许 fallback 时发生运行错误才返回 ``None``，
        由上层决定 CPU 重试或伪分降级；schema/value 契约错误始终抛出，禁止
        fallback 时运行错误也会抛出。CWD 切换仅覆盖官方 NISQA 预测调用。
        """
        if self._nisqa_model is None:
            return None
        try:
            output_dir = self._nisqa_output_dir()
            with _CWD_LOCK:
                # args 赋值必须放在锁内：args 是共享模型状态，锁外赋值会被并发线程
                # 互相覆盖，导致两个文件拿到同一份分数（P1 修复）
                self._nisqa_model.args["deg"] = audio_path
                self._nisqa_model.args["output_dir"] = output_dir
                original_cwd = os.getcwd()
                os.chdir(self.nisqa_dir)
                try:
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                self._nisqa_model._loadDatasetsFile()
                                result_df = self._nisqa_model.predict()
                finally:
                    os.chdir(original_cwd)

            if result_df is not None and len(result_df) > 0:
                row = result_df.iloc[0]
                try:
                    score = row['mos_pred']
                except (KeyError, TypeError, IndexError) as exc:
                    raise MosResultContractError(
                        "NISQA result must contain mos_pred"
                    ) from exc
                if (
                    isinstance(score, bool)
                    or not isinstance(score, Real)
                    or not math.isfinite(float(score))
                ):
                    raise MosResultContractError(
                        "NISQA mos_pred must be a finite non-bool real number"
                    )
                return {'nisqa_mos': float(score)}
        except MosResultContractError:
            raise
        except Exception as exc:
            if not self._allow_fallback:
                raise RuntimeError(
                    "NISQA inference failed and fallback is forbidden: "
                    f"{_bounded_error_detail(exc)}"
                ) from exc
            logger.debug("NISQA computation failed: %s, using fallback", exc)
        return None

    @staticmethod
    def _fallback(audio: np.ndarray, sr: int) -> Dict:
        audio = _first_channel_audio(audio)
        energy = np.sqrt(np.mean(audio ** 2))
        noise_floor = np.percentile(np.abs(audio), 5)
        snr = 20 * np.log10(energy / (noise_floor + 1e-10))
        mos = np.clip(3.0 + snr / 20, 1, 5)
        # 标记为 SNR 伪分数，过滤投票与分数融合时不计入有效模型。
        return {'nisqa_mos': float(mos), 'fallback': True}

    def _downgrade_to_cpu(self, reason: str = "") -> None:
        """GPU 推理失败（典型为显存 OOM）时尝试重建 CPU 模型。

        NISQA 模型 device 在初始化时由 tr_device 决定，无法原地迁移，因此以
        ``use_gpu=False`` 按同一显式 checkpoint 重载。重载失败后保留失败或
        fallback 状态，由上层依据 ``allow_fallback`` 决定返回伪分或抛错。
        """
        if not cpu_downgrade_allowed():
            raise RuntimeError(_cpu_downgrade_disallowed_message(self.name, reason))
        if not self.use_gpu or self._is_fallback:
            return
        if not getattr(self, 'model_path', None):
            logger.warning("NISQA 降级 CPU 失败：缺少 model_path")
            return
        try:
            self.use_gpu = False
            self._load_model(self.model_path)
            if self._nisqa_model is not None:
                logger.warning("NISQA 已降级 CPU 推理%s", f"（{reason}）" if reason else "")
            else:
                logger.warning("NISQA 降级 CPU 重载失败，继续使用伪分数")
        except Exception as e:
            logger.warning("NISQA 降级 CPU 失败: %s", e)

    def runtime_attestation(self) -> Dict[str, Any]:
        """校验 NISQA device 与 checkpoint，并返回既有 wire payload。"""
        if self._is_fallback or self._nisqa_model is None:
            raise ValueError("Step3 nisqa estimator is a fallback")
        requested = "cuda" if self.use_gpu else "cpu"
        device = str(getattr(self._nisqa_model, "dev", "")).lower()
        parameter_device = _model_parameters_device(
            getattr(self._nisqa_model, "model", None), "nisqa",
        )
        if requested not in device or requested not in parameter_device:
            raise ValueError(f"Step3 nisqa device is not {requested}")
        return {
            "device": device,
            "parameters": parameter_device,
            "checkpoint_sha256": _attested_checkpoint_sha256(self, "nisqa"),
        }

    def _ensure_cleanup_state(self) -> None:
        """兼容旧测试替身，并为失败清理持有强引用路径。"""
        if not hasattr(self, "_nisqa_cleanup_lock"):
            self._nisqa_cleanup_lock = threading.RLock()
        if not hasattr(self, "_pending_tmp_paths"):
            self._pending_tmp_paths = {}
        if not hasattr(self, "_closed"):
            self._closed = False

    def _cleanup_tmp_path(self, path: str, *, is_dir: bool) -> None:
        """转移临时路径所有权后清理；失败时保留到后续重试。"""
        self._ensure_cleanup_state()
        with self._nisqa_cleanup_lock:
            self._pending_tmp_paths[path] = is_dir
        try:
            if is_dir:
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except FileNotFoundError:
            pass
        except BaseException:
            raise
        with self._nisqa_cleanup_lock:
            self._pending_tmp_paths.pop(path, None)

    def _retry_pending_tmp_cleanup(self) -> None:
        """重试此前失败的输入清理，全部成功前不丢失任何路径。"""
        self._ensure_cleanup_state()
        with self._nisqa_cleanup_lock:
            pending = tuple(self._pending_tmp_paths.items())
        first_error = None
        for path, is_dir in pending:
            try:
                self._cleanup_tmp_path(path, is_dir=is_dir)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    _add_exception_note(
                        first_error, f"additional NISQA temporary cleanup failed: {exc}",
                    )
        if first_error is not None:
            raise first_error

    def compute(self, audio_path: str = None, *, audio_data: tuple = None) -> Dict:
        """Score audio.  Accepts file path or ``(array, sr)`` tuple."""
        self._retry_pending_tmp_cleanup()
        tmp_path = None
        try:
            if audio_data is None and audio_path is not None:
                audio, sr = sf.read(audio_path)
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="nisqa_in_", dir=self._tmpdir_base,
                )
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, _first_channel_audio(audio), sr)
                audio_path = tmp_path
            if audio_data is not None and audio_path is None:
                audio, sr = audio_data
                audio = _first_channel_audio(audio)
                # 与 batch 路径一致，输入临时 wav 也写到 _tmpdir_base（优先 /dev/shm），
                # 避免默认 /tmp 与输出目录分散两处
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="nisqa_in_", dir=self._tmpdir_base,
                )
                tmp_path = tmp.name
                tmp.close()
                sf.write(tmp_path, audio, sr)
                audio_path = tmp_path

            result = self._run_nisqa(audio_path)
            if result is not None:
                return result

            # GPU 推理失败（含显存 OOM）：降级 CPU 后重试一次，用真实模型重算
            if self.use_gpu and not self._is_fallback:
                if not cpu_downgrade_allowed():
                    # fail-closed：禁止降级时直接抛明确错误，不再打印
                    # "降级 CPU 重试"误导日志
                    raise RuntimeError(
                        _cpu_downgrade_disallowed_message(self.name, "GPU inference error")
                    )
                logger.warning("NISQA GPU inference failed，降级 CPU 重试")
                self._downgrade_to_cpu(reason="GPU inference error")
                result = self._run_nisqa(audio_path)
                if result is not None:
                    return result

            if not self._allow_fallback:
                raise RuntimeError(
                    "NISQA inference produced no result and fallback is forbidden"
                )
            if audio_data is not None:
                audio, sr = audio_data
            else:
                audio, sr = sf.read(audio_path)
            return self._fallback(_first_channel_audio(audio), sr)
        finally:
            if tmp_path:
                primary_error = sys.exc_info()[1]
                try:
                    self._cleanup_tmp_path(tmp_path, is_dir=False)
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    _add_exception_note(
                        primary_error,
                        f"NISQA temporary WAV cleanup failed: {cleanup_error}",
                    )

    def _run_nisqa_batch(self, paths: List[str]) -> Optional[Dict]:
        """对同一目录下的多个 wav 路径一次性跑 NISQA batch 预测。

        直接构造 DataFrame 与 SpeechQualityDataset，复用已加载模型，并按 batch
        大小调整 ``tr_bs_val``。模型不可用或输入为空时返回 ``None``；运行错误
        仅在允许 fallback 时返回 ``None``，禁止 fallback 时抛出。返回结果的
        schema/value 契约由 ``compute_batch`` 严格解析，错误不做降级。
        """
        if self._nisqa_model is None or not paths:
            return None
        try:
            import pandas as pd
            from nisqa import NISQA_lib as NL

            data_dir = os.path.dirname(paths[0])
            filenames = [os.path.basename(p) for p in paths]
            df = pd.DataFrame({'deg': filenames})
            output_dir = self._nisqa_output_dir()

            with _CWD_LOCK:
                # args/ds_val 赋值与 predict 必须整体放在锁内：这些都是共享模型
                # 状态，锁外赋值会被并发线程互相覆盖（与 _run_nisqa 相同的 P1 竞态）
                self._nisqa_model.args["deg"] = paths[0]
                self._nisqa_model.args["output_dir"] = output_dir

                # 动态调整 NISQA 的 validation batch size；原初始化时为 1（单文件模式）
                old_tr_bs_val = self._nisqa_model.args.get('tr_bs_val', 1)
                # CPU 下单进程 DataLoader 过大 batch 反而因内存/同步开销变慢；GPU 下可放大
                max_bs = 16 if self.use_gpu else 4
                new_tr_bs_val = max(1, min(len(paths), max_bs))

                original_cwd = os.getcwd()
                try:
                    self._nisqa_model.args['tr_bs_val'] = new_tr_bs_val

                    self._nisqa_model.ds_val = NL.SpeechQualityDataset(
                        df,
                        df_con=None,
                        data_dir=data_dir,
                        filename_column='deg',
                        mos_column='predict_only',
                        seg_length=self._nisqa_model.args['ms_seg_length'],
                        max_length=self._nisqa_model.args['ms_max_segments'],
                        to_memory=None,
                        to_memory_workers=None,
                        seg_hop_length=self._nisqa_model.args['ms_seg_hop_length'],
                        transform=None,
                        ms_n_fft=self._nisqa_model.args['ms_n_fft'],
                        ms_hop_length=self._nisqa_model.args['ms_hop_length'],
                        ms_win_length=self._nisqa_model.args['ms_win_length'],
                        ms_n_mels=self._nisqa_model.args['ms_n_mels'],
                        ms_sr=self._nisqa_model.args['ms_sr'],
                        ms_fmax=self._nisqa_model.args['ms_fmax'],
                        ms_channel=self._nisqa_model.args['ms_channel'],
                        double_ended=self._nisqa_model.args['double_ended'],
                        dim=self._nisqa_model.args['dim'],
                        filename_column_ref=None,
                    )

                    os.chdir(self.nisqa_dir)
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                result_df = self._nisqa_model.predict()
                finally:
                    # 无论成败都恢复 cwd 与共享模型状态：dataset 构造抛异常时
                    # 原先 tr_bs_val 会残留脏值污染后续推理；ds_val 若残留坏
                    # dataset，会被下次单条 _loadDatasetsFile()/新 batch 重建覆盖
                    os.chdir(original_cwd)
                    self._nisqa_model.args['tr_bs_val'] = old_tr_bs_val
            return result_df
        except Exception as exc:
            if not self._allow_fallback:
                raise RuntimeError(
                    "NISQA batch inference failed and fallback is forbidden: "
                    f"{_bounded_error_detail(exc)}"
                ) from exc
            logger.debug("NISQA batch computation failed: %s, using fallback", exc)
            return None

    def compute_batch(self, items: list) -> list:
        """批量评分；``items`` 为路径或 ``(array, sr)`` 元组列表。

        batch 运行失败时，仅在允许 fallback 的 direct/兼容调用中逐条调用
        ``compute``；禁止 fallback 时抛出。返回 DataFrame 的 schema/value 契约
        错误始终直接抛出，不做逐条降级。
        """
        self._retry_pending_tmp_cleanup()
        if not items:
            return []
        if self._nisqa_model is None:
            return [
                self.compute(audio_path=it, audio_data=None)
                if isinstance(it, str)
                else self.compute(audio_data=it)
                for it in items
            ]

        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="nisqa_batch_in_", dir=self._tmpdir_base)
            paths = []
            for i, it in enumerate(items):
                if isinstance(it, str):
                    src = it
                    ext = os.path.splitext(src)[1] or '.wav'
                    dst = os.path.join(tmp_dir, f"batch_{i:04d}{ext}")
                    audio, sr = sf.read(src)
                    sf.write(dst, _first_channel_audio(audio), sr)
                    paths.append(dst)
                else:
                    audio, sr = it
                    dst = os.path.join(tmp_dir, f"batch_{i:04d}.wav")
                    sf.write(dst, _first_channel_audio(audio), sr)
                    paths.append(dst)

            result_df = self._run_nisqa_batch(paths)
            if result_df is None or len(result_df) != len(items):
                raise RuntimeError("NISQA batch result mismatch")

            out = []
            for _, row in result_df.iterrows():
                try:
                    score = row['mos_pred']
                except (KeyError, TypeError, IndexError) as exc:
                    raise MosResultContractError(
                        "NISQA batch result must contain mos_pred"
                    ) from exc
                if (
                    isinstance(score, bool)
                    or not isinstance(score, Real)
                    or not math.isfinite(float(score))
                ):
                    raise MosResultContractError(
                        "NISQA batch mos_pred must be a finite non-bool real number"
                    )
                out.append({'nisqa_mos': float(score)})
            return out
        except MosResultContractError:
            raise
        except Exception as exc:
            if not self._allow_fallback:
                raise RuntimeError(
                    "NISQA batch inference failed and fallback is forbidden: "
                    f"{_bounded_error_detail(exc)}"
                ) from exc
            logger.debug("NISQA batch failed (%s), falling back to per-item", exc)
            return [
                self.compute(audio_path=it, audio_data=None)
                if isinstance(it, str)
                else self.compute(audio_data=it)
                for it in items
            ]
        finally:
            if tmp_dir:
                primary_error = sys.exc_info()[1]
                try:
                    self._cleanup_tmp_path(tmp_dir, is_dir=True)
                except BaseException as cleanup_error:
                    if primary_error is None:
                        raise
                    _add_exception_note(
                        primary_error,
                        "NISQA batch temporary directory cleanup failed: "
                        f"{cleanup_error}",
                    )

    def close(self):
        """收敛全部临时路径、输出目录和模型资源后提交关闭状态。"""
        self._ensure_cleanup_state()
        if self._closed:
            return

        self._retry_pending_tmp_cleanup()
        tmpdir = getattr(self, '_nisqa_tmpdir', None)
        if tmpdir:
            try:
                shutil.rmtree(tmpdir)
            except FileNotFoundError:
                pass
            except BaseException:
                raise
            self._nisqa_tmpdir = None

        nisqa_model = getattr(self, "_nisqa_model", None)
        if nisqa_model is not None:
            if hasattr(nisqa_model, "ds_val"):
                nisqa_model.ds_val = None
            if hasattr(nisqa_model, "model"):
                nisqa_model.model = None
            self._nisqa_model = None

        callback = getattr(self, "_atexit_callback", None)
        if callback is not None:
            atexit.unregister(callback)
            try:
                _NISQA_ATEXIT_CALLBACKS.remove(callback)
            except ValueError:
                pass
            self._atexit_callback = None
        self._closed = True


@register_mos("utmos", MOSSpec(
    name="utmos", target_sr=16000, score_kind="mos",
    primary_score_key="utmos_score", safe_total_seconds=300.0,
    speed_per_second=133.0, selectable=True, languages=("multilingual",),
    version="UTMOS22",
))
class UTMOSEstimator:
    """UTMOS22 scorer with GPU batch support.

    注意（语言适配性）：UTMOS22 在 VoiceMOS Challenge 2022 的语料上训练/评估
    （Blizzard / VCC 等，以英文为主），官方说明仅在英文数据集上进行训练和评估，
    对中文等非英文语料不友好——中文音频的 UTMOS 分数可能系统性偏低，筛选中文
    数据集时请结合该偏差评估阈值。
    """
    name = "utmos"

    def __init__(self, model_path: str = None, use_gpu: bool = False,
                 checkpoint_sha256: str = None,
                 allow_fallback: bool = True):
        self.use_gpu = use_gpu
        self._allow_fallback = bool(allow_fallback)
        self._expected_checkpoint_sha256 = checkpoint_sha256
        self.model = None
        self._device = None
        self._is_fallback = True
        self.utmos_dir = os.path.join(THIRD_PARTY_DIR, 'UTMOS', 'SpeechMOS')
        self._load_model(model_path)

    def _load_model(self, model_path: str = None):
        try:
            if self.utmos_dir not in sys.path:
                sys.path.insert(0, self.utmos_dir)

            import torch
            from speechmos.utmos22.strong.model import UTMOS22Strong
        except Exception as exc:
            self.model = None
            self._is_fallback = True
            if not self._allow_fallback:
                raise RuntimeError(
                    "UTMOS model import failed and fallback is forbidden: "
                    f"{_bounded_error_detail(exc)}"
                ) from exc
            # UTMOS/SpeechMOS 是可选依赖；允许 fallback 时不影响管线导入。
            logger.warning("Failed to import UTMOS model: %s, using fallback", exc)
            return

        try:
            if model_path is None:
                potential_paths = [
                    os.path.join(os.path.dirname(self.utmos_dir), 'utmos22_strong_step7459_v1.pt'),
                    os.path.join(self.utmos_dir, 'utmos22_strong_step7459_v1.pt'),
                ]
                for p in potential_paths:
                    if os.path.exists(p):
                        model_path = p
                        break

            self.model = UTMOS22Strong()

            if model_path and os.path.exists(model_path):
                model_path = os.path.realpath(os.path.abspath(model_path))
                with _open_local_checkpoint_snapshot(model_path) as checkpoint_snapshot:
                    self._checkpoint_sha256 = checkpoint_snapshot.sha256
                    _verify_expected_checkpoint_digest(
                        "UTMOS", self._checkpoint_sha256,
                        self._expected_checkpoint_sha256,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        state_dict = safe_torch_load(
                            checkpoint_snapshot.descriptor_path,
                            map_location="cpu",
                            purpose="load bundled UTMOS weights safely",
                        )
                    if not isinstance(state_dict, Mapping):
                        raise ValueError("UTMOS checkpoint 必须是 state_dict mapping")
                    expected = self.model.state_dict()
                    missing = sorted(set(expected) - set(state_dict))
                    unexpected = sorted(set(state_dict) - set(expected))
                    shape_mismatches = []
                    for key in set(expected) & set(state_dict):
                        value = state_dict[key]
                        if not isinstance(value, torch.Tensor):
                            shape_mismatches.append((key, type(value).__name__, tuple(expected[key].shape)))
                        elif value.shape != expected[key].shape:
                            shape_mismatches.append((key, tuple(value.shape), tuple(expected[key].shape)))
                    if missing or unexpected or shape_mismatches:
                        raise RuntimeError(
                            "UTMOS state_dict 不匹配: "
                            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                            f"shape_mismatches={shape_mismatches[:5]}"
                        )
                    self.model.load_state_dict(state_dict, strict=True)
                logger.debug("UTMOS: Loaded model from %s", model_path)
            else:
                # fail-closed：不在运行时联网下载模型（生产环境外网依赖，
                # 与 DNSMOS/NISQA 的本地加载约定对齐）。预下载：
                #   wget https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt
                # 放到 third_party_scripts/UTMOS/ 下（见 tools/audio_filter_pipeline/README.md）
                raise FileNotFoundError(
                    f"UTMOS 模型未找到: {model_path}；已禁用运行时联网下载，"
                    f"请按注释预下载 utmos22_strong_step7459_v1.pt"
                )

            self.model.eval()
            if self.use_gpu and torch.cuda.is_available():
                self._device = torch.device("cuda")
                self.model = self.model.to(self._device)
            else:
                self._device = torch.device("cpu")
            self._is_fallback = False

        except Exception as exc:
            self.model = None
            self._is_fallback = True
            if not self._allow_fallback or (model_path and os.path.exists(model_path)):
                raise
            logger.warning("Failed to load UTMOS model: %s, using fallback", exc)

    def _load_waveform(self, audio_path: str = None, audio_data: tuple = None):
        """Load and preprocess first-channel waveform from file or memory.

        文件路径与内存输入（audio_data）统一选择 channel 1（Python 索引 0），再重采样到模型目标采样率（model
        target sr 为 16k）：输入 sr != 16k 时两种来源都会重采样，不再沿用
        输入采样率（不信任调用方采样率）。
        """
        import torch
        import torchaudio

        # 统一行为：文件路径与内存输入都重采样到模型目标采样率（utmos=16k，
        # 与 UTMOS22Strong.forward 内部工作采样率一致）；内存输入若 sr 非 16k
        # 也重采样（不信任调用方采样率），两种来源在 model_specific_sr=True 时
        # 收敛到同一采样率（16k）。
        if audio_data is not None:
            audio, sr = audio_data
            waveform = torch.from_numpy(_first_channel_audio(audio)).float().unsqueeze(0)
            inference_sr = get_model_target_sr(self.name)
        else:
            # 用 soundfile 读文件，避免 torchaudio 新版对 torchcodec 的运行时依赖
            audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(np.ascontiguousarray(audio_np.T))[:1]
            inference_sr = get_model_target_sr(self.name)

        if sr != inference_sr:
            waveform = torchaudio.functional.resample(waveform, sr, inference_sr)
        if waveform.shape[1] < inference_sr:
            waveform = torch.nn.functional.pad(waveform, (0, inference_sr - waveform.shape[1]))
        return waveform, inference_sr

    def _downgrade_to_cpu(self, reason: str = "") -> None:
        """GPU 推理失败（典型为显存 OOM）时把模型迁移到 CPU 降级。

        与 DNSMOS/NISQA 同一约定：降级后本实例后续一律 CPU 推理真实模型，
        避免反复 OOM 或退化为能量伪分数。模型权重已在内存，to('cpu') 开销小。
        """
        import torch
        if not cpu_downgrade_allowed():
            raise RuntimeError(_cpu_downgrade_disallowed_message(self.name, reason))
        if not self.use_gpu or self.model is None:
            return
        try:
            self.model = self.model.to(torch.device('cpu'))
            self._device = torch.device('cpu')
            logger.warning("UTMOS 已降级 CPU 推理%s", f"（{reason}）" if reason else "")
        except Exception as e:
            logger.warning("UTMOS 降级 CPU 失败: %s", e)
        finally:
            # fail-safe（P1-1）：无论迁移成败都终止 GPU 重试——
            # 否则 compute 的降级分支会无限递归（降级失败 → use_gpu 仍 True → 递归）。
            # device 一并统一为 cpu（P3-1）：即便 to 失败，递归第二次行为确定
            # （waveform→cpu + 模型在 GPU → 设备不匹配 → 落入 fallback，而非再试 GPU）
            self.use_gpu = False
            self._device = torch.device('cpu')

    def runtime_attestation(self) -> Dict[str, Any]:
        """校验 UTMOS device 与 checkpoint，并返回既有 wire payload。"""
        if self._is_fallback or self.model is None:
            raise ValueError("Step3 utmos estimator is a fallback")
        requested = "cuda" if self.use_gpu else "cpu"
        device = str(self._device).lower()
        parameter_device = _model_parameters_device(self.model, "utmos")
        if requested not in device or requested not in parameter_device:
            raise ValueError(f"Step3 utmos device is not {requested}")
        return {
            "device": device,
            "parameters": parameter_device,
            "checkpoint_sha256": _attested_checkpoint_sha256(self, "utmos"),
        }

    def compute(self, audio_path: str = None, *, audio_data: tuple = None) -> Dict:
        """Score audio.  Accepts file path or ``(array, sr)`` tuple."""
        if self.model is not None:
            try:
                import torch
                waveform, inference_sr = self._load_waveform(audio_path, audio_data)

                with torch.inference_mode():
                    waveform = waveform.to(self._device)
                    score = self.model(waveform, inference_sr).mean().item()

                return {'utmos_score': float(score)}

            except Exception as exc:
                if self.use_gpu and self.model is not None:
                    if not cpu_downgrade_allowed():
                        # fail-closed：禁止降级时直接抛明确错误，不再打印
                        # "降级 CPU 重试"误导日志
                        raise RuntimeError(
                            _cpu_downgrade_disallowed_message(self.name, f"GPU inference error: {exc}")
                        ) from exc
                    # GPU 推理失败（含显存 OOM）：降级 CPU 后递归重试一次，
                    # 用真实模型重算而非能量伪分数
                    logger.warning("UTMOS GPU inference failed (%s)，降级 CPU 重试", exc)
                    self._downgrade_to_cpu(reason="GPU inference error")
                    return self.compute(audio_path=audio_path, audio_data=audio_data)
                if not self._allow_fallback:
                    raise RuntimeError(
                        "UTMOS inference failed and fallback is forbidden: "
                        f"{_bounded_error_detail(exc)}"
                    ) from exc
                logger.debug("UTMOS computation failed: %s, using fallback", exc)

        if not self._allow_fallback:
            raise RuntimeError("UTMOS model unavailable and fallback is forbidden")
        if audio_data is not None:
            audio, sr = audio_data
        else:
            audio, sr = sf.read(audio_path)
        audio = _first_channel_audio(audio)
        energy = np.sqrt(np.mean(audio ** 2))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        quality = 3.0 + energy * 0.5 - zero_crossings * 0.1
        # 标记为能量伪分数，过滤投票与分数融合时不计入有效模型（P2 修复）
        return {'utmos_score': float(np.clip(quality, 1, 5)), 'fallback': True}

    def compute_batch(self, items: list) -> List[Dict]:
        """Batch GPU inference.

        *items* is a list where each element is either an ``audio_path`` string
        or an ``(audio_array, sr)`` tuple for in-memory data.
        """
        if self.model is None:
            return [
                self.compute(audio_path=it) if isinstance(it, str)
                else self.compute(audio_data=it)
                for it in items
            ]

        import torch

        # 按有效采样率分组：mos_use_model_specific_sr=False 时同一 batch 可能混入
        # 不同 sr 的音频，整批共用第一个 item 的 sr 会让其余 item 以错误 sr 评分（P2 修复）
        sr_groups: Dict[int, List] = {}
        fallback_results = {}

        for i, item in enumerate(items):
            try:
                waveform, inference_sr = self._load_waveform(
                    audio_path=item if isinstance(item, str) else None,
                    audio_data=item if not isinstance(item, str) else None,
                )
                sr_groups.setdefault(inference_sr, []).append((i, waveform.squeeze(0)))
            except Exception as e:
                logger.debug("UTMOS batch load failed for item %d: %s", i, e)
                if isinstance(item, str):
                    fallback_results[i] = self.compute(audio_path=item)
                else:
                    fallback_results[i] = self.compute(audio_data=item)

        results = [None] * len(items)
        for i, res in fallback_results.items():
            results[i] = res

        for group_sr, group in sr_groups.items():
            # UTMOS 的第三方 strong model 不接收 length mask，且使用双向 LSTM 后
            # 对全帧 mean；因此必须按预处理后的确切长度分桶，不能让异长样本共享
            # padding。每个 bucket 仍受条数及真实 B×T 分配上限约束。
            length_buckets: Dict[int, List] = {}
            for entry in group:
                length_buckets.setdefault(entry[1].shape[0], []).append(entry)
            bucket_size = 16 if self.use_gpu else 4
            safe_samples = get_mos_spec(self.name).safe_total_seconds * group_sr
            for exact_len, length_bucket in length_buckets.items():
                safe_count = max(1, int(safe_samples // max(exact_len, 1)))
                max_items = min(bucket_size, safe_count)
                for start in range(0, len(length_bucket), max_items):
                    bucket = length_bucket[start:start + max_items]
                    padded = torch.stack([waveform for _, waveform in bucket])
                    with torch.inference_mode():
                        scores = self.model(padded.to(self._device), group_sr)
                    for batch_idx, (orig_idx, _) in enumerate(bucket):
                        results[orig_idx] = {
                            'utmos_score': float(scores[batch_idx].item()),
                        }

        return results


_SCOREQ_VRAM_LIMIT_BYTES = 16 * 1024 * 1024 * 1024


def _harden_scoreq_gpu_session(scorer, model_dir: str) -> None:
    """兼容旧 scorer；真实 core scorer 已在一次构造中创建严格 CUDA-primary session。"""
    if callable(getattr(scorer, "bind_session_path_lease", None)):
        return
    try:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.add_session_config_entry(
            "gpu.mem_limit", str(_SCOREQ_VRAM_LIMIT_BYTES)
        )
        options.add_session_config_entry(
            "gpu.arena_extend_strategy", "kSameAsRequested"
        )
        scorer._sess = ort.InferenceSession(
            os.path.join(model_dir, "adapt_nr_telephone.onnx"),
            options,
            providers=[
                ("CUDAExecutionProvider", {
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": _SCOREQ_VRAM_LIMIT_BYTES,
                }),
                "CPUExecutionProvider",
            ],
        )
        providers = tuple(scorer._sess.get_providers())
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError("SCOREQ CUDA-primary provider registration failed")
        scorer._device_str = "GPU"
    except Exception as exc:
        raise RuntimeError(
            "SCOREQ CUDA-primary hybrid session initialization failed"
        ) from exc


@register_mos("scoreq", MOSSpec(
    name="scoreq", target_sr=16000, score_kind="mos",
    primary_score_key="scoreq_mos", safe_total_seconds=300.0,
    speed_per_second=235.0, selectable=True, languages=("multilingual",),
    version="SCOREQ",
))
class SCOREQEstimator:
    """SCOREQ ONNX 评分器，direct/兼容调用可显式允许能量 fallback。

    只有 ``allow_fallback=True`` 时，加载或推理失败才返回能量伪分；canonical
    Step3 runtime/preflight 始终传入 ``allow_fallback=False``，因此生产路径
    fail-closed。
    """
    name = "scoreq"

    def __init__(
        self,
        model_path: str = None,
        use_gpu: bool = False,
        model_dir=None,
        verified_model_tree_sha256: str = None,
        allow_fallback: bool = True,
    ):
        self.use_gpu = use_gpu
        self._scoreq = None
        self._is_fallback = True
        self._allow_fallback = bool(allow_fallback)
        self._scoreq_dir = _canonical_model_dir(
            model_dir, os.path.join(THIRD_PARTY_DIR, "SCOREQ"),
        )
        self._verified_model_tree_sha256 = _verified_model_tree_digest(
            "SCOREQ", verified_model_tree_sha256,
        )
        self._snapshot = None
        self._closed = False
        if _USE_TRAIN_ROOT not in sys.path:
            sys.path.insert(0, _USE_TRAIN_ROOT)
        snapshot = None
        try:
            snapshot = _open_verified_model_tree_snapshot(
                "SCOREQ", self._scoreq_dir,
                self._verified_model_tree_sha256,
            )
            self._snapshot = snapshot
            replacement = self._new_scoreq(full_verify=True)
        except BaseException as exc:
            try:
                if snapshot is not None:
                    snapshot.close()
            except BaseException as cleanup:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"SCOREQ load rollback failed: {cleanup}")
            self._snapshot = None
            if (
                isinstance(exc, MosModelAssetBindingError)
                or not isinstance(exc, Exception)
                or not self._allow_fallback
            ):
                raise
            logger.warning(
                "SCOREQ ONNX not loaded (%s), using energy fallback. "
                "Place ONNX in third_party_scripts/SCOREQ/ or download from https://zenodo.org/records/15739280",
                exc,
            )
            return
        self._scoreq = replacement
        self._is_fallback = False
        logger.debug("SCOREQ: Loaded ONNX from owned snapshot descriptor")

    def _new_scoreq(self, *, full_verify: bool = False):
        """在同一 loader lease 内用验证夹住 SCOREQ scorer 构造。"""
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("SCOREQ owned snapshot is unavailable")
        from core.metrics.mos_models import load_scoreq

        with snapshot.loader_path_lease() as descriptor_root:
            if full_verify:
                _verify_owned_model_tree_snapshot("SCOREQ", snapshot)
            else:
                _verify_owned_model_tree_manifest("SCOREQ", snapshot)
            replacement = load_scoreq(
                model_dir=descriptor_root, use_gpu=self.use_gpu,
            )
            if self.use_gpu:
                bind_lease = getattr(replacement, "bind_session_path_lease", None)
                if callable(bind_lease):
                    bind_lease(snapshot.loader_path_lease)
                _harden_scoreq_gpu_session(replacement, descriptor_root)
            _verify_owned_model_tree_manifest("SCOREQ", snapshot)
        return replacement

    def reset(self) -> None:
        """从同一 owned snapshot 重建 SCOREQ onnx session。"""
        if self._scoreq is None:
            return
        try:
            replacement = self._new_scoreq()
            self._scoreq = replacement
            logger.info("SCOREQ session 已重置（清空 arena）")
        except Exception as exc:
            self._scoreq = None
            state = _estimator_lifecycle_state(self)
            with state.condition:
                state.phase = "broken"
                state.condition.notify_all()
            raise RuntimeError(
                "SCOREQ session reset failed; scorer is broken and fallback is forbidden"
            ) from exc

    def close(self) -> None:
        """幂等释放 scorer；snapshot 清理成功后才提交关闭状态。"""
        if self._closed:
            return
        # scorer 持有 descriptor model path；必须先撤销引用，再关闭 fd。
        self._scoreq = None
        snapshot = self._snapshot
        if snapshot is not None:
            snapshot.close()
        self._snapshot = None
        self._closed = True

    def runtime_attestation(self) -> Dict[str, Any]:
        """校验 SCOREQ provider/device 并返回既有 wire payload。"""
        if self._is_fallback or self._scoreq is None:
            raise ValueError("Step3 scoreq estimator is a fallback")
        session = getattr(self._scoreq, "_sess", None)
        providers = tuple(session.get_providers()) if session is not None else ()
        device = str(getattr(self._scoreq, "_device_str", "")).lower()
        if self.use_gpu:
            if not providers or providers[0] != "CUDAExecutionProvider":
                raise ValueError("Step3 scoreq provider is not cuda-primary")
            if "gpu" not in device and "cuda" not in device:
                raise ValueError("Step3 scoreq device is not cuda")
            execution_mode = (
                "cuda_primary_hybrid_ep" if len(providers) > 1
                else "cuda_only_ep"
            )
        else:
            if "CPUExecutionProvider" not in providers:
                raise ValueError("Step3 scoreq provider is not cpu")
            execution_mode = "cpu_ep"
        return {
            "execution_mode": execution_mode,
            "providers": list(providers),
            "device": device,
        }

    def _compute_fallback(self, audio: np.ndarray, sr: int) -> Dict:
        """Energy-based fallback when ONNX is not available."""
        audio = audio.astype(np.float64)
        frame_length = int(sr * 0.02)
        hop_length = int(sr * 0.01)
        n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
        if n_frames == 0:
            return {
                'scoreq_mos': 3.0,
                'scoreq_stability': 0.5,
                'scoreq_activity': 0.5,
                'scoreq_energy': 0.5,
                # 标记为能量伪分数，过滤投票与分数融合时不计入有效模型（P2 修复）
                'fallback': True,
            }
        from numpy.lib.stride_tricks import as_strided
        shape = (n_frames, frame_length)
        strides = (audio.strides[0] * hop_length, audio.strides[0])
        frames = as_strided(audio, shape=shape, strides=strides)
        energy = np.sum(frames ** 2, axis=1)
        energy = np.maximum(energy, 1e-10)
        energy_db = 10 * np.log10(energy)
        max_db = np.max(energy_db)
        min_db = np.min(energy_db)
        if max_db - min_db > 1e-6:
            normalized_energy = (energy_db - min_db) / (max_db - min_db)
        else:
            normalized_energy = np.ones_like(energy_db)
        activity_ratio = np.mean(normalized_energy > 0.1)
        e_mean = np.mean(normalized_energy)
        e_std = np.std(normalized_energy)
        stability_score = 1.0 - min(e_std / (e_mean + 1e-10), 1.0)
        activity_score = min(activity_ratio * 2, 1.0)
        energy_score = min(e_mean * 1.5, 1.0)
        scoreq_mos = 1.0 + 2.0 * stability_score + 1.5 * activity_score + 0.5 * energy_score
        scoreq_mos = np.clip(scoreq_mos, 1.0, 5.0)
        return {
            'scoreq_mos': float(scoreq_mos),
            'scoreq_stability': float(stability_score),
            'scoreq_activity': float(activity_score),
            'scoreq_energy': float(energy_score),
            # 标记为能量伪分数，过滤投票与分数融合时不计入有效模型（P2 修复）
            'fallback': True,
        }

    def _fail_closed_if_scoreq_session_broken(self) -> None:
        """把 core scorer 的 reset 失败升级为 estimator 不可恢复的运行状态。"""
        scorer = self._scoreq
        error = getattr(scorer, "_broken_error", None)
        if error is None:
            return
        self._scoreq = None
        state = _estimator_lifecycle_state(self)
        with state.condition:
            state.phase = "broken"
            state.condition.notify_all()
        raise RuntimeError(
            "SCOREQ GPU session reset failed; estimator is broken"
        ) from error

    @staticmethod
    def _parse_result(result: Any) -> Dict[str, float]:
        """严格解析 SCOREQ schema，禁止缺失 MOS 或隐式类型转换。"""
        if not isinstance(result, Mapping) or "MOS" not in result:
            raise MosResultContractError(
                "SCOREQ result must be a mapping containing MOS"
            )
        score = result["MOS"]
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise MosResultContractError(
                "SCOREQ MOS must be a finite non-bool real number"
            )
        return {"scoreq_mos": float(score)}

    def compute(self, audio_path: str = None, *, audio_data: tuple = None) -> Dict:
        if audio_data is not None:
            audio, sr = audio_data
        else:
            audio, sr = sf.read(audio_path)
        audio = _first_channel_audio(audio).astype(np.float64, copy=False)
        if self._scoreq is not None:
            try:
                res = self._scoreq(audio, sr)
                return self._parse_result(res)
            except Exception as exc:
                self._fail_closed_if_scoreq_session_broken()
                if not self._allow_fallback:
                    raise RuntimeError(
                        f"SCOREQ inference failed and energy fallback is forbidden: {exc}"
                    ) from exc
                logger.debug("SCOREQ ONNX inference failed (%s), using fallback", exc)
        if not self._allow_fallback:
            raise RuntimeError("SCOREQ model is unavailable and energy fallback is forbidden")
        return self._compute_fallback(audio, sr)

    def compute_batch(self, items: list) -> List[Dict]:
        """批打分：优先使用 _SCOREQScorer.run_batch 做真正 ONNX batch 推理。

        direct fallback 边界允许在 ONNX 未加载或 run_batch 失败时逐条调用
        ``compute``，但这不代表 canonical production fallback；canonical 路径
        禁止 fallback，因此同类失败会抛出。输出 schema/count/value 契约错误
        始终直接抛出，不做逐条重试。
        """
        if not items:
            return []
        # GPU arena reset 按 core _SCOREQScorer 的实际 ONNX run 计数执行；
        # 外层 batch 的分片数与实际 sess.run 数不等，不能据此提前或延后 reset。
        if self._scoreq is None or not hasattr(self._scoreq, 'run_batch'):
            return [
                self.compute(audio_path=it, audio_data=None) if isinstance(it, str) else self.compute(audio_data=it)
                for it in items
            ]

        # 统一把 path items 加载为 (audio, sr)；加载失败的条目不进 run_batch
        # （(None, None) 会毒化整批导致回退逐条），直接产 error 结果，只把
        # 有效条目送批。
        results = [None] * len(items)
        audio_sr_list = []
        valid_idx = []
        for i, it in enumerate(items):
            if isinstance(it, str):
                try:
                    audio, sr = sf.read(it)
                    audio_sr_list.append((_first_channel_audio(audio), sr))
                    valid_idx.append(i)
                except Exception as e:
                    logger.debug("SCOREQ batch load failed for %s: %s", it, e)
                    results[i] = {'error': str(e)}
            else:
                audio, sr = it
                audio_sr_list.append((_first_channel_audio(audio), sr))
                valid_idx.append(i)

        try:
            batch_scores = self._scoreq.run_batch(audio_sr_list)
            if (
                isinstance(batch_scores, (str, bytes))
                or not isinstance(batch_scores, (list, tuple))
                or len(batch_scores) != len(valid_idx)
            ):
                raise MosResultContractError(
                    "SCOREQ batch result count must match input count"
                )
            for gi, score_dict in zip(valid_idx, batch_scores):
                results[gi] = self._parse_result(score_dict)
            return results
        except MosResultContractError:
            raise
        except Exception as exc:
            self._fail_closed_if_scoreq_session_broken()
            if not self._allow_fallback:
                raise RuntimeError(
                    "SCOREQ batch inference failed and energy fallback is forbidden: "
                    f"{exc}"
                ) from exc
            logger.debug("SCOREQ run_batch failed (%s), falling back to per-item", exc)
            for gi in valid_idx:
                it = items[gi]
                results[gi] = (
                    self.compute(audio_path=it, audio_data=None)
                    if isinstance(it, str)
                    else self.compute(audio_data=it)
                )
            return results


@register_mos("urgent_pk", MOSSpec(
    name="urgent_pk", target_sr=16000, score_kind="predicted_mos",
    primary_score_key="score", safe_total_seconds=32.0,
    speed_per_second=75.0, selectable=True, languages=("multilingual",),
    version="URGENT-PK",
))
class URGENTPKEstimator:
    """URGENT-PK 单音频复制策略估计器。

    按论文 III-F3 与官方 ``rank.py`` 的 ``get_mos_dup`` vote 语义，将同一条 16 kHz
    波形复制为 pair 两侧输入（[B, 2, T]）送入 ``AbTestModel.model``，取输出的两个
    per-side predicted MOS 及其均值；不使用 pairwise logit。
    外部源码和 checkpoint 都必须由配置显式提供，且绝不在运行期下载资产。
    """
    name = "urgent_pk"

    def __init__(self, checkpoint: str = None, source_dir: str = None,
                 encoder: str = None, backbone: str = None,
                 score_range: Any = None, use_gpu: bool = False,
                 expected_sha256: Optional[str] = None,
                 expected_commit: Optional[str] = None):
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"urgent_pk_checkpoint 不存在或不是文件: {checkpoint!r}")
        if not source_dir or not os.path.isdir(source_dir):
            raise FileNotFoundError(f"urgent_pk_source_dir 不存在或不是目录: {source_dir!r}")
        if not encoder:
            raise ValueError("urgent_pk_encoder 必须显式配置")
        if not backbone:
            raise ValueError("urgent_pk_backbone 必须显式配置")
        self.checkpoint = os.path.abspath(checkpoint)
        self.source_dir = os.path.abspath(source_dir)
        self.encoder = str(encoder)
        self.backbone = str(backbone)
        self.score_range = score_range
        # 期望资产身份（urgent_pk_expected_sha256 / urgent_pk_expected_commit）：
        # 由配置显式声明；可信 pickle 加载门禁强校验（verify_urgent_pk_trusted_assets）
        # 在 metadata 读取的 weights_only=False 回退处与 _load_model 的
        # load_from_checkpoint 之前都会执行，期望值缺失一律 fail-closed 拒绝。
        # digest/commit 只是必要门禁，source tree 的无 symlink 普通文件扫描独立执行。
        self.expected_sha256 = expected_sha256
        self.expected_commit = expected_commit
        self.resolved_score_range = normalize_urgent_pk_score_range(score_range)
        self.use_gpu = bool(use_gpu)
        self._model = None
        self._device = None
        self._is_fallback = False
        self._allow_fallback = True
        self._load_error = None
        # checkpoint 元数据可用时用于审计/一致性校验；文件名仅为无 metadata 时的
        # 可复现本地标识，绝不宣称其为模型训练版本。
        self._model_version = os.path.basename(self.checkpoint)

    def _load_bound_checkpoint(self, model_class):
        """让 digest、metadata 与 Lightning loader 消费同一私有快照。"""
        with _open_local_checkpoint_snapshot(self.checkpoint) as snapshot:
            verified = verify_urgent_pk_trusted_assets(
                snapshot.descriptor_path, self.source_dir,
                self.expected_sha256, self.expected_commit,
            )
            _URGENT_PK_VERIFIED_ASSETS[
                (os.path.realpath(self.checkpoint), os.path.realpath(self.source_dir))
            ] = dict(verified)
            metadata = read_urgent_pk_checkpoint_metadata(snapshot.descriptor_path)
            model = model_class.load_from_checkpoint(snapshot.descriptor_path)
            return metadata, model

    def _load_model(self) -> None:
        """仅在首次推理加载本地源码与 checkpoint，失败留给 fallback 策略处理。"""
        if self._model is not None:
            verified = verify_urgent_pk_trusted_assets(
                self.checkpoint, self.source_dir,
                self.expected_sha256, self.expected_commit,
            )
            if urgent_pk_verified_assets(self.checkpoint, self.source_dir) != verified:
                raise UrgentPKTrustError(
                    "URGENT-PK cached estimator asset identity is unavailable"
                )
            return
        try:
            module = load_urgent_pk_model_module(
                self.source_dir, self.expected_commit,
            )
            AbTestModel = module.AbTestModel
            import torch
            metadata, model = self._load_bound_checkpoint(AbTestModel)
            metadata_range = normalize_urgent_pk_score_range(
                metadata.get("urgent_pk_score_range", metadata.get("score_range"))
            )
            if self.resolved_score_range is not None and metadata_range is not None and self.resolved_score_range != metadata_range:
                raise ValueError(
                    f"URGENT-PK explicit score range {self.resolved_score_range} 与 checkpoint metadata {metadata_range} 不一致"
                )
            self.resolved_score_range = self.resolved_score_range or metadata_range
            if self.resolved_score_range is None:
                raise ValueError("URGENT-PK checkpoint 未声明 score range，且未显式配置 urgent_pk_score_range")
            for field, configured in (("encoder", self.encoder), ("backbone", self.backbone)):
                declared = metadata.get(field)
                if declared is not None and str(declared) != configured:
                    raise ValueError(
                        f"URGENT-PK checkpoint metadata {field}={declared!r} 与配置 {configured!r} 不一致"
                    )
            for key in ("model_version", "version", "git_commit", "commit"):
                if metadata.get(key):
                    self._model_version = str(metadata[key])
                    break
            # model 与 metadata 已在 _load_bound_checkpoint 中消费同一快照。
            model.eval()
            self._device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            self._model = model.to(self._device)
            self._is_fallback = False
        except ValueError:
            # 配置/metadata 契约冲突不可被 fallback 掩盖。
            raise
        except Exception as exc:
            self._model = None
            self._is_fallback = True
            self._load_error = str(exc)
            logger.warning("URGENT-PK 加载失败（source=%s, checkpoint=%s）: %s", self.source_dir, self.checkpoint, exc)

    @staticmethod
    def _fallback(audio: np.ndarray) -> Dict:
        energy = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2))) if len(audio) else 0.0
        score = float(np.clip(3.0 + energy * 0.5, 1.0, 5.0))
        return {"score": score, "fallback": True}

    def runtime_attestation(self) -> Dict[str, Any]:
        """校验 URGENT-PK checkpoint/source 与 device，返回既有 wire payload。"""
        verified = verify_urgent_pk_trusted_assets(
            self.checkpoint, self.source_dir,
            self.expected_sha256, self.expected_commit,
        )
        self._load_model()
        if self._is_fallback or self._model is None:
            raise ValueError("Step3 urgent_pk estimator is a fallback")
        requested = "cuda" if self.use_gpu else "cpu"
        device = str(self._device).lower()
        parameter_device = _model_parameters_device(self._model, "urgent_pk")
        if requested not in device or requested not in parameter_device:
            raise ValueError(f"Step3 urgent_pk device is not {requested}")
        if urgent_pk_verified_assets(self.checkpoint, self.source_dir) != verified:
            raise ValueError("Step3 URGENT-PK verified asset identity is unavailable")
        return {"device": device, "parameters": parameter_device}

    def _prepare(self, audio_path: str = None, audio_data: tuple = None) -> np.ndarray:
        if audio_data is None:
            audio, sr = sf.read(audio_path)
        else:
            audio, sr = audio_data
        audio = _first_channel_audio(audio)
        audio, _ = resample_to_target_sr(audio, int(sr), 16000)
        return np.asarray(audio, dtype=np.float32)

    def _predict_batch(self, waveforms: List[np.ndarray]) -> List[Dict]:
        self._load_model()
        if self._model is None:
            if not self._allow_fallback:
                raise RuntimeError(f"URGENT-PK 未成功加载真实权重: {self._load_error or 'unknown error'}")
            return [self._fallback(w) for w in waveforms]
        try:
            import torch
            max_len = max(len(w) for w in waveforms)
            padded = np.stack([np.pad(w, (0, max_len - len(w))) for w in waveforms])
            wav = torch.from_numpy(padded).to(self._device)
            # 官方 rank.py 的 get_mos_dup 语义：同一波形复制为 pair 两侧输入
            # （[B, 2, T]），``AbTestModel.model`` 返回 (pairwise logit, per-side
            # predicted MOS[B,2])；不使用 logit，取两侧 MOS 均值。
            audios = torch.stack([wav, wav], dim=1)
            with torch.inference_mode():
                _predicted, scores = self._model.model(audios)
            if not torch.is_tensor(scores):
                raise TypeError("URGENT-PK model 输出的 scores 必须是张量")
            scores = scores.detach().float().cpu()
            if scores.ndim != 2 or scores.shape[0] != len(waveforms) or scores.shape[1] != 2:
                raise ValueError(f"URGENT-PK model 输出 scores 形状异常: {tuple(scores.shape)}")
            left = scores[:, 0].tolist()
            right = scores[:, 1].tolist()
            out = []
            for l, r in zip(left, right):
                l, r = float(l), float(r)
                if not (math.isfinite(l) and math.isfinite(r)):
                    raise ValueError("URGENT-PK predicted MOS 必须为有限数")
                # 真实模型分数不在 estimator 内钳制；artifact 边界统一检查 [1, 5]。
                out.append({"score": (l + r) / 2.0})
            return out
        except ValueError:
            # 非有限或输出契约错误是模型结果无效，不能静默替换为 fallback。
            raise
        except Exception as exc:
            if self.use_gpu:
                if not cpu_downgrade_allowed():
                    raise RuntimeError(_cpu_downgrade_disallowed_message(self.name, str(exc))) from exc
                self._model = self._model.to("cpu")
                self._device = "cpu"
                self.use_gpu = False
                return self._predict_batch(waveforms)
            if not self._allow_fallback:
                raise RuntimeError(f"URGENT-PK 推理失败且禁止 fallback: {exc}") from exc
            logger.warning("URGENT-PK 推理失败，使用 fallback: %s", exc)
            self._is_fallback = True
            return [self._fallback(w) for w in waveforms]

    def compute(self, audio_path: str = None, *, audio_data: tuple = None) -> Dict:
        return self._predict_batch([self._prepare(audio_path, audio_data)])[0]

    def compute_batch(self, items: list) -> List[Dict]:
        """按精确 waveform length 分桶，避免无 mask 接口受到跨长度补零影响。"""
        waveforms = [self._prepare(item if isinstance(item, str) else None,
                                   None if isinstance(item, str) else item) for item in items]
        buckets: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for index, waveform in enumerate(waveforms):
            buckets.setdefault(len(waveform), []).append((index, waveform))
        results = [None] * len(waveforms)
        for bucket in buckets.values():
            scores = self._predict_batch([waveform for _, waveform in bucket])
            for (index, _), score in zip(bucket, scores):
                results[index] = score
        return results


# ── Model cache ──
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()
_RETIRED_ESTIMATORS = {}
_PENDING_ESTIMATOR_CLEANUP = {}
_PENDING_ESTIMATOR_CLEANUP_LOCK = threading.RLock()
_CACHED_ESTIMATOR_SCOPE = contextvars.ContextVar(
    "audio_filter_cached_estimators", default=None,
)
_UNCACHED_ESTIMATOR_SCOPE = contextvars.ContextVar(
    "audio_filter_uncached_estimators", default=None,
)


def _reset_mos_state_after_fork() -> None:
    """丢弃父进程运行时对象；child 绝不清理 PID-owned estimator/snapshot。"""
    global _URGENT_PK_RUNTIME_LOCK
    global _URGENT_PK_METADATA_CACHE, _URGENT_PK_METADATA_CACHE_LOCK
    global _URGENT_PK_DIGEST_CACHE, _URGENT_PK_DIGEST_CACHE_LOCK
    global _URGENT_PK_VERIFIED_ASSETS
    global _URGENT_PK_COMMIT_CACHE, _URGENT_PK_COMMIT_CACHE_LOCK
    global _CWD_LOCK, _DNSMOS_MODULE_COUNTER, _DNSMOS_MODULE_LOCK
    global _DNSMOS_PRIVATE_MODULES, _URGENT_PK_PRIVATE_MODULES
    global _URGENT_PK_PRIVATE_SNAPSHOT
    global _WORKER_DNSMOS, _WORKER_DNSMOS_ASSET_KEY
    global _WORKER_DNSMOS_ATEXIT_REGISTERED, _WORKER_DNSMOS_FINALIZER
    global _NISQA_ATEXIT_CALLBACKS, _MODEL_CACHE, _MODEL_CACHE_LOCK
    global _RETIRED_ESTIMATORS
    global _PENDING_ESTIMATOR_CLEANUP, _PENDING_ESTIMATOR_CLEANUP_LOCK
    global _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP
    global _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP_LOCK
    global _ESTIMATOR_LIFECYCLE_INIT_LOCK
    global _ALLOW_CPU_DOWNGRADE_CONTEXT, _CACHED_ESTIMATOR_SCOPE
    global _UNCACHED_ESTIMATOR_SCOPE

    # fork 可能发生在其他线程持锁期间；child 必须获得全新的 unlocked 锁。
    _URGENT_PK_RUNTIME_LOCK = threading.RLock()
    _URGENT_PK_METADATA_CACHE_LOCK = threading.RLock()
    _URGENT_PK_DIGEST_CACHE_LOCK = threading.RLock()
    _URGENT_PK_COMMIT_CACHE_LOCK = threading.RLock()
    _CWD_LOCK = threading.Lock()
    _DNSMOS_MODULE_LOCK = threading.Lock()
    _MODEL_CACHE_LOCK = threading.Lock()
    _PENDING_ESTIMATOR_CLEANUP_LOCK = threading.RLock()
    _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP_LOCK = threading.RLock()
    _ESTIMATOR_LIFECYCLE_INIT_LOCK = threading.Lock()

    # 这些缓存含 estimator、snapshot 或进程内可信状态，直接替换，禁止 close。
    _MODEL_CACHE = {}
    _RETIRED_ESTIMATORS = {}
    _PENDING_ESTIMATOR_CLEANUP = {}
    _PENDING_CHECKPOINT_SNAPSHOT_CLEANUP = {}
    _URGENT_PK_METADATA_CACHE = OrderedDict()
    _URGENT_PK_DIGEST_CACHE = OrderedDict()
    _URGENT_PK_VERIFIED_ASSETS = {}
    _URGENT_PK_COMMIT_CACHE = OrderedDict()
    _WORKER_DNSMOS = None
    _WORKER_DNSMOS_ASSET_KEY = None

    # 仅解绑本模块创建且仍指向父进程对象的私有 module。
    for name, module in tuple(_DNSMOS_PRIVATE_MODULES.items()):
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
    _DNSMOS_PRIVATE_MODULES = {}
    for name, module in tuple(_URGENT_PK_PRIVATE_MODULES.items()):
        if sys.modules.get(name) is module:
            sys.modules.pop(name, None)
    _URGENT_PK_PRIVATE_MODULES = {}
    _URGENT_PK_PRIVATE_SNAPSHOT = None
    _DNSMOS_MODULE_COUNTER = 0

    # 注销继承的退出回调只释放 child 引用，不调用父进程对象清理。
    atexit.unregister(_cleanup_dnsmos_worker_estimator)
    inherited_finalizer = _WORKER_DNSMOS_FINALIZER
    if inherited_finalizer is not None:
        inherited_finalizer.cancel()
    _WORKER_DNSMOS_FINALIZER = None
    _WORKER_DNSMOS_ATEXIT_REGISTERED = False
    for callback in tuple(_NISQA_ATEXIT_CALLBACKS):
        atexit.unregister(callback)
    _NISQA_ATEXIT_CALLBACKS = []

    # 不继承活跃 compute scope；配置镜像与各模型 options 保持原值。
    _ALLOW_CPU_DOWNGRADE_CONTEXT = contextvars.ContextVar(
        "audio_filter_allow_cpu_downgrade", default=_UNSET,
    )
    _CACHED_ESTIMATOR_SCOPE = contextvars.ContextVar(
        "audio_filter_cached_estimators", default=None,
    )
    _UNCACHED_ESTIMATOR_SCOPE = contextvars.ContextVar(
        "audio_filter_uncached_estimators", default=None,
    )


if "_MOS_AT_FORK_DISPATCH" not in globals():
    def _mos_at_fork_dispatch(module_name=__name__) -> None:
        """稳定 dispatch：reload 后动态调用当前模块的 reset 实现。"""
        module = sys.modules.get(module_name)
        reset = getattr(module, "_reset_mos_state_after_fork", None)
        if reset is not None:
            reset()

    _MOS_AT_FORK_DISPATCH = _mos_at_fork_dispatch
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_MOS_AT_FORK_DISPATCH)


def _forget_retired_estimator(identity: int) -> None:
    with _MODEL_CACHE_LOCK:
        _RETIRED_ESTIMATORS.pop(identity, None)


def _close_estimators(estimators) -> None:
    """尽量关闭全部 estimator；失败对象留在进程级 registry 供后续重试。"""
    seen = set()
    first_error = None
    for estimator in tuple(estimators):
        if estimator is None or id(estimator) in seen:
            continue
        identity = id(estimator)
        seen.add(identity)
        close = getattr(estimator, "close", None)
        state = getattr(estimator, "_mos_lifecycle_state", None)
        if not callable(close):
            with _PENDING_ESTIMATOR_CLEANUP_LOCK:
                _PENDING_ESTIMATOR_CLEANUP.pop(identity, None)
            _forget_retired_estimator(identity)
            if state is not None:
                with state.condition:
                    state.retirement_cleanup_finished = True
                    state.condition.notify_all()
            continue
        try:
            close()
        except BaseException as exc:
            with _PENDING_ESTIMATOR_CLEANUP_LOCK:
                _PENDING_ESTIMATOR_CLEANUP[identity] = estimator
            _forget_retired_estimator(identity)
            if first_error is None:
                first_error = exc
            elif hasattr(first_error, "add_note"):
                first_error.add_note(f"additional estimator close failure: {exc}")
        else:
            with _PENDING_ESTIMATOR_CLEANUP_LOCK:
                _PENDING_ESTIMATOR_CLEANUP.pop(identity, None)
            _forget_retired_estimator(identity)
        finally:
            if state is not None:
                with state.condition:
                    state.retirement_cleanup_finished = True
                    state.condition.notify_all()
    if first_error is not None:
        raise first_error


def _track_cached_estimator(estimator) -> None:
    """为当前 canonical compute scope 获取一个缓存 estimator 租约。"""
    scope = _CACHED_ESTIMATOR_SCOPE.get()
    if scope is None:
        return
    state = _estimator_lifecycle_state(estimator)
    with state.condition:
        if state.retired or state.phase != "open":
            raise RuntimeError("MOS estimator cache entry is unavailable")
        state.cache_leases += 1
    scope.append(estimator)


def _retire_estimators(estimators, *, wait_for_leases: bool) -> None:
    """标记缓存对象 retired；租约归零后才允许进入 close。"""
    closable = []
    seen = set()
    for estimator in tuple(estimators):
        if estimator is None or id(estimator) in seen:
            continue
        identity = id(estimator)
        seen.add(identity)
        state = _estimator_lifecycle_state(estimator)
        with _MODEL_CACHE_LOCK:
            _RETIRED_ESTIMATORS[identity] = estimator
        with state.condition:
            state.retired = True
            if wait_for_leases:
                while state.cache_leases:
                    state.condition.wait()
            if state.cache_leases == 0 and not state.retirement_cleanup_started:
                state.retirement_cleanup_started = True
                closable.append(estimator)
            elif wait_for_leases and state.retirement_cleanup_started:
                while not state.retirement_cleanup_finished:
                    state.condition.wait()
    _close_estimators(closable)


def _release_cached_estimators(estimators) -> None:
    """释放 scope 租约，并由最后一个租约持有者回收 retired estimator。"""
    closable = []
    for estimator in reversed(tuple(estimators)):
        state = _estimator_lifecycle_state(estimator)
        with state.condition:
            state.cache_leases -= 1
            if state.cache_leases < 0:
                raise RuntimeError("MOS estimator cache lease underflow")
            if (
                state.retired
                and state.cache_leases == 0
                and not state.retirement_cleanup_started
            ):
                state.retirement_cleanup_started = True
                closable.append(estimator)
            state.condition.notify_all()
    _close_estimators(closable)


def _retry_pending_estimator_cleanup() -> None:
    """重试此前 close 失败对象；成功前始终保留强引用。"""
    with _PENDING_ESTIMATOR_CLEANUP_LOCK:
        pending = tuple(_PENDING_ESTIMATOR_CLEANUP.values())
    _close_estimators(pending)


def close_mos_estimators(estimators) -> None:
    """关闭临时 estimator；失败对象进入进程级 registry 等待重试。"""
    _close_estimators(estimators)


def _close_uncached_estimators(function):
    """管理一次 canonical compute 的缓存租约与临时 estimator。"""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        leased = []
        owned = []
        cached_token = _CACHED_ESTIMATOR_SCOPE.set(leased)
        uncached_token = _UNCACHED_ESTIMATOR_SCOPE.set(owned)
        primary = None
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _CACHED_ESTIMATOR_SCOPE.reset(cached_token)
            _UNCACHED_ESTIMATOR_SCOPE.reset(uncached_token)
            for cleanup_name, cleanup in (
                ("cached estimator retirement", lambda: _release_cached_estimators(leased)),
                ("temporary estimator", lambda: _close_estimators(owned)),
            ):
                try:
                    cleanup()
                except BaseException as cleanup_error:
                    if primary is None:
                        primary = cleanup_error
                    else:
                        _add_exception_note(
                            primary,
                            f"{cleanup_name} cleanup failed: {cleanup_error}",
                        )
            if primary is not None and sys.exc_info()[0] is None:
                raise primary
    return wrapped


def _get_cache_key(mos_type: str, use_gpu: bool, allow_fallback: bool = True):
    if mos_type == "urgent_pk":
        options = _URGENT_PK_OPTIONS
        identity = tuple(repr(options.get(k)) for k in (
            "checkpoint", "source_dir", "encoder", "backbone", "score_range",
            "expected_sha256", "expected_commit",
        ))
        # 每次访问 cache 前由 get_mos_estimator 重验可信 assets；这里保留
        # verifier 产出的 package digest，确保相同路径/commit 的不同 package
        # 身份不会复用缓存对象。
        verified = urgent_pk_verified_assets(
            options.get("checkpoint"), options.get("source_dir"),
        )
        return (
            mos_type, bool(use_gpu), bool(allow_fallback), *identity,
            verified.get("checkpoint_sha256"),
            verified.get("source_commit"),
            verified.get("source_package_sha256"),
        )
    if mos_type in {"scoreq", "dnsmos"}:
        options = _SCOREQ_OPTIONS if mos_type == "scoreq" else _DNSMOS_OPTIONS
        default_dir = os.path.join(
            THIRD_PARTY_DIR, "SCOREQ" if mos_type == "scoreq" else "DNSMOS",
        )
        return (
            mos_type,
            bool(use_gpu),
            bool(allow_fallback),
            _canonical_model_dir(options.get("model_dir"), default_dir),
            _verified_model_tree_digest(
                mos_type.upper(), options.get("verified_model_tree_sha256"),
            ),
        )
    if mos_type in {"nisqa", "utmos"}:
        options = _NISQA_OPTIONS if mos_type == "nisqa" else _UTMOS_OPTIONS
        # 只冻结调用方显式绑定的纯数据 identity；这里不得访问 pathname、扫描模型
        # 目录或读取 checkpoint，确保已加载对象不依赖 public asset 继续存在。
        identity_fields = (
            "model_path", "model_version", "model_sha256",
            "checkpoint_sha256", "checkpoint_size_bytes",
        )
        return (
            mos_type, bool(use_gpu), bool(allow_fallback),
            *((field, type(options.get(field)), options.get(field))
              for field in identity_fields),
        )
    return f"{mos_type}_{use_gpu}_{allow_fallback}"


def _verified_tree_model_kwargs(
    options: Mapping[str, Any], model: str,
) -> Dict[str, Any]:
    """只投影 SCOREQ/DNSMOS estimator 可消费的绑定字段。"""
    default_dir = os.path.join(THIRD_PARTY_DIR, model.upper())
    return {
        "model_dir": _canonical_model_dir(
            options.get("model_dir"), default_dir,
        ),
        "verified_model_tree_sha256": _verified_model_tree_digest(
            model.upper(), options.get("verified_model_tree_sha256"),
        ),
    }


def _verified_local_model_kwargs(options: Mapping[str, Any], model: str) -> Dict[str, Any]:
    """验证本地单文件模型目录身份，并只返回 estimator 可消费参数。"""
    path = options.get("model_path")
    version = options.get("model_version")
    sha256 = options.get("model_sha256")
    if not path or not version or not sha256:
        raise ValueError(f"{model} requires model_path/model_version/model_sha256")
    canonical = os.path.realpath(os.path.abspath(os.fspath(path)))
    if canonical != path or not os.path.isfile(canonical):
        raise ValueError(f"{model} model_path must be a canonical existing file")
    verify_local_model_asset(LocalModelAsset(
        local_path=os.path.dirname(canonical), version=version, sha256=sha256,
    ))
    checkpoint_sha256 = options.get("checkpoint_sha256")
    if checkpoint_sha256 is not None and (
        type(checkpoint_sha256) is not str
        or len(checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
    ):
        raise ValueError(f"{model} checkpoint_sha256 must be lowercase SHA-256 hex")
    return {
        "model_path": canonical,
        "checkpoint_sha256": checkpoint_sha256,
    }


def get_mos_estimator(mos_type: str, use_gpu: bool = False, use_cache: bool = True, allow_fallback: bool = True):
    local_options = None
    if mos_type == "urgent_pk":
        kwargs = dict(_URGENT_PK_OPTIONS)
        verified = verify_urgent_pk_trusted_assets(
            kwargs.get("checkpoint"), kwargs.get("source_dir"),
            kwargs.get("expected_sha256"), kwargs.get("expected_commit"),
        )
        if (
            verified["checkpoint_sha256"] != kwargs.get("expected_sha256")
            or verified["source_commit"] != kwargs.get("expected_commit")
        ):
            raise UrgentPKTrustError("URGENT-PK verified asset identity is detached from options")
    elif mos_type == "scoreq":
        kwargs = _verified_tree_model_kwargs(_SCOREQ_OPTIONS, mos_type)
    elif mos_type == "dnsmos":
        kwargs = _verified_tree_model_kwargs(_DNSMOS_OPTIONS, mos_type)
    elif mos_type == "nisqa":
        local_options = _NISQA_OPTIONS
        kwargs = None
    elif mos_type == "utmos":
        local_options = _UTMOS_OPTIONS
        kwargs = None
    else:
        kwargs = {}
    if not use_cache:
        if local_options is not None:
            kwargs = _verified_local_model_kwargs(local_options, mos_type)
        estimator = create_mos_estimator(
            mos_type, use_gpu=use_gpu, allow_fallback=allow_fallback, **kwargs,
        )
        scope = _UNCACHED_ESTIMATOR_SCOPE.get()
        if scope is not None:
            scope.append(estimator)
        return estimator

    cache_key = _get_cache_key(mos_type, use_gpu, allow_fallback)
    evicted = []
    with _MODEL_CACHE_LOCK:
        # 命中、租约获取与 identity 替换必须同锁原子完成，避免返回即被关闭。
        est = _MODEL_CACHE.get(cache_key)
        if est is not None and getattr(est, "use_gpu", use_gpu) == use_gpu:
            # NISQA/UTMOS 热命中只依赖显式 identity，不触碰 public pathname。
            _track_cached_estimator(est)
            return est
        if local_options is not None:
            kwargs = _verified_local_model_kwargs(local_options, mos_type)
        replacement = create_mos_estimator(
            mos_type, use_gpu=use_gpu,
            allow_fallback=allow_fallback, **kwargs,
        )
        for key, cached in tuple(_MODEL_CACHE.items()):
            if isinstance(key, tuple):
                same_slot = key[:2] == cache_key[:2]
            else:
                same_slot = key.startswith(f"{mos_type}_{use_gpu}_")
            if same_slot:
                _MODEL_CACHE.pop(key, None)
                if cached is not replacement:
                    evicted.append(cached)
        _MODEL_CACHE[cache_key] = replacement
        _track_cached_estimator(replacement)
    # 新对象完整构造并发布后才 retire 旧 identity；活跃租约归零前不执行 close。
    _retire_estimators(evicted, wait_for_leases=False)
    return replacement


def clear_model_cache():
    global _MODEL_CACHE
    with _MODEL_CACHE_LOCK:
        old = _MODEL_CACHE
        _MODEL_CACHE = {}
        retired = tuple(_RETIRED_ESTIMATORS.values())
    with _PENDING_ESTIMATOR_CLEANUP_LOCK:
        pending = tuple(_PENDING_ESTIMATOR_CLEANUP.values())
    first_error = None
    retire_succeeded = False
    try:
        # 显式 clear 保持同步语义：等待 canonical compute 租约释放后再关闭。
        _retire_estimators((*retired, *old.values()), wait_for_leases=True)
        retire_succeeded = True
    except BaseException as exc:
        first_error = exc
    cleanup_candidates = pending
    if retire_succeeded:
        # 等待 lease 期间，最后一个 lease 持有者可能 close 失败并新登记 pending；
        # 必须在返回前重新取快照，不能把这类失败留到下一次 clear。
        with _PENDING_ESTIMATOR_CLEANUP_LOCK:
            cleanup_candidates = tuple(_PENDING_ESTIMATOR_CLEANUP.values())
    try:
        _close_estimators(cleanup_candidates)
    except BaseException as exc:
        if first_error is None:
            first_error = exc
        else:
            _add_exception_note(first_error, f"pending estimator cleanup failed: {exc}")
    try:
        _retry_pending_checkpoint_snapshot_cleanup()
    except BaseException as exc:
        if first_error is None:
            first_error = exc
        else:
            _add_exception_note(first_error, f"checkpoint snapshot cleanup failed: {exc}")
    try:
        clear_urgent_pk_checkpoint_metadata_cache()
        with _URGENT_PK_RUNTIME_LOCK:
            _unload_urgent_pk_private_modules()
            snapshot = _detach_urgent_pk_private_snapshot()
            if snapshot is not None:
                snapshot.close()
    except BaseException as exc:
        if first_error is None:
            first_error = exc
        else:
            _add_exception_note(first_error, f"metadata/module cache cleanup failed: {exc}")
    import gc
    gc.collect()
    logger.debug("Model and URGENT-PK metadata caches cleared")
    if first_error is not None:
        raise first_error


def _cleanup_mos_at_exit() -> None:
    """退出时清空缓存，并对瞬时 estimator/snapshot 失败各重试一次。"""
    try:
        clear_model_cache()
    except BaseException:
        for label, retry in (
            ("estimator", _retry_pending_estimator_cleanup),
            ("checkpoint snapshot", _retry_pending_checkpoint_snapshot_cleanup),
        ):
            try:
                retry()
            except BaseException as exc:
                logger.warning("MOS %s atexit cleanup failed: %s", label, exc)


if "_MOS_CACHE_ATEXIT_REGISTERED" not in globals():
    atexit.register(_cleanup_mos_at_exit)
    _MOS_CACHE_ATEXIT_REGISTERED = True


def check_mos_estimator_status(mos_type: str, use_gpu: bool = False) -> dict:
    """预加载并检查单个 MOS 模型是否可用（使用 get_mos_estimator 以预热缓存）。返回 {'ok': bool, 'is_fallback': bool, 'message': str}。"""
    try:
        est = get_mos_estimator(mos_type, use_gpu=use_gpu, use_cache=True)
    except Exception as e:
        return {'ok': False, 'is_fallback': True, 'message': f'加载失败: {e}'}
    is_fallback = getattr(est, '_is_fallback', False)
    if is_fallback:
        return {'ok': True, 'is_fallback': True, 'message': '无真实权重，使用能量 fallback（非真实 MOS）'}
    if mos_type == 'dnsmos':
        has_real = getattr(est, 'compute_score', None) is not None
        if has_real and hasattr(est.compute_score, 'onnx_sess'):
            prov = est.compute_score.onnx_sess.get_providers()
            dev_str = 'GPU' if 'CUDAExecutionProvider' in prov else 'CPU'
            msg = f'ONNX 已加载 ({dev_str})'
        else:
            msg = 'ONNX 已加载 (默认 provider)' if has_real else 'ONNX 未找到，使用 fallback'
        return {'ok': True, 'is_fallback': not has_real, 'message': msg}
    if mos_type == 'nisqa':
        has_real = getattr(est, '_nisqa_model', None) is not None
        dev = getattr(est._nisqa_model, 'dev', None) if has_real else None
        dev_str = 'GPU' if (dev is not None and 'cuda' in str(dev)) else 'CPU'
        msg = f'nisqa.tar 已加载 ({dev_str})' if has_real else '权重未找到，使用 fallback'
        return {'ok': True, 'is_fallback': not has_real, 'message': msg}
    if mos_type == 'utmos':
        has_real = getattr(est, 'model', None) is not None
        dev = getattr(est, '_device', None) if has_real else None
        dev_str = 'GPU' if (dev is not None and 'cuda' in str(dev)) else 'CPU'
        msg = f'UTMOS22 已加载 ({dev_str})' if has_real else '加载失败，使用 fallback'
        return {'ok': True, 'is_fallback': not has_real, 'message': msg}
    if mos_type == 'scoreq':
        has_real = getattr(est, '_scoreq', None) is not None
        dev_str = getattr(est._scoreq, '_device_str', '未知') if has_real else '未知'
        msg = f'SCOREQ ONNX 已加载 ({dev_str})' if has_real else 'ONNX 未找到，使用 fallback'
        return {'ok': True, 'is_fallback': not has_real, 'message': msg}
    if mos_type == 'urgent_pk':
        has_real = getattr(est, '_model', None) is not None
        version = getattr(est, '_model_version', '未知')
        verified = urgent_pk_verified_assets(getattr(est, 'checkpoint', None), getattr(est, 'source_dir', None))
        if not has_real and not is_fallback:
            msg = f'URGENT-PK 已配置，等待首次推理懒加载 ({version})'
        else:
            dev = getattr(est, '_device', None)
            msg = f'URGENT-PK 已加载 ({dev}, {version})' if has_real else f'URGENT-PK 加载失败，使用 fallback: {getattr(est, "_load_error", "未知错误")}'
        return {'ok': True, 'is_fallback': is_fallback, 'loaded': has_real, 'message': msg, **verified}
    return {'ok': True, 'is_fallback': is_fallback, 'message': '已加载'}


def validate_mos_estimators(mos_types: List[str], use_gpu: bool = False) -> Dict[str, dict]:
    """预加载并校验 config 中的全部 MOS 模型；URGENT-PK 保持首次推理懒加载。"""
    out = {}
    for mt in mos_types:
        out[mt] = check_mos_estimator_status(mt, use_gpu=use_gpu)
    return out


def validate_mos_estimators_eager(
    mos_types: List[str], use_gpu: bool = False, allow_fallback: bool = True,
) -> Dict[str, dict]:
    """显式 eager 校验入口；用于启动期必须确认真实权重可加载的模型。"""
    out = validate_mos_estimators(
        [mt for mt in mos_types if mt != "urgent_pk"], use_gpu=use_gpu,
    )
    if "urgent_pk" not in mos_types:
        return out
    try:
        estimator = get_mos_estimator(
            "urgent_pk", use_gpu=use_gpu, use_cache=True,
            allow_fallback=allow_fallback,
        )
        estimator._load_model()
        if estimator._model is None:
            raise RuntimeError(estimator._load_error or "未成功加载真实权重")
        device = getattr(estimator, "_device", None)
        version = getattr(estimator, "_model_version", "未知")
        verified = urgent_pk_verified_assets(
            getattr(estimator, "checkpoint", None), getattr(estimator, "source_dir", None),
        )
        out["urgent_pk"] = {
            "ok": True,
            "is_fallback": bool(getattr(estimator, "_is_fallback", False)),
            "loaded": True,
            "message": f"URGENT-PK 已加载 ({device}, {version})",
            **verified,
        }
    except Exception as exc:
        out["urgent_pk"] = {
            "ok": False, "is_fallback": True, "loaded": False,
            "message": f"加载失败: {exc}",
        }
    return out


def create_mos_estimator(mos_type: str, allow_fallback: bool = True, **kwargs):
    """使用注册表创建 MOS 评估器（支持插件化扩展）。

    allow_fallback=False 时，若模型权重未加载成功（_is_fallback=True）则直接抛异常，
    避免把能量/简单特征伪分数当成真实 MOS 参与后续过滤。

    注意：不开启 ``torch.backends.cudnn.benchmark`` —— 本 pipeline 的 torch 模型
    （UTMOS/NISQA）批内时长随长度分桶 / DataLoader 变化，shape 不稳定，autotune
    会反复重跑卷积搜索反而变慢；DNSMOS/SCOREQ 走 ONNX Runtime，不受该开关影响。
    """
    if mos_type not in _MOS_REGISTRY:
        raise ValueError(f"Unknown MOS type: {mos_type}. Available: {sorted(_MOS_REGISTRY.keys())}")
    if mos_type in {"scoreq", "dnsmos", "nisqa", "utmos"}:
        kwargs = dict(kwargs)
        kwargs["allow_fallback"] = bool(allow_fallback)
    est = _MOS_REGISTRY[mos_type](**kwargs)
    if mos_type == "urgent_pk":
        # URGENT-PK 必须保持首次推理才 import/load 的 lazy 语义；因此把
        # fail-closed 标记下沉给推理路径，而非在构造器中抢先加载模型。
        est._allow_fallback = bool(allow_fallback)
    elif mos_type == "scoreq":
        # CUDA-primary hybrid EP 是静态 graph partition，不是运行时 fallback；
        # pipeline 禁止 estimator 加载/推理失败后改用能量伪分。
        est._allow_fallback = bool(allow_fallback)
    if not allow_fallback and getattr(est, '_is_fallback', False):
        raise RuntimeError(
            f"MOS model '{mos_type}' 未成功加载真实权重，且当前禁止 fallback。"
            f"请检查 third_party_scripts/{mos_type.upper()}/ 下的模型文件。"
        )
    return est


@_close_uncached_estimators
def compute_mos_scores(audio_path: str = None, mos_types: List[str] = None,
                       use_gpu: bool = False, use_cache: bool = True,
                       model_specific_sr: bool = False,
                       early_exit_below: Optional[float] = None,
                       early_exit_first_model_only: bool = False,
                       allow_fallback: bool = True,
                       urgent_options: Optional[Mapping[str, Any]] = None,
                       scoreq_options: Optional[Mapping[str, Any]] = None,
                       dnsmos_options: Optional[Mapping[str, Any]] = None,
                       nisqa_options: Optional[Mapping[str, Any]] = None,
                       utmos_options: Optional[Mapping[str, Any]] = None,
                       *, audio_data: tuple = None) -> Dict:
    """Compute MOS scores.  Accepts file path or ``(array, sr)`` in-memory data.

    When model_specific_sr is True, audio is resampled per model: 48k for nisqa, 16k for others (utmos included).

    early_exit_below: 若设置，按 mos_types 顺序逐模型计算；当「当前模型」标量分 < 该阈值时不再计算后续模型，
    未跑模型写入 0 分。
    early_exit_first_model_only: 为 True 时仅用 mos_types 中第一个模型的分数做早停判断（通常即 dnsmos），
    先算第一个，若 < 阈值则其余不算、填 0；若 >= 阈值则继续算完其余模型，不再做早停。
    """
    results = {'audio_path': audio_path or '<in-memory>'}
    mt_list = list(mos_types or [])
    # P-3：统一按模型实测速度排序执行（scoreq 235 / utmos 133 / dnsmos 105 /
    # nisqa 25 条/s，10s 段、80GB 卡实测）：早停时快的先跑并淘汰低分、慢的
    # 少跑；非早停时执行顺序一致（结果相同，scores 键顺序按速度序）。
    if len(mt_list) > 1:
        mt_list = sorted(mt_list, key=lambda mt: -_mos_speed_per_second(mt))
    first_model = mt_list[0] if mt_list else None
    for mos_type in mt_list:
        try:
            if mos_type == "urgent_pk":
                binding = bind_urgent_pk_options(urgent_options)
            elif mos_type == "scoreq":
                binding = bind_scoreq_options(scoreq_options)
            elif mos_type == "dnsmos":
                binding = bind_dnsmos_options(dnsmos_options)
            elif mos_type == "nisqa":
                binding = bind_nisqa_options(nisqa_options)
            elif mos_type == "utmos":
                binding = bind_utmos_options(utmos_options)
            else:
                binding = contextlib.nullcontext()
            with binding:
                estimator = get_mos_estimator(mos_type, use_gpu=use_gpu, use_cache=use_cache, allow_fallback=allow_fallback)
                if audio_data is not None:
                    if model_specific_sr:
                        target_sr = get_model_target_sr(mos_type)
                        audio, sr = audio_data
                        audio, sr = resample_to_target_sr(audio, sr, target_sr)
                        results[mos_type] = estimator.compute(audio_data=(audio, sr))
                    else:
                        results[mos_type] = estimator.compute(audio_data=audio_data)
                else:
                    results[mos_type] = estimator.compute(audio_path)
        except Exception as exc:
            results[mos_type] = {'error': _bounded_error_detail(exc)}
        if early_exit_below is not None:
            only_first = early_exit_first_model_only and first_model is not None
            if only_first and mos_type != first_model:
                continue
            s = extract_mos_scalar(mos_type, results[mos_type])
            if s is not None and s < float(early_exit_below):
                results['_early_exit'] = True
                break
    if early_exit_below is not None:
        for mt in mt_list:
            if mt not in results:
                results[mt] = _early_exit_zero_score(mt)
    return results


# 兼容旧的模块级常量；唯一数据源为 registry 的 MOSSpec。
SCORE_KEY_MAP = MappingProxyType({
    name: spec.primary_score_key for name, spec in _MOS_SPECS.items()
})
MODEL_TARGET_SR = MappingProxyType({
    name: spec.target_sr for name, spec in _MOS_SPECS.items()
})
_MOS_SAFE_TOTAL_SEC = MappingProxyType({
    name: spec.safe_total_seconds for name, spec in _MOS_SPECS.items()
})

# DNSMOS P.835 四维度分键映射。筛选判定时（mos_thresholds['dnsmos'] 为 dict
# 配置）按维度逐项过线，OVRL/SIG/BAK/P808 四维全过（AND）才计 DNSMOS 这一票
# 通过；代表分（SCORE_KEY_MAP）仍用于 fused_mos_score 融合与早停。
DNSMOS_DIMENSION_KEYS = {
    'ovrl': 'dnsmos_ovr_mos',
    'sig':  'dnsmos_sig_mos',
    'bak':  'dnsmos_bak_mos',
    'p808': 'dnsmos_p808_mos',
}


def _early_exit_zero_score(mos_type: str) -> dict:
    """早停时未跑的模型写入 0 分并标记 skipped，使其在融合与 Step3 判定时不计入有效模型数。"""
    try:
        key = get_mos_spec(mos_type).primary_score_key
    except ValueError:
        key = 'score'
    return {key: 0.0, 'skipped': True}


def extract_mos_scalar(mos_type: str, result) -> Optional[float]:
    """从单模型结果中取出标量 MOS；error/skipped/fallback/缺键时返回 None。"""
    if not isinstance(result, dict) or result.get('error') or result.get('skipped') or result.get('fallback'):
        return None
    try:
        key = get_mos_spec(mos_type).primary_score_key
    except ValueError:
        return None
    v = result.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _is_valid_mos_fusion_entry(result) -> bool:
    # fallback 伪分数与 error/skipped 一样不参与融合，避免被当成真实 MOS
    return (
        isinstance(result, dict)
        and not result.get('error')
        and not result.get('skipped')
        and not result.get('fallback')
    )


def fuse_mos_scores(mos_results: Dict, strategy: str = "average") -> float:
    scores = []
    for mos_type, result in mos_results.items():
        if mos_type == 'audio_path':
            continue
        if _is_valid_mos_fusion_entry(result):
            score = extract_mos_scalar(mos_type, result)
            if score is not None:
                scores.append(score)
    if not scores:
        return 0.0
    if strategy == "average":
        return np.mean(scores)
    elif strategy == "min":
        return np.min(scores)
    elif strategy == "max":
        return np.max(scores)
    return np.mean(scores)


def _fuse_entry_dimensions(mos_type: str, result: Dict, threshold_cfg, base_threshold: float):
    """计算单个模型在阈值配置下的维度级判定。

    返回 ``(score, threshold, passed, dimensions)``，分数缺失/无效时返回 None：
    - 普通模型，或 dnsmos 为 float 阈值：单一维度判定（代表分，SCORE_KEY_MAP），
      ``dimensions`` 为 None（与旧行为完全一致）；
    - dnsmos 且阈值为 dict：对 dict 中列出的每个维度取分、逐项过线，
      ``passed`` = 所有列出维度全过（AND），``dimensions`` 携带逐维度明细；
      dict 中未列出的维度不参与判定。dict 为空（防御）时退回代表分判定。
    """
    score = extract_mos_scalar(mos_type, result)
    if score is None:
        return None

    if isinstance(threshold_cfg, dict) and mos_type == 'dnsmos':
        dimensions = []
        all_passed = True
        for dim_name, dim_threshold in threshold_cfg.items():
            dim_key = DNSMOS_DIMENSION_KEYS.get(dim_name)
            if dim_key is None:
                # config.validate 已拦截未知维度键；此处防御性跳过，避免静默计入
                continue
            dim_score = result.get(dim_key)
            if dim_score is None:
                all_passed = False
                dimensions.append({
                    'name': dim_name, 'score': None,
                    'threshold': float(dim_threshold), 'passed': False,
                })
                continue
            dim_score = float(dim_score)
            dim_passed = dim_score >= float(dim_threshold)
            all_passed = all_passed and dim_passed
            dimensions.append({
                'name': dim_name, 'score': dim_score,
                'threshold': float(dim_threshold), 'passed': dim_passed,
            })
        if not dimensions:
            # 空 dict：退回单维度（代表分）判定
            return score, float(base_threshold), score >= float(base_threshold), None
        # 顶层 threshold 展示用 ovrl 阈值（未列时用基础阈值）
        ovrl_threshold = threshold_cfg.get('ovrl', base_threshold)
        return score, float(ovrl_threshold), all_passed, dimensions

    threshold_val = float(threshold_cfg) if isinstance(threshold_cfg, (int, float)) else float(base_threshold)
    return score, threshold_val, score >= threshold_val, None


def fuse_mos_with_threshold(mos_results: Dict, threshold: float = 3.0,
                            mos_thresholds: Dict[str, float] = None) -> Dict:
    if mos_thresholds is None:
        mos_thresholds = {}

    scores = []
    for mos_type, result in mos_results.items():
        if mos_type == 'audio_path':
            continue
        if not _is_valid_mos_fusion_entry(result):
            continue
        entry = _fuse_entry_dimensions(
            mos_type, result, mos_thresholds.get(mos_type, threshold), threshold,
        )
        if entry is None:
            continue
        score, threshold_val, passed, dimensions = entry
        item = {
            'type': mos_type,
            'score': score,
            'threshold': threshold_val,
            'passed': passed,
        }
        if dimensions is not None:
            item['dimensions'] = dimensions
        scores.append(item)

    total = len(scores)
    passed = sum(1 for s in scores if s['passed'])

    return {
        'scores': scores,
        'total_models': total,
        'passed_models': passed,
        'pass_rate': passed / total if total > 0 else 0,
        'fused_score': np.mean([s['score'] for s in scores]) if scores else 0,
        'majority_passed': passed > total // 2
    }


def compute_quality_passed(
    mos_results: Dict,
    config: Any,
) -> Tuple[bool, float, Dict]:
    """集中通过判定：根据 MOS 结果与配置返回 (quality_passed, fused_score, fusion_result)。

    config 可为 PipelineConfig 或 dict（子进程用）。FilterStage 与 Step3 行级
    判定共用本函数，避免判定逻辑在 filter.py / pipeline.py 多处漂移。

    与 FilterStage 历史口径一致：
    - fallback 伪分数不计入有效模型数（不参与过线计数与融合，fuse 层已过滤）；
    - enable_multi_mos=true 时「过各自阈值的有效模型数 ≥ min_mos_models」即通过
      （min_mos_models=模型总数时等价全票通过；旧版「多数通过 + 有效数下限」
      双重规则已简化为单一口径）；
    - 否则按融合分 ≥ 基础阈值判定。
    """

    def _thresholds(cfg):
        if isinstance(cfg, dict):
            return cfg.get('effective_mos_thresholds') or cfg.get('mos_thresholds') or {}
        dataset_name = getattr(cfg, 'dataset_name', None)
        if hasattr(cfg, 'get_effective_mos_thresholds'):
            return cfg.get_effective_mos_thresholds(dataset_name)
        return getattr(cfg, 'mos_thresholds', None) or {}

    def _mos_threshold(cfg):
        if isinstance(cfg, dict):
            return cfg.get('mos_threshold', 3.0)
        return getattr(cfg, 'mos_threshold', 3.0)

    def _enable_multi_mos(cfg):
        if isinstance(cfg, dict):
            return cfg.get('enable_multi_mos', True)
        return getattr(cfg, 'enable_multi_mos', True)

    def _min_mos_models(cfg):
        if isinstance(cfg, dict):
            return cfg.get('min_mos_models', 2)
        return getattr(cfg, 'min_mos_models', 2)

    def _mos_fusion(cfg):
        if isinstance(cfg, dict):
            return cfg.get('mos_fusion', 'average')
        return getattr(cfg, 'mos_fusion', 'average')

    thresholds = _thresholds(config)
    base_threshold = _mos_threshold(config)
    fusion_result = fuse_mos_with_threshold(mos_results, base_threshold, thresholds)
    fused_score = fuse_mos_scores(mos_results, _mos_fusion(config))
    if _enable_multi_mos(config):
        # 过线模型数（仅计 error/skipped/fallback 除外的有效分数，fuse 层已过滤）
        # 达到 min_mos_models 即通过：min_mos_models=模型总数时等价全票通过
        quality_passed = fusion_result['passed_models'] >= _min_mos_models(config)
    else:
        quality_passed = fused_score >= base_threshold
    return quality_passed, fused_score, fusion_result


def _prepare_batch_items_for_model(batch_items: list, target_sr: int, resample_method: str = "auto") -> list:
    """Resample each item to target_sr. Path items are loaded then resampled."""
    out = []
    for it in batch_items:
        if isinstance(it, str):
            audio, sr = sf.read(it)
            audio, sr = resample_to_target_sr(
                _first_channel_audio(audio), sr, target_sr, method=resample_method,
            )
            out.append((audio, sr))
        else:
            audio, sr = it
            audio, sr = resample_to_target_sr(
                _first_channel_audio(audio), sr, target_sr, method=resample_method,
            )
            out.append((audio, sr))
    return out


def _item_duration_sec(item) -> Optional[float]:
    """取条目的时长（秒）；路径条目长度未知返回 None。"""
    if isinstance(item, str):
        return None
    try:
        audio, sr = item
        return len(audio) / float(sr) if sr else None
    except Exception:
        return None


def _dnsmos_segment_estimate(dur_sec: float) -> float:
    """估算 DNSMOS run_batch_audio 内部为一条音频生成的 segment 数（显存权重）。

    dnsmos_local.run_batch_audio 以 9.01s 为窗、1s 为 hop 滑窗切 segment，且
    短音频会被重复拼接补长到 ≥ 9.01s：显存峰值 ∝ Σ num_hops，与"总时长"
    不呈线性关系（20s→11 seg、10s→1 seg、8s 补长到 16s→7 seg）。按总时长
    切批在长音频下会把 segment 数放大数倍（64s 预算装 3 条 20s → 33 seg），
    顶爆显存反复 OOM。故 DNSMOS 的批预算必须按 segment 数而非秒数计算，
    本函数复刻其 num_hops 计算作为每条目权重。

    注意（分段打分模式）：2-9s 的段会被官方实现重复拼接补长到 ≥ 9.01s 再
    打分，其分数语义弱于 ≥9s 的段（重复自身可能放大伪影/能量）；此行为为
    官方做法，保留。
    """
    if dur_sec <= 0:
        return 1.0
    if dur_sec < 9.01:
        # 短音频补长：重复拼接自身直到 ≥ 9.01s
        while dur_sec < 9.01:
            dur_sec *= 2
    return max(1, int(np.floor(dur_sec) - 9.01) + 1)


def _log_gpu_contention_diagnostic():
    """OOM 时记录当前卡上所有进程的显存占用，供排查叠加/挤卡。

    单进程修复后（DNSMOS 11.5GB + torch ~18GB ≈ 30GB）不应 OOM；若仍出现
    "Available memory ... smaller than requested"，多为多任务叠加（每进程
    15-40GB，2 个即超 80GB）。此诊断在每次 OOM 时打印卡上全部进程与显存，
    使下次 OOM 日志直接可见真相（谁占的显存）。
    """
    try:
        import subprocess
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,process_name',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if out:
            logger.warning("OOM 诊断 - 本卡全部进程与显存:\n%s", out)
    except Exception:
        pass


def _apply_dnsmos_vram_fix(dnsmos_local) -> None:
    """给 dnsmos_local.ComputeScore 打显存修复补丁（幂等，固化到仓库）。

    dnsmos_local.py 属 third_party（被 gitignore，不随仓库提交），其 session
    默认无 gpu_mem_limit：onnxruntime 1.23 下 provider options 的 gpu_mem_limit
    被静默忽略，DNSMOS arena 初始预分配总显存 ~30%（24.8GB）、推理可涨到
    51.7GB，多任务叠加即 OOM。本补丁在 mos.py 侧注入：重建 session 时用
    SessionOptions + provider options 双保险传 gpu.mem_limit（主模型 16GB /
    p808 4GB，见 _DNSMOS_*_VRAM_LIMIT_MB 常量）+ arena_extend_strategy=kSameAsRequested，
    任何环境加载 DNSMOS
    都强制显存受限。已含修复的 dnsmos_local 会被重复硬化一次（幂等无害）。
    """
    if getattr(dnsmos_local.ComputeScore, '_vram_hardened', False):
        return
    _orig_init = dnsmos_local.ComputeScore.__init__

    def _hardened_init(self, primary_model_path, p808_model_path, use_gpu=False):
        _orig_init(self, primary_model_path, p808_model_path, use_gpu)
        if use_gpu and not getattr(self, '_cpu_ep_fallback_disabled', False):
            try:
                import onnxruntime as ort

                def _mk(path, limit_mb):
                    opts = ort.SessionOptions()
                    opts.add_session_config_entry(
                        "gpu.mem_limit", str(limit_mb * 1024 * 1024)
                    )
                    opts.add_session_config_entry(
                        "gpu.arena_extend_strategy", "kSameAsRequested"
                    )
                    opts.add_session_config_entry(
                        "session.disable_cpu_ep_fallback", "1"
                    )
                    return ort.InferenceSession(path, opts, providers=[
                        ("CUDAExecutionProvider", {
                            "arena_extend_strategy": "kSameAsRequested",
                            "gpu_mem_limit": limit_mb * 1024 * 1024,
                        }),
                    ])

                # 4 段/run（max_seg_per_run=4，shape 固定）峰值 ~2.8GB；主 session
                # arena 限 16GB：shape 固定后 arena 不累积（实测 3.7GB 稳定），
                # arena 大则 4 段推理的 0.7GB 中间 buffer 请求更易满足
                # （8 段/run 的 1.4GB 请求在 arena 碎片化时软 OOM，BFCArena
                # 只打 E 日志不抛异常，任务“看似完成但全程 OOM”）
                self.onnx_sess = _mk(primary_model_path, _DNSMOS_PRIMARY_VRAM_LIMIT_MB)
                self.p808_onnx_sess = _mk(p808_model_path, _DNSMOS_P808_VRAM_LIMIT_MB)
                self._vram_limited = True
                self._cpu_ep_fallback_disabled = True
            except Exception as exc:
                raise RuntimeError(
                    "DNSMOS GPU session hardening failed; CPU fallback is forbidden"
                ) from exc

    dnsmos_local.ComputeScore.__init__ = _hardened_init
    dnsmos_local.ComputeScore._vram_hardened = True


def _is_oom_error(exc: BaseException) -> bool:
    """判断异常是否为显存 OOM。

    覆盖：ONNXRuntime Conv 分配失败（BFCArena）、torch "CUDA out of memory"、
    CUDNN_STATUS_ALLOC_FAILED、"cudaMalloc returned ..." 等变体。
    依赖错误文本子串匹配：onnxruntime/torch 版本更新可能改变文案导致漏判
    （此时按普通错误处理，走 per-item fallback，不致命）。若新版本 OOM
    未触发预算减半，请在此补充新变体。
    """
    msg = str(exc).lower()
    return (
        'failed to allocate memory' in msg
        or 'out of memory' in msg
        or 'alloc_failed' in msg
        or 'cudamalloc' in msg
        # fail-closed 包装：_downgrade_to_cpu 把 OOM 包装成
        # "已禁用 CPU 降级...: GPU batch OOM"，消息含 " OOM"
        or ' oom' in msg
    )


def _default_batch_weight(item, fallback_sec: float) -> float:
    """默认单条批权重：时长（秒）；路径条目（时长未知）按 fallback_sec 计。"""
    d = _item_duration_sec(item)
    return d if d is not None else fallback_sec


def resolve_early_exit_threshold(cfg) -> float:
    """解析 Step2 早停阈值（统一实现，供 pipeline._step2_early_exit_threshold 与
    MOSStage 共用，避免 "dnsmos" 字符串解析逻辑多处漂移）。

    支持：数值直接返回；字符串 "dnsmos" → 生效 dnsmos 阈值（mos_thresholds 中）；
    未设置/非法 → mos_threshold（默认 3.0）。
    """
    dataset_name = getattr(cfg, "dataset_name", None)
    if hasattr(cfg, "get_effective_early_exit_threshold"):
        raw = cfg.get_effective_early_exit_threshold(dataset_name)
    else:
        raw = getattr(cfg, "step2_mos_early_exit_threshold", None)
    default = float(getattr(cfg, "mos_threshold", 3.0))
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip().lower() == "dnsmos":
        if hasattr(cfg, "get_effective_mos_thresholds"):
            thresholds = cfg.get_effective_mos_thresholds(dataset_name)
        else:
            thresholds = getattr(cfg, "mos_thresholds", None) or {}
        dnsmos_cfg = thresholds.get("dnsmos", default)
        if isinstance(dnsmos_cfg, dict):
            # dnsmos 为四维 dict 阈值时，早停取 OVRL 维度值（与筛选代表分一致），
            # 避免 float(dict) 报错
            dnsmos_cfg = dnsmos_cfg.get("ovrl", default)
        return float(dnsmos_cfg)
    return default


def _default_batch_budget(budget_sec: float, safe_total_sec: Dict[str, float]) -> float:
    """默认批预算（权重单位=秒）：等于时长预算（已由外层按显存上限 clamp）。"""
    return max(1.0, budget_sec)


# DNSMOS 主/p808 session 显存上限（MB）。
_DNSMOS_PRIMARY_VRAM_LIMIT_MB = 16 * 1024
_DNSMOS_P808_VRAM_LIMIT_MB = 4 * 1024

# 加载全部 MOS 模型（dnsmos/nisqa/utmos/scoreq）所需的最小可用显存（MB）：
# 模型权重（utmos 0.9 + nisqa 4.0 + dnsmos 4.0 + scoreq 6.1 ≈ 15GB）
# + torch/onnxruntime context（~2GB）+ 推理余量（~3GB）。
# Step2 启动时（模型加载前）探测，低于该值直接 fail-fast 报错，
# 避免多进程挤卡把卡占满后 OOM 风暴（加载后探测只见本进程剩余，救不了权重占满）。
_MOS_MODELS_NEED_MB = 15 * 1024

# 模型加载后、推理前所需的最小可用显存（MB）：DNSMOS 30s 音频（21 段）与其余
# 模型并行推理的实测峰值 ~7.3GB，留余量取 10GB。8 进程同时加载 4 模型时，
# 加载前探测（启动瞬间卡未满）无法捕获「加载后卡被占满」，此二次探测在加载
# 完成后立即执行，不足则 fail-fast 报错而非推理时 OOM 风暴。
_MOS_POST_LOAD_MIN_FREE_MB = 10 * 1024

def probe_gpu_free_mem_mb() -> Optional[Tuple[float, float]]:
    """探测当前进程可见 GPU 的可用/总显存（MB）。失败返回 None。

    优先 ``torch.cuda.mem_get_info``：遵守 CUDA_VISIBLE_DEVICES（多卡分片下
    返回映射后物理卡的驱动层可用显存，含其他进程占用，也覆盖 onnxruntime
    等非 torch 分配器的视角）；``nvidia-smi`` 不遵守 CUDA_VISIBLE_DEVICES
    （会测到别的卡），仅作无 torch 时的回退。

    注意：多进程挤卡时，各进程同时探测会看到相同的「总-已用」空闲值，探测
    结果只用于**收紧**预算（不放大），配合 OOM 预算减半兜底形成双层保护。
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 ** 2), total / (1024 ** 2)
    except Exception:
        pass
    try:
        import subprocess
        # nvidia-smi 不遵守 CUDA_VISIBLE_DEVICES（默认列全卡），多卡分片下会测到
        # 别的卡：解析环境变量第一张卡并加 -i 限定；环境变量未设置时保持原行为
        # （第一张物理卡）。
        cmd = ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        first_gpu = next((d.strip() for d in cvd.split(",") if d.strip()), None)
        if first_gpu:
            cmd = ["nvidia-smi", "-i", first_gpu] + cmd[1:]
        out = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        if lines:
            free, total = (float(x) for x in lines[0].split(","))
            return free, total
    except Exception:
        pass
    return None


@_close_uncached_estimators
def compute_mos_scores_batch(items: list, mos_types: List[str],
                             use_gpu: bool = False, use_cache: bool = True,
                             batch_size: int = 32,
                             mos_parallel_workers: int = 0,
                             model_specific_sr: bool = False,
                             overlap_resample: bool = False,
                             early_exit_below: Optional[float] = None,
                             early_exit_first_model_only: bool = False,
                             allow_fallback: bool = True,
                             urgent_options: Optional[Mapping[str, Any]] = None,
                             scoreq_options: Optional[Mapping[str, Any]] = None,
                             dnsmos_options: Optional[Mapping[str, Any]] = None,
                             nisqa_options: Optional[Mapping[str, Any]] = None,
                             utmos_options: Optional[Mapping[str, Any]] = None,
                             resample_method: str = "auto",
                             max_batch_seconds: float = 10.0,
                             safe_total_sec: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Batch MOS scoring with true GPU batching for UTMOS and per-item
    in-memory support. When mos_parallel_workers > 0, per-item estimators
    run in parallel (e.g. SCOREQ or NISQA when not using batch path).

    *items*: list of ``audio_path`` strings **或** ``(audio_array, sr)``
    tuples.  Mixing both types in a single call is supported.

    When model_specific_sr is True, each model receives audio at its target sr (48k for nisqa, 16k else).

    overlap_resample: 仅对非早停多模型路径生效。启用后，在模型 N inference 的同时，后台线程提前 resample
    模型 N+1 的输入，使 CPU resample 与 GPU/CPU inference 重叠。GPU 模式下收益更明显。

    early_exit_below: 按 mos_types 顺序逐模型推理；仅对「尚未因低于阈值而淘汰」的样本继续下一模型，
    未跑模型记为 0 分（保持 GPU 批处理：每轮只对存活子集算分）。
    early_exit_first_model_only: 为 True 时仅用第一个模型的分数做淘汰，后续模型只对存活样本算分、不再淘汰。
    max_batch_seconds: 切批的时长约束——批内条目总时长不超过 ``batch_size × max_batch_seconds``，
    长音频（whole-file）自动缩小批次以控制显存峰值；路径条目时长未知时按该值计。
    safe_total_sec: 按模型覆盖「批内总时长显存安全上限」；缺省用内置 ``_MOS_SAFE_TOTAL_SEC``。
    """
    if not items:
        return []

    # 防御：batch_size<=0 会让时长感知切批的 while 循环无法推进（config 层已有 max(1,...) 兜底，
    # 此处保护绕过 config 的直接调用方）
    batch_size = max(1, int(batch_size))

    logger.debug("compute_mos_scores_batch: %s 条, batch_size=%s, early_exit_below=%s", len(items), batch_size, early_exit_below)

    # 合并用户级安全上限覆盖（config mos_batch_total_seconds_cap）。
    safe_total_overrides = safe_total_sec or {}
    safe_total_sec = {
        name: spec.safe_total_seconds for name, spec in list_mos_specs().items()
    }
    safe_total_sec.update(safe_total_overrides)

    estimators = {}
    estimator_errors = {}
    for mos_type in mos_types:
        if mos_type == "urgent_pk" and urgent_options is not None:
            # estimator identity 依赖任务级 options；延迟到绑定锁内创建并完成推理。
            estimators[mos_type] = None
            continue
        if mos_type == "urgent_pk":
            binding = bind_urgent_pk_options(urgent_options)
        elif mos_type == "scoreq":
            binding = bind_scoreq_options(scoreq_options)
        elif mos_type == "dnsmos":
            binding = bind_dnsmos_options(dnsmos_options)
        elif mos_type == "nisqa":
            binding = bind_nisqa_options(nisqa_options)
        elif mos_type == "utmos":
            binding = bind_utmos_options(utmos_options)
        else:
            binding = contextlib.nullcontext()
        try:
            with binding:
                estimators[mos_type] = get_mos_estimator(mos_type, use_gpu=use_gpu, use_cache=use_cache, allow_fallback=allow_fallback)
        except Exception as exc:
            estimator_errors[mos_type] = _bounded_error_detail(exc)
            logger.warning("Failed to load %s: %s", mos_type, exc)

    results = [
        {'audio_path': it if isinstance(it, str) else '<in-memory>'}
        for it in items
    ]

    # 按目标采样率缓存重采样结果，避免多个 16k 模型（dnsmos/utmos/scoreq）重复重采样。
    # overlap_resample 后台 prepare 线程与主线程并发读写，用锁保护 get/set
    # （计算在锁外执行，锁内只做读写，保持轻量；重复计算无害，赋值幂等）。
    _sr_prepared_cache: Dict[int, list] = {}
    _sr_prepared_cache_lock = threading.Lock()

    def _prepare_cached(batch_items: list, target_sr: int) -> list:
        with _sr_prepared_cache_lock:
            if target_sr in _sr_prepared_cache:
                return _sr_prepared_cache[target_sr]
        prepared = _prepare_batch_items_for_model(batch_items, target_sr, resample_method)
        with _sr_prepared_cache_lock:
            _sr_prepared_cache[target_sr] = prepared
        return prepared

    def _compute_one(est, it, idx):
        try:
            if isinstance(it, str):
                return idx, est.compute(it)
            return idx, est.compute(audio_data=it)
        except Exception as exc:
            return idx, {'error': _bounded_error_detail(exc)}

    def _compute_batch_oom_split(estimator, batch):
        """调 estimator.compute_batch；显存 OOM 时递归减半重试。

        多模型共存一卡时 batch 工作区可能 OOM（ONNXRuntimeError Conv 分配失败 /
        torch CUDA out of memory），整批直接退化为逐条会拖垮吞吐；先减半重试，
        仅在最小批次仍失败时才抛给外层走 per-item fallback。
        """
        batch_scores = None
        try:
            batch_scores = estimator.compute_batch(batch)
        except Exception as e:
            if _is_oom_error(e) and len(batch) > 1:
                mid = len(batch) // 2
                logger.warning(
                    "compute_batch OOM（batch=%d），减半重试", len(batch)
                )
                return (_compute_batch_oom_split(estimator, batch[:mid])
                        + _compute_batch_oom_split(estimator, batch[mid:]))
            raise
        # 批结果长度不一致属于模型输出契约错误，逐条重试不能证明该 batch 有效。
        if len(batch_scores) != len(batch):
            raise MosResultContractError(
                f"compute_batch returned {len(batch_scores)} scores for {len(batch)} items"
            )
        return batch_scores

    def _run_one_model_on_subset_impl(mos_type: str, estimator, act_list: List[int], batch_input: list):
        """Writes results[gi][mos_type] for each global index gi in act_list."""
        n_sub = len(act_list)
        if n_sub == 0:
            return
        has_batch = hasattr(estimator, 'compute_batch')
        if has_batch and (use_gpu or mos_type in ('dnsmos',)):
            # 条数 + 权重双重约束切批（权重由 estimator 声明，见 batch_weight）：
            # DNSMOS 按 run_batch_audio 的 segment 数（9.01s 窗 / 1s hop / 短音频
            # 补长，显存 ∝ segment，64s 秒预算在长音频下放大数倍顶爆显存），
            # 其余模型按时长（秒）；显存上限由 mos_batch_total_seconds_cap /
            # 内置 _MOS_SAFE_TOTAL_SEC 决定。路径条目（时长未知）按
            # max_batch_seconds 计（退化为纯条数约束，行为与旧版一致）。
            budget_sec = batch_size * max_batch_seconds
            budget_sec = max(1.0, min(budget_sec, safe_total_sec.get(mos_type, 300.0)))
            batch_weight = getattr(estimator, 'batch_weight', _default_batch_weight)
            batch_budget = getattr(estimator, 'batch_budget', _default_batch_budget)
            # 条数上限按显存比例自动放大（P-2）：以「批内显存安全上限最小」的模型
            # （显存成本最高，通常 DNSMOS）为基线，其他模型按 cap 比例同倍放大——
            # 轻量模型（cap 大）条数上限自动更大，无需逐模型手动配置。批内总时长
            # 仍由 batch_budget（cap 秒数）封顶，条数上限只是防权重过小时批无限大。
            model_cap = safe_total_sec.get(mos_type, 300.0)
            # 仅将当前调用实际启用的模型纳入批条数比例基线；registry 中未启用的
            # URGENT-PK 等保守 cap 不应悄然改变 DNSMOS/UTMOS 等既有调度行为。
            active_caps = [safe_total_sec.get(name, 300.0) for name in mos_types]
            min_cap = min(active_caps) if active_caps else model_cap
            max_items = max(1, int(math.ceil(batch_size * model_cap / min_cap)))
            # WeightedBatchSplitter.budget 可变：OOM 时下调，后续批次自动收缩（显存自适应下限）。
            splitter = WeightedBatchSplitter(
                batch_input,
                lambda it: batch_weight(it, max_batch_seconds),
                batch_budget(budget_sec, safe_total_sec),
                max_items=max_items,
            )
            i = 0
            for batch in splitter:
                i_end = i + len(batch)
                try:
                    batch_scores = _compute_batch_oom_split(estimator, batch)
                    for k, score in enumerate(batch_scores):
                        gi = act_list[i + k]
                        results[gi][mos_type] = score
                except MosResultContractError:
                    raise
                except Exception as e:
                    if _is_oom_error(e):
                        _log_gpu_contention_diagnostic()
                        # OOM：批预算减半（显存自适应下限），后续批次用更小预算；
                        # 当前批次走 per-item fallback（_compute_batch_oom_split
                        # 已先做条数减半重试，到此处说明最小批次仍 OOM）
                        splitter.budget = max(1.0, splitter.budget / 2)
                        logger.warning(
                            f"{mos_type} batch OOM，批预算下调至 {splitter.budget:.1f}"
                            f"（权重），当前批次走 per-item fallback"
                        )
                    else:
                        logger.warning(f"{mos_type} batch failed, falling back to per-item: {e}")
                    for k, item in enumerate(batch):
                        gi = act_list[i + k]
                        try:
                            if isinstance(item, str):
                                results[gi][mos_type] = estimator.compute(item)
                            else:
                                results[gi][mos_type] = estimator.compute(audio_data=item)
                        except Exception as item_error:
                            results[gi][mos_type] = {
                                'error': _bounded_error_detail(item_error),
                            }
                i = i_end
        else:
            nw = min(mos_parallel_workers, n_sub) if mos_parallel_workers and n_sub > 1 else 0
            if nw >= 2:
                with ThreadPoolExecutor(max_workers=min(nw, n_sub)) as ex:
                    futs = [
                        ex.submit(_compute_one, estimator, batch_input[j], j)
                        for j in range(n_sub)
                    ]
                    for fut in as_completed(futs):
                        j, res = fut.result()
                        results[act_list[j]][mos_type] = res
            else:
                for j in range(n_sub):
                    _, res = _compute_one(estimator, batch_input[j], j)
                    results[act_list[j]][mos_type] = res

    def _run_one_model_on_subset(mos_type: str, estimator, act_list: List[int], batch_input: list):
        if mos_type != "urgent_pk" or urgent_options is None:
            return _run_one_model_on_subset_impl(mos_type, estimator, act_list, batch_input)
        with bind_urgent_pk_options(urgent_options):
            bound_estimator = get_mos_estimator(
                mos_type, use_gpu=use_gpu, use_cache=use_cache,
                allow_fallback=allow_fallback,
            )
            return _run_one_model_on_subset_impl(
                mos_type, bound_estimator, act_list, batch_input,
            )

    th = float(early_exit_below) if early_exit_below is not None else None

    if th is not None:
        active = set(range(len(items)))
        # P-3：早停路径按模型速度排序执行——快的模型（scoreq 235 条/s）先跑并
        # 淘汰低分样本，慢的模型（nisqa 25、dnsmos 105 条/s）只处理存活子集，
        # 整体加速。初筛模型随之从 dnsmos 变为最快的 scoreq（阈值统一 3.0）。
        ordered = sorted(
            mos_types,
            key=lambda mt: -_mos_speed_per_second(mt),
        )
        first_model = ordered[0] if ordered else None
        # per-item 重采样缓存：{(target_sr): {id(item): (audio, sr)}}。
        # 子集随模型淘汰变化，不能走 _sr_prepared_cache（按 sr 整批缓存会串味）；
        # 但同一 item 的同一目标采样率重采样结果可跨模型复用（scoreq/utmos/dnsmos
        # 同为 16k 时只重采样一次）。items 在本次调用内全程存活，id 不会被复用；
        # 缓存随调用结束释放，峰值内存与单模型 prepare 同量级。
        item_prep_cache: Dict[int, Dict[int, tuple]] = {}
        for mos_type in ordered:
            estimator = estimators.get(mos_type)
            if not active or (estimator is None and not (mos_type == "urgent_pk" and urgent_options is not None)):
                continue
            act_list = sorted(active)
            if model_specific_sr:
                target_sr = get_model_target_sr(mos_type)
                # 先按存活子集切片再重采样：避免 48k（nisqa）等重采样浪费在已淘汰
                # 样本上；同一 item 的重采样经 item_prep_cache 跨模型复用。
                act_items = [items[gi] for gi in act_list]
                prep_cache = item_prep_cache.setdefault(target_sr, {})
                batch_input = []
                for it in act_items:
                    prepared = prep_cache.get(id(it))
                    if prepared is None:
                        prepared = _prepare_batch_items_for_model([it], target_sr, resample_method)[0]
                        prep_cache[id(it)] = prepared
                    batch_input.append(prepared)
            else:
                batch_input = [items[gi] for gi in act_list]
            _run_one_model_on_subset(mos_type, estimator, act_list, batch_input)
            # 仅第一个模型做淘汰时，后续模型不再淘汰
            if early_exit_first_model_only and mos_type != first_model:
                continue
            new_active = set()
            for gi in act_list:
                res = results[gi].get(mos_type)
                if res is None:
                    new_active.add(gi)
                    continue
                if isinstance(res, dict) and res.get('error'):
                    new_active.add(gi)
                    continue
                s = extract_mos_scalar(mos_type, res)
                if s is not None and s < th:
                    continue
                new_active.add(gi)
            active = new_active
        for gi in range(len(items)):
            had_early_exit = False
            for mt in mos_types:
                if mt in results[gi]:
                    continue
                if mt not in estimators:
                    # 模型加载失败不是早停：保留有界根因且不计入早停统计。
                    results[gi][mt] = {
                        'error': estimator_errors.get(mt, 'mos_estimator_unavailable'),
                    }
                    continue
                results[gi][mt] = _early_exit_zero_score(mt)
                had_early_exit = True
            if had_early_exit:
                results[gi]['_early_exit'] = True
        return results

    # P-3：统一按模型实测速度排序执行（与早停路径一致）：scoreq 先跑，
    # nisqa 最后。非早停时结果相同，仅执行顺序与 scores 键顺序按速度序。
    estimator_items = sorted(
        estimators.items(),
        key=lambda kv: -_mos_speed_per_second(kv[0]),
    )
    # CPU 下 resample 与 inference 共享 CPU，重叠反而因抢占导致严重退化；
    # 仅在 GPU 模式下启用 overlap，让 CPU resample 与 GPU inference 真正重叠。
    do_overlap = overlap_resample and use_gpu and model_specific_sr and len(estimator_items) > 1
    prepare_executor: Optional[ThreadPoolExecutor] = None
    future_input = None
    future_mos_type = None

    primary = None
    try:
        for model_idx, (mos_type, estimator) in enumerate(estimator_items):
            if model_specific_sr:
                if future_input is not None and future_mos_type == mos_type:
                    batch_input = future_input.result()
                    future_input = None
                    future_mos_type = None
                else:
                    batch_input = _prepare_cached(items, get_model_target_sr(mos_type))
            else:
                batch_input = items

            # 启动下一个模型的输入 resample，与当前模型 inference 重叠
            next_idx = model_idx + 1
            if do_overlap and next_idx < len(estimator_items):
                next_mos_type, _ = estimator_items[next_idx]
                next_sr = get_model_target_sr(next_mos_type)
                if prepare_executor is None:
                    prepare_executor = ThreadPoolExecutor(max_workers=1)
                future_input = prepare_executor.submit(
                    _prepare_cached,
                    items,
                    next_sr,
                )
                future_mos_type = next_mos_type

            # 与早停路径共用同一实现（时长感知切批 + OOM 减半重试 + per-item fallback），
            # 避免两处循环逻辑漂移（OOM 修复曾只覆盖一处的教训）
            _run_one_model_on_subset(
                mos_type, estimator, list(range(len(batch_input))), batch_input
            )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if prepare_executor is not None:
            try:
                # wait=True：异常路径也等待预取结束，禁止白跑或悬挂线程。
                prepare_executor.shutdown(wait=True)
            except BaseException as cleanup:
                if primary is None:
                    raise
                _add_exception_note(
                    primary, f"overlap resample executor shutdown failed: {cleanup}",
                )

    # 模型加载失败的 mos_type 不在 estimator_items 中，results 会缺该 key；
    # 与早停路径一致补有界根因，避免下游静默缺模型分数。
    for gi in range(len(items)):
        for mt in mos_types:
            if mt not in results[gi]:
                results[gi][mt] = {
                    'error': estimator_errors.get(mt, 'mos_estimator_unavailable'),
                }

    return results
