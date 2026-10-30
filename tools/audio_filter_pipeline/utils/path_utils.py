"""音频筛选流程的路径规划与不可变 artifact 目录工具。

production 写入统一采用 generation-3 的 ``<output_root>/stepN/<dataset>`` 布局
（见 ``v3_artifact_dir``）；legacy、sr-first 与 stage-v2 相关 helper 已移除。
"""

import ctypes
import errno
import hashlib
import os
import re
import secrets
import stat
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import quote

# 支持的音频扩展名（Step1/2/3 共用，统一维护避免多处漂移）
AUDIO_EXTENSIONS = frozenset({'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'})

# v3 不可变 artifact 目录使用的生产布局版本。
PRODUCTION_LAYOUT_VERSION = 3


def expand_audio_files(paths, recursive: bool = True) -> list:
    """将输入路径（文件/目录列表）展开为音频文件列表。

    返回值排序并去重：多卡按序号切分分片时，无序/重复列表会导致分片不稳定或重复处理。
    供 main.Step1 分片与 pipeline.Step2 展开共用（原两处重复实现合并至此）。
    """
    from pathlib import Path
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            if Path(p).suffix.lower() in AUDIO_EXTENSIONS:
                out.append(p)
        elif os.path.isdir(p):
            root = Path(p)
            it = root.rglob('*') if recursive else root.glob('*')
            for f in it:
                if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
                    out.append(str(f))
    return sorted(set(out))


# 文件/目录名允许字符：仅 ASCII 字母、数字、下划线、连字符、点号。
# 中文、空格及其他特殊字符一律替换为下划线，保证跨平台/跨工具链可读。
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_path_component(name: str, fallback: str = "x") -> str:
    """清洗单个路径组件（文件名或单层目录名），去除中文、空格、特殊字符。

    非法字符替换为下划线并合并连续下划线；首尾的下划线/点号被剥离。
    结果为空时回退为 ``fallback``。
    """
    if not name:
        return fallback
    cleaned = _SAFE_NAME_PATTERN.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_.") or fallback


def sanitize_relative_path(relative_path: str) -> str:
    """清洗相对路径的每一层目录组件，保持层级结构不变。"""
    if not relative_path:
        return ""
    # 统一按 / 与 os.sep 切分，兼容 manifest 中跨平台写入的路径
    parts = re.split(r"[/\\]+", str(relative_path))
    cleaned = [sanitize_path_component(p) for p in parts if p and p not in (".", "..")]
    return os.sep.join(cleaned)


# source_key 截断长度（hex 字符数）：64 bit 足够在千万级来源上保持碰撞概率可忽略
_SOURCE_KEY_HEX_LEN = 16


@dataclass(frozen=True)
class SourceIdentity:
    """稳定的输入来源身份标识。

    - ``dataset_id``: 数据集标识（Task 5 起由配置显式指定；此前取 ``dataset_name``
      或输入根目录名作为占位）。
    - ``language``: 语言码；未标注时为空字符串。
    - ``original_relative_path``: 源文件在 ``input_root`` 下的完整原始父目录，
      保留 UTF-8 原义（不做 sanitize），分隔符统一为正斜杠以保证跨平台可复现。
      用于输出树（Task 2 OutputPathPlan）。
    - ``source_relative_path``: 源文件相对 ``input_root`` 的完整路径（父目录 +
      原始文件名，UTF-8 原义、正斜杠归一化）；source_key 的唯一性 key material。
      单文件输入（file-as-root）时取文件绝对路径（去前导分隔符），保证同
      dataset_id 下不同位置的同名文件不碰撞。
    - ``source_key``: 由 ``dataset_id + NUL + input_root + NUL +
      source_relative_path`` 的 UTF-8 bytes 派生的截断 SHA-256 hex；唯一标识
      source file（不同输入 root 共享 dataset_id 时也互不碰撞），同源不同
      chunk 共享同一 key（key 不含 chunk_index），同一来源在任何机器/任何
      轮次都稳定一致。
    - ``input_root``: canonical absolute 输入根（key material 的 origin
      discriminator；Task 5 registry 落地后由 DatasetSpec 提供）。
    """
    dataset_id: str
    language: str
    original_relative_path: str
    source_relative_path: str
    source_key: str
    input_root: str = ""


def make_source_identity(
    input_root: str,
    source_path: str,
    dataset_id: str,
    language: str = "",
) -> SourceIdentity:
    """构造 ``source_path`` 相对 ``input_root`` 的 SourceIdentity。

    ``original_relative_path`` 为源文件所在父目录相对 input_root 的完整路径
    （源文件直接在根目录时为 ``""``）；``source_relative_path`` 为其加原始
    文件名。source 不在 input_root 下时抛出 ``ValueError``，绝不回退到
    basename——身份必须可追溯到真实来源。

    单文件输入约定：``source_path == input_root``（文件自身即身份根）时，
    ``original_relative_path`` 为空字符串，``source_relative_path`` 取源文件
    绝对路径（去前导分隔符），使 ``/a/same.wav`` 与 ``/b/same.wav`` 在相同
    dataset_id 下得到不同 source_key。

    ``source_key`` 的 key material 额外纳入 canonical absolute ``input_root``
    （``dataset_id + NUL + input_root + NUL + source_relative_path``）：两个
    独立输入 root 下的同名同 dataset 来源不会碰撞。
    """
    root = os.path.abspath(input_root)
    src = os.path.abspath(source_path)
    real_root = os.path.realpath(root)
    real_src = os.path.realpath(src)
    try:
        real_contained = os.path.commonpath((real_root, real_src)) == real_root
    except (OSError, ValueError):
        real_contained = False
    if real_src != real_root and not real_contained:
        raise ValueError(
            f"source_path realpath 不在 input_root 下（可能是越界 symlink）: "
            f"{source_path!r} (root={input_root!r})"
        )
    parent = os.path.dirname(src)
    try:
        rel = os.path.relpath(parent, root)
    except ValueError:
        # 跨盘符（Windows）等无法计算相对路径的情况
        rel = None
    if rel is None or rel == os.pardir or rel.startswith(os.pardir + os.sep):
        if src == root:
            # 单文件输入：source 即 input root，身份域为该文件自身
            original_rel = ""
            source_rel = src.replace(os.sep, "/").lstrip("/")
        else:
            raise ValueError(
                f"source_path 不在 input_root 下: {source_path!r} (root={input_root!r})"
            )
    else:
        # 统一正斜杠：Windows 反斜杠与 Linux 正斜杠得到同一 identity/key
        original_rel = "" if rel == os.curdir else rel.replace(os.sep, "/")
        source_rel = "/".join(filter(None, [original_rel, os.path.basename(src)]))
    key_material = f"{dataset_id}\0{root}\0{source_rel}".encode("utf-8")
    source_key = hashlib.sha256(key_material).hexdigest()[:_SOURCE_KEY_HEX_LEN]
    return SourceIdentity(
        dataset_id=dataset_id or "",
        language=language or "",
        original_relative_path=original_rel,
        source_relative_path=source_rel,
        source_key=source_key,
        input_root=root,
    )


def canonical_path(path: str) -> str:
    """返回不跟随最终文件语义的 canonical absolute path。"""
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise ValueError(f"path must be path-like, got {path!r}") from exc
    if type(value) is not str or not value.strip():
        raise ValueError("path must be a nonblank path-like string")
    return os.path.realpath(os.path.abspath(value))


def roots_overlap(first: str, second: str) -> bool:
    """按真实路径判断两个根是否相同或互为父子目录。"""
    first_real = canonical_path(first)
    second_real = canonical_path(second)
    try:
        common = os.path.commonpath((first_real, second_real))
    except ValueError:
        return False
    return common == first_real or common == second_real


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_OPEN_TARGET_FLAGS = _FILE_FLAGS | getattr(os, "O_NONBLOCK", 0)


def _absolute_path(value, field: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be a path-like string") from exc
    if type(path) is not str or not path.strip():
        raise ValueError(f"{field} must be a nonblank path-like string")
    return os.path.abspath(path)


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _directory_component_error(
    parent_fd: int, name: str, exc: OSError,
) -> ValueError:
    # 必须带上底层 errno：EMFILE/ENOTDIR/EBADF/EIO 等真实原因此前被统一文案
    # 吞掉，排障时无法区分资源耗尽、fd 竞争与文件系统故障。
    detail = (
        f"open failed: [{errno.errorcode.get(exc.errno, exc.errno)}] "
        f"{exc.strerror or exc}"
    )
    resource_errors = {
        errno.EMFILE,
        errno.ENFILE,
        errno.EBADF,
        errno.ENOMEM,
        errno.EIO,
    }
    if exc.errno in resource_errors:
        return ValueError(
            f"safe directory component open failed: {name!r} ({detail})"
        )
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as stat_exc:
        entry = None
        detail += (
            f"; stat failed: [{errno.errorcode.get(stat_exc.errno, stat_exc.errno)}] "
            f"{stat_exc.strerror or stat_exc}"
        )
    if entry is not None and stat.S_ISLNK(entry.st_mode):
        return ValueError(
            f"safe directory component contains symlink: {name!r} ({detail})"
        )
    return ValueError(
        f"safe directory component is not a real directory: {name!r} ({detail})"
    )


def _open_directory_child(parent_fd: int, name: str, *, create: bool) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        try:
            return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise _directory_component_error(parent_fd, name, exc) from exc
    except OSError as exc:
        raise _directory_component_error(parent_fd, name, exc) from exc


def _open_directory_chain(path: str, *, create: bool) -> tuple[list[int], list[str]]:
    absolute = _absolute_path(path, "directory path")
    drive, tail = os.path.splitdrive(absolute)
    root_name = drive + os.sep if drive else os.sep
    descriptors = [os.open(root_name, _DIRECTORY_FLAGS)]
    names: list[str] = []
    try:
        for component in (part for part in tail.split(os.sep) if part):
            child = _open_directory_child(descriptors[-1], component, create=create)
            names.append(component)
            descriptors.append(child)
        return descriptors, names
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def open_safe_directory(path: str, *, create: bool = False) -> int:
    """逐层以 ``O_DIRECTORY|O_NOFOLLOW`` 打开目录，返回调用方所有的 fd。

    ``create=True`` 只创建缺失的目录层；每层创建后都会再次以 descriptor
    打开，因此竞态替换为 symlink、普通文件或特殊文件都会 fail-closed。
    """
    descriptors, _names = _open_directory_chain(path, create=create)
    for descriptor in descriptors[:-1]:
        os.close(descriptor)
    return descriptors[-1]


def _validate_existing_directory_prefix(path: str) -> None:
    """只打开已存在的目录前缀，拒绝其中任意 symlink 或非目录。"""
    absolute = _absolute_path(path, "directory path")
    drive, tail = os.path.splitdrive(absolute)
    root_name = drive + os.sep if drive else os.sep
    descriptors = [os.open(root_name, _DIRECTORY_FLAGS)]
    try:
        for component in (part for part in tail.split(os.sep) if part):
            try:
                child = _open_directory_child(
                    descriptors[-1], component, create=False,
                )
            except FileNotFoundError:
                break
            descriptors.append(child)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


class SafeOutputPath:
    """绑定到已打开目录链的输出目标。

    ``parent_fd`` 与 ``target_name`` 是供发布方使用的核心契约：后续
    ``stat/open/link/unlink`` 应使用对应的 ``dir_fd``，而不是重新拼接绝对路径。
    对象由 ``open_safe_output_path`` 创建，必须显式关闭或用作上下文管理器。
    """

    def __init__(
        self,
        root: str,
        target: str,
        descriptors: list[int],
        names: list[str],
        root_index: int,
    ) -> None:
        self.root = root
        self.target = target
        self._descriptors = descriptors
        self._names = names
        self._root_index = root_index
        self.target_name = os.path.basename(target)
        self.canonical_target = os.path.realpath(target)

    def _ensure_open(self) -> None:
        if not self._descriptors:
            raise ValueError("safe output path descriptor is closed")

    @property
    def root_fd(self) -> int:
        self._ensure_open()
        return self._descriptors[self._root_index]

    @property
    def parent_fd(self) -> int:
        self._ensure_open()
        return self._descriptors[-1]

    def verify(self) -> None:
        """复验目录链的 name-to-inode 绑定，发现 rename 替换即拒绝。"""
        self._ensure_open()
        for parent_fd, name, child_fd in zip(
            self._descriptors, self._names, self._descriptors[1:], strict=False,
        ):
            try:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise ValueError(f"safe output directory changed: {name!r}") from exc
            if not stat.S_ISDIR(named.st_mode) or not _same_inode(named, os.fstat(child_fd)):
                raise ValueError(f"safe output directory changed: {name!r}")

    def target_stat(self) -> os.stat_result | None:
        self.verify()
        try:
            value = os.stat(
                self.target_name, dir_fd=self.parent_fd, follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(value.st_mode):
            raise ValueError(f"safe output target must not be a symlink: {self.target}")
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"safe output target has unsupported type: {self.target}")
        return value

    def open_target(self, flags: int, mode: int = 0o600) -> int:
        self.verify()
        try:
            descriptor = os.open(
                self.target_name, flags | _OPEN_TARGET_FLAGS,
                mode, dir_fd=self.parent_fd,
            )
        except OSError as exc:
            raise ValueError(f"cannot safely open output target: {self.target}") from exc
        try:
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                raise ValueError(f"safe output target has unsupported type: {self.target}")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def create_temp_file(
        self, prefix: str, suffix: str = "", mode: int = 0o600,
    ) -> tuple[int, str]:
        """在绑定的父目录中以 ``O_EXCL`` 创建临时普通文件。"""
        if (
            type(prefix) is not str or not prefix or "/" in prefix or "\\" in prefix
            or "\x00" in prefix or type(suffix) is not str
            or "/" in suffix or "\\" in suffix or "\x00" in suffix
        ):
            raise ValueError("temporary file prefix and suffix must be basename fragments")
        self.verify()
        for _ in range(128):
            name = f"{prefix}{secrets.token_hex(16)}{suffix}"
            try:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_FLAGS,
                    mode,
                    dir_fd=self.parent_fd,
                )
            except FileExistsError:
                continue
            except OSError as exc:
                raise ValueError(f"cannot safely create temporary output: {self.target}") from exc
            value = os.fstat(descriptor)
            if not stat.S_ISREG(value.st_mode):
                os.close(descriptor)
                raise ValueError("created temporary output is not a regular file")
            return descriptor, name
        raise OSError("cannot allocate unique temporary output")

    def link_from(self, source_fd: int, source_name: str) -> None:
        """从固定源目录 fd 链接到目标目录 fd，目标绝不覆盖。"""
        if (
            type(source_name) is not str
            or not source_name
            or source_name in {".", ".."}
            or "/" in source_name
            or "\\" in source_name
            or "\x00" in source_name
        ):
            raise ValueError("source_name must be a single basename")
        self.verify()
        os.link(
            source_name,
            self.target_name,
            src_dir_fd=source_fd,
            dst_dir_fd=self.parent_fd,
            follow_symlinks=False,
        )

    def rename_from_noreplace(self, source: "SafeOutputPath") -> None:
        """将同一安全根内的已绑定临时文件无覆盖重命名为当前目标。"""
        if not isinstance(source, SafeOutputPath):
            raise TypeError("source must be a SafeOutputPath")
        source.verify()
        self.verify()
        _rename_noreplace(
            source.parent_fd,
            source.target_name,
            self.parent_fd,
            self.target_name,
        )

    def unlink_target(self, expected_identity: tuple[int, int] | None = None) -> bool:
        """按绑定父目录隔离并删除目标；竞争方 entry 永不覆盖或删除。"""
        self.verify()
        value = self.target_stat()
        if value is None:
            return True
        identity = (value.st_dev, value.st_ino)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(f"safe output target changed: {self.target}")
        if not _quarantine_remove_entry(
            self.parent_fd,
            self.target_name,
            self.target,
            identity,
        ):
            raise ValueError(f"safe output target changed during unlink: {self.target}")
        self.verify()
        if self.target_stat() is not None:
            raise ValueError(f"safe output target was replaced during unlink: {self.target}")
        return True

    def remove_empty_parent(self) -> bool:
        """安全移除当前目标所在的空目录；竞争替换时保留现场。"""
        self.verify()
        if not self._names:
            return False
        directory_fd = self.parent_fd
        directory_stat = os.fstat(directory_fd)
        identity = (directory_stat.st_dev, directory_stat.st_ino)
        if not stat.S_ISDIR(directory_stat.st_mode) or os.listdir(directory_fd):
            return False
        outer_fd = self._descriptors[-2]
        directory_name = self._names[-1]
        named = os.stat(directory_name, dir_fd=outer_fd, follow_symlinks=False)
        if not stat.S_ISDIR(named.st_mode) or _same_inode(named, directory_stat) is False:
            return False
        tomb_name = _claim_cleanup_entry(outer_fd, directory_name)
        if tomb_name is None:
            return False
        tomb_fd = None
        try:
            tomb_fd = _open_cleanup_entry_fd(outer_fd, tomb_name, self.target)
            tomb_stat = os.fstat(tomb_fd)
            if (
                not stat.S_ISDIR(tomb_stat.st_mode)
                or (tomb_stat.st_dev, tomb_stat.st_ino) != identity
                or not _cleanup_entry_is_owned(
                    outer_fd, tomb_name, tomb_fd, identity, self.target,
                )
                or os.listdir(tomb_fd)
            ):
                return False
            os.rmdir(tomb_name, dir_fd=outer_fd)
            os.fsync(outer_fd)
            return True
        finally:
            if tomb_fd is not None:
                os.close(tomb_fd)

    def fsync_parent(self) -> None:
        self.verify()
        os.fsync(self.parent_fd)

    def close(self) -> None:
        descriptors = self._descriptors
        self._descriptors = []
        first_error = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as exc:
                # 单个 fd 关闭失败不能跳过其余 descriptor，首个错误仍需返回。
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def open_safe_output_path(
    output_root: str, target_path: str, *, create: bool = False,
) -> SafeOutputPath:
    """打开 output root 与 target parent 的真实目录链并返回 descriptor-bound 目标。"""
    root = _absolute_path(output_root, "output_root")
    target = _absolute_path(target_path, "target_path")
    try:
        if os.path.commonpath((root, target)) != root:
            raise ValueError(f"output target escapes output_root: {target!r}")
    except ValueError as exc:
        raise ValueError(f"output target escapes output_root: {target!r}") from exc
    parent = os.path.dirname(target)
    relative_parent = os.path.relpath(parent, root)
    if relative_parent == os.pardir or relative_parent.startswith(os.pardir + os.sep):
        raise ValueError(f"output target escapes output_root: {target!r}")
    descriptors, names = _open_directory_chain(root, create=create)
    try:
        root_index = len(descriptors) - 1
        if relative_parent != os.curdir:
            for component in relative_parent.split(os.sep):
                child = _open_directory_child(
                    descriptors[-1], component, create=create,
                )
                descriptors.append(child)
                names.append(component)
        safe = SafeOutputPath(root, target, descriptors, names, root_index)
        safe.target_stat()
        return safe
    except BaseException:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _validate_output_target_strings(output_root: str, target_path: str) -> tuple[str, str]:
    root = _absolute_path(output_root, "output_root")
    target = _absolute_path(target_path, "target_path")
    try:
        contained = os.path.commonpath((root, target)) == root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"output target escapes output_root: {target!r}")
    return root, target


def _validate_existing_output_entry(path: str, value: os.stat_result) -> None:
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f"output path contains symlink: {path!r}")
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"output target has unsupported type: {path!r}")


def validate_safe_output_path(output_root: str, target_path: str) -> str:
    """只校验输出路径，不创建目录；拒绝中间目录和非普通目标。"""
    root, target = _validate_output_target_strings(output_root, target_path)
    # 预检不创建目录，但必须逐层打开所有已经存在的前缀；否则
    # ``root/link/missing/file`` 这类深层 symlink 会在字符串 realpath 预检中漏过。
    _validate_existing_directory_prefix(root)
    _validate_existing_directory_prefix(os.path.dirname(target))
    canonical_root = os.path.realpath(root)
    canonical_target = os.path.realpath(target)
    try:
        contained = os.path.commonpath((canonical_root, canonical_target)) == canonical_root
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"canonical output target escapes output_root: {target!r}")
    if os.path.lexists(target):
        value = os.lstat(target)
        _validate_existing_output_entry(target, value)
    return canonical_target


def prepare_safe_output_path(output_root: str, target_path: str) -> str:
    """逐层以 descriptor-bound 方式创建真实目录并返回 canonical target。"""
    root, target = _validate_output_target_strings(output_root, target_path)
    with open_safe_output_path(root, target, create=True) as safe:
        return safe.canonical_target


def _encoded_component(value: str, field: str) -> str:
    """按 path-component layout encoding v2 校验并 percent-encode 单层组件。

    此处 ``v2`` 是路径布局的编码版本，不是 pipeline generation。
    """
    value = str(value or "")
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"invalid {field}: {value!r}")
    return quote(value, safe="-._")


def _encoded_relative_tree(relative_path: str) -> list:
    """编码原始相对目录树；空树代表输入根目录，拒绝穿越组件。"""
    value = str(relative_path or "")
    if not value:
        return []
    parts = re.split(r"[/\\\\]+", value)
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"invalid original_relative_path: {relative_path!r}")
    return [_encoded_component(part, "original_relative_path component") for part in parts]


def v3_artifact_dir(output_root, step, dataset_id) -> str:
    """纯计算 v3 artifact 目录，不访问或创建文件系统。"""
    step_value = getattr(step, "value", step)
    step_value = str(step_value or "")
    if step_value not in {f"step{number}" for number in range(7)}:
        raise ValueError(f"invalid step: {step!r}")
    if type(dataset_id) is not str or not dataset_id.strip():
        raise ValueError(
            f"invalid dataset_id: expected non-empty string, got {dataset_id!r}"
        )
    if dataset_id in (".", "..") or "/" in dataset_id or "\\" in dataset_id:
        raise ValueError(f"invalid dataset_id: {dataset_id!r}")
    encoded_dataset = quote(dataset_id, safe="-._")
    # 无中间 artifacts 层：output_root 下直接是 step -> dataset。
    return os.path.join(
        os.fspath(output_root), step_value, encoded_dataset,
    )


def source_identity_from_metadata(metadata: Mapping) -> SourceIdentity:
    """从 item metadata 重建 identity；缺项或非法值一律抛 ValueError。"""
    try:
        identity = SourceIdentity(
            dataset_id=metadata["dataset_id"],
            language=metadata["language"],
            original_relative_path=metadata["original_relative_path"],
            source_relative_path=metadata["source_relative_path"],
            source_key=metadata["source_key"],
            input_root=metadata.get("input_root", ""),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("missing source identity") from exc
    _encoded_component(identity.language, "language")
    _encoded_component(identity.dataset_id, "dataset_id")
    _encoded_component(identity.source_key, "source_key")
    _encoded_relative_tree(identity.original_relative_path)
    if not identity.source_relative_path:
        raise ValueError("missing source_relative_path")
    return identity


_DEFAULT_TEMP_PREFIXES = ("audio-filter-",)
_TEMP_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _cleanup_prefixes(prefixes: Optional[Iterable[str]]) -> tuple[str, ...]:
    """校验只允许匹配 basename 的临时 artifact prefixes。"""
    if prefixes is None:
        return _DEFAULT_TEMP_PREFIXES
    if isinstance(prefixes, (str, bytes)) or not isinstance(prefixes, Iterable):
        raise ValueError("prefixes must be a non-empty iterable of strings")
    values = tuple(prefixes)
    if not values:
        raise ValueError("prefixes must be a non-empty iterable of strings")
    for prefix in values:
        if (
            type(prefix) is not str
            or not prefix
            or prefix != prefix.strip()
            or prefix in {".", ".."}
            or "/" in prefix
            or "\\" in prefix
            or "\x00" in prefix
        ):
            raise ValueError(
                "each prefix must be a nonblank basename prefix without path separators"
            )
    if len(set(values)) != len(values):
        raise ValueError("prefixes must not contain duplicates")
    return values


def _cleanup_path_string(path: str) -> str:
    """校验清理根路径，但不把不存在路径解析到系统临时目录。"""
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise ValueError(f"temp_dir must be a path-like string, got {path!r}") from exc
    if type(value) is not str or not value.strip():
        raise ValueError("temp_dir must be a nonblank path-like string")
    return os.path.abspath(value)


def _cleanup_entry_stat(parent_fd: int, name: str, label: str) -> os.stat_result:
    try:
        value = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot inspect temporary artifact {label!r}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f"temporary artifact must not be a symlink: {label!r}")
    if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
        raise ValueError(f"temporary artifact has unsupported type: {label!r}")
    return value


_CLEANUP_FILE_FLAGS = (
    getattr(os, "O_PATH", os.O_RDONLY)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _cleanup_stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_cleanup_entry_fd(parent_fd: int, name: str, label: str) -> int:
    """打开并持有待清理 entry 的固定 fd，拒绝 symlink 与特殊类型。"""
    expected = _cleanup_entry_stat(parent_fd, name, label)
    flags = _TEMP_DIRECTORY_FLAGS if stat.S_ISDIR(expected.st_mode) else _CLEANUP_FILE_FLAGS
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"cannot safely open temporary artifact {label!r}") from exc
    try:
        value = os.fstat(descriptor)
        if (
            not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode))
            or stat.S_IFMT(value.st_mode) != stat.S_IFMT(expected.st_mode)
        ):
            raise ValueError(f"temporary artifact changed while opening: {label!r}")
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise


def _cleanup_entry_is_owned(
    parent_fd: int,
    name: str,
    descriptor: int,
    identity: tuple[int, int],
    label: str,
) -> bool:
    """在破坏 entry 前同时复验固定 fd 与当前 name binding。"""
    try:
        held = os.fstat(descriptor)
        if (held.st_dev, held.st_ino) != identity:
            return False
        if not (stat.S_ISREG(held.st_mode) or stat.S_ISDIR(held.st_mode)):
            return False
        named = _cleanup_entry_stat(parent_fd, name, label)
        if (
            not _same_inode(named, held)
            or stat.S_IFMT(named.st_mode) != stat.S_IFMT(held.st_mode)
        ):
            return False
        # 再读一次 name，覆盖 stat helper 返回后到破坏操作前的替换窗口。
        latest = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        return (
            _same_inode(latest, held)
            and stat.S_IFMT(latest.st_mode) == stat.S_IFMT(held.st_mode)
        )
    except BaseException:
        # 任意复验异常都表示 ownership 不确定，调用方必须保留现场。
        return False


def _rename_noreplace(
    source_fd: int, source_name: str, target_fd: int, target_name: str,
) -> None:
    """在固定目录 fd 间执行不覆盖 rename；不支持时拒绝降级。"""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for safe cleanup")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_fd, os.fsencode(source_name), target_fd, os.fsencode(target_name), 1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target_name)
    raise OSError(error_number, os.strerror(error_number), target_name)


def _open_cleanup_quarantine(
    parent_fd: int, original_name: str,
) -> tuple[str, int, tuple[int, int]]:
    for _ in range(128):
        quarantine_name = f".{original_name}.cleanup-{secrets.token_hex(16)}"
        try:
            os.mkdir(quarantine_name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        descriptor = None
        quarantine_identity = None
        try:
            descriptor = os.open(
                quarantine_name, _TEMP_DIRECTORY_FLAGS, dir_fd=parent_fd,
            )
            snapshot = os.fstat(descriptor)
            quarantine_identity = (snapshot.st_dev, snapshot.st_ino)
            if not stat.S_ISDIR(snapshot.st_mode) or os.listdir(descriptor):
                raise ValueError("temporary quarantine is not an empty directory")
            return quarantine_name, descriptor, quarantine_identity
        except BaseException:
            if descriptor is not None and quarantine_identity is not None:
                # 创建阶段失败也必须通过固定 fd 与 inode 绑定删除，不能按名称
                # 直接 rmdir 可能已被竞争方替换的目录。
                _cleanup_quarantine_directory(
                    parent_fd, descriptor, quarantine_name, quarantine_identity,
                )
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
            raise
    raise OSError("cannot allocate temporary cleanup quarantine")


def _claim_cleanup_entry(directory_fd: int, source_name: str) -> str | None:
    """以不覆盖 rename 原子认领待删除名称，失败时不触碰竞争 entry。"""
    for _ in range(128):
        tomb_name = f".{source_name}.remove-{secrets.token_hex(16)}"
        try:
            _rename_noreplace(
                directory_fd, source_name, directory_fd, tomb_name,
            )
        except FileExistsError:
            continue
        except FileNotFoundError:
            return None
        return tomb_name
    raise OSError("cannot allocate temporary cleanup tomb")


def _cleanup_quarantine_directory(
    parent_fd: int,
    quarantine_fd: int,
    quarantine_name: str,
    quarantine_identity: tuple[int, int],
) -> bool:
    """先原子认领 quarantine 名称，再绑定 inode 后删除空目录。"""
    tomb_name = None
    try:
        held = os.fstat(quarantine_fd)
        if (
            not stat.S_ISDIR(held.st_mode)
            or (held.st_dev, held.st_ino) != quarantine_identity
            or os.listdir(quarantine_fd)
        ):
            return False
        current = _cleanup_stat_at(parent_fd, quarantine_name)
        if (
            current is None
            or not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != quarantine_identity
        ):
            return False
        # 名称复验后再次检查固定 fd；不确定时保留 quarantine 现场。
        held_again = os.fstat(quarantine_fd)
        current_again = _cleanup_stat_at(parent_fd, quarantine_name)
        if (
            not stat.S_ISDIR(held_again.st_mode)
            or (held_again.st_dev, held_again.st_ino) != quarantine_identity
            or current_again is None
            or not stat.S_ISDIR(current_again.st_mode)
            or (current_again.st_dev, current_again.st_ino) != quarantine_identity
            or os.listdir(quarantine_fd)
        ):
            return False

        # 原子移走名称，避免 rmdir 直接跟随已被竞争方替换的 quarantine。
        tomb_name = _claim_cleanup_entry(parent_fd, quarantine_name)
        if tomb_name is None:
            return False
        try:
            tomb_fd = _open_cleanup_entry_fd(parent_fd, tomb_name, quarantine_name)
        except (OSError, ValueError):
            return False
        try:
            tomb_held = os.fstat(tomb_fd)
            if (
                not stat.S_ISDIR(tomb_held.st_mode)
                or (tomb_held.st_dev, tomb_held.st_ino) != quarantine_identity
                or not _cleanup_entry_is_owned(
                    parent_fd, tomb_name, tomb_fd, quarantine_identity,
                    quarantine_name,
                )
                or os.listdir(tomb_fd)
            ):
                return False
            os.rmdir(tomb_name, dir_fd=parent_fd)
        finally:
            os.close(tomb_fd)
        removed_fd_stat = os.fstat(quarantine_fd)
        if (
            not stat.S_ISDIR(removed_fd_stat.st_mode)
            or (removed_fd_stat.st_dev, removed_fd_stat.st_ino) != quarantine_identity
        ):
            return False
    except BaseException:
        return False
    try:
        return (
            _cleanup_stat_at(parent_fd, quarantine_name) is None
            and _cleanup_stat_at(parent_fd, tomb_name) is None
        )
    except BaseException:
        # 删除后无法确认名称状态时也必须报告失败，不能把不确定当成功。
        return False


def _restore_cleanup_entry(
    parent_fd: int,
    quarantine_fd: int,
    quarantine_name: str,
    original_name: str,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    """仅恢复仍属于本次操作的 entry，竞争方现场永不覆盖或搬运。"""
    current = _cleanup_stat_at(quarantine_fd, "entry")
    if current is None:
        return False
    if (
        expected_identity is not None
        and (current.st_dev, current.st_ino) != expected_identity
    ):
        return False
    try:
        _rename_noreplace(quarantine_fd, "entry", parent_fd, original_name)
    except (FileExistsError, FileNotFoundError, OSError):
        return False
    restored = _cleanup_stat_at(parent_fd, original_name)
    if restored is None:
        return False
    if expected_identity is not None:
        return (restored.st_dev, restored.st_ino) == expected_identity
    return _cleanup_stat_at(quarantine_fd, "entry") is None


def _validate_cleanup_tree(directory_fd: int, label: str) -> None:
    """在隔离或删除目录前递归确认其内容没有链接或特殊文件。"""
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise ValueError(f"cannot inspect temporary directory {label!r}") from exc
    for name in names:
        child_label = os.path.join(label, name)
        child_stat = _cleanup_entry_stat(directory_fd, name, child_label)
        if stat.S_ISDIR(child_stat.st_mode):
            try:
                child_fd = os.open(name, _TEMP_DIRECTORY_FLAGS, dir_fd=directory_fd)
            except OSError as exc:
                raise ValueError(f"cannot open temporary directory {child_label!r}") from exc
            try:
                _validate_cleanup_tree(child_fd, child_label)
            finally:
                os.close(child_fd)


def _quarantine_remove_entry(
    parent_fd: int,
    name: str,
    label: str,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    """隔离 entry 后以固定 fd 复验并递归删除；失败时保留现场。"""
    initial = _cleanup_entry_stat(parent_fd, name, label)
    identity = (initial.st_dev, initial.st_ino)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"temporary artifact changed before cleanup: {label!r}")

    quarantine_name, quarantine_fd, quarantine_identity = _open_cleanup_quarantine(
        parent_fd, name,
    )
    moved = False
    removed = False
    try:
        try:
            _rename_noreplace(parent_fd, name, quarantine_fd, "entry")
            moved = True
        except FileNotFoundError:
            removed = _cleanup_quarantine_directory(
                parent_fd, quarantine_fd, quarantine_name, quarantine_identity,
            )
            return removed

        # 隔离后立即持有固定 fd；不能先按名称复验再打开，否则 inode
        # 被替换并复用时会丢失原始 ownership 证据。
        entry_fd = _open_cleanup_entry_fd(quarantine_fd, "entry", label)
        claimed_name = None
        try:
            held = os.fstat(entry_fd)
            isolated = _cleanup_entry_stat(quarantine_fd, "entry", label)
            if (
                (held.st_dev, held.st_ino) != identity
                or stat.S_IFMT(held.st_mode) != stat.S_IFMT(isolated.st_mode)
                or not _cleanup_entry_is_owned(
                    quarantine_fd, "entry", entry_fd, identity, label,
                )
            ):
                _restore_cleanup_entry(
                    parent_fd, quarantine_fd, quarantine_name, name, identity,
                )
                return False

            if stat.S_ISDIR(held.st_mode):
                for child_name in os.listdir(entry_fd):
                    child_label = os.path.join(label, child_name)
                    child = _cleanup_entry_stat(entry_fd, child_name, child_label)
                    if not _quarantine_remove_entry(
                        entry_fd, child_name, child_label,
                        (child.st_dev, child.st_ino),
                    ):
                        _restore_cleanup_entry(
                            parent_fd, quarantine_fd, quarantine_name, name, identity,
                        )
                        return False
            if not _cleanup_entry_is_owned(
                quarantine_fd, "entry", entry_fd, identity, label,
            ):
                _restore_cleanup_entry(
                    parent_fd, quarantine_fd, quarantine_name, name, identity,
                )
                return False

            # 先把待删除名称原子认领到随机 tomb；若名称已被替换，
            # tomb 中的竞争方只保留现场，绝不能按原名误删。
            claimed_name = _claim_cleanup_entry(quarantine_fd, "entry")
            if claimed_name is None:
                return False
            try:
                claimed_fd = _open_cleanup_entry_fd(
                    quarantine_fd, claimed_name, label,
                )
            except (OSError, ValueError):
                # tomb 无法重新绑定时保留整个 quarantine 现场，绝不按名称删除。
                return False
            try:
                claimed = os.fstat(claimed_fd)
                if (
                    (claimed.st_dev, claimed.st_ino) != identity
                    or stat.S_IFMT(claimed.st_mode) != stat.S_IFMT(held.st_mode)
                    or not _cleanup_entry_is_owned(
                        quarantine_fd, claimed_name, claimed_fd, identity, label,
                    )
                ):
                    return False
                if stat.S_ISDIR(claimed.st_mode):
                    if os.listdir(claimed_fd):
                        return False
                    os.rmdir(claimed_name, dir_fd=quarantine_fd)
                else:
                    os.unlink(claimed_name, dir_fd=quarantine_fd)
            finally:
                os.close(claimed_fd)
        finally:
            try:
                os.close(entry_fd)
            except BaseException:
                # 主流程已经 fail-closed；仍让外层 finally 尝试关闭 quarantine fd。
                if claimed_name is None:
                    raise

        removed = _cleanup_quarantine_directory(
            parent_fd, quarantine_fd, quarantine_name, quarantine_identity,
        )
        return removed
    except ValueError:
        if moved:
            _restore_cleanup_entry(
                parent_fd, quarantine_fd, quarantine_name, name, identity,
            )
        raise
    except OSError:
        if moved:
            _restore_cleanup_entry(
                parent_fd, quarantine_fd, quarantine_name, name, identity,
            )
        return False
    finally:
        if not removed and moved:
            # 若隔离 entry 已被恢复，quarantine 应为空；若恢复遇到竞争方，
            # 保留非空 quarantine，下一次人工/显式恢复可继续审计。
            _cleanup_quarantine_directory(
                parent_fd, quarantine_fd, quarantine_name, quarantine_identity,
            )
        os.close(quarantine_fd)


def _remove_cleanup_entry(
    parent_fd: int,
    name: str,
    label: str,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    """兼容旧内部调用，使用隔离 rename 而不是 stat 后直接 unlink。"""
    if not _quarantine_remove_entry(parent_fd, name, label, expected_identity):
        raise ValueError(f"temporary artifact changed during cleanup: {label!r}")


def cleanup_expired_temp(
    temp_dir: str,
    max_age_hours: int = 24,
    *,
    prefixes: Optional[Iterable[str]] = None,
) -> int:
    """清理指定根目录下过期且属于 pipeline 的临时 artifact。

    只匹配顶层 basename 以 ``prefixes`` 开头的 regular file/目录；目录会在
    删除前递归检查内容，任何 symlink 或特殊文件都会 fail-closed。函数不会
    扫描或创建系统 ``/tmp``，返回成功删除的顶层 artifact 数量。
    """
    import logging

    if type(max_age_hours) is not int or max_age_hours < 0:
        raise ValueError("max_age_hours must be a non-negative Python integer")
    root = _cleanup_path_string(temp_dir)
    allowed_prefixes = _cleanup_prefixes(prefixes)
    # 即使目标叶节点不存在，也必须检查已有父级，避免通过
    # ``symlink/missing`` 路径绕过 symlink 拒绝后静默返回。
    _validate_existing_directory_prefix(root)
    if not os.path.lexists(root):
        return 0
    if os.path.realpath(root) != root:
        raise ValueError(f"temp_dir path contains a symlink: {root!r}")
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"cannot inspect temp_dir {root!r}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"temp_dir must not be a symlink: {root!r}")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"temp_dir must be a directory: {root!r}")

    logger = logging.getLogger(__name__)
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    count = 0
    try:
        root_fd = open_safe_directory(root, create=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot open temp_dir {root!r}") from exc
    try:
        opened_root = os.fstat(root_fd)
        if not _same_inode(opened_root, root_stat):
            raise ValueError(f"temp_dir changed while opening: {root!r}")
    except (OSError, ValueError) as exc:
        try:
            os.close(root_fd)
        except OSError:
            pass
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot verify temp_dir {root!r}") from exc
    try:
        for name in os.listdir(root_fd):
            if not name.startswith(allowed_prefixes):
                continue
            label = os.path.join(root, name)
            value = _cleanup_entry_stat(root_fd, name, label)
            # 目录内容也必须安全，即使目录本身尚未过期；不能把 symlink
            # 藏在一个新目录里，等未来的清理调用误删其目标。
            if stat.S_ISDIR(value.st_mode):
                child_fd = os.open(name, _TEMP_DIRECTORY_FLAGS, dir_fd=root_fd)
                try:
                    _validate_cleanup_tree(child_fd, label)
                finally:
                    os.close(child_fd)
            if now - value.st_mtime <= max_age_seconds:
                continue
            try:
                _remove_cleanup_entry(
                    root_fd, name, label, (value.st_dev, value.st_ino),
                )
            except (OSError, ValueError) as exc:
                # 单个外来/竞态条目（删除阶段身份复验失败）只告警并跳过，
                # 不让整次清理中止；symlink/特殊类型仍在 stat 阶段 fail-closed。
                logger.warning("Failed to cleanup pipeline temp artifact %s: %s", label, exc)
                continue
            count += 1
    finally:
        os.close(root_fd)
    if count:
        logger.info("Cleaned up %d expired pipeline temp artifacts under %s", count, root)
    return count
