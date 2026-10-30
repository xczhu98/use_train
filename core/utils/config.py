"""配置加载与继承工具。"""

from __future__ import annotations

import hashlib
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from core.utils.artifact_identity import SourceBinding
from core.utils.inference_artifacts import fingerprint_json


BASE_CONFIG_KEY = "base_config"


def _config_root(config_path: Path) -> Path:
    """仓库配置优先以 ``configs/`` 为根；独立配置以入口目录为根。"""
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent
    return config_path.parent


def _resolve_base_config_path(base: str, *, config_path: Path, root_dir: Path) -> Path:
    """词法解析 base 路径；允许根内 ``..``，拒绝绝对路径与越界。"""
    raw_path = Path(base)
    if raw_path.is_absolute():
        raise ValueError(f"base_config must be relative, got absolute path: {base}")

    normalized = Path(os.path.abspath(config_path.parent / raw_path))
    try:
        normalized.relative_to(root_dir)
    except ValueError as exc:
        raise ValueError(
            f"base_config escapes config root {root_dir}: {base} -> {normalized}"
        ) from exc
    return normalized


def _open_directory_without_symlinks(directory: Path) -> int:
    """从根目录逐级打开稳定 dirfd，拒绝任一 symlink ancestor。"""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in directory.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        if exc.errno == getattr(os, "ELOOP", 40):
            raise ValueError(
                f"config paths must not contain symbolic links: {directory}"
            ) from exc
        raise


def _open_config_at(config_path: Path, *, root_dir: Path, root_fd: int):
    """通过稳定 root dirfd 打开配置；校验与读取复用同一个文件描述符。"""
    try:
        relative = config_path.relative_to(root_dir)
    except ValueError as exc:
        raise ValueError(
            f"config path escapes config root {root_dir}: {config_path}"
        ) from exc
    if not relative.parts:
        raise ValueError(f"config path must be a file: {config_path}")

    directory_fd = os.dup(root_fd)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(relative.name, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno == getattr(os, "ELOOP", 40):
            raise ValueError(
                f"config paths must not contain symbolic links: {config_path}"
            ) from exc
        raise
    finally:
        os.close(directory_fd)

    file_stat = os.fstat(file_fd)
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_fd)
        raise ValueError(f"config path must be a regular file: {config_path}")
    return os.fdopen(file_fd, "rb"), (file_stat.st_dev, file_stat.st_ino)


def _read_config_bytes(config_file, config_path: Path) -> tuple[bytes, dict[str, Any]]:
    before = os.fstat(config_file.fileno())
    content = config_file.read()
    after = os.fstat(config_file.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError(f"config source changed during read: {config_path}")
    if len(content) != after.st_size:
        raise RuntimeError(f"config source size changed during read: {config_path}")
    return content, {
        "path": str(config_path), "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _load_raw_merged(
    config_path: Path,
    visited: frozenset[tuple[int, int]],
    chain: tuple[Path, ...],
    *,
    root_dir: Path,
    root_fd: int,
    file_identities: dict[str, dict[str, Any]] | None = None,
) -> DictConfig:
    """递归加载 ``config_path`` 及其 ``base_config`` 继承链，返回未解析插值的 OmegaConf 对象。

    ``base_config`` 自身也可以再指向上一层 base（多层继承），因此每一层都递归调用本函数，
    而不是只 ``OmegaConf.load()`` 一次——否则更上层 base 的字段会被静默丢弃。
    """
    config_file, identity = _open_config_at(
        config_path, root_dir=root_dir, root_fd=root_fd
    )
    if identity in visited:
        config_file.close()
        inheritance_chain = " -> ".join(str(path) for path in (*chain, config_path))
        raise ValueError(
            f"Circular base_config inheritance detected: {inheritance_chain}"
        )
    visited = visited | {identity}
    chain = (*chain, config_path)

    with config_file:
        content, fingerprint = _read_config_bytes(config_file, config_path)
    if file_identities is not None:
        key = str(config_path)
        previous = file_identities.setdefault(key, fingerprint)
        if previous != fingerprint:
            raise RuntimeError(f"config source changed between inherited reads: {config_path}")
    cfg = OmegaConf.load(io.StringIO(content.decode("utf-8")))
    if BASE_CONFIG_KEY in cfg:
        base_raw = cfg[BASE_CONFIG_KEY]
        if isinstance(base_raw, (list, tuple)) or OmegaConf.is_list(base_raw):
            base_paths = [str(p) for p in base_raw]
        else:
            base_paths = [str(base_raw)]

        base_cfgs = []
        for base in base_paths:
            base_path = _resolve_base_config_path(
                base, config_path=config_path, root_dir=root_dir
            )
            base_cfgs.append(
                _load_raw_merged(
                    base_path,
                    visited,
                    chain,
                    root_dir=root_dir,
                    root_fd=root_fd,
                    file_identities=file_identities,
                )
            )

        cfg = OmegaConf.merge(*base_cfgs, cfg)

    return cfg


def load_merged_config(config_path: str | Path, *, pop_base: bool = True) -> dict[str, Any]:
    """加载 YAML 配置，并合并可选的 ``base_config`` 继承（支持多层递归继承）。

    ``base_config`` 可以是单个路径字符串，也可以是路径列表；相对路径基于所在配置文件
    所在目录解析。仓库内配置以最近的 ``configs/`` 祖先为安全根，因此允许规范化后仍在
    该根内的 ``..``；独立配置以入口目录为根。绝对路径、越界路径及任一 symlink 路径
    都会被拒绝。配置校验与读取复用同一稳定文件描述符，避免路径 reopen TOCTOU。
    ``base_config`` 自身也可以继续继承，循环引用会抛出 ``ValueError``。

    合并规则采用 ``OmegaConf.merge``：dict 按 key 递归深合并，子配置覆盖 base 配置的
    同名字段；**列表按 OmegaConf 默认语义整体替换**（子配置的列表完全覆盖 base 配置的
    同名列表，不会按元素合并/追加）。

    Args:
        config_path: 子配置文件路径。
        pop_base: 合并后是否从结果中删除 ``base_config`` 键（默认 True）。

    Returns:
        合并后的普通 dict，所有 OmegaConf 插值已解析（插值在整条继承链合并完成后
        统一解析一次，因此可以跨层引用）。
    """
    return _load_merged_config(config_path, pop_base=pop_base)


def _load_merged_config(
    config_path: str | Path,
    *,
    pop_base: bool,
    file_identities: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config_path = Path(os.path.abspath(Path(config_path).expanduser()))
    root_dir = _config_root(config_path)
    root_fd = _open_directory_without_symlinks(root_dir)
    try:
        cfg = _load_raw_merged(
            config_path,
            frozenset(),
            (),
            root_dir=root_dir,
            root_fd=root_fd,
            file_identities=file_identities,
        )
    finally:
        os.close(root_fd)

    if pop_base and BASE_CONFIG_KEY in cfg:
        OmegaConf.set_struct(cfg, False)
        del cfg[BASE_CONFIG_KEY]

    resolved = OmegaConf.to_container(cfg, resolve=True)
    from core.data.simulation_schema import normalize_config_simulation_sections

    return normalize_config_simulation_sections(resolved)


@dataclass(frozen=True)
class LoadedConfigSnapshot:
    """实际解析配置及完整继承链的不可变快照；各属性返回深隔离视图。

    ``file_identities`` 以展开用户目录、词法规范化后的绝对路径字符串为键，
    不解析或接受 symlink；值为同一路径的 ``path/size/sha256`` 内容身份。
    重复继承同一路径只保留一条，读取间内容变化则拒绝构造快照。
    ``resolved_identity`` 使用 ``sha256-json-v1`` 标识已解析的 effective config。
    """

    _binding: SourceBinding

    @property
    def config(self) -> dict[str, Any]:
        return self._binding.as_dict()["parameters"]["config"]

    @property
    def file_identities(self) -> dict[str, dict[str, Any]]:
        return self._binding.as_dict()["files"]

    @property
    def resolved_identity(self) -> dict[str, str]:
        return {"algorithm": "sha256-json-v1", "sha256": fingerprint_json(self.config)}

    def verify_current_sources(self) -> None:
        """复核完整绑定及当前文件内容；不以新路径内容刷新加载身份。"""
        self._binding.verify()


def load_merged_config_snapshot(
    config_path: str | Path, *, pop_base: bool = True,
) -> LoadedConfigSnapshot:
    """按 ``load_merged_config`` 规则加载，同时冻结供解析的原始 FD 字节身份。

    所有入口和 base 文件在同一继承加载过程中记录，绝不事后重读路径补 hash。
    返回后文件仍可能变化；消费者应在加载/发布边界调用 ``verify_current_sources``。
    """
    file_identities: dict[str, dict[str, Any]] = {}
    config = _load_merged_config(
        config_path, pop_base=pop_base, file_identities=file_identities,
    )
    if not isinstance(config, dict):
        raise ValueError("config snapshot must resolve to a mapping")
    binding = SourceBinding.bind(
        {path: path for path in file_identities},
        parameters={"config": config}, precomputed=file_identities,
    )
    return LoadedConfigSnapshot(binding)
