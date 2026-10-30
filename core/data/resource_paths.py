"""Dataset 资源路径解析。

配置中的 ``data/...`` 是仓库内资源引用；manifest 内部相对路径则锚定到
直接载体文件所在目录。这里只解析路径，不检查或读取音频 payload。
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DATA_SECTION_RESOURCE_KEYS = frozenset(
    {
        "speech_scp",
        "noise_scp",
        "rir_scp",
        "windnoise_scp",
        "speech_length_file",
        "music_scp",
        "clean_scp",
        "noisy_scp",
        "mic_scp",
        "farend_scp",
        "nearend_scp",
        "error_scp",
        "real_farend_echo_pair_json",
        "samples_list",
        "samples_jsonl",
        "cues_yaml",
        "spk2enroll_json",
    }
)

_SIMULATION_RESOURCE_KEYS = frozenset(
    {
        "spk2enroll_json",
    }
)


def repository_root() -> Path:
    """返回由本模块位置确定的仓库根目录。"""
    return _REPO_ROOT


def resolve_config_resource_path(
    value: str | os.PathLike[str], *, repo_root: Path | None = None
) -> str:
    """解析单个配置资源路径。

    仅 ``data/...`` 引用锚定仓库根。最终 canonical path 必须仍在仓库内，
    因而 ``..`` 越界和现有 symlink 跳出仓库都会 fail-closed。其他相对路径
    与绝对路径原样保留，继续沿用调用进程原有语义。
    """
    raw = os.fspath(value)
    if not (raw == "data" or raw.startswith("data/")):
        return raw

    root = (repo_root or repository_root()).resolve(strict=True)
    resolved = (root / raw).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"repository data resource escapes repository root: {raw!r} -> {resolved}"
        ) from exc
    return os.fspath(resolved)


def resolve_manifest_resource_path(
    value: str | os.PathLike[str], manifest_path: str | os.PathLike[str]
) -> str:
    """将 manifest 内部相对路径锚定到直接载体文件的 parent。

    绝对路径与 ``~`` 开头路径保持不变；函数不要求资源存在，也不读取资源。
    """
    raw = os.fspath(value)
    if os.path.isabs(raw) or raw.startswith("~"):
        return raw
    manifest_parent = Path(os.path.abspath(os.fspath(manifest_path))).parent
    return os.path.normpath(os.fspath(manifest_parent / raw))


def resolve_data_section_resource_paths(section: MutableMapping) -> None:
    """原地解析 Dataset section 中白名单资源键，不触碰模型/checkpoint 字段。"""
    for key in _DATA_SECTION_RESOURCE_KEYS:
        value = section.get(key)
        if isinstance(value, (str, os.PathLike)):
            section[key] = resolve_config_resource_path(value)

    simulation_cfg = section.get("simulation_cfg")
    if not isinstance(simulation_cfg, MutableMapping):
        return
    for key in _SIMULATION_RESOURCE_KEYS:
        value = simulation_cfg.get(key)
        if isinstance(value, (str, os.PathLike)):
            simulation_cfg[key] = resolve_config_resource_path(value)
