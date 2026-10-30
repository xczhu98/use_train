#!/usr/bin/env python3
"""JAE-M03 AEC OOD stress set 生成器。

只生成固定 OOD stress set（场景定义见 ``_ood_stress_scenarios.py``），
不改训练数据分布（dynamic_dataset / 训练管线不动）。每场景固定 seed
样本可回放：逐样本 rng 由（master seed + 样本全局索引）派生，同参重跑
逐位一致。事件时间与 oracle stems（nearend/echo）随样本导出。

用法::

    # 零数据依赖 smoke（内置确定性合成源，seed 驱动）
    python -m tools.prepare_data_tools.generators.aec.generate_ood_stress_set \\
        --output-dir /tmp/ood

    # 使用真实语料（两列 wav scp: key path）
    python -m tools.prepare_data_tools.generators.aec.generate_ood_stress_set \\
        --farend-scp far.scp --nearend-scp near.scp --rir-scp rir.scp \\
        --scenarios all --num-per-scenario 8 --seed 0 --output-dir out/ood

输出结构::

    <out>/
      manifest.json                       # git commit / 实际成功样本目录 / 生成参数 / seed
      <scenario>/
        mic.scp farend.scp nearend.scp    # 4 列 scp（uid sr frames path），供现有 eval 管线
        sample_000/
          mic.wav farend.wav nearend.wav echo.wav meta.json

用户决策（2026-07-15）：这些 OOD 场景在实际业务中大概率碰不到，**不纳入
训练仿真链**；本工具仅作为评估侧能力保留，供需要时生成 stress set 复测。

发布状态：GenerationManifest v1 publication 已下线，``main`` 在场景解析前显式 fail-closed。
迁移计划：AEC schema v2 reload 就绪后，把 ``main`` 的发布改走
``tools.prepare_data_tools.offline_publication.publish_owned_offline_dataset_roots``；
迁移完成前本文件保留的生成骨架仅供内部测试，不得作为生产发布路径。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from tools.prepare_data_tools._common import _git_commit
from tools.shared.scp_io import write_scp_entries

# 生成器身份采用 canonical 脚本名；版本只随数据内容或正式发布语义变化。
GENERATOR_ID = "generate_ood_stress_set.py"
# 1.1: 增加业务 manifest inventory 与 source path-set identity。
# 1.2: 接入的 GenerationManifest publication 已下线，保留业务生成版本标识。
GENERATOR_VERSION = "1.2"
MANIFEST_SCHEMA_VERSION = 2
SOURCE_IDENTITY_SCHEMA_VERSION = 1
_AUDIO_PAYLOAD_SUFFIXES = frozenset({".wav", ".flac", ".rir"})
OWNER_FILE = ".generate_ood_stress_set.owner.json"
OWNER_SCHEMA = 1
PUBLICATION_OWNER_ID = "tasks/aec/generate_ood_stress_set.py"
_PUBLICATION_CONTROL_FILES = frozenset()


# ----------------------------------------------------------------------
# 源加载：scp 或确定性合成
# ----------------------------------------------------------------------
def _resolve_regular_file_no_symlink(path: Path, *, base: Path | None = None) -> Path:
    """解析普通文件路径，拒绝路径任一层为 symlink，且不读取 payload。"""
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (base if base is not None else Path.cwd()) / candidate
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    current = Path(absolute.anchor)
    final_stat = None
    for part in absolute.parts[1:]:
        current /= part
        final_stat = current.lstat()
        if stat.S_ISLNK(final_stat.st_mode):
            raise ValueError(f"拒绝 symlink 源路径: {current}")
    if final_stat is None or not stat.S_ISREG(final_stat.st_mode):
        raise ValueError(f"源路径不是普通文件: {absolute}")
    return absolute


def _ordinary_file_fingerprint(path: Path) -> dict[str, object]:
    """对 SCP 等非音频普通文件做稳定内容 fingerprint。"""
    resolved = _resolve_regular_file_no_symlink(path)
    if resolved.suffix.lower() in _AUDIO_PAYLOAD_SUFFIXES:
        raise ValueError(f"禁止对音频 payload 计算 fingerprint: {resolved}")
    before = resolved.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"source is not a regular file: {resolved}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"source replaced before read: {resolved}")
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = resolved.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if identity(before) != identity(after_open) or identity(before) != identity(after):
        raise RuntimeError(f"source changed while reading: {resolved}")
    return {
        "path": str(resolved),
        "device": before.st_dev,
        "inode": before.st_ino,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _source_identity(
    scp: Path | None, entries: list[tuple[str, Path]] | None
) -> dict[str, object] | None:
    """记录 SCP 与音频路径集合；音频/RIR payload 不做额外打开或 hash。"""
    if scp is None or entries is None:
        return None
    source = _resolve_regular_file_no_symlink(scp)
    paths = sorted({path for _, path in entries}, key=lambda path: str(path))
    files = []
    for path in paths:
        resolved = _resolve_regular_file_no_symlink(path)
        info = resolved.stat(follow_symlinks=False)
        files.append(
            {
                "path": str(resolved),
                "device": info.st_dev,
                "inode": info.st_ino,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
        )
    return {
        "schema_version": SOURCE_IDENTITY_SCHEMA_VERSION,
        "kind": "regular_file_path_set",
        "scp": _ordinary_file_fingerprint(source),
        "files": files,
    }


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    """记录业务发布结构；共享 publication 控制文件由 envelope 管理。"""
    files: dict[str, dict[str, object]] = {}
    excluded = {"manifest.json", OWNER_FILE} | _PUBLICATION_CONTROL_FILES
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"inventory 拒绝 symlink: {path}")
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in _AUDIO_PAYLOAD_SUFFIXES:
            files[relative] = {"kind": "waveform"}
            continue
        fingerprint = _ordinary_file_fingerprint(path)
        files[relative] = {
            "kind": "metadata",
            "size": fingerprint["size"],
            "sha256": fingerprint["sha256"],
        }
    return files


def _load_wav(path: Path, sr: int, n: int | None) -> np.ndarray:
    """读 wav → float32 单声道；sr 不一致时线性插值重采样；n 给定时定长（重复/裁切）。"""
    x, x_sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        # 空音频（0 长度文件）会以空数组流入后续卷积，直接报错，由外层按失败样本处理
        raise ValueError(f"空音频文件（0 长度）: {path}")
    if x_sr != sr:
        # 避免额外重采样依赖，stress set 用线性插值即可
        m = int(round(len(x) * sr / x_sr))
        pos = np.arange(m, dtype=np.float64) * (x_sr / sr)
        x = np.interp(pos, np.arange(len(x), dtype=np.float64), x.astype(np.float64))
        x = x.astype(np.float32)
    if n is not None:
        if len(x) < n:
            reps = int(np.ceil(n / len(x)))
            x = np.tile(x, reps)
        x = x[:n]
    return x


# ----------------------------------------------------------------------
# 确定性 / 输出工具
# ----------------------------------------------------------------------
def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _is_owned(root: Path) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
        owner_path = root / OWNER_FILE
        if owner_path.is_symlink() or not owner_path.is_file():
            return False
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        return owner == {"generator": GENERATOR_ID, "schema_version": OWNER_SCHEMA}
    except (OSError, json.JSONDecodeError):
        return False


@contextmanager
def _output_lock(out_root: Path) -> Iterator[None]:
    out_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_root.parent / f".{out_root.name}.{GENERATOR_ID}.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _remove_owned_tree(path: Path) -> None:
    if not _is_owned(path):
        raise RuntimeError(f"拒绝清理 ownership 不匹配的目录: {path}")
    shutil.rmtree(path)


def _publish_staging(staging: Path, out_root: Path) -> None:
    backup = out_root.parent / f".{out_root.name}.{GENERATOR_ID}.backup"
    if backup.exists() or backup.is_symlink():
        if not _is_owned(backup):
            raise RuntimeError(f"backup ownership 不匹配，拒绝处理: {backup}")
        if out_root.exists() or out_root.is_symlink():
            _remove_owned_tree(backup)
        else:
            os.replace(backup, out_root)
    if out_root.exists() or out_root.is_symlink():
        if not _is_owned(out_root):
            raise RuntimeError(f"输出目录不属于 {GENERATOR_ID}，拒绝替换: {out_root}")
        os.replace(out_root, backup)
    try:
        os.replace(staging, out_root)
    except BaseException:
        if backup.exists() and not out_root.exists() and not out_root.is_symlink():
            os.replace(backup, out_root)
        raise
    if backup.exists():
        try:
            _remove_owned_tree(backup)
        except (OSError, RuntimeError) as exc:
            print(
                f"[WARN] 新集合已提交，但旧 owned backup 清理失败: {backup}: {exc}",
                file=sys.stderr,
            )


def _validate_positive_parameters(args: argparse.Namespace) -> None:
    if args.num_per_scenario <= 0:
        raise ValueError("--num-per-scenario 必须为正整数")
    if args.sr <= 0:
        raise ValueError("--sr 必须为正整数")
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0.0:
        raise ValueError("--duration-sec 必须为正有限数")
    frames = args.duration_sec * args.sr
    if not math.isfinite(frames) or frames < 1.0:
        raise ValueError("--duration-sec × --sr 必须至少产生 1 帧")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JAE-M03 AEC OOD stress set 生成器（固定 seed 可回放，不改训练数据分布）"
    )
    parser.add_argument("--farend-scp", type=Path, default=None,
                        help="far-end 两列 wav scp（key path）；缺省用合成 chirp 源")
    parser.add_argument("--nearend-scp", type=Path, default=None,
                        help="near-end 两列 wav scp；缺省用合成语音形噪声")
    parser.add_argument("--rir-scp", type=Path, default=None,
                        help="RIR 两列 wav scp；缺省用合成指数衰减噪声 0.2s")
    parser.add_argument("--scenarios", type=str, default="all",
                        help="逗号分隔场景名，默认 all（全部 9 个场景）")
    parser.add_argument("--num-per-scenario", type=int, default=4,
                        help="每场景样本数（默认 4）")
    parser.add_argument("--seed", type=int, default=0, help="master seed（默认 0）")
    parser.add_argument("--sr", type=int, default=16000, help="采样率（默认 16000）")
    parser.add_argument("--duration-sec", type=float, default=4.0,
                        help="每样本时长秒（默认 4.0）")
    parser.add_argument("--output-dir", dest="out_dir", type=Path,
                        required=True, help="输出根目录")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        _validate_positive_parameters(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(
        "[ERROR] GenerationManifest v1 publication 已下线；schema v2 reload 当前仅支持 SE；"
        "AEC offline publication 需要迁移后重新生成。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
