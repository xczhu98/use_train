#!/usr/bin/env python3
"""
清理 noise_filter_pipeline 运行后残留的临时目录/文件。

扫描目标：
  - output_root/stage_b/ 及 stage_b.temp_dir 下以 ``stage_b_unified_`` 开头的目录
    （stage_b_unified.py 中 TemporaryDirectory 的 temp_parent 默认是 stage_b 输出目录）
  - 系统临时目录及 stage_b.b3.temp_dir 下以 ``vad_shared_`` 开头的文件
    （vad/multi.py 中 mkstemp 的 dir 为 VAD temp_dir，未配置时落在系统临时目录）
  - output_root 下 tmp+replace 原子写的崩溃残留（递归）：
      - audio_utils mkstemp 隐藏临时文件（``.{basename}.<rand>{ext}``，如 ``.x.flac.a1b2.flac``）
      - ``*.jsonl.tmp``（manifest/merged 原子写）
      - ``*.complete.tmp``（stage 完成 marker 原子写）
      - ``*.json.*.tmp``（llm_cache 原子写，亦扫描配置的 cache_dir）

只删除超过指定年龄（默认 24 小时）的残留，避免误删正在运行的任务。
目录年龄按子树内最新 mtime 判定：顶层目录 mtime 只在直接增删子项时变化，
运行中的 stage_b_unified_* 持续往子目录写文件时顶层 mtime 恒定，只看顶层
会把活跃目录误判为过期残留。
脚本启动时会获取文件锁，防止多个实例并发清理。``--purge-intermediate``
还会按固定顺序持有 stage_d run 锁与 marker v4 绑定的 final-root 锁，排斥
同一 run 的 writer/promote 以及同一 final 的并发 promote。

``--purge-intermediate``（promote 语义配套，见重构设计文档第三节）：
完整有效 marker v4 仅共同授权 ``stage_b/audio`` 与 marker 指定的
``stage_d/D1/audio`` 或 ``stage_d/D2/audio`` 两棵树；无 marker 时任一棵都不删。
锁内复验 marker、final 资产、双树 identity、年龄和 symlink 后，先把两棵树
rename 到本次 owned quarantine；任一隔离失败即恢复，全部隔离成功后才删除
quarantine。dry-run 执行相同验证但不 rename。最终音频 promote 到 final/ 后，
这些目录已无消费方，runs/ 只留文本元数据。本开关删除量大，建议先 dry-run。

用法：
  python tools/noise_filter_pipeline/scripts/cleanup_pipeline_temp.py \
      --config tools/noise_filter_pipeline/config/pipeline.yaml \
      --min-age-hours 24

加 cron（每天 3:17 执行一次；可按环境设置仓库根目录和日志路径）：
  17 3 * * * cd "${USE_TRAIN_ROOT:-$HOME/use_train}" && \
    python tools/noise_filter_pipeline/scripts/cleanup_pipeline_temp.py \
      --config tools/noise_filter_pipeline/config/pipeline.yaml \
      >> "${PIPELINE_CLEANUP_LOG:-runs/pipeline_cleanup.log}" 2>&1
"""
from __future__ import annotations

import argparse
import fnmatch
import inspect
import json
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 让 noise_filter_pipeline 包可导入
_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig  # noqa: E402
from noise_filter_pipeline.core.run_lock import (  # noqa: E402
    acquire_lock,
    cleanup_lock_path,
    release_lock,
    stage_d_run_lock,
)
from noise_filter_pipeline.scripts.promote_run_to_final import (  # noqa: E402
    PROMOTE_JOURNAL_NAME,
    _acquire_promote_lock,
    _release_promote_lock,
    read_validated_promote_marker,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理 pipeline 残留临时目录/文件")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="pipeline.yaml 路径；默认使用 tools/noise_filter_pipeline/config/pipeline.yaml",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=24.0,
        help="只删除修改时间超过 N 小时的残留（默认 24）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要删除的内容，不实际删除",
    )
    parser.add_argument(
        "--purge-intermediate",
        action="store_true",
        help="额外删除 stage_b/audio 与 marker 指定的 Stage D 最终 audio 整棵树"
        "（promote 到 final/ 后使用；受 --min-age-hours 年龄保护，建议先 --dry-run）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="打印更多信息",
    )
    return parser.parse_args()


def _collect_scan_roots(config_path: str) -> Tuple[Set[str], Set[str], str]:
    """从配置文件收集需要扫描的根目录。

    返回 (浅层扫描根, 递归扫描根, output_root)：
    浅层根用于 stage_b_unified_*/vad_shared_* 的 glob；递归根（output_root
    与配置的 LLM cache_dir）用于原子写临时文件残留的 rglob。
    """
    roots: Set[str] = set()
    recursive_roots: Set[str] = set()
    cfg = UnifiedPipelineConfig.from_yaml(config_path, active_stages=())

    # output_root/stage_b：Stage B unified 默认临时目录的父目录
    # （temp_parent 默认 output_dir，即 output_root/stage_b）
    output_root = os.path.abspath(cfg.output_root)
    roots.add(os.path.join(output_root, "stage_b"))
    # 原子写临时文件散落在各 stage 输出目录（含音频输出）下，需递归扫描
    recursive_roots.add(output_root)

    # 系统临时目录：VAD 共享临时文件（vad_shared_*）在 temp_dir 未配置时落在这里
    roots.add(os.path.abspath(tempfile.gettempdir()))

    # 显式配置的临时目录（B3 VAD 配置在 stage_b.b3 下）
    stage_b = cfg.stage_b or {}
    b3 = stage_b.get("b3", {})

    stage_b_temp = stage_b.get("temp_dir")
    if stage_b_temp:
        roots.add(os.path.abspath(stage_b_temp))

    b3_temp = b3.get("temp_dir")
    if b3_temp:
        roots.add(os.path.abspath(b3_temp))

    # 各 stage 配置里显式配置的 LLM cache_dir（含嵌套子配置，如 stage_c.c2）。
    # 相对路径按 CWD 解析（abspath），与 pipeline.yaml 加载侧口径一致；
    # 当前 pipeline.yaml 各 cache_dir 均为 null，实际不受该口径影响
    def _collect_cache_dirs(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("cache_dir", "qwen_cache_dir") and isinstance(value, str) and value:
                    recursive_roots.add(os.path.abspath(value))
                else:
                    _collect_cache_dirs(value)
        elif isinstance(node, list):
            for value in node:
                _collect_cache_dirs(value)

    for stage_name in ("A", "B", "C", "D"):
        _collect_cache_dirs(cfg.get_stage_config(stage_name))

    return roots, recursive_roots, output_root


def _acquire_lock(output_root: str) -> int:
    """获取 output-root 粒度的持久清理锁。"""
    lock_path = cleanup_lock_path(output_root)
    try:
        return acquire_lock(lock_path, purpose="清理锁")
    except RuntimeError as exc:
        raise RuntimeError(
            f"无法获取清理锁 {lock_path}，可能已有其他清理进程在运行: {exc}"
        ) from exc


def _release_lock(fd: int) -> None:
    release_lock(fd)


def _subtree_latest_mtime(path: Path) -> Optional[float]:
    """返回子树最新 mtime；任一遍历或 stat 错误均 fail closed 返回 None。"""
    try:
        latest = path.stat().st_mtime
    except OSError:
        return None
    walk_failed = False

    def _on_walk_error(_error: OSError) -> None:
        nonlocal walk_failed
        walk_failed = True

    for dirpath, dirnames, filenames in os.walk(path, onerror=_on_walk_error):
        for name in dirnames + filenames:
            try:
                latest = max(latest, os.stat(os.path.join(dirpath, name)).st_mtime)
            except OSError:
                return None
    return None if walk_failed else latest


# 递归扫描的原子写临时文件残留模式：
#   audio_utils mkstemp：.{basename}.<rand>{ext}（隐藏文件，含音频扩展名）
#   manifest/merged 原子写：*.jsonl.tmp
#   stage 完成 marker 原子写：*.complete.tmp（含 .complete.tmp 本身）
#   llm_cache 原子写：<key>.json.<pid>.tmp
#   checkpoint 原子写：checkpoint*.tmp（checkpoint.py 的 checkpoint.json.tmp；
#     正常产物为 checkpoint.json，无 .tmp 后缀，不会误匹配）
#   relayout 原子写：*.relayout_tmp（relayout_stage_d_audio.py 的
#     d2_merged.jsonl.relayout_tmp、d_manifest_<ds>.jsonl.relayout_tmp；
#     正常产物无 .relayout_tmp 后缀，不会误匹配）
_TMP_FILE_PATTERNS = (
    [f".*{ext}.*" for ext in (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".aac")]
    + [
        "*.jsonl.tmp",
        "*.complete.tmp",
        "*.json.*.tmp",
        "*.relayout_tmp",
        "checkpoint*.tmp",
    ]
)

# 扫描结果仍以 Path 暴露以保持 API 兼容；最终删除以候选及其直接 parent
# identity 为安全锚点，不保留无法约束最终 rename 边界的 scan-root identity。
_SCAN_IDENTITIES: Dict[str, Tuple[int, int, int, int, int]] = {}
_SCAN_PARENT_IDENTITIES: Dict[str, Tuple[int, int]] = {}


def _lstat_identity(path: str, *, expect_dir: Optional[bool] = None) -> Tuple[int, int, int, int, int]:
    """读取候选 lstat identity，拒绝 symlink/特殊节点。"""
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"扫描路径不得是符号链接: {path}")
    if expect_dir is True and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"扫描根必须是目录: {path}")
    if expect_dir is False and not stat.S_ISREG(info.st_mode):
        raise ValueError(f"扫描候选必须是 regular file: {path}")
    return stat.S_IFMT(info.st_mode), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _register_scan_candidate(path: str, *, expect_dir: bool) -> None:
    absolute = os.path.abspath(path)
    if absolute != os.path.realpath(absolute):
        raise ValueError(f"扫描路径 ancestor 不得是符号链接: {absolute}")
    identity = _lstat_identity(absolute, expect_dir=expect_dir)
    parent = os.lstat(os.path.dirname(absolute))
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise ValueError(f"候选 parent 不安全: {absolute}")
    _SCAN_IDENTITIES[absolute] = identity
    _SCAN_PARENT_IDENTITIES[absolute] = (parent.st_dev, parent.st_ino)


def _find_residuals(
    roots: Set[str],
    min_age_hours: float,
    recursive_roots: Optional[Set[str]] = None,
    excluded_subtrees: Optional[Set[str]] = None,
) -> List[Path]:
    """找出所有超过年龄的残留路径，可对大目录做递归剪枝。"""
    now = time.time()
    min_age_sec = min_age_hours * 3600
    residuals: List[Path] = []

    _SCAN_IDENTITIES.clear()
    _SCAN_PARENT_IDENTITIES.clear()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            logger.debug("扫描根目录不存在，跳过: %s", root)
            continue
        try:
            _lstat_identity(str(root_path), expect_dir=True)
        except (OSError, ValueError) as exc:
            logger.error("拒绝不安全扫描根 %s: %s", root, exc)
            continue

        # Stage B unified 残留目录（年龄按子树最新 mtime 判定）
        for item in root_path.glob("stage_b_unified_*"):
            if not item.is_dir():
                continue
            latest = _subtree_latest_mtime(item)
            if latest is None:
                # 子树根 stat 失败（并发删除）：跳过
                continue
            if (now - latest) > min_age_sec:
                try:
                    _register_scan_candidate(str(item), expect_dir=True)
                except (OSError, ValueError) as exc:
                    logger.error("拒绝不安全清理候选 %s: %s", item, exc)
                    continue
                residuals.append(item)

        # VAD 共享临时文件
        for item in root_path.glob("vad_shared_*"):
            if not item.is_file():
                continue
            try:
                is_stale = (now - item.stat().st_mtime) > min_age_sec
            except OSError:
                # glob 后 stat 前被并发删除：跳过而非 traceback
                continue
            if is_stale:
                try:
                    _register_scan_candidate(str(item), expect_dir=False)
                except (OSError, ValueError) as exc:
                    logger.error("拒绝不安全清理候选 %s: %s", item, exc)
                    continue
                residuals.append(item)

    excluded_real = {os.path.realpath(path) for path in excluded_subtrees or ()}

    def _is_excluded(path: str) -> bool:
        real = os.path.realpath(path)
        return any(os.path.commonpath((real, excluded)) == excluded for excluded in excluded_real)

    # tmp+replace 原子写的崩溃残留（递归扫描 output_root / cache_dir）
    for root in recursive_roots or ():
        if _is_excluded(root):
            logger.debug("递归扫描根位于保护子树内，跳过: %s", root)
            continue
        root_path = Path(root)
        if not root_path.exists():
            logger.debug("递归扫描根目录不存在，跳过: %s", root)
            continue
        try:
            _lstat_identity(str(root_path), expect_dir=True)
        except (OSError, ValueError) as exc:
            logger.error("拒绝不安全递归扫描根 %s: %s", root, exc)
            continue
        # 单次全树遍历 + basename 模式匹配：逐模式 rglob 在大输出库上相当于
        # 多遍全树扫描（旧实现对 9 个模式各扫一遍），改为一遍 os.walk 收集
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames
                if not _is_excluded(os.path.join(dirpath, name))
            ]
            for name in filenames:
                if not any(fnmatch.fnmatch(name, pat) for pat in _TMP_FILE_PATTERNS):
                    continue
                item = Path(dirpath) / name
                try:
                    if (now - item.stat().st_mtime) > min_age_sec:
                        try:
                            _register_scan_candidate(str(item), expect_dir=False)
                        except (OSError, ValueError) as exc:
                            logger.error("拒绝不安全清理候选 %s: %s", item, exc)
                            continue
                        residuals.append(item)
                except OSError:
                    # stat 前被并发删除：跳过而非 traceback
                    continue

    return residuals


def _promoted_audio_target(stage_d_dir: str) -> Tuple[Optional[str], str]:
    """从已验证 marker v4 解析当前 run 可清理的最终子阶段 audio 根。"""
    valid, reason, marker = read_validated_promote_marker(stage_d_dir)
    if not valid or marker is None:
        return None, reason
    return os.path.join(marker["stage_d_dir"], marker["final_stage"], "audio"), "ok"


def _journal_is_unfinished(path: str) -> Tuple[bool, str]:
    """保守判断 promote journal；不可读或状态未知均视为未完成。"""
    try:
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            return True, f"promote journal 不是安全 regular file: {path}"
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        return True, f"promote journal 不可安全读取: {path}（{exc}）"
    if not isinstance(payload, dict):
        return True, f"promote journal 必须是 JSON object: {path}"
    states = [payload.get(key) for key in ("status", "state", "phase")]
    completed = {"complete", "completed", "committed", "done"}
    if any(isinstance(value, str) and value.lower() in completed for value in states):
        return False, "ok"
    return True, f"promote journal 尚未完成: {path}"


def _find_unfinished_promote_journal(stage_d_dir: str, final_root: str) -> Optional[str]:
    """检查 promoter 固定 journal；存在未完成事务即拒绝 cleanup。"""
    journal_path = os.path.join(stage_d_dir, PROMOTE_JOURNAL_NAME)
    if not os.path.lexists(journal_path):
        return None
    unfinished, reason = _journal_is_unfinished(journal_path)
    return reason if unfinished else None


def _inspect_audio_tree(path: str) -> Tuple[Tuple[int, int], float]:
    """校验整棵音频树无 symlink/特殊节点，并返回根 identity 与最新 mtime。"""
    absolute = os.path.abspath(path)
    if absolute != os.path.realpath(absolute):
        raise ValueError(f"音频树路径不得包含符号链接: {path}")
    root_stat = os.lstat(absolute)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"音频树必须是目录: {path}")
    latest = root_stat.st_mtime
    walk_errors: List[OSError] = []

    def _on_walk_error(error: OSError) -> None:
        walk_errors.append(error)

    for dirpath, dirnames, filenames in os.walk(
        absolute, topdown=True, followlinks=False, onerror=_on_walk_error
    ):
        for name in dirnames + filenames:
            child = os.path.join(dirpath, name)
            child_stat = os.lstat(child)
            if stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"音频树内不得包含符号链接: {child}")
            if name in dirnames and not stat.S_ISDIR(child_stat.st_mode):
                raise ValueError(f"音频树目录项类型异常: {child}")
            if name in filenames and not stat.S_ISREG(child_stat.st_mode):
                raise ValueError(f"音频树文件项必须是 regular file: {child}")
            latest = max(latest, child_stat.st_mtime)
    if walk_errors:
        raise walk_errors[0]
    return (root_stat.st_dev, root_stat.st_ino), latest


def _require_marker_final_asset(marker: Dict[str, Any]) -> None:
    """确认 marker 绑定的 final stats 是无 symlink 的 regular file。"""
    final_root = marker["final_root"]
    stats_path = os.path.join(final_root, "library_stats.json")
    if stats_path != os.path.realpath(stats_path):
        raise ValueError(f"marker 绑定的 final 资产路径不得包含符号链接: {stats_path}")
    stats = os.lstat(stats_path)
    if not stat.S_ISREG(stats.st_mode):
        raise ValueError(f"marker 绑定的 final 资产必须是 regular file: {stats_path}")


def _validate_intermediate_pair(
    stage_d_dir: str,
    marker: Dict[str, Any],
    min_age_hours: float,
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """在锁内复验双树 identity、年龄与 symlink 契约。"""
    output_root = os.path.dirname(marker["stage_d_dir"])
    paths = [
        os.path.join(output_root, "stage_b", "audio"),
        os.path.join(marker["stage_d_dir"], marker["final_stage"], "audio"),
    ]
    if os.path.realpath(stage_d_dir) != marker["stage_d_dir"]:
        raise ValueError("promote marker 不属于当前 cleanup run")
    if os.path.commonpath(tuple(os.path.realpath(path) for path in paths)) in {
        os.path.realpath(path) for path in paths
    }:
        raise ValueError("Stage B 与 Stage D 音频树不得重叠")
    identities: Dict[str, Tuple[int, int]] = {}
    threshold = time.time() - min_age_hours * 3600
    for path in paths:
        identity, latest = _inspect_audio_tree(path)
        if latest >= threshold:
            raise ValueError(
                f"中间音频子树年龄不足 {min_age_hours:.1f} 小时: {path}"
            )
        identities[path] = identity
    _require_marker_final_asset(marker)
    return paths, identities


def _identity_matches(path: str, expected: Tuple[int, int]) -> bool:
    try:
        current = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == expected


# 保留版本能力探针供回归测试与诊断使用；实际删除统一走下方自有 dirfd 实现。
_RMTREE_SUPPORTS_DIR_FD = "dir_fd" in inspect.signature(shutil.rmtree).parameters
_DIR_FD_REMOVE_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.rmdir in os.supports_dir_fd
)


def _identity_from_stat(info: os.stat_result) -> Tuple[int, int, int, int, int]:
    """把 stat_result 规范化为扫描与最终边界共用的 identity。"""
    return stat.S_IFMT(info.st_mode), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _subtree_latest_mtime_fd(directory_fd: int) -> float:
    """只通过已打开目录 fd 计算子树最新 mtime；遍历竞态一律抛错。"""
    root = os.fstat(directory_fd)
    if not stat.S_ISDIR(root.st_mode):
        raise RuntimeError("候选 fd 不再是目录")
    latest = root.st_mtime
    for entry in os.scandir(directory_fd):
        child = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
        latest = max(latest, child.st_mtime)
        if not stat.S_ISDIR(child.st_mode):
            continue
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        child_fd = os.open(entry.name, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(child_fd)
            if (opened.st_dev, opened.st_ino) != (child.st_dev, child.st_ino):
                raise RuntimeError(f"子目录 identity 在遍历时变化: {entry.name}")
            latest = max(latest, _subtree_latest_mtime_fd(child_fd))
        finally:
            os.close(child_fd)
    return latest


def _rmtree_via_dir_fd(
    name: str,
    parent_fd: int,
    expected_identity: Optional[Tuple[int, int]] = None,
) -> None:
    """通过 dir_fd 递归删除目录树，并复验打开后的目录 identity。"""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    target_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(target_fd)
        if expected_identity is not None and (
            opened.st_dev, opened.st_ino
        ) != expected_identity:
            raise RuntimeError(f"待删除目录 identity 已变化: {name}")
        for entry in os.scandir(target_fd):
            child = os.stat(entry.name, dir_fd=target_fd, follow_symlinks=False)
            if stat.S_ISDIR(child.st_mode):
                _rmtree_via_dir_fd(
                    entry.name, target_fd, (child.st_dev, child.st_ino)
                )
            else:
                # 文件与符号链接都直接 unlink，不会跟随链接进入外部目录
                os.unlink(entry.name, dir_fd=target_fd)
    finally:
        os.close(target_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _restore_quarantines(
    moved: List[Tuple[str, str, Tuple[int, int]]],
) -> Optional[str]:
    """按逆序恢复已隔离树；任一步失败时返回错误。"""
    errors: List[str] = []
    for source, quarantine, identity in reversed(moved):
        try:
            if os.path.lexists(source) or not _identity_matches(quarantine, identity):
                raise RuntimeError("源已存在或 quarantine identity 已变化")
            os.rename(quarantine, source)
            if not _identity_matches(source, identity):
                raise RuntimeError("恢复后的源 identity 不匹配")
        except (OSError, RuntimeError) as exc:
            errors.append(f"{quarantine} -> {source}: {exc}")
    return "; ".join(errors) if errors else None


def _remove_owned_quarantine_at(
    name: str,
    parent_fd: int,
    identity: Tuple[int, int],
    display_path: str,
) -> None:
    """通过调用方持有的稳定 parent dirfd 删除 owned 目录 quarantine。"""
    if not _DIR_FD_REMOVE_SUPPORTED:
        raise RuntimeError("当前平台不支持 symlink-safe rmtree")
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != identity:
        raise RuntimeError(f"quarantine identity 已变化: {display_path}")
    _rmtree_via_dir_fd(name, parent_fd, identity)


def _remove_owned_quarantine(path: str, identity: Tuple[int, int]) -> None:
    """通过安全 parent dirfd 删除本次创建且 identity 未变化的 quarantine。"""
    parent, name = os.path.split(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    parent_fd = os.open(parent, flags)
    try:
        _remove_owned_quarantine_at(name, parent_fd, identity, path)
    finally:
        os.close(parent_fd)


def _purge_intermediate_audio(
    stage_d_dir: str,
    min_age_hours: float,
    dry_run: bool,
) -> int:
    """在 run→final 双锁内验证并事务隔离 Stage B/Stage D 两棵音频树。"""
    try:
        with stage_d_run_lock(stage_d_dir):
            valid, reason, marker = read_validated_promote_marker(stage_d_dir)
            if not valid or marker is None:
                logger.error("拒绝清理中间音频：promote marker 校验失败（%s）", reason)
                return 0
            promote_lock_fd = _acquire_promote_lock(marker["final_root"])
            try:
                valid, reason, locked_marker = read_validated_promote_marker(stage_d_dir)
                if not valid or locked_marker is None or locked_marker != marker:
                    logger.error("拒绝清理中间音频：锁内 marker 复验失败（%s）", reason)
                    return 0
                journal_error = _find_unfinished_promote_journal(
                    locked_marker["stage_d_dir"], locked_marker["final_root"]
                )
                if journal_error:
                    logger.error("拒绝清理中间音频：%s", journal_error)
                    return 0
                paths, identities = _validate_intermediate_pair(
                    stage_d_dir, locked_marker, min_age_hours
                )
                if dry_run:
                    for path in paths:
                        logger.info("[dry-run] 将隔离并删除: %s", path)
                    return len(paths)

                moved: List[Tuple[str, str, Tuple[int, int]]] = []
                try:
                    for source in paths:
                        identity = identities[source]
                        if not _identity_matches(source, identity):
                            raise RuntimeError(f"隔离前源路径 identity 已变化: {source}")
                        quarantine = os.path.join(
                            os.path.dirname(source),
                            f".{os.path.basename(source)}.cleanup-{uuid.uuid4().hex}.quarantine",
                        )
                        if os.path.lexists(quarantine):
                            raise RuntimeError(f"owned quarantine 已存在: {quarantine}")
                        os.rename(source, quarantine)
                        moved.append((source, quarantine, identity))
                        if os.path.lexists(source) or not _identity_matches(
                            quarantine, identity
                        ):
                            raise RuntimeError(
                                f"隔离时检测到路径替换 TOCTOU: {source}"
                            )
                except (OSError, RuntimeError) as exc:
                    restore_error = _restore_quarantines(moved)
                    logger.error("双树隔离失败，已尝试恢复: %s", exc)
                    if restore_error:
                        logger.critical("双树恢复失败，需要人工处理: %s", restore_error)
                    return 0

                removed = 0
                for _source, quarantine, identity in moved:
                    try:
                        _remove_owned_quarantine(quarantine, identity)
                        logger.info("已删除 owned quarantine: %s", quarantine)
                        removed += 1
                    except (OSError, RuntimeError) as exc:
                        logger.error("owned quarantine 删除失败 %s: %s", quarantine, exc)
                return removed
            finally:
                _release_promote_lock(promote_lock_fd)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("拒绝清理中间音频：%s", exc)
        return 0


def _find_intermediate_audio(output_root: str, min_age_hours: float) -> List[Path]:
    """兼容候选发现调用；仅完整有效 marker 且双树均过期时返回两棵树。"""
    stage_d_dir = os.path.join(output_root, "stage_d")
    valid, _reason, marker = read_validated_promote_marker(stage_d_dir)
    if not valid or marker is None:
        return []
    try:
        paths, _identities = _validate_intermediate_pair(
            stage_d_dir, marker, min_age_hours
        )
    except (OSError, ValueError):
        return []
    return [Path(path) for path in paths]


def _remove_owned_file_quarantine_at(
    name: str,
    parent_fd: int,
    identity: Tuple[int, int, int, int, int],
    display_path: str,
) -> None:
    """通过调用方持有的稳定 parent dirfd 删除 owned 文件 quarantine。"""
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode):
        raise RuntimeError(f"quarantine 类型已变化: {display_path}")
    if _identity_from_stat(current) != identity:
        raise RuntimeError(f"quarantine identity 已变化: {display_path}")
    os.unlink(name, dir_fd=parent_fd)


def _remove_residuals(
    paths: List[Path],
    dry_run: bool,
    protected_d2_stage_d: Optional[str] = None,
    min_age_hours: Optional[float] = None,
) -> int:
    """删除普通残留；最终检查、rename 与恢复均锚定稳定 parent dirfd。"""
    removed = 0
    min_age_sec = min_age_hours * 3600 if min_age_hours is not None else None
    protected: Set[str] = set()
    if protected_d2_stage_d:
        output_root = os.path.dirname(os.path.abspath(protected_d2_stage_d))
        protected = {
            os.path.abspath(os.path.join(output_root, "stage_b", "audio")),
            *{
                os.path.abspath(os.path.join(protected_d2_stage_d, stage_name, "audio"))
                for stage_name in ("D1", "D2")
            },
        }

    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    for path in paths:
        absolute = os.path.abspath(path)
        if any(
            os.path.commonpath((absolute, protected_root)) == protected_root
            for protected_root in protected
        ):
            logger.error("拒绝单独删除受保护音频树或其子项: %s", path)
            continue

        parent, name = os.path.split(absolute)
        quarantine_name = f".{name}.cleanup-{uuid.uuid4().hex}.quarantine"
        quarantine = os.path.join(parent, quarantine_name)
        parent_fd: Optional[int] = None
        candidate_fd: Optional[int] = None
        renamed = False
        try:
            if not _DIR_FD_REMOVE_SUPPORTED:
                raise RuntimeError("当前平台不支持稳定 dirfd cleanup")
            parent_fd = os.open(parent, parent_flags)
            parent_stat = os.fstat(parent_fd)
            expected_parent = _SCAN_PARENT_IDENTITIES.get(absolute)
            if expected_parent is not None and (
                parent_stat.st_dev, parent_stat.st_ino
            ) != expected_parent:
                raise RuntimeError("候选 parent identity 已变化")

            entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not (stat.S_ISREG(entry_stat.st_mode) or stat.S_ISDIR(entry_stat.st_mode)):
                raise RuntimeError("候选必须是 regular file 或目录，且不得是 symlink")
            current = _identity_from_stat(entry_stat)
            expected = _SCAN_IDENTITIES.get(absolute)
            if expected is not None and current != expected:
                raise RuntimeError("候选 identity 已变化")
            is_dir = stat.S_ISDIR(entry_stat.st_mode)

            candidate_flags = (
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
            )
            if is_dir:
                candidate_flags |= os.O_DIRECTORY
            candidate_fd = os.open(name, candidate_flags, dir_fd=parent_fd)
            opened = os.fstat(candidate_fd)
            if _identity_from_stat(opened) != current:
                raise RuntimeError("候选 identity 在 open 时变化")
            latest = (
                _subtree_latest_mtime_fd(candidate_fd)
                if is_dir
                else opened.st_mtime
            )
            if min_age_sec is not None and (time.time() - latest) <= min_age_sec:
                raise RuntimeError(
                    f"最终边界年龄不足 {min_age_hours:.1f} 小时"
                )
            if dry_run:
                logger.info("[dry-run] 将删除: %s", path)
                removed += 1
                continue

            try:
                os.stat(quarantine_name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError(f"owned quarantine 已存在: {quarantine}")
            os.rename(
                name,
                quarantine_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            renamed = True
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("隔离后源路径仍存在")

            quarantine_stat = os.stat(
                quarantine_name, dir_fd=parent_fd, follow_symlinks=False
            )
            final_fd_stat = os.fstat(candidate_fd)
            final_identity = _identity_from_stat(final_fd_stat)
            if _identity_from_stat(quarantine_stat) != final_identity:
                raise RuntimeError("quarantine 与候选 fd identity 不一致")
            if expected is not None and final_identity != expected:
                raise RuntimeError("候选 identity 在隔离时变化")
            final_latest = (
                _subtree_latest_mtime_fd(candidate_fd)
                if is_dir
                else final_fd_stat.st_mtime
            )
            if min_age_sec is not None and (time.time() - final_latest) <= min_age_sec:
                raise RuntimeError(
                    f"隔离后年龄不足 {min_age_hours:.1f} 小时"
                )

            if is_dir:
                _remove_owned_quarantine_at(
                    quarantine_name,
                    parent_fd,
                    (final_fd_stat.st_dev, final_fd_stat.st_ino),
                    quarantine,
                )
            else:
                _remove_owned_file_quarantine_at(
                    quarantine_name, parent_fd, final_identity, quarantine
                )
            renamed = False
            logger.info("已隔离并删除: %s", path)
            removed += 1
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("隔离或删除失败 %s: %s", path, exc)
            if renamed and parent_fd is not None and candidate_fd is not None:
                try:
                    try:
                        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise RuntimeError("源路径已被占用")
                    quarantine_stat = os.stat(
                        quarantine_name, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if _identity_from_stat(quarantine_stat) != _identity_from_stat(
                        os.fstat(candidate_fd)
                    ):
                        raise RuntimeError("quarantine identity 已变化")
                    os.rename(
                        quarantine_name,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                except (OSError, RuntimeError):
                    logger.critical("无法恢复 owned quarantine: %s", quarantine)
        finally:
            if candidate_fd is not None:
                os.close(candidate_fd)
            if parent_fd is not None:
                os.close(parent_fd)
    return removed


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    config_path = args.config
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config", "pipeline.yaml")
    config_path = os.path.abspath(config_path)

    if not os.path.isfile(config_path):
        logger.error("配置文件不存在: %s", config_path)
        return 1

    roots, recursive_roots, output_root = _collect_scan_roots(config_path)

    try:
        lock_fd = _acquire_lock(output_root)
    except RuntimeError as e:
        logger.error("%s", e)
        return 2

    logger.info("扫描目录: %s", sorted(roots))
    logger.info("递归扫描目录: %s", sorted(recursive_roots))

    try:
        stage_d_dir = os.path.join(output_root, "stage_d")
        # 通用 tmp 扫描永不进入 Stage B 或 Stage D 音频树；这些树只能由
        # 已验证 marker 授权的 purge-intermediate 双树事务整体处理。
        excluded_intermediate_audio = {
            os.path.join(output_root, "stage_b", "audio"),
            *{
                os.path.join(stage_d_dir, stage_name, "audio")
                for stage_name in ("D1", "D2")
            },
        }
        residuals = _find_residuals(
            roots,
            args.min_age_hours,
            recursive_roots=recursive_roots,
            excluded_subtrees=excluded_intermediate_audio,
        )
        logger.info("发现 %d 个超过 %.1f 小时的普通残留", len(residuals), args.min_age_hours)

        removed = _remove_residuals(
            residuals,
            args.dry_run,
            protected_d2_stage_d=stage_d_dir,
            min_age_hours=args.min_age_hours,
        )
        if args.purge_intermediate:
            removed += _purge_intermediate_audio(
                stage_d_dir,
                args.min_age_hours,
                args.dry_run,
            )
        logger.info("%s完成，清理 %d 个残留", "[dry-run] " if args.dry_run else "", removed)
    finally:
        _release_lock(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
