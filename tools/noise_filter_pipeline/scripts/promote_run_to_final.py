#!/usr/bin/env python3
"""把一次运行的 Stage D 最终产物 promote 到 final/（音频单份原则）。

语义：
  - 完成事实：只从 ``<stage_d>/d_completion.json`` 读取最终子阶段及 exact manifest
    集合，并复验其绑定的 ``d_stats.json``；显式 config 仅做一致性复验。兼容参数
    ``--force`` 不能绕过 completion、绑定内容或零候选门禁；
  - manifest：任何移动或 final 写入前 exact 解析 run/final ``AudioRecord``，再复用
    strict JSONL schema v6 构造器验证 C1/C2/C3/D2、execution contract、policy v4、
    evidence v5 与 model/label-space identity；正式 ``dataset_label`` 的路径、
    provenance、命名空间和配置扩展名对候选及已有 final 记录执行同一门禁。旧 C4/C/D、
    未知字段和旧 schema 均 fail-closed；验证后过滤 D2 passed、改写 ``audio.path``，
    并按 ``record_id`` 合并；
  - 事务：全部候选全局预检后，以 owned staging、backup 和持久 journal 提交音频、
    manifest 与 stats。journal 绑定预提交 source fingerprint 及 source quarantine_root，
    并在恢复复制前持久化 pending 状态、复制后持久化新 identity；连续恢复失败后仍可
    重试，未知 source 替换保持 fail-closed。source 缺失时只接受内容匹配的
    quarantine/destination/staged 副本；
  - 原子性：复制使用目标同目录临时文件，完整校验后 ``fsync``、``os.replace`` 并
    同步父目录；quarantine 恢复使用 no-replace hard link，目标竞争时保留证据；失败
    保留 journal 并阻断新 promote/cleanup。marker v4 最后发布；
  - 并发：固定先持有 stage_d run 排它锁，再持有 final-root 排它锁；两者以 realpath
    为 identity，符号链接别名共用锁，冲突直接失败。

用法：
  python tools/noise_filter_pipeline/scripts/promote_run_to_final.py \
      --stage-d-dir runs/<run_id>/stage_d \
      --final-root runs/final [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import logging
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
for _path in (_REPO_ROOT, _TOOLS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from noise_filter_pipeline.core.manifest_io import (  # noqa: E402
    AudioRecord,
    config_hash,
    parse_audio_record_exact,
    read_manifest_strict,
    write_manifest,
)
from noise_filter_pipeline.core.manifest_per_dataset import validate_dataset_id  # noqa: E402
from core.data.noise_admission import NoiseAdmissionContext  # noqa: E402
from core.data.noise_manifest_adapter import (  # noqa: E402
    NOISE_SOURCE_PARSER_VERSION,
    read_file_bytes_snapshot,
)
from noise_filter_pipeline.core.pipeline_config import (  # noqa: E402
    UnifiedPipelineConfig,
    resolve_stage_d_final_stage,
)
from noise_filter_pipeline.core.run_lock import (  # noqa: E402
    acquire_lock,
    acquire_run_lock,
    promote_lock_path,
    release_lock,
    release_run_lock,
)
from noise_filter_pipeline.core.library_stats import LibraryStats  # noqa: E402
from noise_filter_pipeline.core.stage_d_noise_manifest_contract import (  # noqa: E402
    GENERATOR_VERSION,
    build_validated_noise_record,
)
from noise_filter_pipeline.utils.sanitize import sanitize_basename_for_rename  # noqa: E402

logger = logging.getLogger(__name__)

PROMOTE_MARKER_NAME = ".promote_complete.json"
PROMOTE_MARKER_VERSION = 4
PROMOTE_JOURNAL_NAME = ".promote_transaction.json"
_COMPARE_CHUNK = 1 << 20

_PROMOTE_LOCK_PREFIX = "pipeline_promote_"
_EXPECTED_MANIFEST_SCHEMA_VERSION = 6
_EXPECTED_ADMISSION_POLICY_VERSION = 4
_EXPECTED_EVIDENCE_CONTRACT_VERSION = 5
_EXPECTED_PARSER_VERSION = 10
_EXPECTED_GENERATOR_VERSION = "6.1"
_JOURNAL_VERSION = 2
_STAGE_D_COMPLETION_NAME = "d_completion.json"
_STAGE_D_COMPLETION_SCHEMA = "noise_filter_stage_d_completion"
_STAGE_D_COMPLETION_VERSION = 1
_STAGE_D_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "version",
        "final_stage",
        "manifest_prefix",
        "stage_d_config_hash",
        "d2_passed",
        "manifest_files",
        "d_stats",
    }
)
_LEGACY_PROMOTE_MARKER_FIELDS = frozenset(
    {
        "admission_calibration_version",
        "admission_calibration_sha256",
        "calibration_approved",
        "calibration_status",
        "approved",
        "approval_blockers",
        "review_required",
        "approval_bundle",
        "approval_bundle_sha256",
        "attestation",
        "attestation_sha256",
        "threshold_hits",
        "content_kind",
        "margin",
        "quarantine",
        "reason_codes",
        "risk",
        "max_risk",
        "sustain",
    }
)
_DEFAULT_CONFIG_PATH = os.path.join(
    _TOOLS_DIR, "noise_filter_pipeline", "config", "pipeline.yaml"
)


def _resolve_final_stage_output(
    stage_d_dir: str, config_path: str | None
) -> Tuple[str, str, str]:
    """读取 Stage D completion fact，并解析正式 D2 output_format。"""
    stage_d_dir = os.path.abspath(stage_d_dir)
    completion_path = os.path.join(stage_d_dir, _STAGE_D_COMPLETION_NAME)
    snapshot = read_file_bytes_snapshot(completion_path)
    try:
        completion = json.loads(snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Stage D completion fact 非法: {completion_path}: {exc}") from exc
    if not isinstance(completion, dict) or set(completion) != _STAGE_D_COMPLETION_FIELDS:
        raise ValueError("Stage D completion fact 字段必须与受支持 schema 精确一致")
    if (
        completion.get("schema") != _STAGE_D_COMPLETION_SCHEMA
        or completion.get("version") != _STAGE_D_COMPLETION_VERSION
    ):
        raise ValueError("Stage D completion fact schema/version 不受支持")
    stage_name = completion.get("final_stage")
    manifest_prefix = completion.get("manifest_prefix")
    if (stage_name, manifest_prefix) not in {("D1", "d1"), ("D2", "d2")}:
        raise ValueError("Stage D completion fact 的 final_stage/manifest_prefix 非法")
    config_digest = completion.get("stage_d_config_hash")
    if not isinstance(config_digest, str) or re.fullmatch(r"[0-9a-f]{12}", config_digest) is None:
        raise ValueError("Stage D completion fact 的 stage_d_config_hash 非法")
    d2_passed = completion.get("d2_passed")
    if type(d2_passed) is not int or d2_passed <= 0:
        raise ValueError("Stage D completion fact 的 d2_passed 必须是严格正整数")

    expected_manifests = completion.get("manifest_files")
    if not isinstance(expected_manifests, list) or not expected_manifests:
        raise ValueError("Stage D completion fact 的 manifest_files 必须是非空列表")
    normalized_manifests = []
    for item in expected_manifests:
        if not isinstance(item, dict) or set(item) != {"name", "size", "sha256"}:
            raise ValueError("Stage D completion fact 的 manifest_files 项非法")
        name, size, digest = item["name"], item["size"], item["sha256"]
        filename_prefix = f"{manifest_prefix}_manifest_"
        if (
            not isinstance(name, str)
            or not name.startswith(filename_prefix)
            or not name.endswith(".jsonl")
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("Stage D completion fact 的 manifest_files 项非法")
        validate_dataset_id(name[len(filename_prefix) : -len(".jsonl")])
        normalized_manifests.append({"name": name, "size": size, "sha256": digest})
    if normalized_manifests != sorted(normalized_manifests, key=lambda item: item["name"]):
        raise ValueError("Stage D completion fact 的 manifest_files 必须按文件名排序")
    if len({item["name"] for item in normalized_manifests}) != len(normalized_manifests):
        raise ValueError("Stage D completion fact 的 manifest_files 不得重复")

    stage_output_dir = os.path.join(stage_d_dir, stage_name)
    actual_manifests = []
    for path in sorted(glob(os.path.join(stage_output_dir, f"{manifest_prefix}_manifest_*.jsonl"))):
        content = _content_fingerprint(path)
        actual_manifests.append(
            {"name": os.path.basename(path), "size": content["size"], "sha256": content["sha256"]}
        )
    if actual_manifests != normalized_manifests:
        raise ValueError("Stage D completion fact 绑定的 exact manifest 集合或内容已变化")

    expected_stats = completion.get("d_stats")
    if not isinstance(expected_stats, dict) or set(expected_stats) != {"size", "sha256"}:
        raise ValueError("Stage D completion fact 的 d_stats 项非法")
    actual_stats = _content_fingerprint(os.path.join(stage_d_dir, "d_stats.json"))
    if (
        not actual_stats["exists"]
        or expected_stats != {"size": actual_stats["size"], "sha256": actual_stats["sha256"]}
    ):
        raise ValueError("Stage D completion fact 绑定的 d_stats.json 已变化或缺失")

    config = UnifiedPipelineConfig.from_yaml(
        config_path or _DEFAULT_CONFIG_PATH, active_stages=()
    )
    stage_cfg = config.get_stage_config("D")
    configured_stage, configured_prefix = resolve_stage_d_final_stage(stage_cfg)
    if (configured_stage, configured_prefix) != (stage_name, manifest_prefix):
        if config_path is not None:
            raise ValueError("显式配置与 Stage D completion fact 的最终子阶段不一致")
        raise ValueError("默认 Stage D 配置与 completion fact 的最终子阶段不一致")
    if config_path is not None and config_hash(stage_cfg) != config_digest:
        raise ValueError("显式配置与 Stage D completion fact 的 stage_d_config_hash 不一致")
    d2_config = stage_cfg.get("d2", {})
    output_format = d2_config.get("output_format")
    if (
        not isinstance(output_format, str)
        or not output_format
        or output_format != output_format.strip()
        or re.fullmatch(r"[A-Za-z0-9]+", output_format) is None
    ):
        raise ValueError("Stage D d2.output_format 必须是安全文件扩展名 token")
    return stage_output_dir, manifest_prefix, output_format.lower()


def _reject_symlink_ancestors(path: str, *, allow_missing_leaf: bool = True) -> None:
    """拒绝路径任一级既有 symlink 或非目录 ancestor。"""
    absolute = os.path.abspath(path)
    missing_allowed = allow_missing_leaf
    current = absolute
    components: List[str] = []
    while True:
        parent, name = os.path.split(current)
        if name:
            components.append(name)
            current = parent
            continue
        root = parent or current
        break
    probe = root
    for index, name in enumerate(reversed(components)):
        probe = os.path.join(probe, name)
        is_leaf = index == len(components) - 1
        if not os.path.lexists(probe):
            if is_leaf and missing_allowed:
                return
            missing_allowed = True
            continue
        mode = os.lstat(probe).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"路径祖先不得是符号链接: {probe}")
        if not is_leaf and not stat.S_ISDIR(mode):
            raise ValueError(f"路径祖先必须是目录: {probe}")


# ---------------------------------------------------------------------------
# final-root 粒度并发锁（统一 registry 与 run_lock）
# ---------------------------------------------------------------------------
def _promote_lock_path(final_root: str) -> str:
    """按 final_root realpath 返回持久锁路径，symlink alias 共锁。"""
    return promote_lock_path(final_root)


def _acquire_promote_lock(final_root: str) -> int:
    """获取 final_root 排它锁；调用方必须先持有对应 stage_d run 锁。"""
    lock_path = _promote_lock_path(final_root)
    try:
        return acquire_lock(lock_path, purpose="promote 锁")
    except RuntimeError as exc:
        raise RuntimeError(
            f"无法获取 promote 锁 {lock_path}，可能已有其他 promote 在运行: {exc}"
        ) from exc


def _release_promote_lock(fd: int) -> None:
    release_lock(fd)


# ---------------------------------------------------------------------------
# 单文件移动（rename 优先，跨盘 copy+校验+删源）
# ---------------------------------------------------------------------------
def _same_content_exact(a: str, b: str) -> bool:
    """分块完整比较两个文件；读取错误由调用方按失败处理。"""
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            chunk_a = fa.read(_COMPARE_CHUNK)
            chunk_b = fb.read(_COMPARE_CHUNK)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def _move_file(src: str, dst: str, dry_run: bool) -> Optional[str]:
    """把 src 移动到 dst，返回错误信息（None 表示成功/无需移动）。

    幂等语义见模块 docstring。dst 已存在且内容不一致时，把 src 先原子
    搬入 dst 同目录临时文件（同盘 rename / 跨盘 copy+完整校验），再
    os.replace 原子替换 dst——任意时刻 dst 要么是旧内容要么是新内容，
    不存在"先删旧 dst 再失败"造成的缺失窗口。dry-run 只做校验不落任何变更。
    """
    src_exists = os.path.isfile(src)
    dst_exists = os.path.isfile(dst)
    if dst_exists:
        if not src_exists:
            return (
                "source 缺失时不能仅凭 destination 存在判定幂等；"
                f"必须由完整 marker/journal SHA 证明: {src} vs {dst}"
            )
        try:
            contents_equal = _same_content_exact(src, dst)
        except OSError as e:
            return f"src/dst 完整内容读取失败，均保留: {src} vs {dst}（{e}）"
        if not contents_equal:
            # 内容不一致：新 run 的音频覆盖旧 dst。先原子搬入同目录临时文件，
            # 校验后 os.replace 原子替换，避免"删旧 dst 后失败"的缺失窗口。
            if dry_run:
                return None
            logger.warning("dst 内容不一致，将用新 run 音频原子替换: %s", dst)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            fd, tmp_dst = tempfile.mkstemp(
                prefix=f".{os.path.basename(dst)}.", suffix=".tmp", dir=os.path.dirname(dst)
            )
            os.close(fd)
            src_moved = False
            try:
                try:
                    os.rename(src, tmp_dst)
                    src_moved = True
                except OSError as e:
                    if e.errno != errno.EXDEV:
                        return f"替换失败: {src} → {dst}（{e}）"
                    source_fingerprint = _file_fingerprint(src)
                    shutil.copy2(src, tmp_dst)
                    if not _same_content_exact(src, tmp_dst):
                        return f"跨盘替换拷贝内容校验失败: {src} → {dst}"
                os.replace(tmp_dst, dst)
                tmp_dst = ""
                if not src_moved:
                    _unlink_source_if_unchanged(
                        src,
                        source_fingerprint,
                        transaction_id=uuid.uuid4().hex,
                    )
                return None
            except (OSError, RuntimeError, ValueError) as e:
                return f"跨盘替换失败: {src} → {dst}（{e}）"
            finally:
                if tmp_dst:
                    try:
                        os.remove(tmp_dst)
                    except OSError:
                        logger.exception("无法删除替换临时文件: %s", tmp_dst)
        # 同一批次的重复残留（跨盘 copy 中断等）：删 src
        logger.warning("dst 已存在且内容一致，删除 src 残留: %s", src)
        if not dry_run:
            try:
                _unlink_source_if_unchanged(
                    src,
                    _file_fingerprint(src),
                    transaction_id=uuid.uuid4().hex,
                )
            except (OSError, RuntimeError, ValueError) as e:
                return f"删除 src 残留失败: {src}（{e}）"
        return None
    if not src_exists:
        return f"passed 记录引用的音频缺失: {src}"
    if dry_run:
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.rename(src, dst)
        return None
    except OSError as e:
        if e.errno != errno.EXDEV:
            return f"移动音频失败: {src} → {dst}（{e}）"

    # 跨盘：先复制到 dst 同目录临时文件，完整校验后才原子发布正式 dst。
    fd, tmp_dst = tempfile.mkstemp(
        prefix=f".{os.path.basename(dst)}.", suffix=".tmp", dir=os.path.dirname(dst)
    )
    os.close(fd)
    try:
        source_fingerprint = _file_fingerprint(src)
        shutil.copy2(src, tmp_dst)
        if not _same_content_exact(src, tmp_dst):
            return f"跨盘拷贝内容校验失败: {src} → {dst}"
        os.replace(tmp_dst, dst)
        tmp_dst = ""
        _unlink_source_if_unchanged(
            src,
            source_fingerprint,
            transaction_id=uuid.uuid4().hex,
        )
        return None
    except (OSError, RuntimeError, ValueError) as e:
        return f"跨盘拷贝失败: {src} → {dst}（{e}）"
    finally:
        if tmp_dst:
            try:
                os.remove(tmp_dst)
            except OSError:
                logger.exception("无法删除跨盘拷贝临时文件: %s", tmp_dst)


# ---------------------------------------------------------------------------
# 路径改写（run audio_root → final audio_root，相对布局原样保留）
# ---------------------------------------------------------------------------
def _normalize_absolute_path(path: str) -> str:
    """返回用于路径所有权比较的规范化绝对路径。"""
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def _final_path(src: str, run_audio_root: str, final_audio_root: str) -> Tuple[Optional[str], Optional[str]]:
    """计算 final 目标路径，并拒绝源/目标树中的 symlink escape。"""
    if not isinstance(src, str) or not src:
        return None, "passed 记录的音频路径必须是非空字符串"
    try:
        _reject_symlink_ancestors(src, allow_missing_leaf=True)
    except (OSError, ValueError) as exc:
        return None, f"passed 记录的源音频路径非法: {exc}"
    root_abs = os.path.realpath(run_audio_root)
    src_real = os.path.realpath(src)
    try:
        if os.path.commonpath((src_real, root_abs)) != root_abs:
            return None, f"passed 记录的音频不在 run 音频根目录下: {src}（根: {run_audio_root}）"
    except ValueError:
        return None, f"passed 记录的音频不在 run 音频根目录下: {src}（根: {run_audio_root}）"
    src_exists = os.path.isfile(src) and stat.S_ISREG(
        os.stat(src, follow_symlinks=False).st_mode
    )
    if os.path.lexists(src) and not src_exists:
        return None, f"passed 记录的源音频必须是 regular file: {src}"
    rel = os.path.relpath(src_real, root_abs)
    dst = os.path.join(final_audio_root, rel)
    final_root_abs = os.path.realpath(final_audio_root)
    dst_abs = os.path.abspath(dst)
    try:
        if os.path.commonpath((dst_abs, final_root_abs)) != final_root_abs:
            return None, f"final 音频目标逃逸 final 音频根目录: {dst}"
    except ValueError:
        return None, f"final 音频目标逃逸 final 音频根目录: {dst}"
    try:
        _reject_symlink_ancestors(dst, allow_missing_leaf=True)
    except (OSError, ValueError) as exc:
        return None, f"final 音频目标路径非法: {exc}"
    if os.path.lexists(dst) and not stat.S_ISREG(os.lstat(dst).st_mode):
        return None, f"final 音频目标必须是 regular file: {dst}"
    return dst, None


def _prune_empty_dirs(root: str) -> int:
    """自底向上删除 root 下的空目录（移动后的空壳），返回删除数。

    直接对每个目录尝试 rmdir（非空即失败跳过），不依赖 os.walk 的
    dirnames 快照——底向上遍历时快照在子目录删除前已生成，靠它判断
    会漏删"因子目录被删而变空"的父目录。
    """
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        try:
            os.rmdir(dirpath)
            removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# manifest 严格读取与合并
# ---------------------------------------------------------------------------
def _validate_audio_file_path(
    audio_path: str,
    root: str,
    *,
    location: str,
    require_exists: bool = True,
) -> bool:
    """验证音频路径位于指定根目录，并拒绝任一级 symlink/escape。"""
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError("audio.path 必须是非空字符串")
    _reject_symlink_ancestors(audio_path, allow_missing_leaf=not require_exists)
    root_real = os.path.realpath(root)
    path_abs = os.path.abspath(audio_path)
    try:
        lexical_inside = os.path.commonpath((path_abs, root_real)) == root_real
    except ValueError:
        lexical_inside = False
    if not lexical_inside:
        raise ValueError(f"audio.path 不在音频根目录下: {audio_path}")
    if os.path.islink(audio_path):
        raise ValueError(f"audio.path 不得是符号链接: {audio_path}")
    resolved = os.path.realpath(audio_path)
    try:
        resolved_inside = os.path.commonpath((resolved, root_real)) == root_real
    except ValueError:
        resolved_inside = False
    if not resolved_inside:
        raise ValueError(f"audio.path 解析后逃逸音频根目录: {audio_path}")
    if not os.path.lexists(audio_path):
        if require_exists:
            raise ValueError(f"audio.path 文件不存在: {audio_path}")
        return False
    if not stat.S_ISREG(os.stat(audio_path, follow_symlinks=False).st_mode):
        raise ValueError(f"audio.path 必须是 regular file: {audio_path}")
    return True


_DATASET_LABEL_NAMING_VERSION = 1
_DATASET_LABEL_SHARD_SIZE = 10000
_DATASET_LABEL_SOURCE_IDENTITY_VERSION = 1
_DATASET_LABEL_SOURCE_IDENTITY_KEY = "dataset_label_source_identity"


def _validate_dataset_label_audio_path(
    rec: AudioRecord,
    audio_path: str,
    audio_root: str,
    *,
    location: str,
    seen_namespaces: Dict[str, Dict[str, str]],
    output_extension: List[str],
    expected_output_format: str,
) -> None:
    """校验正式 dataset_label 的路径、ordinal 和源身份契约。"""
    _validate_audio_file_path(
        audio_path, audio_root, location=location, require_exists=False
    )
    c3 = rec.stages.get("C3") if isinstance(rec.stages, dict) else None
    if not isinstance(c3, dict) or c3.get("status") != "passed":
        raise ValueError("dataset_label audio.path requires passed C3")
    category = c3.get("balance_category")
    if not isinstance(category, str) or not category or category != category.strip():
        raise ValueError("dataset_label audio.path requires C3 balance_category")
    dataset_dir = sanitize_basename_for_rename(rec.dataset_id, is_file=False)
    category_dir = sanitize_basename_for_rename(category, is_file=False)
    for namespace, raw, value in (
        ("dataset", rec.dataset_id, dataset_dir),
        (f"category:{rec.dataset_id}", category, category_dir),
    ):
        previous = seen_namespaces.setdefault(namespace, {}).get(value)
        if previous is not None and previous != raw:
            raise ValueError(
                f"dataset_label {namespace} sanitize collision: {previous!r} and {raw!r} -> {value!r}"
            )
        seen_namespaces[namespace][value] = raw

    provenance = rec.provenance
    if not isinstance(provenance, dict):
        raise ValueError("dataset_label provenance must be an object")
    if provenance.get("dataset_label_naming_version") != _DATASET_LABEL_NAMING_VERSION:
        raise ValueError("dataset_label provenance naming_version 不匹配")
    if provenance.get("dataset_label_shard_size") != _DATASET_LABEL_SHARD_SIZE:
        raise ValueError("dataset_label provenance shard_size 不匹配")
    ordinal = provenance.get("dataset_label_ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("dataset_label provenance ordinal 必须是非负整数")
    source_identity = provenance.get(_DATASET_LABEL_SOURCE_IDENTITY_KEY)
    if not isinstance(source_identity, dict):
        raise ValueError("dataset_label provenance 缺少 source identity")
    if source_identity.get("version") != _DATASET_LABEL_SOURCE_IDENTITY_VERSION:
        raise ValueError("dataset_label source identity version 不匹配")
    if source_identity.get("source_id") != rec.source_id:
        raise ValueError("dataset_label provenance source_id 不匹配")
    if source_identity.get("chunk_id") != rec.chunk_id:
        raise ValueError("dataset_label provenance chunk_id 不匹配")
    if source_identity.get("split") != rec.split:
        raise ValueError("dataset_label provenance split 不匹配")
    source_audio = source_identity.get("audio")
    if (
        not isinstance(source_audio, dict)
        or not isinstance(source_audio.get("path"), str)
        or not source_audio.get("path")
        or not isinstance(source_audio.get("fingerprint"), dict)
    ):
        raise ValueError("dataset_label provenance source audio identity 非法")
    d2 = rec.stages.get("D2")
    if not isinstance(d2, dict) or d2.get("status") != "passed":
        raise ValueError("dataset_label audio.path requires passed D2")
    if d2.get("input_path") != source_audio["path"]:
        raise ValueError("dataset_label provenance source path 与 D2 input_path 不一致")
    if d2.get("input_fingerprint") != source_audio["fingerprint"]:
        raise ValueError("dataset_label provenance source fingerprint 与 D2 不一致")

    relative = os.path.relpath(os.path.abspath(audio_path), os.path.realpath(audio_root))
    parts = relative.split(os.sep)
    if len(parts) != 4 or parts[0] != dataset_dir or parts[1] != category_dir:
        raise ValueError(
            f"audio.path 不匹配 dataset_label dataset/category 契约: {audio_path}"
        )
    shard_match = re.fullmatch(r"shard_(\d{5})", parts[2])
    slot_match = re.fullmatch(r"(\d{4})\.([^./]+)", parts[3])
    if shard_match is None or slot_match is None:
        raise ValueError(f"audio.path 不匹配 dataset_label shard/slot 契约: {audio_path}")
    shard = int(shard_match.group(1))
    slot = int(slot_match.group(1))
    extension = slot_match.group(2).lower()
    if extension != expected_output_format:
        raise ValueError(
            f"audio.path 扩展名与 stage_d.d2.output_format 不一致: "
            f"{extension!r} != {expected_output_format!r}"
        )
    if output_extension and output_extension[0] != extension:
        raise ValueError(
            f"dataset_label output extension 不一致: {output_extension[0]!r} != {extension!r}"
        )
    if not output_extension:
        output_extension.append(extension)
    expected_shard, expected_slot = divmod(ordinal, _DATASET_LABEL_SHARD_SIZE)
    if (shard, slot) != (expected_shard, expected_slot):
        raise ValueError(
            f"audio.path shard/slot 与 provenance ordinal 不一致: "
            f"path=({shard},{slot}), expected=({expected_shard},{expected_slot})"
        )


def _load_manifest_strict(
    path: str,
    expected_dataset_id: str,
    *,
    manifest_bytes: bytes | None = None,
    audio_root: str | None = None,
    alternate_audio_root: str | None = None,
    all_records_publishable: bool = False,
    seen_record_ids: set[str] | None = None,
    admission_context: NoiseAdmissionContext | None = None,
) -> List[AudioRecord]:
    """严格读取并验证发布候选；任一坏行或准入失败均立即失败。

    Stage D manifest 只对 D2 passed 候选执行完整准入验证，允许 rejected 记录
    作为运行审计保留；已有 final manifest 的每条记录都必须是合法发布记录。
    """
    records: List[AudioRecord] = []
    unique_ids = seen_record_ids if seen_record_ids is not None else set()
    expected_artifacts: Dict[str, str] | None = None
    if manifest_bytes is None:
        manifest_bytes = read_file_bytes_snapshot(path).data
    with io.StringIO(manifest_bytes.decode("utf-8")) as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            location = f"{path}:{line_number}"
            try:
                data = json.loads(stripped)
                rec = parse_audio_record_exact(
                    data, location=location, require_audio_path=False
                )
                if rec.record_id in unique_ids:
                    raise ValueError(f"duplicate record_id: {rec.record_id!r}")
                unique_ids.add(rec.record_id)
                audio_path = rec.audio.get("path")
                if rec.dataset_id != expected_dataset_id:
                    raise ValueError(
                        f"manifest 文件名 dataset={expected_dataset_id!r} 与记录 "
                        f"dataset_id={rec.dataset_id!r} 不一致"
                    )
                if all_records_publishable or rec.is_stage_passed("D2"):
                    if not isinstance(audio_path, str) or not audio_path.strip():
                        raise ValueError("audio.path 必须是非空字符串")
                    if audio_root is None:
                        raise ValueError("发布记录必须提供 audio_root 进行路径校验")
                    source_exists = _validate_audio_file_path(
                        audio_path,
                        audio_root,
                        location=location,
                        require_exists=alternate_audio_root is None,
                    )
                    if not source_exists and alternate_audio_root is not None:
                        relative = os.path.relpath(
                            os.path.abspath(audio_path), os.path.realpath(audio_root)
                        )
                        _validate_audio_file_path(
                            os.path.join(alternate_audio_root, relative),
                            alternate_audio_root,
                            location=location,
                        )
                    legacy_stages = sorted({"C4", "C", "D"} & set(rec.stages))
                    if legacy_stages:
                        raise ValueError(
                            f"legacy stages are not supported: {legacy_stages!r}"
                        )
                    if not rec.is_stage_passed("C3"):
                        raise ValueError("C3 stage must be passed")
                    operation = admission_context or NoiseAdmissionContext.load()
                    output = build_validated_noise_record(
                        data,
                        location=location,
                        context=operation,
                    )
                    artifacts = output["verification"]["artifact_fingerprints"]
                    if expected_artifacts is None:
                        expected_artifacts = artifacts
                    elif artifacts != expected_artifacts:
                        raise ValueError(
                            "artifact fingerprints differ within one manifest"
                        )
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(f"manifest 非法: {location}: {e}") from e
            records.append(rec)
    return records


def _preflight_manifests(
    stage_output_dir: str,
    manifest_prefix: str,
    final_root: str,
    *,
    admission_context: NoiseAdmissionContext,
    expected_output_format: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], int]:
    """全局预检并冻结 manifest bytes、稳定身份与已验证记录。"""
    datasets: Dict[str, Dict[str, Any]] = {}
    seen_record_ids: set[str] = set()
    candidate_owners: Dict[str, str] = {}
    expected_artifacts: Dict[str, str] | None = None
    run_audio_root = os.path.join(stage_output_dir, "audio")
    final_audio_root = os.path.join(final_root, "audio")
    filename_prefix = f"{manifest_prefix}_manifest_"
    candidate_count = 0
    dataset_label_namespaces: Dict[str, Dict[str, str]] = {"dataset": {}}
    dataset_label_extension: List[str] = []

    def _check_artifact_identity(rec: AudioRecord) -> None:
        nonlocal expected_artifacts
        artifacts = rec.stages["C3"]["artifact_fingerprints"]
        if expected_artifacts is None:
            expected_artifacts = artifacts
        elif artifacts != expected_artifacts:
            raise ValueError("artifact fingerprints differ across publication candidates")

    for path in sorted(glob(os.path.join(stage_output_dir, f"{filename_prefix}*.jsonl"))):
        _reject_symlink_ancestors(path, allow_missing_leaf=False)
        name = os.path.basename(path)
        dataset_id = validate_dataset_id(name[len(filename_prefix):-len(".jsonl")])
        snapshot = read_file_bytes_snapshot(path)
        records = _load_manifest_strict(
            path,
            dataset_id,
            manifest_bytes=snapshot.data,
            audio_root=run_audio_root,
            alternate_audio_root=final_audio_root,
            seen_record_ids=seen_record_ids,
            admission_context=admission_context,
        )
        candidates = [rec for rec in records if rec.is_stage_passed("D2")]
        for rec in candidates:
            candidate_count += 1
            _check_artifact_identity(rec)
            _validate_dataset_label_audio_path(
                rec,
                rec.audio.get("path") or "",
                run_audio_root,
                location=f"{path}:{rec.record_id}",
                seen_namespaces=dataset_label_namespaces,
                output_extension=dataset_label_extension,
                expected_output_format=expected_output_format,
            )
            candidate_owners[rec.record_id] = dataset_id
            _dst, error = _final_path(
                rec.audio.get("path") or "", run_audio_root, final_audio_root
            )
            if error:
                raise ValueError(error)
        if candidates:
            datasets[dataset_id] = {
                "path": path,
                "fingerprint": _file_fingerprint_from_snapshot(snapshot),
                "records": candidates,
            }

    existing: Dict[str, Dict[str, Any]] = {}
    existing_seen_ids: set[str] = set()
    manifests_dir = os.path.join(final_root, "manifests")
    if os.path.isdir(manifests_dir):
        for name in sorted(os.listdir(manifests_dir)):
            if not name.endswith(".jsonl"):
                continue
            dataset_id = validate_dataset_id(name[:-len(".jsonl")])
            path = os.path.join(manifests_dir, name)
            _reject_symlink_ancestors(path, allow_missing_leaf=False)
            snapshot = read_file_bytes_snapshot(path)
            records = _load_manifest_strict(
                path,
                dataset_id,
                manifest_bytes=snapshot.data,
                audio_root=final_audio_root,
                all_records_publishable=True,
                seen_record_ids=existing_seen_ids,
                admission_context=admission_context,
            )
            for rec in records:
                _check_artifact_identity(rec)
                _validate_dataset_label_audio_path(
                    rec,
                    rec.audio.get("path") or "",
                    final_audio_root,
                    location=f"{path}:{rec.record_id}",
                    seen_namespaces=dataset_label_namespaces,
                    output_extension=dataset_label_extension,
                    expected_output_format=expected_output_format,
                )
                candidate_owner = candidate_owners.get(rec.record_id)
                if candidate_owner is not None and candidate_owner != dataset_id:
                    raise ValueError(
                        f"record_id {rec.record_id!r} 跨 dataset 冲突: "
                        f"run={candidate_owner!r}, final={dataset_id!r}"
                    )
            existing[dataset_id] = {
                "path": path,
                "fingerprint": _file_fingerprint_from_snapshot(snapshot),
                "records": records,
            }
    return datasets, existing, candidate_count


def _build_final_target_ownership(
    existing_final: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """从全部 final 记录建立 target path -> record_id 所有权索引。"""
    ownership: Dict[str, str] = {}
    for meta in existing_final.values():
        for rec in meta["records"]:
            target = rec.audio.get("path")
            normalized_target = _normalize_absolute_path(target)
            owner = ownership.get(normalized_target)
            if owner is not None and owner != rec.record_id:
                raise ValueError(
                    "existing final target path 被多个 record_id 占用: "
                    f"{normalized_target}: {owner!r}, {rec.record_id!r}"
                )
            ownership[normalized_target] = rec.record_id
    return ownership


def merge_dataset_manifest(
    final_manifest_path: str,
    new_records: List[AudioRecord],
    dry_run: bool,
    existing_records: Optional[List[AudioRecord]] = None,
) -> int:
    """把 new_records 合并进 final manifest，返回合并后总记录数。"""
    merged: Dict[str, AudioRecord] = {}
    order: List[str] = []
    if existing_records is None and os.path.isfile(final_manifest_path):
        expected_dataset_id = os.path.basename(final_manifest_path)[:-len(".jsonl")]
        final_audio_root = os.path.join(
            os.path.dirname(os.path.dirname(final_manifest_path)), "audio"
        )
        existing_records = _load_manifest_strict(
            final_manifest_path,
            expected_dataset_id,
            audio_root=final_audio_root,
            all_records_publishable=True,
        )
    existing_ids: set[str] = set()
    for rec in existing_records or []:
        if rec.record_id in existing_ids:
            raise ValueError(f"duplicate record_id in existing manifest: {rec.record_id!r}")
        existing_ids.add(rec.record_id)
        order.append(rec.record_id)
        merged[rec.record_id] = rec
    new_ids: set[str] = set()
    for rec in new_records:
        if rec.record_id in new_ids:
            raise ValueError(f"duplicate record_id in new records: {rec.record_id!r}")
        new_ids.add(rec.record_id)
        if rec.record_id not in merged:
            order.append(rec.record_id)
        # 与已有 final 的同 ID 覆盖是显式 promote 语义，不是静默重复。
        merged[rec.record_id] = rec
    if not dry_run:
        os.makedirs(os.path.dirname(final_manifest_path), exist_ok=True)
        write_manifest((merged[rid] for rid in order), final_manifest_path)
    return len(order)


# ---------------------------------------------------------------------------
# 统计聚合（全量重算 final/manifests）
# ---------------------------------------------------------------------------
def _file_fingerprint_from_snapshot(snapshot: Any) -> Dict[str, Any]:
    identity = snapshot.identity
    return {
        "exists": True,
        "size": identity.size,
        "mtime_ns": identity.mtime_ns,
        "ctime_ns": identity.ctime_ns,
        "inode": identity.inode,
        "device": identity.device,
        "sha256": snapshot.sha256,
    }


def _file_fingerprint(path: str) -> Dict[str, Any]:
    """在同一 FD 上返回稳定身份与 SHA-256；变化或读取失败均 fail closed。"""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except FileNotFoundError:
        return {
            "exists": False,
            "size": 0,
            "mtime_ns": 0,
            "ctime_ns": 0,
            "inode": 0,
            "device": 0,
            "sha256": "",
        }
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"fingerprint path must be a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := handle.read(_COMPARE_CHUNK):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    try:
        visible = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError(f"file changed while fingerprinting: {path}") from exc
    identity = lambda value: (
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_ino,
        value.st_dev,
    )
    if (
        not stat.S_ISREG(visible.st_mode)
        or identity(before) != identity(after)
        or identity(after) != identity(visible)
    ):
        raise ValueError(f"file changed while fingerprinting: {path}")
    return {
        "exists": True,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "inode": before.st_ino,
        "device": before.st_dev,
        "sha256": digest.hexdigest(),
    }


def _content_fingerprint(path: str) -> Dict[str, Any]:
    """返回仅由文件 bytes 决定、可跨 rename/copy 稳定比较的身份。"""
    fingerprint = _file_fingerprint(path)
    return {
        "exists": fingerprint["exists"],
        "size": fingerprint["size"],
        "sha256": fingerprint["sha256"],
    }


def _file_fingerprint_from_fd(descriptor: int, display_path: str) -> Dict[str, Any]:
    """在已打开的 regular-file FD 上读取并复验完整身份。"""
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"fingerprint path must be a regular file: {display_path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, _COMPARE_CHUNK):
        digest.update(chunk)
    after = os.fstat(descriptor)
    identity = lambda value: (
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_ino,
        value.st_dev,
    )
    if identity(before) != identity(after):
        raise ValueError(f"file changed while fingerprinting: {display_path}")
    return {
        "exists": True,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "inode": before.st_ino,
        "device": before.st_dev,
        "sha256": digest.hexdigest(),
    }


def _file_fingerprint_at(name: str, directory_fd: int, display_path: str) -> Dict[str, Any]:
    """相对稳定目录 FD 打开文件并读取，避免重新解析绝对路径。"""
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        return _file_fingerprint_from_fd(descriptor, display_path)
    finally:
        os.close(descriptor)


def _tree_identity(
    root: str,
    *,
    exclude_names: set[str] | None = None,
    exclude_relative_roots: set[str] | None = None,
) -> Dict[str, Any]:
    """按相对路径、大小与 SHA-256 绑定 regular-file 树。"""
    if not os.path.exists(root):
        return {"exists": False, "files": [], "sha256": hashlib.sha256(b"").hexdigest()}
    _reject_symlink_ancestors(root, allow_missing_leaf=False)
    excluded = exclude_names or set()
    excluded_roots = {os.path.normpath(value) for value in (exclude_relative_roots or set())}
    files: List[Dict[str, Any]] = []
    walk_errors: List[OSError] = []
    for dirpath, dirnames, filenames in os.walk(
        root, onerror=walk_errors.append, followlinks=False
    ):
        relative_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if os.path.normpath(os.path.join(relative_dir, dirname)) not in excluded_roots
        ]
        for dirname in dirnames:
            candidate = os.path.join(dirpath, dirname)
            if os.path.islink(candidate):
                raise ValueError(f"tree identity 不接受符号链接目录: {candidate}")
        for name in filenames:
            if name in excluded:
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                raise ValueError(f"tree identity 不接受符号链接文件: {path}")
            content = _content_fingerprint(path)
            files.append({"path": os.path.relpath(path, root), **content})
    if walk_errors:
        raise walk_errors[0]
    files.sort(key=lambda item: item["path"])
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"exists": True, "files": files, "sha256": digest}


def _stage_manifest_fingerprint(
    stage_output_dir: str, manifest_prefix: str
) -> Dict[str, Any]:
    """返回最终子阶段每个 per-dataset manifest 的独立内容指纹。"""
    paths = sorted(
        glob(os.path.join(stage_output_dir, f"{manifest_prefix}_manifest_*.jsonl"))
    )
    return {
        "count": len(paths),
        "files": [
            {"name": os.path.basename(path), **_file_fingerprint(path)}
            for path in paths
        ],
    }


def _audio_tree_fingerprint(audio_root: str) -> Dict[str, Any]:
    """返回剩余音频树的轻量 stat 指纹；任一遍历/stat 错误均抛出。"""
    if not os.path.exists(audio_root):
        return {"exists": False, "files": []}
    files: List[Dict[str, Any]] = []
    walk_error: List[OSError] = []

    def _on_walk_error(error: OSError) -> None:
        walk_error.append(error)

    for dirpath, _dirnames, filenames in os.walk(audio_root, onerror=_on_walk_error):
        for name in filenames:
            path = os.path.join(dirpath, name)
            stat = os.stat(path)
            files.append({
                "path": os.path.relpath(path, audio_root),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    if walk_error:
        raise walk_error[0]
    files.sort(key=lambda item: item["path"])
    return {"exists": True, "files": files}


def _atomic_write_json_durable(payload: Dict[str, Any], path: str) -> None:
    """原子写 JSON，并 fsync 文件与父目录。"""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _atomic_copy_verified(
    source: str,
    target: str,
    *,
    expected_content: Dict[str, Any] | None = None,
) -> None:
    """从稳定 regular-file FD 复制，并在目标目录原子发布。"""
    _reject_symlink_ancestors(source, allow_missing_leaf=False)
    target_parent = os.path.dirname(target)
    _reject_symlink_ancestors(target, allow_missing_leaf=True)
    os.makedirs(target_parent, exist_ok=True)
    source_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    temporary = ""
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"atomic copy source must be a regular file: {source}")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=target_parent
        )
        digest = hashlib.sha256()
        size = 0
        try:
            while chunk := os.read(source_fd, _COMPARE_CHUNK):
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("atomic copy short write")
                    view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        after = os.fstat(source_fd)
        path_after = os.stat(source, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(path_after):
            raise OSError(f"atomic copy source changed while reading: {source}")
        actual_content = {"exists": True, "size": size, "sha256": digest.hexdigest()}
        if expected_content is not None and actual_content != expected_content:
            raise OSError(f"atomic copy source fingerprint mismatch: {source}")
        if _content_fingerprint(temporary) != actual_content:
            raise OSError(f"atomic copy temporary fingerprint mismatch: {source} -> {target}")
        os.replace(temporary, target)
        temporary = ""
        _fsync_directories([target])
    finally:
        os.close(source_fd)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _copy_owned(source: str, target: str) -> None:
    """兼容事务准备路径：以原子、可验证复制创建 owned artifact。"""
    _atomic_copy_verified(source, target, expected_content=_content_fingerprint(source))


def _open_directory_without_symlinks(path: str) -> int:
    """逐级打开实体目录，返回绑定最终目录 inode 的稳定 dirfd。"""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in os.path.abspath(path).split(os.path.sep):
            if not component:
                continue
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _source_identity_matches(info: os.stat_result, expected: Dict[str, Any]) -> bool:
    """比较预检完整身份；内容 SHA 已由同一 stat identity 的 fingerprint 绑定。"""
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and info.st_dev == expected["device"]
        and info.st_ino == expected["inode"]
        and info.st_size == expected["size"]
        and info.st_mtime_ns == expected["mtime_ns"]
        and info.st_ctime_ns == expected["ctime_ns"]
    )


def _restore_file_noreplace(
    source: str,
    target: str,
    *,
    source_dir_fd: int,
    target_dir_fd: int,
) -> None:
    """通过原子 hard link 恢复 regular file，绝不覆盖竞争目标。"""
    if not hasattr(os, "link"):
        raise OSError("当前平台不支持 hard-link no-replace 恢复")
    try:
        os.link(
            source,
            target,
            src_dir_fd=source_dir_fd,
            dst_dir_fd=target_dir_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise OSError(f"恢复目标已被并发占用: {target}") from exc


def _cleanup_source_quarantine(
    quarantine_root: str,
    expected_content: Dict[str, Any],
    *,
    parent_fd: int | None = None,
    quarantine_fd: int | None = None,
    quarantine_identity: tuple[int, int] | None = None,
) -> None:
    """以稳定 parent/quarantine FD 校验并删除已恢复的 source quarantine。"""
    parent = os.path.dirname(quarantine_root)
    name = os.path.basename(quarantine_root)
    owns_parent = parent_fd is None
    owns_quarantine = quarantine_fd is None
    if parent_fd is None:
        parent_fd = _open_directory_without_symlinks(parent)
    source_fd: int | None = None
    try:
        if quarantine_fd is None:
            try:
                quarantine_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return
        info = os.fstat(quarantine_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or (
                quarantine_identity is not None
                and (info.st_dev, info.st_ino) != quarantine_identity
            )
        ):
            raise RuntimeError(f"source quarantine 目录身份非法: {quarantine_root}")
        source_fd = os.open(
            "source",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=quarantine_fd,
        )
        actual = _file_fingerprint_from_fd(source_fd, f"{quarantine_root}/source")
        if actual["exists"] is not True or {
            "exists": True,
            "size": actual["size"],
            "sha256": actual["sha256"],
        } != expected_content:
            raise RuntimeError(f"source quarantine 内容已变化: {quarantine_root}")
        os.close(source_fd)
        source_fd = None
        # 目录名也必须仍指向最初打开的 quarantine inode，避免删除替换目录。
        visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (visible.st_dev, visible.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError(f"source quarantine 目录在删除前被替换: {quarantine_root}")
        os.unlink("source", dir_fd=quarantine_fd)
        os.fsync(quarantine_fd)
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if owns_quarantine and quarantine_fd is not None:
            os.close(quarantine_fd)
        if owns_parent:
            os.close(parent_fd)


def _restore_source_from_quarantine(
    quarantine_root: str,
    source: str,
    expected_content: Dict[str, Any],
) -> Dict[str, Any]:
    """从已记录的 quarantine 原子恢复 source，并在成功后清理证据目录。"""
    parent = os.path.dirname(source)
    source_name = os.path.basename(source)
    if os.path.dirname(os.path.abspath(quarantine_root)) != os.path.abspath(parent):
        raise ValueError(f"source quarantine 路径必须位于 source parent: {quarantine_root}")
    parent_fd = _open_directory_without_symlinks(parent)
    quarantine_fd: int | None = None
    try:
        quarantine_fd = os.open(
            os.path.basename(quarantine_root),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        info = os.fstat(quarantine_fd)
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
            raise RuntimeError(f"source quarantine 目录身份非法: {quarantine_root}")
        quarantine_identity = (info.st_dev, info.st_ino)
        _restore_file_noreplace(
            "source",
            source_name,
            source_dir_fd=quarantine_fd,
            target_dir_fd=parent_fd,
        )
        restored = _file_fingerprint_at(source_name, parent_fd, source)
        if {
            "exists": True,
            "size": restored["size"],
            "sha256": restored["sha256"],
        } != expected_content:
            raise RuntimeError(f"quarantine 恢复 source 内容不匹配: {source}")
        _cleanup_source_quarantine(
            quarantine_root,
            expected_content,
            parent_fd=parent_fd,
            quarantine_fd=quarantine_fd,
            quarantine_identity=quarantine_identity,
        )
        return restored
    finally:
        if quarantine_fd is not None:
            os.close(quarantine_fd)
        os.close(parent_fd)


def _unlink_source_if_unchanged(
    source: str,
    expected: Dict[str, Any],
    *,
    transaction_id: str,
    quarantine_root: str | None = None,
) -> None:
    """仅删除预检绑定的 source inode；无法证明同一对象时保留 pathname。

    source 先经稳定 parent dirfd 打开，再原子隔离到同一文件系统内的事务私有
    目录。隔离后同时复验已打开 FD 与 quarantine 路径的 inode，消除 fingerprint
    后直接按原 pathname unlink 的替换窗口。身份不符时尽力恢复刚隔离的对象；
    若 pathname 已被并发占用，则保留 quarantine 证据并 fail closed。
    """
    parent = os.path.dirname(source)
    source_name = os.path.basename(source)
    parent_fd = _open_directory_without_symlinks(parent)
    source_fd: int | None = None
    quarantine_fd: int | None = None
    quarantine_name = "source"
    quarantine_root = quarantine_root or os.path.join(
        parent,
        f".promote-source-{transaction_id}-{secrets.token_hex(16)}",
    )
    if os.path.dirname(os.path.abspath(quarantine_root)) != os.path.abspath(parent):
        raise ValueError(f"source quarantine 路径必须位于 source parent: {quarantine_root}")
    moved = False
    try:
        source_fd = os.open(
            source_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        if not _source_identity_matches(os.fstat(source_fd), expected):
            raise RuntimeError(f"source 在删除前变化，保留文件: {source}")

        os.mkdir(
            os.path.basename(quarantine_root),
            mode=0o700,
            dir_fd=parent_fd,
        )
        quarantine_fd = os.open(
            os.path.basename(quarantine_root),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        quarantine_info = os.fstat(quarantine_fd)
        if not stat.S_ISDIR(quarantine_info.st_mode) or stat.S_IMODE(quarantine_info.st_mode) != 0o700:
            raise RuntimeError(f"source quarantine 目录身份非法: {quarantine_root}")

        os.rename(
            source_name,
            quarantine_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=quarantine_fd,
        )
        moved = True
        opened = os.fstat(source_fd)
        isolated = os.stat(
            quarantine_name,
            dir_fd=quarantine_fd,
            follow_symlinks=False,
        )
        os.lseek(source_fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(source_fd, _COMPARE_CHUNK):
            digest.update(chunk)
        after_read = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != expected["device"]
            or opened.st_ino != expected["inode"]
            or opened.st_size != expected["size"]
            or opened.st_mtime_ns != expected["mtime_ns"]
            or isolated.st_dev != opened.st_dev
            or isolated.st_ino != opened.st_ino
            or not stat.S_ISREG(isolated.st_mode)
            or after_read.st_size != opened.st_size
            or after_read.st_mtime_ns != opened.st_mtime_ns
            or digest.hexdigest() != expected["sha256"]
        ):
            raise RuntimeError(f"source 隔离对象与预检 inode 不一致，保留文件: {source}")

        os.unlink(quarantine_name, dir_fd=quarantine_fd)
        moved = False
        os.fsync(quarantine_fd)
        os.rmdir(os.path.basename(quarantine_root), dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if moved and quarantine_fd is not None:
            try:
                _restore_file_noreplace(
                    quarantine_name,
                    source_name,
                    source_dir_fd=quarantine_fd,
                    target_dir_fd=parent_fd,
                )
                restored = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(restored.st_mode)
                    or restored.st_dev != expected["device"]
                    or restored.st_ino != expected["inode"]
                    or restored.st_size != expected["size"]
                    or restored.st_mtime_ns != expected["mtime_ns"]
                ):
                    raise RuntimeError(f"source 恢复对象身份不一致，保留文件: {source}")
                os.unlink(quarantine_name, dir_fd=quarantine_fd)
                os.fsync(quarantine_fd)
                os.rmdir(os.path.basename(quarantine_root), dir_fd=parent_fd)
                os.fsync(parent_fd)
                moved = False
            except OSError:
                pass
        raise
    finally:
        if quarantine_fd is not None:
            os.close(quarantine_fd)
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)


def _fsync_directories(paths: List[str]) -> None:
    """持久化目录项更新；重复父目录只 fsync 一次。"""
    for directory in sorted({os.path.dirname(path) for path in paths}):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _restore_transaction(journal: Dict[str, Any], journal_path: str) -> None:
    """按持久 journal 恢复 source 与发布前 destination 集合。"""
    errors: List[str] = []
    restored_sources: set[str] = set()

    def persist_source_restore_state(
        item: Dict[str, Any], *, pending: bool, fingerprint: Dict[str, Any] | None
    ) -> None:
        """同步 audio/targets 镜像，避免 journal 重载后 exact 校验漂移。"""
        mirrors = [
            candidate
            for candidate in journal.get("targets", [])
            if candidate.get("source") == item.get("source")
            and candidate.get("target") == item.get("target")
            and candidate.get("record_id") == item.get("record_id")
        ]
        if not mirrors:
            raise OSError("audio target mirror is unavailable")
        for candidate in [item, *mirrors]:
            candidate["source_restore_pending"] = pending
            if fingerprint is None:
                candidate.pop("restored_source_fingerprint", None)
            else:
                candidate["restored_source_fingerprint"] = fingerprint
        _atomic_write_json_durable(journal, journal_path)

    for item in journal.get("audio", []):
        source = item["source"]
        destination = item["target"]
        expected = item["new_content"]
        expected_source = item["source_fingerprint"]
        restored_source = item.get("restored_source_fingerprint")
        restore_pending = item.get("source_restore_pending", False)
        try:
            current_source = _file_fingerprint(source)
            if current_source["exists"]:
                if current_source in (expected_source, restored_source):
                    quarantine_root = item.get("quarantine_root")
                    if (
                        isinstance(quarantine_root, str)
                        and os.path.lexists(quarantine_root)
                        and _content_fingerprint(source) == expected
                    ):
                        restored_sources.add(source)
                    continue
                if not restore_pending or _content_fingerprint(source) != expected:
                    raise OSError("source contains unknown concurrent content or identity")
                persist_source_restore_state(
                    item, pending=False, fingerprint=current_source
                )
                restored_sources.add(source)
                continue
            quarantine_root = item.get("quarantine_root")
            restored_fingerprint: Dict[str, Any] | None = None
            if isinstance(quarantine_root, str):
                persist_source_restore_state(item, pending=True, fingerprint=None)
                try:
                    restored_fingerprint = _restore_source_from_quarantine(
                        quarantine_root, source, expected
                    )
                except FileNotFoundError:
                    pass
            if restored_fingerprint is None:
                recovery_source = (
                    destination
                    if _content_fingerprint(destination) == expected
                    else item["staged"]
                )
                if _content_fingerprint(recovery_source) != expected:
                    raise OSError("published audio, quarantine and staging are all unavailable")
                if not isinstance(quarantine_root, str):
                    persist_source_restore_state(item, pending=True, fingerprint=None)
                _atomic_copy_verified(
                    recovery_source,
                    source,
                    expected_content=expected,
                )
                restored_fingerprint = _file_fingerprint(source)
            if {
                "exists": True,
                "size": restored_fingerprint["size"],
                "sha256": restored_fingerprint["sha256"],
            } != expected:
                raise OSError("restored source content fingerprint mismatch")
            persist_source_restore_state(
                item, pending=False, fingerprint=restored_fingerprint
            )
            restored_sources.add(source)
        except (OSError, ValueError) as exc:
            errors.append(f"restore source {source}: {exc}")
    for item in reversed(journal.get("targets", [])):
        target = item["target"]
        old_content = item["old_content"]
        backup = item.get("backup")
        try:
            current = _content_fingerprint(target)
            new_content = item["new_content"]
            if old_content["exists"]:
                if current == old_content:
                    continue
                if current["exists"] and current != new_content:
                    raise OSError("target contains unknown concurrent content")
                if not backup or _content_fingerprint(backup) != old_content:
                    raise OSError("valid backup unavailable")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.replace(backup, target)
            elif current["exists"]:
                if current != new_content:
                    raise OSError("target contains unknown concurrent content")
                os.unlink(target)
        except (OSError, ValueError) as exc:
            errors.append(f"restore target {target}: {exc}")
    for item in journal.get("audio", []):
        source = item.get("source")
        quarantine_root = item.get("quarantine_root")
        if (
            not isinstance(source, str)
            or source not in restored_sources
            or not isinstance(quarantine_root, str)
            or not os.path.lexists(quarantine_root)
        ):
            continue
        try:
            _cleanup_source_quarantine(quarantine_root, item["new_content"])
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"cleanup source quarantine {quarantine_root}: {exc}")
    if errors:
        raise RuntimeError("promote transaction recovery failed: " + "; ".join(errors))
    for owned_root in journal.get("owned_roots", []):
        if isinstance(owned_root, str) and os.path.isdir(owned_root):
            shutil.rmtree(owned_root)
    os.unlink(journal_path)


def _validate_physical_containment(
    path: str,
    root: str,
    *,
    purpose: str,
    allow_missing_leaf: bool,
) -> None:
    """同时校验词法与物理 containment，并拒绝完整路径上的 symlink。"""
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"{purpose}路径非法")
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    try:
        if os.path.commonpath((path_abs, root_abs)) != root_abs:
            raise RuntimeError(f"{purpose}词法路径越界")
    except ValueError as exc:
        raise RuntimeError(f"{purpose}路径跨设备或格式非法") from exc
    _reject_symlink_ancestors(path_abs, allow_missing_leaf=allow_missing_leaf)
    root_real = os.path.realpath(root_abs)
    resolved = os.path.realpath(path_abs)
    try:
        if os.path.commonpath((resolved, root_real)) != root_real:
            raise RuntimeError(f"{purpose}物理路径越界")
    except ValueError as exc:
        raise RuntimeError(f"{purpose}物理路径跨设备或格式非法") from exc


def _validate_recovery_regular_path(
    path: str,
    root: str,
    *,
    purpose: str,
    allow_missing_leaf: bool,
) -> None:
    """校验恢复路径的 domain、无 symlink 与 existing regular-file 契约。"""
    _validate_physical_containment(
        path,
        root,
        purpose=purpose,
        allow_missing_leaf=allow_missing_leaf,
    )
    if os.path.lexists(path) and not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError(f"{purpose}必须是 regular file: {path}")


def _validate_content_fingerprint_value(value: Any, *, purpose: str) -> None:
    if not isinstance(value, dict) or set(value) != {"exists", "size", "sha256"}:
        raise RuntimeError(f"{purpose}内容指纹非法")
    exists, size, digest = value["exists"], value["size"], value["sha256"]
    if (
        type(exists) is not bool
        or type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or (exists and re.fullmatch(r"[0-9a-f]{64}", digest) is None)
        or (not exists and (size != 0 or digest != ""))
    ):
        raise RuntimeError(f"{purpose}内容指纹非法")


def _validate_file_fingerprint_value(value: Any, *, purpose: str) -> None:
    required = {"exists", "size", "mtime_ns", "ctime_ns", "inode", "device", "sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError(f"{purpose}完整指纹非法")
    _validate_content_fingerprint_value(
        {"exists": value["exists"], "size": value["size"], "sha256": value["sha256"]},
        purpose=purpose,
    )
    if any(type(value[field]) is not int or value[field] < 0 for field in ("mtime_ns", "ctime_ns", "inode", "device")):
        raise RuntimeError(f"{purpose}完整指纹非法")


def _validate_recovery_journal(
    journal: Dict[str, Any], stage_d_dir: str, final_root: str
) -> None:
    """校验 journal 恢复权限范围，禁止利用持久文件触碰事务域外路径。"""
    stage_d_real = os.path.realpath(stage_d_dir)
    final_root_real = os.path.realpath(final_root)
    transaction_id = journal.get("transaction_id")
    if (
        journal.get("version") != _JOURNAL_VERSION
        or journal.get("stage_d_dir") != stage_d_real
        or journal.get("final_root") != final_root_real
        or journal.get("phase") not in {
            "preparing", "prepared", "published", "publishing_marker", "committed"
        }
        or not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise RuntimeError("未完成 promote journal 身份非法，拒绝自动恢复")

    owned_roots = journal.get("owned_roots")
    audio = journal.get("audio")
    targets = journal.get("targets")
    if not isinstance(owned_roots, list) or not isinstance(audio, list) or not isinstance(targets, list):
        raise RuntimeError("未完成 promote journal 集合字段非法")
    expected_owned_name = f".promote-owned-{transaction_id}"
    normalized_owned: set[str] = set()
    for owned_root in owned_roots:
        if not isinstance(owned_root, str) or os.path.basename(owned_root) != expected_owned_name:
            raise RuntimeError("未完成 promote journal owned staging 路径越界")
        if not (
            os.path.commonpath((os.path.abspath(owned_root), stage_d_real)) == stage_d_real
            or os.path.commonpath((os.path.abspath(owned_root), final_root_real)) == final_root_real
        ):
            raise RuntimeError("未完成 promote journal owned staging 路径越界")
        _validate_physical_containment(
            owned_root,
            stage_d_real if os.path.commonpath((os.path.abspath(owned_root), stage_d_real)) == stage_d_real else final_root_real,
            purpose="owned staging",
            allow_missing_leaf=True,
        )
        normalized_owned.add(os.path.abspath(owned_root))
    marker_path = os.path.join(stage_d_real, PROMOTE_MARKER_NAME)
    for item in targets:
        if not isinstance(item, dict):
            raise RuntimeError("未完成 promote journal target 项非法")
        target = item.get("target")
        if not isinstance(target, str):
            raise RuntimeError("未完成 promote journal target 路径非法")
        if os.path.abspath(target) == marker_path:
            _validate_recovery_regular_path(
                target, stage_d_real, purpose="marker target", allow_missing_leaf=True
            )
        else:
            _validate_recovery_regular_path(
                target, final_root_real, purpose="final target", allow_missing_leaf=True
            )
        _validate_content_fingerprint_value(
            item.get("old_content"), purpose="target old_content"
        )
        _validate_content_fingerprint_value(
            item.get("new_content"), purpose="target new_content"
        )
        for field in ("staged", "backup"):
            candidate = item.get(field)
            if candidate is None:
                continue
            if not isinstance(candidate, str):
                raise RuntimeError("未完成 promote journal owned artifact 路径非法")
            candidate_parent = os.path.abspath(os.path.dirname(candidate))
            if candidate_parent not in normalized_owned:
                raise RuntimeError("未完成 promote journal owned artifact 路径越界")
            _validate_recovery_regular_path(
                candidate,
                candidate_parent,
                purpose="owned artifact",
                allow_missing_leaf=True,
            )
    for index, item in enumerate(audio):
        if not isinstance(item, dict) or not any(item == target_item for target_item in targets):
            raise RuntimeError("未完成 promote journal audio 项非法")
        source = item.get("source")
        _validate_recovery_regular_path(
            source, stage_d_real, purpose="audio source", allow_missing_leaf=True
        )
        quarantine_root = item.get("quarantine_root")
        expected_quarantine_name = f".promote-source-{transaction_id}-{index:06d}"
        if (
            not isinstance(quarantine_root, str)
            or os.path.basename(quarantine_root) != expected_quarantine_name
            or os.path.abspath(os.path.dirname(quarantine_root))
            != os.path.abspath(os.path.dirname(source))
        ):
            raise RuntimeError("未完成 promote journal quarantine 路径非法")
        _validate_physical_containment(
            quarantine_root,
            stage_d_real,
            purpose="source quarantine",
            allow_missing_leaf=True,
        )
        target = item.get("target")
        _validate_recovery_regular_path(
            target, final_root_real, purpose="audio target", allow_missing_leaf=True
        )
        _validate_file_fingerprint_value(
            item.get("source_fingerprint"), purpose="audio source_fingerprint"
        )
        restore_pending = item.get("source_restore_pending", False)
        if type(restore_pending) is not bool:
            raise RuntimeError("audio source_restore_pending 必须是 bool")
        restored_source = item.get("restored_source_fingerprint")
        if restored_source is not None:
            _validate_file_fingerprint_value(
                restored_source, purpose="audio restored_source_fingerprint"
            )
            if {
                "exists": restored_source["exists"],
                "size": restored_source["size"],
                "sha256": restored_source["sha256"],
            } != item.get("new_content"):
                raise RuntimeError("audio restored_source_fingerprint 内容不匹配")


def _recover_or_block(stage_d_dir: str, final_root: str) -> bool:
    """恢复遗留事务；本次调用仍被阻断，要求操作者复核后重跑。"""
    journal_path = os.path.join(stage_d_dir, PROMOTE_JOURNAL_NAME)
    if not os.path.lexists(journal_path):
        return False
    _reject_symlink_ancestors(journal_path, allow_missing_leaf=False)
    snapshot = read_file_bytes_snapshot(journal_path)
    journal = json.loads(snapshot.data)
    if not isinstance(journal, dict):
        raise RuntimeError("未完成 promote journal 必须是 JSON object")
    _validate_recovery_journal(journal, stage_d_dir, final_root)
    if journal.get("phase") == "committed":
        for owned_root in journal.get("owned_roots", []):
            if isinstance(owned_root, str) and os.path.isdir(owned_root):
                shutil.rmtree(owned_root)
        os.unlink(journal_path)
    else:
        _restore_transaction(journal, journal_path)
    return True


def _records_identity(records: List[AudioRecord]) -> List[Dict[str, str]]:
    identities: List[Dict[str, str]] = []
    for rec in sorted(records, key=lambda item: (item.dataset_id, item.record_id)):
        payload = json.dumps(
            rec.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        identities.append(
            {
                "dataset_id": rec.dataset_id,
                "record_id": rec.record_id,
                "audio_path": rec.audio["path"],
                "record_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return identities


def _owned_path(target: str, transaction_id: str, role: str) -> str:
    root = os.path.join(os.path.dirname(target), f".promote-owned-{transaction_id}")
    os.makedirs(root, mode=0o700, exist_ok=True)
    return os.path.join(root, f"{role}-{os.path.basename(target)}")


def _write_bytes_durable(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _manifest_bytes(records: List[AudioRecord]) -> bytes:
    return b"".join(
        (
            json.dumps(rec.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        for rec in records
    )


def _merge_records(existing: List[AudioRecord], incoming: List[AudioRecord]) -> List[AudioRecord]:
    merged = {record.record_id: record for record in existing}
    order = [record.record_id for record in existing]
    for record in incoming:
        if record.record_id not in merged:
            order.append(record.record_id)
        merged[record.record_id] = record
    return [merged[record_id] for record_id in order]


def _stats_from_records(records_by_dataset: Dict[str, List[AudioRecord]]) -> Dict[str, Any]:
    stats = LibraryStats()
    for dataset_id, records in sorted(records_by_dataset.items()):
        for rec in records:
            c3 = rec.stages["C3"]
            label_display = c3["balance_category"]
            stats.add(
                dataset_id,
                sanitize_basename_for_rename(label_display, is_file=False),
                label_display,
                rec.audio.get("sr"),
                rec.audio.get("duration"),
            )
    return stats.to_dict()


def read_validated_promote_marker(
    stage_d_dir: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """验证完整 marker v4；未完成 journal、缺字段或任一内容变化均失效。"""
    stage_d_real = os.path.realpath(stage_d_dir)
    journal_path = os.path.join(stage_d_real, PROMOTE_JOURNAL_NAME)
    if os.path.lexists(journal_path):
        return False, f"存在未完成 promote journal: {journal_path}", None
    marker_path = os.path.join(stage_d_real, PROMOTE_MARKER_NAME)
    try:
        snapshot = read_file_bytes_snapshot(marker_path)
        marker = json.loads(snapshot.data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return False, f"promote marker 不存在或不可读: {marker_path}（{exc}）", None
    if not isinstance(marker, dict):
        return False, "promote marker 必须是 JSON object", None
    expected_contract_versions = {
        "version": PROMOTE_MARKER_VERSION,
        "manifest_schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
        "admission_policy_version": _EXPECTED_ADMISSION_POLICY_VERSION,
        "evidence_contract_version": _EXPECTED_EVIDENCE_CONTRACT_VERSION,
        "parser_version": _EXPECTED_PARSER_VERSION,
        "generator_version": _EXPECTED_GENERATOR_VERSION,
    }
    legacy_fields = sorted(_LEGACY_PROMOTE_MARKER_FIELDS & set(marker))
    if legacy_fields:
        return False, f"promote marker 含 legacy 门禁字段: {legacy_fields!r}", None
    for field, expected in expected_contract_versions.items():
        if marker.get(field) != expected:
            return False, f"promote marker 的 {field} 版本不受支持", None
    if marker.get("stage_d_dir") != stage_d_real:
        return False, "promote marker 不属于当前 stage_d", None
    final_root = marker.get("final_root")
    if not isinstance(final_root, str) or final_root != os.path.realpath(final_root):
        return False, "promote marker 的 final_root 非法", None
    counts = (marker.get("dataset_count"), marker.get("record_count"), marker.get("file_count"))
    try:
        created = datetime.fromisoformat(marker.get("created_at"))
    except (TypeError, ValueError):
        return False, "promote marker 契约字段非法", None
    if (
        any(type(value) is not int or value < 0 for value in counts)
        or marker.get("record_count") == 0
        or created.tzinfo is None
    ):
        return False, "promote marker 契约字段非法", None
    stage_name = marker.get("final_stage")
    if stage_name not in {"D1", "D2"}:
        return False, "promote marker 的 final_stage 非法", None
    required_objects = (
        "d_stats_fingerprint",
        "d2_manifest_fingerprint",
        "d2_audio_fingerprint",
        "library_stats_fingerprint",
        "stage_b_tree_identity",
        "final_stage_tree_identity",
    )
    if any(not isinstance(marker.get(field), dict) for field in required_objects):
        return False, "promote marker 缺少完整树或文件 identity", None
    published_audio = marker.get("published_audio")
    final_manifests = marker.get("final_manifest_fingerprints")
    published_records = marker.get("published_records")
    if (
        not isinstance(published_audio, list)
        or len(published_audio) != marker["file_count"]
        or not isinstance(final_manifests, list)
        or len(final_manifests) != marker["dataset_count"]
        or not isinstance(published_records, list)
        or len(published_records) != marker["record_count"]
    ):
        return False, "promote marker 发布集合字段不完整", None
    stage_output_dir = os.path.join(stage_d_real, stage_name)
    try:
        current_d_stats = _file_fingerprint(os.path.join(stage_d_real, "d_stats.json"))
        current_manifests = _stage_manifest_fingerprint(stage_output_dir, stage_name.lower())
        current_audio_tree = _audio_tree_fingerprint(os.path.join(stage_output_dir, "audio"))
        current_stats = _file_fingerprint(os.path.join(final_root, "library_stats.json"))
        stage_b_identity = _tree_identity(
            os.path.join(os.path.dirname(stage_d_real), "stage_b")
        )
        final_stage_identity = _tree_identity(stage_output_dir)
        admission_context = NoiseAdmissionContext.load()
        expected_records: List[AudioRecord] = []
        expected_audio: List[Dict[str, Any]] = []
        expected_dataset_ids: List[str] = []
        prefix = f"{stage_name.lower()}_manifest_"
        for manifest_path in sorted(glob(os.path.join(stage_output_dir, f"{prefix}*.jsonl"))):
            dataset_id = os.path.basename(manifest_path)[len(prefix):-len(".jsonl")]
            candidates = [
                record
                for record in _load_manifest_strict(
                    manifest_path,
                    dataset_id,
                    audio_root=os.path.join(stage_output_dir, "audio"),
                    alternate_audio_root=os.path.join(final_root, "audio"),
                    admission_context=admission_context,
                )
                if record.is_stage_passed("D2")
            ]
            if not candidates:
                continue
            expected_dataset_ids.append(dataset_id)
            for record in candidates:
                target, error = _final_path(
                    record.audio["path"],
                    os.path.join(stage_output_dir, "audio"),
                    os.path.join(final_root, "audio"),
                )
                if error:
                    raise ValueError(error)
                record.audio["path"] = target
                expected_records.append(record)
                expected_audio.append(
                    {"path": target, "content": _content_fingerprint(target)}
                )
        if expected_audio != published_audio:
            return False, "promote marker 绑定的发布音频集合不精确", None
        if _records_identity(expected_records) != published_records:
            return False, "promote marker 绑定的 published record identity 不精确", None
        expected_manifest_paths = [
            os.path.join(final_root, "manifests", f"{dataset_id}.jsonl")
            for dataset_id in expected_dataset_ids
        ]
        if [entry.get("path") for entry in final_manifests if isinstance(entry, dict)] != expected_manifest_paths:
            return False, "promote marker 绑定的 final manifest 集合不精确", None
        final_record_identities: Dict[Tuple[str, str], Dict[str, str]] = {}
        for entry in final_manifests:
            if not isinstance(entry, dict) or _content_fingerprint(entry.get("path", "")) != entry.get("content"):
                return False, "promote marker 绑定的 final manifest 已变化", None
            dataset_id = os.path.basename(entry["path"])[:-len(".jsonl")]
            for identity in _records_identity(list(read_manifest_strict(entry["path"], reject_duplicate_ids=True))):
                final_record_identities[(dataset_id, identity["record_id"])] = identity
        for identity in published_records:
            key = (identity["dataset_id"], identity["record_id"])
            if final_record_identities.get(key) != identity:
                return False, "promote marker 绑定的 published record 已变化", None
    except (OSError, TypeError, ValueError) as exc:
        return False, f"promote 输入、audio 或发布 identity 读取失败: {exc}", None
    checks = (
        ("d_stats_fingerprint", current_d_stats, "d_stats.json 已在 promote 后变化"),
        ("d2_manifest_fingerprint", current_manifests, f"{stage_name} manifest 集合已在 promote 后变化"),
        ("d2_audio_fingerprint", current_audio_tree, f"{stage_name}/audio 树已在 promote 后变化"),
        ("library_stats_fingerprint", current_stats, "promote 目标的 library_stats.json 已变化"),
        ("stage_b_tree_identity", stage_b_identity, "Stage B 树已在 promote 后变化"),
        ("final_stage_tree_identity", final_stage_identity, "最终 Stage D 树已在 promote 后变化"),
    )
    for field, current, reason in checks:
        if marker.get(field) != current:
            return False, reason, None
    return True, "ok", dict(marker)


def validate_promote_marker(stage_d_dir: str) -> Tuple[bool, str]:
    """兼容既有调用，仅返回 marker 校验状态与原因。"""
    valid, reason, _marker = read_validated_promote_marker(stage_d_dir)
    return valid, reason


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def promote_run(
    stage_d_dir: str,
    final_root: str,
    dry_run: bool = False,
    force: bool = False,
    config_path: str | None = None,
) -> int:
    """执行 promote，返回退出码（0 成功；1 校验/移动失败或并发锁冲突；2 完成门槛未过）。

    固定按 run 锁、final-root promote 锁的顺序获取，反向释放。run 锁覆盖
    Stage D 预检、音频移动、final 发布与 marker 写入，避免 writer/cleanup
    交错；final 锁避免多个 run 对同一 final 发生 lost update。dry-run 同样持锁。
    """
    run_lock_fd = None
    promote_lock_fd = None
    try:
        run_lock_fd = acquire_run_lock(stage_d_dir)
        promote_lock_fd = _acquire_promote_lock(final_root)
    except RuntimeError as e:
        logger.error("%s", e)
        if run_lock_fd is not None:
            release_run_lock(run_lock_fd)
        return 1
    try:
        return _promote_run_unlocked(
            stage_d_dir,
            final_root,
            dry_run=dry_run,
            force=force,
            config_path=config_path,
        )
    finally:
        _release_promote_lock(promote_lock_fd)
        release_run_lock(run_lock_fd)


def _promote_run_unlocked(
    stage_d_dir: str,
    final_root: str,
    dry_run: bool = False,
    force: bool = False,
    config_path: str | None = None,
) -> int:
    """全局预检后，以 owned staging、backup 与 journal 执行可恢复提交。"""
    stage_d_dir = os.path.abspath(stage_d_dir)
    final_root = os.path.abspath(final_root)
    marker_path = os.path.join(stage_d_dir, PROMOTE_MARKER_NAME)
    journal_path = os.path.join(stage_d_dir, PROMOTE_JOURNAL_NAME)
    stats_path = os.path.join(stage_d_dir, "d_stats.json")
    try:
        if _recover_or_block(stage_d_dir, final_root):
            logger.error("检测到未完成 promote journal，已恢复并阻止本次 promote；请复核后重跑")
            return 1
        valid_marker, _reason, marker = read_validated_promote_marker(stage_d_dir)
        if valid_marker and marker is not None:
            if marker["final_root"] == os.path.realpath(final_root):
                logger.info("当前 run 已由完整 marker v4 证明发布成功，幂等返回")
                return 0
            raise ValueError("当前 run 已发布到其他 final_root")
        stage_output_dir, manifest_prefix, output_format = _resolve_final_stage_output(
            stage_d_dir, config_path
        )
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        logger.error("promote 恢复或配置检查失败: %s", exc)
        return 1

    stage_name = os.path.basename(stage_output_dir)
    run_audio_root = os.path.join(stage_output_dir, "audio")
    final_audio_root = os.path.join(final_root, "audio")
    if NOISE_SOURCE_PARSER_VERSION != _EXPECTED_PARSER_VERSION or GENERATOR_VERSION != _EXPECTED_GENERATOR_VERSION:
        logger.error("runtime parser/generator contract 版本漂移")
        return 1

    # 全局预检：此区间不得建 final 目录、staging、backup、journal 或失效 marker。
    try:
        for protected in (
            stage_d_dir, stage_output_dir, run_audio_root, final_root,
            final_audio_root, os.path.join(final_root, "manifests"), marker_path, journal_path,
        ):
            _reject_symlink_ancestors(protected, allow_missing_leaf=True)
        if not os.path.isdir(run_audio_root):
            raise ValueError(f"run 音频根目录不存在: {run_audio_root}")

        def _paths_overlap(first: str, second: str) -> bool:
            try:
                return os.path.commonpath((first, second)) in (first, second)
            except ValueError:
                return False

        if _paths_overlap(os.path.realpath(run_audio_root), os.path.realpath(final_audio_root)):
            raise ValueError("run 音频根与 final 音频根存在重叠")
        if _paths_overlap(os.path.realpath(os.path.dirname(stage_d_dir)), os.path.realpath(final_root)):
            raise ValueError("当前 run 根与 final 根存在重叠")

        admission_context = NoiseAdmissionContext.load()
        datasets, existing_final, candidate_count = _preflight_manifests(
            stage_output_dir,
            manifest_prefix,
            final_root,
            admission_context=admission_context,
            expected_output_format=output_format,
        )
        if candidate_count == 0:
            raise ValueError("Stage D 最终子阶段没有 publish candidates")

        final_target_owners = _build_final_target_ownership(existing_final)
        plans: List[Dict[str, Any]] = []
        planned_sources: set[str] = set()
        planned_targets: set[str] = set()
        records_by_dataset = {
            dataset_id: list(meta["records"])
            for dataset_id, meta in existing_final.items()
        }
        published_records: List[AudioRecord] = []
        for dataset_id, meta in sorted(datasets.items()):
            if _file_fingerprint(meta["path"]) != meta["fingerprint"]:
                raise ValueError(f"全局预检期间 manifest 已变化: {meta['path']}")
            incoming = meta["records"]
            for rec in incoming:
                source = rec.audio["path"]
                target, error = _final_path(source, run_audio_root, final_audio_root)
                if error:
                    raise ValueError(error)
                source_identity = os.path.abspath(source)
                target_identity = os.path.abspath(target)
                normalized_target = _normalize_absolute_path(target)
                existing_owner = final_target_owners.get(normalized_target)
                if os.path.lexists(target) and existing_owner is None:
                    raise ValueError(
                        "incoming target path 已存在但未由 final ownership 索引归属: "
                        f"{normalized_target}: incoming={rec.record_id!r}"
                    )
                if existing_owner is not None and existing_owner != rec.record_id:
                    raise ValueError(
                        "incoming target path 已由不同 record_id 占用: "
                        f"{normalized_target}: existing={existing_owner!r}, incoming={rec.record_id!r}"
                    )
                if source_identity in planned_sources or target_identity in planned_targets:
                    raise ValueError(
                        f"多个发布记录映射到重复 source/destination: {source} -> {target}"
                    )
                planned_sources.add(source_identity)
                planned_targets.add(target_identity)
                if not os.path.isfile(source):
                    raise ValueError(
                        f"source 缺失时不得仅凭任意 destination 幂等: {source} -> {target}"
                    )
                source_fingerprint = _file_fingerprint(source)
                if not source_fingerprint["exists"]:
                    raise ValueError(f"发布源不存在: {source}")
                rec.audio["path"] = target
                plans.append(
                    {
                        "dataset_id": dataset_id,
                        "record_id": rec.record_id,
                        "source": source,
                        "target": target,
                        "source_fingerprint": source_fingerprint,
                        "new_content": {
                            "exists": True,
                            "size": source_fingerprint["size"],
                            "sha256": source_fingerprint["sha256"],
                        },
                    }
                )
            published_records.extend(incoming)
            records_by_dataset[dataset_id] = _merge_records(
                existing_final.get(dataset_id, {}).get("records", []), incoming
            )
        for meta in existing_final.values():
            if _file_fingerprint(meta["path"]) != meta["fingerprint"]:
                raise ValueError(f"全局预检期间 final manifest 已变化: {meta['path']}")
        stats_dict = _stats_from_records(records_by_dataset)
        if not stats_dict:
            raise ValueError("final library stats 为空")
        stage_b_identity_before = _tree_identity(
            os.path.join(os.path.dirname(stage_d_dir), "stage_b")
        )
        d_stats_fingerprint = _file_fingerprint(stats_path)
        stage_manifest_fingerprint = _stage_manifest_fingerprint(stage_output_dir, manifest_prefix)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.error("promote 全局预检失败，未修改任何发布数据: %s", exc)
        return 1

    logger.info(
        "%s全局预检通过: %d 数据集 / %d 条记录",
        "[dry-run] " if dry_run else "", len(datasets), candidate_count,
    )
    if dry_run:
        return 0

    transaction_id = uuid.uuid4().hex
    owned_roots = sorted(
        {
            os.path.join(os.path.dirname(target), f".promote-owned-{transaction_id}")
            for target in [
                *(plan["target"] for plan in plans),
                *(os.path.join(final_root, "manifests", f"{dataset_id}.jsonl") for dataset_id in datasets),
                os.path.join(final_root, "library_stats.json"),
                marker_path,
            ]
        }
    )
    for index, plan in enumerate(plans):
        plan["quarantine_root"] = os.path.join(
            os.path.dirname(plan["source"]),
            f".promote-source-{transaction_id}-{index:06d}",
        )
    journal: Dict[str, Any] = {
        "version": _JOURNAL_VERSION,
        "transaction_id": transaction_id,
        "phase": "preparing",
        "stage_d_dir": os.path.realpath(stage_d_dir),
        "final_root": os.path.realpath(final_root),
        "owned_roots": owned_roots,
        "audio": [],
        "targets": [],
    }
    journal_written = False
    try:
        _atomic_write_json_durable(journal, journal_path)
        journal_written = True
        # 完整 staging/backup 先于正式 destination/source 修改。
        for plan in plans:
            target = plan["target"]
            staged = _owned_path(target, transaction_id, "staged")
            owned_root = os.path.dirname(staged)
            if owned_root not in journal["owned_roots"]:
                journal["owned_roots"].append(owned_root)
            _copy_owned(plan["source"], staged)
            old_content = _content_fingerprint(target)
            backup = None
            if old_content["exists"]:
                backup = _owned_path(target, transaction_id, "backup")
                _copy_owned(target, backup)
            entry = {
                **plan,
                "staged": staged,
                "old_content": old_content,
                "backup": backup,
            }
            journal["audio"].append(entry)
            journal["targets"].append(entry)

        touched_manifest_entries: List[Dict[str, Any]] = []
        for dataset_id in sorted(datasets):
            target = os.path.join(final_root, "manifests", f"{dataset_id}.jsonl")
            staged = _owned_path(target, transaction_id, "staged")
            owned_root = os.path.dirname(staged)
            if owned_root not in journal["owned_roots"]:
                journal["owned_roots"].append(owned_root)
            _write_bytes_durable(staged, _manifest_bytes(records_by_dataset[dataset_id]))
            old_content = _content_fingerprint(target)
            backup = None
            if old_content["exists"]:
                backup = _owned_path(target, transaction_id, "backup")
                _copy_owned(target, backup)
            entry = {
                "kind": "manifest", "dataset_id": dataset_id, "target": target,
                "staged": staged, "old_content": old_content, "backup": backup,
                "new_content": _content_fingerprint(staged),
            }
            journal["targets"].append(entry)
            touched_manifest_entries.append(entry)

        stats_target = os.path.join(final_root, "library_stats.json")
        stats_staged = _owned_path(stats_target, transaction_id, "staged")
        stats_owned_root = os.path.dirname(stats_staged)
        if stats_owned_root not in journal["owned_roots"]:
            journal["owned_roots"].append(stats_owned_root)
        _write_bytes_durable(
            stats_staged,
            (json.dumps(stats_dict, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        stats_old = _content_fingerprint(stats_target)
        stats_backup = None
        if stats_old["exists"]:
            stats_backup = _owned_path(stats_target, transaction_id, "backup")
            _copy_owned(stats_target, stats_backup)
        stats_entry = {
            "kind": "stats", "target": stats_target, "staged": stats_staged,
            "old_content": stats_old, "backup": stats_backup,
            "new_content": _content_fingerprint(stats_staged),
        }
        journal["targets"].append(stats_entry)

        marker_old = _content_fingerprint(marker_path)
        marker_backup = None
        marker_staged = _owned_path(marker_path, transaction_id, "staged")
        marker_owned_root = os.path.dirname(marker_staged)
        if marker_owned_root not in journal["owned_roots"]:
            journal["owned_roots"].append(marker_owned_root)
        if marker_old["exists"]:
            marker_backup = _owned_path(marker_path, transaction_id, "backup")
            _copy_owned(marker_path, marker_backup)
        marker_entry = {
            "kind": "marker", "target": marker_path, "staged": marker_staged,
            "old_content": marker_old, "backup": marker_backup,
            "new_content": {"exists": False, "size": 0, "sha256": ""},
        }
        journal["targets"].append(marker_entry)
        journal["phase"] = "prepared"
        _atomic_write_json_durable(journal, journal_path)
        journal_written = True

        # dataset 音频与 manifest 先提交；stats 后发布。
        for entry in journal["audio"]:
            if _file_fingerprint(entry["source"]) != entry["source_fingerprint"]:
                raise RuntimeError(f"source 在提交前变化: {entry['source']}")
            os.makedirs(os.path.dirname(entry["target"]), exist_ok=True)
            os.replace(entry["staged"], entry["target"])
        for entry in touched_manifest_entries:
            os.replace(entry["staged"], entry["target"])
        os.replace(stats_entry["staged"], stats_entry["target"])
        _fsync_directories(
            [entry["target"] for entry in journal["audio"]]
            + [entry["target"] for entry in touched_manifest_entries]
            + [stats_entry["target"]]
        )
        journal["phase"] = "published"
        _atomic_write_json_durable(journal, journal_path)

        for entry in journal["audio"]:
            if _content_fingerprint(entry["target"]) != entry["new_content"]:
                raise RuntimeError(f"destination SHA 校验失败: {entry['target']}")
            _unlink_source_if_unchanged(
                entry["source"],
                entry["source_fingerprint"],
                transaction_id=transaction_id,
                quarantine_root=entry["quarantine_root"],
            )

        marker_payload = {
            "version": PROMOTE_MARKER_VERSION,
            "manifest_schema_version": _EXPECTED_MANIFEST_SCHEMA_VERSION,
            "admission_policy_version": _EXPECTED_ADMISSION_POLICY_VERSION,
            "evidence_contract_version": _EXPECTED_EVIDENCE_CONTRACT_VERSION,
            "parser_version": _EXPECTED_PARSER_VERSION,
            "generator_version": _EXPECTED_GENERATOR_VERSION,
            "stage_d_dir": os.path.realpath(stage_d_dir),
            "final_root": os.path.realpath(final_root),
            "final_stage": stage_name,
            "d_stats_fingerprint": d_stats_fingerprint,
            "d2_manifest_fingerprint": stage_manifest_fingerprint,
            "d2_audio_fingerprint": _audio_tree_fingerprint(run_audio_root),
            "library_stats_fingerprint": _file_fingerprint(stats_target),
            "published_audio": [
                {"path": entry["target"], "content": entry["new_content"]}
                for entry in journal["audio"]
            ],
            "final_manifest_fingerprints": [
                {"path": entry["target"], "content": entry["new_content"]}
                for entry in touched_manifest_entries
            ],
            "published_records": _records_identity(published_records),
            "stage_b_tree_identity": stage_b_identity_before,
            "final_stage_tree_identity": _tree_identity(stage_output_dir),
            "dataset_count": len(datasets),
            "record_count": candidate_count,
            "file_count": len(plans),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_bytes_durable(
            marker_staged,
            (json.dumps(marker_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        marker_entry["new_content"] = _content_fingerprint(marker_staged)
        journal["phase"] = "publishing_marker"
        _atomic_write_json_durable(journal, journal_path)
        os.replace(marker_staged, marker_path)
        journal["phase"] = "committed"
        _atomic_write_json_durable(journal, journal_path)
    except BaseException as exc:
        logger.error("promote 事务失败，正在恢复 source/final: %s", exc)
        try:
            if journal_written:
                _restore_transaction(journal, journal_path)
            else:
                for owned_root in journal["owned_roots"]:
                    if os.path.isdir(owned_root):
                        shutil.rmtree(owned_root)
        except BaseException as recovery_exc:
            logger.critical("promote 恢复失败，保留 journal 阻止后续 promote/cleanup: %s", recovery_exc)
        return 1

    cleanup_errors: List[str] = []
    for owned_root in journal["owned_roots"]:
        try:
            if os.path.isdir(owned_root):
                shutil.rmtree(owned_root)
        except OSError as exc:
            cleanup_errors.append(f"{owned_root}: {exc}")
    if cleanup_errors:
        logger.error("promote 已提交但 owned artifact 清理失败；journal 保留并阻断后续操作: %s", cleanup_errors)
        return 1
    os.unlink(journal_path)
    _prune_empty_dirs(run_audio_root)
    logger.info("promote 完成: %d 条记录 / %d 个音频文件", candidate_count, len(plans))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-d-dir", required=True, help="本次运行的 stage_d 目录")
    parser.add_argument("--final-root", required=True, help="final/ 根目录")
    parser.add_argument(
        "--config",
        default=None,
        help="可选 pipeline.yaml 路径；仅用于复验 completion 中的 Stage D 配置身份",
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验与统计，不移动/不写盘")
    parser.add_argument(
        "--force",
        action="store_true",
        help="兼容参数；不能绕过 d_completion.json 或其绑定内容门禁",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    return promote_run(
        args.stage_d_dir,
        args.final_root,
        dry_run=args.dry_run,
        force=args.force,
        config_path=args.config,
    )


if __name__ == "__main__":
    sys.exit(main())
