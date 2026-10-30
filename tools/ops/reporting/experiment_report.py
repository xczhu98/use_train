#!/usr/bin/env python3
"""实验报告自动生成工具。

从训练实验目录自动发现资产、运行评估、解析结果，生成/更新累积式 Markdown 报告。

用法::

    # 一键全流程
    python -m tools.ops.reporting.experiment_report run exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

    # 仅发现资产
    python -m tools.ops.reporting.experiment_report discover exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

    # 仅运行评估（已有资产）
    python -m tools.ops.reporting.experiment_report evaluate exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

    # 仅更新报告（评估已跑完）
    python -m tools.ops.reporting.experiment_report report exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

    # 手动指定实验目的
    python -m tools.ops.reporting.experiment_report run exp/aec/xxx --purpose "添加 state detector"

    # 跳过评估（已跑过）
    python -m tools.ops.reporting.experiment_report run exp/aec/xxx --skip-eval
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator, Mapping, Optional, Union
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import torch
import torchaudio
from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluate.artifact_lifecycle import (
    ARTIFACT_SCHEMA,
    EVALUATION_RESERVATION_TOKEN_ENV,
    load_committed_artifact,
    release_artifact_reservation,
    reserve_artifact_run,
)
from evaluate.common import load_normalized_score, validate_result_manifest
from evaluate.eval_io import validate_inference_artifacts
from evaluate.eval_pipeline_common import (
    DEFAULT_TEST_DATA_ROOT,
    default_eval_output_root,
    detect_aec_subtype,
    detect_model_type,
    effective_config_fingerprint,
    find_exp_assets,
    guard_explicit_eval_output,
)
from evaluate.multi_gpu_inference import normalize_inference_devices
from evaluate.pipeline_runner import PipelineTaskAdapter
from evaluate.process_runner import EVAL_SUBPROCESS_TIMEOUT_S, death_signal_preexec
from evaluate.urgent_score import (
    NORMALIZED_METRICS,
    SCORE_FILENAME,
    project_existing_score,
    validate_normalized_score,
)
from core.utils.config import load_merged_config
from core.utils.file_transaction import write_text_files_transactionally
from core.utils.inference_artifacts import (
    ManifestValidationError,
    atomic_write_text,
    build_input_inventory,
    fingerprint_file,
    fingerprint_path,
    load_run_manifest,
)
from tools.model_tools.extract_model_checkpoint import (
    checkpoint_family_from_config,
    default_model_only_path,
    is_model_only_checkpoint_path,
    materialize_model_only_checkpoint,
    validate_model_only_checkpoint,
)
from tools.ops.reporting.experiment_report_identity import (
    V3_SCHEMA,
    build_identity_record,
    identity_from_payload,
)

_REAL_LOAD_RUN_MANIFEST = load_run_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("experiment_report")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
REPORT_DIR = _PROJECT_ROOT / "docs" / "reports" / "experiments"
REPORT_FILE = REPORT_DIR / "experiment_report.md"
INDEX_FILE = REPORT_DIR / "experiments.json"
REPORT_LOCK_FILE = REPORT_DIR / ".experiment_report.lock"
# 家族级人工对比分析备注；按 `## <model_family>` 分节，渲染在对应系列对比表之后。
ANALYSIS_NOTES_FILE = REPORT_DIR / "analysis_notes.md"

_PATH_TRAILING_PUNCTUATION = ".;:!?，。；：！？"
_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_MAX_PERCENT_DECODE_LAYERS = 4
_MAX_PERCENT_DECODE_LENGTH = 4096
_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_MAILTO_RE = re.compile(r"mailto:[^\s<>\"']+", re.IGNORECASE)
_API_ROUTE_RE = re.compile(
    r"(?<![\w/])/(?:api(?:/|$)|v\d+(?:/|$))[^\s<>\"']*",
    re.IGNORECASE,
)
_QUOTED_MACHINE_PATH_RE = re.compile(
    r"(?P<quote>[\"'])(?P<path>(?:file:|[A-Za-z]:[\\/]+|\\{2,}|/)[^\n]*?)(?P=quote)",
    re.IGNORECASE,
)
_FILE_URI_RE = re.compile(
    r"(?<![\w>])file:[^\s`\"'()<>\[\]{},]+",
    re.IGNORECASE,
)
_WINDOWS_PATH_RE = re.compile(
    r"(?<![\w>])(?:[A-Za-z]:[\\/]+|\\{2,})[^\s`\"'()<>\[\]{},]+"
)
_POSIX_PATH_RE = re.compile(r"(?<![:/\\\w><.])/(?!/)[^\s`\"'()<>\[\]{},]+")
_PUBLIC_PATH_KEYS = {
    "exp_dir", "config_path", "checkpoint_path", "source_checkpoint_path",
    "model_only_checkpoint_path", "expected_model_only_checkpoint_path",
    "log_dir", "val_metrics_csv", "train_val_curve_csv", "path",
    "expected_path", "reported_path",
    "onnx_temp_path", "json_path", "log_path", "result_dir", "output_dir",
    "delay_summary", "inference_log", "inf_scp", "output_path", "model", "config",
}


_SAFE_ARTIFACT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]*\Z")


def _validated_child_asset(value: Union[str, Path], *, label: str) -> Path:
    configured = Path(value).expanduser()
    if configured.is_symlink():
        raise ValueError(f"评估子进程 {label} 不允许符号链接")
    try:
        resolved = configured.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"评估子进程 {label} 不存在: {configured}") from exc
    if not resolved.is_file():
        raise ValueError(f"评估子进程 {label} 必须是普通文件: {resolved}")
    return resolved


def _evaluation_reservation_adapter(
    *,
    task_type: str,
    output_name: str,
    config_path: Optional[Path],
    checkpoint_path: Optional[Path],
    device: str,
    inference_devices: Optional[Union[str, list[str], tuple[str, ...]]],
    effective_config_identity: Optional[Mapping[str, str]] = None,
) -> PipelineTaskAdapter:
    model_type = task_type
    model_subtype: Optional[str] = None
    if config_path is not None:
        try:
            effective_config = load_merged_config(str(config_path))
            model_type = detect_model_type(
                effective_config, config_path=config_path
            )
            if task_type == "aec":
                model_subtype = detect_aec_subtype(
                    effective_config, config_path=config_path
                )
        except (OSError, TypeError, ValueError):
            # 子 pipeline 会在真正加载配置时报告配置错误；父进程仍需能释放 reservation。
            logger.debug("无法在父进程解析评估模型类型", exc_info=True)
    provenance: dict[str, Any] = {
        "parameters": {
            "step": "all",
            "model_type": model_type,
            "device": device,
            "inference_devices": normalize_inference_devices(
                inference_devices, fallback=device
            ),
            "output_name": output_name,
        }
    }
    if task_type == "aec":
        provenance["parameters"]["model_subtype"] = model_subtype or "unknown"
    if config_path is not None and checkpoint_path is not None:
        if effective_config_identity is None:
            effective_config_identity = effective_config_fingerprint(config_path)
        provenance.update(
            config=config_path,
            checkpoint=checkpoint_path,
            resolved_config_fingerprint=dict(effective_config_identity),
        )
    provenance["entry"] = f"eval_pipeline_{task_type}"
    return PipelineTaskAdapter(
        task=task_type,
        pipeline=f"eval_pipeline_{task_type}",
        step="all",
        output_dir=default_eval_output_root(_PROJECT_ROOT) / task_type / output_name,
        provenance=provenance,
    )


def _bounded_percent_decode(raw: str) -> tuple[str, bool]:
    """有界递归 percent-decode；超长或达到层数上限时标记未穷尽。"""
    current = raw[:_MAX_PERCENT_DECODE_LENGTH]
    exhausted = len(raw) > _MAX_PERCENT_DECODE_LENGTH
    for _ in range(_MAX_PERCENT_DECODE_LAYERS):
        decoded = unquote_plus(current)
        if decoded == current:
            return decoded, exhausted
        current = decoded[:_MAX_PERCENT_DECODE_LENGTH]
        exhausted = exhausted or len(decoded) > _MAX_PERCENT_DECODE_LENGTH
    return current, exhausted or bool(re.search(r"(?i)%[0-9a-f]{2}", current))


def _is_encoded_file_locator(raw: str) -> bool:
    decoded, exhausted = _bounded_percent_decode(raw)
    has_percent_octet = bool(re.search(r"(?i)%[0-9a-f]{2}", raw))
    return decoded.casefold().startswith("file:") or (exhausted and has_percent_octet)


def _is_canonical_external_placeholder(raw: str) -> bool:
    """仅信任无编码、无层级分隔符的稳定 external basename。"""
    prefix = "<external>/"
    if not raw.startswith(prefix):
        return False
    suffix = raw[len(prefix):]
    return bool(
        suffix
        and suffix not in {".", ".."}
        and not any(char in suffix for char in "/\\%<>")
        and not any(char in "\r\n" for char in suffix)
    )


def _sanitize_external_placeholders_linear(value: str) -> str:
    """单遍收敛伪造的 external placeholder，保留严格 canonical 形式。"""
    prefix = "<external>/"
    pieces: list[str] = []
    emit_start = 0
    index = 0
    while index < len(value):
        if not value.startswith(prefix, index):
            index += 1
            continue
        end = index + len(prefix)
        while end < len(value) and not value[end].isspace() and value[end] not in "`\"'()[]{} ,":
            end += 1
        candidate = value[index:end]
        pieces.append(value[emit_start:index])
        pieces.append(candidate if _is_canonical_external_placeholder(candidate) else "<external>/path")
        emit_start = end
        index = end
    pieces.append(value[emit_start:])
    return "".join(pieces)


def _scan_encoded_tokens_linear(value: str, render_token: Any) -> tuple[str, int]:
    """按有限字段边界单遍扫描含 percent octet 的 token。"""
    delimiters = "`\"'()<>[]{} ,=:;&?#"
    hex_digits = "0123456789abcdefABCDEF"
    pieces: list[str] = []
    token_start = 0
    token_has_octet = False
    operations = 0

    def emit_token(end: int) -> None:
        raw = value[token_start:end]
        pieces.append(render_token(raw) if raw and token_has_octet else raw)

    index = 0
    while index < len(value):
        operations += 1
        char = value[index]
        is_file_scheme_colon = (
            char == ":" and value[token_start:index].casefold() == "file"
        )
        if (char.isspace() or char in delimiters) and not is_file_scheme_colon:
            emit_token(index)
            pieces.append(char)
            token_start = index + 1
            token_has_octet = False
        elif (
            char == "%"
            and index + 2 < len(value)
            and value[index + 1] in hex_digits
            and value[index + 2] in hex_digits
        ):
            token_has_octet = True
        index += 1
    emit_token(len(value))
    return "".join(pieces), operations


def _contains_encoded_absolute_locator(value: str) -> bool:
    """检查任一有限字段片段的 encoded view 是否为绝对 locator。"""
    found = False

    def inspect(raw: str) -> str:
        nonlocal found
        found = found or _is_absolute_locator(raw)
        return raw

    _scan_encoded_tokens_linear(value, inspect)
    return found


def _is_absolute_locator(raw: str) -> bool:
    """判断字符串或其有界解码视图是否为绝对 machine locator。"""
    decoded, exhausted = _bounded_percent_decode(raw)
    has_percent_octet = bool(re.search(r"(?i)%[0-9a-f]{2}", raw))
    return (
        _is_encoded_file_locator(raw)
        or Path(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or raw.startswith("\\\\")
        or Path(decoded).is_absolute()
        or PureWindowsPath(decoded).is_absolute()
        or decoded.startswith("\\\\")
        or (exhausted and has_percent_octet)
    )


def _render_path(path: Union[str, Path]) -> str:
    """将运行时 locator 渲染为可发布形式，不改变实际文件访问路径。"""
    raw = str(path)
    if not raw:
        return raw
    if raw.startswith("<external>/"):
        return raw if _is_canonical_external_placeholder(raw) else "<external>/path"

    decoded, _ = _bounded_percent_decode(raw)
    if _is_encoded_file_locator(raw) or (
        decoded != raw
        and (
            Path(decoded).is_absolute()
            or PureWindowsPath(decoded).is_absolute()
            or decoded.startswith("\\\\")
        )
    ):
        return "<external>/path"

    windows_path = PureWindowsPath(raw)
    if windows_path.is_absolute() or raw.startswith("\\\\"):
        return f"<external>/{windows_path.name or 'path'}"

    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(_PROJECT_ROOT).as_posix()
        except ValueError:
            return f"<external>/{candidate.name or 'path'}"

    normalized = Path(os.path.normpath(raw))
    if normalized == Path("..") or ".." in normalized.parts[:1]:
        return f"<external>/{normalized.name or 'path'}"
    return normalized.as_posix()


def _replace_public_path(match: re.Match[str]) -> str:
    raw = match.group(0)
    path_text = raw.rstrip(_PATH_TRAILING_PUNCTUATION)
    suffix = raw[len(path_text):]
    return _render_path(path_text) + suffix


def _is_safe_repo_relative_target(target: str) -> bool:
    """判断 Markdown destination 及其有界解码视图是否为安全相对 target。"""
    decoded, exhausted = _bounded_percent_decode(target)
    has_percent_octet = bool(re.search(r"(?i)%[0-9a-f]{2}", target))
    if (
        not target
        or _is_absolute_locator(target)
        or _contains_encoded_absolute_locator(target)
        or (exhausted and has_percent_octet)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", decoded)
    ):
        return False
    for candidate in (target, decoded):
        path_text = candidate.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
        normalized = Path(os.path.normpath(path_text))
        if normalized == Path("..") or normalized.parts[:1] == ("..",):
            return False
    return True


def _is_sensitive_query_key(raw_key: str) -> bool:
    """判断 query key 是否可能承载认证凭据。"""
    decoded, exhausted = _bounded_percent_decode(raw_key)
    decoded = decoded.casefold()
    parts = {part for part in re.split(r"[^a-z0-9]+", decoded) if part}
    compact = re.sub(r"[^a-z0-9]", "", decoded)
    return bool(
        exhausted
        or parts & {"token", "key", "secret", "password", "passwd", "signature", "credential"}
        or compact in {
            "apikey", "accesskey", "privatekey", "clientsecret", "auth",
            "authorization", "sig", "bearer",
        }
    )


def _classification_has_sensitive_assignment(value: str) -> bool:
    """仅在 decoded classification view 中识别敏感赋值，不发布该视图。"""
    for field in re.split(r"[&;]", value):
        key, separator, _ = field.partition("=")
        if separator and _is_sensitive_query_key(key):
            return True
    return False


def _sanitize_public_component(value: str) -> Optional[str]:
    """清洗 URI query/fragment；不安全的 body 结构返回失败。"""
    decoded, exhausted = _bounded_percent_decode(value)
    if exhausted:
        return "REDACTED"
    introduced_delimiter = any(
        decoded.count(delimiter) > value.count(delimiter)
        for delimiter in ("&", ";", "=")
    )
    if introduced_delimiter and _classification_has_sensitive_assignment(decoded):
        return "REDACTED"

    pieces = re.split(r"([&;])", value)
    normalized = []
    for index, piece in enumerate(pieces):
        if index % 2:
            normalized.append(piece)
            continue
        raw_key, separator, raw_value = piece.partition("=")
        decoded_key, _ = _bounded_percent_decode(raw_key)
        if separator and decoded_key.casefold() == "body":
            decoded_value, body_exhausted = _bounded_percent_decode(raw_value)
            sanitized = (
                "<external>/path"
                if body_exhausted
                else _sanitize_free_text(decoded_value)
            )
            body = raw_value if sanitized == decoded_value else sanitized
            structure_view = _markdown_structure_view(body)
            if any(char in structure_view for char in " \t\r\n\"'<>[]()"):
                return None
            normalized.append(f"{raw_key}={body}")
        elif "<external>/" in piece:
            return None
        elif separator and _is_sensitive_query_key(raw_key):
            normalized.append(f"{raw_key}=REDACTED")
        else:
            normalized.append(piece)
    return "".join(normalized)


def _sanitize_public_query(query: str) -> Optional[str]:
    """兼容入口：清洗 public URI query，保留无歧义的原始编码。"""
    return _sanitize_public_component(query)


def _has_nested_markdown_structure(value: str) -> bool:
    """识别不允许出现在 Markdown 纯文本字段或 destination 中的内链结构。"""
    return value.lstrip(" \t").startswith("[") or "](" in value


def _markdown_structure_view(value: str) -> str:
    """移除可信 external placeholder，保留其余 Markdown 结构字符。"""
    return re.sub(
        r"<external>/[^\s`\"'()\[\]{},]+",
        lambda match: ""
        if _is_canonical_external_placeholder(match.group(0))
        else match.group(0),
        value,
    )


def _guard_normalized_public_destination(
    original: str, normalized: str
) -> Optional[str]:
    """拒绝 URI 规范化新引入的 Markdown destination 边界。"""
    original_view = _markdown_structure_view(original)
    normalized_view = _markdown_structure_view(normalized)
    if _has_nested_markdown_structure(normalized_view):
        return None
    boundary_chars = " \t\r\n\"'<>[]()"
    if any(
        normalized_view.count(char) > original_view.count(char)
        for char in boundary_chars
    ):
        return None
    return normalized


def _public_uri_path_has_machine_locator(path: str) -> bool:
    """检查 public URI path 的原始与有界解码视图是否嵌入机器 locator。"""
    def has_machine_locator(candidate: str) -> bool:
        return bool(
            "\\" in candidate
            or "<external>/" in candidate
            or re.search(r"(?i)(?:^|/)file:", candidate)
            or re.search(r"(?:^|/)[A-Za-z]:[/\\]", candidate)
        )

    decoded, exhausted = _bounded_percent_decode(path)
    if exhausted and re.search(r"(?i)%[0-9a-f]{2}", path):
        return True
    if any(has_machine_locator(candidate) for candidate in (path, decoded)):
        return True

    for segment in path.split("/"):
        has_percent_octet = bool(re.search(r"(?i)%[0-9a-f]{2}", segment))
        decoded_segment, segment_exhausted = _bounded_percent_decode(segment)
        if has_percent_octet and (
            segment_exhausted
            or decoded_segment.startswith("/")
            or has_machine_locator(segment)
            or has_machine_locator(decoded_segment)
        ):
            return True
    return False


def _normalize_public_api_route(destination: str) -> Optional[str]:
    """规范化 public API route；canonical placeholder 仅允许出现在字段值。"""
    before_fragment, marker, fragment = destination.partition("#")
    path, query_marker, query = before_fragment.partition("?")
    if not _API_ROUTE_RE.fullmatch(path) or _public_uri_path_has_machine_locator(path):
        return None
    normalized = path
    if query_marker:
        sanitized_query = _sanitize_public_query(query)
        if sanitized_query is None:
            return None
        normalized += f"?{sanitized_query}"
    if marker:
        sanitized_fragment = _sanitize_public_component(fragment)
        if sanitized_fragment is None:
            return None
        normalized += f"#{sanitized_fragment}"
    return _guard_normalized_public_destination(destination, normalized)


def _normalize_public_http_url(destination: str) -> Optional[str]:
    """规范化可发布 HTTP(S) URL；拒绝内部 host，并移除凭据。"""
    try:
        parsed = urlsplit(destination)
        host = parsed.hostname
        authority = parsed.netloc.rsplit("@", 1)[-1]
        parsed.port  # 触发非法端口校验。
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not destination.casefold().startswith(f"{parsed.scheme.casefold()}://")
        or not host
        or any(char.isspace() for char in host)
        or "\\" in parsed.netloc
        or re.fullmatch(r"[A-Za-z]:", authority)
    ):
        return None

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if re.fullmatch(r"[0-9.]+", host):
            return None
        address = None
    if address is not None:
        if (
            not address.is_global
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            return None
    else:
        labels = host.split(".")
        if (
            len(labels) < 2
            or host.casefold() == "localhost"
            or host.casefold().endswith(
                (".localhost", ".local", ".internal", ".localdomain", ".lan", ".home", ".corp", ".intranet")
            )
            or not all(
                re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                for label in labels
            )
        ):
            return None
    if _public_uri_path_has_machine_locator(parsed.path):
        return None

    scheme = destination.split(":", 1)[0]
    sanitized_query = _sanitize_public_query(parsed.query)
    sanitized_fragment = _sanitize_public_component(parsed.fragment)
    if sanitized_query is None or sanitized_fragment is None:
        return None
    normalized = urlunsplit(
        (
            scheme,
            authority,
            parsed.path,
            sanitized_query,
            sanitized_fragment,
        )
    )
    return _guard_normalized_public_destination(destination, normalized)


def _is_safe_http_destination(destination: str) -> bool:
    """判断 HTTP(S) destination 是否可规范化为 public URL。"""
    return _normalize_public_http_url(destination) is not None


def _normalize_public_mailto(destination: str) -> Optional[str]:
    """验证 mailto address，并清洗 query 中的凭据与 body locator。"""
    scheme, separator, payload = destination.partition(":")
    if not separator or scheme.casefold() != "mailto":
        return None
    before_fragment, fragment_marker, fragment = payload.partition("#")
    address, query_marker, query = before_fragment.partition("?")
    if not re.fullmatch(r"[^@/\\\s:]+@[^@/\\\s:]+", address):
        return None
    normalized = f"{scheme}:{address}"
    if query_marker:
        sanitized_query = _sanitize_public_query(query)
        if sanitized_query is None:
            return None
        normalized += f"?{sanitized_query}"
    if fragment_marker:
        sanitized_fragment = _sanitize_public_component(fragment)
        if sanitized_fragment is None:
            return None
        normalized += f"#{sanitized_fragment}"
    return _guard_normalized_public_destination(destination, normalized)


def _is_safe_mailto_destination(destination: str) -> bool:
    """验证 mailto destination 的 address 部分不是 machine locator。"""
    return _normalize_public_mailto(destination) is not None


def _render_markdown_destination(destination: str, protect: Any) -> str:
    """分类单个 Markdown destination，只保护明确安全的 public target。"""
    if destination.startswith("<external>/"):
        return protect(destination) if _is_canonical_external_placeholder(destination) else "<external>/path"
    angle_wrapped = destination.startswith("<") and destination.endswith(">")
    if destination.startswith("<") != destination.endswith(">"):
        return "<external>/path"
    inner = destination[1:-1] if angle_wrapped else destination
    if "](" in inner or inner.startswith("["):
        return "<external>/path"

    normalized_http = _normalize_public_http_url(inner)
    if normalized_http is not None:
        normalized = f"<{normalized_http}>" if angle_wrapped else normalized_http
        return protect(normalized)
    normalized_mailto = _normalize_public_mailto(inner)
    if normalized_mailto is not None:
        normalized = f"<{normalized_mailto}>" if angle_wrapped else normalized_mailto
        return protect(normalized)
    normalized_api_route = _normalize_public_api_route(inner)
    if normalized_api_route is not None:
        normalized = f"<{normalized_api_route}>" if angle_wrapped else normalized_api_route
        return protect(normalized)
    if "<external>/" in inner:
        return "<external>/path"
    if inner.startswith("#"):
        return protect(destination) if not _contains_encoded_absolute_locator(inner) else "<external>/path"
    if _is_safe_repo_relative_target(inner):
        return protect(destination)

    if angle_wrapped:
        return "<external>/path"
    if inner.startswith(("../", "..\\")):
        name = PureWindowsPath(inner).name if "\\" in inner else Path(inner).name
        return f"<external>/{name or 'path'}"
    if _is_absolute_locator(inner):
        return "<external>/path" if any(char.isspace() for char in inner) else _render_path(inner)
    return "<external>/path"


def _split_markdown_target_bounds(
    value: str, start: int, end: int
) -> tuple[Optional[tuple[int, int, Optional[int], Optional[int]]], int]:
    """在线性时间内返回 target 各字段边界，不复制可能很深的 title。"""
    operations = 0
    index = start
    while index < end and value[index] in " \t":
        operations += 1
        index += 1
    destination_start = index
    if index == end:
        return None, operations

    if value.startswith("<external>/", index):
        while index < end and not value[index].isspace():
            operations += 1
            index += 1
    elif value[index] == "<":
        index += 1
        while index < end and value[index] != ">":
            operations += 1
            index += 1
        if index == end:
            return None, operations
        index += 1
    else:
        while index < end and not value[index].isspace():
            operations += 1
            index += 1
    destination_end = index

    while index < end and value[index] in " \t":
        operations += 1
        index += 1
    if index == end:
        return (destination_start, destination_end, None, None), operations

    quote_start = index
    if destination_end == quote_start or value[quote_start] not in {"\"", "'"}:
        return None, operations

    content_end = end
    while content_end > quote_start and value[content_end - 1] in " \t":
        operations += 1
        content_end -= 1
    quote_end = content_end - 1
    if quote_end <= quote_start or value[quote_end] != value[quote_start]:
        return None, operations
    return (destination_start, destination_end, quote_start, quote_end), operations


def _split_markdown_target_linear(
    target: str,
) -> tuple[Optional[tuple[str, str, str, str, str]], int]:
    """线性拆分 Markdown destination 与可选 quoted title。"""
    bounds, operations = _split_markdown_target_bounds(target, 0, len(target))
    if bounds is None:
        return None, operations

    destination_start, destination_end, quote_start, quote_end = bounds
    leading = target[:destination_start]
    destination = target[destination_start:destination_end]
    if quote_start is None or quote_end is None:
        return (leading, destination, "", "", target[destination_end:]), operations
    return (
        leading,
        destination,
        target[destination_end:quote_start],
        target[quote_start:quote_end + 1],
        target[quote_end + 1:],
    ), operations


def _render_markdown_target(target: str, protect: Any) -> str:
    """拆分 Markdown destination、语法空白和 quoted title 后分别清洗。"""
    parts, _ = _split_markdown_target_linear(target)
    if parts is None:
        return "<external>/path"

    leading, raw_destination, title_separator, quoted_title, trailing = parts
    destination = _render_markdown_destination(raw_destination, protect)
    rendered = f"{leading}{destination}"
    if quoted_title:
        quote = quoted_title[0]
        title = _sanitize_markdown_text_field(quoted_title[1:-1])
        rendered += f"{title_separator}{quote}{title}{quote}"
    return f"{rendered}{trailing}"


def _scan_markdown_links(
    value: str,
    render_target: Any,
    protect_fragment: Optional[Any] = None,
    render_label: Optional[Any] = None,
) -> tuple[str, int]:
    """单次线性扫描 Markdown link，并分别清洗 label 与 target。"""
    pieces: list[str] = []
    emit_start = 0
    label_start: Optional[int] = None
    label_depth = 0
    index = 0
    operations = 0
    length = len(value)

    while index < length:
        operations += 1
        char = value[index]
        if char == "\n":
            label_start = None
            label_depth = 0
        elif char == "[":
            if label_start is None:
                label_start = index
            label_depth += 1
        elif char == "]" and label_start is not None:
            label_depth = max(0, label_depth - 1)
            if (
                label_depth == 0
                and index + 1 < length
                and value[index + 1] == "("
            ):
                target_start = index + 2
                cursor = target_start
                depth = 0
                closing: Optional[int] = None
                while cursor < length and value[cursor] != "\n":
                    operations += 1
                    current = value[cursor]
                    if current == "(":
                        depth += 1
                    elif current == ")":
                        if depth == 0:
                            closing = cursor
                            break
                        depth -= 1
                    cursor += 1

                target_end = closing if closing is not None else cursor
                target = value[target_start:target_end]
                rendered_target = render_target(target)
                raw_label = value[label_start + 1:index]
                label = render_label(raw_label) if render_label is not None else raw_label
                suffix = ")" if closing is not None else ""
                fragment = f"[{label}](" + rendered_target + suffix
                if protect_fragment is not None:
                    fragment = protect_fragment(fragment)
                pieces.append(value[emit_start:label_start])
                pieces.append(fragment)
                emit_start = target_end + (1 if closing is not None else 0)
                index = emit_start
                label_start = None
                label_depth = 0
                continue
            if label_depth == 0:
                label_start = None
        index += 1

    pieces.append(value[emit_start:])
    return "".join(pieces), operations


def _sanitize_markdown_text_field(value: str) -> str:
    """将 label/title 作为纯文本清洗；嵌套 Markdown 结构整体收敛。"""
    if _has_nested_markdown_structure(value):
        return "<external>/path"
    return _sanitize_free_text(value, scan_markdown=False)


def _protect_public_tokens(
    value: str, *, scan_markdown: bool = True
) -> tuple[str, dict[str, str], re.Pattern[str]]:
    """用每次调用不可预测的 token 域保护已验证 public 片段。"""
    while True:
        nonce = secrets.token_hex(16)
        token_prefix = f"\ue000{nonce}:"
        if token_prefix not in value:
            break
    protected: dict[str, str] = {}

    def protect(raw: str) -> str:
        token = f"{token_prefix}{len(protected)}\ue001"
        protected[token] = raw
        return token

    def protect_url(match: re.Match[str]) -> str:
        matched = match.group(0)
        raw = matched
        depth = 0
        for index, char in enumerate(raw):
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    raw = raw[:index]
                    break
                depth -= 1
        suffix = matched[len(raw):]
        punctuation = raw[len(raw.rstrip(_PATH_TRAILING_PUNCTUATION + ",，")):]
        url_raw = raw[:-len(punctuation)] if punctuation else raw
        normalized = _normalize_public_http_url(url_raw)
        if normalized is not None:
            return protect(normalized) + punctuation + suffix
        return f"<external>/path{punctuation}{suffix}"

    if scan_markdown:
        value, _ = _scan_markdown_links(
            value,
            lambda target: _render_markdown_target(target, lambda raw: raw),
            protect,
            _sanitize_markdown_text_field,
        )
    value = _HTTP_URL_RE.sub(protect_url, value)
    value = _MAILTO_RE.sub(
        lambda match: protect(normalized)
        if (normalized := _normalize_public_mailto(match.group(0))) is not None
        else "<external>/path",
        value,
    )
    value = _API_ROUTE_RE.sub(
        lambda match: protect(normalized)
        if (normalized := _normalize_public_api_route(match.group(0))) is not None
        else "<external>/path",
        value,
    )
    token_re = re.compile(re.escape(token_prefix) + r"\d+\ue001")
    return value, protected, token_re


def _restore_public_tokens(
    value: str, protected: dict[str, str], token_re: re.Pattern[str]
) -> str:
    """单次线性替换恢复当前调用域中的 public token。"""
    return token_re.sub(lambda match: protected[match.group(0)], value)


def _sanitize_unquoted_path_chunk(chunk: str) -> tuple[str, bool]:
    """清洗不含 protected token 的片段，并标记是否因歧义截断。"""
    matches = [
        match
        for pattern in (_FILE_URI_RE, _WINDOWS_PATH_RE, _POSIX_PATH_RE)
        for match in pattern.finditer(chunk)
    ]
    if not matches:
        return chunk, False

    first = min(matches, key=lambda match: match.start())
    remainder = chunk[first.end():]
    if remainder.strip(_PATH_TRAILING_PUNCTUATION + " \t"):
        prefix = chunk[:first.start()].rstrip()
        separator = " " if prefix else ""
        return f"{prefix}{separator}<external>/path", True

    chunk = _FILE_URI_RE.sub(_replace_public_path, chunk)
    chunk = _WINDOWS_PATH_RE.sub(_replace_public_path, chunk)
    return _POSIX_PATH_RE.sub(_replace_public_path, chunk), False


def _sanitize_unquoted_machine_paths(
    line: str, token_re: Optional[re.Pattern[str]] = None
) -> str:
    """显式迭代清洗单行 locator，protected token 将扫描域切成互斥片段。"""
    if token_re is None:
        return _sanitize_unquoted_path_chunk(line)[0]

    pieces: list[str] = []
    emit_start = 0
    for token_match in token_re.finditer(line):
        rendered, truncated = _sanitize_unquoted_path_chunk(
            line[emit_start:token_match.start()]
        )
        pieces.append(rendered)
        if truncated:
            pieces.append(" ")
        pieces.append(token_match.group(0))
        emit_start = token_match.end()
    pieces.append(_sanitize_unquoted_path_chunk(line[emit_start:])[0])
    return "".join(pieces)


def _sanitize_free_text(value: str, *, scan_markdown: bool = True) -> str:
    """清洗机器 locator；内部 title/label 按纯文本保守清洗。"""
    value = _sanitize_external_placeholders_linear(value)
    value, protected, token_re = _protect_public_tokens(
        value, scan_markdown=scan_markdown
    )

    def replace_quoted(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{quote}{_render_path(match.group('path'))}{quote}"

    def replace_encoded_locator(raw: str) -> str:
        return "<external>/path" if _is_absolute_locator(raw) else raw

    value = _QUOTED_MACHINE_PATH_RE.sub(replace_quoted, value)
    value, _ = _scan_encoded_tokens_linear(value, replace_encoded_locator)
    lines = value.splitlines(keepends=True)
    sanitized_lines = []
    for line in lines:
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        sanitized_lines.append(_sanitize_unquoted_machine_paths(body, token_re) + ending)
    return _restore_public_tokens("".join(sanitized_lines), protected, token_re)


def _sanitize_public_argv_value(raw: str) -> str:
    """按普通 argv 的同一策略清洗 option value。"""
    if _is_absolute_locator(raw):
        normalized_route = _normalize_public_api_route(raw)
        return normalized_route if normalized_route is not None else _render_path(raw)
    return _sanitize_free_text(raw)


def _sanitize_public_argv(argv: list[Any]) -> list[Any]:
    """逐 argv 元素清洗 locator，避免把命令整体当自由文本解析。"""
    rendered: list[Any] = []
    for arg in argv:
        if not isinstance(arg, (str, Path)):
            rendered.append(arg)
            continue
        raw = str(arg)
        if raw.startswith("--") and "=" in raw:
            option, value = raw.split("=", 1)
            rendered.append(f"{option}={_sanitize_public_argv_value(value)}")
        elif _is_absolute_locator(raw):
            rendered.append(_sanitize_public_argv_value(raw))
        else:
            rendered.append(_sanitize_free_text(raw))
    return rendered


def _is_public_path_key(key: Any) -> bool:
    return isinstance(key, str) and (
        key in _PUBLIC_PATH_KEYS or key.endswith(("_path", "_dir", "_file"))
    )


def _canonical_non_string_key(key: Any) -> str:
    """序列化受支持的非字符串动态 key；不可稳定表示时显式拒绝。"""
    if key is None:
        return "none"
    if type(key) is str:
        return f"str:{len(key)}:{key}"
    if type(key) is bool:
        return f"bool:{str(key).lower()}"
    if type(key) is int:
        return f"int:{key}"
    if type(key) is float:
        if not math.isfinite(key):
            raise ValueError("unsupported dynamic key: non-finite float")
        return f"float:{key.hex()}"
    if type(key) is tuple:
        parts = [_canonical_non_string_key(item) for item in key]
        return "tuple:[" + ",".join(f"{len(part)}:{part}" for part in parts) + "]"
    raise ValueError(f"unsupported dynamic key type: {type(key).__module__}.{type(key).__qualname__}")


def _dynamic_key_sort_identity(key: Any) -> str:
    """返回与插入顺序无关的动态 key 排序身份。"""
    if type(key) is str:
        return f"str:{len(key)}:{key}"
    return _canonical_non_string_key(key)


def _sanitize_public_key(key: Any) -> str:
    """清洗动态展示 key；危险 key 使用固定名称与完整稳定摘要。"""
    if type(key) is not str:
        identity = _canonical_non_string_key(key)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"<external>/key~{digest}"
    if re.fullmatch(r"<external>/key~[0-9a-f]{64}(?:~[1-9]\d*)?", key):
        return key

    decoded, exhausted = _bounded_percent_decode(key)
    decoded_views = [key, decoded]

    rendered = _sanitize_free_text(key)
    has_embedded_locator = any(
        view.startswith("<external>/")
        or re.search(r"(?i)file:", view)
        or re.search(r"[A-Za-z]:[\\/]", view)
        or re.search(r"\\{2,}[^\\\s]+[\\/]", view)
        or re.search(r"(?:^|[.\s=:(\[])/(?!api(?:/|$)|v\d+(?:/|$))", view, re.IGNORECASE)
        for view in decoded_views
    )
    has_sensitive_assignment = any(
        _is_sensitive_query_key(assignment.group("name"))
        for view in decoded_views
        for assignment in re.finditer(r"(?P<name>[A-Za-z0-9_%.-]+)\s*[:=]", view)
    ) or any(re.search(r"(?i)\bbearer\s+\S+", view) for view in decoded_views)
    has_percent_octet = bool(re.search(r"(?i)%[0-9a-f]{2}", key))
    if (
        rendered == key
        and not exhausted
        and not has_embedded_locator
        and not has_sensitive_assignment
        and not has_percent_octet
    ):
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"<external>/key~{digest}"


def _to_public_value(value: Any, key: str = "") -> Any:
    """将 runtime 值转换为只供 Markdown、JSON stdout 使用的 public DTO。"""
    if isinstance(value, dict):
        prepared = [
            (item_key, _sanitize_public_key(item_key), item)
            for item_key, item in value.items()
        ]
        groups: dict[str, list[Any]] = {}
        for item_key, public_key, _ in prepared:
            groups.setdefault(public_key, []).append(item_key)

        reserved = set(groups)
        assigned: dict[tuple[str, Any], str] = {}
        for public_key, item_keys in groups.items():
            if len(item_keys) == 1:
                assigned[(public_key, item_keys[0])] = public_key
                continue

            canonical_owner = next(
                (
                    item_key
                    for item_key in item_keys
                    if type(item_key) is str and item_key == public_key
                ),
                _MISSING,
            )
            if canonical_owner is not _MISSING:
                assigned[(public_key, canonical_owner)] = public_key
            ordinal = 1
            for item_key in sorted(
                (
                    item_key
                    for item_key in item_keys
                    if canonical_owner is _MISSING or item_key != canonical_owner
                ),
                key=_dynamic_key_sort_identity,
            ):
                candidate = f"{public_key}~{ordinal}"
                while candidate in reserved:
                    ordinal += 1
                    candidate = f"{public_key}~{ordinal}"
                assigned[(public_key, item_key)] = candidate
                reserved.add(candidate)
                ordinal += 1

        rendered: dict[str, Any] = {}
        ordered = sorted(
            (
                (assigned[(public_key, item_key)], item_key, item)
                for item_key, public_key, item in prepared
            ),
            key=lambda entry: entry[0],
        )
        for final_key, item_key, item in ordered:
            rendered[final_key] = _to_public_value(item, item_key)
        if len(rendered) != len(prepared):
            raise RuntimeError("public key collision after deterministic allocation")
        return rendered
    if isinstance(value, list):
        if key == "command":
            return _sanitize_public_argv(value)
        if key == "_config_diffs":
            public_diffs = []
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    path, values = item
                    public_diffs.append(
                        (_sanitize_public_key(path), _to_public_value(values))
                    )
                else:
                    public_diffs.append(_to_public_value(item))
            return public_diffs
        return [_to_public_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_public_value(item) for item in value)
    if isinstance(value, Path):
        return _render_path(value)
    if isinstance(value, str):
        return _render_path(value) if _is_public_path_key(key) else _sanitize_free_text(value)
    return value


def _public_dynamic_keys(mapping: dict[Any, Any]) -> list[str]:
    """在 stringify 前生成确定顺序的 public dynamic keys。"""
    return list(_to_public_value(mapping))


def _to_public_record(runtime_record: dict) -> dict:
    """深拷贝 runtime record 并转换为不含私有 locator 的 public DTO。"""
    runtime_copy = copy.deepcopy(runtime_record, {id(_MISSING): _MISSING})
    return _to_public_value(runtime_copy)


def _sanitize_published_paths(value: Any) -> Any:
    """兼容旧调用：把任意展示值转换为 public DTO/value。"""
    return _to_public_value(value)


class _PublishedPathFilter(logging.Filter):
    """统一清洗本工具输出到终端的日志文本。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _sanitize_free_text(str(record.msg))
        if record.args:
            record.args = _to_public_value(record.args)
        return True


logger.addFilter(_PublishedPathFilter())

# 默认数据集与原始数据源配置
AEC_DATASETS = "aec_official,urgent,dns_challenge"
SE_DATASETS = "voicebank,urgent,dns_challenge"
DEFAULT_AEC_TESTSET_ROOT = "data/aec/AEC_Challenge/datasets"
DEFAULT_URGENT_NOISY_SCP = (
    "data/shared/urgent26_track1_se_dataset/urgent26_nonblind_with_label/"
    "urgent26_noisy_nonblind.scp"
)
DEFAULT_URGENT_CLEAN_SCP = (
    "data/shared/urgent26_track1_se_dataset/urgent26_nonblind_with_label/"
    "urgent26_clean_nonblind.scp"
)
DEFAULT_VOICEBANK_CLEAN_SCP = "data/se/voicebank_demand/voicebank_clean_val.scp"
DEFAULT_VOICEBANK_NOISY_SCP = "data/se/voicebank_demand/voicebank_noisy_val.scp"
DEFAULT_DNS_CHALLENGE_NOISY_DIR = (
    "data/se/DNS_Challenge_test/V2_V3_Blindset/"
    "noisy_blind_testset_v3_challenge_withSNR_16k"
)

# 盲测 / 非盲测分类（按实际 eval 目录名子串匹配；dns_challenge 保留前缀以兼容后续年份后缀）
BLIND_DATASETS = {"blind_test_set_icassp2023", "dns_challenge"}

# 模型架构名提取规则：从 model.type 映射到展示名
MODEL_FAMILY_MAP = {
    "DBGNet": "DBGNet",
    "CAGCRN": "CAGCRN",
    "DeepVQE": "DeepVQE",
    "GTCRN": "GTCRN",
    "MossFormer": "MossFormer",
    "MP-SENet": "MP-SENet",
    "MambAttention": "MambAttention",
    "RWSA_Mamba_UNet": "RWSA Mamba UNet",
    "ZipEnhancer": "ZipEnhancer",
    "DPDFNet2": "DPDFNet2",
    "BSRNN": "USR-BSRNN",
    "USR_BSRNN": "USR-BSRNN",
}


# ---------------------------------------------------------------------------
# 资产发现
# ---------------------------------------------------------------------------
def _runtime_model_config(config: dict) -> dict:
    """返回 inference.py / onnx/export.py 实际构造的 model 配置。"""
    model = config.get("model", {})
    if isinstance(model, dict) and "generator" in model:
        model = model.get("generator") or {}
    return model if isinstance(model, dict) else {}


def _is_waveform_native_config(config: dict) -> bool:
    """与 inference.py 保持一致：allow_no_stft 且无顶层 STFT 即 waveform-native。"""
    return bool(config.get("meta", {}).get("allow_no_stft", False)) and not config.get(
        "stft"
    )


def _normalize_experiment_root(exp_dir: Union[str, Path]) -> Path:
    """将 checkpoint/log run 目录归一到词法实验根目录，保留 symlink 审计信息。"""
    path = Path(exp_dir).expanduser()
    if path.parent.name in {"checkpoints", "logs"}:
        return path.parent.parent
    return path


_CONFIG_FILENAMES = (
    "resolved_config.yaml",
    "resolved_config_resumed.yaml",
    "config.yaml",
)


def _find_same_run_config(run_dir: Path) -> Path:
    """返回 checkpoint run 内唯一按优先级发现的普通配置文件。"""
    config = next(
        (
            candidate
            for name in _CONFIG_FILENAMES
            if (candidate := run_dir / name).is_file() and not candidate.is_symlink()
        ),
        None,
    )
    if config is None:
        raise FileNotFoundError(f"checkpoint run 缺少同目录 config: {run_dir}")
    return config.resolve(strict=True)


def _report_asset_search_path(exp_dir: Path) -> Path:
    """选择报告默认发现的完整 run，排序不受 model-only 派生产物影响。"""
    candidates: list[tuple[int, int, int, str, Path]] = []

    def add_candidate(run_dir: Path) -> None:
        if not run_dir.is_dir() or run_dir.is_symlink():
            return
        checkpoint = run_dir / "best_model.tar"
        if not checkpoint.is_file() or checkpoint.is_symlink():
            return
        try:
            config = _find_same_run_config(run_dir)
            checkpoint_mtime = checkpoint.stat().st_mtime_ns
            config_mtime = config.stat().st_mtime_ns
        except (OSError, FileNotFoundError):
            return
        candidates.append(
            (
                max(checkpoint_mtime, config_mtime),
                checkpoint_mtime,
                config_mtime,
                str(run_dir),
                run_dir,
            )
        )

    add_candidate(exp_dir)
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.is_dir() and not checkpoints_dir.is_symlink():
        for run_dir in checkpoints_dir.iterdir():
            add_candidate(run_dir)
    if not candidates:
        return exp_dir
    return max(candidates)[-1]


def _resolve_explicit_checkpoint(exp_dir: Path, checkpoint: Union[str, Path]) -> tuple[Path, Path]:
    """严格绑定用户指定的训练 checkpoint 与其同 run 配置。"""
    configured_exp_dir = Path(exp_dir).expanduser()
    if configured_exp_dir.is_symlink():
        raise ValueError("实验目录不允许符号链接")
    canonical_exp_dir = configured_exp_dir.resolve(strict=True)
    checkpoints_configured = configured_exp_dir / "checkpoints"
    if checkpoints_configured.is_symlink():
        raise ValueError("实验 checkpoints 目录不允许符号链接")
    try:
        checkpoints_root = checkpoints_configured.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"实验目录缺少 checkpoints: {checkpoints_configured}") from exc
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"实验目录缺少 checkpoints: {checkpoints_root}")
    try:
        checkpoints_root.relative_to(canonical_exp_dir)
    except ValueError as exc:
        raise ValueError("实验 checkpoints 目录超出实验根目录") from exc

    configured = Path(checkpoint).expanduser()
    if configured.is_symlink():
        raise ValueError("显式 checkpoint 不允许符号链接")
    if is_model_only_checkpoint_path(configured):
        raise ValueError("显式 checkpoint 不能使用 model-only 派生文件")
    try:
        source = configured.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"显式 checkpoint 不存在: {configured}") from exc
    if not source.is_file():
        raise ValueError(f"显式 checkpoint 必须是普通文件: {source}")

    configured_run_dir = configured.parent
    if configured_run_dir.is_symlink():
        raise ValueError("显式 checkpoint 所在 run 不允许符号链接")
    try:
        source.relative_to(checkpoints_root)
    except ValueError as exc:
        raise ValueError(
            f"显式 checkpoint 必须位于该实验 checkpoints 目录: {source}"
        ) from exc
    run_dir = source.parent
    if run_dir.parent != checkpoints_root:
        raise ValueError(
            "显式 checkpoint 必须直接位于 checkpoints/<run>/，不允许跨 run 或嵌套目录"
        )
    return _find_same_run_config(run_dir), source


def _identity_label(value: str, *, fallback: str) -> str:
    """为显式 artifact identity 生成有限且 ASCII 安全的可读标签。"""
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return (label[:48] or fallback)


def _checkpoint_identity_record(
    exp_name: str,
    source_checkpoint: Path,
    source_fingerprint: Mapping[str, Any],
    effective_config_identity: Mapping[str, Any],
    *,
    explicit_checkpoint: bool,
) -> dict[str, Any]:
    """为显式 checkpoint 构造可持久化、可重算的 v3 identity record。"""
    if not explicit_checkpoint:
        return {"checkpoint_identity": exp_name}
    return build_identity_record(
        schema=V3_SCHEMA,
        experiment=exp_name,
        source_path=str(source_checkpoint.resolve(strict=True)),
        source_sha256=source_fingerprint.get("sha256"),
        config_sha256=effective_config_identity.get("sha256"),
    )


def _checkpoint_identity(
    exp_name: str,
    source_checkpoint: Path,
    source_fingerprint: Mapping[str, Any],
    effective_config_identity: Mapping[str, Any],
    *,
    explicit_checkpoint: bool,
) -> str:
    """兼容返回 identity 字符串；实际构造由纯逻辑模块负责。"""
    return _checkpoint_identity_record(
        exp_name,
        source_checkpoint,
        source_fingerprint,
        effective_config_identity,
        explicit_checkpoint=explicit_checkpoint,
    )["checkpoint_identity"]


def _artifact_identity(exp_data: Mapping[str, Any]) -> str:
    """返回本次 checkpoint 绑定的输出与索引身份。"""
    identity = exp_data.get("checkpoint_identity") or exp_data.get("identity")
    if isinstance(identity, str) and identity:
        return identity
    exp_name = exp_data.get("exp_name")
    if not isinstance(exp_name, str) or not exp_name:
        raise ValueError("实验缺少有效 exp_name")
    return exp_name


def _uses_checkpoint_specific_identity(exp_data: Mapping[str, Any]) -> bool:
    """判断是否必须使用显式 checkpoint 专属评估隔离。"""
    if exp_data.get("explicit_checkpoint"):
        return True
    identity = exp_data.get("checkpoint_identity") or exp_data.get("identity")
    exp_name = exp_data.get("exp_name")
    return isinstance(identity, str) and bool(identity) and identity != exp_name


def _has_valid_v3_identity_record(exp_data: Mapping[str, Any]) -> bool:
    """验证 v3 payload、digest 与当前显式 checkpoint 谱系完全一致。"""
    if not exp_data.get("explicit_checkpoint"):
        return False
    if exp_data.get("checkpoint_identity_schema") != V3_SCHEMA:
        return False
    payload = exp_data.get("checkpoint_identity_payload")
    if not isinstance(payload, Mapping):
        return False
    try:
        identity = identity_from_payload(payload)
    except (TypeError, ValueError):
        return False
    source_fingerprint = exp_data.get("source_checkpoint_fingerprint")
    resolved_fingerprint = exp_data.get("resolved_config_fingerprint")
    return (
        identity == exp_data.get("checkpoint_identity")
        and payload.get("experiment") == exp_data.get("exp_name")
        and payload.get("source_path") == exp_data.get("source_checkpoint_path")
        and isinstance(source_fingerprint, Mapping)
        and payload.get("source_sha256") == source_fingerprint.get("sha256")
        and isinstance(resolved_fingerprint, Mapping)
        and payload.get("effective_config_sha256")
        == resolved_fingerprint.get("sha256")
    )


def _has_consumed_checkpoint_binding(exp_data: Mapping[str, Any]) -> bool:
    """判断调用方是否提供了可严格复核的 discovered checkpoint 谱系。"""
    return all(
        isinstance(exp_data.get(key), Mapping)
        for key in (
            "source_checkpoint_fingerprint",
            "config_fingerprint",
            "resolved_config_fingerprint",
        )
    )


def _validate_consumed_checkpoint_before_child(exp_data: Mapping[str, Any]) -> None:
    """在启动评估子进程前复核已发现的 checkpoint 消费 binding。"""
    if exp_data.get("waveform_native"):
        raise ValueError(
            "waveform-native 模型没有可绑定的 root checkpoint；"
            "为避免子进程回退发现其他 checkpoint，报告评估不支持启动该模型"
        )
    if _has_consumed_checkpoint_binding(exp_data):
        _validate_consumed_checkpoint_binding(exp_data)
        return
    if exp_data.get("explicit_checkpoint"):
        raise ValueError("显式 checkpoint 缺少可复核的消费 binding")


def _checkpoint_specific_identity(exp_data: Mapping[str, Any]) -> str:
    """返回经验证的显式 checkpoint 专属安全 identity。"""
    identity = _artifact_identity(exp_data)
    if not _uses_checkpoint_specific_identity(exp_data):
        return identity
    if identity == exp_data.get("exp_name"):
        raise ValueError("显式 checkpoint 缺少专属 artifact identity")
    if _SAFE_ARTIFACT_NAME_RE.fullmatch(identity) is None:
        raise ValueError(f"checkpoint identity 不安全: {identity!r}")
    return identity


def _display_experiment_name(exp_data: Mapping[str, Any]) -> str:
    """返回报告中的实验基名；显式 checkpoint 附带其隔离 identity。"""
    exp_name = str(exp_data.get("exp_name", "unknown"))
    identity = _artifact_identity(exp_data)
    return exp_name if identity == exp_name else f"{exp_name} ({identity})"


def _eval_root_for(exp_data: Mapping[str, Any]) -> Path:
    """返回 checkpoint identity 对应的评估根目录。"""
    identity = _checkpoint_specific_identity(exp_data)
    return default_eval_output_root(_PROJECT_ROOT) / str(exp_data["task_type"]) / identity


def _config_dependency_fingerprints(config_path: Path) -> dict[str, dict]:
    """记录入口配置与实际声明的 base_config 文件身份，不重复执行 merged loader。"""
    root = next(
        (parent for parent in config_path.parents if parent.name == "configs"),
        config_path.parent,
    ).resolve()
    pending = [config_path.resolve(strict=True)]
    fingerprints: dict[str, dict] = {}
    while pending:
        current = pending.pop()
        current.relative_to(root)
        if current.is_symlink() or not current.is_file():
            raise ValueError(f"config dependency 必须是根内普通文件: {current}")
        key = str(current)
        if key in fingerprints:
            continue
        fingerprints[key] = fingerprint_file(current)
        raw = OmegaConf.load(current)
        base_value = raw.get("base_config")
        if base_value is None:
            continue
        base_paths = (
            list(base_value)
            if isinstance(base_value, (list, tuple)) or OmegaConf.is_list(base_value)
            else [base_value]
        )
        for value in base_paths:
            candidate = (current.parent / str(value)).resolve(strict=True)
            candidate.relative_to(root)
            pending.append(candidate)
    return fingerprints


def _validate_config_dependency_fingerprints(exp_data: Mapping[str, Any]) -> None:
    expected = exp_data.get("config_dependency_fingerprints")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("缺少 config dependency fingerprints")
    for path, fingerprint in expected.items():
        if not isinstance(path, str) or not isinstance(fingerprint, Mapping):
            raise ValueError("config dependency fingerprint 结构非法")
        if fingerprint_file(path) != dict(fingerprint):
            raise RuntimeError(f"effective config dependency 在发现后发生变化: {path}")


def discover_exp_assets(
    exp_dir: Union[str, Path], checkpoint: Optional[Union[str, Path]] = None
) -> dict:
    """从实验目录发现所有资产。

    checkpoint/logs 下的 run 目录会归一为实验根目录，但资产严格绑定用户指定的
    run；传入 logs run 时使用同名 checkpoints run，避免回退到根目录旧 alias。

    Returns:
        dict with keys: exp_dir, exp_name, config_path, checkpoint_path,
                        task_type, log_dir, git_info, val_metrics_csv,
                        train_val_curve_csv, resolved_config
    """
    configured_path = Path(exp_dir).expanduser()
    configured_exp_dir = _normalize_experiment_root(configured_path)
    if not configured_exp_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {configured_exp_dir}")
    requested_path = configured_path.resolve(strict=True)
    exp_dir = configured_exp_dir.resolve(strict=True)

    asset_search_path = requested_path
    if configured_path.parent.name == "logs":
        asset_search_path = exp_dir / "checkpoints" / configured_path.name
        if not asset_search_path.is_dir():
            raise FileNotFoundError(
                f"日志 run 缺少同名 checkpoint run: {asset_search_path}"
            )
    elif configured_path.parent.name != "checkpoints":
        asset_search_path = _report_asset_search_path(exp_dir)

    result = {
        "exp_dir": str(exp_dir),
        "exp_name": exp_dir.name,
    }

    # 报告必须发现原始完整 checkpoint，不能把已有派生物再次当作提取来源。
    explicit_checkpoint = checkpoint is not None
    if explicit_checkpoint:
        config_path, source_checkpoint_path = _resolve_explicit_checkpoint(
            configured_exp_dir, checkpoint
        )
    else:
        config_path, source_checkpoint_path = find_exp_assets(
            asset_search_path, prefer_model_only=False
        )
    source_checkpoint_path = Path(source_checkpoint_path).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    source_fingerprint = fingerprint_file(source_checkpoint_path)
    config_fingerprint = fingerprint_file(config_path)
    cfg = load_merged_config(str(config_path))
    resolved_config_fingerprint = effective_config_fingerprint(cfg)
    result["config_path"] = str(config_path)
    result["config_fingerprint"] = config_fingerprint
    result["config_dependency_fingerprints"] = _config_dependency_fingerprints(
        config_path
    )
    result["resolved_config_fingerprint"] = resolved_config_fingerprint
    result["source_checkpoint_path"] = str(source_checkpoint_path)
    result["checkpoint_path"] = None
    result["source_checkpoint_fingerprint"] = source_fingerprint
    result["explicit_checkpoint"] = explicit_checkpoint
    result.update(_checkpoint_identity_record(
        result["exp_name"],
        source_checkpoint_path,
        source_fingerprint,
        resolved_config_fingerprint,
        explicit_checkpoint=explicit_checkpoint,
    ))
    expected_model_only = default_model_only_path(source_checkpoint_path)
    result["expected_model_only_checkpoint_path"] = str(expected_model_only)
    result["model_only_checkpoint_path"] = None
    result["model_only_checkpoint_fingerprint"] = None

    # 检测任务类型
    result["task_type"] = detect_model_type(cfg, config_path=config_path)

    # 加载配置
    result["resolved_config"] = cfg
    result["checkpoint_family"] = checkpoint_family_from_config(cfg).value
    result["waveform_native"] = _is_waveform_native_config(cfg)
    if not result["waveform_native"] and expected_model_only.is_file():
        try:
            validate_model_only_checkpoint(
                expected_model_only,
                source_path=source_checkpoint_path,
                family=result["checkpoint_family"],
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        else:
            result["model_only_checkpoint_path"] = str(expected_model_only)
            result["model_only_checkpoint_fingerprint"] = fingerprint_file(
                expected_model_only
            )
            result["checkpoint_path"] = str(expected_model_only)
    if result["waveform_native"]:
        result["checkpoint_path"] = None

    # 显式 checkpoint 只能使用同名 run 的训练元数据，不能把较新的 run 日志或
    # git_info 写进当前 checkpoint 专属报告；默认发现保持历史的“最新 run”策略。
    logs_base = exp_dir / "logs"
    result["log_dir"] = None
    result["val_metrics_csv"] = None
    result["train_val_curve_csv"] = None
    source_run_dir = source_checkpoint_path.parent
    source_is_run = source_run_dir.parent.name == "checkpoints"
    selected_log = None
    if explicit_checkpoint:
        candidate = logs_base / source_run_dir.name
        if candidate.is_dir() and not candidate.is_symlink():
            selected_log = candidate.resolve(strict=True)
    elif logs_base.exists():
        log_dirs = sorted(
            [d for d in logs_base.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if log_dirs:
            selected_log = log_dirs[0]
    if selected_log is not None:
        result["log_dir"] = str(selected_log)
        vm = selected_log / "val_metrics.csv"
        if vm.is_file() and (not explicit_checkpoint or not vm.is_symlink()):
            result["val_metrics_csv"] = str(vm)
        tvc = selected_log / "train_val_curve.csv"
        if tvc.is_file() and (not explicit_checkpoint or not tvc.is_symlink()):
            result["train_val_curve_csv"] = str(tvc)

    # 显式 checkpoint 同样只读取来源 run 的 git_info；默认路径保留最新可用记录。
    ckpt_base = exp_dir / "checkpoints"
    result["git_info"] = None
    if explicit_checkpoint and source_is_run:
        gi = source_run_dir / "git_info.json"
        if gi.is_file() and not gi.is_symlink():
            with open(gi) as f:
                result["git_info"] = json.load(f)
    elif not explicit_checkpoint and ckpt_base.exists():
        for sub in sorted(ckpt_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if sub.is_dir():
                gi = sub / "git_info.json"
                if gi.exists():
                    with open(gi) as f:
                        result["git_info"] = json.load(f)
                    break

    # 实验时间（从来源 checkpoint 目录时间戳获取）
    result["timestamp"] = None
    ckpt_parent = Path(source_checkpoint_path).parent
    # checkpoint 通常在 checkpoints/<timestamp>/ 下
    if ckpt_parent.parent.name == "checkpoints":
        ts_str = ckpt_parent.name  # e.g., "2026-07-03_17-47-51"
        try:
            result["timestamp"] = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            result["timestamp"] = datetime.fromtimestamp(ckpt_parent.stat().st_mtime)

    if result["timestamp"] is None:
        result["timestamp"] = datetime.now()

    return result


def _validate_consumed_checkpoint_binding(exp_data: Mapping[str, Any]) -> Path:
    """验证当前消费 model-only checkpoint 仍严格绑定发现时 source/config。"""
    if exp_data.get("waveform_native"):
        raise ValueError("waveform-native 模型不消费 root model-only checkpoint")
    source_value = exp_data.get("source_checkpoint_path")
    config_value = exp_data.get("config_path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("缺少来源训练 checkpoint")
    if not isinstance(config_value, str) or not config_value:
        raise ValueError("缺少同 run config")
    source_configured = Path(source_value).expanduser()
    config_configured = Path(config_value).expanduser()
    if source_configured.is_symlink() or config_configured.is_symlink():
        raise ValueError("消费 checkpoint/config 不允许符号链接")
    source = source_configured.resolve(strict=True)
    config_path = config_configured.resolve(strict=True)
    if not source.is_file() or not config_path.is_file():
        raise ValueError("消费 checkpoint/config 必须是普通文件")
    expected_source_fingerprint = exp_data.get("source_checkpoint_fingerprint")
    if not isinstance(expected_source_fingerprint, Mapping) or (
        fingerprint_file(source) != dict(expected_source_fingerprint)
    ):
        raise RuntimeError("来源训练 checkpoint 在发现后发生变化")
    expected_config_fingerprint = exp_data.get("config_fingerprint")
    if not isinstance(expected_config_fingerprint, Mapping) or (
        fingerprint_file(config_path) != dict(expected_config_fingerprint)
    ):
        raise RuntimeError("同 run config 在发现后发生变化")
    _validate_config_dependency_fingerprints(exp_data)
    expected_resolved_config_fingerprint = exp_data.get("resolved_config_fingerprint")
    resolved_config = exp_data.get("resolved_config")
    if (
        not isinstance(expected_resolved_config_fingerprint, Mapping)
        or not isinstance(resolved_config, Mapping)
        or effective_config_fingerprint(resolved_config)
        != dict(expected_resolved_config_fingerprint)
    ):
        raise RuntimeError("effective merged config mapping 在发现后发生变化")

    expected_model_only = default_model_only_path(source)
    expected_value = exp_data.get("expected_model_only_checkpoint_path")
    if expected_value is not None and Path(str(expected_value)).expanduser().resolve() != expected_model_only:
        raise ValueError("记录的 model-only checkpoint 路径与来源不一致")
    consumed_value = exp_data.get("checkpoint_path") or exp_data.get(
        "model_only_checkpoint_path"
    )
    if not isinstance(consumed_value, str) or not consumed_value:
        raise ValueError("缺少已 materialize 的 model-only checkpoint")
    consumed_configured = Path(consumed_value).expanduser()
    if consumed_configured.is_symlink():
        raise ValueError("消费 model-only checkpoint 不允许符号链接")
    consumed = consumed_configured.resolve(strict=True)
    if consumed != expected_model_only or not consumed.is_file():
        raise ValueError("消费 model-only checkpoint 与来源专属路径不一致")
    validate_model_only_checkpoint(
        consumed,
        source_path=source,
        family=exp_data["checkpoint_family"],
    )
    expected_model_only_fingerprint = exp_data.get("model_only_checkpoint_fingerprint")
    if isinstance(expected_model_only_fingerprint, Mapping) and (
        fingerprint_file(consumed) != dict(expected_model_only_fingerprint)
    ):
        raise RuntimeError("消费 model-only checkpoint 在发现后发生变化")
    return consumed


def _materialize_consumed_checkpoint(
    exp_data: dict, *, trusted_training_checkpoint: bool
) -> None:
    """在任何模型执行前固化实际消费 checkpoint 与内容身份。"""
    if "source_checkpoint_path" not in exp_data:
        return
    if exp_data.get("waveform_native"):
        exp_data["checkpoint_path"] = None
        exp_data["model_only_checkpoint_path"] = None
        exp_data["model_only_checkpoint_fingerprint"] = None
        return
    source = exp_data["source_checkpoint_path"]
    expected_source_fingerprint = exp_data.get("source_checkpoint_fingerprint")
    if not isinstance(expected_source_fingerprint, Mapping) or (
        fingerprint_file(source) != dict(expected_source_fingerprint)
    ):
        raise RuntimeError("来源训练 checkpoint 在发现后发生变化")
    config_path = exp_data.get("config_path")
    expected_config_fingerprint = exp_data.get("config_fingerprint")
    if not isinstance(expected_config_fingerprint, Mapping) or not config_path or (
        fingerprint_file(config_path) != dict(expected_config_fingerprint)
    ):
        raise RuntimeError("同 run config 在发现后发生变化")
    _validate_config_dependency_fingerprints(exp_data)
    expected_resolved_config_fingerprint = exp_data.get("resolved_config_fingerprint")
    resolved_config = exp_data.get("resolved_config")
    if (
        not isinstance(expected_resolved_config_fingerprint, Mapping)
        or not isinstance(resolved_config, Mapping)
        or effective_config_fingerprint(resolved_config)
        != dict(expected_resolved_config_fingerprint)
    ):
        raise RuntimeError("effective merged config mapping 在发现后发生变化")
    output = materialize_model_only_checkpoint(
        source,
        family=exp_data["checkpoint_family"],
        trusted_training_checkpoint=trusted_training_checkpoint,
    )
    exp_data["checkpoint_path"] = str(output)
    exp_data["model_only_checkpoint_path"] = str(output)
    exp_data["model_only_checkpoint_fingerprint"] = fingerprint_file(output)
    _validate_consumed_checkpoint_binding(exp_data)


# ---------------------------------------------------------------------------
# 实验目的推断
# ---------------------------------------------------------------------------
def infer_experiment_purpose(exp_data: dict) -> str:
    """从训练配置、git commit、exp_name 等自动推断实验目的。"""
    purposes = []
    cfg = exp_data.get("resolved_config", {})

    # 1. 模型类型 + variant/规模
    runtime_model = _runtime_model_config(cfg)
    model_type = runtime_model.get("type", "unknown")
    model_args = runtime_model.get("args", {})
    variant = model_args.get("variant", "")
    compression = model_args.get("compression", None)
    model_desc = f"模型: {model_type}"
    if variant:
        model_desc += f" ({variant})"
    if compression is not None:
        model_desc += f", compression={compression}"
    purposes.insert(0, model_desc)

    # 2. 训练时随机延时配置
    sim_cfg = cfg.get("data", {}).get("train", {}).get("simulation_cfg", {})
    delay_prob = sim_cfg.get("aec_ref_echo_random_delay_prob", None)
    _delay_range = sim_cfg.get("aec_ref_echo_delay_range_ms", None)
    delay_min = None if _delay_range is None else _delay_range[0]
    delay_max = None if _delay_range is None else _delay_range[1]
    delay_dist = sim_cfg.get("aec_ref_echo_random_delay_distribution", None)

    if delay_prob is not None and delay_prob > 0:
        delay_desc = f"训练随机延时: {delay_dist}({delay_min}~{delay_max}ms), prob={delay_prob}"
        purposes.append(delay_desc)
    elif delay_prob is not None:
        purposes.append("训练: 无随机延时")

    # 3. 训练时是否启用 AECDeploymentFrontend 的 Streaming GCC-PHAT。
    delay_alignment_cfg = _get_cfg_value(
        cfg, "aec_frontend", "delay_alignment", default=None
    )
    if isinstance(delay_alignment_cfg, dict):
        align_enabled = delay_alignment_cfg.get("enabled", False)
        if align_enabled:
            purposes.append("训练时: 启用 AECDeploymentFrontend Streaming GCC-PHAT 对齐")
        else:
            purposes.append("训练时: 关闭 AECDeploymentFrontend Streaming GCC-PHAT 对齐")

    # 5. 模型规模: 参数量
    model_size_desc = _infer_model_size(exp_data)
    if model_size_desc:
        purposes.append(model_size_desc)

    # 6. 与同系列历史实验的对比
    comparison_desc = _compare_with_previous(exp_data)
    if comparison_desc:
        purposes.append(comparison_desc)

    # 7. 从 git commit message（仅保留与实验设计相关的 commit）
    git_info = exp_data.get("git_info")
    if git_info:
        commit_msg = git_info.get("commit_message", "")
        if not commit_msg:
            commit_hash = git_info.get("commit", "")
            if commit_hash:
                try:
                    result = subprocess.run(
                        ["git", "log", "-1", "--format=%s", commit_hash],
                        capture_output=True, text=True, cwd=_PROJECT_ROOT,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        commit_msg = result.stdout.strip()
                except Exception:
                    pass
        # 只保留与训练/模型/实验设计相关的 commit，过滤 pure bugfix/refactor
        if commit_msg:
            _RELEVANT_COMMIT_PREFIXES = [
                "feat(", "feat:", "exp(", "exp:", "train(", "train:",
                "model(", "model:", "loss(", "loss:", "perf(",
            ]
            if any(commit_msg.startswith(p) for p in _RELEVANT_COMMIT_PREFIXES):
                purposes.append(f"commit: {commit_msg}")

    # 8. 从 exp_name 提取关键特征
    exp_name = exp_data.get("exp_name", "")
    features = _extract_features_from_name(exp_name)
    if features:
        purposes.append(f"特征: {', '.join(features)}")

    if not purposes:
        purposes.append("(未指定)")

    return " | ".join(purposes)


def _infer_model_size(exp_data: dict) -> str:
    """推断模型规模（参数量、FLOPs）。"""
    try:
        checkpoint_path = exp_data.get("checkpoint_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return ""
        from core.utils.safe_load import safe_torch_load
        # 参数量推断只使用默认安全模式；该字段缺失会让报告少一行，
        # 但不会掩盖 checkpoint 本身的问题（加载失败在此记录后跳过）
        ckpt = safe_torch_load(checkpoint_path, purpose="param count inference")
        # Count parameters - check 'model', 'model_state_dict', 'state_dict' keys
        total_params = 0
        for key in ["model", "model_state_dict", "state_dict"]:
            state = ckpt.get(key, {})
            if state:
                for v in state.values():
                    if hasattr(v, "numel"):
                        total_params += v.numel()
                if total_params > 0:
                    break
        if total_params > 0:
            if total_params >= 1e6:
                return f"参数量: {total_params/1e6:.1f}M"
            else:
                return f"参数量: {total_params/1e3:.0f}K"
    except Exception as exc:  # noqa: BLE001
        # 参数量只是报告的可选字段：记录失败原因后返回空串，报告主体继续生成
        logger.debug("参数数量推断失败，跳过该字段: %s", exc)
    return ""


def _compare_with_previous(exp_data: dict) -> str:
    """与同系列历史实验对比配置差异。"""
    try:
        index_file = REPORT_DIR / "experiments.json"
        if not index_file.exists():
            return ""
        with open(index_file) as f:
            history = json.load(f)

        model_family = exp_data.get("model_family", "")
        identity = _artifact_identity(exp_data)
        this_cfg = exp_data.get("resolved_config", {})

        # 找同系列、不是相同 checkpoint identity 的实验。
        siblings = [
            h for h in history
            if h.get("model_family") == model_family
            and _artifact_identity(h) != identity
        ]
        if not siblings:
            return ""

        # 找最近的同系列实验
        recent = max(siblings, key=lambda h: h.get("timestamp", ""))
        recent_name = recent.get("exp_name", "")

        # 加载历史实验配置进行对比
        prev_cfg = {}
        prev_cfg_path = recent.get("config_path")
        if prev_cfg_path:
            try:
                prev_cfg = load_merged_config(prev_cfg_path)
            except Exception:
                prev_cfg = {}

        # 对比关键配置差异
        diffs = _diff_configs(this_cfg, prev_cfg)
        if diffs:
            return f"vs {recent_name[:30]}: {diffs}"
        else:
            return f"同系列对比: {recent_name[:30]} (配置相近)"
    except Exception:
        return ""


def _get_cfg_value(cfg: dict, *keys, default=None):
    """安全嵌套取值。"""
    v = cfg
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k, default)
        else:
            return default
    return v


def _diff_configs(this_cfg: dict, prev_cfg: dict) -> str:
    """简单对比两个实验的关键配置差异。"""
    diffs = []

    # 1. 最大延时范围
    this_delay = _extract_delay_ms(this_cfg)
    prev_delay = _extract_delay_ms(prev_cfg)
    if this_delay is not None and prev_delay is not None and this_delay != prev_delay:
        diffs.append(f"delay={this_delay}ms (was {prev_delay}ms)")
    elif this_delay is not None and prev_delay is None:
        diffs.append(f"delay={this_delay}ms")

    # 2. 模型 variant
    this_variant = _get_cfg_value(this_cfg, "model", "args", "variant")
    prev_variant = _get_cfg_value(prev_cfg, "model", "args", "variant")
    if this_variant and this_variant != prev_variant:
        diffs.append(f"variant={this_variant}")

    # 3. AECDeploymentFrontend 的 Streaming GCC-PHAT 开关。
    this_align = _get_cfg_value(
        this_cfg, "aec_frontend", "delay_alignment", "enabled"
    )
    prev_align = _get_cfg_value(
        prev_cfg, "aec_frontend", "delay_alignment", "enabled"
    )
    if isinstance(this_align, bool) and this_align != prev_align:
        diffs.append(
            "AECDeploymentFrontend Streaming GCC-PHAT=on"
            if this_align
            else "AECDeploymentFrontend Streaming GCC-PHAT=off"
        )

    # 4. Loss 类型（取 optimizer.loss 或 train.loss 的第一个 key）
    this_loss = _get_primary_loss_type(this_cfg)
    prev_loss = _get_primary_loss_type(prev_cfg)
    if this_loss and this_loss != prev_loss:
        diffs.append(f"loss={this_loss}")

    return ", ".join(diffs)


def _get_primary_loss_type(cfg: dict) -> str | None:
    """从配置中提取主 loss 类型名称（简化展示）。"""
    loss_cfg = _get_cfg_value(cfg, "optimizer", "loss") or _get_cfg_value(cfg, "train", "loss")
    if isinstance(loss_cfg, dict) and loss_cfg:
        return next(iter(loss_cfg.keys()), None)
    if isinstance(loss_cfg, str):
        return loss_cfg
    return None


def _extract_delay_ms(cfg: dict) -> int | None:
    """从配置中提取最大延时。"""
    sim_cfg = cfg.get("data", {}).get("train", {}).get("simulation_cfg", {})
    _delay_range = sim_cfg.get("aec_ref_echo_delay_range_ms", None)
    if isinstance(_delay_range, (list, tuple)) and len(_delay_range) >= 2:
        return _delay_range[1]
    return None


def _extract_features_from_name(exp_name: str) -> list[str]:
    """从实验名中提取有意义的特征标签。"""
    features = []
    keywords = {
        "causal": "因果",
        "noncausal": "非因果",
        "offline": "离线",
        "dynamic": "动态模拟",
        "small": "小模型",
        "large": "大模型",
        "gan": "GAN训练",
        "metricgan": "MetricGAN",
        "delay0ms": "无延迟",
        "delay120ms": "120ms延迟",
        "delay400ms": "400ms延迟",
        "hybrid": "混合Loss",
        "fwsdr": "fwSDR",
        "statece": "StateCE",
        "joint": "Joint",
        "denoise": "降噪",
        "10ms": "10ms帧",
        "16ms": "16ms帧",
    }
    name_lower = exp_name.lower()
    for key, label in keywords.items():
        if key in name_lower:
            features.append(label)
    return features


def infer_model_family(config: dict) -> str:
    """从配置推断模型架构族名称；GAN 展示 generator 类型。"""
    model_type = _runtime_model_config(config).get("type", "Unknown")
    # 检查映射表（优先匹配更长的 key）
    for key, family in sorted(MODEL_FAMILY_MAP.items(), key=lambda x: -len(x[0])):
        if key.lower() in model_type.lower():
            return family
    return model_type


# ---------------------------------------------------------------------------
# 训练指标解析
# ---------------------------------------------------------------------------
def parse_training_metrics(exp_data: dict) -> dict:
    """解析训练日志中的指标。

    Returns:
        dict with: last_epoch（最后一行 epoch，不是最优 epoch）、
        best_val_loss_epoch（val_loss 最低的 epoch）、total_epochs、
        best_val_loss、final_epoch_metrics。
    """
    result = {
        "last_epoch": None,
        "total_epochs": None,
        "best_val_loss": None,
        "best_val_metrics": {},
        "final_epoch_metrics": {},
    }

    # 解析 val_metrics.csv
    vm_path = exp_data.get("val_metrics_csv")
    if vm_path and Path(vm_path).exists():
        rows = []
        with open(vm_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows:
            result["total_epochs"] = len(rows)

            # 最后一行 (latest epoch)
            last = rows[-1]
            result["final_epoch_metrics"] = {k: _safe_float(v) for k, v in last.items()}
            result["last_epoch"] = int(float(last.get("epoch", len(rows))))

            # 找 val loss 最低的 epoch
            if "val_loss" in rows[0]:
                best_row = min(rows, key=lambda r: float(r.get("val_loss", float("inf"))))
                result["best_val_loss"] = float(best_row["val_loss"])
                result["best_val_loss_epoch"] = int(float(best_row.get("epoch", 0)))

    # 解析 train_val_curve.csv 补充 val_loss
    tvc_path = exp_data.get("train_val_curve_csv")
    if tvc_path and Path(tvc_path).exists():
        rows = []
        with open(tvc_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            # 找 val_loss 最低的 epoch (如果 val_metrics.csv 没提供)
            if result["best_val_loss"] is None and "val_loss" in rows[0]:
                best_row = min(rows, key=lambda r: float(r.get("val_loss", float("inf"))))
                result["best_val_loss"] = float(best_row["val_loss"])
                result["best_val_loss_epoch"] = int(float(best_row.get("epoch", 0)))

            # total epochs
            if result["total_epochs"] is None:
                result["total_epochs"] = len(rows)

    return result


def _positive_epoch(value: Any) -> Optional[int]:
    """将 checkpoint/指标中的 epoch 收敛为正整数。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    return None


def _resolve_checkpoint_epoch(exp_data: dict) -> tuple[Optional[int], str]:
    """优先安全读取来源训练 epoch，必要时回退最佳验证 epoch。"""
    checkpoint_path = exp_data.get("source_checkpoint_path") or exp_data.get(
        "checkpoint_path"
    )
    if checkpoint_path:
        try:
            from core.utils.safe_load import safe_torch_load

            checkpoint = safe_torch_load(
                checkpoint_path,
                purpose="experiment report checkpoint epoch",
            )
            if isinstance(checkpoint, dict):
                epoch = _positive_epoch(checkpoint.get("epoch"))
                if epoch is not None:
                    return epoch, "checkpoint.epoch"
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("无法安全读取 checkpoint epoch: %s", exc)

    checkpoint_name = Path(checkpoint_path).name if checkpoint_path else ""
    training_metrics = exp_data.get("training_metrics") or {}
    if checkpoint_name == "best_model.tar":
        epoch = _positive_epoch(training_metrics.get("best_val_loss_epoch"))
        if epoch is not None:
            return epoch, "training_metrics.best_val_loss_epoch"
    return None, "epoch_unknown"


def _safe_float(v: str) -> Union[float, str]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ---------------------------------------------------------------------------
# 评估结果解析
# ---------------------------------------------------------------------------
def parse_all_results(eval_root: Union[str, Path]) -> dict[str, dict]:
    """解析 eval 目录下所有 RESULTS.txt。

    Returns:
        {dataset_name: {metric_name: value, ...}}
        对于 aec_sim，结果按 scenario 分组。
    """
    eval_root = Path(eval_root)
    results = {}

    if not eval_root.exists():
        return results

    expected_run_id = None
    expected_summary = None
    for summary_name in ("summary.json", "evaluation_artifact.json"):
        summary_path = eval_root / summary_name
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation summary: {summary_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid evaluation summary: {summary_path}")
        candidate = payload.get("run_id")
        if expected_run_id is None:
            expected_run_id = candidate
        elif candidate != expected_run_id:
            raise ValueError("evaluation summary run_id mismatch")
        candidate_summary = payload.get("summary", payload)
        if not isinstance(candidate_summary, dict):
            raise ValueError(f"invalid evaluation summary: {summary_path}")
        if expected_summary is None or summary_name == "summary.json":
            expected_summary = candidate_summary
    manifest = validate_result_manifest(
        eval_root,
        expected_run_id=expected_run_id,
        expected_summary=expected_summary,
    )
    allowed = set(manifest["files"])

    def parse_manifested(results_file: Path) -> dict[str, Any]:
        metrics = _parse_results_txt(results_file)
        score_file = results_file.with_name(SCORE_FILENAME)
        relative_score = score_file.relative_to(eval_root).as_posix()
        if relative_score in allowed:
            loaded = load_normalized_score(results_file.parent)
            if loaded is None:
                raise ValueError(
                    f"manifested URGENT score sidecar is missing or not bound to RESULTS: {score_file}"
                )
            payload, loaded_path = loaded
            if loaded_path != score_file.resolve():
                raise ValueError(
                    f"manifested URGENT score sidecar is not a RESULTS sibling: {score_file}"
                )
            metrics["_normalized_score"] = payload
        return metrics

    for item in sorted(eval_root.iterdir()):
        if item.is_dir():
            results_file = item / "RESULTS.txt"
            relative_results = results_file.relative_to(eval_root).as_posix()
            if relative_results in allowed:
                results[item.name] = parse_manifested(results_file)

            sim_results = {}
            for sub in sorted(item.iterdir()):
                if sub.is_dir():
                    rf = sub / "RESULTS.txt"
                    relative = rf.relative_to(eval_root).as_posix()
                    if relative in allowed:
                        sim_results[sub.name] = parse_manifested(rf)
            if sim_results:
                results[item.name] = sim_results

    return results


def _parse_results_txt(path: Path) -> dict[str, Any]:
    """原样解析 RESULTS.txt，不从舍入值重算 normalized score。"""
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    metrics[key.strip()] = val.strip()
    return metrics


def _validate_checkpoint_specific_eval_artifact(
    eval_root: Path, exp_data: Mapping[str, Any]
) -> dict[str, Any]:
    """只接受同一 config/model-only pair 的已提交 checkpoint 专属评估。"""
    expected_root = _eval_root_for(exp_data).resolve()
    if eval_root.is_symlink() or eval_root.resolve() != expected_root:
        raise ValueError("checkpoint 专属评估目录与当前 identity 不一致")
    try:
        checkpoint_path = _validate_consumed_checkpoint_binding(exp_data)
        config_path = Path(str(exp_data["config_path"])).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint 专属评估消费 binding 无效") from exc
    try:
        envelope = load_committed_artifact(eval_root)
    except RuntimeError as exc:
        raise ValueError(
            f"checkpoint 专属评估 artifact 未提交或无效: {eval_root}"
        ) from exc
    if envelope.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError("checkpoint 专属评估 artifact schema 不匹配")
    if envelope.get("task") != exp_data.get("task_type"):
        raise ValueError("checkpoint 专属评估 task 与当前实验不一致")
    if envelope.get("status") not in {"done", "partial"}:
        raise ValueError("checkpoint 专属评估状态不可复用")
    provenance = envelope.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint 专属评估缺少 provenance")
    expected = {
        "config": (str(config_path), fingerprint_path(config_path)),
        "checkpoint": (str(checkpoint_path), fingerprint_path(checkpoint_path)),
    }
    for key, (path, fingerprint) in expected.items():
        if provenance.get(key) != path or provenance.get(f"{key}_fingerprint") != fingerprint:
            raise ValueError(f"checkpoint 专属评估 {key} fingerprint 与当前资产不一致")
    if provenance.get("resolved_config_fingerprint") != exp_data.get(
        "resolved_config_fingerprint"
    ):
        raise ValueError("checkpoint 专属评估 effective config fingerprint 与当前资产不一致")
    parameters = provenance.get("parameters")
    if not isinstance(parameters, Mapping) or parameters.get("output_name") != _artifact_identity(exp_data):
        raise ValueError("checkpoint 专属评估 output_name 与当前 identity 不一致")
    return envelope


def _find_eval_root(exp_data: dict) -> Optional[Path]:
    """找到评估输出目录；发现阶段不读取或信任其中的指标值。"""
    if _uses_checkpoint_specific_identity(exp_data):
        eval_root = _eval_root_for(exp_data)
        return eval_root if eval_root.exists() else None

    exp_name = exp_data["exp_name"]
    task_type = exp_data["task_type"]
    eval_base = default_eval_output_root(_PROJECT_ROOT) / task_type
    eval_root = eval_base / exp_name
    if eval_root.exists() and any(eval_root.rglob("RESULTS.txt")):
        return eval_root

    checkpoint_value = exp_data.get("source_checkpoint_path") or exp_data.get(
        "checkpoint_path", ""
    )
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _PROJECT_ROOT / checkpoint_path
    if checkpoint_path.parent.parent.name == "checkpoints":
        legacy_root = eval_base / checkpoint_path.parent.name
        if legacy_root != eval_root and legacy_root.exists():
            legacy_owned = True
            summary_path = legacy_root / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    summary = {}
                reported_checkpoint = summary.get("checkpoint")
                if reported_checkpoint:
                    reported_path = Path(reported_checkpoint)
                    if not reported_path.is_absolute():
                        reported_path = _PROJECT_ROOT / reported_path
                    try:
                        legacy_owned = reported_path.resolve() == checkpoint_path.resolve()
                    except OSError:
                        legacy_owned = reported_path == checkpoint_path
            if legacy_owned and any(legacy_root.rglob("RESULTS.txt")):
                return legacy_root

    return eval_root if eval_root.exists() else None


# ---------------------------------------------------------------------------
# 算法未来依赖验证
# ---------------------------------------------------------------------------
def _build_latency_verify_cmd(exp_data: dict, device: str, json_path: Path) -> list[str]:
    """构建延迟验证命令，固定输出机器可读 JSON。"""
    return [
        sys.executable, "-m", "evaluate.latency_verification",
        "--task", exp_data["task_type"],
        "--model", exp_data["checkpoint_path"],
        "--config", exp_data["config_path"],
        "--device", device,
        "--seed", "0",
        "--json-output", str(json_path),
    ]


def run_latency_verification(exp_data: dict, device: str = "cuda:0") -> dict:
    """运行未来依赖验证并校验本次 owned JSON 契约；失败仅记录。"""
    eval_root = _eval_root_for(exp_data)
    eval_root.mkdir(parents=True, exist_ok=True)
    json_path = eval_root / "latency_verify.json"
    log_path = eval_root / "latency_verify.log"
    fd, owned_name = tempfile.mkstemp(prefix=".latency_verify_", suffix=".json", dir=eval_root)
    os.close(fd)
    owned_json = Path(owned_name)
    owned_json.unlink()
    cmd = _build_latency_verify_cmd(exp_data, device, owned_json)
    result = {"status": "ERROR", "command": cmd, "json_path": str(json_path), "log_path": str(log_path)}
    try:
        completed = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=900)
        stdout, stderr = completed.stdout or "", completed.stderr or ""
        log_path.write_text(stdout + ("\n--- stderr ---\n" + stderr if stderr else ""), encoding="utf-8")
        if completed.returncode != 0:
            result["error"] = f"exit code: {completed.returncode}"
            return result
        payload = json.loads(owned_json.read_text(encoding="utf-8"))
        required = ("min_future_dependency_ms", "mean_future_dependency_ms", "max_future_dependency_ms")
        valid_number = lambda value: (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        )
        if (
            not isinstance(payload, dict)
            or payload.get("metric_kind") != "future_dependency"
            or payload.get("unit") != "ms"
            or any(not valid_number(payload.get(key)) for key in required)
            or payload.get("task") != exp_data["task_type"]
            or payload.get("model") != exp_data["checkpoint_path"]
            or payload.get("config") != exp_data["config_path"]
        ):
            result["error"] = "invalid latency JSON schema or command identity"
            return result
        minimum, mean, maximum = (float(payload[key]) for key in required)
        if not minimum <= mean <= maximum:
            result["error"] = "invalid latency JSON ordering"
            return result
        os.replace(owned_json, json_path)
        result.update(payload)
        result["status"] = "PASS"
    except Exception as e:
        result["error"] = str(e)
        if not log_path.exists():
            log_path.write_text("", encoding="utf-8")
    finally:
        owned_json.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------------------
# ONNX 三路精度验证（离线 PT / 流式 PT / 流式 ORT）
# ---------------------------------------------------------------------------
# model.type 子串（归一化后）→ verify 脚本；匹配时长 key 优先
ONNX_VERIFY_SCRIPTS = {
    "deepvqe": "onnx/verify/verify_3way_deepvqe.py",
    "dbgnet": "onnx/verify/verify_3way_dbgnet.py",
    "cagcrn": "onnx/verify/verify_3way_cagcrn.py",
    "fastenhancer": "onnx/verify/verify_3way_fastenhancer.py",
    "ultradpc": "onnx/verify/verify_3way_ultra_dpc.py",
}


def _match_onnx_verify_script(model_type: str) -> Optional[tuple[str, Path]]:
    """按 model.type 匹配对应的三路验证脚本（长 key 优先）。"""
    norm = model_type.lower().replace("_", "").replace("-", "")
    for key in sorted(ONNX_VERIFY_SCRIPTS, key=len, reverse=True):
        if key in norm:
            return key, _PROJECT_ROOT / ONNX_VERIFY_SCRIPTS[key]
    return None


def _safe_onnx_target(exp_data: dict, key: str) -> Path:
    """生成受限于 onnx/onnx_model 的稳定导出目标。

    显式 checkpoint 的 identity 本身已带内容 digest（``<实验>@<checkpoint>-<digest>``），
    文件名直接复用该 identity；``@`` 是 identity 合同内的合法字符（评估目录同名），
    保留它即可保证不同 identity 不撞名，无需追加第二层哈希。
    """
    def sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._@-]+", "_", Path(value).name).strip("._") or "model"

    family = infer_model_family(exp_data.get("resolved_config", {})) or key
    try:
        identity = _artifact_identity(exp_data)
    except ValueError:
        # 兼容仅构建 ONNX 命令的历史调用方；完整报告流程始终提供 identity。
        identity = Path(str(exp_data.get("checkpoint_path", "model"))).stem
    name = sanitize(identity)
    return _PROJECT_ROOT / "onnx" / "onnx_model" / f"{name}_{sanitize(family)}_streaming.onnx"


def _build_onnx_verify_cmd(
    key: str, script: Path, exp_data: dict, device: str, save_onnx_path: Optional[Path] = None,
) -> list[str]:
    """根据架构类型构建验证命令，参数从 resolved_config 提取。

    PyTorch 参考统一固定 ``--device cpu``（``device`` 形参保留调用方兼容）：
    三路阈值是 CPU 口径，GPU 上离线（整段）与流式（逐帧）会选到不同 cuDNN
    算法并走 TF32，设备差必然超阈 FAIL（DBGNet 实测 max_abs ~6.5e-2）；ORT
    侧本来就是 CPUExecutionProvider，ONNX 部署也在 CPU 上。
    """
    cfg = exp_data.get("resolved_config", {})
    model_args = cfg.get("model", {}).get("args", {}) or {}
    stft = cfg.get("stft", {}) or {}
    verify_device = "cpu"
    cmd = [sys.executable, str(script), "--checkpoint", exp_data["checkpoint_path"]]

    if key == "deepvqe":
        variant = model_args.get("variant") or "base"
        cmd += ["--variant", str(variant)]
        if stft.get("n_fft"):
            cmd += ["--n-fft", str(stft["n_fft"])]
        if stft.get("hop_length"):
            cmd += ["--hop-length", str(stft["hop_length"])]
        align_delay = model_args.get("align_delay")
        if align_delay is not None:
            cmd += ["--align-delay", str(align_delay)]
        # 发布 ONNX 必须走 widened 口径：2 次 reset 将序列切为 3 段；每段至少
        # align_delay + 1 帧，避免 A/C 跳过对齐延迟后没有有效比较而出现 vacuous PASS。
        # 配置未指定时保留 verifier 的 --align-delay 默认值 25。
        effective_align_delay = int(align_delay) if align_delay is not None else 25
        cmd += ["--resets", "2"]
        cmd += ["--frames", str(max(1000, 3 * (effective_align_delay + 1)))]
        if stft.get("center", True):  # 训练默认 center=True
            cmd += ["--center"]
        # 用户决策：deepvqe float32 档阈值（MSE 1e-9 / max_abs 5e-4，JAE-M10
        # 预注册）是 CPU 口径；GPU 上 PyTorch 与 CPU ORT 的设备差（TF32 等，
        # 实测 ~1.8e-7）必然超阈 FAIL。
        cmd += ["--device", verify_device]
    elif key == "dbgnet":
        # DBGNet 发布验证固定 widened 口径：1200 帧、2 次同步 cold reset。
        cmd += [
            "-C", exp_data["config_path"], "--frames", "1200", "--resets", "2",
            "--device", verify_device,
        ]
    elif key == "fastenhancer":
        variant = model_args.get("variant") or "B"
        cmd += ["--variant", str(variant).upper()]
        if stft.get("n_fft"):
            cmd += ["--n-fft", str(stft["n_fft"])]
        if stft.get("hop_length"):
            cmd += ["--hop-size", str(stft["hop_length"])]
        if stft.get("win_length"):
            cmd += ["--win-size", str(stft["win_length"])]
        cmd += ["--device", verify_device]
    elif key == "ultradpc":
        if stft.get("n_fft"):
            cmd += ["--n-fft", str(stft["n_fft"])]
        if stft.get("hop_length"):
            cmd += ["--hop-size", str(stft["hop_length"])]
        if stft.get("win_length"):
            cmd += ["--win-size", str(stft["win_length"])]
        cmd += ["--device", verify_device]
    else:  # cagcrn: 仅 checkpoint + device
        cmd += ["--device", verify_device]
    cmd += ["--save-onnx", str(save_onnx_path or _safe_onnx_target(exp_data, key))]
    return cmd


def _parse_onnx_verify_output(stdout: str) -> tuple[str, list[dict]]:
    """解析 verify 脚本 stdout，返回 (总状态, 各检查项明细)。"""
    all_pass = "总结: ALL PASS" in stdout
    checks = []
    pending_label = None
    for line in stdout.splitlines():
        s = line.strip()
        if re.match(r"^\([A-Z]\)\s*\S", s):
            pending_label = s
            continue
        m = re.search(r"\[(PASS|FAIL)\]$", s)
        if m:
            checks.append({
                "label": pending_label or f"check {len(checks) + 1}",
                "metric": s,
                "status": m.group(1),
            })
            pending_label = None
    return ("PASS" if all_pass else "FAIL"), checks


def _parse_dbgnet_release_gates(stdout: str) -> dict:
    """结构化解析 DBGNet state 与外部 iSTFT 独立发布门禁。"""
    state_enabled = None
    state_metrics = []
    state_gate = None
    external_istft = None
    comparison = None

    for line in stdout.splitlines():
        text = line.strip()
        label_match = re.match(r"^(\([A-Z]\)\s*[^:]+):", text)
        if label_match:
            comparison = label_match.group(1)

        enabled_match = re.search(r"\buse_state_detector=(True|False)\b", text)
        if enabled_match:
            state_enabled = enabled_match.group(1) == "True"

        metric_match = re.search(
            r"\bState MSE=(\S+)\s+max_abs=(\S+)", text
        )
        if metric_match:
            try:
                mse = float(metric_match.group(1))
                max_abs = float(metric_match.group(2))
            except ValueError:
                mse = max_abs = None
            state_metrics.append({
                "comparison": comparison or "unknown",
                "mse": mse,
                "max_abs": max_abs,
            })

        state_gate_match = re.search(
            r"\bState gate:\s*(PASS|FAIL)(?:\s*\((.*)\))?", text
        )
        if state_gate_match:
            state_gate = {
                "status": state_gate_match.group(1),
                "detail": state_gate_match.group(2) or "",
            }

        istft_match = re.search(
            r"\bExternal iSTFT synthesis capability:\s*(PASS|FAIL)"
            r"(?:\s*\((.*)\))?",
            text,
        )
        if istft_match:
            external_istft = {
                "status": istft_match.group(1),
                "detail": istft_match.group(2) or "",
            }

    missing = []
    if state_enabled is None:
        missing.append("use_state_detector")
    elif state_enabled:
        by_comparison = {
            metric["comparison"][:3]: metric for metric in state_metrics
        }
        for required in ("(A)", "(B)"):
            metric = by_comparison.get(required)
            if metric is None or not all(
                isinstance(metric.get(key), float) and math.isfinite(metric[key])
                for key in ("mse", "max_abs")
            ):
                missing.append(f"State MSE/max_abs {required}")
        if state_gate is None:
            missing.append("State gate")
    if external_istft is None:
        missing.append("External iSTFT synthesis capability")

    explicit_failure = (
        state_gate is not None and state_gate["status"] != "PASS"
    ) or (
        external_istft is not None and external_istft["status"] != "PASS"
    )
    audit_status = (
        "UNAUDITABLE" if missing else "FAIL" if explicit_failure else "PASS"
    )
    return {
        "audit_status": audit_status,
        "state_detector_enabled": state_enabled,
        "state_metrics": state_metrics,
        "state_gate": state_gate,
        "external_istft_synthesis_capability": external_istft,
        "missing": missing,
    }


def _artifact_provenance(path: Union[str, Path]) -> dict:
    """计算产物谱系：SHA256 + size + mtime（JAE-M10）。

    用于绑定"被验证的到底是哪个 checkpoint/ONNX"，失败时仅记录错误信息。
    """
    p = Path(path)
    prov = {"path": str(p)}
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        prov["sha256"] = h.hexdigest()
        prov["size_bytes"] = p.stat().st_size
        prov["mtime"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
    except Exception as e:
        prov["error"] = str(e)
    return prov


def _parse_onnx_artifact_from_stdout(stdout: str) -> tuple[Optional[str], bool, Optional[str]]:
    """从 verify 脚本 stdout 解析被验证 ONNX 的路径 / 保留状态 / SHA256（JAE-M10）。

    优先级：脚本明确保留的 artifact（"ONNX artifact kept: <path> ... sha256=<digest>"）
    > 复用的已有 ONNX（"Using existing ONNX: <path>"）
    > 导出的临时 ONNX（"Exporting ONNX to <path>"，验完即删的旧行为）。
    """
    m = re.search(r"ONNX artifact kept:\s*(\S+?)\s*\(([^)]*)\)", stdout)
    if m:
        digest_m = re.search(r"sha256=([0-9a-f]+)", m.group(2))
        return m.group(1), True, (digest_m.group(1) if digest_m else None)
    m = re.search(r"Using existing ONNX:\s*(\S+)", stdout)
    if m:
        return m.group(1), True, None
    m = re.search(r"Exporting ONNX to\s*(\S+)", stdout)
    if m:
        return m.group(1), False, None
    return None, False, None


def run_onnx_verification(exp_data: dict, device: str = "cuda:0") -> Optional[dict]:
    """运行 ONNX 三路精度验证（离线 PT / 流式 PT / 流式 ORT）。

    完整输出写入 eval 目录下的 onnx_verify.log。验证 FAIL 不中断报告流程，
    仅记录到结果中。返回 JSON 可序列化的结果 dict；不支持的架构返回 None。

    结果中绑定产物谱系与验证口径（JAE-M10）：checkpoint 的 SHA256/size/mtime、
    被验证 ONNX 的路径与保留状态、frames/reset/尾块覆盖情况。DeepVQE 与 DBGNet
    按实际命令标记 widened 或 narrow synthetic；所有口径均不代表 deployment PASS。

    注：三路阈值是 CPU 口径，``_build_onnx_verify_cmd`` 对所有架构固定
    ``--device cpu``（GPU 上 TF32 设备差会误 FAIL），忽略本函数的 device 参数；
    该参数仍用于指标、延迟与 badcase。
    """
    model_type = _runtime_model_config(
        exp_data.get("resolved_config", {})
    ).get("type", "")
    matched = _match_onnx_verify_script(model_type)
    if matched is None:
        logger.info(f"  ONNX 三路验证: 架构 {model_type or 'unknown'} 无对应 verify 脚本，跳过")
        return None
    key, script = matched
    expected_onnx_path = _safe_onnx_target(exp_data, key)
    cmd = _build_onnx_verify_cmd(key, script, exp_data, device, expected_onnx_path)
    logger.info(f"  ONNX 三路验证 ({key}): {' '.join(cmd)}")

    # 从实际命令提取验证口径；reset 将谱图帧序列切为三段并同步清空 cache，
    # 不代表音频 waveform 分块、尾块或流式 STFT/iSTFT 已覆盖。
    frames = cmd[cmd.index("--frames") + 1] if "--frames" in cmd else None
    resets = cmd[cmd.index("--resets") + 1] if "--resets" in cmd else None
    widened_family = key in {"deepvqe", "dbgnet"}
    widened = widened_family and frames is not None and resets is not None and (
        int(frames) >= 1000 and int(resets) >= 2
    )
    scope = {
        "frames": frames,
        "reset": resets if widened_family else "未覆盖",
        "tail_block": (
            "整段 STFT 预计算谱图，逐帧送模；未覆盖 waveform 尾块/任意分块"
            if widened_family else "未覆盖"
        ),
        "inputs": "随机白噪声 mic/far waveform，经配置 STFT 预计算谱图",
        "provider": "CPU ORT",
        "onnx_artifact": "临时 float ONNX",
        "validation_scope": "widened" if widened else "narrow" if widened_family else None,
    }
    checkpoint_artifact = _artifact_provenance(exp_data["checkpoint_path"])

    try:
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=900)
    except Exception as e:
        logger.warning(f"  ONNX 三路验证执行失败: {e}")
        return {"family": key, "script": ONNX_VERIFY_SCRIPTS[key], "status": "ERROR",
                "error": str(e), "checks": [], "log_path": None,
                "checkpoint_artifact": checkpoint_artifact,
                "onnx_artifact": {"expected_path": str(expected_onnx_path), "published": False},
                "scope": scope}

    stdout = result.stdout or ""
    status, checks = _parse_onnx_verify_output(stdout)
    release_gates = _parse_dbgnet_release_gates(stdout) if key == "dbgnet" else None
    if result.returncode != 0 and status == "PASS":
        status = "FAIL"  # 退出码非零优先

    # stdout 的 ALL PASS 只是汇总提示；DBGNet 的 state/iSTFT 独立门禁，以及
    # artifact 路径、落盘文件和完整 SHA256，必须分别可审计且全部通过。
    onnx_path, onnx_kept, onnx_sha256 = _parse_onnx_artifact_from_stdout(stdout)
    reported_candidate = Path(onnx_path) if onnx_path else None
    reported_path = None
    if reported_candidate is not None:
        reported_path = (
            reported_candidate
            if reported_candidate.is_absolute()
            else _PROJECT_ROOT / reported_candidate
        ).resolve()
    actual_artifact = _artifact_provenance(expected_onnx_path)
    binding_errors = []
    if status == "PASS":
        if release_gates is not None and release_gates["audit_status"] != "PASS":
            if release_gates["missing"]:
                binding_errors.append(
                    "DBGNet 独立发布门禁不可审计，stdout 缺少: "
                    + ", ".join(release_gates["missing"])
                )
            else:
                binding_errors.append("DBGNet state/iSTFT 独立发布门禁未全部通过")
        if onnx_path is None:
            binding_errors.append("verifier stdout 未报告 ONNX artifact 路径")
        elif reported_path != expected_onnx_path.resolve():
            binding_errors.append(
                f"verifier 报告的 ONNX artifact 路径不匹配: {onnx_path}"
            )
        if "sha256" not in actual_artifact:
            detail = actual_artifact.get("error", "文件不存在或不可读")
            binding_errors.append(f"预期 ONNX artifact 不存在或不可读: {detail}")
        if onnx_sha256 is None:
            binding_errors.append("verifier stdout 未报告 ONNX artifact SHA256")
        elif re.fullmatch(r"[0-9a-f]{64}", onnx_sha256) is None:
            binding_errors.append("verifier stdout 的 ONNX artifact SHA256 格式非法")
        elif (
            "sha256" in actual_artifact
            and onnx_sha256 != actual_artifact["sha256"]
        ):
            binding_errors.append("verifier stdout 的 ONNX artifact SHA256 与落盘文件不一致")
        if binding_errors:
            status = "FAIL"

    published = status == "PASS"
    onnx_artifact = {
        "expected_path": str(expected_onnx_path),
        "reported_path": onnx_path,
        "published": published,
    }
    if published:
        onnx_artifact.update(
            sha256=actual_artifact["sha256"], size_bytes=actual_artifact["size_bytes"]
        )

    # 保存完整日志到 eval 目录
    eval_root = _eval_root_for(exp_data)
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = eval_root / "onnx_verify.log"
    log_path.write_text(stdout + ("\n--- stderr ---\n" + result.stderr if result.stderr else ""))

    verify = {
        "family": key,
        "script": ONNX_VERIFY_SCRIPTS[key],
        "status": status,
        "checks": checks,
        "log_path": str(log_path),
        "checkpoint_artifact": checkpoint_artifact,
        "onnx_temp_path": onnx_path,
        "onnx_artifact_kept": onnx_kept,
        "onnx_artifact_sha256": onnx_sha256,
        "onnx_artifact": onnx_artifact,
        "scope": scope,
    }
    if release_gates is not None:
        verify["release_gates"] = release_gates
    if binding_errors:
        verify["error"] = "; ".join(binding_errors)
    log_fn = logger.info if status == "PASS" else logger.warning
    log_fn(f"  ONNX 三路验证: {status} (日志: {log_path})")
    return verify


# ---------------------------------------------------------------------------
# 评估运行
# ---------------------------------------------------------------------------
def _resolve_eval_source_arg(
    value: Optional[Union[str, Path]], default: str
) -> Optional[str]:
    """固化显式源路径；未显式指定时按环境或仓库默认值处理。"""
    if value is None:
        if os.environ.get("USE_TRAIN_TEST_DATA_ROOT", "").strip():
            return None
        return str((_PROJECT_ROOT / default).resolve())

    source = Path(value).expanduser()
    return str(source.resolve())


def _reject_unsupported_eval_options(
    args: argparse.Namespace, task_type: str
) -> bool:
    """在 parent 拒绝 child 尚不支持的任务专属参数。"""
    if task_type == "aec":
        return False
    unsupported = [
        option
        for option, enabled in (
            ("--aec-no-stream-align", getattr(args, "aec_no_stream_align", False)),
            ("--skip-existing-inference", getattr(args, "skip_existing_inference", False)),
        )
        if enabled
    ]
    if not unsupported:
        return False
    logger.error(
        "参数 %s 仅支持 AEC 任务，当前任务类型为 %s；未启动评估子进程",
        ", ".join(unsupported),
        task_type.upper(),
    )
    return True


def _build_eval_pipeline_cmd(
    exp_dir: str,
    task_type: str,
    datasets: str,
    audio_quality: bool = True,
    device: str = "cuda:0",
    inference_devices: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    aec_no_stream_align: bool = False,
    skip_existing_inference: bool = False,
    aec_testset_root: Optional[Union[str, Path]] = None,
    urgent_noisy_scp: Optional[Union[str, Path]] = None,
    urgent_clean_scp: Optional[Union[str, Path]] = None,
    voicebank_clean_scp: Optional[Union[str, Path]] = None,
    voicebank_noisy_scp: Optional[Union[str, Path]] = None,
    dns_challenge_noisy_dir: Optional[Union[str, Path]] = None,
    output_name: Optional[str] = None,
    expected_effective_config_sha256: Optional[str] = None,
) -> list[str]:
    """构造评估流水线命令，不启动子进程。"""
    if task_type != "aec" and (aec_no_stream_align or skip_existing_inference):
        raise ValueError("AEC-only 评估参数不能传给非 AEC pipeline")
    if output_name is not None and (
        not isinstance(output_name, str)
        or _SAFE_ARTIFACT_NAME_RE.fullmatch(output_name) is None
    ):
        raise ValueError(f"评估输出名称必须是安全 basename: {output_name!r}")

    script = (
        "evaluate/eval_pipeline_aec.py"
        if task_type == "aec"
        else "evaluate/eval_pipeline_se.py"
    )
    cmd = [
        sys.executable, str(_PROJECT_ROOT / script),
        "--exp_dir", exp_dir,
        "--datasets", datasets,
        "--step", "all",
        "--device", device,
    ]
    if inference_devices is not None:
        cmd += ["--inference-devices", ",".join(normalize_inference_devices(inference_devices, fallback=device))]
    if expected_effective_config_sha256 is not None:
        cmd += [
            "--expected-effective-config-sha256",
            expected_effective_config_sha256,
        ]
    if output_name is not None:
        cmd += ["--output-name", output_name]
    source_args = [
        (
            "--voicebank_clean_scp",
            _resolve_eval_source_arg(voicebank_clean_scp, DEFAULT_VOICEBANK_CLEAN_SCP),
        ),
        (
            "--voicebank_noisy_scp",
            _resolve_eval_source_arg(voicebank_noisy_scp, DEFAULT_VOICEBANK_NOISY_SCP),
        ),
        (
            "--dns_challenge_noisy_dir",
            _resolve_eval_source_arg(
                dns_challenge_noisy_dir, DEFAULT_DNS_CHALLENGE_NOISY_DIR
            ),
        ),
    ]
    if task_type == "aec":
        source_args.extend(
            [
                (
                    "--aec_testset_root",
                    _resolve_eval_source_arg(aec_testset_root, DEFAULT_AEC_TESTSET_ROOT),
                ),
                (
                    "--urgent_noisy_scp",
                    _resolve_eval_source_arg(urgent_noisy_scp, DEFAULT_URGENT_NOISY_SCP),
                ),
                (
                    "--urgent_clean_scp",
                    _resolve_eval_source_arg(urgent_clean_scp, DEFAULT_URGENT_CLEAN_SCP),
                ),
            ]
        )
    for option, source in source_args:
        if source is not None:
            cmd.extend([option, source])
    if audio_quality:
        cmd.append("--audio_quality")
    if aec_no_stream_align:
        cmd.append("--aec_no_stream_align")
    if skip_existing_inference:
        cmd.append("--skip-existing-inference")
    return cmd


def run_eval_pipeline(
    exp_dir: str, task_type: str, datasets: str,
    audio_quality: bool = True, device: str = "cuda:0",
    inference_devices: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    aec_no_stream_align: bool = False,
    skip_existing_inference: bool = False,
    aec_testset_root: Optional[Union[str, Path]] = None,
    urgent_noisy_scp: Optional[Union[str, Path]] = None,
    urgent_clean_scp: Optional[Union[str, Path]] = None,
    voicebank_clean_scp: Optional[Union[str, Path]] = None,
    voicebank_noisy_scp: Optional[Union[str, Path]] = None,
    dns_challenge_noisy_dir: Optional[Union[str, Path]] = None,
    output_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    expected_effective_config_identity: Optional[Mapping[str, str]] = None,
) -> int:
    """运行评估流水线（subprocess），并成对绑定父进程选定的模型资产。"""
    if bool(config_path) != bool(checkpoint_path):
        raise ValueError("config_path 与 checkpoint_path 必须成对提供")
    if output_name is not None and (config_path is None or checkpoint_path is None):
        raise ValueError("显式评估输出必须提供已验证的 config/checkpoint pair")
    config = None if config_path is None else _validated_child_asset(
        config_path, label="config"
    )
    checkpoint = None if checkpoint_path is None else _validated_child_asset(
        checkpoint_path, label="checkpoint"
    )
    if config is not None and expected_effective_config_identity is None:
        expected_effective_config_identity = effective_config_fingerprint(config)
    expected_effective_config_sha256 = (
        None
        if expected_effective_config_identity is None
        else expected_effective_config_identity.get("sha256")
    )
    if expected_effective_config_identity is not None and not isinstance(
        expected_effective_config_sha256, str
    ):
        raise ValueError("expected effective config identity 缺少 sha256")
    cmd = _build_eval_pipeline_cmd(
        exp_dir=exp_dir,
        task_type=task_type,
        datasets=datasets,
        audio_quality=audio_quality,
        device=device,
        inference_devices=inference_devices,
        aec_no_stream_align=aec_no_stream_align,
        skip_existing_inference=skip_existing_inference,
        aec_testset_root=aec_testset_root,
        urgent_noisy_scp=urgent_noisy_scp,
        urgent_clean_scp=urgent_clean_scp,
        voicebank_clean_scp=voicebank_clean_scp,
        voicebank_noisy_scp=voicebank_noisy_scp,
        dns_challenge_noisy_dir=dns_challenge_noisy_dir,
        output_name=output_name,
        expected_effective_config_sha256=expected_effective_config_sha256,
    )
    log_name = output_name if output_name is not None else Path(exp_dir).name
    reservation_adapter = _evaluation_reservation_adapter(
        task_type=task_type,
        output_name=log_name,
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
        inference_devices=inference_devices,
        effective_config_identity=expected_effective_config_identity,
    )
    try:
        reservation = reserve_artifact_run(
            reservation_adapter.output_dir,
            reservation_adapter.ownership_binding(),
            preflight=(
                (lambda: guard_explicit_eval_output(reservation_adapter))
                if output_name is not None
                else None
            ),
        )
    except RuntimeError as exc:
        if output_name is None or "active in-progress run" not in str(exc):
            raise
        raise ValueError(str(exc)) from exc
    temporary_log: Optional[Path] = None
    try:
        logger.info(f"运行评估: {' '.join(cmd)}")
        logger.info("这可能需要较长时间，请耐心等待...")

        child_env = os.environ.copy()
        child_env.pop("USE_TRAIN_EVAL_CONFIG", None)
        child_env.pop("USE_TRAIN_EVAL_CHECKPOINT", None)
        child_env.pop("USE_TRAIN_EXPECTED_EFFECTIVE_CONFIG_SHA256", None)
        child_env.pop(EVALUATION_RESERVATION_TOKEN_ENV, None)
        child_env[EVALUATION_RESERVATION_TOKEN_ENV] = reservation.token
        if config is not None and checkpoint is not None:
            child_env["USE_TRAIN_EVAL_CONFIG"] = str(config)
            child_env["USE_TRAIN_EVAL_CHECKPOINT"] = str(checkpoint)
        if expected_effective_config_sha256 is not None:
            child_env["USE_TRAIN_EXPECTED_EFFECTIVE_CONFIG_SHA256"] = (
                expected_effective_config_sha256
            )

        private_log = reservation.output_dir / "eval_pipeline.log"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{private_log.name}.", suffix=".tmp", dir=private_log.parent
        )
        temporary_log = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                preexec_fn=death_signal_preexec,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    log_file.write(line)
                    print(_sanitize_free_text(line), end="")
            try:
                returncode = process.wait(timeout=EVAL_SUBPROCESS_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # 整组杀：子进程已 start_new_session，killpg 覆盖其所有后代。
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise RuntimeError(
                    f"评估子进程超时({EVAL_SUBPROCESS_TIMEOUT_S}s)被终止: {cmd[0]}"
                )
        os.replace(temporary_log, private_log)
        temporary_log = None
        return returncode
    finally:
        if temporary_log is not None:
            try:
                temporary_log.unlink()
            except FileNotFoundError:
                pass
        release_artifact_reservation(reservation)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def _report_lock_path() -> Path:
    """返回持久化全局索引锁；锁文件不得在解锁后删除。"""
    return INDEX_FILE.parent / REPORT_LOCK_FILE.name


@contextmanager
def _persistent_flock(path: Path) -> Iterator[None]:
    """在持久锁文件上持有进程级独占 ``flock``。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_experiments_index_unlocked() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    with INDEX_FILE.open(encoding="utf-8") as index_file:
        value = json.load(index_file)
    if not isinstance(value, list):
        raise ValueError(f"实验索引必须是 list: {INDEX_FILE}")
    return value


def load_experiments_index() -> list[dict]:
    """在持久全局锁下加载 ignored private runtime index。"""
    with _persistent_flock(_report_lock_path()):
        return _load_experiments_index_unlocked()


def _serialize_experiments_index(exps: list[dict]) -> str:
    return json.dumps(exps, ensure_ascii=False, indent=2, default=str) + "\n"


def _save_experiments_index_unlocked(exps: list[dict]) -> None:
    atomic_write_text(INDEX_FILE, _serialize_experiments_index(exps))


def save_experiments_index(exps: list[dict]) -> None:
    """兼容保存 API：持久 ``flock`` 下原子替换 private runtime index。"""
    with _persistent_flock(_report_lock_path()):
        _save_experiments_index_unlocked(exps)


_CHECKPOINT_INDEX_LINEAGE_KEYS = (
    "config_fingerprint",
    "config_dependency_fingerprints",
    "resolved_config_fingerprint",
    "source_checkpoint_fingerprint",
    "model_only_checkpoint_fingerprint",
    "checkpoint_family",
    "waveform_native",
)


def _matches_checkpoint_index_lineage(
    entry: Mapping[str, Any], exp_data: Mapping[str, Any]
) -> bool:
    """判断索引条目是否仍属于当前显式 checkpoint 消费链路。"""
    return bool(entry.get("explicit_checkpoint")) and not any(
        entry.get(key) != exp_data.get(key)
        for key in _CHECKPOINT_INDEX_LINEAGE_KEYS
    )


def _index_entry_for_identity(
    exps: list[dict], exp_data: Mapping[str, Any]
) -> Optional[dict]:
    """按 identity 查询索引；显式 checkpoint 只复用可重算的 v3 条目。"""
    identity = _artifact_identity(exp_data)
    if _uses_checkpoint_specific_identity(exp_data) and not _has_valid_v3_identity_record(
        exp_data
    ):
        return None
    for entry in exps:
        if _artifact_identity(entry) != identity:
            continue
        if not _uses_checkpoint_specific_identity(exp_data):
            return entry
        if (
            _has_valid_v3_identity_record(entry)
            and _matches_checkpoint_index_lineage(entry, exp_data)
        ):
            return entry
    return None


def _merge_experiment_index(
    exps: list[dict], exp_data: dict, eval_results: dict
) -> list[dict]:
    """基于传入快照合并一个实验条目，不执行 I/O。"""
    exp_name = exp_data["exp_name"]
    identity = _artifact_identity(exp_data)
    if _uses_checkpoint_specific_identity(exp_data) and not _has_valid_v3_identity_record(
        exp_data
    ):
        raise ValueError("显式 checkpoint identity 缺少可重算的 v3 record")
    entry = {
        "exp_name": exp_name,
        "checkpoint_identity": identity,
        "checkpoint_identity_schema": exp_data.get("checkpoint_identity_schema"),
        "checkpoint_identity_payload": exp_data.get("checkpoint_identity_payload"),
        "explicit_checkpoint": exp_data.get("explicit_checkpoint", False),
        "config_fingerprint": exp_data.get("config_fingerprint"),
        "config_dependency_fingerprints": exp_data.get(
            "config_dependency_fingerprints"
        ),
        "resolved_config_fingerprint": exp_data.get("resolved_config_fingerprint"),
        "exp_dir": exp_data["exp_dir"],
        "task_type": exp_data["task_type"],
        "model_family": infer_model_family(exp_data.get("resolved_config", {})),
        "model_type": _runtime_model_config(
            exp_data.get("resolved_config", {})
        ).get("type", "unknown"),
        "timestamp": exp_data["timestamp"].isoformat() if hasattr(exp_data["timestamp"], "isoformat") else str(exp_data["timestamp"]),
        "purpose": exp_data.get("purpose", ""),
        "config_path": exp_data.get("config_path", ""),
        "source_checkpoint_path": exp_data.get("source_checkpoint_path", ""),
        "source_checkpoint_fingerprint": exp_data.get("source_checkpoint_fingerprint"),
        "expected_model_only_checkpoint_path": exp_data.get("expected_model_only_checkpoint_path", ""),
        "model_only_checkpoint_path": exp_data.get("model_only_checkpoint_path"),
        "model_only_checkpoint_fingerprint": exp_data.get("model_only_checkpoint_fingerprint"),
        "checkpoint_path": exp_data.get("checkpoint_path"),
        "checkpoint_family": exp_data.get("checkpoint_family"),
        "waveform_native": exp_data.get("waveform_native", False),
        "metric_device": exp_data.get("metric_device"),
        "inference_devices": exp_data.get("inference_devices"),
        "training_metrics": exp_data.get("training_metrics", {}),
        "eval_results": _simplify_eval_results(eval_results),
        "onnx_verify": exp_data.get("onnx_verify"),
        "latency_verify": exp_data.get("latency_verify"),
        "cpu_rtf": exp_data.get("cpu_rtf"),
        "badcase": exp_data.get("badcase"),
        "added_at": datetime.now().isoformat(),
    }

    if _uses_checkpoint_specific_identity(entry):
        conflicting = [
            existing
            for existing in exps
            if _artifact_identity(existing) == identity
            and not _matches_checkpoint_index_lineage(existing, entry)
        ]
        if conflicting:
            raise ValueError("checkpoint identity 与既有索引来源谱系冲突")
    old_entry = _index_entry_for_identity(exps, entry) or {}
    checkpoint_lineage_keys = {
        "checkpoint_path",
        "source_checkpoint_path",
        "source_checkpoint_fingerprint",
        "expected_model_only_checkpoint_path",
        "model_only_checkpoint_path",
        "model_only_checkpoint_fingerprint",
        "checkpoint_family",
        "waveform_native",
    }
    for key, value in old_entry.items():
        if key not in entry or (
            entry[key] is None and key not in checkpoint_lineage_keys
        ):
            entry[key] = value

    existing = [e for e in exps if _artifact_identity(e) != identity]
    existing.append(entry)
    existing.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return existing


def update_experiment_index(exp_data: dict, eval_results: dict) -> None:
    """兼容更新 API：在同一锁周期内执行 read-modify-atomic-write。"""
    with _persistent_flock(_report_lock_path()):
        existing = _merge_experiment_index(
            _load_experiments_index_unlocked(), exp_data, eval_results
        )
        _save_experiments_index_unlocked(existing)


def _simplify_eval_results(results: dict) -> dict:
    """简化评估结果用于 JSON 存储（提取关键指标）。

    除核心指标外保留统计/谱系注解键（JAE-M06）：`_count_*` 样本数与
    `Normalized_Average_Denominator` 分母，供报告表格展示。
    """
    keep_keys = ("PESQ", "STOI", "ESTOI", "SI-SDR", "DNSMOS",
                 "NISQA", "UTMOS", "SCOREQ", "Normalized_Average",
                 "ERLE_dB", "NEAREND_SISDR_dB",
                 "NOISE_REDUCTION_dB", "AECMOS_ECHO", "AECMOS_DEG",
                 # 下游任务指标（--downstream 启用时产生）
                 "WER", "CER", "SpeakerSimilarity", "EmotionSimilarity",
                 "PhonemeSimilarity", "SpeechBERTScore", "LAcc")

    def _keep(k: str) -> bool:
        return k in keep_keys or _is_annotation_key(k)

    simplified = {}
    for ds_name, ds_result in results.items():
        if isinstance(ds_result, dict):
            # 检查是否是场景嵌套结构 (aec_sim)
            if _is_nested_scenario_result(ds_result):
                # 嵌套场景
                simplified[ds_name] = {}
                for scen, metrics in ds_result.items():
                    if isinstance(metrics, dict):
                        simplified[ds_name][scen] = {
                            k: v for k, v in metrics.items() if _keep(k)
                        }
            else:
                simplified[ds_name] = {
                    k: v for k, v in ds_result.items() if _keep(k)
                }
    return simplified


def collect_existing_eval_results(exp_data: dict) -> dict:
    """收集已存在的评估结果（如果 eval 已跑过）。"""
    eval_root = _find_eval_root(exp_data)
    if eval_root is None:
        return {}
    if _uses_checkpoint_specific_identity(exp_data):
        _validate_checkpoint_specific_eval_artifact(eval_root, exp_data)
    return parse_all_results(eval_root)


def _load_eval_device_provenance(exp_data: dict) -> None:
    """从受信评估 envelope 恢复指标设备与模型推理设备。"""
    eval_root = _find_eval_root(exp_data)
    if eval_root is None:
        return
    if _uses_checkpoint_specific_identity(exp_data):
        envelope = _validate_checkpoint_specific_eval_artifact(eval_root, exp_data)
    else:
        envelope_path = eval_root / "evaluation_artifact.json"
        try:
            envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
    provenance = envelope.get("provenance") if isinstance(envelope, dict) else None
    parameters = provenance.get("parameters") if isinstance(provenance, dict) else None
    if not isinstance(parameters, dict):
        return
    metric_device = parameters.get("device")
    inference_devices = parameters.get("inference_devices")
    if isinstance(metric_device, str) and metric_device:
        exp_data["metric_device"] = metric_device
    if isinstance(inference_devices, list) and all(
        isinstance(device, str) and device for device in inference_devices
    ):
        exp_data["inference_devices"] = inference_devices


def _historical_score_coverage(metrics: Mapping[str, Any]) -> Optional[dict[str, dict]]:
    """仅恢复 RESULTS 明示的 computed，绝不从其他指标猜 eligible。"""
    coverage: dict[str, dict] = {}
    for metric in NORMALIZED_METRICS:
        count = metrics.get(f"_count_{metric}")
        if (
            metric in metrics
            and isinstance(count, (int, float))
            and not isinstance(count, bool)
            and math.isfinite(float(count))
            and float(count).is_integer()
            and count >= 0
        ):
            coverage[metric] = {"computed": int(count)}
    return coverage or None


def _annotate_normalized_score_projections(node: Any) -> None:
    if not isinstance(node, dict):
        return
    if "_normalized_score" in node:
        node["_normalized_score"] = validate_normalized_score(
            node["_normalized_score"]
        )
        return
    if "Normalized_Average" in node or any(
        metric in node for metric in NORMALIZED_METRICS
    ):
        node["_normalized_score_projection"] = project_existing_score(
            node, _historical_score_coverage(node)
        )
        return
    for value in node.values():
        _annotate_normalized_score_projections(value)


def _prepare_public_experiments(runtime_exps: list[dict]) -> list[dict]:
    """先用 private locator 完成配置读取，再构造仅供渲染的 public DTO。"""
    enriched = copy.deepcopy(runtime_exps)
    for exp in enriched:
        _annotate_normalized_score_projections(exp.get("eval_results"))
    task_families: dict[tuple[str, str], list[dict]] = {}
    for exp in enriched:
        key = (exp.get("task_type", ""), exp.get("model_family", "Other"))
        task_families.setdefault(key, []).append(exp)

    for family_exps in task_families.values():
        for members in _group_ablation_families(family_exps).values():
            config_diffs = _diff_family_configs(members)
            public_diffs = [
                (
                    path,
                    {
                        name: value if value is _MISSING else _to_public_value(value)
                        for name, value in values.items()
                    },
                )
                for path, values in config_diffs
            ]
            for member in members:
                member["_config_diffs"] = public_diffs

    return [_to_public_record(exp) for exp in enriched]


def _build_report_from_experiments(runtime_exps: list[dict]) -> str:
    """从调用方持有的 private index 快照构建 public Markdown。"""
    if not runtime_exps:
        return _empty_report()
    exps = _prepare_public_experiments(runtime_exps)

    lines = []
    lines.append("# 实验训练报告")
    lines.append("")
    lines.append(f"> 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 总实验数: {len(exps)}")
    lines.append("")

    # 按 task_type 分组
    aec_exps = [e for e in exps if e["task_type"] == "aec"]
    se_exps = [e for e in exps if e["task_type"] == "se"]

    if aec_exps:
        lines.extend(_build_task_section("AEC 实验", aec_exps))
    if se_exps:
        lines.extend(_build_task_section("SE 实验", se_exps))

    # 附录：指标说明
    lines.extend(_build_metric_legend())

    return "\n".join(lines)


def build_report() -> str:
    """从兼容加载 API 读取一个 index 快照并构建报告。"""
    return _build_report_from_experiments(load_experiments_index())


def _build_task_section(title: str, exps: list[dict]) -> list[str]:
    """构建一个任务大类的报告 section。"""
    lines = []
    lines.append(f"## {title}")
    lines.append("")

    # 按模型架构分组
    families: dict[str, list[dict]] = {}
    for e in exps:
        fam = e.get("model_family", "Other")
        families.setdefault(fam, []).append(e)

    for fam_name in sorted(families.keys()):
        fam_exps = _sort_experiments_chronologically(families[fam_name])
        lines.append(f"### {fam_name} 系列 ({len(fam_exps)}个实验)")
        lines.append("")

        # 实验编号图例：表格内一律用 expN 指代，避免长名称截断不可读
        lines.extend(_build_exp_legend(fam_exps))

        # 按数据集/场景拆分的对比表
        lines.extend(_build_comparison_table(fam_exps))
        lines.append("")

        # 家族级人工对比分析备注（analysis_notes.md 按 model_family 分节）
        lines.extend(_build_family_analysis_note(fam_name))
        lines.append("")

        # 同一家族内实验的消融分析（基于配置差异与关键指标）
        lines.extend(_build_ablation_analysis(fam_exps))
        lines.append("")

        # 每个实验的详情
        for exp in fam_exps:
            lines.extend(_build_experiment_detail(exp))
            lines.append("")

    return lines


def _sort_experiments_chronologically(exps: list[dict]) -> list[dict]:
    """按时间戳升序排列实验（最早的在前），便于按时间线观察迭代效果。"""
    return sorted(exps, key=lambda e: e.get("timestamp", "") or "")


def _build_exp_legend(fam_exps: list[dict]) -> list[str]:
    """生成实验编号图例表，并为每个实验赋 ``_exp_id``（exp1, exp2, ...，按时间序）。

    报告各表格行以编号指代实验，长名称只在此表出现一次（可读、可追溯）。
    """
    lines = ["| 编号 | 实验全称 |", "|---|---|"]
    for i, e in enumerate(fam_exps, 1):
        e["_exp_id"] = f"exp{i}"
        lines.append(f"| {e['_exp_id']} | `{_display_experiment_name(e)}` |")
    lines.append("")
    return lines


_FAMILY_NOTES_CACHE: dict[str, Any] = {"mtime_ns": None, "notes": {}}


def _load_family_analysis_notes() -> dict[str, str]:
    """读取 analysis_notes.md，返回 ``{model_family: markdown}``。

    文件按 ``## <model_family>`` 分节；缺失文件视为无备注。按 mtime 缓存，
    同一进程内重复构建报告不重复读盘。
    """
    try:
        stat = ANALYSIS_NOTES_FILE.stat()
    except OSError:
        return {}
    if _FAMILY_NOTES_CACHE["mtime_ns"] == stat.st_mtime_ns:
        return _FAMILY_NOTES_CACHE["notes"]
    notes: dict[str, list[str]] = {}
    current: str | None = None
    try:
        text = ANALYSIS_NOTES_FILE.read_text(encoding="utf-8")
    except OSError:
        return {}
    for raw in text.splitlines():
        if raw.startswith("## "):
            current = raw[3:].strip()
            notes.setdefault(current, [])
        elif current is not None:
            notes[current].append(raw)
    result = {name: "\n".join(lines).strip("\n") for name, lines in notes.items()}
    _FAMILY_NOTES_CACHE["mtime_ns"] = stat.st_mtime_ns
    _FAMILY_NOTES_CACHE["notes"] = result
    return result


def _build_family_analysis_note(fam_name: str) -> list[str]:
    """渲染单模型族的人工对比分析备注（存在才输出）。"""
    content = _load_family_analysis_notes().get(fam_name)
    if not content:
        return []
    lines = ["#### 对比分析", ""]
    lines.extend(content.splitlines())
    lines.append("")
    return lines


def _build_comparison_table(exps: list[dict]) -> list[str]:
    """构建同一架构族内、按数据集/场景拆分的对比表。

    实验按时间升序排列；每个数据集（或 aec_sim 的每个场景）独立成表，
    避免不同 testset 口径混在同一列。
    """
    lines: list[str] = []
    dataset_groups = _collect_dataset_groups(exps)
    task_type = exps[0].get("task_type", "") if exps else ""

    main = _select_main_dataset(dataset_groups, exps)
    used_flat: set[str] = set()
    used_nested: set[str] = set()
    used_prefixed: set[str] = set()

    # 1. 主指标表
    if main:
        kind, name = main
        lines.append("#### 主指标对比")
        if kind == "flat":
            lines.extend(_build_flat_dataset_table(exps, name))
            used_flat.add(name)
        elif kind == "prefixed":
            lines.extend(_build_prefixed_group_tables(
                exps, name, dataset_groups["prefixed_groups"][name], heading_level=5
            ))
            used_prefixed.add(name)
        elif kind == "nested":
            lines.extend(_build_nested_dataset_tables(
                exps, name, dataset_groups["nested"][name], heading_level=5
            ))
            used_nested.add(name)
        lines.append("")

    # AEC 优先展示 AEC Sim / AEC Challenge，SE 数据集放后面；SE 保持 flat -> nested -> prefixed
    if task_type == "aec":
        # 2. 嵌套场景数据集（AEC Sim 参考集）
        for ds_name, scenarios in sorted(dataset_groups.get("nested", {}).items()):
            if ds_name in used_nested:
                continue
            lines.extend(_build_nested_dataset_tables(exps, ds_name, scenarios, heading_level=4))

        # 3. 盲测集分组（AEC Challenge）
        for group_prefix, sub_datasets in sorted(dataset_groups.get("prefixed_groups", {}).items()):
            if group_prefix in used_prefixed:
                continue
            lines.extend(_build_prefixed_group_tables(exps, group_prefix, sub_datasets, heading_level=4))

        # 4. 其他 flat 数据集（通常是 SE 数据集）
        for ds_name in sorted(dataset_groups.get("flat", set())):
            if ds_name in used_flat:
                continue
            display = _dataset_display_name(ds_name)
            lines.append(f"#### {display}")
            lines.extend(_build_flat_dataset_table(exps, ds_name))
            lines.append("")
    else:
        # 2. 其他 flat 数据集
        for ds_name in sorted(dataset_groups.get("flat", set())):
            if ds_name in used_flat:
                continue
            display = _dataset_display_name(ds_name)
            lines.append(f"#### {display}")
            lines.extend(_build_flat_dataset_table(exps, ds_name))
            lines.append("")

        # 3. 嵌套场景数据集
        for ds_name, scenarios in sorted(dataset_groups.get("nested", {}).items()):
            if ds_name in used_nested:
                continue
            lines.extend(_build_nested_dataset_tables(exps, ds_name, scenarios, heading_level=4))

        # 4. 盲测集分组
        for group_prefix, sub_datasets in sorted(dataset_groups.get("prefixed_groups", {}).items()):
            if group_prefix in used_prefixed:
                continue
            lines.extend(_build_prefixed_group_tables(exps, group_prefix, sub_datasets, heading_level=4))

    return lines


def _build_nested_dataset_tables(
    exps: list[dict], ds_name: str, scenarios: set[str], heading_level: int = 4
) -> list[str]:
    """构建嵌套数据集（如 aec_sim）的所有场景对比表。"""
    lines: list[str] = []
    display = _dataset_display_name(ds_name)
    lines.append(f"{'#' * heading_level} {display}")
    sub_level = heading_level + 1
    for scen_name in _scenario_display_order(scenarios):
        lines.append(f"{'#' * sub_level} {scen_name}")
        lines.extend(_build_scenario_table(exps, ds_name, scen_name))
        lines.append("")
    return lines


def _build_prefixed_group_tables(
    exps: list[dict], prefix: str, sub_datasets: set[str], heading_level: int = 4
) -> list[str]:
    """构建前缀分组数据集（如 blind_test_set_icassp2023_*）的所有子集对比表。"""
    lines: list[str] = []
    display = _dataset_display_name(prefix)
    lines.append(f"{'#' * heading_level} {display}")
    sub_level = heading_level + 1
    for sub_ds in _prefixed_display_order(prefix, sub_datasets):
        suffix = sub_ds[len(prefix):].lstrip("_-")
        suffix_display = suffix or sub_ds
        lines.append(f"{'#' * sub_level} {suffix_display}")
        lines.extend(_build_flat_dataset_table(exps, sub_ds))
        lines.append("")
    return lines


def _is_nested_scenario_result(ds_result: dict) -> bool:
    """判断数据集结果是否为场景嵌套结构（如 aec_sim）。"""
    if not ds_result:
        return False
    return all(isinstance(v, dict) for v in ds_result.values())


# ---------------------------------------------------------------------------
# 实验家族消融分析
# ---------------------------------------------------------------------------
_ABLATION_SUFFIX_RE = re.compile(r"_(v\d+|run\d+)$")


def _experiment_base_name(exp_name: str) -> str:
    """去掉 _v2/_v3/_run1 等版本后缀，得到实验家族基名。"""
    return _ABLATION_SUFFIX_RE.sub("", exp_name)


def _group_ablation_families(exps: list[dict]) -> dict[str, list[dict]]:
    """按基名把实验分组，只返回包含 >=2 个成员的家族。"""
    groups: dict[str, list[dict]] = {}
    for e in exps:
        base = _experiment_base_name(e.get("exp_name", ""))
        if not base:
            continue
        groups.setdefault(base, []).append(e)
    return {
        base: _sort_experiments_chronologically(members)
        for base, members in groups.items()
        if len(members) >= 2
    }


def _flatten_dict(obj: Any, parent_key: str = "") -> dict[str, Any]:
    """把嵌套 dict/list 展平为点分路径。"""
    items: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            items.update(_flatten_dict(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten_dict(v, f"{parent_key}[{i}]"))
    else:
        items[parent_key] = obj
    return items


_MISSING = object()


def _diff_family_configs(members: list[dict]) -> list[tuple[str, dict[str, Any]]]:
    """比较同一家族各实验的 resolved_config.yaml，返回有差异的配置项。

    只关注 data.train.simulation_cfg / model.args / stft 等与消融相关的分支，
    过滤 exp_path 等无关路径。
    """
    flat_cfgs: dict[str, dict[str, Any]] = {}
    all_paths: set[str] = set()
    for e in members:
        name = _artifact_identity(e)
        cfg_path = e.get("config_path")
        cfg: dict = {}
        if cfg_path:
            try:
                cfg = load_merged_config(str(cfg_path))
            except Exception as err:
                logger.debug(f"加载 config 失败 {cfg_path}: {err}")
        flat = _flatten_dict(cfg)
        flat_cfgs[name] = flat
        all_paths.update(flat.keys())

    interesting_prefixes = (
        "data.train.simulation_cfg.",
        "model.args.",
        "stft.",
    )
    ignore_suffixes = (
        "exp_path",
        "exp_name",
        "log_dir",
    )

    diffs: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(all_paths):
        values = {
            name: flat.get(path, _MISSING)
            for name, flat in flat_cfgs.items()
        }
        str_values = {str(v) for v in values.values() if v is not _MISSING}
        if len(str_values) <= 1:
            continue
        if not any(path.startswith(p) for p in interesting_prefixes):
            continue
        if any(path.endswith(s) for s in ignore_suffixes):
            continue
        diffs.append((path, values))
    return diffs


def _fmt_diff_value(value: Any) -> str:
    """把配置值格式化为 Markdown 表格单元格内容。"""
    if value is _MISSING:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list) and len(value) > 4:
        return "[" + ", ".join(_fmt_diff_value(v) for v in value[:4]) + ", ...]"
    return json.dumps(value, ensure_ascii=False)


def _extract_main_metrics(exp: dict) -> dict[str, Any]:
    """提取用于消融分析的关键指标。"""
    results = exp.get("eval_results", {})
    metrics: dict[str, Any] = {}

    # AEC Challenge doubletalk DNSMOS
    bt = results.get("blind_test_set_icassp2023_doubletalk") or {}
    if isinstance(bt, dict):
        metrics["doubletalk_DNSMOS"] = bt.get("DNSMOS", "-")

    # aec_sim prob_target_only NormAvg（与主表格第一行口径一致，避免跨场景平均失真）
    aec_sim = results.get("aec_sim") or {}
    if isinstance(aec_sim, dict):
        prob_target_only = aec_sim.get("prob_target_only") or {}
        if isinstance(prob_target_only, dict) and "Normalized_Average" in prob_target_only:
            metrics["aec_sim_clean_NormAvg"] = _format_normalized_score(prob_target_only)

    # prob_echo_only / prob_noise_echo_only ERLE / AECMOS_ECHO
    for scen in ["prob_echo_only", "prob_noise_echo_only"]:
        scen_res = aec_sim.get(scen) or {}
        if isinstance(scen_res, dict):
            metrics[f"{scen}_ERLE"] = scen_res.get("ERLE_dB", "-")
            metrics[f"{scen}_AECMOS_ECHO"] = scen_res.get("AECMOS_ECHO", "-")

    # prob_noise_only NR
    noise = aec_sim.get("prob_noise_only") or {}
    if isinstance(noise, dict):
        metrics["prob_noise_only_NR"] = noise.get("NOISE_REDUCTION_dB", "-")

    return metrics


def _build_ablation_analysis(exps: list[dict]) -> list[str]:
    """为同一模型架构族内的实验家族生成消融分析段落。"""
    families = _group_ablation_families(exps)
    if not families:
        return []

    lines: list[str] = []
    lines.append("#### 消融分析")
    lines.append("")

    for base_name, members in sorted(families.items()):
        # 优先用实验编号（expN）指代，无编号（旧索引）回退 checkpoint-aware 名称。
        identities = [_artifact_identity(member) for member in members]
        labels = [member.get("_exp_id") or _display_experiment_name(member) for member in members]
        lines.append(f"**{base_name}** 系列包含 {len(members)} 个实验：")
        lines.append(", ".join(f"`{n}`" for n in labels))
        lines.append("")

        # 配置读取已在 public DTO 构造前完成，这里不再使用已去敏 locator 做 I/O。
        cfg_diffs = members[0].get("_config_diffs", [])
        if cfg_diffs:
            lines.append("配置差异：")
            lines.append("")
            header = "| 配置项 | " + " | ".join(labels) + " |"
            lines.append(header)
            lines.append("|" + "|".join(["------"] * (len(labels) + 1)) + "|")
            for path, values in cfg_diffs[:15]:  # 最多展示 15 项差异
                row_values = [
                    _fmt_diff_value(values.get(identity, _MISSING))
                    for identity in identities
                ]
                lines.append(f"| `{path}` | " + " | ".join(row_values) + " |")
            lines.append("")

        # 关键指标对比表
        metrics_list = [_extract_main_metrics(e) for e in members]
        lines.append("关键指标对比：")
        lines.append("")
        metric_keys = [
            ("doubletalk_DNSMOS", "doubletalk DNSMOS"),
            ("aec_sim_clean_NormAvg", "aec_sim prob_target_only NormAvg"),
            ("prob_echo_only_ERLE", "prob_echo_only ERLE(dB)"),
            ("prob_noise_echo_only_ERLE", "prob_noise_echo_only ERLE(dB)"),
            ("prob_echo_only_AECMOS_ECHO", "prob_echo_only AECMOS_ECHO"),
            ("prob_noise_only_NR", "prob_noise_only NR(dB)"),
        ]
        lines.append("| 指标 | " + " | ".join(labels) + " |")
        lines.append("|" + "|".join(["------"] * (len(labels) + 1)) + "|")
        for key, display in metric_keys:
            row = [display]
            for m in metrics_list:
                v = m.get(key, "-")
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # 结论与原因分析
        best_idx = -1
        best_val = None
        for i, m in enumerate(metrics_list):
            val = m.get("doubletalk_DNSMOS")
            if isinstance(val, (int, float)):
                if best_val is None or val > best_val:
                    best_val = val
                    best_idx = i

        if best_idx >= 0:
            best_name = labels[best_idx]
            lines.append(
                f"在该家族中，`{best_name}` 的 AEC Challenge doubletalk DNSMOS 最高（{best_val:.4f}）。"
            )
            lines.append("")

        # 辅助：从配置差异中提取某个路径的数值
        def _cfg_value(identity: str, path: str) -> Any:
            for p, values in cfg_diffs:
                if p == path:
                    return values.get(identity, _MISSING)
            return _MISSING

        def _metric_value(idx: int, key: str) -> Any:
            return metrics_list[idx].get(key, _MISSING)

        # 以首个非 _runN 成员为基线；若全是 run，则取首个成员
        baseline_idx = next(
            (i for i, n in enumerate(labels) if not n.endswith(("_run1", "_run2", "_run3"))),
            0,
        )
        baseline_identity = identities[baseline_idx]

        for idx, (name, identity) in enumerate(zip(labels, identities)):
            if idx == baseline_idx:
                continue

            # v2 类：jitter 加量
            jitter_prob = _cfg_value(identity, "data.train.simulation_cfg.aec_time_delay_jitter.prob")
            baseline_jitter_prob = _cfg_value(baseline_identity, "data.train.simulation_cfg.aec_time_delay_jitter.prob")
            jump_lo = _cfg_value(identity, "data.train.simulation_cfg.aec_time_delay_jitter.jump_ms_range[0]")
            baseline_jump_lo = _cfg_value(baseline_identity, "data.train.simulation_cfg.aec_time_delay_jitter.jump_ms_range[0]")

            if (isinstance(jitter_prob, float) and isinstance(baseline_jitter_prob, float) and jitter_prob > baseline_jitter_prob) or \
               (isinstance(jump_lo, (int, float)) and isinstance(baseline_jump_lo, (int, float)) and abs(jump_lo) > abs(baseline_jump_lo)):
                bt_dnsmos = _metric_value(idx, "doubletalk_DNSMOS")
                base_bt = _metric_value(baseline_idx, "doubletalk_DNSMOS")
                if isinstance(bt_dnsmos, (int, float)) and isinstance(base_bt, (int, float)) and bt_dnsmos < base_bt:
                    lines.append(
                        f"`{name}` 加大了 delay jitter 剂量，但 doubletalk DNSMOS 从 {base_bt:.4f} "
                        f"降至 {bt_dnsmos:.4f}。说明当前小模型对更大范围的时变抖动适应能力有限，"
                        f"jitter 增强已超过收益拐点。"
                    )
                    lines.append("")

            # v3 类：音量范围扩展
            src_vol_lo = _cfg_value(identity, "data.train.simulation_cfg.source_volume_db_range[0]")
            baseline_src_vol_lo = _cfg_value(baseline_identity, "data.train.simulation_cfg.source_volume_db_range[0]")
            far_lo = _cfg_value(identity, "data.train.simulation_cfg.far_db_range[0]")
            baseline_far_lo = _cfg_value(baseline_identity, "data.train.simulation_cfg.far_db_range[0]")

            if (isinstance(src_vol_lo, (int, float)) and isinstance(baseline_src_vol_lo, (int, float)) and src_vol_lo < baseline_src_vol_lo) or \
               (isinstance(far_lo, (int, float)) and isinstance(baseline_far_lo, (int, float)) and far_lo < baseline_far_lo):
                echo_erle = _metric_value(idx, "prob_echo_only_ERLE")
                base_echo_erle = _metric_value(baseline_idx, "prob_echo_only_ERLE")
                ne_erle = _metric_value(idx, "prob_noise_echo_only_ERLE")
                base_ne_erle = _metric_value(baseline_idx, "prob_noise_echo_only_ERLE")
                bt_dnsmos = _metric_value(idx, "doubletalk_DNSMOS")
                base_bt = _metric_value(baseline_idx, "doubletalk_DNSMOS")

                explanations = []
                if isinstance(echo_erle, (int, float)) and isinstance(base_echo_erle, (int, float)) and echo_erle > base_echo_erle:
                    explanations.append(
                        f"纯回声场景 ERLE 从 {base_echo_erle:.2f} dB 提升到 {echo_erle:.2f} dB"
                    )
                if isinstance(ne_erle, (int, float)) and isinstance(base_ne_erle, (int, float)) and ne_erle < base_ne_erle:
                    explanations.append(
                        f"但噪声+回声场景 ERLE 从 {base_ne_erle:.2f} dB 降至 {ne_erle:.2f} dB"
                    )
                if explanations:
                    lines.append(
                        f"`{name}` 扩展了低音量覆盖，" + "，".join(explanations) +
                        "；低音量样本信噪比恶化，导致混合场景回声抑制能力下降。"
                    )
                    lines.append("")

                if isinstance(bt_dnsmos, (int, float)) and isinstance(base_bt, (int, float)) and bt_dnsmos < base_bt:
                    lines.append(
                        f"同时 doubletalk DNSMOS 从 {base_bt:.4f} 降至 {bt_dnsmos:.4f}，"
                        f"说明音量范围扩展未能转化为盲测集主指标收益。"
                    )
                    lines.append("")

            # val_loss 与主指标背离提示
            train = members[idx].get("training_metrics", {})
            base_train = members[baseline_idx].get("training_metrics", {})
            val_loss = train.get("best_val_loss")
            base_val_loss = base_train.get("best_val_loss")
            bt_dnsmos = _metric_value(idx, "doubletalk_DNSMOS")
            base_bt = _metric_value(baseline_idx, "doubletalk_DNSMOS")
            if (
                isinstance(val_loss, (int, float))
                and isinstance(base_val_loss, (int, float))
                and isinstance(bt_dnsmos, (int, float))
                and isinstance(base_bt, (int, float))
                and val_loss < base_val_loss - 0.05
                and bt_dnsmos < base_bt
            ):
                lines.append(
                    f"`{name}` 的 best_val_loss（{val_loss:.4f}）低于基线（{base_val_loss:.4f}），"
                    f"但 doubletalk DNSMOS 反而更低，说明训练/验证 loss 的数值变化与盲测感知质量不一致。"
                )
                lines.append("")

    return lines


# 盲测集前缀分组规则：共同前缀的多个子数据集合并为一个 section
_BLIND_DATASET_PREFIXES = ("blind_test_set_icassp2023",)


def _extract_blind_prefix(ds_name: str) -> str:
    """提取盲测集分组前缀。"""
    low = ds_name.lower()
    for prefix in _BLIND_DATASET_PREFIXES:
        if low.startswith(prefix):
            return prefix
    return ds_name


def _collect_dataset_groups(exps: list[dict]) -> dict:
    """汇总所有实验中出现的数据集，并按类型分组。

    Returns:
        {"flat": set(...), "nested": {ds_name: set(scenarios)}, "prefixed_groups": {prefix: set(sub_datasets)}}
    """
    flat: set[str] = set()
    nested: dict[str, set[str]] = {}
    prefixed_groups: dict[str, set[str]] = {}

    for exp in exps:
        for ds_name, ds_result in exp.get("eval_results", {}).items():
            if not isinstance(ds_result, dict):
                continue
            if _is_nested_scenario_result(ds_result):
                nested.setdefault(ds_name, set()).update(ds_result.keys())
            else:
                prefix = _extract_blind_prefix(ds_name)
                if prefix != ds_name:
                    prefixed_groups.setdefault(prefix, set()).add(ds_name)
                else:
                    flat.add(ds_name)

    return {"flat": flat, "nested": nested, "prefixed_groups": prefixed_groups}


def _select_main_dataset(dataset_groups: dict, exps: list[dict]) -> Optional[tuple[str, str]]:
    """选择主指标数据集，返回 (kind, name)。

    - AEC 主指标：AEC Challenge 盲测集（`blind_test_set_icassp2023_*` 前缀分组）。
    - AEC 参考集：`aec_sim`（嵌套场景，不作为主指标表，但在报告中优先展示）。
    - SE 主指标：`urgent2026` / `dns_challenge` / `voicebank`（flat 数据集）。
    """
    task_type = exps[0].get("task_type", "") if exps else ""
    flat = dataset_groups.get("flat", set())
    nested = dataset_groups.get("nested", {})
    prefixed_groups = dataset_groups.get("prefixed_groups", {})

    if task_type == "aec":
        # AEC 主指标是 AEC Challenge 盲测集（前缀分组）
        if prefixed_groups:
            prefix = sorted(prefixed_groups.keys())[0]
            return ("prefixed", prefix)
        # 回退到 aec_sim 参考集
        if nested:
            ds_name = sorted(nested.keys())[0]
            return ("nested", ds_name)
        # 其他 AEC flat 数据集
        aec_flat = [d for d in flat if d not in {"voicebank", "urgent", "dns_challenge"}]
        if aec_flat:
            return ("flat", sorted(aec_flat)[0])
        return None

    if task_type == "se":
        for preferred in ("urgent", "dns_challenge", "voicebank"):
            if preferred in flat:
                return ("flat", preferred)
        if flat:
            return ("flat", sorted(flat)[0])
        return None

    # 未知任务类型通用回退
    if flat:
        return ("flat", sorted(flat)[0])
    if prefixed_groups:
        return ("prefixed", sorted(prefixed_groups.keys())[0])
    if nested:
        return ("nested", sorted(nested.keys())[0])
    return None


_DATASET_DISPLAY_NAMES = {
    "urgent": "URGENT2026 (SE, 有参考)",
    "voicebank": "VoiceBank-Demand (SE, 有参考)",
    "dns_challenge": "DNS Challenge (SE, 盲测)",
    "aec_sim": "AEC Sim (参考集)",
    "blind_test_set_icassp2023": "AEC Challenge (ICASSP 2023 盲测集)",
}


def _dataset_display_name(ds_name: str) -> str:
    """数据集展示名（含前缀分组）。"""
    if ds_name in _DATASET_DISPLAY_NAMES:
        return _DATASET_DISPLAY_NAMES[ds_name]
    low = ds_name.lower()
    for prefix, display in _DATASET_DISPLAY_NAMES.items():
        if low.startswith(prefix):
            suffix = ds_name[len(prefix):].lstrip("_-")
            return f"{display} — {suffix}" if suffix else display
    return ds_name


from core.data.simulation.scene_names import AEC_SCENE_ORDER

_AEC_SIM_SCENARIO_ORDER = list(AEC_SCENE_ORDER)


def _scenario_display_order(scenarios: set[str]) -> list[str]:
    """aec_sim 等嵌套场景展示顺序。"""
    ordered = [s for s in _AEC_SIM_SCENARIO_ORDER if s in scenarios]
    ordered.extend(sorted(s for s in scenarios if s not in _AEC_SIM_SCENARIO_ORDER))
    return ordered


def _prefixed_display_order(prefix: str, sub_datasets: set[str]) -> list[str]:
    """盲测集子数据集展示顺序。"""
    order = ["doubletalk", "farend-singletalk", "nearend-singletalk"]
    prefix_with_sep = prefix + "_"
    ordered = []
    for suffix in order:
        full = prefix_with_sep + suffix
        if full in sub_datasets:
            ordered.append(full)
    ordered.extend(sorted(s for s in sub_datasets if s not in ordered))
    return ordered


def _order_metrics_for_dataset(ds_name: str, metrics: set[str]) -> list[str]:
    """根据数据集类型选择并排序指标列。"""
    low = ds_name.lower()

    if "dns_challenge" in low:
        priority = ["Normalized_Average", "DNSMOS", "NISQA", "UTMOS", "SCOREQ"]
    elif low.startswith("blind_test_set_icassp2023"):
        if "farend" in low:
            priority = ["Normalized_Average", "DNSMOS", "NISQA", "UTMOS", "SCOREQ", "ERLE_dB", "AECMOS_ECHO", "AECMOS_DEG"]
        else:
            priority = ["Normalized_Average", "DNSMOS", "NISQA", "UTMOS", "SCOREQ", "AECMOS_ECHO", "AECMOS_DEG"]
    else:
        # urgent / voicebank / 其他有参考集
        priority = ["Normalized_Average", "PESQ", "SI-SDR", "DNSMOS",
                    "NISQA", "UTMOS", "SCOREQ", "STOI", "ESTOI", "AECMOS_DEG"]

    ordered = [m for m in priority if m in metrics]
    ordered.extend(sorted(m for m in metrics if m not in priority))
    return ordered


def _order_metrics_for_scenario(scen_name: str, metrics: set[str]) -> list[str]:
    """根据 aec_sim 场景类型选择并排序指标列。"""
    echo_only_scenarios = {"prob_echo_only", "prob_noise_echo_only"}

    if scen_name in echo_only_scenarios:
        priority = ["ERLE_dB", "AECMOS_ECHO", "AECMOS_DEG"]
    elif scen_name == "prob_noise_only":
        priority = ["NOISE_REDUCTION_dB"]
    elif scen_name in {"prob_target_only", "prob_target_noise"}:
        priority = ["PESQ", "SI-SDR", "Normalized_Average",
                    "STOI", "ESTOI", "DNSMOS", "NISQA", "UTMOS", "SCOREQ"]
    else:
        # prob_target_echo / prob_target_noise_echo
        priority = ["PESQ", "SI-SDR", "Normalized_Average", "ERLE_dB",
                    "NEAREND_SISDR_dB", "AECMOS_ECHO", "AECMOS_DEG",
                    "STOI", "ESTOI", "DNSMOS", "NISQA", "UTMOS", "SCOREQ"]

    ordered = [m for m in priority if m in metrics]
    ordered.extend(sorted(m for m in metrics if m not in priority))
    return ordered


def _build_flat_dataset_table(exps: list[dict], ds_name: str) -> list[str]:
    """单个 flat 数据集跨实验对比表。"""
    metrics: set[str] = set()
    for exp in exps:
        ds_result = exp.get("eval_results", {}).get(ds_name, {})
        if isinstance(ds_result, dict):
            metrics.update(k for k in ds_result.keys() if not _is_annotation_key(k))

    ordered = _order_metrics_for_dataset(ds_name, metrics)
    if not ordered:
        return []
    return _build_metric_table_rows(exps, ds_name, ordered, is_scenario=False)


def _build_scenario_table(exps: list[dict], ds_name: str, scen_name: str) -> list[str]:
    """嵌套数据集（aec_sim）单个场景跨实验对比表。"""
    metrics: set[str] = set()
    for exp in exps:
        ds_result = exp.get("eval_results", {}).get(ds_name, {})
        if isinstance(ds_result, dict) and scen_name in ds_result:
            scen = ds_result[scen_name]
            if isinstance(scen, dict):
                metrics.update(k for k in scen.keys() if not _is_annotation_key(k))

    ordered = _order_metrics_for_scenario(scen_name, metrics)
    if not ordered:
        return []
    return _build_metric_table_rows(exps, (ds_name, scen_name), ordered, is_scenario=True)


def _build_metric_table_rows(
    exps: list[dict],
    ds_or_scen: Union[str, tuple[str, str]],
    metrics: list[str],
    is_scenario: bool,
) -> list[str]:
    """通用跨实验指标表行构建。"""
    lines: list[str] = []
    header = ["实验", "日期"] + [_metric_display_name(m) for m in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["------"] * len(header)) + "|")

    for exp in exps:
        ts = exp.get("timestamp", "")[:10] if exp.get("timestamp") else "-"
        # 有编号时用编号指代（图例表给出全称）；无编号时显示 checkpoint identity。
        display = _display_experiment_name(exp)
        display_name = exp.get("_exp_id") or (
            display if len(display) <= 55 else display[:52] + "..."
        )

        if isinstance(ds_or_scen, tuple):
            ds_name, scen_name = ds_or_scen
            metric_values = exp.get("eval_results", {}).get(ds_name, {}).get(scen_name, {})
        else:
            metric_values = exp.get("eval_results", {}).get(ds_or_scen, {})

        row = [display_name, ts]
        for metric in metrics:
            row.append(_format_metric_cell(metric_values, metric))
        lines.append("| " + " | ".join(row) + " |")

    return lines


def _build_experiment_detail(exp: dict) -> list[str]:
    """从 public DTO 构建单个实验的详情折叠块。"""
    lines = []
    exp_name = exp["exp_name"]
    display_name = _display_experiment_name(exp)
    exp_id = exp.get("_exp_id")

    lines.append(f"<details>")
    if exp_id:
        lines.append(f"<summary><b>{exp_id}</b> <code>{display_name}</code></summary>")
    else:
        lines.append(f"<summary><b>{display_name}</b></summary>")
    lines.append("")

    # 基本信息
    ts = exp.get("timestamp", "")[:16].replace("T", " ") if exp.get("timestamp") else "-"
    lines.append(f"- **时间**: {ts}")
    lines.append(f"- **任务**: {exp.get('task_type', '-').upper()}")
    lines.append(f"- **模型**: {exp.get('model_type', '-')}")
    if _uses_checkpoint_specific_identity(exp):
        lines.append(f"- **Checkpoint identity**: `{_artifact_identity(exp)}`")
    lines.append(f"- **配置**: `{exp.get('config_path', '-')}`")
    lines.append(f"- **来源训练 Checkpoint**: `{exp.get('source_checkpoint_path', '-')}`")
    source_fingerprint = exp.get("source_checkpoint_fingerprint") or {}
    if source_fingerprint.get("sha256"):
        lines.append(f"  SHA256=`{source_fingerprint['sha256']}`")
    if exp.get("waveform_native"):
        lines.append("- **消费 Checkpoint**: waveform-native；权重由配置 artifact 注入，未消费 root checkpoint")
    else:
        lines.append(f"- **消费 Model-only Checkpoint**: `{exp.get('model_only_checkpoint_path') or '-'}`")
        model_only_fingerprint = exp.get("model_only_checkpoint_fingerprint") or {}
        if model_only_fingerprint.get("sha256"):
            lines.append(f"  SHA256=`{model_only_fingerprint['sha256']}`")

    # 实验目的
    purpose = exp.get("purpose", "")
    if purpose:
        lines.append(f"- **实验目的**: {purpose}")
    metric_device = exp.get("metric_device")
    inference_devices = exp.get("inference_devices")
    if metric_device or inference_devices:
        lines.append(
            f"- **评估设备**: 指标/延迟/ONNX/badcase=`{metric_device or '-'}`；"
            f"模型推理=`{','.join(inference_devices) if isinstance(inference_devices, list) else inference_devices or '-'}`"
        )
    lines.append("")

    # 训练指标
    train_metrics = exp.get("training_metrics", {})
    if train_metrics and train_metrics.get("total_epochs"):
        lines.append("**训练指标**:")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        lines.append(f"| 总 Epochs | {train_metrics.get('total_epochs', '-')} |")
        if train_metrics.get("best_val_loss") is not None:
            lines.append(f"| Best Val Loss (epoch {train_metrics.get('best_val_loss_epoch', '-')}) | {train_metrics['best_val_loss']:.4f} |")
        for k, v in train_metrics.get("final_epoch_metrics", {}).items():
            # 跳过 epoch 列、非数值、以及过于冗长的 loss 组件名
            if k == "epoch" or not isinstance(v, (int, float)):
                continue
            if "val_loss_comp" in k and "::" in k:
                continue
            lines.append(f"| {k} (final epoch) | {v:.4f} |")
        lines.append("")

    # 评估结果
    eval_results = exp.get("eval_results", {})
    if eval_results:
        lines.append("**评估结果**:")
        lines.append("")
        _append_eval_tables(lines, eval_results)
        # 逐样本谱系（JAE-M06）：均值/样本数在 RESULTS.txt，
        # 下游与音频质量指标逐样本值在同目录 per_utterance.json（若存在）
        eval_root_path = _find_eval_root(exp)
        eval_root = (
            _render_path(eval_root_path)
            if eval_root_path is not None
            else _render_path(
                default_eval_output_root(_PROJECT_ROOT)
                / str(exp.get("task_type", "-"))
                / exp_name
            )
        )
        lines.append(
            f"- 逐样本记录: `{eval_root}/<dataset>/RESULTS.txt`（均值 + `_count_` 样本数）；"
            "下游与音频质量（PESQ/MOS）指标逐样本值见同目录 `per_utterance.json`"
            "（新口径评估产生；历史目录无此文件）。"
        )
        lines.append("")
    else:
        lines.append("> ⚠️ 暂无评估结果，请运行 `evaluate` 步骤。")
        lines.append("")

    latency_verify = exp.get("latency_verify")
    if latency_verify:
        status = latency_verify.get("status", "-")
        values = [latency_verify.get(key) for key in (
            "min_future_dependency_ms", "mean_future_dependency_ms", "max_future_dependency_ms",
        )]
        values_txt = " / ".join(
            f"{value:.3f} ms" if isinstance(value, (int, float)) else "N/A" for value in values
        )
        lines.append(f"**算法未来依赖验证**: **{status}**（最小/均值/最大: {values_txt}）")
        threshold = latency_verify.get("threshold")
        sample_rate = latency_verify.get("sample_rate")
        n_perturbations = latency_verify.get("n_perturbations")
        if threshold is not None or sample_rate is not None or n_perturbations is not None:
            lines.append(
                "- 参数: "
                f"threshold={threshold if threshold is not None else 'N/A'}，"
                f"{sample_rate if sample_rate is not None else 'N/A'} Hz，"
                f"n={n_perturbations if n_perturbations is not None else 'N/A'}"
            )
        lines.append("")
        lines.append(
            "> 正值表示 look-ahead，负值表示输出缓冲或刻意延时；该指标衡量输出对未来"
            "输入的算法依赖，不是 wall-clock 推理耗时，不是 RTF，也不是 GCC-PHAT 输入对齐时延。"
        )
        if exp.get("task_type") == "aec":
            lines.append("> AEC 测量仅扰动 mic、固定 far-end ref，用于检查对 mic 的未来依赖。")
        if latency_verify.get("error"):
            lines.append(f"> ⚠️ 执行错误: {latency_verify['error']}")
        if latency_verify.get("log_path"):
            lines.append(f"- 日志: `{latency_verify['log_path']}`")
        lines.append("")

    cpu_rtf = exp.get("cpu_rtf")
    if cpu_rtf:
        rtf = cpu_rtf.get("rtf_median")
        mean_rtf = cpu_rtf.get("rtf_mean")
        median_seconds = cpu_rtf.get("median_seconds")
        duration_seconds = cpu_rtf.get("duration_seconds")
        def _fmt_cpu_rtf(value, fmt, suffix=""):
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                return f"{value:{fmt}}{suffix}"
            return "N/A"

        lines.append("**CPU 推理 RTF**:")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        lines.append(f"| RTF（中位数） | {_fmt_cpu_rtf(rtf, '.4f')} |")
        lines.append(f"| RTF（平均） | {_fmt_cpu_rtf(mean_rtf, '.4f')} |")
        lines.append(f"| 30 秒输入中位数纯推理耗时 | {_fmt_cpu_rtf(median_seconds, '.3f', ' s')} |")
        lines.append(f"| 测试输入时长 | {_fmt_cpu_rtf(duration_seconds, '.1f', ' s')} |")
        lines.append(f"| CPU 线程 / interop 线程 | {cpu_rtf.get('threads', 'N/A')} / {cpu_rtf.get('interop_threads', 'N/A')} |")
        lines.append("")
        lines.append(
            "> 仅统计 CPU 上的 STFT、模型前向和 iSTFT wall-clock 时间；不包含模型加载、音频读取或 WAV 写盘。"
        )
        lines.append(
            "> RTF 越低越快；该指标与算法未来依赖验证的 look-ahead 延迟不是同一个指标。"
        )
        lines.append("")

    # ONNX 三路精度验证
    onnx_verify = exp.get("onnx_verify")
    if onnx_verify:
        status = onnx_verify.get("status", "-")
        is_dbgnet = "dbgnet" in str(
            onnx_verify.get("family") or exp.get("model_type") or ""
        ).casefold()
        release_gates = onnx_verify.get("release_gates")
        display_status = status
        if is_dbgnet and not isinstance(release_gates, dict):
            display_status = "不可审计"
        elif is_dbgnet and release_gates.get("audit_status") == "UNAUDITABLE":
            display_status = "不可审计"
        elif is_dbgnet and release_gates.get("audit_status") == "FAIL":
            display_status = "FAIL"
        # validation_scope 为 widened 时展示发布级合成验证范围；字段缺失或 narrow
        # 保持历史报告的窄口径表述。
        scope = onnx_verify.get("scope") or {}
        frames = scope.get("frames") or "脚本默认"
        resets = scope.get("reset") or "未覆盖"
        inputs = scope.get("inputs") or "随机白噪声 mic/far waveform"
        tail_block = scope.get("tail_block") or "未覆盖 waveform 尾块/任意分块"
        validation_scope = scope.get("validation_scope")
        lines.append(
            f"**ONNX 三路精度验证** (离线 PT / 流式 PT / 流式 ORT): **{display_status}**"
            + (
                " — widened synthetic PASS"
                if display_status == "PASS" and validation_scope == "widened"
                else " — narrow synthetic PASS" if display_status == "PASS" else ""
            )
        )
        lines.append("")
        artifact = onnx_verify.get("onnx_artifact") or {}
        artifact_scope = "已发布的同一验证 ONNX" if artifact.get("published") else "验证 ONNX"
        if validation_scope == "widened":
            lines.append(
                f"> ⚠️ 验证口径: widened synthetic —— {inputs}、{artifact_scope}、CPU ORT、"
                f"frames={frames}、resets={resets}（3 段谱图帧序列；段间同步 cold reset cache）。"
                f"{tail_block}，且不覆盖流式 STFT/iSTFT、量化或目标 provider；"
                "PASS 不代表 deployment PASS。"
            )
        else:
            lines.append(
                f"> ⚠️ 验证口径: narrow synthetic —— {inputs}、{artifact_scope}、CPU ORT、"
                f"单次 cold start（frames={frames}，无 reset 覆盖）。{tail_block}，且不覆盖长序列漂移、"
                "流式 STFT/iSTFT、量化或目标 provider；PASS 不代表 deployment PASS。"
            )
        lines.append("")
        checks = onnx_verify.get("checks", [])
        if checks:
            lines.append("| 检查项 | 指标 | 结果 |")
            lines.append("|------|------|------|")
            for chk in checks:
                lines.append(f"| {chk.get('label', '-')} | `{chk.get('metric', '-')}` | {chk.get('status', '-')} |")
            lines.append("")

        if is_dbgnet:
            release_gates = onnx_verify.get("release_gates")
            lines.append("**DBGNet 独立发布门禁**:")
            lines.append("")
            if not isinstance(release_gates, dict):
                lines.append(
                    "> ⚠️ 不可审计：记录缺少结构化 State MSE/max_abs、State gate 与 "
                    "External iSTFT synthesis capability；不得依据尾部汇总 PASS 推断这些门禁通过。"
                )
                lines.append("")
            else:
                state_enabled = release_gates.get("state_detector_enabled")
                lines.append("| 独立门禁 | 指标/范围 | 结果 |")
                lines.append("|------|------|------|")
                lines.append(
                    "| State detector | "
                    f"enabled={state_enabled if state_enabled is not None else '未知'} | "
                    f"{'可审计' if state_enabled is not None else '不可审计'} |"
                )
                for metric in release_gates.get("state_metrics") or []:
                    mse = metric.get("mse")
                    max_abs = metric.get("max_abs")
                    metric_text = (
                        f"MSE={mse:.6e}, max_abs={max_abs:.6e}"
                        if isinstance(mse, (int, float)) and isinstance(max_abs, (int, float))
                        else "MSE/max_abs 不可解析"
                    )
                    lines.append(
                        f"| State MSE/max_abs {metric.get('comparison', '-')} | "
                        f"`{metric_text}` | 记录 |"
                    )
                state_gate = release_gates.get("state_gate")
                if state_enabled is False:
                    lines.append("| State gate | state detector disabled | N/A |")
                else:
                    lines.append(
                        f"| State gate | `{(state_gate or {}).get('detail') or '-'}` | "
                        f"{(state_gate or {}).get('status') or '不可审计'} |"
                    )
                external_istft = release_gates.get(
                    "external_istft_synthesis_capability"
                ) or {}
                lines.append(
                    f"| External iSTFT synthesis capability | "
                    f"`{external_istft.get('detail') or '-'}` | "
                    f"{external_istft.get('status') or '不可审计'} |"
                )
                if release_gates.get("audit_status") == "UNAUDITABLE":
                    missing = ", ".join(release_gates.get("missing") or [])
                    lines.append("")
                    lines.append(
                        f"> ⚠️ 不可审计：DBGNet verifier stdout 缺少 `{missing or '独立门禁字段'}`；"
                        "不得依据尾部汇总 PASS 推断发布门禁通过。"
                    )
                lines.append("")
        # 产物谱系（JAE-M10）：被验证 checkpoint 的 SHA256 与临时 ONNX 去向
        ckpt_prov = onnx_verify.get("checkpoint_artifact") or {}
        if ckpt_prov.get("sha256"):
            size_mb = ckpt_prov.get("size_bytes", 0) / 1024 / 1024
            lines.append(f"- Checkpoint: `{ckpt_prov.get('path', '-')}`")
            lines.append(f"  SHA256=`{ckpt_prov['sha256']}`（{size_mb:.1f} MB）")
        artifact = onnx_verify.get("onnx_artifact")
        if artifact:
            state = "已发布" if artifact.get("published") else "未发布"
            lines.append(f"- ONNX 目标: `{artifact.get('expected_path', '-')}`（{state}）")
            if artifact.get("published"):
                size_mb = artifact.get("size_bytes", 0) / 1024 / 1024
                lines.append(
                    f"  已发布文件 SHA256=`{artifact.get('sha256', '-')}`（{size_mb:.1f} MB）"
                )
        if onnx_verify.get("onnx_temp_path"):
            kept = onnx_verify.get("onnx_artifact_kept")
            if kept is True:
                fate = "（verify 脚本已保留，可复核）"
            elif kept is False:
                fate = "（verify 脚本创建的临时文件，验证结束后由脚本删除）"
            else:
                fate = ""
            sha = onnx_verify.get("onnx_artifact_sha256")
            sha_txt = f" SHA256=`{sha}`" if sha else ""
            lines.append(f"- 被验证 ONNX: `{onnx_verify['onnx_temp_path']}`{sha_txt}{fate}")
        if onnx_verify.get("error"):
            lines.append(f"> ⚠️ 执行错误: {onnx_verify['error']}")
            lines.append("")
        if onnx_verify.get("log_path"):
            lines.append(f"- 日志: `{onnx_verify['log_path']}`")
            lines.append("")

    # Badcase 快速推理
    badcase = exp.get("badcase")
    if badcase and badcase.get("status") in {"done", "partial", "error"}:
        lines.append("**Badcase 快速推理**")
        lines.append("")
        lines.append(f"- 状态: `{badcase['status']}`")
        checkpoint_epoch = badcase.get("checkpoint_epoch")
        epoch_text = f"{checkpoint_epoch:04d}" if isinstance(checkpoint_epoch, int) else "unknown"
        lines.append(
            f"- Checkpoint Epoch: `{epoch_text}`（来源: "
            f"`{badcase.get('checkpoint_epoch_source', 'epoch_unknown')}`）"
        )
        result_dir = badcase.get("result_dir")
        if result_dir:
            lines.append(f"- 输出根目录: `{result_dir}`")
        n_samples = badcase.get("n_samples", 0)
        lines.append(f"- 样本数: {n_samples}")
        for mode, mode_info in badcase.get("modes", {}).items():
            mode_status = mode_info.get("status")
            if mode_status is None:
                lines.append(
                    f"- `{mode}`: {mode_info.get('n_files', 0)} 个文件 → "
                    f"`{mode_info.get('output_dir', '-')}`（历史记录，无 manifest 状态）"
                )
            else:
                lines.append(
                    f"- `{mode}`: status=`{mode_status}`，"
                    f"manifest 成功/期望={mode_info.get('n_files', 0)}/"
                    f"{mode_info.get('n_expected', 0)}，failed={mode_info.get('n_failed', 0)} → "
                    f"`{mode_info.get('output_dir', '-')}`"
                )
            if mode_info.get("manifest"):
                lines.append(f"  - 运行清单: `{mode_info['manifest']}`")
            if mode_info.get("inf_scp"):
                lines.append(f"  - 输出清单: `{mode_info['inf_scp']}`")
            if mode_info.get("delay_summary"):
                lines.append(f"  - badcase 摘要: `{mode_info['delay_summary']}`")
            if mode_info.get("inference_log"):
                lines.append(f"  - 推理日志: `{mode_info['inference_log']}`")
            if mode_info.get("error"):
                lines.append(f"  - 错误: {mode_info['error']}")
        lines.append("")

    lines.append("</details>")
    return lines


def _append_eval_tables(lines: list[str], eval_results: dict) -> None:
    """追加评估结果表格。盲测和非盲测分开。"""
    blind_results = {}
    nonblind_results = {}

    for ds_name, ds_result in eval_results.items():
        if not isinstance(ds_result, dict):
            continue
        # 判断盲测/非盲测
        is_blind = any(b in ds_name.lower() for b in BLIND_DATASETS)

        # 检查是否是嵌套场景（aec_sim）
        if _is_nested_scenario_result(ds_result):
            # 嵌套场景 - 展开
            if is_blind:
                blind_results[ds_name] = ds_result
            else:
                nonblind_results[ds_name] = ds_result
        else:
            if is_blind or ds_name.startswith("blind"):
                blind_results[ds_name] = ds_result
            else:
                nonblind_results[ds_name] = ds_result

    # 非盲测集
    if nonblind_results:
        lines.append("##### 非盲测集")
        lines.append("")
        _build_metrics_table(lines, nonblind_results)
        lines.append("")

    # 盲测集
    if blind_results:
        lines.append("##### 盲测集 (无参考，仅MOS)")
        lines.append("")
        _build_metrics_table(lines, blind_results)
        lines.append("")


def _format_normalized_score(metric_values: dict) -> str:
    score = metric_values.get("_normalized_score")
    projection = False
    if not isinstance(score, Mapping):
        score = metric_values.get("_normalized_score_projection")
        projection = isinstance(score, Mapping)
    if not isinstance(score, Mapping):
        score = project_existing_score(
            metric_values, _historical_score_coverage(metric_values)
        )
        projection = True

    value = score.get("value")
    cell = f"{float(value):.4f}" if isinstance(value, (int, float)) else "N/A"
    dimensions = score.get("dimensions") if isinstance(score.get("dimensions"), Mapping) else {}
    status = str(score.get("status", "legacy-unknown"))
    if projection:
        projection_status = "unknown" if status == "legacy-unknown" else "partial"
        label = f"legacy/{projection_status} projection"
    else:
        label = status
    details = [
        f"dimensions={dimensions.get('valid', 0)}/{dimensions.get('required', 6)}"
    ]
    coverage = score.get("sample_coverage")
    if projection:
        observed = []
        missing = list(dimensions.get("missing", []))
        if isinstance(coverage, Mapping):
            for name in NORMALIZED_METRICS:
                item = coverage.get(name)
                if isinstance(item, Mapping) and item.get("status") == "observed":
                    observed.append(f"{name}={item.get('computed')}")
        if observed:
            details.append(f"observed={','.join(observed)}")
        if missing:
            details.append(f"missing={','.join(missing)}")
    elif status == "partial" and isinstance(coverage, Mapping):
        for name in NORMALIZED_METRICS:
            item = coverage.get(name)
            if not isinstance(item, Mapping) or item.get("status") == "complete":
                continue
            computed, eligible = item.get("computed"), item.get("eligible")
            if isinstance(computed, int) and isinstance(eligible, int):
                details.append(f"{name}={computed}/{eligible}")
    return f"{cell} ({label}; {'; '.join(details)})"


def _format_metric_cell(metric_values: dict, metric: str) -> str:
    """渲染指标值；normalized score 只校验 sidecar 或投影历史舍入值。"""
    if metric == "Normalized_Average":
        return _format_normalized_score(metric_values)

    v = metric_values.get(metric, "-")
    if isinstance(v, float) and math.isfinite(v):
        cell = f"{v:.4f}"
    elif isinstance(v, int):
        cell = f"{v:.4f}"
    else:
        cell = str(v) if v != "-" else "-"
    if cell == "-":
        return cell
    n = metric_values.get(f"_count_{metric}")
    if n is not None:
        n_disp = int(n) if isinstance(n, float) and n.is_integer() else n
        cell += f" (n={n_disp})"
    return cell


def _build_metrics_table(lines: list[str], dataset_results: dict) -> None:
    """构建指标表格。"""
    # 收集所有指标（注解键如 _count_* / Normalized_Average_Denominator 不作为列）
    all_metrics = set()
    for ds_result in dataset_results.values():
        if isinstance(ds_result, dict):
            # 检查嵌套
            if _is_nested_scenario_result(ds_result):
                for sub in ds_result.values():
                    if isinstance(sub, dict):
                        all_metrics.update(k for k in sub.keys() if not _is_annotation_key(k))
            else:
                all_metrics.update(k for k in ds_result.keys() if not _is_annotation_key(k))

    metric_order = ["PESQ", "STOI", "ESTOI", "SI-SDR",
                    "DNSMOS", "NISQA", "UTMOS", "SCOREQ",
                    "ERLE_dB", "NEAREND_SISDR_dB",
                    "NOISE_REDUCTION_dB", "AECMOS_ECHO", "AECMOS_DEG",
                    "Normalized_Average"]
    ordered = [m for m in metric_order if m in all_metrics]
    ordered.extend(sorted(m for m in all_metrics if m not in metric_order))

    if not ordered:
        return

    # 表头（key 不变，展示名按实际口径映射，如 SCOREQ → SCOREQ-NR）
    header = ["数据集"] + [_metric_display_name(m) for m in ordered]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["------"] * len(header)) + "|")

    for ds_name, ds_result in sorted(dataset_results.items()):
        if isinstance(ds_result, dict):
            if _is_nested_scenario_result(ds_result):
                # 嵌套（aec_sim 场景）
                for scen_name, scen_metrics in sorted(ds_result.items()):
                    if isinstance(scen_metrics, dict):
                        short_name = scen_name[:25]
                        row = [f"  ↳ {short_name}"]
                        for m in ordered:
                            row.append(_format_metric_cell(scen_metrics, m))
                        lines.append("| " + " | ".join(row) + " |")
            else:
                row = [ds_name[:30]]
                for m in ordered:
                    row.append(_format_metric_cell(ds_result, m))
                lines.append("| " + " | ".join(row) + " |")


def _scoreq_provenance() -> dict:
    """从 core.metrics.mos_models.load_scoreq 的实际默认值取 SCOREQ 谱系（JAE-L02）。

    报告元数据必须与加载器一致：mode / data_domain 取函数签名默认值，
    model filename 由 _SCOREQ_ONNX_FILES 映射推导；读取失败时回退 "unknown"。
    """
    try:
        import inspect
        from core.metrics.mos_models import _SCOREQ_ONNX_FILES, load_scoreq
        params = inspect.signature(load_scoreq).parameters
        mode = params["mode"].default
        domain = params["data_domain"].default
        fname = _SCOREQ_ONNX_FILES.get((domain, mode), "unknown")
        # 版本依据见 core/metrics/mos_models.py SCOREQ 节注释（NeurIPS 2024）
        return {"model_file": fname, "mode": mode, "data_domain": domain,
                "version": "NeurIPS 2024"}
    except Exception:
        return {"model_file": "unknown", "mode": "unknown",
                "data_domain": "unknown", "version": "unknown"}


# 指标展示名映射：RESULTS.txt 的 key 保持不变（历史解析兼容），
# 仅报告表头按实际运行口径展示（JAE-L02：SCOREQ 实际为 NR 模式）。
_METRIC_DISPLAY_NAME = {"SCOREQ": "SCOREQ-NR"}


def _metric_display_name(metric: str) -> str:
    return _METRIC_DISPLAY_NAME.get(metric, metric)


_ANNOTATION_KEYS = {
    "Normalized_Average_Denominator",
    "Normalized_Average_Status",
    "_normalized_score",
    "_normalized_score_projection",
}


def _is_annotation_key(key: str) -> bool:
    return (
        key in _ANNOTATION_KEYS
        or key.startswith("_count_")
        or key.startswith("Normalized_Average_Coverage_")
    )


def _build_metric_legend() -> list[str]:
    """构建指标说明。"""
    # SCOREQ 谱系从加载器实际默认值取（JAE-L02）：评估链路只以 NR 模式运行，
    # 指标标为 SCOREQ-NR，不再使用 "Reference-based" 全称。
    sq = _scoreq_provenance()
    scoreq_desc = (
        f"SCOREQ No-Reference MOS（无参考评分；model={sq['model_file']}, "
        f"mode={sq['mode']}, domain={sq['data_domain']}, version={sq['version']}）"
    )
    return [
        "",
        "---",
        "",
        "### 指标说明",
        "",
        "| 指标 | 说明 | 方向 |",
        "|------|------|------|",
        "| PESQ | Perceptual Evaluation of Speech Quality | ↑ |",
        "| STOI | Short-Time Objective Intelligibility | ↑ |",
        "| ESTOI | Extended STOI | ↑ |",
        "| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio (dB) | ↑ |",
        "| DNSMOS | DNS Challenge MOS predictor (OVRL) | ↑ |",
        "| NISQA | Non-Intrusive Speech Quality Assessment | ↑ |",
        "| UTMOS | UTokyo MOS predictor | ↑ |",
        f"| SCOREQ-NR | {scoreq_desc} | ↑ |",
        "| Normalized_Average | 固定 6 项指标的归一化均值。`official` 要求 dimensions=6/6 且六项样本覆盖全部完整；`partial` 表示本轮 sidecar 的维度或样本覆盖不完整；历史 projection 只展示 RESULTS 明示的 `observed` 成功计数与 `missing` 维度，不推断 eligible 或覆盖率；`legacy/unknown projection` 表示历史舍入值没有可验证的覆盖分母。报告端不从舍入指标重算该值 | ↑ |",
        "| ERLE | Echo Return Loss Enhancement (dB) | ↑ |",
        "| NEAREND_SISDR_dB | 近端保持 SI-SDR (dB)，仿真 doubletalk 场景与 ERLE 成对解读（口径见 evaluate/eval_pipeline_aec.py） | ↑ |",
        "| NOISE_REDUCTION_dB | 噪声抑制量 (dB)，prob_noise_only 场景 | ↑ |",
        "| AECMOS_ECHO | AEC MOS 回声抑制评分 | ↑ |",
        "| AECMOS_DEG | AEC MOS 语音质量评分 | ↑ |",
    ]


def _empty_report() -> str:
    return textwrap.dedent("""\
    # 实验训练报告

    > 暂无实验数据。

    运行训练并执行以下命令来添加实验:

    ```bash
    # SE 实验（单层目录）
    python -m tools.ops.reporting.experiment_report run exp/se/<exp_name>

    # AEC 实验（两层目录，必须传到叶子实验目录）
    python -m tools.ops.reporting.experiment_report run exp/aec/<model_group>/<exp_name>
    ```
    """)


def _publish_experiment_documents_unlocked(
    experiments: list[dict], *, include_index: bool
) -> None:
    """由一个已持锁 index 快照构建并发布报告文档。"""
    report = _build_report_from_experiments(experiments)
    outputs = [(REPORT_FILE, lambda stream: stream.write(report))]
    if include_index:
        index_content = _serialize_experiments_index(experiments)
        outputs.insert(0, (INDEX_FILE, lambda stream: stream.write(index_content)))
    write_text_files_transactionally(outputs, overwrite=True)


def save_report(_content: str) -> Path:
    """兼容保存 API：锁内重读当前 index，拒绝写入锁外陈旧 content。"""
    with _persistent_flock(_report_lock_path()):
        _publish_experiment_documents_unlocked(
            _load_experiments_index_unlocked(), include_index=False
        )
    return REPORT_FILE


def update_report_snapshot(exp_data: dict, eval_results: dict) -> Path:
    """在业务锁内通过统一发布 seam 更新 index/Markdown 文件对。"""
    with _persistent_flock(_report_lock_path()):
        existing = _merge_experiment_index(
            _load_experiments_index_unlocked(), exp_data, eval_results
        )
        _publish_experiment_documents_unlocked(existing, include_index=True)
    return REPORT_FILE


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _discover_from_args(args: argparse.Namespace) -> dict:
    """从 CLI 参数发现资产；未选 checkpoint 时保留旧调用形式。"""
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None:
        return discover_exp_assets(args.exp_dir)
    return discover_exp_assets(args.exp_dir, checkpoint=checkpoint)


def cmd_discover(args: argparse.Namespace) -> int:
    """发现实验资产。"""
    exp_data = _discover_from_args(args)
    exp_data["purpose"] = infer_experiment_purpose(exp_data)
    exp_data["training_metrics"] = parse_training_metrics(exp_data)
    exp_data["model_family"] = infer_model_family(exp_data.get("resolved_config", {}))

    # 检查已有 eval 结果
    eval_results = collect_existing_eval_results(exp_data)

    summary = {
        "exp_name": exp_data["exp_name"],
        "checkpoint_identity": _artifact_identity(exp_data),
        "explicit_checkpoint": exp_data.get("explicit_checkpoint", False),
        "task_type": exp_data["task_type"],
        "model_type": _runtime_model_config(
            exp_data.get("resolved_config", {})
        ).get("type", "unknown"),
        "model_family": exp_data["model_family"],
        "purpose": exp_data["purpose"],
        "config_path": exp_data["config_path"],
        "source_checkpoint_path": exp_data.get(
            "source_checkpoint_path", exp_data.get("checkpoint_path")
        ),
        "source_checkpoint_fingerprint": exp_data.get("source_checkpoint_fingerprint"),
        "expected_model_only_checkpoint_path": exp_data.get("expected_model_only_checkpoint_path"),
        "model_only_checkpoint_path": exp_data.get("model_only_checkpoint_path"),
        "model_only_checkpoint_fingerprint": exp_data.get("model_only_checkpoint_fingerprint"),
        "checkpoint_path": exp_data.get("checkpoint_path"),
        "log_dir": exp_data.get("log_dir"),
        "timestamp": exp_data["timestamp"].isoformat() if exp_data["timestamp"] else None,
        "training_metrics": {
            "total_epochs": exp_data["training_metrics"].get("total_epochs"),
            "best_val_loss": exp_data["training_metrics"].get("best_val_loss"),
            "best_val_loss_epoch": exp_data["training_metrics"].get("best_val_loss_epoch"),
        },
        "has_eval_results": bool(eval_results),
        "eval_datasets": _public_dynamic_keys(eval_results) if eval_results else [],
    }
    print(json.dumps(_sanitize_published_paths(summary), ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """运行评估流水线。"""
    exp_data = _discover_from_args(args)
    task_type = exp_data["task_type"]
    if _reject_unsupported_eval_options(args, task_type):
        return 2
    if exp_data.get("waveform_native"):
        _validate_consumed_checkpoint_before_child(exp_data)
    _materialize_consumed_checkpoint(
        exp_data,
        trusted_training_checkpoint=getattr(
            args, "trusted_training_checkpoint", False
        ),
    )
    datasets = args.datasets or (AEC_DATASETS if task_type == "aec" else SE_DATASETS)
    inference_devices = normalize_inference_devices(
        getattr(args, "inference_devices", None), fallback=args.device
    )
    exp_data["metric_device"] = args.device
    exp_data["inference_devices"] = inference_devices

    logger.info(f"任务类型: {task_type.upper()}")
    logger.info("指标/延迟/ONNX/badcase device: %s | inference devices: %s", args.device, inference_devices)
    logger.info(f"实验名称: {exp_data['exp_name']}")
    logger.info(f"Checkpoint identity: {_artifact_identity(exp_data)}")
    logger.info(f"数据集: {datasets}")
    _validate_consumed_checkpoint_before_child(exp_data)

    return run_eval_pipeline(
        exp_dir=exp_data["exp_dir"],
        task_type=task_type,
        datasets=datasets,
        audio_quality=not args.no_audio_quality,
        device=args.device,
        inference_devices=inference_devices,
        aec_no_stream_align=getattr(args, "aec_no_stream_align", False),
        skip_existing_inference=getattr(args, "skip_existing_inference", False),
        aec_testset_root=getattr(args, "aec_testset_root", None),
        urgent_noisy_scp=getattr(args, "urgent_noisy_scp", None),
        urgent_clean_scp=getattr(args, "urgent_clean_scp", None),
        voicebank_clean_scp=getattr(args, "voicebank_clean_scp", None),
        voicebank_noisy_scp=getattr(args, "voicebank_noisy_scp", None),
        dns_challenge_noisy_dir=getattr(args, "dns_challenge_noisy_dir", None),
        output_name=(
            _checkpoint_specific_identity(exp_data)
            if _uses_checkpoint_specific_identity(exp_data)
            else None
        ),
        config_path=exp_data.get("config_path"),
        checkpoint_path=(
            exp_data.get("checkpoint_path")
            if exp_data.get("config_path") is not None
            else None
        ),
        expected_effective_config_identity=exp_data.get(
            "resolved_config_fingerprint"
        ),
    )


def cmd_report(args: argparse.Namespace) -> int:
    """生成/更新报告。"""
    exp_data = _discover_from_args(args)

    # 推断实验目的
    if args.purpose:
        exp_data["purpose"] = args.purpose
    else:
        exp_data["purpose"] = infer_experiment_purpose(exp_data)

    # 解析训练指标
    exp_data["training_metrics"] = parse_training_metrics(exp_data)

    # 收集评估结果及其设备谱系
    eval_results = collect_existing_eval_results(exp_data)
    _load_eval_device_provenance(exp_data)
    if not eval_results and not args.skip_eval_check:
        logger.warning("未找到评估结果。请先运行 evaluate 步骤，或使用 --skip-eval-check")
        logger.info("运行: python -m tools.ops.reporting.experiment_report evaluate <exp_dir>")
        return 1

    # index 与 Markdown 在同一业务锁内成对发布；普通异常会回滚旧文件对。
    path = update_report_snapshot(exp_data, eval_results)
    logger.info(f"报告已保存: {path}")

    # 打印摘要
    _print_summary(exp_data, eval_results)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """一键全流程。"""
    # 1. Discover
    logger.info("=" * 60)
    logger.info("Step 1/6: 发现实验资产")
    logger.info("=" * 60)
    exp_data = _discover_from_args(args)
    if _reject_unsupported_eval_options(args, exp_data["task_type"]):
        return 2
    if exp_data.get("waveform_native") and not getattr(args, "skip_eval", False):
        _validate_consumed_checkpoint_before_child(exp_data)
    _materialize_consumed_checkpoint(
        exp_data,
        trusted_training_checkpoint=getattr(
            args, "trusted_training_checkpoint", False
        ),
    )

    if args.purpose:
        exp_data["purpose"] = args.purpose
    else:
        exp_data["purpose"] = infer_experiment_purpose(exp_data)

    exp_data["training_metrics"] = parse_training_metrics(exp_data)
    exp_data["model_family"] = infer_model_family(exp_data.get("resolved_config", {}))

    logger.info(f"  实验: {exp_data['exp_name']}")
    logger.info(f"  Checkpoint identity: {_artifact_identity(exp_data)}")
    logger.info(f"  任务: {exp_data['task_type'].upper()}")
    runtime_model_type = _runtime_model_config(
        exp_data.get("resolved_config", {})
    ).get("type", "unknown")
    logger.info(
        f"  模型: {runtime_model_type} ({exp_data['model_family']} 系列)"
    )
    logger.info(f"  目的: {exp_data['purpose']}")
    logger.info(f"  Config: {exp_data['config_path']}")
    logger.info(f"  Checkpoint: {exp_data['checkpoint_path']}")

    train = exp_data["training_metrics"]
    if train.get("total_epochs"):
        logger.info(f"  训练: {train['total_epochs']} epochs, "
                    f"best val_loss={train.get('best_val_loss', 'N/A')}")

    inference_devices = normalize_inference_devices(
        getattr(args, "inference_devices", None), fallback=args.device
    )
    exp_data["metric_device"] = args.device
    exp_data["inference_devices"] = inference_devices

    # 2. Evaluate
    if args.skip_eval:
        logger.info("")
        logger.info("Step 2/6: 跳过评估 (--skip-eval)")
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 2/6: 运行评估")
        logger.info("=" * 60)
        task_type = exp_data["task_type"]
        datasets = args.datasets or (AEC_DATASETS if task_type == "aec" else SE_DATASETS)
        logger.info(f"  数据集: {datasets}")
        logger.info("  指标/延迟/ONNX/badcase device: %s | inference devices: %s", args.device, inference_devices)
        _validate_consumed_checkpoint_before_child(exp_data)

        ret = run_eval_pipeline(
            exp_dir=exp_data["exp_dir"],
            task_type=task_type,
            datasets=datasets,
            audio_quality=not args.no_audio_quality,
            device=args.device,
            inference_devices=inference_devices,
            aec_no_stream_align=getattr(args, "aec_no_stream_align", False),
            skip_existing_inference=getattr(args, "skip_existing_inference", False),
            aec_testset_root=getattr(args, "aec_testset_root", None),
            urgent_noisy_scp=getattr(args, "urgent_noisy_scp", None),
            urgent_clean_scp=getattr(args, "urgent_clean_scp", None),
            voicebank_clean_scp=getattr(args, "voicebank_clean_scp", None),
            voicebank_noisy_scp=getattr(args, "voicebank_noisy_scp", None),
            dns_challenge_noisy_dir=getattr(args, "dns_challenge_noisy_dir", None),
            output_name=(
                _checkpoint_specific_identity(exp_data)
                if _uses_checkpoint_specific_identity(exp_data)
                else None
            ),
            config_path=exp_data["config_path"],
            checkpoint_path=exp_data["checkpoint_path"],
            expected_effective_config_identity=exp_data.get(
                "resolved_config_fingerprint"
            ),
        )
        if ret != 0:
            logger.error(f"评估失败 (exit code: {ret})")
            return ret

    # 无已有评估结果或 waveform-native 不消费 root checkpoint 时，跳过 root checkpoint 下游。
    eval_results = collect_existing_eval_results(exp_data)
    if args.skip_eval:
        _load_eval_device_provenance(exp_data)
        if _uses_checkpoint_specific_identity(exp_data) and not eval_results:
            logger.error(
                "显式 checkpoint 的 --skip-eval 只能复用同 identity 的已提交评估结果: %s",
                _eval_root_for(exp_data),
            )
            return 1
    skip_inference_steps = (args.skip_eval and not eval_results) or exp_data.get(
        "waveform_native", False
    )
    if exp_data.get("waveform_native"):
        logger.info("waveform-native 模型权重由配置 artifact 注入，跳过 root checkpoint 验证与 badcase")
    elif skip_inference_steps:
        logger.warning("--skip-eval 指定且未找到已有评估结果，跳过算法未来依赖验证、ONNX 验证和 Badcase 推理")

    # 3. 算法未来依赖验证
    logger.info("")
    logger.info("=" * 60)
    if args.skip_latency_verify or skip_inference_steps:
        logger.info("Step 3/6: 跳过算法未来依赖验证 (--skip-latency-verify)")
        existing = _index_entry_for_identity(load_experiments_index(), exp_data)
        exp_data["latency_verify"] = (existing or {}).get("latency_verify")
    else:
        logger.info("Step 3/6: 算法未来依赖验证")
        exp_data["latency_verify"] = run_latency_verification(exp_data, device=args.device)
    logger.info("=" * 60)

    # 4. ONNX 三路精度验证
    logger.info("")
    logger.info("=" * 60)
    if args.skip_onnx_verify or skip_inference_steps:
        logger.info("Step 4/6: 跳过 ONNX 三路验证 (--skip-onnx-verify)")
        # 保留同一 checkpoint identity 的已有验证结果，避免被 None 覆盖。
        existing = _index_entry_for_identity(load_experiments_index(), exp_data)
        exp_data["onnx_verify"] = (existing or {}).get("onnx_verify")
    else:
        logger.info("Step 4/6: ONNX 三路精度验证")
        exp_data["onnx_verify"] = run_onnx_verification(exp_data, device=args.device)
    logger.info("=" * 60)

    # 5. Report
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 5/6: 生成报告")
    logger.info("=" * 60)

    if eval_results:
        logger.info("  评估数据集: %s", _public_dynamic_keys(eval_results))
    else:
        logger.warning("  未找到评估结果")

    path = update_report_snapshot(exp_data, eval_results)
    logger.info(f"  报告已保存: {path}")

    _print_summary(exp_data, eval_results)

    # 6. Badcase test inference (AEC only)
    if not args.skip_badcase and not skip_inference_steps and exp_data.get("task_type") == "aec":
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 6/6: Badcase 快速推理")
        logger.info("=" * 60)
        exp_data["badcase"] = _run_badcase_inference(
            exp_data, device=args.device, skip_existing=args.skip_existing_badcase,
            with_align=args.badcase_with_align,
        )
        # badcase 信息进入 index 后，与 Markdown 在同一业务锁内成对发布。
        path = update_report_snapshot(exp_data, eval_results)
        logger.info(f"  报告已更新: {path}")

    return 0


# ---------------------------------------------------------------------------
# Badcase 音频快速推理测试
# ---------------------------------------------------------------------------
# 输入样本统一置于 USE_TRAIN_TEST_DATA_ROOT/badcase/samples/ 下，允许按来源目录递归组织；每个 mode
# 的 output_dir 同时保存 inference.log、delay_summary.json、run_manifest.json、inf.scp 和 WAV。
# 样本通过 rglob 递归收集（见 _collect_badcase_samples），输出按来源目录保留层级。
BADCASE_TEST_DIR = DEFAULT_TEST_DATA_ROOT / "badcase" / "samples"
BADCASE_RESULT_DIR = DEFAULT_TEST_DATA_ROOT / "badcase" / "outputs"
BADCASE_CPU_LONG_AUDIO_SECONDS = 30.0

# 单文件 badcase 由 _collect_badcase_samples 的通用规则识别，无需硬编码清单：
# 任何文件名不以 *_mic / *_ref 结尾的音频，都视为近端单文件 badcase，文件本身即 mic，
# ref 固定使用等长全零信号。*_mic 与 *_ref 必须同目录同名（仅后缀不同），异名文件不会猜配。

def _collect_badcase_samples(test_dir: Path, tmp_dir: Path) -> list[tuple[str, Path, Path]]:
    """发现 badcase 样本；缺少 ref 时生成等长空信号临时文件。

    递归遍历 samples/ 下的来源目录，识别两类样本：
      1) 标准配对：``*_mic.wav`` 及同目录同名 ``*_ref.wav``；缺少 ref 时生成等长空信号。
      2) 单文件样本：文件名不以 ``*_mic`` / ``*_ref`` 结尾的音频，文件本身即 mic，
         ref 恒为等长全零信号（无独立 _mic/_ref 拆分）。

    ``*_mic`` 与 ``*_ref`` 按同目录同名配对；当前资产中的 ``[ans-mic]`` /
    ``[ans-ref]`` 使用明确的字符串替换规则。其他孤立的 ``*_ref.wav`` 不会被猜配，
    并会记录 warning，避免静默丢失新命名规则下的参考。

    Returns:
        [(stem, mic_path, ref_path), ...]。返回空列表表示无有效样本。
    """
    samples: list[tuple[str, Path, Path]] = []
    matched_refs: set[Path] = set()

    # 1) 标准 *_mic.wav / *_ref.wav 配对（递归到子目录）
    for mic_path in sorted(test_dir.rglob("*_mic.wav")):
        stem = mic_path.stem[:-4]  # 去掉末尾 "_mic"
        ref_path = mic_path.parent / f"{stem}_ref.wav"
        # 当前 badcase 资产还使用时间戳序号递增的 [ans-mic] / [ans-ref] 命名；
        # 仅按这个明确格式推导，不根据目录顺序或近似字符串做模糊猜配。
        if not ref_path.exists() and "[ans-mic]" in stem:
            match = re.match(r"^(\d{8}-\d{6}-)(\d+)(-.*)\[ans-mic\]$", stem)
            if match:
                alt_stem = (
                    f"{match.group(1)}{int(match.group(2)) + 1}"
                    f"{match.group(3)}[ans-ref]"
                )
                alt_ref_path = mic_path.parent / f"{alt_stem}_ref.wav"
                if alt_ref_path.exists():
                    ref_path = alt_ref_path
        if ref_path.exists():
            matched_refs.add(ref_path.resolve())
        else:
            empty_ref = _create_empty_reference(mic_path, tmp_dir)
            if empty_ref is None:
                continue
            ref_path = empty_ref
            logger.info(f"  {stem}: 未找到参考，使用等长空信号")
        samples.append((stem, mic_path, ref_path))

    for ref_path in sorted(test_dir.rglob("*_ref.wav")):
        if ref_path.resolve() not in matched_refs:
            logger.warning(
                "发现未配对的 badcase 参考文件，未自动猜配: %s",
                ref_path,
            )

    # 2) 单文件 badcase：文件本身即 mic，ref 恒为等长全零信号
    #    递归发现所有不是 *_mic/*_ref 配对的音频（如 delay_* 下的 mixture、meeting_women）。
    for wav in sorted(test_dir.rglob("*.wav")):
        name = wav.name
        if name.endswith("_mic.wav") or name.endswith("_ref.wav"):
            continue
        empty_ref = _create_empty_reference(wav, tmp_dir)
        if empty_ref is None:
            continue
        samples.append((wav.stem, wav, empty_ref))
        logger.info(f"  {wav.name}: 单文件 badcase，ref 使用等长全零信号")

    if not samples:
        logger.info("跳过 badcase 推理: 未找到有效样本")
        return []

    # 去除重复 stem（同名不同目录的极端情况按目录扁平化去重，保留首个）
    dedup: dict[str, tuple[Path, Path]] = {}
    for stem, mic_path, ref_path in samples:
        dedup.setdefault(stem, (mic_path, ref_path))
    return [(s, m, r) for s, (m, r) in dedup.items()]



def _create_empty_reference(mic_path: Path, tmp_dir: Path) -> Path | None:
    """生成与输入等长的空信号参考文件。"""
    try:
        wav, sr = torchaudio.load(str(mic_path))
        empty = torch.zeros_like(wav)
        tmp_ref = tmp_dir / f"empty_ref_{mic_path.stem}.wav"
        torchaudio.save(str(tmp_ref), empty, sr)
        return tmp_ref
    except Exception as e:
        logger.warning(f"生成空参考失败 {mic_path}: {e}")
        return None


def _create_scp_files(samples: list[tuple[str, Path, Path]]) -> tuple[str, str]:
    """创建 shell 引号安全的 mic/far SCP 临时文件。"""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".scp", delete=False) as f_mic:
        for stem, mic_path, _ in samples:
            f_mic.write(f"{shlex.quote(stem)} {shlex.quote(str(mic_path))}\n")
        mic_scp = f_mic.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".scp", delete=False) as f_far:
        for stem, _, ref_path in samples:
            f_far.write(f"{shlex.quote(stem)} {shlex.quote(str(ref_path))}\n")
        far_scp = f_far.name

    return mic_scp, far_scp


def _build_badcase_output_relpaths(
    samples: list[tuple[str, Path, Path]],
    cat_map: dict[str, str | None],
) -> dict[str, str]:
    """按 UID 和输入来源目录构建安全的相对输出 WAV 路径。"""
    output_relpaths: dict[str, str] = {}
    for uid, _, _ in samples:
        filename = f"{uid}.wav"
        category = cat_map.get(uid)
        relative = Path(category) / filename if category else Path(filename)
        if (
            not isinstance(uid, str)
            or not uid
            or Path(filename).name != filename
            or relative.is_absolute()
            or relative.parts[:1] == ("..",)
            or ".." in relative.parts
            or any(part in {"", "."} for part in relative.parts)
            or any(char in str(relative) for char in "\\\r\n\0")
        ):
            raise ValueError(f"badcase 输出相对路径不安全: uid={uid!r}, path={relative}")
        output_relpaths[uid] = relative.as_posix()
    return output_relpaths


def _create_badcase_output_relpath_scp(
    output_relpaths: dict[str, str],
) -> str:
    """创建临时两列 ``uid output_relpath`` 映射，供 inference.py 消费。"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".scp", delete=False) as output_scp:
        for uid, relative in output_relpaths.items():
            output_scp.write(f"{shlex.quote(uid)} {shlex.quote(relative)}\n")
        return output_scp.name


def _validate_badcase_review_audio(
    output_path: Path,
    mic_path: Path,
    ref_path: Path,
    expected_sample_rate: int,
) -> dict[str, Any]:
    """读取实际审听 WAV，校验采样率、帧数和 ``mic[/ref]/enhanced`` 通道合同。"""
    output, output_sr = torchaudio.load(str(output_path))
    mic, mic_sr = torchaudio.load(str(mic_path))
    ref, ref_sr = torchaudio.load(str(ref_path))
    if mic.shape[-1] <= 0 or ref.shape[-1] <= 0:
        raise ManifestValidationError("badcase 输入音频帧数必须为正")
    if not torch.isfinite(mic).all() or not torch.isfinite(ref).all():
        raise ManifestValidationError("badcase 输入音频包含 NaN 或 Inf")
    if output.ndim != 2 or not torch.isfinite(output).all():
        raise ManifestValidationError("badcase 审听输出必须是有限值二维音频")
    mic = mic.mean(0, keepdim=True)
    ref = ref.mean(0, keepdim=True)
    if ref_sr != mic_sr:
        ref = torchaudio.functional.resample(ref.float(), ref_sr, mic_sr)
    if mic_sr != expected_sample_rate:
        mic = torchaudio.functional.resample(mic.float(), mic_sr, expected_sample_rate)
        ref = torchaudio.functional.resample(ref.float(), mic_sr, expected_sample_rate)
    mic = mic.squeeze(0)[: min(mic.shape[-1], ref.shape[-1])]
    ref = ref.squeeze(0)[: mic.shape[-1]]
    expected_frames = mic.shape[-1]
    if output_sr != expected_sample_rate:
        raise ManifestValidationError(
            "badcase 输出采样率不匹配: "
            f"expected={expected_sample_rate}, actual={output_sr}"
        )
    if output.shape[-1] != expected_frames:
        raise ManifestValidationError(
            f"badcase 输出帧数不匹配: expected={expected_frames}, actual={output.shape[-1]}"
        )
    ref_all_zero = bool(torch.count_nonzero(ref).item() == 0)
    expected_channels = 2 if ref_all_zero else 3
    channels = int(output.shape[0])
    if channels != expected_channels:
        raise ManifestValidationError(
            "badcase 审听输出通道数不匹配: "
            f"expected={expected_channels}, actual={channels}"
        )
    if not torch.allclose(output[0], mic, atol=1e-4, rtol=1e-4):
        raise ManifestValidationError("badcase 审听输出第 1 通道不是 mic")
    if channels == 3 and not torch.allclose(output[1], ref, atol=1e-4, rtol=1e-4):
        raise ManifestValidationError("badcase 审听输出第 2 通道不是 ref")
    channel_layout = ["mic", "enhanced"]
    if channels == 3:
        channel_layout = ["mic", "ref", "enhanced"]
    return {
        "channels": channels,
        "channel_layout": channel_layout,
        "sample_rate": int(output_sr),
        "num_frames": int(output.shape[-1]),
        "ref_all_zero": ref_all_zero,
    }


def _validate_badcase_manifest_identity(
    manifest: dict,
    results: list[Any],
    *,
    exp_data: dict,
    samples: list[tuple[str, Path, Path]],
    mode: str,
    output_dir: Path,
    device: str,
    output_relpaths: Optional[dict[str, str]] = None,
) -> None:
    """校验 manifest 身份，并严格绑定 UID 对应的 mode-root 相对输出路径。"""
    expected_uids = [uid for uid, _, _ in samples]
    if manifest["expected_uids"] != expected_uids:
        raise ManifestValidationError("badcase manifest UID 与本次样本不一致")
    if Path(manifest["config"]).resolve() != Path(exp_data["config_path"]).resolve():
        raise ManifestValidationError("badcase manifest config 与本次运行不一致")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, str) or (
        Path(checkpoint).resolve() != Path(exp_data["checkpoint_path"]).resolve()
    ):
        raise ManifestValidationError("badcase manifest checkpoint 与本次运行不一致")

    parameters = manifest["artifact_fingerprint"]["parameters"]
    expected_flags = {
        "aec_stream_align": mode == "with_align",
        "aec_no_stream_align": mode == "no_align",
        "aec_review_multichannel": True,
        "requested_device": device,
        "cpu_long_audio_seconds": BADCASE_CPU_LONG_AUDIO_SECONDS,
    }
    for key, expected in expected_flags.items():
        if parameters.get(key) != expected:
            raise ManifestValidationError(
                f"badcase manifest 参数 {key} 与本次运行不一致"
            )

    output_relpaths = output_relpaths or {
        uid: f"{uid}.wav" for uid in expected_uids
    }
    if set(output_relpaths) != set(expected_uids):
        raise ManifestValidationError("badcase 输出相对路径映射与样本 UID 不一致")
    root = output_dir.resolve()
    for result in results:
        if result.status == "failed":
            continue
        relative_text = output_relpaths[result.uid]
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative.name != f"{result.uid}.wav"
            or ".." in relative.parts
            or any(char in relative_text for char in "\\\r\n\0")
        ):
            raise ManifestValidationError(
                f"badcase manifest 输出相对路径不安全: {result.uid}"
            )
        expected_output = (root / relative).resolve()
        if expected_output != root and not expected_output.is_relative_to(root):
            raise ManifestValidationError(
                f"badcase manifest 输出越界: {result.uid}"
            )
        actual_output = Path(result.output_path)
        if actual_output.resolve() != expected_output:
            raise ManifestValidationError(
                f"badcase manifest 输出越界或文件名不匹配: {result.uid}"
            )
        if not actual_output.is_file():
            raise ManifestValidationError(
                f"badcase manifest 输出不存在: {result.uid}"
            )


def _validate_badcase_artifacts(
    output_dir: Path,
    *,
    exp_data: dict,
    samples: list[tuple[str, Path, Path]],
    mode: str,
    device: str,
    output_relpaths: dict[str, str],
) -> tuple[dict, list[Any]]:
    """校验 mode 根的 manifest、inf.scp、输入输出身份及模型 provenance。"""
    # 旧测试通过替换本模块的 loader 提供轻量 mock；生产路径始终复用统一校验。
    if load_run_manifest is _REAL_LOAD_RUN_MANIFEST:
        manifest, results = validate_inference_artifacts(
            output_dir,
            config=exp_data["config_path"],
            checkpoint=exp_data["checkpoint_path"],
        )
        expected_inventory = build_input_inventory(
            (uid, {"mic": mic_path, "farend": ref_path})
            for uid, mic_path, ref_path in samples
        )
        if manifest["input_inventory"] != expected_inventory:
            raise ManifestValidationError(
                "badcase manifest 输入 identity 与本次样本不一致"
            )
    else:
        manifest, results = load_run_manifest(output_dir / "run_manifest.json")
    _validate_badcase_manifest_identity(
        manifest,
        results,
        exp_data=exp_data,
        samples=samples,
        mode=mode,
        output_dir=output_dir,
        device=device,
        output_relpaths=output_relpaths,
    )
    return manifest, results


def _badcase_metadata_snapshot(output_dir: Path) -> dict[str, tuple[int, int, int, int] | None]:
    """记录 manifest/inf.scp inode 与时间，识别子进程未发布新事务。"""
    snapshot: dict[str, tuple[int, int, int, int] | None] = {}
    for name in ("run_manifest.json", "inf.scp"):
        try:
            file_stat = (output_dir / name).stat()
        except OSError:
            snapshot[name] = None
        else:
            snapshot[name] = (
                int(file_stat.st_dev),
                int(file_stat.st_ino),
                int(file_stat.st_size),
                int(file_stat.st_mtime_ns),
            )
    return snapshot


def _cleanup_stale_badcase_wavs(output_dir: Path, results: list[Any]) -> None:
    """仅保留可信 manifest 成功输出；任何 stale WAV 清理失败均 fail closed。

    只允许在本轮 manifest 通过校验后调用：推理失败、manifest 无效或审阅音频
    合同失败等错误路径必须保留既有产物，避免用一次失败运行抹掉上次成功输出。
    """
    current_outputs = {
        Path(result.output_path).resolve()
        for result in results
        if result.status in {"ok", "skipped"}
    }
    failures: list[str] = []
    for wav_path in output_dir.rglob("*.wav"):
        if wav_path.resolve() in current_outputs:
            continue
        try:
            wav_path.unlink()
        except OSError as exc:
            failures.append(f"{wav_path}: {exc}")
    if failures:
        raise OSError("清理旧 badcase WAV 失败: " + "; ".join(failures))


def _run_badcase_mode(
    exp_data: dict,
    samples: list[tuple[str, Path, Path]],
    mic_scp: str,
    far_scp: str,
    mode: str,
    mode_flag: list[str],
    output_dir: Path,
    device: str,
    skip_existing: bool,
    cat_map: Optional[dict[str, str | None]] = None,
) -> dict:
    """持久 mode 锁下运行推理，并严格按 mode 根 ``run_manifest.json`` 汇总。

    inference.py 通过临时 UID→相对路径映射直接把 WAV 写入 mode 根；因此根目录
    同时保存 manifest、``inf.scp``、日志和摘要，分类 WAV 只生成一个实际文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "delay_summary.json"
    cat_map = cat_map or {}
    output_relpaths = _build_badcase_output_relpaths(samples, cat_map)

    with _persistent_flock(output_dir / ".badcase_mode.lock"):
        output_relpath_scp = _create_badcase_output_relpath_scp(output_relpaths)
        try:
            metadata_before = _badcase_metadata_snapshot(output_dir)
            logger.info(f"badcase 推理 ({mode})...")
            cmd = [
                sys.executable, str(_PROJECT_ROOT / "inference.py"),
                "-C", exp_data["config_path"],
                "--checkpoint", exp_data["checkpoint_path"],
                "--input_mic_scp", mic_scp,
                "--input_far_scp", far_scp,
                "--output_dir", str(output_dir),
                "--output_relpath_scp", output_relpath_scp,
                "--device", device,
                "--cpu-long-audio-seconds", f"{BADCASE_CPU_LONG_AUDIO_SECONDS:g}",
                "--aec-review-multichannel",
                "--allow-partial",
            ] + mode_flag
            if skip_existing:
                cmd.append("--skip-existing")

            try:
                process = subprocess.run(
                    cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True
                )
            except Exception as exc:
                combined_log = f"inference.py 启动失败: {exc}"
                atomic_write_text(output_dir / "inference.log", combined_log + "\n")
                mode_info = {
                    "status": "error",
                    "output_dir": str(output_dir),
                    "n_files": 0,
                    "n_expected": len(samples),
                    "n_failed": len(samples),
                    "records": [],
                    "error": str(exc),
                    "delay_summary": str(summary_path),
                    "inference_log": str(output_dir / "inference.log"),
                    "manifest": str(output_dir / "run_manifest.json"),
                    "inf_scp": str(output_dir / "inf.scp"),
                }
                atomic_write_text(
                    summary_path,
                    json.dumps(mode_info, ensure_ascii=False, indent=2) + "\n",
                )
                logger.warning("badcase (%s) inference.py 启动失败: %s", mode, exc)
                return mode_info

            combined_log = process.stdout + "\n" + process.stderr
            atomic_write_text(output_dir / "inference.log", combined_log)
            if process.returncode != 0:
                mode_info = {
                    "status": "error",
                    "output_dir": str(output_dir),
                    "n_files": 0,
                    "n_expected": len(samples),
                    "n_failed": len(samples),
                    "records": [],
                    "error": f"inference.py exit code: {process.returncode}",
                    "delay_summary": str(summary_path),
                    "inference_log": str(output_dir / "inference.log"),
                    "manifest": str(output_dir / "run_manifest.json"),
                    "inf_scp": str(output_dir / "inf.scp"),
                }
                atomic_write_text(
                    summary_path,
                    json.dumps(mode_info, ensure_ascii=False, indent=2) + "\n",
                )
                logger.warning(
                    "badcase (%s) inference.py exit code: %s", mode, process.returncode
                )
                return mode_info

            try:
                metadata_after = _badcase_metadata_snapshot(output_dir)
                if load_run_manifest is _REAL_LOAD_RUN_MANIFEST and any(
                    metadata_after[name] is None
                    or metadata_after[name] == metadata_before[name]
                    for name in ("run_manifest.json", "inf.scp")
                ):
                    raise ManifestValidationError(
                        "badcase 子进程未发布本次 manifest/inf.scp"
                    )
                manifest, manifest_results = _validate_badcase_artifacts(
                    output_dir,
                    exp_data=exp_data,
                    samples=samples,
                    mode=mode,
                    device=device,
                    output_relpaths=output_relpaths,
                )
                _cleanup_stale_badcase_wavs(output_dir, manifest_results)
            except (ManifestValidationError, OSError, ValueError) as exc:
                mode_info = {
                    "status": "error",
                    "output_dir": str(output_dir),
                    "n_files": 0,
                    "n_expected": len(samples),
                    "n_failed": len(samples),
                    "records": [],
                    "error": str(exc),
                    "delay_summary": str(summary_path),
                    "inference_log": str(output_dir / "inference.log"),
                    "manifest": str(output_dir / "run_manifest.json"),
                    "inf_scp": str(output_dir / "inf.scp"),
                }
                atomic_write_text(
                    summary_path,
                    json.dumps(mode_info, ensure_ascii=False, indent=2) + "\n",
                )
                logger.warning("badcase (%s) manifest 无效: %s", mode, exc)
                return mode_info

            records: list[dict] = []
            sample_paths = {
                uid: (mic_path, ref_path) for uid, mic_path, ref_path in samples
            }
            sample_rate = _positive_epoch(
                (exp_data.get("resolved_config", {}).get("data") or {}).get("sample_rate")
            )
            if sample_rate is None:
                raise ManifestValidationError("badcase 配置缺少正整数 data.sample_rate")
            try:
                for manifest_result in manifest_results:
                    if manifest_result.status == "failed":
                        continue
                    relative = output_relpaths[manifest_result.uid]
                    output_path = Path(manifest_result.output_path)
                    mic_path, ref_path = sample_paths[manifest_result.uid]
                    record = {
                        "uid": manifest_result.uid,
                        "status": manifest_result.status,
                        "output_relpath": relative,
                        "output_path": str(output_path),
                        **_validate_badcase_review_audio(
                            output_path, mic_path, ref_path, sample_rate
                        ),
                    }
                    if cat_map.get(manifest_result.uid):
                        record["category"] = cat_map[manifest_result.uid]
                    records.append(record)
            except (OSError, RuntimeError, ValueError) as exc:
                mode_info = {
                    "status": "error",
                    "output_dir": str(output_dir),
                    "n_files": 0,
                    "n_expected": len(samples),
                    "n_failed": len(samples),
                    "records": [],
                    "error": str(exc),
                    "delay_summary": str(summary_path),
                    "inference_log": str(output_dir / "inference.log"),
                    "manifest": str(output_dir / "run_manifest.json"),
                    "inf_scp": str(output_dir / "inf.scp"),
                }
                atomic_write_text(
                    summary_path,
                    json.dumps(mode_info, ensure_ascii=False, indent=2) + "\n",
                )
                logger.warning("badcase (%s) 审阅音频合同无效: %s", mode, exc)
                return mode_info

            n_failed = len(manifest["failed_uids"])
            status = "error" if process.returncode != 0 else ("partial" if n_failed else "done")
            mode_info = {
                "status": status,
                "output_dir": str(output_dir),
                "n_files": len(records),
                "n_expected": len(manifest["expected_uids"]),
                "n_ok": len(manifest["ok_uids"]),
                "n_skipped": len(manifest["skipped_uids"]),
                "n_failed": n_failed,
                "failed_uids": list(manifest["failed_uids"]),
                "records": records,
                "delay_summary": str(summary_path),
                "inference_log": str(output_dir / "inference.log"),
                "manifest": str(output_dir / "run_manifest.json"),
                "inf_scp": str(output_dir / "inf.scp"),
            }
            if process.returncode != 0:
                mode_info["error"] = f"inference.py exit code: {process.returncode}"
            atomic_write_text(
                summary_path,
                json.dumps(mode_info, ensure_ascii=False, indent=2) + "\n",
            )
            logger.info(
                "badcase (%s): status=%s, manifest=%d/%d, failed=%d",
                mode,
                status,
                len(records),
                len(manifest["expected_uids"]),
                n_failed,
            )
            return mode_info
        finally:
            Path(output_relpath_scp).unlink(missing_ok=True)


def _badcase_result_paths(
    exp_data: dict,
) -> tuple[Path, Path, Optional[int], str]:
    """返回 checkpoint identity 隔离的 epoch 化输出目录及 provenance。"""
    identity = _artifact_identity(exp_data)
    if not identity or Path(identity).name != identity:
        raise ValueError(f"非法 checkpoint identity: {identity!r}")
    if _uses_checkpoint_specific_identity(exp_data):
        identity = _checkpoint_specific_identity(exp_data)
    checkpoint_epoch, source = _resolve_checkpoint_epoch(exp_data)
    suffix = (
        f"epoch{checkpoint_epoch:04d}"
        if checkpoint_epoch is not None
        else "epoch_unknown"
    )
    return (
        BADCASE_RESULT_DIR / f"{identity}_{suffix}",
        BADCASE_RESULT_DIR / identity,
        checkpoint_epoch,
        source,
    )


def _cleanup_legacy_badcase_result(result_dir: Path, legacy_dir: Path) -> None:
    """新 epoch 目录成功后仅删除同实验旧无 epoch 输出目录。"""
    root = BADCASE_RESULT_DIR.resolve()
    resolved_result = result_dir.resolve()
    resolved_legacy = legacy_dir.resolve()
    if (
        resolved_result == resolved_legacy
        or resolved_legacy.parent != root
        or resolved_result.parent != root
        or legacy_dir.is_symlink()
        or not resolved_legacy.is_dir()
    ):
        return
    shutil.rmtree(resolved_legacy)
    logger.info("已清理旧无 epoch badcase 输出目录: %s", resolved_legacy)


def _run_badcase_inference(
    exp_data: dict, device: str = "cuda:0", skip_existing: bool = False, with_align: bool = False,
) -> dict:
    """对 badcase 目录下的样本做快速推理验证，默认仅运行 ``no_align``。

    规则:
      - 输入目录: USE_TRAIN_TEST_DATA_ROOT/badcase/samples，递归发现样本并保留来源目录相对路径
      - 输出目录: USE_TRAIN_TEST_DATA_ROOT/badcase/outputs/<checkpoint_identity>_epochNNNN/，epoch 未知时使用
        ``_epoch_unknown``；每个 mode 根保存 manifest、inf.scp、日志、摘要和 WAV，
        WAV 按输入来源目录保留相对层级
      - 自动发现各目录下所有 *_mic.wav，仅与同目录同名 *_ref.wav 配对；
        缺少参考时，使用与输入等长的空信号作为临时参考，异名参考不会猜配
      - 单文件 badcase（文件名不以 *_mic/*_ref 结尾的音频，如 delay_* 中的 mixture、
        meeting_women.wav）：文件本身即 mic，ref 固定使用等长全零信号
      - 每个模式只启动一次 ``inference.py`` 子进程；固定传递 30 秒长音频 CPU
        阈值。AEC 使用原始 mic 帧数/采样率严格判断 ``>30`` 秒：超过 30 秒
        的文件切换到 CPU，等于或低于 30 秒的文件沿用 ``device``。规则同时
        适用于 ``no_align``、``with_align`` 以及新增的递归样本。
      - 每个文件记录最终审听 WAV 的相对/绝对路径、通道合同与来源分类

    返回:
      包含 result_dir、样本数、各模式输出目录及审听记录的字典，供报告索引使用。
    """
    import tempfile

    result_dir, legacy_dir, checkpoint_epoch, epoch_source = _badcase_result_paths(
        exp_data
    )
    result: dict = {
        "status": "skipped",
        "result_dir": str(result_dir),
        "n_samples": 0,
        "modes": {},
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_epoch_source": epoch_source,
    }

    test_dir = BADCASE_TEST_DIR
    if not test_dir.exists():
        logger.info("跳过 badcase 推理: 测试目录不存在")
        return result

    result_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="badcase_refs_", dir=result_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        samples = _collect_badcase_samples(test_dir, tmp_dir)
        result["n_samples"] = len(samples)
        if not samples:
            logger.info("跳过 badcase 推理: 没有有效样本")
            return result

        mic_scp, far_scp = _create_scp_files(samples)
        # 记录样本相对 samples/ 根的完整来源目录，作为 output-relpath 的前缀。
        cat_map: dict[str, str | None] = {}
        for stem, mic_path, _ in samples:
            try:
                relative_parent = mic_path.parent.resolve().relative_to(test_dir.resolve())
            except ValueError:
                # 测试替身或外部调用可能提供不在 samples 根下的路径；不创建分类 link。
                relative_parent = Path()
            cat_map[stem] = relative_parent.as_posix() if relative_parent != Path() else None


        try:
            modes = [("no_align", ["--aec_no_stream_align"], "no_align")]
            if with_align:
                modes.insert(0, ("with_align", ["--aec_stream_align"], "with_align"))
            for mode, mode_flag, mode_dir in modes:
                output_dir = result_dir / mode_dir
                result["modes"][mode] = _run_badcase_mode(
                    exp_data,
                    samples,
                    mic_scp,
                    far_scp,
                    mode,
                    mode_flag,
                    output_dir,
                    device,
                    skip_existing,
                    cat_map,
                )
            statuses = {mode_info["status"] for mode_info in result["modes"].values()}
            if "error" in statuses:
                result["status"] = "error"
            elif "partial" in statuses:
                result["status"] = "partial"
            else:
                result["status"] = "done"
                if not _uses_checkpoint_specific_identity(exp_data):
                    _cleanup_legacy_badcase_result(result_dir, legacy_dir)
        finally:
            Path(mic_scp).unlink(missing_ok=True)
            Path(far_scp).unlink(missing_ok=True)

    return result


def _print_summary(exp_data: dict, eval_results: dict) -> None:
    """打印不含本机绝对路径的评估结果摘要。"""
    if not eval_results:
        return
    exp_data = _sanitize_published_paths(exp_data)
    eval_results = _sanitize_published_paths(eval_results)

    print()
    print("=" * 60)
    print("评估结果摘要")
    print("=" * 60)

    for ds_name, ds_result in sorted(eval_results.items()):
        if not isinstance(ds_result, dict):
            continue

        # 判断嵌套
        if _is_nested_scenario_result(ds_result):
            print(f"\n[{ds_name}]")
            for scen_name, scen_metrics in sorted(ds_result.items()):
                if isinstance(scen_metrics, dict):
                    na = (
                        _format_normalized_score(scen_metrics)
                        if "Normalized_Average" in scen_metrics
                        else "-"
                    )
                    pesq = scen_metrics.get("PESQ", "-")
                    si_sdr = scen_metrics.get("SI-SDR", "-")
                    print(f"  {scen_name}: PESQ={pesq}, SI-SDR={si_sdr}, NormAvg={na}")
        else:
            na = (
                _format_normalized_score(ds_result)
                if "Normalized_Average" in ds_result
                else "-"
            )
            pesq = ds_result.get("PESQ", "-")
            si_sdr = ds_result.get("SI-SDR", "-")
            dnsmos = ds_result.get("DNSMOS", "-")
            print(f"[{ds_name}] PESQ={pesq}, SI-SDR={si_sdr}, DNSMOS={dnsmos}, NormAvg={na}")

    onnx_verify = exp_data.get("onnx_verify")
    if onnx_verify:
        print(f"\n[onnx_verify] {onnx_verify.get('family', '-')}: {onnx_verify.get('status', '-')}")

    print()
    print(f"完整报告: {_render_path(REPORT_FILE)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_checkpoint_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        help=(
            "显式训练 checkpoint；必须位于该实验 checkpoints/<run>/，"
            "并与同 run 配置成对使用"
        ),
    )


def _add_trusted_checkpoint_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trusted-training-checkpoint",
        action="store_true",
        help="显式信任自生成训练 checkpoint；仅安全提取失败时允许 unsafe fallback",
    )


def _add_eval_data_source_args(parser: argparse.ArgumentParser) -> None:
    """为 evaluate/run 注册显式原始评估数据源参数。"""
    parser.add_argument(
        "--aec-testset-root",
        help="AEC official 原始测试集根目录（仅传给 AEC pipeline）",
    )
    parser.add_argument(
        "--urgent-noisy-scp",
        help="URGENT nonblind noisy SCP（仅传给 AEC pipeline，须与 clean 成对）",
    )
    parser.add_argument(
        "--urgent-clean-scp",
        help="URGENT nonblind clean SCP（仅传给 AEC pipeline，须与 noisy 成对）",
    )
    parser.add_argument(
        "--voicebank-clean-scp",
        help="VoiceBank clean SCP",
    )
    parser.add_argument(
        "--voicebank-noisy-scp",
        help="VoiceBank noisy SCP",
    )
    parser.add_argument(
        "--dns-challenge-noisy-dir",
        help="DNS Challenge noisy 测试集目录",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="实验报告自动生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        示例:
          # 一键全流程
          %(prog)s run exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

          # 仅发现资产
          %(prog)s discover exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

          # 仅运行评估
          %(prog)s evaluate exp/aec/dbgnet_small_16k_10ms_delay80ms_loss_hybrid/xxx

          # 仅更新报告（评估已跑完）
          %(prog)s report exp/aec/xxx --skip-eval-check

          # 显式评估某个同 run 的 epoch checkpoint
          %(prog)s run exp/aec/xxx --checkpoint exp/aec/xxx/checkpoints/<run>/model_0060.pth
        """),
    )

    sub = p.add_subparsers(dest="command", help="子命令")

    # discover
    p_discover = sub.add_parser("discover", help="发现实验资产")
    p_discover.add_argument("exp_dir", help="实验目录路径")
    _add_checkpoint_arg(p_discover)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="运行评估流水线")
    p_eval.add_argument("exp_dir", help="实验目录路径")
    _add_checkpoint_arg(p_eval)
    p_eval.add_argument("--datasets", help="数据集标签（逗号分隔），默认根据任务类型自动选择")
    p_eval.add_argument("--no-audio-quality", action="store_true", help="禁用 MOS 质量评估")
    p_eval.add_argument("--device", default="cuda:0", help="指标、延迟和 badcase 使用的设备（ONNX 三路验证固定用 CPU 参考）")
    p_eval.add_argument(
        "--inference-devices",
        default=None,
        help="仅模型推理使用的设备列表，例如 cuda:0,cuda:1；默认使用 --device",
    )
    p_eval.add_argument("--aec-no-stream-align", action="store_true", help="AEC: 禁用流式延时对齐")
    p_eval.add_argument("--skip-existing-inference", action="store_true", help="跳过已有增强音频的文件（断点续传）")
    _add_trusted_checkpoint_arg(p_eval)
    _add_eval_data_source_args(p_eval)

    # report
    p_report = sub.add_parser("report", help="生成/更新报告")
    p_report.add_argument("exp_dir", help="实验目录路径")
    _add_checkpoint_arg(p_report)
    p_report.add_argument("--purpose", help="实验目的/描述（留空则自动推断）")
    p_report.add_argument("--skip-eval-check", action="store_true", help="即使无评估结果也更新报告")

    # run (全流程)
    p_run = sub.add_parser(
        "run",
        help="一键全流程: discover → evaluate → future dependency → ONNX publish → report → badcase",
    )
    p_run.add_argument("exp_dir", help="实验目录路径")
    _add_checkpoint_arg(p_run)
    p_run.add_argument("--purpose", help="实验目的/描述（留空则自动推断）")
    p_run.add_argument("--datasets", help="数据集标签（逗号分隔），默认根据任务类型自动选择")
    p_run.add_argument("--skip-eval", action="store_true", help="跳过评估步骤（仅更新报告）")
    p_run.add_argument("--no-audio-quality", action="store_true", help="禁用 MOS 质量评估")
    p_run.add_argument("--device", default="cuda:0", help="指标、延迟和 badcase 使用的设备（ONNX 三路验证固定用 CPU 参考）")
    p_run.add_argument(
        "--inference-devices",
        default=None,
        help="仅模型推理使用的设备列表，例如 cuda:0,cuda:1；默认使用 --device",
    )
    p_run.add_argument("--aec-no-stream-align", action="store_true", help="AEC: 禁用流式延时对齐")
    p_run.add_argument("--skip-existing-inference", action="store_true", help="跳过已有增强音频的文件（断点续传）")
    _add_trusted_checkpoint_arg(p_run)
    _add_eval_data_source_args(p_run)
    p_run.add_argument("--skip-latency-verify", action="store_true", help="跳过算法未来依赖验证")
    p_run.add_argument("--skip-onnx-verify", action="store_true", help="跳过 ONNX 三路精度验证")
    p_run.add_argument(
        "--skip-badcase", "--skip-self-recorded",
        action="store_true",
        dest="skip_badcase",
        help="跳过 badcase 快速推理测试",
    )
    p_run.add_argument(
        "--badcase-with-align",
        action="store_true",
        help="badcase 额外运行显式 --aec_stream_align 模式",
    )
    p_run.add_argument(
        "--skip-existing-badcase",
        action="store_true",
        help="跳过 badcase 中已存在增强音频的文件（断点续传）",
    )

    return p.parse_args()


def _dispatch(args: argparse.Namespace) -> int:
    """分发已解析的 CLI 子命令。"""
    if args.command == "discover":
        return cmd_discover(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "report":
        return cmd_report(args)
    return cmd_run(args)


def main() -> int:
    """CLI 异常边界：终端仅输出去敏摘要，不暴露 traceback。"""
    try:
        return _dispatch(parse_args())
    except Exception as err:
        message = _sanitize_free_text(str(err))
        print(f"{type(err).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
