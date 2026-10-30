"""离线物化 stereo echo/ref 的 windowed GCC 预对齐结果。

默认保持源采样率与容器格式不变（低置信文件 byte copy）；显式指定
``--output-sample-rate`` / ``--output-format`` / ``--output-subtype`` 时，
所有文件（含低置信）统一解码 → 对齐 → soxr VHQ 高保真重采样 → 重编码，
输出扩展名按目标格式替换。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import stat
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import soundfile as sf

try:
    import soxr
except ImportError:  # pragma: no cover - soxr 是 requirements 声明的重采样依赖
    soxr = None

from tasks.aec.inference.gcc_phat_core import (
    GccAlignEstimate,
    clamp_gcc_delay_samples,
    estimate_gcc_tau_windowed,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - 本工具仅支持 Linux 事务语义。
    fcntl = None

GENERATOR_ID = "tasks/aec/prealign_stereo_echo_ref.py"
GENERATOR_VERSION = "3.2"
RECORDS_NAME = "gcc_prealign_records.jsonl"
SUMMARY_NAME = "gcc_prealign_summary.json"
OWNER_FILE = ".prealign_stereo_echo_ref.owner.json"
OWNER_SCHEMA = 1


@dataclass(frozen=True)
class PrealignParams:
    """worker 所需的稳定参数快照。"""

    echo_channel: int
    farend_channel: int
    downsample: int | None
    max_tau_seconds: float
    peak_threshold: float
    prominence_db: float
    tv_iqr_ms: float
    max_windows: int
    bias_seconds: float
    max_pad_seconds: float
    min_pad_samples: int
    max_residual_error_ms: float
    on_residual_failure: str
    output_sample_rate: int | None
    output_format: str | None
    output_subtype: str | None
    overwrite: bool


@dataclass(frozen=True)
class WorkItem:
    """单个输入文件、owned staging 输出与扫描时身份。"""

    source: str
    target: str
    relative_path: str
    source_identity: tuple[int, int, int, int]


def clamp_tau_samples(
    tau_sec: float,
    sample_rate: int,
    *,
    bias_seconds: float,
    max_pad_seconds: float,
    min_pad_samples: int,
) -> int:
    """调用训练/离线/预物化共用的 bias/max-pad/min-pad helper。"""
    return clamp_gcc_delay_samples(
        tau_sec,
        sample_rate,
        bias_seconds=bias_seconds,
        max_delay_seconds=max_pad_seconds,
        min_delay_samples=min_pad_samples,
    )


def apply_constant_delay(farend: np.ndarray, delay_samples: int) -> np.ndarray:
    """farend 前补零后裁回原长度。"""
    delay = int(delay_samples)
    if delay <= 0:
        return np.asarray(farend, dtype=np.float32).copy()
    output = np.zeros_like(farend, dtype=np.float32)
    if delay < farend.size:
        output[delay:] = farend[:-delay]
    return output


def resample_audio(
    audio: np.ndarray, source_sr: int, target_sr: int
) -> np.ndarray:
    """高保真重采样（soxr VHQ，多声道沿 axis=0），返回 float32。

    soxr 缺失时 fail-closed：重采样质量直接影响物化语料内容，禁止静默
    降级到低质量实现。
    """
    if source_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    if soxr is None:
        raise RuntimeError(
            "high-fidelity resampling requires soxr (requirements.txt: soxr>=0.3)"
        )
    return soxr.resample(
        np.asarray(audio, dtype=np.float32),
        int(source_sr),
        int(target_sr),
        quality="VHQ",
    ).astype(np.float32, copy=False)


def _estimate_gcc(
    farend: np.ndarray,
    echo: np.ndarray,
    sample_rate: int,
    params: PrealignParams,
) -> GccAlignEstimate:
    """按生成和验收共用的参数调用 windowed GCC。"""
    return estimate_gcc_tau_windowed(
        farend,
        echo,
        sample_rate,
        downsample=params.downsample,
        max_tau_seconds=params.max_tau_seconds,
        peak_threshold=params.peak_threshold,
        prominence_db=params.prominence_db,
        tv_iqr_ms=params.tv_iqr_ms,
        max_windows=params.max_windows,
    )


def _initial_trajectory_point(estimate: GccAlignEstimate) -> tuple[float, float]:
    """返回按时间排序后的第一可靠 GCC 轨迹点。"""
    if not estimate.trajectory:
        raise RuntimeError("time-varying estimate has no reliable trajectory points")
    points = sorted((float(point[0]), float(point[1])) for point in estimate.trajectory)
    start_sec, tau_sec = points[0]
    if not np.isfinite(start_sec) or start_sec < 0.0 or not np.isfinite(tau_sec):
        raise RuntimeError("time-varying estimate has an invalid initial trajectory point")
    return start_sec, tau_sec


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _estimate_record(estimate: GccAlignEstimate) -> dict[str, Any]:
    return {
        "tau_seconds": float(estimate.tau_sec),
        "time_varying": bool(estimate.time_varying),
        "low_confidence": bool(estimate.low_confidence),
        "mode": str(estimate.mode),
        "peak": float(estimate.peak),
        "prominence": float(estimate.prominence),
        "trajectory": _json_value(estimate.trajectory),
    }


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    """返回足以发现替换/截断/覆写的文件身份。"""
    return (int(value.st_dev), int(value.st_ino), int(value.st_size), int(value.st_mtime_ns))


def _verify_source_identity(path: Path, expected: tuple[int, int, int, int]) -> None:
    current = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != expected:
        raise RuntimeError(f"input identity changed while processing: {path}")


def _audio_header(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": str(info.format),
        "subtype": str(info.subtype),
        "endian": str(info.endian),
    }


def _validate_staged_header(staged: Path, expected: dict[str, Any]) -> None:
    actual = _audio_header(staged)
    for key in ("sample_rate", "channels", "frames", "format", "subtype", "endian"):
        # 转换容器格式时 endian 由 soundfile 按目标格式决定（如 FLAC 恒为
        # FILE），expected 置 None 表示该校验项不适用。
        if expected.get(key) is None:
            continue
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"staged audio header mismatch for {staged}: {key} "
                f"expected {expected[key]!r}, got {actual[key]!r}"
            )


def _write_audio_staged(
    staged: Path,
    audio: np.ndarray,
    header: dict[str, Any],
) -> None:
    kwargs: dict[str, Any] = {
        "samplerate": header["sample_rate"],
        "channels": header["channels"],
        "format": header["format"],
        "subtype": header["subtype"],
    }
    if header.get("endian") is not None:
        kwargs["endian"] = header["endian"]
    with sf.SoundFile(str(staged), mode="w", **kwargs) as output:
        output.write(audio)
        output.flush()
    descriptor = os.open(staged, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_staged(source: Path, staged: Path) -> None:
    with source.open("rb") as input_file, staged.open("wb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
        output_file.flush()
        os.fsync(output_file.fileno())


def _snapshot_source(
    source: Path,
    target_parent: Path,
    expected_identity: tuple[int, int, int, int],
) -> Path:
    """复制一个稳定输入快照，并在读取前后复核源身份。"""
    _ensure_directory_without_symlinks(target_parent)
    descriptor, raw_path = tempfile.mkstemp(
        dir=target_parent, prefix=".source.", suffix=source.suffix
    )
    snapshot = Path(raw_path)
    source_fd = -1
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or _stat_identity(before) != expected_identity:
            raise RuntimeError(f"input identity changed before read: {source}")
        with os.fdopen(source_fd, "rb", closefd=True) as input_file, os.fdopen(
            descriptor, "wb", closefd=True
        ) as output_file:
            source_fd = -1
            descriptor = -1
            shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
            after = os.fstat(input_file.fileno())
        if _stat_identity(after) != expected_identity:
            raise RuntimeError(f"input identity changed during read: {source}")
        _verify_source_identity(source, expected_identity)
        return snapshot
    except BaseException:
        if source_fd >= 0:
            os.close(source_fd)
        if descriptor >= 0:
            os.close(descriptor)
        try:
            snapshot.unlink()
        except FileNotFoundError:
            pass
        raise


def _ensure_directory_without_symlinks(directory: Path) -> None:
    """逐层创建目录，并拒绝任一现有 symlink 或非目录节点。"""
    absolute = _absolute_lexical(directory)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            node_stat = os.lstat(cursor)
        except FileNotFoundError:
            try:
                os.mkdir(cursor, 0o755)
            except FileExistsError:
                node_stat = os.lstat(cursor)
            else:
                node_stat = os.lstat(cursor)
        if stat.S_ISLNK(node_stat.st_mode):
            raise RuntimeError(
                f"output path must not contain symbolic-link ancestors: {cursor}"
            )
        if not stat.S_ISDIR(node_stat.st_mode):
            raise NotADirectoryError(
                f"output path ancestor is not a directory: {cursor}"
            )


def _publish_audio(
    source: Path,
    target: Path,
    audio: np.ndarray | None,
    header: dict[str, Any],
    *,
    overwrite: bool,
    validate_content: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """把单文件写入 owned staging tree；最终 output 此时尚不可见。"""
    del overwrite
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"owned staging target unexpectedly exists: {target}")
    _ensure_directory_without_symlinks(target.parent)
    if audio is None:
        _copy_staged(source, target)
    else:
        _write_audio_staged(target, audio, header)
    _validate_staged_header(target, header)
    return validate_content(target) if validate_content is not None else None


def _unchecked_residual(reason: str) -> dict[str, Any]:
    return {"checked": False, "passed": False, "reason": reason, "scope": None}


class ResidualGateError(RuntimeError):
    """staged 残差校验未通过；携带校验结果供 copy 策略记录。

    ``abort``（默认）下与此前行为一致（整轮 fail-closed）；``copy`` 策略下
    由 ``_process_one`` 捕获并回退为不对齐保留样本。
    """

    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


def _process_one(item: WorkItem, params: PrealignParams) -> dict[str, Any]:
    original_source = Path(item.source)
    target = Path(item.target)
    source = _snapshot_source(original_source, target.parent, item.source_identity)
    audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    info = sf.info(str(source))
    header = _audio_header(source)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"invalid sample rate: {source}")
    if audio.shape[0] <= 0:
        raise ValueError(f"empty audio file: {source}")
    if audio.shape[1] != 2 or info.channels != 2:
        raise ValueError(f"expected exactly 2 channels, got {audio.shape[1]}: {source}")
    if not np.isfinite(audio).all():
        raise ValueError(f"audio contains NaN or Inf: {source}")
    for channel in (params.echo_channel, params.farend_channel):
        if channel < 0 or channel >= 2:
            raise ValueError(f"channel index must be 0 or 1, got {channel}")
    if params.echo_channel == params.farend_channel:
        raise ValueError("echo and farend channels must be different")

    echo = audio[:, params.echo_channel]
    farend = audio[:, params.farend_channel]
    estimate = _estimate_gcc(farend, echo, sample_rate, params)

    applied_tau_samples = 0
    output_audio: np.ndarray | None = None
    policy = "copy_low_confidence"
    residual_scope: str | None = None
    residual_start_sec = 0.0
    original_tau_sec = float(estimate.tau_sec)
    residual_check = _unchecked_residual("original_estimate_low_confidence")
    if not estimate.low_confidence:
        if estimate.time_varying:
            residual_start_sec, original_tau_sec = _initial_trajectory_point(estimate)
            policy = "apply_time_varying_initial_front_pad_crop"
            residual_scope = "initial_window"
        else:
            policy = "apply_constant_front_pad_crop"
            residual_scope = "full_pair"
        applied_tau_samples = clamp_tau_samples(
            original_tau_sec,
            sample_rate,
            bias_seconds=params.bias_seconds,
            max_pad_seconds=params.max_pad_seconds,
            min_pad_samples=params.min_pad_samples,
        )
        if applied_tau_samples > 0:
            output_audio = audio.copy()
            output_audio[:, params.farend_channel] = apply_constant_delay(
                farend, applied_tau_samples
            )
        else:
            policy = "copy_zero_applied_delay"
            reason = (
                "clamped_initial_tau_is_zero"
                if estimate.time_varying
                else "clamped_constant_tau_is_zero"
            )
            residual_check = _unchecked_residual(reason)
            residual_scope = None

    # ---- 可选的输出采样率 / 容器格式转换（默认完全透传源 header）----
    target_sample_rate = params.output_sample_rate or sample_rate
    target_format = params.output_format or header["format"]
    if params.output_format is not None:
        target_subtype = params.output_subtype or sf.default_subtype(target_format)
        # 目标容器（如 FLAC）没有源 endian 语义，交由 soundfile 按格式决定
        target_endian = None
    else:
        target_subtype = header["subtype"]
        target_endian = header["endian"]
    conversion_requested = (
        target_sample_rate != sample_rate or params.output_format is not None
    )
    if output_audio is None and conversion_requested:
        # copy 策略（低置信 / 钳制为零）在转换模式下同样重采样重编码
        output_audio = audio
    if output_audio is not None:
        output_audio = resample_audio(output_audio, sample_rate, target_sample_rate)
        target_header = {
            "sample_rate": target_sample_rate,
            "channels": header["channels"],
            "frames": int(output_audio.shape[0]),
            "format": target_format,
            "subtype": target_subtype,
            "endian": target_endian,
        }
    else:
        target_header = header

    validate_content: Callable[[Path], dict[str, Any]] | None = None
    if output_audio is not None and residual_scope is not None:
        # expected_tau_ms 是时间量，applied_tau_samples 按源采样率记账，与重采样无关
        expected_tau_ms = max(
            0.0,
            original_tau_sec * 1000.0
            - applied_tau_samples * 1000.0 / float(sample_rate),
        )

        def validate_content(staged: Path) -> dict[str, Any]:
            staged_audio, staged_sr = sf.read(
                str(staged), dtype="float32", always_2d=True
            )
            if int(staged_sr) != target_sample_rate:
                raise RuntimeError(
                    f"staged residual sample rate mismatch: {staged_sr} != {target_sample_rate}"
                )
            residual_farend = staged_audio[:, params.farend_channel]
            residual_echo = staged_audio[:, params.echo_channel]
            if residual_scope == "initial_window":
                start = int(round(residual_start_sec * target_sample_rate))
                stop = min(staged_audio.shape[0], start + target_sample_rate)
                if start < 0 or stop - start < 8:
                    raise ResidualGateError(
                        "initial residual window is unavailable in staged audio",
                        {
                            "checked": True,
                            "passed": False,
                            "scope": residual_scope,
                            "reason": "initial_window_unavailable",
                        },
                    )
                residual_farend = residual_farend[start:stop]
                residual_echo = residual_echo[start:stop]
            residual_estimate = _estimate_gcc(
                residual_farend, residual_echo, target_sample_rate, params
            )
            if residual_estimate.low_confidence:
                raise ResidualGateError(
                    f"{residual_scope} staged residual estimate is low-confidence",
                    {
                        "checked": True,
                        "passed": False,
                        "scope": residual_scope,
                        "reason": "residual_low_confidence",
                        "low_confidence": True,
                    },
                )
            if residual_scope == "full_pair" and residual_estimate.time_varying:
                raise ResidualGateError(
                    "full-pair staged residual estimate is time-varying",
                    {
                        "checked": True,
                        "passed": False,
                        "scope": residual_scope,
                        "reason": "residual_time_varying",
                        "time_varying": True,
                    },
                )
            if residual_estimate.time_varying:
                _, residual_tau_sec = _initial_trajectory_point(residual_estimate)
            else:
                residual_tau_sec = float(residual_estimate.tau_sec)
            estimated_tau_ms = residual_tau_sec * 1000.0
            error_ms = abs(estimated_tau_ms - expected_tau_ms)
            result = {
                "checked": True,
                "scope": residual_scope,
                "expected_tau_ms": float(expected_tau_ms),
                "estimated_tau_ms": float(estimated_tau_ms),
                "error_ms": float(error_ms),
                "low_confidence": bool(residual_estimate.low_confidence),
                "time_varying": bool(residual_estimate.time_varying),
                "passed": bool(error_ms <= params.max_residual_error_ms),
            }
            if not result["passed"]:
                raise ResidualGateError(
                    f"{residual_scope} staged residual error {error_ms:.3f} ms exceeds "
                    f"{params.max_residual_error_ms:.3f} ms",
                    result,
                )
            return result

    try:
        validation = _publish_audio(
            source,
            target,
            output_audio,
            target_header,
            overwrite=params.overwrite,
            validate_content=validate_content,
        )
    except ResidualGateError as error:
        if params.on_residual_failure != "copy":
            raise
        # copy 回退：已写入的对齐结果无法验证，改写为不对齐（仅转换）的音频。
        # 与低置信同一哲学——样本保留，不施加无法验证的对齐。
        target.unlink()
        fallback_audio = (
            resample_audio(audio, sample_rate, target_sample_rate)
            if conversion_requested
            else None
        )
        _publish_audio(
            source,
            target,
            fallback_audio,
            target_header,
            overwrite=params.overwrite,
            validate_content=None,
        )
        residual_check = error.result
        applied_tau_samples = 0
        policy = "copy_residual_gate_failure"
    else:
        if validation is not None:
            residual_check = validation
    # low-confidence copy、GCC 估计与发布均绑定 snapshot 的同一 bytes；发布前再
    # 复核原路径，禁止输入在处理期间被替换后仍生成“成功”集合。
    _verify_source_identity(original_source, item.source_identity)
    source.unlink()
    return {
        "generator_version": GENERATOR_VERSION,
        "relative_path": item.relative_path,
        "sample_rate": header["sample_rate"],
        "frames": header["frames"],
        "channels": header["channels"],
        "format": header["format"],
        "subtype": header["subtype"],
        "endian": header["endian"],
        "output": {
            "sample_rate": int(target_header["sample_rate"]),
            "frames": int(target_header["frames"]),
            "format": str(target_header["format"]),
            "subtype": str(target_header["subtype"]),
        },
        "estimate": _estimate_record(estimate),
        "applied": {
            "tau_samples": int(applied_tau_samples),
            "tau_seconds": float(applied_tau_samples / sample_rate),
            "trajectory": None,
        },
        "policy": policy,
        "residual_check": residual_check,
        "status": "success",
    }


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_ancestors(output_root: Path) -> None:
    """只读校验现有输出祖先；缺失部分留待正式运行时创建。"""
    absolute = _absolute_lexical(output_root)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            node_stat = os.lstat(cursor)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(node_stat.st_mode):
            raise RuntimeError(
                f"output path must not contain symbolic-link ancestors: {cursor}"
            )
        if not stat.S_ISDIR(node_stat.st_mode):
            raise NotADirectoryError(
                f"output path ancestor is not a directory: {cursor}"
            )


def _validate_root_paths(input_root: Path, output_root: Path) -> tuple[Path, Path]:
    input_abs = _absolute_lexical(input_root)
    output_abs = _absolute_lexical(output_root)
    if not input_abs.exists() or not input_abs.is_dir():
        raise NotADirectoryError(f"input root is not a directory: {input_abs}")
    _validate_output_ancestors(output_abs)
    input_real = input_abs.resolve(strict=True)
    output_real = output_abs.resolve(strict=False)
    if (
        input_real == output_real
        or _is_relative_to(input_real, output_real)
        or _is_relative_to(output_real, input_real)
    ):
        raise ValueError("input and output roots must be different and non-overlapping")
    return input_abs, output_abs


def _scan_inputs(
    input_root: Path,
    extensions: Sequence[str],
    limit: int | None,
) -> list[Path]:
    normalized = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }
    files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in normalized
    ]
    files.sort(key=lambda path: path.relative_to(input_root).as_posix())
    if limit is not None:
        files = files[:limit]
    return files


def _json_dump(value: Any, output: Any, *, lines: bool = False) -> None:
    if lines:
        for record in value:
            json.dump(record, output, ensure_ascii=False, sort_keys=True, allow_nan=False)
            output.write("\n")
    else:
        json.dump(
            value,
            output,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output.write("\n")


def _owner_payload() -> dict[str, object]:
    return {"generator": GENERATOR_ID, "schema_version": OWNER_SCHEMA}


def _write_owner(root: Path) -> None:
    path = root / OWNER_FILE
    with path.open("x", encoding="utf-8") as output:
        _json_dump(_owner_payload(), output)
        output.flush()
        os.fsync(output.fileno())


def _is_owned(root: Path) -> bool:
    try:
        root_stat = root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            return False
        owner_path = root / OWNER_FILE
        try:
            owner_stat = owner_path.lstat()
        except FileNotFoundError:
            summary_path = root / SUMMARY_NAME
            if summary_path.is_symlink() or not summary_path.is_file():
                return False
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return (
                summary.get("generator_version") == GENERATOR_VERSION
                and summary.get("status") == "complete"
            )
        if not stat.S_ISREG(owner_stat.st_mode):
            return False
        return json.loads(owner_path.read_text(encoding="utf-8")) == _owner_payload()
    except (OSError, json.JSONDecodeError):
        return False


def _remove_owned_tree(path: Path) -> None:
    if not _is_owned(path):
        raise RuntimeError(f"refusing to remove unowned prealign directory: {path}")
    shutil.rmtree(path)


def _publish_sidecars(
    output_root: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """在 owned staging tree 内写完并 fsync 两个 sidecar。"""
    del overwrite
    for path, value, lines in (
        (output_root / RECORDS_NAME, records, True),
        (output_root / SUMMARY_NAME, summary, False),
    ):
        with path.open("x", encoding="utf-8") as output:
            _json_dump(value, output, lines=lines)
            output.flush()
            os.fsync(output.fileno())


@contextmanager
def _output_lock(output_root: Path) -> Iterator[None]:
    """用 output sibling lock 串行化 owned staging 与最终发布。"""
    if fcntl is None:
        raise RuntimeError("prealign transactional publishing requires fcntl")
    _ensure_directory_without_symlinks(output_root.parent)
    lock_path = output_root.parent / f".{output_root.name}.prealign.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_staging_tree(root: Path) -> None:
    """自底向上 fsync owned staging 的全部目录项。"""
    directories = [Path(directory) for directory, _, _ in os.walk(root)]
    for directory in reversed(directories):
        _fsync_directory(directory)


def _exchange_directories(first: Path, second: Path) -> None:
    """用 Linux renameat2(RENAME_EXCHANGE) 原子交换两个完整目录集合。"""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic overwrite requires Linux renameat2")
    result = renameat2(
        -100,
        os.fsencode(first),
        -100,
        os.fsencode(second),
        2,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{first} <-> {second}")


def _publish_staging_tree(staging_root: Path, output_root: Path, *, overwrite: bool) -> None:
    """一次原子提交完整音频+sidecar 集合，失败恢复旧集合。"""
    if not _is_owned(staging_root):
        raise RuntimeError(f"staging root is not owned: {staging_root}")
    output_stat = None
    try:
        output_stat = output_root.lstat()
    except FileNotFoundError:
        pass
    if output_stat is not None and not stat.S_ISDIR(output_stat.st_mode):
        raise RuntimeError(f"output root must be a real directory: {output_root}")
    if output_stat is not None and not overwrite:
        raise FileExistsError(
            f"output root already exists; pass --overwrite to replace: {output_root}"
        )
    if output_stat is not None and not _is_owned(output_root):
        raise RuntimeError(f"refusing to overwrite unowned output root: {output_root}")
    _fsync_staging_tree(staging_root)
    if output_stat is None:
        os.rename(staging_root, output_root)
        try:
            _fsync_directory(output_root.parent)
        except BaseException:
            os.rename(output_root, staging_root)
            _fsync_directory(output_root.parent)
            raise
        return

    _exchange_directories(staging_root, output_root)
    try:
        _fsync_directory(output_root.parent)
    except BaseException:
        _exchange_directories(staging_root, output_root)
        _fsync_directory(output_root.parent)
        raise
    # 首次 parent fsync 成功即完成 commit。此后 staging_root 指向旧 owned 集合；
    # 清理失败不能把已提交的新集合误报为发布失败，只保留旧目录并告警。
    try:
        _remove_owned_tree(staging_root)
        _fsync_directory(output_root.parent)
    except (OSError, RuntimeError) as error:
        print(
            f"WARNING: committed output but failed to remove old collection "
            f"{staging_root}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-align stereo AEC echo/ref files with offline windowed GCC-PHAT."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extensions", nargs="+", default=[".wav", ".flac"])
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
        help="并行进程数（默认 min(8, CPU 数)）",
    )
    parser.add_argument("--echo-channel", type=int, default=0)
    parser.add_argument("--farend-channel", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-tau-seconds",
        type=float,
        default=0.3,
        help="GCC 搜索上限秒数（默认 0.3，与当前 AEC 正式配置一致）",
    )
    parser.add_argument("--downsample", type=int)
    parser.add_argument("--peak-threshold", type=float, default=0.05)
    parser.add_argument("--prominence-db", type=float, default=6.0)
    parser.add_argument("--tv-iqr-ms", type=float, default=10.0)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--bias-seconds", type=float, default=0.001)
    parser.add_argument(
        "--max-pad-seconds",
        type=float,
        default=0.3,
        help="farend 最大补零秒数（默认 0.3，与当前 AEC 正式配置一致）",
    )
    parser.add_argument("--min-pad-samples", type=int, default=0)
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        help="输出采样率（Hz，如 16000/32000）；缺省保持源采样率不重采样",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        help="输出容器格式（soundfile 格式名，如 FLAC/WAV）；缺省保持源格式。"
        "指定后输出扩展名替换为 .{format}（如 .flac），低置信文件也由 byte copy "
        "改为重编码",
    )
    parser.add_argument(
        "--output-subtype",
        type=str,
        help="输出编码 subtype（如 PCM_16/PCM_24）；需搭配 --output-format，"
        "缺省用 soundfile 对该格式的默认 subtype",
    )
    parser.add_argument(
        "--max-residual-error-ms",
        type=float,
        default=5.0,
        help="staged 编码后 GCC residual 最大允许误差（ms，默认 5.0）",
    )
    parser.add_argument(
        "--on-residual-failure",
        choices=["abort", "copy"],
        default="copy",
        help="staged 残差校验未通过时的策略：copy（默认）该文件回退为不对齐、"
        "原样保留（与低置信文件同一哲学），records 标记 copy_residual_gate_failure "
        "并继续；abort 整轮 fail-closed 中止",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    finite_values = {
        "--max-tau-seconds": args.max_tau_seconds,
        "--peak-threshold": args.peak_threshold,
        "--prominence-db": args.prominence_db,
        "--tv-iqr-ms": args.tv_iqr_ms,
        "--bias-seconds": args.bias_seconds,
        "--max-pad-seconds": args.max_pad_seconds,
        "--max-residual-error-ms": args.max_residual_error_ms,
    }
    invalid = [name for name, value in finite_values.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError("arguments must be finite: " + ", ".join(invalid))
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.downsample is not None and args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    if (
        args.max_tau_seconds < 0
        or args.max_pad_seconds < 0
        or args.max_residual_error_ms < 0
    ):
        raise ValueError("tau, pad, and residual limits must be non-negative")
    if not (0.0 < args.peak_threshold <= 1.0):
        raise ValueError("--peak-threshold must be in (0.0, 1.0]")
    if args.prominence_db < 0.0:
        raise ValueError("--prominence-db must be >= 0")
    if args.tv_iqr_ms <= 0.0:
        raise ValueError("--tv-iqr-ms must be > 0")
    if args.max_windows < 1:
        raise ValueError("--max-windows must be >= 1")
    if args.min_pad_samples < 0:
        raise ValueError("--min-pad-samples must be >= 0")
    if args.output_sample_rate is not None and not (
        1000 <= args.output_sample_rate <= 768000
    ):
        raise ValueError("--output-sample-rate must be in [1000, 768000]")
    if args.output_format is not None:
        args.output_format = args.output_format.upper()
        if args.output_format not in sf.available_formats():
            raise ValueError(f"unknown --output-format: {args.output_format!r}")
    if args.output_subtype is not None:
        if args.output_format is None:
            raise ValueError("--output-subtype requires --output-format")
        args.output_subtype = args.output_subtype.upper()
        if not sf.check_format(args.output_format, args.output_subtype):
            raise ValueError(
                f"subtype {args.output_subtype!r} is not valid for "
                f"format {args.output_format!r}"
            )
    if args.echo_channel == args.farend_channel:
        raise ValueError("echo and farend channels must be different")
    if {args.echo_channel, args.farend_channel} != {0, 1}:
        raise ValueError("stereo channel indices must be 0 and 1")


def _run_staged(
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    files: list[Path],
    staging_root: Path,
) -> int:
    """在 owned staging tree 内生成完整集合，成功后一次提交。"""
    params = PrealignParams(
        echo_channel=args.echo_channel,
        farend_channel=args.farend_channel,
        downsample=args.downsample,
        max_tau_seconds=args.max_tau_seconds,
        peak_threshold=args.peak_threshold,
        prominence_db=args.prominence_db,
        tv_iqr_ms=args.tv_iqr_ms,
        max_windows=args.max_windows,
        bias_seconds=args.bias_seconds,
        max_pad_seconds=args.max_pad_seconds,
        min_pad_samples=args.min_pad_samples,
        max_residual_error_ms=args.max_residual_error_ms,
        on_residual_failure=args.on_residual_failure,
        output_sample_rate=args.output_sample_rate,
        output_format=args.output_format,
        output_subtype=args.output_subtype,
        overwrite=args.overwrite,
    )
    # 指定输出容器格式时，输出相对路径的扩展名替换为该格式的规范后缀
    output_suffix = f".{args.output_format.lower()}" if args.output_format else None

    def _target_relative(source: Path) -> Path:
        relative = source.relative_to(input_root)
        if output_suffix is not None:
            relative = relative.with_suffix(output_suffix)
        return relative

    items = [
        WorkItem(
            source=str(source),
            target=str(staging_root / _target_relative(source)),
            relative_path=_target_relative(source).as_posix(),
            source_identity=_stat_identity(source.stat(follow_symlinks=False)),
        )
        for source in files
    ]
    targets = [item.target for item in items]
    if len(set(targets)) != len(targets):
        raise ValueError(
            "output extension replacement collapses distinct inputs "
            "onto the same target path"
        )
    print(
        f"starting: files={len(items)} workers={args.workers} "
        f"input={input_root} output={output_root}"
    )

    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    if args.workers == 1:
        for item in items:
            try:
                records.append(_process_one(item, params))
            except Exception as error:
                failures.append((item.relative_path, f"{type(error).__name__}: {error}"))
                break
            if len(records) % 1000 == 0 or len(records) == len(items):
                print(f"progress: {len(records)}/{len(items)}")
    else:
        # 只保留 2×workers 个 in-flight future，避免全量语料一次性提交造成内存膨胀。
        item_iter = iter(items)
        in_flight: dict[Any, WorkItem] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(min(len(items), 2 * args.workers)):
                item = next(item_iter)
                in_flight[executor.submit(_process_one, item, params)] = item
            while in_flight and not failures:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    item = in_flight.pop(future)
                    try:
                        records.append(future.result())
                    except Exception as error:
                        failures.append(
                            (item.relative_path, f"{type(error).__name__}: {error}")
                        )
                        break
                    try:
                        next_item = next(item_iter)
                    except StopIteration:
                        pass
                    else:
                        in_flight[
                            executor.submit(_process_one, next_item, params)
                        ] = next_item
                if len(records) % 1000 == 0 or len(records) == len(items):
                    print(f"progress: {len(records)}/{len(items)}")
            if failures:
                for future in in_flight:
                    future.cancel()

    if failures:
        for relative_path, message in sorted(failures):
            print(f"ERROR {relative_path}: {message}", file=sys.stderr)
        return 1

    records.sort(key=lambda record: record["relative_path"])
    parameter_record = asdict(params)
    parameter_record.update(
        {
            "extensions": list(args.extensions),
            "workers": int(args.workers),
            "limit": args.limit,
            "overwrite": bool(args.overwrite),
        }
    )
    counts: dict[str, int] = {}
    for record in records:
        policy = record["policy"]
        counts[policy] = counts.get(policy, 0) + 1
    checked = [record["residual_check"] for record in records if record["residual_check"]["checked"]]
    # 早期校验失败（residual_low_confidence 等）没有 error_ms，只统计完整误差记录
    residual_errors = np.asarray(
        [check["error_ms"] for check in checked if "error_ms" in check],
        dtype=np.float64,
    )
    residual_error_stats = {"max": None, "mean": None, "p50": None, "p95": None}
    if residual_errors.size:
        residual_error_stats = {
            "max": float(np.max(residual_errors)),
            "mean": float(np.mean(residual_errors)),
            "p50": float(np.percentile(residual_errors, 50.0)),
            "p95": float(np.percentile(residual_errors, 95.0)),
        }
    summary = {
        "generator_version": GENERATOR_VERSION,
        "status": "complete",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_file_count": len(items),
        "success_count": len(records),
        "parameters": parameter_record,
        "policy_counts": counts,
        "residual_checked_count": len(checked),
        "residual_passed_count": sum(bool(check["passed"]) for check in checked),
        "residual_error_ms": residual_error_stats,
        "records_file": RECORDS_NAME,
    }
    _publish_sidecars(staging_root, records, summary, overwrite=False)
    _publish_staging_tree(staging_root, output_root, overwrite=args.overwrite)
    print(f"completed: {len(records)} files -> {output_root}")
    return 0


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    input_root, output_root = _validate_root_paths(args.input_root, args.output_root)
    files = _scan_inputs(input_root, args.extensions, args.limit)
    if not files:
        raise RuntimeError("no matching audio files found")
    if args.dry_run:
        print(f"dry-run: found {len(files)} files under {input_root}")
        return 0

    with _output_lock(output_root):
        if output_root.exists() and not args.overwrite:
            raise FileExistsError(
                f"output root already exists; pass --overwrite to replace: {output_root}"
            )
        staging_root = Path(
            tempfile.mkdtemp(
                dir=output_root.parent,
                prefix=f".{output_root.name}.owned-staging.",
            )
        )
        _write_owner(staging_root)
        try:
            result = _run_staged(
                args, input_root, output_root, files, staging_root
            )
            if result != 0 and (staging_root.exists() or staging_root.is_symlink()):
                _remove_owned_tree(staging_root)
            return result
        except BaseException:
            if staging_root.exists() or staging_root.is_symlink():
                _remove_owned_tree(staging_root)
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
