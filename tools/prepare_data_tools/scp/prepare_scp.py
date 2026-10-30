#!/usr/bin/env python3
"""从音频目录生成训练用严格四列 SCP 索引。

================================================================================
使用说明（概要）
================================================================================

**输出**
    默认 SCP 每行四列：``uid 采样率 采样点数 manifest-relative-path``，与 YAML 里
    ``dynamic.*_scp`` 字段常见写法一致；读取端仍兼容历史完整绝对路径。

**模式 A：单目录**
    把 ``<音频根目录>`` 扫成一套索引；``-o`` 是**文件前缀**（无扩展名），例如::

        python prepare_scp.py /data/speech -o data/src/speech_sources
        # -> data/src/speech_sources.scp

    默认递归子目录；只要当前层用 ``--no_recursive``。

**模式 B：批处理**
    出现 ``--speech`` / ``--noise`` / ``--rir`` 等任一即进入批处理；``-o`` 是**输出目录**。
    每种资源写到固定文件名（与配置里字段名对应）。默认按 ``--val_ratio``（默认 0.05）
    随机划分出 train/val 两份；``--val_ratio 0`` 关闭划分、只写合并文件，例如::

        speech          -> <out>/train_speech_sources.scp + val_speech_sources.scp   （划分时）
                           <out>/speech_sources.scp                                  （--val_ratio 0 时）
        noise           -> <out>/train_noise_sources.scp + val_noise_sources.scp
        rir             -> <out>/train_rirs.scp + val_rirs.scp
        wind            -> <out>/train_wind_noise_sources.scp + val_wind_noise_sources.scp
        music           -> <out>/train_music_sources.scp + val_music_sources.scp     （TSE music 池）

    每个前缀固定输出 ``<prefix>.scp``（四列）。本 CLI 不产出 JSONL——
    分类噪声 JSONL（逐行 object）由 ``generators/noise/build_stage_d_noise_manifests.py`` 发布。

    可只给 ``--speech`` 不写 noise/rir/wind；同类型多个路径会合并进同一个 scp。

**划分验证集**
    * 目录已分开（仅 speech 支持显式划分）：``--speech_train`` + ``--speech_val``，
      此时 speech 不再随机划分，其余资源仍按 ``--val_ratio`` 随机划分。
    * 随机划分：所有资源共用 ``--val_ratio``（默认 0.05）与 ``--seed``（默认 42），
      每个资源独立 shuffle，结果完全可复现；``--val_ratio 0`` 表示不划分。
    * 验证集消费方式：``data.validation`` 块（``inherit_simulation: true`` 时）
      可覆盖 ``speech_scp / noise_scp / rir_scp / windnoise_scp``，指向对应的
      ``val_*`` 文件即可（见脚本末尾打印的 YAML snippet）。

**RIR**
    目录里可以是 **波形**（wav/flac/mp3/…，与语音同一套后缀）或 **``.pkl``**
   （numpy/字典，受限反序列化）。仅可信数据可在环境中设
    ``PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1`` 回退完整 pickle。

**其它常用参数**
    ``--sample_rate`` 强制写死采样率（仍读长度；与真实采样率不一致时默认计入
    validation summary 的 ``fs_mismatch``，加 ``--strict_sample_rate`` 改为直接报错）；
    ``--workers`` 并行 ``sf.info``（默认 8，1=串行）；
    ``--analyze_sr`` 扫本轮输入根目录并统计采样率（路径去重）；详见 ``--help``。

**失败行为**
    目录不存在、目录下无音频、``.pkl`` 无有效 RIR、校验后零条目等会**直接报错退出**，
    不写残缺 SCP。校验开启（``--validate_audio``，默认关闭）时单条坏音频（读失败/静音/越界等）为**跳过 + warning**，
    不会中止整个扫描。已移除的 deprecated flag（``--no_validate_audio``）在解析期直接报错，
    不再静默 no-op。

================================================================================
English summary
================================================================================
Build index files from audio roots. Default SCP: ``uid fs num_samples manifest-relative-path``;
legacy absolute paths remain readable. Single-dir: ``<dir> -o <prefix>``. Batch:
``--speech`` / ``--noise`` / … and ``-o <dir>``. Each fixed prefix writes one four-column
``.scp``. No ``.jsonl`` is emitted here — categorized noise JSONL comes from
``generators/noise/build_stage_d_noise_manifests.py``.
RIR: waveform files or restricted ``.pkl``. See ``--help`` for full flags and epilog examples.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.prepare_data_tools._common import setup_logging  # noqa: E402
from tools.prepare_data_tools.scp._analyze_sr import analyze_audio_sr  # noqa: E402
from tools.shared.audio_io import load_audio  # noqa: E402
from tools.shared.pickle_io import load_pkl_file as _load_pkl_file  # noqa: E402
from tools.shared.scp_io import (  # noqa: E402
    serialize_manifest_relative_path,
    serialize_scp_entries,
)


# -----------------------------------------------------------------------------
# Logging & constants
# -----------------------------------------------------------------------------

logger = setup_logging("prepare_scp")

# Index scan: speech / noise / wind / RIR-as-waveform share this set; RIR may also be .pkl (see below).
AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".sph",
    ".wv",
    ".aif",
    ".aiff",
    ".aifc",
    ".pkl",
}

Entry = tuple[str, int, int, str]

# 默认验证集随机划分比例与种子（--val_ratio 0 关闭划分）
DEFAULT_VAL_RATIO = 0.05
DEFAULT_SEED = 42

# 音频校验默认阈值
DEFAULT_MAX_PEAK = 1.05
DEFAULT_MIN_MEAN_ABS = 1e-6
DEFAULT_DC_OFFSET_THRESHOLD = 0.5

# 采样率分析报告中错误文件列表截断条数
ANALYZE_ERROR_LIST_LIMIT = 200

BATCH_KEY_TO_PREFIX = {
    "speech": "speech_sources",
    "speech_train": "train_speech_sources",
    "speech_val": "val_speech_sources",
    "noise": "noise_sources",
    "rir": "rirs",
    "wind": "wind_noise_sources",
    "music": "music_sources",
}

CLI_EPILOG = r"""
Examples:
  python prepare_scp.py /data/speech -o data/train_sources/speech_sources

  # Batch: only speech -> train/val SCPs by default; use --val_ratio 0 for one file
  python prepare_scp.py --speech /data/speech -o data/train_sources

  python prepare_scp.py --speech /data/s1 /data/s2 --noise /data/n1 -o data/train_sources

  python prepare_scp.py --speech /data/sp --noise /data/n --rir /data/r --wind /data/w \\
      -o data/train_sources

  python prepare_scp.py --speech_train /data/tr --speech_val /data/va \\
      --noise /data/n --rir /data/r --wind /data/w -o data/sources

  python prepare_scp.py --speech /data/sp --noise /data/n --rir /data/r --wind /data/w \\
      -o data/sources --val_ratio 0.05 --seed 42 --analyze_sr

  python prepare_scp.py --speech /data/sp --noise /data/n -o data/sources --validate_audio
"""


class PrepareScpError(RuntimeError):
    """Fatal indexing error (abort run, no partial SCP)."""


@dataclass
class ProbeOptions:
    """音频探测/校验选项组：收敛 validate/阈值/stats/allow_pkl 的层层透传。"""

    validate: bool = False
    allow_pkl: bool = True
    max_peak: float = DEFAULT_MAX_PEAK
    min_mean_abs: float = DEFAULT_MIN_MEAN_ABS
    dc_offset_threshold: float = DEFAULT_DC_OFFSET_THRESHOLD
    stats: ValidationStats | None = None
    # --sample_rate 声明值与真实采样率不一致时直接报错（默认仅计数 + debug 日志）
    strict_sample_rate: bool = False

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        allow_pkl: bool = True,
        stats: ValidationStats | None = None,
    ) -> "ProbeOptions":
        """从 argparse 结果构造；共享 ``stats`` 汇总本轮全部资源。"""
        return cls(
            validate=bool(args.validate_audio),
            allow_pkl=allow_pkl,
            max_peak=args.validate_max_peak,
            min_mean_abs=args.validate_min_mean_abs,
            dc_offset_threshold=args.validate_dc_offset_threshold,
            stats=stats,
            strict_sample_rate=getattr(args, "strict_sample_rate", False),
        )


# -----------------------------------------------------------------------------
# Audio quality validation
# -----------------------------------------------------------------------------

class ValidationStats:
    """Simple counters for audio validation results (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.ok = 0
        self.zero_length = 0
        self.nan_inf = 0
        self.too_silent = 0
        self.peak_overflow = 0
        self.dc_offset = 0
        self.read_error = 0
        self.fs_mismatch = 0

    def inc(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def summary(self) -> str:
        skipped = self.total - self.ok
        parts = [f"total={self.total}, kept={self.ok}, skipped={skipped}"]
        if self.zero_length:
            parts.append(f"zero_length={self.zero_length}")
        if self.nan_inf:
            parts.append(f"nan_inf={self.nan_inf}")
        if self.too_silent:
            parts.append(f"too_silent={self.too_silent}")
        if self.peak_overflow:
            parts.append(f"peak_overflow={self.peak_overflow}")
        if self.dc_offset:
            parts.append(f"dc_offset={self.dc_offset}")
        if self.read_error:
            parts.append(f"read_error={self.read_error}")
        if self.fs_mismatch:
            parts.append(f"fs_mismatch={self.fs_mismatch}")
        return "; ".join(parts)


def validate_audio_file(
    path: str,
    max_peak: float = 1.05,
    min_mean_abs: float = 1e-6,
    dc_offset_threshold: float = 0.5,
) -> tuple[bool, str]:
    """Return (is_valid, reason) after fully decoding the file.

    Checks: readable, non-empty, finite, peak, mean absolute energy, DC offset.
    """
    try:
        load_audio(path, max_peak=max_peak, min_mean_abs=min_mean_abs, dc_offset_threshold=dc_offset_threshold)
        return True, ""
    except ValueError as e:
        # 结构化异常优先用 reason 归类；旧版消息（或第三方抛出的 ValueError）回退子串匹配
        reason = getattr(e, "reason", None)
        if reason:
            return False, reason
        msg = str(e)
        if "为空" in msg or "Zero-length" in msg or "empty" in msg.lower():
            return False, "zero_length"
        if "NaN/Inf" in msg or "nan" in msg.lower():
            return False, "nan_inf"
        if "过静音" in msg or "silence" in msg.lower() or "mean_abs" in msg:
            return False, "too_silent"
        if "幅度越界" in msg or "peak" in msg.lower():
            return False, "peak_overflow"
        if "DC offset" in msg or "dc_offset" in msg.lower():
            return False, "dc_offset"
        return False, "read_error"
    except Exception:
        return False, "read_error"


# -----------------------------------------------------------------------------
# Restricted pickle (RIR .pkl)
# -----------------------------------------------------------------------------

# 受限反序列化实现已提取到 tools.shared.pickle_io（load_pkl_file，上方已导入为
# _load_pkl_file），与 generate_simulated_rirs.py / generate_simulated_farend_echo_pairs.py 共用；
# 环境变量 PREPARE_SCP_ALLOW_UNSAFE_PICKLE 语义不变。


# -----------------------------------------------------------------------------
# RIR from pickle (.pkl only)
# -----------------------------------------------------------------------------

def _probe_pkl_file(path: str, default_sr: int = 16000) -> tuple[int, int, int | None]:
    """返回 ``(fallback_or_metadata_fs, num_samples, metadata_fs)``。

    ``metadata_fs`` 为 ``None`` 表示 pickle 没有 fs metadata；此时保留历史
    16000 Hz fallback（或调用方传入的 ``default_sr``），因此不能把 fallback
    当作真实采样率参与 ``--sample_rate`` mismatch 比较。
    """
    data = _load_pkl_file(
        path, purpose="probe RIR metadata while preparing SCP index"
    )
    metadata_fs: int | None = None

    if isinstance(data, np.ndarray):
        rir = data
        fs = default_sr
    elif isinstance(data, dict):
        rir = None
        for key in ("rir", "impulse_response", "h", "source_rir", "data", "ir", "signal", "audio"):
            v = data.get(key)
            if v is not None and isinstance(v, np.ndarray) and getattr(v, "ndim", 0) >= 1:
                rir = v
                break
        if rir is None:
            for v in data.values():
                if isinstance(v, np.ndarray) and getattr(v, "ndim", 0) >= 1:
                    rir = v
                    break
        if rir is None:
            raise PrepareScpError(f"No RIR array in pickle (keys: {list(data.keys())}): {path}")
        for key in ("fs", "sample_rate", "sr"):
            if key in data and data[key] is not None:
                try:
                    metadata_fs = int(data[key])
                except (TypeError, ValueError) as exc:
                    raise PrepareScpError(
                        f"Invalid sample-rate metadata {data[key]!r} in pickle: {path}"
                    ) from exc
                if metadata_fs <= 0:
                    raise PrepareScpError(
                        f"Sample-rate metadata must be positive in pickle: {path}"
                    )
                break
        fs = metadata_fs if metadata_fs is not None else default_sr
    else:
        raise PrepareScpError(f"Unsupported pickle root type {type(data)}: {path}")

    if not isinstance(rir, np.ndarray):
        raise PrepareScpError(f"RIR is not ndarray: {path}")
    if rir.ndim > 1:
        rir = rir[:, 0] if rir.shape[1] < rir.shape[0] else rir[0, :]
    return int(fs), len(rir), metadata_fs


# -----------------------------------------------------------------------------
# Directory scan & UID
# -----------------------------------------------------------------------------

def scan_audio_dir(
    root: Path,
    recursive: bool = True,
    exts: set[str] | None = None,
    followlinks: bool = False,
) -> list[Path]:
    """扫描目录下的音频文件；``exts`` 默认 ``AUDIO_EXTS``。

    非 RIR 资源（speech/noise/wind/extra）传入去掉 ``.pkl`` 的扩展名集合，
    避免把混入的 pickle 当波形/RIR 索引。``followlinks=True`` 时递归进入
    软连接目录（os.walk 默认跳过）。
    """
    exts = exts or AUDIO_EXTS
    files: list[Path] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root, followlinks=followlinks):
            for fn in filenames:
                if Path(fn).suffix.lower() in exts:
                    files.append(Path(dirpath) / fn)
    elif root.is_dir():
        for fp in root.iterdir():
            if fp.is_file() and fp.suffix.lower() in exts:
                files.append(fp)
    files.sort()
    return files


def make_uid(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "".join(
        "_" if c.isspace() else c for c in str(rel.with_suffix("")).replace(os.sep, "_").replace("/", "_")
    )


def _audio_fs_nf(path: str, sample_rate: int | None) -> tuple[int, int, int]:
    """返回 (声明 fs, frames, 真实 fs)；``sample_rate`` 覆盖时声明值可能与真实值不同。"""
    info = sf.info(path)
    fs = sample_rate if sample_rate is not None else info.samplerate
    return int(fs), int(info.frames), int(info.samplerate)


def _entry_from_file(
    fp: Path,
    audio_dir: Path,
    sample_rate: int | None,
    uid: str,
    probe: ProbeOptions | None = None,
) -> Entry | None:
    """One SCP row: ``.pkl`` -> numpy RIR dict/array; anything else in ``AUDIO_EXTS`` -> waveform via soundfile.

    ``probe.validate`` 时音频被全解码并检查零长/NaN/Inf/越界/静音/DC；非法文件跳过
    （返回 None）并更新 ``probe.stats`` 计数。
    """
    probe = probe or ProbeOptions()
    stats = probe.stats
    path = str(fp)
    if fp.suffix.lower() == ".pkl":
        fallback_fs, nf, metadata_fs = _probe_pkl_file(path)
        fs = int(sample_rate) if sample_rate is not None else fallback_fs
        if sample_rate is not None and metadata_fs is not None and metadata_fs != fs:
            msg = f"--sample_rate declares {fs} but real sample rate is {metadata_fs}: {path}"
            if probe.strict_sample_rate:
                raise PrepareScpError(msg)
            if stats is not None:
                stats.inc("fs_mismatch")
            logger.debug("%s", msg)
        return (uid, fs, nf, path)
    try:
        fs, nf, real_fs = _audio_fs_nf(path, sample_rate)
    except Exception as e:
        raise PrepareScpError(f"Cannot read audio (soundfile): {path}: {e}") from e

    if sample_rate is not None and real_fs != fs:
        msg = f"--sample_rate declares {fs} but real sample rate is {real_fs}: {path}"
        if probe.strict_sample_rate:
            raise PrepareScpError(msg)
        if stats is not None:
            stats.inc("fs_mismatch")
        logger.debug("%s", msg)
    if probe.validate:
        if stats is not None:
            stats.inc("total")
        if nf == 0:
            if stats is not None:
                stats.inc("zero_length")
            logger.warning("Skipping zero-length audio: %s", path)
            return None
        ok, reason = validate_audio_file(
            path,
            max_peak=probe.max_peak,
            min_mean_abs=probe.min_mean_abs,
            dc_offset_threshold=probe.dc_offset_threshold,
        )
        if not ok:
            if stats is not None:
                stats.inc(reason)
            logger.warning("Skipping %s audio: %s", reason, path)
            return None
        if stats is not None:
            stats.inc("ok")

    return (uid, fs, nf, path)


def _build_entries_parallel(
    files: list[Path],
    audio_dir: Path,
    sample_rate: int | None,
    workers: int,
    uid_for: Callable[[Path], str] | None = None,
    probe: ProbeOptions | None = None,
) -> list[Entry]:
    """Thread pool for non-.pkl files; .pkl stays on main thread; output order matches ``files``."""
    probe = probe or ProbeOptions()
    if uid_for is None:
        uid_for = lambda fp: make_uid(fp, audio_dir)

    if workers <= 1:
        return [
            e for e in (
                _entry_from_file(fp, audio_dir, sample_rate, uid_for(fp), probe=probe)
                for fp in files
            )
            if e is not None
        ]

    entries: list[Entry | None] = [None] * len(files)
    pkl_indices = [i for i, f in enumerate(files) if f.suffix.lower() == ".pkl"]
    audio_indices = [i for i, f in enumerate(files) if f.suffix.lower() != ".pkl"]

    for i in pkl_indices:
        fp = files[i]
        entries[i] = _entry_from_file(
            fp,
            audio_dir,
            sample_rate,
            uid_for(fp),
            probe=probe,
        )

    if not audio_indices:
        out = [e for e in entries if e is not None]
        return out  # type: ignore[return-value]

    def job(idx: int) -> tuple[int, Entry | None]:
        fp = files[idx]
        uid = uid_for(fp)
        ent = _entry_from_file(fp, audio_dir, sample_rate, uid, probe=probe)
        return idx, ent

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(job, idx) for idx in audio_indices]
        for fut in as_completed(futures):
            try:
                idx, ent = fut.result()
                entries[idx] = ent
            except Exception as e:
                for f2 in futures:
                    f2.cancel()
                raise PrepareScpError(f"Parallel probe failed: {e}") from e

    out = [e for e in entries if e is not None]
    return out  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Build entries & write index files
# -----------------------------------------------------------------------------

def _check_uid_collisions(entries: list[Entry], source_desc: str) -> None:
    """同目录内 uid 碰撞会静默覆盖下游数据（simulation_utils 按 uid 建 dict），

    这里显式报错：同一资源扫描出的 uid 必须唯一（多目录合并已带 ``dir{i}_`` 前缀，
    一般不会跨目录碰撞）。
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for uid, _fs, _nf, path in entries:
        if uid in seen:
            collisions.append(f"{uid!r}: {seen[uid]} <-> {path}")
        else:
            seen[uid] = path
    if collisions:
        preview = "; ".join(collisions[:5])
        more = f" ... and {len(collisions) - 5} more" if len(collisions) > 5 else ""
        raise PrepareScpError(
            f"Duplicate uids in {source_desc}: {preview}{more}. "
            "Rename conflicting files (e.g. a.wav/a.flac in the same dir)."
        )


def build_entries(
    dir_paths: list[Path],
    recursive: bool = True,
    workers: int = 1,
    sample_rate: int | None = None,
    probe: ProbeOptions | None = None,
    followlinks: bool = False,
) -> list[Entry]:
    """扫描并构建一个资源的全部条目。

    单目录 uid 无前缀；多目录按 ``dir{i}_`` 前缀防跨目录碰撞。
    空目录 / 无音频 / 校验后零条目 / uid 碰撞均直接报错。
    ``probe.allow_pkl=False`` 时跳过 ``.pkl`` 文件（非 RIR 资源）。
    """
    probe = probe or ProbeOptions()
    exts = AUDIO_EXTS if probe.allow_pkl else AUDIO_EXTS - {".pkl"}
    all_entries: list[Entry] = []

    for i, audio_dir in enumerate(dir_paths):
        audio_dir = audio_dir.resolve()
        if not audio_dir.is_dir():
            raise PrepareScpError(f"Missing or not a directory: {audio_dir}")
        if len(dir_paths) > 1:
            logger.info("  [%d/%d] Scanning %s", i + 1, len(dir_paths), audio_dir)

        files = scan_audio_dir(audio_dir, recursive=recursive, exts=exts, followlinks=followlinks)
        if not files:
            raise PrepareScpError(f"No audio/pkl files under: {audio_dir}")

        if len(dir_paths) > 1:
            logger.info("    Found %d files", len(files))

        def uid_for(fp: Path, _i: int = i) -> str:
            if len(dir_paths) == 1:
                return make_uid(fp, audio_dir)
            return f"dir{_i}_{make_uid(fp, audio_dir)}"

        chunk = _build_entries_parallel(
            files,
            audio_dir,
            sample_rate,
            workers,
            uid_for=uid_for,
            probe=probe,
        )
        all_entries.extend(chunk)

    if not all_entries:
        raise PrepareScpError("No entries after scanning directories")

    source_desc = f"{dir_paths[0].resolve()} (+{len(dir_paths) - 1} more)" if len(dir_paths) > 1 else str(dir_paths[0].resolve())
    _check_uid_collisions(all_entries, source_desc)
    return all_entries


def process_dir(
    audio_dir: Path,
    out_prefix: Path,
    sample_rate: int | None = None,
    fmt: str = "scp",
    recursive: bool = True,
    workers: int = 1,
    probe: ProbeOptions | None = None,
    followlinks: bool = False,
) -> list[Entry]:
    """单目录模式：扫描 -> 写索引（不划分）。"""
    audio_dir = audio_dir.resolve()
    logger.info("  Scanned %s", audio_dir)

    entries = build_entries(
        [audio_dir],
        recursive=recursive,
        workers=workers,
        sample_rate=sample_rate,
        probe=probe,
        followlinks=followlinks,
    )
    _write_outputs(entries, out_prefix, fmt)
    sr_info = ", ".join(f"{s}Hz" for s in sorted({e[1] for e in entries}))
    logger.info("  -> %d entries, sample rates: %s", len(entries), sr_info)
    return entries


def process_multiple_dirs(
    dir_list: list[Path],
    out_prefix: Path,
    sample_rate: int | None = None,
    fmt: str = "scp",
    recursive: bool = True,
    workers: int = 1,
    probe: ProbeOptions | None = None,
    followlinks: bool = False,
) -> list[Entry]:
    """多目录模式：扫描合并 -> 写索引（不划分）。"""
    all_entries = build_entries(
        dir_list,
        recursive=recursive,
        workers=workers,
        sample_rate=sample_rate,
        probe=probe,
        followlinks=followlinks,
    )
    _write_outputs(all_entries, out_prefix, fmt)
    sr_info = ", ".join(f"{s}Hz" for s in sorted({e[1] for e in all_entries}))
    logger.info("  -> Total: %d entries, sample rates: %s", len(all_entries), sr_info)
    return all_entries


def split_entries_random(
    entries: list[Entry],
    val_ratio: float,
    seed: int,
) -> tuple[list[Entry], list[Entry]]:
    if not entries:
        return [], []
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    if n_val >= len(shuffled):
        raise ValueError(
            f"val_ratio={val_ratio} leaves no samples for training "
            f"(total={len(shuffled)}). Reduce --val_ratio, or pass --val_ratio 0 "
            "to disable splitting."
        )
    val_entries = sorted(shuffled[:n_val], key=lambda e: e[0])
    train_entries = sorted(shuffled[n_val:], key=lambda e: e[0])
    return train_entries, val_entries


def _write_outputs(entries: list[Entry], out_prefix: Path, fmt: str) -> None:
    _publish_output_plans([(entries, out_prefix, fmt)])


def _output_payloads(entries: list[Entry], out_prefix: Path, fmt: str) -> list[tuple[Path, str]]:
    """序列化一组 canonical 四列 SCP，路径相对具体 manifest parent。"""
    if fmt != "scp":
        raise PrepareScpError(f"unsupported source manifest format: {fmt!r}")
    output_path = Path(f"{out_prefix}.scp")
    portable_entries = [
        (uid, fs, frames, serialize_manifest_relative_path(path, output_path))
        for uid, fs, frames, path in entries
    ]
    return [(output_path, serialize_scp_entries(portable_entries, four_col=True))]


def _publish_output_plans(plans: list[tuple[list[Entry], Path, str]]) -> None:
    outputs = [
        output
        for entries, out_prefix, fmt in plans
        for output in _output_payloads(entries, out_prefix, fmt)
    ]
    try:
        write_text_files_transactionally(
            [
                (path, lambda handle, payload=payload: handle.write(payload))
                for path, payload in outputs
            ],
            overwrite=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise PrepareScpError(f"Cannot publish index files transactionally: {exc}") from exc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan audio directories and generate strict four-column SCP index files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLI_EPILOG,
    )
    p.add_argument("input_dir", nargs="?", default=None, help="Single-dir mode: audio root")
    p.add_argument("-o", "--output-dir", required=True, help="Output prefix (single-dir) or directory (batch)")
    p.add_argument("--sample_rate", type=int, default=None, help="Force sample rate (faster; still reads length)")
    p.add_argument(
        "--strict_sample_rate",
        action="store_true",
        help="With --sample_rate: fail if the declared rate differs from a file's real sample rate "
        "(default: count as fs_mismatch in the validation summary)",
    )
    p.add_argument("--no_recursive", action="store_true", help="Only scan files directly under each root")
    p.add_argument(
        "--follow_links",
        action="store_true",
        help="Recursive scan follows symlinked directories (os.walk skips them by default)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread workers for sf.info on wav/flac/etc. (1=sequential). Pickle always on main thread.",
    )
    p.add_argument(
        "--analyze_sr",
        action="store_true",
        help="After run: aggregate sample-rate stats for all batch input roots (or single input_dir)",
    )
    p.add_argument(
        "--analyze_sr_log",
        type=str,
        default=None,
        help="Write analyze_sr report to this file (defaults next to -o)",
    )
    p.add_argument(
        "--validate_audio",
        action="store_true",
        help="Enable full-decode validation per file (zero-length/NaN/Inf/clipping/silence/DC); "
        "disabled by default — enable it for unverified raw sources, skip it for pipeline-verified bulk data",
    )
    p.add_argument(
        "--no_validate_audio",
        action="store_true",
        help="Removed: audio validation is disabled by default; use --validate_audio to enable it",
    )
    p.add_argument(
        "--validate_max_peak",
        type=float,
        default=1.05,
        help="Max allowed absolute peak during validation (default: 1.05)",
    )
    p.add_argument(
        "--validate_min_mean_abs",
        type=float,
        default=1e-6,
        help="Min mean absolute amplitude during validation; lower -> treated as silence (default: 1e-6)",
    )
    p.add_argument(
        "--validate_dc_offset_threshold",
        type=float,
        default=0.5,
        help="Max allowed DC offset during validation (default: 0.5)",
    )

    g = p.add_argument_group("Batch mode")
    g.add_argument(
        "--speech",
        nargs="+",
        default=None,
        help="Speech roots only is OK: writes speech_sources.scp under -o; see --val_ratio for train/val split",
    )
    g.add_argument("--speech_train", nargs="+", default=None, help="Train speech -> train_speech_sources")
    g.add_argument("--speech_val", nargs="+", default=None, help="Val speech -> val_speech_sources")
    g.add_argument("--noise", nargs="+", default=None, help="Noise roots -> noise_sources")
    g.add_argument(
        "--rir",
        nargs="+",
        default=None,
        help="RIR roots -> rirs.scp; same extensions as speech (wav/flac/...) or .pkl numpy/dict RIR",
    )
    g.add_argument("--wind", nargs="+", default=None, help="Wind roots -> wind_noise_sources")
    g.add_argument(
        "--music",
        nargs="+",
        default=None,
        help="Music roots -> music_sources (TSE music_scp 池；与分类 noise JSONL 的 music role 合并)",
    )
    g.add_argument("--extra", action="append", default=[], metavar="name=path", help="Extra name=path[,path...]; repeatable")
    g.add_argument(
        "--val_ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="随机划分验证集比例（默认 0.05，作用于所有资源）；0 关闭划分、只写合并 scp；"
        "与 --speech_train/--speech_val 不冲突（speech 用显式划分，其余仍随机划分）",
    )
    g.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for --val_ratio (default 42)")

    return p


def _validate_numeric_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.val_ratio) or not (0.0 <= args.val_ratio < 1.0):
        p.error("--val_ratio must be finite and in [0, 1) (0 disables the train/val split)")
    if args.workers <= 0:
        p.error("--workers must be positive")
    if args.sample_rate is not None and args.sample_rate <= 0:
        p.error("--sample_rate must be positive")
    if args.strict_sample_rate and args.sample_rate is None:
        p.error("--strict_sample_rate requires --sample_rate")
    if not math.isfinite(args.validate_max_peak) or args.validate_max_peak <= 0:
        p.error("--validate_max_peak must be finite and positive")
    if not math.isfinite(args.validate_min_mean_abs) or args.validate_min_mean_abs < 0:
        p.error("--validate_min_mean_abs must be finite and non-negative")
    if (
        not math.isfinite(args.validate_dc_offset_threshold)
        or args.validate_dc_offset_threshold < 0
    ):
        p.error("--validate_dc_offset_threshold must be finite and non-negative")


def _reject_removed_flags(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """已移除的 deprecated flag 显式拒绝，避免静默 no-op 掩盖调用方意图。"""
    if args.no_validate_audio:
        p.error(
            "--no_validate_audio 已移除：音频校验默认关闭，需要校验时请显式使用 --validate_audio"
        )


def _validate_speech_mutex(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.speech is not None and (args.speech_train is not None or args.speech_val is not None):
        p.error(
            "--speech cannot be used together with --speech_train or --speech_val "
            "(use --speech with --val_ratio for a random split)"
        )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for r in paths:
        try:
            k = str(r.resolve())
        except OSError:
            k = str(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _collect_analyze_roots_from_args(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    for key in ("speech", "speech_train", "speech_val", "noise", "rir", "wind", "music"):
        paths = getattr(args, key, None)
        if paths:
            roots.extend(Path(x) for x in paths)
    for paths in args.extra_sources.values():
        roots.extend(paths)
    return _dedupe_paths(roots)


def _build_batch_sources(args: argparse.Namespace) -> dict[str, tuple[list[Path], bool]]:
    """构造 资源名(输出前缀) -> (根目录列表, 是否跳过随机划分)。

    ``--speech_train/--speech_val`` 直接映射为 ``train_speech_sources/val_speech_sources``
    前缀（显式划分，跳过随机划分）；其余资源由调用方按 ``--val_ratio`` 决定是否划分。
    """
    sources: dict[str, tuple[list[Path], bool]] = {}
    for key, scp_name in BATCH_KEY_TO_PREFIX.items():
        skip_split = key in ("speech_train", "speech_val")
        if key == "speech" and (args.speech_train is not None or args.speech_val is not None):
            continue
        dir_paths = getattr(args, key, None)
        if dir_paths:
            sources[scp_name] = ([Path(p) for p in dir_paths], skip_split)

    for name, paths in args.extra_sources.items():
        sources[name] = (paths, False)

    return sources


def _validate_extra_names(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """严格解析 ``--extra``，拒绝路径逃逸、空路径与重复输出名。"""
    reserved = set(BATCH_KEY_TO_PREFIX.values())
    parsed: dict[str, list[Path]] = {}
    for extra in args.extra:
        if "=" not in extra:
            p.error(f"--extra expected name=path, got: {extra!r}")
        name, raw_paths = extra.split("=", 1)
        if (
            not name
            or name in {".", ".."}
            or Path(name).is_absolute()
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or any(char in name for char in "\r\n\0")
        ):
            p.error(f"--extra name must be a safe non-empty basename, got: {name!r}")
        if name in reserved:
            p.error(f"--extra name {name!r} conflicts with a built-in resource prefix")
        if name.startswith(("train_", "val_")):
            p.error(
                f"--extra name {name!r} uses the reserved train_/val_ prefix "
                "(reserved for split outputs; pick another name)"
            )
        if name in parsed:
            p.error(f"duplicate --extra name: {name!r}")
        path_parts = [part.strip() for part in raw_paths.split(",")]
        if not path_parts or any(not part for part in path_parts):
            p.error(f"--extra {name!r} must contain only non-empty paths")
        parsed[name] = [Path(part) for part in path_parts]
    args.extra_sources = parsed


def _validate_output_prefixes(
    out_dir: Path,
    sources: dict[str, tuple[list[Path], bool]],
    *,
    split_enabled: bool,
) -> None:
    """确保全部派生输出前缀唯一且词法包含于输出目录。"""
    root = Path(os.path.abspath(out_dir))
    prefixes: list[Path] = []
    for name, (_paths, skip_split) in sources.items():
        names = [f"train_{name}", f"val_{name}"] if split_enabled and not skip_split else [name]
        for output_name in names:
            prefix = Path(os.path.abspath(root / output_name))
            try:
                prefix.relative_to(root)
            except ValueError as exc:
                raise PrepareScpError(
                    f"Output prefix escapes --output-dir: {output_name!r}"
                ) from exc
            if prefix.parent != root:
                raise PrepareScpError(
                    f"Output prefix must be a basename under --output-dir: {output_name!r}"
                )
            prefixes.append(prefix)
    if len(prefixes) != len(set(prefixes)):
        raise PrepareScpError("Duplicate output prefixes after resolving batch sources")


def _print_yaml_snippet(
    out_dir: Path,
    sources: dict[str, list[Path]],
    split_enabled: bool,
) -> None:
    def fname(prefix: str) -> Path:
        return out_dir / f"{prefix}.scp"

    lines = [
        "data:",
        "  mode: dynamic",
        "  # 4-column SCP: uid fs length path",
        "  train:",
    ]
    # 资源名 -> 训练配置字段名
    field_map = {
        "speech_sources": "speech_scp",
        "train_speech_sources": "speech_scp",
        "val_speech_sources": "speech_scp",
        "noise_sources": "noise_scp",
        "rirs": "rir_scp",
        "wind_noise_sources": "windnoise_scp",
        "music_sources": "music_scp",
    }
    for scp_name in sources:
        if scp_name == "val_speech_sources":
            continue
        field = field_map.get(scp_name)
        if field is None:
            continue  # --extra 资源名与字段无固定映射，不打印提示
        if split_enabled and not sources[scp_name][1]:
            lines.append(f"    {field}: {fname(f'train_{scp_name}')}")
        else:
            lines.append(f"    {field}: {fname(scp_name)}")

    # validation 块：inherit_simulation: true 时可覆盖全部数据源键
    if split_enabled or "val_speech_sources" in sources:
        lines.extend(
            [
                "  validation:",
                "    mode: dynamic",
                "    inherit_simulation: true  # 仅覆盖以下数据源键",
            ]
        )
        val_paths: dict[str, str] = {}
        for scp_name in sources:
            field = field_map.get(scp_name)
            if field is None:
                continue
            if split_enabled and not sources[scp_name][1]:
                val_paths[field] = fname(f"val_{scp_name}")
            elif scp_name == "val_speech_sources":
                val_paths["speech_scp"] = fname(scp_name)
        for field in (
            "speech_scp",
            "noise_scp",
            "music_scp",
            "rir_scp",
            "windnoise_scp",
        ):
            if field in val_paths:
                lines.append(f"    {field}: {val_paths[field]}")

    print("\n" + "=" * 60)
    print("YAML snippet (paste into config)")
    print("=" * 60)
    print("\n".join(lines))
    print("=" * 60 + "\n")


def _plan_random_split(
    scp_name: str,
    dir_paths: list[Path],
    args: argparse.Namespace,
    out_dir: Path,
    recursive: bool,
    workers: int,
    stats: ValidationStats,
) -> list[tuple[list[Entry], Path, str]]:
    """扫描一个资源并返回 train_/val_ 集合发布计划。"""
    probe = ProbeOptions.from_args(
        args, allow_pkl=(scp_name == "rirs"), stats=stats
    )
    entries = build_entries(
        dir_paths,
        recursive=recursive,
        workers=workers,
        sample_rate=args.sample_rate,
        probe=probe,
        followlinks=args.follow_links,
    )
    try:
        train_entries, val_entries = split_entries_random(entries, args.val_ratio, args.seed)
    except ValueError as e:
        raise PrepareScpError(str(e)) from e
    logger.info(
        "  [%s] split -> train %d, val %d (val_ratio=%s, seed=%s)",
        scp_name,
        len(train_entries),
        len(val_entries),
        args.val_ratio,
        args.seed,
    )
    return [
        (train_entries, out_dir / f"train_{scp_name}", "scp"),
        (val_entries, out_dir / f"val_{scp_name}", "scp"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _reject_removed_flags(parser, args)
    _validate_speech_mutex(parser, args)
    _validate_numeric_args(parser, args)

    workers = args.workers

    recursive = not args.no_recursive
    is_batch = any(
        [
            args.speech,
            args.speech_train,
            args.speech_val,
            args.noise,
            args.rir,
            args.wind,
            args.music,
            args.extra,
        ]
    )

    if not is_batch and args.input_dir is None:
        parser.error("Need input_dir (single mode) or batch flags (--speech, --noise, ...)")

    _validate_extra_names(parser, args)
    split_enabled = args.val_ratio > 0.0
    stats = ValidationStats()

    try:
        if not is_batch:
            logger.info("Single directory mode: %s", args.input_dir)
            process_dir(
                Path(args.input_dir),
                Path(args.output_dir),
                sample_rate=args.sample_rate,
                fmt="scp",
                recursive=recursive,
                workers=workers,
                probe=ProbeOptions.from_args(args, stats=stats),
                followlinks=args.follow_links,
            )
            if args.analyze_sr:
                log = args.analyze_sr_log or str(Path(args.output_dir).resolve().parent / "sr_analysis_report.txt")
                analyze_audio_sr([Path(args.input_dir)], save_log=Path(log), recursive=recursive)
        else:
            out_dir = Path(args.output_dir)
            sources = _build_batch_sources(args)

            if not sources:
                raise PrepareScpError("Batch mode needs at least one source directory")
            _validate_output_prefixes(
                out_dir, sources, split_enabled=split_enabled
            )

            plans: list[tuple[list[Entry], Path, str]] = []
            for scp_name, (dir_paths, skip_split) in sources.items():
                if split_enabled and not skip_split:
                    plans.extend(
                        _plan_random_split(
                            scp_name,
                            dir_paths,
                            args,
                            out_dir,
                            recursive,
                            workers,
                            stats,
                        )
                    )
                    continue
                logger.info("Processing [%s]: %d root(s)", scp_name, len(dir_paths))
                probe = ProbeOptions.from_args(
                    args, allow_pkl=(scp_name == "rirs"), stats=stats
                )
                entries = build_entries(
                    dir_paths,
                    recursive=recursive,
                    workers=workers,
                    sample_rate=args.sample_rate,
                    probe=probe,
                    followlinks=args.follow_links,
                )
                plans.append((entries, out_dir / scp_name, "scp"))
                sr_info = ", ".join(f"{s}Hz" for s in sorted({e[1] for e in entries}))
                logger.info("  -> Total: %d entries, sample rates: %s", len(entries), sr_info)

            _publish_output_plans(plans)
            if split_enabled:
                logger.info(
                    "Random split enabled for all resources: val_ratio=%s seed=%s",
                    args.val_ratio,
                    args.seed,
                )
            _print_yaml_snippet(out_dir, sources, split_enabled)

            if args.analyze_sr:
                log = args.analyze_sr_log or str(out_dir / "sr_analysis_report.txt")
                analyze_audio_sr(_collect_analyze_roots_from_args(args), save_log=Path(log), recursive=recursive)
    except PrepareScpError as e:
        logger.error("%s", e)
        return 1

    if stats is not None:
        logger.info("Validation summary: %s", stats.summary())

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
