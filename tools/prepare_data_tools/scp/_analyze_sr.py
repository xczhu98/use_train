"""采样率审计（--analyze_sr）实现，自 scp/prepare_scp.py 拆出。"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import soundfile as sf

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    import librosa
except ImportError:
    librosa = None

# 审计口径的扩展名集合：镜像 prepare_scp.AUDIO_EXTS（去掉 .pkl）再加 m4a/aac/wma。
# 独立定义避免与 prepare_scp 形成循环导入。
ANALYZE_EXTS = {
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
    ".m4a",
    ".aac",
    ".wma",
}

from tools.prepare_data_tools._common import setup_logging  # noqa: E402

logger = setup_logging("prepare_scp")


def _get_audio_info_for_analyze(file_path: Path) -> tuple[int | None, int | None]:
    path = str(file_path)
    try:
        with sf.SoundFile(path) as f:
            return int(f.samplerate), int(f.frames)
    except Exception:
        pass
    if librosa is None:
        return None, None
    try:
        if hasattr(librosa, "get_duration") and hasattr(librosa, "get_samplerate"):
            dur = float(librosa.get_duration(path=path))
            sr = int(librosa.get_samplerate(path))
            return sr, int(dur * sr)
    except Exception:
        pass
    try:
        y, sr = librosa.load(path, sr=None, mono=False)
        n = int(y.shape[-1]) if y.ndim > 1 else int(len(y))
        return int(sr), n
    except Exception:
        return None, None


def scan_audio_files_for_analyze(root: Path, recursive: bool = True) -> list[Path]:
    files: list[Path] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                suf = Path(fn).suffix.lower()
                if suf in ANALYZE_EXTS:
                    files.append(Path(dirpath) / fn)
    elif root.is_dir():
        for fp in root.iterdir():
            if fp.is_file() and fp.suffix.lower() in ANALYZE_EXTS:
                files.append(fp)
    files.sort()
    return files


def _unique_analyze_files(target_dirs: list[Path], recursive: bool) -> list[Path]:
    """Deduplicate by resolved path so nested/overlapping roots are not double-counted."""
    roots = [d.resolve() for d in target_dirs if d is not None]
    roots = [d for d in roots if d.exists()]
    seen: set[str] = set()
    ordered: list[Path] = []
    for r in roots:
        for p in scan_audio_files_for_analyze(r, recursive=recursive):
            try:
                key = str(p.resolve())
            except OSError:
                key = str(p)
            if key not in seen:
                seen.add(key)
                ordered.append(p)
    ordered.sort()
    return ordered


def analyze_audio_sr(
    target_dirs: list[Path],
    save_log: Path | None = None,
    recursive: bool = True,
) -> None:
    roots = [d.resolve() for d in target_dirs if d is not None]
    roots = [d for d in roots if d.exists()]
    if not roots:
        logger.warning("analyze_sr: no existing directories to scan")
        return

    audio_files = _unique_analyze_files(roots, recursive)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit(f"\n开始分析：唯一音频文件 {len(audio_files)} 个（{len(roots)} 个根目录，路径已去重）\n")

    sr_stats: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "total_frames": 0})
    error_files: list[str] = []

    for file in tqdm(audio_files, desc="采样率分析"):
        sr, frames = _get_audio_info_for_analyze(file)
        if sr is not None and frames is not None:
            sr_stats[sr]["count"] += 1
            sr_stats[sr]["total_frames"] += frames
        else:
            error_files.append(str(file))

    total_files = len(audio_files)
    stats: dict[int, dict[str, float | int]] = {}
    total_all_seconds = 0.0
    for sr in sorted(sr_stats.keys()):
        count = sr_stats[sr]["count"]
        total_frames = sr_stats[sr]["total_frames"]
        total_seconds = total_frames / sr
        total_all_seconds += total_seconds
        stats[sr] = {
            "音频个数": count,
            "总帧数": total_frames,
            "总时长(秒)": round(total_seconds, 2),
            "总时长(小时)": round(total_seconds / 3600.0, 4),
        }

    summary = {
        "根目录数": len(roots),
        "唯一音频文件数": total_files,
        "有效音频数": total_files - len(error_files),
        "损坏/无法识别文件数": len(error_files),
        "有效音频总时长(小时)": round(total_all_seconds / 3600.0, 4),
        "涉及采样率种类": len(stats),
    }

    emit("\n" + "=" * 80)
    emit("【音频采样率统计汇总】")
    for k, v in summary.items():
        emit(f"{k:<22}: {v}")
    emit("=" * 80)
    emit("【各采样率详细统计】")
    emit(f"{'采样率(Hz)':<12} {'音频个数':<10} {'总帧数':<15} {'总时长(秒)':<15} {'总时长(小时)':<15}")
    emit("-" * 70)
    for sr in sorted(stats.keys()):
        info = stats[sr]
        emit(
            f"{sr:<12} {info['音频个数']:<10} {info['总帧数']:<15} "
            f"{info['总时长(秒)']:<15} {info['总时长(小时)']:<15}"
        )
    emit("=" * 80)

    if error_files:
        emit(f"\n【读取失败的文件列表（共{len(error_files)}个）】")
        for fpath in error_files[:200]:
            emit(f"- {fpath}")
        if len(error_files) > 200:
            emit(f"... 另有 {len(error_files) - 200} 个未列出")

    if save_log is not None:
        save_log = Path(save_log)
        save_log.parent.mkdir(parents=True, exist_ok=True)
        save_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("采样率统计已写入: %s", save_log)
