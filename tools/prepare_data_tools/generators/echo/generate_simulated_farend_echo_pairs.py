#!/usr/bin/env python3
"""离线批量生成仿真回声对（farend + RIR → echo），输出与训练 100% 兼容的 stereo WAV + JSON。

仿真链路（与 ``monaural_simu_dataset.py`` 对齐）::

    farend ──→ [EQ 扰动] ──→ [非线性失真] ──→ [RIR 卷积] ──→ [线路噪声] ──→ [SER 缩放] ──→ echo
                                                    ↑
                                            随机选真实 RIR

输出为 stereo 单文件 WAV（ch0=echo, ch1=ref），JSON 格式与
``build_real_farend_echo_pair_manifest.py`` 完全一致，可直接作为
``real_farend_echo_pair_json`` 使用。

用法::

    # 1) 写 YAML 配置（见下方示例）
    vim tools/prepare_data_tools/generators/echo/generate_simulated_farend_echo_pairs.yaml

    # 2) 生成
    python -m tools.prepare_data_tools.generators.echo.generate_simulated_farend_echo_pairs \
        -C tools/prepare_data_tools/generators/echo/generate_simulated_farend_echo_pairs.yaml

    # 3) 与真实 manifest 合并（可选）
    python -m tools.prepare_data_tools.generators.echo.merge_farend_echo_pair_manifests \
        --inputs real_manifest.json synthetic_manifest.json \
        --output merged_manifest.json

    # 4) 诊断验证
    python tools/verify/aec/diagnose_aec_stereo_data.py \
        --json data/synthetic_echo_pairs/real_farend_echo_pair_manifest.json

**YAML 示例**

.. code-block:: yaml

    farend_scp: data/lite_realtime_denoie_dynamic_16k/train_speech_sources.scp
    rir_scp: data/lite_realtime_denoie_dynamic_16k/rirs.scp
    noise_scp: data/lite_realtime_denoie_dynamic_16k/noise_sources.scp   # 可选：线路噪声源
    output_dir: data/synthetic_echo_pairs
    num_pairs: 10000
    sample_rate: 16000
    max_duration_seconds: 10
    seed: 42
    workers: 8          # 并行进程数；0=单进程

    # 物理参数
    ser_range_db: [-20, 20]
    nonlinear_distortion:
      enabled: true
      types: [soft_clip, hard_clip, harmonic, none]   # 按概率均匀抽取
      prob: 0.5
    line_noise:
      enabled: true
      snr_db: [20, 40]      # 线路噪声 vs echo 的 SNR
      prob: 0.3
    random_delay_ms: [0, 50]   # 额外随机延迟（模拟设备 buffer）
    eq_perturb:
      enabled: false
      prob: 0.0

    # 输出控制
    # 输出固定为 stereo WAV（subtype=PCM_16），无 format/bit_depth 配置项
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import fftconvolve

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.prepare_data_tools._common import (  # noqa: E402
    ensure_project_root,
    setup_logging,
)
from tools.shared.scp_io import read_scp  # noqa: E402
from tools.shared.pickle_io import load_pkl_file  # noqa: E402
from core.data.simulation.urgent_impl import detect_non_silence  # noqa: E402

ensure_project_root()
logger = setup_logging("generate_simulated_farend_echo_pairs")

# Canonical 输出文件名：与 build_real_farend_echo_pair_manifest.py 的无划分输出对齐。
DEFAULT_MANIFEST_NAME = "real_farend_echo_pair_manifest.json"
DEFAULT_METADATA_NAME = "real_farend_echo_pair_metadata.json"
GENERATOR_ID = "echo.generate_simulated_farend_echo_pairs"
GENERATOR_VERSION_HISTORY = ("1.0", "1.1")
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]


def _sha256_file(path: Path) -> str:
    """返回 SCP 或发布音频的内容 SHA-256 身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _load_audio(path: str, target_sr: int, max_len: int | None = None) -> np.ndarray:
    """Load mono audio, resample if needed, optionally truncate.

    max_len 按目标采样率计（样本数）：读取时先按源采样率折算帧数
    （ceil 保证重采样后不短于 max_len），重采样后再截断。
    """
    info = sf.info(path)
    sr = info.samplerate
    if max_len is not None and sr != target_sr:
        read_frames = min(info.frames, int(math.ceil(max_len * sr / target_sr)))
    elif max_len is not None:
        read_frames = max_len
    else:
        read_frames = info.frames
    data, sr_read = sf.read(path, dtype="float32", frames=read_frames)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr_read != target_sr:
        import librosa
        # librosa 对短信号会发 UserWarning；只在重采样这一段局部抑制，
        # 不再用模块级 filterwarnings 掩盖整个进程的数值告警。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            data = librosa.resample(
                data, orig_sr=sr_read, target_sr=target_sr, res_type="soxr_hq"
            )
    data = np.asarray(data, dtype=np.float32).ravel()
    if max_len is not None:
        data = data[:max_len]
    return data


def _load_rir(path: str) -> np.ndarray:
    """Load RIR; support .wav/.flac and .pkl (pickle).

    Returns:
        rir_full  1-D float32（去前导静音 + 归一化）
    """
    p = Path(path)
    if p.suffix == ".pkl":
        # 受限反序列化（仅 numpy/基础类型）；可信数据可用
        # PREPARE_SCP_ALLOW_UNSAFE_PICKLE=1 回退完整 pickle
        obj = load_pkl_file(
            path, purpose="load RIR for simulated echo generation"
        )
        if isinstance(obj, dict):
            rir = obj.get("source_rir")
            if rir is None:
                rir = obj.get("rir")
        else:
            rir = obj
        if rir is None:
            raise ValueError(f"RIR pickle 缺少 rir/source_rir 字段: {path}")
    else:
        rir, _ = sf.read(path, dtype="float32")
        if rir.ndim == 2:
            rir = rir.mean(axis=1)

    rir = np.asarray(rir, dtype=np.float32)
    if rir.ndim > 1:
        rir = rir[:, 0]
    rir = rir.ravel()
    if rir.size == 0:
        raise ValueError(f"RIR 为空: {path}")
    # Crop from peak (remove leading silence) and normalize
    peak_idx = int(np.argmax(np.abs(rir)))
    rir = rir[peak_idx:]
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir


# ---------------------------------------------------------------------------
# Nonlinear distortion (loudspeaker simulation)
# ---------------------------------------------------------------------------
def _distort_soft_clip(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    return x * threshold / (threshold + np.abs(x) + 1e-8)


def _distort_hard_clip(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    return np.clip(x, -threshold, threshold)


def _distort_harmonic(x: np.ndarray, gain: float = 0.1) -> np.ndarray:
    return x + gain * (x ** 2)


_DISTORTERS: dict[str, Any] = {
    "soft_clip": _distort_soft_clip,
    "hard_clip": _distort_hard_clip,
    "harmonic": _distort_harmonic,
    "none": lambda x: x,
}


# ---------------------------------------------------------------------------
# Core echo generation
# ---------------------------------------------------------------------------
def generate_one_pair(
    farend_path: str,
    rir_path: str,
    noise_path: str | None,
    target_sr: int,
    max_len: int,
    ser_db: float,
    distort_type: str | None,
    line_noise_snr_db: float | None,
    random_delay_ms: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one synthetic echo/ref pair.

    Returns:
        (ref_1d, echo_1d, meta_dict)  both float32, length <= max_len
    """
    # 1. Load farend
    ref = _load_audio(farend_path, target_sr, max_len=max_len)
    if ref.size == 0:
        raise ValueError(f"Empty farend: {farend_path}")

    # 2. EQ perturb (placeholder; currently no-op unless external config enables it)
    #    (can be extended with phone_DL_EQ_perturb from NS/se_train)

    # 3. Nonlinear distortion
    echo = ref.copy()
    if distort_type and distort_type != "none":
        echo = _DISTORTERS[distort_type](echo)

    # 4. RIR convolution
    rir_full = _load_rir(rir_path)
    echo = fftconvolve(echo, rir_full, mode="full")[: len(ref)]

    # 5. Line noise
    if line_noise_snr_db is not None and noise_path is not None:
        noise = _load_audio(noise_path, target_sr, max_len=len(ref))
        if noise.size < len(ref):
            noise = np.pad(noise, (0, len(ref) - noise.size))
        else:
            noise = noise[: len(ref)]
        noise_power = float(np.mean(noise ** 2)) + 1e-10
        echo_power = float(np.mean(echo ** 2)) + 1e-10
        scale = np.sqrt(echo_power / (noise_power * (10 ** (line_noise_snr_db / 10.0))))
        echo = echo + scale * noise

    # 6. SER scaling (active-segment power, same as scale_echo_for_ser_db)
    ref_active = ref[detect_non_silence(ref.reshape(1, -1))]
    echo_active = echo[detect_non_silence(echo.reshape(1, -1))]
    power_ref = float(np.mean(ref_active ** 2)) if ref_active.size > 0 else 1e-10
    power_echo = float(np.mean(echo_active ** 2)) if echo_active.size > 0 else 1e-10
    power_ref = max(power_ref, 1e-10)
    power_echo = max(power_echo, 1e-10)
    scale_ser = 10.0 ** (-float(ser_db) / 20.0) * np.sqrt(power_ref) / np.sqrt(power_echo)
    echo = echo * scale_ser

    # 7. Random delay (prepend zeros to echo)
    delay_lo, delay_hi = random_delay_ms
    delay_ms = float(rng.uniform(delay_lo, delay_hi))
    delay_samp = max(0, int(round(delay_ms * target_sr / 1000.0)))
    if delay_samp > 0:
        echo = np.concatenate([np.zeros(delay_samp, dtype=np.float32), echo])[:max_len]
        # Pad ref to same length if needed
        if ref.size < echo.size:
            ref = np.pad(ref, (0, echo.size - ref.size))
        else:
            ref = ref[: echo.size]
    else:
        ref = ref[:max_len]
        echo = echo[:max_len]

    meta = {
        "ser_db": float(ser_db),
        "distort_type": distort_type or "none",
        "line_noise_snr_db": line_noise_snr_db,
        "random_delay_ms": delay_ms,
        "rir_path": str(rir_path),
        "farend_path": str(farend_path),
        "noise_path": str(noise_path) if noise_path else None,
    }
    return ref, echo, meta


# ---------------------------------------------------------------------------
# Worker (for multiprocessing)
# ---------------------------------------------------------------------------
# 进程级全局配置：由 Pool/Executor 的 initializer 一次性注入，
# 避免把整份 farend/rir/noise 路径列表嵌入每个 task tuple 重复 pickle。
_WORKER_CFG: dict | None = None


def _worker_init(cfg: dict):
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _iter_pair_seeds(seed: int, num_pairs: int):
    """逐条生成与旧版向量化调用完全相同的 ``(idx, seed)`` 序列。"""
    rng = np.random.default_rng(seed)
    for idx in range(num_pairs):
        yield idx, int(rng.integers(0, 2**31))


def _iter_bounded_futures(executor, tasks, max_outstanding: int):
    """提交有界数量任务，并逐个返回 ``(idx, future)``。"""
    from concurrent.futures import FIRST_COMPLETED, wait

    if max_outstanding < 1:
        raise ValueError("max_outstanding must be >= 1")

    task_iter = iter(tasks)
    pending = {}

    def _submit_one() -> bool:
        try:
            idx, task_seed = next(task_iter)
        except StopIteration:
            return False
        pending[executor.submit(_worker_generate, idx, task_seed)] = idx
        return True

    for _ in range(max_outstanding):
        if not _submit_one():
            break

    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield pending.pop(future), future
        while len(pending) < max_outstanding and _submit_one():
            pass


def _worker_generate(idx: int, seed: int) -> dict:
    """Worker that receives a seed integer and builds its own RNG.

    Using integer seeds avoids numpy Generator pickling issues across processes;
    worker 配置经 initializer 注入进程全局 ``_WORKER_CFG``。
    """
    cfg = _WORKER_CFG
    if cfg is None:
        raise RuntimeError("worker 全局配置未初始化（缺少 initializer）")
    rng = np.random.default_rng(seed)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Random selections
        farend_path = str(rng.choice(cfg["farend_paths"]))
        rir_path = str(rng.choice(cfg["rir_paths"]))
        noise_path = str(rng.choice(cfg["noise_paths"])) if cfg.get("noise_paths") else None

        ser_db = float(rng.uniform(*cfg["ser_range_db"]))

        distort_type = None
        if cfg.get("nonlinear_enabled") and rng.random() < cfg.get("nonlinear_prob", 0.5):
            distort_type = str(rng.choice(cfg.get("nonlinear_types", ["none"])))

        line_noise_snr = None
        if cfg.get("line_noise_enabled") and rng.random() < cfg.get("line_noise_prob", 0.3):
            line_noise_snr = float(rng.uniform(*cfg.get("line_noise_snr_db", [20, 40])))

        ref, echo, meta = generate_one_pair(
            farend_path=farend_path,
            rir_path=rir_path,
            noise_path=noise_path,
            target_sr=cfg["sample_rate"],
            max_len=cfg["max_len"],
            ser_db=ser_db,
            distort_type=distort_type,
            line_noise_snr_db=line_noise_snr,
            random_delay_ms=tuple(cfg["random_delay_ms"]),
            rng=rng,
        )

        # Write stereo WAV: ch0=echo, ch1=ref
        # Pad both to same length
        target_len = max(len(ref), len(echo))
        if len(ref) < target_len:
            ref = np.pad(ref, (0, target_len - len(ref)))
        if len(echo) < target_len:
            echo = np.pad(echo, (0, target_len - len(echo)))
        stereo = np.stack([echo, ref], axis=1).astype(np.float32)

        out_path = out_dir / f"pair_{idx:08d}.wav"
        sf.write(str(out_path), stereo, cfg["sample_rate"], subtype="PCM_16")

        return {
            "path": str(out_path.resolve()),
            "meta": meta,
        }
    except Exception:
        # Log full traceback so we can diagnose ambiguous-array errors
        logger.error("Pair %08d failed:\n%s", idx, traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _finite_pair(value: Any, name: str, *, nonnegative: bool = False) -> tuple[float, float]:
    """规范化有限递增二元区间。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} 必须是包含两个数值的区间")
    lo, hi = (float(item) for item in value)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError(f"{name} 必须是有限且 lo <= hi 的区间")
    if nonnegative and lo < 0.0:
        raise ValueError(f"{name} 下界必须 >= 0")
    return lo, hi


def _validate_generation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """在读源文件或创建输出目录前验证所有生成参数。"""
    num_pairs = int(cfg["num_pairs"])
    sample_rate = int(cfg.get("sample_rate", 16000))
    max_duration = float(cfg.get("max_duration_seconds", 10.0))
    workers = int(cfg.get("workers", 0))
    if num_pairs <= 0:
        raise ValueError("num_pairs 必须 > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须 > 0")
    if not math.isfinite(max_duration) or max_duration <= 0.0:
        raise ValueError("max_duration_seconds 必须是有限正数")
    if workers < 0:
        raise ValueError("workers 必须 >= 0")

    nonlinear = cfg.get("nonlinear_distortion", {})
    line_noise = cfg.get("line_noise", {})
    for name, probability in (
        ("nonlinear_distortion.prob", float(nonlinear.get("prob", 0.5))),
        ("line_noise.prob", float(line_noise.get("prob", 0.3))),
    ):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} 必须是 [0, 1] 内有限数值")
    distortion_types = list(nonlinear.get("types", ["none"]))
    unknown = sorted(set(distortion_types) - set(_DISTORTERS))
    if not distortion_types or unknown:
        raise ValueError(f"nonlinear_distortion.types 含未知类型: {unknown}")

    _finite_pair(cfg.get("ser_range_db", [-20, 20]), "ser_range_db")
    _finite_pair(line_noise.get("snr_db", [20, 40]), "line_noise.snr_db")
    _finite_pair(
        cfg.get("random_delay_ms", [0, 50]),
        "random_delay_ms",
        nonnegative=True,
    )
    return cfg

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="离线生成仿真回声对（farend + RIR → echo），输出 stereo WAV + JSON manifest。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例::
    python -m tools.prepare_data_tools.generators.echo.generate_simulated_farend_echo_pairs \\
        -C tools/prepare_data_tools/generators/echo/generate_simulated_farend_echo_pairs.yaml
        """,
    )
    ap.add_argument(
        "-C",
        "--config",
        type=Path,
        required=True,
        help="YAML 配置文件路径",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印参数和抽样计划，不写文件",
    )
    args = ap.parse_args(argv)

    # Load YAML
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置根必须是 mapping")
    cfg = _validate_generation_config(cfg)

    # Resolve paths
    farend_scp = Path(cfg["farend_scp"])
    rir_scp = Path(cfg["rir_scp"])
    noise_scp = Path(cfg["noise_scp"]) if cfg.get("noise_scp") else None
    output_dir = Path(cfg["output_dir"])
    num_pairs = int(cfg["num_pairs"])
    sample_rate = int(cfg.get("sample_rate", 16000))
    max_dur = float(cfg.get("max_duration_seconds", 10.0))
    max_len = int(max_dur * sample_rate)
    seed = int(cfg.get("seed", 42))
    workers = int(cfg.get("workers", 0))

    # Read SCPs
    farend_paths = read_scp(farend_scp)
    rir_paths = read_scp(rir_scp)
    noise_paths = read_scp(noise_scp) if noise_scp else []

    if not farend_paths:
        raise SystemExit(f"farend SCP 为空或不存在: {farend_scp}")
    if not rir_paths:
        raise SystemExit(f"RIR SCP 为空或不存在: {rir_scp}")

    # Build worker config
    worker_cfg = {
        "output_dir": str(output_dir),
        "farend_paths": farend_paths,
        "rir_paths": rir_paths,
        "noise_paths": noise_paths,
        "sample_rate": sample_rate,
        "max_len": max_len,
        "ser_range_db": cfg.get("ser_range_db", [-20, 20]),
        "nonlinear_enabled": bool(cfg.get("nonlinear_distortion", {}).get("enabled", False)),
        "nonlinear_prob": float(cfg.get("nonlinear_distortion", {}).get("prob", 0.5)),
        "nonlinear_types": list(cfg.get("nonlinear_distortion", {}).get("types", ["none"])),
        "line_noise_enabled": bool(cfg.get("line_noise", {}).get("enabled", False)),
        "line_noise_prob": float(cfg.get("line_noise", {}).get("prob", 0.3)),
        "line_noise_snr_db": cfg.get("line_noise", {}).get("snr_db", [20, 40]),
        "random_delay_ms": cfg.get("random_delay_ms", [0, 50]),
    }

    logger.info("=" * 60)
    logger.info("生成计划")
    logger.info("  farend 源: %d 条 (%s)", len(farend_paths), farend_scp)
    logger.info("  RIR 源:    %d 条 (%s)", len(rir_paths), rir_scp)
    logger.info("  噪声源:    %d 条 (%s)", len(noise_paths), noise_scp or "N/A")
    logger.info("  生成对数:  %d", num_pairs)
    logger.info("  采样率:    %d Hz", sample_rate)
    logger.info("  最大长度:  %d samples (%.1f s)", max_len, max_dur)
    logger.info("  SER 范围:  %.1f ~ %.1f dB", *worker_cfg["ser_range_db"])
    logger.info("  非线性:    %s (prob=%.2f, types=%s)",
                worker_cfg["nonlinear_enabled"], worker_cfg["nonlinear_prob"],
                worker_cfg["nonlinear_types"])
    logger.info("  线路噪声:  %s (prob=%.2f, SNR=%s dB)",
                worker_cfg["line_noise_enabled"], worker_cfg["line_noise_prob"],
                worker_cfg["line_noise_snr_db"])
    logger.info("  随机延迟:  %s ms", worker_cfg["random_delay_ms"])
    logger.info("  并行数:    %d (%s)", workers, "多进程" if workers > 0 else "单进程")
    logger.info("  输出目录:  %s", output_dir.resolve())
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[dry-run] 已跳过文件写入")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate。逐条派生 seed，保持旧版向量化序列但避免分配完整数组。
    tasks = _iter_pair_seeds(seed, num_pairs)
    results: list[dict] = []
    failed = 0
    if workers > 0:
        from concurrent.futures import ProcessPoolExecutor

        # 配置经 initializer 一次性注入进程全局；outstanding futures 保持有界。
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(worker_cfg,),
        ) as exe:
            completed = 0
            for idx, future in _iter_bounded_futures(exe, tasks, max_outstanding=workers * 2):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error("Pair %08d failed: %s", idx, exc)
                    failed += 1
                completed += 1
                if completed % 100 == 0:
                    logger.info("Progress: %d / %d", completed, num_pairs)
    else:
        _worker_init(worker_cfg)
        for i, pair_seed in tasks:
            try:
                res = _worker_generate(i, pair_seed)
                results.append(res)
            except Exception as exc:
                logger.error("Pair %08d failed: %s", i, exc)
                failed += 1
            if (i + 1) % 100 == 0:
                logger.info("Progress: %d / %d", i + 1, num_pairs)

    # 失败时不发布 manifest/meta，避免下游把不完整音频集合误认为完整数据集。
    if failed > 0:
        logger.error("失败摘要: %d / %d 对生成失败，请检查上方日志", failed, num_pairs)
        return 1

    # Sort by path order
    results.sort(key=lambda x: x["path"])

    # 发布 manifest 绑定规范化参数、输入 SCP 内容和输出 inventory，防止参数或输入
    # 变化后误把历史产物当作可复用的同一数据集。
    input_fingerprints = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in (("farend_scp", farend_scp), ("rir_scp", rir_scp))
    }
    if noise_scp is not None:
        input_fingerprints["noise_scp"] = {
            "path": str(noise_scp.resolve()), "sha256": _sha256_file(noise_scp)
        }
    output_inventory = [
        {"path": r["path"], "sha256": _sha256_file(Path(r["path"]))}
        for r in results
    ]
    manifest = {
        "pairs": [r["path"] for r in results],
        "meta": {
            "source": "synthetic",
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "generator_version_history": list(GENERATOR_VERSION_HISTORY),
            "seed": seed,
            "input_fingerprints": input_fingerprints,
            "output_inventory": output_inventory,
            "num_pairs": len(results),
            "sample_rate": sample_rate,
            "max_duration_seconds": max_dur,
            "config": {k: v for k, v in worker_cfg.items() if k not in ("farend_paths", "rir_paths", "noise_paths")},
        },
    }
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )

    # manifest 与逐条元信息必须作为一个发布集合提交。
    meta_path = output_dir / DEFAULT_METADATA_NAME
    meta_payload = (
        json.dumps(
            {r["path"]: r["meta"] for r in results},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    write_text_files_transactionally(
        [
            (manifest_path, lambda handle: handle.write(manifest_payload)),
            (meta_path, lambda handle: handle.write(meta_payload)),
        ],
        overwrite=True,
    )

    logger.info("=" * 60)
    logger.info("完成: %d / %d 对生成成功", len(results), num_pairs)
    logger.info("Manifest: %s", manifest_path.resolve())
    logger.info("Meta:     %s", meta_path.resolve())
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass
