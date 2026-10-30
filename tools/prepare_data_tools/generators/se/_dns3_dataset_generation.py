#!/usr/bin/env python3
"""DNS Challenge 3 source scan、DSP、split 与 SCP 生成实现。

复现 GTCRN (ICASSP 2024) Section 3.1：全混响语音与噪声按
``[-5, 15]`` dB SNR 混合，以 RIR 前 100 ms 生成低混响目标。train/val/test
从同一 clean/noise/RIR 文件池采样，不做源级 hold-out；该数据仅用于论文复现。

CLI、参数校验、锁、owned staging 与事务发布由
``generate_dns3_offline_dataset`` 独占管理。
"""

from __future__ import annotations

import csv
import hashlib
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
from tools.prepare_data_tools._common import setup_logging  # noqa: E402
from tools.shared.file_scan import scan_audio_files  # noqa: E402
from tools.shared.scp_io import probe_audio_meta, write_scp_entries  # noqa: E402

logger = setup_logging("gen_dns3")
# ─── Paper params (GTCRN, Section 3.1) ───
FS = 16000
WAV_LEN_SEC = 10
SNR_RANGE = (-5, 15)
EARLY_RIR_MS = 100
NUM_TRAIN = 720000
NUM_VAL = 840
NUM_TEST = 800
SEED = 42


@dataclass(frozen=True)
class DNS3GenerationConfig:
    """一次 DNS3 waveform/SCP 生成所需的显式 library 参数。"""

    data_root: Path
    output_dir: Path
    scp_dir: Path | None = None
    num_train: int = NUM_TRAIN
    num_val: int = NUM_VAL
    num_test: int = NUM_TEST
    fs: int = FS
    wav_len: int = WAV_LEN_SEC
    num_workers: int = 4
    seed: int = SEED
    write_info_csv: bool = False


def check_data_root(data_root: str) -> dict[str, Path]:
    """Check if DNS3 data directory exists, return subdirectory paths."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {root}\n"
            "Please download DNS Challenge 3 data first:\n"
            "  git clone https://github.com/microsoft/DNS-Challenge.git\n"
            "  cd DNS-Challenge\n"
            "  python download-dns-challenge-3.py\n"
            f"Place data at: {root}"
        )

    candidates = {
        "clean": [
            root / "datasets_fullband" / "clean_fullband",
            root / "clean",
            root / "clean_fullband",
        ],
        "noise": [
            root / "datasets_fullband" / "noise_fullband",
            root / "noise",
            root / "noise_fullband",
        ],
        "rir": [
            root / "datasets_fullband" / "impulse_responses",
            root / "impulse_responses",
            root / "rir",
        ],
    }

    paths = {}
    for key, options in candidates.items():
        found = None
        for p in options:
            if p.exists():
                found = p
                break
        if found is None:
            raise FileNotFoundError(
                f"Data directory for {key} not found. Checked paths:\n"
                + "\n".join(f"  - {p}" for p in options)
                + f"\nPlease confirm DNS3 data is downloaded to {root}"
            )
        paths[key] = found
        logger.info(f"  {key}: {found}")

    return paths


def build_csv_lists(data_root: str):
    """Scan DNS3 data directory, generate clean/noise/rir CSV lists."""
    logger.info("Scanning data directory...")
    paths = check_data_root(data_root)

    root = Path(data_root)
    for key, dir_path in paths.items():
        # 复用 tools/shared/file_scan.scan_audio_files（返回绝对路径 Path 列表，已排序）
        files = scan_audio_files(dir_path, exts=(".wav", ".flac"))
        # 注意：CSV 列表写到数据根目录（与 main 读取路径保持一致）。
        # 若数据目录为只读挂载，会在此处 PermissionError，
        # 请改用可写副本或在有写权限的位置手动执行 --scan-and-build-csv。
        csv_path = root / f"{key}_file_list.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_dir"])
            for fp in files:
                writer.writerow([str(fp).replace("\\", "/")])
        logger.info(f"  {key}: {len(files)} files -> {csv_path}")

    logger.info("CSV list generation complete.")


def add_pyreverb(clean_speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve clean speech with RIR to generate reverberant speech (same length as input)."""
    reverb = signal.fftconvolve(clean_speech, rir, mode="full")
    return reverb[: clean_speech.shape[0]]


def mk_mixture(
    s1: np.ndarray,
    s2: np.ndarray,
    s1_ref: np.ndarray,
    snr: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix reverberant speech with noise.

    Args:
        s1:     Reverberant speech (full reverb)
        s2:     Noise signal
        s1_ref: Low-reverb speech (early reflection target)
        snr:    Target SNR (dB)

    Returns:
        (mixture, target): Noisy mixture and training target
    """
    amp = 0.5 * np.random.rand() + 0.01
    s1_ref = amp * s1_ref / (np.max(np.abs(s1)) + eps)
    s1 = amp * s1 / (np.max(np.abs(s1)) + eps)

    norm_sig2 = s2 * np.sqrt(np.sum(s1**2) + eps) / np.sqrt(np.sum(s2**2) + eps)
    alpha = 10 ** (-snr * 1.5 / 20)

    mix = s1 + alpha * norm_sig2

    M = max(np.max(np.abs(mix)), np.max(np.abs(s1)), np.max(np.abs(alpha * norm_sig2))) + eps
    if M > 1.0:
        mix = mix / M
        s1 = s1 / M
        s1_ref = s1_ref / M

    return mix, s1_ref


def process_one(
    idx: int,
    clean_list: list[str],
    noise_list: list[str],
    rir_list: list[str],
    snr_list: np.ndarray,
    save_dir_noisy: str,
    save_dir_clean: str,
    nfill: int,
    fs: int,
    wav_len: int,
    random_start: bool,
    seed: int = SEED,
) -> tuple[int, bool, str]:
    """Process one sample (worker function).

    Returns:
        (idx, success, error_msg)
    """
    # 每个 worker 独立进程时 numpy 随机状态来自 OS 熵；按 idx+seed 重新播种保证可复现
    np.random.seed((seed + idx) % (2 ** 32))
    try:
        total_samples = wav_len * fs

        clean_path = clean_list[idx % len(clean_list)]
        noise_path = noise_list[idx % len(noise_list)]
        rir_path = rir_list[idx % len(rir_list)]
        snr = snr_list[idx]

        clean_info = sf.info(clean_path)
        noise_info = sf.info(noise_path)

        if random_start and clean_info.frames > total_samples:
            max_start = max(0, clean_info.frames - total_samples)
            start_s = np.random.randint(0, max_start + 1)
        else:
            start_s = 0

        if random_start and noise_info.frames > total_samples:
            max_start_n = max(0, noise_info.frames - total_samples)
            start_n = np.random.randint(0, max_start_n + 1)
        else:
            start_n = 0

        clean = sf.read(clean_path, dtype="float32", start=start_s, stop=start_s + total_samples)[0]
        noise = sf.read(noise_path, dtype="float32", start=start_n, stop=start_n + total_samples)[0]
        rir_data = sf.read(rir_path, dtype="float32")[0]

        if clean.ndim > 1:
            clean = clean[:, 0]
        if noise.ndim > 1:
            noise = noise[:, 0]
        if rir_data.ndim > 1:
            rir_data = rir_data[:, 0]

        if len(clean) < total_samples:
            clean = np.pad(clean, (0, total_samples - len(clean)), mode="wrap")
        if len(noise) < total_samples:
            noise = np.pad(noise, (0, total_samples - len(noise)), mode="wrap")

        max_index = np.argmax(np.abs(rir_data))
        rir_full = rir_data[max_index:]
        early_samples = min(int(EARLY_RIR_MS * fs / 1000), len(rir_full))
        rir_early = rir_full[:early_samples]

        rev_clean = add_pyreverb(clean, rir_full)
        drb_clean = add_pyreverb(clean, rir_early)

        mixture, target = mk_mixture(rev_clean, noise, drb_clean, snr)

        fname = str(idx + 1).zfill(nfill) + ".wav"
        sf.write(os.path.join(save_dir_noisy, fname), mixture, fs)
        sf.write(os.path.join(save_dir_clean, fname), target, fs)

        return (idx, True, "")

    except Exception as e:
        return (idx, False, str(e))


# 模块级 worker 全局状态：由 mp.Pool initializer 一次性注入，
# 避免 partial 把完整文件列表随每个任务 pickle（数据量大时 IPC 开销显著）
_WORKER_ARGS: dict = {}


def _init_worker(
    clean_list: list[str],
    noise_list: list[str],
    rir_list: list[str],
    snr_list: np.ndarray,
    save_dir_noisy: str,
    save_dir_clean: str,
    nfill: int,
    fs: int,
    wav_len: int,
    random_start: bool,
    seed: int,
) -> None:
    """mp.Pool worker 初始化：把大列表等只读参数注入各 worker 进程。"""
    _WORKER_ARGS.update(
        clean_list=clean_list,
        noise_list=noise_list,
        rir_list=rir_list,
        snr_list=snr_list,
        save_dir_noisy=save_dir_noisy,
        save_dir_clean=save_dir_clean,
        nfill=nfill,
        fs=fs,
        wav_len=wav_len,
        random_start=random_start,
        seed=seed,
    )


def _process_index(idx: int) -> tuple[int, bool, str]:
    """worker 入口：从全局状态取参数后调用 process_one。"""
    return process_one(idx, **_WORKER_ARGS)


def generate_split(
    split: str,
    num: int,
    clean_list: list[str],
    noise_list: list[str],
    rir_list: list[str],
    output_dir: str,
    fs: int = FS,
    wav_len: int = WAV_LEN_SEC,
    num_workers: int = 4,
    seed: int = SEED,
    write_info_csv: bool = False,
):
    """Generate one data split (train/val/test)."""
    # 使用稳定哈希替代 Python 内置 hash()，避免 PYTHONHASHSEED 导致跨运行不可复现
    split_hash = int(hashlib.md5(split.encode("utf-8")).hexdigest(), 16)
    rng = np.random.RandomState(seed + split_hash % 10000)
    snr_list = rng.uniform(SNR_RANGE[0], SNR_RANGE[1], size=num)

    indices = list(range(len(clean_list)))
    rng.shuffle(indices)
    shuffled_clean = [clean_list[indices[i % len(clean_list)]] for i in range(num)]

    indices_n = list(range(len(noise_list)))
    rng.shuffle(indices_n)
    shuffled_noise = [noise_list[indices_n[i % len(noise_list)]] for i in range(num)]

    indices_r = list(range(len(rir_list)))
    rng.shuffle(indices_r)
    shuffled_rir = [rir_list[indices_r[i % len(rir_list)]] for i in range(num)]

    save_noisy = os.path.join(output_dir, f"{split}_noisy")
    save_clean = os.path.join(output_dir, f"{split}_clean")
    os.makedirs(save_noisy, exist_ok=True)
    os.makedirs(save_clean, exist_ok=True)

    nfill = len(str(num))
    random_start = (split == "train")

    logger.info(f"[{split}] Generating {num} pairs -> {output_dir}/{split}_*")

    if num_workers <= 1:
        from tqdm import tqdm
        success, fail = 0, 0
        for idx in tqdm(range(num), desc=split):
            _, ok, err = process_one(
                idx, shuffled_clean, shuffled_noise, shuffled_rir, snr_list,
                save_noisy, save_clean, nfill, fs, wav_len, random_start,
                seed=seed + split_hash,
            )
            if ok:
                success += 1
            else:
                fail += 1
                if fail <= 10:
                    logger.warning(f"  [{split}] idx={idx} failed: {err}")
    else:
        from tqdm import tqdm
        success, fail = 0, 0
        with mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(
                shuffled_clean, shuffled_noise, shuffled_rir, snr_list,
                save_noisy, save_clean, nfill, fs, wav_len, random_start,
                seed + split_hash,
            ),
        ) as pool:
            for _, ok, err in tqdm(
                pool.imap_unordered(_process_index, range(num)),
                total=num,
                desc=split,
            ):
                if ok:
                    success += 1
                else:
                    fail += 1
                    if fail <= 10:
                        logger.warning(f"  [{split}] failed: {err}")

    logger.info(f"  [{split}] Done: {success} success, {fail} failed")

    if write_info_csv:
        info_path = os.path.join(output_dir, f"{split}_INFO.csv")
        with open(info_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "clean", "noise", "snr"])
            for idx in range(num):
                fname = str(idx + 1).zfill(nfill) + ".wav"
                writer.writerow([fname, shuffled_clean[idx], shuffled_noise[idx], f"{snr_list[idx]:.2f}"])
        logger.info(f"  [{split}] INFO -> {info_path}")


def _path(p: Path) -> str:
    return str(p).replace("\\", "/")


def generate_scp(
    output_dir: str,
    scp_dir: str,
    fs: int = FS,
    *,
    waveform_root_alias: str | None = None,
):
    """为 use_train 生成严格四列 clean/noisy SCP；默认拒绝 unmatched。"""
    scp_path = Path(scp_dir)
    waveform_root = Path(output_dir).resolve(strict=True)
    scp_path.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        clean_dir = Path(output_dir) / f"{split}_clean"
        noisy_dir = Path(output_dir) / f"{split}_noisy"

        if clean_dir.exists() != noisy_dir.exists():
            raise ValueError(
                f"[SCP] {split}: clean/noisy split directories must both exist or both be absent"
            )
        if not clean_dir.exists():
            continue

        clean_by_stem = {fp.stem: fp for fp in clean_dir.glob("*.wav")}
        noisy_by_stem = {fp.stem: fp for fp in noisy_dir.glob("*.wav")}
        common_stems = sorted(set(clean_by_stem) & set(noisy_by_stem))

        if len(common_stems) != len(clean_by_stem) or len(common_stems) != len(noisy_by_stem):
            raise ValueError(
                f"[SCP] {split}: clean/noisy must match 1:1: "
                f"clean={len(clean_by_stem)}, noisy={len(noisy_by_stem)}, "
                f"matched={len(common_stems)}"
            )
        if not common_stems:
            logger.warning(f"[SCP] {split}: no matched clean/noisy pairs, skipping")
            continue

        split_dir = scp_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        clean_entries: list[tuple[str, int, int, str]] = []
        noisy_entries: list[tuple[str, int, int, str]] = []
        for uid in common_stems:
            clean_fp = clean_by_stem[uid]
            noisy_fp = noisy_by_stem[uid]
            clean_meta = probe_audio_meta(clean_fp, fallback_sr=fs)
            noisy_meta = probe_audio_meta(noisy_fp, fallback_sr=fs)
            if clean_meta["frames"] <= 0 or noisy_meta["frames"] <= 0:
                raise ValueError(f"[SCP] {split}: invalid audio metadata for uid={uid}")
            clean_relative = clean_fp.resolve(strict=True).relative_to(
                waveform_root
            ).as_posix()
            noisy_relative = noisy_fp.resolve(strict=True).relative_to(
                waveform_root
            ).as_posix()
            clean_path = (
                f"{waveform_root_alias}:{clean_relative}"
                if waveform_root_alias is not None
                else _path(clean_fp)
            )
            noisy_path = (
                f"{waveform_root_alias}:{noisy_relative}"
                if waveform_root_alias is not None
                else _path(noisy_fp)
            )
            clean_entries.append(
                (uid, clean_meta["sr"], clean_meta["frames"], clean_path)
            )
            noisy_entries.append(
                (uid, noisy_meta["sr"], noisy_meta["frames"], noisy_path)
            )

        write_scp_entries(clean_entries, split_dir / "clean.scp", four_col=True)
        write_scp_entries(noisy_entries, split_dir / "noisy.scp", four_col=True)
        logger.info(
            f"  [SCP] {split}: {len(common_stems)} entries -> {split_dir} "
            "(four-column clean.scp, noisy.scp)"
        )


def run_generation(config: DNS3GenerationConfig) -> None:
    """按显式配置扫描 source lists，并生成 waveform 与可选 SCP。"""
    logger.info("=== DNS Challenge 3 offline data generation ===")
    logger.warning(
        "Data split note: train/val/test 均从同一 clean/noise/rir 池采样"
        "（无源级 hold-out，与 GTCRN 论文一致），val/test 指标可能偏乐观，"
        "仅用于复现论文结果，不代表真实泛化性能"
    )
    logger.info(f"Raw data: {config.data_root}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Params: fs={config.fs}, wav_len={config.wav_len}s, SNR={SNR_RANGE}")
    logger.info(
        f"Counts: train={config.num_train}, val={config.num_val}, "
        f"test={config.num_test}"
    )

    check_data_root(str(config.data_root))
    clean_csv = config.data_root / "clean_file_list.csv"
    noise_csv = config.data_root / "noise_file_list.csv"
    rir_csv = config.data_root / "rir_file_list.csv"
    if not all(path.exists() for path in (clean_csv, noise_csv, rir_csv)):
        logger.info("CSV file lists do not exist, auto-scanning...")
        build_csv_lists(str(config.data_root))

    try:
        import pandas as pd
    except ImportError as exc:
        logger.error(
            "pandas is required for DNS3 offline generation. "
            "Please install it first: pip install pandas"
        )
        raise ImportError("pandas is required") from exc
    clean_list = pd.read_csv(clean_csv)["file_dir"].tolist()
    noise_list = pd.read_csv(noise_csv)["file_dir"].tolist()
    rir_list = pd.read_csv(rir_csv)["file_dir"].tolist()
    logger.info(f"Clean: {len(clean_list)}, Noise: {len(noise_list)}, RIR: {len(rir_list)}")
    if not clean_list or not noise_list or not rir_list:
        raise ValueError(
            "Data file lists are empty. Please confirm data is downloaded and extracted correctly."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    generated_splits: list[str] = []
    split_counts = (
        ("train", config.num_train),
        ("val", config.num_val),
        ("test", config.num_test),
    )
    for split, num in split_counts:
        if num <= 0:
            continue
        generate_split(
            split,
            num,
            clean_list,
            noise_list,
            rir_list,
            str(config.output_dir),
            config.fs,
            config.wav_len,
            config.num_workers,
            config.seed,
            write_info_csv=config.write_info_csv,
        )
        generated_splits.append(split)

    if config.scp_dir is not None:
        generate_scp(
            str(config.output_dir),
            str(config.scp_dir),
            config.fs,
        )

    logger.info("=== Generation complete ===")
    if generated_splits:
        num_by_split = dict(split_counts)
        logger.info("Output directory structure:")
        logger.info(f"  {config.output_dir}/")
        for split in generated_splits:
            logger.info(f"  ├── {split}_clean/  ({num_by_split[split]} files)")
            logger.info(f"  ├── {split}_noisy/  ({num_by_split[split]} files)")
        if config.write_info_csv:
            for split in generated_splits:
                logger.info(f"  └── {split}_INFO.csv")
    else:
        logger.info("No split generated (all split counts <= 0)")
