#!/usr/bin/env python3
"""对四个 REAL-T 官方 checkpoint 执行完整 TSEExtractor 端到端验证。

每个变体使用独立 use_train infer YAML，经安全 loader、完整 state 预校验、
STFT/TSE/iSTFT 和真实 REAL-T 波形前向。重复前向必须逐值确定，输出必须有限、
等长且满足配置的 0.9 峰值归一。官方主 checkpoint 内含 speaker encoder，验证
期间不得加载独立 ECAPA artifact。

Usage::

    cd use_train
    python tools/verify/verify_tse_official_ckpts.py \
        --zip /tmp/real-tse-ckpt.zip \
        --ckpt_root tasks/tse/pretrained_models
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import stat
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torchaudio

_VARIANTS = {
    "spk_emb_100": "tse_bsrnn_spk_realt_baseline_infer.yaml",
    "spk_emb_causal_100": "tse_bsrnn_spk_realt_baseline_causal_infer.yaml",
    "tfmap_context_100": "tse_bsrnn_spk_realt_tfmap_context_infer.yaml",
    "tfmap_context_causal_100": "tse_bsrnn_spk_realt_tfmap_context_causal_infer.yaml",
}


def _unzip_if_needed(zip_path: str | None, ckpt_root: Path) -> None:
    """官方 zip 解压到 ckpt_root（已存在同名目录则跳过）。

    越界成员路径与 symlink 成员一律拒绝；stdlib 支持 ``filter="data"`` 时叠加使用。
    """
    if not zip_path:
        return
    z = Path(zip_path)
    if not z.is_file():
        raise FileNotFoundError(f"zip 不存在: {z}")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        root = ckpt_root.resolve()
        for member in zf.infolist():
            destination = (root / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"zip 成员越界: {member.filename!r}")
        top = {n.split("/")[0] for n in zf.namelist() if n.strip("/")}
        print(f"[INFO] zip 顶层目录: {sorted(top)}")
        # 上方成员路径校验只约束常规文件；symlink 成员仍能指向目录之外。
        # 逐成员显式拒绝 symlink（全版本生效），并在 stdlib 支持时再用
        # filter="data" 兜底（部分发行版 stdlib 无该参数，故按签名探测）
        for member in zf.infolist():
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"zip 成员为 symlink，拒绝解压: {member.filename!r}")
        if "filter" in inspect.signature(zipfile.ZipFile.extractall).parameters:
            zf.extractall(ckpt_root, filter="data")
        else:
            zf.extractall(ckpt_root)
    print(f"[INFO] 已解压到 {ckpt_root}")


def _load_real_trial(realt_root: Path, num_samples: int):
    """从排序后的首条 DEV meta 读取一个稳定 mix/enrollment 片段。"""
    meta_paths = sorted((realt_root / "DEV").glob("*_meta.csv"))
    if not meta_paths:
        raise FileNotFoundError(f"REAL-T DEV meta 不存在: {realt_root / 'DEV'}")
    with meta_paths[0].open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"REAL-T DEV meta 为空: {meta_paths[0]}")
    mix_path = realt_root / "mixtures" / f"{row['mixture_utterance']}.wav"
    enroll_path = (
        realt_root
        / "enrolment_speakers"
        / f"{row['enrolment_speakers_utterance']}.wav"
    )
    mix, mix_sr = torchaudio.load(mix_path)
    enroll, enroll_sr = torchaudio.load(enroll_path)
    if mix_sr != 16000 or enroll_sr != 16000:
        raise ValueError(
            f"REAL-T trial 必须为 16kHz，got mix={mix_sr}, enroll={enroll_sr}"
        )
    return (
        mix.mean(dim=0)[:num_samples].contiguous(),
        enroll.mean(dim=0)[:num_samples].contiguous(),
        mix_path,
        enroll_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="任一 checkpoint/config/state/真实波形/确定性 gate 失败均返回非零。",
    )
    ap.add_argument("--zip", default=None, help="官方 real-tse-ckpt.zip 路径（可选，已解压则省略）")
    ap.add_argument("--ckpt_root", default="tasks/tse/pretrained_models", help="ckpt 根目录")
    ap.add_argument(
        "--realt_root",
        default="data/tse/REAL-TSE/REAL-T-dev",
        help="REAL-T DEV 根目录",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=4096,
        help="真实 mix/enrollment 各取的确定性前缀样本数",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.num_samples < 400:
        ap.error("--num_samples must be >= 400 for speaker fbank")

    from tasks.tse.inference import TSEExtractor

    ckpt_root = Path(args.ckpt_root)
    _unzip_if_needed(args.zip, ckpt_root)

    mix, enroll, mix_path, enroll_path = _load_real_trial(
        Path(args.realt_root), args.num_samples
    )
    config_root = Path(_REPO_ROOT) / "configs/tse/inference"
    print(f"[INPUT] mix={mix_path} enroll={enroll_path} samples={mix.numel()}")

    ok = True
    for name, config_name in _VARIANTS.items():
        ckpt = ckpt_root / name / "avg_model.pt"
        config = config_root / config_name
        if not config.is_file():
            print(f"[MISS] {name}: 配置 {config} 不存在")
            ok = False
            continue
        if not ckpt.is_file():
            print(f"[MISS] {name}: {ckpt} 不存在")
            ok = False
            continue
        extractor = TSEExtractor(
            str(ckpt), config_path=str(config), device=args.device
        )
        first = extractor.extract_speech_from_pcm(mix, 16000, enroll, 16000)
        second = extractor.extract_speech_from_pcm(mix, 16000, enroll, 16000)
        finite = bool(np.isfinite(first).all())
        deterministic = bool(np.array_equal(first, second))
        expected_peak = 0.9 if np.any(first) else 0.0
        peak = float(np.max(np.abs(first))) if first.size else 0.0
        passed = (
            finite
            and deterministic
            and first.shape == (mix.numel(),)
            and abs(peak - expected_peak) <= 1e-5
            and extractor.external_speaker_encoder_pretrained is None
        )
        print(
            f"[E2E] {name}: keys={len(extractor.model.state_dict())} "
            f"out={first.shape} finite={finite} deterministic={deterministic} "
            f"peak={peak:.6f} passed={passed}"
        )
        ok = ok and passed

    print("[RESULT]", "四 checkpoint 端到端验证通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
