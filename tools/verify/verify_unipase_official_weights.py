#!/usr/bin/env python3
"""端到端验证 UniPASE 移植与官方模型输出一致性。

对照对象：通过 `--official_repo` 或 `UNIPASE_SOURCE_REPO` 指定的官方 repo
（只读参考）及其 demo 音频 `audio/noisy/* -> audio/enh/*`。

验证内容（复现官方 inference/inference.py 的 demo 设置：sr_out=输入采样率、
enable_plc=True、输出峰值归一化到输入峰值）：

1. 端到端：本框架 UniPASE 对 audio/noisy/* 推理，与官方 audio/enh/* 对比，
   报告 max_abs_diff / SNR / SI-SNR（双方都做了输入峰值归一化，波形可直接比较）。
2. 与官方代码现场推理结果对比（隔离"移植 vs 官方代码"与
   "官方代码 vs 存档 demo 文件"两种误差来源）。
3. 组件级 parity：同 checkpoint + 同随机输入，直接对比本框架与官方 repo 的
   VocosAdapter / VocosVocoder / TFGridNet(PostNet) 输出（隔离 WavLM 差异）。
4. --dump-intermediates：逐阶段（encoder L1/L24、adapter、vocoder、postnet）
   对比双方中间激活，定位分歧位置。

用法：
    python tools/verify/verify_unipase_official_weights.py
    python tools/verify/verify_unipase_official_weights.py --dump-intermediates --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# 项目根目录加入 sys.path，保证可以 import tasks.se.models...
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 官方 repo 只读对照路径必须由调用方显式提供，避免绑定本机目录。
DEFAULT_OFFICIAL_REPO = os.environ.get("UNIPASE_SOURCE_REPO")
DEFAULT_CKPT_DIR = "tasks/se/models/pase_family/unipase/open_model_checkpoint"

CKPT_FILES = {
    "dewavlm": "DeWavLM-Omni.pt",
    "adapter": "Adapter.pt",
    "vocoder": "Vocoder_DWO-L1.pt",
    "postnet": "PostNet.pt",
}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """SNR = 10·log10(||ref||² / ||ref - est||²)。"""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    est = np.asarray(est, dtype=np.float64).ravel()
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    err = ref - est
    denom = float(np.sum(err * err)) + 1e-12
    return float(10.0 * np.log10(np.sum(ref * ref) / denom + 1e-12))


def _si_snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """SI-SNR（尺度不变），float64 计算。"""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    est = np.asarray(est, dtype=np.float64).ravel()
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    dot = float(np.dot(est, ref))
    ref_energy = float(np.dot(ref, ref)) + 1e-12
    target = (dot / ref_energy) * ref
    noise = est - target
    return float(10.0 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-12) + 1e-12))


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    return float(np.max(np.abs(a[:n] - b[:n])))


def _report_pair(name: str, ref: np.ndarray, est: np.ndarray) -> dict:
    """打印一组 ref/est 对比指标并返回 dict。"""
    m = {
        "name": name,
        "max_abs_diff": _max_abs_diff(ref, est),
        "snr_db": _snr_db(ref, est),
        "si_snr_db": _si_snr_db(ref, est),
    }
    print(
        f"  [{name}] max_abs_diff={m['max_abs_diff']:.3e}  "
        f"SNR={m['snr_db']:.2f} dB  SI-SNR={m['si_snr_db']:.2f} dB"
    )
    return m


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------

def _build_ours(ckpt_dir: Path, device: torch.device):
    """用官方 checkpoint 构建本框架 UniPASE。"""
    from tasks.se.models.pase_family.unipase.unipase_model import UniPASE

    model = UniPASE(
        dewavlm_ckpt_path=str(ckpt_dir / CKPT_FILES["dewavlm"]),
        adapter_ckpt_path=str(ckpt_dir / CKPT_FILES["adapter"]),
        vocoder_ckpt_path=str(ckpt_dir / CKPT_FILES["vocoder"]),
        postnet_ckpt_path=str(ckpt_dir / CKPT_FILES["postnet"]),
        sr_out=16000,  # 逐文件覆盖为输入采样率
        enable_plc=True,
    ).to(device).eval()
    return model


def _build_official(ckpt_dir: Path, official_repo: Path, device: torch.device):
    """通过 sys.path 注入官方 repo，构建官方 UniPASE。"""
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    # 官方 inference 以 repo 根为 import 根（from models.unipase import UniPASE）
    from models.unipase import UniPASE as OfficialUniPASE  # noqa: E402

    model = OfficialUniPASE(
        dewavlm_ckpt_path=str(ckpt_dir / CKPT_FILES["dewavlm"]),
        adapter_ckpt_path=str(ckpt_dir / CKPT_FILES["adapter"]),
        vocoder_ckpt_path=str(ckpt_dir / CKPT_FILES["vocoder"]),
        postnet_ckpt_path=str(ckpt_dir / CKPT_FILES["postnet"]),
    ).to(device).eval()
    return model


# ---------------------------------------------------------------------------
# 推理（复现官方 inference_file 行为）
# ---------------------------------------------------------------------------

def _run_ours(model, audio: np.ndarray, fs: int, device: torch.device) -> np.ndarray:
    """本框架前向；forward 内部已做输入峰值归一化。"""
    x = torch.from_numpy(audio).float()[None].to(device)
    model.sr_out = fs  # 官方 demo：sr_out = 输入采样率
    with torch.inference_mode():
        y = model(x, fs=fs)
    return y.squeeze(0).cpu().numpy()


def _run_official(model, audio: np.ndarray, fs: int, device: torch.device) -> np.ndarray:
    """官方前向 + 官方 inference/inference.py:84-85 的峰值归一化。"""
    x = torch.from_numpy(audio).float()[None].to(device)
    with torch.inference_mode():
        y = model(x, sr_in=fs, sr_out=fs, enable_plc=True)
    y = y.cpu().numpy().squeeze()
    scale = np.max(np.abs(audio))
    y = y / (np.max(np.abs(y)) + 1e-8) * scale
    return y


# ---------------------------------------------------------------------------
# 组件级 parity
# ---------------------------------------------------------------------------

def _component_parity(ckpt_dir: Path, official_repo: Path, device: torch.device) -> None:
    """同 checkpoint + 同随机输入，对比官方与本框架的 Adapter/Vocoder/PostNet。"""
    if str(official_repo) not in sys.path:
        sys.path.insert(0, str(official_repo))
    from models.adapter.vocos.adapter import VocosAdapter as OffAdapter  # noqa: E402
    from models.vocoder.vocos.vocoder import VocosVocoder as OffVocoder  # noqa: E402
    from models.postnet.tfgrid_cws_res import TFGridNet as OffPostNet  # noqa: E402
    from tasks.se.models.pase_family.unipase.adapter import VocosAdapter  # noqa: E402
    from tasks.se.models.pase_family.unipase.vocoder import VocosVocoder  # noqa: E402
    from tasks.se.models.pase_family.unipase.postnet import TFGridNet  # noqa: E402

    torch.manual_seed(0)
    print("\n=== 组件级 parity（同 checkpoint、同随机输入） ===")

    # --- Adapter ---
    off = OffAdapter.from_pretrained(str(ckpt_dir / CKPT_FILES["adapter"])).to(device)
    ours = VocosAdapter.from_pretrained(str(ckpt_dir / CKPT_FILES["adapter"])).to(device)
    ea = torch.randn(1, 100, 1024, device=device)
    ep = torch.randn(1, 100, 1024, device=device)
    with torch.inference_mode():
        d = (off(ea, ep) - ours(ea, ep)).abs().max().item()
    print(f"  [VocosAdapter] max_abs_diff={d:.3e}")

    # --- Vocoder ---
    off = OffVocoder.from_pretrained(str(ckpt_dir / CKPT_FILES["vocoder"])).to(device)
    ours = VocosVocoder.from_pretrained(str(ckpt_dir / CKPT_FILES["vocoder"])).to(device)
    feat = torch.randn(1, 1024, 100, device=device)
    with torch.inference_mode():
        yo = off(feat)
        ym = ours(feat)
    n = min(yo.shape[-1], ym.shape[-1])
    d = (yo[..., :n] - ym[..., :n]).abs().max().item()
    print(f"  [VocosVocoder] max_abs_diff={d:.3e}  (len off={yo.shape[-1]}, ours={ym.shape[-1]})")

    # --- PostNet (TFGridNet.infer) ---
    off = OffPostNet.from_pretrained(str(ckpt_dir / CKPT_FILES["postnet"])).to(device)
    ours = TFGridNet.from_pretrained(str(ckpt_dir / CKPT_FILES["postnet"])).to(device)
    wav16k = torch.randn(1, 16000, device=device) * 0.1
    with torch.inference_mode():
        yo = off.infer(wav16k, sr_in=16000, sr_out=48000)
        ym = ours.infer(wav16k, sr_in=16000, sr_out=48000)
    n = min(yo.shape[-1], ym.shape[-1])
    d = (yo[..., :n] - ym[..., :n]).abs().max().item()
    print(f"  [TFGridNet.infer 16k->48k] max_abs_diff={d:.3e}  (len off={yo.shape[-1]}, ours={ym.shape[-1]})")


# ---------------------------------------------------------------------------
# 中间激活对比
# ---------------------------------------------------------------------------

def _dump_intermediates(ours, official, audio: np.ndarray, fs: int, device: torch.device) -> None:
    """逐阶段对比 encoder(L1/L24) / adapter / vocoder / postnet 中间激活。"""
    x = torch.from_numpy(audio).float()[None].to(device)

    with torch.inference_mode():
        fa_off, fp_off = official.encoder(x, sr=fs, mask=True)
        fa_our, fp_our = ours.encoder(x, sr=fs, mask=True)
        for tag, a, b in [
            ("encoder L1 (noisy_feat_a)", fa_off, fa_our),
            ("encoder L24 (enh_feat_p)", fp_off, fp_our),
        ]:
            n_t = min(a.shape[1], b.shape[1])
            d = (a[:, :n_t] - b[:, :n_t]).abs().max().item()
            print(f"  [{tag}] shape off={tuple(a.shape)} ours={tuple(b.shape)} max_abs_diff={d:.3e}")

        ga_off = official.adapter(fa_off, fp_off)
        ga_our = ours.adapter(fa_our, fp_our)
        n_t = min(ga_off.shape[1], ga_our.shape[1])
        d = (ga_off[:, :n_t] - ga_our[:, :n_t]).abs().max().item()
        print(f"  [adapter out] shape off={tuple(ga_off.shape)} ours={tuple(ga_our.shape)} max_abs_diff={d:.3e}")

        w_off = official.decoder(ga_off.transpose(1, 2))
        w_our = ours.decoder(ga_our.transpose(1, 2))
        n = min(w_off.shape[-1], w_our.shape[-1])
        d = (w_off[..., :n] - w_our[..., :n]).abs().max().item()
        print(f"  [vocoder out 16k] len off={w_off.shape[-1]} ours={w_our.shape[-1]} max_abs_diff={d:.3e}")

        p_off = official.postnet.infer(w_off, sr_in=16000, sr_out=fs)
        p_our = ours.postnet.infer(w_our, sr_in=16000, sr_out=fs)
        n = min(p_off.shape[-1], p_our.shape[-1])
        d = (p_off[..., :n] - p_our[..., :n]).abs().max().item()
        print(f"  [postnet out {fs}Hz] len off={p_off.shape[-1]} ours={p_our.shape[-1]} max_abs_diff={d:.3e}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="官方 checkpoint 目录")
    parser.add_argument(
        "--official_repo",
        default=DEFAULT_OFFICIAL_REPO,
        help="官方 repo 路径（只读；也可设置 UNIPASE_SOURCE_REPO）",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dump-intermediates", action="store_true",
                        help="逐阶段对比中间激活（首个 demo 文件）")
    parser.add_argument("--skip-official", action="store_true",
                        help="不运行官方代码（仅对比存档 enh 文件）")
    args = parser.parse_args()
    if not args.official_repo:
        parser.error("必须通过 --official_repo 或 UNIPASE_SOURCE_REPO 提供官方 repo 路径")

    ckpt_dir = Path(args.ckpt_dir)
    official_repo = Path(args.official_repo)
    noisy_dir = official_repo / "audio" / "noisy"
    enh_dir = official_repo / "audio" / "enh"
    device = torch.device(args.device)

    for k, f in CKPT_FILES.items():
        if not (ckpt_dir / f).is_file():
            raise FileNotFoundError(f"缺少 checkpoint: {ckpt_dir / f}")
    print(f"[*] device={device}  ckpt_dir={ckpt_dir}  official_repo={official_repo}")

    ours = _build_ours(ckpt_dir, device)
    official = None if args.skip_official else _build_official(ckpt_dir, official_repo, device)

    files = sorted(p for p in noisy_dir.iterdir() if p.suffix.lower() in (".wav", ".flac"))
    if not files:
        raise FileNotFoundError(f"{noisy_dir} 下没有音频文件")

    print("\n=== 端到端对比（官方 demo 设置：sr_out=输入采样率, enable_plc=True） ===")
    results = []
    live_results = []
    for noisy_path in files:
        enh_path = enh_dir / noisy_path.name
        audio, fs = sf.read(str(noisy_path), dtype="float32")
        print(f"\n[*] {noisy_path.name}  sr={fs}  n={audio.size} ({audio.size / fs:.2f}s)")

        y_ours = _run_ours(ours, audio, fs, device)

        if enh_path.is_file():
            y_ref, fs_ref = sf.read(str(enh_path), dtype="float32")
            if fs_ref != fs:
                raise ValueError(f"enh 采样率不匹配: {fs_ref} vs {fs}")
            m = _report_pair("ours vs 官方存档 enh", y_ref, y_ours)
            m["file"] = noisy_path.name
            results.append(m)
        else:
            print(f"  [WARN] 无对应 enh 文件: {enh_path}")

        if official is not None:
            y_off = _run_official(official, audio, fs, device)
            m = _report_pair("ours vs 官方代码现场输出", y_off, y_ours)
            m["file"] = noisy_path.name
            live_results.append(m)

    if results:
        print("\n=== 汇总（ours vs 官方存档 enh，仅作参考） ===")
        for m in results:
            print(f"  {m['file']}: SI-SNR={m['si_snr_db']:.2f} dB, max_abs_diff={m['max_abs_diff']:.3e}")
        # 注意：官方存档 enh 是发布时生成的，与当前官方代码/设置存在已知偏差
        # （如长音频分块），不作为判定门槛；判定以「vs 官方代码现场输出」为准。

    # 判定：与官方代码现场输出应接近逐比特一致（SI-SNR > 60 dB）；
    # 非零退出让 CI/脚本调用方能捕获回归（review P3-8）
    if live_results:
        worst_live = min(live_results, key=lambda m: m["si_snr_db"])
        verdict = "PASS" if worst_live["si_snr_db"] > 60.0 else "BELOW TARGET"
        print(f"[*] 判定（vs 官方代码现场输出）: 最差 SI-SNR = "
              f"{worst_live['si_snr_db']:.2f} dB ({worst_live['file']})，目标 >60 dB -> {verdict}")
        if worst_live["si_snr_db"] <= 60.0:
            print("[ERROR] 与官方代码输出偏差超阈，判定失败", file=sys.stderr)
            sys.exit(1)
    elif args.skip_official:
        # 跳过官方现场对比时没有判定基准：退出码 0 不代表 PASS，必须显式说明
        print(
            "[WARN] --skip-official：未执行官方代码现场对比，本次运行不构成 PASS 判定",
            file=sys.stderr,
        )
    else:
        print(
            "[ERROR] 未产生任何「ours vs 官方代码现场输出」对比结果，无法给出判定",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dump_intermediates and official is not None:
        print(f"\n=== 中间激活对比（{files[0].name}） ===")
        audio, fs = sf.read(str(files[0]), dtype="float32")
        _dump_intermediates(ours, official, audio, fs, device)

    if not args.skip_official:
        _component_parity(ckpt_dir, official_repo, device)


if __name__ == "__main__":
    main()
