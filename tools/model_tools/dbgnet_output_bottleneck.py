"""JAE-M08：DBGNet 输出表达瓶颈实验 harness（只出证据，不改模型架构）。

背景
----
审计 JAE-M08 指出 ``tasks/aec/models/discriminative/dbgnet.py::forward_spectra``
（574-583 行）的最终输出为 ``m = sigmoid(y)``（ERB 域单通道实数 mask，
bound [0,1]），``enhanced = mic_complex * m[..., None]``。实值增益 m∈[0,1]
意味着模型**不能旋转相位、也不能放大**。参考复现 DBGNet（github 工作副本）
network.py:343-352 的输出表达与本仓库
完全一致（同为 sigmoid real mask），因此该瓶颈不是移植引入的偏差。

审计要求：先以 reference sensitivity 与 oracle mask 实验证明瓶颈，再决定是否上
complex ratio mask 或 bounded residual head。本脚本即该证据实验。

四个对照（复谱 (B,F,T,2)）
--------------------------
A. model            ：模型原始输出。
B. oracle_real_mask ：m* = clamp(|near|/(|mic|+1e-8), 0, 1)，enh = mic × m*。
                      与模型相同表达形式（实数 [0,1] mask）的估计上限。
C. oracle_crm       ：m_c = near/(mic+1e-8)（复数除法，无界），enh = mic × m_c。
                      数学上应精确还原 nearend，作为表达上限 sanity check。
D. reference sensitivity：模型输出在 (a) 正确 far、(b) zeros far、
                      (c) far 循环移位 800 样点（50ms）三种 reference 下的对比。

判定
----
- ``expression_gap_db = mean(C - B)``：oracle CRM 显著高于 oracle real mask
  → 表达受限证据（实数 mask 形式本身丢失了相位/放大能力）。
- ``estimation_gap_db = mean(B - A)``：A 接近 B → 瓶颈在估计不在表达；
  A 远落后于 B → 即使同一表达形式，估计器也远未达上限。
- ``reference_sensitivity_db = mean(D_a - D_b)``：模型对 reference 的依赖程度。

用法
----
  python tools/model_tools/dbgnet_output_bottleneck.py \
      --checkpoint <best_model.tar> --config <resolved_config.yaml> \
      --data-dir <含 mic.wav/nearend.wav/farend.wav 的样本目录> \
      --device cpu --num-samples 4 --out report.json

用户决策（2026-07-15）
----------------------
实验结论：estimation_gap 仅 ~1.1 dB（模型 9.53 vs oracle real-mask 10.91），
瓶颈在输出表达形式而非估计器；reference_sensitivity ~4.3 dB，模型确实利用
reference。用户据此决策**不修改模型架构**（维持 sigmoid real mask，与参考
复现一致），本 harness 保留供后续复测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 脚本位于 tools/model_tools/，需回退两级到 use_train 根目录
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

REPO_ROOT = _REPO_ROOT

# JAE-M08 审计实验使用 dbgnet_small_10ms_delay400ms 的 best_model.tar /
# resolved_config.yaml，但带时间戳的 checkpoint 目录属本机实验产物
# （AGENTS.md 红线：checkpoint/日志路径不入版本控制），故不设默认路径，
# 复测时需显式传 --checkpoint/--config。
DEFAULT_CHECKPOINT = None
DEFAULT_CONFIG = None

# reference sensitivity 实验中 far 循环移位的样点数（16kHz 下 800 样点 = 50ms）
FAR_SHIFT_SAMPLES = 800
_EPS = 1e-8


# ---------------------------------------------------------------------------
# 纯函数（供 tests 直接 import，不触碰 checkpoint / 模型）
# ---------------------------------------------------------------------------

def oracle_real_mask_enh(mic_c: np.ndarray, near_c: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """对照 B：oracle 实数 mask 增强后的复谱。

    m* = clamp(|near| / (|mic| + eps), 0, 1)，enh = mic × m*。
    与 DBGNet 输出表达形式一致（实数 [0,1] 增益），代表该形式下的估计上限。
    """
    m = np.clip(np.abs(near_c) / (np.abs(mic_c) + eps), 0.0, 1.0)
    return mic_c * m


def oracle_crm_enh(mic_c: np.ndarray, near_c: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """对照 C：oracle complex ratio mask 增强后的复谱。

    m_c = near / (mic + eps)（复数除法，无界），enh = mic × m_c。
    数学上精确还原 near（至 eps 数值误差），作为表达上限 sanity check。
    """
    return mic_c * (near_c / (mic_c + eps))


def make_reference_variants(far: np.ndarray, shift_samples: int = FAR_SHIFT_SAMPLES) -> dict:
    """对照 D 的三档 reference：(a) 正确 far、(b) zeros far、(c) 循环移位 far。"""
    return {
        "correct": far,
        "zeros": np.zeros_like(far),
        "shifted": np.roll(far, shift_samples),
    }


def sisdr_db(ref: np.ndarray, enh: np.ndarray, fs: int = 16000) -> float:
    """SI-SDR(dB) 包装：复用 core.metrics.intrusive 的 SI-SDR 实现。

    支持实数波形与复数谱。输入先按原始最后一维对齐长度；复数再按
    real/imag 展开拼接，保证谱域的时间长度语义不被展平顺序破坏。
    """
    from core.metrics.intrusive import _INTRUSIVE_FNS

    fn, _ = _INTRUSIVE_FNS["SI-SDR"]
    ref = np.asarray(ref)
    enh = np.asarray(enh)
    n = min(ref.shape[-1], enh.shape[-1])
    ref = ref[..., :n]
    enh = enh[..., :n]
    if np.iscomplexobj(ref) or np.iscomplexobj(enh):
        # 复数谱：频率及其它非时间维必须一致，否则 flatten 后向量长度不同。
        if ref.shape[:-1] != enh.shape[:-1]:
            raise ValueError(
                "复数谱 ref 与 enh 的非时间维必须一致，实际为 "
                f"{ref.shape[:-1]} 和 {enh.shape[:-1]}"
            )
        # 复数谱：先裁共同时间长度，再把 real/imag 展开为同一实向量。
        ref = np.concatenate([ref.real.ravel(), ref.imag.ravel()])
        enh = np.concatenate([enh.real.ravel(), enh.imag.ravel()])
    v = fn(ref.ravel().astype(np.float64), enh.ravel().astype(np.float64), fs)
    return float(v)


# ---------------------------------------------------------------------------
# 以下为 harness 主流程（加载 checkpoint / 跑真实样本）
# ---------------------------------------------------------------------------

def _load_cfg(path: str) -> dict:
    """加载 resolved_config.yaml（OmegaConf，与训练入口同一解析路径）。"""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def load_model(cfg: dict, ckpt_path: str, device: str):
    """按 config 构建 DBGNet 并严格加载 checkpoint。

    - 模型构建复用 ``core.models.build.build_model``（会把顶层 stft 的
      n_fft/hop/win 合并进 model.args，与训练入口一致，即等效
      ``DBGNet(**model.args)``）；
    - checkpoint 加载复用 ``onnx.verify.verify_common.load_checkpoint_strict``
      的模式（内部走 ``onnx.export.load_checkpoint``，key/参数量不一致即报错）。
    """
    import torch

    from core.models import register_task_models
    from core.models.build import build_model
    from onnx.verify.verify_common import load_checkpoint_strict

    model_type = (cfg.get("model") or {}).get("type")
    register_task_models("aec", only=str(model_type) if model_type else "DBGNet")
    model = build_model(cfg["model"], stft_cfg=cfg.get("stft"))
    load_checkpoint_strict(model, ckpt_path)
    model.to(device).eval()
    return model


def load_wav(path: Path, target_sr: int) -> np.ndarray:
    """读 wav（soundfile），多声道取第一声道，必要时重采样到 target_sr。"""
    import soundfile as sf

    from core.metrics.utils import _resample_np

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav[:, 0]
    if sr != target_sr:
        wav = _resample_np(wav, sr, target_sr).astype(np.float32)
    return wav


def _stft_np(stft_mod, wav: np.ndarray, device: str) -> np.ndarray:
    """波形 (T,) → 复谱 (F, T)（numpy complex）。"""
    import torch

    spec = stft_mod(torch.from_numpy(wav).unsqueeze(0).to(device))  # (1, F, T, 2)
    spec = torch.view_as_complex(spec.squeeze(0).contiguous().float().cpu())
    return spec.numpy()


def _istft_np(stft_mod, spec_c: np.ndarray, length: int, device: str) -> np.ndarray:
    """复谱 (F, T) → 波形 (length,)。"""
    import torch

    spec_t = torch.from_numpy(spec_c).unsqueeze(0).to(device)  # (1, F, T) complex
    wav = stft_mod.inverse(spec_t, length)
    return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)


def enhance_with_model(model, mic_c: np.ndarray, far_c: np.ndarray, device: str) -> np.ndarray:
    """DBGNet forward_spectra：mic/far 复谱 (F,T) → enhanced 复谱 (F,T)。"""
    import torch

    mic_t = torch.view_as_real(torch.from_numpy(mic_c)).unsqueeze(0).to(device)
    far_t = torch.view_as_real(torch.from_numpy(far_c)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.forward_spectra(mic_t.float(), far_t.float())
    if isinstance(out, tuple):  # use_state_detector=True 时附带帧级状态 logits
        out = out[0]
    enh = torch.view_as_complex(out.squeeze(0).contiguous().float().cpu())
    return enh.numpy()


def _metric_pair(stft_mod, near_wav, enh_spec_c, length, device, fs, with_pesq):
    """对一条 enhanced 复谱做 ISTFT 并算指标。"""
    enh_wav = _istft_np(stft_mod, enh_spec_c, length, device)
    out = {"sisdr_db": sisdr_db(near_wav, enh_wav, fs)}
    if with_pesq:
        from core.metrics.intrusive import _INTRUSIVE_FNS

        fn, _ = _INTRUSIVE_FNS["PESQ"]
        n = min(len(near_wav), len(enh_wav))
        v = fn(near_wav[:n], enh_wav[:n], fs)
        out["pesq"] = None if v is None else float(v)
    return out


def process_sample(sample_dir: Path, model, stft_mod, device: str, fs: int,
                   with_pesq: bool, shift_samples: int) -> dict:
    """单样本四对照流程：A 模型输出 / B oracle real mask / C oracle CRM / D reference 敏感性。"""
    mic = load_wav(sample_dir / "mic.wav", fs)
    near = load_wav(sample_dir / "nearend.wav", fs)
    far = load_wav(sample_dir / "farend.wav", fs)

    # oracle stems 对齐：mic/nearend（连同 farend）裁剪到最短，保证谱域逐点对齐
    n = min(len(mic), len(near), len(far))
    mic, near, far = mic[:n], near[:n], far[:n]

    mic_c = _stft_np(stft_mod, mic, device)
    near_c = _stft_np(stft_mod, near, device)

    result = {"sample": str(sample_dir), "num_samples_wav": int(n)}

    # A. model：模型原始输出
    enh_a = enhance_with_model(model, mic_c, _stft_np(stft_mod, far, device), device)
    result["model"] = _metric_pair(stft_mod, near, enh_a, n, device, fs, with_pesq)

    # B. oracle_real_mask：与模型同表达形式的 oracle 上限
    enh_b = oracle_real_mask_enh(mic_c, near_c)
    result["oracle_real_mask"] = _metric_pair(stft_mod, near, enh_b, n, device, fs, with_pesq)

    # C. oracle_crm：复数无界 mask，数学上精确还原 nearend（表达上限 sanity check）
    enh_c = oracle_crm_enh(mic_c, near_c)
    result["oracle_crm"] = _metric_pair(stft_mod, near, enh_c, n, device, fs, with_pesq)

    # D. reference sensitivity：correct / zeros / 循环移位 三档 far
    ref_metrics = {}
    for tag, far_var in make_reference_variants(far, shift_samples).items():
        enh_d = enhance_with_model(model, mic_c, _stft_np(stft_mod, far_var, device), device)
        ref_metrics[tag] = _metric_pair(stft_mod, near, enh_d, n, device, fs, with_pesq)
    result["reference_sensitivity"] = ref_metrics
    return result


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def aggregate(samples: list[dict], with_pesq: bool) -> dict:
    """聚合 per-sample 明细并写判定字段。"""
    a = [s["model"]["sisdr_db"] for s in samples]
    b = [s["oracle_real_mask"]["sisdr_db"] for s in samples]
    c = [s["oracle_crm"]["sisdr_db"] for s in samples]
    d_correct = [s["reference_sensitivity"]["correct"]["sisdr_db"] for s in samples]
    d_zeros = [s["reference_sensitivity"]["zeros"]["sisdr_db"] for s in samples]
    d_shifted = [s["reference_sensitivity"]["shifted"]["sisdr_db"] for s in samples]

    agg = {
        "model_sisdr_db": _mean(a),
        "oracle_real_mask_sisdr_db": _mean(b),
        "oracle_crm_sisdr_db": _mean(c),
        "ref_correct_sisdr_db": _mean(d_correct),
        "ref_zeros_sisdr_db": _mean(d_zeros),
        "ref_shifted_sisdr_db": _mean(d_shifted),
    }
    if with_pesq:
        agg["model_pesq"] = _mean([s["model"].get("pesq") for s in samples])
        agg["oracle_real_mask_pesq"] = _mean([s["oracle_real_mask"].get("pesq") for s in samples])
        agg["oracle_crm_pesq"] = _mean([s["oracle_crm"].get("pesq") for s in samples])

    judgement = {
        "expression_gap_db": _mean([ci - bi for bi, ci in zip(b, c)]),
        "estimation_gap_db": _mean([bi - ai for ai, bi in zip(a, b)]),
        "reference_sensitivity_db": _mean([x - y for x, y in zip(d_correct, d_zeros)]),
        "note": (
            "oracle CRM > oracle real mask 显著 → 表达受限证据（sigmoid 实数 mask "
            "不能旋转相位也不能放大）；A 接近 B → 瓶颈在估计不在表达；"
            "reference_sensitivity_db 为模型在正确 far 与 zeros far 下的 SI-SDR 差，"
            "衡量输出对 reference 的实际依赖。"
        ),
    }
    return {"aggregate": agg, "judgement": judgement}


def _positive_int(value: str) -> int:
    """argparse 类型：仅接受有限的正整数文本。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _find_sample_dirs(data_dir: Path, num_samples: int) -> list[Path]:
    """递归枚举 canonical AEC testset 中的样本目录。"""
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"num_samples 必须是正整数，实际为 {num_samples!r}")

    required = ("mic.wav", "nearend.wav", "farend.wav")
    sample_dirs = []
    # canonical testset 常见布局为 scenario/sample_xxx，按稳定顺序遍历并尽早停止。
    for root, subdirs, _files in os.walk(data_dir, topdown=True):
        subdirs.sort()
        path = Path(root)
        if all((path / name).is_file() for name in required):
            sample_dirs.append(path)
        if len(sample_dirs) >= num_samples:
            break
    return sample_dirs


def main():
    ap = argparse.ArgumentParser(
        description="JAE-M08 DBGNet 输出表达瓶颈实验：oracle mask/CRM + reference sensitivity"
    )
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="训练 checkpoint（best_model.tar），必传")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="resolved_config.yaml，必传")
    ap.add_argument("--data-dir", required=True,
                    help="样本目录（canonical AEC simulated testset 产物：每样本子目录含 mic.wav/nearend.wav/farend.wav）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-samples", type=_positive_int, default=8)
    ap.add_argument("--out", default="report.json", help="report.json 输出路径")
    ap.add_argument("--with-pesq", action="store_true", help="附加 PESQ（需要 pesq 包）")
    ap.add_argument("--far-shift-samples", type=int, default=FAR_SHIFT_SAMPLES,
                    help="reference sensitivity 中 far 循环移位样点数（默认 800 = 16kHz 50ms）")
    args = ap.parse_args()

    # checkpoint / config 无默认路径（本机实验产物），缺省时给出清晰报错
    if args.checkpoint is None or args.config is None:
        ap.error(
            "请显式传 --checkpoint/--config（checkpoint 与 resolved_config 为本机实验产物，"
            "不设默认路径；参考 JAE-M08 审计对应 dbgnet_small_10ms_delay400ms 实验目录）"
        )

    from core.stft import build_stft_istft

    cfg = _load_cfg(args.config)
    fs = int(cfg.get("data", {}).get("sample_rate", 16000))
    stft_mod = build_stft_istft(cfg["stft"]).to(args.device).eval()
    model = load_model(cfg, args.checkpoint, args.device)

    sample_dirs = _find_sample_dirs(Path(args.data_dir), args.num_samples)
    if not sample_dirs:
        raise SystemExit(
            f"在 {args.data_dir} 下未找到含 mic.wav/nearend.wav/farend.wav 的样本目录"
        )
    print(f"[dbgnet_output_bottleneck] {len(sample_dirs)} 个样本, device={args.device}")

    samples = []
    for sd in sample_dirs:
        r = process_sample(sd, model, stft_mod, args.device, fs,
                           args.with_pesq, args.far_shift_samples)
        samples.append(r)
        print(
            f"  {Path(sd).name}: "
            f"A(model)={r['model']['sisdr_db']:.2f}dB "
            f"B(oracle_real)={r['oracle_real_mask']['sisdr_db']:.2f}dB "
            f"C(oracle_crm)={r['oracle_crm']['sisdr_db']:.2f}dB "
            f"D(correct/zeros/shifted)="
            f"{r['reference_sensitivity']['correct']['sisdr_db']:.2f}/"
            f"{r['reference_sensitivity']['zeros']['sisdr_db']:.2f}/"
            f"{r['reference_sensitivity']['shifted']['sisdr_db']:.2f}dB"
        )

    report = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "data_dir": args.data_dir,
        "device": args.device,
        "num_samples": len(samples),
        "samples": samples,
        **aggregate(samples, args.with_pesq),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    j = report["judgement"]
    print(
        f"[done] expression_gap={j['expression_gap_db']:.2f}dB "
        f"estimation_gap={j['estimation_gap_db']:.2f}dB "
        f"reference_sensitivity={j['reference_sensitivity_db']:.2f}dB -> {out_path}"
    )


if __name__ == "__main__":
    main()
