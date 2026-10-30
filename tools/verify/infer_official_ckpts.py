"""官方开源权重端到端验证：六模型加载 + 测试音频推理 + 输出保存。

覆盖模型与权重：
  - GTCRN / UL-UNAS（tasks/se/models/discriminative，谱域接口，16kHz；
    官方 .tar 为 ``{'model': state_dict, ...}`` 训练 ckpt）
  - PASE / StuPASE / UniPASE / GAP-URGENet（tasks/se/models/pase_family，
    waveform-native 级联；artifact 为 ``{'cfg', 'model'}``，from_pretrained
    fail-closed，任何 missing/unexpected 直接抛错）

用法（输入/输出路径由参数或环境变量注入，不硬编码）：
  PYTHONPATH=. python tools/verify/infer_official_ckpts.py \
      --input /path/to/noisy.wav --output_dir /path/to/out_dir
  PYTHONPATH=. python tools/verify/infer_official_ckpts.py \
      --input /path/to/noisy.wav --models GTCRN UL-UNAS
  （--output_dir 未给时读环境变量 SE_VERIFY_OUTPUT_DIR；两者都缺则报错）

谱域模型（GTCRN/UL-UNAS）用 core/stft.py 的 StftIstftModule（512/256/512, hann）
包波形→谱→波形，44.1k 输入先重采样到 16k、输出回 44.1k；PASE/StuPASE/UniPASE
内部 16k 处理；GAP-URGENet 内部 16k 双支路 + 48k 融合、输出与输入同采样率。
所有输出统一重采样回 44.1k、长度与输入对齐后保存。

推理模式（chunked vs whole，官方依据）：
  - UniPASE / GAP-URGENet：长音频按官方 inference/inference_long.py 语义分块
    （seg=8s, hop=4s, 中点拼接，拼接后全局峰值归一），复用根 inference.py 的
    ``_chunked_waveform_forward``（其尾块策略已修复官方 end_tmp bug），
    避免 TFGridNetV3 注意力 T×T 的整条 OOM；
  - PASE / StuPASE：官方 pase/inference/inference.py、stupase/inference/*.py
    均无分块逻辑（已 grep 核实），但官方短音频路径会把输出峰值归一到输入峰值；
    此处按用户要求在框架层对齐同一 inference_long 配方（8s/4s 中点拼接 +
    拼接后全局峰值归一）。StuPASE.forward 支持 ``normalize_output``，分块时
    传 False，由脚本外层统一归一；PASE 无该参数，直接在脚本侧归一；
  - GTCRN / UL-UNAS：因果流式设计（可整流处理任意长度），保持整条推理。
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.utils.safe_load import safe_torch_load  # noqa: E402

GTCRN_CKPT = REPO_ROOT / "tasks/se/models/discriminative/gtcrn/open_model_checkpoint/model_trained_on_dns3.tar"
ULUNAS_CKPT = REPO_ROOT / "tasks/se/models/discriminative/ulunas/open_model_checkpoint/model_trained_on_dns3.tar"
PASE_CKPT_DIR = REPO_ROOT / "tasks/se/models/pase_family/pase/open_model_checkpoint"
STUPASE_CKPT_DIR = REPO_ROOT / "tasks/se/models/pase_family/stupase/open_model_checkpoint"
UNIPASE_CKPT_DIR = REPO_ROOT / "tasks/se/models/pase_family/unipase/open_model_checkpoint"
GAP_CKPT_DIR = REPO_ROOT / "tasks/se/models/pase_family/gap_urgenet/open_model_checkpoint"

_PROC_SR = 16000  # 谱域模型与 PASE/StuPASE/UniPASE 的内部处理采样率
_STFT_CFG = dict(n_fft=512, hop_length=256, win_length=512, window="hann")

ALL_MODELS = ["GTCRN", "UL-UNAS", "PASE", "StuPASE", "UniPASE", "GAP-URGENet"]


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def _resample(wav: torch.Tensor, orig: int, new: int) -> torch.Tensor:
    if orig == new:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=orig, new_freq=new)


def _align(wav: torch.Tensor, n_samples: int) -> torch.Tensor:
    if wav.shape[-1] < n_samples:
        wav = torch.nn.functional.pad(wav, (0, n_samples - wav.shape[-1]))
    return wav[..., :n_samples]


def _sanity(y: torch.Tensor, x_peak: float, x_rms: float) -> dict:
    y = y.detach().float().cpu()
    return {
        "nan_inf": bool((~torch.isfinite(y)).any().item()),
        "rms_in": x_rms,
        "rms_out": y.pow(2).mean().sqrt().item(),
        "peak_in": x_peak,
        "peak_out": y.abs().max().item(),
    }


def _report_sanity(name: str, s: dict, sr_out: int, dur_in: float, dur_out: float, elapsed: float) -> bool:
    """打印 sanity 报告；返回是否通过（NaN/Inf 或静音为 FAIL）。"""
    status = "OK"
    if s["nan_inf"]:
        status = "FAIL(NaN/Inf)"
    elif s["rms_out"] < 1e-6:
        status = "FAIL(静音)"
    print(
        f"[{name}] {status} | {dur_in:.2f}s -> {dur_out:.2f}s @{sr_out}Hz | "
        f"耗时 {elapsed:.1f}s | RMS {s['rms_in']:.4f}->{s['rms_out']:.4f} | "
        f"peak {s['peak_in']:.4f}->{s['peak_out']:.4f}"
    )
    return status == "OK"


def _map_gtcrn_official_keys(sd: dict) -> dict:
    """官方 gtcrn-main checkpoint -> 框架 GTCRN 的 key 映射（仅命名差异，结构同构）。

    官方 ``encoder.en_convs`` / ``decoder.de_convs`` 在框架中统一为 ``layers``；
    官方 GTConvBlock 内部 ``point_conv1/point_bn1/point_act/depth_conv/depth_bn/
    depth_act/point_conv2/point_bn2`` 在框架中缩写为 ``pc1/bn1/act1/dc/dbn/dact/
    pc2/bn2``（sfe/tra 同名；ConvBlock 的 conv/bn/act 与 DPGRNN/erb/mask 均同名）。
    已实测映射后 matched=271/271, missing=0, unexpected=0。
    """
    rename = {
        "en_convs": "layers", "de_convs": "layers",
        "point_conv1": "pc1", "point_bn1": "bn1", "point_act": "act1",
        "depth_conv": "dc", "depth_bn": "dbn", "depth_act": "dact",
        "point_conv2": "pc2", "point_bn2": "bn2",
    }
    mapped = {".".join(rename.get(seg, seg) for seg in k.split(".")): v for k, v in sd.items()}
    # 防静默覆盖：映射必须是 1:1（key 数不变、无碰撞）
    assert len(mapped) == len(sd), (
        f"GTCRN key 映射发生碰撞: {len(sd)} -> {len(mapped)}（有 official key 被覆盖）"
    )
    return mapped


def _load_spectrum_model(model: torch.nn.Module, ckpt_path: Path, name: str, key_mapper=None) -> None:
    """官方 .tar（{'model': state_dict}）加载 + missing/unexpected 报告；
    任何 missing/unexpected 非零即抛错（计入该模型 FAIL），与 artifact 加载的
    fail-closed 语义一致。"""
    ckpt = safe_torch_load(
        ckpt_path,
        map_location="cpu",
        purpose=f"verify official {name} checkpoint safely",
    )
    sd = ckpt["model"]
    if key_mapper is not None:
        sd = key_mapper(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{name}] 权重加载: matched={len(sd) - len(unexpected)}/{len(sd)}, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(
            f"[{name}] state_dict 不匹配: matched={len(sd) - len(unexpected)}/{len(sd)}, "
            f"missing[:5]={missing[:5]}, unexpected[:5]={unexpected[:5]}"
        )


def _run_spectrum_model(model_cls, ckpt_path: Path, name: str, x44: torch.Tensor, sr_in: int, device: torch.device, key_mapper=None):
    """谱域模型公共路径：16k 重采样 -> StftIstftModule -> model(spec) -> istft -> 回 44.1k。"""
    from core.stft import StftIstftModule

    model = model_cls().eval().to(device)
    _load_spectrum_model(model, ckpt_path, name, key_mapper=key_mapper)
    stft = StftIstftModule(**_STFT_CFG).to(device)

    n_samples = x44.shape[-1]
    x16 = _resample(x44, sr_in, _PROC_SR).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        spec = stft(x16)
        enh_spec = model(spec, None)
        y16 = stft.inverse(enh_spec, x16.shape[-1])
    y = _resample(y16.cpu(), _PROC_SR, sr_in)
    elapsed = time.perf_counter() - t0
    y = _align(y, n_samples)
    del model, stft
    return y, elapsed, f"44.1k->16k->44.1k"


# ---------------------------------------------------------------------------
# 各模型 runner：输入 x44 (1, T) @sr_in，返回 (y (1, T) @sr_in, elapsed, sr_path)
# ---------------------------------------------------------------------------

def run_gtcrn(x44, sr_in, device):
    """GTCRN：整条推理（因果流式设计，可整流处理任意长度，无需分块）。"""
    from tasks.se.models.discriminative.gtcrn import GTCRN

    return _run_spectrum_model(GTCRN, GTCRN_CKPT, "GTCRN", x44, sr_in, device, key_mapper=_map_gtcrn_official_keys)


def run_ulunas(x44, sr_in, device):
    """UL-UNAS：整条推理（因果流式设计，可整流处理任意长度，无需分块）。"""
    from tasks.se.models.discriminative.ulunas import ULUNAS

    return _run_spectrum_model(ULUNAS, ULUNAS_CKPT, "UL-UNAS", x44, sr_in, device)


def _global_peak_norm(y: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """全局峰值归一（官方 inference_long.py:123 同公式）：y 峰值归一到 x_ref 峰值。

    用于 forward 签名无 ``normalize_output`` 参数的模型（PASE），或分块路径
    主动传 ``normalize_output=False`` 的模型（StuPASE）——
    ``_chunked_waveform_forward`` 只对有该参数的模型做拼接后归一，这里在脚本侧
    补做。x_ref 为模型输入域波形（本脚本为 16k 重采样后的 wav）。"""
    in_peak = x_ref.abs().amax(dim=-1, keepdim=True)
    return y / (y.abs().amax(dim=-1, keepdim=True) + 1e-8) * in_peak


def run_pase(x44, sr_in, device):
    """PASE 固定 16k 处理（官方推理语义：encoder 提 noisy 的 L1+L24 -> decoder 生成），
    输入先重采样到 16k，输出回 44.1k。

    分块说明：官方 pase/inference/inference.py 无分块逻辑（已 grep 核实）；
    此处按用户要求在框架层对齐 UniPASE inference_long 完整配方（8s/4s 中点
    拼接 + 拼接后全局峰值归一）。全局归一会把输出电平拉回输入峰值（PASE 官方
    短音频路径无峰值归一，整条旧输出 peak≈0.897）——此为按用户要求的
    inference_long 配方行为，偏离官方短音频电平。"""
    from inference import _chunked_waveform_forward
    from tasks.se.models.pase_family.pase import PASE

    model = PASE(
        wavlm_ckpt_path=str(PASE_CKPT_DIR / "DeWavLM.pt"),
        vocoder_ckpt_path=str(PASE_CKPT_DIR / "Vocoder_Dual.pt"),
    ).eval().to(device)
    print("[PASE] 权重加载: from_pretrained fail-closed（任何 key 不匹配会直接抛错），加载成功即 0/0")

    n_samples = x44.shape[-1]
    x16 = _resample(x44, sr_in, _PROC_SR).to(device)
    t0 = time.perf_counter()
    y16 = _chunked_waveform_forward(model, x16, _PROC_SR, chunk_seconds=8.0, multi_sr=True)
    y16 = _global_peak_norm(y16, x16)  # PASE 无 normalize_output 参数，脚本侧补做
    y = _resample(y16.cpu(), _PROC_SR, sr_in)
    elapsed = time.perf_counter() - t0
    y = _align(y, n_samples)
    del model
    return y, elapsed, "44.1k->16k->44.1k (chunked 8s/4s)"


def run_stupase(x44, sr_in, device):
    """StuPASE 固定 16k 处理（CFM 的 cond 必须是 16k 波形），输入先重采样到 16k。

    分块说明：官方 stupase/inference/*.py 无分块逻辑（已 grep 核实）；此处按
    用户要求在框架层对齐 UniPASE inference_long 完整配方（8s/4s 中点拼接 +
    拼接后全局峰值归一）。StuPASE.forward 已支持 ``normalize_output``，分块辅助
    函数会对它传 False，避免每块各自归一；拼接后由本脚本统一归一。
    官方短音频路径也会把输出峰值拉回输入峰值（inference.py:84-85）。"""
    from inference import _chunked_waveform_forward
    from tasks.se.models.pase_family.stupase.stupase_model import StuPASE

    model = StuPASE(
        dewavlm_ckpt_path=str(STUPASE_CKPT_DIR / "DeWavLM-R.pt"),
        cfm_ckpt_path=str(STUPASE_CKPT_DIR / "CFM.pt"),
        vocoder_ckpt_path=str(STUPASE_CKPT_DIR / "Vocoder_Mel-16k.pt"),
    ).eval().to(device)
    print("[StuPASE] 权重加载: from_pretrained fail-closed，加载成功即 0/0")

    n_samples = x44.shape[-1]
    x16 = _resample(x44, sr_in, _PROC_SR).to(device)
    t0 = time.perf_counter()
    y16 = _chunked_waveform_forward(model, x16, _PROC_SR, chunk_seconds=8.0, multi_sr=True)
    y16 = _global_peak_norm(y16, x16)  # 分块路径传 normalize_output=False，脚本侧做全局归一
    y = _resample(y16.cpu(), _PROC_SR, sr_in)
    elapsed = time.perf_counter() - t0
    y = _align(y, n_samples)
    del model
    return y, elapsed, "44.1k->16k->44.1k (chunked 8s/4s)"


def run_unipase(x44, sr_in, device):
    """UniPASE：任意采样率输入（内部重采样到 16k），sr_out=16000（不走 PostNet，
    对齐 configs/se/urgent2026/pase/unipase/unipase_infer.yaml 默认），输出回 44.1k。
    长音频按官方 inference/inference_long.py 语义分块（8s/4s 中点拼接，
    拼接后全局峰值归一），复用 inference._chunked_waveform_forward。"""
    from inference import _chunked_waveform_forward
    from tasks.se.models.pase_family.unipase.unipase_model import UniPASE

    model = UniPASE(
        dewavlm_ckpt_path=str(UNIPASE_CKPT_DIR / "DeWavLM-Omni.pt"),
        adapter_ckpt_path=str(UNIPASE_CKPT_DIR / "Adapter.pt"),
        vocoder_ckpt_path=str(UNIPASE_CKPT_DIR / "Vocoder_DWO-L1.pt"),
        postnet_ckpt_path=str(UNIPASE_CKPT_DIR / "PostNet.pt"),
        sr_out=16000,
        enable_plc=True,
    ).eval().to(device)
    print("[UniPASE] 权重加载: from_pretrained fail-closed，加载成功即 0/0")

    x = x44.to(device)
    t0 = time.perf_counter()
    # model.sr_out=16000：_chunked_waveform_forward 在输出域（16k）换算拼接 boundary，
    # 输出 16k、拼接后已做全局峰值归一
    y16 = _chunked_waveform_forward(model, x, sr_in, chunk_seconds=8.0, multi_sr=True)
    y = _resample(y16.cpu(), _PROC_SR, sr_in)
    elapsed = time.perf_counter() - t0
    y = _align(y, x44.shape[-1])
    del model
    return y, elapsed, f"44.1k->16k(内部)->16k out->44.1k (chunked 8s/4s)"


def run_gap_urgenet(x44, sr_in, device):
    """GAP-URGENet：任意采样率输入（内部 16k 双支路 + 48k PostNet 融合），
    sr_out=None 输出与输入同采样率（对齐 configs/se/urgent2026/pase/gap_urgenet/gap_urgenet_infer.yaml）。
    长音频按官方 inference/inference_long.py 语义分块（8s/4s 中点拼接，
    拼接后全局峰值归一），复用 inference._chunked_waveform_forward——
    避免 TFGridNetV3 注意力 T×T 的整条 OOM（180s@48k 整条约需 35GB+）。"""
    from inference import _chunked_waveform_forward
    from tasks.se.models.pase_family.gap_urgenet.gap_urgenet_model import GAP_URGENet

    model = GAP_URGENet(
        dewavlm_ckpt_path=str(GAP_CKPT_DIR / "DeWavLM-Omni.pt"),
        adapter_ckpt_path=str(GAP_CKPT_DIR / "Adapter.pt"),
        vocoder_ckpt_path=str(GAP_CKPT_DIR / "Vocoder.pt"),
        predictor_ckpt_path=str(GAP_CKPT_DIR / "Predictor.pt"),
        postnet_ckpt_path=str(GAP_CKPT_DIR / "PostNet.pt"),
        sr_out=None,
        enable_plc=True,
    ).eval().to(device)
    print("[GAP-URGENet] 权重加载: from_pretrained fail-closed，加载成功即 0/0")

    x = x44.to(device)
    t0 = time.perf_counter()
    # model.sr_out=None：_chunked_waveform_forward 回退到输入域（sr_in）换算拼接
    # boundary，输出与输入同采样率、拼接后已做全局峰值归一
    y = _chunked_waveform_forward(model, x, sr_in, chunk_seconds=8.0, multi_sr=True)
    elapsed = time.perf_counter() - t0
    y = _align(y.cpu(), x44.shape[-1])
    del model
    return y, elapsed, "44.1k->16k 双支路->48k 融合->44.1k (chunked 8s/4s)"


RUNNERS = {
    "GTCRN": run_gtcrn,
    "UL-UNAS": run_ulunas,
    "PASE": run_pase,
    "StuPASE": run_stupase,
    "UniPASE": run_unipase,
    "GAP-URGENet": run_gap_urgenet,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="测试音频（任意采样率，自动处理；必填）")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="输出目录（自动创建）；未给时读环境变量 SE_VERIFY_OUTPUT_DIR")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS, help="只跑指定模型子集")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output_dir is None:
        env_dir = os.environ.get("SE_VERIFY_OUTPUT_DIR")
        if not env_dir:
            parser.error("--output_dir 必填（或设环境变量 SE_VERIFY_OUTPUT_DIR）")
        args.output_dir = Path(env_dir)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio, sr_in = sf.read(str(args.input), dtype="float32", always_2d=True)  # (T, C)
    x44 = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)  # (1, T) mono
    n_samples = x44.shape[-1]
    x_peak = x44.abs().max().item()
    x_rms = x44.pow(2).mean().sqrt().item()
    print(f"输入: {args.input.name} | {sr_in}Hz | {n_samples / sr_in:.2f}s | "
          f"RMS={x_rms:.4f} peak={x_peak:.4f} | device={device}")

    failures: list[str] = []
    for name in args.models:
        print(f"\n===== {name} =====")
        try:
            y, elapsed, sr_path = RUNNERS[name](x44, sr_in, device)
        except Exception as e:  # 单模型失败不阻断其余模型，但计入退出码
            print(f"[{name}] FAIL: {type(e).__name__}: {e}")
            failures.append(name)
            continue
        s = _sanity(y, x_peak, x_rms)
        ok = _report_sanity(name, s, sr_in, n_samples / sr_in, y.shape[-1] / sr_in, elapsed)
        print(f"[{name}] 采样率路径: {sr_path}")
        if not ok:
            failures.append(name)
        if s["nan_inf"]:
            print(f"[{name}] 输出含 NaN/Inf，跳过写盘")
        else:
            out_path = args.output_dir / f"{name}.wav"
            # subtype FLOAT：PCM_16 会把 peak>1 的输出（GTCRN/UL-UNAS 等无峰值归一化的
            # 模型）静默截幅到 ±1，FLOAT 保留原始动态范围
            sf.write(str(out_path), y.squeeze(0).numpy(), sr_in, subtype="FLOAT")
            print(f"[{name}] 已保存: {out_path}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if failures:
        print(f"\n完成，但有 FAIL: {failures}")
        sys.exit(1)
    print("\n全部完成，无 FAIL。")


if __name__ == "__main__":
    main()
