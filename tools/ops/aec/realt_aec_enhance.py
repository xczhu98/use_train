#!/usr/bin/env python3
"""用训练好的 DeepVQE AEC 模型对单声道 mixture 做近端增强推理并保存结果。

背景
----
DeepVQE 是双路输入 AEC 模型，正常推理需要 mic + farend（回声参考）配对。
REAL-TSE 等单声道 mixture 没有 farend 参考，本脚本采用 **zero-farend** 约定：
把 mixture 当作近端（mic）输入，farend 用与 mic 等长的全零参考，模型输出
nearend_estimate 作为"增强/清晰化"后的近端语音保存。

注意（项目 .research/findings.md 已提示）：zero-farend 对训练在非零 ref 上的
联合 AEC+NS 模型属于 OOD 分布外输入，直接作为近端增强器使用属于近似用法，
输出质量可能无法与标准双路 AEC 场景等价。

用法:
    python -m tools.ops.aec.realt_aec_enhance \
        --config   exp/aec/<model>/checkpoints/<run>/resolved_config.yaml \
        --checkpoint exp/aec/<model>/checkpoints/<run>/best_model.tar \
        --input_dir  data/tse/REAL-TSE/REAL-T-EVAL_20260531/REAL-T-eval1/mixtures \
        --output_dir exp/aec/<model>/realt_eval1_out \
        --device cuda:0

任一文件失败均被计数，全部处理结束后若有失败则返回非零退出码。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# 注册 AEC 模型到共享 MODEL_REGISTRY（懒加载 map）
import tasks.aec.models  # noqa: F401
from core.models.build import build_model
from core.stft import build_stft_istft
from core.utils.config import load_merged_config
from core.utils.safe_load import safe_torch_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("realt_aec_enhance")

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def parse_args():
    p = argparse.ArgumentParser(description="DeepVQE AEC 对单声道 mixture 做 zero-farend 近端增强推理")
    p.add_argument("--config", required=True, help="训练 resolved_config.yaml")
    p.add_argument("--checkpoint", required=True, help="best_model.tar")
    p.add_argument("--input_dir", required=True, help="mixture 目录")
    p.add_argument("--output_dir", required=True, help="增强结果输出目录")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    cfg = load_merged_config(args.config)
    stft_cfg = cfg.get("stft")
    if not stft_cfg:
        raise ValueError("config 缺少顶层 'stft'（AEC 谱进谱出需要 STFT 配置）")

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type")
    if not model_type:
        raise ValueError("config 缺少 model.type")
    tasks.aec.models.register_aec_models(only=str(model_type))

    model = build_model(model_cfg, stft_cfg=stft_cfg).to(device).eval()
    logger.info("built model: %s", model_type)

    # 安全加载训练 checkpoint（含 optimizer/scheduler 等非 tensor 对象，
    # 来自项目训练产物 best_model.tar，属可信来源，显式注明 purpose）
    ck = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        allow_unsafe=True,
        purpose=f"加载已训练 AEC 推理权重用于 REAL-TSE mixture 增强 ({args.checkpoint})",
    )
    sd = ck.get("model_state_dict") or ck.get("model") or ck.get("state_dict") or ck
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(
            f"checkpoint 与模型不匹配: missing={len(res.missing_keys)}, "
            f"unexpected={len(res.unexpected_keys)}"
        )
    logger.info("loaded checkpoint: %s", args.checkpoint)

    stft = build_stft_istft(stft_cfg).to(device)
    data_sr = int(cfg.get("data", {}).get("sample_rate", 16000))
    logger.info("model fixed sample rate=%d, n_fft=%d hop=%d",
                data_sr, stft_cfg["n_fft"], stft_cfg["hop_length"])

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in _AUDIO_EXTS)
    if not files:
        raise ValueError(f"input_dir 下没有可处理音频: {in_dir}")
    logger.info("processing %d audio files", len(files))

    it = tqdm(files, desc="zero-farend AEC enhance", unit="file", disable=args.no_progress)
    failures: list[str] = []
    for path in it:
        try:
            wav, fs = torchaudio.load(str(path))          # [C, N]
            wav = wav.mean(0, keepdim=True)               # 混音取均值 → 单声道
            if fs != data_sr:
                wav = torchaudio.functional.resample(wav.float(), fs, data_sr).float()
            wav = wav.to(device)
            n = wav.shape[-1]

            mic_spec = stft(wav)
            # zero-farend：与 mic 等长的全零参考
            far_spec = stft(torch.zeros_like(wav))
            out = model({"mic_spectra": mic_spec, "farend_spectra": far_spec})
            ne_spec = out["nearend_estimate_spectra"]
            ne_wav = stft.inverse(ne_spec, n)[..., :n].squeeze(0).cpu()

            out_path = out_dir / f"{path.stem}.wav"
            torchaudio.save(str(out_path), ne_wav.unsqueeze(0), data_sr)
            it.set_postfix(uid=path.stem)
        except Exception as e:  # noqa: BLE001
            failures.append(path.name)
            logger.warning("failed on %s: %s", path.name, e)

    if failures:
        # 部分失败的批处理不得被调用方当成已完成
        logger.error("done with %d/%d failures -> %s: %s",
                     len(failures), len(files), out_dir, failures[:10])
        return 1
    logger.info("done -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
