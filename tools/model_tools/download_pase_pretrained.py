#!/usr/bin/env python3
"""Download official PASE / StuPASE / UniPASE pretrained checkpoints from HuggingFace Hub.

PASE and StuPASE weights are hosted in separate HF repos (审计 T-01 修复:
此前误以为两者同在 ``cisco-ai/pase``，且 PASE 用了不存在的 ``.tar`` 扩展名):
  PASE    (cisco-ai/pase):     DeWavLM.pt, Vocoder_Dual.pt
  StuPASE (cisco-ai/stupase):   DeWavLM-R.pt, CFM.pt, Vocoder_Mel-16k.pt
  UniPASE (Xiaobin-Rong/unipase): DeWavLM-Omni.pt, Adapter.pt,
              Vocoder_DWO-L1.pt, Vocoder_WavLM-L24.pt, PostNet.pt

WavLM-Large.pt (the WavLM base checkpoint PASE/StuPASE fine-tune from) is NOT
hosted on cisco-ai/pase — it must be obtained separately from Microsoft's
official WavLM release: https://github.com/microsoft/unilm/tree/master/wavlm
(UniPASE 的 DeWavLM-Omni.pt 已内置 WavLM 权重，无需单独下载。)

网络备注（2026-07 实测）：本机直连 huggingface.co 的 ``hf_hub_download``
会中途停滞；设置 ``HF_ENDPOINT=https://hf-mirror.com`` 后该版本
huggingface_hub(1.16.1) 的 HEAD 请求仍失败。实测可靠的方式是 aria2c 多连接
断点续传（``aria2c -c -x 8 -s 8``）从 hf-mirror 的 resolve URL 下载，如
    aria2c -c -x 8 -s 8 -o DeWavLM-Omni.pt \
        https://hf-mirror.com/Xiaobin-Rong/unipase/resolve/main/DeWavLM-Omni.pt

Usage:
    python tools/model_tools/download_pase_pretrained.py --model pase --output_dir checkpoints/pase
    python tools/model_tools/download_pase_pretrained.py --model stupase  # 默认落到 tasks/se/models/pase_family/stupase/open_model_checkpoint
    python tools/model_tools/download_pase_pretrained.py --model unipase  # 默认落到 tasks/se/models/pase_family/unipase/open_model_checkpoint
    python tools/model_tools/download_pase_pretrained.py --model all --output_dir checkpoints/pase_family
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 审计 T-01: PASE 与 StuPASE 在不同 HF repo；PASE 文件扩展名是 .pt（非 .tar）
_REPOS = {
    "pase": "cisco-ai/pase",
    "stupase": "cisco-ai/stupase",
    "unipase": "Xiaobin-Rong/unipase",
}

_FILES = {
    "pase": ["DeWavLM.pt", "Vocoder_Dual.pt"],
    "stupase": ["DeWavLM-R.pt", "CFM.pt", "Vocoder_Mel-16k.pt"],
    "unipase": [
        "DeWavLM-Omni.pt",
        "Adapter.pt",
        "Vocoder_DWO-L1.pt",
        "Vocoder_WavLM-L24.pt",
        "PostNet.pt",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download official PASE/StuPASE/UniPASE checkpoints from HF Hub"
    )
    parser.add_argument("--model", choices=["pase", "stupase", "unipase", "all"], default="all")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Local directory to save checkpoints (默认 checkpoints/<model>)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    models = ["pase", "stupase", "unipase"] if args.model == "all" else [args.model]
    # 未显式指定时的默认目录：pase -> checkpoints/pase（PASE 配置经 ~/use_train_checkpoints/pase 软链读取）；
    # stupase/unipase -> 模型包内的 open_model_checkpoint（权重随模型目录管理）
    _DEFAULT_DIRS = {
        "pase": Path("checkpoints/pase"),
        "stupase": Path("tasks/se/models/pase_family/stupase/open_model_checkpoint"),
        "unipase": Path("tasks/se/models/pase_family/unipase/open_model_checkpoint"),
    }
    for model in models:
        out_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_DIRS[model]
        out_dir.mkdir(parents=True, exist_ok=True)
        repo = _REPOS[model]
        for filename in _FILES[model]:
            print(f"[*] Downloading {filename} from {repo} ...")
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(out_dir),
            )
            print(f"[OK] Saved: {path}")

    if "pase" in models or "stupase" in models:
        print(
            "\n[NOTE] WavLM-Large.pt is NOT hosted on cisco-ai/pase or cisco-ai/stupase — "
            "obtain it separately from https://github.com/microsoft/unilm/tree/master/wavlm "
            "and place it under the checkpoint dir before running Phase-1/2 configs "
            "that reference wavlm_ckpt_path."
        )


if __name__ == "__main__":
    main()
