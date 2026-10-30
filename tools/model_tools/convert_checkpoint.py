#!/usr/bin/env python3
"""Official HF checkpoint -> use_train weight conversion tool.

Supports two sources:
  1. pase_official (PASE/StuPASE) — official HF checkpoints (cisco-ai/pase, cisco-ai/stupase)
  2. unipase_official (UniPASE)   — official HF checkpoints (Xiaobin-Rong/unipase, 1:1 vendored passthrough)

Usage:
    # PASE official WavLM -> use_train PASE encoder (passthrough)
    python tools/model_tools/convert_checkpoint.py \
        --source pase_official \
        --model_type PASEEncoder \
        --input  /path/to/WavLM-Large.pt \
        --output checkpoints/pase_wavlm_converted.pth

    # View key mapping only, do not save
    python tools/model_tools/convert_checkpoint.py \
        --source pase_official --model_type PASEEncoder \
        --input /path/to/WavLM-Large.pt --dry-run

Legacy GTCRN / BSRNN 转换（外部 se_train / urgent2026 / use_train_plus 项目）已移除：
use_train 用自身 checkpoint 训练与推理，不再提供旧项目权重入口。
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import torch


def _extract_model_state(ckpt: dict) -> dict:
    """Extract model state_dict from various checkpoint formats."""
    if "model" in ckpt:
        return ckpt["model"]
    if "generator" in ckpt:
        # GAN 训练产物 / 部分发布 artifact 的 generator 槽位
        return ckpt["generator"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    if any(k.startswith(("erb.", "encoder.", "bsrnn.")) for k in ckpt):
        return ckpt
    return ckpt


def _strip_prefix(sd: dict, prefix: str) -> dict:
    """Remove key prefix (e.g. 'module.' or 'model.')."""
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_sd[new_key] = v
    return new_sd


# ──────────────────────────────────────────────────────────
# Official PASE / StuPASE (cisco-ai/pase on HF Hub) → use_train
# ──────────────────────────────────────────────────────────
# Both the source repo's WavLM_feat and WavLMDec and this framework's ported
# tasks/se/models/pase_family/{wavlm_frontend.py,vocoder_backend.py} are
# 1:1 vendored ports with identical submodule names, so their checkpoints are
# key-compatible without remapping.
#
# The MPD+MBD discriminator is the one exception: source's CombinedDiscriminator
# wraps a ModuleList indexed [mpd, mbd] (state_dict keys "discrimiantor.0.*" /
# "discrimiantor.1.*" -- note the source repo's own typo, "discrimiantor"), while
# the ported version names the submodules directly ("mpd.*" / "mbd.*").

def convert_pase_official_wavlm(src_sd: dict) -> dict:
    """Official WavLM-Large.pt / DeWavLM.tar['model'] -> WavLM_feat.wavlm state_dict (passthrough)."""
    return OrderedDict(src_sd)


def convert_pase_official_vocoder(src_sd: dict) -> dict:
    """Official Vocoder_Dual.tar['model'] -> WavLMDec state_dict (passthrough)."""
    return OrderedDict(src_sd)


def convert_pase_official_discriminator(src_sd: dict) -> dict:
    """Official CombinedDiscriminator([mpd, mbd]) state_dict -> ported CombinedDiscriminator.

    Remaps "discrimiantor.0.*" -> "mpd.*" and "discrimiantor.1.*" -> "mbd.*".
    """
    new_sd = OrderedDict()
    for k, v in src_sd.items():
        if k.startswith("discrimiantor.0."):
            new_sd["mpd." + k[len("discrimiantor.0."):]] = v
        elif k.startswith("discrimiantor.1."):
            new_sd["mbd." + k[len("discrimiantor.1."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def convert_stupase_official_dewavlm_r(src_sd: dict) -> dict:
    """Official DeWavLM-R.pt -> WavLM_feat.wavlm state_dict (passthrough; verify after download)."""
    return OrderedDict(src_sd)


def convert_stupase_official_cfm(src_sd: dict) -> dict:
    """Official CFM.pt -> ported CFM state_dict (passthrough; verify after download)."""
    return OrderedDict(src_sd)


def convert_stupase_official_vocoder_mel(src_sd: dict) -> dict:
    """Official Vocoder_Mel-16k.pt -> ported VocosVocoder state_dict (passthrough; verify after download)."""
    return OrderedDict(src_sd)


# ──────────────────────────────────────────────────────────
# Official UniPASE (Xiaobin-Rong/unipase on HF Hub) → use_train
# ──────────────────────────────────────────────────────────
# 与 PASE 相同：tasks/se/models/pase_family/unipase/ 是 1:1 vendored 移植，
# 子模块命名与官方一致，checkpoint 无需 remap，直接 passthrough。
# 各模块也支持 from_pretrained 原生加载官方 {cfg, model} 格式，这里的转换器
# 主要用于 key 匹配校验与统一导出。

def convert_unipase_official_passthrough(src_sd: dict) -> dict:
    """Official DeWavLM-Omni/Adapter/Vocoder/PostNet .pt['model'] -> 移植版 state_dict (passthrough)."""
    return OrderedDict(src_sd)


CONVERTERS = {
    ("pase_official", "PASEEncoder"): convert_pase_official_wavlm,
    ("pase_official", "PASEVocoder"): convert_pase_official_vocoder,
    ("pase_official", "PASEDiscriminator"): convert_pase_official_discriminator,
    ("pase_official", "StuPASEEncoder"): convert_stupase_official_dewavlm_r,
    ("pase_official", "StuPASECFM"): convert_stupase_official_cfm,
    ("pase_official", "StuPASEVocoderMel"): convert_stupase_official_vocoder_mel,
    ("unipase_official", "UniPASEEncoder"): convert_unipase_official_passthrough,
    ("unipase_official", "UniPASEAdapter"): convert_unipase_official_passthrough,
    ("unipase_official", "UniPASEVocoder"): convert_unipase_official_passthrough,
    ("unipase_official", "UniPASEPostNet"): convert_unipase_official_passthrough,
}


def _verify_with_model(converted_sd: dict, model_type: str, src_cfg: dict = None) -> tuple[int, int, list[str], list[str]]:
    """Verify key matching with use_train model instance.

    Args:
        src_cfg: 源 checkpoint 的模型配置（如 WavLM 系 ckpt 的 ``ckpt['cfg']``）。
            特征编码器结构依赖配置（如 extractor_mode=layer_norm 时 LayerNorm
            嵌套在 Sequential 内，key 为 ``conv_layers.*.2.1.*``），必须按源配置
            构建目标模型才能得到真实的 key 匹配结果。

    Returns:
        (matched_count, total_target_keys, missing_keys, extra_keys)，或 ``None``
        表示该 model_type 尚未实现验证器（调用方须显式提示未校验）。
    """
    if model_type in ("PASEEncoder", "StuPASEEncoder"):
        from tasks.se.models.pase_family.common.wavlm_frontend import WavLM_feat
        target_keys = set(WavLM_feat(load_pretrained=False, frozen=False).wavlm.state_dict().keys())
    elif model_type == "PASEVocoder":
        from tasks.se.models.pase_family.common.vocoder_backend import WavLMDec
        target_keys = set(WavLMDec().state_dict().keys())
    elif model_type == "PASEDiscriminator":
        from tasks.se.models.pase_family.common.discriminator import CombinedDiscriminator
        target_keys = set(CombinedDiscriminator().state_dict().keys())
    elif model_type == "UniPASEEncoder":
        from tasks.se.models.pase_family.unipase.wavlm_frontend_plc import WavLM_feat_PLC
        if src_cfg is not None:
            # 按源 ckpt 的 cfg 构建（extractor_mode 影响 conv_layers 结构）
            from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig
            target_keys = set(WavLM(WavLMConfig(src_cfg)).state_dict().keys())
        else:
            target_keys = set(WavLM_feat_PLC(load_pretrained=False, frozen=False).wavlm.state_dict().keys())
    elif model_type == "UniPASEAdapter":
        from tasks.se.models.pase_family.unipase.adapter import VocosAdapter
        target_keys = set(VocosAdapter().state_dict().keys())
    elif model_type == "UniPASEVocoder":
        from tasks.se.models.pase_family.unipase.vocoder import VocosVocoder
        target_keys = set(VocosVocoder().state_dict().keys())
    elif model_type == "UniPASEPostNet":
        from tasks.se.models.pase_family.unipase.postnet import TFGridNet
        target_keys = set(TFGridNet().state_dict().keys())
    else:
        # StuPASECFM / StuPASEVocoderMel: submodules not yet ported (Stage 4) or too
        # parameter-heavy to instantiate here with defaults; no verifier exists for
        # these types. 返回 None 与「验证失败」区分，由调用方显式提示未校验。
        return None

    source_keys = set(converted_sd.keys())
    matched = target_keys & source_keys
    missing_in_source = target_keys - source_keys
    extra_in_source = source_keys - target_keys

    return len(matched), len(target_keys), sorted(missing_in_source), sorted(extra_in_source)


def main():
    parser = argparse.ArgumentParser(description="Official HF checkpoints → use_train weight conversion")
    parser.add_argument("--source", required=True,
                        choices=["pase_official", "unipase_official"],
                        help="Source framework")
    parser.add_argument("--model_type", required=True,
                        choices=[
                            "PASEEncoder", "PASEVocoder", "PASEDiscriminator",
                            "StuPASEEncoder", "StuPASECFM", "StuPASEVocoderMel",
                            "UniPASEEncoder", "UniPASEAdapter", "UniPASEVocoder", "UniPASEPostNet",
                        ],
                        help="Target model type")
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", default=None, help="Output path (auto-named if not specified)")
    parser.add_argument("--dry-run", action="store_true", help="Print key mapping only, do not save")
    parser.add_argument("--no-verify", action="store_true", help="Skip model verification (no model instantiation)")
    args = parser.parse_args()

    converter_key = (args.source, args.model_type)
    if converter_key not in CONVERTERS:
        print(f"[ERROR] Conversion from {args.source} → {args.model_type} is not supported.")
        print(f"        Supported combinations: {list(CONVERTERS.keys())}")
        sys.exit(1)

    print(f"[INFO] Loading checkpoint: {args.input}")
    from core.utils.safe_load import safe_torch_load
    # NOTE: allow_unsafe=True bypasses torch's weights_only guard so that old-format
    # checkpoints can be converted. Only run this on a TRUSTED input file — loading an
    # untrusted checkpoint here allows arbitrary code execution.
    ckpt = safe_torch_load(args.input, allow_unsafe=True, purpose="checkpoint conversion (may contain old format)")
    src_sd = _extract_model_state(ckpt)
    print(f"[INFO] Source state_dict key count: {len(src_sd)}")

    src_sd = _strip_prefix(src_sd, "module.")

    converter = CONVERTERS[converter_key]
    converted_sd = converter(src_sd)
    print(f"[INFO] Converted key count: {len(converted_sd)}")

    if args.dry_run:
        print("\n[DRY RUN] Key mapping (first 30):")
        src_keys = list(src_sd.keys())
        cvt_keys = list(converted_sd.keys())
        for i, (s, c) in enumerate(zip(src_keys, cvt_keys)):
            mark = " ✓" if s == c else f" ← {s}"
            print(f"  {c}{mark}")
            if i >= 29:
                print(f"  ... {len(cvt_keys)} keys total")
                break

    if args.no_verify:
        # 显式跳过语义保留，但必须让用户看到产物未经 key 匹配校验
        print("\n[WARN] --no-verify: model verification disabled; the saved checkpoint was NOT checked against the target model.")
    else:
        try:
            verify_result = _verify_with_model(
                converted_sd, args.model_type, src_cfg=ckpt.get("cfg") if isinstance(ckpt, dict) else None
            )
        except Exception as e:
            print(f"\n[ERROR] Verification failed (cannot instantiate target model): {type(e).__name__}: {e}")
            print("[ERROR] Nothing was saved. Fix the verification failure, or pass --no-verify to skip verification explicitly.")
            sys.exit(1)
        if verify_result is None:
            print(f"\n[Verify] SKIPPED: no verifier implemented for model_type={args.model_type}; converted keys were NOT validated.")
        else:
            matched, total, missing, extra = verify_result
            print(f"\n[Verify] Matched: {matched}/{total} keys")
            if missing:
                print(f"[Verify] Target model missing {len(missing)} keys (not found in source):")
                for k in missing[:10]:
                    print(f"  - {k}")
                if len(missing) > 10:
                    print(f"  ... {len(missing)} total")
            if extra:
                print(f"[Verify] Source has {len(extra)} extra keys (not in target model):")
                for k in extra[:10]:
                    print(f"  - {k}")
                if len(extra) > 10:
                    print(f"  ... {len(extra)} total")
            if matched != total or extra:
                # 验证不通过不得写出产物：不完整/错位的 state_dict 会让下游加载失败
                # 或静默部分初始化
                print(
                    f"[ERROR] Verification failed: matched {matched}/{total} target keys, "
                    f"{len(missing)} missing, {len(extra)} extra. Nothing was saved."
                )
                sys.exit(1)
            print("[Verify] Perfect match!")

    if args.dry_run:
        print("\n[DRY RUN] File not saved.")
        return

    out_path = args.output
    if out_path is None:
        out_path = f"checkpoints/{args.model_type.lower()}_{args.source}_converted.pth"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_ckpt = {"model": converted_sd}
    # UniPASE/PASE 系 from_pretrained 需要 {cfg, model} 完整 artifact——
    # 源 ckpt 带 cfg 时一并拷贝，否则转换产物无法被 from_pretrained 加载（KeyError）
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        out_ckpt["cfg"] = ckpt["cfg"]
    if "epoch" in ckpt:
        out_ckpt["source_epoch"] = ckpt["epoch"]
    if "n_steps" in ckpt:
        out_ckpt["source_n_steps"] = ckpt["n_steps"]
    out_ckpt["source_framework"] = args.source

    torch.save(out_ckpt, out_path)
    print(f"\n[OK] Saved: {out_path}")
    # 产物统一经 safe_torch_load 读取（weights_only=True 安全模式；purpose 供审计记录）
    print("     Load: from core.utils.safe_load import safe_torch_load")
    print(f"           state = safe_torch_load('{out_path}', purpose='converted checkpoint model weights')['model']")


if __name__ == "__main__":
    main()
