#!/usr/bin/env python3
"""Export PASE/StuPASE training checkpoints → standalone inference artifacts.

训练保存的是 wrapper state_dict（多数阶段带 ``teacher.``/``student.``/
``encoder.``/``cfm.``/``vocoder.``/``decoder.``/``postnet.`` 前缀；GAP Phase-1
Predictor 等单模型阶段无前缀，整体导出），而下阶段初始化与推理的
``from_pretrained`` 期望独立 artifact ``{cfg, model}``（无前缀）。本脚本剥离
前缀并配 cfg，闭合训练→下阶段/推理的产物链（审计 R-04）。

参考官方 ``pase/utils/create_ckpt*.py``。对应关系::

    Phase-2 (DeWavLMDistillModel, 单模型)     → DeWavLM.pt
    StuP0  (DeWavLMDistillModel, 单模型)      → DeWavLM-R.pt
    Phase-3 (PASE GAN, model_generator)       → Vocoder_Dual.pt
    StuP1  (StuPASEMelVocoderGenerator GAN)   → Vocoder_Mel-16k.pt
    StuP2  (StuPASEFlowTrainModel, 单模型)    → CFM.pt
    UniP1  (DeWavLMDistillModel PLC, 单模型)  → DeWavLM-Omni.pt
    UniP2  (UniPASEVocoderGenerator GAN)      → Vocoder_DWO-L1.pt
    UniP3  (UniPASEAdapterGenerator GAN)      → Adapter.pt
    UniP4  (UniPASEPostNetGenerator GAN)      → PostNet.pt
    GAP-P1 (GAPURGENetPredictor, 单模型)       → Predictor.pt
    GAP-P2 (GAPURGENetPostNetGenerator GAN)   → PostNet.pt

GAP 两阶段的 ckpt 结构与 PASE 家族一致：Phase-1 单模型使用顶层 ``model``
（Predictor 无 wrapper，state_dict 无前缀，artifact 整体复制）；Phase-2 GAN
使用 ``model_generator``，只导出 ``postnet.*`` 子模块。

用法::

    python tools/model_tools/export_pase_artifacts.py \\
        --stage dewavlm \\
        --train-ckpt exp/se/phase2_dewavlm_distill/checkpoints/best_model.tar \\
        --output checkpoints/pase/DeWavLM.pt --verify

    # GAP-URGENet Phase-1：单模型 checkpoint，state_dict 无前缀
    python tools/model_tools/export_pase_artifacts.py \\
        --stage gap_predictor \\
        --train-ckpt exp/se/gap_urgenet_phase1_predictor/checkpoints/best_model.tar \\
        --output checkpoints/pase/Predictor.pt --verify

    # GAP-URGENet Phase-2：GAN generator checkpoint，仅导出 postnet.* 子模块
    python tools/model_tools/export_pase_artifacts.py \\
        --stage gap_postnet \\
        --train-ckpt exp/se/gap_urgenet_phase2_postnet_gan/checkpoints/best_model.tar \\
        --output checkpoints/pase/PostNet.pt --verify
"""

from __future__ import annotations

import argparse
import logging
import os
import stat
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils.file_transaction import publish_staged_files, unlink_file_if_exists  # noqa: E402

logger = logging.getLogger(__name__)

# 各阶段 → (训练 ckpt 顶层 key, wrapper 内子模块前缀)
STAGE_SPEC = {
    "dewavlm":         ("model",           "student.wavlm."),  # PASE Phase-2
    "dewavlm_r":       ("model",           "student.wavlm."),  # StuPASE Phase-0
    "pase_vocoder":    ("model_generator", "decoder."),        # PASE Phase-3
    "stupase_vocoder": ("model_generator", "vocoder."),        # StuPASE Phase-1
    "cfm":             ("model",           "cfm."),            # StuPASE Phase-2
    "unipase_dewavlm_omni": ("model",      "student.wavlm."),  # UniPASE Stage-1 (PLC)
    "unipase_vocoder": ("model_generator", "vocoder."),        # UniPASE Stage-2
    "unipase_adapter": ("model_generator", "adapter."),        # UniPASE Stage-3
    "unipase_postnet": ("model_generator", "postnet."),        # UniPASE Stage-4
    "gap_predictor":   ("model",           ""),                # GAP-URGENet Phase-1（单模型，无前缀）
    "gap_postnet":     ("model_generator", "postnet."),        # GAP-URGENet Phase-2
}


def _strip_prefix(sd: dict, prefix: str) -> dict:
    """返回 key 去掉 *prefix* 的子集（只保留以 prefix 开头的 key）。"""
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _load_train_ckpt(path: str) -> dict:
    # 与同目录 convert_checkpoint.py 一致：经 safe_torch_load 加载用户提供的
    # ckpt（本框架训练 ckpt 的 config 为 plain dict，weights_only 安全路径可用）
    from core.utils.safe_load import safe_torch_load

    ckpt = safe_torch_load(path, purpose="export training artifact")
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError(f"{path} 不是本框架训练 checkpoint（缺 'config' key）")
    return ckpt


def _cfg_from_ckpt(ckpt: dict) -> dict:
    """训练 checkpoint 内的完整 resolved config（dict）。"""
    cfg = ckpt.get("resolved_config") or ckpt.get("config")
    return OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)


def _extract(ckpt: dict, stage: str) -> tuple[dict, dict]:
    """返回 (子模块 state_dict(无前缀), 训练 config)。

    空前缀表示顶层 state_dict 本身就是目标模块（单模型、无 wrapper，如
    GAP-URGENet Phase-1 Predictor），此时整体复制所有 key，不走前缀匹配。
    """
    top_key, prefix = STAGE_SPEC[stage]
    if top_key not in ckpt:
        raise ValueError(
            f"checkpoint 缺顶层 '{top_key}' key（现有: {list(ckpt.keys())}）。"
            f"阶段 '{stage}' 期望 {'GAN' if top_key == 'model_generator' else '单模型'} 训练 checkpoint。"
        )
    if prefix:
        submodule_sd = _strip_prefix(ckpt[top_key], prefix)
        if not submodule_sd:
            raise ValueError(
                f"checkpoint['{top_key}'] 没有 '{prefix}*' 前缀的 key——阶段/前缀不匹配。"
                f"现有前缀样本: {sorted({k.split('.')[0] for k in ckpt[top_key]})[:5]}"
            )
    else:
        submodule_sd = dict(ckpt[top_key])
        if not submodule_sd:
            raise ValueError(
                f"checkpoint['{top_key}'] 的 state_dict 为空——无法导出阶段 '{stage}'。"
            )
    return submodule_sd, _cfg_from_ckpt(ckpt)


_WAVLM_LARGE_CFG_CACHE: dict | None = None


def _wavlm_large_cfg() -> dict:
    """WavLM-Large config（PASE/StuPASE 固定用 Large，结构不随训练改变）。

    直接构造 WavLMConfig + Large override，不实例化 315M WavLM（避免 review #5 性能浪费）。
    ``extractor_mode="layer_norm"`` 必须与官方 WavLM-Large.pt 的 cfg 一致：该模式决定
    conv_layers 归一化层嵌套结构（``conv_layers.*.2.1.*``），否则 state_dict key 形态不匹配；
    ``max_distance=800`` 同样取自官方 cfg，影响 relative position bucket 映射。模块级缓存。
    """
    global _WAVLM_LARGE_CFG_CACHE
    if _WAVLM_LARGE_CFG_CACHE is None:
        from tasks.se.models.pase_family.common.wavlm.WavLM import WavLMConfig
        cfg = WavLMConfig()
        cfg.extractor_mode = "layer_norm"
        cfg.encoder_layers = 24
        cfg.encoder_embed_dim = 1024
        cfg.encoder_ffn_embed_dim = 4096
        cfg.encoder_attention_heads = 16
        cfg.layer_norm_first = True
        cfg.gru_rel_pos = True
        cfg.relative_position_embedding = True
        cfg.max_distance = 800
        cfg.normalize = True
        cfg.mask_prob = 0.8
        # 官方 WavLM-Large.pt / DeWavLM*.pt 的 cfg 为 0.0；WavLMConfig 默认 0.1。
        # 作为 frozen encoder 时 eval 无影响，但作为 StuPASE P0 可训练 student
        # 热启动时会引入官方没有的 dropout，必须显式归零。
        cfg.dropout = 0.0
        cfg.attention_dropout = 0.0
        _WAVLM_LARGE_CFG_CACHE = dict(cfg.__dict__)
    return _WAVLM_LARGE_CFG_CACHE


def _build_cfg(stage: str, config: dict) -> dict:
    """从训练 config 提取目标模块的 ``__init__`` kwargs。"""
    if stage in ("dewavlm", "dewavlm_r", "unipase_dewavlm_omni"):
        # 三个阶段的 encoder 都是官方 WavLM-Large.pt（结构和 cfg 相同），
        # 返回副本，避免调用方修改模块级缓存。
        return dict(_wavlm_large_cfg())
    if stage == "cfm":
        return config["model"]["args"]["cfm_kwargs"]
    if stage == "stupase_vocoder":
        gen_args = config["model"]["generator"]["args"]
        keys = ["input_channels", "dim", "intermediate_dim", "num_layers",
                "num_res", "num_attn", "n_fft", "hop_length"]
        return {k: gen_args[k] for k in keys if k in gen_args}
    if stage == "unipase_vocoder":
        gen_args = config["model"]["generator"]["args"]
        keys = ["input_channels", "dim", "intermediate_dim", "num_layers",
                "num_res", "num_attn", "n_fft", "hop_length"]
        return {k: gen_args[k] for k in keys if k in gen_args}
    if stage == "unipase_adapter":
        gen_args = config["model"]["generator"]["args"]
        keys = ["input_channels", "dim", "intermediate_dim", "num_layers",
                "num_res", "num_attn", "output_channels"]
        return {k: gen_args[k] for k in keys if k in gen_args}
    if stage == "unipase_postnet":
        # 硬索引：缺 postnet_kwargs 时显式 KeyError，而非静默回退全默认参数
        # （review P3-7，与 cfm 分支行为一致）
        return config["model"]["generator"]["args"]["postnet_kwargs"]
    if stage == "gap_predictor":
        # Phase-1 单模型：model.args 即 Predictor TFGridNet 的构造参数
        return dict(config["model"]["args"])
    if stage == "gap_postnet":
        # Phase-2 GAN：PostNet 结构参数位于 generator.args.postnet_kwargs。
        # 剔除本仓库新增的 normalization_mode：官方 TFGridNet 构造器没有该形参，
        # artifact 必须能被官方 from_pretrained/推理链直接加载。本仓库该类默认值
        # 即 legacy_centered，剔除后本地 verify 仍等价。
        cfg = dict(config["model"]["generator"]["args"]["postnet_kwargs"])
        cfg.pop("normalization_mode", None)
        return cfg
    if stage == "pase_vocoder":
        g = config["model"]["generator"]["args"]
        # PASE 配置字段名 → WavLMDec __init__ 参数名映射
        return {
            "input_channels": g["input_channels"],
            "dim": g["vocoder_dim"],
            "intermediate_dim": g["vocoder_intermediate_dim"],
            "num_layers": g["vocoder_num_layers"],
            "n_fft": g["n_fft"],
            "hop_length": g["hop_length"],
        }
    raise ValueError(stage)


def _prepare_output_path(output: str) -> Path:
    """词法绝对化并拒绝 output 路径中的 symlink/特殊节点。"""
    out = Path(os.path.abspath(output))
    current = Path(out.anchor)
    for component in out.parent.parts[1:]:
        current /= component
        try:
            node_stat = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            node_stat = current.lstat()
        if stat.S_ISLNK(node_stat.st_mode):
            raise RuntimeError(f"artifact output path must not contain symlinks: {current}")
        if not stat.S_ISDIR(node_stat.st_mode):
            raise NotADirectoryError(f"artifact output ancestor is not a directory: {current}")
    try:
        target_stat = out.lstat()
    except FileNotFoundError:
        return out
    if stat.S_ISLNK(target_stat.st_mode):
        raise RuntimeError(f"artifact output must not be a symbolic link: {out}")
    if not stat.S_ISREG(target_stat.st_mode):
        raise RuntimeError(f"artifact output must be a regular file: {out}")
    return out


def export(stage: str, train_ckpt: str, output: str, verify: bool) -> dict:
    """导出单个 artifact；验证成功后才事务替换目标文件。"""
    ckpt = _load_train_ckpt(train_ckpt)
    model_sd, config = _extract(ckpt, stage)
    cfg = _build_cfg(stage, config)
    artifact = {"cfg": cfg, "model": model_sd}
    out = _prepare_output_path(output)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=out.parent, prefix=f".{out.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(artifact, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if verify:
            _verify(stage, str(temporary_path))
        publish_staged_files([temporary_path], [out])
    except BaseException:
        unlink_file_if_exists(temporary_path)
        raise
    logger.info("[%s] 导出 %s（%d 个 state_dict key）", stage, out, len(model_sd))
    return artifact


def _verify(stage: str, artifact_path: str) -> None:
    """用 ``from_pretrained`` 语义加载导出的 artifact，确认零 missing/unexpected key。

    from_pretrained 内部用 ``strict=False``，故这里手动检查 missing/unexpected
    列表（审计 R-04 验收：strict reload round-trip，零 key 命中必须 fail closed）。
    """
    from core.utils.safe_load import safe_torch_load

    ckpt = safe_torch_load(
        artifact_path,
        map_location="cpu",
        purpose="verify exported PASE artifact",
    )
    if stage in ("dewavlm", "dewavlm_r", "unipase_dewavlm_omni"):
        # 按 artifact 自带 cfg 构建：extractor_mode 决定 conv_layers key 形态，
        # 用默认 cfg 会与 layer_norm 模式的权重失配。
        from tasks.se.models.pase_family.common.wavlm.WavLM import WavLM, WavLMConfig
        m = WavLM(WavLMConfig(ckpt["cfg"]))
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "cfm":
        from tasks.se.models.pase_family.stupase.flow.cfm import CFM
        m = CFM(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "stupase_vocoder":
        from tasks.se.models.pase_family.stupase.vocos_vocoder import VocosVocoder
        m = VocosVocoder(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "unipase_vocoder":
        from tasks.se.models.pase_family.unipase.vocoder import VocosVocoder
        m = VocosVocoder(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "unipase_adapter":
        from tasks.se.models.pase_family.unipase.adapter import VocosAdapter
        m = VocosAdapter(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "unipase_postnet":
        from tasks.se.models.pase_family.unipase.postnet import TFGridNet
        m = TFGridNet(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "gap_predictor":
        from tasks.se.models.pase_family.gap_urgenet.predictor.tfgridnet import TFGridNet
        m = TFGridNet(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "gap_postnet":
        from tasks.se.models.pase_family.gap_urgenet.postnet.tfgrid_cws import TFGridNet
        m = TFGridNet(**ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    elif stage == "pase_vocoder":
        from tasks.se.models.pase_family.common.vocoder_backend import WavLMDec
        # 单流 artifact 没有 proj.*；WavLMDec 默认 use_dual_stream=True，直接
        # 构造会把单流 Vocoder_L24 误判为双流并报 missing proj。按 state_dict
        # 自动识别，与 WavLMDec.from_pretrained 的语义一致。
        dual = any(key.startswith("proj.") for key in ckpt["model"])
        m = WavLMDec(frozen=True, use_dual_stream=dual, **ckpt["cfg"])
        result = m.load_state_dict(ckpt["model"], strict=False)
    else:
        raise ValueError(stage)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(
            f"[verify {stage}] key 不匹配: missing={len(missing)} unexpected={len(unexpected)} "
            f"(missing[:3]={missing[:3]} unexpected[:3]={unexpected[:3]})"
        )
    logger.info("[verify %s] ✓ strict reload 成功（零 missing/unexpected）", stage)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=list(STAGE_SPEC),
                   help="导出哪个阶段的 artifact")
    p.add_argument("--train-ckpt", required=True,
                   help="训练 checkpoint 路径 (latest_model.tar / best_model.tar / model_XXXX.pth)")
    p.add_argument("--output", required=True, help="输出 artifact 路径 (如 DeWavLM.pt)")
    p.add_argument("--verify", action="store_true",
                   help="导出后用 from_pretrained strict reload 验证（零 missing/unexpected key）")
    args = p.parse_args()
    export(args.stage, args.train_ckpt, args.output, args.verify)


if __name__ == "__main__":
    main()
