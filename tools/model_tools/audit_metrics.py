"""按统一合同审计 SE/AEC/TSE 模型的 Params、MACs 与 RTF。

YAML 始终通过 ``load_merged_config`` 读取，不修改或回写配置。``--task`` 控制模型
注册域，``--scope generator`` 可只审计 GAN generator。参数统计与 profile 分阶段执行：
profile 失败仍保留 Params 行。FLOPs 没有独立可靠测量时固定为 ``not measured``，
不从 MACs 推算；MOS 只允许在 ``evaluate/eval_pipeline_*.py`` 离线评估。

RTF 语义：按 ``--rtf_duration`` 秒整段波形经任务 adapter（waveform→STFT→模型→iSTFT）
一次前向计时，属于**整段/批**口径，不是逐帧流式口径。对 OnlineSpatialNet 这类模型，逐帧
``forward_streaming`` 的 CPU RTF 会明显更高（单帧算子数多、cache 更新与形状操作占主导，
实测 4 s/8 kHz 整段 3,815 次 aten 调用 vs 逐帧累计 989,853 次）；mamba 分支在 CPU 上走
纯 PyTorch 参考 step，其 RTF 不代表 CUDA 内核成本。比较流式成本时请单独测量逐帧路径。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# 脚本位于 tools/model_tools/，需回退两级到仓库根目录。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from core.data.constants import normalize_data_mode
from core.utils.config import load_merged_config

# 项目根目录（use_train），用于解析相对路径的 YAML。
REPO_ROOT = _REPO_ROOT


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _load_cfg(path: str) -> dict:
    """只读加载完整继承链；审计器不向 YAML 回写任何字段。"""
    return load_merged_config(path)


# ---------------------------------------------------------------------------
# 内置审计列表：官方参数量/算力来自论文报告值，仅填数字不引用论文与仓库。
# 每条：label, yaml 或 inline, official_params_M, official_macs_G（单位：M 参数、G MACs/1s）。
# 本脚本按 1s 波形经任务 adapter 运行（声明 n_channel>1 的空间模型使用 [1, C, 1s] 多通道波形，
# 逐通道 STFT 后堆叠为 [B, C, F, T, 2]）；THOP 对 torch.stft/istft、functional、复数与自定义算子覆盖不完整。
# 因此 MACs 始终按 partial/unverified 披露，不能声称等于完整端到端算力。
# ---------------------------------------------------------------------------

# STFT 预设（16kHz 常用）
STFT_16K_512 = {"n_fft": 512, "hop_length": 256, "win_length": 512, "window": "hann"}
STFT_16K_320 = {"n_fft": 320, "hop_length": 160, "win_length": 320, "window": "hann"}
STFT_48K_960 = {"n_fft": 960, "hop_length": 480, "win_length": 960, "window": "hann"}
# 论文 bsrnn_sfi: 2048-point window, 512-point hop, Hanning → 41 subbands
STFT_48K_2048 = {"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann"}

AUDIT_ENTRIES = [
    # -------- 1. URGENT2026 BSRNN baseline（模型 A：960/480, 34 子带, N=196, mask+residual）--------
    {"label": "URGENT2026 BSRNN baseline", "yaml": "configs/se/urgent2026/usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml", "official_params_M": 37.8, "official_macs_G": None},

    # -------- 2. USR-BSRNN / SFI-BSRNN（模型 B：2048/512, 41 子带, N=16, 仅 mask；论文 0.35M / 0.52G）--------
    {"label": "USR-BSRNN causal URGENT26", "yaml": "configs/se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml", "official_params_M": 0.35, "official_macs_G": 0.52},

    # -------- 3. DPDFNet-2 (论文表: 2.49M, 1.35G MACs) --------
    {"label": "DPDFNet-2 (urgent26)", "yaml": "configs/se/urgent2026/dpdfnet2_causal_urgent26_offline_hybrid.yaml", "official_params_M": 2.49, "official_macs_G": 1.35},

    # -------- 4. DeepFilterNet2（原版，Rikorose/DeepFilterNet-main）--------
    # 8/8、gru_variant=squeezed，论文 +Simplified DNN ~2.31M/0.36G；与 config 表、tasks/se/models/discriminative/deepfilternet.py 一致。
    {
        "label": "DeepFilterNet2 (inline)",
        "inline": {"model": {"type": "DeepFilterNet", "args": {"samplerate": 16000, "dprnn_num_blocks": 0, "enc_concat": True, "enc_lin_groups": 8, "lin_groups": 8, "erb_apply_above_bins": None, "gru_variant": "squeezed"}}, "stft": STFT_16K_320, "data": {"sample_rate": 16000}},
        "official_params_M": 2.31,
        "official_macs_G": 0.36,
    },
    # DeepFilterNet3（官方 48kHz；STFT 960/480；论文 ~None M / NoneG）
    {
        "label": "DeepFilterNet3 (48k inline)",
        "inline": {"model": {"type": "DeepFilterNet", "args": {"samplerate": 48000, "dprnn_num_blocks": 0, "enc_concat": False, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": 96}}, "stft": STFT_48K_960, "data": {"sample_rate": 48000}},
        "official_params_M": None,
        "official_macs_G": None,
    },
    # DPDFNet 系列（DPDFNet 官方表: baseline 2.31/0.36, dpdfnet2 2.49/1.35, dpdfnet4 2.84/2.36, dpdfnet8 3.54/4.37）
    {
        "label": "DPDFNet-2 (inline)",
        "inline": {"model": {"type": "DeepFilterNet", "args": {"samplerate": 16000, "dprnn_num_blocks": 2, "enc_concat": True, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": None}}, "stft": STFT_16K_320, "data": {"sample_rate": 16000}},
        "official_params_M": 2.49,
        "official_macs_G": 1.35,
    },
    {
        "label": "DPDFNet-4 (inline)",
        "inline": {"model": {"type": "DeepFilterNet", "args": {"samplerate": 16000, "dprnn_num_blocks": 4, "enc_concat": True, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": None}}, "stft": STFT_16K_320, "data": {"sample_rate": 16000}},
        "official_params_M": 2.84,
        "official_macs_G": 2.36,
    },
    {
        "label": "DPDFNet-8 (inline)",
        "inline": {"model": {"type": "DeepFilterNet", "args": {"samplerate": 16000, "dprnn_num_blocks": 8, "enc_concat": True, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": None}}, "stft": STFT_16K_320, "data": {"sample_rate": 16000}},
        "official_params_M": 3.54,
        "official_macs_G": 4.37,
    },

    # -------- 5. GTCRN (gtcrn-main: 23.67K trainable, 33 MMACs；论文 39.6 MMACs/s) --------
    # 本实现 trainable=23.67K、total~48K（含固定 ERB），与官方一致。详见 docs/architecture/SE_Architecture.md。
    {"label": "GTCRN (voicebank_demand)", "yaml": "configs/se/voicebank_demand/gtcrn_causal_offline.yaml", "official_params_M": 0.0237, "official_macs_G": 0.0396},

    # -------- 6. FastEnhancer (官方 https://github.com/aask1357/fastenhancer Table 1/4, 16kHz) --------
    {"label": "FastEnhancer-T (inline)", "inline": {"model": {"type": "FastEnhancer", "args": {"variant": "T"}}, "stft": STFT_16K_512, "data": {"sample_rate": 16000}}, "official_params_M": 0.022, "official_macs_G": 0.06},
    {"label": "FastEnhancer-B (inline)", "inline": {"model": {"type": "FastEnhancer", "args": {"variant": "B"}}, "stft": STFT_16K_512, "data": {"sample_rate": 16000}}, "official_params_M": 0.091, "official_macs_G": 0.262},
    {"label": "FastEnhancer-S (inline)", "inline": {"model": {"type": "FastEnhancer", "args": {"variant": "S"}}, "stft": STFT_16K_512, "data": {"sample_rate": 16000}}, "official_params_M": 0.194, "official_macs_G": 0.664},
    {"label": "FastEnhancer-M (inline)", "inline": {"model": {"type": "FastEnhancer", "args": {"variant": "M"}}, "stft": STFT_16K_512, "data": {"sample_rate": 16000}}, "official_params_M": 0.492, "official_macs_G": 2.9},
    {"label": "FastEnhancer-L (inline)", "inline": {"model": {"type": "FastEnhancer", "args": {"variant": "L"}}, "stft": STFT_16K_512, "data": {"sample_rate": 16000}}, "official_params_M": 1.105, "official_macs_G": 12.0},
]


def _collect_from_entries(entries: list[dict]) -> list[tuple[str, dict, dict]]:
    """从内置条目生成 (label, cfg, official_info)。"""
    result = []
    for e in entries:
        label = e["label"]
        official = {"params_M": e.get("official_params_M"), "macs_G": e.get("official_macs_G")}
        if "yaml" in e and e["yaml"]:
            path = os.path.join(REPO_ROOT, e["yaml"])
            if not os.path.isfile(path):
                print(f"[WARN] Skip {label}: YAML not found {path}", file=sys.stderr)
                continue
            try:
                cfg = _load_cfg(path)
            except Exception as err:
                print(f"[WARN] Skip {label}: {err}", file=sys.stderr)
                continue
            result.append((label, cfg, official))
        elif "inline" in e and e["inline"]:
            cfg = e["inline"]
            if "model" not in cfg or "stft" not in cfg:
                print(f"[WARN] Skip {label}: inline must have model and stft", file=sys.stderr)
                continue
            result.append((label, cfg, official))
        else:
            print(f"[WARN] Skip {label}: need yaml or inline", file=sys.stderr)
    return result


def _config_paths(configs_dir: str, recursive: bool) -> list[str]:
    """稳定枚举目录中的 YAML；递归模式同时支持 ``.yaml`` 与 ``.yml``。"""
    paths = []
    if recursive:
        for root, _, filenames in os.walk(configs_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in {".yaml", ".yml"}:
                    paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(configs_dir):
            path = os.path.join(configs_dir, filename)
            if (
                os.path.isfile(path)
                and os.path.splitext(filename)[1].lower() in {".yaml", ".yml"}
            ):
                paths.append(path)
    return sorted(paths)


def _load_cli_candidate(path: str, label: str) -> tuple[str, dict | None, dict]:
    """加载单个候选；失败仍返回占位项，由审计输出结构化失败行。"""
    try:
        return label, _load_cfg(path), {}
    except Exception as exc:
        return label, None, {
            "audit_error": f"{type(exc).__name__}: {exc}",
            "audit_stage": "config_load",
        }


def _collect_configs_cli(args) -> list[tuple[str, dict | None, dict]] | None:
    """枚举 CLI 配置；同一规范路径只保留首次发现的候选。"""
    out = []
    seen_paths: set[str] = set()

    def append_candidate(path: str, label: str) -> None:
        path = os.path.abspath(path)
        canonical_path = os.path.realpath(path)
        if canonical_path in seen_paths:
            return
        seen_paths.add(canonical_path)
        if not os.path.isfile(path):
            out.append((
                label,
                None,
                {
                    "audit_error": f"FileNotFoundError: config not found: {path}",
                    "audit_stage": "config_load",
                },
            ))
            return
        out.append(_load_cli_candidate(path, label))

    if args.config:
        for raw_path in args.config:
            path = os.path.abspath(raw_path)
            label = os.path.splitext(os.path.basename(path))[0]
            append_candidate(path, label)
    if args.configs_dir:
        directory = os.path.abspath(args.configs_dir)
        if not os.path.isdir(directory):
            print(f"[WARN] Configs dir not found: {directory}", file=sys.stderr)
        else:
            for path in _config_paths(directory, args.recursive):
                relative = os.path.relpath(path, directory)
                label = os.path.splitext(relative)[0].replace(os.sep, "/")
                append_candidate(path, label)
    if not out:
        return None
    return out


def _build_stft(stft_cfg):
    from core.stft import build_stft_istft

    return build_stft_istft({**stft_cfg, "window": stft_cfg.get("window", "hann")})


def _model_channels(model) -> int:
    """模型声明的原生麦克风通道数；mono 模型返回 1。"""
    channels = getattr(model, "n_channel", 1)
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        return 1
    return channels


def _stack_channel_spectra(stft, wav):
    """[B, C, T] 波形逐通道 STFT 后堆叠为 [B, C, F, T, 2]。"""
    return torch.stack([stft(wav[:, index]) for index in range(wav.shape[1])], dim=1)


class _SEProfileAdapter(nn.Module):
    """SE waveform → STFT → dict/谱模型 → iSTFT adapter；原生多通道模型接受 [B, C, T]。"""

    def __init__(self, model, stft_cfg, channels: int = 1):
        super().__init__()
        self.model = model
        self.stft = _build_stft(stft_cfg)
        self.channels = channels

    def forward(self, wav):
        spec = _stack_channel_spectra(self.stft, wav) if wav.ndim == 3 else self.stft(wav)
        out = self.model(spec)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return self.stft.inverse(out, wav.shape[-1])


class _AECProfileAdapter(nn.Module):
    """AEC 的 1 秒 mic + farend dict 谱进谱出 adapter。"""

    def __init__(self, model, stft_cfg):
        super().__init__()
        self.model = model
        self.stft = _build_stft(stft_cfg)

    def forward(self, packed_wav):
        mic = packed_wav[:, 0]
        farend = packed_wav[:, 1]
        batch = {
            "mic_spectra": self.stft(mic),
            "farend_spectra": self.stft(farend),
        }
        out = self.model(batch)["nearend_estimate_spectra"]
        return self.stft.inverse(out, mic.shape[-1])


class _SSProfileAdapter(nn.Module):
    """SS adapter：支持 waveform-native TDANet 与 external-STFT TIGER。"""

    def __init__(self, model, stft_cfg=None, sample_rate: int = 16000, channels: int = 1):
        super().__init__()
        self.model = model
        self.input_domain = getattr(model, "input_domain", "waveform")
        self.stft = _build_stft(stft_cfg) if self.input_domain == "spectrum" else None
        self.sample_rate = sample_rate
        self.channels = channels

    def forward(self, mixture):
        if mixture.ndim not in (2, 3):
            raise ValueError(f"SS mixture must have shape [B, T] or [B, C, T], got {tuple(mixture.shape)}")
        if mixture.ndim == 3 and self.stft is None:
            raise ValueError("SS waveform-native model does not accept multichannel waveform input")
        lengths = torch.full(
            (mixture.shape[0],), mixture.shape[-1], dtype=torch.int64, device=mixture.device
        )
        if mixture.ndim == 3:
            model_input = _stack_channel_spectra(self.stft, mixture)
        else:
            model_input = self.stft(mixture) if self.stft is not None else mixture
        output = self.model(model_input, lengths, fs=self.sample_rate)
        if not isinstance(output, dict):
            raise TypeError("SS model output must be a mapping")
        result = output.get("est_sources")
        if result is None:
            result = output.get("est_sources_spectra")
        if result is None:
            raise KeyError("SS model output requires est_sources or est_sources_spectra")
        return result


class _TSEProfileAdapter(nn.Module):
    """TSE mixture + enrollment 的 cue-aware dict adapter。"""

    def __init__(self, model, stft_cfg, mix_samples: int, enroll_samples: int):
        super().__init__()
        self.model = model
        self.stft = _build_stft(stft_cfg)
        self.mix_samples = mix_samples
        self.enroll_samples = enroll_samples
        self.features = model.features_config
        if self.features["spkemb"]["enabled"] and not model.joint_training:
            raise ValueError(
                "Params-only: joint_training=false spkemb requires a real pre-extracted "
                "spk_embedding; synthetic enrollment is not equivalent"
            )

    def forward(self, packed_wav):
        mixture = packed_wav[:, 0, :self.mix_samples]
        enrollment = packed_wav[:, 1, :self.enroll_samples]
        batch = {
            "mix_spectra": self.stft(mixture),
            "mixture": mixture,
            "sample_rate": self.model.sample_rate,
        }
        if any(self.features[name]["enabled"] for name in ("listen", "usef", "tfmap")):
            batch["enroll_spectra"] = self.stft(enrollment)
            batch["enroll_spectra_lengths"] = torch.full(
                (enrollment.shape[0],), batch["enroll_spectra"].shape[-2],
                dtype=torch.int32, device=enrollment.device,
            )
        if self.features["context"]["enabled"] or (
            self.features["spkemb"]["enabled"] and self.model.joint_training
        ) or (self.features["tfmap"]["enabled"] and self.features["tfmap"].get("type") == "embedding"):
            batch["enroll_wav"] = enrollment
            batch["enroll_lengths"] = torch.full(
                (enrollment.shape[0],), enrollment.shape[-1],
                dtype=torch.int32, device=enrollment.device,
            )
        out = self.model(batch)["target_estimate_spectra"]
        return self.stft.inverse(out, mixture.shape[-1])


def _named_profile_input(task, sample_rate, duration, enroll_duration, channels: int = 1):
    """生成不依赖模型前向是否成功的命名输入合同。"""
    mix_samples = int(sample_rate * duration)
    waveform_shape = [1, channels, mix_samples] if channels > 1 else [1, mix_samples]
    if task == "aec":
        return {
            "mic": {"shape": [1, mix_samples], "sample_rate_hz": sample_rate, "duration_seconds": duration},
            "farend": {"shape": [1, mix_samples], "sample_rate_hz": sample_rate, "duration_seconds": duration},
        }
    if task == "tse":
        enroll_samples = int(sample_rate * enroll_duration)
        return {
            "mixture": {"shape": [1, mix_samples], "sample_rate_hz": sample_rate, "duration_seconds": duration},
            "enrollment": {"shape": [1, enroll_samples], "sample_rate_hz": sample_rate, "duration_seconds": enroll_duration},
        }
    if task == "ss":
        return {
            "mixture": {"shape": waveform_shape, "sample_rate_hz": sample_rate, "duration_seconds": duration},
        }
    return {
        "waveform": {"shape": waveform_shape, "sample_rate_hz": sample_rate, "duration_seconds": duration},
    }


def _make_profile_adapter(model, task, stft_cfg, sample_rate, duration, enroll_duration, device):
    """返回真实任务 adapter、打包输入张量与命名输入合同。"""
    channels = _model_channels(model)
    named_input = _named_profile_input(task, sample_rate, duration, enroll_duration, channels)
    mix_samples = int(sample_rate * duration)
    if task == "aec":
        adapter = _AECProfileAdapter(model, stft_cfg)
        profile_input = torch.zeros(1, 2, mix_samples, device=device)
    elif task == "ss":
        if getattr(model, "input_domain", "waveform") == "spectrum" and not stft_cfg:
            raise ValueError("SS spectrum profile requires stft")
        sample_rate_value = int(getattr(model, "sample_rate", sample_rate))
        adapter = _SSProfileAdapter(model, stft_cfg, sample_rate_value, channels)
        profile_input = (
            torch.zeros(1, channels, mix_samples, device=device)
            if channels > 1
            else torch.zeros(1, mix_samples, device=device)
        )
    elif task == "tse":
        enroll_samples = int(sample_rate * enroll_duration)
        adapter = _TSEProfileAdapter(model, stft_cfg, mix_samples, enroll_samples)
        profile_input = torch.zeros(1, 2, max(mix_samples, enroll_samples), device=device)
    else:
        adapter = _SEProfileAdapter(model, stft_cfg, channels)
        profile_input = (
            torch.zeros(1, channels, mix_samples, device=device)
            if channels > 1
            else torch.zeros(1, mix_samples, device=device)
        )
    return adapter.to(device).eval(), profile_input, named_input


def _synchronize_cuda(device: str | torch.device) -> None:
    """对 ``cuda`` 与 ``cuda:N`` 同步实际计时设备。"""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def _measure_rtf(adapter, profile_input, duration, repeats, device):
    """只测 adapter 前向耗时，不重复运行 MACs profiler。"""
    with torch.no_grad():
        adapter(profile_input)
        _synchronize_cuda(device)
        started = time.perf_counter()
        for _ in range(repeats):
            adapter(profile_input)
        _synchronize_cuda(device)
    latency_ms = (time.perf_counter() - started) / repeats * 1000
    return latency_ms / (duration * 1000)


def _resolve_task(cfg: dict, task: str = "auto") -> str:
    """按显式参数、标准化 data.mode 或模型 type 解析任务域。"""
    if task != "auto":
        return task
    mode = (cfg.get("data") or {}).get("mode")
    if mode is not None:
        return normalize_data_mode(mode).task
    model_cfg = cfg.get("model") or {}
    probe_cfg = model_cfg.get("generator", model_cfg)
    model_type = str(probe_cfg.get("type", ""))
    if model_type in {"TDANet", "TIGER"}:
        return "ss"
    if model_type in {"DeepVQE", "CAGCRN", "UltraDPCAEC", "DBGNet", "LocalVQE"}:
        return "aec"
    if model_type == "TSE_BSRNN_SPK":
        return "tse"
    return "se"


def _build_audit_models(cfg: dict, task: str, scope: str, device: str):
    """构建参数统计域及可 profile 的主模型。"""
    from functools import partial
    from core.models import register_task_models
    from core.models.build import build_model

    model_cfg = cfg.get("model") or {}
    stft_cfg = cfg.get("stft")
    task = _resolve_task(cfg, task)
    register = partial(register_task_models, task)

    if "generator" not in model_cfg:
        if scope == "generator":
            raise ValueError("--scope generator requires model.generator; non-GAN config has only task scope")
        selected = model_cfg
        model_type = selected.get("type")
        register(only=str(model_type)) if model_type else register()
        model = build_model(selected, stft_cfg=stft_cfg).to(device).eval()
        return [model], model, task, "task", "task"

    generator_cfg = model_cfg["generator"]
    generator_type = generator_cfg.get("type")
    register(only=str(generator_type)) if generator_type else register()
    generator = build_model(generator_cfg, stft_cfg=stft_cfg).to(device).eval()
    if scope == "generator":
        return [generator], generator, task, "generator", "generator"

    models = [generator]
    discriminator_cfg = model_cfg.get("discriminator")
    if discriminator_cfg:
        discriminator_type = discriminator_cfg.get("type")
        register(only=str(discriminator_type)) if discriminator_type else register()
        models.append(build_model(discriminator_cfg).to(device).eval())
    return models, generator, task, "task", "generator"


def _audit_one(
    label: str,
    cfg: dict,
    device: str,
    duration: float,
    profile: bool,
    profile_verbose: bool = False,
    rtf_duration: float = 30.0,
    rtf_repeats: int = 30,
    task: str = "auto",
    scope: str = "task",
    contract_out: dict | None = None,
    enroll_duration: float = 3.0,
) -> tuple[str, int, int, float | None, str | None, float | None, int]:
    """先统计 Params，再分别测 MACs 与长时 RTF；七元组返回保持兼容。"""
    from core.utils.model_profile import NOT_MEASURED, make_complexity_contract, profile_model

    for name, value in (
        ("duration", duration),
        ("rtf_duration", rtf_duration),
        ("enroll_duration", enroll_duration),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive, got {value!r}")
    if (
        not isinstance(rtf_repeats, int)
        or isinstance(rtf_repeats, bool)
        or rtf_repeats <= 0
    ):
        raise ValueError(
            f"rtf_repeats must be a positive integer, got {rtf_repeats!r}"
        )

    models, model, resolved_task, params_scope, forward_scope = _build_audit_models(
        cfg, task, scope, device
    )
    total_p = sum(count_params(item)[0] for item in models)
    train_p = sum(count_params(item)[1] for item in models)
    data = cfg.get("data") or {}
    runtime_sample_rate = next(
        (
            getattr(model, name)
            for name in ("target_fs", "sample_rate", "samplerate", "model_sr")
            if getattr(model, name, None) is not None
        ),
        None,
    )
    sample_rate = int(data.get("sample_rate") or runtime_sample_rate or 16000)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    named_input = _named_profile_input(
        resolved_task, sample_rate, duration, enroll_duration
    )
    macs = None
    macs_error = None
    rtf = None
    rtf_error = None
    rtf_measured_duration = None
    stft_cfg = cfg.get("stft")
    macs_scope = f"{resolved_task} {forward_scope} adapter path"

    if profile:
        if not stft_cfg and resolved_task != "ss":
            macs_error = "Params-only: profile requires stft for spectrum task adapter"
            rtf_error = macs_error
        else:
            try:
                adapter, profile_input, named_input = _make_profile_adapter(
                    model, resolved_task, stft_cfg, sample_rate, duration,
                    enroll_duration, device,
                )
                ret = profile_model(
                    adapter,
                    sample_rate=sample_rate,
                    duration=duration,
                    device=device,
                    warmup=0,
                    repeats=rtf_repeats if rtf_duration == duration else 1,
                    label=label,
                    quiet=not profile_verbose,
                    params_scope=params_scope,
                    params_source="runtime model.parameters()",
                    macs_scope=macs_scope,
                    profile_input=profile_input,
                    input_description=named_input,
                )
                macs = ret.get("macs")
                macs_error = ret.get("macs_error")
                rtf = ret.get("rtf")
                rtf_error = ret.get("rtf_error")
                if rtf is not None:
                    rtf_measured_duration = float(duration)
                if contract_out is not None:
                    contract_out.update(ret["complexity"])
                    contract_out["Params"].update(
                        total=total_p, trainable=train_p, scope=params_scope
                    )
                    if len(models) > 1:
                        contract_out["MACs"]["coverage"] += (
                            "; discriminator excluded from forward-path MACs"
                        )
            except Exception as exc:
                macs_error = f"Params-only: {type(exc).__name__}: {exc}"
                rtf_error = f"RTF profile failed: {type(exc).__name__}: {exc}"

            if rtf_duration != duration:
                rtf = None
                rtf_measured_duration = None
                try:
                    rtf_adapter, rtf_input, _ = _make_profile_adapter(
                        model, resolved_task, stft_cfg, sample_rate, rtf_duration,
                        enroll_duration, device,
                    )
                    rtf = _measure_rtf(
                        rtf_adapter, rtf_input, rtf_duration, rtf_repeats, device
                    )
                    rtf_error = None
                    rtf_measured_duration = float(rtf_duration)
                except Exception as exc:
                    rtf_error = f"RTF profile failed: {type(exc).__name__}: {exc}"

    if contract_out is not None and not contract_out:
        coverage = macs_error or NOT_MEASURED
        if len(models) > 1:
            coverage += "; discriminator excluded from forward-path MACs"
        contract_out.update(
            make_complexity_contract(
                total_params=total_p,
                trainable_params=train_p,
                params_scope=params_scope,
                params_source="runtime model.parameters()",
                macs=macs,
                macs_backend=NOT_MEASURED,
                macs_input=named_input,
                macs_scope=macs_scope,
                macs_coverage=coverage,
            )
        )
    if contract_out is not None:
        contract_out["RTF"] = {
            "value": rtf,
            "error": rtf_error,
            "duration_seconds": rtf_measured_duration,
        }

    return label, total_p, train_p, macs, macs_error, rtf, sample_rate


def _fmt_official(official: dict) -> str:
    """仅输出官方参数量与算力数字，不引用论文/仓库。"""
    parts = []
    if official.get("params_M") is not None:
        parts.append(f"{official['params_M']}M")
    if official.get("macs_G") is not None:
        parts.append(f"{official['macs_G']}G")
    return " / ".join(str(p) for p in parts) if parts else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="按统一合同审计 SE/AEC/TSE/SS Params、MACs 与 RTF；不回写 YAML。")
    parser.add_argument("--config", "-C", nargs="+", metavar="YAML", help="仅审计这些 YAML（不跑内置列表）")
    parser.add_argument("--configs_dir", metavar="DIR", help="仅审计该目录下 YAML（不跑内置列表）")
    parser.add_argument("--recursive", "-r", action="store_true", help="与 --configs_dir 同用：递归子目录")
    parser.add_argument("--task", choices=("auto", "se", "aec", "tse", "ss"), default="auto", help="模型注册任务域；默认从配置自动推断")
    parser.add_argument("--scope", choices=("task", "generator"), default="task", help="Params 统计任务模型或仅 GAN generator；MACs 始终对应 generator/单模型前向")
    parser.add_argument("--device", default="cpu", help="device: cpu 或 cuda")
    parser.add_argument("--duration", type=float, default=1.0, help="profile 输入时长(秒)，用于 MACs")
    parser.add_argument("--enroll_duration", type=float, default=3.0, help="TSE enrollment 时长(秒)，默认 3.0")
    parser.add_argument(
        "--rtf_duration",
        type=float,
        default=30.0,
        help="RTF 测试时长(秒)：按该时长整段 wav 一次前向计时，默认 30.0；改小可加快试跑",
    )
    parser.add_argument("--rtf_repeats", type=int, default=30, help="RTF 计时重复次数，默认 30")
    parser.add_argument("--no_profile", action="store_true", help="只统计参数量，不测 MACs/RTF")
    parser.add_argument("--verbose", "-v", action="store_true", help="每个模型单独打印 profile 详情")
    args = parser.parse_args()

    # 配置来源：优先 CLI，否则内置列表。CLI 目录模式对每个发现文件保留一项，
    # 包括配置加载或模型构建失败，禁止因单项异常缩短覆盖范围。
    if args.config or args.configs_dir:
        configs = _collect_configs_cli(args)
        mode = "cli"
    else:
        configs = _collect_from_entries(AUDIT_ENTRIES)
        mode = "built_in"
    if not configs:
        sys.exit(1)

    records = []
    for label, cfg, official in configs:
        if cfg is None:
            records.append({
                "label": label,
                "status": "failed",
                "stage": official.get("audit_stage", "config_load"),
                "error": official.get("audit_error", "unknown config load error"),
            })
            continue
        try:
            contract = {}
            row = _audit_one(
                label, cfg,
                device=args.device,
                duration=args.duration,
                profile=not args.no_profile,
                profile_verbose=args.verbose,
                rtf_duration=args.rtf_duration,
                rtf_repeats=args.rtf_repeats,
                task=args.task,
                scope=args.scope,
                contract_out=contract,
                enroll_duration=args.enroll_duration,
            )
            name, total, trainable, macs, macs_error, rtf, sample_rate = row
            records.append({
                "label": name,
                "status": "ok",
                "sample_rate_hz": sample_rate,
                "params": {"total": total, "trainable": trainable},
                "macs": macs,
                "macs_error": macs_error,
                "rtf": rtf,
                "rtf_error": contract["RTF"]["error"],
                "contract": contract,
                "official": {
                    "params_M": official.get("params_M"),
                    "macs_G": official.get("macs_G"),
                },
            })
        except Exception as exc:
            records.append({
                "label": label,
                "status": "failed",
                "stage": "model_build_or_params",
                "error": f"{type(exc).__name__}: {exc}",
            })

    for record in records:
        print("[AUDIT] " + json.dumps(record, ensure_ascii=False, sort_keys=True))
    failed = sum(record["status"] == "failed" for record in records)
    print(
        "[SUMMARY] "
        + json.dumps(
            {
                "mode": mode,
                "discovered": len(configs),
                "reported": len(records),
                "ok": len(records) - failed,
                "failed": failed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    # 有失败项即非零退出，调用方（CI/脚本）才能感知覆盖缺口
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
