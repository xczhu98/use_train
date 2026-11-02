#!/usr/bin/env python3
"""use_train – unified inference entry（支持 SE、AEC 与 TSE）。

SS/SSL 配置会在任务解析后 fail-fast：SS 使用 ``tasks.ss.inference`` 的 adapter/extractor，
SSL 当前仅支持 scene-manifest/dataset 训练与合同校验；二者均不使用根入口的 ``--input_dir``。

Usage — Speech enhancement (GTCRN / BSRNN / DeepFilterNet / FastEnhancer / FastULCNet / ULUNAS):
    python inference.py -C configs/se/voicebank_demand/gtcrn_causal_offline.yaml \
        --checkpoint exp/.../best_model.tar \
        --input_dir  data/test/noisy \
        --output_dir results/enhanced

Usage — Echo cancellation (DeepVQE AEC):
    python inference.py -C configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml \
        --checkpoint exp/.../best_model.tar \
        --input_mic_dir  data/aec_test/mic \
        --input_far_dir  data/aec_test/farend \
        --output_dir results/aec_enhanced

    # Microsoft ICASSP blind layout: one folder with paired *_mic.wav + *_lpb.wav
    # (see https://github.com/microsoft/AEC-Challenge )
    python inference.py -C configs/aec/microsoft_challenge/deepvqe/deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml \
        --checkpoint exp/.../best_model.tar \
        --aec_flat_dir /path/to/blind_test_set_icassp2023/doubletalk \
        --output_dir results/aec_out

Usage — Target speaker extraction (REAL-TSE / use_train TSE):
    python inference.py -C configs/tse/inference/tse_bsrnn_spk_realt_baseline_infer.yaml \
        --checkpoint tasks/tse/pretrained_models/spk_emb_100/avg_model.pt \
        --tse-manifest data/tse/infer.jsonl \
        --output_dir results/tse

The TSE JSONL manifest uses one object per line with ``uid`` and ``mixture`` plus
the model-declared cue roles (for example ``speaker_waveform``). Relative paths
are resolved against the manifest directory; inline cues live under
``attributes.cues``.

Sample rate:
  - SE: if ``data.sample_rate`` is set (or ``--sr``), inputs are resampled to that rate (fixed-SR), regardless of
    whether ``forward`` accepts ``fs``.
  - SE: if neither is set and the model's ``forward`` accepts ``fs``, keep each file's native SR and pass ``fs``.
  - SE: if neither is set and the model has no ``fs``, resample to 16000 Hz.
  - AEC: runs the AEC input/deployment frontend before full-file STFT + one model forward.
    ``--aec_stream_align`` / ``--aec_no_stream_align`` only override the resolved
    ``delay_alignment.enabled``; all other frontend settings keep their resolved values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import inspect
import logging
import math

from core.data.constants import normalize_data_mode
from tasks.registry import build_inference_adapter, get_task
from core.inference import (
    ArtifactBinding,
    FileInferenceItem,
    InferenceReport,
    InferenceSpec,
    ReusePolicy,
    bind_artifacts,
    run_inference_report,
)
from core.utils.inference_artifacts import (
    build_input_inventory,
)
from core.utils.python_version import check_python_version
from evaluate.checkpoint_loading import CheckpointFamily, load_checkpoint
from evaluate.inference_components import (
    build_artifact_fingerprint,
    configured_checkpoint_files,
    inference_source_files,
    load_effective_config,
    verify_configured_checkpoint_fingerprints,
)
from pathlib import Path, PureWindowsPath
from tools.shared.scp_io import parse_scp_fields

import torch
import torchaudio

from tasks.aec.inference.aec_frontend import (
    AECFrontendConfig,
    resolve_aec_frontend_config,
)
from tasks.aec.inference.device_policy import (
    validate_cpu_long_audio_seconds as _validate_cpu_long_audio_seconds,
)
from core.stft import build_stft_istft
from evaluate.aec_pairing import resolve_farend_path

# 根 CLI 保留 SE 旧符号作为兼容别名；具体任务输入处理由 descriptor factory 构造。
from tasks.se.inference.adapter import SEInferenceAdapter as _SEInferenceAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def _artifact_binding_from_fingerprint(fingerprint: dict) -> ArtifactBinding:
    """将入口层已计算的 source identity 原样转入 runner。"""
    return ArtifactBinding.from_fingerprint(fingerprint)


def parse_args():
    p = argparse.ArgumentParser(description="use_train inference (SE + AEC + TSE)")
    p.add_argument("-C", "--config", required=True)
    p.add_argument(
        "--expected-effective-config-sha256",
        default=None,
        help="可选父进程 effective merged config SHA-256；模型构造前严格校验。",
    )
    p.add_argument("--checkpoint", required=False, help="训练 checkpoint 路径；仅当 waveform-native 模型把权重写在 model 配置的 artifact 路径时（PASE/StuPASE）可省略，模型仍设 meta.allow_no_stft=true")
    # SE inputs
    p.add_argument("--input_dir", default=None, help="SE: noisy audio directory")
    p.add_argument("--input_scp", default=None, help="SE: noisy scp file")
    p.add_argument("--scp_root_dir", default=None, help="require manifest/SCP paths to be under this dir (path traversal guard)")
    # TSE inputs
    p.add_argument(
        "--tse-manifest",
        "--tse_manifest",
        dest="tse_manifest",
        default=None,
        help="TSE: strict JSONL manifest with uid, mixture, and canonical cue roles",
    )
    p.add_argument(
        "--tse-apply-vad",
        "--tse_apply_vad",
        dest="tse_apply_vad",
        action="store_true",
        help="TSE: apply the optional silero-VAD enrollment preprocessing",
    )
    # AEC inputs
    p.add_argument("--input_mic_dir", default=None, help="AEC: mic audio directory")
    p.add_argument("--input_far_dir", default=None, help="AEC: far-end audio directory")
    p.add_argument("--input_mic_scp", default=None, help="AEC: mic scp file")
    p.add_argument("--input_far_scp", default=None, help="AEC: far-end scp file")
    p.add_argument(
        "--aec_flat_dir",
        default=None,
        help="AEC: single directory with Microsoft-style pairs (*_mic.wav + *_lpb.wav); "
        "cannot be combined with other AEC input options.",
    )
    # output & runtime
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--output_relpath_scp",
        default=None,
        help="可选 UID 到 output_dir 内相对 .wav 文件路径的两列映射文件；未映射 UID 使用默认平铺路径。",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--cpu-long-audio-seconds",
        type=float,
        default=0.0,
        help="AEC: mic 原始时长严格超过该阈值时切换到 CPU；0=关闭。",
    )
    p.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Fixed model sample rate. SE: overrides data.sample_rate. "
        "AEC: may only restate the exact configured data.sample_rate. "
        "TSE: must match model.args.sr/data.sample_rate.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm file progress bar")
    p.add_argument(
        "--chunk_seconds",
        type=float,
        default=0.0,
        help="Waveform-native 模型的长音频分块推理（>0 启用，如 8.0）：seg=chunk_seconds, "
        "hop=seg/2 中点拼接，等价官方 unipase inference_long（默认 0=整段推理）",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse only outputs verified by a matching strict run_manifest.json; "
        "missing, stale, or invalid manifests cause re-inference.",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许部分文件推理失败仍正常退出（默认 fail closed：有失败即写出 "
        "run_manifest.json 后以非零码退出）。",
    )
    p.add_argument(
        "--aec_stream_align",
        action="store_true",
        help="AEC: force-enable canonical delay_alignment.enabled. "
        "Mutually exclusive with --aec_no_stream_align.",
    )
    p.add_argument(
        "--aec_no_stream_align",
        action="store_true",
        help="AEC: force-disable canonical delay_alignment.enabled. "
        "Default (neither flag): keep the resolved canonical frontend value.",
    )
    p.add_argument(
        "--aec-review-multichannel",
        action="store_true",
        help="AEC only: write review audio as [mic, enhanced] when the consumed "
        "reference is exactly zero, otherwise [mic, reference, enhanced].",
    )
    return p.parse_args()


def _load_model(
    cfg,
    ckpt_path,
    device,
    stft_cfg=None,
    *,
    expected_checkpoint_fingerprint=None,
):
    """Load model and apply checkpoint + EMA through the strict safe loader.

    For SE models (spectrum-in/out), pass stft_cfg so n_fft/hop_length/win_length
    match config. GAN-style configs build only the generator. Checkpoint key/shape
    mismatches always fail closed; inference never retries unsafe or partial loading.
    """
    from core.models.build import build_model
    model_type = cfg.get("model", {}).get("type")
    if model_type is None and isinstance(cfg.get("model"), dict):
        model_type = (cfg["model"].get("generator") or {}).get("type")
    model_type = str(model_type) if model_type else None
    from tasks.aec.models import AEC_MODEL_TYPES, register_aec_models
    if model_type in AEC_MODEL_TYPES:
        register_aec_models(only=model_type)
    else:
        from tasks.se.models import register_se_models

        register_se_models(only=model_type)
    ckpt_path = Path(ckpt_path).expanduser()
    model_cfg = deepcopy(cfg.get("model", {}))
    # 支持时，推理默认使用固定 positional embedding。
    _inf_pe_types = {"FastEnhancer"}
    _mcfg = model_cfg.get("generator", model_cfg) if isinstance(model_cfg, dict) else {}
    if isinstance(_mcfg, dict) and _mcfg.get("type") in _inf_pe_types:
        _args = _mcfg.setdefault("args", {})
        if "positional_embedding" not in _args:
            _args["positional_embedding"] = "fixed"
    # GAN-style config: build only generator for inference
    if isinstance(model_cfg, dict) and "generator" in model_cfg:
        model = build_model(model_cfg["generator"], stft_cfg=stft_cfg).to(device).eval()
        is_gan = True
    else:
        model = build_model(model_cfg, stft_cfg=stft_cfg).to(device).eval()
        is_gan = False
    family = (
        CheckpointFamily.GAN_GENERATOR
        if is_gan
        else CheckpointFamily.SINGLE_MODEL
    )
    _, ckpt = load_checkpoint(
        model,
        ckpt_path,
        family=family,
        map_location="cpu",
        purpose="inference model checkpoint (safe mode)",
        expected_fingerprint=expected_checkpoint_fingerprint,
    )
    logger.info("Loaded %s checkpoint: %s", family.value, ckpt_path)
    if "ema" in ckpt:
        logger.info("Applied EMA weights")
    return model


def _validate_uid(uid: str, *, source: str) -> None:
    """UID 会直接用于输出文件名，拒绝空值、路径分隔符与 ``.``/``..``。"""
    if (
        not isinstance(uid, str)
        or not uid
        or uid != uid.strip()
        or any(char.isspace() for char in uid)
        or uid in {".", ".."}
        or "\x00" in uid
        or "/" in uid
        or "\\" in uid
        or Path(uid).name != uid
    ):
        raise ValueError(f"{source} 包含非法 uid（不得为空或包含路径分隔符）: {uid!r}")


def _ensure_unique_uids(files, *, source: str):
    seen = set()
    duplicates = set()
    for uid, _ in files:
        _validate_uid(uid, source=source)
        if uid in seen:
            duplicates.add(uid)
        else:
            seen.add(uid)
    dup = sorted(duplicates)
    if dup:
        raise ValueError(
            f"{source} 产生重复 uid（会导致输出互相覆盖）: {dup[:5]}"
            f"{' ...' if len(dup) > 5 else ''}（共 {len(dup)} 个）"
        )
    return files


def _load_output_relpath_scp(path: str | Path, out_dir: Path) -> dict[str, Path]:
    """读取 UID 到 output_dir 内相对 ``.wav`` 文件路径的严格映射。"""
    scp_path = Path(path).expanduser().resolve(strict=True)
    output_root = Path(out_dir).resolve()
    mapping: dict[str, Path] = {}
    targets: dict[Path, str] = {}
    with scp_path.open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                # 必须先解析 shell quoting，再校验解析后的路径安全性。
                parts = parse_scp_fields(line, scp_path, lineno, allowed_columns=(2,))
            except ValueError as exc:
                raise ValueError(f"输出路径映射格式错误 {exc}") from exc
            uid, relative_text = parts
            _validate_uid(uid, source=f"输出路径映射 {scp_path}:{lineno}")
            if not relative_text or not relative_text.strip():
                raise ValueError(
                    f"输出路径映射 {scp_path}:{lineno} 的 relative path 不能为空"
                )
            if (
                PureWindowsPath(relative_text).is_absolute()
                or PureWindowsPath(relative_text).drive
                or Path(relative_text).is_absolute()
                or "\\" in relative_text
                or relative_text in {".", ".."}
                or any(part in {"", ".", ".."} for part in Path(relative_text).parts)
                or relative_text.endswith(("/", "\\"))
                or Path(relative_text).suffix.lower() != ".wav"
            ):
                raise ValueError(
                    f"输出路径映射 {scp_path}:{lineno} 必须是 output_dir 内的相对 .wav 文件路径: {relative_text!r}"
                )
            target = (output_root / Path(relative_text)).resolve()
            try:
                target.relative_to(output_root)
            except ValueError as exc:
                raise ValueError(
                    f"输出路径映射 {scp_path}:{lineno} 路径穿越 output_dir: {relative_text!r}"
                ) from exc
            if uid in mapping:
                raise ValueError(f"输出路径映射存在重复 uid: {uid!r}")
            if target in targets:
                raise ValueError(
                    f"输出路径映射存在重复目标: {relative_text!r} 与 {targets[target]!r}"
                )
            mapping[uid] = target
            targets[target] = uid
    return mapping


def _build_output_paths(
    out_dir: Path, expected_uids: list[str], mapped_paths: dict[str, Path]
) -> dict[str, Path]:
    """为每个 UID 生成唯一输出路径，并拒绝未知 UID 或目标冲突。"""
    unknown = sorted(set(mapped_paths) - set(expected_uids))
    if unknown:
        raise ValueError(f"输出路径映射包含未知 uid: {unknown[:5]}")
    output_paths: dict[str, Path] = {}
    targets: dict[Path, str] = {}
    for uid in expected_uids:
        target = mapped_paths.get(uid, out_dir / f"{uid}.wav").resolve()
        try:
            target.relative_to(out_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"输出路径超出 output_dir: {target}") from exc
        owner = targets.get(target)
        if owner is not None:
            raise ValueError(f"输出目标重复: uid={uid!r} 与 uid={owner!r} 共用 {target}")
        output_paths[uid] = target
        targets[target] = uid
    return output_paths


def _collect_files(input_dir=None, input_scp=None, scp_root_dir=None):
    """严格收集 ``(uid, path)``：坏行、越界路径、缺失文件与重复 UID 均报错。

    uid 规则：SCP 模式取每行第一个 token（保持不变）；SCP 中的相对音频路径
    按清单文件所在目录解析；目录模式下，平铺文件 uid==stem，嵌套子目录用相对路径拼接
    （如 ``a/x.wav`` → ``a_x``），避免不同子目录同 stem 文件在 out_dir 中互相覆盖
    （与 ``evaluate/eval_pipeline_common.py`` 口径一致）。
    """
    if input_dir is not None and input_scp is not None:
        raise ValueError("不能同时传入输入目录和 SCP；请只保留其中一个")

    files = []
    scp_root = Path(scp_root_dir).resolve() if scp_root_dir else None
    if input_scp:
        scp_path = Path(input_scp).resolve(strict=True)
        with scp_path.open(encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    parts = parse_scp_fields(
                        line, scp_path, lineno, allowed_columns=(2, 4)
                    )
                except ValueError as exc:
                    raise ValueError(f"SCP 格式错误 {exc}") from exc
                uid = parts[0]
                _validate_uid(uid, source=f"SCP {scp_path}:{lineno}")
                if len(parts) == 4:
                    try:
                        sample_rate, frames = int(parts[1]), int(parts[2])
                    except ValueError as exc:
                        raise ValueError(
                            f"SCP 格式错误 {scp_path}:{lineno}: sample_rate/frames 必须为整数"
                        ) from exc
                    if sample_rate <= 0 or frames < 0:
                        raise ValueError(
                            f"SCP 格式错误 {scp_path}:{lineno}: sample_rate 必须 > 0 且 frames >= 0"
                        )
                raw_path = Path(parts[-1]).expanduser()
                if not raw_path.is_absolute():
                    raw_path = scp_path.parent / raw_path
                path = raw_path.resolve(strict=False)
                if scp_root is not None:
                    try:
                        path.relative_to(scp_root)
                    except ValueError as exc:
                        raise ValueError(
                            f"SCP path 超出 --scp_root_dir {scp_root}: {path} "
                            f"({scp_path}:{lineno})"
                        ) from exc
                if not path.is_file():
                    raise FileNotFoundError(f"SCP path 不存在或不是文件: {path} ({scp_path}:{lineno})")
                files.append((uid, str(path)))
    elif input_dir:
        root = Path(input_dir).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"输入目录不是目录: {root}")
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
                rel = p.relative_to(root).with_suffix("")
                uid = rel.name if len(rel.parts) == 1 else "_".join(rel.parts)
                files.append((uid, str(p.resolve())))
    return _ensure_unique_uids(files, source="输入目录/清单")


def _collect_aec_flat_ms_challenge(flat_dir: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Pair mic + loopback (farend) from one folder (ICASSP AEC blind / Microsoft AEC-Challenge layout).

    Collects only stems ending with ``_mic`` / ``_lpb`` (before extension), e.g.
    ``uuid_doubletalk_mic.wav`` ↔ ``uuid_doubletalk_lpb.wav``. Skips ``*_enrl.*`` and other roles.
    """
    root = Path(flat_dir).resolve()
    mic_files: list[tuple[str, str]] = []
    far_files: list[tuple[str, str]] = []
    if not root.is_dir():
        return mic_files, far_files
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _AUDIO_EXTS:
            continue
        name_lower = p.name.lower()
        if name_lower.endswith("_mic.wav") or name_lower.endswith("_mic.flac"):
            mic_files.append((p.stem, str(p)))
        elif name_lower.endswith("_lpb.wav") or name_lower.endswith("_lpb.flac"):
            far_files.append((p.stem, str(p)))
    return mic_files, far_files


def _resolve_inference_task(cfg) -> str:
    """按 ``data.mode`` 路由，并拒绝任务与模型类型不一致的配置。"""
    from tasks.aec.models import AEC_MODEL_TYPES
    from tasks.ss.models import SS_MODEL_TYPES
    from tasks.ssl.models import SSL_MODEL_TYPES
    from tasks.tse.models import TSE_MODEL_TYPES

    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("inference config requires a mapping data section with data.mode")
    task = normalize_data_mode(data_cfg.get("mode")).task

    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("inference config requires a mapping model section")
    model_type = model_cfg.get("type")
    if model_type is None and isinstance(model_cfg.get("generator"), dict):
        model_type = model_cfg["generator"].get("type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("inference config requires a non-empty model.type")

    if task == "aec" and model_type not in AEC_MODEL_TYPES:
        raise ValueError(
            f"data.mode routes to AEC, but model.type={model_type!r} is not an AEC model"
        )
    if task == "tse" and model_type not in TSE_MODEL_TYPES:
        raise ValueError(
            f"data.mode routes to TSE, but model.type={model_type!r} is not a TSE model"
        )
    if task == "ss" and model_type not in SS_MODEL_TYPES:
        raise ValueError(
            f"data.mode routes to SS, but model.type={model_type!r} is not an SS model"
        )
    if task == "ssl" and model_type not in SSL_MODEL_TYPES:
        raise ValueError(
            f"data.mode routes to SSL, but model.type={model_type!r} is not an SSL model"
        )
    if task == "se" and model_type in AEC_MODEL_TYPES | TSE_MODEL_TYPES | SS_MODEL_TYPES | SSL_MODEL_TYPES:
        raise ValueError(
            f"data.mode routes to SE, but model.type={model_type!r} belongs to another task"
        )
    return task


def _se_model_accepts_fs(model) -> bool:
    return "fs" in inspect.signature(model.forward).parameters


def _resolve_aec_sample_rate(configured_sr, cli_sr=None) -> int:
    """AEC 采样率只允许来自配置；``--sr`` 仅可重复声明同一值。"""
    if (
        isinstance(configured_sr, bool)
        or not isinstance(configured_sr, int)
        or configured_sr <= 0
    ):
        raise ValueError(
            "AEC inference requires data.sample_rate to be a positive integer"
        )
    if cli_sr is None:
        return configured_sr
    if isinstance(cli_sr, bool) or not isinstance(cli_sr, int) or cli_sr <= 0:
        raise ValueError("AEC --sr must be a positive integer")
    if cli_sr != configured_sr:
        raise ValueError(
            "AEC --sr must match configured data.sample_rate exactly: "
            f"--sr={cli_sr}, data.sample_rate={configured_sr}"
        )
    return configured_sr


def _resolve_se_sample_rate(configured_sr, cli_sr=None) -> int | None:
    """解析 SE 固定采样率；未配置时保留 native-SR 分支。"""
    sample_rate = cli_sr if cli_sr is not None else configured_sr
    if sample_rate is None:
        return None
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        source = "--sr" if cli_sr is not None else "data.sample_rate"
        raise ValueError(f"SE {source} must be a positive integer")
    return sample_rate


@torch.inference_mode()
def _chunked_waveform_forward(model, wav, fs, chunk_seconds, multi_sr):
    """长音频分块推理。

    seg = chunk_seconds，hop = seg/2；相邻块在重叠区的全局中点建立 seam，
    每个输入 sample 只归属一个块；无重叠时 seam 即相邻边界。输出采样率不同
    时按 ``sr_out / fs`` 将全局 seam 映射到各块输出坐标。模型支持
    ``normalize_output`` 参数时，分块时关闭内部峰值归一，拼接后统一归一。
    ``chunk_seconds`` 必须有限且为正，并在当前采样率下至少量化为 1 sample。
    """
    import inspect

    chunk_seconds = float(chunk_seconds)
    if not math.isfinite(chunk_seconds):
        raise ValueError("chunk_seconds must be finite")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    chunk_samples = chunk_seconds * fs
    if not math.isfinite(chunk_samples):
        raise ValueError("chunk_seconds * fs must be finite")
    seg = int(chunk_samples)
    if seg < 1:
        raise ValueError("chunk_seconds is too small to produce at least 1 sample")

    norm_supported = "normalize_output" in set(inspect.signature(model.forward).parameters)
    n = wav.shape[-1]
    hop = max(1, seg // 2)
    sr_out = int(getattr(model, "sr_out", None) or fs)

    def map_sample(position: int) -> int:
        return position * sr_out // fs

    expected_output_len = map_sample(n)
    if expected_output_len < 1:
        raise ValueError(
            f"chunked waveform output would contain 0 samples "
            f"(input_samples={n}, fs={fs}, sr_out={sr_out})"
        )

    spans = []
    start = 0
    while start < n:
        end = min(start + seg, n)
        spans.append((start, end))
        if end == n:
            break
        start += hop

    outs = []
    for s, e in spans:
        patch = wav[:, s:e]
        lengths = torch.tensor([e - s], device=wav.device)
        kw = {"fs": fs} if multi_sr else {}
        if norm_supported:
            kw["normalize_output"] = False
        outs.append(model(patch, lengths, **kw))

    # 在输出域分配分块边界，处理 floor 相位差时保持 slice 长度。
    global_cuts = [0]
    global_cuts.extend(
        (left[1] + right[0]) // 2 for left, right in zip(spans, spans[1:])
    )
    global_cuts.append(n)
    target_cuts = [map_sample(cut) for cut in global_cuts]

    parts = []
    for i, ((s, _), out) in enumerate(zip(spans, outs)):
        required = target_cuts[i + 1] - target_cuts[i]
        out_len = out.shape[-1]
        if out_len < required:
            raise ValueError(
                f"chunk output too short for target slice: block={i}, "
                f"available={out_len}, required={required}, fs={fs}, sr_out={sr_out}"
            )
        ideal_left = (global_cuts[i] - s) * sr_out // fs
        local_left = min(max(ideal_left, 0), out_len - required)
        parts.append(out[:, local_left : local_left + required])
    y = torch.cat(parts, dim=-1)
    if y.shape[-1] != expected_output_len:
        raise ValueError(
            f"chunked waveform output length mismatch: got={y.shape[-1]}, "
            f"expected={expected_output_len}, fs={fs}, sr_out={sr_out}"
        )

    if norm_supported:
        in_peak = wav.abs().amax(dim=-1, keepdim=True)
        y = y / (y.abs().amax(dim=-1, keepdim=True) + 1e-8) * in_peak
    return y


def _resolve_aec_frontend_with_cli(
    cfg: dict,
    *,
    force_enable: bool = False,
    force_disable: bool = False,
) -> AECFrontendConfig:
    """复用 canonical resolver，并仅允许 CLI 覆盖对齐开关。"""
    if force_enable and force_disable:
        raise ValueError(
            "--aec_stream_align and --aec_no_stream_align are mutually exclusive"
        )
    frontend_cfg = resolve_aec_frontend_config(cfg)
    if not force_enable and not force_disable:
        return frontend_cfg

    resolved = frontend_cfg.to_dict()
    resolved["delay_alignment"]["enabled"] = bool(force_enable)
    return AECFrontendConfig.from_dict(resolved)


def _build_aec_input_inventory(mic_files, far_files):
    """按实际配对结果绑定 AEC mic/far-end 内容；缺配对也进入 inventory。"""
    return build_input_inventory(
        (
            uid,
            {
                "mic": mic_path,
                "farend": resolve_farend_path(mic_path, far_files, mic_uid=uid),
            },
        )
        for uid, mic_path in mic_files
    )


def main():
    check_python_version()
    args = parse_args()
    # 公共 runner 负责整个输出目录生命周期和锁；根 CLI 不得先持有同一把锁，
    # 否则主路径会在真正调用 runner 时触发同进程重入。
    return _main_locked(args, Path(args.output_dir).expanduser().resolve())


def _apply_meta_tf32_policy(cfg) -> None:
    """显式声明 ``meta.allow_tf32`` 时对齐推理侧 TF32 开关。

    训练侧由 ``BaseTrainer`` 应用同一开关；推理侧此前静默忽略该字段，导致
    GPU 推理（含评估与 badcase）与配置声明不一致。未声明的配置保持 PyTorch
    默认，避免推理单方面改变全局后端精度。
    """
    meta = cfg.get("meta") or {}
    if "allow_tf32" not in meta:
        return
    allow_tf32 = bool(meta["allow_tf32"])
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def _tse_root_artifact_binding(
    snapshot,
    *,
    requested_device: str,
    effective_device: str,
    apply_vad: bool,
    effective_config_identity,
):
    """把 TSE loader 冻结的身份转成公共 runner 的 source binding。"""

    from tasks.tse.inference import artifacts as tse_artifacts

    identities = snapshot.source_file_identities()
    if "checkpoint" not in identities:
        raise ValueError("TSE loader snapshot has no model checkpoint identity")
    # checkpoint 内嵌配置时没有独立 config 文件；保留一个同身份占位，令
    # InferenceSpec.config 仍有稳定的 source path，同时不重新 fingerprint。
    identities.setdefault("config", dict(identities["checkpoint"]))
    files = {name: identity["path"] for name, identity in identities.items()}
    config_identity = (
        dict(effective_config_identity)
        if isinstance(effective_config_identity, Mapping)
        else effective_config_identity
    )
    return bind_artifacts(
        files,
        precomputed=identities,
        parameters={
            "mode": "tse-root",
            "requested_device": requested_device,
            "effective_device": effective_device,
            "apply_vad": bool(apply_vad),
            "effective_config_identity": config_identity,
            "implementation": tse_artifacts.implementation_fingerprint(),
            "tse_artifact_snapshot": snapshot.meta,
        },
    )


def _run_tse_root_inference(
    args,
    out_dir: Path,
    *,
    effective_identity,
):
    """通过公共 ``InferenceSpec`` 执行 TSE 根文件推理并发布 sidecar。"""

    from tasks.tse.inference import (
        TSEExtractor,
        TSEInferenceArtifactSnapshot,
        publish_sidecars_from_report,
        read_tse_root_manifest,
    )

    manifest_path = getattr(args, "tse_manifest", None)
    if not manifest_path:
        raise ValueError("TSE root inference requires --tse-manifest")
    if not getattr(args, "checkpoint", None):
        raise ValueError("TSE root inference requires --checkpoint (file or model directory)")

    invalid_options = []
    for name in (
        "input_dir",
        "input_scp",
        "input_mic_dir",
        "input_far_dir",
        "input_mic_scp",
        "input_far_scp",
        "aec_flat_dir",
    ):
        if getattr(args, name, None):
            invalid_options.append(f"--{name}")
    if invalid_options:
        raise ValueError(
            "TSE root inference only accepts --tse-manifest; unsupported input options: "
            + ", ".join(invalid_options)
        )
    if float(getattr(args, "chunk_seconds", 0.0) or 0.0) > 0:
        raise ValueError("--chunk_seconds is only supported by waveform-native SE inference")
    if float(getattr(args, "cpu_long_audio_seconds", 0.0) or 0.0) > 0:
        raise ValueError("--cpu-long-audio-seconds is only supported by AEC inference")
    if any(
        bool(getattr(args, name, False))
        for name in (
            "aec_stream_align",
            "aec_no_stream_align",
            "aec_review_multichannel",
        )
    ):
        raise ValueError("AEC-only options cannot be used for TSE inference")

    requested_device = str(getattr(args, "device", "cuda:0"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "请求的设备 %s 不可用（torch.cuda.is_available() 为 False），本次 TSE 推理回退到 CPU",
            requested_device,
        )
    effective_device = requested_device if torch.cuda.is_available() else "cpu"
    apply_vad = bool(getattr(args, "tse_apply_vad", False))
    extractor = TSEExtractor(
        args.checkpoint,
        config_path=args.config,
        device=requested_device,
        sample_rate=getattr(args, "sr", None),
        apply_vad=apply_vad,
    )
    snapshot = TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor,
        {
            "requested_device": requested_device,
            "effective_device": str(extractor.device),
            "apply_vad": apply_vad,
            "effective_config_identity": effective_identity,
        },
    )
    snapshot.verify_global_identity()
    adapter = build_inference_adapter("tse", extractor)
    entries = read_tse_root_manifest(
        manifest_path,
        root_dir=getattr(args, "scp_root_dir", None),
    )
    mapped_output_paths = (
        _load_output_relpath_scp(args.output_relpath_scp, out_dir)
        if getattr(args, "output_relpath_scp", None) is not None
        else {}
    )
    output_paths = _build_output_paths(
        out_dir, [entry.uid for entry in entries], mapped_output_paths
    )
    items: list[FileInferenceItem] = []
    expected_sidecars: dict[str, dict] = {}
    for entry in entries:
        item = FileInferenceItem(
            uid=entry.uid,
            inputs=entry.inputs,
            output=output_paths[entry.uid],
            attributes=entry.attributes,
        )
        cue_files, declaration = adapter.prepare_item(item)
        expected_sidecars[entry.uid] = snapshot.bind_inputs(
            {"mixture": Path(item.inputs["mixture"]), **cue_files},
            cues=declaration,
        )
        items.append(item)

    artifact_fingerprint = _tse_root_artifact_binding(
        snapshot,
        requested_device=requested_device,
        effective_device=str(extractor.device),
        apply_vad=apply_vad,
        effective_config_identity=effective_identity,
    )
    source_identities = snapshot.source_file_identities()
    model_identity = source_identities["checkpoint"]
    config_identity = source_identities.get("config")
    spec = InferenceSpec(
        config=config_identity["path"] if config_identity else model_identity["path"],
        checkpoint=model_identity["path"],
        output_dir=out_dir,
        items=items,
        artifacts=_artifact_binding_from_fingerprint(artifact_fingerprint),
        reuse=ReusePolicy(enabled=bool(getattr(args, "skip_existing", False))),
        # 与 SE/AEC 根入口保持一致：先发布含 failed 结果的公共 manifest，
        # 再由 CLI 的 allow-partial 选项决定退出码。
        allow_partial=True,
        run_metadata={"task": "tse", "entry": "inference.py"},
    )
    report: InferenceReport = run_inference_report(spec, adapter)
    publish_sidecars_from_report(report, expected_sidecars, output_dir=out_dir)
    results = list(report.results)
    n_fail = sum(result.status == "failed" for result in results)
    n_ok = sum(result.status in {"ok", "skipped"} for result in results)
    logger.info("TSE done → %s (%d ok, %d failed)", out_dir, n_ok, n_fail)
    if n_fail > 0 and not getattr(args, "allow_partial", False):
        raise SystemExit(
            f"[ERROR] {n_fail} TSE file(s) failed inference (see {out_dir / 'run_manifest.json'}); "
            "use --allow-partial to tolerate partial failures."
        )
    return results


def _main_locked(args, out_dir: Path):
    cpu_long_audio_seconds = _validate_cpu_long_audio_seconds(
        getattr(args, "cpu_long_audio_seconds", 0.0)
    )
    cfg, effective_identity = load_effective_config(
        args.config,
        expected=getattr(args, "expected_effective_config_sha256", None),
    )
    _apply_meta_tf32_policy(cfg)
    task = _resolve_inference_task(cfg)
    descriptor = get_task(task)
    if task == "tse":
        return _run_tse_root_inference(
            args,
            out_dir,
            effective_identity=effective_identity,
        )
    if descriptor.file_inference_adapter_factory is None:
        raise ValueError(descriptor.root_file_inference_error)

    mapped_output_paths = (
        _load_output_relpath_scp(args.output_relpath_scp, out_dir)
        if getattr(args, "output_relpath_scp", None) is not None
        else {}
    )
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "请求的设备 %s 不可用（torch.cuda.is_available() 为 False），本次推理回退到 CPU",
            requested_device,
        )
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

    is_aec = task == "aec"
    if args.aec_review_multichannel and not is_aec:
        raise ValueError("--aec-review-multichannel is only valid for AEC inference")
    if is_aec:
        import tasks.aec.models  # noqa: F401 — register AEC models before build_model

        _mt = cfg.get("model", {}).get("type")
        tasks.aec.models.register_aec_models(only=str(_mt)) if _mt else tasks.aec.models.register_aec_models()
    stft_cfg = cfg.get("stft")
    allow_no_stft = cfg.get("meta", {}).get("allow_no_stft", False)
    is_waveform_native = allow_no_stft and not stft_cfg
    has_model_artifacts = is_waveform_native and bool(
        configured_checkpoint_files(cfg.get("model", {}))
    )
    project_root = Path(__file__).resolve().parent
    artifact_fingerprint = build_artifact_fingerprint(
        args,
        cfg,
        project_root=project_root,
        inference_entry=Path(__file__),
        is_aec=is_aec,
        is_waveform_native=is_waveform_native,
        has_model_artifacts=has_model_artifacts if is_waveform_native else None,
        effective_device=str(device),
        source_files=inference_source_files(project_root, is_aec=is_aec),
        effective_identity=effective_identity,
    )
    if is_waveform_native and has_model_artifacts:
        from tasks.se.models import build_model
        _mcfg = cfg.get("model", {})
        _waveform_checkpoint_files = configured_checkpoint_files(_mcfg)
        _gen_cfg = _mcfg["generator"] if (isinstance(_mcfg, dict) and "generator" in _mcfg) else _mcfg
        model = build_model(deepcopy(_gen_cfg), stft_cfg=None).to(device).eval()
        verify_configured_checkpoint_fingerprints(
            _waveform_checkpoint_files,
            artifact_fingerprint,
        )
        logger.info("Loaded waveform-native model (weights via from_pretrained artifacts; --checkpoint ignored).")
    else:
        if not args.checkpoint:
            raise ValueError(
                "非 waveform-native 模型需要 --checkpoint（waveform-native artifact 模型设 "
                "meta.allow_no_stft=true 可免，权重由模型配置内的 artifact 注入；"
                "checkpoint 训练的 waveform-native 模型仍需 --checkpoint）"
            )
        model = _load_model(
            cfg,
            args.checkpoint,
            device,
            stft_cfg=stft_cfg,
            expected_checkpoint_fingerprint=artifact_fingerprint["files"]["checkpoint"],
        )

    data_cfg = cfg.get("data") or {}
    cfg_sr = data_cfg.get("sample_rate", None)
    override_sr = args.sr
    show_progress = not args.no_progress

    if is_aec:
        if args.aec_flat_dir:
            if (
                args.input_mic_dir
                or args.input_far_dir
                or args.input_mic_scp
                or args.input_far_scp
            ):
                raise ValueError(
                    "--aec_flat_dir cannot be combined with --input_mic_dir / "
                    "--input_far_dir / --input_mic_scp / --input_far_scp"
                )
            mic_files, far_files = _collect_aec_flat_ms_challenge(args.aec_flat_dir)
            _ensure_unique_uids(mic_files, source="AEC flat mic 输入")
            _ensure_unique_uids(far_files, source="AEC flat far-end 输入")
        else:
            mic_files = _collect_files(args.input_mic_dir, args.input_mic_scp, args.scp_root_dir)
            far_files = _collect_files(args.input_far_dir, args.input_far_scp, args.scp_root_dir)
        if not mic_files:
            raise ValueError(
                "AEC mode: provide --aec_flat_dir, or --input_mic_dir / --input_mic_scp"
            )
        if not far_files:
            raise ValueError(
                "AEC mode: provide --aec_flat_dir with *_lpb.wav, or --input_far_dir / --input_far_scp"
            )

        model_fixed_sr = _resolve_aec_sample_rate(cfg_sr, override_sr)

        stft_aec = cfg.get("stft")
        if not stft_aec:
            raise ValueError(
                "AEC inference requires top-level 'stft' in config (same as training)."
            )
        mr = model_fixed_sr

        # 三态解析：共享 resolver 决定完整 frontend，CLI 只覆盖唯一对齐开关。
        frontend_cfg = _resolve_aec_frontend_with_cli(
            cfg,
            force_enable=args.aec_stream_align,
            force_disable=args.aec_no_stream_align,
        )
        align_enabled = frontend_cfg.delay_alignment.enabled
        logger.info(
            f"Resolved aec_frontend.delay_alignment.enabled={align_enabled} "
            f"(cli_stream_align={args.aec_stream_align}, "
            f"cli_no_stream_align={args.aec_no_stream_align})"
        )

        logger.info(
            f"AEC inference (batch): {len(mic_files)} mic | fixed_sr={mr} | "
            f"streaming_align={'enabled' if align_enabled else 'disabled'} | "
            f"input_routing={frontend_cfg.input_routing}"
        )
        expected_uids = [uid for uid, _ in mic_files]
        output_paths = _build_output_paths(out_dir, expected_uids, mapped_output_paths)
        input_inventory = _build_aec_input_inventory(mic_files, far_files)
        adapter = build_inference_adapter(
            task,
            model,
            device=device,
            stft_cfg=stft_aec,
            frontend_cfg=frontend_cfg,
            model_fixed_sr=mr,
            cpu_long_audio_seconds=cpu_long_audio_seconds,
            review_multichannel=args.aec_review_multichannel,
        )
    else:
        files = _collect_files(args.input_dir, args.input_scp, args.scp_root_dir)
        if not files:
            if args.input_dir:
                raise ValueError(
                    f"SE mode: --input_dir {args.input_dir} 中没有可处理的音频文件"
                    f"（支持的扩展名: {sorted(_AUDIO_EXTS)}）"
                )
            raise ValueError("SE mode: provide --input_dir or --input_scp")
        stft_cfg = cfg.get("stft")
        allow_no_stft = cfg.get("meta", {}).get("allow_no_stft", False)
        if not stft_cfg and not allow_no_stft:
            raise ValueError(
                "SE inference requires top-level 'stft' in config (n_fft, hop_length, win_length), "
                "or meta.allow_no_stft=true for waveform-native models (PASE/StuPASE)."
            )

        model_accepts_fs = _se_model_accepts_fs(model)
        fixed_from_cfg = _resolve_se_sample_rate(cfg_sr, override_sr)
        multi_sr = model_accepts_fs and fixed_from_cfg is None
        if multi_sr:
            logger.info(
                f"SE inference (multi-SR): {len(files)} files | data.sample_rate unset, native SR + fs=..."
            )
            fixed = 16000  # unused when multi_sr=True
        else:
            fixed = int(override_sr if override_sr is not None else (cfg_sr or 16000))
            src = "--sr" if override_sr is not None else ("data.sample_rate" if cfg_sr is not None else "default")
            logger.info(
                f"SE inference (fixed-SR): {len(files)} files | resample to {fixed} Hz ({src})"
            )

        expected_uids = [uid for uid, _ in files]
        output_paths = _build_output_paths(out_dir, expected_uids, mapped_output_paths)
        input_inventory = build_input_inventory(
            (uid, {"noisy": path}) for uid, path in files
        )
        adapter = build_inference_adapter(
            task,
            model,
            device=device,
            stft=stft_cfg,
            stft_module=build_stft_istft(stft_cfg) if stft_cfg else None,
            multi_sr=multi_sr,
            model_fixed_sr=fixed,
            chunk_seconds=args.chunk_seconds,
        )

    runner_items = [
        FileInferenceItem(
            uid=uid,
            inputs={
                "mic": path,
                "farend": resolve_farend_path(path, far_files, mic_uid=uid),
            } if is_aec else {"noisy": path},
            output=output_paths[uid],
        )
        for uid, path in (mic_files if is_aec else files)
    ]
    spec = InferenceSpec(
        config=args.config,
        output_dir=out_dir,
        items=runner_items,
        checkpoint=args.checkpoint,
        artifacts=_artifact_binding_from_fingerprint(artifact_fingerprint),
        reuse=ReusePolicy(enabled=args.skip_existing),
        # 先发布包含 failed 结果的 manifest，再在函数末尾保持 CLI 的
        # fail-closed 退出语义。
        allow_partial=True,
        run_metadata={"task": task},
    )
    report: InferenceReport = run_inference_report(spec, adapter)
    results = list(report.results)
    n_fail = sum(result.status == "failed" for result in results)
    n_ok = sum(result.status in {"ok", "skipped"} for result in results)
    logger.info(f"Done → {out_dir} ({n_ok} ok, {n_fail} failed)")
    if n_fail > 0 and not args.allow_partial:
        raise SystemExit(
            f"[ERROR] {n_fail} file(s) failed inference (see {out_dir / 'run_manifest.json'}); "
            f"use --allow-partial to tolerate partial failures."
        )
    return results


if __name__ == "__main__":
    main()
