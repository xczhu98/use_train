#!/usr/bin/env python3
"""REAL-TSE 评测流水线 — 批量 TSE 推理 + 比赛四指标.

输入 REAL-TSE-Challenge 的 ``*_meta.csv``（必含 ``mixture_utterance`` /
``enrolment_speakers_utterance`` 两列；可选 ``language`` /
``ground_truth_transcript`` 列用于 TER，``speaker`` 列用于 timing 官方口径，
``target_utterance`` 列用于 SI-SDR/PESQ/STOI 有参考指标），逐行调
:class:`TSEExtractor` 推理写 wav，再按 ``--metrics`` 单选/组合计算指标。

四指标与 REAL-TSE-Challenge 官方实现（``utils/`` 下各评测脚本）逐项对齐：

- **spksim**：spk cosine similarity —— 对齐官方 ``utils/spk_similarity_eval.py``：
  ``wespeakerruntime.Speaker(lang=...)``（en=voxceleb_resnet34_LM /
  chs=cnceleb_resnet34_LM，ONNX），输出 enroll↔tse（官方 ``tse_enrol``）与
  enroll↔mixture（官方 ``mixture_enrol``）两组余弦均值。语种按 meta
  ``language`` 列（zh/ch→chs、en→en），缺失时按 ``--dataset_lang_overrides``
  （默认 ``AISHELL-4:chs,AliMeeting:chs``，与官方一致）再退回
  ``--wespeaker_lang``。模型文件约定存放于
  ``third_party_scripts/wespeaker/{en,chs}/model.onnx``（不进 git），缺失时
  按官方 Hub 机制自动下载到 ``~/.wespeaker/``。``--spksim_backend ecapa``
  可退回旧 vendored ECAPA-TDNN 路径：非官方口径，仅作后备与快速自检，
  其结果不可与官方 wespeaker 数值直接比较；
- **dnsmos**：DNSMOS P.835（SIG/BAK/OVRL）+ P808 —— 对齐官方
  ``utils/dnsmos_eval.py``：``sig_bak_ovr.onnx``（非个性化 polyfit，
  INPUT_LENGTH=9.01s、hop=1s、短音频倍增补齐）与 ``model_v8.onnx``
  （P808，输入为 seg[:-160] 的 120 维 melspec：n_fft=321/hop=160/
  power_to_db(ref=max)+40)/40。模型位于 ``third_party_scripts/DNSMOS/DNSMOS``；
- **timing**：TSE timing —— 对齐官方 ``utils/tse_timing_evaluation.py``
  （+ ``vad_inference_firered.py`` / ``prepare_label_segments.py``）：提供
  ``--timing_gt_json_base_dir``（官方 GT overlap json 根目录，含
  ``<dataset>/overlap_records.json``）后走官方口径——GT label 取目标说话人
  overlap 段（相对 mixture 起点、clip 后加 collar=0.05s），hyp 为 VAD(tse 输出)
  的语音段，10ms 帧级掩码统计 TP/FP/FN 时长，汇总为 micro
  precision/recall/F1。VAD 后端默认 FireRedVAD（官方同款配置：
  speech_threshold=0.5、smooth=5、min_speech=20、max_speech=2000、
  min_silence=10、merge_silence=50、extend_speech=5、chunk_max=30000，
  输入峰值归一化到 0.9 后转 int16）；FireRedASR2S 代码/模型缺失时降级
  silero-vad（日志注明非官方口径）。未提供 GT json 时退回旧的
  mixture-VAD 近似口径（ref=VAD(mixture)，仅衡量"语音保留"，非官方口径）；
- **ter**：TER（zh=CER / en=WER）—— sherpa-onnx Zipformer transducer，
  文本归一化对齐官方 ``utils/asr_metrics.py``（hyp 走 whisper-normalizer；
  zh GT 仅 opencc t2s 简体化后逐字切分，en GT 仅去 ``.``/``…`` 与压缩空白，
  **不**额外小写化——与官方一致），错误率用 meeteval
  ``siso_word_error_rate``（缺失时退回等价本地 Levenshtein 实现）。
  **可选依赖（决策 D3）**：未安装 sherpa-onnx 时跳过 TER 并 warning，
  其余指标照常。安装方式::

      pip install sherpa-onnx whisper-normalizer opencc-python-reimplemented meeteval

  另需 Zipformer 模型文件（encoder/decoder/joiner ONNX + tokens.txt），
  经 ``--ter_model_dir``（或按语言 ``--ter_model_dir_zh/en``）注入；
  本仓库约定存放于 ``third_party_scripts/zipformer/``（模型文件不进 git，
  下载方式见该目录 README.md）::

      third_party_scripts/zipformer/
      ├── sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/   # 中文
      └── sherpa-onnx-zipformer-gigaspeech-2023-12-12/    # 英文
- **sisdr / pesq / stoi**：常规有参考指标，复用 ``core.metrics``；
  需要 meta 含 ``target_utterance`` 列（或 ``--ref_root`` 提供参考目录）。

输出：``tse_eval_results.csv``（逐行明细）+ ``tse_eval_summary.json``
（均值汇总；timing 官方口径为 micro P/R/F1）；同时按官方
``aggregate_eval_summary.py`` 的消费约定自动导出五个指标 CSV 到
``eval_metrics/<目录名>_*.csv``（TER / spk_similarity /
spk_similarity_mixture_enrol / dnsmos / TSE_TIMING，仅导出实际计算过的
指标），可直接用 ``python -m evaluate.aggregate_eval_summary --output_dir
<目录>`` 生成按 dataset/language/overall 分组的官方格式汇总报表。
因果/延迟验证用 ``python -m evaluate.latency_verification``（官方
latency_verification.py 移植版）。

数据身份：运行前按内容绑定所有 ``--meta``、显式 ``--mapping_csv``、timing
实际消费的 ``overlap_records.json``、EVAL2 ``scene_channel.csv`` 与 REAL-T release
README。该 identity 进入 inference sidecar/manifest、评测 summary 和 lifecycle
provenance；文件变化会使旧 artifact 失配，同次运行拒绝混用不同 split/release root。

路径解析：meta 中 utterance 列为绝对路径且存在则直接用；否则依次尝试
``<audio_root>/<value>[.wav]``，以及官方布局下的
``<audio_root>/{mixtures,enrolment_speakers}/<value>[.wav]``；亦可用
``--mapping_csv``（两列 utterance,path，比赛 mapping.csv 格式）显式映射。
解析到的音频必须是 ``audio_root`` 内不含符号链接的普通文件；mapping 相对路径按
``audio_root`` 解析，绝对路径也必须落在该根目录内。

Usage::

    python -m evaluate.eval_pipeline_tse \\
        --meta data/tse/REAL-TSE/REAL-T-dev/DEV/AISHELL-4_meta.csv \\
        --audio_root data/tse/REAL-TSE/REAL-T-dev \\
        --model exp/tse/dynamic/tse_bsrnn_spk/checkpoints/<run-id>/best_model.tar \\
        --output_dir exp/eval/tse/tse_bsrnn_spk \\
        --metrics spksim,dnsmos,timing,ter \\
        --timing_gt_json_base_dir data/tse/REAL-TSE/REAL-T-dev/DEV/json \\
        --ter_model_dir_zh third_party_scripts/zipformer/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2 \\
        --ter_model_dir_en third_party_scripts/zipformer/sherpa-onnx-zipformer-gigaspeech-2023-12-12
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils.inference_artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_content_fingerprint,
    fingerprint_file,
)
from tasks.tse.inference import artifacts as tse_inference_artifacts
from tasks.tse.models.speaker.config import resolve_external_speaker_checkpoint
from evaluate.artifact_lifecycle import (
    attach_run_id,
    begin_artifact_run,
    new_run_id,
    publish_artifact_commit,
)
from evaluate.common import publish_result_manifest
from core.inference import (
    FileInferenceItem,
    InferenceSpec,
    ReusePolicy,
    bind_artifacts,
    run_inference,
)
from evaluate.pipeline_runner import PipelineTaskAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_pipeline_tse")

_ALL_METRICS = ("spksim", "dnsmos", "timing", "ter", "sisdr", "pesq", "stoi")


def _make_uid(row: dict) -> str:
    """返回 dataset/mixture/enrollment canonical SHA-256 UID。"""
    return tse_inference_artifacts.make_uid(row)


def _out_name(uid: str) -> str:
    """canonical UID 本身就是安全且唯一的输出 basename。"""
    return uid


_ARTIFACT_SCHEMA = tse_inference_artifacts.ARTIFACT_SCHEMA
_TSE_IMPLEMENTATION_PATHS = tse_inference_artifacts.IMPLEMENTATION_PATHS


def _tse_implementation_fingerprint(
    project_root: str | Path = _PROJECT_ROOT,
    source_paths: tuple[str, ...] = _TSE_IMPLEMENTATION_PATHS,
) -> dict[str, Any]:
    """委托 TSE artifact 模块计算源码身份。"""
    return tse_inference_artifacts.implementation_fingerprint(project_root, source_paths)


def _artifact_meta(artifacts: dict[str, Any]) -> dict:
    """委托 TSE artifact 模块封装 metadata。"""
    return tse_inference_artifacts._artifact_meta(artifacts)


def _model_meta(
    model_path: str,
    config_path: str | None = None,
    inference_params: dict[str, Any] | None = None,
    input_paths: dict[str, Path] | None = None,
    *,
    model_checkpoint: str | Path | None = None,
    model_checkpoint_identity: Mapping[str, Any] | None = None,
    config_identity: Mapping[str, Any] | None = None,
    resolved_config_identity: Mapping[str, Any] | None = None,
    speaker_checkpoint: str | Path | None = None,
    speaker_checkpoint_identity: Mapping[str, Any] | None = None,
    implementation_project_root: str | Path = _PROJECT_ROOT,
    implementation_source_paths: tuple[str, ...] = _TSE_IMPLEMENTATION_PATHS,
    dataset_identity: Mapping[str, Any] | None = None,
) -> dict:
    """委托 TSE artifact 模块构造 metadata。"""
    snapshot = tse_inference_artifacts.TSEInferenceArtifactSnapshot._build(
        model_checkpoint=(
            model_checkpoint
            if model_checkpoint is not None
            else tse_inference_artifacts.model_checkpoint_path(model_path)
        ),
        model_checkpoint_identity=model_checkpoint_identity,
        config_path=config_path,
        config_identity=config_identity,
        resolved_config_identity=resolved_config_identity,
        speaker_checkpoint=speaker_checkpoint,
        speaker_checkpoint_identity=speaker_checkpoint_identity,
        inference_params=inference_params or {},
        dataset_identity=dataset_identity,
        mode="compatibility",
        configured_speaker_encoder_pretrained=None,
        implementation_project_root=implementation_project_root,
        implementation_source_paths=implementation_source_paths,
    )
    return snapshot.bind_inputs(input_paths or {})


def _build_dataset_identity(args: argparse.Namespace, rows: list[dict]) -> dict[str, Any]:
    """绑定本次评测实际消费的 release、split 与 annotation 内容身份。"""
    files: dict[str, Path] = {}
    meta_paths: list[Path] = []
    for index, value in enumerate(args.meta):
        snapshot = tse_inference_artifacts.capture_immutable_file(
            value, label=f"dataset meta[{index}]"
        )
        meta_paths.append(snapshot.path)
    mapping_csv = getattr(args, "mapping_csv", None)
    if mapping_csv:
        mapping = tse_inference_artifacts.capture_immutable_file(
            mapping_csv, label="dataset mapping"
        )
        files["mapping"] = mapping.path

    split_names = {
        path.parent.name
        for path in meta_paths
        if path.parent.name in {"DEV", "EVAL1", "EVAL2"}
    }
    if len(split_names) > 1:
        raise ValueError(f"REAL-T evaluation cannot mix splits: {sorted(split_names)}")
    split = next(iter(split_names), None)
    official_roots = {
        path.parent.parent.resolve(strict=True)
        for path in meta_paths
        if path.parent.name in {"DEV", "EVAL1", "EVAL2"}
    }
    if len(official_roots) > 1:
        raise ValueError(
            "REAL-T meta files from different release roots cannot be mixed: "
            + ", ".join(str(path) for path in sorted(official_roots))
        )
    official_root = next(iter(official_roots), None)
    for path in sorted(meta_paths):
        dataset = path.name.removesuffix("_meta.csv")
        role = f"meta:{dataset}"
        if role in files:
            raise ValueError(f"duplicate dataset meta role: {role}")
        files[role] = path
    audio_root_value = getattr(args, "audio_root", None)
    audio_root = (
        Path(audio_root_value).expanduser().resolve(strict=True)
        if audio_root_value
        else None
    )
    if official_root is not None and audio_root is not None and official_root != audio_root:
        raise ValueError(
            f"REAL-T meta root {official_root} does not match --audio_root {audio_root}"
        )

    timing_root_value = getattr(args, "timing_gt_json_base_dir", None)
    if timing_root_value:
        timing_root = Path(timing_root_value).expanduser().resolve(strict=True)
        if official_root is not None and split is not None:
            expected_timing_root = (official_root / split / "json").resolve(strict=True)
            if timing_root != expected_timing_root:
                raise ValueError(
                    f"REAL-T timing root {timing_root} does not match current release "
                    f"root {expected_timing_root}"
                )
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            overlap = timing_root / dataset / "overlap_records.json"
            snapshot = tse_inference_artifacts.capture_immutable_file(
                overlap, label=f"timing overlap {dataset}"
            )
            files[f"timing:{dataset}:overlap_records.json"] = snapshot.path

    scene_found = False
    for parent in sorted({path.parent for path in meta_paths}):
        scene_channel = parent / "scene_channel.csv"
        if scene_channel.is_file() or scene_channel.is_symlink():
            snapshot = tse_inference_artifacts.capture_immutable_file(
                scene_channel, label="REAL-T scene_channel"
            )
            files[f"annotation:{parent.name}:scene_channel.csv"] = snapshot.path
            scene_found = True
    if split == "EVAL2" and not scene_found:
        raise FileNotFoundError("REAL-T EVAL2 requires scene_channel.csv")

    release = None
    release_date = None
    release_readme = official_root.parent / "README.md" if official_root else None
    if release_readme is not None and release_readme.is_file():
        snapshot = tse_inference_artifacts.capture_immutable_file(
            release_readme, label="REAL-T release README"
        )
        text = snapshot.content.decode("utf-8")
        if "# REAL-T:" in text:
            files["release:README.md"] = snapshot.path
            match = re.search(r"\*\*Release:\*\*\s*([^\s(]+)\s*\(([^)]+)\)", text)
            if match:
                release, release_date = match.groups()
    if official_root is not None and (release is None or release_date is None):
        raise ValueError("REAL-T release README is missing or has no parseable release identity")

    parameters = {
        "schema": "tse-evaluation-dataset-identity-v1",
        "benchmark": "REAL-T" if official_root is not None else "custom",
        "release": release,
        "release_date": release_date,
        "split": split,
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "audio_root": str(audio_root) if audio_root is not None else None,
        "official_split_root": str(official_root) if official_root is not None else None,
    }
    return build_content_fingerprint(files, parameters=parameters)


def _model_config_path_for_artifacts(
    model_path: str | Path,
    config_path: str | Path | None = None,
) -> Path | None:
    """按 TSE artifact 模块规则解析外部配置。"""
    return tse_inference_artifacts.model_config_path(model_path, config_path)


def _load_model_artifacts_for_skip(
    model_path: str | Path,
    config_path: str | Path | None = None,
) -> tuple[
    dict[str, Any] | None,
    Path | None,
    dict[str, Any] | None,
    Path,
    dict[str, Any],
    dict[str, str] | None,
]:
    """委托 TSE artifact inspection（不构造完整模型）。"""
    return tse_inference_artifacts._inspect_model_artifacts(model_path, config_path)


def _load_model_config_for_artifacts(
    model_path: str | Path,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any] | None]:
    """恢复配置、入口配置路径及其 loader 身份。"""
    config, config_file, config_identity, _, _, _ = (
        _load_model_artifacts_for_skip(model_path, config_path)
    )
    return config, config_file, config_identity


def _resolve_external_speaker_checkpoint_for_artifacts(
    model_path: str | Path,
    config_path: str | Path | None = None,
    *,
    strict: bool = False,
    loaded_config: Mapping[str, Any] | None = None,
) -> Path | None:
    """恢复并解析非 joint + ``spkemb`` 的外部 speaker checkpoint。

    ``strict=True`` 用于 ``--skip_inference``，避免在无法确认 speaker 权重身份
    时复用已有 wav；正常推理的实际配置由 TSEExtractor 自己严格校验。
    """
    if loaded_config is None:
        config, _, _ = _load_model_config_for_artifacts(model_path, config_path)
    else:
        config = loaded_config
    if config is None:
        if strict:
            raise ValueError(
                "无法恢复模型配置，不能在 --skip_inference 时验证 speaker artifact"
            )
        return None
    model_config = config.get("model")
    # 缺少 ``model.args`` 时无法确认外部 speaker encoder，记录 None。
    if not isinstance(model_config, Mapping):
        return None
    model_args = model_config.get("args")
    if not isinstance(model_args, Mapping):
        return None
    try:
        return resolve_external_speaker_checkpoint(
            model_args, repo_root=_PROJECT_ROOT
        )
    except (TypeError, ValueError):
        # 已识别 ``model.args`` 时，strict skip 模式必须原样传播路径错误。
        if strict:
            raise
        return None


def _bind_input_artifacts(
    base_meta: dict[str, Any],
    input_paths: dict[str, Path],
    *,
    cues: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """委托 TSE artifact 模块绑定输入 stat 与 cue 声明。"""
    return tse_inference_artifacts.bind_input_artifacts(base_meta, input_paths, cues=cues)


def _meta_matches(
    meta: dict | None, expected: dict, output_path: Path | None = None
) -> bool:
    """委托 TSE artifact 模块校验 sidecar。"""
    return tse_inference_artifacts.meta_matches(meta, expected, output_path)


def _resolve_ckpt_path(p: str) -> Path:
    """相对 ckpt 路径先按 CWD 判存在，再相对仓库根解析一次（缓解 CWD 依赖）。"""
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    alt = _PROJECT_ROOT / path
    return alt if alt.exists() else path


def _escape_csv_formula(v):
    """CSV 公式注入防护：=+-@ 开头的文本前加单引号。"""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
        return "'" + v
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _finite_nonnegative(value: str) -> float:
    """解析允许为零的有限 timing 秒数参数。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite number >= 0") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return parsed


def _finite_positive(value: str) -> float:
    """解析必须为正的有限 timing 秒数参数。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite number > 0") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return parsed


def _validate_timing_parameters(args: argparse.Namespace) -> None:
    """在 CLI 与测试构造的 Namespace 两条入口统一校验 timing 边界。"""
    frame_shift = getattr(args, "timing_frame_shift", 0.01)
    collar = getattr(args, "timing_collar", 0.05)
    tolerance = getattr(args, "timing_match_tolerance", 0.02)
    if isinstance(frame_shift, bool) or not isinstance(frame_shift, (int, float)):
        raise ValueError("timing_frame_shift must be a finite number > 0")
    if not math.isfinite(frame_shift) or frame_shift <= 0:
        raise ValueError("timing_frame_shift must be a finite number > 0")
    for name, value in (("timing_collar", collar), ("timing_match_tolerance", tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number >= 0")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="REAL-TSE 评测流水线（推理 + 比赛四指标）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_data = p.add_argument_group("Data")
    g_data.add_argument("--meta", action="append", required=True,
                        help="比赛 *_meta.csv 路径（可重复传入多个）")
    g_data.add_argument("--audio_root", default=None,
                        help="音频根目录：meta 中相对路径/utterance 名相对它解析")
    g_data.add_argument("--mapping_csv", default=None,
                        help="可选比赛 mapping.csv（utterance,path 两列）显式路径映射")
    g_data.add_argument("--ref_root", default=None,
                        help="可选参考（target）音频根目录，用于 sisdr/pesq/stoi")
    g_data.add_argument(
        "--cue_root", default=None,
        help="可选 cue 资源根目录（speaker embedding/spatial/visual/textual）；"
             "缺省时按 audio_root 及其 cues/ 子目录解析。模型声明的 cue role "
             "缺少真实资源时显式报错，不做静默跳过",
    )

    g_model = p.add_argument_group("Model")
    g_model.add_argument("--model", required=True,
                         help="use_train 训练 checkpoint，或比赛兼容 model_dir")
    g_model.add_argument("--config", default=None, help="显式模型配置 YAML（可选）")
    g_model.add_argument("--device", default="cuda:0")
    g_model.add_argument("--apply_vad", action="store_true",
                         help="推理时用 silero-VAD 裁 enroll 静音（未安装 silero-vad 自动跳过）")

    g_met = p.add_argument_group("Metrics")
    g_met.add_argument("--metrics", default="spksim,timing",
                       help="逗号分隔: spksim,dnsmos,timing,ter,sisdr,pesq,stoi")
    g_met.add_argument("--spksim_backend", default="wespeakerruntime",
                       choices=["wespeakerruntime", "ecapa"],
                       help="spksim 后端：wespeakerruntime（官方口径，默认）/ "
                            "ecapa（vendored ECAPA-TDNN 后备，非官方口径，"
                            "结果不可与官方数值比较，仅作快速自检）")
    g_met.add_argument("--spk_encoder_ckpt", default=None,
                       help="spksim_backend=ecapa 时的 SpeakerEncoder 预训练权重；"
                            "缺省时取模型配置 speaker_encoder.pretrained")
    g_met.add_argument("--wespeaker_model_dir", default=None,
                       help="wespeaker ONNX 根目录（含 en/chs 子目录下 model.onnx）；"
                            "缺省 third_party_scripts/wespeaker；文件缺失时按官方 "
                            "Hub 机制自动下载到 ~/.wespeaker/")
    g_met.add_argument("--wespeaker_lang", default="en", choices=["en", "chs"],
                       help="meta 无 language 列且数据集不在 overrides 时的默认语种")
    g_met.add_argument("--dataset_lang_overrides",
                       default="AISHELL-4:chs,AliMeeting:chs",
                       help="逗号分隔 dataset:lang 覆盖（与官方默认一致）")
    g_met.add_argument("--dnsmos_model_dir", default=None,
                       help="DNSMOS ONNX 目录（含 sig_bak_ovr.onnx 与 model_v8.onnx）；"
                            "缺省 third_party_scripts/DNSMOS/DNSMOS")
    g_met.add_argument("--timing_gt_json_base_dir", default=None,
                       help="官方 GT overlap json 根目录（含 <dataset>/overlap_records.json）；"
                            "提供后 timing 走官方口径（FireRedVAD + GT label）")
    g_met.add_argument("--timing_vad_backend", default="firered",
                       choices=["firered", "silero", "energy"],
                       help="timing VAD 后端：firered（官方口径，默认；依赖缺失时"
                            "自动降级 silero → energy 并日志注明）")
    g_met.add_argument("--fireredasr2s_root", default=None,
                       help="FireRedASR2S 仓库根目录（缺省 third_party_scripts/FireRedASR2S）")
    g_met.add_argument("--timing_frame_shift", type=_finite_positive, default=0.01,
                       help="timing 帧移（秒），官方默认 0.01；必须 > 0")
    g_met.add_argument("--timing_collar", type=_finite_nonnegative, default=0.05,
                       help="GT label 双侧 collar（秒），官方默认 0.05；必须 >= 0")
    g_met.add_argument("--timing_match_tolerance", type=_finite_nonnegative, default=0.02,
                       help="overlap record 起止时间匹配容差（秒），官方默认 0.02；必须 >= 0")
    g_met.add_argument("--ter_model_dir", default=None,
                       help="sherpa-onnx Zipformer transducer 目录"
                            "（含 encoder/decoder/joiner ONNX 与 tokens.txt；双语共用）")
    g_met.add_argument("--ter_model_dir_zh", default=None, help="中文 Zipformer 模型目录（覆盖 --ter_model_dir）")
    g_met.add_argument("--ter_model_dir_en", default=None, help="英文 Zipformer 模型目录（覆盖 --ter_model_dir）")
    g_met.add_argument("--default_lang", default="en", choices=["en", "zh"],
                       help="meta 无 language 列时 TER 的默认语种")

    g_run = p.add_argument_group("Run control")
    g_run.add_argument("--output_dir", required=True)
    g_run.add_argument("--skip_inference", action="store_true",
                       help="跳过推理，直接评测 output_dir 下已有 wav（断点续评）")
    g_run.add_argument(
        "--allow_partial", action="store_true",
        help="显式允许推理/指标部分失败并产出 partial summary；默认任一失败即报错",
    )
    g_run.add_argument("--max_samples", type=int, default=None,
                       help="只处理前 N 行（省略=全部；0 会因无样本而报错）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# meta 读取与路径解析
# ---------------------------------------------------------------------------
# 模型声明的 cue role → meta 列名；speaker_waveform 复用 REAL-TSE 的 enrollment 列。
_CUE_META_COLUMNS = {
    "speaker_waveform": "enrolment_speakers_utterance",
    "speaker_embedding": "speaker_embedding_cue",
    "spatial": "spatial_cue",
    "visual": "visual_cue",
    "textual": "textual_cue",
}


def _read_meta(meta_paths: list[str]) -> list[dict]:
    """读取一个或多个 meta CSV，返回行 dict 列表。

    除 REAL-TSE 必需列外，可选读取 cue 资源列（``speaker_embedding_cue`` /
    ``spatial_cue`` / ``visual_cue`` / ``textual_cue``）；模型声明了对应 role
    但该行没有资源时由 cue 解析边界显式报错。
    """
    import pandas as pd

    rows: list[dict] = []
    for mp in meta_paths:
        df = pd.read_csv(mp)
        for col in ("mixture_utterance", "enrolment_speakers_utterance"):
            if col not in df.columns:
                raise ValueError(f"{mp} 缺少必需列 {col!r}（现有列: {list(df.columns)}）")
        dataset = Path(mp).name.split("_meta.csv")[0]
        for r in df.to_dict("records"):
            row = {
                "dataset": dataset,
                "mixture_utterance": str(r["mixture_utterance"]),
                "enrolment_speakers_utterance": str(r["enrolment_speakers_utterance"]),
                "target_utterance": (
                    str(r["target_utterance"])
                    if "target_utterance" in df.columns and not pd.isna(r.get("target_utterance"))
                    else None
                ),
                "language": (
                    str(r["language"]).lower()
                    if "language" in df.columns and not pd.isna(r.get("language"))
                    else None
                ),
                # timing 官方口径需要目标说话人 ID（overlap_records.json 按 speaker 过滤）
                "speaker": (
                    str(r["speaker"])
                    if "speaker" in df.columns and not pd.isna(r.get("speaker"))
                    else None
                ),
                "ground_truth_transcript": (
                    str(r["ground_truth_transcript"])
                    if "ground_truth_transcript" in df.columns
                    and not pd.isna(r.get("ground_truth_transcript"))
                    else None
                ),
            }
            for column in _CUE_META_COLUMNS.values():
                if column == "enrolment_speakers_utterance":
                    continue
                value = r.get(column) if column in df.columns else None
                row[column] = (
                    str(value).strip()
                    if value is not None and not pd.isna(value) and str(value).strip()
                    else None
                )
            rows.append(row)
    return rows


def _load_mapping(mapping_csv: str | None) -> dict[str, str]:
    """读取比赛 mapping.csv（utterance,path）。"""
    if not mapping_csv:
        return {}
    mapping: dict[str, str] = {}
    with open(mapping_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            utterance = r.get("utterance")
            path = r.get("path")
            if not isinstance(utterance, str) or not utterance.strip():
                raise ValueError(f"mapping.csv contains an empty utterance: {mapping_csv}")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"mapping.csv contains an empty path: {mapping_csv}")
            utterance = utterance.strip()
            path = path.strip()
            if utterance in mapping:
                raise ValueError(f"mapping.csv contains duplicate utterance: {utterance!r}")
            mapping[utterance] = path
    return mapping


def _validate_audio_file(path: Path, root: Path | None = None, *, label: str = "音频") -> Path:
    """验证输入路径是 root 内不含符号链接的普通文件。"""
    path = Path(path)
    if root is not None:
        root = root.expanduser().resolve()
        try:
            path.expanduser().resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label}路径超出 audio_root: {path}") from exc
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label}文件不存在: {path}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label}路径不允许符号链接: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"{label}路径必须是普通文件: {path}")
    return path


def _resolve_audio_path(value: str, audio_root: str | None, mapping: dict[str, str]) -> Path:
    """utterance 值 → 音频路径，兼容 REAL-T 官方双音频目录布局。"""
    root = Path(audio_root).expanduser() if audio_root else None
    if value in mapping:
        mapped = Path(mapping[value]).expanduser()
        if not mapped.is_absolute() and root is not None:
            mapped = root / mapped
        if not mapped.exists() and not mapped.is_symlink():
            raise FileNotFoundError(f"mapping 音频文件不存在: {mapped}")
        return _validate_audio_file(mapped, root)
    p = Path(value).expanduser()
    if p.is_absolute() and (p.exists() or p.is_symlink()):
        return _validate_audio_file(p, root)
    if root:
        candidates = []
        for parent in (root, root / "mixtures", root / "enrolment_speakers"):
            candidates.extend((parent / value, parent / f"{value}.wav"))
        for cand in candidates:
            if cand.exists() or cand.is_symlink():
                return _validate_audio_file(cand, root)
    raise FileNotFoundError(
        f"无法解析音频路径: {value!r}（audio_root={audio_root}, "
        f"mapping {'有' if mapping else '无'} 匹配）"
    )


def _resolve_cue_path(value: str, args: argparse.Namespace, mapping: dict[str, str]) -> Path:
    """utterance/相对路径 → cue 资源文件，按 cue_root → audio_root 顺序解析。"""
    roots = []
    for raw in (getattr(args, "cue_root", None), getattr(args, "audio_root", None)):
        if raw:
            roots.append(Path(raw).expanduser())
    if value in mapping:
        mapped = Path(mapping[value]).expanduser()
        if not mapped.is_absolute() and roots:
            mapped = roots[0] / mapped
        return _validate_audio_file(mapped, roots[0] if roots else None, label="cue")
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and (candidate.exists() or candidate.is_symlink()):
        return _validate_audio_file(candidate, None, label="cue")
    for root in roots:
        for parent in (root, root / "cues"):
            cand = parent / value
            if cand.exists() or cand.is_symlink():
                return _validate_audio_file(cand, root, label="cue")
    raise FileNotFoundError(
        f"无法解析 cue 资源路径: {value!r}（cue_root={getattr(args, 'cue_root', None)}, "
        f"audio_root={getattr(args, 'audio_root', None)}）"
    )


def _row_cue_inputs(
    row: dict,
    roles: Mapping[str, Any] | None,
    args: argparse.Namespace,
    mapping: dict[str, str],
) -> tuple[dict[str, Path], dict[str, Any] | None]:
    """按模型声明的 cue role 解析该行资源；缺列或缺文件时显式失败。

    模型声明的 role 必须有对应 meta 列与真实文件；否则报告缺少的 role 及其
    期望列名，不做静默跳过或占位。
    """
    if not roles:
        return {}, None
    files: dict[str, Path] = {}
    missing_columns: list[str] = []
    for role in sorted(roles):
        column = _CUE_META_COLUMNS.get(role)
        if column is None:
            raise ValueError(
                f"模型声明了未知 cue role {role!r}；支持的 role 为 "
                f"{sorted(_CUE_META_COLUMNS)}"
            )
        value = row.get(column)
        if not value:
            missing_columns.append(f"{role}（列 {column}）")
            continue
        if role == "speaker_waveform":
            # 注册音与旧 enrollment 入口共用同一解析规则（含官方双目录布局）。
            files[role] = _resolve_audio_path(
                str(value), getattr(args, "audio_root", None), mapping
            )
        else:
            files[role] = _resolve_cue_path(str(value), args, mapping)
    if missing_columns:
        raise ValueError(
            "meta 未提供模型声明的 cue 资源: " + ", ".join(missing_columns)
        )
    from tasks.tse.inference.adapter import cue_declaration

    return files, cue_declaration(files, {})


def _format_row_errors(rows: list[dict], limit: int = 3) -> str:
    """把逐行 inference_error 汇总进失败信息，避免只报计数不报原因。"""
    errors = [
        f"{row.get('mixture_utterance')}: {row['inference_error']}"
        for row in rows
        if row.get("inference_error")
    ]
    if not errors:
        return ""
    head = errors[:limit]
    tail = "" if len(errors) <= limit else f"（共 {len(errors)} 条）"
    return "；原因: " + " | ".join(head) + tail


def _uid_cue_identity(declaration: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """只有 speaker_waveform 之外的 cue 才进入 UID。

    纯 enrollment 入口（speaker_waveform 来自 REAL-TSE 的 enrollment 列）继续使用
    dataset/mixture/enrollment identity，使既有 REAL-TSE 输出 basename 保持稳定；
    出现新模态 cue 时 UID 绑定全部实际 cue 文件路径与 inline cue JSON。
    """
    if not declaration:
        return None
    extra = {
        role: entry
        for role, entry in declaration.items()
        if role != "speaker_waveform"
    }
    return extra or None


# ---------------------------------------------------------------------------
# VAD（timing 指标）
# ---------------------------------------------------------------------------
def _energy_vad_frames(wav: np.ndarray, sr: int, frame_shift: float = 0.01,
                       frame_len: float = 0.025, thresh_db: float = -40.0) -> np.ndarray:
    """简单能量 VAD：帧 RMS 相对峰值低 thresh_db 以上判为语音帧。"""
    shift = max(1, int(frame_shift * sr))
    win = max(1, int(frame_len * sr))
    n_frames = max(0, (len(wav) - win) // shift + 1)
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    frames = np.stack([wav[i * shift: i * shift + win] for i in range(n_frames)])
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak = rms.max()
    if peak <= 0:
        return np.zeros(n_frames, dtype=bool)
    return rms > peak * (10 ** (thresh_db / 20.0))


def _silero_vad_frames(wav: np.ndarray, sr: int, vad_model,
                       frame_shift: float = 0.01) -> np.ndarray:
    """silero-VAD 语音段时间戳 → 10ms 帧级 bool 掩码。"""
    from silero_vad import get_speech_timestamps

    wav_t = torch.from_numpy(wav.astype(np.float32))
    if sr != 16000:
        import torchaudio

        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    segments = get_speech_timestamps(wav_t, vad_model, return_seconds=True)
    n_frames = int(np.ceil(len(wav) / sr / frame_shift))
    mask = np.zeros(n_frames, dtype=bool)
    for seg in segments:
        s = int(seg["start"] / frame_shift)
        e = int(np.ceil(seg["end"] / frame_shift))
        mask[s:e] = True
    return mask


class _VAD:
    """VAD 后端封装：silero 可用则用，否则能量 VAD。"""

    def __init__(self):
        self._silero = None
        try:
            from silero_vad import load_silero_vad

            self._silero = load_silero_vad()
            logger.info("timing VAD 后端: silero-vad")
        except ImportError:
            logger.info("timing VAD 后端: 能量 VAD（未安装 silero-vad）")

    def frames(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if self._silero is not None:
            return _silero_vad_frames(wav, sr, self._silero)
        return _energy_vad_frames(wav, sr)


def _timing_prf(ref_mask: np.ndarray, hyp_mask: np.ndarray) -> dict:
    """帧级 precision/recall/F1（对齐到同长度）。"""
    n = min(len(ref_mask), len(hyp_mask))
    if n == 0:
        return {"timing_precision": float("nan"), "timing_recall": float("nan"),
                "timing_f1": float("nan")}
    ref, hyp = ref_mask[:n], hyp_mask[:n]
    tp = int(np.sum(ref & hyp))
    fp = int(np.sum(~ref & hyp))
    fn = int(np.sum(ref & ~hyp))
    # tp+fp==0 时 precision 无定义：记 NaN（汇总口径对 NaN 统一剔除，见 run_evaluation）
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if np.isfinite(precision):
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    else:
        f1 = float("nan")
    return {"timing_precision": precision, "timing_recall": recall, "timing_f1": f1}


# ---------------------------------------------------------------------------
# timing 官方口径（对齐 REAL-TSE-Challenge utils/tse_timing_evaluation.py
# 与 utils/vad_inference_firered.py / utils/prepare_label_segments.py）
# ---------------------------------------------------------------------------
_TIMING_EPS = 1e-8


def _parse_mixture_utterance(mixture_utterance: str) -> tuple[str, float, float]:
    """``{session}_mixture_{start}_{end}`` → (session_id, start, end)。官方同款正则。"""
    pattern = r"^(.+)_mixture_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)$"
    match = re.match(pattern, mixture_utterance)
    if match is None:
        raise ValueError(f"Invalid mixture_utterance format: {mixture_utterance}")
    return match.group(1), float(match.group(2)), float(match.group(3))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """重叠/相邻区间合并（官方 merge_intervals）。"""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for cur_start, cur_end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))
    return merged


def _clip_intervals(intervals, start: float, end: float):
    """区间裁剪到 [start, end] 并合并（官方 clip_intervals）。"""
    clipped = []
    for seg_start, seg_end in intervals:
        seg_start = max(start, seg_start)
        seg_end = min(end, seg_end)
        if seg_end > seg_start:
            clipped.append((seg_start, seg_end))
    return _merge_intervals(clipped)


def _apply_collar_to_segments(intervals, collar: float, duration: float):
    """GT 段双侧扩 collar 后裁剪（官方 apply_collar_to_segments）。"""
    if isinstance(collar, bool) or not isinstance(collar, (int, float)):
        raise ValueError("collar must be a finite number >= 0")
    if not math.isfinite(collar) or collar < 0:
        raise ValueError("collar must be a finite number >= 0")
    expanded = [(s - collar, e + collar) for s, e in intervals]
    return _clip_intervals(expanded, 0.0, duration)


def _segments_to_frame_mask(intervals, duration: float, frame_shift: float) -> np.ndarray:
    """语音段 → 10ms 帧级 bool 掩码（官方 segments_to_mask：
    起点 floor、终点 ceil）。"""
    if isinstance(frame_shift, bool) or not isinstance(frame_shift, (int, float)):
        raise ValueError("frame_shift must be a finite number > 0")
    if not math.isfinite(frame_shift) or frame_shift <= 0:
        raise ValueError("frame_shift must be a finite number > 0")
    if duration <= 0:
        return np.zeros(0, dtype=bool)
    n_frames = int(math.ceil(duration / frame_shift))
    mask = np.zeros(n_frames, dtype=bool)
    for seg_start, seg_end in intervals:
        seg_start = max(0.0, min(duration, seg_start))
        seg_end = max(0.0, min(duration, seg_end))
        if seg_end <= seg_start:
            continue
        start_idx = max(0, int(math.floor(seg_start / frame_shift)))
        end_idx = min(n_frames, int(math.ceil(seg_end / frame_shift)))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True
    return mask


def _timing_frame_metrics(pred_segments, gt_segments, duration: float,
                          frame_shift: float) -> dict:
    """帧级 TP/FP/FN 时长与 P/R/F1（官方 compute_frame_metrics）。"""
    pred_mask = _segments_to_frame_mask(pred_segments, duration, frame_shift)
    gt_mask = _segments_to_frame_mask(gt_segments, duration, frame_shift)

    def _div(num: float, den: float) -> float:
        return 0.0 if den <= _TIMING_EPS else num / den

    tp = float(np.sum(pred_mask & gt_mask) * frame_shift)
    fp = float(np.sum(pred_mask & (~gt_mask)) * frame_shift)
    fn = float(np.sum((~pred_mask) & gt_mask) * frame_shift)
    return {
        "timing_tp_dur": tp,
        "timing_fp_dur": fp,
        "timing_fn_dur": fn,
        "timing_precision": _div(tp, tp + fp),
        "timing_recall": _div(tp, tp + fn),
        "timing_f1": _div(2 * tp, 2 * tp + fp + fn),
    }


def _timing_micro_prf(rows: list[dict]) -> dict | None:
    """micro 聚合：先累加各行 TP/FP/FN 时长再算 P/R/F1（官方 summarize_micro）。"""
    tp = sum(r["timing_tp_dur"] for r in rows if "timing_tp_dur" in r)
    fp = sum(r["timing_fp_dur"] for r in rows if "timing_fp_dur" in r)
    fn = sum(r["timing_fn_dur"] for r in rows if "timing_fn_dur" in r)
    if not any("timing_tp_dur" in r for r in rows):
        return None

    def _div(num: float, den: float) -> float:
        return 0.0 if den <= _TIMING_EPS else num / den

    return {
        "timing_precision": _div(tp, tp + fp),
        "timing_recall": _div(tp, tp + fn),
        "timing_f1": _div(2 * tp, 2 * tp + fp + fn),
        "timing_tp_dur_total": tp,
        "timing_fp_dur_total": fp,
        "timing_fn_dur_total": fn,
    }


class _TimingGTLabels:
    """官方 GT label：读 ``<base>/<dataset>/overlap_records.json``，按目标说话人
    取 overlap 段 → 相对 mixture 起点 → clip → 加 collar（官方
    prepare_label_segments.py 的 phase-1 路径，逐行即时计算，不落盘）。"""

    def __init__(self, gt_json_base_dir: str, collar: float, match_tolerance: float):
        for name, value in (("collar", collar), ("match_tolerance", match_tolerance)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number >= 0")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0")
        self._base = Path(gt_json_base_dir)
        self._collar = collar
        self._tol = match_tolerance
        self._cache: dict[str, dict | None] = {}

    def _overlap_data(self, dataset: str) -> dict | None:
        if dataset not in self._cache:
            path = self._base / dataset / "overlap_records.json"
            if not path.exists():
                logger.warning("timing GT json 缺失，跳过数据集 %s: %s", dataset, path)
                self._cache[dataset] = None
            else:
                self._cache[dataset] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[dataset]

    def label_for_row(self, row: dict) -> tuple[float, list[tuple[float, float]]] | None:
        """返回 (mix_duration, gt_segments_with_collar)；无法构造时返回 None。"""
        speaker = row.get("speaker")
        if not speaker:
            return None
        data = self._overlap_data(row["dataset"])
        if data is None:
            return None
        try:
            session_id, mix_start, mix_end = _parse_mixture_utterance(
                row["mixture_utterance"])
        except ValueError:
            return None
        records = data.get(session_id)
        if not records:
            return None
        mix_duration = max(0.0, mix_end - mix_start)
        matched = None
        for item in records:
            info = item.get("overlap_info", {})
            try:
                start = float(info.get("overlap_start_time"))
                end = float(info.get("overlap_end_time"))
            except (TypeError, ValueError):
                continue
            if abs(start - mix_start) <= self._tol and abs(end - mix_end) <= self._tol:
                matched = item
                break
        if matched is None:
            return None
        gt_abs = []
        for seg in matched.get("overlap_segments", []):
            if seg.get("speaker") != speaker:
                continue
            try:
                s = float(seg.get("start_time"))
                e = float(seg.get("end_time"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(s) and np.isfinite(e):
                gt_abs.append((s, e))
        gt_rel = _clip_intervals(
            [(s - mix_start, e - mix_start) for s, e in gt_abs], 0.0, mix_duration)
        return mix_duration, _apply_collar_to_segments(
            gt_rel, self._collar, mix_duration)


def _peak_normalize_to_int16(wav: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """峰值归一化到 peak 后转 int16（官方 vad_inference_firered.py 的
    peak_normalize_audio：FireRedVAD 的 fbank/cmvn 按 int16 幅度标定）。"""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    max_abs = float(np.max(np.abs(wav))) if wav.size > 0 else 0.0
    if max_abs > 0:
        wav = wav * (peak / max_abs)
    return np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)


class _FireRedVAD:
    """FireRedVAD 后端（官方 vad_inference_firered.py 同款配置与输入约定）。

    依赖 FireRedASR2S 仓库（``--fireredasr2s_root``，缺省
    ``third_party_scripts/FireRedASR2S``）与模型
    ``<root>/pretrained_models/FireRedVAD/VAD``（cmvn.ark + model.pth.tar）。
    """

    def __init__(self, root: Path, use_gpu: bool):
        root = root.resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

        model_dir = root / "pretrained_models" / "FireRedVAD" / "VAD"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"FireRedVAD 模型目录不存在: {model_dir}")
        # 官方 vad_and_evaluation.sh 注入的 VAD 配置（注意与 FireRedASR2S
        # 自带默认值不同：threshold 0.5 / min_silence 10 / merge_silence 50 / extend 5）
        cfg = FireRedVadConfig(
            use_gpu=use_gpu,
            smooth_window_size=5,
            speech_threshold=0.5,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=10,
            merge_silence_frame=50,
            extend_speech_frame=5,
            chunk_max_frame=30000,
        )
        self._vad = FireRedVad.from_pretrained(str(model_dir), cfg)
        logger.info("timing VAD 后端: FireRedVAD（官方口径）model_dir=%s", model_dir)

    def segments(self, wav: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """波形 → 语音段（秒）。输入重采样到 16k 后峰值归一化转 int16。"""
        if sr != 16000:
            import librosa

            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        result, _ = self._vad.detect(_peak_normalize_to_int16(wav))
        return [(float(s), float(e)) for s, e in result.get("timestamps", [])]


class _SileroVADSegments:
    """silero-vad 段级后端（FireRedVAD 不可用时的降级，非官方口径）。"""

    def __init__(self):
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()

    def segments(self, wav: np.ndarray, sr: int) -> list[tuple[float, float]]:
        from silero_vad import get_speech_timestamps

        wav_t = torch.from_numpy(wav.astype(np.float32))
        if sr != 16000:
            import torchaudio

            wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        segs = get_speech_timestamps(wav_t, self._model, return_seconds=True)
        return [(float(s["start"]), float(s["end"])) for s in segs]


class _EnergyVADSegments:
    """能量 VAD 段级后端（最终兜底，非官方口径）。"""

    def segments(self, wav: np.ndarray, sr: int) -> list[tuple[float, float]]:
        mask = _energy_vad_frames(wav, sr)
        segments: list[tuple[float, float]] = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                segments.append((start * 0.01, i * 0.01))
                start = None
        if start is not None:
            segments.append((start * 0.01, len(mask) * 0.01))
        return segments


def _build_timing_vad_backend(args, device: torch.device):
    """构造段级 VAD 后端：firered 优先，依赖缺失时降级 silero → energy。"""
    backend = args.timing_vad_backend
    if backend == "firered":
        root = Path(args.fireredasr2s_root) if args.fireredasr2s_root else (
            _PROJECT_ROOT / "third_party_scripts" / "FireRedASR2S")
        try:
            return _FireRedVAD(root, use_gpu=device.type == "cuda")
        except Exception as e:
            logger.warning(
                "FireRedVAD 不可用（%s），timing 降级 silero-vad（非官方口径）", e)
            backend = "silero"
    if backend == "silero":
        try:
            vad = _SileroVADSegments()
            logger.info("timing VAD 后端: silero-vad（非官方口径降级）")
            return vad
        except Exception as e:
            logger.warning("silero-vad 不可用（%s），timing 降级能量 VAD", e)
            backend = "energy"
    logger.info("timing VAD 后端: 能量 VAD（非官方口径兜底）")
    return _EnergyVADSegments()


# ---------------------------------------------------------------------------
# TER（sherpa-onnx，可选依赖）
# ---------------------------------------------------------------------------
def _find_ter_model_files(model_dir: str) -> dict | None:
    """在模型目录中定位 encoder/decoder/joiner ONNX 与 tokens.txt。

    与官方 asr/backends/zipformer_sherpa.py 的 ``_pick_first`` 行为一致：
    sorted glob 下 ``*.int8.onnx`` 排在 fp32 前（'i' < 'o'），故两种文件
    并存时实际选中 int8——官方文档虽写 "prefer fp32"，其 glob 顺序实际同样
    命中 int8，此处保持行为一致。
    """
    d = Path(model_dir)
    if not d.is_dir():
        return None

    def _glob1(*patterns: str) -> Path | None:
        for pat in patterns:
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
        return None

    encoder = _glob1("encoder*.onnx")
    decoder = _glob1("decoder*.onnx")
    joiner = _glob1("joiner*.onnx")
    tokens = _glob1("tokens.txt")
    if not all((encoder, decoder, joiner, tokens)):
        return None
    return {"encoder": str(encoder), "decoder": str(decoder),
            "joiner": str(joiner), "tokens": str(tokens)}


def _build_ter_recognizers(args) -> dict[str, object] | None:
    """构建 sherpa-onnx Zipformer 识别器（按语种）。可选依赖，缺失时返回 None。"""
    try:
        import sherpa_onnx
    except ImportError:
        logger.warning(
            "未安装 sherpa-onnx，跳过 TER 指标（其余指标照常）。"
            "可选安装: pip install sherpa-onnx"
        )
        return None
    dirs = {
        "zh": args.ter_model_dir_zh or args.ter_model_dir,
        "en": args.ter_model_dir_en or args.ter_model_dir,
    }
    recognizers: dict[str, object] = {}
    for lang, mdir in dirs.items():
        if not mdir:
            continue
        files = _find_ter_model_files(mdir)
        if files is None:
            logger.warning("TER 模型目录不完整（需 encoder/decoder/joiner ONNX + tokens.txt）: %s", mdir)
            continue
        recognizers[lang] = sherpa_onnx.OfflineRecognizer.from_transducer(
            **files, num_threads=2, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search",
        )
        logger.info("TER 识别器已加载（%s）: %s", lang, mdir)
    if not recognizers:
        logger.warning("无可用 TER 模型目录（--ter_model_dir[_zh/_en]），跳过 TER 指标")
        return None
    return recognizers


def _ter_transcribe(recognizer, wav: np.ndarray, sr: int) -> str:
    """sherpa-onnx 单句识别（重采样到 16k）。"""
    from core.metrics import _resample_np

    if sr != 16000:
        wav = _resample_np(wav, sr, 16000)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, wav.astype(np.float32))
    recognizer.decode_stream(stream)
    return stream.result.text


# TER 文本归一化：逐行对齐官方 utils/asr_metrics.py
# （whisper-normalizer + opencc t2s；GT 与 hyp 走不同分支，勿"顺手"统一）。
_WHISPER_NORMALIZERS: tuple | None = None
_ZH_CONVERTER = None


def _get_whisper_normalizers():
    """懒加载 whisper_normalizer 的 English/Basic 归一化器（官方同款）。"""
    global _WHISPER_NORMALIZERS
    if _WHISPER_NORMALIZERS is None:
        try:
            from whisper_normalizer.basic import BasicTextNormalizer
            from whisper_normalizer.english import EnglishTextNormalizer
        except ImportError as e:
            raise ImportError(
                "TER 官方归一化需要 whisper-normalizer：pip install whisper-normalizer"
            ) from e
        _WHISPER_NORMALIZERS = (EnglishTextNormalizer(), BasicTextNormalizer())
    return _WHISPER_NORMALIZERS


def _get_zh_converter():
    """懒加载 opencc t2s 转换器（官方同款）。"""
    global _ZH_CONVERTER
    if _ZH_CONVERTER is None:
        try:
            import opencc
        except ImportError as e:
            raise ImportError(
                "TER 官方归一化需要 opencc：pip install opencc-python-reimplemented"
            ) from e
        _ZH_CONVERTER = opencc.OpenCC("t2s")
    return _ZH_CONVERTER


def _whisper_normalize(transcript: str, language: str) -> str:
    """官方 whisper_normalize：en 用 EnglishTextNormalizer，否则 BasicTextNormalizer。"""
    transcript = transcript.replace("(", "").replace(")", "")
    en, basic = _get_whisper_normalizers()
    normalized = en(transcript) if language == "en" else basic(transcript)
    return " ".join(normalized.split()).strip()


def _normalize_transcript(transcript: str, option: str, language: str) -> str:
    """对齐官方 normalizer_for_transcript。

    - zh：hyp 先 whisper_normalize(basic)；GT 不归一化（保留数据集原样数字/
      标点）；随后统一 opencc t2s 简体化、去全部空白、逐字符空格分隔（字级 token）。
    - en：hyp 先 whisper_normalize(english)；GT **不** whisper 归一化（官方如此，
      保留大小写与逗号等）；随后统一去 "." 与 "…"、压缩空白。
    """
    if option not in ("Predicted", "Ground Truth"):
        raise ValueError(f"Invalid option: {option}")
    if language not in ("zh", "en"):
        raise ValueError(f"Invalid language: {language!r}")
    if language == "zh":
        if option == "Predicted":
            transcript = _whisper_normalize(transcript, language="zh")
        transcript = _get_zh_converter().convert(transcript)
        transcript = re.sub(r"\s+", "", transcript)
        return " ".join(transcript.strip()).strip()
    # en
    transcript = transcript.strip().replace("…", "")
    if option == "Predicted":
        transcript = _whisper_normalize(transcript, language="en")
    transcript = transcript.replace(".", "")
    transcript = re.sub(r"\s+", " ", transcript.strip())
    return transcript.strip()


def _edit_error_rate(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """编辑距离错误率（zh=CER / en=WER 统一实现）。

    与 meeteval ``siso_word_error_rate`` 的逐句 error_rate 等价
    （Levenshtein 距离 / len(ref)），作为 meeteval 缺失时的兜底。
    """
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    n, m = len(ref_tokens), len(hyp_tokens)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def _ter_error_rate(gt: str, pred: str) -> float:
    """TER 单句错误率：优先 meeteval ``siso_word_error_rate``（官方
    calculate_TER.py 同款），缺失时退回等价的本地 Levenshtein 实现。
    GT 为空（无 token）时官方 meeteval 返回 None → 记 NaN（汇总剔除）。"""
    ref_tokens = gt.split()
    if not ref_tokens:
        return float("nan")
    try:
        from meeteval.wer.wer.siso import siso_word_error_rate
    except ImportError:
        return _edit_error_rate(ref_tokens, pred.split())
    error_rate = siso_word_error_rate(gt, pred).error_rate
    return float("nan") if error_rate is None else float(error_rate)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def _load_spk_encoder(ckpt_path: str, device: torch.device):
    from tasks.tse.models.speaker.encoder import SpeakerEncoder

    enc = SpeakerEncoder(pretrained=ckpt_path, freeze=True).to(device).eval()
    logger.info("spksim SpeakerEncoder 已加载: %s", ckpt_path)
    return enc


@torch.inference_mode()
def _spk_embedding(encoder, wav: np.ndarray, sr: int, device: torch.device) -> torch.Tensor:
    import torchaudio

    wav_t = torch.from_numpy(wav.astype(np.float32))
    if sr != 16000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
    return encoder(wav_t.unsqueeze(0).to(device)).squeeze(0).float().cpu()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom <= 0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _parse_dataset_lang_overrides(raw: str) -> dict[str, str]:
    """``AISHELL-4:chs,AliMeeting:chs`` → dict（官方 parse_dataset_lang_overrides）。"""
    overrides: dict[str, str] = {}
    if not raw or not raw.strip():
        return overrides
    for item in raw.strip().split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        dataset, lang = part.split(":", 1)
        dataset, lang = dataset.strip(), lang.strip()
        if dataset and lang in ("en", "chs"):
            overrides[dataset] = lang
    return overrides


def _wespeaker_lang_for_row(row: dict, overrides: dict[str, str],
                            default_lang: str) -> str:
    """行的 wespeaker 语种：meta language 列优先（zh/ch→chs、en→en），
    其次 dataset overrides（官方默认 AISHELL-4/AliMeeting→chs），最后默认。"""
    lang = (row.get("language") or "").lower()
    if lang.startswith(("zh", "ch")):
        return "chs"
    if lang.startswith("en"):
        return "en"
    return overrides.get(row.get("dataset") or "", default_lang)


class _WeSpeakerScorer:
    """wespeakerruntime spksim 后端（官方 spk_similarity_eval.py 同款）。

    - 模型：``<model_dir>/<lang>/model.onnx``（en=voxceleb_resnet34_LM /
      chs=cnceleb_resnet34_LM）；本地缺失时按官方 ``Hub.get_model_by_lang``
      自动下载到 ``~/.wespeaker/``。
    - 打分：``extract_embedding(wav_path)`` + ``compute_cosine_score``，
      embedding 按 (lang, path) 缓存（enroll 在多行间复用，官方同策略）。
    """

    def __init__(self, model_dir: Path | None):
        try:
            import wespeakerruntime as wespeaker
        except ImportError as e:
            raise ImportError(
                "spksim 官方口径需要 wespeakerruntime：pip install wespeakerruntime"
                "（或 --spksim_backend ecapa 退回 vendored ECAPA 后备）"
            ) from e
        self._wespeaker = wespeaker
        self._model_dir = model_dir
        self._speakers: dict[str, object] = {}
        self._emb_cache: dict[tuple[str, str], np.ndarray] = {}

    def _speaker(self, lang: str):
        if lang not in self._speakers:
            onnx_path = None
            if self._model_dir is not None:
                cand = self._model_dir / lang / "model.onnx"
                if cand.exists():
                    onnx_path = str(cand)
            if onnx_path is None:
                # 官方 init_speaker_model 的默认路径：Hub 自动下载
                from wespeakerruntime.hub import Hub

                onnx_path = Hub.get_model_by_lang(lang)
                logger.info("wespeaker 本地模型缺失，Hub 下载/复用: %s", onnx_path)
            self._speakers[lang] = self._wespeaker.Speaker(
                onnx_path=onnx_path, lang=lang)
            logger.info("wespeaker 已加载 lang=%s onnx=%s", lang, onnx_path)
        return self._speakers[lang]

    def _embedding(self, lang: str, wav_path: str) -> np.ndarray:
        key = (lang, wav_path)
        if key not in self._emb_cache:
            emb = self._speaker(lang).extract_embedding(wav_path)
            self._emb_cache[key] = np.asarray(emb).squeeze()
        return self._emb_cache[key]

    def cosine(self, lang: str, est_path: str, ref_path: str) -> float:
        """官方 speaker_cosine_similarity 口径（extract_embedding + cosine）。"""
        speaker = self._speaker(lang)
        est_emb = self._embedding(lang, est_path)
        ref_emb = self._embedding(lang, ref_path)
        return float(speaker.compute_cosine_score(est_emb, ref_emb))


# ---------------------------------------------------------------------------
# DNSMOS 官方口径（对齐 REAL-TSE-Challenge utils/dnsmos_eval.py，
# 该脚本与 Microsoft DNS-Challenge dnsmos_local.py 一致）
# ---------------------------------------------------------------------------
_DNSMOS_SR = 16000
_DNSMOS_INPUT_LENGTH = 9.01


def _dnsmos_melspec(audio: np.ndarray, n_mels: int = 120, frame_size: int = 320,
                    hop_length: int = 160, sr: int = 16000,
                    to_db: bool = True) -> np.ndarray:
    """P808 输入特征（官方 _audio_melspec：n_fft=321、power_to_db(ref=max)+40)/40）。"""
    import librosa

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels)
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T


class _OfficialDNSMOSScorer:
    """DNSMOS SIG/BAK/OVRL + P808（官方 DNSMOSComputeScore 逐行对齐）。

    - 主模型 ``sig_bak_ovr.onnx``：非个性化 polyfit（p_ovr/p_sig/p_bak 二次拟合）；
    - P808 模型 ``model_v8.onnx``：输入为 seg[:-160] 的 120 维 melspec；
    - 滑窗：INPUT_LENGTH=9.01s、hop=1s；短音频 ``append`` 倍增补齐；
    - 非 16k 输入用 librosa.resample（官方同款）。
    """

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        primary = model_dir / "sig_bak_ovr.onnx"
        p808 = model_dir / "model_v8.onnx"
        missing = [p for p in (primary, p808) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"DNSMOS 模型缺失: {missing}")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self._sess = ort.InferenceSession(str(primary), providers=providers)
        self._p808_sess = ort.InferenceSession(str(p808), providers=providers)
        # 官方 _get_polyfit_val(is_personalized_MOS=False)
        self._p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        self._p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        self._p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        logger.info("DNSMOS 已加载（官方口径 SIG/BAK/OVRL + P808）: %s", model_dir)

    def __call__(self, audio: np.ndarray, fs: int) -> dict[str, float]:
        audio = np.asarray(audio, dtype=np.float64)
        if audio.size == 0:
            return {
                "SIG": float("nan"),
                "BAK": float("nan"),
                "OVRL": float("nan"),
                "P808": float("nan"),
            }

        import librosa

        if fs != _DNSMOS_SR:
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=fs, target_sr=_DNSMOS_SR)
        len_samples = int(_DNSMOS_INPUT_LENGTH * _DNSMOS_SR)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / _DNSMOS_SR) - _DNSMOS_INPUT_LENGTH) + 1
        pred_sig, pred_bak, pred_ovr, pred_p808 = [], [], [], []
        for idx in range(num_hops):
            seg = audio[idx * _DNSMOS_SR: int((idx + _DNSMOS_INPUT_LENGTH) * _DNSMOS_SR)]
            if len(seg) < len_samples:
                continue
            input_features = np.array(seg, dtype=np.float32)[np.newaxis, :]
            p808_input = np.array(
                _dnsmos_melspec(audio=seg[:-160]), dtype=np.float32)[np.newaxis, :, :]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self._sess.run(
                None, {"input_1": input_features})[0][0]
            pred_sig.append(float(self._p_sig(float(mos_sig_raw))))
            pred_bak.append(float(self._p_bak(float(mos_bak_raw))))
            pred_ovr.append(float(self._p_ovr(float(mos_ovr_raw))))
            pred_p808.append(
                float(self._p808_sess.run(None, {"input_1": p808_input})[0][0][0]))
        return {
            "SIG": float(np.mean(pred_sig)),
            "BAK": float(np.mean(pred_bak)),
            "OVRL": float(np.mean(pred_ovr)),
            "P808": float(np.mean(pred_p808)),
        }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _write_inference_manifest_locked(
    rows: list[dict],
    path: Path,
    *,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """调用方持有 TSE output-dir 锁时发布 inference manifest。"""
    return tse_inference_artifacts._write_inference_manifest_locked(
        rows, path, _make_uid, artifact_meta=artifact_meta
    )


def _write_inference_manifest(
    rows: list[dict],
    path: Path,
    *,
    artifact_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """兼容公开入口：获取 TSE output-dir 锁后发布 inference manifest。"""
    return tse_inference_artifacts.write_inference_manifest(
        rows, path, _make_uid, artifact_meta=artifact_meta
    )


def _write_tse_summary_locked(
    output_dir: str | Path,
    payload: dict[str, Any],
    *,
    model: str | Path | None = None,
    config: str | Path | None = None,
    meta: list[str] | None = None,
    artifact_fingerprint: dict[str, Any] | None = None,
    step: str = "all",
    active_run: bool = False,
    inference_manifest_path: str | Path | None = None,
) -> Path:
    """调用方持锁时发布 summary/envelope，并在 CURRENT 前完成复核。"""
    output = tse_inference_artifacts.ensure_output_directory(output_dir)
    path = output / "tse_eval_summary.json"
    run_id = str(
        payload.get("run_id")
        or payload.get("summary", {}).get("run_id")
        or new_run_id()
    )
    summary = attach_run_id(payload.get("summary", {}), run_id)
    bound_payload = {**payload, "run_id": run_id, "summary": summary}
    if not active_run:
        begin_artifact_run(output, run_id)
    atomic_write_json(path, bound_payload)
    PipelineTaskAdapter(
        task="tse",
        pipeline=str(payload.get("pipeline", "eval_pipeline_tse")),
        step=step,
        output_dir=output,
        provenance={
            "model": model if model is not None else payload.get("model"),
            "config": config,
            "meta": meta,
            "parameters": (
                artifact_fingerprint.get("artifacts", {}).get("parameters", {})
                if artifact_fingerprint is not None
                else {}
            ),
            "entry": "eval_pipeline_tse",
            "artifact_fingerprint": artifact_fingerprint,
            "dataset_identity": (
                artifact_fingerprint.get("artifacts", {}).get("dataset")
                if artifact_fingerprint is not None
                else None
            ),
        },
        summary_filename=path.name,
    ).publish(summary, run_id=run_id)

    def verify_current() -> None:
        if artifact_fingerprint is not None:
            tse_inference_artifacts.verify_artifact_meta(artifact_fingerprint)
        if inference_manifest_path is not None:
            tse_inference_artifacts.verify_inference_manifest(
                inference_manifest_path, artifact_meta=artifact_fingerprint
            )

    publish_artifact_commit(
        output,
        run_id=run_id,
        summary_filename=path.name,
        precommit=verify_current,
        require_result_manifest=summary.get("status") in {"done", "partial"},
        lock_held=True,
    )
    return path


def _write_tse_summary(
    output_dir: str | Path,
    payload: dict[str, Any],
    **kwargs,
) -> Path:
    """公开 writer：持有 TSE output-dir 锁后发布 summary/CURRENT。"""
    with tse_inference_artifacts.exclusive_output_run(output_dir) as output:
        return _write_tse_summary_locked(output, payload, **kwargs)


def _coverage_status(coverage: dict[str, dict[str, Any]]) -> str:
    incomplete = [
        item for item in coverage.values()
        if item["eligible"] > item["computed"]
    ]
    if not incomplete:
        return "done"
    if any(item["computed"] for item in coverage.values()):
        return "partial"
    return "error"


def run_evaluation(rows, args, metrics: list[str],
                   audio_root, mapping, device) -> tuple[list[dict], dict]:
    """按 metrics 列表逐行计算指标，返回 (行结果, 含 coverage 的汇总)。"""
    _validate_timing_parameters(args)
    from evaluate.eval_io import load_audio

    requested_metrics = list(metrics)
    skipped_metrics: list[dict[str, str]] = []
    spk_encoder = None
    wespeaker_scorer = None
    wespeaker_overrides: dict[str, str] = {}
    dnsmos_model = None
    vad = None
    timing_gt = None
    ter_recognizers = None

    def _skip_metric(metric: str, reason: str) -> None:
        metrics.remove(metric)
        skipped_metrics.append({"metric": metric, "reason": reason})

    if "spksim" in metrics:
        if args.spksim_backend == "wespeakerruntime":
            model_dir = (Path(args.wespeaker_model_dir) if args.wespeaker_model_dir
                         else _PROJECT_ROOT / "third_party_scripts" / "wespeaker")
            try:
                wespeaker_scorer = _WeSpeakerScorer(model_dir)
                wespeaker_overrides = _parse_dataset_lang_overrides(
                    args.dataset_lang_overrides)
            except ImportError as e:
                logger.warning("%s，跳过 spksim", e)
                _skip_metric("spksim", f"{type(e).__name__}: {e}")
        else:  # ecapa 后备（非官方口径）
            ckpt = args.spk_encoder_ckpt
            if ckpt is None:
                ckpt = getattr(args, "_model_spk_pretrained", None)
            ckpt_path = _resolve_ckpt_path(str(ckpt)) if ckpt else None
            if ckpt_path is None or not ckpt_path.exists():
                reason = "SpeakerEncoder checkpoint unavailable"
                logger.warning("无可用 SpeakerEncoder 权重（--spk_encoder_ckpt），跳过 spksim")
                _skip_metric("spksim", reason)
            else:
                spk_encoder = _load_spk_encoder(str(ckpt_path), device)
    if "dnsmos" in metrics:
        dnsmos_dir = (Path(args.dnsmos_model_dir) if args.dnsmos_model_dir
                      else _PROJECT_ROOT / "third_party_scripts" / "DNSMOS" / "DNSMOS")
        try:
            dnsmos_model = _OfficialDNSMOSScorer(dnsmos_dir)
        except (FileNotFoundError, ImportError) as e:
            # 模型缺失时剔除该指标，其余指标继续评测。
            logger.warning("DNSMOS 不可用，跳过 dnsmos 指标（其余指标照常）: %s", e)
            _skip_metric("dnsmos", f"{type(e).__name__}: {e}")
    if "timing" in metrics:
        if args.timing_gt_json_base_dir:
            # 官方口径：GT label + 段级 VAD（FireRedVAD 优先）
            timing_gt = _TimingGTLabels(
                args.timing_gt_json_base_dir,
                collar=args.timing_collar,
                match_tolerance=args.timing_match_tolerance)
            vad = _build_timing_vad_backend(args, device)
        else:
            # 无 GT label 的近似口径（非官方）：ref 用 mixture VAD
            logger.warning(
                "未提供 --timing_gt_json_base_dir，timing 走 mixture-VAD 近似口径"
                "（非比赛官方口径，仅衡量语音保留）")
            vad = _VAD()
    if "ter" in metrics:
        ter_recognizers = _build_ter_recognizers(args)
        if ter_recognizers is None:
            _skip_metric("ter", "TER backend or language model unavailable")

    intrusive = [m for m in metrics if m in ("sisdr", "pesq", "stoi")]
    if intrusive:
        from core.metrics import (
            _resample_np,
            compute_pesq,
            compute_sisdr,
            compute_stoi,
        )

    from tqdm import tqdm

    def _eligible(row: dict, metric: str) -> bool:
        if metric == "ter":
            return bool(row.get("ground_truth_transcript"))
        if metric in ("sisdr", "pesq", "stoi"):
            return bool(row.get("target_utterance"))
        return True

    def _computed(row: dict, metric: str) -> bool:
        keys = {
            "spksim": ("spksim_tse_enroll", "spksim_mix_enroll"),
            "dnsmos": ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"),
            "timing": ("timing_precision", "timing_recall", "timing_f1"),
            "ter": ("ter",), "sisdr": ("sisdr",), "pesq": ("pesq",),
            "stoi": ("stoi",),
        }[metric]
        return all(_finite(row.get(key)) for key in keys)

    for row in rows:
        row["_metric_tracking"] = {
            metric: {"eligible": _eligible(row, metric), "attempted": False}
            for metric in requested_metrics
        }

    for row in tqdm(rows, desc="Evaluate", unit="utt"):
        eligible_active = [
            metric for metric in metrics
            if row["_metric_tracking"][metric]["eligible"]
        ]
        if not row.get("tse_path") or not Path(row["tse_path"]).exists():
            continue
        try:
            tse_wav, tse_sr = load_audio(row["tse_path"])
        except Exception as e:
            for metric in eligible_active:
                row["_metric_tracking"][metric]["attempted"] = True
            row.setdefault("_metric_errors", {})["audio"] = f"{type(e).__name__}: {e}"
            logger.warning("评测音频读取失败 %s: %s", row["mixture_utterance"], e)
            continue

        mix_wav_t: np.ndarray | None = None
        mix_sr_t: int = 0

        def _get_mix() -> tuple[np.ndarray, int]:
            nonlocal mix_wav_t, mix_sr_t
            if mix_wav_t is None:
                mix_path = _resolve_audio_path(
                    row["mixture_utterance"], audio_root, mapping)
                mix_wav_t, mix_sr_t = load_audio(str(mix_path))
            return mix_wav_t, mix_sr_t

        for metric in eligible_active:
            row["_metric_tracking"][metric]["attempted"] = True
            try:
                if metric == "spksim":
                    enroll_path = _resolve_audio_path(
                        row["enrolment_speakers_utterance"], audio_root, mapping)
                    if wespeaker_scorer is not None:
                        ws_lang = _wespeaker_lang_for_row(
                            row, wespeaker_overrides, args.wespeaker_lang)
                        row["spksim_tse_enroll"] = wespeaker_scorer.cosine(
                            ws_lang, row["tse_path"], str(enroll_path))
                        mix_path = _resolve_audio_path(
                            row["mixture_utterance"], audio_root, mapping)
                        row["spksim_mix_enroll"] = wespeaker_scorer.cosine(
                            ws_lang, str(mix_path), str(enroll_path))
                    else:
                        enroll_wav, enroll_sr = load_audio(str(enroll_path))
                        e_emb = _spk_embedding(
                            spk_encoder, enroll_wav, enroll_sr, device)
                        t_emb = _spk_embedding(spk_encoder, tse_wav, tse_sr, device)
                        row["spksim_tse_enroll"] = _cosine(e_emb, t_emb)
                        mix_wav, mix_sr = _get_mix()
                        m_emb = _spk_embedding(spk_encoder, mix_wav, mix_sr, device)
                        row["spksim_mix_enroll"] = _cosine(e_emb, m_emb)
                elif metric == "dnsmos":
                    scores = dnsmos_model(tse_wav, tse_sr)
                    for key, value in scores.items():
                        row[f"dnsmos_{key.lower()}"] = value
                elif metric == "timing":
                    if timing_gt is not None:
                        label = timing_gt.label_for_row(row)
                        if label is None:
                            raise ValueError("timing GT label missing")
                        mix_duration, gt_segments = label
                        pred_segments = _clip_intervals(
                            vad.segments(tse_wav, tse_sr), 0.0, mix_duration)
                        row.update(_timing_frame_metrics(
                            pred_segments, gt_segments, mix_duration,
                            args.timing_frame_shift))
                    else:
                        hyp = vad.frames(tse_wav, tse_sr)
                        ref_wav_v, ref_sr_v = _get_mix()
                        row.update(_timing_prf(
                            vad.frames(ref_wav_v, ref_sr_v), hyp))
                elif metric == "ter":
                    lang = row.get("language") or args.default_lang
                    lang = "zh" if lang.startswith(("zh", "ch")) else "en"
                    rec = ter_recognizers.get(lang)
                    if rec is None:
                        raise ValueError(f"TER recognizer missing for {lang}")
                    hyp_text = _ter_transcribe(rec, tse_wav, tse_sr)
                    gt_text = _normalize_transcript(
                        row["ground_truth_transcript"], "Ground Truth", lang)
                    pred_text = _normalize_transcript(hyp_text, "Predicted", lang)
                    row["ter"] = _ter_error_rate(gt_text, pred_text)
                    row["ter_hypothesis"] = hyp_text
                else:
                    ref_path = _resolve_audio_path(
                        row["target_utterance"], args.ref_root or audio_root, mapping)
                    ref_wav, ref_sr = load_audio(str(ref_path))
                    if ref_sr != tse_sr:
                        ref_wav = _resample_np(ref_wav, ref_sr, tse_sr)
                    n = min(len(ref_wav), len(tse_wav))
                    if n == 0:
                        raise ValueError("参考或估计音频为空（0 样本）")
                    fn = {"sisdr": compute_sisdr, "pesq": compute_pesq,
                          "stoi": compute_stoi}[metric]
                    row[metric] = fn(ref_wav[:n], tse_wav[:n], tse_sr)
                if not _computed(row, metric):
                    raise ValueError("metric returned no finite result")
            except Exception as e:
                row.setdefault("_metric_errors", {})[metric] = f"{type(e).__name__}: {e}"
                logger.warning("指标 %s 计算失败 %s: %s",
                               metric, row["mixture_utterance"], e)

    # ---- 汇总（均值；timing 官方口径为 micro 聚合）----
    result_keys = {
        "spksim_tse_enroll", "spksim_mix_enroll",
        "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808",
        "timing_precision", "timing_recall", "timing_f1",
        "ter", "sisdr", "pesq", "stoi",
    }
    summary: dict[str, Any] = {}
    for key in sorted(result_keys):
        vals = [row[key] for row in rows if _finite(row.get(key))]
        if vals:
            summary[key] = float(np.mean(vals))
    # timing 官方口径：micro P/R/F1（先累加 TP/FP/FN 时长再算比值）覆盖逐行均值
    micro = _timing_micro_prf(rows)
    if micro is not None:
        summary.update(micro)

    coverage: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        eligible = sum(
            bool(row["_metric_tracking"][metric]["eligible"]) for row in rows
        )
        attempted = sum(
            bool(row["_metric_tracking"][metric]["attempted"]) for row in rows
        )
        computed = sum(_computed(row, metric) for row in rows)
        failed = max(0, attempted - computed)
        coverage[metric] = {
            "eligible": eligible,
            "attempted": attempted,
            "computed": computed,
            "failed": failed,
            "coverage": computed / eligible if eligible else None,
        }

    summary["status"] = _coverage_status(coverage)
    summary["metrics"] = {
        "requested": requested_metrics,
        "active": list(metrics),
        "skipped": skipped_metrics,
    }
    summary["coverage"] = coverage
    summary["num_utterances"] = sum(
        1 for row in rows if row.get("tse_path") and Path(row["tse_path"]).exists()
    )
    summary["num_rows"] = len(rows)
    return rows, summary


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and np.isfinite(v)


def _export_official_format_csvs(rows: list[dict], out_dir: Path) -> list[Path]:
    """导出官方 REAL-TSE-Challenge 约定的指标 CSV（``eval_metrics/<目录名>_*.csv``）。

    列名与文件名逐一对齐官方 ``utils/aggregate_eval_summary.py`` 的消费约定，
    使本 pipeline 的评测产物可直接被官方汇总工具（及本仓库移植版
    ``evaluate/aggregate_eval_summary.py``）读取。仅导出实际计算过的指标
    （对应列至少一行有限值）；TER 文件名无 ``_csv`` 之前的指标区分，
    官方统一为 ``<name>_TER.csv``（zh=CER / en=WER 同列 ``wer_or_cer``）。
    """
    import pandas as pd

    name = out_dir.name
    sub = out_dir / "eval_metrics"
    written: list[Path] = []

    def _write(suffix: str, records: list[dict], columns: list[str]) -> None:
        if not records:
            return
        sub.mkdir(parents=True, exist_ok=True)
        p = sub / f"{name}_{suffix}.csv"
        atomic_write_text(
            p, pd.DataFrame(records, columns=columns).to_csv(index=False)
        )
        written.append(p)

    def _base(r: dict) -> dict:
        return {
            "utterance": r["mixture_utterance"],
            "dataset": r.get("dataset") or "",
            "language": r.get("language") or "",
        }

    # ---- TER（zh=CER / en=WER 同列 wer_or_cer；source=dataset，官方列名）----
    _write(
        "TER",
        [{"mixture_utterance": r["mixture_utterance"],
          "source": r.get("dataset") or "",
          "language": r.get("language") or "",
          "wer_or_cer": r["ter"]}
         for r in rows if _finite(r.get("ter"))],
        ["mixture_utterance", "source", "language", "wer_or_cer"],
    )

    # ---- SpkSim（enrol-tse 与 enrol-mixture 两个文件；status=ok）----
    for suffix, key in (
        ("spk_similarity", "spksim_tse_enroll"),
        ("spk_similarity_mixture_enrol", "spksim_mix_enroll"),
    ):
        _write(
            suffix,
            [{**_base(r), "speaker_cosine_similarity": r[key], "status": "ok"}
             for r in rows if _finite(r.get(key))],
            ["utterance", "dataset", "language", "speaker_cosine_similarity", "status"],
        )

    # ---- DNSMOS（P.835 SIG/BAK/OVRL + P808；status=ok；P808 未算时留空列）----
    _write(
        "dnsmos",
        [{"utterance": r["mixture_utterance"],
          "dataset": r.get("dataset") or "",
          "SIG": r.get("dnsmos_sig"), "BAK": r.get("dnsmos_bak"),
          "OVRL": r.get("dnsmos_ovrl"),
          "P808": r.get("dnsmos_p808") if _finite(r.get("dnsmos_p808")) else None,
          "status": "ok"}
         for r in rows if _finite(r.get("dnsmos_ovrl"))],
        ["utterance", "dataset", "SIG", "BAK", "OVRL", "P808", "status"],
    )

    # ---- TSE_TIMING（逐行 P/R/F1；汇总侧官方口径为 micro，见 summary）----
    _write(
        "TSE_TIMING",
        [{**_base(r), "precision": r["timing_precision"],
          "recall": r["timing_recall"], "f1": r["timing_f1"]}
         for r in rows if _finite(r.get("timing_f1"))],
        ["utterance", "dataset", "language", "precision", "recall", "f1"],
    )

    return written


def _main_locked(args: argparse.Namespace, out_dir: Path) -> int:
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in _ALL_METRICS]
    if unknown:
        raise SystemExit(f"未知指标: {unknown}（可选: {_ALL_METRICS}）")
    try:
        _validate_timing_parameters(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.max_samples is not None and args.max_samples < 0:
        raise SystemExit("--max_samples must be >= 0")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wav_dir = out_dir / "wav"
    run_id = new_run_id()

    try:
        rows = _read_meta(args.meta)
    except Exception as exc:
        reason = f"meta_read_failed: {type(exc).__name__}: {exc}"
        begin_artifact_run(out_dir, run_id)
        _write_tse_summary_locked(
            out_dir,
            {
                "pipeline": "eval_pipeline_tse",
                "model": str(Path(args.model).expanduser().resolve()),
                "metrics": metrics,
                "run_id": run_id,
                "summary": {
                    "status": "error",
                    "reason": reason,
                    "coverage": {},
                    "inference": {
                        "expected": 0,
                        "ok": 0,
                        "failed": 0,
                        "coverage": None,
                    },
                },
            },
            model=args.model,
            config=args.config,
            meta=args.meta,
            step="all",
            active_run=True,
        )
        logger.error("读取 meta 失败: %s", reason)
        return 2
    try:
        dataset_identity = _build_dataset_identity(args, rows)
    except Exception as exc:
        reason = f"dataset_identity_failed: {type(exc).__name__}: {exc}"
        begin_artifact_run(out_dir, run_id)
        _write_tse_summary_locked(
            out_dir,
            {
                "pipeline": "eval_pipeline_tse",
                "model": str(Path(args.model).expanduser().resolve()),
                "metrics": metrics,
                "run_id": run_id,
                "summary": {
                    "status": "error",
                    "reason": reason,
                    "coverage": {},
                    "inference": {
                        "expected": 0,
                        "ok": 0,
                        "failed": 0,
                        "coverage": None,
                    },
                },
            },
            model=args.model,
            config=args.config,
            meta=args.meta,
            step="all",
            active_run=True,
        )
        logger.error("构造 dataset identity 失败: %s", reason)
        return 2
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        reason = (
            "max_samples_selected_zero_rows"
            if args.max_samples == 0
            else "meta_contains_no_rows"
        )
        begin_artifact_run(out_dir, run_id)
        _write_tse_summary_locked(
            out_dir,
            {
                "pipeline": "eval_pipeline_tse",
                "model": str(Path(args.model).expanduser().resolve()),
                "metrics": metrics,
                "run_id": run_id,
                "summary": {
                    "status": "error",
                    "reason": reason,
                    "dataset_identity": dataset_identity,
                    "coverage": {},
                    "inference": {
                        "expected": 0,
                        "ok": 0,
                        "failed": 0,
                        "coverage": None,
                    },
                },
            },
            model=args.model,
            config=args.config,
            meta=args.meta,
            step="all",
            active_run=True,
        )
        logger.error("无可评测 meta 记录: %s", reason)
        return 2
    tse_inference_artifacts.validate_batch(rows)
    begin_artifact_run(out_dir, run_id)
    mapping = _load_mapping(args.mapping_csv)
    logger.info("共 %d 条 meta 记录，指标: %s", len(rows), metrics)

    inference_params = {
        "device": str(device),
        "apply_vad": bool(args.apply_vad),
    }
    inference_manifest_path = out_dir / "tse_inference_manifest.json"
    if not args.skip_inference:
        raise RuntimeError(
            "_main_locked requires public TSE inference to run before evaluation; "
            "call main() instead of invoking the legacy evaluation lifecycle directly"
        )
    snapshot = getattr(args, "_public_tse_snapshot", None)
    if snapshot is None:
        snapshot = tse_inference_artifacts.TSEInferenceArtifactSnapshot.inspect_for_skip(
            args.model,
            args.config,
            inference_params,
            dataset_identity=dataset_identity,
        )
    args._model_spk_pretrained = snapshot.configured_speaker_encoder_pretrained
    base_meta = snapshot.meta
    # cue 角色由公共推理入口（_run_public_tse_inference）在验证前记录；未记录
    # 角色说明绕过了公共 runner，直接 fail-fast 要求重新推理。
    cue_roles = getattr(args, "_public_tse_cue_roles", None)
    if cue_roles is None:
        raise RuntimeError(
            "--skip_inference 复核需要公共推理入口记录的 cue roles；"
            "请通过 main() 运行或重新推理"
        )
    for row in rows:
        row["tse_path"] = None
        row["inference_status"] = "failed"
        row["inference_reused"] = False
        try:
            mix_path = _resolve_audio_path(
                row["mixture_utterance"], args.audio_root, mapping)
            cue_files, declaration = _row_cue_inputs(row, cue_roles, args, mapping)
            row["cues"] = _uid_cue_identity(declaration)
            if row["cues"] is None:
                row.pop("cues")
            out_path = wav_dir / f"{_out_name(_make_uid(row))}.wav"
            meta_path = out_path.with_name(out_path.stem + ".meta.json")
            input_paths = {"mixture": mix_path, **cue_files}
            expected_meta = snapshot.bind_inputs(input_paths, cues=declaration)
            if not out_path.exists():
                raise FileNotFoundError(f"inference wav missing: {out_path}")
            cached = tse_inference_artifacts.load_sidecar(meta_path)
            if not snapshot.matches_sidecar(cached, expected_meta, out_path):
                raise ValueError("sidecar missing, legacy, or fingerprint mismatch")
            row["tse_path"] = str(out_path)
            row["inference_status"] = "ok"
            row["inference_reused"] = True
            row["_inference_artifacts"] = expected_meta
        except Exception as exc:
            row["inference_error"] = f"{type(exc).__name__}: {exc}"
    manifest = _write_inference_manifest_locked(
        rows, inference_manifest_path, artifact_meta=base_meta
    )
    if manifest["failed"] and not args.allow_partial:
        _write_tse_summary_locked(
            out_dir,
            {
                "pipeline": "eval_pipeline_tse",
                "model": str(Path(args.model).expanduser().resolve()),
                "metrics": metrics,
                "run_id": run_id,
                "summary": {
                    "status": "error",
                    "reason": "inference_artifact_validation_failed",
                    "dataset_identity": dataset_identity,
                    "coverage": {},
                    "inference": {
                        "expected": manifest["expected"],
                        "ok": manifest["ok"],
                        "failed": manifest["failed"],
                        "coverage": manifest["ok"] / manifest["expected"]
                        if manifest["expected"] else None,
                    },
                },
            },
            model=args.model,
            config=args.config,
            meta=args.meta,
            artifact_fingerprint=base_meta,
            step="evaluate",
            active_run=True,
            inference_manifest_path=inference_manifest_path,
        )
        raise SystemExit(
            f"--skip_inference 无法验证 {manifest['failed']}/{manifest['expected']} 个 artifact；"
            "旧 sidecar 不可验证。请重新推理，或显式传 --allow_partial。"
            + _format_row_errors(rows)
        )

    rows, summary = run_evaluation(
        rows, args, metrics, args.audio_root, mapping, device)
    inference_expected = len(rows)
    inference_ok = sum(row.get("inference_status") == "ok" for row in rows)
    inference_failed = inference_expected - inference_ok
    summary["inference"] = {
        "expected": inference_expected,
        "ok": inference_ok,
        "failed": inference_failed,
        "coverage": inference_ok / inference_expected if inference_expected else None,
    }
    summary["dataset_identity"] = dataset_identity
    if inference_failed:
        if not inference_ok or summary["status"] == "error":
            summary["status"] = "error"
        else:
            summary["status"] = "partial"

    # ---- 写出结果 ----
    import pandas as pd

    drop_cols = {
        "tse_path",
        "_metric_tracking",
        "_metric_errors",
        "_inference_artifacts",
    }
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in drop_cols}
                       for r in rows])
    # CSV 公式注入防护：=+-@ 开头的文本单元格前加单引号
    df = df.map(_escape_csv_formula)
    results_csv = out_dir / "tse_eval_results.csv"
    atomic_write_text(results_csv, df.to_csv(index=False))
    # 官方格式五 CSV（eval_metrics/<目录名>_*.csv），供官方/移植版汇总工具消费
    official_csvs = _export_official_format_csvs(rows, out_dir)
    summary = attach_run_id(summary, run_id)
    publish_result_manifest(
        out_dir,
        summary,
        artifact_paths=[results_csv, *official_csvs],
    )
    summary_payload = {
        "pipeline": "eval_pipeline_tse",
        "model": str(Path(args.model).expanduser().resolve()),
        # 实际生效指标（run_evaluation 就地剔除降级项后的列表）
        "metrics": metrics,
        "run_id": run_id,
        "summary": summary,
    }
    _write_tse_summary_locked(
        out_dir,
        summary_payload,
        model=args.model,
        config=args.config,
        meta=args.meta,
        artifact_fingerprint=base_meta,
        step="evaluate" if args.skip_inference else "all",
        active_run=True,
        inference_manifest_path=inference_manifest_path,
    )

    logger.info("==== 汇总 ====")
    for k, v in summary.items():
        logger.info("  %-24s %.4f" % (k, v) if isinstance(v, float) else "  %-24s %s" % (k, v))
    logger.info("明细: %s", results_csv)
    logger.info("汇总: %s", out_dir / "tse_eval_summary.json")
    if official_csvs:
        logger.info("官方格式 CSV（可用 evaluate.aggregate_eval_summary 汇总）: %s",
                    ", ".join(str(p) for p in official_csvs))
    if summary["status"] != "done" and not args.allow_partial:
        logger.error(
            "评测 coverage 不完整（status=%s）；显式传 --allow_partial 可接受部分结果",
            summary["status"],
        )
        return 2
    return 0



def _public_inference_artifact_binding(
    snapshot: tse_inference_artifacts.TSEInferenceArtifactSnapshot,
    dataset_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """将 TSE loader 的实际 source identities 交给公共推理 runner。"""
    identities = snapshot.source_file_identities()
    identities.setdefault("config", dict(identities["checkpoint"]))
    files = {name: identity["path"] for name, identity in identities.items()}
    return bind_artifacts(
        files,
        precomputed=identities,
        parameters={
            "mode": "tse",
            "dataset_identity": dataset_identity,
            "implementation": _tse_implementation_fingerprint(),
            "tse_artifact_snapshot": snapshot.meta,
        },
    )


def _run_public_tse_inference(args: argparse.Namespace) -> None:
    """通过公共 Interface 完成 TSE WAV 推理，再从 report 发布 sidecar。

    公共 ``InferenceReport`` 是 WAV 执行和 reuse 的唯一事实来源；TSE sidecar
    不参与 WAV reuse 判定，是否跳过推理只由公共 runner 决定。
    """
    from tasks.tse.inference import TSEExtractor, TSEInferenceAdapter
    from tasks.tse.inference.sidecar import publish_sidecars_from_report

    rows = _read_meta(args.meta)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        return
    mapping = _load_mapping(args.mapping_csv)
    extractor = TSEExtractor(
        args.model,
        config_path=args.config,
        device=str(args.device),
        apply_vad=args.apply_vad,
    )
    dataset_identity = _build_dataset_identity(args, rows)
    snapshot = tse_inference_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor,
        {"device": str(args.device), "apply_vad": bool(args.apply_vad)},
        dataset_identity=dataset_identity,
    )
    args._public_tse_snapshot = snapshot
    args._public_tse_base_meta = snapshot.meta
    adapter = TSEInferenceAdapter(extractor)
    cue_roles = adapter.declared_cue_roles()
    args._public_tse_cue_roles = cue_roles
    wav_dir = Path(args.output_dir).expanduser().resolve() / "wav"
    items: list[FileInferenceItem] = []
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        mixture = _resolve_audio_path(row["mixture_utterance"], args.audio_root, mapping)
        cue_files, declaration = _row_cue_inputs(row, cue_roles, args, mapping)
        row["cues"] = _uid_cue_identity(declaration)
        if row["cues"] is None:
            row.pop("cues")
        uid = _make_uid(row)
        inputs: dict[str, Any] = {"mixture": mixture, **cue_files}
        expected[uid] = snapshot.bind_inputs(inputs, cues=declaration)
        items.append(
            FileInferenceItem(
                uid=uid,
                inputs=inputs,
                output=wav_dir / f"{_out_name(uid)}.wav",
                attributes={"dataset": row.get("dataset")},
            )
        )
    artifact_fingerprint = _public_inference_artifact_binding(
        snapshot, dataset_identity
    )
    source_identities = snapshot.source_file_identities()
    model_identity = source_identities["checkpoint"]
    config_identity = source_identities.get("config")
    spec = InferenceSpec(
        config=config_identity["path"] if config_identity else model_identity["path"],
        checkpoint=model_identity["path"],
        output_dir=wav_dir,
        items=items,
        artifacts=None,
        reuse=ReusePolicy(enabled=True),
        allow_partial=args.allow_partial,
        run_metadata={
            "evaluate_entry": "tse_adapter",
            "sidecar_schema": _ARTIFACT_SCHEMA,
        },
    )
    report = run_inference(
        spec,
        adapter,
        artifact_fingerprint=artifact_fingerprint,
    )
    publish_sidecars_from_report(report, expected, output_dir=wav_dir)



def main() -> int:
    """先经公共 runner 完成 TSE 音频推理，再进入既有 sidecar/评测生命周期。"""
    args = parse_args()
    if getattr(args, "max_samples", None) is not None and args.max_samples < 0:
        raise SystemExit("--max_samples must be >= 0")
    if not getattr(args, "skip_inference", False):
        _run_public_tse_inference(args)
        # 后续生命周期分支严格验证公共 runner 生成的 WAV 与 sidecar，并继续
        # 生成 TSE inference manifest、CSV 和评测 summary。
        args.skip_inference = True
    with tse_inference_artifacts.exclusive_output_run(args.output_dir) as out_dir:
        return _main_locked(args, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
