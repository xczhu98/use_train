"""Config regression tests: load and validate every training YAML under configs/."""

from __future__ import annotations
import pytest

import copy
import inspect
import json
import re
import unittest
from pathlib import Path

from tasks.aec.data.dynamic_aec_build import resolve_aec_dataset_kwargs
from tasks.aec.trainer.aec_trainer import AECTrainer
from tasks.aec.trainer.entry import AECTrainingEntry
from core.data.config_flags import resolve_simulation_cfg_strict
from core.data.constants import _SUPPORTED_DATA_MODES, normalize_data_mode
from core.data.dataset_build import (
    expand_simulation_cfg_layer_a,
    expand_simulation_cfg_se_pase_dataset_keys,
    expand_simulation_cfg_tse_dataset_keys,
    merge_validation_section,
    validate_layer_a_runtime_config,
)
from core.data.simulation_config_contract import validate_simulation_config_contract
from core.data.simulation_configs import SimulationConfigs
from core.trainer.base_trainer import BaseTrainer
from core.trainer.metrics_trainer import MetricsCapableTrainer
from core.utils.config import load_merged_config
from core.utils.registry import MODEL_REGISTRY
from core.trainer.entry import detect_task
from tasks.se.models.base_model import BaseSEModel
from tasks.se.trainer.entry import SETrainingEntry
from tasks.se.trainer.se_trainer import SETrainer
from tasks.ss.trainer.entry import SSTrainingEntry
from tasks.tse.trainer.entry import TSETrainingEntry
from tasks.tse.trainer.tse_trainer import TSETrainer
from tasks.ssl.trainer.entry import SSLTrainingEntry
from tools.verify.check_config_comments import check_file, iter_config_paths

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
TSE_TEMPLATE_CONFIGS = {
    Path("tse/dynamic/tse_bsrnn_spk.yaml"),
}
# 动态 SE/TSE 训练配置必须全部按 runtime 顺序 strict 解析 simulation_cfg；
# 新增/重命名配置必须显式登记，失败信息直接指出差异配置。
STRICT_SIMULATION_PARSE_CONFIGS = {
    "se/urgent2026/dpdfnet2_causal_urgent26_dynamic_hybrid.yaml",
    "se/urgent2026/gan/gtcrn_metricdiscriminator_dynamic.yaml",
    "se/urgent2026/gan/mambattention_metricgan_dynamic.yaml",
    "se/urgent2026/gan/mossformer_gan_metricgan_dynamic.yaml",
    "se/urgent2026/gan/mp_senet_metricgan_dynamic.yaml",
    "se/urgent2026/gan/rwsa_mamba_unet_metricgan_dynamic.yaml",
    "se/urgent2026/pase/gap_urgenet/gap_urgenet_phase1_predictor.yaml",
    "se/urgent2026/pase/gap_urgenet/gap_urgenet_phase2_postnet_gan.yaml",
    "se/urgent2026/pase/gap_urgenet/gap_urgenet_phase2_postnet_finetune.yaml",
    "se/urgent2026/pase/phase1_vocoder_single.yaml",
    "se/urgent2026/pase/phase2_dewavlm_distill.yaml",
    "se/urgent2026/pase/phase3_vocoder_dual_gan.yaml",
    "se/urgent2026/pase/stupase_phase0_dewavlm_r_distill.yaml",
    "se/urgent2026/pase/stupase_phase1_melvocoder_gan.yaml",
    "se/urgent2026/pase/stupase_phase2_flow.yaml",
    "se/urgent2026/pase/unipase/unipase_phase1_dewavlm_omni.yaml",
    "se/urgent2026/pase/unipase/unipase_phase2_vocoder_gan.yaml",
    "se/urgent2026/pase/unipase/unipase_phase3_adapter_gan.yaml",
    "se/urgent2026/pase/unipase/unipase_phase4_postnet_gan.yaml",
    "se/spatial/spatialnet_dynamic.yaml",
    "se/spatial/online_spatialnet_dynamic_mhsa.yaml",
    "se/spatial/online_spatialnet_dynamic_ret.yaml",
    "se/spatial/online_spatialnet_dynamic_mamba.yaml",
    "se/spatial/online_spatialnet_dynamic_mhsa_lf32s.yaml",
    "se/spatial/online_spatialnet_dynamic_mhsa_lf10s_36e.yaml",
    "se/spatial/online_spatialnet_dynamic_ret_lf32s.yaml",
    "se/spatial/online_spatialnet_dynamic_mamba_lf32s.yaml",
    "tse/dynamic/tse_bsrnn_spk.yaml",
}
DNS3_ISOLATED_CONFIG = Path("se/dns_challenge3/gtcrn_causal_offline.yaml")
DNS3_ISOLATED_EXP_PATH = "exp/se/dns_challenge3_gtcrn_causal_offline"


@pytest.mark.parametrize("field", ("n_fft", "hop_length", "win_length"))
@pytest.mark.parametrize("invalid_value", (True, False, 512.0, 511.5, "512", 0, -1))
def test_stft_config_build_rejects_non_positive_exact_integers(
    field: str, invalid_value
):
    """STFT 构建必须拒绝 bool、浮点数、数字字符串和非正整数。"""
    from core.stft import build_stft_istft

    stft_cfg = {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": "hann",
        "center": True,
    }
    stft_cfg[field] = invalid_value

    with pytest.raises(ValueError, match=rf"stft\.{field} must be an exact positive int"):
        build_stft_istft(stft_cfg)


@pytest.mark.parametrize("invalid_center", (0, 1, "true", None, 1.0))
def test_stft_config_build_rejects_non_bool_center(invalid_center):
    """center 只接受精确 bool，不接受等价真值。"""
    from core.stft import build_stft_istft

    with pytest.raises(ValueError, match=r"stft\.center must be a bool"):
        build_stft_istft(
            {
                "n_fft": 512,
                "hop_length": 128,
                "window": "hann",
                "center": invalid_center,
            }
        )


@pytest.mark.parametrize("invalid_window", ("", "blackman", None, 1, True))
def test_stft_config_build_rejects_invalid_window(invalid_window):
    """window 只允许框架明确支持的名称。"""
    from core.stft import build_stft_istft

    with pytest.raises(ValueError, match=r"stft\.window must be one of"):
        build_stft_istft(
            {
                "n_fft": 512,
                "hop_length": 128,
                "window": invalid_window,
            }
        )


def test_stft_config_keeps_effective_defaults_and_accepts_legal_short_window():
    """win_length 默认等于 n_fft，但 center=True 时允许 PyTorch 可重建的短窗。"""
    from core.stft import resolve_stft_config

    assert resolve_stft_config({"n_fft": 512, "hop_length": 128}) == {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": "hann",
        "center": True,
        "normalized": False,
    }
    assert resolve_stft_config(
        {"n_fft": 512, "hop_length": 128, "win_length": 400}
    )["win_length"] == 400


@pytest.mark.parametrize(
    "stft_cfg",
    [
        {"n_fft": 512, "hop_length": 401, "win_length": 400},
        {"n_fft": 400, "hop_length": 128, "win_length": 512},
    ],
)
def test_stft_config_rejects_invalid_ordered_geometry(stft_cfg):
    """共享 STFT/iSTFT 必须满足 hop_length <= win_length <= n_fft。"""
    from core.stft import resolve_stft_config

    with pytest.raises(ValueError, match=r"hop_length <= win_length <= n_fft"):
        resolve_stft_config(stft_cfg)


@pytest.mark.parametrize(
    ("stft_cfg", "accepted"),
    [
        (
            {
                "n_fft": 64,
                "hop_length": 32,
                "win_length": 64,
                "window": "hann",
                "center": False,
            },
            False,
        ),
        (
            {
                "n_fft": 64,
                "hop_length": 32,
                "win_length": 64,
                "window": "hamming",
                "center": False,
            },
            True,
        ),
        (
            {
                "n_fft": 64,
                "hop_length": 16,
                "win_length": 48,
                "window": "hamming",
                "center": False,
            },
            False,
        ),
        (
            {
                "n_fft": 64,
                "hop_length": 64,
                "win_length": 64,
                "window": "hann",
                "center": True,
            },
            False,
        ),
        (
            {
                "n_fft": 64,
                "hop_length": 64,
                "win_length": 64,
                "window": "hamming",
                "center": True,
            },
            True,
        ),
    ],
)
def test_stft_config_matches_torch_istft_nola_contract(stft_cfg, accepted):
    """center/window 组合按 torch.istft 的 NOLA 边界行为 fail-fast。"""
    from core.stft import resolve_stft_config

    if accepted:
        assert resolve_stft_config(stft_cfg) == {**stft_cfg, "normalized": False}
    else:
        with pytest.raises(ValueError, match="NOLA"):
            resolve_stft_config(stft_cfg)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (("n_fft", 512.5), ("hop_length", 128.5), ("win_length", 512.5)),
)
def test_loss_build_fails_before_fractional_stft_values_can_be_truncated(
    field: str, invalid_value: float
):
    """loss 构建必须复用严格解析，不能通过 int() 截断小数。"""
    from core.losses.build import build_loss

    stft_cfg = {"n_fft": 512, "hop_length": 128, "win_length": 512}
    stft_cfg[field] = invalid_value
    with pytest.raises(ValueError, match=rf"stft\.{field} must be an exact positive int"):
        build_loss({"type": "SISNRLoss"}, stft_cfg=stft_cfg)


def test_noise_manifest_admission_contract_versions_and_verification_fields():
    """strict 噪声清单只接受 schema v6 与固定 top-5 C3 准入契约。"""
    from core.data.noise_admission import (
        NOISE_MANIFEST_SCHEMA_VERSION,
        REQUIRED_ARTIFACT_FINGERPRINTS,
        load_canonical_voice_policy,
        validate_noise_manifest_records,
    )
    from core.data.noise_admission_policy import (
        ADMISSION_POLICY_VERSION,
        CLASSIFIER_LABEL_COUNT,
        EVIDENCE_CONTRACT_VERSION,
        SCORE_CONTRACT,
    )

    assert NOISE_MANIFEST_SCHEMA_VERSION == 6
    assert ADMISSION_POLICY_VERSION == 4
    assert EVIDENCE_CONTRACT_VERSION == 5
    assert REQUIRED_ARTIFACT_FINGERPRINTS == {
        "voice_policy_sha256",
        "audiotag_label_dict_sha256",
        "audiotag_checkpoint_sha256",
        "qwen_model_sha256",
        "c1_prompt_sha256",
    }

    policy, policy_sha = load_canonical_voice_policy()
    artifacts = {
        "voice_policy_sha256": policy_sha,
        "audiotag_label_dict_sha256": policy.label_space_sha256,
        "audiotag_checkpoint_sha256": "2" * 64,
        "qwen_model_sha256": "3" * 64,
        "c1_prompt_sha256": "4" * 64,
    }
    noise_mids = [
        mid
        for _, mid in sorted(policy.index_to_mid.items())
        if policy.class_of(mid) == "noise"
    ][:5]
    top5_hits = [
        {
            "index": policy.mid_to_index[mid],
            "mid": mid,
            "label": policy.label_of(mid),
            "kind": "noise",
            "prob": 1.0 - rank * 0.1,
        }
        for rank, mid in enumerate(noise_mids)
    ]
    verification = {
        "c1_status": "passed",
        "c1_intelligibility": "noise",
        "c2_status": "passed",
        "c3_status": "passed",
        "d2_status": "passed",
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "score_contract": SCORE_CONTRACT,
        "class_count": CLASSIFIER_LABEL_COUNT,
        "label_space_sha256": policy.label_space_sha256,
        "artifact_fingerprints": artifacts,
        "top5_hits": top5_hits,
    }
    assert set(verification) == {
        "c1_status",
        "c1_intelligibility",
        "c2_status",
        "c3_status",
        "d2_status",
        "evidence_contract_version",
        "score_contract",
        "class_count",
        "label_space_sha256",
        "artifact_fingerprints",
        "top5_hits",
    }

    balance_mid = noise_mids[0]
    balance_label = policy.label_of(balance_mid)
    record = {
        "uid": "noise-contract",
        "fs": 16000,
        "length": 16000,
        "path": "~/placeholder.wav",
        "manifest_schema_version": NOISE_MANIFEST_SCHEMA_VERSION,
        "balance_category_mid": balance_mid,
        "balance_category": balance_label,
        "balance_category_source": "noise_top5_hit",
        "category_name": balance_label,
        "category_id": 0,
        "admission_allowed": True,
        "admission_decision": "accept_noise",
        "admission_reason_codes": ["admitted_noise"],
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "admission_policy_sha256": policy_sha,
        "verification": verification,
        "dataset_id": "contract-fixture",
    }
    validate_noise_manifest_records(
        {record["uid"]: record}, source_path="tests/test_configs.py contract fixture"
    )

    legacy_schema = copy.deepcopy(record)
    legacy_schema["manifest_schema_version"] = 5
    with pytest.raises(ValueError, match="manifest_schema_version must be exact int 6"):
        validate_noise_manifest_records(
            {legacy_schema["uid"]: legacy_schema}, source_path="legacy schema fixture"
        )

    legacy_evidence = copy.deepcopy(record)
    legacy_evidence["verification"]["evidence_contract_version"] = 4
    with pytest.raises(
        ValueError, match="verification.evidence_contract_version must be exact int 5"
    ):
        validate_noise_manifest_records(
            {legacy_evidence["uid"]: legacy_evidence},
            source_path="legacy evidence fixture",
        )

    legacy_calibration = copy.deepcopy(record)
    legacy_calibration["admission_calibration_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="invalid manifest fields"):
        validate_noise_manifest_records(
            {legacy_calibration["uid"]: legacy_calibration},
            source_path="legacy calibration fixture",
        )

    legacy_content_kind = copy.deepcopy(record)
    legacy_content_kind["content_kind"] = "noise"
    with pytest.raises(ValueError, match="invalid manifest fields"):
        validate_noise_manifest_records(
            {legacy_content_kind["uid"]: legacy_content_kind},
            source_path="legacy content_kind fixture",
        )

    legacy_c4 = copy.deepcopy(record)
    legacy_c4["verification"].pop("c3_status")
    legacy_c4["verification"]["c4_status"] = "passed"
    with pytest.raises(ValueError, match="invalid verification fields"):
        validate_noise_manifest_records(
            {legacy_c4["uid"]: legacy_c4}, source_path="legacy C4 fixture"
        )


def test_noise_filter_canonical_config_freezes_v6_fixed_top5_contract():
    """canonical policy 与 Stage C 配置冻结无阈值固定 top-5 契约。"""
    from core.data.noise_admission_policy import (
        AUDIOTAG_TOP_K,
        CLASSIFIER_LABEL_COUNT,
        EVIDENCE_CONTRACT_VERSION,
        MID_CLASSES,
        SCORE_CONTRACT,
        load_admission_policy,
    )
    from tools.noise_filter_pipeline.core.pipeline_config import UnifiedPipelineConfig
    from tools.noise_filter_pipeline.core.stage_d_noise_manifest_contract import (
        GENERATOR_VERSION,
    )
    from tools.noise_filter_pipeline.core.stage_orchestrator import _MARKER_VERSION
    from tools.noise_filter_pipeline.scripts.promote_run_to_final import (
        _EXPECTED_PARSER_VERSION,
    )

    policy = load_admission_policy()
    canonical_mids = set(policy.mid_to_index)

    assert len(canonical_mids) == CLASSIFIER_LABEL_COUNT == 527
    assert EVIDENCE_CONTRACT_VERSION == 5
    assert AUDIOTAG_TOP_K == 5
    assert SCORE_CONTRACT == "audioset_527_independent_sigmoid_v1"
    assert set(policy.mid_to_class) == canonical_mids
    assert set(policy.mid_to_class.values()) == set(MID_CLASSES)

    expected_classes = {
        "/m/0261r1": "speech",  # AudioSet Babbling
        "/m/07rkbfh": "babble",  # Chatter
        "/m/07qfr4h": "babble",  # Hubbub, speech noise, speech babble
        "/m/03qtwd": "babble",  # Crowd
        "/m/04rlf": "noise",  # Music
        "/m/025td0t": "noise",  # Background music
        "/t/dd00127": "noise",  # Inside, public space
        "/m/07c52": "speech",  # Television
        "/m/06bz3": "speech",  # Radio
        "/m/07hvw1": "speech",  # Field recording
    }
    assert {mid: policy.class_of(mid) for mid in expected_classes} == expected_classes

    pipeline_path = (
        CONFIG_ROOT.parent / "tools/noise_filter_pipeline/config/pipeline.yaml"
    )
    pipeline = UnifiedPipelineConfig.from_yaml(str(pipeline_path), active_stages=[])
    assert pipeline.pipeline_version == "4.1.0"
    assert set(pipeline.stage_c) == {"c1", "c2", "c3"}
    c2 = pipeline.stage_c["c2"]
    for removed_key in (
        "output_label_key",
        "topk_output_key",
        "topk_keep_k",
        "policy_path",
        "calibration_path",
    ):
        assert removed_key not in c2
    assert set(pipeline.stage_c["c3"]) == {"enable", "policy_path"}
    from core.data.noise_manifest_adapter import NOISE_SOURCE_PARSER_VERSION

    assert NOISE_SOURCE_PARSER_VERSION == 10
    assert _EXPECTED_PARSER_VERSION == NOISE_SOURCE_PARSER_VERSION
    canonical_docs = (
        "README.md",
        "README_CN.md",
        "tasks/aec/README.md",
        "tasks/tse/README.md",
        "configs/README.md",
        "configs/se/README.md",
        "configs/se/README_CN.md",
        "configs/aec/microsoft_challenge/README.md",
        "docs/guides/simulation/Simulation_Flow_Logic.md",
        "tools/noise_filter_pipeline/README.md",
        "tools/noise_filter_pipeline/assets/noise_filter_pipeline_overview.md",
        "third_party_scripts/README.md",
        "third_party_scripts/Qwen2.5-Omni-7B/README.md",
    )
    for relative_path in canonical_docs:
        text = (CONFIG_ROOT.parent / relative_path).read_text(encoding="utf-8")
        assert "parser v10" in text, relative_path
        assert "parser v9" not in text, relative_path
    assert GENERATOR_VERSION == "6.1"
    assert _MARKER_VERSION == 4


TSE_TEMPLATE_NULL_FIELDS = (
    (("data", "train", "speech_scp"), r"data\.train\.speech_scp"),
    (
        ("data", "train", "spk2enroll_json"),
        r"data\.train\.spk2enroll_json",
    ),
    (("data", "validation", "speech_scp"), r"data\.validation\.speech_scp"),
)


def _nested_value(cfg: dict, keys: tuple[str, ...]):
    value = cfg
    for key in keys:
        value = value[key]
    return value


def _set_nested_value(cfg: dict, keys: tuple[str, ...], value) -> None:
    parent = cfg
    for key in keys[:-1]:
        parent = parent[key]
    parent[keys[-1]] = value


def _hydrate_tse_template(cfg: dict) -> dict:
    """只替换三个 strict JSONL/nested enrollment 数据占位，不访问文件系统。"""
    hydrated = copy.deepcopy(cfg)
    replacements = (
        "placeholder/train.jsonl",
        "placeholder/spk2enroll_nested.json",
        "placeholder/val.jsonl",
    )
    for (keys, _), value in zip(TSE_TEMPLATE_NULL_FIELDS, replacements):
        _set_nested_value(hydrated, keys, value)
    return hydrated


def _hydrate_tse_strict_template(cfg: dict) -> dict:
    """切换为 strict JSONL manifest。"""
    hydrated = copy.deepcopy(cfg)
    hydrated["data"]["train"]["speech_scp"] = "placeholder/train.jsonl"
    hydrated["data"]["train"]["spk2enroll_json"] = (
        "placeholder/spk2enroll_nested.json"
    )
    hydrated["data"]["validation"]["speech_scp"] = "placeholder/val.jsonl"
    return hydrated


def _validate_template_metadata(rel: Path, meta: dict) -> None:
    if "training_template" in meta:
        raise AssertionError(
            "training templates are identified only by the exact test allowlist"
        )
    if rel in TSE_TEMPLATE_CONFIGS and meta.get("inference_only") is True:
        raise AssertionError("allowlisted TSE training templates cannot be inference_only")


_RESOURCE_KEY_MARKERS = (
    "_scp",
    "_json",
    "samples_list",
    "cues_yaml",
    "speech_length_file",
    "pretrained",
)


def _is_resource_key(key: str) -> bool:
    key_text = key.lower()
    if key_text == "pretrained_prefix":
        return False
    return (
        any(marker in key_text for marker in _RESOURCE_KEY_MARKERS)
        or key_text.endswith(("checkpoint", "checkpoint_path", "ckpt", "ckpt_path"))
    )


def _iter_resource_values(node, path=()):
    """递归枚举配置中由键名标识的数据清单和权重路径。"""
    if isinstance(node, dict):
        for key, value in node.items():
            key_text = str(key)
            if _is_resource_key(key_text):
                yield path + (key_text,), value
            yield from _iter_resource_values(value, path + (key_text,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_resource_values(value, path + (str(index),))


def test_detect_task_reports_channel_mode():
    """启动报告须能区分 mono 与 spatial（含 C 与阵列 profile）。"""
    from core.trainer.entry import _detect_channels

    # mono dynamic：无 spatial 开关。
    assert _detect_channels(
        {"mode": "dynamic", "train": {"simulation_cfg": {"scene": {}}}},
        {"type": "GTCRN"},
    ) == "mono"
    # 分层 spatial（8 个 SE spatial 配方的结构）。
    assert _detect_channels(
        {
            "mode": "dynamic",
            "train": {
                "simulation_cfg": {
                    "spatial": {
                        "enabled": True,
                        "geometry": {"profile": "circular_8ch_r80mm"},
                    }
                }
            },
        },
        {"type": "OnlineSpatialNet_SE", "args": {"n_channel": 8}},
    ) == "spatial (C=8, circular_8ch_r80mm)"
    # 平面 spatial_enabled + 无模型 n_channel。
    assert _detect_channels(
        {"mode": "dynamic", "train": {"simulation_cfg": {"spatial_enabled": True}}},
        {"type": "M"},
    ) == "spatial"
    # offline spatial：split 级 spatial_enabled（mc_sim 结构）。
    assert _detect_channels(
        {
            "mode": "offline",
            "train": {"spatial_enabled": True, "array_profile": "linear_2ch_80mm"},
        },
        {"type": "SpatialNet_SE", "args": {"n_channel": 2}},
    ) == "spatial (C=2, linear_2ch_80mm)"
    # 全量配置端到端：36e 配方检出 spatial。
    cfg = load_merged_config(
        str(CONFIG_ROOT / "se/spatial/online_spatialnet_dynamic_mhsa_lf10s_36e.yaml")
    )
    assert detect_task(cfg).channels == "spatial (C=8, circular_8ch_r80mm)"
    # spatial 背景三门 nominal 值进入数据分布摘要。
    from core.trainer.entry import _spatial_background_gate_nominals

    assert _spatial_background_gate_nominals(cfg["data"]) == [
        "point_source=80.0%",
        "wind_background=50.0%",
        "colored=20.0%",
    ]
    # mono 配方无 spatial 门输出。
    mono_cfg = load_merged_config(
        str(CONFIG_ROOT / "se/voicebank_demand/gtcrn_causal_offline.yaml")
    )
    assert _spatial_background_gate_nominals(mono_cfg["data"]) == []


def _validate_inference_only_contract(rel: Path, cfg: dict) -> None:
    """静态校验纯推理配置，不实例化模型或访问 checkpoint。"""
    task_info = detect_task(cfg)
    expected_task = task_info.task

    model_cfg = cfg.get("model") or {}
    model_type = model_cfg.get("type")
    assert isinstance(model_type, str) and model_type, f"missing model.type: {rel}"
    if expected_task == "tse":
        from tasks.tse.models import register_tse_models

        register_tse_models(only=model_type)
    elif expected_task == "aec":
        from tasks.aec.models import register_aec_models

        register_aec_models(only=model_type)
    elif expected_task == "ss":
        from tasks.ss.models import register_ss_models

        register_ss_models(only=model_type)
    else:
        import tasks.se.models  # noqa: F401

        MODEL_REGISTRY.ensure_registered(model_type)
    model_cls = MODEL_REGISTRY.get(model_type)

    init_signature = inspect.signature(model_cls.__init__)
    init_params = init_signature.parameters
    for arg_name in (model_cfg.get("args") or {}):
        assert arg_name in init_params, (
            f"model arg {arg_name!r} is not explicitly declared by "
            f"{model_type}: {rel}"
        )

    stft_cfg = cfg.get("stft")
    if expected_task == "tse":
        assert stft_cfg is not None, f"TSE inference config requires stft: {rel}"
        assert set(("n_fft", "hop_length", "win_length", "window")) <= set(stft_cfg), (
            f"incomplete TSE inference stft contract: {rel}"
        )
        forward_params = list(inspect.signature(model_cls.forward).parameters)
        assert forward_params[:1] == ["self"] and forward_params[1:2] in (
            ["batch"],
            ["data"],
        ), f"TSE inference model must use dict batch input: {rel}"
    elif expected_task == "se":
        assert cfg.get("meta", {}).get("allow_no_stft") is True, str(rel)
        assert stft_cfg is None, f"waveform-native inference must not define stft: {rel}"
        assert issubclass(model_cls, BaseSEModel), str(rel)
        fs = inspect.signature(model_cls.forward).parameters["fs"]
        assert fs.kind is inspect.Parameter.KEYWORD_ONLY, str(rel)


def _discover_config_files(root: Path = CONFIG_ROOT) -> tuple[Path, ...]:
    """稳定发现 YAML 两种标准扩展名，且不依赖文件系统遍历顺序。"""
    return tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _config_header(path: Path) -> str:
    """返回首个 YAML key 前的文件头注释。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    first_key = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith("#")
        ),
        len(lines),
    )
    return "\n".join(lines[:first_key])


_REQUIRED_CONFIG_HEADER_FIELDS = (
    "Params",
    "MACs",
    "FLOPs",
    "Trainer MOS boundary",
)


def _validate_config_header_contract(path: Path) -> None:
    """配置头保留四字段人工摘要，不承载结构化 audit schema。"""
    header = _config_header(path)
    summary_lines = [
        line.lstrip()[1:].strip()
        for line in header.splitlines()
        if line.lstrip().startswith("#") and line.lstrip()[1:].strip()
    ]
    assert summary_lines, f"{path}: 首个 YAML key 前必须保留自由文本配置摘要"
    assert "[AUDIT]" not in header, (
        f"{path}: 结构化 audit JSON 应由 audit_metrics.py 输出，不得写入配置头"
    )
    missing_fields = [
        field for field in _REQUIRED_CONFIG_HEADER_FIELDS if field not in header
    ]
    assert not missing_fields, (
        f"{path}: 配置头缺少人工摘要字段: {', '.join(missing_fields)}"
    )


def _strict_parse_runtime_simulation_cfg(cfg: dict) -> int:
    """按 SE/TSE runtime 的继承与专属键提升顺序 strict 解析 simulation_cfg。"""
    if (cfg.get("meta") or {}).get("evaluation_only") is True:
        return 0
    data_cfg = cfg.get("data") or {}
    mode = normalize_data_mode(data_cfg.get("mode"))
    if mode.storage != "dynamic" or mode.task not in {"se", "tse"}:
        return 0
    train_section = data_cfg.get("train")
    if not train_section:
        return 0

    validate_simulation_config_contract(
        task=mode.task,
        split="train",
        train_section=train_section,
    )
    sections = [copy.deepcopy(train_section)]
    validation = data_cfg.get("validation")
    if validation:
        val_mode = validation.get("mode")
        val_storage = (
            normalize_data_mode(val_mode).storage if val_mode is not None else None
        )
        if val_storage == "dynamic" or validation.get("use_dynamic") is True:
            validate_simulation_config_contract(
                task=mode.task,
                split="val",
                train_section=train_section,
                validation_section=validation,
            )
            sections.append(merge_validation_section(train_section, validation))

    parsed = 0
    for section in sections:
        expand_simulation_cfg_layer_a(section)
        if mode.task == "se":
            expand_simulation_cfg_se_pase_dataset_keys(section)
        else:
            expand_simulation_cfg_tse_dataset_keys(section)
        validate_layer_a_runtime_config(
            section,
            config_path=f"config {mode.task} dynamic section",
        )
        simulation_cfg = section.get("simulation_cfg")
        assert isinstance(simulation_cfg, dict), "dynamic runtime requires simulation_cfg"
        assert resolve_simulation_cfg_strict(cfg) is True
        SimulationConfigs.from_dict(simulation_cfg, strict=True)
        parsed += 1
    return parsed


def test_config_discovery_accepts_yml_and_sorts_by_relative_path(tmp_path: Path):
    """发现器必须真实覆盖 ``.yml``，且过滤其他后缀。"""
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "z.yml").write_text("meta: {}\n", encoding="utf-8")
    (nested / "a.yaml").write_text("meta: {}\n", encoding="utf-8")
    (nested / "ignored.txt").write_text("meta: {}\n", encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in _discover_config_files(tmp_path)] == [
        "nested/a.yaml",
        "z.yml",
    ]


def test_data_mode_contract_covers_five_tasks_and_nine_modes():
    """配置路由必须覆盖 SE/AEC/TSE/SS/SSL；SSL 仅保留 dynamic 仿真模式。"""
    expected = {
        "dynamic": "se",
        "offline": "se",
        "dynamic_aec": "aec",
        "offline_aec": "aec",
        "dynamic_tse": "tse",
        "offline_tse": "tse",
        "dynamic_ss": "ss",
        "offline_ss": "ss",
        "dynamic_ssl": "ssl",
    }
    assert set(_SUPPORTED_DATA_MODES) == set(expected)
    assert {
        mode: normalize_data_mode(mode).task for mode in expected
    } == expected
    assert {normalize_data_mode(mode).storage for mode in expected} == {
        "dynamic",
        "offline",
    }
    assert {normalize_data_mode(mode).task for mode in expected} == {
        "se",
        "aec",
        "tse",
        "ss",
        "ssl",
    }


def test_public_docs_keep_mode_spatial_and_layer_f_contracts():
    """公开入口文档必须与五任务、空间通道和 CPU Layer F 合同同步。"""
    modes = {
        "dynamic",
        "offline",
        "dynamic_aec",
        "offline_aec",
        "dynamic_tse",
        "offline_tse",
        "dynamic_ss",
        "offline_ss",
        "dynamic_ssl",
    }
    root_docs = (
        CONFIG_ROOT.parent / "README.md",
        CONFIG_ROOT.parent / "README_CN.md",
    )
    for path in root_docs:
        text = path.read_text(encoding="utf-8")
        documented_modes = set(
            re.findall(r"\bdynamic(?:_[a-z]+)?\b|\boffline(?:_[a-z]+)?\b", text)
        )
        assert modes <= documented_modes
        assert all(task in text for task in ("SE", "AEC", "TSE", "SS", "SSL"))

    config_text = (CONFIG_ROOT / "README.md").read_text(encoding="utf-8")
    for profile in ("linear_2ch_80mm", "linear_4ch_40mm", "linear_6ch_40mm", "circular_8ch_r80mm"):
        assert profile in config_text
    for fragment in (
        "waveform target/output 保持 `[T]`",
        "AEC `farend` 与 TSE `enrollment` 始终为单通道 `[T]`",
        "`C=1` 继续走 legacy mono 路径",
        "Layer F 是 Dataset 内 CPU-only",
        "TSE/SS/SSL 多源相对窗口",
        "CanvasSpec",
        "prob_wind_noise_background",
        "wind_background_prob",
    ):
        assert fragment in config_text

    repo_root = CONFIG_ROOT.parent
    ssl_text = (repo_root / "docs/guides/simulation/Simulation_Flow_Logic.md").read_text(
        encoding="utf-8"
    )
    for fragment in ("dynamic_ssl", "direct-path", "scene manifest"):
        assert fragment in ssl_text

    active_docs = {
        "configs": config_text,
        "simulation": ssl_text,
        "multi_source": (repo_root / "docs/guides/simulation/Multi_Source_Simulation.md").read_text(
            encoding="utf-8"
        ),
        "ssl": (repo_root / "tasks/ssl/README.md").read_text(encoding="utf-8"),
        "diagram_index": (repo_root / "docs/diagrams/README.md").read_text(encoding="utf-8"),
    }
    for name, text in active_docs.items():
        assert "五任务" in text or "五个任务" in text or "SE/AEC/TSE/SS/SSL" in text, name
        assert "dynamic_ssl" in text or "TSE/SS/SSL" in text or "SE/AEC/TSE/SS/SSL" in text, name

    canonical_dot = (repo_root / "docs/diagrams/se_aec_tse_simulation_layers_current.dot").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "九种 data.mode",
        "dynamic_ss",
        "dynamic_ssl",
        "SSBatch",
        "SSLBatch",
        "CanvasSpec",
    ):
        assert fragment in canonical_dot


def test_active_documentation_does_not_link_removed_filter_artifacts():
    """活动入口不得继续链接已删除的 clean/noise filter 文档或图源。"""
    active_docs = (
        CONFIG_ROOT.parent / "README.md",
        CONFIG_ROOT.parent / "README_CN.md",
        CONFIG_ROOT / "README.md",
        CONFIG_ROOT.parent / "docs/README.md",
        CONFIG_ROOT.parent / "docs/guides/README.md",
        CONFIG_ROOT.parent / "docs/diagrams/README.md",
        CONFIG_ROOT.parent / "tools/noise_filter_pipeline/README.md",
        CONFIG_ROOT.parent / "tools/noise_filter_pipeline/assets/noise_filter_pipeline_overview.md",
    )
    removed_refs = (
        "clean_filter/",
        "guides/noise_filter/",
        "noise_filter_admission_pipeline.dot",
        "balanced_noise_sampling_flow",
        "balanced_speech_sampling_flow",
    )
    for path in active_docs:
        text = path.read_text(encoding="utf-8")
        for reference in removed_refs:
            assert reference not in text, f"{path}: stale reference {reference!r}"


def test_config_header_contract_accepts_required_manual_summary_fields(tmp_path: Path):
    """配置头必须在首个 YAML key 前保留四个人工摘要标签。"""
    accepted = tmp_path / "accepted.yaml"
    accepted.write_text(
        "# VoiceBank GTCRN 离线训练摘要：16 kHz、8 秒片段。\n"
        "# Params: 0.0482M total / 0.0237M trainable\n"
        "# MACs: 0.0341G @ 1 s / 16 kHz\n"
        "# FLOPs: not measured\n"
        "# Trainer MOS boundary: Trainer 不计算 MOS\n"
        "meta: {}\n",
        encoding="utf-8",
    )

    _validate_config_header_contract(accepted)


@pytest.mark.parametrize("missing_field", _REQUIRED_CONFIG_HEADER_FIELDS)
def test_config_header_contract_rejects_each_missing_manual_field(
    tmp_path: Path, missing_field: str
):
    fields = {
        "Params": "# Params: not measured\n",
        "MACs": "# MACs: not measured\n",
        "FLOPs": "# FLOPs: not measured\n",
        "Trainer MOS boundary": "# Trainer MOS boundary: offline only\n",
    }
    config = tmp_path / f"missing-{missing_field.replace(' ', '-')}.yaml"
    config.write_text(
        "# 人工摘要\n"
        + "".join(value for field, value in fields.items() if field != missing_field)
        + "meta: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match=re.escape(missing_field)):
        _validate_config_header_contract(config)


def test_config_header_contract_requires_summary_before_first_key(tmp_path: Path):
    config = tmp_path / "missing-header.yaml"
    config.write_text("meta: {}\n# 这里已经太晚\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="自由文本配置摘要"):
        _validate_config_header_contract(config)


def test_config_header_contract_keeps_structured_audit_json_out_of_yaml(tmp_path: Path):
    config = tmp_path / "embedded-audit.yaml"
    config.write_text(
        '# [AUDIT] {"status": "ok", "params": {"total": 1}}\nmeta: {}\n',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="audit JSON"):
        _validate_config_header_contract(config)


def test_yaml_comment_contract():
    """configs/ 下全部 YAML 必须满足注释规范（configs/README.md「YAML 注释规范」）。"""
    config_files = list(iter_config_paths(CONFIG_ROOT))
    assert config_files, "configs/ 下未发现 YAML 配置"
    violations = [
        f"{path.relative_to(CONFIG_ROOT.parent)}:{violation}"
        for path in config_files
        for violation in check_file(path)
    ]
    assert not violations, "YAML 注释规范违规：\n" + "\n".join(violations)


class TestConfigLoading(unittest.TestCase):
    """Smoke-test that every committed training config is loadable and valid."""

    def test_config_discovery_is_stable_and_covers_yaml_and_yml(self):
        """配置清单非空、顺序稳定，并同时覆盖 ``.yaml`` / ``.yml``。"""
        config_files = _discover_config_files()
        self.assertTrue(config_files)
        self.assertEqual(config_files, _discover_config_files())
        self.assertEqual(
            [path.relative_to(CONFIG_ROOT).as_posix() for path in config_files],
            sorted(path.relative_to(CONFIG_ROOT).as_posix() for path in config_files),
        )

    def test_all_config_headers_keep_four_field_summary_outside_audit_schema(self):
        """所有配置头保留四字段人工摘要；完整合同只属于 audit JSON。"""
        config_files = _discover_config_files()
        self.assertTrue(config_files, "config discovery must not be empty")
        for cfg_path in config_files:
            with self.subTest(config=str(cfg_path.relative_to(CONFIG_ROOT))):
                _validate_config_header_contract(cfg_path)

    def test_se_aec_offline_sections_do_not_use_retired_exact_keys(self):
        """现役 SE/AEC offline split 禁止退役 sidecar 与 section SR alias。"""
        retired = {"utt2fs", "speech_length", "sample_rate"}
        for cfg_path in _discover_config_files():
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            data_cfg = cfg.get("data") or {}
            mode = normalize_data_mode(data_cfg.get("mode"))
            if mode.task not in {"se", "aec"}:
                continue
            for split in ("train", "validation"):
                section = data_cfg.get(split)
                if not isinstance(section, dict):
                    continue
                is_offline = mode.storage == "offline"
                if split == "validation" and mode.storage == "dynamic":
                    val_mode = section.get("mode")
                    is_offline = not (
                        section.get("use_dynamic") is True
                        or (
                            val_mode is not None
                            and normalize_data_mode(val_mode).storage == "dynamic"
                        )
                    )
                if not is_offline:
                    continue
                with self.subTest(config=str(rel), split=split):
                    self.assertFalse(
                        retired & set(section),
                        f"{rel}: data.{split} contains retired exact keys "
                        f"{sorted(retired & set(section))}",
                    )

    def test_dynamic_configs_pass_spatial_capability_gate_at_load_time(self):
        """每份 dynamic 配置的 train simulation_cfg 在配置加载期即须通过 spatial
        capability 门禁（capabilities.SPATIAL_FEATURE_CAPABILITIES 为单一事实源），
        避免违规配置拖到 DDP loader startup preflight 才失败。"""
        from core.data.simulation.spatial_background import (
            validate_spatial_background_capability,
        )
        from core.data.simulation_configs import SimulationConfigs
        from core.data.simulation_schema import normalize_simulation_cfg

        checked = 0
        for cfg_path in _discover_config_files():
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            data_cfg = cfg.get("data") or {}
            mode = normalize_data_mode(data_cfg.get("mode"))
            if mode.storage != "dynamic":
                continue
            sim_cfg = (data_cfg.get("train") or {}).get("simulation_cfg")
            if not sim_cfg:
                continue
            with self.subTest(config=str(rel)):
                # strict=False：任务扩展键（AEC echo/TSE cue/PASE preprocess 等）由
                # 各自 task parser 校验；capability 门禁只消费共享 spatial 字段。
                sim = SimulationConfigs.from_dict(
                    normalize_simulation_cfg(sim_cfg), strict=False
                )
                validate_spatial_background_capability(mode.task, sim)
                checked += 1
        self.assertGreater(checked, 0)

    def test_active_aec_configs_explicitly_declare_erle_when_metrics_enabled(self):
        """现役 AEC 配置必须在非空指标列表中显式声明 ERLE。"""
        checked_configs = 0
        for cfg_path in _discover_config_files(CONFIG_ROOT / "aec"):
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            if (cfg.get("meta") or {}).get("inference_only") is True:
                continue
            val_metrics = (cfg.get("trainer") or {}).get("val_metrics")
            if not isinstance(val_metrics, dict) or val_metrics.get("enabled") is not True:
                continue
            with self.subTest(config=str(rel)):
                self.assertIn(
                    "metrics",
                    val_metrics,
                    f"{rel}: enabled AEC val_metrics must explicitly declare metrics",
                )
                metrics = val_metrics["metrics"]
                if metrics:
                    checked_configs += 1
                    self.assertIn(
                        "ERLE",
                        metrics,
                        f"{rel}: non-empty AEC val_metrics.metrics must explicitly include ERLE",
                    )
        self.assertGreater(checked_configs, 0)

    def test_all_training_configs_load_and_validate(self):
        """发现全部 YAML/YML；仅一个精确 TSE 模板可 hydrate 后走完整 entry。"""
        config_files = _discover_config_files()
        self.assertGreater(len(config_files), 0, f"No configs found under {CONFIG_ROOT}")

        for cfg_path in config_files:
            rel = cfg_path.relative_to(CONFIG_ROOT)
            with self.subTest(config=str(rel)):
                cfg = load_merged_config(cfg_path)
                meta = cfg.get("meta") or {}
                _validate_template_metadata(rel, meta)
                runtime_cfg = cfg
                if rel in TSE_TEMPLATE_CONFIGS:
                    self.assertEqual(detect_task(cfg).task, "tse")
                    for keys, error_path in TSE_TEMPLATE_NULL_FIELDS:
                        with self.subTest(config=str(rel), invalid_field=".".join(keys)):
                            candidate = _hydrate_tse_template(cfg)
                            _set_nested_value(candidate, keys, None)
                            with self.assertRaisesRegex(ValueError, error_path):
                                TSETrainingEntry("TSE_Test")._validate_config(candidate)
                    runtime_cfg = _hydrate_tse_template(cfg)
                # 非训练配置不走训练字段校验，但必须满足静态模型契约。
                elif meta.get("inference_only", False) or meta.get(
                    "evaluation_only", False
                ):
                    _validate_inference_only_contract(rel, cfg)
                    continue

                task_info = detect_task(runtime_cfg)
                if task_info.task == "aec":
                    entry = AECTrainingEntry("AEC_Test")
                elif task_info.task == "tse":
                    entry = TSETrainingEntry("TSE_Test")
                elif task_info.task == "ss":
                    entry = SSTrainingEntry("SS_Test")
                elif task_info.task == "ssl":
                    entry = SSLTrainingEntry("SSL_Test")
                else:
                    entry = SETrainingEntry("SE_Test")
                entry._validate_config(runtime_cfg)

    def test_ssl_moving_config_resource_gate(self):
        """SSL 仅保留 moving 动态配方：geometry 来自 simulation_cfg，offline 资源不得出现。"""
        moving = load_merged_config(
            CONFIG_ROOT / "ssl/fixed_array_ipdnet_dynamic_moving.yaml"
        )
        self.assertEqual(moving["meta"]["ssl_profile"], "moving_shared_simulation")
        self.assertEqual(moving["meta"]["ssl_version"], "v2_mobile")
        for split in ("train", "validation"):
            section = moving["data"][split]
            self.assertEqual(section["simulation_cfg"]["prob_moving_rir"], 1.0)
            for resource in ("speech_scp", "noise_scp"):
                self.assertTrue(section[resource])
            self.assertNotIn("rir_scp", section)

    def test_se_tse_dynamic_configs_use_runtime_equivalent_strict_simulation_parse(self):
        """SE/TSE 动态配置按 runtime 的继承和专属键提升后 strict 解析。"""
        parsed_configs: set[str] = set()
        parsed_sections = 0
        for cfg_path in _discover_config_files():
            cfg = load_merged_config(cfg_path)
            with self.subTest(config=str(cfg_path.relative_to(CONFIG_ROOT))):
                parsed = _strict_parse_runtime_simulation_cfg(cfg)
            if parsed:
                parsed_configs.add(cfg_path.relative_to(CONFIG_ROOT).as_posix())
                parsed_sections += parsed
        self.assertEqual(parsed_configs, STRICT_SIMULATION_PARSE_CONFIGS)
        self.assertGreaterEqual(parsed_sections, len(parsed_configs))

    def test_dynamic_aec_configs_preserve_effective_validation_simulation_contract(self):
        """八份生产动态 AEC 配置 + 一份论文对齐配置的 validation 继承 train 的 C-aec/Layer F。"""
        target_configs = (
            Path(
                "aec/microsoft_challenge/cagcrn/"
                "cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_32k_10ms_delay80ms_loss_mbrcl.yaml"
            ),
            Path(
                "aec/microsoft_challenge/deepvqe/"
                "deepvqe_small_32k_10ms_delay80ms_loss_MBRCL.yaml"
            ),
            Path(
                "aec/microsoft_challenge/deepvqe/"
                "deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ),
            Path(
                "aec/microsoft_challenge/deepvqe/"
                "deepvqe_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml"
            ),
            Path(
                "aec/microsoft_challenge/localvqe/"
                "localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_base_32k_10ms_delay80ms_loss_mr_si_l1_ss.yaml"
            ),
        )
        discovered_dynamic_aec = {
            path.relative_to(CONFIG_ROOT)
            for path in _discover_config_files(CONFIG_ROOT / "aec")
            if (load_merged_config(path).get("meta") or {}).get("evaluation_only") is not True
            and normalize_data_mode(
                (load_merged_config(path).get("data") or {}).get("mode")
            ).storage
            == "dynamic"
        }
        self.assertEqual(discovered_dynamic_aec, set(target_configs))
        expected_layer_f = {
            "noisy_post_mix_degradations_enabled": True,
            "prob_clip": 0.15,
            "clip_variants": ["fixed_rail"],
            "clip_variant_probs": {"fixed_rail": 1.0},
            "clip_pre_drive_db_range": [0.0, 9.0],
            "prob_post_eq": 0.15,
            "post_eq_db_range": [-3, 3],
        }
        expected_c_aec = {
            "ser_range": [-20, 20],
            "far_db_range": [-42, -12],
            "erl_range": [-20, 40],
            "erl_distribution": "beta",
            "erl_beta_a": 1.5,
            "erl_beta_b": 1.5,
            "aec_echo_device_eq": {
                "enabled": True,
                "prob": 0.25,
                "variant_probs": {"random_eq": 0.75, "bass_boost": 0.25},
                "random_eq_db_range": [-6.0, 6.0],
                "bass_boost_highpass_cutoff_range": [200.0, 2000.0],
                "bass_boost_attenuation_min_db": -12.0,
            },
            "aec_echo_clipping": {
                "enabled": True,
                "prob": 0.15,
                "path_probs": {
                    "post_ser_distortion": 0.80,
                    "pre_ser_overload": 0.15,
                    "synced_ref_clip": 0.05,
                },
                "paths": {
                    "post_ser_distortion": {
                        "variant_probs": {"tanh": 0.35, "hard": 0.65},
                        "peak_relative_threshold_range": [0.90, 0.99],
                    },
                    "pre_ser_overload": {
                        "variant_probs": {"fixed_rail": 0.80, "tanh": 0.20},
                        "overload_boost_db_range": [0.0, 6.0],
                        "rail_dbfs_range": [-3.0, -0.5],
                    },
                    "synced_ref_clip": {
                        "variant_probs": {"fixed_rail": 0.95, "tanh": 0.05},
                        "rail_dbfs_range": [-3.0, -0.5],
                    },
                },
            },
        }
        layer_f_fields = tuple(expected_layer_f)
        c_aec_simulation_fields = tuple(expected_c_aec)[:-2]

        # 论文对齐配置（DBGNet DBG-Mag，论文 Table I / Eq.10-12 / Sec. III-C）：
        # Layer F 整层关闭、C-aec 只走 post_ser_distortion、SER 均匀 [-15, 15]、
        # 播放设备 EQ 关闭（论文只建模扬声器非线性）。
        expected_paper_layer_f = {
            **expected_layer_f,
            "noisy_post_mix_degradations_enabled": False,
            "prob_clip": 0.0,
            "prob_post_eq": 0.0,
        }
        expected_paper_c_aec = {
            **{field: expected_c_aec[field] for field in c_aec_simulation_fields},
            "ser_range": [-15, 15],
            "aec_echo_device_eq": {
                "enabled": False,
                "prob": 0.0,
                "variant_probs": {"random_eq": 0.75, "bass_boost": 0.25},
                "random_eq_db_range": [-6.0, 6.0],
                "bass_boost_highpass_cutoff_range": [200.0, 2000.0],
                "bass_boost_attenuation_min_db": -12.0,
            },
            "aec_echo_clipping": {
                "enabled": True,
                "prob": 0.5,
                "path_probs": {
                    "post_ser_distortion": 1.0,
                    "pre_ser_overload": 0.0,
                    "synced_ref_clip": 0.0,
                },
                "paths": {
                    "post_ser_distortion": {
                        "variant_probs": {"tanh": 0.5, "hard": 0.5},
                        "peak_relative_threshold_range": [0.90, 0.99],
                    },
                    "pre_ser_overload": {
                        "variant_probs": {"fixed_rail": 0.80, "tanh": 0.20},
                        "overload_boost_db_range": [0.0, 6.0],
                        "rail_dbfs_range": [-3.0, -0.5],
                    },
                    "synced_ref_clip": {
                        "variant_probs": {"fixed_rail": 0.95, "tanh": 0.05},
                        "rail_dbfs_range": [-3.0, -0.5],
                    },
                },
            },
        }

        for rel in target_configs:
            cfg = load_merged_config(CONFIG_ROOT / rel)
            data_cfg = cfg["data"]
            mode = normalize_data_mode(data_cfg["mode"])
            self.assertEqual((mode.task, mode.storage), ("aec", "dynamic"))
            train_runtime = resolve_aec_dataset_kwargs(cfg, "train", cfg.get("stft"))
            validation_runtime = resolve_aec_dataset_kwargs(
                cfg, "val", cfg.get("stft")
            )
            train_simulation = SimulationConfigs.from_dict(
                train_runtime["simulation_cfg"], strict=True
            )
            validation_simulation = SimulationConfigs.from_dict(
                validation_runtime["simulation_cfg"], strict=True
            )

            def effective_contract(runtime: dict, simulation: SimulationConfigs) -> dict:
                return {
                    "layer_f": {
                        field: getattr(simulation, field) for field in layer_f_fields
                    },
                    "c_aec": {
                        **{
                            field: getattr(simulation, field)
                            for field in c_aec_simulation_fields
                        },
                        "aec_echo_device_eq": runtime["aec_echo_device_eq"],
                        "aec_echo_clipping": runtime["aec_echo_clipping"],
                    },
                }

            train_contract = effective_contract(train_runtime, train_simulation)
            validation_contract = effective_contract(
                validation_runtime, validation_simulation
            )
            config_expected_c_aec = dict(expected_c_aec)
            if rel.parts[-2] == "cagcrn":
                # CAGCRN 论文 recipe 的 SER 范围为 [-10, 10] dB。
                config_expected_c_aec["ser_range"] = [-10, 10]
            if rel.name.startswith("dbgnet_paper_"):
                expected_contract = {
                    "layer_f": expected_paper_layer_f,
                    "c_aec": expected_paper_c_aec,
                }
            else:
                expected_contract = {
                    "layer_f": expected_layer_f,
                    "c_aec": config_expected_c_aec,
                }
            with self.subTest(config=str(rel), contract="effective_validation"):
                self.assertTrue(data_cfg["validation"]["inherit_simulation"])
                self.assertEqual(train_contract, expected_contract)
                self.assertEqual(validation_contract, train_contract)
                self.assertEqual(
                    cfg["trainer"]["val_metrics"]["max_utterances"], 2200
                )

    def test_dynamic_training_configs_follow_rir_and_volume_contract_after_merge(self):
        """现役 dynamic 训练配置保持 RIR 与 post-RIR 音量字段合同。"""
        raw_rir_gain_allowlist = {
            Path("se/urgent2026/pase/stupase_phase2_flow.yaml"),
            Path("se/urgent2026/pase/stupase_phase1_melvocoder_gan.yaml"),
            Path("se/urgent2026/pase/stupase_phase0_dewavlm_r_distill.yaml"),
            Path("se/urgent2026/pase/phase3_vocoder_dual_gan.yaml"),
            Path("se/urgent2026/pase/phase2_dewavlm_distill.yaml"),
            Path("se/urgent2026/pase/phase1_vocoder_single.yaml"),
            Path("se/urgent2026/pase/unipase/unipase_phase4_postnet_gan.yaml"),
            Path("se/urgent2026/pase/unipase/unipase_phase3_adapter_gan.yaml"),
            Path("se/urgent2026/pase/unipase/unipase_phase2_vocoder_gan.yaml"),
            Path("se/urgent2026/pase/unipase/unipase_phase1_dewavlm_omni.yaml"),
            Path("se/urgent2026/pase/gap_urgenet/gap_urgenet_phase2_postnet_gan.yaml"),
            Path("se/urgent2026/pase/gap_urgenet/gap_urgenet_phase2_postnet_finetune.yaml"),
            Path("se/urgent2026/pase/gap_urgenet/gap_urgenet_phase1_predictor.yaml"),
        }
        seen_raw_rir_gain_configs = set()
        volume_fields = {
            "source_volume_db_range",
            "source_volume_use_rms",
            "source_volume_distribution",
            "source_volume_beta_a",
            "source_volume_beta_b",
        }
        for cfg_path in _discover_config_files():
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            if (cfg.get("meta") or {}).get("inference_only") is True:
                continue
            mode = normalize_data_mode((cfg.get("data") or {}).get("mode"))
            if mode.storage != "dynamic":
                continue

            simulation_cfg = (
                ((cfg.get("data") or {}).get("train") or {}).get("simulation_cfg")
                or {}
            )
            rir_field_path = "data.train.simulation_cfg.rir_normalize"
            rir_normalize = simulation_cfg.get(
                "rir_normalize", SimulationConfigs.rir_normalize
            )
            expected_rir_normalize = rel not in raw_rir_gain_allowlist
            with self.subTest(config=str(rel), field=rir_field_path):
                self.assertIs(
                    type(rir_normalize),
                    bool,
                    f"{rel}: effective {rir_field_path} must be a strict bool; "
                    f"got {rir_normalize!r}",
                )
                self.assertIs(
                    rir_normalize,
                    expected_rir_normalize,
                    f"{rel}: rir_normalize=false is reserved for the closed "
                    "PASE-family compatibility allowlist",
                )
            if not rir_normalize:
                seen_raw_rir_gain_configs.add(rel)

            if simulation_cfg.get("source_volume_normalization_enabled") is not True:
                continue
            missing_volume_fields = volume_fields - set(simulation_cfg)
            with self.subTest(config=str(rel), contract="post_rir_target_volume"):
                self.assertFalse(
                    missing_volume_fields,
                    f"{rel}: enabled post-RIR target volume is missing fields: "
                    f"{sorted(missing_volume_fields)}",
                )

        self.assertEqual(seen_raw_rir_gain_configs, raw_rir_gain_allowlist)

    def test_training_configs_do_not_declare_removed_trainer_mos_keys(self):
        """训练配置不得再声明已移除的 Trainer MOS 选项。"""
        removed_keys = {"mos_models", "mos_interval", "mos_batch_size"}
        for cfg_path in _discover_config_files():
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            if (cfg.get("meta") or {}).get("inference_only") is True:
                continue
            val_metrics = (cfg.get("trainer") or {}).get("val_metrics") or {}
            with self.subTest(config=str(rel)):
                self.assertFalse(
                    removed_keys & set(val_metrics),
                    f"removed Trainer MOS keys remain in {rel}: "
                    f"{sorted(removed_keys & set(val_metrics))}",
                )

    def test_removed_trainer_mos_keys_are_rejected_unconditionally(self):
        """旧 Trainer MOS 配置即使未启用 strict 也必须 fail-fast。"""
        cfg = load_merged_config(
            CONFIG_ROOT / "se/voicebank_demand/gtcrn_causal_offline.yaml"
        )
        entry = SETrainingEntry("SE_Test")
        cases = {
            "mos_models": ["DNSMOS"],
            "mos_interval": 1,
            "mos_batch_size": 16,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                candidate = copy.deepcopy(cfg)
                candidate.setdefault("trainer", {}).setdefault("val_metrics", {})[
                    key
                ] = value
                with self.assertRaisesRegex(
                    ValueError,
                    rf"trainer\.val_metrics\.{key}.*evaluate/",
                ):
                    entry._validate_config(candidate)

    def test_direct_trainer_construction_rejects_removed_mos_keys_before_base_init(self):
        """绕过 Entry 直构造任一 Trainer 时也必须复用相同 fail-fast 契约。"""
        trainer_classes = (
            MetricsCapableTrainer,
            SETrainer,
            AECTrainer,
            TSETrainer,
        )
        for trainer_cls in trainer_classes:
            for key, value in {
                "mos_models": ["DNSMOS"],
                "mos_interval": 1,
                "mos_batch_size": 16,
            }.items():
                with self.subTest(trainer=trainer_cls.__name__, key=key):
                    config = {"trainer": {"val_metrics": {key: value}}}
                    with unittest.mock.patch.object(
                        BaseTrainer,
                        "__init__",
                        side_effect=AssertionError("BaseTrainer init must not run"),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"trainer\.val_metrics\.{key}.*evaluate/",
                        ):
                            trainer_cls(None, None, config=config)

    def test_removed_trainer_mos_metric_names_are_rejected_unconditionally(self):
        """MOS 名称不能借 objective metrics 列表重新进入 Trainer。"""
        cfg = load_merged_config(
            CONFIG_ROOT / "se/voicebank_demand/gtcrn_causal_offline.yaml"
        )
        entry = SETrainingEntry("SE_Test")
        for metric in ("DNSMOS", "DNSMOS_OVRL", "AECMOS"):
            with self.subTest(metric=metric):
                candidate = copy.deepcopy(cfg)
                candidate.setdefault("trainer", {})["val_metrics"] = {
                    "enabled": False,
                    "strict": False,
                    "metrics": [metric],
                }
                with self.assertRaisesRegex(
                    ValueError,
                    rf"trainer\.val_metrics\.metrics.*{metric}.*evaluate/",
                ):
                    entry._validate_config(candidate)

    def test_direct_trainer_construction_rejects_removed_mos_metric_names(self):
        """直构造 Trainer 时也必须拒绝 objective metrics 中的 MOS 名称。"""
        trainer_classes = (
            MetricsCapableTrainer,
            SETrainer,
            AECTrainer,
            TSETrainer,
        )
        for trainer_cls in trainer_classes:
            for metric in ("DNSMOS", "DNSMOS_OVRL", "AECMOS"):
                with self.subTest(trainer=trainer_cls.__name__, metric=metric):
                    config = {
                        "trainer": {
                            "val_metrics": {
                                "enabled": False,
                                "strict": False,
                                "metrics": [metric],
                            }
                        }
                    }
                    with unittest.mock.patch.object(
                        BaseTrainer,
                        "__init__",
                        side_effect=AssertionError("BaseTrainer init must not run"),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"trainer\.val_metrics\.metrics.*{metric}.*evaluate/",
                        ):
                            trainer_cls(None, None, config=config)

    def test_config_path_contracts(self):
        """合并后的训练配置必须遵守实验目录、推理命名和资源占位约定。"""
        for cfg_path in _discover_config_files():
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            meta = cfg.get("meta") or {}
            is_infer = meta.get("inference_only") is True
            self.assertEqual(
                is_infer,
                cfg_path.stem.endswith("_infer"),
                f"inference_only/name mismatch: {rel}",
            )
            if not is_infer:
                task = detect_task(cfg).task
                trainer = cfg.get("trainer") or {}
                self.assertIsInstance(trainer.get("type"), str, str(rel))
                self.assertTrue(trainer["type"].strip(), str(rel))
                expected_prefix = {
                    "se": "exp/se/",
                    "aec": "exp/aec/",
                    "tse": "exp/tse/",
                    "ss": "exp/ss/",
                    "ssl": "exp/ssl/",
                }[task]
                if task == "tse":
                    expected_prefix += f"{rel.parts[1]}/"
                expected_exp_path = (
                    DNS3_ISOLATED_EXP_PATH
                    if rel == DNS3_ISOLATED_CONFIG
                    else expected_prefix + cfg_path.stem
                )
                self.assertEqual(
                    trainer.get("exp_path"),
                    expected_exp_path,
                    f"unexpected exp_path: {rel}",
                )

            for resource_path, value in _iter_resource_values(cfg):
                with self.subTest(config=str(rel), resource=".".join(resource_path)):
                    if value is None:
                        continue
                    self.assertIsInstance(value, str)
                    if value.startswith("exp/se/"):
                        resource_key = ".".join(resource_path).lower()
                        self.assertEqual(
                            rel.parts[0],
                            "se",
                            f"exp/se artifact is only valid in SE configs: {value}",
                        )
                        self.assertTrue(
                            "checkpoint" in resource_key or "ckpt" in resource_key,
                            f"exp/se is only valid for checkpoint artifacts: {value}",
                        )
                    else:
                        self.assertTrue(
                            value.startswith(
                                (
                                    # 本机实际数据根或相对仓库根目录路径
                                    "/data_nvme0n1/zxc/code/use_train/data/",
                                    "data/",
                                    # 仓库内随 TSE 模块提供、由 .gitignore 隔离的 speaker encoder 权重
                                    "tasks/tse/speaker_embedding_models/",
                                    # REAL-T 官方完整模型 checkpoint（可显式选择 speaker 子模块）
                                    "tasks/tse/pretrained_models/",
                                    # PASE 系列官方发布权重（由 .gitignore 隔离，不随 Git 分发）
                                    "tasks/se/models/pase_family/",
                                    "~/use_train_data/",
                                    "~/use_train_checkpoints/",
                                )
                            ),
                            f"resource must use the machine data root or a quoted "
                            f"external placeholder: {value}",
                        )

    def test_gtcrn_hybrid_configs_use_explicit_two_component_pair(self):
        """GTCRN 配置不重复声明 args，并共享顶层 STFT 上下文；论文对齐配置显式豁免。"""
        from core.losses.build import build_loss
        from core.stft import build_stft_istft

        expected_configs = {
            Path("se/voicebank_demand/gtcrn_causal_offline.yaml"),
            Path("se/dns_challenge3/gtcrn_causal_offline.yaml"),
            Path("se/urgent2026/gan/gtcrn_metricdiscriminator_dynamic.yaml"),
            Path("se/urgent2026/dpdfnet2_causal_urgent26_dynamic_hybrid.yaml"),
            Path("se/urgent2026/dpdfnet2_causal_urgent26_offline_hybrid.yaml"),
            Path("se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml"),
            Path("se/spatial/spatialnet_dynamic.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_mhsa.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_ret.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_mamba.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_mhsa_lf32s.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_mhsa_lf10s_36e.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_ret_lf32s.yaml"),
            Path("se/spatial/online_spatialnet_dynamic_mamba_lf32s.yaml"),
            Path("aec/offline_aec_template.yaml"),
            Path(
                "aec/microsoft_challenge/localvqe/"
                "localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml"
            ),
            Path(
                "aec/microsoft_challenge/deepvqe/"
                "deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml"
            ),
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ),
        }
        # 论文对齐配置复用同一 loss 类表达 L_recon = 0.7·L_mag + 0.3·L_com，权重与项目
        # GTCRN 配方（30/70 + log-ratio 对）不同，显式豁免；豁免集合在末尾精确断言。
        paper_alignment_exempt = {
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_paper_16k_16ms_delay1000ms_loss_recon.yaml"
            ),
        }
        migrated_configs = set()
        skipped_exempt = set()
        for cfg_path in _discover_config_files():
            rel = cfg_path.relative_to(CONFIG_ROOT)
            cfg = load_merged_config(cfg_path)
            loss_cfg = cfg.get("loss") or {}
            components = loss_cfg.get("components") or []
            component_types = [
                component.get("type")
                for component in components
                if isinstance(component, dict)
            ]
            with self.subTest(config=str(rel), contract="no_legacy_hybrid"):
                self.assertNotEqual(loss_cfg.get("type"), "HybridLoss")
                self.assertNotIn("HybridLoss", component_types)
            if "GTCRNCompressedSpectralLoss" not in component_types:
                continue
            if rel in paper_alignment_exempt:
                skipped_exempt.add(rel)
                continue

            migrated_configs.add(rel)
            spectral_index = component_types.index("GTCRNCompressedSpectralLoss")
            with self.subTest(config=str(rel), contract="explicit_gtcrn_pair"):
                self.assertIn(rel, expected_configs)
                self.assertEqual(loss_cfg.get("type"), "CompositeLoss")
                self.assertEqual(
                    component_types[spectral_index : spectral_index + 2],
                    ["GTCRNCompressedSpectralLoss", "GTCRNLogRatioLoss"],
                )
                self.assertEqual(component_types.count("GTCRNCompressedSpectralLoss"), 1)
                self.assertEqual(component_types.count("GTCRNLogRatioLoss"), 1)
                pair = components[spectral_index : spectral_index + 2]
                self.assertEqual([component["weight"] for component in pair], [1.0, 1.0])
                self.assertEqual(
                    pair[0].get("args"),
                    {"compress_factor": 0.3, "lamda_ri": 30.0, "lamda_mag": 70.0},
                )
                self.assertNotIn("args", pair[1])
                stft_keys = {
                    "n_fft",
                    "hop_length",
                    "win_length",
                    "window",
                    "center",
                    "normalized",
                }
                for component in pair:
                    self.assertTrue(stft_keys.isdisjoint(component.get("args") or {}))

                stft_cfg = cfg["stft"]
                self.assertEqual(stft_cfg.get("window", "hann"), "hann")
                self.assertIs(stft_cfg.get("center", True), True)
                self.assertIs(stft_cfg.get("normalized", False), False)
                built = build_loss(
                    loss_cfg,
                    stft_cfg=stft_cfg,
                    sample_rate=(cfg.get("data") or {}).get("sample_rate"),
                )
                spectral = built.losses[spectral_index]
                ratio = built.losses[spectral_index + 1]
                model_stft = build_stft_istft(stft_cfg)
                for component in (spectral, ratio):
                    self.assertEqual(
                        (
                            component.n_fft,
                            component.hop_length,
                            component.win_length,
                        ),
                        (
                            model_stft.n_fft,
                            model_stft.hop_length,
                            model_stft.win_length,
                        ),
                    )
                    self.assertEqual(component.window_name, model_stft.window_name)
                    self.assertIs(component.center, model_stft.center)
                    self.assertIs(component.normalized, model_stft.normalized)
                self.assertTrue(spectral._window.equal(model_stft._win_buf))

                if rel.name == "localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml":
                    self.assertEqual(
                        component_types,
                        [
                            "GTCRNCompressedSpectralLoss",
                            "GTCRNLogRatioLoss",
                            "FWNegSISDRLoss",
                        ],
                    )
                    fw_sdr = components[2]
                    self.assertEqual(fw_sdr["weight"], 0.2)
                    self.assertEqual(
                        fw_sdr["args"],
                        {
                            "n_bins": 18,
                            "scale": "mel",
                            "weights_type": "log_snr_speech_norm",
                            "gamma": 0.2,
                            "sdr_min_db": -20.0,
                            "sdr_max_db": 40.0,
                        },
                    )
                else:
                    self.assertEqual(
                        component_types,
                        ["GTCRNCompressedSpectralLoss", "GTCRNLogRatioLoss"],
                    )

        self.assertEqual(migrated_configs, expected_configs)
        self.assertEqual(skipped_exempt, paper_alignment_exempt)

    def test_repaired_se_config_fields(self):
        """被同行注释吞掉的关键训练字段必须按实验设定保留。"""
        expected_fields = {
            Path("se/dns_challenge3/gtcrn_causal_offline.yaml"): {
                ("trainer", "type"): "SETrainer",
                ("trainer", "epochs"): 200,
                ("trainer", "max_val_batches"): 0,
                ("trainer", "validation_epoch_interval"): 1,
                ("trainer", "save_checkpoint_interval"): 5,
            },
            Path("se/voicebank_demand/gtcrn_causal_offline.yaml"): {
                ("trainer", "type"): "SETrainer",
                ("trainer", "epochs"): 200,
                ("trainer", "max_val_batches"): 0,
                ("trainer", "validation_epoch_interval"): 1,
                ("trainer", "save_checkpoint_interval"): 10,
            },
            Path("se/urgent2026/usr_bsrnn_causal_urgent26_offline_hybrid.yaml"): {
                ("trainer", "type"): "SETrainer",
                ("trainer", "epochs"): 100,
                ("trainer", "max_val_batches"): 0,
                ("trainer", "validation_epoch_interval"): 1,
                ("trainer", "save_checkpoint_interval"): 5,
                ("distributed", "nccl_timeout_minutes"): 10,
            },
            Path(
                "se/urgent2026/"
                "usr_bsrnn_noncausal_urgent26_offline_multires_l1_spec.yaml"
            ): {
                ("model", "args", "num_layer"): 6,
                ("trainer", "type"): "SETrainer",
                ("trainer", "epochs"): 80,
                ("trainer", "max_val_batches"): 0,
                ("trainer", "validation_epoch_interval"): 1,
                ("trainer", "save_checkpoint_interval"): 5,
            },
            Path(
                "aec/microsoft_challenge/deepvqe/"
                "deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ): {
                ("data", "sample_rate"): 32000,
                ("data", "train", "rir_scp"): "data/shared/rirs/gpurir_static/gpurir/fs32000/train_rirs.scp",
                ("data", "validation", "rir_scp"): "data/shared/rirs/gpurir_static/gpurir/fs32000/val_rirs.scp",
                ("data", "train", "simulation_cfg", "rir_normalize"): True,
                # 纯 SCP 噪声（非均衡对照）：无 category metadata，不得保留 sampling.noise 均衡块
                ("data", "train", "noise_scp"): "data/shared/noise/filter/v2/train_noise_sources.scp",
                ("data", "validation", "noise_scp"): "data/shared/noise/filter/v2/val_noise_sources.scp",
                ("data", "dataloader", "batch_size"): 16,
                ("data", "dataloader", "num_workers"): 4,
                ("data", "dataloader", "val_num_workers"): 1,
                ("data", "dataloader", "timeout"): 120,
                ("data", "dataloader", "prefetch_factor"): 2,
                ("data", "dataloader", "pin_memory"): True,
                ("data", "dataloader", "persistent_workers"): True,
                ("data", "dataloader", "val_persistent_workers"): False,
                ("model", "args", "align_delay"): 10,
                ("trainer", "exp_path"): "exp/aec/deepvqe_small_32k_10ms_delay80ms_loss_hybrid",
                ("trainer", "epochs"): 60,
                ("trainer", "max_val_batches"): 0,
                ("trainer", "validation_epoch_interval"): 1,
                ("trainer", "validation_steps_interval"): 0,
                ("trainer", "val_metrics", "metrics"): ["SI-SDR", "ESTOI", "ERLE"],
                ("trainer", "val_metrics", "interval"): 1,
                ("trainer", "val_metrics", "max_utterances"): 2200,
                ("trainer", "val_metrics", "metrics_workers"): 2,
            },
            Path(
                "tse/inference/"
                "tse_bsrnn_spk_realt_baseline_infer.yaml"
            ): {
                ("model", "args", "speaker_model", "fbank", "dither"): 1.0,
                ("model", "args", "speaker_model", "fbank", "sample_rate"): 16000,
            },
        }

        for rel, fields in expected_fields.items():
            cfg = load_merged_config(CONFIG_ROOT / rel)
            for keys, expected in fields.items():
                with self.subTest(config=str(rel), field=".".join(keys)):
                    self.assertEqual(_nested_value(cfg, keys), expected)

        meeting_cfg = load_merged_config(
            CONFIG_ROOT
            / "aec/microsoft_challenge/deepvqe/"
            "deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml"
        )
        self.assertNotIn("noise", meeting_cfg["data"].get("sampling") or {})

    def test_dbgnet_32k_configuration_contract(self):
        """32 kHz DBGNet 必须固定训练频谱、ERB、manifest 与 delay 合同。"""
        rel = Path(
            "aec/microsoft_challenge/dbgnet/"
            "dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml"
        )
        cfg = load_merged_config(CONFIG_ROOT / rel)
        from core.losses.build import build_loss

        loss = build_loss(
            cfg["loss"],
            stft_cfg=cfg["stft"],
            sample_rate=cfg["data"]["sample_rate"],
        )
        self.assertEqual(cfg["loss"]["type"], "CompositeLoss")
        self.assertEqual(
            [component.__class__.__name__ for component in loss.losses],
            ["GTCRNCompressedSpectralLoss", "GTCRNLogRatioLoss"],
        )
        self.assertEqual(
            [component["weight"] for component in cfg["loss"]["components"]],
            [1.0, 1.0],
        )
        spectral, ratio = loss.losses
        for key in ("n_fft", "hop_length", "win_length", "center"):
            self.assertEqual(getattr(spectral, key), cfg["stft"][key])
        self.assertEqual(spectral._window.shape[0], cfg["stft"]["win_length"])
        for key in ("n_fft", "hop_length", "win_length", "center"):
            self.assertEqual(getattr(ratio, key), cfg["stft"][key])
        self.assertEqual(ratio.window_name, cfg["stft"]["window"])
        self.assertIs(ratio.normalized, False)
        self.assertEqual(ratio._window.shape[0], cfg["stft"]["win_length"])
        self.assertEqual(
            cfg["stft"],
            {
                "n_fft": 640,
                "hop_length": 320,
                "win_length": 640,
                "window": "hann",
                "center": True,
            },
        )
        self.assertEqual(cfg["data"]["sample_rate"], 32000)
        self.assertEqual(cfg["model"]["args"]["sample_rate"], 32000)
        cfg_16k = load_merged_config(
            CONFIG_ROOT
            / "aec/microsoft_challenge/dbgnet/"
            "dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml"
        )
        alpha_16k = cfg_16k["data"]["train"]["state_detector_pre_emphasis"]
        alpha_32k = cfg["data"]["train"]["state_detector_pre_emphasis"]
        self.assertEqual(alpha_16k, 0.97)
        self.assertEqual(alpha_32k, alpha_16k)
        self.assertEqual(cfg["model"]["args"]["erb_low_bins"], 81)
        self.assertEqual(cfg["model"]["args"]["erb_bands"], 80)
        self.assertEqual(
            cfg["data"]["train"]["rir_scp"],
            "data/shared/rirs/gpurir_static/gpurir/fs32000/train_rirs.scp",
        )
        self.assertEqual(
            cfg["data"]["validation"]["rir_scp"],
            "data/shared/rirs/gpurir_static/gpurir/fs32000/val_rirs.scp",
        )
        self.assertEqual(
            cfg["data"]["train"]["real_farend_echo_pair_json"],
            "data/aec/echo/echo_prealign_32k/real_farend_echo_pair_train.json",
        )
        self.assertEqual(
            cfg["data"]["validation"]["real_farend_echo_pair_json"],
            "data/aec/echo/echo_prealign_32k/real_farend_echo_pair_val.json",
        )
        self.assertEqual(
            cfg["aec_frontend"],
            {
                "delay_alignment": {"enabled": False},
                "input_routing": "mic_farend",
            },
        )
        self.assertFalse(cfg["model"]["args"]["use_state_detector"])
        self.assertEqual(cfg["model"]["args"]["state_nclass"], 3)
        self.assertEqual(
            cfg["data"]["train"]["state_detector_label_type"], "fst_dt_nst"
        )
        self.assertEqual(
            cfg["trainer"]["exp_path"],
            "exp/aec/dbgnet_small_32k_10ms_delay80ms_loss_hybrid",
        )

        simulation_cfg = cfg["data"]["train"]["simulation_cfg"]
        self.assertFalse(simulation_cfg["aec_ref_echo_fine_gcc_align"])
        self.assertNotIn("jitter_type_probs", simulation_cfg["aec_time_delay_jitter"])
        self.assertEqual(
            {key: simulation_cfg[key] for key in (
                "aec_ref_echo_random_delay_prob",
                "aec_ref_echo_random_delay_mode",
                "aec_ref_echo_random_delay_distribution",
                "aec_ref_echo_random_delay_beta_a",
                "aec_ref_echo_random_delay_beta_b",
                "aec_ref_echo_random_delay_uniform_prob",
                "aec_ref_echo_delay_range_ms",
            )},
            {
                "aec_ref_echo_random_delay_prob": 1.0,
                "aec_ref_echo_random_delay_mode": "echo_only",
                "aec_ref_echo_random_delay_distribution": "beta",
                "aec_ref_echo_random_delay_beta_a": 2.0,
                "aec_ref_echo_random_delay_beta_b": 4.0,
                "aec_ref_echo_random_delay_uniform_prob": 0.0,
                "aec_ref_echo_delay_range_ms": [0.0, 80.0],
            },
        )
        self.assertEqual(
            simulation_cfg["aec_time_delay_jitter"],
            {
                "enabled": True,
                "prob": 0.03,
                "jitter_type": "sudden_jump",
                "jitter_rate": [0.02, 0.10],
                "jitter_corr_length": [100.0, 500.0],
                "jitter_max_ms": 20.0,
                "jump_ms_range": [-20.0, 20.0],
                "min_segment_ms": 1000.0,
                "transition_ms": 0.0,
                "edge_mode": "nearest",
            },
        )
        self.assertNotIn("max_relative_jitter_ms", simulation_cfg["aec_time_delay_jitter"])
        self.assertNotIn("max_abs_delay_ms", simulation_cfg["aec_time_delay_jitter"])

    def test_aec_frontend_defaults_baseline_and_model_enablement_matrix(self):
        """AEC input frontend 默认值与正式配置矩阵均受合同保护。"""
        from dataclasses import asdict

        from tasks.aec.inference.aec_frontend import AECFrontendConfig
        from tasks.aec.inference.streaming_delay_estimator import AECStreamAlignConfig
        from tools.verify.streaming_gcc_benchmark import BASELINE

        expected_defaults = {
            "enabled": False,
            "algorithm": "gcc_phat",
            "buffer_ms": 500.0,
            "coarse_ms": 50.0,
            "residual_ms": 50.0,
            "update_ms": 100.0,
            "alpha": 0.15,
            "acps_alpha": 0.8,
            "ref_thresh_db": -50.0,
            "min_obs_ms": 200.0,
            "peak_thresh": 0.05,
            "prominence_db": 6.0,
            "false_accept_prob": 1e-4,
            "frame_size": 160,
        }
        self.assertEqual(AECStreamAlignConfig().to_dict(), expected_defaults)

        baseline_align = AECStreamAlignConfig.from_dict(asdict(BASELINE))
        self.assertEqual(baseline_align.to_dict(), expected_defaults)
        self.assertEqual(
            baseline_align.to_estimator_config(sample_rate=16000),
            BASELINE.config(sample_rate=16000),
        )
        for invalid in (0.0, 1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(false_accept_prob=invalid):
                with self.assertRaisesRegex(ValueError, "false_accept_prob"):
                    AECStreamAlignConfig.from_dict({"false_accept_prob": invalid})

        expected_matrix = {
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_16k_10ms_delay80ms_loss_hybrid.yaml"
            ): "DBGNet",
            Path(
                "aec/microsoft_challenge/dbgnet/"
                "dbgnet_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ): "DBGNet",
            Path(
                "aec/microsoft_challenge/cagcrn/"
                "cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml"
            ): "CAGCRN",
            Path(
                "aec/microsoft_challenge/deepvqe/"
                "deepvqe_small_32k_10ms_delay80ms_loss_hybrid.yaml"
            ): "DeepVQE",
            Path(
                "aec/microsoft_challenge/localvqe/"
                "localvqe_16k_20ms_delay80ms_loss_hybrid_fwsdr.yaml"
            ): "LocalVQE",
        }
        for rel, model_type in expected_matrix.items():
            cfg = load_merged_config(CONFIG_ROOT / rel)
            raw_frontend = cfg["aec_frontend"]
            effective_frontend = AECFrontendConfig.from_dict(raw_frontend).to_dict()
            with self.subTest(config=str(rel), model=model_type):
                self.assertEqual(cfg["model"]["type"], model_type)
                self.assertIn("aec_frontend", cfg)
                self.assertNotIn("aec_stream_align", cfg)
                self.assertNotIn("linear_aec", cfg)
                self.assertNotIn("aec_stream_align", cfg["data"])
                self.assertNotIn("aec_stream_align", cfg["data"]["train"])
                self.assertNotIn("aec_stream_align", cfg["data"]["train"]["simulation_cfg"])
                self.assertFalse(effective_frontend["delay_alignment"]["enabled"])
                self.assertIsNone(effective_frontend["linear_aec"])
                self.assertEqual(effective_frontend["input_routing"], "mic_farend")
                # 五份当前 AEC 生产配置共用原始 validation SCP 的全局指标 quota 上限。
                self.assertEqual(cfg["trainer"]["val_metrics"]["max_utterances"], 2200)

        shared_delay = {
            "aec_ref_echo_random_delay_prob": 1.0,
            "aec_ref_echo_random_delay_mode": "echo_only",
            "aec_ref_echo_random_delay_distribution": "beta",
            "aec_ref_echo_random_delay_beta_a": 2.0,
            "aec_ref_echo_random_delay_beta_b": 4.0,
            "aec_ref_echo_random_delay_uniform_prob": 0.0,
            "aec_ref_echo_delay_range_ms": [0.0, 80.0],
        }
        shared_jitter = {
            "enabled": True,
            "prob": 0.03,
            "jitter_type": "sudden_jump",
            "jitter_rate": [0.02, 0.10],
            "jitter_corr_length": [100.0, 500.0],
            "jitter_max_ms": 20.0,
            "jump_ms_range": [-20.0, 20.0],
            "min_segment_ms": 1000.0,
            "transition_ms": 0.0,
            "edge_mode": "nearest",
        }
        expected_delay_jitter = {
            model_type: {"delay": shared_delay, "jitter": shared_jitter}
            for model_type in ("DBGNet", "DeepVQE", "CAGCRN", "LocalVQE")
        }
        for rel, model_type in expected_matrix.items():
            if model_type not in expected_delay_jitter:
                continue
            simulation_cfg = load_merged_config(CONFIG_ROOT / rel)["data"]["train"][
                "simulation_cfg"
            ]
            expected = expected_delay_jitter[model_type]
            with self.subTest(config=str(rel), delay_jitter=model_type):
                for key, value in expected["delay"].items():
                    self.assertEqual(simulation_cfg[key], value)
                jitter_cfg = simulation_cfg["aec_time_delay_jitter"]
                self.assertEqual(jitter_cfg, expected["jitter"])
                self.assertNotIn("max_relative_jitter_ms", jitter_cfg)
                self.assertNotIn("max_abs_delay_ms", jitter_cfg)
                self.assertNotIn("jitter_type_probs", jitter_cfg)

        offline_cfg = load_merged_config(CONFIG_ROOT / "aec/offline_aec_template.yaml")
        offline_frontend = AECFrontendConfig.from_dict(
            offline_cfg["aec_frontend"]
        ).to_dict()
        self.assertFalse(offline_frontend["delay_alignment"]["enabled"])
        self.assertIsNone(offline_frontend["linear_aec"])
        self.assertEqual(offline_frontend["input_routing"], "mic_farend")

    def test_base_se_model_forward_declares_optional_keyword_only_fs(self):
        signature = inspect.signature(BaseSEModel.forward)
        fs = signature.parameters["fs"]
        self.assertEqual(fs.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(fs.default)

    def test_tse_template_strict_hydrate_and_validate(self):
        """strict JSONL 模板经 hydrate 后通过入口校验。"""
        for rel in sorted(TSE_TEMPLATE_CONFIGS):
            with self.subTest(config=str(rel)):
                cfg = load_merged_config(CONFIG_ROOT / rel)
                strict_cfg = _hydrate_tse_strict_template(cfg)
                self.assertNotIn(
                    "utt2spk", strict_cfg["data"]["train"]
                )
                TSETrainingEntry("TSE_Test")._validate_config(strict_cfg)

    def test_tse_template_allowlist_rejects_inference_only_short_circuit(self):
        rel = Path("tse/dynamic/tse_bsrnn_spk.yaml")
        with self.assertRaisesRegex(AssertionError, "inference_only"):
            _validate_template_metadata(rel, {"inference_only": True})

    def test_tse_template_allowlist_does_not_hide_other_config_errors(self):
        cases = (
            (
                ("data", "train", "spk2enroll_json"),
                [],
                r"data\.train\.spk2enroll_json",
            ),
            (
                ("data", "validation", "speech_scp"),
                [],
                r"data\.validation\.speech_scp",
            ),
            (
                ("data", "validation", "mode"),
                "offline",
                "offline validation",
            ),
        )
        for rel in sorted(TSE_TEMPLATE_CONFIGS):
            cfg = load_merged_config(CONFIG_ROOT / rel)
            for keys, invalid_value, error in cases:
                with self.subTest(config=str(rel), invalid_field=".".join(keys)):
                    candidate = _hydrate_tse_template(cfg)
                    _set_nested_value(candidate, keys, invalid_value)
                    with self.assertRaisesRegex((TypeError, ValueError), error):
                        TSETrainingEntry("TSE_Test")._validate_config(candidate)


def _write_placeholder_scp(path: Path, uid: str, n_samples: int = 160000) -> None:
    """生成四列占位 SCP（uid fs length path）；path 无需真实存在（构建期只解析清单）。"""
    path.write_text(f"{uid} 16000 {n_samples} /nonexistent/{uid}.wav\n", encoding="utf-8")


def _write_placeholder_strict_jsonl(path: Path, uid: str, speaker_id: str) -> None:
    """生成 dynamic TSE strict speech manifest 的单条占位记录。"""
    path.write_text(
        json.dumps(
            {
                "uid": uid,
                "fs": 16000,
                "length": 160000,
                "path": f"/nonexistent/{uid}.wav",
                "language": "zh",
                "dataset_id": "fixture",
                "speaker_id": speaker_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_tse_config_val_dataset_build_smoke(tmp_path):
    """TSE 配置 val 数据集构建冒烟：防止"校验通过但 build_dataset(mode='val') 必崩"回归。

    加载真实 TSE YAML，把数据清单路径替换为 tmp_path 下的占位文件，
    实测 val 走动态仿真分支构建出 TSEDynamicMixingDataset。
    """
    from tasks.tse.data import build_dataset
    from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset

    cfg_path = CONFIG_ROOT / "tse" / "dynamic" / "tse_bsrnn_spk.yaml"
    cfg = load_merged_config(cfg_path)

    data_cfg = cfg["data"]
    train_cfg = data_cfg["train"]
    sim_cfg = train_cfg["simulation_cfg"]

    speech_jsonl = tmp_path / "speech.jsonl"
    _write_placeholder_strict_jsonl(speech_jsonl, "utt1", "spk1")
    train_cfg["speech_scp"] = str(speech_jsonl)
    for key in ("noise_scp", "rir_scp", "windnoise_scp"):
        scp = tmp_path / f"{key}.scp"
        _write_placeholder_scp(scp, f"{key}_1")
        train_cfg[key] = str(scp)
    # 冒烟测试使用 ordinary SCP；noise runtime 不执行 strict v6 admission。
    music_scp = tmp_path / "music.scp"
    _write_placeholder_scp(music_scp, "music1")
    train_cfg["music_scp"] = str(music_scp)

    spk2enroll = tmp_path / "spk2enroll.json"
    spk2enroll.write_text(
        json.dumps({"fixture": {"spk1": ["/nonexistent/enroll1.wav"]}}),
        encoding="utf-8",
    )
    train_cfg["spk2enroll_json"] = str(spk2enroll)

    validation_cfg = data_cfg["validation"]
    val_speech_jsonl = tmp_path / "val_speech.jsonl"
    _write_placeholder_strict_jsonl(val_speech_jsonl, "val_utt1", "spk1")
    validation_cfg["speech_scp"] = str(val_speech_jsonl)
    for key in ("noise_scp", "rir_scp", "windnoise_scp"):
        scp = tmp_path / f"val_{key}.scp"
        _write_placeholder_scp(scp, f"val_{key}_1")
        validation_cfg[key] = str(scp)
    validation_sim_cfg = validation_cfg["simulation_cfg"]
    validation_cfg["music_scp"] = str(music_scp)
    validation_cfg["spk2enroll_json"] = str(spk2enroll)
    # 注册音 provider 是独立资源，也必须由 fixture 接管，不能读真实训练资产。
    for section, simulation in ((train_cfg, sim_cfg), (validation_cfg, validation_sim_cfg)):
        simulation.setdefault("enrollment_augment", {}).update(
            rir_scp=section["rir_scp"], noise_scp=section["noise_scp"],
        )

    TSETrainingEntry("TSE_Test")._validate_config(cfg)

    val_ds = build_dataset(data_cfg, mode="val", meta=cfg.get("meta"))
    assert isinstance(val_ds, TSEDynamicMixingDataset)
    assert not val_ds.enrollment_augmentor.enabled
    assert len(val_ds) > 0


def test_removed_allow_unverified_noise_manifest_config_key_is_rejected():
    """旧 runtime admission 逃生键不能在任意配置层静默遗留。"""
    from tasks.se.data import build_dataset

    cfg = {
        "mode": "dynamic",
        "train": {
            "speech_scp": "speech.scp",
            "noise_scp": "noise.scp",
            "rir_scp": "rir.scp",
            "simulation_cfg": {"prob_reverberation": 0.0},
            "allow_unverified_noise_manifest": True,
        },
    }
    with pytest.raises(ValueError, match="allow_unverified_noise_manifest is no longer supported"):
        build_dataset(cfg, mode="train")


if __name__ == "__main__":
    unittest.main()

pytestmark = [pytest.mark.unit]
