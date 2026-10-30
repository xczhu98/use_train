"""SE spatial C-3 彩色噪声（BackgroundSpec 链渲染）与 capability 矩阵合同测试。

接入 checklist 三件套（capabilities 模块 docstring）的本文件覆盖：

1. target protection：背景只进入 observation，输入 speech image 不被改写；
2. facts/audit：渲染 policy、SNR、颜色类型进入 BackgroundMix facts 与 acoustic audit；
3. RNG：prob=0 时零随机数消耗；mono 路径走原实现不受影响。
"""

from __future__ import annotations

import numpy as np
import pytest

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.simulation.background import mix_background
from core.data.simulation.capabilities import (
    SPATIAL_FEATURE_CAPABILITIES,
    triggered_spatial_features,
)
from core.data.simulation.point_source import PointSourceRIRProvider
from core.data.simulation.role_policy import BackgroundPolicy, BackgroundSpec
from core.data.simulation.spatial_background import (
    validate_spatial_background_capability,
)
from core.data.simulation_configs import SimulationConfigs

pytestmark = [pytest.mark.unit]

_MIC_POSITIONS = [[1.0, 1.0, 1.5], [1.1, 1.0, 1.5]]


def _capability(feature: str):
    matches = [c for c in SPATIAL_FEATURE_CAPABILITIES if c.feature == feature]
    assert len(matches) == 1
    return matches[0]


def _spatial_cfg(**overrides) -> SimulationConfigs:
    values = {
        "spatial_enabled": True,
        "custom_geometry": {"positions_m": _MIC_POSITIONS},
        "reference_channel": 0,
        "prob_colored_noise": 0.0,
        "colored_noise_snr_range": [10.0, 50.0],
        "colored_noise_types": ["white", "pink", "brown", "equalized"],
        "noise_mix_prob": 0.0,
        "prob_wind_noise_sidechain": 0.0,
        "snr_low": 0.0,
        "snr_high": 10.0,
        "snr_reference": "all_speech",
        "reuse_noise": False,
    }
    values.update(overrides)
    return SimulationConfigs.from_dict(values, strict=True)


def _dataset(cfg: SimulationConfigs, monkeypatch, noise: np.ndarray):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = cfg
    dataset.export_simulation_stems = True
    dataset._noise_single_reuse_cache = None
    dataset._noise_dual_reuse_cache = None
    noise_2d = noise[None, :]
    monkeypatch.setattr(
        dataset,
        "_read_one_additive_noise_scp_only_with_identity",
        lambda sr, max_dur, rng: ("noise_scp:fake", noise_2d),
    )
    monkeypatch.setattr(
        dataset,
        "_mix_acoustic_read_additive",
        lambda mode, sr, max_dur, rng, reuse: noise_2d,
    )
    return dataset


# ---------------------------------------------------------------------------
# capability 矩阵（单一事实源）
# ---------------------------------------------------------------------------


def test_capability_matrix_pinned():
    assert _capability("colored_noise").spatial_tasks == frozenset(
        {"se", "aec", "tse", "ss", "ssl"}
    )
    assert _capability("wind_sidechain").spatial_tasks == frozenset({"ssl"})
    assert _capability("rts_target").spatial_tasks == frozenset({"se"})
    assert _capability("rts_target").is_triggered(_spatial_cfg(target_type="rts"))
    assert not _capability("rts_target").is_triggered(_spatial_cfg(target_type="early"))


@pytest.mark.parametrize("task", ["se", "aec", "tse", "ss", "ssl"])
def test_gate_allows_colored_noise_for_all_tasks(task):
    validate_spatial_background_capability(task, _spatial_cfg(prob_colored_noise=0.2))


@pytest.mark.parametrize("task", ["aec", "tse", "ss", "ssl"])
def test_gate_rejects_rts_target_outside_se(task):
    cfg = _spatial_cfg(target_type="rts")
    with pytest.raises(ValueError, match=r"does not preserve target_type='rts'"):
        validate_spatial_background_capability(task, cfg)


@pytest.mark.parametrize("task", ["se", "aec", "tse", "ss"])
def test_gate_rejects_wind_sidechain_outside_ssl(task):
    cfg = _spatial_cfg(prob_wind_noise_sidechain=0.5)
    with pytest.raises(ValueError, match="does not preserve prob_wind_noise_sidechain"):
        validate_spatial_background_capability(task, cfg)


def test_gate_allows_registered_combinations():
    validate_spatial_background_capability("se", _spatial_cfg(prob_colored_noise=1.0))
    validate_spatial_background_capability("ssl", _spatial_cfg(prob_wind_noise_sidechain=1.0))
    validate_spatial_background_capability("se", _spatial_cfg(target_type="rts"))


def test_gate_noop_for_mono_or_zero_probability():
    mono = SimulationConfigs.from_dict({"prob_colored_noise": 1.0}, strict=True)
    validate_spatial_background_capability("aec", mono)
    assert triggered_spatial_features("aec", _spatial_cfg(prob_colored_noise=0.0)) == []
    # 非五任务不受 spatial capability 管辖。
    assert triggered_spatial_features("other", _spatial_cfg(prob_colored_noise=1.0)) == []


# ---------------------------------------------------------------------------
# spatial 渲染集成：diffuse 与 point_source 两种 policy
# ---------------------------------------------------------------------------


def _run_mix(cfg: SimulationConfigs, monkeypatch, policy_spec: BackgroundSpec, seed: int = 0):
    speech = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    speech_2d = np.stack([speech, speech * 0.5]).astype(np.float32)
    noise = np.random.default_rng(99).normal(0.0, 0.1, 4000).astype(np.float32)
    dataset = _dataset(cfg, monkeypatch, noise)
    original = speech_2d.copy()
    result = mix_background(
        dataset,
        speech_2d.copy(),
        speech_2d,
        16000,
        speech_2d.shape[1],
        speech_2d.shape[1],
        np.random.default_rng(seed),
        True,
        5.0,
        image_geometry=_MIC_POSITIONS,
        background_spec=policy_spec,
    )
    return result, original, speech_2d


def test_spatial_colored_noise_diffuse_renders_c_t_image(monkeypatch):
    cfg = _spatial_cfg(prob_colored_noise=1.0)
    spec = BackgroundSpec(BackgroundPolicy.DIFFUSE_COHERENCE, geometry_bound=True)
    result, original, speech_2d = _run_mix(cfg, monkeypatch, spec)

    assert result.used_colored_noise is True
    assert result.observation.shape == speech_2d.shape
    assert result.observation.dtype == np.float32
    assert result.additive.shape == speech_2d.shape
    assert not np.allclose(result.additive, 0.0)
    # target protection：输入 speech image 未被改写。
    np.testing.assert_array_equal(speech_2d, original)
    # facts/audit 三件套：policy、SNR、颜色类型齐全。
    facts = result.noise_image_provenance
    assert facts["colored_noise_applied"] is True
    assert facts["colored_noise"]["background_policy"] == "diffuse_coherence"
    assert facts["colored_noise"]["shape"] == [2, 4000]
    assert facts["colored_noise"]["color"] in {"white", "pink", "brown", "equalized"}
    assert result.audit is not None
    assert result.audit["colored_noise"]["applied"] is True
    assert result.audit["colored_noise"]["spatial"] is True
    assert result.audit["colored_noise"]["snr_db"] == pytest.approx(
        facts["colored_noise"]["snr_db"]
    )
    assert any(
        "colored_noise:synthetic" in facts_entry.get("component_identities", ())
        for facts_entry in facts["ordinary_components"]
    )


def test_spatial_colored_noise_point_source_renders_c_t_image(monkeypatch):
    cfg = _spatial_cfg(prob_colored_noise=1.0)
    rirs = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    point_source = PointSourceRIRProvider(
        rirs,
        {"positions_m": _MIC_POSITIONS},
        16000,
        "background_noise/0",
    )
    spec = BackgroundSpec(
        BackgroundPolicy.POINT_SOURCE_RIR,
        geometry_bound=True,
        shared_scene=True,
        point_source_provider=point_source,
    )
    result, original, speech_2d = _run_mix(cfg, monkeypatch, spec)

    assert result.used_colored_noise is True
    assert result.observation.shape == speech_2d.shape
    np.testing.assert_array_equal(speech_2d, original)
    facts = result.noise_image_provenance
    assert facts["colored_noise_applied"] is True
    assert facts["colored_noise"]["background_policy"] == "point_source_rir"


def test_spatial_colored_noise_miss_keeps_scene_unchanged(monkeypatch):
    """触发未命中（概率 < 1 且本次抽样未中）时不施加彩色噪声、不产生彩色 facts。"""
    cfg = _spatial_cfg(prob_colored_noise=0.3)
    spec = BackgroundSpec(BackgroundPolicy.DIFFUSE_COHERENCE, geometry_bound=True)
    # 找一个触发未命中的 seed：逐个尝试直到 used_colored_noise=False。
    for seed in range(200):
        result, _, _ = _run_mix(cfg, monkeypatch, spec, seed=seed)
        if not result.used_colored_noise:
            break
    else:
        pytest.fail("200 个 seed 均未触发未命中分支")
    assert result.noise_image_provenance["colored_noise_applied"] is False
    assert "colored_noise" not in (result.audit or {})


def test_spatial_colored_noise_zero_prob_consumes_no_rng(monkeypatch):
    """prob=0 时彩色噪声分支零随机数消耗：调用计数与未声明该语义时一致。"""
    import core.data.simulation.background as background_module

    counts = {"random": 0, "uniform": 0}
    real_random = background_module.rng_random
    real_uniform = background_module.rng_uniform

    def counting_random(rng):
        counts["random"] += 1
        return real_random(rng)

    def counting_uniform(rng, lo, hi):
        counts["uniform"] += 1
        return real_uniform(rng, lo, hi)

    spec = BackgroundSpec(BackgroundPolicy.DIFFUSE_COHERENCE, geometry_bound=True)

    monkeypatch.setattr(background_module, "rng_random", counting_random)
    monkeypatch.setattr(background_module, "rng_uniform", counting_uniform)
    baseline = _run_mix(_spatial_cfg(prob_colored_noise=0.0), monkeypatch, spec)[0]
    counts_zero = dict(counts)

    counts.update({"random": 0, "uniform": 0})
    triggered = _run_mix(_spatial_cfg(prob_colored_noise=1.0), monkeypatch, spec)[0]
    counts_one = dict(counts)

    assert triggered.used_colored_noise is True
    # 触发分支只多消耗 2 次背景级抽样（trigger + colored SNR）；
    # colored 波形合成走 post_augment 内部 rng 路径，不经 background 包装器。
    assert counts_one["random"] - counts_zero["random"] == 1
    assert counts_one["uniform"] - counts_zero["uniform"] == 1
    # 零概率与未命中都不改变观测形状/类型。
    assert baseline.observation.shape == triggered.observation.shape


def test_mono_colored_noise_path_unchanged(monkeypatch):
    """mono（image_geometry=None）仍走原 _mix_acoustic_colored_noise 实现。"""
    speech = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)
    speech_2d = speech[None, :].astype(np.float32)
    noise = np.random.default_rng(99).normal(0.0, 0.1, 4000).astype(np.float32)
    cfg = SimulationConfigs.from_dict(
        {
            "prob_colored_noise": 1.0,
            "colored_noise_snr_range": [10.0, 50.0],
            "noise_mix_prob": 0.0,
            "snr_low": 0.0,
            "snr_high": 10.0,
            "snr_reference": "all_speech",
            "reuse_noise": False,
        },
        strict=True,
    )
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = cfg
    dataset.export_simulation_stems = True
    dataset._noise_single_reuse_cache = None
    dataset._noise_dual_reuse_cache = None
    monkeypatch.setattr(
        dataset,
        "_mix_acoustic_read_additive",
        lambda mode, sr, max_dur, rng, reuse: noise[None, :],
    )
    result = mix_background(
        dataset,
        speech_2d.copy(),
        speech_2d,
        16000,
        speech_2d.shape[1],
        speech_2d.shape[1],
        np.random.default_rng(3),
        True,
        5.0,
        image_geometry=None,
        background_spec=BackgroundSpec(BackgroundPolicy.ADDITIVE_MONO),
    )
    assert result.used_colored_noise is False
    assert result.audit is not None
    # mono 路径沿用 _mix_acoustic_colored_noise：audit 记录颜色与 SNR（无 spatial 标记）。
    assert result.audit["colored_noise"]["color"] in {"white", "pink", "brown", "equalized"}
    assert "spatial" not in result.audit["colored_noise"]
