"""共享 observation channel dispatcher 与 AEC 空间 E/F 合同测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.data.dynamic_dataset_base import LayerEOperationResult, LayerEResult
from core.data.simulation.channel_effects import (
    aggregate_layer_e_results,
    apply_post_augmentations_channels,
    dispatch_observation_channels,
)
from core.data.simulation_configs import SimulationConfigs
from core.data.waveform_bundle import WaveformBundle
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset
from tasks.aec.data.post_mix_adapter import (
    AECPostMixContext,
    TaskPostMixAdapter,
)


pytestmark = pytest.mark.unit


def _layer_e_result(waveform, *, clipping=False, codec=False):
    operations = tuple(
        LayerEOperationResult(name, True, True)
        for name, enabled in (("clipping", clipping), ("codec", codec))
        if enabled
    )
    return LayerEResult(
        waveform=np.asarray(waveform, dtype=np.float32),
        enabled=bool(operations),
        num_augmentations_drawn=len(operations),
        selected=tuple(operation.name for operation in operations),
        operations=operations,
        observation_changed=bool(operations),
    )


def test_spatial_layer_f_shared_policy_uses_canonical_channel_dispatch():
    cfg = SimpleNamespace(
        noisy_post_mix_degradations_enabled=True,
        prob_drc=0.0, prob_codecs=0.0, prob_clip=1.0,
        clip_pre_drive_db_range=(0.0, 0.0), clip_hard_portion=1.0,
        clip_hard_on_rate=False, clip_rate_range=(0.2, 0.2),
        clip_threshold_db_range=(-6.0, -6.0), clip_soft_slope_range=(2.0, 2.0),
        clip_sigmoid_slope_range=(2.0, 2.0), clip_sox_threshold_db_range=(-6.0, -6.0),
        clip_pedal_threshold_db_range=(-6.0, -6.0), clip_variants=("fixed_rail",),
        clip_variant_probs={"fixed_rail": 1.0}, prob_post_eq=0.0,
        prob_lowpass=0.0, prob_band_reject=0.0, prob_dc_offset=0.0,
        prob_bit_crush=0.0, prob_spectral_holes=0.0, prob_webrtc_ns=0.0,
        prob_webrtc_agc=0.0, prob_bass_boost=0.0, prob_spectral_leakage=0.0,
        prob_loudness_perturb=0.0,
    )
    observation = np.asarray([[0.8, 0.2, -0.8], [0.4, 0.1, -0.4]], np.float32)
    result = apply_post_augmentations_channels(
        observation,
        16000,
        cfg,
        policy="shared",
        rng=np.random.default_rng(7),
        layer_e_results=tuple(_layer_e_result(channel) for channel in observation),
    )

    assert result.waveform.shape == observation.shape
    assert (
        result.channel_facts[0]["clipping"]["pre_drive_db"]
        == result.channel_facts[1]["clipping"]["pre_drive_db"]
    )


def test_spatial_layer_f_rejects_unknown_channel_policy():
    observation = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="channel policy"):
        apply_post_augmentations_channels(
            observation,
            16000,
            SimulationConfigs.from_dict({}),
            policy="broadcast",
            rng=np.random.default_rng(11),
            layer_e_results=tuple(_layer_e_result(channel) for channel in observation),
        )


def test_shared_rng_replays_one_parameter_plan_and_advances_once():
    observation = np.ones((2, 4), dtype=np.float32)
    rng = np.random.default_rng(17)
    expected = np.random.default_rng(17)
    expected_draw = expected.random()
    result = dispatch_observation_channels(
        observation,
        policy="shared",
        rng=rng,
        apply_channel=lambda channel, channel_rng, _index: channel + channel_rng.random(),
    )

    np.testing.assert_allclose(result.waveform[0], result.waveform[1])
    assert result.waveform[0, 0] == pytest.approx(1.0 + expected_draw)
    assert rng.random() == pytest.approx(expected.random())


def test_per_channel_rng_consumption_is_sequential():
    observation = np.ones((2, 2), dtype=np.float32)
    rng = np.random.default_rng(19)
    expected = np.random.default_rng(19)
    first, second = expected.random(2)
    result = dispatch_observation_channels(
        observation,
        policy="per_channel",
        rng=rng,
        apply_channel=lambda channel, channel_rng, _index: channel + channel_rng.random(),
    )

    assert result.waveform[0, 0] == pytest.approx(1.0 + first)
    assert result.waveform[1, 0] == pytest.approx(1.0 + second)
    assert rng.random() == pytest.approx(expected.random())


def test_shared_policy_rejects_data_dependent_rng_consumption():
    with pytest.raises(RuntimeError, match="different RNG counts"):
        dispatch_observation_channels(
            np.ones((2, 2), dtype=np.float32),
            policy="shared",
            rng=np.random.default_rng(23),
            apply_channel=lambda channel, channel_rng, index: channel + (
                channel_rng.random()
                if index == 0
                else (channel_rng.random(), channel_rng.random())[0]
            ),
        )


def test_layer_e_aggregation_keeps_one_clipped_channel_and_one_unclipped_channel():
    observation = np.asarray([[1.0, -1.0], [0.1, -0.1]], dtype=np.float32)
    dispatch = dispatch_observation_channels(
        observation,
        policy="per_channel",
        rng=np.random.default_rng(29),
        apply_channel=lambda channel, _rng, _index: _layer_e_result(
            channel, clipping=float(np.max(np.abs(channel))) > 0.5
        ),
    )
    aggregate = aggregate_layer_e_results(dispatch)

    assert aggregate.operation_applied("clipping") is True
    assert dispatch.channel_facts[0]["operations"][0]["applied"] is True
    assert dispatch.channel_facts[1]["operations"] == []


def test_aec_layer_e_real_clipping_aggregates_channel_dependent_facts():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "spatial_layer_e_policy": "per_channel",
            "prob_channel_enabled": True,
            "augmentations_name": ["clipping"],
            "num_augmentations": {1: 1.0},
        }
    )
    dataset = DynamicMixingDataset.__new__(DynamicMixingDataset)
    dataset.sim_cfg = cfg
    dataset._aug_names = ["clipping"]
    dataset._aug_weights = np.asarray([1.0], dtype=np.float64)
    dataset._gen_clipping_params = lambda rng: (0.25, 0.75)
    observation = np.asarray(
        [[1.0, -1.0, 0.8, -0.8], [0.1, -0.1, 0.1, -0.1]],
        dtype=np.float32,
    )

    result = dataset._apply_channel_degradation_result(
        observation, 16000, observation.shape[-1], np.random.default_rng(41), False
    )

    assert result.operation_applied("clipping") is True
    assert result.operations[0].reason == "channel_dependent"
    assert not np.array_equal(result.waveform[0], observation[0])
    np.testing.assert_array_equal(result.waveform[1], observation[1])


def test_aec_spatial_f_uses_real_lowpass_and_per_channel_layer_e_gates():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "spatial_layer_e_policy": "per_channel",
            "spatial_layer_f_policy": "per_channel",
            "noisy_post_mix_degradations_enabled": True,
            "prob_lowpass": 1.0,
            "lowpass_cutoff_range": [500.0, 500.0],
            "lowpass_order_range": [4, 4],
        }
    )

    class Dataset:
        sim_cfg = cfg

        @staticmethod
        def _apply_channel_degradation_result(waveform, _sr, _length, _rng, _wind):
            return _layer_e_result(waveform)

    context = AECPostMixContext(
        branch="prob_target_only",
        task="aec",
        aec_disk_role=None,
        aec_disk_extras={},
        used_sidechain_wind=False,
        joint_target_rir_mismatch=False,
        sidechain_wind_mismatch_reason=None,
        index=None,
    )
    adapter = TaskPostMixAdapter(
        dataset=Dataset(), rng=np.random.default_rng(31), sample_rate=16000,
        actual_len=256, speech_uid="test", context=context,
    )
    t = np.arange(256, dtype=np.float32) / 16000.0
    mic = np.stack([np.sin(2 * np.pi * 4000 * t), np.sin(2 * np.pi * 200 * t)], axis=0)
    bundle = WaveformBundle(
        {
            "nearend": np.zeros(256, dtype=np.float32),
            "mic": mic,
            "farend": np.zeros(256, dtype=np.float32),
            "echo": np.zeros(256, dtype=np.float32),
            "noise": np.zeros(256, dtype=np.float32),
        }
    )
    state = adapter.state_from_bundle(bundle)
    state = adapter.apply_stage("E", state).state
    state = adapter.apply_stage("F", state).state

    assert state.waveforms["mic"].shape == (2, 256)
    assert not np.allclose(state.waveforms["mic"][0], mic[0])
    assert not np.allclose(state.waveforms["mic"][1], mic[1])


def test_aec_layer_d_reference_gain_follows_spatial_reference_channel():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "spatial_layer_d_policy": "per_channel",
            "peak_normalization_mode": "joint",
            "peak_normalization_scale_range": [0.5, 0.5],
        }
    )

    class Dataset:
        sim_cfg = cfg

        @staticmethod
        def _export_layer_d_stems(**_kwargs):
            pass

    context = AECPostMixContext(
        branch="prob_target_only", task="aec", aec_disk_role=None,
        aec_disk_extras={}, used_sidechain_wind=False,
        joint_target_rir_mismatch=False, sidechain_wind_mismatch_reason=None, index=None,
    )
    adapter = TaskPostMixAdapter(
        dataset=Dataset(), rng=np.random.default_rng(37), sample_rate=16000,
        actual_len=4, speech_uid="test", context=context,
    )
    bundle = WaveformBundle(
        {
            "nearend": np.asarray([0.2, -0.2, 0.2, -0.2], dtype=np.float32),
            "mic": np.asarray([[1.0, -1.0, 1.0, -1.0], [0.25, -0.25, 0.25, -0.25]], dtype=np.float32),
            "farend": np.asarray([0.1, -0.1, 0.1, -0.1], dtype=np.float32),
            "echo": np.asarray([0.1, -0.1, 0.1, -0.1], dtype=np.float32),
            "noise": np.asarray([0.1, -0.1, 0.1, -0.1], dtype=np.float32),
        }
    )
    state = adapter.state_from_bundle(bundle)
    result = adapter.apply_stage("D-normalize", state)
    normalized = result.state

    reference_scale = float(adapter._layer_d_normalize_stage_facts["reference_scale"])
    np.testing.assert_allclose(
        normalized.waveforms["nearend"], bundle.require("nearend") * reference_scale
    )
    assert adapter._layer_d_normalize_stage_facts["policy"] == "per_channel"
