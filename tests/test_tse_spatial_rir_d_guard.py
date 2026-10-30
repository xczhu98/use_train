"""TSE spatial RIR tail 与多目标 post-mix D/E/F 合同。"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.data.dynamic_dataset_base import LayerEResult
from core.data.simulation.observation_post_mix import (
    ObservationPostMixRenderer,
    ObservationPostMixRequest,
)
from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
from core.data.waveform_bundle import WaveformBundle
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset


pytestmark = pytest.mark.unit


class _MethodProvider:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def sample_spatial_scene(
        self, sample_rate, source_count, rng, *, source_uids, normalize
    ):
        self.calls.append(
            (sample_rate, source_count, rng, tuple(source_uids), normalize)
        )
        return self.result

    def sample_scene_tracks(self, plan, *, normalize=True, track_kinds=("full",)):
        self.calls.append((tuple(plan.source_uids), tuple(track_kinds), normalize))
        if isinstance(self.result, SpatialRIRTrackSet):
            return self.result
        return SpatialRIRTrackSet(full=self.result)


def _dataset(provider):
    dataset = object.__new__(TSEDynamicMixingDataset)
    dataset.spatial_rir_provider = provider
    dataset._spatial_rir_provider = None
    dataset._simulation_view = SimpleNamespace(
        shared=SimpleNamespace(
            spatial=SimpleNamespace(
                spatial_enabled=True,
                rir_backend="offline",
                channels=2,
            )
        )
    )
    dataset.sim_cfg = SimpleNamespace(rir_backend="offline")
    return dataset


def test_tse_spatial_rir_consumes_typed_full_tail_without_impulse_probe():
    rir = np.zeros((2, 2, 9), dtype=np.float32)
    rir[0, 0, 0] = 0.25
    rir[0, 1, 8] = -0.75
    rir[1, 0, 4] = 0.5
    provider = _MethodProvider(
        SpatialRIRTrackSet(
            full=SpatialRIRResult(
                rir,
                geometry={"array_profile": "fixture"},
                provenance={"source_order": ["target", "interference"]},
                generator_version="fixture-v1",
            )
        )
    )

    class Plan:
        source_uids = ("target", "interference")
    tracks = provider.sample_scene_tracks(Plan(), normalize=False, track_kinds=("full",))
    output = tracks.full.canonical_rirs

    np.testing.assert_array_equal(output, rir)
    assert output.shape == (2, 2, 9)
    assert provider.calls == [(("target", "interference"), ("full",), False)]


def test_tse_spatial_rir_extracts_full_track_from_typed_track_set():
    geometry = {"array_profile": "fixture"}
    provenance = {"source_order": ["target", "interference"]}
    full = np.zeros((2, 2, 11), dtype=np.float32)
    full[1, 1, -1] = 0.9
    direct = np.zeros_like(full)
    direct[:, :, 0] = 1.0
    provider = _MethodProvider(
        SpatialRIRTrackSet(
            full=SpatialRIRResult(full, geometry, provenance, "fixture-full"),
            direct=SpatialRIRResult(direct, geometry, provenance, "fixture-direct"),
        )
    )

    output = provider.result.full.canonical_rirs

    np.testing.assert_array_equal(output, full)
    assert output.shape[-1] == 11


def _post_mix_policy():
    cfg = SimpleNamespace(
        peak_normalization_mode="joint",
        peak_normalization_scale_range=(0.25, 0.25),
        prob_channel_enabled=False,
        noisy_post_mix_degradations_enabled=False,
    )
    spatial = SimpleNamespace(
        spatial_enabled=True,
        spatial_layer_d_policy="shared",
        spatial_layer_e_policy="shared",
        spatial_layer_f_policy="shared",
        reference_channel=0,
        channels=1,
    )
    policy = SimpleNamespace(
        sim_cfg=cfg,
        _simulation_view=SimpleNamespace(
            shared=SimpleNamespace(
                spatial=spatial,
                peak=SimpleNamespace(
                    peak_normalization_mode="joint",
                    peak_normalization_scale_range=(0.25, 0.25),
                ),
                post_mix=SimpleNamespace(noisy_post_mix_degradations_enabled=False),
            )
        ),
        used_sidechain_wind=False,
    )
    policy._apply_channel_degradation_result = (
        lambda waveform, sample_rate, length, rng, sidechain: LayerEResult(
            waveform=np.asarray(waveform, dtype=np.float32),
            enabled=False,
            num_augmentations_drawn=0,
            selected=(),
            operations=(),
            observation_changed=False,
            skip_reason="test",
        )
    )
    return policy


def test_tse_post_mix_scales_unselected_targets_in_d_only():
    request = ObservationPostMixRequest(
        scene_bundle=WaveformBundle(
            {
                "mixture": np.full((1, 4), 0.5, dtype=np.float32),
                "background": np.full((1, 4), 0.25, dtype=np.float32),
                "source_full_rir/0": np.full((1, 4), 0.25, dtype=np.float32),
                "target/0": np.full(4, 0.25, dtype=np.float32),
                "target/1": np.full(4, 0.125, dtype=np.float32),
            }
        ),
        sample_rate=16000,
        rng=np.random.default_rng(0),
        task_policy=_post_mix_policy(),
        observation_roles=("mixture", "background"),
        # target/1 模拟未选作监督输出的备用目标。
        protected_roles=("target/0",),
        amplitude_roles=("source_full_rir/0",),
    )

    result = ObservationPostMixRenderer().apply(request)

    np.testing.assert_allclose(result.bundle.require("target/0"), 0.125)
    np.testing.assert_allclose(result.bundle.require("target/1"), 0.0625)
    np.testing.assert_allclose(result.bundle.require("source_full_rir/0"), 0.125)
    assert result.facts["adapter"]["D-normalize"]["coupled_roles"] == [
        "target/0",
        "target/1",
    ]
    assert result.facts["adapter"]["E"]["changed_roles"] == []
    assert result.facts["adapter"]["F"]["changed_roles"] == []
