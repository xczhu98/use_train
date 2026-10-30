"""RTS target derivation and spatial renderer integration."""

import numpy as np
import pytest

from core.data.acoustic_scene_plan import (
    AcousticActorPlan,
    AcousticScenePlan,
    NoisePlan,
    RIRPlan,
    SourceGainPlan,
)
from core.data.simulation.array_profiles import (
    ArrayTopology,
    MicrophoneArrayProfile,
    ResolvedMicrophoneArray,
)
from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
from core.data.simulation.spatial_scene import SpatialSceneRenderer, SpatialSceneRequest
from core.data.spatial import derive_rts_rir_bank, render_spatial_scene

pytestmark = pytest.mark.unit


def test_rts_window_matches_reference_mic_exponential_decay():
    rirs = np.zeros((1, 2, 10), dtype=np.float32)
    rirs[0, 0, 2:] = 1.0
    rirs[0, 1, 1:] = 0.5

    shortened = derive_rts_rir_bank(
        rirs,
        reference_channel=0,
        original_t60_sec=0.8,
        target_t60_sec=0.1,
        sample_rate=1000,
        time_after_max_sec=0.002,
    )

    q = 3.0 / (0.1 * 1000) - 3.0 / (0.8 * 1000)
    expected_window = np.ones(10, dtype=np.float64)
    expected_window[4:] = 10.0 ** (-q * np.arange(6))
    np.testing.assert_allclose(shortened[0, 0, 2:] / rirs[0, 0, 2:], expected_window[2:])
    np.testing.assert_allclose(shortened[0, 1, 1:] / rirs[0, 1, 1:], expected_window[1:])


def test_rts_is_unchanged_when_target_t60_is_not_shorter():
    rirs = np.random.default_rng(2).normal(size=(3, 2, 24)).astype(np.float32)

    shortened = derive_rts_rir_bank(
        rirs,
        reference_channel=1,
        original_t60_sec=0.2,
        target_t60_sec=0.5,
        sample_rate=8000,
    )

    np.testing.assert_array_equal(shortened, rirs)


def test_static_spatial_renderer_returns_separate_rts_tracks():
    positions = ((0.0, 0.0, 0.0),)
    array = ResolvedMicrophoneArray(
        MicrophoneArrayProfile("rts-fixture", ArrayTopology(positions)), 0
    )
    room = {"room_uid": "rts-room", "rt60": 0.8}
    geometry = {"positions_m": [list(positions[0])], "reference_channel": 0}
    provenance = {
        "scene_uid": "rts-scene",
        "source_order": ["target"],
        "room": room,
    }
    full_rirs = np.zeros((1, 1, 64), dtype=np.float32)
    full_rirs[0, 0, 2:] = np.exp(-np.arange(62, dtype=np.float32) / 12.0)
    full = SpatialRIRResult(full_rirs, geometry, provenance, "fixture")
    early = SpatialRIRResult(full_rirs[:, :, :16], geometry, provenance, "fixture")
    plan = AcousticScenePlan(
        sample_rate=1000,
        num_samples=16,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={"scene_uid": "rts-scene"},
        room=room,
        array=array,
        actors=(AcousticActorPlan("target", "target", 0, (1.0, 1.0, 1.0)),),
    )
    request = SpatialSceneRequest(
        sources=np.pad(np.ones((1, 1), dtype=np.float32), ((0, 0), (0, 15))),
        sample_rate=1000,
        provider=type("Provider", (), {"sample_scene_tracks": lambda self, *_args, **_kwargs: SpatialRIRTrackSet(full=full, early=early)})(),
        source_uids=("target",),
        track_kinds=("full", "early"),
        early_boundary=16,
        num_samples=16,
        rts_target_t60_sec=0.15,
        rts_time_after_max_sec=0.0,
        scene_plan=plan,
    )

    rendered = SpatialSceneRenderer().render(request)

    expected_rirs = derive_rts_rir_bank(
        full_rirs,
        reference_channel=0,
        original_t60_sec=0.8,
        target_t60_sec=0.15,
        sample_rate=1000,
        time_after_max_sec=0.0,
    )
    expected = render_spatial_scene(
        request.sources,
        full_rirs,
        early_boundary=16,
        reference_channel=0,
        num_samples=16,
        rts_rirs=expected_rirs,
    )
    assert rendered.rts_tracks is not None
    np.testing.assert_allclose(rendered.rts_tracks, expected.rts_tracks)

    from types import SimpleNamespace

    from core.data.simulation.scene_renderer import render_speech_scene
    from core.data.simulation_configs import SimulationConfigs
    from core.data.simulation_views import project_simulation_view
    from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy

    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "custom_geometry": {"positions_m": [list(positions[0])]},
            "reference_channel": 0,
            "target_type": "rts",
            "prob_channel_enabled": False,
        },
        strict=True,
    )
    dataset = SimpleNamespace(
        sim_cfg=cfg,
        export_simulation_stems=False,
        _simulation_view=project_simulation_view(cfg, "se"),
        _apply_channel_degradation_result=lambda *_args, **_kwargs: None,
    )
    speech_scene = render_speech_scene(
        dataset,
        (
            AcousticActor(
                uid="target",
                role=AcousticRole.TARGET_SPEECH,
                rir_policy=RIRPolicy.SPATIAL_SOURCE,
                waveform=request.sources[0],
            ),
        ),
        1000,
        np.random.default_rng(5),
        sir_range=None,
        sir_mode="none",
        need_noise=False,
        num_samples=16,
        pre_rendered_spatial=rendered,
        sources_preplaced=True,
    )
    target = speech_scene.bundle.require("target/0")
    np.testing.assert_allclose(target, rendered.rts_tracks[0, 0])
    assert not np.allclose(target, rendered.early_tracks[0, 0])


def test_rts_target_is_rejected_for_non_se_spatial_tasks():
    from core.data.simulation_configs import SimulationConfigs
    from core.data.simulation_views import project_simulation_view

    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 0,
            "target_type": "rts",
        },
        strict=True,
    )
    with pytest.raises(ValueError, match="does not preserve target_type='rts'"):
        project_simulation_view(cfg, "tse")
