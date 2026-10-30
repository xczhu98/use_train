"""空间 observation 与 C-1 reference-channel SNR 回归测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation.scene_renderer import render_speech_scene
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.spatial_background import render_diffuse_background
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view

pytestmark = pytest.mark.unit


def _rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def test_diffuse_background_c1_is_calibrated_at_configured_reference_channel():
    source = np.random.default_rng(7).normal(0.0, 0.2, 512).astype(np.float32)
    positions = np.asarray(
        [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]], dtype=np.float64
    )
    rendered = render_diffuse_background(
        source,
        positions,
        16000,
        np.random.default_rng(8),
        reference_channel=1,
    )

    assert _rms(rendered.waveform[1]) == pytest.approx(_rms(source), rel=1e-6)
    assert not np.array_equal(rendered.waveform[0], rendered.waveform[1])


def test_spatial_target_only_keeps_rendered_source_microphone_observation():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 1,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
        }
    )
    dataset = SimpleNamespace(
        sim_cfg=cfg,
        export_simulation_stems=False,
        _simulation_view=project_simulation_view(cfg, "se"),
    )
    dataset._apply_channel_degradation_result = lambda waveform, *args: None
    source = np.zeros(32, dtype=np.float32)
    source[0] = 1.0
    rirs = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)

    scene = render_speech_scene(
        dataset,
        (
            AcousticActor(
                uid="target",
                role=AcousticRole.TARGET_SPEECH,
                rir_policy=RIRPolicy.SPATIAL_SOURCE,
                waveform=source,
            ),
        ),
        8000,
        np.random.default_rng(9),
        sir_range=None,
        sir_mode="none",
        need_noise=False,
        spatial_rirs=rirs,
    )

    observation = scene.bundle.require("mixture")
    target = scene.bundle.require("target/0")
    assert observation.shape == (2, 32)
    assert target.shape == (32,)
    assert not np.array_equal(observation[0], observation[1])
    np.testing.assert_array_equal(target, observation[1])
