from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.data.acoustic_scene_plan import (
    AcousticActorPlan,
    AcousticScenePlan,
    NoisePlan,
    RIRPlan,
    SourceGainPlan,
)
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.trajectory import SourceTrajectory, TrajectoryTimebase
from tasks.tse.cue_contract import TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.tse.data.scene_dataset import _build_target_spatial_cue
from tasks.tse.model_io import build_tse_model_inputs
from tasks.tse.models.discriminative.tse_modular import TSE_Modular


pytestmark = pytest.mark.unit


class _RuntimeSTFT:
    n_fft = 16
    hop_length = 4

    def __call__(self, waveform):
        return torch.randn(waveform.shape[0], 9, 3, 2, dtype=waveform.dtype)


def _scene(*, moving: bool) -> AcousticScenePlan:
    array = resolve_microphone_array(
        custom_geometry={"positions_m": ((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0))},
        reference_channel=0,
    )
    assert array is not None
    timebase = TrajectoryTimebase(16000, 48, 16, 3) if moving else None
    trajectory = (
        SourceTrajectory(
            np.asarray([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[-1.0, 0.0, 0.0]]], dtype=np.float32),
            timebase,
        )
        if moving
        else None
    )
    return AcousticScenePlan(
        sample_rate=16000,
        num_samples=48,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={"scene_uid": "dynamic-scene"},
        room={"coordinate_system": "cartesian_xyz_m", "sound_speed": 340.0},
        array=array,
        timebase=timebase,
        actors=(
            AcousticActorPlan(
                "target", "target", position_m=(1.0, 0.0, 0.0), trajectory=trajectory
            ),
        ),
    )


def _dataset_stub(scene, representation):
    return SimpleNamespace(
        spatial_representation=representation,
        resolved_microphone_array=scene.array,
        spatial_speed_of_sound_mps=340.0,
    )


def test_dynamic_static_scene_cue_reaches_modular_forward_backward():
    scene = _scene(moving=False)
    cue, metadata = _build_target_spatial_cue(
        _dataset_stub(scene, "static"), scene, "target", scene.num_samples
    )
    assert cue is not None and cue.present and cue.representation == "static"
    assert metadata["actor_uid"] == "target"

    sample = make_tse_sample(
        key="static",
        mixture=torch.randn(2, 48),
        sr=16000,
        length=48,
        targets=[
            make_tse_target(
                target=torch.randn(48),
                cues=TSECues(spatial=cue),
                speaker_id="target",
            )
        ],
        geometry=scene.array,
        timebase=None,
        provenance={"scene_uid": "dynamic-scene"},
    )
    batch = tse_collate_fn(
        [sample], cue_specs=TSECueSpecs(spatial=TSECueSpec("static", (2,)))
    )
    model = TSE_Modular(
        backbone="bsrnn",
        separator={"feature_dim": 4, "num_repeat": 1, "band_width": [4, 5]},
        cues={"spatial": {"representation": "static", "features": {"cdf": {}}}},
        sample_rate=16000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        n_channel=2,
        ref_channel=0,
        mic_positions_m=scene.array.positions_m,
        speed_of_sound_mps=340.0,
    )
    inputs = build_tse_model_inputs(
        model=model,
        stft=_RuntimeSTFT(),
        mixture=batch.mixture,
        sample_rate=16000,
        cues=batch.cues,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
    )
    estimate = model(inputs)["target_estimate_spectra"]
    assert estimate.shape == (1, 9, 3, 2)
    estimate.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dynamic_moving_scene_track_preserves_renderer_timebase_in_model_input():
    scene = _scene(moving=True)
    cue, metadata = _build_target_spatial_cue(
        _dataset_stub(scene, "track"), scene, "target", scene.num_samples
    )
    assert cue is not None and cue.present and cue.representation == "track"
    assert cue.length == 3
    torch.testing.assert_close(cue.timebase, torch.tensor([0, 16, 32]))
    assert metadata["actor_uid"] == "target"

    sample = make_tse_sample(
        key="moving",
        mixture=torch.zeros(2, 48),
        sr=16000,
        length=48,
        targets=[
            make_tse_target(
                target=torch.zeros(48),
                cues=TSECues(spatial=cue),
                speaker_id="target",
            )
        ],
        geometry=scene.array,
        timebase=scene.timebase,
        provenance=metadata,
    )
    batch = tse_collate_fn(
        [sample], cue_specs=TSECueSpecs(spatial=TSECueSpec("track", (2, 1)))
    )
    assert batch.cues.spatial is not None
    assert batch.cues.spatial.lengths.tolist() == [3]
    assert batch.cues.spatial.timebase.tolist() == [[0, 16, 32]]

    model = TSE_Modular(
        backbone="bsrnn",
        separator={"feature_dim": 4, "num_repeat": 1, "band_width": [4, 5]},
        cues={"spatial": {"representation": "track", "features": {"cdf": {}}}},
        sample_rate=16000,
        n_fft=16,
        hop_length=4,
        win_length=16,
        n_channel=2,
        ref_channel=0,
        mic_positions_m=scene.array.positions_m,
        speed_of_sound_mps=340.0,
    )
    inputs = build_tse_model_inputs(
        model=model,
        stft=_RuntimeSTFT(),
        mixture=batch.mixture,
        sample_rate=16000,
        cues=batch.cues,
        lengths=batch.lengths,
        num_targets=batch.num_targets,
    )
    assert inputs["spatial"].shape == (1, 2, 3)
    assert inputs["spatial_lengths"].tolist() == [3]
    assert inputs["spatial_timebase"].tolist() == [[0, 16, 32]]
