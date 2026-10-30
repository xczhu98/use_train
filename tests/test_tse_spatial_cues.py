from __future__ import annotations

import math

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
from tasks.tse.data.spatial_cues import (
    align_spatial_cue_to_frames,
    azimuth_elevation_from_positions,
    build_static_spatial_cue_from_scene,
    build_track_spatial_cue_from_scene,
)
from tasks.tse.models.modules.separator.bsrnn import BSRNN_Separator
from tasks.tse.models.modules.spatial.fixed_features import (
    CDFSpatialFeature,
    DeltaSTFTFeature,
    IPDFeature,
    SDFSpatialFeature,
)
from tasks.tse.models.modules.spatial.spatial_frontend import SpatialFrontend


pytestmark = pytest.mark.unit


def _complex_spectrum(batch: int, channels: int, bins: int, frames: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(71)
    real = torch.randn(batch, channels, bins, frames, generator=generator)
    imag = torch.randn(batch, channels, bins, frames, generator=generator)
    return torch.complex(real, imag)


def _upstream_ipd(spectrum: torch.Tensor, pairs: tuple[tuple[int, int], ...]) -> torch.Tensor:
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    phase = (
        torch.angle(spectrum[:, pair_tensor[:, 0]])
        - torch.angle(spectrum[:, pair_tensor[:, 1]])
    )
    return torch.remainder(phase + math.pi, 2 * math.pi) - math.pi


def _upstream_tpd(
    direction_rad: torch.Tensor,
    pairs: tuple[tuple[int, int], ...],
    positions_m: tuple[tuple[float, float, float], ...],
    *,
    sample_rate: int,
    n_fft: int,
    sound_speed_mps: float,
) -> torch.Tensor:
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    positions = torch.tensor(positions_m)
    baselines = positions[pair_tensor[:, 0]] - positions[pair_tensor[:, 1]]
    azimuth = direction_rad[:, 0]
    elevation = direction_rad[:, 1]
    direction = torch.stack(
        (
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ),
        dim=-1,
    )
    delays_m = torch.einsum("btc,pc->bpt", direction, baselines)
    angular = 2 * math.pi * torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return angular.view(1, 1, -1, 1) * delays_m.unsqueeze(2) / sound_speed_mps


def _static_scene(
    *,
    microphone_positions: tuple[tuple[float, float, float], ...],
    target_position: tuple[float, float, float],
    reference_channel: int,
    room_extra: dict[str, object] | None = None,
) -> AcousticScenePlan:
    array = resolve_microphone_array(
        custom_geometry={"positions_m": microphone_positions},
        reference_channel=reference_channel,
    )
    assert array is not None
    return AcousticScenePlan(
        sample_rate=16000,
        num_samples=1600,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={"scene_uid": "scene-7"},
        room={
            "coordinate_system": "cartesian_xyz_m",
            "sound_speed": 340.0,
            **(room_extra or {}),
        },
        array=array,
        actors=(AcousticActorPlan("target", "target", position_m=target_position),),
    )


def test_static_geometry_uses_array_origin_and_horizontal_elevation() -> None:
    positions = ((0.96, 1.0, 1.0), (1.04, 1.0, 1.0))
    scene = _static_scene(
        microphone_positions=positions,
        target_position=(1.0, 2.0, 2.0),
        reference_channel=0,
    )

    cue = build_static_spatial_cue_from_scene(scene, actor_uid="target")

    assert cue.actor_uid == "target"
    assert cue.scene_uid == "scene-7"
    assert cue.coordinate_frame == "cartesian_xyz_m"
    assert cue.sound_speed_mps == 340.0
    assert cue.array_origin_m == (1.0, 1.0, 1.0)
    assert cue.reference_channel == 0
    torch.testing.assert_close(
        cue.value,
        torch.tensor([math.pi / 2, math.pi / 4]),
        rtol=0,
        atol=1e-6,
    )
    assert cue.to_tse_cue().representation == "static"
    explicit_origin = build_static_spatial_cue_from_scene(
        scene,
        actor_uid="target",
        array_origin_m=(1.0, 1.0, 0.0),
    )
    assert explicit_origin.array_origin_m == (1.0, 1.0, 0.0)
    torch.testing.assert_close(
        explicit_origin.value,
        torch.tensor([math.pi / 2, math.atan2(2.0, 1.0)]),
        rtol=0,
        atol=1e-6,
    )
    with pytest.raises(KeyError, match="missing"):
        build_static_spatial_cue_from_scene(scene, actor_uid="missing")


def test_static_doa_is_invariant_to_reference_channel_and_global_translation() -> None:
    positions = ((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0))
    target = (1.0, 2.0, 1.0)
    reference_zero = build_static_spatial_cue_from_scene(
        _static_scene(
            microphone_positions=positions,
            target_position=target,
            reference_channel=0,
        ),
        actor_uid="target",
    )
    reference_one = build_static_spatial_cue_from_scene(
        _static_scene(
            microphone_positions=positions,
            target_position=target,
            reference_channel=1,
        ),
        actor_uid="target",
    )
    shift = (4.0, -3.0, 2.5)
    shifted_positions = tuple(
        tuple(position[axis] + shift[axis] for axis in range(3))
        for position in positions
    )
    shifted_target = tuple(target[axis] + shift[axis] for axis in range(3))
    translated = build_static_spatial_cue_from_scene(
        _static_scene(
            microphone_positions=shifted_positions,
            target_position=shifted_target,
            reference_channel=1,
        ),
        actor_uid="target",
    )

    torch.testing.assert_close(reference_zero.value, reference_one.value)
    torch.testing.assert_close(reference_zero.value, translated.value)
    assert reference_zero.reference_channel == 0
    assert reference_one.reference_channel == 1


def test_scene_geometry_rejects_unmodeled_array_rotation() -> None:
    scene = _static_scene(
        microphone_positions=((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0)),
        target_position=(1.0, 0.0, 0.0),
        reference_channel=0,
        room_extra={"array_rotation_rad": [0.0, 0.0, 0.0]},
    )
    with pytest.raises(ValueError, match="rotation is not modeled"):
        build_static_spatial_cue_from_scene(scene, actor_uid="target")


def test_position_conversion_rejects_actor_at_array_origin() -> None:
    with pytest.raises(ValueError, match="differ"):
        azimuth_elevation_from_positions((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_track_alignment_uses_explicit_renderer_segment_sample_anchors() -> None:
    cue = torch.tensor(
        [[[0.0, 1.0, 2.0], [0.1, 0.2, 0.3]]],
        dtype=torch.float32,
    )
    anchors = torch.tensor([[0, 100, 250]], dtype=torch.int64)
    frames = torch.tensor([0, 99, 100, 249, 250, 400], dtype=torch.int64)

    aligned = align_spatial_cue_to_frames(
        cue,
        representation="track",
        present=torch.tensor([True]),
        frame_sample_indices=frames,
        sample_anchors=anchors,
        lengths=torch.tensor([3], dtype=torch.int32),
    )

    torch.testing.assert_close(
        aligned,
        torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]]]
        ),
    )
    with pytest.raises(ValueError, match="point-count resizing"):
        align_spatial_cue_to_frames(
            cue,
            representation="track",
            present=torch.tensor([True]),
            frame_sample_indices=frames,
            sample_anchors=anchors,
            lengths=torch.tensor([3], dtype=torch.int32),
            track_alignment="linear",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="start at sample 0"):
        align_spatial_cue_to_frames(
            cue,
            representation="track",
            present=torch.tensor([True]),
            frame_sample_indices=frames,
            sample_anchors=torch.tensor([[1, 100, 250]], dtype=torch.int64),
            lengths=torch.tensor([3], dtype=torch.int32),
        )


def test_track_geometry_uses_actor_identity_and_renderer_crop_boundaries() -> None:
    timebase = TrajectoryTimebase(
        sample_rate=16000,
        num_samples=400,
        hop_samples=100,
        frame_count=4,
    )
    target_positions = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[-1.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    other_positions = np.asarray(
        [[[2.0, 0.0, 0.0]]] * timebase.frame_count,
        dtype=np.float32,
    )
    array = resolve_microphone_array(
        custom_geometry={"positions_m": ((-0.05, 0.0, 0.0), (0.05, 0.0, 0.0))},
        reference_channel=1,
    )
    assert array is not None
    scene = AcousticScenePlan(
        sample_rate=16000,
        num_samples=400,
        need_noise=False,
        rir=RIRPlan.disabled(),
        source_gain=SourceGainPlan.disabled(),
        noise=NoisePlan.disabled(),
        identity={"scene_uid": "moving-scene"},
        room={"coordinate_system": "cartesian_xyz_m", "sound_speed": 341.0},
        array=array,
        timebase=timebase,
        actors=(
            AcousticActorPlan(
                "other",
                "interference",
                position_m=(2.0, 0.0, 0.0),
                trajectory=SourceTrajectory(other_positions, timebase),
            ),
            AcousticActorPlan(
                "target",
                "target",
                position_m=(1.0, 0.0, 0.0),
                trajectory=SourceTrajectory(target_positions, timebase),
            ),
        ),
    )

    track = build_track_spatial_cue_from_scene(
        scene,
        actor_uid="target",
        output_start_sample=150,
        output_num_samples=200,
    )

    cue = track.to_tse_cue()
    assert cue.representation == "track"
    assert cue.length == 3
    torch.testing.assert_close(cue.timebase, torch.tensor([0, 50, 150]))
    torch.testing.assert_close(
        cue.value,
        torch.tensor(
            [
                [math.pi / 2, math.pi, -math.pi / 2],
                [0.0, math.pi / 4, 0.0],
            ]
        ),
        rtol=0,
        atol=1e-6,
    )
    assert track.source_sample_anchors == (100, 200, 300)
    assert track.sound_speed_mps == 341.0
    assert track.reference_channel == 1
    assert track.metadata["renderer"] == "moving_segment_overlap_add"
    aligned = align_spatial_cue_to_frames(
        cue.value.unsqueeze(0),
        representation="track",
        present=torch.tensor([True]),
        frame_sample_indices=torch.tensor([0, 49, 50, 149, 150, 199]),
        sample_anchors=cue.timebase.unsqueeze(0),
        lengths=torch.tensor([cue.length], dtype=torch.int32),
    )
    torch.testing.assert_close(
        aligned[0, 0],
        torch.tensor(
            [math.pi / 2, math.pi / 2, math.pi, math.pi, -math.pi / 2, -math.pi / 2]
        ),
        rtol=0,
        atol=1e-6,
    )
    with pytest.raises(KeyError, match="missing"):
        build_track_spatial_cue_from_scene(scene, actor_uid="missing")


def test_fixed_features_match_pinned_upstream_formulas_for_complex_and_ri() -> None:
    sample_rate = 16000
    n_fft = 16
    sound_speed = 340.0
    pairs = ((0, 1), (2, 0))
    positions = ((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.03, 0.0))
    spectrum = _complex_spectrum(2, 3, n_fft // 2 + 1, 5)
    direction = torch.tensor(
        [
            [[0.2] * 5, [0.1] * 5],
            [[-1.0] * 5, [-0.3] * 5],
        ],
        dtype=torch.float32,
    )
    present = torch.ones(2, dtype=torch.bool)
    upstream_ipd = _upstream_ipd(spectrum, pairs)
    upstream_tpd = _upstream_tpd(
        direction,
        pairs,
        positions,
        sample_rate=sample_rate,
        n_fft=n_fft,
        sound_speed_mps=sound_speed,
    )

    ipd = IPDFeature(pairs, 3)
    delta = DeltaSTFTFeature(pairs, 3)
    cdf = CDFSpatialFeature(
        pairs=pairs,
        mic_positions_m=positions,
        sample_rate=sample_rate,
        n_fft=n_fft,
        speed_of_sound_mps=sound_speed,
    )
    sdf = SDFSpatialFeature(
        pairs=pairs,
        mic_positions_m=positions,
        sample_rate=sample_rate,
        n_fft=n_fft,
        speed_of_sound_mps=sound_speed,
    )

    torch.testing.assert_close(ipd.compute(spectrum), upstream_ipd)
    torch.testing.assert_close(cdf.compute(spectrum, direction, present), torch.cos(upstream_ipd - upstream_tpd))
    torch.testing.assert_close(sdf.compute(spectrum, direction, present), torch.sin(upstream_ipd - upstream_tpd))
    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    upstream_delta = spectrum[:, pair_tensor[:, 0]] - spectrum[:, pair_tensor[:, 1]]
    expected_delta = torch.cat([upstream_delta.real, upstream_delta.imag], dim=1)
    torch.testing.assert_close(delta.compute(spectrum), expected_delta)

    ri = torch.view_as_real(spectrum)
    torch.testing.assert_close(ipd.compute(ri), upstream_ipd)
    torch.testing.assert_close(delta.compute(ri), expected_delta)


def test_plane_wave_direction_and_missing_presence_keep_observed_features() -> None:
    sample_rate = 16000
    n_fft = 16
    sound_speed = 340.0
    positions = ((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0))
    pairs = ((0, 1),)
    azimuth = 0.0
    elevation = 0.0
    direction_vector = torch.tensor([1.0, 0.0, 0.0])
    frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    microphone_phase = torch.stack(
        [
            2 * math.pi * frequencies * torch.dot(direction_vector, torch.tensor(position)) / sound_speed
            for position in positions
        ]
    )
    one_frame = torch.polar(torch.ones_like(microphone_phase), microphone_phase)
    spectrum = one_frame[None, :, :, None].expand(2, -1, -1, 4).clone()
    static_cue = torch.tensor([[azimuth, elevation], [0.0, 0.0]])
    present = torch.tensor([True, False])
    frontend = SpatialFrontend(
        representation="static",
        features={"ipd": {}, "cdf": {}, "sdf": {}, "delta_stft": {}},
        pairs=pairs,
        num_mics=2,
        sample_rate=sample_rate,
        n_fft=n_fft,
        mic_positions_m=positions,
        speed_of_sound_mps=sound_speed,
    )

    output = frontend(spectrum, spatial_cue=static_cue, present=present)

    assert output.spectral.shape == (2, 5, n_fft // 2 + 1, 4)
    torch.testing.assert_close(output.spectral_by_name["cdf"][0], torch.ones_like(output.spectral_by_name["cdf"][0]), atol=2e-6, rtol=0)
    torch.testing.assert_close(output.spectral_by_name["sdf"][0], torch.zeros_like(output.spectral_by_name["sdf"][0]), atol=2e-6, rtol=0)
    assert torch.count_nonzero(output.spectral_by_name["cdf"][1]) == 0
    assert torch.count_nonzero(output.spectral_by_name["sdf"][1]) == 0
    torch.testing.assert_close(output.spectral_by_name["ipd"][1], output.spectral_by_name["ipd"][0])
    torch.testing.assert_close(output.spectral_by_name["delta_stft"][1], output.spectral_by_name["delta_stft"][0])
    assert torch.count_nonzero(output.spectral_by_name["delta_stft"][1]) > 0


def test_all_missing_direction_accepts_none_and_keeps_ipd_delta() -> None:
    spectrum = _complex_spectrum(2, 2, 9, 3)
    frontend = SpatialFrontend(
        representation="static",
        features={"ipd": {}, "cdf": {}, "delta_stft": {}},
        pairs=((0, 1),),
        num_mics=2,
        sample_rate=16000,
        n_fft=16,
        mic_positions_m=((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0)),
        speed_of_sound_mps=343.0,
    )

    output = frontend(
        spectrum,
        spatial_cue=None,
        present=torch.tensor([False, False]),
    )

    assert torch.count_nonzero(output.spectral_by_name["cdf"]) == 0
    assert torch.count_nonzero(output.spectral_by_name["ipd"]) > 0
    assert torch.count_nonzero(output.spectral_by_name["delta_stft"]) > 0


def test_cyclic_and_initstate_shapes_match_bsrnn_initial_state_interface() -> None:
    mix_dim = 8
    num_layers = 2
    frames = 5
    frontend = SpatialFrontend(
        representation="track",
        features={
            "cyc_doaemb": {
                "alpha": 20.0,
                "embedding_dim": 8,
                "use_elevation": True,
                "fusion": "multiply",
            },
            "initstate_emb": {
                "alpha": 20.0,
                "embedding_dim": 8,
                "use_elevation": True,
            },
        },
        pairs=((0, 1),),
        num_mics=2,
        sample_rate=16000,
        n_fft=16,
        mix_dim=mix_dim,
        num_layers=num_layers,
    )
    spectrum = _complex_spectrum(2, 2, 9, frames)
    cue = torch.tensor(
        [
            [[0.0, 0.5], [0.0, 0.1]],
            [[1.0, 1.2], [0.2, 0.3]],
        ]
    )
    output = frontend(
        spectrum,
        spatial_cue=cue,
        present=torch.tensor([True, False]),
        frame_sample_indices=torch.tensor([0, 64, 128, 192, 256]),
        sample_anchors=torch.tensor([[0, 128], [0, 128]], dtype=torch.int64),
        lengths=torch.tensor([2, 0], dtype=torch.int32),
    )

    assert output.spectral.shape == (2, 0, 9, frames)
    assert output.cyclic_embedding is not None
    assert output.cyclic_embedding.shape == (2, 1, mix_dim, frames)
    assert output.cyclic_fusion == "multiply"
    assert torch.count_nonzero(output.cyclic_embedding[1]) == 0
    assert output.initial_states is not None
    assert len(output.initial_states) == num_layers
    for state in output.initial_states:
        for key in ("band", "communication"):
            for tensor in state[key]:
                assert tensor.shape == (2, mix_dim * 2)
                assert torch.count_nonzero(tensor[1]) == 0

    separator = BSRNN_Separator(
        nband=3,
        num_repeat=num_layers,
        feature_dim=mix_dim,
        causal=False,
    )
    separator_input = torch.randn(2, 3, mix_dim, frames)
    separator_output = separator(separator_input, initial_states=output.initial_states)
    assert separator_output.shape == separator_input.shape


def test_spatial_frontend_unknown_config_and_missing_physics_fail_fast() -> None:
    common = dict(
        representation="static",
        pairs=((0, 1),),
        num_mics=2,
        sample_rate=16000,
        n_fft=16,
    )
    with pytest.raises(ValueError, match="unknown features"):
        SpatialFrontend(features={"mystery": {}}, **common)
    with pytest.raises(ValueError, match="does not accept options"):
        SpatialFrontend(features={"ipd": {"enabled": True}}, **common)
    with pytest.raises(ValueError, match="actual speed_of_sound_mps"):
        SpatialFrontend(
            features={"cdf": {}},
            mic_positions_m=((-0.04, 0.0, 0.0), (0.04, 0.0, 0.0)),
            **common,
        )
    with pytest.raises(ValueError, match="unknown keys"):
        SpatialFrontend(
            features={"cyc_doaemb": {"width": 8}},
            **common,
        )
