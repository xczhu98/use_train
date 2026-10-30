"""多通道 spatial foundation 的纯合同测试。"""

import numpy as np
import pytest

from core.data.spatial import (
    LayerPolicy,
    SpatialChannelLayout,
    SpatialLayerPolicy,
    TrackProjection,
    apply_spatial_layer_d,
    apply_spatial_layer_ef,
    project_channels,
    project_se_scene,
    project_ss_scene,
    project_tracks,
    render_spatial_scene,
    validate_projection,
)


@pytest.mark.unit
def test_layout_keeps_microphone_channels_and_tracks_independent():
    layout = SpatialChannelLayout(num_channels=4, num_tracks=2, reference_channel=2)
    assert (layout.C, layout.K, layout.reference) == (4, 2, 2)
    signal = np.arange(8).reshape(4, 2)
    np.testing.assert_array_equal(project_channels(signal, layout), signal[2])


@pytest.mark.unit
def test_projection_validates_bounds_and_preserves_track_order():
    layout = SpatialChannelLayout(3, 2, 1)
    projection = TrackProjection.from_mapping({"speech_b": (1, 2), "speech_a": (0, 0)})
    validate_projection(layout, projection)
    assert list(project_tracks({"speech_a": 1, "speech_b": 2}, projection)) == [
        "speech_b",
        "speech_a",
    ]
    with pytest.raises(ValueError, match="exceeds K"):
        validate_projection(layout, TrackProjection.from_mapping({"bad": (2, 0)}))


@pytest.mark.unit
def test_layer_policy_is_explicit_and_coerces_serializable_values():
    policy = SpatialLayerPolicy(layer_d="shared", layer_e="per_channel", layer_f=LayerPolicy.SHARED)
    assert policy == SpatialLayerPolicy(LayerPolicy.SHARED, LayerPolicy.PER_CHANNEL, LayerPolicy.SHARED)


@pytest.mark.unit
def test_source_scene_contract_keeps_layer_a_b_separate_from_spatial_geometry():
    from core.data.simulation.scene_spec import SceneSpec
    from core.data.spatial import SpatialSourceSceneContract

    scene = SceneSpec("prob_target_only", True, False, False, False, 2, task="ss")
    contract = SpatialSourceSceneContract(scene, ("source_a", "source_b"))
    assert contract.K == scene.source_count == 2
    with pytest.raises(ValueError, match="source_names length"):
        SpatialSourceSceneContract(scene, ("source_a",))


@pytest.mark.unit
def test_track_projection_rejects_malformed_pairs_and_layout_types():
    with pytest.raises(ValueError, match="pair"):
        TrackProjection.from_mapping({"speech": (0,)})
    with pytest.raises(TypeError, match="layout"):
        validate_projection(object(), TrackProjection.from_mapping({}))


@pytest.mark.unit
def test_renderer_returns_decomposable_full_early_reference_and_mixture_tracks():
    sources = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rirs = np.array([[[1.0, 0.5], [0.0, 1.0]], [[2.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    result = render_spatial_scene(sources, rirs, early_boundary=1, reference_channel=1)
    assert result.full_tracks.shape == (2, 2, 2)
    np.testing.assert_allclose(result.mixture, result.full_tracks.sum(axis=0))
    np.testing.assert_array_equal(result.reference_tracks, result.full_tracks[:, 1])
    np.testing.assert_allclose(result.mixture[0], [7.0, 10.5])
    np.testing.assert_allclose(result.mixture[1], [3.0, 5.0])


@pytest.mark.unit
def test_layer_d_is_joint_and_ef_supports_per_channel_waveforms_without_target_changes():
    observation = np.arange(6, dtype=np.float32).reshape(2, 3)
    target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scaled = apply_spatial_layer_d(observation, scale=2.0)
    output = apply_spatial_layer_ef(scaled, waveform=lambda channel: channel + 1, per_channel=True)
    np.testing.assert_array_equal(output, scaled + 1)
    _, projected_target = project_se_scene(output, target, reference_channel=1)
    np.testing.assert_array_equal(projected_target, target)
    mixture, targets = project_ss_scene(output, np.stack([target, target]))
    assert mixture.shape == (2, 3) and targets.shape == (2, 3)


@pytest.mark.unit
def test_renderer_keeps_full_direct_and_early_tracks_distinct_with_scene_identity():
    sources = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    full_rirs = np.array([[[1.0, 0.5, 0.25], [0.5, 0.25, 0.125]]], dtype=np.float32)
    direct_rirs = np.array([[[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]]], dtype=np.float32)
    result = render_spatial_scene(
        sources,
        full_rirs,
        early_boundary=1,
        direct_rirs=direct_rirs,
        geometry={"array_profile": "fixture"},
        provenance={"scene_id": "scene-1"},
    )
    assert result.full_tracks.shape == result.early_tracks.shape == result.direct_tracks.shape == (1, 2, 3)
    assert result.has_direct_path
    np.testing.assert_allclose(result.early_tracks[0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.direct_tracks[0, 0], [0.0, 1.0, 0.0])
    assert result.geometry == {"array_profile": "fixture"}
    assert result.provenance == {"scene_id": "scene-1"}


@pytest.mark.unit
def test_renderer_does_not_alias_early_as_direct_when_direct_rir_is_absent():
    sources = np.array([[1.0, 0.0]], dtype=np.float32)
    rirs = np.array([[[1.0, 0.5]]], dtype=np.float32)
    result = render_spatial_scene(sources, rirs, early_boundary=1)
    assert result.direct_tracks is None
    assert not result.has_direct_path


@pytest.mark.unit
def test_spatial_rir_track_set_requires_matching_geometry_and_axes():
    from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet

    full = SpatialRIRResult(np.ones((1, 2, 3), dtype=np.float32), {"array": "a"}, {}, "fixture")
    direct = SpatialRIRResult(np.ones((1, 2, 2), dtype=np.float32), {"array": "a"}, {}, "fixture")
    tracks = SpatialRIRTrackSet(full=full, direct=direct)
    assert tracks.full.canonical_rirs.shape == (1, 2, 3)
    assert tracks.direct.canonical_rirs.shape == (1, 2, 2)
    assert tracks.geometry == {"array": "a"}

    bad_geometry = SpatialRIRResult(np.ones((1, 2, 2), dtype=np.float32), {"array": "b"}, {}, "fixture")
    with pytest.raises(ValueError, match="geometry"):
        SpatialRIRTrackSet(full=full, direct=bad_geometry)

    bad_axes = SpatialRIRResult(np.ones((2, 2, 2), dtype=np.float32), {"array": "a"}, {}, "fixture")
    with pytest.raises(ValueError, match="source/channel"):
        SpatialRIRTrackSet(full=full, direct=bad_axes)
