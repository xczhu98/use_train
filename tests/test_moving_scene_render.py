"""移动 waveform renderer 的独立线性卷积 oracle 与分段时间轴门禁。"""

import numpy as np
import pytest

import core.data.simulation.moving_scene as moving
from core.data.spatial import SpatialRenderResult, render_spatial_scene


pytestmark = pytest.mark.unit


def _oracle(sources, rirs, starts, length):
    """按每个发射 sample 累加脉冲响应，不复用 FFT/分段 renderer。"""
    output = np.zeros((sources.shape[0], rirs.shape[2], length), dtype=np.float64)
    for sample in range(sources.shape[1]):
        point = np.searchsorted(starts, sample, side="right") - 1
        count = min(rirs.shape[-1], length - sample)
        for track in range(sources.shape[0]):
            output[track, :, sample:sample + count] += (
                sources[track, sample] * rirs[point, track, :, :count]
            )
    return output


def _inputs():
    rng = np.random.default_rng(190)
    sources = rng.normal(size=(2, 23))
    full = rng.normal(size=(3, 2, 3, 13))
    direct = np.zeros((3, 2, 3, 7))
    direct[..., 4] = rng.uniform(0.2, 1.0, size=(3, 2, 3))
    return sources, full, direct, np.array([0, 5, 16])


@pytest.mark.parametrize("output_length", [None, 23, 27, 45])
def test_irregular_segments_match_independent_emission_oracle(output_length):
    sources, full, direct, starts = _inputs()
    originals = [value.copy() for value in (sources, full, direct, starts)]
    result = moving.render_moving_spatial_scene(
        sources, full, direct_rirs=direct, frame_samples=starts, early_boundary=6,
        reference_channel=2, num_samples=output_length,
    )
    length = 35 if output_length is None else output_length
    assert isinstance(result, SpatialRenderResult)
    for actual, bank in ((result.full_tracks, full), (result.direct_tracks, direct),
                         (result.early_tracks, full[..., :6])):
        np.testing.assert_allclose(actual, _oracle(sources, bank, starts, length), atol=3e-6, rtol=2e-6)
        assert actual.shape == (2, 3, length)
        assert actual.dtype == np.float32
        assert not actual.flags.writeable
    np.testing.assert_array_equal(result.reference_tracks, result.full_tracks[:, 2])
    np.testing.assert_allclose(result.mixture, result.full_tracks.sum(axis=0))
    assert result.provenance["frame_samples"] == starts.tolist()
    for actual, original in zip((sources, full, direct, starts), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("starts", [[0], [0, 1, 7, 14]])
@pytest.mark.parametrize("early_boundary", [0, 3, 99])
def test_single_point_and_stationary_trajectory_reduce_to_static_convolution(starts, early_boundary):
    rng = np.random.default_rng(80)
    sources = rng.normal(size=(2, 19)).astype(np.float32)
    full = rng.normal(size=(2, 3, 11)).astype(np.float32)
    direct = rng.normal(size=(2, 3, 5)).astype(np.float32)
    static = render_spatial_scene(
        sources, full, direct_rirs=direct, early_boundary=early_boundary,
        num_samples=29, reference_channel=1,
    )
    result = moving.render_moving_spatial_scene(
        sources, np.repeat(full[None], len(starts), axis=0),
        direct_rirs=np.repeat(direct[None], len(starts), axis=0),
        frame_samples=starts, early_boundary=early_boundary, reference_channel=1,
    )
    for name in ("full_tracks", "direct_tracks", "early_tracks", "mixture", "reference_tracks"):
        np.testing.assert_allclose(getattr(result, name), getattr(static, name), atol=3e-6, rtol=2e-6)


def test_changing_rir_keeps_old_segment_tail_and_direct_delay_without_wrap():
    sources = np.zeros((1, 8), dtype=np.float32)
    sources[0, [3, 4, 7]] = [1, 2, 3]
    full = np.zeros((2, 1, 1, 7), dtype=np.float32)
    full[0, 0, 0, [0, 6]] = [1, 5]
    full[1, 0, 0, [0, 6]] = [2, 7]
    direct = np.zeros((2, 1, 1, 4), dtype=np.float32)
    direct[:, 0, 0, 3] = [1, 4]
    result = moving.render_moving_spatial_scene(
        sources, full, direct_rirs=direct, frame_samples=[0, 4], early_boundary=1,
    )
    expected_full = np.zeros(14)
    expected_full[[3, 4, 7, 9, 10, 13]] = [1, 4, 6, 5, 14, 21]
    expected_direct = np.zeros(14)
    expected_direct[[6, 7, 10]] = [1, 8, 12]
    np.testing.assert_allclose(result.full_tracks[0, 0], expected_full, atol=1e-6)
    np.testing.assert_allclose(result.direct_tracks[0, 0], expected_direct, atol=1e-6)
    np.testing.assert_allclose(result.direct_tracks[..., :6], 0, atol=1e-6)
    np.testing.assert_allclose(result.early_tracks[..., 8:], 0, atol=1e-6)
    assert result.full_tracks[0, 0, 9] == pytest.approx(5)
    assert result.full_tracks[0, 0, -1] == pytest.approx(21)


def test_longer_direct_rir_retains_its_tail_and_early_is_independent():
    result = moving.render_moving_spatial_scene(
        np.array([[0.0, 1.0]]), np.array([[[[2.0]]]]),
        direct_rirs=np.array([[[[0.0, 0.0, 0.0, 3.0]]]]),
        frame_samples=[0], early_boundary=0,
    )
    np.testing.assert_allclose(result.direct_tracks[0, 0], [0, 0, 0, 0, 3], atol=1e-6)
    np.testing.assert_array_equal(result.early_tracks, 0)
    np.testing.assert_allclose(result.full_tracks[0, 0], [0, 2, 0, 0, 0], atol=1e-6)


def test_moving_rts_bank_is_rendered_as_a_separate_track():
    sources = np.zeros((1, 8), dtype=np.float32)
    sources[0, [1, 5]] = [1.0, 2.0]
    full = np.ones((2, 1, 1, 4), dtype=np.float32)
    direct = np.zeros_like(full)
    rts = np.zeros_like(full)
    rts[0, 0, 0] = [1.0, 0.5, 0.25, 0.125]
    rts[1, 0, 0] = [2.0, 1.0, 0.5, 0.25]
    starts = np.asarray([0, 4], dtype=np.int64)

    result = moving.render_moving_spatial_scene(
        sources, full, direct_rirs=direct, rts_rirs=rts,
        frame_samples=starts, early_boundary=2, num_samples=8,
    )
    assert result.rts_tracks is not None
    np.testing.assert_allclose(
        result.rts_tracks, _oracle(sources, rts, starts, 8), atol=1e-6
    )


def test_moving_renderer_can_skip_direct_track():
    result = moving.render_moving_spatial_scene(
        np.array([[1.0, 2.0]]), np.array([[[[1.0, 0.5]]]]),
        frame_samples=[0], early_boundary=1,
    )
    assert result.direct_tracks is None
    np.testing.assert_allclose(result.full_tracks[0, 0], [1.0, 2.5, 1.0], atol=1e-6)

def test_fft_workspace_is_bounded_and_block_boundaries_keep_convolution_tail(monkeypatch):
    rng = np.random.default_rng(42)
    sources = rng.normal(size=(1, moving._SOURCE_BLOCK_SAMPLES * 2 + 29))
    full = rng.normal(size=(1, 1, 1, 31))
    direct = full * 0.7
    original_rfft = moving.rfft
    observed = []

    def checked_rfft(signal, *, n):
        observed.append((signal.size, n))
        assert signal.ndim == 1
        assert n <= moving.next_fast_len(moving._SOURCE_BLOCK_SAMPLES + 30, real=True)
        return original_rfft(signal, n=n)

    monkeypatch.setattr(moving, "rfft", checked_rfft)
    result = moving.render_moving_spatial_scene(
        sources, full, direct_rirs=direct, frame_samples=[0], early_boundary=4,
    )
    np.testing.assert_allclose(
        result.full_tracks[0, 0], np.convolve(sources[0], full[0, 0, 0]), atol=4e-6, rtol=2e-6,
    )
    assert len(observed) >= 12


@pytest.mark.parametrize("change, message", [
    ({"sources": np.ones(5)}, "sources"),
    ({"sources": np.ones((2, 0))}, "non-empty"),
    ({"sources": np.ones((1, 23))}, "source K"),
    ({"sources": np.full((2, 23), np.nan)}, "finite"),
    ({"sources": np.ones((2, 23), dtype=complex)}, "real numeric"),
    ({"full_rirs": np.ones((3, 2, 3))}, "full_rirs"),
    ({"full_rirs": np.ones((3, 2, 0, 4))}, "non-empty"),
    ({"direct_rirs": np.ones((2, 2, 3, 7))}, "P/K/C"),
    ({"direct_rirs": np.ones((3, 1, 3, 7))}, "P/K/C"),
    ({"direct_rirs": np.ones((3, 2, 4, 7))}, "P/K/C"),
    ({"direct_rirs": np.full((3, 2, 3, 7), np.inf)}, "finite"),
    ({"frame_samples": [0, 5]}, "integer.*P"),
    ({"frame_samples": [[0, 5, 16]]}, "integer.*P"),
    ({"frame_samples": [0.0, 5.0, 16.0]}, "integer.*P"),
    ({"frame_samples": [False, True, True]}, "integer.*P"),
    ({"frame_samples": [1, 5, 16]}, "start at 0"),
    ({"frame_samples": [0, -1, 16]}, "lie in"),
    ({"frame_samples": [0, 5, 23]}, "lie in"),
    ({"frame_samples": [0, 5, 5]}, "strictly increasing"),
    ({"frame_samples": np.array([0, 8, 3], dtype=np.uint64)}, "strictly increasing"),
    ({"frame_samples": np.array([0, 8, 2**64 - 1], dtype=np.uint64)}, "lie in"),
    ({"early_boundary": -1}, "early_boundary"),
    ({"early_boundary": True}, "early_boundary"),
    ({"reference_channel": 3}, "reference_channel"),
    ({"reference_channel": -1}, "reference_channel"),
    ({"reference_channel": False}, "reference_channel"),
    ({"num_samples": 22}, "num_samples"),
    ({"num_samples": 24.0}, "num_samples"),
    ({"num_samples": True}, "num_samples"),
])
def test_invalid_shapes_values_and_shared_timebase_fail_fast(change, message):
    sources, full, direct, starts = _inputs()
    kwargs = dict(sources=sources, full_rirs=full, direct_rirs=direct, frame_samples=starts, early_boundary=3)
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        moving.render_moving_spatial_scene(**kwargs)
