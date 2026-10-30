"""SSL direct-path target projection contracts."""

import numpy as np
import pytest
import torch

from tasks.ssl.data import SSLProjectionConfig, SSLSceneInput, prepare_ssl_scene, project_ssl_scene
from tasks.ssl.data import ssl_collate_fn


pytestmark = pytest.mark.unit


def _scene(*, moving=False, moving_points=7):
    length = 64
    t = torch.arange(length, dtype=torch.float32)
    source = torch.sin(2 * torch.pi * 4 * t / length)
    direct = torch.stack((source, torch.roll(source, 1))).unsqueeze(0)
    observation = direct[0]
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    source_positions = torch.tensor([[1.0, 0.0, 1.0]])
    if moving:
        source_positions = source_positions.repeat(moving_points, 1, 1)
        source_positions[:, 0, 1] = torch.linspace(-1.0, 1.0, moving_points)
    return SSLSceneInput(observation, direct, positions, source_positions, 16000, {"backend": "fixture"})


def test_projection_uses_spatial_scene_contract_and_keeps_batch_layout():
    item = project_ssl_scene(_scene(), SSLProjectionConfig(n_fft=16, hop_length=8, win_length=16))
    assert item["features"].shape == (7, 9, 4)
    assert item["target_dpipd"].shape == (7, 9, 1, 1, 2)
    assert item["doa"].shape == (7, 1, 2)
    assert item["vad"].shape == (7, 1)
    assert item["input_lengths"] == 7
    assert item["output_lengths"] == 7
    assert item["pair_index"].shape == (1, 2)
    batch = ssl_collate_fn([item])
    batch.validate()


def test_projection_supports_frame_aligned_doa():
    item = project_ssl_scene(_scene(moving=True), SSLProjectionConfig(n_fft=16, hop_length=8, win_length=16))
    assert item["doa"].shape[0] == item["output_lengths"]
    assert item["source_positions"].shape == (item["input_lengths"], 1, 3)
    assert not torch.allclose(item["doa"][0], item["doa"][-1])


def test_projection_interpolates_sparse_moving_positions_to_final_stft_frames():
    item = project_ssl_scene(
        _scene(moving=True, moving_points=4),
        SSLProjectionConfig(n_fft=16, hop_length=8, win_length=16),
    )
    expected = torch.linspace(-1.0, 1.0, item["input_lengths"])
    torch.testing.assert_close(item["source_positions"][:, 0, 1], expected)
    assert item["source_positions"].shape == (item["input_lengths"], 1, 3)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("observation", np.zeros((64,), dtype=np.float32), "observation must have shape"),
        ("direct_tracks", np.zeros((2, 64), dtype=np.float32), "direct_tracks must have shape"),
        ("source_positions", np.zeros((2, 2), dtype=np.float32), "source_positions must have shape"),
    ],
)
def test_scene_input_rejects_ambiguous_shapes(field, value, match):
    values = _scene().__dict__
    values[field] = value
    with pytest.raises(ValueError, match=match):
        values = SSLSceneInput(**values)
        values.tensors()


def test_prepare_scene_rejects_observation_direct_length_mismatch():
    """scene 输入长度不一致时必须报错，不能静默裁短 supervision。"""
    with pytest.raises(ValueError, match="share waveform length"):
        prepare_ssl_scene(
            np.zeros((2, 32), dtype=np.float32),
            np.zeros((1, 2, 16), dtype=np.float32),
            sample_rate=16000,
            mic_positions=np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]], dtype=np.float32),
            source_positions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            reference_channel=0,
            n_fft=16,
            hop_length=8,
            win_length=16,
            window="hamming",
            center=False,
            expected_channels=2,
            expected_tracks=1,
            expected_freqs=9,
            expected_input_size=4,
            backend="online",
            provenance={},
        )
