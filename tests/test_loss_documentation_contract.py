"""Loss 文档事实与迁移措辞回归测试。"""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dbgnet_mbrcl_config_keeps_paper_weights_and_32k_schedule():
    """DBGNet MBRCL 配方数值必须与文档声明的 32 kHz 工程移植一致。"""
    from core.losses.build import build_loss
    from core.utils.config import load_merged_config

    config_path = ROOT / (
        "configs/aec/microsoft_challenge/dbgnet/"
        "dbgnet_small_32k_10ms_delay80ms_loss_mbrcl.yaml"
    )
    cfg = load_merged_config(config_path)
    components = cfg["loss"]["components"]
    assert [component["type"] for component in components] == [
        "SISNRLoss",
        "MultiResSTFTLoss",
        "MultiBandRelativeContrastiveLoss",
    ]
    assert [component["weight"] for component in components] == [0.5, 1.0, 1.0]
    assert components[1]["args"] == {
        "fft_sizes": [1024, 2048, 4096],
        "win_lengths": [640, 1280, 2560],
        "hop_sizes": [320, 640, 1280],
    }
    assert components[2]["args"] == {
        "num_bands": 16,
        "margin": 0.06,
        "eps": 1.0e-8,
        "band_weight": 0.5,
        "envelope_weight": 0.2,
    }

    loss = build_loss(
        cfg["loss"],
        stft_cfg=cfg["stft"],
        sample_rate=cfg["data"]["sample_rate"],
    )
    assert [component.__class__.__name__ for component in loss.losses] == [
        "SISNRLoss",
        "MultiResSTFTLoss",
        "MultiBandRelativeContrastiveLoss",
    ]
    assert [
        (resolution.n_fft, resolution.win_length, resolution.hop_length)
        for resolution in loss.losses[1].losses
    ] == [
        (1024, 640, 320),
        (2048, 1280, 640),
        (4096, 2560, 1280),
    ]
    mbrcl = loss.losses[2]
    assert (
        mbrcl.n_fft,
        mbrcl.win_length,
        mbrcl.hop_length,
        mbrcl.window_name,
        mbrcl.sample_rate,
    ) == (640, 640, 320, "hann", 32000)


pytestmark = [pytest.mark.unit]
