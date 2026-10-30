"""NBSS 论文口径 Params/FLOPs 复核 CLI 的合同测试。"""

import sys
from pathlib import Path

import pytest

from tools.verify import verify_nbss_paper_alignment as alignment

pytestmark = pytest.mark.unit


def test_default_configs_exist_and_are_covered_by_paper_values():
    for relative in alignment.DEFAULT_CONFIGS:
        assert Path(alignment._REPO_ROOT, relative).is_file(), relative
        label = Path(relative).stem
        assert label in alignment.PAPER_VALUES, label


def test_frames_follow_paper_protocol():
    cfg = {"data": {"sample_rate": 8000}, "stft": {"hop_length": 128}}
    assert alignment._frames(cfg, 4.0) == 251
    cfg_16k = {"data": {"sample_rate": 16000}, "stft": {"hop_length": 256}}
    assert alignment._frames(cfg_16k, 4.0) == 251


def test_cli_reports_aligned_params_and_flops(capsys, monkeypatch):
    monkeypatch.setattr(
        alignment, "DEFAULT_CONFIGS", ("configs/se/spatial/online_spatialnet_dynamic_mhsa.yaml",)
    )
    monkeypatch.setattr(sys, "argv", ["verify_nbss_paper_alignment"])
    exit_code = alignment.main()
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "online_spatialnet_dynamic_mhsa.yaml" in captured
    assert "failures=0" in captured

    result = alignment._measure("configs/se/spatial/online_spatialnet_dynamic_mhsa.yaml")
    assert result["params"] == 1190898
    assert result["n_channel"] == 6
    assert result["n_fft"] == 256
    assert result["frames"] == 251
    # 论文 oSpatialNet-MSA 报告 23.1 G/s（4 s/8 kHz/6 ch）
    assert result["streaming_gflops"] / 4.0 == pytest.approx(23.1, rel=0.05)
