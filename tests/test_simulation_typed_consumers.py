"""Layer A typed view 的共享消费者合同测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation.speech_preparation import SpeechPreparation
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view


pytestmark = pytest.mark.unit


class _RejectConfigReads:
    """验证已投影的 Dataset 不再从原始 sim_cfg 读取 Layer A 字段。"""

    def __getattr__(self, name: str):
        raise AssertionError(f"Layer A read bypassed its typed view: {name}")


def _dataset(config):
    clean = np.array([0.2, -0.4, 0.6, -0.8], dtype=np.float32)
    dataset = SimpleNamespace(
        sim_cfg=config,
        speech_flt={"speech": "speech.wav"},
        _read_one_speech=lambda _path, _sample_rate: clean.copy(),
        repeat_to_max_duration=True,
        repeat_tile_silence_gap=False,
        repeat_tile_silence_gap_range=(0.02, 0.2),
        export_simulation_stems=False,
        _last_simulation_stems=None,
        _hp_taps=None,
        _repeat_export_info=lambda *args: {"repeat_args": args},
        _speech_splice_export_info=lambda cfg, _sr, _n: {
            "enabled": bool(cfg.speech_splice),
            "target_seconds": float(cfg.speech_splice_seconds),
        },
    )
    return dataset


def test_speech_preparation_consumes_layer_a_view_without_changing_rng_or_output():
    config = SimulationConfigs.from_dict(
        {
            "speech_splice": True,
            "speech_splice_seconds": 0.4,
            "silence_pad_prob": 1.0,
            "silence_pad_position": "scene_aware",
            "use_high_pass": False,
        }
    )
    view_dataset = _dataset(_RejectConfigReads())
    view_dataset._simulation_task = "se"
    view_dataset._simulation_view = project_simulation_view(config, "se")
    direct_dataset = _dataset(config)
    view_rng = np.random.default_rng(923)
    direct_rng = np.random.default_rng(923)
    preparation = SpeechPreparation()

    view_result = preparation.prepare_result(
        view_dataset, "speech", 10, 10, view_rng
    )
    direct_result = preparation.prepare_result(
        direct_dataset, "speech", 10, 10, direct_rng
    )

    np.testing.assert_array_equal(view_result.speech_1d, direct_result.speech_1d)
    np.testing.assert_array_equal(view_result.speech_2d, direct_result.speech_2d)
    assert view_result.speech_splice_export == direct_result.speech_splice_export
    assert view_result.repeat_export == direct_result.repeat_export
    assert view_rng.bit_generator.state == direct_rng.bit_generator.state
