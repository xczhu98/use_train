"""BaseDynamicMixingDataset 分层组件的最小合同测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.data.simulation.acoustic_renderer import AcousticRenderer
from core.data.simulation.resource_store import ResourceStore
from core.data.simulation.scenario_sampler import ScenarioSampler
from core.data.simulation.speech_preparation import SpeechPreparation


pytestmark = pytest.mark.unit


def test_resource_store_forwards_strict_audio_read():
    captured = {}

    def fake_read(path, sample_rate, strict):
        captured.update(path=path, sample_rate=sample_rate, strict=strict)
        return np.ones(4, dtype=np.float32)

    result = ResourceStore(strict_load=False, speech_reader=fake_read).read_speech(
        "speech.wav", 16000
    )

    assert result.shape == (4,)
    assert captured == {"path": "speech.wav", "sample_rate": 16000, "strict": False}


def test_scenario_sampler_force_branch_does_not_consume_rng():
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    branch, draw = ScenarioSampler.select_branch(
        (("target", 0.5), ("noise", 0.5)),
        rng,
        force_branch="target",
    )

    assert (branch, draw) == ("target", None)
    assert rng.bit_generator.state == before


def test_speech_preparation_single_utterance_uses_resource_hook():
    expected = np.arange(4, dtype=np.float32)
    dataset = SimpleNamespace(
        sim_cfg=SimpleNamespace(speech_splice=True),
        speech_flt={"u": "speech.wav"},
        _read_one_speech=lambda path, sample_rate: expected.copy(),
    )

    result = SpeechPreparation().get_clean_speech(
        dataset, "u", 16000, np.random.default_rng(0), single_utterance=True
    )

    np.testing.assert_array_equal(result, expected)


def test_acoustic_renderer_is_public_component():
    assert callable(AcousticRenderer().render_bundle)
