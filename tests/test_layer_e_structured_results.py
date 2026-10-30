"""Layer E selector/execution 结构化结果与兼容 Adapter 单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import core.data.dynamic_dataset_base as dynamic_base
import core.data.simulation.urgent_impl as urgent_impl
from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.simulation.urgent_impl import CodecCompressionResult
from core.data.simulation_configs import SimulationConfigs


pytestmark = [pytest.mark.unit]


class _ChoiceRng:
    def __init__(self, *values):
        self._values = iter(values)

    def choice(self, *args, **kwargs):
        return next(self._values)


def _dataset(*, enabled: bool = True, names=("clipping",)):
    dataset = BaseDynamicMixingDataset.__new__(BaseDynamicMixingDataset)
    dataset.sim_cfg = SimulationConfigs.from_dict(
        {
            "prob_channel_enabled": enabled,
            "num_augmentations": {0: 0.5, 1: 0.5},
            "noisy_post_mix_degradations_enabled": False,
        },
        strict=True,
    )
    dataset._aug_names = list(names)
    dataset._aug_weights = np.full(len(names), 1.0 / len(names))
    return dataset


def test_disabled_and_zero_draw_do_not_create_selected_operations():
    waveform = np.linspace(-1.0, 1.0, 16, dtype=np.float32)

    disabled = _dataset(enabled=False)._apply_channel_degradation_result(
        waveform, 16000, len(waveform), _ChoiceRng(), False
    )
    assert disabled.selected == ()
    assert disabled.operations == ()
    assert disabled.skip_reason == "disabled"

    zero_draw = _dataset()._apply_channel_degradation_result(
        waveform, 16000, len(waveform), _ChoiceRng(0), False
    )
    assert zero_draw.selected == ()
    assert zero_draw.operations == ()
    assert zero_draw.skip_reason == "draw_not_selected"


def test_selected_conflict_is_skipped_without_losing_selector_fact():
    waveform = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    result = _dataset()._apply_channel_degradation_result(
        waveform,
        16000,
        len(waveform),
        _ChoiceRng(1, np.asarray(["clipping"])),
        True,
    )

    assert result.selected == ("clipping",)
    assert result.applied == ()
    assert result.skipped == ("clipping",)
    assert result.operations[0].reason == "conflict"
    assert result.observation_changed is False


@pytest.mark.parametrize(
    ("name", "setup", "reason"),
    [
        (
            "bandwidth_limitation",
            lambda dataset: setattr(
                dataset, "_gen_bandwidth_params", lambda sr, rng: (None, None)
            ),
            "no_lower_sr",
        ),
        (
            "packet_loss",
            lambda dataset: setattr(
                dataset,
                "_gen_packet_loss_indices",
                lambda length, sr, rng: ([], 20),
            ),
            "no_packets",
        ),
    ],
)
def test_selected_precondition_skip_reasons(name, setup, reason):
    dataset = _dataset(names=(name,))
    setup(dataset)
    waveform = np.linspace(-1.0, 1.0, 16, dtype=np.float32)

    result = dataset._apply_channel_degradation_result(
        waveform,
        16000,
        len(waveform),
        _ChoiceRng(1, np.asarray([name])),
        False,
    )

    assert result.operations[0].reason == reason
    assert result.observation_changed is False


def test_no_change_is_skipped_but_real_clipping_is_applied():
    dataset = _dataset()
    dataset._gen_clipping_params = lambda rng: (0.25, 0.75)

    constant = np.ones(16, dtype=np.float32)
    no_change = dataset._apply_channel_degradation_result(
        constant,
        16000,
        len(constant),
        _ChoiceRng(1, np.asarray(["clipping"])),
        False,
    )
    assert no_change.operations[0].reason == "no_change"
    assert no_change.operation_applied("clipping") is False

    ramp = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    applied = dataset._apply_channel_degradation_result(
        ramp,
        16000,
        len(ramp),
        _ChoiceRng(1, np.asarray(["clipping"])),
        False,
    )
    assert applied.operations[0].reason is None
    assert applied.operation_applied("clipping") is True
    assert applied.observation_changed is True


def test_codec_result_reports_empty_dependency_error_and_no_change(monkeypatch):
    empty = urgent_impl.codec_compression_result(
        np.zeros((1, 0), dtype=np.float32), 16000, "mp3"
    )
    assert empty.reason == "empty"

    waveform = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(1, -1)
    monkeypatch.setattr(urgent_impl, "HAS_TORCHAUDIO_IO", False)
    with pytest.warns(UserWarning):
        unavailable = urgent_impl.codec_compression_result(waveform, 16000, "mp3")
    assert unavailable.reason == "dependency_unavailable"

    class _Effector:
        def __init__(self, **kwargs):
            pass

        def apply(self, samples, fs):
            return samples

    monkeypatch.setattr(urgent_impl, "HAS_TORCHAUDIO_IO", True)
    monkeypatch.setattr(urgent_impl, "CodecConfig", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(urgent_impl, "AudioEffector", _Effector, raising=False)
    unchanged = urgent_impl.codec_compression_result(waveform, 16000, "mp3")
    assert unchanged.reason == "no_change"
    assert unchanged.observation_changed is False

    class _BrokenEffector(_Effector):
        def apply(self, samples, fs):
            raise RuntimeError("codec failed")

    monkeypatch.setattr(urgent_impl, "AudioEffector", _BrokenEffector)
    failed = urgent_impl.codec_compression_result(waveform, 16000, "mp3")
    assert failed.reason == "codec_error"
    assert failed.observation_changed is False


def test_dataloader_worker_codec_uses_external_ffmpeg_backend(monkeypatch):
    """worker 内不能触发 torchaudio native codec，必须隔离到子进程。"""
    waveform = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(1, -1)
    calls = []

    def _run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is False
        if "pcm_f32le" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=np.arange(16, dtype=np.float32).tobytes(),
                stderr=b"",
            )
        return SimpleNamespace(returncode=0, stdout=b"encoded", stderr=b"")

    monkeypatch.setattr(urgent_impl, "_in_dataloader_worker", lambda: True)
    monkeypatch.setattr(urgent_impl, "HAS_TORCHAUDIO_IO", True)
    monkeypatch.setattr(
        urgent_impl,
        "AudioEffector",
        lambda **_kwargs: pytest.fail("worker must not construct AudioEffector"),
    )
    monkeypatch.setattr(urgent_impl.subprocess, "run", _run)

    result = urgent_impl.codec_compression_result(
        waveform, 16000, "mp3", qscale=5
    )

    assert len(calls) == 2
    assert calls[0][0] == "ffmpeg" and calls[1][0] == "ffmpeg"
    np.testing.assert_array_equal(
        result.waveform, np.arange(16, dtype=np.float32).reshape(1, -1)
    )
    assert result.observation_changed is True


def test_layer_f_redundant_skip_requires_actual_layer_e_application(monkeypatch):
    dataset = _dataset(names=("codec",))
    dataset._gen_codec_params = lambda rng: ("mp3", None, 1)
    observed = []

    def post_adapter(noisy, sr, cfg, **kwargs):
        observed.append((kwargs["skip_codec"], kwargs["skip_clip"]))
        return noisy

    monkeypatch.setattr(dynamic_base, "apply_post_augmentations", post_adapter)
    waveform = np.linspace(-0.5, 0.5, 16, dtype=np.float32)

    monkeypatch.setattr(
        dynamic_base,
        "codec_compression_result",
        lambda *args, **kwargs: CodecCompressionResult(
            args[0], False, "dependency_unavailable"
        ),
    )
    legacy = dataset._apply_layer_ef(
        waveform,
        16000,
        len(waveform),
        _ChoiceRng(1, np.asarray(["codec"])),
        False,
    )
    assert legacy[1:] == (["codec"], False, False)
    assert observed[-1] == (False, False)

    monkeypatch.setattr(
        dynamic_base,
        "codec_compression_result",
        lambda *args, **kwargs: CodecCompressionResult(
            args[0] + 0.125, True, None
        ),
    )
    legacy = dataset._apply_layer_ef(
        waveform,
        16000,
        len(waveform),
        _ChoiceRng(1, np.asarray(["codec"])),
        False,
    )
    assert legacy[1:] == (["codec"], True, False)
    assert observed[-1] == (True, False)
