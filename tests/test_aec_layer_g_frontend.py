"""AEC input/deployment frontend 的直接单测。"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import tasks.aec.inference.aec_frontend as frontend_module
from tasks.aec.inference.aec_frontend import (
    AECDeploymentFrontend,
    AECFrontendConfig,
    resolve_aec_frontend_config,
)


@pytest.mark.unit
def test_config_defaults_and_strict_gcc_contract():
    config = AECFrontendConfig.from_dict({})

    assert config.delay_alignment.enabled is False
    assert config.delay_alignment.algorithm == "gcc_phat"
    assert config.input_routing == "mic_farend"
    assert config.linear_aec is None

    with pytest.raises(ValueError, match="unknown key"):
        AECFrontendConfig.from_dict({"unexpected": True})
    with pytest.raises(ValueError, match="fixed to 'gcc_phat'"):
        AECFrontendConfig.from_dict(
            {"delay_alignment": {"algorithm": "matched_filter"}}
        )


@pytest.mark.unit
def test_config_linear_defaults_strict_schema_and_routing_consistency():
    config = AECFrontendConfig.from_dict(
        {
            "linear_aec": {"backend": "pfdkf_torch"},
            "input_routing": "mic_farend_error",
        }
    )
    assert config.linear_aec is not None
    assert config.linear_aec.N == 5

    for legacy_key in ("enable_tdc", "tdc_max_delay_ms", "tdc_win_seconds"):
        with pytest.raises(ValueError, match="unknown key"):
            AECFrontendConfig.from_dict(
                {
                    "linear_aec": {"backend": "pfdkf", legacy_key: True},
                    "input_routing": "mic_farend_error",
                }
            )
    with pytest.raises(ValueError, match="requires a LinearAEC backend"):
        AECFrontendConfig.from_dict({"input_routing": "error_farend"})
    with pytest.raises(ValueError, match="would not be consumed"):
        AECFrontendConfig.from_dict({"linear_aec": {"backend": "pfdkf"}})

    no_op = AECFrontendConfig.from_dict(
        {"linear_aec": {"backend": "none"}, "input_routing": "mic_farend"}
    )
    assert no_op.linear_aec is None


@pytest.mark.unit
def test_resolver_accepts_only_top_level_frontend_config():
    config = resolve_aec_frontend_config(
        {
            "aec_frontend": {
                "delay_alignment": {"enabled": True, "frame_size": 80},
            }
        }
    )

    assert config.delay_alignment.enabled is True
    assert config.delay_alignment.algorithm == "gcc_phat"
    assert config.delay_alignment.frame_size == 80

    legacy_configs = [
        {"aec_stream_align": {"enabled": True}},
        {"inference": {"aec_stream_align": {"enabled": True}}},
        {"data": {"aec_stream_align": {"enabled": True}}},
        {
            "data": {
                "train": {
                    "simulation_cfg": {"aec_stream_align": {"enabled": True}}
                }
            }
        },
        {
            "data": {
                "dynamic": {
                    "simulation_cfg": {"aec_stream_align": {"enabled": True}}
                }
            }
        },
    ]
    for legacy_config in legacy_configs:
        with pytest.raises(
            ValueError,
            match="legacy aec_stream_align.*aec_frontend.delay_alignment.enabled",
        ):
            resolve_aec_frontend_config(legacy_config)

    with pytest.raises(ValueError, match="configure them under aec_frontend"):
        resolve_aec_frontend_config({"delay_alignment": {"enabled": True}})


@pytest.mark.unit
def test_resolver_rejects_top_level_linear_aec():
    for config in (
        {"linear_aec": {"backend": "pfdkf"}},
        {
            "aec_frontend": {"delay_alignment": {"enabled": False}},
            "linear_aec": {"backend": "pfdkf"},
        },
    ):
        with pytest.raises(ValueError, match="top-level linear_aec.*aec_frontend.linear_aec"):
            resolve_aec_frontend_config(config)


@pytest.mark.unit
def test_disabled_frontend_is_valid_region_passthrough_without_backend_creation(
    monkeypatch,
):
    monkeypatch.setattr(
        frontend_module,
        "create_delay_estimator",
        lambda _cfg: pytest.fail("disabled alignment must not create an estimator"),
    )
    monkeypatch.setattr(
        frontend_module,
        "LinearAECFrontend",
        lambda *_args, **_kwargs: pytest.fail(
            "absent linear_aec must not create a backend"
        ),
    )
    frontend = AECDeploymentFrontend()
    mic = torch.tensor([[1.0, 2.0, 99.0], [3.0, 88.0, 77.0]], requires_grad=True)
    farend = torch.tensor([[4.0, 5.0, float("nan")], [6.0, 66.0, 55.0]])
    mic_before = mic.detach().clone()

    aligned, error, mask, facts = frontend.process_batch(
        mic, farend, torch.tensor([2, 1]), 16000
    )

    torch.testing.assert_close(
        aligned, torch.tensor([[4.0, 5.0, 0.0], [6.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(mic.detach(), mic_before)
    assert aligned.requires_grad is False
    assert error is None
    assert mask.tolist() == [False, False]
    assert facts["delay_alignment"]["backend_identity"] is None
    assert json.loads(json.dumps(facts)) == facts


class _FakeEstimator:
    def __init__(self, created_index: int):
        self.created_index = created_index
        self.backend_identity = "cpp" if created_index == 0 else "python"
        self.delay_samples = created_index + 1
        self.delay_ms = float(created_index + 1) / 16.0
        self.ref_frames: list[np.ndarray] = []
        self.mic_frames: list[np.ndarray] = []

    def push_ref(self, frame):
        self.ref_frames.append(np.asarray(frame).copy())

    def push_mic(self, frame):
        self.mic_frames.append(np.asarray(frame).copy())

    def get_aligned_ref_frame(self, frame_size):
        return self.ref_frames[-1][:frame_size].copy()


@pytest.mark.unit
def test_enabled_alignment_uses_independent_estimators_and_reports_identity(monkeypatch):
    estimators: list[_FakeEstimator] = []
    seen_configs = []

    def create(config):
        seen_configs.append(config)
        estimator = _FakeEstimator(len(estimators))
        estimators.append(estimator)
        return estimator

    monkeypatch.setattr(frontend_module, "create_delay_estimator", create)
    frontend = AECDeploymentFrontend(
        {"delay_alignment": {"enabled": True, "frame_size": 2}}
    )
    mic = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    farend = mic + 10

    aligned, error, mask, facts = frontend.process_batch(
        mic, farend, torch.tensor([5, 3]), 16000
    )

    assert len(estimators) == 2
    assert [len(item.ref_frames) for item in estimators] == [3, 2]
    assert all(config.algorithm == "gcc_phat" for config in seen_configs)
    torch.testing.assert_close(aligned[0], farend[0])
    torch.testing.assert_close(aligned[1], torch.tensor([15.0, 16.0, 17.0, 0.0, 0.0]))
    assert error is None
    assert mask.tolist() == [False, False]
    assert facts["delay_alignment"]["backend_identity"] == "mixed"
    assert facts["delay_alignment"]["backend_identity_per_utterance"] == [
        "cpp",
        "python",
    ]
    assert facts["delay_alignment"]["delay_samples"] == [1, 2]


@pytest.mark.unit
def test_real_delay_factory_reports_cpp_or_explicit_python_fallback():
    frontend = AECDeploymentFrontend(
        {
            "delay_alignment": {
                "enabled": True,
                "frame_size": 160,
                "buffer_ms": 100.0,
                "residual_ms": 20.0,
                "min_obs_ms": 20.0,
            }
        }
    )
    mic = torch.zeros(1, 320)
    farend = torch.zeros(1, 320)

    _, _, _, facts = frontend.process_batch(
        mic, farend, torch.tensor([320]), 16000
    )

    assert facts["delay_alignment"]["backend_identity"] in {"cpp", "python"}


@pytest.mark.unit
def test_cpu_linear_backend_is_per_utterance_and_padding_is_zero(monkeypatch):
    frontends = []

    class FakeLinearFrontend:
        def __init__(self, cfg, sample_rate):
            self.cfg = cfg
            self.sample_rate = sample_rate
            frontends.append(self)

        def process(self, mic, ref):
            return mic - ref

    monkeypatch.setattr(frontend_module, "LinearAECFrontend", FakeLinearFrontend)
    frontend = AECDeploymentFrontend(
        {
            "linear_aec": {"backend": "pfdkf"},
            "input_routing": "error_farend",
        }
    )
    mic = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    farend = torch.tensor([[1.0, 2.0, 30.0], [3.0, 40.0, 50.0]])

    aligned, error, mask, facts = frontend.process_batch(
        mic, farend, torch.tensor([2, 1]), 16000
    )

    assert len(frontends) == 2
    assert error is not None
    torch.testing.assert_close(error, torch.tensor([[3.0, 3.0, 0.0], [4.0, 0.0, 0.0]]))
    torch.testing.assert_close(aligned, torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]]))
    assert mask.tolist() == [True, True]
    assert facts["linear_aec"]["backend_identity"] == "pfdkf"


@pytest.mark.unit
def test_torch_pfdkf_batches_on_input_device_and_enables_res(monkeypatch):
    import tasks.aec.data.pfdkf_torch as pfdkf_torch

    instances = []

    class FakeRES(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.process_calls = []
            instances.append(self)

        def process(self, mic, ref):
            self.process_calls.append((mic.device, mic.shape))
            return mic - ref

    monkeypatch.setattr(pfdkf_torch, "PFDKFRESTorch", FakeRES)
    frontend = AECDeploymentFrontend(
        {
            "linear_aec": {
                "backend": "pfdkf_torch",
                "use_res": True,
                "N": 5,
            },
            "input_routing": "mic_farend_error",
        }
    )
    mic = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [4.0, 3.0, 2.0]])
    farend = torch.tensor([[1.0, 2.0, 30.0], [3.0, 4.0, 50.0], [1.0, 40.0, 50.0]])

    _, error, _, facts = frontend.process_batch(
        mic, farend, torch.tensor([2, 2, 1]), 16000
    )

    assert len(instances) == 1
    assert instances[0].kwargs["N"] == 5
    assert instances[0].kwargs["res"] is True
    assert instances[0].process_calls == [
        (mic.device, torch.Size([2, 2])),
        (mic.device, torch.Size([1, 1])),
    ]
    assert error is not None
    torch.testing.assert_close(
        error, torch.tensor([[4.0, 4.0, 0.0], [5.0, 5.0, 0.0], [3.0, 0.0, 0.0]])
    )
    assert facts["linear_aec"] == {
        "enabled": True,
        "backend_identity": "pfdkf_torch",
        "use_res": True,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mic", "farend", "lengths", "message"),
    [
        (torch.zeros(3), torch.zeros(3), torch.tensor([3]), "shape \\[B, T\\]"),
        (
            torch.zeros(1, 3),
            torch.zeros(1, 2),
            torch.tensor([2]),
            "shape mismatch",
        ),
        (
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.tensor([4]),
            "must be in",
        ),
        (
            torch.tensor([[float("nan"), 0.0]]),
            torch.zeros(1, 2),
            torch.tensor([1]),
            "NaN or Inf",
        ),
    ],
)
def test_batch_input_validation_fails_closed(mic, farend, lengths, message):
    with pytest.raises(ValueError, match=message):
        AECDeploymentFrontend().process_batch(mic, farend, lengths, 16000)


@pytest.mark.unit
def test_nonfinite_backend_output_fails_closed(monkeypatch):
    class BadLinearFrontend:
        def __init__(self, *_args, **_kwargs):
            pass

        def process(self, mic, ref):
            return np.full_like(mic, np.nan)

    monkeypatch.setattr(frontend_module, "LinearAECFrontend", BadLinearFrontend)
    frontend = AECDeploymentFrontend(
        {
            "linear_aec": {"backend": "speex"},
            "input_routing": "mic_farend_error",
        }
    )

    with pytest.raises(RuntimeError, match="invalid output"):
        frontend.process_batch(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([2]), 16000
        )
