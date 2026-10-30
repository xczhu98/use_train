"""SS OnlineSpatialNet：多源输出、几何合同与 SS runtime 集成。"""

from types import SimpleNamespace

import pytest
import torch

from core.data.simulation_configs import SimulationConfigs
from core.models.io_contract import validate_spatial_model_geometry
from core.stft import build_stft_istft
from tasks.ss.model_io import SSModelRuntime, resolve_ss_execution_plan
from tasks.ss.models import build_model

pytestmark = pytest.mark.unit

SS_BASE = dict(
    num_freqs=9,
    n_channel=2,
    num_sources=2,
    dim_hidden=8,
    dim_ffn=8,
    dim_squeeze=2,
    num_layers=1,
    num_heads=2,
    conv_groups=(1, 1),
)

BRANCHES = (
    ("mhsa(3)", {}),
    ("ret(2,not_share_qk)", {"decay": [4, 5]}),
    ("mamba(4,4)", {}),
)


def _build(attention="mhsa(3)", extra=None, **overrides):
    args = {**SS_BASE, "attention": attention, **(extra or {}), **overrides}
    return build_model({"type": "OnlineSpatialNet", "args": args}, num_sources=2).eval()


def _dataset(reference_channel=0):
    return SimpleNamespace(
        sim_cfg=SimulationConfigs.from_dict(
            {
                "spatial_enabled": True,
                "array_profile": "linear_2ch_80mm",
                "reference_channel": reference_channel,
            },
            strict=True,
        )
    )


def test_ss_online_spatialnet_outputs_k_source_spectra():
    torch.manual_seed(7)
    model = _build()
    mixture = torch.randn(1, 2, 9, 5, 2)
    output = model(mixture)
    assert tuple(output["est_sources_spectra"].shape) == (1, 2, 9, 5, 2)


def test_ss_online_spatialnet_keeps_c1_legacy_input():
    model = _build(n_channel=1)
    output = model(torch.randn(1, 9, 5, 2))
    assert tuple(output["est_sources_spectra"].shape) == (1, 2, 9, 5, 2)


@pytest.mark.parametrize("attention,extra", BRANCHES)
def test_ss_online_spatialnet_sequence_matches_streaming(attention, extra):
    torch.manual_seed(11)
    model = _build(attention, extra)
    mixture = torch.randn(1, 2, 9, 4, 2)
    with torch.no_grad():
        full = model(mixture)["est_sources_spectra"]
        cache = model.init_cache(1, torch.device("cpu"), torch.float32)
        frames = []
        for index in range(4):
            frame, cache = model.forward_streaming(mixture[:, :, :, index : index + 1], cache)
            frames.append(frame["est_sources_spectra"])
    torch.testing.assert_close(torch.cat(frames, dim=3), full, rtol=1e-5, atol=1e-6)


def test_ss_online_spatialnet_geometry_contract():
    model = _build()
    validate_spatial_model_geometry(model, _dataset(reference_channel=0))
    with pytest.raises(ValueError, match="reference_channel"):
        validate_spatial_model_geometry(model, _dataset(reference_channel=1))


def test_ss_online_spatialnet_runtime_separates_multichannel_mixture():
    stft_cfg = {
        "n_fft": 16,
        "hop_length": 8,
        "win_length": 16,
        "window": "hamming",
        "center": True,
    }
    model = _build()
    plan = resolve_ss_execution_plan(
        {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}, "stft": stft_cfg},
        model,
    )
    runtime = SSModelRuntime(model, plan=plan, stft_module=build_stft_istft(stft_cfg))
    result = runtime.separate(torch.randn(1, 2, 64), torch.tensor([64]), sample_rate=16000)
    assert tuple(result.sources.shape) == (1, 2, 64)
    assert tuple(result.source_spectra.shape) == (1, 2, 9, 9, 2)


def test_ss_online_spatialnet_rejects_num_sources_conflict():
    with pytest.raises(ValueError, match="num_sources conflicts"):
        build_model(
            {"type": "OnlineSpatialNet", "args": {**SS_BASE, "num_sources": 3}},
            num_sources=2,
        )
