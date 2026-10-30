"""BSRNN 谱域 Adapter 与 WeSep dc11cb18 的固定同权重回归。"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tasks.tse.models.modules.separator.adapters import BSRNNSeparatorAdapter
from tasks.tse.models.modules.separator.bsrnn import BSRNN


pytestmark = [pytest.mark.unit]


_SPECTRAL_REPR = torch.tensor(
    [
        [
            [
                [0.10, -0.20, 0.30],
                [0.40, 0.50, -0.60],
                [0.70, -0.80, 0.90],
                [1.00, 1.10, -1.20],
            ],
            [
                [-0.30, 0.20, 0.10],
                [0.60, -0.50, 0.40],
                [-0.90, 0.80, -0.70],
                [1.20, -1.10, 1.00],
            ],
        ]
    ],
    dtype=torch.float32,
)
_REFERENCE_SPEC = torch.complex(
    torch.tensor(
        [[[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.7, 0.6], [0.8, -0.4, 0.1]]]
    ),
    torch.tensor(
        [[[-0.5, 0.6, 0.2], [0.1, -0.7, 0.3], [0.9, -0.8, 0.4], [-0.2, 0.5, -0.6]]]
    ),
)

# 这些张量由固定上游 commit 的 BandSplit → SubbandNorm →
# BSRNN_Separator → BandMasker 直接生成；测试本身不依赖上游 checkout。
_UPSTREAM_FIXTURES = {
    False: {
        "seed": 2301,
        "latent": [
            -0.7274198532,
            1.0070447922,
            -0.6610959172,
            -0.3810972571,
            0.7684681416,
            -0.6863232851,
            0.2667104006,
            0.4838292599,
            0.0154209733,
            0.0572060272,
            -0.5925596356,
            0.7680672407,
        ],
        "separated": [
            -0.4653261900,
            1.2144349813,
            -0.3787122667,
            -0.0719002262,
            1.1351184845,
            -0.4053866565,
            0.4661233723,
            0.8296054602,
            0.2116515040,
            0.3700895607,
            -0.2466840744,
            1.1872494221,
        ],
        "output": [
            -0.0631826445,
            0.0650374070,
            0.1013145298,
            -0.1607581228,
            -0.0349451527,
            -0.0509029403,
            -0.0310633592,
            -0.0297824331,
            -0.0637443364,
            0.0199891906,
            0.0440831855,
            -0.0087970588,
            0.0461171232,
            0.0588879474,
            -0.0695216358,
            -0.1686789542,
            0.0205010977,
            -0.0191902947,
            -0.0468967445,
            -0.1004169062,
            0.1018847600,
            0.0190833993,
            -0.0637175366,
            -0.0149506629,
        ],
    },
    True: {
        "seed": 2302,
        "latent": [
            0.2280627489,
            0.3866367638,
            0.1081652343,
            0.4252405763,
            -0.7703578472,
            1.2663780451,
            -0.3814128637,
            0.0733379126,
            -0.7957094908,
            0.6211559176,
            -0.5591370463,
            -0.2048030794,
        ],
        "separated": [
            0.2288998663,
            0.2787245810,
            -0.0229901895,
            0.6681114435,
            -0.5282729864,
            1.5922137499,
            -0.4118821025,
            -0.1047547832,
            -0.9505946040,
            0.9263663888,
            -0.2439311594,
            0.2035956681,
        ],
        "output": [
            0.0843537748,
            -0.0853733569,
            -0.0003006179,
            0.0663135424,
            0.0442992672,
            0.0670419335,
            0.0658098534,
            0.0456926301,
            0.1573857814,
            -0.1429434717,
            -0.0569850504,
            0.0407573059,
            0.0069149975,
            -0.0388730206,
            -0.3008116484,
            -0.1904224455,
            -0.0271238871,
            -0.0128456047,
            -0.1990825236,
            0.0208768975,
            0.1096315309,
            0.0208422653,
            -0.0444310792,
            0.1408137381,
        ],
    },
}


def _adapter(causal: bool, *, input_dim: int = 2, num_repeat: int = 1):
    fixture = _UPSTREAM_FIXTURES[causal]
    torch.manual_seed(fixture["seed"])
    return BSRNNSeparatorAdapter(
        sample_rate=16000,
        n_fft=6,
        input_dim=input_dim,
        feature_dim=2,
        num_repeat=num_repeat,
        causal=causal,
        band_width=(2, 2),
    ).eval()


@pytest.mark.parametrize("causal", [False, True])
def test_fixed_upstream_spectral_stages_and_output_match(causal):
    adapter = _adapter(causal)
    fixture = _UPSTREAM_FIXTURES[causal]

    with torch.no_grad():
        latent = adapter.encode(_SPECTRAL_REPR)
        separated = adapter.separate(latent)
        rendered = adapter.render(separated, _REFERENCE_SPEC)
        forwarded = adapter(_SPECTRAL_REPR, _REFERENCE_SPEC)

    expected_latent = torch.tensor(fixture["latent"]).view_as(latent)
    expected_separated = torch.tensor(fixture["separated"]).view_as(separated)
    expected_output = torch.tensor(fixture["output"]).view_as(rendered)
    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=2e-7)
    torch.testing.assert_close(separated, expected_separated, rtol=0, atol=2e-7)
    torch.testing.assert_close(rendered, expected_output, rtol=0, atol=2e-7)
    torch.testing.assert_close(forwarded, expected_output, rtol=0, atol=2e-7)
    assert rendered.shape == (1, 4, 3, 2)


@pytest.mark.parametrize("causal", [False, True])
def test_zero_initial_states_are_identical_to_implicit_lstm_zeros(causal):
    adapter = _adapter(causal)
    latent = adapter.encode(_SPECTRAL_REPR)
    hidden = adapter.feature_dim * 2
    zero = latent.new_zeros(latent.shape[0], hidden)
    initial_states = [
        {"band": (zero, zero), "communication": (zero, zero)}
        for _ in range(adapter.num_blocks)
    ]

    with torch.no_grad():
        implicit = adapter.separate(latent)
        explicit = adapter.separate(latent, initial_states=initial_states)

    torch.testing.assert_close(explicit, implicit, rtol=0, atol=0)


class _LatentFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, latent):
        return latent * self.scale


class _BlockConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = 0

    def forward(self, latent, condition):
        self.calls += 1
        return latent + self.scale * condition


class _ModularOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = _adapter(False, num_repeat=2)
        self.latent_fusion = _LatentFusion()
        self.block_conditioners = nn.ModuleList(
            [_BlockConditioner() for _ in range(self.adapter.num_blocks)]
        )

    def forward(self, spectral_repr, reference_spec, condition):
        return self.adapter(
            spectral_repr,
            reference_spec,
            latent_fusion=self.latent_fusion,
            block_conditioners=self.block_conditioners,
            block_conditions=condition,
        )


def test_latent_and_per_block_condition_hooks_keep_single_parameter_owner():
    model = _ModularOwner()
    latent = model.adapter.encode(_SPECTRAL_REPR)
    condition = torch.ones_like(latent)

    output = model(_SPECTRAL_REPR, _REFERENCE_SPEC, condition)
    output.square().mean().backward()

    assert output.shape == (1, 4, 3, 2)
    assert [layer.calls for layer in model.block_conditioners] == [1, 1]
    assert model.latent_fusion.scale.grad is not None
    assert all(layer.scale.grad is not None for layer in model.block_conditioners)
    names_and_parameters = list(model.named_parameters(remove_duplicate=False))
    assert len({id(parameter) for _, parameter in names_and_parameters}) == len(
        names_and_parameters
    )
    assert all(
        not name.startswith("adapter.latent_fusion")
        and not name.startswith("adapter.block_conditioners")
        for name, _ in names_and_parameters
    )


@pytest.mark.parametrize("causal", [False, True])
def test_existing_bsrnn_default_forward_remains_legacy_stream_path(causal):
    torch.manual_seed(101 + int(causal))
    model = BSRNN(
        sr=16000,
        win=512,
        stride=128,
        feature_dim=4,
        num_repeat=2,
        causal=causal,
        nspk=1,
        spec_dim=2,
    ).eval()
    latent = torch.randn(1, model.nband, model.feature_dim, 3)

    with torch.no_grad():
        expected, _ = model.separator.forward_stream(
            latent, [None] * len(model.separator.separation)
        )
        actual = model.separator(latent)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all("eps" not in key for key in model.state_dict())
    for block in model.separator.separation:
        assert block.band_rnn.norm.eps == pytest.approx(1e-5)
        assert block.band_comm.norm.eps == pytest.approx(1e-5)


def test_adapter_uses_upstream_float32_epsilon_without_changing_legacy_default():
    adapter = _adapter(False)
    upstream_eps = torch.finfo(torch.float32).eps

    assert adapter.eps == upstream_eps
    for block in adapter.separator.separation:
        assert block.band_rnn.norm.eps == upstream_eps
        assert block.band_comm.norm.eps == upstream_eps
    assert BSRNN(feature_dim=2, num_repeat=1).separator.separation[0].band_rnn.norm.eps == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("initial_states", "error", "match"),
    [
        ([None, None], ValueError, "initial_states must contain 1"),
        ({"unknown": (torch.zeros(1, 4), torch.zeros(1, 4))}, ValueError, "未知键"),
        ({"band": torch.zeros(1, 4)}, ValueError, "h0, c0"),
        (
            {"band": (torch.zeros(1, 3), torch.zeros(1, 3))},
            ValueError,
            r"\[1, 4\]",
        ),
        (
            {"communication": (torch.zeros(1, 4, dtype=torch.float64), torch.zeros(1, 4, dtype=torch.float64))},
            TypeError,
            "same device and dtype|相同 device 和 dtype",
        ),
    ],
)
def test_invalid_initial_states_are_rejected(initial_states, error, match):
    adapter = _adapter(False)
    latent = adapter.encode(_SPECTRAL_REPR)

    with pytest.raises(error, match=match):
        adapter.separate(latent, initial_states=initial_states)


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda adapter: adapter.encode(torch.zeros(1, 2, 5, 3)), "F must equal 4"),
        (
            lambda adapter: adapter.render(
                torch.zeros(1, adapter.nband, adapter.feature_dim, 3),
                torch.zeros(1, 4, 2, dtype=torch.complex64),
            ),
            "reference_spec must be complex",
        ),
        (
            lambda adapter: adapter(
                _SPECTRAL_REPR,
                torch.zeros(1, 4, 3),
            ),
            "complex tensor",
        ),
    ],
)
def test_invalid_spectral_shapes_are_rejected(operation, match):
    with pytest.raises((TypeError, ValueError), match=match):
        operation(_adapter(False))


def test_condition_lists_must_match_blocks_and_conditioners():
    adapter = _adapter(False)
    latent = adapter.encode(_SPECTRAL_REPR)
    condition = torch.ones_like(latent)

    with pytest.raises(ValueError, match="block_conditions must contain 1"):
        adapter.separate(latent, block_conditions=[condition, condition])
    with pytest.raises(ValueError, match="without a conditioner"):
        adapter.separate(latent, block_conditions=condition)
    with pytest.raises(ValueError, match="requires a condition"):
        adapter.separate(latent, block_conditioners=[_BlockConditioner()])


def test_adapter_exposes_full_sequence_causal_path_only():
    adapter = _adapter(True)

    assert adapter.causal is True
    assert not hasattr(adapter, "init_streaming_state")
    assert not hasattr(adapter, "forward_stream_chunk")
