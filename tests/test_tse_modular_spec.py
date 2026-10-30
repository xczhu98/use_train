"""TSE_Modular 纯配置解析合同测试。"""

from __future__ import annotations

import copy
import importlib
import sys

import pytest

from tasks.tse.models import modular_spec
from tasks.tse.models.modular_spec import PUBLIC_COMBINATIONS, resolve_modular_spec


pytestmark = pytest.mark.unit

_STFT = {
    "n_fft": 512,
    "hop_length": 128,
    "win_length": 512,
    "window": "hann",
    "center": True,
}


def _speaker(*features: str, representation: str = "waveform") -> dict:
    return {
        "representation": representation,
        "features": {name: {} for name in features},
    }


def _spatial(*features: str, representation: str = "static") -> dict:
    return {
        "representation": representation,
        "features": {name: {} for name in features},
    }


def _visual(representation: str = "features") -> dict:
    return {
        "representation": representation,
        "features": {"muse_visual": {}},
    }


def _textual() -> dict:
    return {
        "representation": "phoneme_ids",
        "features": {"dae_kce": {}},
    }


def _kwargs(
    backbone: str,
    cues: dict,
    *,
    separator: dict | None = None,
    n_channel: int = 1,
    sample_rate: int = 16000,
    composition: dict | None = None,
) -> dict:
    result = {
        "backbone": backbone,
        "separator": {} if separator is None else separator,
        "cues": cues,
        "composition": composition,
        "sample_rate": sample_rate,
        "n_channel": n_channel,
        "ref_channel": 0,
    }
    if backbone != "spex_plus":
        result.update(_STFT)
    return result


_PUBLIC_CASES = (
    (
        "TSE_BSRNN_SPK",
        _kwargs("bsrnn", {"speaker": _speaker("tfmap", "context")}),
    ),
    (
        "TSE_BSRNN_SPATIAL",
        _kwargs(
            "bsrnn",
            {"spatial": _spatial("ipd", "cdf", "sdf", "delta_stft")},
            n_channel=4,
        ),
    ),
    (
        "TSE_BSRNN_SPK_SPATIAL",
        _kwargs(
            "bsrnn",
            {
                "speaker": _speaker("tfmap", "context"),
                "spatial": _spatial("ipd", "cdf", "sdf", "delta_stft"),
            },
            n_channel=4,
        ),
    ),
    (
        "TSE_TFGRIDNET_SPK",
        _kwargs("tfgridnet", {"speaker": _speaker("tfmap", "context")}),
    ),
    (
        "TSE_NBC2_SPATIAL",
        _kwargs(
            "nbc2",
            {"spatial": _spatial("ipd", "cdf", "sdf", "delta_stft", "cyc_doaemb")},
            separator={
                "block": {
                    "n_heads": 2,
                    "dropout": 0.1,
                    "conv_kernel_size": 3,
                    "n_conv_groups": 8,
                    "norms": ["LN", "GBN", "GBN"],
                }
            },
            n_channel=4,
        ),
    ),
    (
        "TSE_DPCCN_SPK",
        _kwargs("dpccn", {"speaker": _speaker("spkemb")}),
    ),
    (
        "TSE_SPEX_PLUS",
        _kwargs(
            "spex_plus",
            {"speaker": _speaker("spex_plus")},
            separator={"kernel_sizes": [20, 80, 160], "stride": 10},
        ),
    ),
    (
        "TSE_BSRNN_TEXTUAL",
        _kwargs("bsrnn", {"textual": _textual()}),
    ),
    (
        "TSE_BSRNN_VISUAL",
        _kwargs("bsrnn", {"visual": _visual()}),
    ),
    (
        "TSE_TFGRIDNET_VISUAL",
        _kwargs("tfgridnet", {"visual": _visual("raw")}),
    ),
)


_WIRED_UPSTREAM_COMBINATIONS = frozenset({
    "TSE_BSRNN_SPK", "TSE_BSRNN_SPATIAL", "TSE_BSRNN_SPK_SPATIAL",
})
# textual/visual 是本地接线：沿用上游注册名做组合键，但不声明 upstream parity。
_NO_UPSTREAM_PARITY_COMBINATIONS = frozenset({
    "TSE_BSRNN_TEXTUAL", "TSE_BSRNN_VISUAL", "TSE_TFGRIDNET_VISUAL",
})
_LOCAL_EXTENSION_COMBINATIONS = frozenset({
    "TSE_BSRNN_TEXTUAL", "TSE_BSRNN_VISUAL",
})


def test_public_support_matrix_is_exactly_the_ten_upstream_registrations():
    assert tuple(PUBLIC_COMBINATIONS) == tuple(name for name, _ in _PUBLIC_CASES)
    assert len(PUBLIC_COMBINATIONS) == 10


@pytest.mark.parametrize(("upstream_name", "kwargs"), _PUBLIC_CASES)
def test_public_combinations_only_claim_wired_bsrnn_features(
    upstream_name: str,
    kwargs: dict,
):
    spec = resolve_modular_spec(**kwargs)

    assert spec.upstream_name == (
        None if upstream_name in _NO_UPSTREAM_PARITY_COMBINATIONS else upstream_name
    )
    assert spec.local_extension is (upstream_name in _LOCAL_EXTENSION_COMBINATIONS)
    assert spec.implementation_available is (
        upstream_name in _WIRED_UPSTREAM_COMBINATIONS | _LOCAL_EXTENSION_COMBINATIONS
    )
    assert spec.requires_geometry is ("spatial" in spec.cues)
    if kwargs["backbone"] == "spex_plus":
        assert spec.separator_input_domain == "waveform"
        assert spec.frontend_owner == "model"
        assert spec.stft is None
    else:
        assert spec.separator_input_domain == "spectrum"
        assert spec.frontend_owner == "runtime"
        assert spec.stft is not None
        assert spec.stft.n_fft == 512


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            _kwargs("bsrnn", {"visual": _visual("raw")}),
            "raw video input is not wired",
        ),
        (
            _kwargs("tfgridnet", {"visual": _visual()}),
            "not available",
        ),
    ],
)
def test_visual_raw_video_and_unwired_backbones_stay_out_of_the_gate(
    kwargs: dict,
    match: str,
):
    spec = resolve_modular_spec(**kwargs)

    assert spec.implementation_available is False
    assert spec.local_extension is False
    with pytest.raises(NotImplementedError, match=match):
        resolve_modular_spec(**kwargs, require_implemented=True)


def test_textual_and_visual_are_local_extensions_not_upstream_parity():
    for kwargs in (
        _kwargs("bsrnn", {"textual": _textual()}),
        _kwargs("bsrnn", {"visual": _visual()}),
    ):
        spec = resolve_modular_spec(**kwargs)

        assert spec.upstream_name is None
        assert spec.local_extension is True
        assert spec.implementation_available is True
        assert resolve_modular_spec(
            **kwargs, require_implemented=True
        ).implementation_available


def test_textual_kce_config_passes_through_frontend_keys():
    textual = _textual()
    textual["features"]["dae_kce"] = {
        "checkpoint": "ckpt/epoch_149.pt",
        "model_config": {"audio_net_config": {"num_mel_bins": 80}},
        "fbank": {"num_mel_bins": 80},
        "eps": 1e-5,
        "adapter": {"output_dim": 64, "num_layers": 2},
        "multi_fuse": False,
    }
    spec = resolve_modular_spec(**_kwargs("bsrnn", {"textual": textual}))

    options = spec.cues["textual"].options["features"]["dae_kce"]
    assert options["checkpoint"] == "ckpt/epoch_149.pt"
    assert options["eps"] == 1e-5
    assert options["multi_fuse"] is False
    with pytest.raises(TypeError):
        options["model_config"] = {}


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("model_config", [1], TypeError),
        ("fbank", 80, TypeError),
        ("eps", 0, ValueError),
    ],
)
def test_textual_kce_pass_through_keys_are_still_validated(key, value, error):
    textual = _textual()
    textual["features"]["dae_kce"] = {key: value}

    with pytest.raises(error, match=f"dae_kce.{key}"):
        resolve_modular_spec(**_kwargs("bsrnn", {"textual": textual}))


def test_textual_and_listen_specs_retain_waveform_mixture_dependency():
    textual = resolve_modular_spec(**_kwargs("bsrnn", {"textual": _textual()}))
    listen = resolve_modular_spec(**_kwargs("bsrnn", {"speaker": _speaker("listen")}))

    assert textual.required_mixture_domains == frozenset({"spectrum", "waveform"})
    assert listen.required_mixture_domains == frozenset({"spectrum", "waveform"})


def test_usef_spatial_append_is_a_legal_local_extension_not_an_upstream_path():
    spec = resolve_modular_spec(
        **_kwargs(
            "bsrnn",
            {
                "speaker": _speaker("usef", "context"),
                "spatial": _spatial("ipd", "cdf", "delta_stft"),
            },
            composition={"usef_mode": "append", "allow_all_missing": True},
            n_channel=4,
        )
    )

    assert spec.upstream_name is None
    assert spec.implementation_available is True
    assert spec.composition == {
        "usef_mode": "append",
        "allow_all_missing": True,
    }


def test_speaker_only_usef_replace_keeps_the_upstream_registered_combination():
    spec = resolve_modular_spec(
        **_kwargs(
            "bsrnn",
            {"speaker": _speaker("usef")},
            composition={"usef_mode": "replace"},
        )
    )

    assert spec.upstream_name == "TSE_BSRNN_SPK"


def test_track_is_legal_but_not_mislabeled_as_the_static_upstream_path():
    spec = resolve_modular_spec(
        **_kwargs(
            "bsrnn",
            {"spatial": _spatial("ipd", "cdf", representation="track")},
            n_channel=2,
        )
    )

    assert spec.upstream_name is None
    assert spec.cues["spatial"].representation == "track"


@pytest.mark.parametrize(
    ("cue_kind", "legacy_representation"),
    [
        ("spatial", "static_doa"),
        ("spatial", "doa_track"),
        ("visual", "raw_video"),
    ],
)
def test_legacy_representation_aliases_are_rejected(
    cue_kind: str,
    legacy_representation: str,
):
    cue = (
        _spatial("ipd", "cdf", representation=legacy_representation)
        if cue_kind == "spatial"
        else _visual(legacy_representation)
    )
    args = _kwargs(
        "bsrnn",
        {cue_kind: cue},
        n_channel=2 if cue_kind == "spatial" else 1,
    )

    with pytest.raises(ValueError, match=f"cues\\.{cue_kind}\\.representation"):
        resolve_modular_spec(**args)


def test_speaker_representation_selects_model_input_without_excluding_data_slots():
    waveform = resolve_modular_spec(
        **_kwargs("dpccn", {"speaker": _speaker("spkemb", representation="waveform")})
    )
    embedding = resolve_modular_spec(
        **_kwargs("dpccn", {"speaker": _speaker("spkemb", representation="embedding")})
    )

    assert waveform.cues["speaker"].representation == "waveform"
    assert embedding.cues["speaker"].representation == "embedding"


def test_spec_snapshots_are_immutable():
    args = _kwargs("bsrnn", {"speaker": _speaker("tfmap")})
    spec = resolve_modular_spec(**args)
    args["cues"]["speaker"]["features"]["tfmap"]["type"] = "changed"

    assert spec.cues["speaker"].options["features"]["tfmap"] == {}
    with pytest.raises(TypeError):
        spec.separator["feature_dim"] = 1


def test_reloading_parser_does_not_import_optional_frontends_or_models():
    before = set(sys.modules)
    importlib.reload(modular_spec)
    loaded = set(sys.modules) - before

    forbidden = (
        "tasks.tse.models.discriminative",
        "tasks.tse.models.modules.speaker.spk_frontend",
        "wesep",
    )
    assert not any(name.startswith(forbidden) for name in loaded)


@pytest.mark.parametrize(
    "alias",
    [
        "sr",
        "win",
        "stride",
        "n_fft",
        "hop_length",
        "win_length",
        "window",
        "center",
        "channels",
        "reference_channel",
        "ref_channel",
    ],
)
def test_spectral_separator_rejects_nested_stft_and_geometry_aliases(alias: str):
    args = _kwargs(
        "bsrnn",
        {"speaker": _speaker("tfmap")},
        separator={alias: 1},
    )

    with pytest.raises(ValueError, match="separator contains unknown keys"):
        resolve_modular_spec(**args)


@pytest.mark.parametrize(
    ("path", "mutator"),
    [
        ("cues", lambda args: args["cues"].update({"audio": {}})),
        ("cues.speaker", lambda args: args["cues"]["speaker"].update({"input": "waveform"})),
        (
            "cues.speaker.features",
            lambda args: args["cues"]["speaker"]["features"].update({"unknown": {}}),
        ),
        (
            "cues.speaker.features.tfmap",
            lambda args: args["cues"]["speaker"]["features"]["tfmap"].update({"enabled": True}),
        ),
        (
            "cues.speaker.encoder",
            lambda args: args["cues"]["speaker"].update({"encoder": {"speaker_model": "x"}}),
        ),
        (
            "cues.speaker.encoder.fbank",
            lambda args: args["cues"]["speaker"].update(
                {"encoder": {"fbank": {"sample_rate": 16000}}}
            ),
        ),
    ],
)
def test_speaker_nested_unknown_keys_fail_at_their_exact_boundary(path: str, mutator):
    args = _kwargs("bsrnn", {"speaker": _speaker("tfmap")})
    mutator(args)

    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        resolve_modular_spec(**args)


def test_speaker_encoder_pretrained_prefix_is_explicitly_preserved():
    args = _kwargs("bsrnn", {"speaker": _speaker("context")})
    args["cues"]["speaker"]["encoder"] = {
        "checkpoint": "tasks/tse/pretrained_models/spk_emb_100/avg_model.pt",
        "pretrained_prefix": "spk_ft.encoder.spk_model.",
    }
    spec = resolve_modular_spec(**args)
    assert (
        spec.cues["speaker"].options["encoder"]["pretrained_prefix"]
        == "spk_ft.encoder.spk_model."
    )


@pytest.mark.parametrize("value", ["", "  prefix", 3])
def test_speaker_encoder_pretrained_prefix_is_strict(value):
    args = _kwargs("bsrnn", {"speaker": _speaker("context")})
    args["cues"]["speaker"]["encoder"] = {
        "pretrained_prefix": value,
    }
    with pytest.raises(ValueError, match="pretrained_prefix"):
        resolve_modular_spec(**args)


@pytest.mark.parametrize(
    ("args", "path"),
    [
        (
            _kwargs(
                "bsrnn",
                {
                    "spatial": {
                        **_spatial("ipd"),
                        "array": {"mic_positions": [[0.0, 0.0, 0.0]]},
                    }
                },
                n_channel=2,
            ),
            "cues.spatial",
        ),
        (
            _kwargs(
                "bsrnn",
                {
                    "visual": {
                        "representation": "features",
                        "features": {
                            "muse_visual": {"adapter": {"unknown": 1}}
                        },
                    }
                },
            ),
            "cues.visual.features.muse_visual.adapter",
        ),
        (
            _kwargs(
                "bsrnn",
                {
                    "textual": {
                        "representation": "phoneme_ids",
                        "features": {
                            "dae_kce": {"tokenizer": {"vocabulary": "vocab.txt"}}
                        },
                    }
                },
            ),
            "cues.textual.features.dae_kce.tokenizer",
        ),
        (
            _kwargs(
                "nbc2",
                {"spatial": _spatial("ipd")},
                separator={"block": {"unknown": 1}},
                n_channel=2,
            ),
            "separator.block",
        ),
    ],
)
def test_other_nested_unknown_keys_are_rejected(args: dict, path: str):
    with pytest.raises(ValueError, match=path.replace(".", r"\.")):
        resolve_modular_spec(**args)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_kwargs("bsrnn", {}), "at least one cue"),
        (
            _kwargs("bsrnn", {"speaker": _speaker("tfmap"), "visual": _visual()}),
            "unsupported TSE_Modular combination",
        ),
        (
            _kwargs("tfgridnet", {"spatial": _spatial("ipd")}, n_channel=2),
            "unsupported TSE_Modular combination",
        ),
        (
            _kwargs("bsrnn", {"speaker": _speaker("spex_plus")}),
            "does not support speaker features",
        ),
        (
            _kwargs("dpccn", {"speaker": _speaker("tfmap")}),
            "does not support speaker features",
        ),
        (
            _kwargs(
                "bsrnn",
                {"speaker": _speaker("tfmap", representation="embedding")},
            ),
            "embedding representation only supports",
        ),
        (
            _kwargs(
                "spex_plus",
                {"speaker": _speaker("spex_plus", representation="embedding")},
            ),
            "embedding representation only supports",
        ),
        (
            _kwargs("bsrnn", {"spatial": _spatial("ipd")}),
            "spatial cue requires n_channel",
        ),
        (
            _kwargs("bsrnn", {"spatial": _spatial("ipd")}, n_channel=2),
            "spatial cue requires at least one target-conditioned feature",
        ),
        (
            _kwargs(
                "nbc2",
                {"spatial": _spatial("ipd", "initstate_emb")},
                n_channel=2,
            ),
            "nbc2 does not support spatial initstate_emb",
        ),
        (
            _kwargs("dpccn", {"speaker": _speaker("spkemb")}, n_channel=2),
            "dpccn requires n_channel=1",
        ),
        (
            _kwargs("spex_plus", {"speaker": _speaker("spex_plus")}, n_channel=2),
            "spex_plus requires n_channel=1",
        ),
        (
            _kwargs("bsrnn", {"textual": _textual()}, n_channel=2),
            "requires n_channel=1",
        ),
        (
            _kwargs("bsrnn", {"textual": _textual()}, sample_rate=8000),
            "requires sample_rate=16000",
        ),
    ],
)
def test_invalid_combination_representation_and_geometry_paths_fail(args: dict, message: str):
    with pytest.raises(ValueError, match=message):
        resolve_modular_spec(**args)


def test_usef_spatial_rejects_replacement_semantics():
    args = _kwargs(
        "bsrnn",
        {
            "speaker": _speaker("usef"),
            "spatial": _spatial("ipd", "cdf"),
        },
        composition={"usef_mode": "replace"},
        n_channel=4,
    )

    with pytest.raises(ValueError, match="requires composition.usef_mode='append'"):
        resolve_modular_spec(**args)


@pytest.mark.parametrize("field", tuple(_STFT))
def test_spectrum_backbone_requires_every_top_level_stft_field(field: str):
    args = _kwargs("bsrnn", {"speaker": _speaker("tfmap")})
    del args[field]

    with pytest.raises(ValueError, match="requires top-level STFT fields"):
        resolve_modular_spec(**args)


@pytest.mark.parametrize(("field", "value"), tuple(_STFT.items()))
def test_waveform_native_backbone_rejects_every_unused_stft_field(field: str, value):
    args = _kwargs("spex_plus", {"speaker": _speaker("spex_plus")})
    args[field] = value

    with pytest.raises(ValueError, match="rejects unused top-level STFT fields"):
        resolve_modular_spec(**args)


@pytest.mark.parametrize(
    "updates",
    [
        {"n_fft": 128, "hop_length": 256, "win_length": 128},
        {"center": False, "win_length": 256},
        {"center": False, "window": "hann"},
    ],
)
def test_invalid_top_level_stft_geometry_is_rejected(updates: dict):
    args = _kwargs("bsrnn", {"speaker": _speaker("tfmap")})
    args.update(updates)

    with pytest.raises(ValueError):
        resolve_modular_spec(**args)


def test_reference_channel_must_belong_to_top_level_channel_geometry():
    args = _kwargs("bsrnn", {"speaker": _speaker("tfmap")}, n_channel=2)
    args["ref_channel"] = 2

    with pytest.raises(ValueError, match="must be smaller than n_channel"):
        resolve_modular_spec(**args)


@pytest.mark.parametrize(
    "pairs",
    [[], [[0, 0]], [[0, 2]], [[0]], [[0, 1], [0, 1]]],
)
def test_spatial_microphone_pairs_are_strictly_validated(pairs: list):
    spatial = _spatial("ipd")
    spatial["pairs"] = pairs
    args = _kwargs("bsrnn", {"spatial": spatial}, n_channel=2)

    with pytest.raises((TypeError, ValueError), match="cues.spatial.pairs"):
        resolve_modular_spec(**args)


def test_implementation_gate_is_separate_from_spec_validation():
    implemented = _kwargs("bsrnn", {"speaker": _speaker("tfmap")})
    assert resolve_modular_spec(**implemented, require_implemented=True).implementation_available
    legal = _kwargs("bsrnn", {"speaker": _speaker("listen", "tfmap")})
    assert not resolve_modular_spec(**legal).implementation_available
    with pytest.raises(NotImplementedError, match="TSE_BSRNN_SPK"):
        resolve_modular_spec(**legal, require_implemented=True)

    malformed = copy.deepcopy(legal)
    malformed["separator"]["win"] = 512
    with pytest.raises(ValueError, match="separator contains unknown keys"):
        resolve_modular_spec(**malformed, require_implemented=True)


def test_tfmap_embedding_requires_centered_noncausal_mono_bsrnn():
    speaker = {
        "representation": "waveform",
        "features": {
            "tfmap": {"type": "embedding"},
            "context": {
                "source": "tfmap_embedding",
                "injection": "latent",
                "attention_dim": 8,
                "num_heads": 1,
                "fusion": "multiply",
            },
        },
        "encoder": {
            "name": "ECAPA_TDNN_GLOB_c512",
            "fbank": {
                "num_mel_bins": 8,
                "frame_length_ms": 32.0,
                "frame_shift_ms": 8.0,
                "dither": 0.0,
            },
        },
    }
    kwargs = _kwargs("bsrnn", {"speaker": speaker})
    spec = resolve_modular_spec(**kwargs)
    assert spec.implementation_available is True
    assert "waveform" in spec.required_mixture_domains

    causal = copy.deepcopy(kwargs)
    causal["separator"] = {"causal": True}
    with pytest.raises(ValueError, match="tfmap embedding does not support causal"):
        resolve_modular_spec(**causal)

    multichannel = copy.deepcopy(kwargs)
    multichannel["n_channel"] = 2
    with pytest.raises(ValueError, match="tfmap embedding requires mono"):
        resolve_modular_spec(**multichannel)
