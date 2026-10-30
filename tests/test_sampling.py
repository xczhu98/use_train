"""Contract tests for Task 1 dynamic balanced sampling primitives."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import soundfile

from core.data.sampling import (
    DatasetSamplingConfig,
    LanguageSamplingConfig,
    SamplingConfig,
    SpeechLeafMetadata,
    SpeechSamplingConfig,
    SpeakerSamplingConfig,
    _normalize_speech_inputs,
    _parse_speech as parse_speech,
    make_speech_plan,
)
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from core.data.simulation_utils import read_source_scp
from tasks.se.data.dynamic_dataset import DynamicMixingDataset as SEDynamicMixingDataset


def make_leaves(*specs):
    leaves = []
    start = 0
    for language, dataset, sr, count, seconds in specs:
        leaves.append(
            SpeechLeafMetadata(
                language=language,
                dataset_id=dataset,
                speaker=None,
                batching_sr=sr,
                uid_indices=tuple(range(start, start + count)),
                effective_seconds=Fraction(str(seconds)),
            )
        )
        start += count
    return leaves


def test_sampling_facade_reexports_schema_and_projector_boundaries():
    import core.data.sampling as facade
    import core.data.sampling_projector as projector
    import core.data.sampling_schema as schema

    for name in (
        "DatasetSamplingConfig",
        "LanguageSamplingConfig",
        "NoiseSamplingConfig",
        "SamplingConfig",
        "SpeakerSamplingConfig",
        "SpeechSamplingConfig",
    ):
        assert getattr(facade, name) is getattr(schema, name)
    for name in (
        "_fraction_ceiling",
        "_project_speaker_margin_actual",
        "_project_transport_actual",
    ):
        assert getattr(facade, name) is getattr(projector, name)
    assert facade._project_dataset_total_cap((1,), Fraction(1), 1, 1) == 1


def test_sampling_projector_has_no_runtime_dataset_dependencies():
    source = Path("core/data/sampling_projector.py").read_text(encoding="utf-8")

    assert "dynamic_dataset" not in source
    assert "simulation_utils" not in source
    assert "core.data.sampling" not in source


def test_manual_language_config_rejects_duplicate_keys():
    cfg = SpeechSamplingConfig(
        language=LanguageSamplingConfig((
            ("en", Fraction(1, 2)),
            ("en", Fraction(1, 4)),
            ("zh", Fraction(1, 4)),
        ))
    )
    with pytest.raises(ValueError, match="duplicate.*en"):
        make_speech_plan(
            make_leaves(("en", "a", 16000, 2, 2.0)),
            requested=1, world_size=1, batch_size=1,
            sampling_config=cfg,
        )


def test_manual_language_config_requires_exact_probability_sum():
    cfg = SpeechSamplingConfig(
        language=LanguageSamplingConfig((
            ("en", Fraction(1, 2)),
            ("zh", Fraction(5_000_000_001, 10_000_000_000)),
        ))
    )

    with pytest.raises(ValueError, match="probabilities must sum to 1"):
        _normalize_speech_inputs(cfg)


@pytest.mark.parametrize(
    "pair",
    [
        None,
        ("zh",),
        ("zh", Fraction(1), "extra"),
        ["zh", Fraction(1)],
        (("nested", "key"), Fraction(1)),
    ],
)
def test_manual_language_config_rejects_malformed_pairs_with_field_index(pair):
    cfg = SpeechSamplingConfig(
        language=LanguageSamplingConfig(probs=(pair,))
    )

    with pytest.raises(ValueError, match=r"language\.probs\[0\]"):
        _normalize_speech_inputs(cfg)


def test_manual_dataset_config_normalizes_real_values_to_fraction():
    cfg = SpeechSamplingConfig(
        dataset=DatasetSamplingConfig(alpha=0.5, max_ratio=0.5)
    )

    normalized = _normalize_speech_inputs(cfg)
    assert normalized is not cfg
    assert normalized.dataset == DatasetSamplingConfig(
        alpha=Fraction(1, 2), max_ratio=Fraction(1, 2)
    )
    assert type(normalized.dataset.alpha) is Fraction
    assert type(normalized.dataset.max_ratio) is Fraction

    plan = make_speech_plan(
        make_leaves(("zh", "a", 16000, 2, 2.0)),
        requested=1, world_size=1, batch_size=1,
        sampling_config=cfg,
    )
    assert plan.actual == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", True),
        ("alpha", float("nan")),
        ("alpha", float("inf")),
        ("alpha", -0.1),
        ("alpha", 1.1),
        ("max_ratio", True),
        ("max_ratio", float("nan")),
        ("max_ratio", float("inf")),
        ("max_ratio", 0),
        ("max_ratio", 1.1),
    ],
)
def test_manual_dataset_config_rejects_invalid_real_values(field, value):
    values = {"alpha": Fraction(1, 2), "max_ratio": Fraction(1, 2)}
    values[field] = value
    cfg = SpeechSamplingConfig(dataset=DatasetSamplingConfig(**values))

    with pytest.raises(ValueError, match=rf"dataset\.{field}"):
        make_speech_plan(
            make_leaves(("zh", "a", 16000, 2, 2.0)),
            requested=1, world_size=1, batch_size=1,
            sampling_config=cfg,
        )


@pytest.mark.parametrize("field", ["alpha", "max_ratio"])
def test_manual_dataset_config_wraps_float_conversion_overflow(field):
    values = {"alpha": Fraction(1, 2), "max_ratio": Fraction(1, 2)}
    values[field] = 10**400
    cfg = SpeechSamplingConfig(dataset=DatasetSamplingConfig(**values))

    with pytest.raises(ValueError, match=rf"dataset\.{field}.*finite"):
        _normalize_speech_inputs(cfg)


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.0, "1", _IntSubclass(1)],
)
def test_manual_speaker_config_requires_exact_positive_builtin_int(value):
    cfg = SpeechSamplingConfig(
        speaker=SpeakerSamplingConfig(max_samples_per_epoch=value)
    )

    with pytest.raises(ValueError, match="speaker.max_samples_per_epoch.*positive int"):
        _normalize_speech_inputs(cfg)


@pytest.mark.parametrize(
    ("field", "value", "expected_type"),
    [
        ("language", object(), "LanguageSamplingConfig"),
        ("dataset", object(), "DatasetSamplingConfig"),
        ("speaker", {"max_samples_per_epoch": 1}, "SpeakerSamplingConfig"),
    ],
)
def test_manual_speech_config_rejects_malformed_wrapper_types(
    field, value, expected_type
):
    cfg = SpeechSamplingConfig(**{field: value})

    with pytest.raises(
        ValueError,
        match=rf"speech\.{field} must be a {expected_type}",
    ):
        _normalize_speech_inputs(cfg)


def test_empty_sampling_blocks_are_disabled_independently():
    config = SamplingConfig.from_dict({"speech": {}, "noise": {}})

    assert config.speech is None
    assert config.noise is None


def test_explicit_empty_speech_config_stays_disabled_at_public_boundary():
    explicit = SpeechSamplingConfig()

    assert explicit.has_controllers is False
    assert _normalize_speech_inputs(explicit) is None
    assert _normalize_speech_inputs(SamplingConfig(speech=explicit)) is None

    with pytest.raises(ValueError, match="controllers=none"):
        make_speech_plan(
            make_leaves(("en", "a", 16000, 2, 2.0)),
            requested=2,
            world_size=1,
            batch_size=1,
            sampling_config=explicit,
        )


def test_flat_language_and_dataset_normalize_to_nested_types():
    cfg = parse_speech(
        {
            "language_probs": {"zh": 0.7, "en": 0.3},
            "dataset_alpha": 0.4,
            "max_dataset_ratio": 0.25,
        }
    )

    assert cfg.language.probs == (
        ("en", Fraction(3, 10)),
        ("other", Fraction(0)),
        ("zh", Fraction(7, 10)),
    )
    assert cfg.dataset == DatasetSamplingConfig(Fraction(2, 5), Fraction(1, 4))
    assert cfg.speaker is None
    assert cfg.has_controllers is True


def test_flat_language_weights_stably_normalize_overflowing_values():
    speech = parse_speech({"language_probs": {"zh": 1e308, "en": 1e308}})

    assert speech is not None
    assert speech.language.probs == (
        ("en", Fraction(1, 2)),
        ("other", Fraction(0)),
        ("zh", Fraction(1, 2)),
    )
    assert speech.dataset is None
    assert speech.speaker is None


def test_flat_language_weights_canonicalize_rounding_residual():
    speech = parse_speech(
        {"language_probs": {"en": 1, "other": 1, "zh": 1}}
    )

    assert sum(probability for _, probability in speech.language.probs) == 1
    assert _normalize_speech_inputs(SamplingConfig(speech=speech)) == speech
    plan = make_speech_plan(
        make_leaves(
            ("en", "a", 16000, 2, 1.0),
            ("other", "b", 16000, 2, 1.0),
            ("zh", "c", 16000, 2, 1.0),
        ),
        requested=2,
        world_size=1,
        batch_size=1,
        sampling_config=speech,
    )
    assert plan.language_quotas == {"en": 1, "other": 1, "zh": 0}


def test_nested_language_parser_canonicalizes_rounding_residual():
    cfg = SamplingConfig.from_dict({
        "speech": {
            "language": {
                "probs": {"en": 1 / 3, "other": 1 / 3, "zh": 1 / 3}
            }
        }
    })

    assert sum(probability for _, probability in cfg.speech.language.probs) == 1
    plan = make_speech_plan(
        make_leaves(
            ("en", "a", 16000, 1, 1.0),
            ("other", "b", 16000, 1, 1.0),
            ("zh", "c", 16000, 1, 1.0),
        ),
        requested=3,
        world_size=1,
        batch_size=1,
        sampling_config=cfg,
    )
    assert plan.actual == 3


def test_flat_migration_preserves_controller_independence():
    language_only = parse_speech({"language_probs": {"zh": 3, "en": 1}})
    assert language_only.language.probs == (
        ("en", Fraction(1, 4)),
        ("other", Fraction(0)),
        ("zh", Fraction(3, 4)),
    )
    assert language_only.dataset is None
    assert language_only.speaker is None
    assert language_only.has_controllers is True

    dataset_only = parse_speech({"max_dataset_ratio": 0.4})
    assert dataset_only.language is None
    assert dataset_only.dataset == DatasetSamplingConfig(
        Fraction(1, 2), Fraction(2, 5)
    )
    assert dataset_only.speaker is None

    both = parse_speech(
        {"language_probs": {"zh": 1}, "dataset_alpha": 0.2}
    )
    assert both.language.probs == (
        ("en", Fraction(0)),
        ("other", Fraction(0)),
        ("zh", Fraction(1)),
    )
    assert both.dataset == DatasetSamplingConfig(Fraction(1, 5), Fraction(3, 10))


def test_empty_speech_sampling_keeps_legacy_disabled():
    assert SamplingConfig.from_dict(None).speech is None
    assert SamplingConfig.from_dict({}).speech is None
    assert SamplingConfig.from_dict({"speech": {}}).speech is None


@pytest.mark.parametrize(
    ("speech", "active"),
    [
        ({}, (False, False, False)),
        ({"language": {"probs": {"zh": 1.0}}}, (True, False, False)),
        ({"dataset": {}}, (False, True, False)),
        ({"speaker": {"max_samples_per_epoch": 3}}, (False, False, True)),
        ({"language": {"probs": {"en": 1.0}}, "dataset": {}}, (True, True, False)),
        ({"dataset": {}, "speaker": {"max_samples_per_epoch": 2}}, (False, True, True)),
        ({"language": {"probs": {"other": 1.0}}, "speaker": {"max_samples_per_epoch": 2}}, (True, False, True)),
        ({"language": {"probs": {"zh": 1.0}}, "dataset": {}, "speaker": {"max_samples_per_epoch": 2}}, (True, True, True)),
    ],
)
def test_nested_speech_controller_presence_is_activation(speech, active):
    cfg = SamplingConfig.from_dict({"speech": speech})
    controllers = cfg.speech
    observed = (
        controllers is not None and controllers.language is not None,
        controllers is not None and controllers.dataset is not None,
        controllers is not None and controllers.speaker is not None,
    )
    assert observed == active


def test_nested_dataset_defaults_match_frozen_contract():
    dataset = parse_speech({"dataset": {}}).dataset

    assert dataset.alpha == Fraction(1, 2)
    assert dataset.max_ratio == Fraction(3, 10)


@pytest.mark.parametrize(
    ("speech", "match"),
    [
        ({"language": {}}, "probs"),
        ({"language": {"probs": {"fr": 1}}}, "unknown"),
        ({"language": {"probs": {"zh": -0.1, "en": 1.1}}}, "non-negative"),
        ({"language": {"probs": {"zh": float("nan")}}}, "finite"),
        ({"language": {"probs": {"zh": float("inf")}}}, "finite"),
        ({"language": {"probs": {"zh": 0.6, "en": 0.3}}}, "sum"),
    ],
)
def test_nested_language_rejects_strict_schema_errors(speech, match):
    with pytest.raises(ValueError, match=match):
        parse_speech(speech)


@pytest.mark.parametrize(
    ("speech", "match"),
    [
        ({"dataset": {"alpha": True}}, "dataset.alpha"),
        ({"dataset": {"alpha": float("nan")}}, "finite"),
        ({"dataset": {"alpha": float("inf")}}, "finite"),
        ({"dataset": {"alpha": -0.1}}, "dataset.alpha"),
        ({"dataset": {"alpha": 1.1}}, "dataset.alpha"),
        ({"dataset": {"max_ratio": True}}, "max_ratio"),
        ({"dataset": {"max_ratio": float("nan")}}, "finite"),
        ({"dataset": {"max_ratio": float("inf")}}, "finite"),
        ({"dataset": {"max_ratio": 0}}, "max_ratio"),
        ({"dataset": {"max_ratio": 1.1}}, "max_ratio"),
        ({"dataset": {"extra": 0.5}}, "unknown"),
    ],
)
def test_nested_dataset_rejects_non_finite_and_out_of_range(speech, match):
    with pytest.raises(ValueError, match=match):
        parse_speech(speech)


@pytest.mark.parametrize(
    ("speech", "match"),
    [
        ({"speaker": {"max_samples_per_epoch": True}}, "positive int"),
        ({"speaker": {"max_samples_per_epoch": False}}, "positive int"),
        ({"speaker": {"max_samples_per_epoch": 0}}, "positive int"),
        ({"speaker": {"max_samples_per_epoch": -1}}, "positive int"),
        ({"speaker": {"max_samples_per_epoch": 1.5}}, "positive int"),
        ({"speaker": {"max_samples_per_epoch": "3"}}, "positive int"),
        ({"speaker": {}}, "required"),
        ({"speaker": {"max_samples_per_epoch": 3, "extra": 1}}, "unknown"),
    ],
)
def test_nested_speaker_requires_positive_int_cap(speech, match):
    with pytest.raises(ValueError, match=match):
        parse_speech(speech)


def test_flat_and_nested_language_conflict_is_rejected():
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_speech({"language_probs": {"zh": 1}, "language": {"probs": {"zh": 1}}})


def test_flat_and_nested_dataset_conflict_is_rejected():
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_speech(
            {"dataset_alpha": 0.5, "dataset": {"alpha": 0.5}}
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_speech(
            {"max_dataset_ratio": 0.3, "dataset": {"max_ratio": 0.3}}
        )


def test_flat_dataset_partial_fields_use_frozen_defaults():
    alpha_only = parse_speech({"dataset_alpha": 0.4})
    assert alpha_only.dataset == DatasetSamplingConfig(Fraction(2, 5), Fraction(3, 10))

    ratio_only = parse_speech({"max_dataset_ratio": 0.25})
    assert ratio_only.dataset == DatasetSamplingConfig(Fraction(1, 2), Fraction(1, 4))


@pytest.mark.parametrize("field", [
    "language_enabled", "dataset_enabled", "speaker_enabled", "speech.enabled"
])
def test_enable_style_fields_are_rejected(field):
    with pytest.raises(ValueError, match="unknown speech sampling field"):
        parse_speech({field: True})


@pytest.mark.parametrize(
    ("sampling", "match"),
    [
        ({"speech": {"unknown": 1}}, "unknown"),
        ({"speech": {"language_probs": {"zh": 1}, "dataset_alpha": True}}, "dataset_alpha"),
        ({"speech": {"language_probs": {"zh": 1}, "dataset_alpha": float("nan")}}, "finite"),
        ({"speech": {"language_probs": {"zh": 0, "en": 0, "other": 0}}}, "positive"),
        ({"speech": {"language_probs": {"zh": 1, "fr": 1}}}, "language_probs"),
        ({"speech": {"language_probs": {"zh": 1}, "max_dataset_ratio": 1.1}}, "max_dataset_ratio"),
        ({"noise": {"balance_categories": 1}}, "balance_categories"),
        ({"noise": {"unknown": False}}, "unknown"),
    ],
)
def test_sampling_config_rejects_invalid_strict_values(sampling, match):
    with pytest.raises(ValueError, match=match):
        SamplingConfig.from_dict(sampling)


def test_dataset_alpha_accepts_unit_interval_and_rejects_above_one():
    for dataset_alpha in (0, 1):
        config = SamplingConfig.from_dict(
            {"speech": {"language_probs": {"zh": 1}, "dataset_alpha": dataset_alpha}}
        )
        assert config.speech.dataset.alpha == Fraction(dataset_alpha)

    with pytest.raises(ValueError, match="dataset_alpha"):
        SamplingConfig.from_dict(
            {"speech": {"language_probs": {"zh": 1}, "dataset_alpha": 1.000001}}
        )


def test_noise_sampling_is_independent_opt_in():
    config = SamplingConfig.from_dict({"noise": {"balance_categories": True}})

    assert config.speech is None
    assert config.noise.balance_categories is True


def test_jsonl_reader_returns_metadata_only_when_requested(tmp_path):
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "uid": "speech-1",
                        "fs": 16000,
                        "length": 32000,
                        "path": "/audio/speech.wav",
                        "language": "zh",
                        "dataset_id": "set-a",
                        "speaker_id": "spk-1",
                        "extra": {"source": "fixture"},
                    }
                ),
                json.dumps(
                    {
                        "uid": "noise-1",
                        "fs": 48000,
                        "length": 96000,
                        "path": "/audio/noise.wav",
                        "category_id": "traffic",
                        "category_name": "Traffic",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    default_result = read_source_scp(manifest)
    result_with_metadata = read_source_scp(manifest, return_metadata=True)

    assert len(default_result) == 4
    by_fs, uids, flat, lengths, metadata = result_with_metadata
    assert by_fs[16000] == {"speech-1": "/audio/speech.wav"}
    assert uids[48000] == ["noise-1"]
    assert flat["noise-1"] == "/audio/noise.wav"
    assert lengths == {"speech-1": 32000, "noise-1": 96000}
    assert metadata["speech-1"] == {
        "uid": "speech-1",
        "fs": 16000,
        "length": 32000,
        "path": "/audio/speech.wav",
        "language": "zh",
        "dataset_id": "set-a",
        "speaker_id": "spk-1",
        "extra": {"source": "fixture"},
    }


@pytest.mark.parametrize(
    "records",
    [
        ["not an object"],
        [{"uid": " uid", "fs": 16000, "length": 1, "path": "/a.wav"}],
        [{"uid": "u", "fs": True, "length": 1, "path": "/a.wav"}],
        [{"uid": "u", "fs": 16000, "length": 0, "path": "/a.wav"}],
        [
            {"uid": "u", "fs": 16000, "length": 1, "path": "/a.wav"},
            {"uid": "u", "fs": 16000, "length": 1, "path": "/b.wav"},
        ],
    ],
)
def test_jsonl_reader_rejects_invalid_records_with_path_and_line(tmp_path, records):
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{manifest}.*line"):
        read_source_scp(manifest, return_metadata=True)


def test_scp_has_no_metadata_but_jsonl_preserves_records(tmp_path):
    scp = tmp_path / "legacy.scp"
    scp.write_text("legacy 16000 100 /legacy.wav\n", encoding="utf-8")
    jsonl_manifest = tmp_path / "canonical.jsonl"
    jsonl_manifest.write_text(
        json.dumps({"uid": "old", "fs": 8000, "length": 5, "path": "/old.wav"})
        + "\n",
        encoding="utf-8",
    )

    assert read_source_scp(scp, return_metadata=True)[-1] == {}
    assert read_source_scp(jsonl_manifest, return_metadata=True)[-1] == {
        "old": {"uid": "old", "fs": 8000, "length": 5, "path": "/old.wav"}
    }


def _write_wav(path: Path, frames: int, sr: int) -> None:
    soundfile.write(path, np.zeros(frames, dtype=np.float32), sr)


def _make_se_sampling_dataset(
    tmp_path: Path,
    speech_records: list[dict],
    *,
    sampling_cfg: SamplingConfig | None,
    target_sr: int | None = None,
    simulation_cfg: dict | None = None,
    max_duration_seconds: float | None = None,
) -> SEDynamicMixingDataset:
    for record in speech_records:
        _write_wav(tmp_path / record["path"], record["length"], record["fs"])
    speech_scp = tmp_path / "speech.jsonl"
    speech_scp.write_text(
        "\n".join(json.dumps(record) for record in speech_records) + "\n", encoding="utf-8"
    )

    aux_paths = {}
    for name in ("noise", "rir", "wind"):
        wav_path = tmp_path / f"{name}.wav"
        _write_wav(wav_path, 16000, 16000)
        scp_path = tmp_path / f"{name}.scp"
        scp_path.write_text(f"{name} 16000 16000 {wav_path}\n", encoding="utf-8")
        aux_paths[name] = scp_path

    return SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(aux_paths["noise"]),
        rir_scp=str(aux_paths["rir"]),
        windnoise_scp=str(aux_paths["wind"]),
        sampling_cfg=sampling_cfg,
        target_sr=target_sr,
        max_duration_seconds=max_duration_seconds,
        filter_by_duration=True,
        filter_min_duration=0.5,
        filter_max_duration=3.0,
        simulation_cfg=simulation_cfg or {"prob_reverberation": 0.0},
    )


def _speech_sampling_config() -> SamplingConfig:
    return SamplingConfig.from_dict({"speech": {"language_probs": {"zh": 1}}})


def test_speech_sampling_cells_follow_duration_filter_and_processing_sr(tmp_path):
    records = [
        {"uid": "short", "fs": 8000, "length": 2000, "path": "short.wav", "language": "zh", "dataset_id": "set-a"},
        {"uid": "kept-zh", "fs": 16000, "length": 16000, "path": "kept-zh.wav", "language": "zh", "dataset_id": "set-a"},
        {"uid": "kept-en", "fs": 8000, "length": 16000, "path": "kept-en.wav", "language": "en", "dataset_id": "set-b"},
    ]

    dataset = _make_se_sampling_dataset(
        tmp_path, records, sampling_cfg=_speech_sampling_config(), target_sr=24000
    )

    assert dataset.balanced_speech_sampling_enabled is True
    cells = dataset.speech_sampling_cells
    assert set(cells) == {("zh", "set-a", 24000), ("en", "set-b", 24000)}
    assert cells[("zh", "set-a", 24000)].uids == ("kept-zh",)
    assert cells[("en", "set-b", 24000)].uids == ("kept-en",)
    assert cells[("zh", "set-a", 24000)].external_indices == (1,)
    assert cells[("en", "set-b", 24000)].external_indices == (0,)


def test_speech_sampling_cell_effective_seconds_uses_raw_metadata(tmp_path):
    records = [
        {"uid": "long", "fs": 16000, "length": 32000, "path": "long.wav", "language": "zh", "dataset_id": "set-a"},
    ]

    dataset = _make_se_sampling_dataset(tmp_path, records, sampling_cfg=_speech_sampling_config())

    cell = dataset.speech_sampling_cells[("zh", "set-a", 16000)]
    assert dataset.source_length["long"] == 32000
    assert cell.effective_seconds == pytest.approx(2.0)


def test_speech_sampling_aggregates_invalid_metadata_after_duration_filter(tmp_path):
    records = [
        {"uid": "bad-language", "fs": 16000, "length": 16000, "path": "bad-language.wav", "language": " zh ", "dataset_id": "set-a"},
        {"uid": "bad-dataset", "fs": 16000, "length": 16000, "path": "bad-dataset.wav", "language": "zh", "dataset_id": ""},
        {"uid": "filtered-out", "fs": 16000, "length": 1000, "path": "filtered-out.wav", "language": "", "dataset_id": ""},
    ]

    with pytest.raises(ValueError, match=r"2.*bad-language.*bad-dataset"):
        _make_se_sampling_dataset(tmp_path, records, sampling_cfg=_speech_sampling_config())


def test_speaker_splice_pool_validates_filtered_out_manifest_metadata(tmp_path):
    """speaker splice 池覆盖完整 manifest，duration-filtered 记录也必须有 identity。"""
    records = [
        {"uid": "kept", "fs": 16000, "length": 16000, "path": "kept.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "filtered-out", "fs": 16000, "length": 1000,
         "path": "filtered-out.wav", "language": "zh", "dataset_id": "set-a"},
    ]
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 2}}}
    )

    with pytest.raises(
        ValueError,
        match=r"speaker sampling splice pool metadata invalid.*filtered-out.*speaker_id",
    ):
        _make_se_sampling_dataset(tmp_path, records, sampling_cfg=sampling_cfg)


def test_speech_sampling_disabled_preserves_manifest_path_without_metadata(tmp_path):
    wav_path = tmp_path / "speech.wav"
    _write_wav(wav_path, 16000, 16000)
    speech_scp = tmp_path / "speech.scp"
    speech_scp.write_text(f"legacy 16000 16000 {wav_path}\n", encoding="utf-8")
    aux = tmp_path / "aux.scp"
    aux.write_text(f"aux 16000 16000 {wav_path}\n", encoding="utf-8")

    dataset = SEDynamicMixingDataset(
        speech_scp=str(speech_scp),
        noise_scp=str(aux),
        rir_scp=str(aux),
        windnoise_scp=str(aux),
        sampling_cfg=None,
        filter_by_duration=True,
        simulation_cfg={"prob_reverberation": 0.0},
    )

    assert dataset.balanced_speech_sampling_enabled is False
    assert not hasattr(dataset, "speech_sampling_cells")


def test_speech_sampling_leaves_use_processing_sr_and_merge_speakers_when_disabled(tmp_path):
    """Task 5 Step 1/2：leaf 的 batching_sr 取 target_sr（处理 SR），不用
    manifest fs；未启用 speaker 时多 speaker 记录聚合为单个 None leaf。"""
    records = [
        {"uid": "spk-a", "fs": 48000, "length": 48000, "path": "spk-a.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "spk-b", "fs": 48000, "length": 48000, "path": "spk-b.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s2"},
    ]

    dataset = _make_se_sampling_dataset(
        tmp_path, records, sampling_cfg=_speech_sampling_config(), target_sr=16000
    )

    leaves = dataset.speech_sampling_leaves
    assert set(leaves) == {("zh", "set-a", None, 16000)}
    leaf = leaves[("zh", "set-a", None, 16000)]
    assert leaf.uids == ("spk-a", "spk-b")
    assert leaf.effective_seconds == Fraction(2)
    # 三元 cell 视图由 leaf 聚合，sampler 校验面不变
    assert set(dataset.speech_sampling_cells) == {("zh", "set-a", 16000)}


def test_speech_sampling_speaker_leaves_carry_canonical_identity(tmp_path):
    """Task 5 Step 2：启用 speaker 时 leaf key 含 canonical identity。"""
    records = [
        {"uid": "spk-a", "fs": 16000, "length": 16000, "path": "spk-a.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "spk-b", "fs": 16000, "length": 16000, "path": "spk-b.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s2"},
    ]
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 10}}}
    )

    dataset = _make_se_sampling_dataset(tmp_path, records, sampling_cfg=sampling_cfg)

    assert set(dataset.speech_sampling_leaves) == {
        ("zh", "set-a", ("set-a", "s1"), 16000),
        ("zh", "set-a", ("set-a", "s2"), 16000),
    }


def test_se_aec_speaker_splice_pool_uses_canonical_identity_manifest_order(tmp_path):
    """SE/AEC speaker controller 只在 canonical identity 内 splice。"""
    records = [
        {"uid": "a1", "fs": 16000, "length": 16000, "path": "a1.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "001"},
        {"uid": "b1", "fs": 16000, "length": 16000, "path": "b1.wav",
         "language": "zh", "dataset_id": "set-b", "speaker_id": "001"},
        {"uid": "a2", "fs": 16000, "length": 16000, "path": "a2.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "001"},
    ]
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 2}}}
    )
    se_dataset = _make_se_sampling_dataset(
        tmp_path, records, sampling_cfg=sampling_cfg
    )

    assert se_dataset._speech_splice_candidate_uids("a1") == ("a1", "a2")
    assert se_dataset._speech_splice_candidate_uids("b1") == ("b1",)
    assert (
        AECDynamicMixingDataset._speech_splice_candidate_uids
        is SEDynamicMixingDataset._speech_splice_candidate_uids
    )


def test_se_speaker_quota_counts_primary_anchors_only(tmp_path):
    """speaker quota 只覆盖 selected primary external indices，不计 splice extra。"""
    records = [
        {"uid": "a1", "fs": 16000, "length": 16000, "path": "a1.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "a2", "fs": 16000, "length": 16000, "path": "a2.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "b1", "fs": 16000, "length": 16000, "path": "b1.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s2"},
    ]
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 1}}}
    )
    dataset = _make_se_sampling_dataset(
        tmp_path, records, sampling_cfg=sampling_cfg
    )

    plan = dataset.get_speech_epoch_plan(
        requested=3,
        world_size=1,
        batch_size=1,
        drop_last=False,
        seed=7,
        epoch=0,
    )

    assert plan.speech_plan.actual == 2
    assert sum(plan.speech_plan.speaker_quotas.values()) == 2
    assert len(plan.selected_indices) == len(plan.selected_uids) == 2
    assert set(plan.selected_uids).issubset({"a1", "a2", "b1"})
    assert plan.speech_plan.summary["joint_solver"]["speaker"]["primary_samples"] == 2


def test_se_splice_trace_accounting_is_sample_local_and_primary_only(tmp_path):
    """每次 trace 只记录本次 primary 与实际 splice extra，不污染下一样本。"""
    records = [
        {"uid": "a1", "fs": 16000, "length": 8000, "path": "a1.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "a2", "fs": 16000, "length": 8000, "path": "a2.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s1"},
        {"uid": "b1", "fs": 16000, "length": 8000, "path": "b1.wav",
         "language": "zh", "dataset_id": "set-a", "speaker_id": "s2"},
    ]
    sampling_cfg = SamplingConfig.from_dict(
        {"speech": {"speaker": {"max_samples_per_epoch": 2}}}
    )
    dataset = _make_se_sampling_dataset(
        tmp_path,
        records,
        sampling_cfg=sampling_cfg,
        target_sr=16000,
        max_duration_seconds=1.25,
        simulation_cfg={
            "prob_target_only": 1.0,
            "prob_noise_only": 0.0,
            "prob_reverberation": 0.0,
            "speech_splice": True,
            "speech_splice_seconds": 1.25,
            "speech_splice_gap_range": [0.01, 0.01],
            "speech_splice_equal_volume": False,
            "speech_splice_max_repeat_times": 1,
            "speech_random_start": False,
            "silence_pad_prob": 0.0,
            "use_high_pass": False,
        },
    )
    dataset.set_epoch(0, base_seed=11)

    first = dataset.get_sample_with_trace(0, force_branch="prob_target_only")
    second = dataset.get_sample_with_trace(2, force_branch="prob_target_only")
    first_export = first.trace.sources["scene"].metadata["speech_splice_export"]
    second_export = second.trace.sources["scene"].metadata["speech_splice_export"]

    assert first_export["primary_uid"] == "a1"
    assert first_export["candidate_scope"] == "canonical_identity"
    assert first_export["attempted_extra_uids"] == first_export["extra_uids"]
    assert len(first_export["extra_uids"]) == 1
    assert set(first_export["extra_uids"]).issubset({"a1", "a2"})
    assert first_export["attempt_count"] == 1
    assert first_export["failure_count"] == 0
    assert first_export["zero_pad_samples"] > 0
    assert first_export["quota_scope"] == "primary_only"

    assert second_export["primary_uid"] == "b1"
    assert second_export["attempted_extra_uids"] == ["b1"]
    assert second_export["extra_uids"] == ["b1"]
    assert first_export is not second_export


pytestmark = [pytest.mark.unit]
