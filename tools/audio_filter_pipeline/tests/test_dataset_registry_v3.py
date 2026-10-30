"""Generation 3 数据集 registry 显式语义测试。"""

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import MappingProxyType

import pytest

from audio_filter_pipeline.config import (
    PipelineConfig,
    SourceSelectorSpec,
    SpeakerMetadataSpec,
    ValidationContext,
)
from audio_filter_pipeline.processors.mos import list_mos_specs

pytestmark = pytest.mark.unit


def _quality_profiles():
    return {
        "quality_balanced": {"thresholds": {
            "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
            "scoreq": 3.6,
            "urgent_pk": 3.5,
        }},
        "quality_high": {"thresholds": {
            "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
            "scoreq": 4.2,
            "urgent_pk": 3.5,
        }},
        "quality_premium": {"thresholds": {
            "dnsmos": {"ovrl": 3.4, "sig": 3.6, "bak": 4.2, "p808": 4.0},
            "scoreq": 4.5,
            "urgent_pk": 3.5,
        }},
    }


def _explicit_row(**overrides):
    row = {
        "input_root": "/data/alpha",
        "language": "EN_us",
        "category": "read_speech",
        "quality_profile": "quality_high",
        "speaker": {
            "source": "path_component",
            "path_component": 0,
        },
    }
    row.update(overrides)
    return row


def _parse(row, selection_policies=None):
    if selection_policies is None:
        selection_policies = {"quality_profiles": _quality_profiles()}
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": row},
        selection_policies=selection_policies,
    )
    return config._dataset_specs()["alpha"]


def test_removed_dead_private_config_helpers_are_not_exposed():
    assert not hasattr(PipelineConfig, "_check_quality_profile_explicitness")
    assert not hasattr(PipelineConfig, "_canonical_selection_policies")


def test_generation3_requires_exact_quality_profile():
    row = _explicit_row()
    del row["quality_profile"]
    with pytest.raises(ValueError, match="missing explicit quality_profile"):
        _parse(row)


@pytest.mark.parametrize("bad", [None, "", " ", 1, "QUALITY_HIGH", "quality-high", "unknown"])
def test_generation3_rejects_invalid_quality_profile(bad):
    with pytest.raises(ValueError, match="quality_profile"):
        _parse(_explicit_row(quality_profile=bad))


@pytest.mark.parametrize("legacy", ["regular", "regular_strict", "special"])
def test_generation3_rejects_legacy_quality_category(legacy):
    with pytest.raises(ValueError, match="semantic category|quality_profile"):
        _parse(_explicit_row(category=legacy))


@pytest.mark.parametrize(
    "category", [None, "read_speech", "expressive", "elderly", "singing", "game_voice"]
)
def test_generation3_accepts_semantic_category_vocabulary(category):
    assert _parse(_explicit_row(category=category)).category == category


def test_bare_generation3_config_preserves_legacy_three_model_default(tmp_path):
    config = PipelineConfig(
        pipeline_generation=3,
        output_root=tmp_path / "output",
    )

    assert config.mos_types == ["scoreq", "urgent_pk", "dnsmos"]
    assert config.min_mos_models == 3


def test_min_source_sample_rate_defaults_to_none_and_preserves_legacy_serialization():
    config = PipelineConfig(pipeline_generation=3)

    assert config.min_source_sample_rate_hz is None
    assert "min_source_sample_rate_hz" not in config.to_dict()
    assert PipelineConfig(**config.to_dict()).min_source_sample_rate_hz is None


def test_min_source_sample_rate_accepts_positive_integer_and_round_trips():
    config = PipelineConfig(
        pipeline_generation=3,
        output_sample_rate_hz=32_000,
        min_source_sample_rate_hz=16_000,
    )

    assert config.min_source_sample_rate_hz == 16_000
    assert config.output_sample_rate_hz == 32_000
    assert config.to_dict()["min_source_sample_rate_hz"] == 16_000
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()


@pytest.mark.parametrize("configured", [None, 16_000])
def test_min_source_sample_rate_is_read_only_after_construction(configured):
    config = PipelineConfig(
        pipeline_generation=3,
        min_source_sample_rate_hz=configured,
    )

    with pytest.raises(AttributeError):
        config.min_source_sample_rate_hz = 32_000

    assert config.min_source_sample_rate_hz == configured


@pytest.mark.parametrize("value", [True, False, 0, -1, 16_000.0, "16000", [], {}])
def test_min_source_sample_rate_rejects_bool_non_integer_and_non_positive(value):
    with pytest.raises(
        ValueError,
        match="min_source_sample_rate_hz.*positive Python integer or None",
    ):
        PipelineConfig(
            pipeline_generation=3,
            min_source_sample_rate_hz=value,
        )


@pytest.mark.parametrize("selection_policies", [None, {}, {"global": {}}])
def test_generation3_empty_global_policy_preserves_legacy_three_of_three(
    tmp_path, selection_policies,
):
    policies = {"quality_profiles": _quality_profiles()}
    if isinstance(selection_policies, dict):
        policies.update(selection_policies)
    config = PipelineConfig(
        pipeline_generation=3,
        output_root=tmp_path / "output",
        datasets={"alpha": _explicit_row()},
        selection_policies=policies,
    )

    policy = config.resolve_selection_policy("alpha")

    assert policy.mos_models == ("scoreq", "urgent_pk", "dnsmos")
    assert policy.min_mos_models == 3
    # 显式 quality_high 的阈值在 global 之后合并；empty global 保留
    # legacy 构造默认，最终阈值等于该三模型 tuple 的全量值。
    assert dict(policy.thresholds) == {
        "dnsmos": {"ovrl": 3.3, "sig": 3.4, "bak": 4.0, "p808": 3.8},
        "scoreq": 4.2,
        "urgent_pk": 3.5,
    }


def test_generation3_default_policy_snapshot_is_stable_across_public_calls(tmp_path):
    config = PipelineConfig(
        pipeline_generation=3,
        output_root=tmp_path / "output",
        datasets={"alpha": _explicit_row()},
        selection_policies={"quality_profiles": _quality_profiles()},
    )
    expected_global = {
        "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
        "thresholds": {},
        "min_mos_models": 3,
        "mos_fusion": "min",
        "early_exit": None,
    }
    serialized = config.to_dict()

    assert serialized["selection_policies"]["global"] == expected_global
    config.resolve_selection_policy("alpha")
    serialized["selection_policies"]["global"]["mos_models"].append("utmos")

    assert config.to_dict()["selection_policies"]["global"] == expected_global
    assert config.mos_types == ["scoreq", "urgent_pk", "dnsmos"]


def test_generation3_legacy_default_does_not_narrow_five_model_registry():
    registered_models = set(list_mos_specs())

    assert registered_models == {
        "scoreq", "urgent_pk", "dnsmos", "nisqa", "utmos",
    }



def test_explicit_registry_semantics_preserve_selection_policy_resolution():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "categories": {"read_speech": {"mos_models": ["dnsmos"]}},
        },
    )

    policy = config.resolve_selection_policy("alpha")

    assert policy.policy_id == "alpha:en"
    assert policy.mos_models == ("dnsmos",)


@pytest.mark.parametrize("missing", ["speaker", "quality_profile"])
def test_generation3_rejects_missing_explicit_registry_field(missing):
    row = _explicit_row()
    del row[missing]

    with pytest.raises(ValueError, match=rf"dataset 'alpha'.*{missing}"):
        _parse(row)


@pytest.mark.parametrize(
    ("dataset_id", "row_overrides", "missing"),
    [
        (
            "voicebank_spk",
            {"input_root": "/data/speakers/voicebank_spk/session-01"},
            "speaker",
        ),
    ],
)
def test_generation3_does_not_infer_missing_semantics_from_bait_fields(
    dataset_id, row_overrides, missing,
):
    row = _explicit_row(**row_overrides)
    del row[missing]

    with pytest.raises(ValueError, match=rf"dataset '{dataset_id}'.*{missing}"):
        PipelineConfig(pipeline_generation=3, datasets={dataset_id: row})


@pytest.mark.parametrize(
    ("speaker", "error_field"),
    [
        (None, "mapping"),
        ({"source": "guessed"}, "source"),
        (
            {"source": "path_component", "path_component": 0, "regex": ".+"},
            "regex",
        ),
        ({"source": "none", "required": False, "unknown": True}, "unknown"),
        (
            {"source": "path_component", "path_component": 0, "required": 1},
            "required",
        ),
    ],
)
def test_explicit_speaker_failures_use_v3_parser(speaker, error_field):
    with pytest.raises(ValueError, match=rf"dataset 'alpha'.*{error_field}"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize("speaker", [None, [], "none", 1])
def test_generation3_requires_speaker_mapping(speaker):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*speaker.*mapping"):
        _parse(_explicit_row(speaker=speaker))


def test_speaker_source_must_be_explicit():
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*speaker.*source"):
        _parse(_explicit_row(speaker={"required": False}))


@pytest.mark.parametrize(
    "source", ["path", "mapping", "NONE", "", None, 1, True, [], {}],
)
def test_speaker_source_accepts_only_declared_values(source):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*speaker.*source"):
        _parse(_explicit_row(speaker={"source": source}))


@pytest.mark.parametrize("path_component", [None, True, False, -1, 1.0, "0"])
def test_path_component_source_requires_non_negative_exact_integer(path_component):
    speaker = {"source": "path_component"}
    if path_component is not None:
        speaker["path_component"] = path_component

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*path_component"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("regex", r"(?P<speaker_id>.+)"),
        ("regex_group", "speaker_id"),
        ("mapping_path", "/data/map.json"),
    ],
)
def test_path_component_source_rejects_other_source_fields(field, value):
    speaker = {"source": "path_component", "path_component": 1, field: value}

    with pytest.raises(ValueError, match=rf"dataset 'alpha'.*{field}"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize("regex", [None, "", "   ", "("])
def test_regex_source_requires_non_empty_compilable_pattern(regex):
    speaker = {"source": "regex", "regex_group": "speaker_id"}
    if regex is not None:
        speaker["regex"] = regex

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*regex"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize(
    ("regex_group", "pattern"),
    [
        ("", r"(?P<speaker_id>.+)"),
        ("   ", r"(?P<speaker_id>.+)"),
        ("speaker_id", r"(?P<other>.+)"),
    ],
)
def test_regex_source_requires_named_group_present_in_pattern(regex_group, pattern):
    speaker = {"source": "regex", "regex": pattern, "regex_group": regex_group}

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*regex_group"):
        _parse(_explicit_row(speaker=speaker))


def test_regex_source_defaults_to_speaker_id_named_group():
    spec = _parse(
        _explicit_row(
            speaker={"source": "regex", "regex": r"(?P<speaker_id>.+)"},
        )
    )

    assert spec.speaker.regex_group == "speaker_id"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path_component", 0),
        ("mapping_path", "/data/map.json"),
    ],
)
def test_regex_source_rejects_other_source_fields(field, value):
    speaker = {
        "source": "regex",
        "regex": r"(?P<speaker_id>.+)",
        field: value,
    }

    with pytest.raises(ValueError, match=rf"dataset 'alpha'.*{field}"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize("mapping_path", [None, "", "   ", 1, True])
def test_mapping_file_source_requires_non_empty_mapping_path(mapping_path):
    speaker = {"source": "mapping_file"}
    if mapping_path is not None:
        speaker["mapping_path"] = mapping_path

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*mapping_path"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path_component", 0),
        ("regex", r"(?P<speaker_id>.+)"),
        ("regex_group", "speaker_id"),
    ],
)
def test_mapping_file_source_rejects_other_source_fields(field, value):
    speaker = {
        "source": "mapping_file",
        "mapping_path": "/data/map.json",
        field: value,
    }

    with pytest.raises(ValueError, match=rf"dataset 'alpha'.*{field}"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize("speaker", [{"source": "none"}, {"source": "none", "required": True}])
def test_none_source_requires_required_false(speaker):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*required.*False"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path_component", 0),
        ("regex", r"(?P<speaker_id>.+)"),
        ("regex_group", "speaker_id"),
        ("mapping_path", "/data/map.json"),
    ],
)
def test_none_source_rejects_all_source_specific_fields(field, value):
    speaker = {"source": "none", "required": False, field: value}

    with pytest.raises(ValueError, match=rf"dataset 'alpha'.*{field}"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize("required", [0, 1, None, "false"])
def test_speaker_required_must_be_exact_bool(required):
    speaker = {
        "source": "path_component",
        "path_component": 0,
        "required": required,
    }

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*required.*bool"):
        _parse(_explicit_row(speaker=speaker))


def test_speaker_mapping_rejects_unknown_fields():
    speaker = {"source": "none", "required": False, "guess_from_id": True}

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*guess_from_id"):
        _parse(_explicit_row(speaker=speaker))


def test_speaker_mapping_rejects_mixed_type_unknown_field_names_clearly():
    speaker = {
        "source": "none",
        "required": False,
        "extra": True,
        1: True,
    }

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*field"):
        _parse(_explicit_row(speaker=speaker))


@pytest.mark.parametrize(
    ("speaker", "expected"),
    [
        (
            {"source": "path_component", "path_component": 2},
            {
                "source": "path_component", "path_component": 2,
                "regex": None, "mapping_path": None,
                "regex_group": "speaker_id", "required": True,
            },
        ),
        (
            {
                "source": "regex",
                "regex": r"/(?P<speaker_id>spk-[^/]+)/",
                "regex_group": "speaker_id",
                "required": False,
            },
            {
                "source": "regex", "path_component": None,
                "regex": r"/(?P<speaker_id>spk-[^/]+)/",
                "mapping_path": None, "regex_group": "speaker_id",
                "required": False,
            },
        ),
        (
            {"source": "mapping_file", "mapping_path": "/data/map.json"},
            {
                "source": "mapping_file", "path_component": None,
                "regex": None, "mapping_path": "/data/map.json",
                "regex_group": "speaker_id", "required": True,
            },
        ),
        (
            {"source": "none", "required": False},
            {
                "source": "none", "path_component": None,
                "regex": None, "mapping_path": None,
                "regex_group": "speaker_id", "required": False,
            },
        ),
    ],
)
def test_all_speaker_sources_parse_to_immutable_metadata(speaker, expected):
    spec = _parse(_explicit_row(speaker=speaker))

    assert asdict(spec.speaker) == expected
    with pytest.raises(FrozenInstanceError):
        spec.speaker.required = not spec.speaker.required


def test_dataset_spec_is_immutable():
    spec = _parse(_explicit_row())

    with pytest.raises(FrozenInstanceError):
        spec.quality_profile = "quality_premium"


@pytest.mark.parametrize("dataset_id", [1, True, "", "   "])
def test_dataset_registry_requires_non_empty_exact_string_ids(dataset_id):
    with pytest.raises(ValueError, match="dataset.*id.*non-empty string"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={dataset_id: _explicit_row()},
        )


def test_dataset_registry_rejects_stringification_collision_channel():
    with pytest.raises(ValueError, match="dataset.*id.*string"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={1: _explicit_row(), "1": _explicit_row()},
        )


@pytest.mark.parametrize("unknown_key", ["langauge", 1, True])
def test_generation3_dataset_rows_reject_unknown_or_non_string_keys(unknown_key):
    row = _explicit_row()
    row[unknown_key] = "unexpected"

    with pytest.raises(ValueError, match=r"dataset 'alpha'.*(field|unknown)"):
        PipelineConfig(pipeline_generation=3, datasets={"alpha": row})



def test_dataset_registry_snapshot_isolated_from_caller_mutation():
    row = _explicit_row(
        speaker={"source": "mapping_file", "mapping_path": "maps/speakers.jsonl"},
    )
    datasets = {"alpha": row}
    config = PipelineConfig(
        pipeline_generation=3,
        datasets=datasets,
        selection_policies={"quality_profiles": _quality_profiles()},
    )
    before = config.to_dict()["datasets"]

    row["language"] = "FR"
    row["speaker"]["mapping_path"] = "changed.jsonl"
    datasets["beta"] = _explicit_row()

    assert config.to_dict()["datasets"] == before
    assert config.dataset_spec("alpha").language == "en"
    assert config.dataset_spec("alpha").speaker.mapping_path == "maps/speakers.jsonl"
    with pytest.raises(ValueError, match="unknown dataset"):
        config.dataset_spec("beta")


def test_public_dataset_accessors_expose_immutable_snapshot():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    specs = config.dataset_specs()
    assert isinstance(specs, MappingProxyType)
    assert specs["alpha"] is config.dataset_spec("alpha")
    with pytest.raises(TypeError):
        specs["beta"] = specs["alpha"]


def test_generation3_dataset_serialization_is_canonical_and_complete():
    first = PipelineConfig(
        pipeline_generation=3,
        datasets={
            "alpha": {
                "speaker": {
                    "mapping_path": "maps/speakers.jsonl",
                    "source": "mapping_file",
                },
                "category": "read_speech",
                "quality_profile": "quality_high",
                "language": "EN_us",
                "input_root": "/data/alpha",
            },
        },
        selection_policies={"quality_profiles": _quality_profiles()},
    )
    second = PipelineConfig(
        pipeline_generation=3,
        datasets={
            "alpha": {
                "input_root": "/data/alpha",
                "language": "en-US",
                "category": "read_speech",
                "quality_profile": "quality_high",
                "speaker": {
                    "source": "mapping_file",
                    "mapping_path": "maps/speakers.jsonl",
                },
            },
        },
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    expected = {
        "alpha": {
            "input_root": "/data/alpha",
            "language": "en",
            "category": "read_speech",
            "quality_profile": "quality_high",
            "speaker": {
                "source": "mapping_file",
                "mapping_path": "maps/speakers.jsonl",
                "required": True,
            },
        },
    }
    assert first.to_dict()["datasets"] == expected
    assert first.to_dict()["datasets"] == second.to_dict()["datasets"]
    assert PipelineConfig(**first.to_dict()).to_dict()["datasets"] == expected


@pytest.mark.parametrize(
    ("source_selector", "expected_sample_rates", "expected_speaker_ids"),
    [
        ({"sample_rates_hz": [48000, 16000, 48000]}, (16000, 48000), ()),
        (
            {"include_speaker_ids": ["speaker-b", "speaker-a", "speaker-b"]},
            (),
            ("speaker-a", "speaker-b"),
        ),
        (
            {
                "sample_rates_hz": [32000, 16000, 32000],
                "include_speaker_ids": ["speaker-b", "speaker-a", "speaker-b"],
            },
            (16000, 32000),
            ("speaker-a", "speaker-b"),
        ),
    ],
)
def test_source_selector_parses_canonical_deduplicated_values(
    source_selector, expected_sample_rates, expected_speaker_ids,
):
    spec = _parse(_explicit_row(source_selector=source_selector))

    assert spec.source_selector.sample_rates_hz == expected_sample_rates
    assert spec.source_selector.include_speaker_ids == expected_speaker_ids


def test_source_selector_parses_canonical_excluded_relative_paths():
    spec = _parse(_explicit_row(source_selector={
        "exclude_relative_paths": [
            "noise/b.flac", "noise/a.flac", "noise/b.flac",
        ],
    }))

    assert spec.source_selector == SourceSelectorSpec(
        exclude_relative_paths=("noise/a.flac", "noise/b.flac"),
    )


@pytest.mark.parametrize("source_selector", [None, [], "16000", 16000, True])
def test_source_selector_requires_mapping(source_selector):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*source_selector.*mapping"):
        _parse(_explicit_row(source_selector=source_selector))


@pytest.mark.parametrize("unknown_key", ["sample_rate_hz", "speaker_ids", 1, True])
def test_source_selector_rejects_unknown_or_non_string_keys(unknown_key):
    source_selector = {"sample_rates_hz": [16000], unknown_key: [48000]}

    with pytest.raises(
        ValueError,
        match=r"dataset 'alpha'.*source_selector.*(field|unknown)",
    ):
        _parse(_explicit_row(source_selector=source_selector))


@pytest.mark.parametrize(
    "exclude_relative_paths",
    [
        None,
        "noise/a.flac",
        [""],
        ["/noise/a.flac"],
        ["noise\\a.flac"],
        ["noise/../a.flac"],
        ["noise//a.flac"],
        [True],
    ],
)
def test_source_selector_rejects_invalid_excluded_relative_paths(
    exclude_relative_paths,
):
    with pytest.raises(
        ValueError,
        match=r"dataset 'alpha'.*exclude_relative_paths.*(list or tuple|normalized relative POSIX)",
    ):
        _parse(_explicit_row(source_selector={
            "exclude_relative_paths": exclude_relative_paths,
        }))


@pytest.mark.parametrize(
    "source_selector",
    [
        {},
        {"sample_rates_hz": []},
        {"include_speaker_ids": []},
        {"sample_rates_hz": [], "include_speaker_ids": []},
    ],
)
def test_source_selector_rejects_empty_selection(source_selector):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*source_selector.*empty"):
        _parse(_explicit_row(source_selector=source_selector))


@pytest.mark.parametrize(
    "sample_rates_hz",
    [[0], [-16000], [True], [False], [16000.0], ["16000"]],
)
def test_source_selector_rejects_non_positive_or_non_exact_integer_sample_rates(
    sample_rates_hz,
):
    with pytest.raises(
        ValueError,
        match=r"dataset 'alpha'.*source_selector.*sample_rates_hz.*positive integer",
    ):
        _parse(
            _explicit_row(
                source_selector={"sample_rates_hz": sample_rates_hz},
            )
        )


@pytest.mark.parametrize("sample_rates_hz", [[8_000], [16_000, 8_000], [48_000, 15_999]])
def test_source_selector_rejects_any_sample_rate_below_configured_floor(
    sample_rates_hz,
):
    with pytest.raises(
        ValueError,
        match=r"dataset 'alpha'.*sample_rates_hz.*min_source_sample_rate_hz.*16000",
    ):
        PipelineConfig(
            pipeline_generation=3,
            min_source_sample_rate_hz=16_000,
            datasets={
                "alpha": _explicit_row(
                    source_selector={"sample_rates_hz": sample_rates_hz},
                ),
            },
            selection_policies={"quality_profiles": _quality_profiles()},
        )


def test_source_selector_accepts_sample_rates_equal_to_or_above_configured_floor():
    config = PipelineConfig(
        pipeline_generation=3,
        min_source_sample_rate_hz=16_000,
        datasets={
            "alpha": _explicit_row(
                source_selector={"sample_rates_hz": [48_000, 16_000]},
            ),
        },
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    assert config.dataset_spec("alpha").source_selector.sample_rates_hz == (
        16_000,
        48_000,
    )


@pytest.mark.parametrize(
    "speaker_id", ["", " ", "\t", " speaker-a", "speaker-a ", "\tspeaker-a"]
)
def test_source_selector_rejects_empty_or_whitespace_padded_speaker_ids(speaker_id):
    with pytest.raises(
        ValueError,
        match=r"dataset 'alpha'.*source_selector.*include_speaker_ids",
    ):
        _parse(
            _explicit_row(
                source_selector={"include_speaker_ids": [speaker_id]},
            )
        )


def test_source_selector_speaker_ids_require_non_none_speaker_source():
    with pytest.raises(
        ValueError,
        match=r"dataset 'alpha'.*include_speaker_ids.*speaker source.*none",
    ):
        _parse(
            _explicit_row(
                speaker={"source": "none", "required": False},
                source_selector={"include_speaker_ids": ["speaker-a"]},
            )
        )


def test_dataset_without_source_selector_preserves_legacy_serialized_shape():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    assert set(config.to_dict()["datasets"]["alpha"]) == {
        "input_root",
        "language",
        "category",
        "quality_profile",
        "speaker",
    }


def test_configured_source_selector_serializes_lists_and_round_trips():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={
            "alpha": _explicit_row(
                source_selector={
                    "sample_rates_hz": [48000, 16000, 48000],
                    "include_speaker_ids": ["speaker-b", "speaker-a", "speaker-b"],
                    "exclude_relative_paths": ["noise/b.flac", "noise/a.flac"],
                },
            ),
        },
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    serialized = config.to_dict()
    assert serialized["datasets"]["alpha"]["source_selector"] == {
        "sample_rates_hz": [16000, 48000],
        "include_speaker_ids": ["speaker-a", "speaker-b"],
        "exclude_relative_paths": ["noise/a.flac", "noise/b.flac"],
    }
    assert PipelineConfig(**serialized).to_dict() == serialized


def _same_canonical_root_config(
    tmp_path,
    first_selector=None,
    second_selector=None,
    first_speaker=None,
    second_speaker=None,
):
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    first = _explicit_row(input_root=str(shared_root))
    second = _explicit_row(input_root=f"{shared_root}/.")
    if first_selector is not None:
        first["source_selector"] = first_selector
    if second_selector is not None:
        second["source_selector"] = second_selector
    if first_speaker is not None:
        first["speaker"] = first_speaker
    if second_speaker is not None:
        second["speaker"] = second_speaker
    return PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": first, "beta": second},
        selection_policies={"quality_profiles": _quality_profiles()},
    )


@pytest.mark.parametrize(
    ("first_selector", "second_selector"),
    [
        (
            {"sample_rates_hz": [16000]},
            {"sample_rates_hz": [48000]},
        ),
        (
            {"include_speaker_ids": ["speaker-a"]},
            {"include_speaker_ids": ["speaker-b"]},
        ),
        (
            {
                "sample_rates_hz": [16000],
                "include_speaker_ids": ["shared-speaker"],
            },
            {
                "sample_rates_hz": [48000],
                "include_speaker_ids": ["shared-speaker"],
            },
        ),
        (
            {
                "sample_rates_hz": [16000],
                "include_speaker_ids": ["speaker-a"],
            },
            {
                "sample_rates_hz": [16000],
                "include_speaker_ids": ["speaker-b"],
            },
        ),
    ],
)
def test_same_canonical_root_accepts_provably_disjoint_source_selectors(
    tmp_path, first_selector, second_selector,
):
    config = _same_canonical_root_config(
        tmp_path,
        first_selector=first_selector,
        second_selector=second_selector,
    )

    assert set(config.dataset_specs()) == {"alpha", "beta"}


def test_same_canonical_root_accepts_disjoint_speakers_with_identical_resolver(tmp_path):
    config = _same_canonical_root_config(
        tmp_path,
        first_selector={"include_speaker_ids": ["speaker-a"]},
        second_selector={"include_speaker_ids": ["speaker-b"]},
        first_speaker={"source": "path_component", "path_component": 1},
        second_speaker={"source": "path_component", "path_component": 1},
    )

    assert set(config.dataset_specs()) == {"alpha", "beta"}


def test_same_canonical_root_accepts_disjoint_speakers_with_identical_regex(tmp_path):
    speaker = {"source": "regex", "regex": r"(?P<speaker_id>[^/]+)"}
    config = _same_canonical_root_config(
        tmp_path,
        first_selector={"include_speaker_ids": ["speaker-a"]},
        second_selector={"include_speaker_ids": ["speaker-b"]},
        first_speaker=speaker,
        second_speaker=speaker,
    )

    assert set(config.dataset_specs()) == {"alpha", "beta"}


def test_same_canonical_root_rejects_disjoint_speakers_with_mapping_file(tmp_path):
    speaker = {"source": "mapping_file", "mapping_path": "maps/speakers.jsonl"}
    with pytest.raises(ValueError, match=r"speaker-set exclusivity.*path_component.*regex"):
        _same_canonical_root_config(
            tmp_path,
            first_selector={"include_speaker_ids": ["speaker-a"]},
            second_selector={"include_speaker_ids": ["speaker-b"]},
            first_speaker=speaker,
            second_speaker=speaker,
        )


def test_same_canonical_root_accepts_disjoint_rates_with_mapping_file(tmp_path):
    speaker = {"source": "mapping_file", "mapping_path": "maps/speakers.jsonl"}
    config = _same_canonical_root_config(
        tmp_path,
        first_selector={
            "sample_rates_hz": [16_000],
            "include_speaker_ids": ["speaker-a"],
        },
        second_selector={
            "sample_rates_hz": [48_000],
            "include_speaker_ids": ["speaker-a"],
        },
        first_speaker=speaker,
        second_speaker=speaker,
    )

    assert set(config.dataset_specs()) == {"alpha", "beta"}


@pytest.mark.parametrize(
    ("first_speaker", "second_speaker"),
    [
        (
            {"source": "path_component", "path_component": 0},
            {"source": "path_component", "path_component": 1},
        ),
        (
            {"source": "path_component", "path_component": 0},
            {"source": "regex", "regex": r"(?P<speaker_id>[^/]+)"},
        ),
    ],
)
def test_same_canonical_root_rejects_disjoint_speakers_with_different_resolvers(
    tmp_path, first_speaker, second_speaker,
):
    with pytest.raises(ValueError, match=r"speaker.*(metadata|resolver)|source_selector"):
        _same_canonical_root_config(
            tmp_path,
            first_selector={"include_speaker_ids": ["speaker-a"]},
            second_selector={"include_speaker_ids": ["speaker-b"]},
            first_speaker=first_speaker,
            second_speaker=second_speaker,
        )


def test_same_canonical_root_disjoint_sample_rates_override_resolver_difference(tmp_path):
    config = _same_canonical_root_config(
        tmp_path,
        first_selector={
            "sample_rates_hz": [16000],
            "include_speaker_ids": ["speaker-a"],
        },
        second_selector={
            "sample_rates_hz": [48000],
            "include_speaker_ids": ["speaker-a"],
        },
        first_speaker={"source": "path_component", "path_component": 0},
        second_speaker={"source": "regex", "regex": r"(?P<speaker_id>[^/]+)"},
    )

    assert set(config.dataset_specs()) == {"alpha", "beta"}


@pytest.mark.parametrize(
    ("first_selector", "second_selector"),
    [
        ({"sample_rates_hz": [16000]}, None),
        ({"sample_rates_hz": [16000]}, {"include_speaker_ids": ["speaker-b"]}),
        ({"sample_rates_hz": [16000]}, {"sample_rates_hz": [16000, 48000]}),
        (
            {"include_speaker_ids": ["speaker-a"]},
            {"include_speaker_ids": ["speaker-a", "speaker-b"]},
        ),
        (
            {
                "sample_rates_hz": [16000, 48000],
                "include_speaker_ids": ["speaker-a", "speaker-b"],
            },
            {
                "sample_rates_hz": [48000],
                "include_speaker_ids": ["speaker-b", "speaker-c"],
            },
        ),
    ],
)
def test_same_canonical_root_rejects_unproven_or_overlapping_selectors(
    tmp_path, first_selector, second_selector,
):
    with pytest.raises(ValueError, match=r"(input_root|source_selector|selector)"):
        _same_canonical_root_config(
            tmp_path,
            first_selector=first_selector,
            second_selector=second_selector,
        )


def test_same_canonical_root_rejects_two_datasets_without_selectors(tmp_path):
    with pytest.raises(ValueError, match=r"source_selectors.*provably disjoint"):
        _same_canonical_root_config(tmp_path)


def test_mapping_path_is_opaque_metadata_owned_by_b2():
    relative_path = "../opaque/speaker-map.jsonl"
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={
            "alpha": _explicit_row(
                speaker={"source": "mapping_file", "mapping_path": relative_path},
            ),
        },
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    assert config.dataset_spec("alpha").speaker.mapping_path == relative_path
    assert not Path(relative_path).is_absolute()
    assert "B2" in (SpeakerMetadataSpec.__doc__ or "")
    assert "不读取" in (SpeakerMetadataSpec.__doc__ or "")


@pytest.mark.parametrize("category", [[], {}, 1, True, ""])
def test_generation3_category_is_none_or_non_empty_exact_string(category):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*category"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={"alpha": _explicit_row(category=category)},
        )


@pytest.mark.parametrize("language", [1, True, [], {}, "   "])
def test_generation3_language_requires_exact_non_empty_string(language):
    with pytest.raises(ValueError, match=r"dataset 'alpha'.*language.*non-empty string"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={"alpha": _explicit_row(language=language)},
        )



def test_v3_selection_policies_are_canonical_immutable_and_round_trip():
    policies = {
        "quality_profiles": _quality_profiles(),
        "languages": {
            "en": {
                "early_exit": {"threshold": 3.5, "enabled": False},
                "mos_fusion": " MIN ",
                "thresholds": {"DNSMOS": {"sig": 3.1, "ovrl": 3.0}},
                "models": ("DNSMOS", "ScoreQ"),
                "min_mos_models": 1,
            },
        },
    }
    config = PipelineConfig(
        pipeline_generation=3,
        step="step1",
        output_root="/tmp/output",
        datasets={"alpha": _explicit_row()},
        selection_policies=policies,
    )
    before = config.resolve_selection_policy("alpha")

    policies["languages"]["en"]["models"] = ["utmos"]
    policies["languages"]["en"]["thresholds"]["DNSMOS"]["ovrl"] = 4.9
    serialized = config.to_dict()
    serialized["selection_policies"]["languages"]["en"]["mos_models"].append("utmos")

    expected = {
        "global": {
            "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
            "thresholds": {},
            "min_mos_models": 3,
            "mos_fusion": "min",
            "early_exit": None,
        },
        "quality_profiles": _quality_profiles(),
        "languages": {
            "en": {
                "mos_models": ["dnsmos", "scoreq"],
                "thresholds": {"dnsmos": {"ovrl": 3.0, "sig": 3.1}},
                "min_mos_models": 1,
                "mos_fusion": "min",
                "early_exit": {"enabled": False, "threshold": 3.5},
            },
        },
    }
    assert config.to_dict()["selection_policies"] == expected
    assert config.resolve_selection_policy("alpha") == before
    assert PipelineConfig(**config.to_dict()).to_dict() == config.to_dict()
    with pytest.raises(TypeError):
        config.selection_policies["languages"]["en"]["thresholds"]["dnsmos"]["ovrl"] = 4.0


def test_v3_semantically_equivalent_selection_policies_serialize_identically():
    first = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {
                "models": ("DNSMOS", "ScoreQ"),
                "thresholds": {"ScoreQ": 3.2, "DNSMOS": 3.0},
            },
        },
    )
    second = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {
                "thresholds": {"dnsmos": 3.0, "scoreq": 3.2},
                "mos_models": ["dnsmos", "scoreq"],
            },
        },
    )

    assert first.to_dict()["selection_policies"] == second.to_dict()["selection_policies"]


@pytest.mark.parametrize(
    ("early_exit", "error"),
    [
        (object(), "canonical JSON"),
        ({1: True}, "keys.*strings"),
        ({"threshold": float("inf")}, "finite"),
    ],
)
def test_v3_early_exit_requires_recursive_canonical_json(early_exit, error):
    with pytest.raises(ValueError, match=error):
        PipelineConfig(
            pipeline_generation=3,
            datasets={"alpha": _explicit_row()},
            selection_policies={
                "quality_profiles": _quality_profiles(),
                "global": {"early_exit": early_exit},
            },
        )


@pytest.mark.parametrize(
    ("early_exit", "error"),
    [
        ({"enabled": "false"}, "enabled.*bool"),
        ({"enabled": True, "threshold": []}, "threshold"),
        ({"enabled": True, "unknown": 3.0}, "unknown.*early_exit"),
        ({"enabled": True, "order": ["dnsmos", "dnsmos"]}, "duplicate"),
    ],
)
def test_v3_validate_rejects_invalid_early_exit_execution_schema(
    early_exit, error,
):
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {
                "mos_models": ["dnsmos"],
                "min_mos_models": 1,
                "early_exit": early_exit,
            },
        },
    )

    with pytest.raises(ValueError, match=error):
        config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])


def test_v3_validate_rejects_enabled_early_exit_until_execution_is_implemented():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {
                "early_exit": {
                    "enabled": True,
                    "threshold": 3.5,
                    "order": ["scoreq", "urgent_pk", "dnsmos"],
                },
            },
        },
    )

    with pytest.raises(ValueError, match="early_exit.*not implemented"):
        config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])


@pytest.mark.parametrize("fusion", ["average", "max", "bogus"])
def test_v3_validate_rejects_unsupported_mos_fusion_semantics(fusion):
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"mos_fusion": fusion},
        },
    )

    with pytest.raises(ValueError, match="mos_fusion must be 'min'.*K/N"):
        config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])


def test_v3_min_fusion_is_canonical_and_absent_from_decision_hash_identity():
    implicit = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={"quality_profiles": _quality_profiles()},
    )
    explicit = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"mos_fusion": " MIN "},
        },
    )

    implicit_policy = implicit.resolve_selection_policy("alpha")
    explicit_policy = explicit.resolve_selection_policy("alpha")
    assert implicit_policy.mos_fusion == explicit_policy.mos_fusion == "min"
    assert implicit_policy.policy_hash == explicit_policy.policy_hash


@pytest.mark.parametrize(
    "thresholds",
    [
        {"dnsmos": True},
        {"dnsmos": {"ovrl": True}},
    ],
)
def test_v3_validate_rejects_boolean_thresholds(thresholds):
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {
                "mos_models": ["dnsmos"],
                "min_mos_models": 1,
            },
            # boolean 阈值放在 dataset 层（最末合并层）：若放在 global 层，
            # 会被显式 quality_high 的合法阈值整体覆盖，失去拒绝语义
            "datasets": {"alpha": {"thresholds": thresholds}},
        },
    )

    with pytest.raises(ValueError, match="阈值"):
        config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])



@pytest.mark.parametrize(
    "selection_policies",
    [
        {"   ": {}},
        {"languages": {"   ": {}}},
        {"global": {"   ": None}},
        {"global": {"early_exit": {"   ": True}}},
    ],
)
def test_v3_selection_policy_keys_reject_whitespace_only_strings(
    selection_policies,
):
    with pytest.raises(ValueError, match="non-empty strings"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={"alpha": _explicit_row()},
            selection_policies={
                "quality_profiles": _quality_profiles(),
                **selection_policies,
            },
        )


def test_v3_selection_policy_attribute_rebinding_cannot_replace_snapshot():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"models": ["DNSMOS"], "min_mos_models": 1},
        },
    )
    snapshot = config.selection_policies
    serialized = config.to_dict()
    resolved = config.resolve_selection_policy("alpha")

    with pytest.raises(AttributeError, match="selection_policies.*read-only"):
        config.selection_policies = {"global": {"models": ["utmos"]}}
    config.__dict__["selection_policies"] = {"global": {"models": ["utmos"]}}

    config.validate(context=ValidationContext.STRUCTURAL)
    assert config.selection_policies is snapshot
    assert config.to_dict() == serialized
    assert config.resolve_selection_policy("alpha") == resolved


def test_v3_selection_policy_snapshot_rejects_in_place_union_mutation():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"early_exit": {"enabled": True}},
        },
    )
    policy = config.selection_policies["global"]
    before = config.to_dict()

    with pytest.raises(TypeError):
        policy |= {"early_exit": object()}

    assert config.to_dict() == before


def test_v3_selection_policy_snapshot_cannot_be_reinitialized():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "global": {"early_exit": {"enabled": True}},
        },
    )
    policy = config.selection_policies["global"]
    before = config.to_dict()

    policy.__init__({"early_exit": object()})
    with pytest.raises(TypeError):
        dict.__setitem__(policy, "early_exit", object())

    assert config.to_dict() == before


def test_v3_language_policy_targets_are_canonical_and_effective():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "languages": {"EN_us": {"models": ("DNSMOS",)}},
        },
    )

    assert config.to_dict()["selection_policies"] == {
        "global": {
            "mos_models": ["scoreq", "urgent_pk", "dnsmos"],
            "thresholds": {},
            "min_mos_models": 3,
            "mos_fusion": "min",
            "early_exit": None,
        },
        "quality_profiles": _quality_profiles(),
        "languages": {"en": {"mos_models": ["dnsmos"]}},
    }
    assert config.resolve_selection_policy("alpha").mos_models == ("dnsmos",)


def test_dataset_local_policy_bindings_include_only_reachable_routes():
    profiles = _quality_profiles()
    profiles["quality_high"]["thresholds"]["utmos"] = 3.5
    policies = {
        "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
        "quality_profiles": profiles,
        "datasets": {
            "beta": {
                "mos_models": ["utmos"],
                "min_mos_models": 1,
                "thresholds": {"utmos": 3.5},
            },
        },
        "sources": {
            "alpha": {
                "speaker/one.wav": {
                    "mos_models": ["scoreq"],
                    "min_mos_models": 1,
                },
            },
        },
    }
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={
            "alpha": _explicit_row(input_root="/data/alpha"),
            "beta": _explicit_row(input_root="/data/beta"),
        },
        selection_policies=policies,
    )

    scoring = config.dataset_mos_scoring_binding("alpha")
    decision = config.dataset_mos_decision_binding("alpha")
    before = (scoring, decision)

    assert scoring["schema"] == "audio_filter_pipeline.dataset_scoring_binding.v1"
    assert scoring["dataset_id"] == "alpha"
    assert [item["route_identity"] for item in scoring["projections"]] == [
        None,
        {"dataset_id": "alpha", "source_relative_path": "speaker/one.wav"},
    ]
    assert decision["schema"] == "audio_filter_pipeline.dataset_decision_binding.v1"
    assert [item["source_relative_path"] for item in decision["policies"]] == [
        None,
        "speaker/one.wav",
    ]
    assert all(item["policy_id"].startswith("alpha:en") for item in decision["policies"])

    policies["datasets"]["beta"]["mos_models"] = ["scoreq"]
    policies["datasets"]["beta"]["thresholds"] = {"scoreq": 4.5}

    assert (config.dataset_mos_scoring_binding("alpha"),
            config.dataset_mos_decision_binding("alpha")) == before
    with pytest.raises(TypeError):
        scoring["projections"] = ()
    with pytest.raises(ValueError, match="unknown dataset"):
        config.dataset_mos_scoring_binding("missing")


def test_v3_unreachable_language_policy_does_not_expand_selected_model_union():
    profiles = _quality_profiles()
    for profile in profiles.values():
        profile["thresholds"]["utmos"] = 3.5
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row(language="en")},
        selection_policies={
            "quality_profiles": profiles,
            "global": {"mos_models": ["scoreq"], "min_mos_models": 1},
            "languages": {"fr": {"mos_models": ["utmos"]}},
        },
    )

    config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])
    assert config.selected_mos_models() == ("scoreq",)


def test_v3_unreachable_language_policy_is_not_cross_applied_to_dataset_semantics():
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row(language="en")},
        selection_policies={
            "quality_profiles": _quality_profiles(),
            "languages": {
                "fr": {
                    "mos_models": ["dnsmos"],
                    "min_mos_models": 2,
                },
            },
            "sources": {
                "alpha": {
                    "speaker/one.wav": {
                        "mos_models": ["scoreq"],
                        "min_mos_models": 1,
                    },
                },
            },
        },
    )

    config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])
    assert config.resolve_selection_policy("alpha").mos_models == (
        "scoreq", "urgent_pk", "dnsmos",
    )
    assert config.resolve_selection_policy(
        "alpha", source_relative_path="speaker/one.wav",
    ).mos_models == ("scoreq",)


def test_v3_rejects_colliding_canonical_language_policy_targets():
    with pytest.raises(ValueError, match="duplicate.*language"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={"alpha": _explicit_row()},
            selection_policies={
                "quality_profiles": _quality_profiles(),
                "languages": {"EN_us": {}, "en-GB": {}},
            },
        )


@pytest.mark.parametrize("language", ["_", "___", "-"])
def test_v3_rejects_language_policy_targets_that_normalize_empty(language):
    with pytest.raises(ValueError, match="language.*non-empty string"):
        PipelineConfig(
            pipeline_generation=3,
            datasets={"alpha": _explicit_row()},
            selection_policies={
                "quality_profiles": _quality_profiles(),
                "languages": {language: {}},
            },
        )


@pytest.mark.parametrize("language", [1, True, [], {}, "", "   ", "_", "___", "-"])
def test_v3_public_policy_resolver_requires_exact_language(language):
    config = PipelineConfig(
        pipeline_generation=3,
        datasets={"alpha": _explicit_row()},
        selection_policies={"quality_profiles": _quality_profiles()},
    )

    with pytest.raises(ValueError, match="language.*non-empty string"):
        config.resolve_selection_policy("alpha", language)
