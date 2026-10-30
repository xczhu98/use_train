"""generation-3 shipped 配置及其生成器的契约测试。"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from audio_filter_pipeline.cli.main import load_config
from audio_filter_pipeline.config import PipelineConfig, ValidationContext

pytestmark = pytest.mark.unit

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_PATH = CONFIG_DIR / "spk_filter.yaml"
VARIANT_CONFIGS = {
    "audio_filter_zh_32k.yaml": ({"zh"}, 1, 2, 32, None, 32_000, 32_000),
    "audio_filter_en_32k.yaml": ({"en"}, 3, 11, 32, None, 32_000, 32_000),
    "audio_filter_mixed_32k.yaml": ({"mixed"}, 1, 2, 32, None, 32_000, 32_000),
    "audio_filter_zh_16k.yaml": ({"zh"}, 0, 4, 0, 32, 16_000, 16_000),
    "audio_filter_en_16k.yaml": ({"en"}, 3, 0, 0, 32, 16_000, 16_000),
}
UNLABELED_VARIANTS = {"audio_filter_zh_16k.yaml", "audio_filter_en_16k.yaml"}
VARIANT_OUTPUT_ROOTS = {
    "audio_filter_zh_32k.yaml": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_zh_32k",
    "audio_filter_en_32k.yaml": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_en_32k",
    "audio_filter_mixed_32k.yaml": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_mixed_32k",
    "audio_filter_zh_16k.yaml": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_zh_16k",
    "audio_filter_en_16k.yaml": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_en_16k",
}
SHIPPED_CONFIG_NAMES = ["spk_filter.yaml", *VARIANT_CONFIGS]
REMOVED_CONFIG_NAMES = {"audio_filter_mixed_16k.yaml"}
DEFAULT_ROUTE_MODELS = ["scoreq", "dnsmos"]
OPTIONAL_MOS_MODELS = {"nisqa", "utmos", "urgent_pk"}
MIN_SOURCE_SAMPLE_RATE_HZ = 16_000
LEGACY_CONFIG_NAMES = {
    "audio_filter.yaml",
    "audio_filter_32k_plus_zh.yaml",
    "audio_filter_32k_plus_en.yaml",
    "audio_filter_32k_plus_others.yaml",
}
LEGACY_SELECTION_FIELDS = {
    "mos_types",
    "mos_threshold",
    "mos_thresholds",
    "enable_multi_mos",
    "min_mos_models",
    "mos_fusion",
    "step2_direct_from_input",
    "step2_mos_early_exit_threshold",
}
KNOWN_LANGUAGES = {"zh", "en", "mixed"}
KNOWN_CATEGORIES = {"read_speech", "expressive", "elderly", "singing", "game_voice"}
KNOWN_PROFILES = {"quality_balanced", "quality_high", "quality_premium"}
URGENT_SHA256 = "266e995f745f265ea52277ca657f86c7d27077b385136edd85f7ef923bdfed5d"
URGENT_COMMIT = "484269f41ef17ea2419952e92ace7109b4b96052"
COMMENTED_DATASETS = {
    "aishell-3_spk", "savee_spk", "nnces_spontaneous_spk", "nnces_read_spk",
    "vocalset_spk", "hifitts2_spk", "libritts_r_spk", "commonvoice_en_spk",
    "bibletts_spk", "dailytalk_spk", "expresso_spk", "hifi_tts_spk", "ears_spk",
    "commonvoice_cn_spk", "vctk_spk", "gtsinger_spk",
}


def _raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_configs", CONFIG_DIR / "generate_configs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_catalog():
    spec = importlib.util.spec_from_file_location("dataset_catalog", CONFIG_DIR / "dataset_catalog.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _commented_ids(path: Path) -> set[str]:
    pattern = re.compile(r"^#\s+([A-Za-z0-9_.-]+): \{input_root:")
    return {
        match.group(1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := pattern.match(line))
    }


def test_shipped_yaml_configs_exist():
    shipped_names = {path.name for path in CONFIG_DIR.glob("*.yaml")}
    assert sorted(shipped_names) == sorted(SHIPPED_CONFIG_NAMES)
    assert not shipped_names & LEGACY_CONFIG_NAMES
    assert not shipped_names & REMOVED_CONFIG_NAMES
    assert not any(path.name.startswith("audio_filter_32k_plus_") for path in CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("filename", SHIPPED_CONFIG_NAMES)
def test_shipped_configs_do_not_use_legacy_selection_fields(filename):
    raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))

    assert raw["pipeline_generation"] == 3
    assert LEGACY_SELECTION_FIELDS.isdisjoint(raw)


def test_shipped_configs_preserve_original_language_rate_and_speaker_boundaries():
    expected_output_rates = {
        "spk_filter.yaml": 16_000,
        **{filename: values[-2] for filename, values in VARIANT_CONFIGS.items()},
    }
    expected_source_floors = {
        "spk_filter.yaml": 16_000,
        **{filename: values[-1] for filename, values in VARIANT_CONFIGS.items()},
    }
    shipped = {
        filename: yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        for filename in SHIPPED_CONFIG_NAMES
    }
    source_floors = {
        filename: raw["min_source_sample_rate_hz"] for filename, raw in shipped.items()
    }
    output_rates = {
        filename: raw["output_sample_rate_hz"] for filename, raw in shipped.items()
    }

    assert source_floors == expected_source_floors
    assert output_rates == expected_output_rates
    assert set(source_floors.values()) == {16_000, 32_000}
    assert set(output_rates.values()) == {16_000, 32_000}
    assert {
        filename for filename, rate in output_rates.items() if rate == 32_000
    } == {filename for filename in VARIANT_CONFIGS if filename.endswith("_32k.yaml")}
    assert {
        filename
        for filename, raw in shipped.items()
        if raw["speaker_consistency"]["enabled"]
    } == {"spk_filter.yaml"}

    zh_32k = shipped["audio_filter_zh_32k.yaml"]["datasets"]
    en_32k = shipped["audio_filter_en_32k.yaml"]["datasets"]
    assert zh_32k["smoothconv"]["language"] == "zh"
    assert "smoothconv" not in en_32k
    assert en_32k["jlcorpus_spk"]["language"] == "en"
    assert "jlcorpus_spk" not in zh_32k
    assert "audio_filter_mixed_16k.yaml" not in shipped


def test_generator_renders_audio_export_concurrency_per_config():
    generator = _load_generator()
    default_text = generator.render_config(datasets=[], commented_datasets=[], step="step4")
    tuned_text = generator.render_config(
        datasets=[], commented_datasets=[], step="step4",
        audio_export_workers=8, audio_export_max_pending=16,
    )

    assert "# 按 32 worker / max_pending 64 配置。" in default_text
    assert "audio_export_workers: 32" in default_text
    assert "audio_export_max_pending: 64" in default_text
    assert "# 按 8 worker / max_pending 16 配置。" in tuned_text
    assert "# 导出对 fd 需求高" in tuned_text
    assert "8 worker" in tuned_text
    assert "32 worker" not in tuned_text
    assert "并发峰值约为 256 个 fd" in tuned_text
    assert "ulimit -n 1048576" in tuned_text


@pytest.mark.parametrize("filename", SHIPPED_CONFIG_NAMES)
def test_shipped_configs_use_expected_audio_export_concurrency(filename):
    raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    expected = (8, 16) if filename == "audio_filter_zh_32k.yaml" else (32, 64)

    assert (raw["audio_export_workers"], raw["audio_export_max_pending"]) == expected


def test_shipped_config_is_complete_v3_schema():
    raw = _raw_config()
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    assert raw["step"] == "all"
    assert raw["step6_mode"] == "filter"
    assert raw["step4_mode"] == "filter"
    assert raw["output_sample_rate_hz"] == 16_000
    assert raw["step4_export_root"] is None
    assert raw["step6_export_root"] is None
    assert raw["audio_export_workers"] == 32
    assert raw["audio_export_max_pending"] == 64
    assert raw["audio_export_max_inflight_bytes"] == 4 * 1024 * 1024 * 1024
    assert raw["output_root"] == "/data_nvme0n1/zxc/code/use_train/data/tse/spk/filter"
    assert "/data_nvme" in config_text
    assert "/root" not in config_text
    assert raw["vad"] == {"runtime": "ten_vad_native_cpu", "hop_size": 256, "threshold": 0.5,
                          "workers": 64, "max_pending": 128}
    assert raw["use_gpu"] is True
    assert raw["step3_gpu_ids"] == list(range(8))
    assert raw["speaker_consistency"] == {
        "enabled": True,
        "runtime": "wespeaker",
        "model_dir": "/REPLACE_ME/audio_filter_pipeline/wespeaker_model",
        "model_version": "replace-me",
        "model_sha256": "0" * 64,
        "runtime_version": "replace-me",
        "target_sample_rate": 16_000,
        "enrollment_bundle_path": "/REPLACE_ME/audio_filter_pipeline/enrollment_bundle.jsonl",
        "enrollment_bundle_sha256": "0" * 64,
        "enrollment_exclusion_path": "/REPLACE_ME/audio_filter_pipeline/enrollment_exclusion.jsonl",
        "enrollment_exclusion_sha256": "0" * 64,
        "min_cosine_similarity": 0.8,
    }
    assert all(
        yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))[
            "speaker_consistency"
        ] == {"enabled": False}
        for filename in VARIANT_CONFIGS
    )
    assert raw["scoreq_model_dir"] == "third_party_scripts/SCOREQ"
    assert raw["dnsmos_model_dir"] == "third_party_scripts/DNSMOS"

    datasets = raw["datasets"]
    generator = _load_generator()
    step4_spk_ids = [
        name
        for members in generator.SR_DATASETS.values()
        for name in members
        if name.endswith("_spk")
    ]
    expected_spk_ids = {
        "genshinvoice_spk" if name == "genshinvoice_48k_spk" else name
        for name in step4_spk_ids
        if name not in generator.SPK_FILTER_EXCLUDED_DATASETS
    }
    assert generator.COMMENTED_DATASETS == COMMENTED_DATASETS
    assert generator.LANGUAGE_MAP["zh"].count("smoothconv") == 1
    assert "smoothconv" not in generator.LANGUAGE_MAP["en"]
    assert generator.LANGUAGE_MAP["en"].count("jlcorpus_spk") == 1
    assert "jlcorpus_spk" not in generator.LANGUAGE_MAP["zh"]
    assert len(step4_spk_ids) == 42
    assert len(expected_spk_ids) == len(datasets) == 36
    assert set(datasets) == expected_spk_ids == set(generator.SPK_FILTER_DATASETS)
    assert datasets == generator.SPK_FILTER_INVENTORY
    excluded_ids = {
        "genshinvoice_spk" if name == "genshinvoice_48k_spk" else name
        for name in generator.SPK_FILTER_EXCLUDED_DATASETS
    }
    assert excluded_ids.isdisjoint(datasets)
    assert not _commented_ids(CONFIG_PATH)
    assert set(raw) >= {"datasets", "selection_policies"}
    assert all(
        spec["speaker"] == {
            "source": "path_component", "path_component": 0, "required": False,
        }
        for spec in datasets.values()
    )
    assert all(Path(spec["input_root"]).is_dir() for spec in datasets.values())
    assert "genshinvoice_spk" not in datasets


@pytest.mark.parametrize("filename", SHIPPED_CONFIG_NAMES)
def test_shipped_default_route_is_strict_scoreq_dnsmos_two_of_two(filename):
    raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    policies = raw["selection_policies"]

    assert policies["global"] == {
        "mos_models": DEFAULT_ROUTE_MODELS,
        "min_mos_models": 2,
        "mos_fusion": "min",
        "early_exit": None,
    }
    if not raw["datasets"]:
        # 零 active 数据集的配置（zh_16k 整体注释）只保留 global route，
        # 运行时 schema 禁止无 datasets 时携带 profile 层。
        assert set(policies) == {"global"}
        return
    if raw["datasets"]:
        assert policies["categories"] == {}
        if filename not in UNLABELED_VARIANTS:
            assert policies["datasets"] == {}
        assert policies["languages"] == {
            language: {} for language in ("en", "zh", "mixed", "fr", "ja", "ko", "ru")
        }
        for profile in KNOWN_PROFILES:
            thresholds = policies["quality_profiles"][profile]["thresholds"]
            assert set(thresholds) == set(DEFAULT_ROUTE_MODELS)
            assert OPTIONAL_MOS_MODELS.isdisjoint(thresholds)


def test_spk_filter_uses_balanced_thresholds_for_every_quality_profile():
    raw = yaml.safe_load((CONFIG_DIR / "spk_filter.yaml").read_text(encoding="utf-8"))
    profiles = raw["selection_policies"]["quality_profiles"]
    balanced_thresholds = profiles["quality_balanced"]["thresholds"]

    assert balanced_thresholds == {
        "dnsmos": {"ovrl": 3.2, "sig": 3.0, "bak": 3.8, "p808": 3.6},
        "scoreq": 3.6,
    }
    assert all(
        profiles[profile]["thresholds"] == balanced_thresholds
        for profile in KNOWN_PROFILES
    )


@pytest.mark.parametrize("filename", SHIPPED_CONFIG_NAMES)
def test_shipped_optional_mos_assets_are_retained_but_not_selected(filename):
    raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))

    assert raw["scoreq_model_dir"] == "third_party_scripts/SCOREQ"
    assert raw["dnsmos_model_dir"] == "third_party_scripts/DNSMOS"
    assert raw["nisqa_model_path"] == "third_party_scripts/NISQA/weights/nisqa.tar"
    assert raw["utmos_model_path"] == (
        "third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt"
    )
    assert raw["urgent_pk_checkpoint"] == (
        "third_party_scripts/URGENT-PK/exp/mel_resnet34_urgent24.ckpt"
    )
    assert raw["urgent_pk_source_dir"] == "third_party_scripts/URGENT-PK"
    assert raw["urgent_pk_expected_sha256"] == URGENT_SHA256
    assert raw["urgent_pk_expected_commit"] == URGENT_COMMIT
    if "selection_policies" in raw:
        assert OPTIONAL_MOS_MODELS.isdisjoint(
            raw["selection_policies"]["global"]["mos_models"]
        )


def test_five_variant_configs_match_inventory_and_fixed_rates():
    generator = _load_generator()
    all_datasets = sum(generator.SR_DATASETS.values(), [])
    full_inventory = generator.dataset_inventory(all_datasets)
    selected_sets = []
    for filename, (languages, active_count, commented_count, minimum_rate,
                   maximum_rate, expected_output_rate,
                   expected_source_floor) in VARIANT_CONFIGS.items():
        raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        datasets = raw["datasets"]
        if filename == "audio_filter_zh_16k.yaml":
            # 2026-09-16 起四个数据集全部注释：筛选产出过低，不再处理；
            # 注释 inventory 仍保留身份与 catalog 注释。
            expected_ids = set()
            expected_commented_ids = set(
                generator.select_datasets_without_speaker_labels(
                    minimum_rate, languages, max_sample_rate_khz=maximum_rate,
                )
            )
            expected_inventory_ids = expected_commented_ids
        elif filename in UNLABELED_VARIANTS:
            expected_ids = set(generator.select_datasets_without_speaker_labels(
                minimum_rate, languages, max_sample_rate_khz=maximum_rate,
            ))
            expected_commented_ids = set()
            expected_inventory_ids = expected_ids
        else:
            expected_ids = set(generator.select_datasets(
                minimum_rate, languages, max_sample_rate_khz=maximum_rate,
            ))
            expected_commented_ids = set(generator.select_commented_datasets(
                minimum_rate, languages, max_sample_rate_khz=maximum_rate,
            ))
            expected_inventory_ids = set(generator.select_datasets(
                minimum_rate, languages, max_sample_rate_khz=maximum_rate,
                include_commented=True,
            ))
        selected_sets.append(expected_ids)
        assert len(datasets) == active_count
        assert len(expected_commented_ids) == commented_count
        assert set(datasets) == expected_ids
        assert _commented_ids(CONFIG_DIR / filename) == expected_commented_ids
        assert expected_ids | expected_commented_ids == expected_inventory_ids
        assert all(spec["language"] in languages for spec in datasets.values())
        assert all(
            dataset_id in full_inventory
            and spec["input_root"] == generator.DATASET_ROOTS[dataset_id]
            and {key: value for key, value in spec.items() if key != "input_root"}
            == {key: value for key, value in full_inventory[dataset_id].items() if key != "input_root"}
            for dataset_id, spec in datasets.items()
        )
        if filename in UNLABELED_VARIANTS:
            assert all(
                spec["speaker"]["source"] == "none"
                for spec in datasets.values()
            )
        assert raw["output_root"] == VARIANT_OUTPUT_ROOTS[filename]
        assert raw["output_sample_rate_hz"] == expected_output_rate
        assert raw["min_source_sample_rate_hz"] == expected_source_floor
        assert all(Path(spec["input_root"]).is_dir() for spec in datasets.values())
        assert raw["step"] == "step4"
        assert raw["step4_mode"] == "filter"
        assert raw["step6_mode"] == "filter"

    expected_step4_ids = (
        set(generator.select_datasets(32, {"zh"}))
        | set(generator.select_datasets(32, {"en"}))
        | set(generator.select_datasets(32, {"mixed"}))
        | set(generator.select_datasets_without_speaker_labels(0, {"en"}, max_sample_rate_khz=32))
    )
    expected_commented_ids = (
        set(generator.select_commented_datasets(32, {"zh"}))
        | set(generator.select_commented_datasets(32, {"en"}))
        | set(generator.select_commented_datasets(32, {"mixed"}))
        | set(generator.select_datasets_without_speaker_labels(0, {"zh"}, max_sample_rate_khz=32))
    )
    assert all(left.isdisjoint(right) for index, left in enumerate(selected_sets) for right in selected_sets[index + 1:])
    assert set().union(*selected_sets) == expected_step4_ids
    assert set().union(
        *(set(_commented_ids(CONFIG_DIR / filename)) for filename in VARIANT_CONFIGS)
    ) == expected_commented_ids


def test_16k_configs_use_requested_quality_standards():
    zh = yaml.safe_load((CONFIG_DIR / "audio_filter_zh_16k.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load((CONFIG_DIR / "audio_filter_en_16k.yaml").read_text(encoding="utf-8"))

    # zh_16k 四个数据集已整体注释（筛选产出过低），不再有任何 active 数据集；
    # selection_policies 只保留 global route（运行时 schema 禁止无 datasets 时携带 profile 层）；
    # 身份与 catalog 注释保留在注释 inventory 中。
    assert zh["datasets"] == {}
    assert set(zh["selection_policies"]) == {"global"}
    assert _commented_ids(CONFIG_DIR / "audio_filter_zh_16k.yaml") == set(
        _load_generator().select_datasets_without_speaker_labels(0, {"zh"}, max_sample_rate_khz=32)
    )

    # 五份 audio_filter 配置共享同一 quality_profiles 阶梯定义；数据集档位只经
    # catalog quality_profile / dataset_quality_profile_overrides 选择，
    # en_16k 不再用 uniform_quality_profile 把三档抹平成 premium。
    raws = {
        filename: yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        for filename in VARIANT_CONFIGS
    }
    profiles_by_config = {
        filename: raw["selection_policies"]["quality_profiles"]
        for filename, raw in raws.items()
        if "quality_profiles" in raw["selection_policies"]
    }
    assert all(
        profiles == profiles_by_config["audio_filter_en_32k.yaml"]
        for profiles in profiles_by_config.values()
    )
    assert en["selection_policies"]["datasets"] == {}


def test_spk_filter_excludes_datasets_already_present_in_clean_filter():
    active_in = set()
    commented_in = set()
    for filename in VARIANT_CONFIGS:
        raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
        if "bibletts_spk" in raw["datasets"]:
            active_in.add(filename)
        if "bibletts_spk" in _commented_ids(CONFIG_DIR / filename):
            commented_in.add(filename)
    assert active_in == set()
    assert commented_in == {"audio_filter_mixed_32k.yaml"}
    assert "bibletts_spk" not in _raw_config()["datasets"]
    assert {
        "aishell-3_spk", "bibletts_spk", "dailytalk_spk", "gtsinger_spk", "hifitts2_spk",
    }.isdisjoint(_raw_config()["datasets"])
    assert "genshinvoice_spk" not in _raw_config()["datasets"]


def test_removed_mixed_16k_config_cannot_be_generated(tmp_path):
    output_dir = tmp_path / "config"
    result = subprocess.run(
        [sys.executable, str(CONFIG_DIR / "generate_configs.py"), "--output-dir", str(output_dir)],
        cwd=CONFIG_DIR.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert not (CONFIG_DIR / "audio_filter_mixed_16k.yaml").exists()
    assert not (output_dir / "audio_filter_mixed_16k.yaml").exists()
    assert {path.name for path in output_dir.glob("*.yaml")} == set(SHIPPED_CONFIG_NAMES)


def test_shipped_config_output_root_and_speaker_preflight_context_boundaries():
    raw = load_config(str(CONFIG_PATH))
    assert raw["urgent_pk_checkpoint"].endswith("third_party_scripts/URGENT-PK/exp/mel_resnet34_urgent24.ckpt")
    parameter_names = set(inspect.signature(PipelineConfig.__init__).parameters) - {"self"}
    config = PipelineConfig(**{key: value for key, value in raw.items() if key in parameter_names})
    config.validate(context=ValidationContext.STRUCTURAL, stage_names=[])
    for context in (
        ValidationContext.V3_STEP0_EXECUTION,
        ValidationContext.V3_STEP1_EXECUTION,
        ValidationContext.V3_STEP2_EXECUTION,
        ValidationContext.V3_STEP3_EXECUTION,
        ValidationContext.V3_STEP4_EXECUTION,
    ):
        config.validate(context=context, stage_names=[])
    for context in (
        ValidationContext.V3_STEP5_EXECUTION,
        ValidationContext.V3_STEP6_EXECUTION,
    ):
        with pytest.raises(ValueError, match="placeholder"):
            config.validate(context=context, stage_names=[])
    assert config.output_root == "/data_nvme0n1/zxc/code/use_train/data/tse/spk/filter"


def test_execution_validation_rejects_null_output_root(tmp_path):
    raw = _raw_config()
    datasets = {
        dataset_id: {**spec, "input_root": str(tmp_path / dataset_id)}
        for dataset_id, spec in raw["datasets"].items()
    }
    config = PipelineConfig(
        pipeline_generation=3,
        step="step0",
        output_root=None,
        datasets=datasets,
        selection_policies=raw["selection_policies"],
    )
    with pytest.raises(ValueError, match="output_root must be a non-empty string"):
        config.validate(context=ValidationContext.V3_STEP0_EXECUTION)


def test_generator_coverage_and_inventory_identities():
    generator = _load_generator()
    generator.check_dataset_coverage(generator.SR_DATASETS, generator.SEMANTIC_CATEGORY_MAP,
                                     generator.QUALITY_PROFILE_MAP, generator.LANGUAGE_MAP)
    datasets = sum(generator.SR_DATASETS.values(), [])
    assert len(datasets) == len(set(datasets)) == 50
    assert {rate: len(items) for rate, items in generator.SR_DATASETS.items()} == {
        "16k": 29, "24k": 1, "32k": 3, "44k1": 8, "48k": 9,
    }
    inventory = generator.dataset_inventory(datasets)
    assert set(inventory) == set(datasets) == set(generator.DATASET_ROOTS)
    expected_languages = {
        "gigaspeech": "en",
        "emov-db_en_16k_spk": "en",
        "ears_spk": "en",
        "jlcorpus_spk": "en",
        "smoothconv": "zh",
        "gtsinger_spk": "mixed",
        "genshinvoice_48k_spk": "mixed",
    }
    assert {name: inventory[name]["language"] for name in expected_languages} == expected_languages
    assert {name for name in datasets if name.startswith("emov-db")} == {
        "emov-db_en_16k_spk",
    }
    assert {name for name in datasets if name.startswith("genshinvoice")} == {
        "genshinvoice_48k_spk",
    }
    expected_spk_ids = {
        "genshinvoice_spk" if name == "genshinvoice_48k_spk" else name
        for name in datasets
        if name.endswith("_spk")
        and name not in generator.SPK_FILTER_EXCLUDED_DATASETS
    }
    assert set(_raw_config()["datasets"]) == expected_spk_ids
    assert len(expected_spk_ids) == 36
    assert "genshinvoice_spk" not in set(datasets)
    assert "genshinvoice_44k1_spk" not in set(datasets)
    assert "genshinvoice_48k_spk" not in expected_spk_ids
    assert "bibletts_spk" not in expected_spk_ids
    assert {profile: len([item for item in items if item.endswith("_spk")])
            for profile, items in generator.QUALITY_PROFILE_MAP.items()} == {
        "quality_balanced": 20, "quality_high": 19, "quality_premium": 3,
    }
    assert {category: len([item for item in items if item.endswith("_spk")])
            for category, items in generator.SEMANTIC_CATEGORY_MAP.items()} == {
        "read_speech": 22, "expressive": 16, "elderly": 1, "singing": 2, "game_voice": 1,
    }
    for name, spec in inventory.items():
        assert spec["input_root"] is None, name
        expected_speaker = (
            {"source": "path_component", "path_component": 0, "required": False}
            if name.endswith("_spk") else {"source": "none", "required": False}
        )
        assert spec["speaker"] == expected_speaker
        assert spec.get("source_selector") == generator.SOURCE_SELECTORS.get(name)


def test_dataset_catalog_covers_inventory_and_matches_runtime_mappings():
    generator = _load_generator()
    catalog = _load_catalog()
    sr_ids = set(sum(generator.SR_DATASETS.values(), []))
    expected_ids = sr_ids | {"genshinvoice_spk"}

    assert set(catalog.DATASET_CATALOG) == expected_ids
    catalog.validate_catalog(
        expected_ids,
        generator.LANGUAGE_MAP,
        generator.SEMANTIC_CATEGORY_MAP,
        generator.QUALITY_PROFILE_MAP,
    )
    for dataset_id, metadata in catalog.DATASET_CATALOG.items():
        assert metadata["display_name"]
        assert metadata["summary"]
        assert metadata["language_bucket"] in generator.LANGUAGE_MAP
        assert metadata["category"] in generator.SEMANTIC_CATEGORY_MAP
        assert metadata["quality_profile"] in generator.QUALITY_PROFILE_MAP
        assert metadata["traits"]
        assert metadata["sample_rate_or_evidence"]
        assert metadata["evidence_status"]

    assert "老人声" in catalog.DATASET_CATALOG["seniortalk_spk"]["traits"]
    assert catalog.DATASET_CATALOG["vocalset_spk"]["category"] == "singing"
    assert catalog.DATASET_CATALOG["gtsinger_spk"]["category"] == "singing"
    assert catalog.DATASET_CATALOG["genshinvoice_48k_spk"]["category"] == "game_voice"
    assert "genshinvoice_44k1_spk" not in catalog.DATASET_CATALOG
    assert catalog.DATASET_CATALOG["smoothconv"]["language_bucket"] == "zh"
    assert "多通道" in catalog.DATASET_CATALOG["smoothconv"]["traits"]
    assert "对话" in catalog.DATASET_CATALOG["smoothconv"]["traits"]
    assert "年龄属性待 manifest 核验" in catalog.DATASET_CATALOG["nnces_read_spk"]["traits"]
    assert "年龄属性待 manifest 核验" in catalog.DATASET_CATALOG["nnces_spontaneous_spk"]["traits"]
    assert all(
        "儿童语音" not in catalog.DATASET_CATALOG[dataset_id]["traits"]
        for dataset_id in ("nnces_read_spk", "nnces_spontaneous_spk")
    )


def test_generated_yaml_has_catalog_annotation_for_every_dataset_entry():
    generator = _load_generator()
    entry_pattern = re.compile(r"^(?:# )?  ([A-Za-z0-9_.-]+): \{input_root:")
    all_expected_ids = set(sum(generator.SR_DATASETS.values(), []))

    all_found_ids = set()
    for filename in SHIPPED_CONFIG_NAMES:
        text = (CONFIG_DIR / filename).read_text(encoding="utf-8")
        lines = text.splitlines()
        found_ids = set()
        for index, line in enumerate(lines):
            match = entry_pattern.match(line)
            if not match:
                continue
            dataset_id = match.group(1)
            found_ids.add(dataset_id)
            metadata = generator.DATASET_CATALOG[dataset_id]
            annotation = "\n".join(
                (
                    f"# dataset_catalog: {metadata['display_name']}（{dataset_id}）",
                    f"#   基本信息：{metadata['summary']}",
                    f"#   language bucket：{metadata['language_bucket']}；category：{metadata['category']}；",
                    f"#   quality_profile：{metadata['quality_profile']}",
                    f"#   特殊属性/备注：{metadata['traits']}",
                    f"#   采样率/证据：{metadata['sample_rate_or_evidence']}",
                    f"#   证据状态：{metadata['evidence_status']}",
                )
            )
            assert index >= 7
            assert "\n".join(lines[index - 7:index]) == annotation
        assert found_ids <= all_expected_ids
        all_found_ids.update(found_ids)

    assert all_found_ids == all_expected_ids


def test_split_roots_have_exact_logical_ids_aliases_and_disjoint_selectors():
    generator = _load_generator()
    speakers_zh = [f"{speaker_id:04d}" for speaker_id in range(1, 11)]
    speakers_en = [f"{speaker_id:04d}" for speaker_id in range(11, 21)]
    expected_selectors = {
        "esd_zh_16k_spk": {"include_speaker_ids": speakers_zh},
        "esd_en_16k_spk": {"include_speaker_ids": speakers_en},
        "genshinvoice_48k_spk": {"sample_rates_hz": [48_000]},
        "aishell5": {
            "exclude_relative_paths": [
                f"noise/{speaker}/noise_{channel}C01.flac"
                for speaker in ("040", "044")
                for channel in ("DX01", "DX02", "DX03", "DX04")
            ],
        },
    }
    assert generator.SOURCE_SELECTORS == expected_selectors

    all_ids = set(sum(generator.SR_DATASETS.values(), []))
    assert {name for name in all_ids if name.startswith("esd_")} == {
        "esd_zh_16k_spk", "esd_en_16k_spk",
    }
    assert generator.LANGUAGE_MAP["zh"].count("esd_zh_16k_spk") == 1
    assert generator.LANGUAGE_MAP["en"].count("esd_en_16k_spk") == 1

    physical_aliases = {
        "emov-db_en_16k_spk": "emov-db_spk",
        "esd_zh_16k_spk": "esd_spk",
        "esd_en_16k_spk": "esd_spk",
        "genshinvoice_48k_spk": "genshinvoice_spk",
    }
    assert {
        logical_id: Path(generator.DATASET_ROOTS[logical_id]).name
        for logical_id in physical_aliases
    } == physical_aliases

    ids_by_root = {}
    for logical_id, root in generator.DATASET_ROOTS.items():
        ids_by_root.setdefault(root, []).append(logical_id)
    for logical_ids in ids_by_root.values():
        if len(logical_ids) < 2:
            continue
        for index, left_id in enumerate(logical_ids):
            left = expected_selectors[left_id]
            for right_id in logical_ids[index + 1:]:
                right = expected_selectors[right_id]
                rates_disjoint = (
                    "sample_rates_hz" in left and "sample_rates_hz" in right
                    and set(left["sample_rates_hz"]).isdisjoint(right["sample_rates_hz"])
                )
                speakers_disjoint = (
                    "include_speaker_ids" in left and "include_speaker_ids" in right
                    and set(left["include_speaker_ids"]).isdisjoint(right["include_speaker_ids"])
                )
                assert rates_disjoint or speakers_disjoint, (left_id, right_id)


def test_step4_active_partition_validation_accepts_five_shipped_partitions():
    generator = _load_generator()
    outputs = {
        "audio_filter_zh_32k.yaml": {
            "step": "step4", "datasets": generator.select_datasets(32, {"zh"}),
        },
        "audio_filter_en_32k.yaml": {
            "step": "step4", "datasets": generator.select_datasets(32, {"en"}),
        },
        "audio_filter_mixed_32k.yaml": {
            "step": "step4", "datasets": generator.select_datasets(32, {"mixed"}),
        },
        "audio_filter_zh_16k.yaml": {
            "step": "step4",
            "datasets": generator.select_datasets_without_speaker_labels(
                0, {"zh"}, max_sample_rate_khz=32,
            ),
        },
        "audio_filter_en_16k.yaml": {
            "step": "step4",
            "datasets": generator.select_datasets_without_speaker_labels(
                0, {"en"}, max_sample_rate_khz=32,
            ),
        },
        "spk_filter.yaml": {"step": "all", "datasets": generator.SPK_FILTER_DATASETS},
    }
    generator.validate_step4_active_partitions(outputs)

    # 同一份 shipped 数据挂到第二个 config 必须被拒绝，
    # 否则上面的"通过"可能只是校验根本没生效。
    duplicated = dict(outputs)
    duplicated["duplicate.yaml"] = dict(outputs["audio_filter_zh_32k.yaml"])
    with pytest.raises(ValueError, match="active in both"):
        generator.validate_step4_active_partitions(duplicated)


def test_step4_active_partition_validation_rejects_duplicate_logical_id():
    generator = _load_generator()
    outputs = {
        "left": {"step": "step4", "datasets": ["casia_spk"]},
        "right": {"step": "step4", "datasets": ["casia_spk"]},
    }
    with pytest.raises(ValueError, match="active in both"):
        generator.validate_step4_active_partitions(outputs)


def test_step4_active_partition_validation_rejects_same_config_shared_root(monkeypatch):
    generator = _load_generator()
    monkeypatch.setitem(
        generator.DATASET_ROOTS,
        "cremad_spk",
        generator.DATASET_ROOTS["casia_spk"],
    )
    outputs = {
        "combined": {
            "step": "step4",
            "datasets": ["casia_spk", "cremad_spk"],
        },
    }
    with pytest.raises(ValueError, match="share canonical root"):
        generator.validate_step4_active_partitions(outputs)


def test_step4_active_partition_validation_requires_identical_speaker_resolver(monkeypatch):
    generator = _load_generator()
    monkeypatch.setitem(
        generator.DATASET_ROOTS,
        "esd_en_16k_spk",
        generator.DATASET_ROOTS["esd_zh_16k_spk"],
    )
    original_speaker_spec = generator._speaker_spec
    monkeypatch.setattr(
        generator,
        "_speaker_spec",
        lambda name: (
            {"source": "path_component", "path_component": 1, "required": False}
            if name == "esd_en_16k_spk" else original_speaker_spec(name)
        ),
    )
    outputs = {
        "left": {"step": "step4", "datasets": ["esd_zh_16k_spk"]},
        "right": {"step": "step4", "datasets": ["esd_en_16k_spk"]},
    }
    with pytest.raises(ValueError, match="share canonical root"):
        generator.validate_step4_active_partitions(outputs)


def test_generator_emits_all_configs_and_is_idempotent(tmp_path):
    output_dir = tmp_path / "config"
    command = [sys.executable, str(CONFIG_DIR / "generate_configs.py"), "--output-dir", str(output_dir)]
    first = subprocess.run(command, cwd=CONFIG_DIR.parent, capture_output=True, text=True, timeout=30)
    assert first.returncode == 0, first.stderr
    assert sorted(path.name for path in output_dir.glob("*.yaml")) == sorted(SHIPPED_CONFIG_NAMES)
    assert not set(path.name for path in output_dir.glob("*.yaml")) & LEGACY_CONFIG_NAMES
    first_bytes = {
        name: (output_dir / name).read_bytes()
        for name in SHIPPED_CONFIG_NAMES
    }
    second = subprocess.run(command, cwd=CONFIG_DIR.parent, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stderr
    assert {
        name: (output_dir / name).read_bytes()
        for name in SHIPPED_CONFIG_NAMES
    } == first_bytes
    assert hashlib.sha256(first_bytes["spk_filter.yaml"]).digest() == hashlib.sha256(CONFIG_PATH.read_bytes()).digest()
    for name in VARIANT_CONFIGS:
        assert hashlib.sha256(first_bytes[name]).digest() == hashlib.sha256((CONFIG_DIR / name).read_bytes()).digest()


def test_enrollment_bundle_step3_loader_imports_cli_main(tmp_path):
    """验证 enrollment_bundle 的懒加载使用 canonical cli.main，而非已移除的 main。"""
    from audio_filter_pipeline.processors.stages_v3.enrollment_bundle import (
        _current_step3_from_config,
    )

    with pytest.raises(ValueError, match="canonical absolute existing file"):
        _current_step3_from_config(str(tmp_path / "missing.yaml"), "dataset")
