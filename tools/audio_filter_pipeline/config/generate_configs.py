#!/usr/bin/env python3
"""生成 generation-3 的五份 Step4 配置和一份 Step6 speaker 配置。

数据集身份由本文件的 inventory 显式维护；生成器不依赖 v2 模板，连续执行必须
产生字节一致的配置。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent


def _load_dataset_catalog():
    """从同目录文件加载 catalog，避免把展示 catalog 混入运行时 schema。"""
    spec = importlib.util.spec_from_file_location(
        "audio_filter_pipeline_dataset_catalog", CONFIG_DIR / "dataset_catalog.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load config/dataset_catalog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATASET_CATALOG_MODULE = _load_dataset_catalog()
DATASET_CATALOG = DATASET_CATALOG_MODULE.DATASET_CATALOG

# 六份筛选配置均绑定已核验的本地 input_root；spk_filter 使用固定的 speaker filter 输出目录。
# 当前仅纳入 16 kHz 及以上源数据，明确排除 8 kHz 数据。

# 采样率分组用于五个语言/采样率 Step4 配置的 inventory coverage。
# 50 个 Step4 逻辑数据集可通过 source_selector 复用 49 个已核验物理目录。
SR_DATASETS = {
    "16k": ["casia_spk", "cremad_spk", "emov-db_en_16k_spk", "esd_zh_16k_spk",
            "esd_en_16k_spk", "gigaspeech", "hq-conversations_spk", "iemocap_spk",
            "librispeech_spk", "m3ed_spk", "magicdata_spk", "mead_spk", "meld_spk",
            "mer2023", "msp-podcast_spk", "seniortalk_spk", "tess_spk", "voicebank_spk",
            "aishell4", "aishell5", "alimeeting", "ami", "cnceleb_spk",
            "cs_dialogue_spk", "dipco", "easyturn_spk", "kespeech_spk",
            "voxceleb_vox1_spk", "voxceleb_vox2_spk"],
    "24k": ["libritts_r_spk"],
    "32k": ["commonvoice_cn_spk", "commonvoice_en_spk", "hifitts2_spk"],
    "44k1": ["aishell-3_spk", "dailytalk_spk", "hifi_tts_spk", "jlcorpus_spk",
              "nnces_read_spk", "nnces_spontaneous_spk", "savee_spk", "vocalset_spk"],
    "48k": ["bibletts_spk", "ears_spk", "emns_spk", "expresso_spk", "gtsinger_spk",
            "ravdess_spk", "vctk_spk", "genshinvoice_48k_spk", "smoothconv"],
}

# 分组键到 kHz 的显式映射：键名只作展示，采样率由 SR_DATASETS_KHZ 显式给出
SR_DATASETS_KHZ = {
    "16k": 16.0,
    "24k": 24.0,
    "32k": 32.0,
    "44k1": 44.1,
    "48k": 48.0,
}


def _dataset_sample_rate_khz(group: str) -> float:
    """返回 SR_DATASETS 分组的 kHz；缺登记即 fail-fast，不做字符串解析。"""
    try:
        return SR_DATASETS_KHZ[group]
    except KeyError as exc:
        raise ValueError(
            f"SR_DATASETS group is missing from SR_DATASETS_KHZ: {group!r}"
        ) from exc


# 已整理完成、保留为 YAML 注释但不进入 active datasets 映射的数据集。
COMMENTED_DATASETS = frozenset({
    "aishell-3_spk", "savee_spk", "nnces_spontaneous_spk", "nnces_read_spk",
    "vocalset_spk", "hifitts2_spk", "libritts_r_spk", "commonvoice_en_spk",
    "bibletts_spk", "dailytalk_spk", "expresso_spk", "hifi_tts_spk", "ears_spk",
    "commonvoice_cn_spk", "vctk_spk", "gtsinger_spk",
})

# Step6 显式排除已经在 clean/filter 中保留筛选结果的 speaker 数据集，避免重复处理。
SPK_FILTER_EXCLUDED_DATASETS = frozenset({
    "aishell-3_spk", "bibletts_spk", "dailytalk_spk", "genshinvoice_48k_spk",
    "gtsinger_spk", "hifitts2_spk",
})

# Step6 纳入 Step4 inventory 中未被显式排除的带 speaker 身份逻辑项；
# GenshinVoice 仅保留 48 kHz Step4 视图，并在 Step6 使用单一 ID。
SPK_FILTER_DATASETS = list(dict.fromkeys(
    "genshinvoice_spk" if name == "genshinvoice_48k_spk" else name
    for members in SR_DATASETS.values()
    for name in members
    if name.endswith("_spk") and name not in SPK_FILTER_EXCLUDED_DATASETS
))

# spk_filter 使用已经整理好的 16 kHz 物理目录；不要复用 Step4 的源数据根。
SPK_FILTER_DATASET_ROOT_BASE = "/data_nvme0n1/zxc/code/use_train/data/tse/spk/ori"
_SPK_FILTER_PHYSICAL_NAMES = {
    "casia_spk": "casia_spk",
    "cremad_spk": "cremad_spk",
    "emov-db_en_16k_spk": "emov-db_spk",
    "esd_zh_16k_spk": "esd_spk",
    "esd_en_16k_spk": "esd_spk",
    "hq-conversations_spk": "hq-conversations_spk",
    "iemocap_spk": "iemocap_spk",
    "librispeech_spk": "librispeech_spk",
    "m3ed_spk": "m3ed_spk",
    "magicdata_spk": "magicdata_spk",
    "mead_spk": "mead_spk",
    "meld_spk": "meld_spk",
    "msp-podcast_spk": "msp-podcast_spk",
    "seniortalk_spk": "seniortalk_spk",
    "tess_spk": "tess_spk",
    "voicebank_spk": "voicebank_spk",
    "cnceleb_spk": "cnceleb_spk",
    "cs_dialogue_spk": "cs_dialogue_spk",
    "easyturn_spk": "easyturn_spk",
    "kespeech_spk": "kespeech_spk",
    "voxceleb_vox1_spk": "voxceleb_vox1_spk",
    "voxceleb_vox2_spk": "voxceleb_vox2_spk",
    "libritts_r_spk": "libritts_r_spk",
    "commonvoice_cn_spk": "commonvoice_cn_spk",
    "commonvoice_en_spk": "commonvoice_en_spk",
    "hifi_tts_spk": "hifi_tts_spk",
    "jlcorpus_spk": "jlcorpus_spk",
    "nnces_read_spk": "nnces_read_spk",
    "nnces_spontaneous_spk": "nnces_spontaneous_spk",
    "savee_spk": "savee_spk",
    "vocalset_spk": "vocalset_spk",
    "ears_spk": "ears_spk",
    "emns_spk": "emns_spk",
    "expresso_spk": "expresso_spk",
    "ravdess_spk": "ravdess_spk",
    "vctk_spk": "vctk_spk",
}
SPK_FILTER_DATASET_ROOTS = {
    dataset_id: f"{SPK_FILTER_DATASET_ROOT_BASE}/{physical_name}"
    for dataset_id, physical_name in _SPK_FILTER_PHYSICAL_NAMES.items()
}

# 已核验的数据集实际路径；逻辑 ID 显式映射到对应的物理目录。
DATASET_ROOTS = {
    **{
        name: "/data_nvme1n1/clean_speech_download/datasets/" + name
        for name in [
            "aishell4", "aishell5", "alimeeting", "ami", "casia_spk", "cnceleb_spk",
            "cremad_spk", "cs_dialogue_spk", "dipco", "easyturn_spk", "gigaspeech",
            "hq-conversations_spk", "iemocap_spk", "kespeech_spk", "librispeech_spk",
            "m3ed_spk", "magicdata_spk", "mead_spk", "meld_spk", "mer2023",
            "msp-podcast_spk", "seniortalk_spk", "tess_spk", "voicebank_spk",
            "voxceleb_vox1_spk",
        ]
    },
    **{
        name: "/data_nvme2n1/clean_speech_download/datasets/" + name
        for name in [
            "aishell-3_spk", "bibletts_spk", "commonvoice_cn_spk", "commonvoice_en_spk",
            "dailytalk_spk", "ears_spk", "emns_spk", "expresso_spk", "gtsinger_spk",
            "hifitts2_spk", "hifi_tts_spk", "jlcorpus_spk", "libritts_r_spk",
            "nnces_read_spk", "nnces_spontaneous_spk", "ravdess_spk", "savee_spk",
            "smoothconv", "vctk_spk", "vocalset_spk", "voxceleb_vox2_spk",
        ]
    },
    "emov-db_en_16k_spk": "/data_nvme1n1/clean_speech_download/datasets/emov-db_spk",
    "esd_zh_16k_spk": "/data_nvme1n1/clean_speech_download/datasets/esd_spk",
    "esd_en_16k_spk": "/data_nvme1n1/clean_speech_download/datasets/esd_spk",
    "genshinvoice_48k_spk": "/data_nvme2n1/clean_speech_download/datasets/genshinvoice_spk",
}
SOURCE_SELECTORS = {
    "esd_zh_16k_spk": {"include_speaker_ids": [f"{speaker_id:04d}" for speaker_id in range(1, 11)]},
    "esd_en_16k_spk": {"include_speaker_ids": [f"{speaker_id:04d}" for speaker_id in range(11, 21)]},
    "genshinvoice_48k_spk": {"sample_rates_hz": [48_000]},
    "aishell5": {
        "exclude_relative_paths": [
            f"noise/{speaker}/noise_{channel}C01.flac"
            for speaker in ("040", "044")
            for channel in ("DX01", "DX02", "DX03", "DX04")
        ],
    },
}
LANGUAGE_MAP = {
    "zh": ["casia_spk", "esd_zh_16k_spk", "magicdata_spk", "mer2023", "m3ed_spk",
           "seniortalk_spk", "aishell-3_spk", "commonvoice_cn_spk",
           "aishell4", "aishell5", "alimeeting", "cnceleb_spk",
           "cs_dialogue_spk", "kespeech_spk", "smoothconv"],
    "en": ["cremad_spk", "emov-db_en_16k_spk", "esd_en_16k_spk",
           "gigaspeech", "librispeech_spk", "hq-conversations_spk", "iemocap_spk",
           "mead_spk", "meld_spk", "msp-podcast_spk", "tess_spk", "voicebank_spk",
           "libritts_r_spk", "dailytalk_spk", "hifi_tts_spk", "hifitts2_spk",
           "nnces_read_spk", "nnces_spontaneous_spk", "savee_spk", "vocalset_spk",
           "ears_spk", "emns_spk", "expresso_spk", "ravdess_spk", "vctk_spk",
           "commonvoice_en_spk", "ami", "dipco", "easyturn_spk",
           "voxceleb_vox1_spk", "voxceleb_vox2_spk", "jlcorpus_spk"],
    "mixed": ["bibletts_spk", "genshinvoice_48k_spk", "gtsinger_spk"],
}
SEMANTIC_CATEGORY_MAP = {
    "read_speech": ["hq-conversations_spk", "librispeech_spk", "magicdata_spk", "voicebank_spk",
        "aishell4", "aishell5", "alimeeting", "ami", "cnceleb_spk", "cs_dialogue_spk", "dipco",
        "easyturn_spk", "kespeech_spk", "voxceleb_vox1_spk", "libritts_r_spk", "commonvoice_cn_spk",
        "aishell-3_spk", "dailytalk_spk", "hifi_tts_spk", "nnces_read_spk", "nnces_spontaneous_spk",
        "bibletts_spk", "ears_spk", "vctk_spk", "smoothconv", "gigaspeech",
        "voxceleb_vox2_spk", "commonvoice_en_spk", "hifitts2_spk"],
    "expressive": ["casia_spk", "cremad_spk", "emov-db_en_16k_spk",
        "esd_zh_16k_spk", "esd_en_16k_spk", "iemocap_spk", "m3ed_spk", "mead_spk",
        "meld_spk", "mer2023", "msp-podcast_spk", "tess_spk", "jlcorpus_spk", "savee_spk",
        "emns_spk", "expresso_spk", "ravdess_spk"],
    "elderly": ["seniortalk_spk"], "singing": ["vocalset_spk", "gtsinger_spk"],
    "game_voice": ["genshinvoice_48k_spk"],
}
QUALITY_PROFILE_MAP = {
    "quality_high": ["hq-conversations_spk", "librispeech_spk", "magicdata_spk", "voicebank_spk",
        "aishell4", "aishell5", "alimeeting", "ami", "cnceleb_spk", "cs_dialogue_spk", "dipco",
        "easyturn_spk", "kespeech_spk", "voxceleb_vox1_spk", "libritts_r_spk", "commonvoice_cn_spk",
        "dailytalk_spk", "hifi_tts_spk", "nnces_read_spk", "nnces_spontaneous_spk", "bibletts_spk",
        "ears_spk", "vctk_spk", "smoothconv", "genshinvoice_48k_spk"],
    "quality_premium": ["gigaspeech", "voxceleb_vox2_spk", "commonvoice_en_spk",
        "hifitts2_spk"],
    "quality_balanced": ["casia_spk", "cremad_spk", "emov-db_en_16k_spk",
        "esd_zh_16k_spk", "esd_en_16k_spk", "iemocap_spk", "m3ed_spk", "mead_spk", "meld_spk",
        "mer2023", "msp-podcast_spk", "tess_spk", "jlcorpus_spk", "savee_spk", "emns_spk",
        "expresso_spk", "ravdess_spk", "seniortalk_spk", "vocalset_spk", "gtsinger_spk",
        "aishell-3_spk"],
}


def check_dataset_coverage(sr_datasets, semantic_category_map, quality_profile_map, language_map) -> None:
    """确保每项恰好拥有采样率、语言、类别和质量档位身份。"""
    ordered = [name for members in sr_datasets.values() for name in members]
    if len(ordered) != len(set(ordered)):
        raise ValueError("SR_DATASETS contains duplicate dataset IDs")
    expected = set(ordered)
    for label, mapping, required_keys in (
        ("SEMANTIC_CATEGORY_MAP", semantic_category_map,
         {"read_speech", "expressive", "elderly", "singing", "game_voice"}),
        ("QUALITY_PROFILE_MAP", quality_profile_map,
         {"quality_balanced", "quality_high", "quality_premium"}),
        ("LANGUAGE_MAP", language_map, set(language_map)),
    ):
        if set(mapping) != required_keys:
            raise ValueError(f"{label} keys mismatch: {set(mapping) ^ required_keys}")
        entries = [name for members in mapping.values() for name in members]
        if len(entries) != len(set(entries)):
            raise ValueError(f"{label} contains duplicate dataset IDs")
        if set(entries) != expected:
            raise ValueError(f"{label} coverage mismatch: {set(entries) ^ expected}")
    unknown_commented = COMMENTED_DATASETS - expected
    if unknown_commented:
        raise ValueError(
            f"COMMENTED_DATASETS contains unknown dataset IDs: {sorted(unknown_commented)!r}"
        )
    if set(DATASET_ROOTS) != expected:
        raise ValueError(f"DATASET_ROOTS coverage mismatch: {set(DATASET_ROOTS) ^ expected}")
    unknown_selectors = set(SOURCE_SELECTORS) - expected
    if unknown_selectors:
        raise ValueError(f"SOURCE_SELECTORS contains unknown dataset IDs: {unknown_selectors}")
    DATASET_CATALOG_MODULE.validate_catalog(
        expected | {"genshinvoice_spk"},
        language_map,
        semantic_category_map,
        quality_profile_map,
    )


def check_spk_filter_root_coverage() -> None:
    """确保 spk_filter 映射完整覆盖现有的 35 个物理目录。"""
    expected_ids = set(SPK_FILTER_DATASETS)
    mapped_ids = set(SPK_FILTER_DATASET_ROOTS)
    if mapped_ids != expected_ids:
        raise ValueError(
            "SPK_FILTER_DATASET_ROOTS coverage mismatch: "
            f"{mapped_ids ^ expected_ids}"
        )

    root_base = Path(SPK_FILTER_DATASET_ROOT_BASE).resolve()
    mapped_roots = {Path(root).resolve() for root in SPK_FILTER_DATASET_ROOTS.values()}
    if len(mapped_roots) != 35:
        raise ValueError(
            "SPK_FILTER_DATASET_ROOTS must cover exactly 35 physical roots; "
            f"got {len(mapped_roots)}"
        )
    invalid_roots = []
    for root in mapped_roots:
        try:
            relative = root.relative_to(root_base)
        except ValueError:
            invalid_roots.append(str(root))
            continue
        if len(relative.parts) != 1 or not root.is_dir():
            invalid_roots.append(str(root))
    if invalid_roots:
        raise ValueError(
            "SPK_FILTER_DATASET_ROOTS contains missing or out-of-tree roots: "
            f"{sorted(invalid_roots)!r}"
        )
    if not root_base.is_dir():
        raise ValueError(f"SPK_FILTER_DATASET_ROOT_BASE is not a directory: {root_base}")
    actual_names = {path.name for path in root_base.iterdir() if path.is_dir()}
    mapped_names = {path.name for path in mapped_roots}
    if actual_names != mapped_names:
        raise ValueError(
            "spk_filter physical root coverage mismatch: "
            f"missing={sorted(mapped_names - actual_names)!r}, "
            f"extra={sorted(actual_names - mapped_names)!r}"
        )


def _speaker_spec(name: str) -> dict[str, object]:
    """_spk 数据集已按说话人重组织，第一层目录即 speaker_id；其余不标注 speaker。"""
    if name.endswith("_spk"):
        return {"source": "path_component", "path_component": 0, "required": False}
    return {"source": "none", "required": False}


def _yaml_speaker_flow(speaker: dict[str, object]) -> str:
    """把 speaker spec 渲染成内联 YAML flow（与 shipped 现有格式一致）。"""
    if speaker.get("source") == "path_component":
        return f'{{source: path_component, path_component: {speaker["path_component"]}, required: {str(speaker["required"]).lower()}}}'
    return '{source: none, required: false}'


def dataset_inventory(
    datasets: list[str], *, input_roots: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """生成 inventory，可选绑定已核验的本地输入根与 source selector。"""
    language_of = {name: value for value, names in LANGUAGE_MAP.items() for name in names}
    category_of = {name: value for value, names in SEMANTIC_CATEGORY_MAP.items() for name in names}
    profile_of = {name: value for value, names in QUALITY_PROFILE_MAP.items() for name in names}
    inventory = {}
    for name in datasets:
        identity_name = "genshinvoice_48k_spk" if name == "genshinvoice_spk" else name
        input_root = (input_roots or {}).get(name)
        if name == "genshinvoice_spk" and input_roots is not None:
            input_root = input_roots[identity_name]
        spec = {
            "input_root": input_root,
            "language": language_of[identity_name],
            "category": category_of[identity_name],
            "quality_profile": profile_of[identity_name],
            "speaker": _speaker_spec(name),
        }
        if name == "smoothconv":
            spec["source_layout"] = "streaming_flac_aggregate"
        if name == "genshinvoice_spk":
            spec["source_selector"] = {"sample_rates_hz": [48_000]}
        elif name in SOURCE_SELECTORS:
            spec["source_selector"] = SOURCE_SELECTORS[name]
        inventory[name] = spec
    return inventory


SPK_FILTER_INVENTORY = dataset_inventory(
    SPK_FILTER_DATASETS, input_roots=SPK_FILTER_DATASET_ROOTS,
)


def _yaml_source_selector_flow(selector: dict[str, list[object]]) -> str:
    """把 source selector 渲染为保留字符串 speaker ID 的内联 YAML flow。"""
    fields = []
    if "sample_rates_hz" in selector:
        values = ", ".join(str(value) for value in selector["sample_rates_hz"])
        fields.append(f"sample_rates_hz: [{values}]")
    if "include_speaker_ids" in selector:
        values = ", ".join(f'"{value}"' for value in selector["include_speaker_ids"])
        fields.append(f"include_speaker_ids: [{values}]")
    if "exclude_relative_paths" in selector:
        values = ", ".join(
            f'"{value}"' for value in selector["exclude_relative_paths"]
        )
        fields.append(f"exclude_relative_paths: [{values}]")
    return "{" + ", ".join(fields) + "}"


def _yaml_dataset_comment_lines(dataset_id: str, *, prefix: str = "") -> list[str]:
    """把 catalog 的只读展示信息渲染为 entry 前的 YAML 注释。"""
    metadata = DATASET_CATALOG_MODULE.get_dataset_metadata(dataset_id)
    return [
        f"{prefix}# dataset_catalog: {metadata['display_name']}（{dataset_id}）",
        f"{prefix}#   基本信息：{metadata['summary']}",
        f"{prefix}#   language bucket：{metadata['language_bucket']}；category：{metadata['category']}；",
        f"{prefix}#   quality_profile：{metadata['quality_profile']}",
        f"{prefix}#   特殊属性/备注：{metadata['traits']}",
        f"{prefix}#   采样率/证据：{metadata['sample_rate_or_evidence']}",
        f"{prefix}#   证据状态：{metadata['evidence_status']}",
    ]


def _yaml_dataset_entry_lines(
    inventory: dict[str, dict[str, object]], *, commented: bool = False,
) -> list[str]:
    lines = []
    for dataset_id, spec in inventory.items():
        input_root = spec["input_root"]
        input_root_yaml = "null" if input_root is None else f'"{input_root}"'
        selector_yaml = (
            f', source_selector: {_yaml_source_selector_flow(spec["source_selector"])}'
            if "source_selector" in spec else ""
        )
        source_layout_yaml = (
            f', source_layout: {spec["source_layout"]}'
            if spec.get("source_layout", "regular_file") != "regular_file" else ""
        )
        line = (
            f'  {dataset_id}: {{input_root: {input_root_yaml}, language: {spec["language"]}, '
            f'category: {spec["category"]}, quality_profile: {spec["quality_profile"]}, '
            f'speaker: {_yaml_speaker_flow(spec["speaker"])}{selector_yaml}{source_layout_yaml}}}'
        )
        if commented:
            lines.extend(_yaml_dataset_comment_lines(dataset_id))
            lines.append(f"# {line}")
        else:
            lines.extend(_yaml_dataset_comment_lines(dataset_id))
            lines.append(line)
    return lines


def select_datasets(
    min_sample_rate_khz: int | None = None,
    languages: set[str] | None = None,
    *,
    max_sample_rate_khz: int | None = None,
    include_commented: bool = False,
) -> list[str]:
    """按采样率范围和语言集合筛选数据集，并保持 inventory 顺序。

    ``max_sample_rate_khz`` 为开区间上界；因此 ``max=32`` 只包含现有
    16 kHz 与 24 kHz 分组，不会把低于 32 kHz 的 inventory 错标成 16 kHz，
    也不会凭空引入未登记的 8 kHz 数据。
    """
    if min_sample_rate_khz is None:
        min_sample_rate_khz = 0
    if max_sample_rate_khz is not None and max_sample_rate_khz <= min_sample_rate_khz:
        raise ValueError("max_sample_rate_khz must be greater than min_sample_rate_khz")
    if min_sample_rate_khz < 0:
        raise ValueError("min_sample_rate_khz must be nonnegative")
    if languages is None:
        languages = set(LANGUAGE_MAP)
    unknown_languages = set(languages) - set(LANGUAGE_MAP)
    if unknown_languages:
        raise ValueError(f"unknown languages: {sorted(unknown_languages)!r}")
    language_of = {name: value for value, names in LANGUAGE_MAP.items() for name in names}
    selected = []
    for sample_rate, members in SR_DATASETS.items():
        sample_rate_khz = _dataset_sample_rate_khz(sample_rate)
        if sample_rate_khz < 16 or sample_rate_khz < min_sample_rate_khz:
            continue
        if max_sample_rate_khz is not None and sample_rate_khz >= max_sample_rate_khz:
            continue
        selected.extend(
            name for name in members
            if language_of[name] in languages
            and (include_commented or name not in COMMENTED_DATASETS)
        )
    return selected


def select_commented_datasets(
    min_sample_rate_khz: int | None = None,
    languages: set[str] | None = None,
    *,
    max_sample_rate_khz: int | None = None,
) -> list[str]:
    """按同一 inventory 分组选择已整理数据集，供 YAML 注释渲染。"""
    return [
        name for name in select_datasets(
            min_sample_rate_khz,
            languages,
            max_sample_rate_khz=max_sample_rate_khz,
            include_commented=True,
        ) if name in COMMENTED_DATASETS
    ]


def select_datasets_without_speaker_labels(
    min_sample_rate_khz: int | None = None,
    languages: set[str] | None = None,
    *,
    max_sample_rate_khz: int | None = None,
) -> list[str]:
    """选择 source=none 的数据集，并将原注释项提升为 active inventory。"""
    return [
        name for name in select_datasets(
            min_sample_rate_khz,
            languages,
            max_sample_rate_khz=max_sample_rate_khz,
            include_commented=True,
        ) if _speaker_spec(name)["source"] == "none"
    ]


def validate_step4_active_partitions(outputs: dict[str, dict[str, object]]) -> None:
    """确保所有 Step4 active 逻辑项不会重复扫描同一来源。"""
    partitions = {
        filename: dataset_inventory(options["datasets"], input_roots=DATASET_ROOTS)
        for filename, options in outputs.items()
        if options["step"] == "step4" and options["datasets"]
    }
    owners: dict[str, str] = {}
    by_root: dict[str, list[tuple[str, str, dict[str, object]]]] = {}
    for filename, inventory in partitions.items():
        for dataset_id, spec in inventory.items():
            if dataset_id in owners:
                raise ValueError(
                    f"Step4 dataset {dataset_id!r} is active in both "
                    f"{owners[dataset_id]!r} and {filename!r}"
                )
            owners[dataset_id] = filename
            canonical_root = os.path.realpath(os.path.abspath(str(spec["input_root"])))
            by_root.setdefault(canonical_root, []).append((filename, dataset_id, spec))

    for canonical_root, entries in by_root.items():
        for index, (left_file, left_id, left) in enumerate(entries):
            for right_file, right_id, right in entries[index + 1:]:
                left_selector = left.get("source_selector", {})
                right_selector = right.get("source_selector", {})
                left_rates = set(left_selector.get("sample_rates_hz", []))
                right_rates = set(right_selector.get("sample_rates_hz", []))
                rates_disjoint = bool(
                    left_rates and right_rates and left_rates.isdisjoint(right_rates)
                )
                left_speakers = set(left_selector.get("include_speaker_ids", []))
                right_speakers = set(right_selector.get("include_speaker_ids", []))
                left_resolver = left["speaker"]
                right_resolver = right["speaker"]
                same_speaker_resolver = bool(
                    (
                        left_resolver.get("source")
                        == right_resolver.get("source")
                        == "path_component"
                        and left_resolver.get("path_component")
                        == right_resolver.get("path_component")
                    )
                    or (
                        left_resolver.get("source")
                        == right_resolver.get("source")
                        == "regex"
                        and left_resolver.get("regex") == right_resolver.get("regex")
                        and left_resolver.get("regex_group", "speaker_id")
                        == right_resolver.get("regex_group", "speaker_id")
                    )
                )
                speakers_disjoint = bool(
                    same_speaker_resolver
                    and left_speakers
                    and right_speakers
                    and left_speakers.isdisjoint(right_speakers)
                )
                if not rates_disjoint and not speakers_disjoint:
                    raise ValueError(
                        f"Step4 datasets {left_id!r} ({left_file}) and {right_id!r} "
                        f"({right_file}) share canonical root {canonical_root!r} without "
                        "provably disjoint source selectors"
                    )


def render_config(
    datasets: list[str] | None = None, *, title: str | None = None,
    input_roots: dict[str, str] | None = None, output_root: str | None = None,
    output_sample_rate_hz: int | None = None,
    min_source_sample_rate_hz: int = 16_000,
    commented_datasets: list[str] | None = None,
    step: str = "all",
    speaker_consistency_enabled: bool = False,
    uniform_quality_profile: str | None = None,
    dataset_quality_profile_overrides: dict[str, str] | None = None,
    threshold_overrides: dict[str, dict[str, str]] | None = None,
    step4_advisory_notes: list[str] | None = None,
    audio_export_workers: int = 32,
    audio_export_max_pending: int = 64,
) -> str:
    """按目标步骤组织 generation-3 配置（字节稳定）。"""
    if datasets is None:
        datasets = select_datasets(include_commented=False)
    if commented_datasets is None:
        commented_datasets = [
            dataset_id for dataset_id in select_datasets(include_commented=True)
            if dataset_id in COMMENTED_DATASETS
        ]
    if step not in {"step4", "all"}:
        raise ValueError("step must be exactly 'step4' or 'all'")
    if set(datasets) & set(commented_datasets):
        raise ValueError("active and commented datasets must be disjoint")
    if output_sample_rate_hz is not None and output_sample_rate_hz not in {16_000, 32_000}:
        raise ValueError("output_sample_rate_hz must be 16000 or 32000")
    if (
        isinstance(min_source_sample_rate_hz, bool)
        or not isinstance(min_source_sample_rate_hz, int)
        or min_source_sample_rate_hz <= 0
    ):
        raise ValueError("min_source_sample_rate_hz must be a positive integer")
    if output_root is not None and input_roots is None:
        raise ValueError("input_roots must be provided for a rooted variant")
    if type(speaker_consistency_enabled) is not bool:
        raise ValueError("speaker_consistency_enabled must be a boolean")
    if type(audio_export_workers) is not int or audio_export_workers <= 0:
        raise ValueError("audio_export_workers must be a positive Python integer")
    if type(audio_export_max_pending) is not int or audio_export_max_pending < audio_export_workers:
        raise ValueError(
            "audio_export_max_pending must be a Python integer >= audio_export_workers"
        )
    allowed_quality_profiles = {"quality_balanced", "quality_high", "quality_premium"}
    if uniform_quality_profile not in {None, *allowed_quality_profiles}:
        raise ValueError(
            "uniform_quality_profile must be one of quality_balanced, quality_high, quality_premium"
        )
    if dataset_quality_profile_overrides is not None:
        if not isinstance(dataset_quality_profile_overrides, dict):
            raise ValueError("dataset_quality_profile_overrides must be a mapping")
        unknown_datasets = set(dataset_quality_profile_overrides) - set(datasets)
        if unknown_datasets:
            raise ValueError(
                "dataset_quality_profile_overrides references inactive datasets: "
                f"{sorted(unknown_datasets)!r}"
            )
        invalid_profiles = {
            profile for profile in dataset_quality_profile_overrides.values()
            if profile not in allowed_quality_profiles
        }
        if invalid_profiles:
            raise ValueError(
                "dataset_quality_profile_overrides uses unknown quality profiles: "
                f"{sorted(invalid_profiles)!r}"
            )
    inventory = dataset_inventory([*datasets, *commented_datasets], input_roots=input_roots)
    active_inventory = {name: inventory[name] for name in datasets}
    commented_inventory = {name: inventory[name] for name in commented_datasets}
    rows = "\n".join([
        *_yaml_dataset_entry_lines(active_inventory),
        *_yaml_dataset_entry_lines(commented_inventory, commented=True),
    ])
    config_title = title or "通用主配置"
    output_root_yaml = "null" if output_root is None else f'"{output_root}"'
    output_root_comment = (
        "# output_root 保留 null 占位；运行前请填写独立的实际输出目录。"
        if output_root is None
        else "# 筛选配置已绑定本地输出根和数据集输入根。"
    )
    input_root_comment = (
        "#   input_root：数据集本地绝对输入根；模板以 null 占位，未填写时消费者会 fail-fast。"
        if input_roots is None
        else "#   input_root：数据集本地绝对输入根；筛选配置使用已核验目录。"
    )
    selected_mos_models = ["scoreq", "dnsmos"]
    min_mos_models = len(selected_mos_models)
    threshold_sets = {
        "quality_balanced": {
            "dnsmos": "{ovrl: 3.2, sig: 3.0, bak: 3.8, p808: 3.6}",
            "scoreq": "3.6",
        },
        "quality_high": {
            "dnsmos": "{ovrl: 3.3, sig: 3.4, bak: 4.0, p808: 3.8}",
            "scoreq": "4.2",
        },
        "quality_premium": {
            "dnsmos": "{ovrl: 3.4, sig: 3.6, bak: 4.2, p808: 4.0}",
            "scoreq": "4.5",
        },
    }
    if uniform_quality_profile is not None:
        uniform_thresholds = threshold_sets[uniform_quality_profile]
        threshold_sets = {
            profile: uniform_thresholds for profile in threshold_sets
        }
    # 单配置阈值放宽（如某语言桶 MOS 打分系统性偏低时）；
    # 显式覆盖指定 profile 的 dnsmos/scoreq 阈值，未知 profile/键 fail-closed。
    # shipped 五份 audio_filter 配置统一使用默认阶梯，不再做 per-config 放宽。
    if threshold_overrides is not None:
        for profile, overrides in threshold_overrides.items():
            if profile not in threshold_sets:
                raise ValueError(
                    f"threshold_overrides references unknown quality profile: {profile!r}"
                )
            unknown_keys = set(overrides) - {"dnsmos", "scoreq"}
            if unknown_keys:
                raise ValueError(
                    f"threshold_overrides[{profile!r}] has unknown threshold keys: {sorted(unknown_keys)}"
                )
            threshold_sets[profile] = {**threshold_sets[profile], **overrides}
    if step4_advisory_notes is not None:
        if not all(isinstance(note, str) and "\n" not in note for note in step4_advisory_notes):
            raise ValueError("step4_advisory_notes must be a list of single-line strings")
    dataset_policy_lines = ["  datasets: {}"]
    if dataset_quality_profile_overrides:
        dataset_policy_lines = [
            "  # 指定 dataset 使用对应 quality profile 的阈值；数据集身份字段保持不变。",
            "  datasets:",
        ]
        for dataset_id in datasets:
            profile = dataset_quality_profile_overrides.get(dataset_id)
            if profile is None:
                continue
            dataset_policy_lines.extend([
                f"  # {dataset_id} 使用 {profile} 阈值标准。",
                f"    {dataset_id}:",
                "      thresholds:",
                f"        dnsmos: {threshold_sets[profile]['dnsmos']}",
                f"        scoreq: {threshold_sets[profile]['scoreq']}",
            ])
    selection_policy_lines = [
        "selection_policies:",
        "  global:",
        f"    mos_models: [{', '.join(selected_mos_models)}]",
        f"    min_mos_models: {min_mos_models}",
        "    mos_fusion: min",
        "    early_exit: null",
    ]
    # quality_profiles 等 profile 层仅在存在 active 数据集时渲染：运行时 schema
    # 规定非 global 策略层必须配 datasets（selection_policy._canonicalize）；
    # global route 始终渲染，零 active 配置（整体注释的 zh_16k）据此保留 route 记录。
    if active_inventory:
        selection_policy_lines.extend([
            "  # shipped 默认 route 仅选 scoreq / dnsmos，严格 2/2 通过（DNSMOS 四维 AND）。",
            "  # NISQA/UTMOS 虽已通过真实 GPU smoke，但生产阈值尚未校准；派生策略 opt-in 时须显式补阈值。",
            *(["  # Step6 全部数据集统一使用 quality_balanced 阈值；dataset quality_profile 身份字段保持不变。"]
              if uniform_quality_profile == "quality_balanced" else []),
            *(["  # 本配置 quality_profiles 阈值经 generate_configs.py threshold_overrides 显式放宽。"]
              if threshold_overrides is not None else []),
            "  quality_profiles:",
            "    quality_balanced:",
            "      thresholds:",
            f"        dnsmos: {threshold_sets['quality_balanced']['dnsmos']}",
            f"        scoreq: {threshold_sets['quality_balanced']['scoreq']}",
            "    quality_high:",
            "      thresholds:",
            f"        dnsmos: {threshold_sets['quality_high']['dnsmos']}",
            f"        scoreq: {threshold_sets['quality_high']['scoreq']}",
            "    quality_premium:",
            "      thresholds:",
            f"        dnsmos: {threshold_sets['quality_premium']['dnsmos']}",
            f"        scoreq: {threshold_sets['quality_premium']['scoreq']}",
            "  categories: {}",
            "  languages: {en: {}, zh: {}, mixed: {}, fr: {}, ja: {}, ko: {}, ru: {}}",
            *dataset_policy_lines,
            "  # 可选 exact-source override（shipped 默认不启用）：sources.<dataset_id>.<source_relative_path>；仅 exact match，禁止绝对路径、反斜杠和 '..' segment。",
            "  # sources:",
            "  #   dataset_id:",
            "  #     speaker/example.wav: {mos_models: [utmos, nisqa], min_mos_models: 1}",
        ])
    speaker_consistency_lines = (
        [
            "# 此 shipped Step6 fail-closed 模板显式启用 speaker consistency，但下列 /REPLACE_ME 路径、",
            "# replace-me 版本与全 0 digest 都是拒绝执行的占位，不代表资产可运行。",
            "# 运行前必须替换为真实 WeSpeaker/enrollment 资产、版本、SHA-256 与校准阈值；",
            "# Step5/Step6 execution preflight 会在任何 stage/factory/GPU worker 启动前拒绝占位。",
            "# 即使其他检查通过，缺失 speaker_id、enrollment 或 speaker score 的输出也会在 Step6 最终拒绝。",
            "speaker_consistency:",
            "  enabled: true",
            "  runtime: wespeaker",
            "  model_dir: /REPLACE_ME/audio_filter_pipeline/wespeaker_model",
            "  model_version: replace-me",
            f'  model_sha256: "{"0" * 64}"',
            "  runtime_version: replace-me",
            "  target_sample_rate: 16000",
            "  enrollment_bundle_path: /REPLACE_ME/audio_filter_pipeline/enrollment_bundle.jsonl",
            f'  enrollment_bundle_sha256: "{"0" * 64}"',
            "  enrollment_exclusion_path: /REPLACE_ME/audio_filter_pipeline/enrollment_exclusion.jsonl",
            f'  enrollment_exclusion_sha256: "{"0" * 64}"',
            "  min_cosine_similarity: 0.8",
        ]
        if speaker_consistency_enabled
        else [
            "# 五份 audio_filter 配置不执行 Step5/6，speaker consistency 保持关闭。",
            "speaker_consistency:",
            "  enabled: false",
        ]
    )
    lines = [
        # 文件头
        f"# Audio Filter Pipeline generation-3 {config_title}。",
        f"# 数据集 identity 由 generate_configs.py 的 {len(datasets)} 项 active inventory 和 {len(commented_datasets)} 项注释 inventory 显式维护；注释条目不进入 active datasets 映射，也不会被 pipeline 处理。",
        "",
        # 全局 / 运行控制
        "# step: all = 请求 Step6 完整依赖闭包（Step0→Step1→Step2→Step3→Step5→Step6，不含 Step4）；step4 = 请求 Step4 完整依赖闭包（Step0→Step1→Step2→Step3→Step4）。",
        "pipeline_generation: 3",
        f"step: {step}",
        "# 统一输出根目录：Step0~Step6 的不可变 artifact 与 Step4/Step6 filter 导出的 FLAC 均落在此目录下。",
        output_root_comment,
        f"output_root: {output_root_yaml}",
        "# Step0 源采样率下限：读头发现源音频采样率低于该正整数时直接跳过，",
        f"# 不进入冻结快照及后续步骤；当前配置下限为 {min_source_sample_rate_hz} Hz。",
        f"min_source_sample_rate_hz: {min_source_sample_rate_hz}",
        *(
            [
                f"# 固定导出采样率（仅筛选配置）：{output_sample_rate_hz} Hz。",
                f"output_sample_rate_hz: {output_sample_rate_hz}",
            ]
            if output_sample_rate_hz is not None
            else []
        ),
        "",
        # Step0：冻结输入快照
        "# ================= Step0：冻结输入快照 =================",
        "# 数据集 registry：每个数据集的显式身份，由本文件 inventory 维护，请勿手改",
        "# （重新生成会覆盖本块）。",
        "#   speaker：仅标注该数据集的说话人身份信息——source 声明「有无标签及如何解析」",
        "#     （none=无标签/不解析；path_component/regex/mapping_file=有标签并按指定方式解析），",
        "#     required 声明缺失标签时是否强制报错；它只影响 Step0 的 speaker_id 标注。",
        "#     是否执行声纹相似度由顶层 speaker_consistency.enabled 独立控制，不放在 datasets 内。",
        "#   speaker.source：_spk 数据集用 path_component=0（第一层目录即 speaker_id）；",
        "#     其余（会议/对话/多说话人类）用 none（不标注 speaker 身份）。",
        "#   audio_filter_<language bucket>_<source floor/output rate>k.yaml：文件名表达路由桶、源采样率下限与输出采样率；",
        "#     例如 audio_filter_mixed_32k.yaml 表示 mixed language bucket、32 kHz source floor 与 32 kHz output。",
        "#   language：仅表示本地五份配置使用的 route 处理桶（zh/en/mixed），",
        "#     不表示通用语言学标签，也不是运行时自动语言识别结果；mixed route 不等同于特殊数据集。",
        "#   category：数据集特殊属性元数据（read_speech/expressive/elderly/singing/game_voice）；",
        "#     selection_policies.categories 当前为空，仅作元数据，不参与阈值。",
        input_root_comment,
        "datasets:" if active_inventory else "datasets: {}",
        rows,
        "",
        # Step0：源扫描与不可变快照
        "# ================= Step0：源扫描与不可变快照 =================",
        "#   step0_workers：并发哈希/读头线程数（os.pread 与 hashlib 释放 GIL，线程共享",
        "#   root descriptor，descriptor-bound fail-closed 校验不变）；训练占 ~30% CPU、",
        "#   空闲核充裕，按 32 并发。",
        "step0_workers: 32",
        "",
        # Step1：VAD / 带宽 / 有效采样率分析
        "# ================= Step1：TEN VAD / 带宽 / 有效采样率分析 =================",
        "#   runtime：当前仅支持 ten_vad_native_cpu。",
        "#   hop_size：VAD 帧移（样本数）；threshold：语音概率阈值。",
        "#   workers：进程池工作进程数，上限 STEP1_VAD_MAX_WORKERS=64（超出会被 config 校验拒绝）。",
        "#   max_pending：同时提交给进程池的 in-flight 任务数上限，必须落在 [workers, workers*2]；",
        "#                限制在途 future 数量，避免大目录下一次性提交全部任务撑爆内存。",
        "#   workers/max_pending（32→64, 64→128）：训练仅占 ~30% CPU、空闲核充裕；",
        "#   每个 worker 已钉死单线程（OMP/OPENBLAS/MKL_NUM_THREADS=1），",
        "#   64 worker ≈ 64 线程，低于空闲核数，不会与训练数据加载超订抢核。",
        "vad:",
        "  runtime: ten_vad_native_cpu",
        "  hop_size: 256",
        "  threshold: 0.5",
        "  workers: 64",
        "  max_pending: 128",
        "",
        # Step2：切段（超长段跳过上限 + 最短段长过滤）
        "# ================= Step2：切段 =================",
        "# 连续语音超过该时长仍无安全静音切点的段跳过：不产生 segment 行、",
        "# 不参与 Step3 打分与后续导出；默认 60 秒。",
        "step2_max_segment_duration_sec: 60",
        "# 低于该时长的段直接丢弃：不产生 segment 行、不参与 Step3 打分与后续",
        "# 导出；过短的 VAD 碎片对训练无价值，且会让 SCOREQ 等 MOS 模型收到",
        "# 退化输入；五份 audio_filter 配置统一为 1.0 秒。",
        "step2_min_segment_duration_sec: 1.0",
        "",
        # Step3：MOS 打分
        "# ================= Step3：MOS 打分 =================",
        "use_gpu: true",
        "step3_gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]",
        "# registry 支持五个 selectable MOS 模型；配置策略决定实际执行 route。",
        "# 仅当前策略选中的模型会在 Step3 preflight 访问，未选资产字段不会被读取或校验。",
        "# 路径均相对仓库根解析（cli/main.py 的 _REPO_RELATIVE_CONFIG_PATH_KEYS）。",
        "# NISQA/UTMOS 已通过真实 GPU smoke，但生产阈值尚未校准，故仅保留可选资产字段。",
        "# URGENT-PK checkpoint/commit 虽匹配，shipped source 中 urgentpk/model_mod.py 是 symlink，",
        "# 会被安全 tree scanner fail-closed 拒绝；当前资产不可 opt-in，commit 匹配不等于 tree 可加载。",
        "urgent_pk_checkpoint: third_party_scripts/URGENT-PK/exp/mel_resnet34_urgent24.ckpt",
        "urgent_pk_source_dir: third_party_scripts/URGENT-PK",
        "urgent_pk_encoder: mel",
        "urgent_pk_backbone: resnet34",
        "urgent_pk_score_range: mos_1_to_5",
        "scoreq_model_dir: third_party_scripts/SCOREQ",
        "dnsmos_model_dir: third_party_scripts/DNSMOS",
        "nisqa_model_path: third_party_scripts/NISQA/weights/nisqa.tar",
        "nisqa_model_version: nisqa.tar",
        "nisqa_model_sha256: 4e69f09f85de3baad68654133046f554b4bd70ca96cf09ce29848a28104586cf",
        "utmos_model_path: third_party_scripts/UTMOS/utmos22_strong_step7459_v1.pt",
        "utmos_model_version: utmos22_strong_step7459_v1",
        "utmos_model_sha256: abe1c1aa15225599fa7189a9887da12842f8beb5541add007855ead9acdb6d9a",
        "# NISQA/UTMOS 的 SHA-256 绑定 checkpoint 父目录完整 tree；目录内任一文件变化都会 fail-closed。",
        "# 可信 pickle 加载（weights_only=False）的 fail-closed 门禁：",
        "#   urgent_pk_expected_sha256 = checkpoint 文件 SHA-256；",
        "#   urgent_pk_expected_commit  = source_dir 的 git HEAD。",
        "# 二者必须显式给出且与实际一致，缺失或不匹配即拒绝加载，绝不默认信任任意本地文件；",
        "# 更换权重或源码版本时必须同步更新这两项。",
        "urgent_pk_expected_sha256: 266e995f745f265ea52277ca657f86c7d27077b385136edd85f7ef923bdfed5d",
        "urgent_pk_expected_commit: 484269f41ef17ea2419952e92ace7109b4b96052",
        "",
        # Step4：MOS K/N keep/reject 筛选；标题随 route 动态生成，避免注释漂移。
        f"# ================= Step4：MOS {min_mos_models}/{len(selected_mos_models)} keep/reject 筛选 =================",
        *selection_policy_lines,
        "",
        # Step5：per-output speaker consistency
        "# ================= Step5：WeSpeaker speaker consistency =================",
        "# 开启后 Step5 用冻结 enrollment bundle 对同一 identity 做 L2-normalized cosine，",
        "# 生成 speaker score；Step6 filter 才据此执行最终声纹筛选。",
        *speaker_consistency_lines,
        "",
        # Step4/Step6：执行模式与导出
        # Step4：MOS 筛选执行模式
        "# ================= Step4：MOS 筛选执行模式 =================",
        *(
            [f"# {note}" for note in step4_advisory_notes]
            if step4_advisory_notes else []
        ),
        "# dry_run：不发布 artifact，仅按当前阈值统计 keep/reject（调阈值用）；",
        "# filter（默认）：发布 Step4 artifact 后把 keep 段导出为 FLAC（SE 干净数据集，",
        "# 采样率路由与 Step6 一致）。导出根由 step4_export_root 显性指定，",
        "# null 时落在 <output_root>/audio_exports/step4/。",
        "step4_mode: filter",
        "step4_export_root: null",
        "",
        "# ================= Step6：最终决策与导出 =================",
        "# dry_run：发布 metadata artifact 并统计，不导出音频；",
        "# filter（默认）：发布后导出 FLAC（TSE 数据集）。声纹未启用时等价纯 MOS 导出。",
        "# 导出根由 step6_export_root 显性指定，null 时落在 <output_root>/audio_exports/step6/。",
        "step6_mode: filter",
        "step6_export_root: null",
        "# Step4/Step6 共用导出并发：解码(libsndfile)/重采样(soxr)/FLAC 编码均释放 GIL；",
        f"# 按 {audio_export_workers} worker / max_pending {audio_export_max_pending} 配置。max_inflight_bytes 是严格的 estimated",
        "# waveform/verification bytes gate，不代表进程总 RSS。",
        f"# 导出对 fd 需求高：每次路径操作都从根逐层打开目录链（每层一个 fd），{audio_export_workers} worker",
        *(
            [
                "# 并发峰值 >1024。运行前必须确认 shell 软限制 ulimit -n 1048576（screen/tmux 老",
            ]
            if audio_export_workers * 32 >= 1024 else [
                f"# 并发峰值约为 {audio_export_workers * 32} 个 fd。运行前必须确认 shell 软限制 ulimit -n 1048576（screen/tmux 老",
            ]
        ),
        "# 会话可能只有 1024，fd 耗尽会以 EMFILE 崩溃）。",
        f"audio_export_workers: {audio_export_workers}",
        f"audio_export_max_pending: {audio_export_max_pending}",
        "audio_export_max_inflight_bytes: 4294967296",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成五份 Step4 audio_filter 配置和一份 Step6 spk_filter 配置"
    )
    parser.add_argument("--output-dir", type=Path, default=CONFIG_DIR)
    args = parser.parse_args()
    check_dataset_coverage(SR_DATASETS, SEMANTIC_CATEGORY_MAP, QUALITY_PROFILE_MAP, LANGUAGE_MAP)
    check_spk_filter_root_coverage()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        # 五个语音增强筛选配置只执行到 Step4，保留整理数据集为注释。
        "audio_filter_zh_32k.yaml": {
            "datasets": select_datasets(32, {"zh"}),
            "commented_datasets": select_commented_datasets(32, {"zh"}),
            "title": "32 kHz 及以上的中文数据集 Step4 配置",
            "step": "step4",
            "output_root": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_zh_32k",
            "output_sample_rate_hz": 32_000,
            "min_source_sample_rate_hz": 32_000,
            "audio_export_workers": 8,
            "audio_export_max_pending": 16,
        },
        "audio_filter_en_32k.yaml": {
            "datasets": select_datasets(32, {"en"}),
            "commented_datasets": select_commented_datasets(32, {"en"}),
            "title": "32 kHz 及以上的英文数据集 Step4 配置",
            "step": "step4",
            "output_root": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_en_32k",
            "output_sample_rate_hz": 32_000,
            "min_source_sample_rate_hz": 32_000,
        },
        "audio_filter_mixed_32k.yaml": {
            "datasets": select_datasets(32, {"mixed"}),
            "commented_datasets": select_commented_datasets(32, {"mixed"}),
            "title": "mixed language bucket，32 kHz source floor / 32 kHz output 的 Step4 配置",
            "step": "step4",
            "output_root": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_mixed_32k",
            "output_sample_rate_hz": 32_000,
            "min_source_sample_rate_hz": 32_000,
        },
        "audio_filter_zh_16k.yaml": {
            # 2026-09-16 dry_run 实测：统一阶梯下筛选产出过低（合计 1.7%、约 22.9 h，
            # 且 aishell4 为 channel 0 远场打分异常），四个数据集全部转为注释 inventory，
            # 不再执行 Step0–Step4；重跑前先按实测分位数重标定阈值。
            "datasets": [],
            "commented_datasets": ["aishell4", "aishell5", "alimeeting", "mer2023"],
            "title": "32 kHz 以下的中文无说话人标签数据集 Step4 配置",
            "step": "step4",
            "output_root": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_zh_16k",
            "output_sample_rate_hz": 16_000,
            "step4_advisory_notes": [
                "2026-09-16 dry_run 实测（统一阶梯，基于已发布 Step3 分数）：合计 keep 6012/346310 段（1.7%）、约 22.9 h。",
                "其中 alimeeting 5997 段 / 22.9 h，占保留量 99.9%；aishell5 4 段、mer2023 11 段、aishell4 0 段。",
                "aishell4 为 channel 0 远场打分异常，降档无法解决。",
                "结论：当前阈值下不值得执行 filter 并把导出数据加入后续实验任务；四个数据集已全部转为下方注释 inventory，恢复处理前先按实测分位数重标定阈值。",
            ],
        },
        "audio_filter_en_16k.yaml": {
            "datasets": select_datasets_without_speaker_labels(0, {"en"}, max_sample_rate_khz=32),
            "commented_datasets": [],
            "title": "32 kHz 以下的英文无说话人标签数据集 Step4 配置",
            "step": "step4",
            "output_root": "/data_nvme0n1/zxc/code/use_train/data/shared/clean/filter/audio_filter_en_16k",
            "output_sample_rate_hz": 16_000,
        },
        # Step6 处理全部带 speaker 身份的数据集；Genshin 仅保留 48 kHz 视图。
        "spk_filter.yaml": {
            "datasets": SPK_FILTER_DATASETS,
            "commented_datasets": [],
            "title": "speaker identity Step6 fail-closed 模板（speaker 缺失最终拒绝）",
            "step": "all",
            "output_root": "/data_nvme0n1/zxc/code/use_train/data/tse/spk/filter",
            "output_sample_rate_hz": 16_000,
            "speaker_consistency_enabled": True,
            "uniform_quality_profile": "quality_balanced",
        },
    }
    validate_step4_active_partitions(outputs)
    for filename, options in outputs.items():
        output_path = output_dir / filename
        input_roots = (
            SPK_FILTER_DATASET_ROOTS if filename == "spk_filter.yaml" else DATASET_ROOTS
        )
        output_path.write_text(
            render_config(input_roots=input_roots, **options), encoding="utf-8",
        )
        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
