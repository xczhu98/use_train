"""Dataset 配置与多层 manifest 资源路径契约。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.aec.data.dynamic_aec_build import resolve_aec_dataset_kwargs
from core.data.dataset_build import (
    DatasetBuildSpec,
    build_dataset,
    resolve_dataset_build_options,
)
from core.data.offline_dataset import _read_scp_four_col
from core.data.resource_paths import (
    repository_root,
    resolve_config_resource_path,
    resolve_data_section_resource_paths,
    resolve_manifest_resource_path,
)
from tasks.tse.manifest_contract import parse_enrollment_manifest
from tasks.tse.data.wesep_adapter import (
    MANIFEST_FORMAT_JSONL,
    MANIFEST_FORMAT_PATH_LIST,
    PATH_BASE_MANIFEST_RELATIVE,
    load_wesep_records,
)


def test_config_data_path_anchors_to_repo_root_and_plain_relative_is_preserved():
    assert resolve_config_resource_path("data/test/example.scp") == str(
        repository_root() / "data/test/example.scp"
    )
    assert resolve_config_resource_path("manifests/example.scp") == (
        "manifests/example.scp"
    )
    assert resolve_config_resource_path("./data/example.scp") == (
        "./data/example.scp"
    )


def test_config_data_path_rejects_parent_and_symlink_escape(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "data"
    outside = tmp_path / "outside"
    data.mkdir(parents=True)
    outside.mkdir()

    with pytest.raises(ValueError, match="escapes repository root"):
        resolve_config_resource_path("data/../../outside/file.scp", repo_root=repo)

    (data / "external").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes repository root"):
        resolve_config_resource_path("data/external/file.scp", repo_root=repo)


def test_section_resolution_is_resource_whitelisted_and_does_not_mutate_config():
    cfg = {
        "data": {
            "mode": "offline_aec",
            "train": {
                "mic_scp": "data/test/mic.scp",
                "farend_scp": "relative/farend.scp",
                "nearend_scp": "/absolute/nearend.scp",
                "model_path": "data/checkpoints/model.pt",
                "checkpoint": "data/checkpoints/resume.pt",
            },
        }
    }

    options = resolve_dataset_build_options(cfg, "train", expected_task="aec")

    assert options.section["mic_scp"] == str(repository_root() / "data/test/mic.scp")
    assert options.section["farend_scp"] == "relative/farend.scp"
    assert options.section["nearend_scp"] == "/absolute/nearend.scp"
    assert options.section["model_path"] == "data/checkpoints/model.pt"
    assert options.section["checkpoint"] == "data/checkpoints/resume.pt"
    assert cfg["data"]["train"]["mic_scp"] == "data/test/mic.scp"


def test_dynamic_aec_builder_resolves_repo_data_manifest_path():
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "train": {
                "real_farend_echo_pair_json": "data/test/pairs.json",
                "simulation_cfg": {},
            },
        }
    }

    kwargs = resolve_aec_dataset_kwargs(cfg, "train")

    assert kwargs["real_farend_echo_pair_json"] == str(
        repository_root() / "data/test/pairs.json"
    )


@pytest.mark.parametrize("task_mode", ["offline", "offline_aec"])
@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("retired_key", ["utt2fs", "speech_length"])
def test_dataset_builder_rejects_retired_offline_sidecar_keys(
    task_mode, split, retired_key
):
    task = "aec" if task_mode.endswith("_aec") else "se"
    cfg = {
        "data": {
            "mode": task_mode,
            "train": {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp"},
            "validation": {
                "clean_scp": "val-clean.scp",
                "noisy_scp": "val-noisy.scp",
            },
        }
    }
    cfg["data"][split][retired_key] = "retired-sidecar.scp"

    with pytest.raises(ValueError, match=rf"{split}.{retired_key}.*four-column"):
        resolve_dataset_build_options(
            cfg, "train" if split == "train" else "val", expected_task=task
        )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_dataset_builder_rejects_offline_section_sample_rate_alias(split):
    cfg = {
        "data": {
            "mode": "offline",
            "sample_rate": 16000,
            "train": {"clean_scp": "clean.scp", "noisy_scp": "noisy.scp"},
            "validation": {
                "clean_scp": "val-clean.scp",
                "noisy_scp": "val-noisy.scp",
            },
        }
    }
    cfg["data"][split]["sample_rate"] = 8000

    with pytest.raises(ValueError, match=rf"{split}.sample_rate.*no longer supported"):
        resolve_dataset_build_options(
            cfg, "train" if split == "train" else "val", expected_task="se"
        )


def test_dynamic_speech_length_metadata_remains_supported():
    cfg = {
        "data": {
            "mode": "dynamic",
            "train": {
                "speech_scp": "data/test/speech.scp",
                "speech_length_file": "data/test/speech.lengths",
                "simulation_cfg": {"metadata": {"speech_length": "diagnostic"}},
            },
        }
    }

    options = resolve_dataset_build_options(cfg, "train", expected_task="se")

    assert options.section["speech_length_file"] == str(
        repository_root() / "data/test/speech.lengths"
    )
    assert options.section["simulation_cfg"]["metadata"]["speech_length"] == (
        "diagnostic"
    )


def test_tse_wesep_metadata_is_not_subject_to_se_aec_offline_sidecar_gate():
    cfg = {
        "data": {
            "mode": "offline_tse",
            "sample_rate": 16000,
            "train": {
                "samples_list": "data/tse/REAL-TSE/train/raw.list",
                "cues_yaml": "data/tse/REAL-TSE/train/cues.yaml",
                "utt2fs": "wesep-owned-metadata",
                "speech_length": "wesep-owned-metadata",
            },
        }
    }

    options = resolve_dataset_build_options(cfg, "train", expected_task="tse")

    assert options.section["utt2fs"] == "wesep-owned-metadata"
    assert options.section["speech_length"] == "wesep-owned-metadata"


def test_section_resolution_covers_tse_simulation_resources_only():
    section = {
        "speech_scp": "data/test/speech.scp",
        "music_scp": "data/test/music.scp",
        "spk2enroll_json": "data/test/enroll.json",
        "simulation_cfg": {
            "checkpoint": "data/checkpoints/model.pt",
        },
    }

    resolve_data_section_resource_paths(section)

    assert section["speech_scp"] == str(repository_root() / "data/test/speech.scp")
    assert section["spk2enroll_json"] == str(
        repository_root() / "data/test/enroll.json"
    )
    assert section["music_scp"] == str(
        repository_root() / "data/test/music.scp"
    )
    assert section["simulation_cfg"]["checkpoint"] == "data/checkpoints/model.pt"


def test_manifest_relative_helper_anchors_only_to_direct_parent(tmp_path):
    manifest = tmp_path / "carrier" / "manifest.scp"
    assert resolve_manifest_resource_path("../audio/a.wav", manifest) == str(
        tmp_path / "audio/a.wav"
    )
    assert resolve_manifest_resource_path("/audio/a.wav", manifest) == "/audio/a.wav"
    assert resolve_manifest_resource_path("~/audio/a.wav", manifest) == "~/audio/a.wav"


def test_offline_scp_paths_use_manifest_parent(tmp_path):
    scp = tmp_path / "scp" / "clean.scp"
    scp.parent.mkdir(parents=True)
    scp.write_text("u 16000 10 ../audio/clean.wav\n", encoding="utf-8")

    scp_paths, _, _ = _read_scp_four_col(scp)

    assert scp_paths["u"] == str(tmp_path / "audio/clean.wav")


def test_tse_raw_list_and_samples_paths_resolve_layer_by_layer(tmp_path):
    raw_list = tmp_path / "lists" / "raw.list"
    samples_jsonl = tmp_path / "manifests" / "samples.jsonl"
    raw_list.parent.mkdir()
    samples_jsonl.parent.mkdir()
    raw_list.write_text("../manifests/samples.jsonl\n", encoding="utf-8")
    sample = {
        "key": "u",
        "spk": ["s"],
        "mix": {"default": ["../audio/mix.wav"]},
        "src": {"s": ["targets/target.wav"]},
    }
    samples_jsonl.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    record = load_wesep_records(
        raw_list,
        manifest_format=MANIFEST_FORMAT_PATH_LIST,
        path_base=PATH_BASE_MANIFEST_RELATIVE,
        mix_variant="default",
        target_selection="all",
    )[0]

    assert record.manifest_path == str(samples_jsonl)
    assert record.mixture_paths == (str(tmp_path / "audio/mix.wav"),)
    assert record.targets[0].source_paths == (
        str(samples_jsonl.parent / "targets/target.wav"),
    )


def test_tse_cues_resource_and_enrollment_paths_resolve_layer_by_layer(tmp_path):
    samples_jsonl = tmp_path / "config" / "samples.jsonl"
    cues_yaml = tmp_path / "config" / "cues.yaml"
    resource_json = tmp_path / "resources" / "audio.json"
    cues_yaml.parent.mkdir()
    resource_json.parent.mkdir()
    samples_jsonl.write_text(
        json.dumps(
            {
                "key": "u",
                "spk": ["s"],
                "mix": {"default": ["mix.wav"]},
                "src": {"s": ["target.wav"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cues_yaml.write_text(
        "cues:\n"
        "  audio:\n"
        "    type: wav\n"
        "    format: waveform\n"
        "    scope: speaker\n"
        "    guaranteed: true\n"
        "    policy:\n"
        "      type: fixed\n"
        "      key: spk_id\n"
        "      resource: ../resources/audio.json\n",
        encoding="utf-8",
    )
    resource_json.write_text(
        json.dumps({"s": [{"utt_id": "e", "path": "../audio/enroll.wav"}]}),
        encoding="utf-8",
    )

    cue = load_wesep_records(
        samples_jsonl,
        manifest_format=MANIFEST_FORMAT_JSONL,
        path_base=PATH_BASE_MANIFEST_RELATIVE,
        mix_variant="default",
        target_selection="all",
        cues_yaml=cues_yaml,
        enabled_cues=("audio",),
    )[0].targets[0].cue("audio")

    assert cue.spec.resource_path == str(resource_json)
    assert cue.items[0]["path"] == str(tmp_path / "audio/enroll.wav")


def test_tse_dynamic_strict_enrollment_paths_use_json_parent(tmp_path):
    strict = tmp_path / "strict" / "enroll.json"
    strict.parent.mkdir()
    strict.write_text(
        json.dumps({"dataset": {"s": ["../audio/e.wav"]}}), encoding="utf-8"
    )

    strict_result = parse_enrollment_manifest(
        str(strict), {("dataset", "s")}
    )

    assert strict_result[("dataset", "s")] == (str(tmp_path / "audio/e.wav"),)


def test_offline_factory_receives_resolved_config_resource_paths():
    requests = []
    spec = DatasetBuildSpec(
        task="aec",
        dynamic_dataset_cls=lambda **_kwargs: None,
        offline_dataset_factory=lambda request: requests.append(request) or request,
    )
    cfg = {
        "data": {
            "mode": "offline_aec",
            "train": {
                "mic_scp": "data/test/mic.scp",
                "farend_scp": "relative/farend.scp",
                "nearend_scp": "/absolute/nearend.scp",
            },
        }
    }

    request = build_dataset(cfg, spec=spec)

    assert request is requests[0]
    assert request.section["mic_scp"] == str(repository_root() / "data/test/mic.scp")
    assert request.section["farend_scp"] == "relative/farend.scp"
    assert request.section["nearend_scp"] == "/absolute/nearend.scp"


pytestmark = [pytest.mark.unit]
