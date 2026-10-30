"""MC-LibriMix 生成、数值资源和 WeSep 索引的契约测试。"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import yaml

from tools.prepare_data_tools.generators.tse.mc_librimix import (
    build_spatial_cues,
    prepare_data,
    scan_librimix,
    simulate_data,
)
from tools.prepare_data_tools.generators.tse.mc_librimix.simulation import create_scenarios

pytestmark = [pytest.mark.unit, pytest.mark.contract]

PACKAGE = "tools.prepare_data_tools.generators.tse.mc_librimix"
CLI_MODULES = (
    "simulation.create_scenarios",
    "simulation.simulate_librimix",
    "simulate_data",
    "build_spatial_cues",
    "scan_librimix",
    "prepare_data",
)
MIXTURE_ID = "123-456-0001_789-321-0002"
SECOND_ID = "123-456-0003_789-321-0004"


@pytest.fixture
def config():
    return {
        "version": "unit-test-only",
        "dataset": {"seed": 17, "splits": ["test"], "max_sampling_attempts": 1000},
        "room": {
            "dimensions": {"min": [5, 5, 3], "max": [6, 6, 3]},
            "rt60": {"min": 0.1, "max": 0.2},
            "boundary_margin": 0.2,
        },
        "array": {
            "mic_positions": [[-0.02, 0, 0], [0.02, 0, 0], [0, -0.02, 0], [0, 0.02, 0]],
            "height": {"min": 1.2, "max": 1.3},
            "rotation_deg": {"min": -90, "max": 90},
        },
        "sources": {
            "num_speakers": 2,
            "azimuth_deg": {"min": -180, "max": 180},
            "horizontal_distance": {"min": 0.3, "max": 0.7},
            "height": {"min": 1.1, "max": 1.5},
            "min_azimuth_difference_deg": 15,
        },
        "audio": {"sample_rate": 16000},
        "rir": {"early_window_ms": 0.25, "preserve_time_of_flight": False, "sound_speed": 343},
        "outputs": {
            "early_source": True,
            "early_mixture": True,
            "reverb_source": True,
            "reverb_mixture": True,
        },
    }


def _write_config(path, config):
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_sources(root, mixture_id=MIXTURE_ID):
    first, second = np.zeros(128, dtype=np.float32), np.zeros(128, dtype=np.float32)
    first[[0, 20, 60]] = [0.75, -0.5, 0.25]
    second[[0, 24, 64]] = [0.5, 0.25, -0.5]
    for name, waveform in (("s1", first), ("s2", second)):
        path = root / "test" / name / f"{mixture_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, waveform, 16000, subtype="FLOAT")
    return first, second


@pytest.fixture
def inputs(tmp_path, config):
    root = tmp_path / "input"
    sources = _write_sources(root)
    config_path = _write_config(tmp_path / "test-only.yaml", config)
    return root, config_path, sources


class FakeGpuRIR:
    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def beta_SabineEstimation(self, room, rt60):
        assert np.all(room > 0) and rt60 > 0
        return np.full(6, 0.5)

    def t2n(self, rt60, room):
        return np.ones(3, dtype=int)

    def simulateRIR(self, room, beta, sources, microphones, image_order, rt60, sample_rate):
        self.calls.append((sources.copy(), microphones.copy(), sample_rate))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("injected RIR failure")
        gains = np.arange(1, len(microphones) + 1, dtype=np.float32)
        rirs = np.zeros((2, len(microphones), 16), dtype=np.float32)
        rirs[0, :, 0] = gains
        rirs[1, :, 0] = gains * 0.5
        rirs[0, :, 12] = gains * 0.25
        rirs[1, :, 12] = -gains * 0.125
        return rirs


@pytest.mark.subprocess
@pytest.mark.parametrize("module", CLI_MODULES)
def test_import_and_module_help_do_not_require_gpurir(tmp_path, module):
    (tmp_path / "gpuRIR.py").write_text(
        "raise ImportError('gpuRIR intentionally unavailable in unit test')\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tmp_path), str(root))))
    qualified = f"{PACKAGE}.{module}"
    code = (
        "import importlib, sys; "
        f"module = importlib.import_module({qualified!r}); "
        "assert 'gpuRIR' not in sys.modules; "
        "module.main(['--help'])"
    )
    for args in (("-c", code), ("-m", qualified, "--help")):
        result = subprocess.run(
            [sys.executable, *args], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout and "--help" in result.stdout


@pytest.mark.parametrize("field", ["version", "dataset", "room", "array", "sources", "audio", "rir", "outputs"])
def test_config_requires_every_section(tmp_path, config, field):
    del config[field]
    path = _write_config(tmp_path / "invalid.yaml", config)
    with pytest.raises(ValueError, match="requires exactly these fields"):
        create_scenarios.load_config(path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("dataset", "seed", True, "integer"),
        ("dataset", "max_sampling_attempts", 0, "integer"),
        ("dataset", "splits", ["typo"], "splits"),
        ("dataset", "splits", ["test", "test"], "splits"),
        ("room", "dimensions", {"min": [6, 5, 3], "max": [5, 5, 3]}, "min must not exceed"),
        ("room", "rt60", {"min": float("nan"), "max": 0.2}, "finite"),
        ("array", "mic_positions", [[0, 0, 0], [0, 0, 0]], "distinct"),
        ("array", "height", {"min": 10, "max": 10}, "outside"),
        ("sources", "num_speakers", 3, "exactly 2"),
        ("sources", "min_azimuth_difference_deg", 181, "infeasible"),
        ("audio", "sample_rate", 0, "positive integer"),
        ("rir", "preserve_time_of_flight", "false", "boolean"),
        ("rir", "early_window_ms", 0.000001, "at least one sample"),
        ("outputs", "early_source", 1, "booleans"),
        ("audio", "typo", 16000, "requires exactly"),
    ],
)
def test_config_rejects_invalid_fields(tmp_path, config, section, field, value, message):
    config[section][field] = value
    with pytest.raises(ValueError, match=message):
        create_scenarios.load_config(_write_config(tmp_path / "invalid.yaml", config))


@pytest.mark.parametrize("split", ["typo", "dev"])
def test_unconfigured_split_fails_before_publication(tmp_path, config, inputs, split):
    root, config_path, _ = inputs
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match="split"):
        create_scenarios.build_scenarios(config, root, split)
    with pytest.raises(ValueError, match="split"):
        simulate_data.run(config_path, root, output, split, backend=FakeGpuRIR())
    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


@pytest.mark.parametrize("splits", [(), ("typo",), ("test", "test"), "test"])
def test_prepare_rejects_invalid_splits(tmp_path, splits):
    with pytest.raises(ValueError, match="splits"):
        prepare_data.prepare_data(tmp_path, tmp_path / "index", splits=splits)


def test_cue_export_rejects_invalid_split_before_opening_manifest(tmp_path):
    with pytest.raises(ValueError, match="Invalid split"):
        build_spatial_cues.export_spatial_cues(tmp_path / "missing.csv", tmp_path / "out", "typo")


def test_cue_export_rejects_out_of_range_azimuth(tmp_path, config, inputs):
    root, _, _ = inputs
    rows = create_scenarios.build_scenarios(config, root)
    rows[0]["source_1_azimuth_deg"] = 400.0
    manifest = tmp_path / "scenarios.csv.gz"
    create_scenarios.write_scenarios(rows, manifest)
    with pytest.raises(ValueError, match="Invalid azimuth"):
        build_spatial_cues.export_spatial_cues(manifest, tmp_path / "cues")
    assert not (tmp_path / "cues").exists()


def test_prepare_rejects_out_of_range_azimuth_cue(tmp_path, config, inputs, monkeypatch):
    root, config_path, _ = inputs
    backend = FakeGpuRIR()
    monkeypatch.setitem(sys.modules, "gpuRIR", backend)
    monkeypatch.chdir(tmp_path)
    simulate_data.run(config_path.name, root.name, "generated", "test")
    cue_path = tmp_path / "generated/test/cues/spatial_npy" / f"{MIXTURE_ID}__123.npy"
    payload = np.load(cue_path, allow_pickle=False)
    payload["azimuth"][0] = 7.0
    np.save(cue_path, payload)
    with pytest.raises(ValueError, match="azimuth must use radians"):
        prepare_data.prepare_data("generated", "index", splits=("test",))
    assert not (tmp_path / "index").exists()


def test_scenarios_have_stable_seed_geometry_and_reproducible_gzip(tmp_path, config, inputs):
    root, config_path, _ = inputs
    assert create_scenarios.load_config(config_path) == config
    rows = create_scenarios.build_scenarios(config, root)
    expected_seed = int.from_bytes(
        hashlib.sha256(f"unit-test-only:17:test:{MIXTURE_ID}".encode()).digest()[:8], "little",
    )
    assert rows[0]["scenario_seed"] == expected_seed
    _write_sources(root, SECOND_ID)
    expanded = create_scenarios.build_scenarios(config, root)
    assert next(row for row in expanded if row["mixture_id"] == MIXTURE_ID) == rows[0]
    changed = copy.deepcopy(config)
    changed["dataset"]["seed"] += 1
    assert create_scenarios.build_scenarios(changed, root)[0] != expanded[0]
    row = rows[0]
    center = np.array([row[f"array_{axis}"] for axis in "xyz"])
    for index in (1, 2):
        position = np.array([row[f"source_{index}_{axis}"] for axis in "xyz"])
        delta = position - center
        assert np.linalg.norm(delta) == pytest.approx(row[f"source_{index}_distance"])
        assert math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))) == pytest.approx(
            row[f"source_{index}_elevation_deg"],
        )
        assert np.all(position >= config["room"]["boundary_margin"])
        assert np.all(position <= np.array([row[f"room_{axis}"] for axis in "xyz"]) - 0.2)
    first, second = tmp_path / "first.csv.gz", tmp_path / "second.csv.gz"
    for path in (first, second):
        create_scenarios.write_scenarios(rows, path)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as stream:
        decoded = list(csv.DictReader(stream))
    assert decoded == create_scenarios.read_scenarios(first)
    assert decoded[0]["scenario_seed"] == str(expected_seed)
    for field in create_scenarios.FIELDS[6:]:
        assert float(decoded[0][field]) == pytest.approx(row[field], abs=1e-8)
    with pytest.raises(FileExistsError):
        create_scenarios.write_scenarios(rows, first)


@pytest.mark.parametrize("problem", ["missing_pair", "frames", "sample_rate", "channels", "empty", "nonfinite"])
def test_generation_rejects_inconsistent_sources_without_partial_output(tmp_path, inputs, problem):
    root, config_path, sources = inputs
    path = root / "test/s2" / f"{MIXTURE_ID}.wav"
    if problem == "missing_pair":
        path.rename(path.with_name(f"{SECOND_ID}.wav"))
    else:
        audio = sources[1]
        if problem == "frames":
            audio = audio[:-1]
        elif problem == "channels":
            audio = np.stack([audio, audio], axis=1)
        elif problem == "empty":
            audio = audio[:0]
        elif problem == "nonfinite":
            audio = audio.copy()
            audio[0] = np.nan
        sf.write(path, audio, 8000 if problem == "sample_rate" else 16000, subtype="FLOAT")
    backend = FakeGpuRIR()
    output = tmp_path / "generated"
    with pytest.raises(ValueError):
        simulate_data.run(config_path, root, output, backend=backend)
    assert not backend.calls
    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def _expected_waveforms(sources):
    expected = {}
    for condition in ("early", "reverb"):
        images = []
        for index, source in enumerate(sources):
            channels = []
            for gain in range(1, 5):
                impulse = np.zeros(16)
                impulse[0] = gain * (1 if index == 0 else 0.5)
                if condition == "reverb":
                    impulse[12] = gain * (0.25 if index == 0 else -0.125)
                channels.append(np.convolve(source, impulse)[:len(source)])
            images.append(np.stack(channels, axis=1))
            expected[f"s{index + 1}_{condition}"] = images[-1]
        expected[f"mix_{condition}"] = images[0] + images[1]
    peak = max(np.abs(audio).max() for audio in expected.values())
    assert peak > 1
    return {name: audio / peak for name, audio in expected.items()}


@pytest.mark.parametrize(("mix_condition", "target_condition"), [("reverb", "early"), ("early", "reverb")])
def test_fake_backend_end_to_end_waveforms_cues_and_wesep_indexes(
    tmp_path, config, inputs, monkeypatch, mix_condition, target_condition,
):
    root, config_path, sources = inputs
    backend = FakeGpuRIR()
    monkeypatch.setitem(sys.modules, "gpuRIR", backend)
    monkeypatch.chdir(tmp_path)
    summary = simulate_data.run(config_path.name, root.name, "generated", "test")
    output = tmp_path / "generated"
    assert summary["counts"] == {"scenarios": 1, "wav_files": 6, "spatial_cues": 2}
    assert summary["config"] == config and summary["seed"] == 17
    assert summary["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert json.loads((output / "metadata/generation_summary.json").read_text()) == summary
    assert len(backend.calls) == 1
    assert backend.calls[0][1].shape == (4, 3)
    assert backend.calls[0][2] == 16000
    actual = {}
    assert len(list(output.rglob("*.wav"))) == 6
    for name, expected in _expected_waveforms(sources).items():
        path = output / "test" / name / f"{MIXTURE_ID}.wav"
        info = sf.info(path)
        assert (info.channels, info.frames, info.samplerate, info.subtype) == (4, 128, 16000, "PCM_16")
        actual[name], _ = sf.read(path, always_2d=True)
        assert np.isfinite(actual[name]).all()
        np.testing.assert_allclose(actual[name], expected, rtol=0, atol=1.1 / 32768)
    for condition in ("early", "reverb"):
        np.testing.assert_allclose(
            actual[f"mix_{condition}"], actual[f"s1_{condition}"] + actual[f"s2_{condition}"],
            rtol=0, atol=2 / 32768,
        )
    assert not np.allclose(actual["s1_early"], actual["s1_reverb"])
    row = create_scenarios.read_scenarios(output / "metadata/scenarios_v1.csv.gz")[0]
    cue_index_path = output / "test/cues/spatial.json"
    cue_index = json.loads(cue_index_path.read_text())
    assert set(cue_index) == {f"{MIXTURE_ID}::123", f"{MIXTURE_ID}::789"}
    for i, speaker in enumerate(("123", "789"), 1):
        relative = Path(cue_index[f"{MIXTURE_ID}::{speaker}"]["path"])
        assert not relative.is_absolute() and relative.suffix == ".npy"
        cue = np.load(cue_index_path.parent / relative, allow_pickle=False)
        assert cue.dtype.names == ("azimuth", "elevation", "distance")
        assert cue.shape == (1,)
        for field in cue.dtype.names:
            assert cue[field].dtype == np.dtype("float32")
            value = float(row[f"source_{i}_{field}" if field == "distance" else f"source_{i}_{field}_deg"])
            expected = value if field == "distance" else math.radians(value)
            assert float(cue[field][0]) == pytest.approx(expected, abs=1e-6)
    stats = prepare_data.prepare_data("generated", "index", mix_condition, target_condition, ("test",))
    assert stats["samples"] == 1 and stats["cues"] == 2
    sample_path = tmp_path / "index/test/samples.jsonl"
    raw_list = sample_path.with_name("raw.list")
    assert not raw_list.is_symlink() and raw_list.read_text() == "samples.jsonl\n"
    sample = json.loads(sample_path.read_text())
    for group in (sample["mix"], sample["src"]):
        for paths in group.values():
            for path in paths:
                assert not Path(path).is_absolute()
                assert (sample_path.parent / path).is_file()
    monkeypatch.chdir(root)
    listed_manifests = [
        (raw_list.parent / line).resolve() for line in raw_list.read_text().splitlines()
    ]
    assert listed_manifests == [sample_path]
    records = [json.loads(line) for line in listed_manifests[0].read_text().splitlines()]
    assert records == [sample]
    assert set(sample) == {"key", "spk", "mix", "src"}
    assert sample["key"] == MIXTURE_ID and sample["spk"] == ["123", "789"]
    assert set(sample["mix"]) == {"default"}
    assert len(sample["mix"]["default"]) == 1
    assert (sample_path.parent / sample["mix"]["default"][0]).resolve() == (
        output / "test" / f"mix_{mix_condition}" / f"{MIXTURE_ID}.wav"
    )
    assert set(sample["src"]) == set(sample["spk"])
    for index, speaker in enumerate(sample["spk"], 1):
        paths = sample["src"][speaker]
        assert len(paths) == 1
        assert (sample_path.parent / paths[0]).resolve() == (
            output / "test" / f"s{index}_{target_condition}" / f"{MIXTURE_ID}.wav"
        )
    cue_config_path = sample_path.with_name("cues.yaml")
    cue_config = yaml.safe_load(cue_config_path.read_text())
    assert set(cue_config) == {"cues"} and set(cue_config["cues"]) == {"spatial"}
    spatial = cue_config["cues"]["spatial"]
    assert set(spatial) == {"type", "format", "guaranteed", "fields", "policy"}
    assert spatial["type"] == "npy" and spatial["format"] == "fields"
    assert spatial["guaranteed"] is True and spatial["fields"] == ["azimuth", "elevation"]
    policy = spatial["policy"]
    assert set(policy) == {"type", "key", "resource"}
    assert policy["type"] == "fixed" and policy["key"] == "mix_spk_id"
    assert not Path(policy["resource"]).is_absolute()
    indexed_cues = (cue_config_path.parent / policy["resource"]).resolve()
    assert indexed_cues == cue_index_path
    resources = json.loads(indexed_cues.read_text())
    assert resources == cue_index
    assert set(resources) == {f"{sample['key']}::{speaker}" for speaker in sample["spk"]}
    for speaker in sample["spk"]:
        item = resources[f"{sample['key']}::{speaker}"]
        assert set(item) == {"path"} and not Path(item["path"]).is_absolute()
        payload_path = (indexed_cues.parent / item["path"]).resolve()
        assert payload_path == indexed_cues.parent / "spatial_npy" / f"{MIXTURE_ID}__{speaker}.npy"
        assert payload_path.is_file()
    assert not list(tmp_path.glob(".*.staging-*"))


@pytest.mark.parametrize("problem", ["frames", "sample_rate", "channels", "ids"])
def test_scan_rejects_inconsistent_audio_and_does_not_publish(tmp_path, problem):
    directories = [tmp_path / name for name in ("mix", "s1", "s2")]
    for index, directory in enumerate(directories):
        directory.mkdir()
        frames, channels, rate, name = 128, 4, 16000, MIXTURE_ID
        if index == 2:
            frames = 127 if problem == "frames" else frames
            channels = 2 if problem == "channels" else channels
            rate = 8000 if problem == "sample_rate" else rate
            name = SECOND_ID if problem == "ids" else name
        sf.write(directory / f"{name}.wav", np.zeros((frames, channels)), rate)
    output = tmp_path / "samples.jsonl"
    with pytest.raises(ValueError, match="mismatch|match exactly"):
        scan_librimix.write_samples(*directories, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".samples.jsonl.*.tmp"))


def test_nonempty_outputs_are_not_overwritten(tmp_path, inputs):
    root, config_path, _ = inputs
    output = tmp_path / "generated"
    simulate_data.run(config_path, root, output, backend=FakeGpuRIR())
    snapshot = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        simulate_data.run(config_path, root, output, backend=FakeGpuRIR())
    with pytest.raises(FileExistsError):
        build_spatial_cues.export_spatial_cues(output / "metadata/scenarios_v1.csv.gz", output)
    index = tmp_path / "index"
    prepare_data.prepare_data(output, index, splits=("test",))
    with pytest.raises(FileExistsError):
        prepare_data.prepare_data(output, index, splits=("test",))
    directories = [output / "test" / name for name in ("mix_reverb", "s1_early", "s2_early")]
    samples = tmp_path / "standalone/samples.jsonl"
    assert scan_librimix.write_samples(*directories, samples) == 1
    assert [json.loads(line) for line in samples.read_text().splitlines()] == scan_librimix.scan_records(*directories, samples)
    original = samples.read_bytes()
    with pytest.raises(FileExistsError):
        scan_librimix.write_samples(*directories, samples)
    assert samples.read_bytes() == original
    assert snapshot == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_mid_simulation_failure_removes_previously_written_assets_and_staging(tmp_path, inputs):
    root, config_path, _ = inputs
    _write_sources(root, SECOND_ID)
    backend = FakeGpuRIR(fail_on_call=2)
    output = tmp_path / "generated"
    with pytest.raises(RuntimeError, match="injected RIR failure"):
        simulate_data.run(config_path, root, output, backend=backend)
    assert len(backend.calls) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
    assert not list(tmp_path.rglob("generation_summary.json"))


def test_mid_prepare_failure_removes_indexes_and_staging(tmp_path, inputs, monkeypatch):
    root, config_path, _ = inputs
    output = tmp_path / "generated"
    simulate_data.run(config_path, root, output, backend=FakeGpuRIR())
    index = tmp_path / "index"

    def fail_dump(*args, **kwargs):
        staged = list(tmp_path.glob(".index.staging-*/test/samples.jsonl"))
        assert len(staged) == 1
        assert json.loads(staged[0].read_text())["key"] == MIXTURE_ID
        raise RuntimeError("injected YAML write failure")

    monkeypatch.setattr(yaml, "safe_dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected YAML write failure"):
        prepare_data.prepare_data(output, index, splits=("test",))
    assert not index.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
    assert len(list(output.rglob("*.wav"))) == 6


def test_cue_publish_failure_rolls_back_partial_links(tmp_path, inputs, config, monkeypatch):
    root, _, _ = inputs
    manifest = tmp_path / "scenarios.csv.gz"
    create_scenarios.write_scenarios(create_scenarios.build_scenarios(config, root), manifest)
    output = tmp_path / "cues"
    link = os.link
    calls = []

    def fail_second_link(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            assert Path(calls[0]).is_file()
            raise OSError("injected cue publish failure")
        return link(source, destination)

    monkeypatch.setattr(build_spatial_cues.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected cue publish failure"):
        build_spatial_cues.export_spatial_cues(manifest, output)
    assert len(calls) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".cues.cues-*"))
