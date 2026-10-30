"""Unit tests for core/data/simulation_utils.py."""

from __future__ import annotations

import builtins
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from core.data.simulation_utils import read_source_scp, rng_uniform_or_beta, weighted_choice


class _RecordingRng:
    def __init__(self, *, uniform_value=0.0, beta_value=0.5, choice_index=0):
        self.uniform_value = uniform_value
        self.beta_value = beta_value
        self.choice_index = choice_index
        self.calls = []

    def uniform(self, lo, hi):
        self.calls.append(("uniform", lo, hi))
        return self.uniform_value

    def beta(self, alpha, beta):
        self.calls.append(("beta", alpha, beta))
        return self.beta_value

    def choice(self, size, *, p):
        self.calls.append(("choice", size, np.asarray(p)))
        return self.choice_index


def test_uniform_delegates_exact_range_to_rng():
    rng = _RecordingRng(uniform_value=-13.0)
    assert rng_uniform_or_beta(rng, -40.0, 0.0, "uniform") == -13.0
    assert rng.calls == [("uniform", -40.0, 0.0)]


def test_uniform_and_beta_have_expected_fixed_seed_distributions():
    lo, hi = -40.0, 0.0
    uniform_rng = np.random.default_rng(123)
    uniform = np.array(
        [rng_uniform_or_beta(uniform_rng, lo, hi, "uniform") for _ in range(20_000)]
    )
    beta_rng = np.random.default_rng(123)
    beta = np.array(
        [rng_uniform_or_beta(beta_rng, lo, hi, "beta") for _ in range(20_000)]
    )

    assert np.all((uniform >= lo) & (uniform <= hi))
    assert np.all((beta >= lo) & (beta <= hi))
    assert uniform.min() < lo + 0.1 * (hi - lo)
    assert uniform.max() > hi - 0.1 * (hi - lo)
    assert beta.mean() == pytest.approx((lo + hi) / 2.0, abs=0.3)
    assert beta.var() < uniform.var()


@pytest.mark.parametrize(
    ("unit_value", "expected"),
    [(0.0, -40.0), (0.5, -20.0), (1.0, 0.0)],
)
def test_beta_maps_unit_interval_to_requested_bounds(unit_value, expected):
    rng = _RecordingRng(beta_value=unit_value)
    assert rng_uniform_or_beta(
        rng, -40.0, 0.0, "beta", beta_a=2.0, beta_b=3.0
    ) == expected
    assert rng.calls == [("beta", 2.0, 3.0)]


def test_rejects_unknown_distribution():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="unknown distribution"):
        rng_uniform_or_beta(rng, -40.0, 0.0, "gaussian")


def test_falls_back_to_global_random_when_rng_is_none(monkeypatch):
    monkeypatch.setattr("core.data.simulation_utils.random.uniform", lambda lo, hi: -12.5)
    monkeypatch.setattr("core.data.simulation_utils.random.betavariate", lambda a, b: 0.25)
    assert rng_uniform_or_beta(None, -40.0, 0.0, "uniform") == -12.5
    assert rng_uniform_or_beta(None, -40.0, 0.0, "beta") == -30.0


def test_weighted_choice_always_picks_the_only_positive_weight_key():
    rng = _RecordingRng(choice_index=0)
    assert weighted_choice(rng, {"a": 1.0, "b": 0.0}) == "a"
    _, size, probs = rng.calls[-1]
    assert size == 2
    np.testing.assert_array_equal(probs, [1.0, 0.0])


def test_weighted_choice_passes_normalized_relative_weights():
    rng = _RecordingRng(choice_index=1)
    assert weighted_choice(rng, {"a": 3.0, "b": 1.0}) == "b"
    _, size, probs = rng.calls[-1]
    assert size == 2
    np.testing.assert_array_equal(probs, [0.75, 0.25])


def test_weighted_choice_normalizes_weights_not_summing_to_one():
    rng = _RecordingRng(choice_index=0)
    assert weighted_choice(rng, {"a": 10.0, "b": 10.0}) == "a"
    _, size, probs = rng.calls[-1]
    assert size == 2
    np.testing.assert_array_equal(probs, [0.5, 0.5])


def test_weighted_choice_rejects_all_zero_weights():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="weights must sum to > 0"):
        weighted_choice(rng, {"a": 0.0, "b": 0.0})


def test_read_source_scp_parses_standard_four_col(tmp_path):
    scp = tmp_path / "sources.scp"
    scp.write_text(
        "uid0 16000 32000 /a/b/c.wav\n"
        "uid1 48000 96000 /x/y/z.flac\n",
        encoding="utf-8",
    )
    by_fs, uids, flat, lengths = read_source_scp(scp)
    assert set(by_fs.keys()) == {16000, 48000}
    assert flat == {
        "uid0": "/a/b/c.wav",
        "uid1": "/x/y/z.flac",
    }
    assert lengths == {"uid0": 32000, "uid1": 96000}


def test_read_source_scp_rejects_unquoted_spaces(tmp_path):
    scp = tmp_path / "sources.scp"
    scp.write_text(
        'dir5_All_Raw Data_0_0hp 44100 783872 /data/noise/MKA/All/Raw Data/0/0hp.wav\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected exactly four fields"):
        read_source_scp(scp)


def test_read_source_scp_handles_quoted_shell_fields(tmp_path):
    """兼容新版带 shell 引号 SCP。"""
    scp = tmp_path / "sources.scp"
    scp.write_text(
        '"dir5_All_Raw Data_0_0hp" 44100 783872 "/data/noise/MKA/All/Raw Data/0/0hp.wav"\n',
        encoding="utf-8",
    )
    by_fs, uids, flat, lengths = read_source_scp(scp)
    assert flat["dir5_All_Raw Data_0_0hp"] == "/data/noise/MKA/All/Raw Data/0/0hp.wav"
    assert lengths["dir5_All_Raw Data_0_0hp"] == 783872


def test_read_source_scp_rejects_three_column_records(tmp_path):
    scp = tmp_path / "sources.scp"
    scp.write_text('uid1 16000 "/data/path with spaces/file.wav"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="expected exactly four fields"):
        read_source_scp(scp)


def test_read_source_scp_accepts_canonical_four_column_records(tmp_path):
    scp = tmp_path / "sources.scp"
    scp.write_text(
        "uid0 192000 384000 /a/b/c.wav\n"
        "'uid 16000' 44100 123 '/123 /audio/quoted path.wav'\n",
        encoding="utf-8",
    )

    by_fs, uids, flat, lengths = read_source_scp(scp)

    assert by_fs == {
        192000: {"uid0": "/a/b/c.wav"},
        44100: {"uid 16000": "/123 /audio/quoted path.wav"},
    }
    assert uids == {192000: ["uid0"], 44100: ["uid 16000"]}
    assert flat == {
        "uid0": "/a/b/c.wav",
        "uid 16000": "/123 /audio/quoted path.wav",
    }
    assert lengths == {"uid0": 384000, "uid 16000": 123}


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("uid 16000 0 /audio.wav", "length must be an exact integer > 0"),
        ("uid 16000 -1 /audio.wav", "length must be an exact integer > 0"),
        ("uid 16000 123", "expected exactly four fields"),
        ("uid 16000 123 456 audio.wav", "expected exactly four fields"),
    ],
)
def test_read_source_scp_rejects_invalid_scp_lines(tmp_path, line, message):
    scp = tmp_path / "sources.scp"
    scp.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_source_scp(scp)


def test_read_source_scp_metadata_only_does_not_load_admission_resources(
    tmp_path, monkeypatch
):
    from core.data.noise_admission import NoiseAdmissionContext

    source = tmp_path / "noise.jsonl"
    source.write_text(
        '{"uid":"id","fs":16000,"length":1,"path":"/audio.wav"}\n',
        encoding="utf-8",
    )

    def unexpected_load(cls):
        raise AssertionError("metadata-only parsing must not load admission resources")

    monkeypatch.setattr(NoiseAdmissionContext, "load", classmethod(unexpected_load))
    assert read_source_scp(source, return_metadata=True)[2] == {"id": "/audio.wav"}

    with pytest.raises(AssertionError, match="must not load admission resources"):
        read_source_scp(source, production_admission=True)


def test_read_source_scp_production_admission_does_not_load_calibration(
    tmp_path, monkeypatch
):
    from core.data import noise_admission_policy

    source = tmp_path / "noise.jsonl"
    source.write_text(
        '{"uid":"id","fs":16000,"length":1,"path":"/audio.wav"}\n',
        encoding="utf-8",
    )

    def unexpected_calibration_load(*args, **kwargs):
        raise AssertionError("production admission must not load calibration")

    monkeypatch.setattr(
        noise_admission_policy,
        "load_admission_calibration",
        unexpected_calibration_load,
        raising=False,
    )

    with pytest.raises(ValueError, match="manifest_schema_version"):
        read_source_scp(source, production_admission=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record["labels"].__setitem__("audioset_top10", []),
        lambda record: record["stages"]["C3"].__setitem__(
            "admission_calibration_version", 2
        ),
    ],
)
def test_strict_final_noise_record_rejects_legacy_admission_fields(
    monkeypatch, mutation
):
    from core.data import simulation_utils

    record = {"labels": {}, "stages": {"C3": {}}}
    mutation(record)
    monkeypatch.setattr(
        simulation_utils,
        "build_validated_noise_record",
        lambda *args, **kwargs: pytest.fail("legacy record must not reach shared builder"),
    )

    with pytest.raises(ValueError, match="legacy admission fields are forbidden"):
        simulation_utils._build_strict_final_noise_record(
            record,
            location="fixture:1",
            context=object(),
        )


def test_manifest_cache_concurrent_writers_use_owned_atomic_temp_files(tmp_path, monkeypatch):
    from core.data import simulation_utils

    source = tmp_path / "sources.scp"
    source.write_text("uid 16000 1 /audio.wav\n", encoding="utf-8")
    cache_path = simulation_utils._manifest_cache_path(source)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    parse_barrier = threading.Barrier(2)
    real_parser = simulation_utils._read_source_manifest_impl

    def synchronized_parser(*args, **kwargs):
        result = real_parser(*args, **kwargs)
        parse_barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(simulation_utils, "_read_source_manifest_impl", synchronized_parser)

    write_test_paths: list[Path] = []
    real_open = builtins.open

    def record_write_test(path, *args, **kwargs):
        path = Path(path)
        if path.name.endswith(".write_test"):
            write_test_paths.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", record_write_test)

    real_unlink = Path.unlink
    cleanup_done = threading.Event()
    failed_source: list[Path] = []

    def track_owned_cleanup(path, *args, **kwargs):
        if path.name.endswith(".write_test"):
            return None
        result = real_unlink(path, *args, **kwargs)
        if failed_source and path == failed_source[0]:
            cleanup_done.set()
        return result

    monkeypatch.setattr(Path, "unlink", track_owned_cleanup)

    real_replace = os.replace
    replace_barrier = threading.Barrier(2)
    replace_sources: list[Path] = []
    replace_lock = threading.Lock()

    def fail_one_adjacent_publish(source_path, target_path):
        source_path = Path(source_path)
        target_path = Path(target_path)
        if target_path != cache_path:
            return real_replace(source_path, target_path)
        with replace_lock:
            writer_index = len(replace_sources)
            replace_sources.append(source_path)
            if writer_index == 0:
                failed_source.append(source_path)
        replace_barrier.wait(timeout=5)
        if writer_index == 0:
            raise OSError("simulated cache publish failure")
        assert cleanup_done.wait(timeout=5)
        return real_replace(source_path, target_path)

    monkeypatch.setattr(simulation_utils.os, "replace", fail_one_adjacent_publish)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: read_source_scp(source), range(2)))

    assert all(result[2] == {"uid": "/audio.wav"} for result in results)
    assert len(set(replace_sources)) == 2
    assert not write_test_paths
    assert cache_path.is_file()


@pytest.mark.parametrize(
    "invalid_field, invalid_value",
    [
        ("_cache_version", "stale"),
        ("_parser_version", "stale"),
        ("_source_inode", -1),
        ("_return_metadata", False),
    ],
)
def test_manifest_cache_reads_valid_fallback_after_invalid_adjacent_cache(
    tmp_path, monkeypatch, invalid_field, invalid_value
):
    from core.data import simulation_utils

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source = tmp_path / "noise.jsonl"
    source.write_text(
        '{"uid":"id","fs":16000,"length":1,"path":"/audio.wav"}\n',
        encoding="utf-8",
    )
    adjacent_cache = simulation_utils._manifest_cache_path(source)
    fallback_cache = simulation_utils._manifest_cache_dir_fallback(source)
    real_named_temporary_file = simulation_utils.tempfile.NamedTemporaryFile

    def reject_read_only_source_directory(*args, **kwargs):
        if Path(kwargs["dir"]) == source.parent:
            raise PermissionError("simulated read-only source directory")
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        simulation_utils.tempfile,
        "NamedTemporaryFile",
        reject_read_only_source_directory,
    )
    expected = read_source_scp(source, return_metadata=True)

    assert not adjacent_cache.exists()
    assert fallback_cache.is_file()

    real_load = simulation_utils._manifest_cache_raw_load
    invalid_adjacent = dict(real_load(fallback_cache))
    invalid_adjacent[invalid_field] = invalid_value
    load_paths: list[Path] = []

    def load_adjacent_then_fallback(cache_path):
        load_paths.append(cache_path)
        if cache_path == adjacent_cache:
            return invalid_adjacent
        return real_load(cache_path)

    monkeypatch.setattr(simulation_utils, "_manifest_cache_raw_load", load_adjacent_then_fallback)

    def fail_if_parsed(*args, **kwargs):
        raise AssertionError("valid fallback cache should avoid parsing")

    monkeypatch.setattr(simulation_utils, "_read_source_manifest_impl", fail_if_parsed)

    assert read_source_scp(source, return_metadata=True) == expected
    assert load_paths == [adjacent_cache, fallback_cache]


def test_manifest_file_identity_rejects_path_replacement_during_same_fd_read(
    tmp_path, monkeypatch
):
    """digest 读取期间 path 换 inode 时，即使旧 FD 稳定也必须 fail-closed。"""
    from core.data import simulation_utils

    source = tmp_path / "manifest.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    source.write_bytes(b'{"uid":"old"}\n')
    replacement.write_bytes(b'{"uid":"new"}\n')
    real_open = Path.open

    class ReplacingReader:
        def __init__(self, handle):
            self._handle = handle
            self._replaced = False

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def fileno(self):
            return self._handle.fileno()

        def read(self, size=-1):
            chunk = self._handle.read(size)
            if chunk and not self._replaced:
                self._replaced = True
                replacement.replace(source)
            return chunk

    def replace_during_source_read(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if path == source and "b" in kwargs.get("mode", args[0] if args else "r"):
            return ReplacingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", replace_during_source_read)

    with pytest.raises(ValueError, match="manifest changed"):
        simulation_utils._manifest_file_identity(source)


def test_read_source_scp_quoted_uid_with_common_rate_token(tmp_path):
    """带引号 uid 含常见采样率 token 时解析结果必须一致。"""
    scp = tmp_path / "sources.scp"
    scp.write_text("'uid 16000' 44100 123 '/tmp/a b.wav'\n", encoding="utf-8")
    by_fs, uids, flat, lengths = read_source_scp(scp)
    assert flat == {"uid 16000": "/tmp/a b.wav"}
    assert set(by_fs) == {44100}
    assert lengths == {"uid 16000": 123}


def test_read_source_scp_quoted_uid_with_non_rate_integer(tmp_path):
    scp = tmp_path / "sources.scp"
    scp.write_text("'rec 2024' 16000 1 /tmp/a.wav\n", encoding="utf-8")
    by_fs, uids, flat, lengths = read_source_scp(scp)
    assert flat == {"rec 2024": "/tmp/a.wav"}
    assert set(by_fs) == {16000}
    assert lengths == {"rec 2024": 1}


def test_read_source_scp_resolves_relative_paths_against_manifest_dir(
    tmp_path, monkeypatch
):
    """相对音频路径相对 manifest 所在目录解析，与进程 cwd 无关。"""
    root = tmp_path / "lib"
    (root / "audio").mkdir(parents=True)
    manifest = root / "sources.scp"
    manifest.write_text("uid0 16000 32000 audio/a.wav\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # cwd 与 manifest 目录不同
    by_fs, uids, flat, lengths = read_source_scp(manifest)
    assert flat["uid0"] == os.path.normpath(str(root / "audio" / "a.wav"))
    assert lengths == {"uid0": 32000}


def test_read_source_jsonl_resolves_relative_paths_against_manifest_dir(tmp_path):
    manifest = tmp_path / "sub" / "noise.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        '{"uid":"u","fs":16000,"length":10,"path":"rel/x.wav"}\n', encoding="utf-8"
    )
    by_fs, uids, flat, lengths = read_source_scp(manifest)
    assert flat["u"] == os.path.normpath(str(manifest.parent / "rel" / "x.wav"))


def test_read_source_scp_keeps_absolute_and_tilde_paths(tmp_path):
    scp = tmp_path / "sources.scp"
    scp.write_text(
        "u1 16000 10 /abs/x.wav\nu2 16000 10 ~/rel/y.wav\n", encoding="utf-8"
    )
    by_fs, uids, flat, lengths = read_source_scp(scp)
    assert flat["u1"] == "/abs/x.wav"
    assert flat["u2"] == "~/rel/y.wav"


pytestmark = [pytest.mark.unit]
