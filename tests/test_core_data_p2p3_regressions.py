"""core/data P2/P3 审计修复的回归覆盖（时长过滤哨兵、validation 继承、RIR 元信息回落）。"""

from __future__ import annotations

import logging

import pytest

from core.data.dataset_build import (
    apply_duration_filter_config,
    merge_validation_section,
)


@pytest.mark.unit
def test_duration_filter_rejects_explicit_false_alongside_range_keys():
    section = {"filter_by_duration": False, "filter_max_duration": 30.0}

    with pytest.raises(ValueError, match="filter_by_duration=false conflicts"):
        apply_duration_filter_config(section)

    assert section["filter_by_duration"] is False


@pytest.mark.unit
def test_duration_filter_rejects_non_bool_flag_instead_of_rewriting_it():
    section = {"filter_by_duration": "false", "filter_min_duration": 1.0}

    with pytest.raises(ValueError, match="filter_by_duration must be a bool"):
        apply_duration_filter_config(section)

    assert section["filter_by_duration"] == "false"


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, 0.0, -1.0])
def test_duration_filter_rejects_non_positive_max_duration(value):
    with pytest.raises(ValueError, match="filter_max_duration must be > 0"):
        apply_duration_filter_config({"filter_max_duration": value})


@pytest.mark.unit
def test_duration_filter_enables_filtering_and_fills_missing_bounds():
    section = {"filter_min_duration": 2.0}

    apply_duration_filter_config(section)

    assert section["filter_by_duration"] is True
    assert section["filter_max_duration"] == 1e6


@pytest.mark.unit
def test_validation_inheritance_warns_about_ignored_section_keys(caplog):
    train_section = {"speech_scp": "train.scp", "segment_seconds": 3.0}
    validation_section = {"speech_scp": "val.scp", "segment_seconds": 5.0}

    with caplog.at_level(logging.WARNING, logger="core.data.dataset_build"):
        merged = merge_validation_section(train_section, validation_section)

    assert merged["speech_scp"] == "val.scp"
    assert merged["segment_seconds"] == 3.0
    assert "segment_seconds" in caplog.text


@pytest.mark.unit
def test_validation_inheritance_does_not_warn_for_builder_only_keys(caplog):
    train_section = {"speech_scp": "train.scp"}
    validation_section = {"num_per_epoch": 8, "strict_load": True, "speech_scp": "val.scp"}

    with caplog.at_level(logging.WARNING, logger="core.data.dataset_build"):
        merged = merge_validation_section(train_section, validation_section)

    assert merged["speech_scp"] == "val.scp"
    assert caplog.text == ""


@pytest.mark.unit
def test_validation_inheritance_disabled_applies_every_explicit_key(caplog):
    train_section = {"speech_scp": "train.scp", "segment_seconds": 3.0}
    validation_section = {
        "inherit_train_simulation": False,
        "segment_seconds": 5.0,
    }

    with caplog.at_level(logging.WARNING, logger="core.data.dataset_build"):
        merged = merge_validation_section(train_section, validation_section)

    assert merged["segment_seconds"] == 5.0
    assert caplog.text == ""


@pytest.mark.unit
def test_rir_meta_missing_file_returns_none_without_warning(tmp_path, caplog):
    from core.data.simulation_audio_utils import _read_rir_meta_segment_sec

    missing = tmp_path / "missing.pkl"
    _read_rir_meta_segment_sec.cache_clear()

    with caplog.at_level(logging.WARNING):
        assert _read_rir_meta_segment_sec(str(missing)) is None

    assert caplog.text == ""


@pytest.mark.unit
def test_rir_meta_corrupt_pickle_warns_and_falls_back(tmp_path, caplog):
    from core.data.simulation_audio_utils import _read_rir_meta_segment_sec

    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle payload")
    _read_rir_meta_segment_sec.cache_clear()

    try:
        with caplog.at_level(logging.WARNING):
            assert _read_rir_meta_segment_sec(str(corrupt)) is None

        assert "falling back to cfg.segment_sec" in caplog.text
    finally:
        _read_rir_meta_segment_sec.cache_clear()


@pytest.mark.unit
def test_output_set_lock_refuses_non_posix_platform(monkeypatch, tmp_path):
    from core.data import simulation_manifest_transaction

    monkeypatch.setattr(simulation_manifest_transaction.os, "name", "nt")

    with pytest.raises(RuntimeError, match="requires POSIX fcntl.flock"):
        with simulation_manifest_transaction.exclusive_output_set_lock(
            [tmp_path / "output.jsonl"]
        ):
            pass
