"""AEC 验证器聚合分母、canonical 配置与最终输出一致性合同。"""

import json

import pytest

from test_aec_echo_clipping_integration import aec_paths, echo_clip_dataset
from tools.verify.aec import verify_aec_echo_clipping as verifier

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _row(applied=False, clipped=None, total=None):
    return {
        "echo_applied": applied, "farend_applied": False, "any_applied": applied,
        "echo_rail_hit": None if total is None else bool(clipped),
        "farend_rail_hit": None,
        "rail_hit": None if total is None else bool(clipped),
        "clipped_sample_count": clipped, "rail_sample_count": total,
    }


def test_rail_denominator_is_observed_samples_and_sample_ratio_is_weighted():
    result = verifier._summarize_clipping([
        _row(), _row(True), _row(True, 1, 10), _row(True, 9, 90), _row(False, 0, 100),
    ])
    assert result["any_applied"] == {"count": 3, "observed": 5, "rate": .6}
    assert result["rail_hit"] == {"count": 2, "observed": 3, "rate": 2 / 3}
    assert result["absolute_rail_samples"] == {"clipped": 10, "total": 200, "ratio": .05}
    assert verifier._summarize_clipping([_row()])["rail_hit"]["rate"] is None


@pytest.mark.parametrize("clipped,total", [(-1, 10), (11, 10), (True, 10), (1, 1.5)])
def test_invalid_sample_counts_fail(clipped, total):
    with pytest.raises(ValueError, match="sample counts"):
        verifier._summarize_clipping([_row(True, clipped, total)])


def test_missing_applied_facts_fail():
    with pytest.raises(KeyError):
        verifier._summarize_clipping([{}])
    with pytest.raises(ValueError, match="missing clipping boolean fact"):
        verifier._summarize_clipping([_row(None)])
    with pytest.raises(ValueError, match="both present or both absent"):
        verifier._summarize_clipping([_row(True, 1, None)])


@pytest.mark.parametrize("path", ["post_ser_distortion", "pre_ser_overload", "synced_ref_clip"])
def test_report_uses_real_trace_and_disk_consistency(echo_clip_dataset, monkeypatch, tmp_path, path):
    def build(_cfg_path, *, echo_clip_enabled):
        echo_clip_dataset._aec_echo_clipping_enabled = echo_clip_enabled
        echo_clip_dataset._aec_echo_clipping_path_probs = {path: 1.0}
        echo_clip_dataset.state_detector_label_type = "fst_dt_nst"
        echo_clip_dataset.state_detector_hop_length = 160
        echo_clip_dataset.state_detector_win_length = 320
        return echo_clip_dataset

    monkeypatch.setattr(verifier, "_build_dataset", build)
    verifier.run("unused", tmp_path, 1)
    report = json.loads((tmp_path / "report.json").read_text())
    for row in report.values():
        assert row["aec_disk_consistency"]["checked_samples"] == 1
        assert set(row["aec_disk_consistency"]["passed"].values()) == {1}
    assert report["echo_clipping_off"]["clipping"]["any_applied"]["count"] == 0
    assert report["echo_clipping_on"]["path_trigger_counts"] == {path: 1}


def test_builder_resolves_canonical_config(monkeypatch):
    from types import SimpleNamespace

    cfg = {"data": {"train": {"simulation_cfg": {
        "aec_echo_clipping": {"enabled": False},
    }}}, "stft": {"n_fft": 320}, "meta": {"seed": 12}}
    seen = {}
    monkeypatch.setattr(verifier, "load_merged_config", lambda path: cfg)

    def build(config, split, *, stft_cfg):
        seen.update(config=config, split=split, stft=stft_cfg)
        return SimpleNamespace(set_epoch=lambda epoch, base_seed: seen.update(seed=base_seed))

    monkeypatch.setattr(verifier, "build_dynamic_aec_dataset", build)
    verifier._build_dataset("config.yaml", echo_clip_enabled=True)
    sim = seen["config"]["data"]["train"]["simulation_cfg"]
    assert sim["prob_target_echo"] == 1.0
    assert sum(value for key, value in sim.items() if key.startswith("prob_")) == 1.0
    assert sim["aec_echo_clipping"]["enabled"] is True
    assert seen["seed"] == 12
    assert seen["stft"] == {"n_fft": 320}
