"""SE 仿真音频 verifier 的 canonical Layer-D stem 合同测试。"""

from __future__ import annotations

import numpy as np
import pytest

from tools.verify import verify_se_simulation_audio


pytestmark = [pytest.mark.unit]


def _synthetic_dataset(stems: dict[str, object]):
    clean = np.full(8, 0.1, dtype=np.float32)
    noisy = np.full(8, 0.2, dtype=np.float32)

    class SyntheticDataset:
        _last_run_branch = None
        _last_simulation_stems = None

        def _getitem_simulation_verify(self, index, force_branch=None):
            assert index == 3
            assert force_branch == "prob_target_noise"
            self._last_run_branch = force_branch
            self._last_simulation_stems = stems
            return clean, noisy, 16000, 8, False, False, False

    return SyntheticDataset()


def _canonical_stems() -> dict[str, object]:
    return {
        "layer_d_noisy": np.full(8, 0.3, dtype=np.float32),
        "layer_d_clean": np.full(8, 0.1, dtype=np.float32),
        "layer_d_noise_additive": np.full(8, 0.2, dtype=np.float32),
        "final_noisy": np.full(8, 0.2, dtype=np.float32),
        "final_clean": np.full(8, 0.1, dtype=np.float32),
        "simulation_by_layer": {
            "B": {"branch": "prob_target_noise", "force_branch": "prob_target_noise"},
            "C": {"need_additive_noise": True},
            "D": {
                "stage_order": ("D-normalize", "D-guard"),
                "normalize": {
                    "mode": "joint",
                    "target_peak": 0.8,
                    "noisy_scale": 0.5,
                    "clean_scale": 0.5,
                    "noise_scale": 0.5,
                    "preserves_decomposition": True,
                },
                "guard": {
                    "roles": ["observation", "target", "additive", "residual"],
                    "threshold": 0.99,
                    "peak": 0.3,
                    "scale": 1.0,
                    "applied": False,
                },
            },
        },
        "final_peak_normalize": {
            "roles": ["observation"],
            "threshold": 0.99,
            "peak": 0.2,
            "scale": 1.0,
            "applied": False,
        },
    }


def test_se_verifier_consumes_canonical_layer_d_stems_with_stable_wav_names(
    tmp_path, monkeypatch
):
    stems = _canonical_stems()
    writes: dict[str, np.ndarray] = {}

    def capture_wav(path, audio, sample_rate):
        assert sample_rate == 16000
        writes[path.name] = np.asarray(audio, dtype=np.float32).copy()

    monkeypatch.setattr(verify_se_simulation_audio, "_save_wav", capture_wav)
    meta = verify_se_simulation_audio._export_one_scenario_sample(
        _synthetic_dataset(stems),
        tmp_path / "sample",
        3,
        "prob_target_noise",
        "synthetic",
        False,
        {},
        {},
    )

    np.testing.assert_array_equal(writes["layer_D_noisy.wav"], stems["layer_d_noisy"])
    np.testing.assert_array_equal(writes["layer_D_clean.wav"], stems["layer_d_clean"])
    np.testing.assert_array_equal(
        writes["layer_D_noise_additive.wav"], stems["layer_d_noise_additive"]
    )
    assert meta["rms"]["noisy_Layer_D"] == pytest.approx(0.3)
    assert meta["rms"]["clean_Layer_D"] == pytest.approx(0.1)
    layer_d = meta["simulation_by_layer"]["D"]
    assert layer_d["stage_order"] == ["D-normalize", "D-guard"]
    assert layer_d["normalize"]["mode"] == "joint"
    assert layer_d["guard"]["applied"] is False
    assert "D-normalize" not in meta["simulation_by_layer"]
    assert meta["simulation_stems_summary"]["layer_d_noise_additive"] == {
        "shape": [8],
        "rms": pytest.approx(0.2),
    }
    canonical_guard_keys = {"roles", "threshold", "peak", "scale", "applied"}
    final_guard = meta["final_peak_normalize"]
    assert set(final_guard) == canonical_guard_keys
    assert final_guard["roles"] == ["observation"]
    assert "peak_before" not in final_guard
    assert "scale_applied" not in final_guard


@pytest.mark.parametrize(
    "missing_key",
    ["layer_d_noisy", "layer_d_clean", "layer_d_noise_additive"],
)
def test_se_verifier_rejects_missing_canonical_layer_d_stem(
    tmp_path, monkeypatch, missing_key
):
    stems = _canonical_stems()
    stems.pop(missing_key)
    writes = []
    monkeypatch.setattr(
        verify_se_simulation_audio,
        "_save_wav",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match=missing_key):
        verify_se_simulation_audio._export_one_scenario_sample(
            _synthetic_dataset(stems),
            tmp_path / "sample",
            3,
            "prob_target_noise",
            "synthetic",
            False,
            {},
            {},
        )

    assert writes == []


@pytest.mark.unit
@pytest.mark.parametrize("unexpected_layer", ["G", "unexpected_fact"])
def test_se_verifier_rejects_non_dataset_simulation_layers(
    tmp_path, monkeypatch, unexpected_layer
):
    stems = _canonical_stems()
    stems["simulation_by_layer"][unexpected_layer] = {}
    with pytest.raises(RuntimeError, match="must contain only Layers A-F"):
        verify_se_simulation_audio._export_one_scenario_sample(
            _synthetic_dataset(stems),
            tmp_path / "sample",
            3,
            "prob_target_noise",
            "synthetic",
            False,
            {},
            {},
        )


@pytest.mark.unit
def test_se_verifier_accepts_producer_nested_aggregate_facts(tmp_path, monkeypatch):
    """生产端把 final_peak_normalize/realized 嵌在 by_layer 内：必须被提升为顶层事实。"""
    stems = _canonical_stems()
    layers = stems["simulation_by_layer"]
    layers["final_peak_normalize"] = stems.pop("final_peak_normalize")
    layers["realized"] = {"mic_dbfs_rms": -20.0, "measure_point": "post_layer_f"}
    monkeypatch.setattr(
        verify_se_simulation_audio, "_save_wav", lambda *args, **kwargs: None
    )

    meta = verify_se_simulation_audio._export_one_scenario_sample(
        _synthetic_dataset(stems),
        tmp_path / "sample",
        3,
        "prob_target_noise",
        "synthetic",
        False,
        {},
        {},
    )

    assert meta["final_peak_normalize"]["applied"] is False
    assert meta["realized"]["measure_point"] == "post_layer_f"
    assert "realized" not in meta["simulation_by_layer"]
    assert set(meta["simulation_by_layer"]) <= {"A", "B", "C", "D", "E", "F"}


@pytest.mark.parametrize("bypass", ["flat_D", "top_level_normalize"])
def test_se_verifier_rejects_layer_d_bypass_before_writing(
    tmp_path, monkeypatch, bypass
):
    stems = _canonical_stems()
    layers = stems["simulation_by_layer"]
    if bypass == "flat_D":
        layers["D"] = dict(layers["D"]["guard"])
    else:
        layers["D-normalize"] = dict(layers["D"]["normalize"])
    writes = []
    monkeypatch.setattr(
        verify_se_simulation_audio,
        "_save_wav",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="simulation_by_layer"):
        verify_se_simulation_audio._export_one_scenario_sample(
            _synthetic_dataset(stems),
            tmp_path / "sample",
            3,
            "prob_target_noise",
            "synthetic",
            False,
            {},
            {},
        )

    assert writes == []
