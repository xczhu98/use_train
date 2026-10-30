"""验证 tools/verify 正式入口的 CLI 兼容性。"""

from __future__ import annotations

import pytest

from tools.verify.aec import compare_pfdkf_erle
from tools.verify.aec import deep_check_aec_simulation
from tools.verify.aec import verify_aec_simulation_audio
from tools.verify.aec import verify_linear_aec_backends


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        compare_pfdkf_erle,
        verify_linear_aec_backends,
        verify_aec_simulation_audio,
        deep_check_aec_simulation,
    ],
)
def test_canonical_cli_accepts_explicit_help_argv(
    module, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])
    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "invalid_args",
    [
        pytest.param(["--num-samples", "0"], id="zero-num-samples"),
        pytest.param(["--num-samples", "-1"], id="negative-num-samples"),
        pytest.param(["--index", "-1"], id="negative-index"),
    ],
)
def test_aec_audio_cli_rejects_invalid_scalar_args_before_output_or_dataset(
    tmp_path, monkeypatch, invalid_args
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: {}\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("invalid scalar arguments must fail before dataset construction")

    monkeypatch.setattr(
        verify_aec_simulation_audio,
        "build_dynamic_aec_dataset",
        fail_if_called,
    )

    assert verify_aec_simulation_audio.main(
        ["-C", str(config_path), "-o", str(output_dir), *invalid_args]
    ) == 1
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "index_args",
    [
        pytest.param(["--index", "3"], id="start-index-out-of-range"),
        pytest.param(
            ["--index", "2", "--num-samples", "2"],
            id="sample-range-out-of-range",
        ),
    ],
)
def test_aec_audio_cli_rejects_out_of_range_indices_before_output(
    tmp_path, monkeypatch, index_args
):
    from core.utils import config as config_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text("data: {}\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    class SyntheticDataset:
        def __len__(self):
            return 3

        def set_epoch(self, epoch, *, base_seed):
            assert (epoch, base_seed) == (0, 123)

    monkeypatch.setattr(
        config_module,
        "load_merged_config",
        lambda _path: {"data": {}, "meta": {"seed": 123}},
    )
    monkeypatch.setattr(
        verify_aec_simulation_audio,
        "build_dynamic_aec_dataset",
        lambda _cfg, _split: SyntheticDataset(),
    )

    assert verify_aec_simulation_audio.main(
        ["-C", str(config_path), "-o", str(output_dir), *index_args]
    ) == 1
    assert not output_dir.exists()


def test_linear_backend_cli_fails_when_every_backend_fails(
    tmp_path, monkeypatch, capsys
):
    import numpy as np

    audio = np.zeros(32, dtype=np.float32)
    monkeypatch.setattr(
        verify_linear_aec_backends,
        "load_pair",
        lambda _path, _idx: (audio, audio, 16000),
    )
    monkeypatch.setattr(verify_linear_aec_backends, "run_speex", lambda *_: None)
    monkeypatch.setattr(verify_linear_aec_backends, "run_pfdkf", lambda *_: None)

    assert verify_linear_aec_backends.main(
        ["--pair", "unused.json", "--out_dir", str(tmp_path)]
    ) == 1
    assert "No backend completed successfully" in capsys.readouterr().err


def test_compare_cli_plot_flag_controls_plot_artifact(tmp_path, monkeypatch):
    import json
    import numpy as np

    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": ["unused.wav"]}), encoding="utf-8")
    audio = np.ones(32, dtype=np.float32)
    monkeypatch.setattr(
        compare_pfdkf_erle,
        "load_pair",
        lambda _path, _idx: (audio, audio, 16000),
    )
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_numpy", lambda *_: audio)
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_torch_cpu", lambda *_: audio)
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_torch_gpu", lambda *_: None)
    plotted = []
    monkeypatch.setattr(
        compare_pfdkf_erle,
        "_save_waveform_plot",
        lambda path, *_: plotted.append(path),
    )

    assert compare_pfdkf_erle.main(
        ["--pair", str(manifest), "--out_dir", str(tmp_path / "plain")]
    ) == 0
    assert plotted == []
    assert compare_pfdkf_erle.main(
        [
            "--pair",
            str(manifest),
            "--out_dir",
            str(tmp_path / "plot"),
            "--plot",
        ]
    ) == 0
    assert [path.name for path in plotted] == ["001_waveform_diff.png"]


@pytest.mark.parametrize(
    ("backend_name", "display_name"),
    [
        ("run_speex", "Speex"),
        ("run_pfdkf", "PFDKF"),
    ],
)
@pytest.mark.parametrize(
    "invalid_output",
    [
        pytest.param([], id="empty"),
        pytest.param([1.0] * 31, id="length-mismatch"),
        pytest.param([1.0] * 31 + [float("nan")], id="non-finite"),
    ],
)
def test_linear_backend_cli_rejects_invalid_output_without_wav_or_summary(
    tmp_path, monkeypatch, capsys, backend_name, display_name, invalid_output
):
    import numpy as np

    audio = np.ones(32, dtype=np.float32)
    invalid = np.asarray(invalid_output, dtype=np.float32)
    monkeypatch.setattr(
        verify_linear_aec_backends,
        "load_pair",
        lambda _path, _idx: (audio, audio, 16000),
    )
    monkeypatch.setattr(verify_linear_aec_backends, "run_speex", lambda *_: None)
    monkeypatch.setattr(verify_linear_aec_backends, "run_pfdkf", lambda *_: None)
    monkeypatch.setattr(
        verify_linear_aec_backends, backend_name, lambda *_, **__: invalid
    )
    writes = []
    monkeypatch.setattr(
        verify_linear_aec_backends.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    assert verify_linear_aec_backends.main(
        ["--pair", "unused.json", "--out_dir", str(tmp_path)]
    ) == 1
    captured = capsys.readouterr()
    assert f"{display_name} output" in captured.err
    assert "Summary" not in captured.out
    assert "Output WAVs written" not in captured.out
    assert writes == []


def test_linear_backend_cli_rejects_non_finite_metric_without_wav_or_summary(
    tmp_path, monkeypatch, capsys
):
    import numpy as np

    audio = np.ones(32, dtype=np.float32)
    monkeypatch.setattr(
        verify_linear_aec_backends,
        "load_pair",
        lambda _path, _idx: (audio, audio, 16000),
    )
    monkeypatch.setattr(verify_linear_aec_backends, "run_speex", lambda *_: audio)
    monkeypatch.setattr(verify_linear_aec_backends, "run_pfdkf", lambda *_: None)
    monkeypatch.setattr(
        verify_linear_aec_backends, "erle_db", lambda *_: float("inf")
    )
    writes = []
    monkeypatch.setattr(
        verify_linear_aec_backends.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    assert verify_linear_aec_backends.main(
        ["--pair", "unused.json", "--out_dir", str(tmp_path)]
    ) == 1
    captured = capsys.readouterr()
    assert "Speex metrics must be finite" in captured.err
    assert "Summary" not in captured.out
    assert "Output WAVs written" not in captured.out
    assert writes == []


@pytest.mark.parametrize(
    "backend_name",
    ["run_pfdkf_numpy", "run_pfdkf_torch_cpu", "run_pfdkf_torch_gpu"],
)
@pytest.mark.parametrize(
    "invalid_output",
    [
        pytest.param([], id="empty"),
        pytest.param([1.0] * 31, id="length-mismatch"),
        pytest.param([1.0] * 31 + [float("inf")], id="non-finite"),
    ],
)
def test_compare_cli_rejects_invalid_backend_output_without_wav_or_summary(
    tmp_path, monkeypatch, capsys, backend_name, invalid_output
):
    import json
    import numpy as np

    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": ["unused.wav"]}), encoding="utf-8")
    audio = np.ones(32, dtype=np.float32)
    invalid = np.asarray(invalid_output, dtype=np.float32)
    monkeypatch.setattr(
        compare_pfdkf_erle,
        "load_pair",
        lambda _path, _idx: (audio, audio, 16000),
    )
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_numpy", lambda *_: audio)
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_torch_cpu", lambda *_: audio)
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_torch_gpu", lambda *_: None)
    monkeypatch.setattr(compare_pfdkf_erle, backend_name, lambda *_: invalid)
    writes = []
    monkeypatch.setattr(
        compare_pfdkf_erle.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    assert compare_pfdkf_erle.main(
        ["--pair", str(manifest), "--out_dir", str(tmp_path / "out")]
    ) == 1
    captured = capsys.readouterr()
    assert "output" in captured.err
    assert "ERLE Summary" not in captured.out
    assert "Output files" not in captured.out
    assert writes == []


def test_compare_cli_rejects_non_finite_metric_without_wav_or_summary(
    tmp_path, monkeypatch, capsys
):
    import json
    import numpy as np

    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": ["unused.wav"]}), encoding="utf-8")
    audio = np.ones(32, dtype=np.float32)
    monkeypatch.setattr(
        compare_pfdkf_erle,
        "load_pair",
        lambda _path, _idx: (audio, audio, 16000),
    )
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_numpy", lambda *_: audio)
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_torch_cpu", lambda *_: audio)
    monkeypatch.setattr(compare_pfdkf_erle, "run_pfdkf_torch_gpu", lambda *_: None)
    monkeypatch.setattr(compare_pfdkf_erle, "erle_db", lambda *_: float("nan"))
    writes = []
    monkeypatch.setattr(
        compare_pfdkf_erle.sf,
        "write",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    assert compare_pfdkf_erle.main(
        ["--pair", str(manifest), "--out_dir", str(tmp_path / "out")]
    ) == 1
    captured = capsys.readouterr()
    assert "ERLE" in captured.err
    assert "ERLE Summary" not in captured.out
    assert "Output files" not in captured.out
    assert writes == []


def test_compare_stat_rejects_empty_and_non_finite_values():
    with pytest.raises(ValueError, match="non-empty"):
        compare_pfdkf_erle._stat([])
    with pytest.raises(ValueError, match="finite"):
        compare_pfdkf_erle._stat([1.0, float("nan")])


def _canonical_aec_layer_d(*, applied=False, scale=1.0):
    return {
        "D": {
            "stage_order": ("D-normalize", "D-guard"),
            "normalize": {
                "mode": "joint",
                "target_peak": 0.8,
                "scale": 0.5,
                "preserves_decomposition": True,
            },
            "guard": {
                "mic_related": {
                    "roles": ["mic", "nearend", "echo", "noise"],
                    "threshold": 0.99,
                    "peak": 1.2 if applied else 0.3,
                    "scale": scale,
                    "applied": applied,
                },
                "farend": {
                    "roles": ["farend"],
                    "threshold": 0.99,
                    "peak": 0.2,
                    "scale": 1.0,
                    "applied": False,
                },
            },
        },
    }


def _complete_aec_verify_stems(branch, audio):
    import numpy as np

    echo = audio * 0.5 if "echo" in branch else np.zeros_like(audio)
    farend = audio if "echo" in branch else np.zeros_like(audio)
    stems = {
        "layer_d_noisy": audio * 2,
        "layer_d_clean": audio,
        "layer_d_noise": audio * 0.5,
        "layer_d_echo": echo,
        "farend_ref": farend,
        "final_noise": audio * 0.5,
        "final_clean": audio,
        "final_noisy": audio * 2,
        "simulation_by_layer": {
            "A": {},
            "B": {"force_branch": branch, "branch": branch},
            "C": {},
            **_canonical_aec_layer_d(),
            "E": {},
            "F": {},
        },
        "realized": {"branch": branch},
        "realized_final": {"branch": branch},
        "aec_frontend_facts": {"input_routing": "mic_farend"},
    }
    if "echo" in branch:
        stems["final_echo"] = echo
    return stems


@pytest.mark.parametrize("tuple_length", [5, 6, 7, 10, 11, 12, 13])
def test_aec_verify_tools_consume_only_first_five_tuple_items(tuple_length):
    sample = ("near", "mic", "far", 16000, "lengths") + tuple(
        object() for _ in range(tuple_length - 5)
    )
    expected = sample[:5]
    assert verify_aec_simulation_audio._unpack_dataset_sample(sample) == expected
    assert deep_check_aec_simulation._unpack_dataset_sample(sample) == expected


@pytest.mark.parametrize(
    "module", [verify_aec_simulation_audio, deep_check_aec_simulation]
)
@pytest.mark.parametrize("tuple_length", [4, 8, 9, 14])
def test_aec_verify_tools_reject_unsupported_dataset_tuple(module, tuple_length):
    sample = tuple(object() for _ in range(tuple_length))
    with pytest.raises(RuntimeError, match="5/6/7/10/11/12/13"):
        module._unpack_dataset_sample(sample)


def test_aec_verify_config_copy_uses_train_and_forces_strict_load():
    cfg = {
        "data": {
            "mode": "dynamic_aec",
            "strict_load": False,
            "train": {"speech_scp": "train.scp"},
        }
    }

    copied = deep_check_aec_simulation._prepare_verification_config(cfg)

    assert copied["data"]["strict_load"] is True
    assert copied["data"]["train"] == {
        "speech_scp": "train.scp",
        "export_simulation_stems": True,
    }
    assert cfg["data"]["strict_load"] is False
    assert "export_simulation_stems" not in cfg["data"]["train"]


@pytest.mark.parametrize("tuple_length", [5, 6, 7, 10, 11, 12, 13])
def test_aec_audio_export_uses_supported_forced_branch_adapter(
    tmp_path, monkeypatch, tuple_length
):
    import numpy as np

    audio = np.full(8, 0.1, dtype=np.float32)
    sample = (audio, audio * 2, audio, 16000, 8) + tuple(
        object() for _ in range(tuple_length - 5)
    )
    calls = []

    class SyntheticDataset:
        task = "joint"
        _last_run_branch = "stale"
        _last_simulation_stems = {"stale": True}

        def _getitem_simulation_verify(self, index, force_branch=None):
            assert self._last_run_branch is None
            assert self._last_simulation_stems is None
            calls.append((index, force_branch))
            self._last_run_branch = force_branch
            self._last_simulation_stems = _complete_aec_verify_stems(
                force_branch, audio
            )
            return sample

    monkeypatch.setattr(verify_aec_simulation_audio, "_save_wav", lambda *_: None)
    meta = verify_aec_simulation_audio._export_one_scenario_sample(
        SyntheticDataset(),
        tmp_path / "sample",
        7,
        "prob_target_noise_echo",
        "synthetic",
        False,
        {},
        {},
    )

    assert calls == [(7, "prob_target_noise_echo")]
    assert meta["task"] == "joint"
    assert meta["force_branch"] == "prob_target_noise_echo"
    assert meta["validation"]["scenario_match"] is True
    assert meta["validation"]["branches"] == {
        "requested": "prob_target_noise_echo",
        "dataset_actual": "prob_target_noise_echo",
        "layer_b_force": "prob_target_noise_echo",
        "layer_b_actual": "prob_target_noise_echo",
        "post_f_realized": "prob_target_noise_echo",
        "realized_final": "prob_target_noise_echo",
    }


@pytest.mark.unit
def test_aec_clipping_audit_reports_fixed_rail_and_cooccurrence():
    by_layer = {
        "C": {
            "c3_echo_clipping": {
                "path": "post_ser_distortion",
                "variant": "hard",
                "echo_applied": True,
                "any_applied": True,
            }
        },
        **_canonical_aec_layer_d(applied=True, scale=0.825),
        "E": {
            "selected": ["clipping"],
            "applied": ["clipping"],
            "operations": [
                {
                    "name": "clipping",
                    "applied": True,
                    "actual_changed": True,
                }
            ],
        },
        "F": {
            "cpu_applied_ops": ["fixed_rail"],
            "observation_clipping": {
                "event": True,
                "applied": True,
                "actual_changed": True,
                "mode": "fixed_rail",
                "rail": 0.75,
                "pre_drive": 2.0,
                "pre_drive_db": 6.0206,
                "peak_before": 0.6,
                "peak_after_pre_drive": 1.2,
                "peak_after": 0.75,
                "changed_sample_ratio": 0.875,
                "clipped_sample_count": 4,
                "clipped_sample_ratio": 0.25,
                "clipped_frame_count": 2,
                "clipped_frame_ratio": 0.5,
                "longest_clipped_run_samples": 16,
                # CLI 必须忽略 sidecar 中预先换算的 ms，统一由 sample_rate 推导。
                "longest_clipped_run_ms": 999.0,
            },
        },
    }

    final_guard = {
        "roles": ["mic"],
        "threshold": 0.99,
        "peak": 1.1,
        "scale": 0.9,
        "applied": True,
    }
    audit = verify_aec_simulation_audio._observation_clipping_audit(
        by_layer, sample_rate=16000, final_guard=final_guard
    )

    assert audit["layer_c_echo_clipping"]["actual_changed"] is True
    assert audit["layer_d_facts"] == by_layer["D"]
    assert "layer_d_normalize_facts" not in audit
    assert "farend_guard_facts" not in audit
    assert audit["final_guard"] == {
        "applied": True,
        "scale": pytest.approx(0.9),
    }
    assert audit["layer_e_channel_clipping"]["actual_changed"] is True
    assert audit["layer_f_observation_clipping"] == {
        "event": True,
        "applied": True,
        "actual_changed": True,
        "mode": "fixed_rail",
        "rail": 0.75,
        "pre_drive": 2.0,
        "pre_drive_db": pytest.approx(6.0206),
        "peak_before": 0.6,
        "peak_after_pre_drive": 1.2,
        "peak_after": 0.75,
        "changed_sample_ratio": 0.875,
        "clipped_sample_count": 4,
        "clipped_sample_ratio": 0.25,
        "clipped_frame_count": 2,
        "clipped_frame_ratio": 0.5,
        "longest_clipped_run_samples": 16,
        "longest_clipped_run_ms": 1.0,
    }
    assert audit["cooccurrence"] == {
        "c_and_e_actual": True,
        "c_and_f_actual": True,
        "e_and_f_actual": True,
        "c_e_f_actual": True,
    }


@pytest.mark.unit
def test_aec_clipping_audit_aggregates_event_and_realized_rates():
    first = verify_aec_simulation_audio._observation_clipping_audit(
        {
            "C": {"c3_echo_clipping": {"path": "post_ser_distortion", "applied": True}},
            **_canonical_aec_layer_d(),
            "E": {},
            "F": {
                "clipping": {
                    "event": True,
                    "actual_changed": True,
                    "changed_sample_ratio": 0.9,
                    "clipped_sample_count": 32,
                    "clipped_sample_ratio": 0.2,
                    "clipped_frame_count": 4,
                    "clipped_frame_ratio": 0.4,
                    "longest_clipped_run_samples": 32,
                }
            },
        },
        sample_rate=16000,
    )
    second = verify_aec_simulation_audio._observation_clipping_audit(
        {
            **_canonical_aec_layer_d(applied=True, scale=0.9),
            "E": {},
            "F": {},
        },
        sample_rate=16000,
    )

    summary = verify_aec_simulation_audio._aggregate_clipping_audits(
        [first, second]
    )

    assert summary["num_samples"] == 2
    assert summary["counts"]["c_actual"] == 1
    assert summary["counts"]["d_guard_applied"] == 1
    assert summary["counts"]["f_event"] == 1
    assert summary["counts"]["f_actual"] == 1
    assert summary["counts"]["c_and_f_actual"] == 1
    assert summary["rates"]["f_actual"] == pytest.approx(0.5)
    assert summary["layer_f_realized_means"] == {
        "clipped_sample_count": pytest.approx(32.0),
        "clipped_sample_ratio": pytest.approx(0.2),
        "clipped_frame_count": pytest.approx(4.0),
        "clipped_frame_ratio": pytest.approx(0.4),
        "longest_clipped_run_samples": pytest.approx(32.0),
        "longest_clipped_run_ms": pytest.approx(2.0),
    }


@pytest.mark.unit
def test_aec_clipping_audit_keeps_missing_rail_stats_none_for_legacy_sidecar():
    audit = verify_aec_simulation_audio._observation_clipping_audit(
        {
            **_canonical_aec_layer_d(),
            "F": {
                "observation_clipping": {
                    "event": True,
                    "applied": True,
                    "actual_changed": True,
                    "changed_sample_ratio": 0.75,
                }
            }
        },
        sample_rate=16000,
    )

    facts = audit["layer_f_observation_clipping"]
    assert facts["changed_sample_ratio"] == pytest.approx(0.75)
    assert facts["clipped_sample_count"] is None
    assert facts["clipped_sample_ratio"] is None
    assert facts["clipped_frame_count"] is None
    assert facts["clipped_frame_ratio"] is None
    assert facts["longest_clipped_run_samples"] is None
    assert facts["longest_clipped_run_ms"] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("invalid_path", "expected_path"),
    [
        ("D", "simulation_by_layer.D"),
        ("normalize", "simulation_by_layer.D"),
        ("guard", "simulation_by_layer.D"),
        ("mic_related", "simulation_by_layer.D.guard"),
        ("farend", "simulation_by_layer.D.guard"),
    ],
)
def test_aec_clipping_audit_rejects_missing_canonical_layer_d_facts(
    invalid_path, expected_path
):
    by_layer = _canonical_aec_layer_d()
    if invalid_path == "D":
        by_layer.pop("D")
    elif invalid_path in {"normalize", "guard"}:
        by_layer["D"].pop(invalid_path)
    else:
        by_layer["D"]["guard"].pop(invalid_path)

    with pytest.raises(RuntimeError, match=expected_path):
        verify_aec_simulation_audio._observation_clipping_audit(by_layer)


@pytest.mark.unit
@pytest.mark.parametrize("bypass", ["flat_D", "top_level_normalize", "farend_guard"])
def test_aec_clipping_audit_rejects_layer_d_bypass(bypass):
    by_layer = _canonical_aec_layer_d()
    if bypass == "flat_D":
        by_layer["D"] = dict(by_layer["D"]["guard"]["mic_related"])
    elif bypass == "top_level_normalize":
        by_layer["D-normalize"] = dict(by_layer["D"]["normalize"])
    else:
        by_layer["aec_context"] = {
            "farend_guard": dict(by_layer["D"]["guard"]["farend"])
        }

    with pytest.raises(RuntimeError):
        verify_aec_simulation_audio._observation_clipping_audit(by_layer)


@pytest.mark.parametrize("tuple_length", [5, 6, 7, 10, 11, 12, 13])
def test_aec_deep_check_uses_supported_forced_branch_adapter(tuple_length):
    import numpy as np

    audio = np.full(8, 0.1, dtype=np.float32)
    sample = (audio * 1.5, audio * 2, audio, 16000, 8) + tuple(
        object() for _ in range(tuple_length - 5)
    )
    calls = []

    class SyntheticDataset:
        task = "aec"
        _last_run_branch = "stale"
        _last_simulation_stems = {"stale": True}

        def __len__(self):
            return 1

        def _getitem_simulation_verify(self, index, force_branch=None):
            assert self._last_run_branch is None
            assert self._last_simulation_stems is None
            calls.append((index, force_branch))
            self._last_run_branch = force_branch
            stems = _complete_aec_verify_stems(force_branch, audio)
            stems["layer_d_clean"] = audio * 1.5
            self._last_simulation_stems = stems
            return sample

    records, _ = deep_check_aec_simulation.check_branch(
        SyntheticDataset(), "prob_target_noise_echo", 1, rng_seed=123
    )

    assert calls == [(0, "prob_target_noise_echo")]
    assert records[0]["scenario_match"] is True
    assert records[0]["layer_d_full_rms_snr_db"] == pytest.approx(6.0206, abs=1e-4)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "dataset_branch",
        "layer_b_force",
        "layer_b_actual",
        "post_f_branch",
        "final_branch",
        "missing_schema",
        "missing_stem",
        "non_finite_waveform",
        "non_finite_stem",
        "silent_placeholder",
        "missing_final_echo",
    ],
)
def test_aec_verify_hard_gate_rejects_invalid_forced_sample(invalid_case):
    import numpy as np

    branch = "prob_target_noise_echo"
    audio = np.full(8, 0.1, dtype=np.float32)

    class InvalidDataset:
        _last_run_branch = None
        _last_simulation_stems = None

        def _getitem_simulation_verify(self, index, force_branch=None):
            self._last_run_branch = branch
            stems = _complete_aec_verify_stems(branch, audio)
            sample_audio = audio.copy()
            if invalid_case == "dataset_branch":
                self._last_run_branch = "prob_target_only"
            elif invalid_case == "layer_b_force":
                stems["simulation_by_layer"]["B"]["force_branch"] = "prob_target_only"
            elif invalid_case == "layer_b_actual":
                stems["simulation_by_layer"]["B"]["branch"] = "prob_target_only"
            elif invalid_case == "post_f_branch":
                stems["realized"]["branch"] = "prob_target_only"
            elif invalid_case == "final_branch":
                stems["realized_final"]["branch"] = "prob_target_only"
            elif invalid_case == "missing_schema":
                stems["simulation_by_layer"].pop("D")
            elif invalid_case == "missing_stem":
                stems.pop("layer_d_noisy")
            elif invalid_case == "non_finite_waveform":
                sample_audio[0] = np.nan
            elif invalid_case == "non_finite_stem":
                stems["layer_d_noisy"][0] = np.inf
            elif invalid_case == "silent_placeholder":
                sample_audio[:] = 0.0
            elif invalid_case == "missing_final_echo":
                stems.pop("final_echo")
            self._last_simulation_stems = stems
            return sample_audio, sample_audio, sample_audio, 16000, 8

    with pytest.raises(RuntimeError):
        deep_check_aec_simulation._get_validated_forced_branch_sample(
            InvalidDataset(), 0, branch
        )


@pytest.mark.unit
@pytest.mark.parametrize("unexpected_layer", ["G", "aec_frontend_facts"])
def test_aec_verify_hard_gate_rejects_non_dataset_simulation_layers(
    unexpected_layer,
):
    import numpy as np

    branch = "prob_target_noise_echo"
    audio = np.full(8, 0.1, dtype=np.float32)

    class InvalidDataset:
        _last_run_branch = None
        _last_simulation_stems = None

        def _getitem_simulation_verify(self, index, force_branch=None):
            self._last_run_branch = branch
            stems = _complete_aec_verify_stems(branch, audio)
            stems["simulation_by_layer"][unexpected_layer] = {}
            self._last_simulation_stems = stems
            return audio, audio, audio, 16000, 8

    with pytest.raises(RuntimeError, match="must contain only Layers A-F"):
        deep_check_aec_simulation._get_validated_forced_branch_sample(
            InvalidDataset(), 0, branch
        )


@pytest.mark.unit
def test_aec_verify_accepts_producer_nested_aggregate_facts(tmp_path, monkeypatch):
    """生产端把 realized/final_peak_normalize 嵌在 by_layer 内：硬门禁与导出都必须接受。"""
    import numpy as np

    branch = "prob_target_noise_echo"
    audio = np.full(8, 0.1, dtype=np.float32)

    class NestedDataset:
        task = "joint"
        _last_run_branch = None
        _last_simulation_stems = None

        def _getitem_simulation_verify(self, index, force_branch=None):
            self._last_run_branch = force_branch
            stems = _complete_aec_verify_stems(force_branch, audio)
            layers = stems["simulation_by_layer"]
            layers["realized"] = stems.pop("realized")
            layers["final_peak_normalize"] = {"applied": False, "scale": 1.0}
            self._last_simulation_stems = stems
            return audio, audio, audio, 16000, 8

    deep_check_aec_simulation._get_validated_forced_branch_sample(
        NestedDataset(), 7, branch
    )
    monkeypatch.setattr(verify_aec_simulation_audio, "_save_wav", lambda *_: None)
    meta = verify_aec_simulation_audio._export_one_scenario_sample(
        NestedDataset(), tmp_path / "sample", 7, branch, "synthetic", False, {}, {}
    )
    assert meta["realized"]["branch"] == branch
    assert "realized" not in meta["simulation_by_layer"]


@pytest.mark.unit
def test_aec_verify_keeps_frontend_facts_outside_simulation_layers():
    import numpy as np

    branch = "prob_target_noise_echo"
    audio = np.full(8, 0.1, dtype=np.float32)

    class ValidDataset:
        _last_run_branch = None
        _last_simulation_stems = None

        def _getitem_simulation_verify(self, index, force_branch=None):
            self._last_run_branch = branch
            self._last_simulation_stems = _complete_aec_verify_stems(branch, audio)
            return audio, audio, audio, 16000, 8

    _sample, stems, _branches = deep_check_aec_simulation._get_validated_forced_branch_sample(
        ValidDataset(), 0, branch
    )
    points = verify_aec_simulation_audio._audit_measurement_points(
        stems, stems["simulation_by_layer"]
    )
    assert points["aec_frontend_facts"] == {"input_routing": "mic_farend"}


def test_aec_verify_hard_gate_rejects_stale_final_echo_without_echo_branch():
    import numpy as np

    branch = "prob_target_only"
    audio = np.full(8, 0.1, dtype=np.float32)

    class StaleEchoDataset:
        _last_run_branch = None
        _last_simulation_stems = None

        def _getitem_simulation_verify(self, index, force_branch=None):
            self._last_run_branch = branch
            stems = _complete_aec_verify_stems(branch, audio)
            stems["final_echo"] = audio.copy()
            self._last_simulation_stems = stems
            return audio, audio, np.zeros_like(audio), 16000, 8

    with pytest.raises(RuntimeError, match="stale final_echo"):
        deep_check_aec_simulation._get_validated_forced_branch_sample(
            StaleEchoDataset(), 0, branch
        )


def test_deep_check_summary_reports_audit_coverage_and_counts():
    records = [
        {
            "branch": "prob_target_noise",
            "colored_noise_color": "pink",
            "layer_c_aggregate_snr_db": 9.0,
            "post_f_snr_db": None,
            "c3_erl_range_passed": True,
            "c3_erl_range_reason_code": None,
        },
        {
            "branch": "prob_target_echo",
            "colored_noise_color": None,
            "layer_c_aggregate_snr_db": 11.0,
            "post_f_snr_db": 7.0,
            "c3_erl_range_passed": False,
            "c3_erl_range_reason_code": "erl_effective_range_mismatch",
        },
        {
            "branch": "prob_target_echo",
            "colored_noise_color": None,
            "layer_c_aggregate_snr_db": None,
            "post_f_snr_db": None,
            "c3_erl_range_passed": None,
            "c3_erl_range_reason_code": "legacy_erl_range_audit_unavailable",
        },
    ]

    summary = deep_check_aec_simulation.summarize(records)

    assert summary["total"] == 3
    assert summary["audited"] == 2
    assert summary["none"] == 1
    assert summary["c3_erl_range_gate_status_counts"] == {
        "false": 1,
        "na": 1,
        "true": 1,
    }
    assert summary["c3_erl_range_gate_reason_counts"] == {
        "erl_effective_range_mismatch": 1,
        "legacy_erl_range_audit_unavailable": 1,
        "none": 1,
    }
    assert summary["color_counts"] == {"none": 2, "pink": 1}
    assert summary["branch_counts"] == {"prob_target_echo": 2, "prob_target_noise": 1}
    aggregate = summary["metrics"]["layer_c_aggregate_snr_db"]
    assert aggregate["total"] == 3
    assert aggregate["present"] == 2
    assert aggregate["finite"] == 2
    assert aggregate["invalid"] == 0
    assert aggregate["missing"] == 1
    assert aggregate["audited"] == 2
    assert aggregate["none"] == 1
    assert aggregate["stats"] == {
        "mean": 10.0,
        "min": 9.0,
        "max": 11.0,
        "p5": 9.1,
        "p50": 10.0,
        "p95": 10.9,
    }


def test_deep_check_summary_excludes_non_finite_values_from_stats():
    import json

    summary = deep_check_aec_simulation.summarize(
        [
            {"branch": "echo_only", "layer_c_aggregate_snr_db": float("nan")},
            {"branch": "echo_only", "layer_c_aggregate_snr_db": float("inf")},
            {"branch": "echo_only", "layer_c_aggregate_snr_db": "invalid"},
        ]
    )

    aggregate = summary["metrics"]["layer_c_aggregate_snr_db"]
    assert aggregate == {
        "total": 3,
        "present": 3,
        "finite": 0,
        "invalid": 3,
        "missing": 0,
        "audited": 3,
        "none": 0,
        "stats": None,
    }
    report = deep_check_aec_simulation._json_finite(
        {"records": [float("nan"), float("inf")], "summary": summary}
    )
    assert report["records"] == [None, None]
    json.dumps(report, allow_nan=False)


def test_canonical_audit_record_preserves_realized_none_and_reason():
    import numpy as np

    stems = {
        "layer_d_clean": np.ones(8, dtype=np.float32),
        "layer_d_noise": np.ones(8, dtype=np.float32) * 0.5,
        "layer_d_echo": np.zeros(8, dtype=np.float32),
        "simulation_by_layer": {
            "C": {
                "acoustic_audit": {
                    "calibration": {
                        "layer_c_aggregate_realized_snr_db": 4.5,
                        "layer_c_aggregate_reason_code": None,
                        "colored_noise": {
                            "color": "brown",
                            "calibration_realized_snr_db": 12.0,
                        },
                    },
                    "c0_full_early": {
                        "full_to_early_rms_db": 1.25,
                        "reason_code": None,
                    },
                }
            },
        },
        "realized": {
            "snr_db": None,
            "snr_reason_code": "mic_not_decomposable",
            "decomposable": False,
            "decomposable_reason_codes": ["layer_f_nonlinear"],
            "reconstruction_residual": {
                "rms": None,
                "reason_code": "mic_not_decomposable",
            },
        },
        "aec_frontend_facts": {"input_routing": "mic_farend"},
        "realized_final": {
            "snr_db": None,
            "snr_reason_code": "mic_not_decomposable",
            "decomposable": False,
            "decomposable_reason_codes": ["layer_f_nonlinear"],
            "reconstruction_residual": {
                "rms": None,
                "reason_code": "mic_not_decomposable",
            },
        },
    }

    record = deep_check_aec_simulation._canonical_audit_record(
        stems, "prob_target_noise_echo"
    )

    assert record["layer_d_full_rms_snr_db"] == pytest.approx(6.0206)
    assert record["layer_d_full_rms_ser_db"] is None
    assert record["colored_noise_color"] == "brown"
    assert record["colored_calibration_snr_db"] == 12.0
    assert record["layer_c_aggregate_snr_db"] == 4.5
    assert record["post_f_snr_db"] is None
    assert record["post_f_snr_reason_code"] == "mic_not_decomposable"
    assert record["realized_final_snr_db"] is None
    assert record["realized_final_snr_reason_code"] == "mic_not_decomposable"

    points = verify_aec_simulation_audio._audit_measurement_points(
        stems, stems["simulation_by_layer"]
    )
    assert points["layer_d_full_rms"]["snr_db"] == pytest.approx(6.0206)
    assert points["layer_d_full_rms"]["ser_db"] is None
    assert points["colored_calibration"]["color"] == "brown"
    assert points["layer_c_aggregate"] == {"snr_db": 4.5, "reason_code": None}
    assert points["post_f_realized"]["snr_reason_code"] == "mic_not_decomposable"
    assert points["realized_final"]["snr_reason_code"] == "mic_not_decomposable"
    assert points["aec_frontend_facts"] == {"input_routing": "mic_farend"}


def test_layer_d_full_rms_uses_physical_nearend_for_aec_task():
    import numpy as np

    stems = {
        # task=aec 的 training target 包含透传 noise，物理近端应为 target-noise。
        "layer_d_clean": np.full(8, 1.5, dtype=np.float32),
        "layer_d_noise": np.full(8, 0.5, dtype=np.float32),
        "layer_d_echo": np.full(8, 0.25, dtype=np.float32),
    }

    joint_points = verify_aec_simulation_audio._audit_measurement_points(
        stems, {}, task="joint"
    )
    aec_points = verify_aec_simulation_audio._audit_measurement_points(
        stems, {}, task="aec"
    )
    aec_record = deep_check_aec_simulation._canonical_audit_record(
        stems, "prob_target_noise_echo", task="aec"
    )

    assert joint_points["layer_d_full_rms"]["snr_db"] == pytest.approx(9.5424, abs=1e-4)
    assert joint_points["layer_d_full_rms"]["ser_db"] == pytest.approx(15.5630, abs=1e-4)
    assert aec_points["layer_d_full_rms"]["snr_db"] == pytest.approx(6.0206, abs=1e-4)
    assert aec_points["layer_d_full_rms"]["ser_db"] == pytest.approx(12.0412, abs=1e-4)
    assert aec_record["layer_d_full_rms_snr_db"] == pytest.approx(6.0206, abs=1e-4)
    assert aec_record["layer_d_full_rms_ser_db"] == pytest.approx(12.0412, abs=1e-4)


@pytest.mark.parametrize(
    "metric_key",
    ["snr_reason_code", "ser_reason_code"],
)
def test_layer_d_full_rms_aec_task_is_na_without_physical_noise_stem(metric_key):
    import numpy as np

    stems = {
        "layer_d_clean": np.ones(8, dtype=np.float32),
        "layer_d_echo": np.full(8, 0.25, dtype=np.float32),
    }
    points = verify_aec_simulation_audio._audit_measurement_points(
        stems, {}, task="aec"
    )["layer_d_full_rms"]
    record = deep_check_aec_simulation._canonical_audit_record(
        stems, "prob_target_echo", task="aec"
    )

    metric = metric_key.removesuffix("_reason_code")
    assert points[f"{metric}_db"] is None
    assert points[metric_key] == "aec_physical_nearend_stem_unavailable"
    assert record[f"layer_d_full_rms_{metric}_db"] is None
    assert (
        record[f"layer_d_full_rms_{metric}_reason_code"]
        == "aec_physical_nearend_stem_unavailable"
    )


def _synthetic_deep_check_dataset(
    c3_far_db,
    *,
    actual_farend_db=None,
    echo_clip_path=None,
):
    import numpy as np

    nearend = np.full(8, 0.01, dtype=np.float32)
    echo = np.full(8, 0.005, dtype=np.float32)
    actual_db = c3_far_db if actual_farend_db is None else actual_farend_db
    farend = np.full(8, 10.0 ** (actual_db / 20.0), dtype=np.float32)
    sample = (nearend, nearend + echo, farend, 16000, 8)

    class SyntheticDataset:
        task = "joint"
        _last_run_branch = None
        _last_simulation_stems = None

        def __len__(self):
            return 1

        def _getitem_simulation_verify(self, index, force_branch=None):
            self._last_run_branch = force_branch
            stems = _complete_aec_verify_stems(force_branch, nearend)
            stems.update(
                {
                    "layer_d_noisy": nearend + echo,
                    "layer_d_noise": np.zeros_like(nearend),
                    "layer_d_echo": echo,
                    "farend_ref": farend,
                    "final_echo": echo,
                }
            )
            stems["simulation_by_layer"]["C"].update(
                {
                    "c3_ser_scaling_used": echo_clip_path != "pre_ser_overload",
                    "c3_far_db": c3_far_db,
                    "c3_echo_clipping": {"path": echo_clip_path},
                }
            )
            stems["realized"].update(
                {
                    "erl_applicable": True,
                    "erl_reason_code": None,
                }
            )
            self._last_simulation_stems = stems
            return sample

    return SyntheticDataset()


def test_deep_check_reads_farend_range_from_merged_train_config():
    cfg = {
        "data": {
            "train": {
                "simulation_cfg": {
                    "far_db_range": [-42, -12],
                }
            }
        }
    }

    assert deep_check_aec_simulation._far_db_range_from_merged_config(cfg) == (
        -42.0,
        -12.0,
    )


def test_deep_check_accepts_c3_far_db_inside_configured_range():
    records, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(-35.3),
        "prob_target_echo",
        1,
        far_db_range=(-42.0, -12.0),
    )

    assert anomalies == []
    assert records[0]["c3_far_db_nominal"] == pytest.approx(-35.3)
    assert records[0]["c3_far_db_range_applicable"] is True
    assert records[0]["c3_far_db_range_reason_code"] is None


def test_deep_check_reports_c3_far_db_outside_configured_range():
    _, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(-42.5),
        "prob_target_echo",
        1,
        far_db_range=(-42.0, -12.0),
    )

    assert len(anomalies) == 1
    assert "nominal c3_far_db=-42.5 dB outside configured [-42, -12]" in anomalies[0]


def test_deep_check_synced_clip_does_not_gate_on_lower_actual_farend_rms():
    records, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(
            -35.3,
            actual_farend_db=-48.0,
            echo_clip_path="synced_ref_clip",
        ),
        "prob_target_echo",
        1,
        far_db_range=(-42.0, -12.0),
    )

    assert anomalies == []
    assert records[0]["c3_far_db_range_applicable"] is True
    assert records[0]["layer_d_actual_farend_rms_db"] == pytest.approx(-48.0)
    assert records[0]["layer_d_actual_farend_rms_gate_role"] == "diagnostic_only"
    assert records[0]["farend_ref_rms_db"] == pytest.approx(-48.0)
    assert records[0]["farend_ref_range_gate_basis"] == "c3_far_db_nominal"


def test_deep_check_nominal_farend_gate_ignores_lower_actual_rms():
    records, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(
            -35.3,
            actual_farend_db=-48.0,
        ),
        "prob_target_echo",
        1,
        far_db_range=(-42.0, -12.0),
    )

    assert anomalies == []
    assert records[0]["c3_far_db_range_applicable"] is True
    assert records[0]["layer_d_actual_farend_rms_db"] == pytest.approx(-48.0)
    assert records[0]["farend_ref_range_gate_basis"] == "c3_far_db_nominal"


def test_deep_check_missing_c3_far_db_is_na_without_false_anomaly():
    records, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(None, actual_farend_db=-48.0),
        "prob_target_echo",
        1,
        far_db_range=(-42.0, -12.0),
    )

    assert anomalies == []
    assert records[0]["c3_far_db_nominal"] is None
    assert records[0]["c3_far_db_range_applicable"] is False
    assert records[0]["c3_far_db_range_reason_code"] == "c3_far_db_nominal_unavailable"


def test_deep_check_missing_farend_range_does_not_report_anomaly():
    records, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(-35.3),
        "prob_target_echo",
        1,
        far_db_range=None,
    )

    assert anomalies == []
    assert records[0]["c3_far_db_nominal"] == pytest.approx(-35.3)
    assert records[0]["c3_far_db_expected_range_db"] is None
    assert records[0]["c3_far_db_range_applicable"] is False
    assert records[0]["c3_far_db_range_reason_code"] == "far_db_range_unavailable"


def test_deep_check_pre_ser_overload_skips_farend_range_check():
    records, anomalies = deep_check_aec_simulation.check_branch(
        _synthetic_deep_check_dataset(-6.0, echo_clip_path="pre_ser_overload"),
        "prob_target_echo",
        1,
        far_db_range=(-42.0, -12.0),
    )

    assert anomalies == []
    assert records[0]["c3_far_db_range_applicable"] is False
    assert (
        records[0]["c3_far_db_range_reason_code"]
        == "pre_ser_overload_bypasses_ser_erl_coupling"
    )


def _erl_range_gate_stems(audit):
    return {
        "simulation_by_layer": {
            "C": {
                "c3_ser_scaling_used": True,
                "c3_echo_clipping": {"path": None},
                "c3_erl_range_audit": audit,
            }
        }
    }


def test_deep_check_erl_range_gate_accepts_valid_closed_intersection():
    gate = deep_check_aec_simulation._c3_erl_range_gate(
        _erl_range_gate_stems(
            {
                "configured_range_db": [-20.0, 40.0],
                "physical_range_db": [-10.0, 18.0],
                "effective_range_db": [-10.0, 18.0],
                "selected_erl_db": 18.0,
            }
        )
    )

    assert gate["applicable"] is True
    assert gate["passed"] is True
    assert gate["reason_code"] is None


@pytest.mark.parametrize(
    "audit,reason",
    [
        (
            {
                "configured_range_db": [-20.0, 40.0],
                "physical_range_db": [-10.0, 18.0],
                "effective_range_db": [-20.0, 40.0],
                "selected_erl_db": 0.0,
            },
            "erl_effective_range_mismatch",
        ),
        (
            {
                "configured_range_db": [-20.0, 40.0],
                "physical_range_db": [-10.0, 18.0],
                "effective_range_db": [-10.0, 18.0],
                "selected_erl_db": 19.0,
            },
            "erl_selected_outside_range",
        ),
    ],
)
def test_deep_check_erl_range_gate_rejects_inconsistent_audit(audit, reason):
    gate = deep_check_aec_simulation._c3_erl_range_gate(
        _erl_range_gate_stems(audit)
    )

    assert gate["applicable"] is True
    assert gate["passed"] is False
    assert gate["reason_code"] == reason


def test_deep_check_legacy_erl_metadata_is_honest_na():
    gate = deep_check_aec_simulation._c3_erl_range_gate(
        _erl_range_gate_stems(None)
    )

    assert gate["applicable"] is False
    assert gate["passed"] is None
    assert gate["reason_code"] == "legacy_erl_range_audit_unavailable"


pytestmark = [pytest.mark.unit]
