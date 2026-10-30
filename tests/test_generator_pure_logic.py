"""Tests for pure-function logic of generate_simulated_farend_echo_pairs / generate_dns3_offline_dataset / prepare_testset_scps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data.simulation_trace import SimulationTrace, TracedSample  # noqa: E402
from tools.prepare_data_tools.generators.se import generate_dns3_offline_dataset  # noqa: E402
from tools.prepare_data_tools.generators.se._dns3_dataset_generation import (  # noqa: E402
    DNS3GenerationConfig,
    generate_split,
)
from tools.prepare_data_tools.generators.aec import generate_simulated_testset as aec_sim  # noqa: E402
from tools.prepare_data_tools.generators.aec.generate_simulated_testset import (  # noqa: E402
    GENERATOR_ID as AEC_SIM_GENERATOR_ID,
    GENERATOR_VERSION as AEC_SIM_GENERATOR_VERSION,
    _build_sample_metadata as build_aec_sim_metadata,
    _build_seeded_dataset as build_seeded_aec_sim_dataset,
    _export_aec_sample as export_aec_sim_sample,
    _source_fingerprint as aec_sim_source_fingerprint,
    _summarize_audit_metadata as summarize_aec_sim_audit,
    _unpack_aec_sample as unpack_aec_sim_sample,
    parse_args as parse_aec_sim_args,
)
from tools.prepare_data_tools.generation_manifest import (  # noqa: E402
    GENERATION_ROOT_MARKER,
    load_generation_manifest,
)
from tools.prepare_data_tools.generators.echo.generate_simulated_farend_echo_pairs import (  # noqa: E402
    DEFAULT_MANIFEST_NAME,
    DEFAULT_METADATA_NAME,
    _iter_bounded_futures,
    _iter_pair_seeds,
    _load_audio,
    _validate_generation_config,
)
from tools.prepare_data_tools.generators.se import generate_offline_dataset  # noqa: E402
from tools.prepare_data_tools.offline_publication import (  # noqa: E402
    OfflinePublicationError,
    write_ownership_marker,
)
from tools.prepare_data_tools.scp.prepare_testset_scps import (  # noqa: E402
    _prepare_aec_sim_scps,
)
from tools.verify.aec import deep_check_aec_simulation  # noqa: E402
from tools.verify.aec import verify_aec_simulation_audio  # noqa: E402


# ---------------------------------------------------------------------------
# canonical generator CLI / identity 契约
# ---------------------------------------------------------------------------

def test_aec_sim_canonical_cli_and_current_identity():
    args = parse_aec_sim_args([
        "-C", "config.yaml",
        "--output-dir", "out",
        "--num-samples", "3",
    ])
    assert args.num_samples == 3
    assert AEC_SIM_GENERATOR_ID == "generate_simulated_testset.py"
    assert tuple(map(int, AEC_SIM_GENERATOR_VERSION.split("."))) >= (2, 18)
    assert aec_sim.MANIFEST_SCHEMA_VERSION == 2


def test_aec_sim_cli_help_has_no_generation_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())
    with pytest.raises(SystemExit) as exc_info:
        parse_aec_sim_args(["--help"])
    assert exc_info.value.code == 0
    assert "--output-dir" in capsys.readouterr().out
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("tuple_length", [5, 6, 7, 10, 11, 12, 13])
def test_aec_sim_tuple_consumes_first_five_items(tuple_length: int):
    nearend = np.arange(8, dtype=np.float32)
    mic = nearend + 10
    farend = nearend + 20
    sample = (nearend, mic, farend, np.int64(24000), np.int32(6))
    sample += tuple(object() for _ in range(tuple_length - 5))

    unpacked = unpack_aec_sim_sample(sample)

    np.testing.assert_array_equal(unpacked[0], nearend[:6])
    np.testing.assert_array_equal(unpacked[1], mic[:6])
    np.testing.assert_array_equal(unpacked[2], farend[:6])
    assert unpacked[3:] == (24000, 6)


@pytest.mark.parametrize("tuple_length", [0, 1, 2, 3, 4, 8, 9, 14])
def test_aec_sim_tuple_rejects_unsupported_lengths(tuple_length: int):
    with pytest.raises(ValueError, match="5, 6, 7, 10, 11, 12, or 13"):
        unpack_aec_sim_sample(tuple(range(tuple_length)))


def _complete_aec_publication_stems(
    branch: str,
    index: int = 0,
    length: int = 4,
) -> dict:
    has_echo = branch in {
        "prob_echo_only",
        "prob_noise_echo_only",
        "prob_target_echo",
        "prob_target_noise_echo",
    }
    has_noise = branch in {
        "prob_noise_only",
        "prob_noise_echo_only",
        "prob_target_noise",
        "prob_target_noise_echo",
    }
    stems = _aec_audit_stems(
        branch=branch,
        ser_applicable=branch in {"prob_target_echo", "prob_target_noise_echo"},
        snr_applicable=branch in {"prob_target_noise", "prob_target_noise_echo"},
    )
    stems["simulation_by_layer"]["B"].update(
        {"force_branch": branch, "dataset_index": index}
    )
    stems["realized"] = {"branch": branch}
    stems["realized_final"]["branch"] = branch
    stems.update(
        {
            "layer_d_noisy": np.full(length, 0.3, dtype=np.float32),
            "layer_d_clean": np.full(length, 0.2, dtype=np.float32),
            "layer_d_noise": np.full(
                length, 0.05 if has_noise else 0.0, dtype=np.float32
            ),
            "layer_d_echo": np.full(
                length, 0.05 if has_echo else 0.0, dtype=np.float32
            ),
            "farend_ref": np.full(
                length, 0.1 if has_echo else 0.0, dtype=np.float32
            ),
            "final_noise": np.full(
                length, 0.05 if has_noise else 0.0, dtype=np.float32
            ),
            "final_clean": np.full(length, 0.2, dtype=np.float32),
            "final_noisy": np.full(length, 0.3, dtype=np.float32),
        }
    )
    if has_echo:
        stems["final_echo"] = np.full(length, 0.05, dtype=np.float32)
    return stems


@pytest.mark.unit
def test_aec_sim_dataset_seed_forces_publication_safety_on_a_copy():
    calls: list[tuple[int, int]] = []
    constructed: list[tuple[str, bool, bool]] = []

    class StubDataset:
        def __init__(
            self,
            *,
            marker: str,
            strict_load: bool,
            export_simulation_stems: bool,
        ):
            self.marker = marker
            constructed.append((marker, strict_load, export_simulation_stems))

        def set_epoch(self, epoch: int, *, base_seed: int) -> None:
            calls.append((epoch, base_seed))

    source_kwargs = {
        "marker": "first",
        "strict_load": False,
        "export_simulation_stems": False,
    }
    first = build_seeded_aec_sim_dataset(StubDataset, source_kwargs, base_seed=17)
    second = build_seeded_aec_sim_dataset(
        StubDataset, {"marker": "second", "strict_load": False}, base_seed=17
    )
    third = build_seeded_aec_sim_dataset(
        StubDataset, {"marker": "third", "strict_load": True}, base_seed=29
    )

    assert [first.marker, second.marker, third.marker] == ["first", "second", "third"]
    assert calls == [(0, 17), (0, 17), (0, 29)]
    assert constructed == [
        ("first", True, True),
        ("second", True, True),
        ("third", True, True),
    ]
    assert source_kwargs["strict_load"] is False
    assert source_kwargs["export_simulation_stems"] is False


@pytest.mark.unit
def test_aec_sim_reuses_seeded_dataset_and_forces_each_scenario_deterministically():
    def run(seed: int):
        constructed: list[tuple[str, bool, bool]] = []
        epoch_calls: list[tuple[int, int]] = []
        forced_calls: list[tuple[int, str | None]] = []

        class SpyDataset:
            def __init__(
                self,
                *,
                marker: str,
                strict_load: bool,
                export_simulation_stems: bool,
            ):
                constructed.append((marker, strict_load, export_simulation_stems))
                self._last_simulation_stems = None
                self._last_run_branch = None

            def __len__(self) -> int:
                return 11

            def set_epoch(self, epoch: int, *, base_seed: int) -> None:
                epoch_calls.append((epoch, base_seed))

            def __getitem__(self, index: int):
                raise AssertionError(f"ordinary __getitem__ must not be called: {index}")

            def get_sample_with_trace(
                self,
                index: int,
                *,
                force_branch: str | None = None,
                capture_stems: bool | None = None,
            ):
                forced_calls.append((index, force_branch))
                assert force_branch is not None
                assert capture_stems is True
                trace = SimulationTrace(
                    index=index,
                    uid=str(index),
                    task="aec",
                    capture_stems=True,
                )
                trace.replace_metadata("branch", force_branch)
                trace.source("scene").ingest_legacy_mapping(
                    _complete_aec_publication_stems(force_branch, index)
                )
                trace.seal()
                waveform = np.full(4, index + 1, dtype=np.float32)
                sample = (
                    waveform,
                    waveform,
                    waveform,
                    np.int64(16000),
                    np.int32(4),
                )
                return TracedSample(sample=sample, trace=trace)

        dataset = build_seeded_aec_sim_dataset(
            SpyDataset, {"marker": "shared"}, base_seed=seed
        )
        records: dict[str, list[tuple[int, int, str | None]]] = {}
        for scenario_name in aec_sim.SCENARIOS:
            records[scenario_name] = [
                (
                    index,
                    int(sample[0][0]),
                    stems["simulation_by_layer"]["B"]["branch"],
                )
                for index, sample, stems in aec_sim._iter_forced_scenario_samples(
                    dataset,
                    scenario_name,
                    num_samples=3,
                    base_seed=seed,
                )
            ]

        assert constructed == [("shared", True, True)]
        assert epoch_calls == [(0, seed)]
        assert [branch for _index, branch in forced_calls] == [
            scenario_name
            for scenario_name in aec_sim.SCENARIOS
            for _ in range(3)
        ]
        return records, forced_calls

    first = run(73)
    second = run(73)

    assert first == second
    for scenario_name, records in first[0].items():
        assert all(branch == scenario_name for _index, _sample_index, branch in records)
        assert all(index + 1 == sample_index for index, sample_index, _branch in records)


@pytest.mark.unit
def test_aec_sim_forced_generator_propagates_dataset_failure():
    class FailingDataset:
        def __len__(self) -> int:
            return 1

        def get_sample_with_trace(
            self,
            index: int,
            *,
            force_branch: str | None = None,
            capture_stems: bool | None = None,
        ):
            raise ValueError(f"forced failure: {force_branch}")

    dataset = FailingDataset()
    with pytest.raises(ValueError, match="forced failure: prob_target_echo"):
        list(
            aec_sim._iter_forced_scenario_samples(
                dataset,
                "prob_target_echo",
                num_samples=1,
                base_seed=7,
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_force_branch",
        "mismatched_force_branch",
        "missing_layer_branch",
        "mismatched_layer_branch",
        "missing_post_f_branch",
        "mismatched_post_f_branch",
        "missing_final_branch",
        "mismatched_final_branch",
        "missing_dataset_branch",
        "mismatched_dataset_branch",
    ],
)
def test_aec_sim_forced_sample_rejects_missing_or_mismatched_branch_audit(
    invalid_case: str,
):
    scenario_name = "prob_target_noise_echo"

    class InvalidBranchDataset:
        def __init__(self):
            self._last_simulation_stems = None
            self._last_run_branch = scenario_name

        def __len__(self) -> int:
            return 1

        def _getitem_simulation_verify(self, index: int, force_branch: str | None = None):
            assert force_branch == scenario_name
            self._last_run_branch = scenario_name
            stems = _complete_aec_publication_stems(scenario_name, index)
            layer_b = stems["simulation_by_layer"]["B"]
            if invalid_case == "missing_force_branch":
                layer_b.pop("force_branch")
            elif invalid_case == "mismatched_force_branch":
                layer_b["force_branch"] = "prob_target_only"
            elif invalid_case == "missing_layer_branch":
                layer_b.pop("branch")
            elif invalid_case == "mismatched_layer_branch":
                layer_b["branch"] = "prob_target_only"
            elif invalid_case == "missing_post_f_branch":
                stems["realized"].pop("branch")
            elif invalid_case == "mismatched_post_f_branch":
                stems["realized"]["branch"] = "prob_target_only"
            elif invalid_case == "missing_final_branch":
                stems["realized_final"].pop("branch")
            elif invalid_case == "mismatched_final_branch":
                stems["realized_final"]["branch"] = "prob_target_only"
            elif invalid_case == "missing_dataset_branch":
                self._last_run_branch = None
            elif invalid_case == "mismatched_dataset_branch":
                self._last_run_branch = "prob_target_only"
            self._last_simulation_stems = stems
            waveform = np.ones(4, dtype=np.float32)
            return waveform, waveform, waveform, 16000, 4

    with pytest.raises(RuntimeError, match="branch"):
        list(
            aec_sim._iter_forced_scenario_samples(
                InvalidBranchDataset(),
                scenario_name,
                num_samples=1,
                base_seed=42,
            )
        )


@pytest.mark.unit
def test_aec_sim_forced_sample_rejects_silence_placeholder():
    scenario_name = "prob_target_only"

    class PlaceholderDataset:
        def __init__(self):
            self._last_simulation_stems = None
            self._last_run_branch = scenario_name

        def __len__(self) -> int:
            return 1

        def _getitem_simulation_verify(self, index: int, force_branch: str | None = None):
            self._last_run_branch = scenario_name
            self._last_simulation_stems = _complete_aec_publication_stems(
                scenario_name, index
            )
            silence = np.zeros(4, dtype=np.float32)
            return silence, silence, silence, 16000, 4

    with pytest.raises(RuntimeError, match="静音 placeholder"):
        list(
            aec_sim._iter_forced_scenario_samples(
                PlaceholderDataset(),
                scenario_name,
                num_samples=1,
                base_seed=42,
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing_key", ["simulation_by_layer", "realized", "layer_d_noisy", "realized_final"]
)
def test_aec_sim_forced_sample_rejects_incomplete_stems(missing_key: str):
    scenario_name = "prob_target_noise"

    class IncompleteStemsDataset:
        def __init__(self):
            self._last_simulation_stems = None
            self._last_run_branch = scenario_name

        def __len__(self) -> int:
            return 1

        def _getitem_simulation_verify(self, index: int, force_branch: str | None = None):
            self._last_run_branch = scenario_name
            self._last_simulation_stems = _complete_aec_publication_stems(
                scenario_name, index
            )
            self._last_simulation_stems.pop(missing_key)
            waveform = np.ones(4, dtype=np.float32)
            return waveform, waveform, waveform, 16000, 4

    with pytest.raises(RuntimeError, match="stems"):
        list(
            aec_sim._iter_forced_scenario_samples(
                IncompleteStemsDataset(),
                scenario_name,
                num_samples=1,
                base_seed=42,
            )
        )


@pytest.mark.unit
def test_aec_sim_forced_sample_rejects_flat_layer_d_facts():
    scenario_name = "prob_target_noise"

    class FlatLayerDDataset:
        _last_simulation_stems = None
        _last_run_branch = scenario_name

        def __len__(self) -> int:
            return 1

        def _getitem_simulation_verify(self, index: int, force_branch: str | None = None):
            stems = _complete_aec_publication_stems(scenario_name, index)
            stems["simulation_by_layer"]["D"] = {"applied": False, "scale": 1.0}
            self._last_simulation_stems = stems
            waveform = np.ones(4, dtype=np.float32)
            return waveform, waveform, waveform, 16000, 4

    with pytest.raises(RuntimeError, match="simulation_by_layer.D"):
        list(
            aec_sim._iter_forced_scenario_samples(
                FlatLayerDDataset(),
                scenario_name,
                num_samples=1,
                base_seed=42,
            )
        )


@pytest.mark.unit
def test_aec_sim_forced_sample_rejects_stale_stems():
    scenario_name = "prob_target_noise"

    class StaleStemsDataset:
        def __init__(self):
            self._last_simulation_stems = _complete_aec_publication_stems(
                scenario_name, 99
            )
            self._last_run_branch = scenario_name

        def __len__(self) -> int:
            return 1

        def _getitem_simulation_verify(self, index: int, force_branch: str | None = None):
            waveform = np.ones(4, dtype=np.float32)
            return waveform, waveform, waveform, 16000, 4

    with pytest.raises(RuntimeError, match="stems"):
        list(
            aec_sim._iter_forced_scenario_samples(
                StaleStemsDataset(),
                scenario_name,
                num_samples=1,
                base_seed=42,
            )
        )


@pytest.mark.unit
def test_aec_sim_forced_sample_rejects_legacy_nested_realized_schema():
    scenario_name = "prob_target_noise"

    class LegacyNestedRealizedDataset:
        def __len__(self) -> int:
            return 1

        def _getitem_simulation_verify(self, index: int, force_branch: str | None = None):
            stems = _complete_aec_publication_stems(scenario_name, index)
            stems["simulation_by_layer"]["realized"] = stems.pop("realized")
            waveform = np.ones(4, dtype=np.float32)
            self._last_simulation_stems = stems
            self._last_run_branch = scenario_name
            return waveform, waveform, waveform, 16000, 4

    with pytest.raises(RuntimeError, match="simulation_by_layer.realized.*顶层 realized"):
        list(
            aec_sim._iter_forced_scenario_samples(
                LegacyNestedRealizedDataset(),
                scenario_name,
                num_samples=1,
                base_seed=42,
            )
        )


@pytest.mark.unit
def test_aec_sim_metadata_rejects_missing_layer_branch_without_scenario_fallback():
    stems = _complete_aec_publication_stems("prob_target_only")
    stems["simulation_by_layer"]["B"].pop("branch")

    with pytest.raises(RuntimeError, match="simulation_by_layer.B.branch"):
        build_aec_sim_metadata("prob_target_only", 0, 16000, 4, stems)


@pytest.mark.parametrize("has_echo", [False, True])
def test_aec_sim_export_helper_writes_aligned_float32_wavs_and_metadata(
    tmp_path: Path,
    has_echo: bool,
):
    sample_dir = tmp_path / ("with_echo" if has_echo else "without_echo")
    sample_dir.mkdir()
    length = 5
    nearend = np.linspace(-0.4, 0.4, 8, dtype=np.float64)
    mic = nearend + 0.1
    farend = nearend - 0.1
    sample = (nearend, mic, farend, np.int64(24000), np.int32(length))
    stems = _aec_audit_stems(
        branch="prob_target_noise_echo" if has_echo else "prob_target_noise",
        ser_applicable=has_echo,
    )
    stems["layer_d_noisy"] = np.linspace(-0.2, 0.2, 7, dtype=np.float64)
    if has_echo:
        stems["layer_d_echo"] = np.full(7, 0.025, dtype=np.float64)
        stems["final_echo"] = np.linspace(0.01, 0.07, 7, dtype=np.float64)

    metadata = export_aec_sim_sample(
        "prob_target_noise_echo" if has_echo else "prob_target_noise",
        3,
        sample,
        stems,
        sample_dir,
    )

    expected_waveforms = {
        "mic.wav": mic[:length],
        "nearend.wav": nearend[:length],
        "farend.wav": farend[:length],
        "noisy_se.wav": stems["layer_d_noisy"][:length]
        - (stems["layer_d_echo"][:length] if has_echo else 0.0),
    }
    if has_echo:
        expected_waveforms["echo.wav"] = stems["final_echo"][:length]
    assert {path.name for path in sample_dir.glob("*.wav")} == set(expected_waveforms)
    for wav_name, expected_waveform in expected_waveforms.items():
        waveform, sample_rate = sf.read(sample_dir / wav_name, dtype="float32")
        assert sample_rate == 24000
        assert sf.info(sample_dir / wav_name).subtype == "FLOAT"
        assert waveform.dtype == np.float32
        assert waveform.shape == (length,)
        np.testing.assert_array_equal(
            waveform,
            np.asarray(expected_waveform, dtype=np.float32),
            err_msg=wav_name,
        )
    persisted = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert persisted == metadata
    assert persisted["sample_rate"] == 24000
    assert persisted["length_samples"] == length
    json.dumps(persisted, allow_nan=False)


@pytest.mark.unit
def test_aec_sim_export_rejects_missing_canonical_layer_d_noisy(
    tmp_path: Path,
):
    sample_dir = tmp_path / "missing_layer_d_noisy"
    sample_dir.mkdir()
    waveform = np.linspace(-0.2, 0.2, 6, dtype=np.float32)
    sample = (waveform, waveform, waveform, 16000, 6)
    stems = _aec_audit_stems(branch="prob_target_noise")
    stems["layer_d_clean"] = np.full(6, 0.2, dtype=np.float32)
    stems["layer_d_noise"] = np.full(6, 0.05, dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match="canonical Layer-D contract requires layer_d_noisy ndarray",
    ):
        export_aec_sim_sample(
            "prob_target_noise",
            0,
            sample,
            stems,
            sample_dir,
        )

    assert not (sample_dir / "noisy_se.wav").exists()


def _aec_audit_stems(
    *,
    branch: str = "prob_target_noise_echo",
    decomposable: bool | None = True,
    ser_applicable: bool = True,
    snr_applicable: bool = True,
) -> dict:
    acoustic_audit = {
        "candidate": {"primary_snr_db": 12.0, "wind_snr_db": 18.0},
        "nominal": {"selected_primary_snr_db": 15.0},
        "calibration": {
            "layer_c_aggregate_realized_snr_db": 11.75,
            "colored_noise": {"color": "pink", "nominal_snr_db": 14.0},
        },
        "c0_full_early": {"reason_code": None},
    }
    simulation_by_layer = {
        "A": {"speech_uid": "utt"},
        "B": {"branch": branch},
        "C": {
            "acoustic_audit": acoustic_audit,
            "source_volume_calibration": {
                "target_db": -24.0,
                "target_requested_db": -24.0,
                "reference_db_before": -30.0,
                "realized_after_db": -24.0,
                "gain": 10.0 ** (6.0 / 20.0),
                "gain_clipped": False,
                "status": "applied",
                "reason": None,
                "forced_gain": False,
                "reference": "early_rir",
                "stage": "post_rir_pre_noise_echo",
            },
            "c3_ser_db_drawn": 6.0,
            "c1_c2_snr_primary_drawn_db": 12.0,
            "custom_future_field": {"kept": True},
        },
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
                    "peak": 0.3,
                    "scale": 1.0,
                    "applied": False,
                },
                "farend": {
                    "roles": ["farend"],
                    "threshold": 0.99,
                    "peak": 0.1,
                    "scale": 1.0,
                    "applied": False,
                },
            },
        },
        "E": {"num_augmentations_drawn": 1},
        "F": {"noisy_post_mix_degradations_enabled": False},
    }
    realized = {
        "ser_db": 5.75 if ser_applicable else None,
        "ser_applicable": ser_applicable,
        "ser_reason_code": None if ser_applicable else "scenario_has_no_nearend",
        "snr_db": 11.5 if snr_applicable else None,
        "snr_applicable": snr_applicable,
        "snr_reason_code": None if snr_applicable else "mic_not_decomposable",
        "decomposable": decomposable,
        "decomposable_reason_codes": [] if decomposable else ["layer_e_applied"],
    }
    realized_final = dict(realized)
    return {
        "simulation_by_layer": simulation_by_layer,
        "realized": realized,
        "realized_final": realized_final,
    }


def test_aec_sim_metadata_preserves_canonical_audits_and_legacy_fields():
    stems = _aec_audit_stems()

    metadata = build_aec_sim_metadata("prob_target_noise_echo", 7, 24000, 320, stems)

    assert metadata["simulation_by_layer"] == json.loads(
        json.dumps(stems["simulation_by_layer"])
    )
    assert metadata["acoustic_audit"] == stems["simulation_by_layer"]["C"]["acoustic_audit"]
    assert metadata["source_volume_calibration"] == stems["simulation_by_layer"]["C"][
        "source_volume_calibration"
    ]
    assert metadata["realized"] == stems["realized"]
    assert metadata["realized_final"] == stems["realized_final"]
    assert metadata["ser_db"] == metadata["ser_db_drawn"] == 6.0
    assert metadata["snr_db"] == metadata["snr_db_selected_nominal"] == 15.0
    assert metadata["snr_db_drawn"] == metadata["snr_db_primary_candidate"] == 12.0
    assert metadata["snr_db_wind_candidate"] == 18.0
    assert metadata["ser_applicable"] is True
    assert metadata["ser_db_realized"] == 5.75
    assert metadata["snr_applicable"] is True
    assert metadata["snr_db_realized"] == 11.5
    assert metadata["sample_rate"] == 24000
    assert metadata["length_samples"] == 320


def test_aec_sim_pure_echo_legacy_ser_is_null_but_draw_is_preserved():
    stems = _aec_audit_stems(
        branch="prob_echo_only", ser_applicable=False, snr_applicable=False
    )

    metadata = build_aec_sim_metadata("prob_echo_only", 0, 16000, 160, stems)

    assert metadata["ser_db"] is None
    assert metadata["ser_db_drawn"] == 6.0
    assert metadata["ser_applicable"] is False
    assert metadata["ser_db_realized"] is None
    assert metadata["snr_db"] is None
    assert metadata["snr_db_selected_nominal"] == 15.0
    assert metadata["snr_applicable"] is False


def test_aec_sim_post_f_stems_infer_applicability_from_canonical_branch():
    stems = _aec_audit_stems(branch="prob_target_noise_echo")
    stems["realized"].pop("ser_applicable")
    stems["realized"].pop("snr_applicable")
    stems.pop("realized_final")

    metadata = build_aec_sim_metadata("prob_target_noise_echo", 0, 16000, 160, stems)

    assert metadata["ser_applicable"] is True
    assert metadata["snr_applicable"] is True
    assert metadata["ser_db"] == metadata["ser_db_drawn"] == 6.0
    assert metadata["snr_db"] == metadata["snr_db_selected_nominal"] == 15.0
    assert metadata["snr_db_drawn"] == 12.0
    assert metadata["ser_db_realized"] == 5.75
    assert metadata["snr_db_realized"] == 11.5


def test_aec_generator_and_verifier_accept_same_canonical_layer_d_fixture():
    stems = _aec_audit_stems(branch="prob_target_only")
    layers = stems["simulation_by_layer"]

    metadata = build_aec_sim_metadata("prob_target_only", 0, 16000, 0, stems)
    audit = verify_aec_simulation_audio._observation_clipping_audit(layers)

    assert metadata["simulation_by_layer"]["D"] == json.loads(
        json.dumps(layers["D"])
    )
    assert audit["layer_d_facts"] == layers["D"]


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
def test_aec_sim_metadata_rejects_missing_canonical_layer_d_facts(
    invalid_path, expected_path
):
    stems = _aec_audit_stems(branch="prob_target_only")
    layers = stems["simulation_by_layer"]
    if invalid_path == "D":
        layers.pop("D")
    elif invalid_path in {"normalize", "guard"}:
        layers["D"].pop(invalid_path)
    else:
        layers["D"]["guard"].pop(invalid_path)

    with pytest.raises(RuntimeError, match=expected_path):
        build_aec_sim_metadata("prob_target_only", 0, 16000, 0, stems)


@pytest.mark.parametrize("bypass", ["flat_D", "top_level_normalize", "farend_guard"])
def test_aec_generator_and_verifier_reject_layer_d_bypass_fixtures(bypass):
    stems = _aec_audit_stems(branch="prob_target_only")
    layers = stems["simulation_by_layer"]
    if bypass == "flat_D":
        layers["D"] = dict(layers["D"]["guard"]["mic_related"])
    elif bypass == "top_level_normalize":
        layers["D-normalize"] = dict(layers["D"]["normalize"])
    else:
        layers["aec_context"] = {
            "farend_guard": dict(layers["D"]["guard"]["farend"])
        }

    with pytest.raises(RuntimeError):
        build_aec_sim_metadata("prob_target_only", 0, 16000, 0, stems)
    with pytest.raises(RuntimeError):
        verify_aec_simulation_audio._observation_clipping_audit(layers)


def test_aec_sim_metadata_normalizes_non_finite_audits():
    stems = _aec_audit_stems()
    stems["simulation_by_layer"]["C"]["c3_ser_db_drawn"] = float("nan")
    stems["simulation_by_layer"]["C"]["acoustic_audit"]["candidate"][
        "primary_snr_db"
    ] = float("inf")
    stems["simulation_by_layer"]["C"]["custom_array"] = np.array(
        [np.nan, np.inf, np.float32(1.5)], dtype=np.float32
    )
    stems["simulation_by_layer"]["C"]["source_volume_calibration"][
        "realized_after_db"
    ] = float("nan")
    stems["simulation_by_layer"]["C"]["source_volume_calibration"]["gain"] = float(
        "inf"
    )
    stems["realized"]["ser_db"] = float("-inf")
    stems["realized"]["snr_db"] = float("nan")
    metadata = build_aec_sim_metadata("prob_target_noise_echo", 1, 16000, 32, stems)
    assert metadata["ser_db_drawn"] is None
    assert metadata["snr_db_drawn"] == 12.0  # candidate 非有限时回退到 C 层 canonical drawn 键
    assert metadata["ser_db_realized"] is None
    assert metadata["snr_db_realized"] is None
    assert metadata["simulation_by_layer"]["C"]["custom_array"] == [None, None, 1.5]
    assert metadata["source_volume_calibration"]["realized_after_db"] is None
    assert metadata["source_volume_calibration"]["gain"] is None
    json.dumps(metadata, allow_nan=False)


def test_aec_sim_audit_summary_covers_empty_and_decomposable_false():
    empty = summarize_aec_sim_audit([])
    assert empty["total_samples"] == 0
    assert empty["audit_coverage"]["realized"] == {"count": 0, "fraction": 0.0}
    assert empty["audit_coverage"]["realized_final"] == {"count": 0, "fraction": 0.0}
    assert empty["audit_coverage"]["source_volume_calibration"] == {
        "count": 0,
        "fraction": 0.0,
    }
    assert empty["finite_stats"]["ser_db_drawn"]["count"] == 0
    assert empty["finite_stats"]["source_volume_gain"]["count"] == 0
    assert empty["source_volume_calibration"] == {
        "status_counts": {},
        "reason_counts": {},
        "gain_clipped_counts": {"true": 0, "false": 0, "missing": 0},
    }
    assert empty["color_counts"] == {}

    first = build_aec_sim_metadata(
        "prob_target_noise_echo",
        0,
        16000,
        32,
        _aec_audit_stems(decomposable=False, ser_applicable=False, snr_applicable=False),
    )
    skipped_stems = _aec_audit_stems(branch="prob_target_only")
    skipped_stems["simulation_by_layer"]["C"]["source_volume_calibration"].update(
        {
            "target_db": -24.0,
            "target_requested_db": -24.0,
            "reference_db_before": -240.0,
            "realized_after_db": -240.0,
            "gain": 2.0,
            "gain_clipped": True,
            "status": "skipped",
            "reason": "reference_all_zero",
        }
    )
    second = build_aec_sim_metadata("prob_target_only", 1, 16000, 32, skipped_stems)
    summary = summarize_aec_sim_audit([first, second])
    assert summary["audit_coverage"]["acoustic_audit"] == {"count": 2, "fraction": 1.0}
    assert summary["audit_coverage"]["source_volume_calibration"] == {
        "count": 2,
        "fraction": 1.0,
    }
    assert summary["audit_coverage"]["decomposable"] == {
        "true": 1,
        "false": 1,
        "missing": 0,
    }
    assert summary["finite_stats"]["ser_db_drawn"] == {
        "count": 2,
        "min": 6.0,
        "max": 6.0,
        "mean": 6.0,
    }
    assert summary["finite_stats"]["source_volume_target_requested_db"] == {
        "count": 2,
        "min": -24.0,
        "max": -24.0,
        "mean": -24.0,
    }
    assert summary["finite_stats"]["source_volume_realized_after_db"] == {
        "count": 2,
        "min": -240.0,
        "max": -24.0,
        "mean": -132.0,
    }
    assert summary["finite_stats"]["source_volume_gain"] == {
        "count": 2,
        "min": pytest.approx(10.0 ** (6.0 / 20.0)),
        "max": 2.0,
        "mean": pytest.approx((10.0 ** (6.0 / 20.0) + 2.0) / 2.0),
    }
    assert summary["source_volume_calibration"] == {
        "status_counts": {"applied": 1, "skipped": 1},
        "reason_counts": {"none": 1, "reference_all_zero": 1},
        "gain_clipped_counts": {"true": 1, "false": 1, "missing": 0},
    }
    assert summary["color_counts"] == {"pink": 2}


def test_aec_verify_tools_forward_source_volume_calibration_and_summarize():
    stems = _aec_audit_stems()
    by_layer = stems["simulation_by_layer"]
    expected = by_layer["C"]["source_volume_calibration"]
    expected.update(
        {
            "reference_db_before": -240.0,
            "realized_after_db": -240.0,
            "gain": 2.0,
            "gain_clipped": True,
            "status": "skipped",
            "reason": "reference_all_zero",
        }
    )

    record = deep_check_aec_simulation._canonical_audit_record(
        stems, "prob_target_noise_echo"
    )
    points = verify_aec_simulation_audio._audit_measurement_points(stems, by_layer)

    assert record["source_volume_target_requested_db"] == -24.0
    assert record["source_volume_realized_after_db"] == -240.0
    assert record["source_volume_gain"] == pytest.approx(2.0)
    assert record["source_volume_gain_clipped"] is True
    assert record["source_volume_status"] == "skipped"
    assert record["source_volume_reason"] == "reference_all_zero"
    assert points["source_volume_calibration"] == expected

    summary = deep_check_aec_simulation.summarize([record, {"branch": "prob_target_only"}])
    assert summary["metrics"]["source_volume_gain"]["finite"] == 1
    assert summary["source_volume_status_counts"] == {"missing": 1, "skipped": 1}
    assert summary["source_volume_reason_counts"] == {
        "none": 1,
        "reference_all_zero": 1,
    }
    assert summary["source_volume_gain_clipped_counts"] == {
        "missing": 1,
        "true": 1,
    }
    json.dumps(deep_check_aec_simulation._json_finite(summary), allow_nan=False)


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".rir"])
def test_aec_sim_source_fingerprint_resolves_paths_without_hashing_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
):
    source_dir = tmp_path / "source dir"
    source_dir.mkdir()
    audio = source_dir / f"audio file{suffix}"
    audio.write_bytes(b"audio-v1")
    scp = source_dir / "source.scp"
    scp.write_text(f"utt 'audio file{suffix}'\n", encoding="utf-8")
    original = aec_sim._file_fingerprint
    with pytest.raises(ValueError, match="禁止对音频 payload"):
        original(audio)
    fingerprint_calls: list[Path] = []

    def reject_audio_fingerprint(path):
        resolved = Path(path).resolve()
        fingerprint_calls.append(resolved)
        assert resolved.suffix.lower() not in {".wav", ".flac", ".rir"}
        return original(path)

    monkeypatch.setattr(aec_sim, "_file_fingerprint", reject_audio_fingerprint)
    first = aec_sim_source_fingerprint(scp)
    assert first["manifest"]["path"] == str(scp.resolve())
    assert first["entries"] == [str(audio.resolve())]

    audio.write_bytes(b"audio-v2-with-different-payload")
    second = aec_sim_source_fingerprint(scp)
    assert second == first
    assert fingerprint_calls == [scp.resolve()] * 4


def test_aec_sim_source_fingerprint_covers_canonical_stereo_pair_without_payload_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stereo = tmp_path / "stereo.wav"
    stereo.write_bytes(b"stereo")
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps({"pairs": [stereo.name]}),
        encoding="utf-8",
    )
    original = aec_sim._file_fingerprint

    def reject_audio_fingerprint(path):
        assert Path(path).suffix.lower() not in {".wav", ".flac", ".rir"}
        return original(path)

    monkeypatch.setattr(aec_sim, "_file_fingerprint", reject_audio_fingerprint)
    fingerprint = aec_sim_source_fingerprint(manifest)

    assert fingerprint["entries"] == [str(stereo.resolve())]


def _write_aec_sim_structural_config(
    tmp_path: Path,
    *,
    include_wind: bool,
) -> tuple[Path, dict[str, Path]]:
    resources = {
        "speech_scp": tmp_path / "speech.scp",
        "noise_scp": tmp_path / "noise.jsonl",
        "rir_scp": tmp_path / "rir.scp",
        "speech_length_file": tmp_path / "speech.lengths",
        "real_farend_echo_pair_json": tmp_path / "pairs.json",
    }
    for name, path in resources.items():
        path.write_text("{}\n" if name.endswith("json") else "", encoding="utf-8")
    embedded_wind = tmp_path / "embedded-wind.wav"
    embedded_wind.write_bytes(b"payload must not be scanned")
    resources["noise_scp"].write_text(
        json.dumps(
            {
                "uid": "wind-1",
                "fs": 16000,
                "length": 160,
                "path": embedded_wind.name,
                "roles": ["wind"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if include_wind:
        resources["windnoise_scp"] = tmp_path / "wind.scp"
        resources["windnoise_scp"].write_text("", encoding="utf-8")
    config = {
        "data": {
            "mode": "dynamic_aec",
            "sample_rate": 16000,
            "train": {
                **{name: str(path) for name, path in resources.items()},
                "simulation_cfg": {},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, resources


def test_aec_sim_speech_lengths_are_structural_and_wind_is_optional(
    tmp_path: Path,
):
    config_path, resources = _write_aec_sim_structural_config(
        tmp_path,
        include_wind=False,
    )

    assert aec_sim.main(
        [
            "-C",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    ) == 0
    fingerprints = aec_sim._source_fingerprints(
        {name: str(path) for name, path in resources.items()}
    )
    assert "windnoise_scp" not in fingerprints
    assert fingerprints["speech_length_file"]["entries"] == []
    assert fingerprints["speech_length_file"]["manifest"]["path"] == str(
        resources["speech_length_file"].resolve()
    )


def test_aec_sim_non_dry_run_publication_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config_path, _resources = _write_aec_sim_structural_config(
        tmp_path,
        include_wind=True,
    )
    output = tmp_path / "out"

    assert aec_sim.main(["-C", str(config_path), "--output-dir", str(output)]) == 1
    assert "GenerationManifest v1 publication 已下线" in capsys.readouterr().err
    assert not output.exists()


def test_aec_sim_source_fingerprint_rechecks_manifest_after_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    scp = tmp_path / "source.scp"
    scp.write_text("utt audio.wav\n", encoding="utf-8")
    original = aec_sim._file_fingerprint
    manifest_reads = 0

    def mutate_before_second_manifest_read(path):
        nonlocal manifest_reads
        if Path(path).resolve() == scp.resolve():
            manifest_reads += 1
            if manifest_reads == 2:
                scp.write_text("utt audio.wav\n# changed\n", encoding="utf-8")
        return original(path)

    monkeypatch.setattr(aec_sim, "_file_fingerprint", mutate_before_second_manifest_read)
    with pytest.raises(RuntimeError, match="扫描与复核"):
        aec_sim_source_fingerprint(scp)


def test_se_offline_cli_uses_kebab_case_and_warns_for_reserved_workers(monkeypatch):
    args = generate_offline_dataset.parse_args([
        "-C", "config.yaml",
        "--output-dir", "out",
        "--num-samples", "5",
        "--num-workers", "2",
        "--save-debug", "1",
    ])
    assert (args.num_samples, args.num_workers, args.save_debug) == (5, 2, 1)

    for legacy_flag in ("--num_samples", "--num_workers", "--save_debug"):
        with pytest.raises(SystemExit):
            generate_offline_dataset.parse_args([
                "-C", "config.yaml",
                "--output-dir", "out",
                legacy_flag, "1",
            ])

    warnings = []

    class _Logger:
        def warning(self, message):
            warnings.append(message)

        def error(self, _message):
            pass

    monkeypatch.setattr(generate_offline_dataset, "logger", _Logger())
    monkeypatch.setattr(
        generate_offline_dataset.OmegaConf,
        "load",
        lambda _path: {"data": {"mode": "offline"}},
    )
    monkeypatch.setattr(
        generate_offline_dataset.OmegaConf,
        "to_container",
        lambda config, resolve: config,
    )
    rc = generate_offline_dataset.main([
        "-C", "config.yaml",
        "--output-dir", "out",
        "--num-workers", "2",
    ])
    assert rc == 1
    assert warnings == ["--num-workers is reserved; generation currently runs serially"]


@pytest.mark.unit
def test_se_offline_generator_version_is_current_2_10():
    assert tuple(map(int, generate_offline_dataset.GENERATOR_VERSION.split("."))) >= (2, 11)


@pytest.mark.unit
def test_se_offline_main_forces_strict_load_and_binds_cli_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_cfg = {
        "data": {
            "mode": "dynamic",
            "strict_load": False,
            "sample_rate": 16000,
            "train": {"simulation_cfg": {}}
        }
    }
    built_configs = []
    epoch_calls = []

    class StubDataset:
        def __len__(self):
            return 1

        def set_epoch(self, epoch: int, *, base_seed: int):
            epoch_calls.append((epoch, base_seed))

    monkeypatch.setattr(generate_offline_dataset.OmegaConf, "load", lambda _path: source_cfg)
    monkeypatch.setattr(
        generate_offline_dataset.OmegaConf,
        "to_container",
        lambda config, resolve: config,
    )
    import tasks.se.data

    def build_dataset(config, mode):
        built_configs.append(config)
        assert mode == "train"
        return StubDataset()

    monkeypatch.setattr(tasks.se.data, "build_dataset", build_dataset)

    rc = generate_offline_dataset.main(
        [
            "-C",
            "config.yaml",
            "--output-dir",
            str(tmp_path / "offline"),
            "--seed",
            "73",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert epoch_calls == [(0, 73)]
    assert built_configs[0]["strict_load"] is True
    assert built_configs[0]["train"]["simulation_cfg"] == {}
    assert source_cfg["data"]["strict_load"] is False
    assert source_cfg["data"]["train"]["simulation_cfg"] == {}


@pytest.mark.unit
@pytest.mark.parametrize("tuple_length", [4, 7])
def test_se_offline_tuple_compatibility_consumes_first_four_items(tuple_length: int):
    canonical = (object(), object(), 16000, 32)
    sample = canonical + tuple(object() for _ in range(tuple_length - 4))

    unpacked = generate_offline_dataset._unpack_se_sample(sample)

    assert unpacked == canonical


@pytest.mark.unit
@pytest.mark.parametrize("tuple_length", [0, 1, 2, 3, 5, 6, 8])
def test_se_offline_tuple_compatibility_rejects_other_lengths(tuple_length: int):
    with pytest.raises(ValueError, match="4 or 7"):
        generate_offline_dataset._unpack_se_sample(tuple(range(tuple_length)))


@pytest.mark.unit
def test_se_offline_rejects_removed_legacy_five_item_tuple():
    """历史 5 元组（含 is_pure_noise）已移除，必须在 sample 边界显式 fail-fast。"""
    with pytest.raises(ValueError, match="4 or 7"):
        generate_offline_dataset._unpack_se_sample(
            (object(), object(), 16000, 32, False)
        )


def _se_sample(value: float = 0.1):
    wav = np.full(32, value, dtype=np.float32)
    return wav, wav.copy(), 16000, len(wav), False, False, False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (_se_sample(0.0), "silence placeholder"),
        (
            (np.full(4, np.nan), np.ones(4), 16000, 4),
            "finite values",
        ),
        ((np.ones(4), np.ones(4), 16000, 5), "exceeds waveform payload"),
    ],
)
def test_se_offline_publication_rejects_invalid_or_placeholder_samples(
    sample, message
):
    with pytest.raises(ValueError, match=message):
        generate_offline_dataset._validate_publication_sample(sample)


class _CurrentSESampleDataset:
    def __init__(self, values: list[float]):
        self.values = values

    def __getitem__(self, index: int):
        return _se_sample(self.values[index])


@pytest.mark.unit
def test_se_offline_four_column_scps_use_in_memory_metadata_without_audio_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def reject_audio_probe(*_args, **_kwargs):
        raise AssertionError("staging WAV must not be reopened for SCP metadata")

    monkeypatch.setattr(generate_offline_dataset.sf, "info", reject_audio_probe)
    staged = tmp_path / "staged"
    published = tmp_path / "offline"
    _write_se_publication_owner(staged)

    generate_offline_dataset._materialize_dataset(
        _CurrentSESampleDataset([0.1, 0.2]),
        staged=staged,
        published=published,
        indices=[0, 1],
        sample_rate=16000,
        save_debug=0,
        seed=42,
    )

    clean_lines = (staged / "clean.scp").read_text(encoding="utf-8").splitlines()
    noisy_lines = (staged / "noisy.scp").read_text(encoding="utf-8").splitlines()
    assert [line.split(maxsplit=3)[:3] for line in clean_lines] == [
        ["utt_00000000", "16000", "32"],
        ["utt_00000001", "16000", "32"],
    ]
    assert [line.split(maxsplit=3)[:3] for line in noisy_lines] == [
        ["utt_00000000", "16000", "32"],
        ["utt_00000001", "16000", "32"],
    ]
    assert [line.split(maxsplit=3)[3] for line in clean_lines] == [
        str(published / "clean" / f"utt_{index:08d}.wav") for index in range(2)
    ]
    assert [line.split(maxsplit=3)[3] for line in noisy_lines] == [
        str(published / "noisy" / f"utt_{index:08d}.wav") for index in range(2)
    ]
    assert (staged / generate_offline_dataset.RELOAD_CLEAN_SCP).read_text(
        encoding="utf-8"
    ).splitlines() == [
        f"utt_00000000 16000 32 dataset:clean/utt_00000000.wav",
        f"utt_00000001 16000 32 dataset:clean/utt_00000001.wav",
    ]
    assert (staged / generate_offline_dataset.RELOAD_NOISY_SCP).read_text(
        encoding="utf-8"
    ).splitlines() == [
        f"utt_00000000 16000 32 dataset:noisy/utt_00000000.wav",
        f"utt_00000001 16000 32 dataset:noisy/utt_00000001.wav",
    ]
    envelope = load_generation_manifest(
        staged, generate_offline_dataset.ENVELOPE_MANIFEST_NAME
    )
    assert envelope.schema_version == 2
    assert envelope.root_bindings[0].alias == "dataset"
    assert (staged / GENERATION_ROOT_MARKER).is_file()
    assert not (staged / "utt2fs").exists()
    assert not (staged / "speech_length.scp").exists()


def _write_se_publication_owner(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_ownership_marker(
        root, owner_id=generate_offline_dataset.PUBLICATION_OWNER_ID
    )
    generate_offline_dataset._write_owner(root)


def test_se_offline_verified_publication_replaces_collection_and_removes_stale(
    tmp_path: Path,
):
    output = tmp_path / "offline"
    staged_initial = tmp_path / "staged-initial"
    _write_se_publication_owner(staged_initial)
    generate_offline_dataset._materialize_dataset(
        _CurrentSESampleDataset([0.1]),
        staged=staged_initial,
        published=output,
        indices=[0],
        sample_rate=16000,
        save_debug=0,
        seed=42,
    )
    generate_offline_dataset._publish_staged_directory(staged_initial, output)

    (output / "debug_stereo").mkdir()
    (output / "clean" / "utt_99999999.wav").write_bytes(b"stale")
    (output / "debug_stereo" / "old_stereo.wav").write_bytes(b"stale")
    staged_replacement = tmp_path / "staged-replacement"
    _write_se_publication_owner(staged_replacement)
    manifest = generate_offline_dataset._materialize_dataset(
        _CurrentSESampleDataset([0.1, 0.2]),
        staged=staged_replacement,
        published=output,
        indices=[0, 1],
        sample_rate=16000,
        save_debug=0,
        seed=42,
    )

    generate_offline_dataset._publish_staged_directory(staged_replacement, output)

    assert not (output / "debug_stereo").exists()
    assert sorted(path.name for path in (output / "clean").glob("*.wav")) == [
        "utt_00000000.wav",
        "utt_00000001.wav",
    ]
    assert sorted(path.name for path in (output / "noisy").glob("*.wav")) == [
        "utt_00000000.wav",
        "utt_00000001.wav",
    ]
    assert manifest["generator_version"] == generate_offline_dataset.GENERATOR_VERSION
    assert manifest["counts"] == {"requested": 2, "success": 2, "failed": 0}
    assert manifest["uid_semantics"] == "successful_samples_contiguous_zero_based"
    persisted = json.loads(
        (output / generate_offline_dataset.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert persisted == manifest
    assert (output / "clean.scp").read_text(encoding="utf-8").splitlines() == [
        f"utt_00000000 16000 32 {output / 'clean' / 'utt_00000000.wav'}",
        f"utt_00000001 16000 32 {output / 'clean' / 'utt_00000001.wav'}",
    ]
    assert (output / "noisy.scp").read_text(encoding="utf-8").splitlines() == [
        f"utt_00000000 16000 32 {output / 'noisy' / 'utt_00000000.wav'}",
        f"utt_00000001 16000 32 {output / 'noisy' / 'utt_00000001.wav'}",
    ]
    assert (output / generate_offline_dataset.RELOAD_CLEAN_SCP).read_text(
        encoding="utf-8"
    ).splitlines() == [
        "utt_00000000 16000 32 dataset:clean/utt_00000000.wav",
        "utt_00000001 16000 32 dataset:clean/utt_00000001.wav",
    ]
    envelope = load_generation_manifest(
        output, generate_offline_dataset.ENVELOPE_MANIFEST_NAME
    )
    assert envelope.schema_version == 2
    assert envelope.root_bindings[0].alias == "dataset"
    assert (output / GENERATION_ROOT_MARKER).is_file()


def test_se_offline_failure_injection_keeps_previous_publication(tmp_path):
    output = tmp_path / "offline"
    output.mkdir()
    marker = output / "previous-success.txt"
    marker.write_text("keep", encoding="utf-8")
    staged = tmp_path / "staged"

    class FailingDataset:
        def __getitem__(self, index):
            if index == 1:
                raise RuntimeError("injected sample failure")
            return _se_sample()

    with pytest.raises(RuntimeError, match="requested=2, success=1, failed=1"):
        generate_offline_dataset._materialize_dataset(
            FailingDataset(),
            staged=staged,
            published=output,
            indices=[0, 1],
            sample_rate=16000,
            save_debug=0,
            seed=42,
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (output / "clean").exists()


def test_se_offline_main_failure_cleans_owned_staging(tmp_path, monkeypatch):
    output = tmp_path / "offline"
    output.mkdir()
    marker = output / "previous-success.txt"
    marker.write_text("keep", encoding="utf-8")

    class FailingDataset:
        def __len__(self):
            return 2

        def set_epoch(self, epoch: int, *, base_seed: int):
            assert (epoch, base_seed) == (0, 42)

        def __getitem__(self, index):
            if index == 1:
                raise RuntimeError("injected sample failure")
            return _se_sample()

    monkeypatch.setattr(
        generate_offline_dataset.OmegaConf,
        "load",
        lambda _path: {
            "data": {"mode": "dynamic", "dynamic": {}, "sample_rate": 16000}
        },
    )
    monkeypatch.setattr(
        generate_offline_dataset.OmegaConf,
        "to_container",
        lambda config, resolve: config,
    )
    import tasks.se.data
    monkeypatch.setattr(tasks.se.data, "build_dataset", lambda *_args, **_kwargs: FailingDataset())

    rc = generate_offline_dataset.main([
        "-C", "config.yaml",
        "--output-dir", str(output),
        "--num-samples", "2",
    ])

    assert rc == 1
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".offline.owned-staging-*"))


def test_se_offline_verified_publish_failure_restores_previous_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "offline"
    staged_initial = tmp_path / "staged-initial"
    _write_se_publication_owner(staged_initial)
    generate_offline_dataset._materialize_dataset(
        _CurrentSESampleDataset([0.1]),
        staged=staged_initial,
        published=output,
        indices=[0],
        sample_rate=16000,
        save_debug=0,
        seed=42,
    )
    generate_offline_dataset._publish_staged_directory(staged_initial, output)
    marker = output / "previous-success.txt"
    marker.write_text("keep", encoding="utf-8")
    staged = tmp_path / "staged"
    _write_se_publication_owner(staged)
    generate_offline_dataset._materialize_dataset(
        _CurrentSESampleDataset([0.2]),
        staged=staged,
        published=output,
        indices=[0],
        sample_rate=16000,
        save_debug=0,
        seed=42,
    )
    real_replace = generate_offline_dataset.os.replace
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(generate_offline_dataset.os, "replace", fail_second_replace)
    with pytest.raises(OfflinePublicationError, match="injected publish failure"):
        generate_offline_dataset._publish_staged_directory(staged, output)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert staged.exists()
    assert not list(tmp_path.glob(".offline.backup-*"))


def test_se_offline_verified_publication_rejects_unowned_output_and_symlink(
    tmp_path: Path,
):
    staged = tmp_path / "staged"
    _write_se_publication_owner(staged)
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    sentinel = unowned / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unowned output"):
        generate_offline_dataset._publish_staged_directory(staged, unowned)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    referent = tmp_path / "referent"
    referent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(referent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unowned output"):
        generate_offline_dataset._publish_staged_directory(staged, link)
    assert link.is_symlink()
    assert not list(referent.iterdir())


# ---------------------------------------------------------------------------
# generate_simulated_farend_echo_pairs._load_audio：截断语义（曾有 48k 源截短的 bug）
# ---------------------------------------------------------------------------

def test_load_audio_truncates_in_target_sr_domain(tmp_path):
    """max_len 按目标采样率计：48k 源 1s，目标 16k、max_len=16000 -> 输出 16000 样本。"""
    src = tmp_path / "src48k.wav"
    x = np.random.randn(48000).astype(np.float32)
    sf.write(src, x, 48000)
    out = _load_audio(str(src), target_sr=16000, max_len=16000)
    assert out.shape == (16000,)
    assert out.dtype == np.float32


def test_load_audio_short_source_not_padded(tmp_path):
    """源短于 max_len 时不补零，只重采样。"""
    src = tmp_path / "short16k.wav"
    sf.write(src, np.random.randn(8000).astype(np.float32), 16000)
    out = _load_audio(str(src), target_sr=16000, max_len=16000)
    assert out.shape == (8000,)


def test_load_audio_no_max_len_keeps_full(tmp_path):
    src = tmp_path / "src8k.wav"
    sf.write(src, np.random.randn(4000).astype(np.float32), 8000)
    out = _load_audio(str(src), target_sr=16000)
    assert out.shape == (8000,)  # 4000 样本 8k 源重采样到 16k = 8000 样本，不截断


def test_echo_pair_seed_iterator_matches_legacy_vectorized_sequence():
    expected = np.random.default_rng(42).integers(0, 2**31, size=257)
    actual = list(_iter_pair_seeds(42, 257))
    assert [idx for idx, _ in actual] == list(range(257))
    np.testing.assert_array_equal([seed for _, seed in actual], expected)


def test_echo_pair_futures_keep_outstanding_bounded():
    from concurrent.futures import Future

    class FakeExecutor:
        def __init__(self):
            self.submitted = 0

        def submit(self, _fn, idx, seed):
            self.submitted += 1
            future = Future()
            future.set_result((idx, seed))
            return future

    executor = FakeExecutor()
    received = []
    max_observed = 0
    tasks = _iter_pair_seeds(7, 23)
    for idx, future in _iter_bounded_futures(executor, tasks, max_outstanding=5):
        received.append(future.result())
        max_observed = max(max_observed, executor.submitted - len(received))
        assert idx == future.result()[0]

    assert sorted(received) == sorted(_iter_pair_seeds(7, 23))
    assert max_observed <= 5
    assert executor.submitted == 23


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_pairs", 0),
        ("sample_rate", 0),
        ("max_duration_seconds", float("nan")),
        ("workers", -1),
        ("ser_range_db", [0, float("inf")]),
        ("random_delay_ms", [-1, 2]),
    ],
)
def test_echo_generation_config_rejects_invalid_values(field, value):
    config = {
        "num_pairs": 1,
        "sample_rate": 16000,
        "max_duration_seconds": 1.0,
        "workers": 0,
    }
    config[field] = value
    with pytest.raises(ValueError):
        _validate_generation_config(config)


# ---------------------------------------------------------------------------
# generate_dns3_offline_dataset canonical 入口与纯逻辑
# ---------------------------------------------------------------------------


def test_dns3_canonical_entry_accepts_only_kebab_case():
    args = generate_dns3_offline_dataset.parse_args([
        "--data-root", "raw", "--output-dir", "out", "--num-workers", "2"
    ])
    assert (args.data_root, args.output_dir, args.num_workers) == (
        Path("raw"), Path("out"), 2
    )
    with pytest.raises(SystemExit):
        generate_dns3_offline_dataset.parse_args([
            "--data_root", "raw", "--output-dir", "out", "--num_workers", "3"
        ])
    with pytest.raises(SystemExit):
        generate_dns3_offline_dataset.parse_args([
            "--data-root", "raw", "--output-dir", "out", "--scp-four-col"
        ])


def test_dns3_generator_identity_tracks_schema_v2_publication():
    assert generate_dns3_offline_dataset.GENERATOR_VERSION == "2.1"
    # 发布 owner identity 由共享 publication 层持有，改名会使既有输出无法被 --scp-only 复用。
    assert generate_dns3_offline_dataset.PUBLICATION_OWNER_ID == (
        "tasks/se/generate_dns3_offline_dataset"
    )


def test_dns3_verified_main_publishes_schema_v2_alias_relative_scp_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "published wavs"
    scp = tmp_path / "published scp"

    def generation_stub(config: DNS3GenerationConfig):
        clean = config.output_dir / "train_clean" / "1.wav"
        noisy = config.output_dir / "train_noisy" / "1.wav"
        clean.parent.mkdir(parents=True)
        noisy.parent.mkdir(parents=True)
        waveform = np.linspace(-0.2, 0.2, 64, dtype=np.float32)
        sf.write(clean, waveform, 16000, subtype="FLOAT")
        sf.write(noisy, waveform * 0.5, 16000, subtype="FLOAT")

    monkeypatch.setattr(
        generate_dns3_offline_dataset.generation,
        "run_generation",
        generation_stub,
    )
    rc = generate_dns3_offline_dataset.main([
        "--data-root", "unused",
        "--output-dir", str(output),
        "--scp-dir", str(scp),
        "--num-train", "1",
        "--num-val", "0",
        "--num-test", "0",
    ])

    assert rc == 0
    clean_line = (scp / "train" / "clean.scp").read_text(encoding="utf-8").strip()
    noisy_line = (scp / "train" / "noisy.scp").read_text(encoding="utf-8").strip()
    assert clean_line == "1 16000 64 waveforms:train_clean/1.wav"
    assert noisy_line == "1 16000 64 waveforms:train_noisy/1.wav"
    assert str(output) not in clean_line + noisy_line
    assert "owned-staging" not in clean_line + noisy_line


@pytest.fixture()
def dns3_sources(tmp_path):
    def _wav(d: Path, name: str, sr: int = 16000) -> str:
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / name, np.random.randn(sr).astype(np.float32), sr)
        return str(d / name)

    clean = [_wav(tmp_path / "clean", f"c{i}.wav") for i in range(2)]
    noise = [_wav(tmp_path / "noise", f"n{i}.wav") for i in range(2)]
    rir = [_wav(tmp_path / "rir", f"r{i}.wav") for i in range(2)]
    return clean, noise, rir


def test_dns3_split_deterministic(tmp_path, dns3_sources):
    clean, noise, rir = dns3_sources
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    kw = dict(
        split="train",
        num=2,
        clean_list=clean,
        noise_list=noise,
        rir_list=rir,
        fs=16000,
        wav_len=1,
        num_workers=1,
        seed=42,
    )
    generate_split(output_dir=str(out1), **kw)
    generate_split(output_dir=str(out2), **kw)
    wavs1 = sorted((out1 / "train_clean").glob("*.wav"))
    wavs2 = sorted((out2 / "train_clean").glob("*.wav"))
    assert len(wavs1) == 2
    for a, b in zip(wavs1, wavs2):
        assert a.read_bytes() == b.read_bytes()


def test_dns3_split_output_layout(tmp_path, dns3_sources):
    clean, noise, rir = dns3_sources
    out = tmp_path / "out"
    generate_split(
        split="val",
        num=1,
        clean_list=clean,
        noise_list=noise,
        rir_list=rir,
        output_dir=str(out),
        fs=16000,
        wav_len=1,
        num_workers=1,
        seed=7,
    )
    assert (out / "val_clean" / "1.wav").exists()
    assert (out / "val_noisy" / "1.wav").exists()


# ---------------------------------------------------------------------------
# prepare_testset_scps._prepare_aec_sim_scps：缺 wav 时计数不一致告警
# ---------------------------------------------------------------------------

def test_aec_sim_scps_count_mismatch_warns(tmp_path, caplog):
    src = tmp_path / "aec_sim_testset"
    scen = src / "prob_target_only"
    (scen / "sample_000").mkdir(parents=True)
    for w in ("mic", "nearend", "farend", "noisy_se"):
        (scen / "sample_000" / f"{w}.wav").touch()
    # sample_001 缺 farend.wav -> 触发计数不一致告警
    (scen / "sample_001").mkdir()
    for w in ("mic", "nearend", "noisy_se"):
        (scen / "sample_001" / f"{w}.wav").touch()

    dst = tmp_path / "out"
    with caplog.at_level("WARNING", logger="prepare_testset_scps"):
        result = _prepare_aec_sim_scps(src, dst)
    assert any("per-wav entry counts differ" in r.message for r in caplog.records)
    assert len(result) == 4
    # mic.scp 2 条，其余 1 条
    assert len((dst / "aec_sim_testset" / "prob_target_only" / "mic.scp").read_text().splitlines()) == 2
    assert len((dst / "aec_sim_testset" / "prob_target_only" / "farend.scp").read_text().splitlines()) == 1


# ---------------------------------------------------------------------------
# generate_simulated_farend_echo_pairs：固定 canonical 输出名
# ---------------------------------------------------------------------------

def test_echo_ref_output_names_are_fixed_canonical():
    assert DEFAULT_MANIFEST_NAME == "real_farend_echo_pair_manifest.json"
    assert DEFAULT_METADATA_NAME == "real_farend_echo_pair_metadata.json"

pytestmark = [pytest.mark.unit]
