import pytest

from core.data.dataset_contract import DatasetContract
from core.data.dataset_contract import (
    build_aec_dataset_contract,
    build_se_dataset_contract,
    build_tse_dataset_contract,
)
from core.losses.loss_contract import LossContract
from core.models.execution_plan import resolve_execution_plan
from core.models.io_contract import ModelIOContract, TensorField
from core.models.legacy_contract import resolve_model_io_contract
from tasks.ss.data.config import build_ss_dataset_contract
from tasks.tse.cue_contract import TSECueSpec, TSECueSpecs


def _ss_dataset_contract():
    return DatasetContract(
        task="ss",
        fields=(
            TensorField("mixture", "waveform", ("batch", "time")),
            TensorField("sources", "waveform", ("batch", "source", "time"), multiple=True),
            TensorField("lengths", "label", ("batch",)),
            TensorField("source_mask", "mask", ("batch", "source")),
        ),
        sample_rate=16000,
        cardinality="fixed_multi",
        num_sources=2,
    )


def _pit_contract():
    return LossContract(
        inputs=("est_sources", "sources", "lengths", "source_mask"),
        permutation_invariant=True,
        supports_lengths=True,
        supports_source_mask=True,
    )


@pytest.mark.unit
def test_execution_plan_resolves_ss_spectrum_runtime():
    model_contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
        outputs=(TensorField("est_sources_spectra", "spectrum", ("batch", "source", "frequency", "frame", "complex"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )
    config = {
        "data": {"mode": "offline_ss", "sample_rate": 16000},
        "ss": {"num_sources": 2},
        "stft": {"n_fft": 640, "hop_length": 160, "win_length": 640, "window": "hann", "center": True},
    }
    plan = resolve_execution_plan(
        config,
        model_contract=model_contract,
        dataset_contract=_ss_dataset_contract(),
        loss_contract=_pit_contract(),
    )
    assert plan.task == "ss"
    assert plan.input_domain == "spectrum"
    assert plan.output_domain == "spectrum"
    assert plan.num_sources == 2
    assert plan.stft["n_fft"] == 640


@pytest.mark.unit
def test_trainer_execution_plan_uses_actual_stft_window_name():
    """兼容路径合成 stft 上下文必须取自真实模块的 window_name，而非硬编码 hann。"""
    import torch

    from core.models.execution_plan import resolve_trainer_execution_plan
    from core.stft import StftIstftModule

    class _SpectrumModel(torch.nn.Module):
        input_domain = "spectrum"
        output_domain = "spectrum"
        frontend_owner = "runtime"

        def forward(self, noisy_spectra):
            return noisy_spectra

    plan = resolve_trainer_execution_plan(
        {"data": {"mode": "dynamic", "sample_rate": 16000}},
        _SpectrumModel(),
        task="se",
        sample_rate=16000,
        stft_module=StftIstftModule(
            n_fft=64, hop_length=16, win_length=64, window="hamming"
        ),
    )

    assert plan.stft["window"] == "hamming"


@pytest.mark.unit
def test_execution_plan_rejects_missing_stft_for_spectrum_model():
    contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture_spectra", "spectrum", ("batch", "frequency", "frame", "complex")),),
        outputs=(TensorField("est_sources_spectra", "spectrum", ("batch", "source", "frequency", "frame", "complex"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="runtime",
    )
    with pytest.raises(ValueError, match="STFT"):
        resolve_execution_plan(
            {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}},
            model_contract=contract,
            dataset_contract=_ss_dataset_contract(),
            loss_contract=_pit_contract(),
        )


@pytest.mark.unit
def test_execution_plan_rejects_dataset_source_count_mismatch():
    contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(TensorField("est_sources", "waveform", ("batch", "source", "time"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )
    with pytest.raises(ValueError, match="num_sources"):
        resolve_execution_plan(
            {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 3}},
            model_contract=contract,
            dataset_contract=_ss_dataset_contract(),
            loss_contract=_pit_contract(),
        )


@pytest.mark.unit
def test_execution_plan_requires_ss_num_sources():
    contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(TensorField("est_sources", "waveform", ("batch", "source", "time"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )
    with pytest.raises(ValueError, match="num_sources"):
        resolve_execution_plan(
            {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {}},
            model_contract=contract,
        )


@pytest.mark.unit
def test_execution_plan_requires_complete_pit_loss_capabilities():
    contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(TensorField("est_sources", "waveform", ("batch", "source", "time"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )
    incomplete = LossContract(inputs=("est_sources", "sources"))
    with pytest.raises(ValueError, match="PIT capabilities"):
        resolve_execution_plan(
            {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}},
            model_contract=contract,
            dataset_contract=_ss_dataset_contract(),
            loss_contract=incomplete,
            require_loss_contract=True,
        )


@pytest.mark.unit
def test_execution_plan_requires_explicit_loss_contract_for_ss_training():
    contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(TensorField("est_sources", "waveform", ("batch", "source", "time"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )
    with pytest.raises(ValueError, match="explicit LossContract"):
        resolve_execution_plan(
            {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}},
            model_contract=contract,
            dataset_contract=_ss_dataset_contract(),
            require_loss_contract=True,
        )


@pytest.mark.unit
def test_execution_plan_rejects_model_sample_rate_mismatch():
    contract = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(TensorField("est_sources", "waveform", ("batch", "source", "time"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )
    with pytest.raises(ValueError, match="model sample_rate"):
        resolve_execution_plan(
            {"data": {"mode": "offline_ss", "sample_rate": 16000}, "ss": {"num_sources": 2}},
            model_contract=contract,
            model_sample_rate=8000,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "stft", "expected_task", "expected_domain"),
    [
        ("offline", True, "se", "spectrum"),
        ("offline_aec", True, "aec", "spectrum"),
        ("offline_tse", True, "tse", "spectrum"),
    ],
)
def test_legacy_contract_resolution_is_centralized(mode, stft, expected_task, expected_domain):
    class LegacyModel:
        pass

    cfg = {"data": {"mode": mode}, "stft": {"n_fft": 512}} if stft else {"data": {"mode": mode}}
    contract = resolve_model_io_contract(LegacyModel(), cfg)
    assert expected_task in contract.tasks
    assert contract.inputs[0].domain == expected_domain


@pytest.mark.unit
def test_legacy_ss_contract_defaults_output_domain_to_input_domain():
    class LegacySS:
        input_domain = "waveform"

    contract = resolve_model_io_contract(
        LegacySS(), {"data": {"mode": "offline_ss"}}
    )

    assert contract.outputs[0].domain == "waveform"
    assert contract.outputs[0].name == "est_sources"


@pytest.mark.unit
def test_task_dataset_contracts_declare_canonical_fields_and_cardinality():
    assert build_se_dataset_contract(16000).cardinality == "single"
    assert build_aec_dataset_contract(16000).task == "aec"
    tse = build_tse_dataset_contract(16000, variable_targets=True)
    assert tse.cardinality == "variable_multi"
    assert {field.name for field in tse.fields} == {
        "mixture",
        "target",
        "sr",
        "lengths",
        "num_targets",
        "spk_labels",
        "target_present",
    }


@pytest.mark.unit
def test_tse_dataset_contract_expands_only_enabled_typed_cues():
    specs = TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (16000,)),
        speaker_embedding=TSECueSpec("embedding", (192,)),
        spatial=TSECueSpec("track", (2, 20)),
        visual=TSECueSpec("features", (256, 12)),
        textual=TSECueSpec("phoneme_ids", (32,)),
    )
    contract = build_tse_dataset_contract(16000, cue_specs=specs)
    fields = {field.name: field for field in contract.fields}
    assert fields["speaker_waveform"].axes == ("target", "cue_time")
    assert fields["speaker_embedding"].domain == "embedding"
    assert fields["spatial"].axes == ("target", "direction", "cue_point")
    assert fields["spatial_present"].axes == ("target",)
    assert fields["spatial_lengths"].axes == ("target",)
    assert fields["spatial_timebase"].axes == ("target", "cue_point")
    assert fields["visual_timebase"].axes == ("target", "frame")
    assert fields["textual_lengths"].axes == ("target",)
    assert "enroll" not in fields and "enroll_lengths" not in fields

    static = build_tse_dataset_contract(
        16000,
        cue_specs=TSECueSpecs(spatial=TSECueSpec("static", (2,))),
    )
    static_fields = {field.name: field for field in static.fields}
    assert static_fields["spatial"].axes == ("target", "direction")
    assert "spatial_lengths" not in static_fields
    assert "spatial_timebase" not in static_fields


@pytest.mark.unit
def test_tse_builder_uses_dataset_cue_specs_as_contract_source(monkeypatch):
    import tasks.tse.data as tse_data

    specs = TSECueSpecs(spatial=TSECueSpec("track", (2, 4)))
    dataset = type(
        "_Dataset",
        (),
        {
            "sample_rate": 16000,
            "targets_per_mixture": 1,
            "cue_specs": specs,
            "resolved_microphone_array": object(),
        },
    )()
    monkeypatch.setattr(
        tse_data,
        "_build_dataset_impl",
        lambda _cfg, *, mode, meta, spec: dataset,
    )

    built = tse_data.build_dataset({"data": {"mode": "offline_tse"}})
    fields = {field.name: field for field in built.contract.fields}
    assert fields["spatial"].axes == ("target", "direction", "cue_point")
    assert fields["spatial_timebase"].axes == ("target", "cue_point")
    assert built.contract.field_axes("mixture") == ("batch", "microphone", "time")


@pytest.mark.unit
@pytest.mark.parametrize("representation", ["static", "track"])
def test_tse_offline_builder_publishes_actual_spatial_cue_contract(tmp_path, representation):
    import numpy as np
    import torch

    from tasks.tse.data import build_dataset, tse_collate_fn
    from test_tse_offline_cues import _SPATIAL_SPLIT, _write_spatial_fixture

    track = representation == "track"
    payload = (
        np.array([[0, 0.1, 0.2], [800, 0.3, 0.4]], dtype=np.float64)
        if track else np.array([0.1, 0.2], dtype=np.float32)
    )
    manifest, cues_yaml = _write_spatial_fixture(
        tmp_path, spatial_values={"s1": payload, "s2": None},
        fields=("sample_anchor", "azimuth", "elevation") if track else ("azimuth", "elevation"),
        track_clock=track,
    )
    dataset = build_dataset({"data": {
        "mode": "offline_tse", "sample_rate": 16000,
        "train": {
            "samples_jsonl": str(manifest), "cues_yaml": str(cues_yaml),
            "manifest_format": "jsonl", "path_base": "manifest_relative",
            "target_selection": "all", "mix_variant": "default",
            "cue_schema": "wesep_v0_1", "enabled_cues": ["spatial"],
            **_SPATIAL_SPLIT, "spatial_representation": representation,
            **({"spatial_track_timebase": "source_samples"} if track else {}),
        },
    }})
    item = dataset[0]
    batch = tse_collate_fn([item], cue_specs=dataset.cue_specs)
    fields = {field.name: field for field in dataset.contract.fields}
    assert dataset.contract.cardinality == "variable_multi"
    assert dataset.contract.sample_rate == int(batch.sr) == 16000
    assert dataset.contract.field_axes("mixture") == ("batch", "microphone", "time")
    assert batch.mixture.shape == (1, 2, 1600)
    assert batch.num_targets.tolist() == [2]
    assert fields["spatial"].multiple and fields["spatial_present"].multiple
    assert len(fields["spatial"].axes) == batch.cues.spatial.value.ndim
    assert batch.cues.spatial.present.tolist() == [True, False]
    assert not any(name.startswith("speaker_") for name in fields)
    assert not {"geometry", "timebase", "provenance"}.intersection(fields)
    assert batch.geometry[0] is item["geometry"]
    assert batch.timebase[0].num_samples == int(batch.lengths[0])
    assert batch.provenance[0]["spatial_cue"]["speed_of_sound_mps"] == 343.0
    if track:
        assert batch.cues.spatial.timebase.dtype == torch.int64
        assert batch.cues.spatial.timebase.tolist() == [[0, 800], [0, 0]]
        assert batch.cues.spatial.lengths.tolist() == [2, 0]
    else:
        assert "spatial_timebase" not in fields and "spatial_lengths" not in fields


@pytest.mark.unit
def test_dataset_contract_exposes_optional_spatial_microphone_axes():
    se = build_se_dataset_contract(16000)
    aec = build_aec_dataset_contract(16000)
    tse = build_tse_dataset_contract(16000)
    ss = build_ss_dataset_contract(sample_rate=16000, num_sources=2)
    assert se.field_axes("noisy") == ("batch", "time")
    assert se.field_axes("noisy", spatial=True) == ("batch", "microphone", "time")
    assert se.field_axes("clean", spatial=True) == ("batch", "time")
    assert aec.field_axes("mic", spatial=True) == ("batch", "microphone", "time")
    assert tse.field_axes("mixture") == ("batch", "microphone", "time")
    assert tse.field_axes("mixture", spatial=True) == ("batch", "microphone", "time")
    assert ss.field_axes("mixture", spatial=True) == ("batch", "microphone", "time")
    with pytest.raises(ValueError, match="unknown fields"):
        DatasetContract(
            task="se",
            fields=(TensorField("clean", "waveform", ("batch", "time")),),
            sample_rate=16000,
            cardinality="single",
            spatial_observation_fields=("noisy",),
        )


@pytest.mark.unit
def test_model_contract_resolution_unwraps_parallel_wrappers():
    explicit = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(TensorField("est_sources", "waveform", ("batch", "source", "time"), multiple=True),),
        forward_kind="estimate",
        frontend_owner="model",
    )

    class Model:
        IO_CONTRACT = explicit

    class Compiled:
        _orig_mod = Model()

    class Distributed:
        module = Compiled()

    assert resolve_model_io_contract(
        Distributed(), {"data": {"mode": "offline_ss"}}
    ) is explicit


@pytest.mark.unit
def test_aec_legacy_contract_uses_declared_input_capabilities():
    class AECModel:
        aec_input_capabilities = frozenset(
            {"mic_spectra", "farend_spectra", "error_spectra"}
        )

    contract = resolve_model_io_contract(
        AECModel(), {"data": {"mode": "offline_aec"}, "stft": {"n_fft": 512}}
    )
    assert [field.name for field in contract.inputs] == [
        "error_spectra",
        "farend_spectra",
        "mic_spectra",
    ]
    assert contract.inputs[0].required is False


@pytest.mark.unit
def test_declared_spectrum_se_model_rejects_missing_runtime_stft():
    class SpectrumSE:
        input_domain = "spectrum"
        output_domain = "spectrum"
        frontend_owner = "runtime"

    config = {"data": {"mode": "offline", "sample_rate": 16000}}
    contract = resolve_model_io_contract(SpectrumSE(), config)

    with pytest.raises(ValueError, match="requires top-level STFT"):
        resolve_execution_plan(config, model_contract=contract)


@pytest.mark.unit
def test_declared_waveform_se_model_owns_frontend_even_with_loss_stft_config():
    class WaveformSE:
        input_domain = "waveform"
        output_domain = "waveform"
        frontend_owner = "model"

    config = {
        "data": {"mode": "offline", "sample_rate": 16000},
        "stft": {"n_fft": 512, "hop_length": 128, "win_length": 512},
    }
    plan = resolve_execution_plan(
        config,
        model_contract=resolve_model_io_contract(WaveformSE(), config),
    )

    assert plan.input_domain == "waveform"
    assert plan.frontend_owner == "model"
    assert plan.stft is None


@pytest.mark.unit
def test_declared_loss_returning_se_contract_preserves_forward_kind():
    class LossReturningSE:
        input_domain = "waveform"
        output_domain = "loss"
        forward_kind = "loss_returning"
        frontend_owner = "model"

    config = {"data": {"mode": "offline", "sample_rate": 16000}}
    plan = resolve_execution_plan(
        config,
        model_contract=resolve_model_io_contract(LossReturningSE(), config),
    )

    assert plan.output_domain == "loss"
    assert plan.forward_kind == "loss_returning"
