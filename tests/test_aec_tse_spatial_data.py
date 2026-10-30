"""AEC/TSE spatial observation 数据合同测试。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.data.dynamic_dataset_base import LayerEResult
from core.data.simulation.array_profiles import resolve_microphone_array
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.scene_renderer import finish_speech_scene, render_speech_scene
from core.data.simulation_views import project_simulation_view
from core.data.simulation_configs import SimulationConfigs
from core.models.io_contract import ModelIOContract, TensorField
from tasks.aec.data.batch import AECBatch
from tasks.aec.data.collate import aec_collate_fn
from tasks.aec.trainer.aec_trainer import AECTrainer
from tasks.aec.trainer.entry import AECTrainingEntry
from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn
from tasks.tse.model_io import build_tse_model_inputs

pytestmark = pytest.mark.unit


def test_aec_collate_preserves_spatial_mic_and_mono_targets():
    mic = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    sample = (
        torch.zeros(8), mic, torch.zeros(8), 16000, 8, torch.zeros(8), False,
        False, False, False, False,
    )
    batch = aec_collate_fn([sample])
    assert batch.nearend.shape == (1, 8)
    assert batch.mic.shape == (1, 2, 8)
    assert batch.farend.shape == (1, 8)


def test_aec_forward_rejects_spatial_mic_for_mono_model():
    class _STFT:
        n_fft = 4
        hop_length = 2

        def __call__(self, waveform):
            assert waveform.ndim == 2
            return torch.zeros(waveform.shape[0], 3, 5, 2)

        def inverse(self, spectra, length):
            return torch.zeros(spectra.shape[0], length)

    class _Model:
        def __init__(self):
            self.seen = None
            self.seen_waveform = None

        def __call__(self, data):
            self.seen_waveform = data["mic"]
            self.seen = data["mic_spectra"]
            return {"nearend_estimate_spectra": data["mic_spectra"]}

        def modules(self):
            return ()

    model = _Model()
    trainer = object.__new__(AECTrainer)
    trainer.stft_module = _STFT()
    trainer.model = model
    trainer._aec_frontend_enabled = False
    batch = AECBatch(
        nearend=torch.zeros(1, 8),
        mic=torch.arange(16, dtype=torch.float32).reshape(1, 2, 8),
        farend=torch.zeros(1, 8),
        fs=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.tensor([8], dtype=torch.int32),
    )
    trainer._assert_batch_valid(batch.mic, batch.farend, batch.nearend, batch.lengths)

    with pytest.raises(ValueError, match="native model IO contract"):
        trainer._forward_aec_batch(batch)


def test_aec_multichannel_without_native_model_contract_fails_closed():
    trainer = object.__new__(AECTrainer)
    trainer.model = object()
    with pytest.raises(ValueError, match="native model IO contract"):
        trainer._adapt_aec_mic(torch.zeros(1, 2, 8))


def test_aec_entry_rejects_removed_multichannel_adapter_config():
    with pytest.raises(ValueError, match="aec_multichannel_adapter is removed"):
        AECTrainingEntry("AEC_Test")._validate_config(
            {"aec_multichannel_adapter": {"type": "reference"}}
        )


def test_tse_spatial_mixture_reaches_spectrum_adapter():
    cue_specs = TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (4,))
    )
    sample = make_tse_sample(
        key="spatial",
        mixture=torch.arange(16, dtype=torch.float32).reshape(2, 8),
        sr=16000,
        length=8,
        targets=[
            make_tse_target(
                target=torch.zeros(8),
                cues=TSECues(
                    speaker_waveform=TSECue(
                        "waveform", torch.zeros(4), True, 4, None
                    )
                ),
                speaker_id=None,
                spk_label=0,
            )
        ],
        geometry=resolve_microphone_array(
            array_profile="linear_2ch_80mm", reference_channel=0
        ),
    )
    batch = tse_collate_fn([sample], cue_specs=cue_specs)

    class _Model:
        sample_rate = 16000
        supports_multichannel_input = True
        reference_channel = 0
        IO_CONTRACT = ModelIOContract(
            tasks=frozenset({"tse"}),
            inputs=(
                TensorField(
                    "mix_spectra",
                    "spectrum",
                    ("batch", "microphone", "frequency", "frame", "complex"),
                ),
                TensorField("speaker_waveform", "waveform", ("batch", "time")),
            ),
            outputs=(
                TensorField(
                    "target_estimate_spectra",
                    "spectrum",
                    ("batch", "frequency", "frame", "complex"),
                ),
            ),
            forward_kind="estimate",
            frontend_owner="runtime",
        )

    class _STFT:
        n_fft = 4
        hop_length = 2

        def __call__(self, waveform):
            return torch.zeros(waveform.shape[0], 3, 5, 2)

    inputs = build_tse_model_inputs(
        model=_Model(), stft=_STFT(), mixture=batch.mixture,
        cues=batch.cues, sample_rate=16000, lengths=batch.lengths,
        num_targets=batch.num_targets,
        reference_channel=batch.geometry[0].reference_channel,
        target=batch.target,
    )
    assert inputs["mixture"].shape == (1, 8)
    assert inputs["speaker_waveform"].shape == (1, 4)
    assert inputs["mix_spectra"].shape == (1, 2, 3, 5, 2)
    assert inputs["target"].shape == (1, 8)


def test_supplied_spatial_scene_keeps_target_reference_and_observation_channels():
    cfg = SimulationConfigs.from_dict({
        "spatial_enabled": True, "array_profile": "linear_2ch_80mm", "reference_channel": 1,
        "prob_reverberation": 0.0, "prob_channel_enabled": False,
    })
    dataset = SimpleNamespace(
        sim_cfg=cfg, export_simulation_stems=False,
        _simulation_view=project_simulation_view(cfg, "tse"),
    )
    dataset._apply_channel_degradation_result = lambda waveform, *args: LayerEResult(
        waveform, False, 0, (), (), False, "disabled"
    )
    source = np.zeros(16, dtype=np.float32)
    source[0] = 1.0
    rirs = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    scene = render_speech_scene(
        dataset,
        (
            AcousticActor(
                uid="target",
                role=AcousticRole.TARGET_SPEECH,
                rir_policy=RIRPolicy.SPATIAL_SOURCE,
                waveform=source,
            ),
        ),
        8000,
        np.random.default_rng(1),
        sir_range=[0.0, 0.0], need_noise=False, spatial_rirs=rirs,
    )
    finished = finish_speech_scene(dataset, scene, 8000, np.random.default_rng(2))
    assert finished.bundle.require("mixture").shape == (2, 16)
    assert finished.bundle.require("target/0").shape == (16,)
    np.testing.assert_array_equal(
        finished.bundle.require("target/0"), finished.bundle.require("source_full_rir/0")[1]
    )


def test_preplaced_spatial_scene_preserves_timeline_facts():
    cfg = SimulationConfigs.from_dict(
        {
            "spatial_enabled": True,
            "array_profile": "linear_2ch_80mm",
            "reference_channel": 1,
            "prob_reverberation": 0.0,
            "prob_channel_enabled": False,
        }
    )
    dataset = SimpleNamespace(
        sim_cfg=cfg,
        export_simulation_stems=False,
        _simulation_view=project_simulation_view(cfg, "tse"),
    )
    source = np.zeros(16, dtype=np.float32)
    source[0] = 1.0
    rirs = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    timeline = (
        {
            "source_length": 16,
            "scene_length": 16,
            "turns": [],
            "mode": "head",
            "requested_ratio": 0.0,
            "source_trim_offset": 0,
        },
    )
    scene = render_speech_scene(
        dataset,
        (
            AcousticActor(
                uid="target",
                role=AcousticRole.TARGET_SPEECH,
                rir_policy=RIRPolicy.SPATIAL_SOURCE,
                waveform=source,
            ),
        ),
        8000,
        np.random.default_rng(1),
        sir_range=[0.0, 0.0],
        need_noise=False,
        spatial_rirs=rirs,
        sources_preplaced=True,
        timeline=timeline,
    )
    assert scene.facts["timeline"] == timeline


def test_aec_input_builder_keeps_spatial_mic_and_mono_farend():
    class _STFT:
        n_fft = 4
        hop_length = 2

        def __call__(self, waveform):
            if waveform.ndim != 2:
                raise AssertionError(f"STFT expects flattened batch/channel waveform, got {waveform.shape}")
            return torch.zeros(waveform.shape[0], 3, 5, 2)

    trainer = object.__new__(AECTrainer)
    trainer.stft_module = _STFT()
    trainer.model = type("_NativeAEC", (), {"supports_multichannel_input": True})()
    trainer._aec_frontend_enabled = False
    batch = AECBatch(
        nearend=torch.zeros(1, 8),
        mic=torch.arange(16, dtype=torch.float32).reshape(1, 2, 8),
        farend=torch.zeros(1, 8),
        fs=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.tensor([8], dtype=torch.int32),
    )
    data = trainer._build_aec_model_input(batch)

    assert data["mic"].shape == (1, 2, 8)
    assert data["farend"].shape == (1, 8)
    assert data["mic_spectra"].shape == (1, 2, 3, 5, 2)
    assert data["farend_spectra"].shape == (1, 3, 5, 2)


def test_offline_spatial_placeholder_keeps_geometry_channels():
    from tasks.aec.data.offline_dataset import AECOfflineDataset
    from tasks.tse.data.offline_dataset import TSEOfflineDataset

    # 通过空对象直接覆盖 placeholder 所需的已解析状态，避免重复构造 manifest。
    aec = object.__new__(AECOfflineDataset)
    aec.uids = ["u"]
    aec.utt2fs = {"u": 16000}
    aec.max_duration_seconds = None
    aec.max_duration = -1
    aec.placeholder_num_samples = 12
    aec.target_sr = None
    aec.resolved_microphone_array = __import__(
        "core.data.simulation.array_profiles", fromlist=["resolve_microphone_array"]
    ).resolve_microphone_array(array_profile="linear_2ch_80mm", reference_channel=0)
    sample = aec._placeholder(0)
    assert sample[0].shape == (1, 12)
    assert sample[1].shape == (2, 12)

    from tasks.tse.data.wesep_adapter import WeSepSampleRecord, WeSepTargetRecord

    tse = object.__new__(TSEOfflineDataset)
    tse.uids = ["u"]
    tse.records = (
        WeSepSampleRecord(
            key="u",
            manifest_path="fixture.jsonl",
            mix_variant="default",
            mixture_paths=("mixture.wav",),
            targets=(WeSepTargetRecord("speaker", ("target.wav",), ()),),
        ),
    )
    tse.utt2fs = {"u": 16000}
    tse.max_duration_seconds = None
    tse.max_duration = -1
    tse.placeholder_num_samples = 12
    tse.target_sr = None
    tse.max_enroll_duration_seconds = None
    tse._audio_representation = None
    tse.spatial_representation = None
    tse.spatial_angle_unit = None
    tse.spatial_coordinate_frame = None
    tse.spatial_speed_of_sound_mps = None
    tse.spatial_track_timebase = None
    tse._spatial_fields = None
    tse._spatial_field_defaults = {}
    tse._spatial_cue_payloads = {}
    tse._mixture_info = {"u": (12, 16000, 2)}
    tse.resolved_microphone_array = aec.resolved_microphone_array
    sample = tse._placeholder(0)
    assert sample["mixture"].shape == (2, 12)
    assert sample["targets"][0]["target"].shape == (12,)


def test_offline_spatial_observation_read_rechecks_geometry_channels(tmp_path):
    from core.data.simulation.array_profiles import resolve_microphone_array
    from tasks.aec.data.offline_dataset import AECOfflineDataset

    dataset = object.__new__(AECOfflineDataset)
    dataset.resolved_microphone_array = resolve_microphone_array(
        array_profile="linear_2ch_80mm", reference_channel=0
    )
    path = tmp_path / "offline_spatial_wrong_channels.wav"
    import soundfile as sf
    sf.write(path, np.zeros((8, 1), dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="expected geometry C=2"):
        dataset._load_observation(str(path))
