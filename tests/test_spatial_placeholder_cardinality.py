"""空间 placeholder 的观测通道与监督源 cardinality 合同。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.data.dynamic_dataset_base import BaseDynamicMixingDataset
from core.data.errors import RecoverableResourceError
from core.data.simulation.array_profiles import resolve_microphone_array
from tasks.aec.data.dynamic_dataset import DynamicMixingDataset as AECDynamicMixingDataset
from tasks.se.data.dynamic_dataset import DynamicMixingDataset as SEDynamicMixingDataset
from tasks.ss.data.dynamic_dataset import SSDynamicMixingDataset
from tasks.ss.data.offline_dataset import SSOfflineDataset
from tasks.tse.cue_contract import TSECueSpec, TSECueSpecs
from tasks.tse.data.collate import tse_collate_fn
from tasks.tse.data.dynamic_dataset import TSEDynamicMixingDataset


pytestmark = pytest.mark.unit


def _spatial_stub(dataset_cls, *, channels=2):
    dataset = object.__new__(dataset_cls)
    positions = tuple((float(index), 0.0, 0.0) for index in range(channels))
    dataset.sim_cfg = SimpleNamespace(spatial_enabled=True)
    dataset._simulation_view = SimpleNamespace(
        shared=SimpleNamespace(
            spatial=SimpleNamespace(
                channels=channels,
                microphone_positions=positions,
            )
        )
    )
    dataset.max_duration = -1
    dataset.max_duration_seconds = None
    dataset.placeholder_num_samples = 12
    return dataset


def test_dynamic_placeholders_keep_spatial_observation_and_mono_supervision():
    se = _spatial_stub(SEDynamicMixingDataset)
    clean, noisy, *_ = se._placeholder_sample(16000, 0, "se")
    assert clean.shape == (12,)
    assert noisy.shape == (2, 12)

    aec = _spatial_stub(AECDynamicMixingDataset, channels=1)
    aec.state_detector_hop_length = None
    nearend, mic, farend = aec._placeholder_sample(16000, 0, "aec")[:3]
    assert nearend.shape == (12,)
    assert mic.shape == (1, 12)
    assert farend.shape == (12,)
    se_only = aec._pack_se_only(
        np.zeros(12, dtype=np.float32),
        np.zeros((1, 12), dtype=np.float32),
        16000,
        "aec",
    )
    assert se_only[0].shape == (12,)
    assert se_only[1].shape == (1, 12)
    assert se_only[2].shape == (12,)

    tse = _spatial_stub(TSEDynamicMixingDataset)
    tse.enrollment_duration = 0.01
    tse.targets_per_mixture = 1
    # 生产 __init__ 在 spatial_enabled=True 时必然写入 static/track；stub 必须补齐
    # 该属性，_missing_cues 才能构造出 spatial slot。
    tse.spatial_representation = "static"
    tse.resolved_microphone_array = resolve_microphone_array(
        custom_geometry={"positions_m": tse._simulation_view.shared.spatial.microphone_positions},
        reference_channel=0,
    )
    sample = tse._placeholder_sample(16000, 0, "tse")
    target = sample["targets"][0]
    speaker_waveform = target["cues"].speaker_waveform
    assert sample["mixture"].shape == (2, 12)
    assert sample["geometry"].channels == 2
    assert target["target"].shape == (12,)
    assert target["target_present"] is True
    assert speaker_waveform is not None and speaker_waveform.present is False
    assert speaker_waveform.value is None and speaker_waveform.length == 0
    cue_specs = TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (160,)),
        spatial=TSECueSpec("static", (2,)),
    )
    batch = tse_collate_fn([sample], cue_specs=cue_specs)
    assert batch.mixture.shape == (1, 2, 12)
    assert batch.target_present.tolist() == [True]
    assert batch.cues.speaker_waveform.present.tolist() == [False]
    assert batch.cues.spatial.present.tolist() == [False]

    ss = _spatial_stub(SSDynamicMixingDataset)
    ss.num_sources = 3
    sample = ss._placeholder_sample(16000, 0, "ss")
    assert sample["mixture"].shape == (2, 12)
    assert sample["sources"].shape == (3, 12)


def test_offline_ss_lenient_recoverable_placeholder_keeps_cardinality():
    dataset = object.__new__(SSOfflineDataset)
    dataset.records = (SimpleNamespace(uid="u", sample_rate=16000),)
    dataset.target_sr = None
    dataset.placeholder_num_samples = 12
    dataset.max_duration_seconds = None
    dataset.strict_load = False
    dataset.spatial = True
    dataset.num_sources = 2
    dataset.sim_cfg = SimpleNamespace(
        array_profile="linear_2ch_80mm",
        custom_geometry=None,
        reference_channel=0,
    )

    def fail(_index):
        raise RecoverableResourceError("temporary read failure")

    dataset._load_item = fail
    sample = dataset[0]

    assert sample["mixture"].shape == (2, 12)
    assert sample["sources"].shape == (2, 12)
    assert torch.count_nonzero(sample["mixture"]) == 0
    assert torch.count_nonzero(sample["sources"]) == 0


def test_placeholder_helper_uses_declared_spatial_cardinality():
    dataset = _spatial_stub(BaseDynamicMixingDataset, channels=3)
    observation, length = dataset._make_observation_placeholder(16000)
    assert observation.shape == (3, 12)
    assert length == 12
    assert observation.dtype == np.float32
