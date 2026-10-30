"""AEC trainer/data 残留审计问题的回归测试。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

import tasks.aec.data as aec_data
from tasks.aec.data.batch import AECBatch
from tasks.aec.data.dynamic_dataset import scale_signal_to_dBfs_rms
from tasks.aec.trainer.aec_trainer import AECTrainer


class _Stft:
    hop_length = 4


def _batch(*, state: torch.Tensor | None = None) -> AECBatch:
    return AECBatch(
        nearend=torch.zeros(2, 8),
        mic=torch.zeros(2, 8),
        farend=torch.zeros(2, 8),
        fs=torch.tensor(16000, dtype=torch.int32),
        lengths=torch.tensor([3, 8], dtype=torch.int32),
        state=state,
    )


@pytest.mark.unit
def test_aec_trainer_marks_state_padding_after_declared_length():
    trainer = object.__new__(AECTrainer)
    trainer.stft_module = _Stft()
    batch = _batch(state=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long))

    trainer._mask_state_padding(batch)

    assert batch.state.tolist() == [[0, -1, -1], [0, 1, 2]]


@pytest.mark.unit
def test_aec_trainer_removes_stale_model_metadata_for_missing_batch_fields():
    batch = _batch()
    data = {
        "lengths": torch.tensor([99]),
        "is_pure_noise": torch.tensor([True, True]),
        "is_pure_echo": torch.tensor([True, True]),
        "is_pure_echo_noise": torch.tensor([True, True]),
        "state": torch.zeros(2, 3, dtype=torch.long),
    }

    AECTrainer._restore_batch_metadata(data, batch)

    assert data["lengths"] is batch.lengths
    assert all(
        key not in data
        for key in ("is_pure_noise", "is_pure_echo", "is_pure_echo_noise", "state")
    )


@pytest.mark.unit
def test_aec_compat_builder_rejects_retired_simulation_cfg_strict_key():
    cfg = {
        "mode": "dynamic",
        "sample_rate": 16000,
        "simulation_cfg_strict": False,
        "train": {
            "speech_scp": "speech.scp",
            "noise_scp": "noise.scp",
            "max_duration_seconds": 10.0,
            "repeat_to_max_duration": True,
            "simulation_cfg": {
                "prob_target_only": 1.0,
            },
        },
    }

    with patch("tasks.aec.data.DynamicMixingDataset") as dataset_cls:
        with pytest.raises(
            ValueError, match="simulation_cfg_strict is no longer supported"
        ):
            aec_data.build_dataset(cfg, mode="train")

        dataset_cls.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("signal", "target_db", "match"),
    [
        (np.array([1e100], dtype=np.float64), -20.0, "representable in float32"),
        (np.ones(8, dtype=np.float32), 1000.0, "non-float32-finite output"),
    ],
)
def test_aec_dynamic_rms_scaling_rejects_unrepresentable_extremes(
    signal: np.ndarray, target_db: float, match: str
):
    with pytest.raises(ValueError, match=match):
        scale_signal_to_dBfs_rms(signal, target_db)
