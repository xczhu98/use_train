"""Tests for tasks/tse/data/collate.py (tse_collate_fn) and tasks/tse/data/batch.py (TSEBatch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.data.simulation.array_profiles import (
    ArrayTopology,
    MicrophoneArrayProfile,
    ResolvedMicrophoneArray,
)
from core.data.simulation.trajectory import TrajectoryTimebase
from tasks.tse.cue_contract import TSECue, TSECueSpec, TSECueSpecs, TSECues
from tasks.tse.data.batch import TSEBatch
from tasks.tse.data.collate import make_tse_sample, make_tse_target, tse_collate_fn


def _speaker_specs(enroll_capacity: int = 6000) -> TSECueSpecs:
    return TSECueSpecs(
        speaker_waveform=TSECueSpec("waveform", (enroll_capacity,))
    )


def _speaker_cues(enroll: np.ndarray | None, *, length: int = 0) -> TSECues:
    return TSECues(
        speaker_waveform=TSECue(
            representation="waveform",
            value=enroll,
            present=enroll is not None,
            length=length,
            timebase=None,
        )
    )


def _geometry(channels: int, *, offset_m: float = 0.0) -> ResolvedMicrophoneArray:
    positions = tuple(
        (offset_m + 0.04 * channel, 0.0, 0.0) for channel in range(channels)
    )
    topology = ArrayTopology(
        positions,
        spacing_m=0.04 if channels > 1 else None,
    )
    return ResolvedMicrophoneArray(
        MicrophoneArrayProfile(f"fixture-{channels}-{offset_m}", topology),
        reference_channel=0,
    )


def _make_sample(
    mix_len: int,
    enroll_len: int,
    *,
    sr: int = 16000,
    seed: int = 0,
    spk_label: int = -1,
    channels: int = 1,
    key: str | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    mixture = rng.standard_normal((channels, mix_len)).astype(np.float32) * 0.01
    target = rng.standard_normal(mix_len).astype(np.float32) * 0.01
    enroll = rng.standard_normal(enroll_len).astype(np.float32) * 0.01
    return make_tse_sample(
        key=key or f"sample-{seed}",
        mixture=mixture,
        sr=sr,
        length=mix_len,
        targets=[
            make_tse_target(
                target=target,
                cues=_speaker_cues(enroll, length=enroll_len),
                speaker_id=f"speaker-{spk_label}",
                spk_label=spk_label,
            )
        ],
        geometry=_geometry(channels) if channels > 1 else None,
    )


class TestTseCollateFn:
    def test_output_shapes_padding_and_dtypes(self):
        result = tse_collate_fn(
            [_make_sample(8000, 4000, seed=1), _make_sample(12000, 6000, seed=2)],
            cue_specs=_speaker_specs(),
        )

        assert isinstance(result, TSEBatch)
        assert result.mixture.shape == (2, 1, 12000)
        assert result.target.shape == (2, 12000)
        assert result.sr.shape == () and result.sr.dtype == torch.int32
        assert result.lengths.tolist() == [8000, 12000]
        assert result.num_targets.dtype == torch.int32
        assert result.num_targets.tolist() == [1, 1]
        assert result.spk_labels.dtype == torch.int64
        assert result.target_present.tolist() == [True, True]
        speaker = result.cues.speaker_waveform
        assert speaker is not None
        assert speaker.value.shape == (2, 6000)
        assert speaker.lengths is not None
        assert speaker.lengths.tolist() == [4000, 6000]
        assert torch.all(result.mixture[0, :, 8000:] == 0)
        assert torch.all(speaker.value[0, 4000:] == 0)
        result.validate()

    def test_full_enrollment_can_exceed_configured_fallback_capacity(self):
        result = tse_collate_fn(
            [_make_sample(8, 10, seed=30)],
            cue_specs=_speaker_specs(4),
        )

        speaker = result.cues.speaker_waveform
        assert speaker is not None
        assert speaker.value.shape == (1, 10)
        assert speaker.lengths is not None and speaker.lengths.tolist() == [10]
        result.validate()

    def test_canonicalizes_declared_length_tails(self):
        sample = _make_sample(12, 8, seed=3)
        sample["length"] = 5
        sample["mixture"][..., 5:] = np.nan
        sample["targets"][0]["target"][5:] = np.nan
        cue = sample["targets"][0]["cues"].speaker_waveform
        assert cue is not None
        sample["targets"][0]["cues"] = _speaker_cues(cue.value, length=3)

        result = tse_collate_fn([sample], cue_specs=_speaker_specs(8))

        assert torch.all(result.mixture[0, :, 5:] == 0)
        assert torch.all(result.target[0, 5:] == 0)
        speaker = result.cues.speaker_waveform
        assert speaker is not None
        assert torch.all(speaker.value[0, 3:] == 0)

    def test_multichannel_mixture_keeps_channel_axis(self):
        result = tse_collate_fn(
            [_make_sample(8, 4, seed=4, channels=2)],
            cue_specs=_speaker_specs(4),
        )
        assert result.mixture.shape == (1, 2, 8)

    def test_scalar_tensor_metadata_is_accepted(self):
        sample = _make_sample(8, 4, seed=5, spk_label=2)
        sample["sr"] = torch.tensor(16000, dtype=torch.int32)
        sample["length"] = torch.tensor(8, dtype=torch.int64)
        sample["targets"][0]["spk_label"] = torch.tensor(2, dtype=torch.int64)
        result = tse_collate_fn([sample], cue_specs=_speaker_specs(4))
        assert result.spk_labels.tolist() == [2]

    def test_one_element_metadata_tensor_is_rejected(self):
        sample = _make_sample(8, 4, seed=6)
        sample["sr"] = torch.tensor([16000])
        with pytest.raises(TypeError, match="scalar integer tensor"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

    def test_none_samples_rejected_without_shrinking_batch(self):
        with pytest.raises(RuntimeError, match="refusing to shrink"):
            tse_collate_fn([None, _make_sample(8, 4, seed=7)], cue_specs=_speaker_specs(4))

    @pytest.mark.parametrize("raise_on_all_failed", [False, True])
    def test_all_failed_raises(self, raise_on_all_failed):
        with pytest.raises(RuntimeError, match="all 2 samples"):
            tse_collate_fn([None, None], raise_on_all_failed=raise_on_all_failed)

    def test_empty_batch_rejected(self):
        with pytest.raises(RuntimeError, match="empty batch"):
            tse_collate_fn([])

    def test_mismatched_sr_raises(self):
        with pytest.raises(RuntimeError, match="Sample rates must match"):
            tse_collate_fn(
                [_make_sample(8, 4, sr=16000), _make_sample(8, 4, sr=8000, seed=1)],
                cue_specs=_speaker_specs(4),
            )

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [("sr", 0, "sample rate must be > 0"), ("length", 0, "length must be > 0")],
    )
    def test_invalid_metadata_fails_before_canonicalization(self, field, value, match):
        sample = _make_sample(8, 4, seed=8)
        sample[field] = value
        sample["mixture"][0, 0] = np.nan
        with pytest.raises(RuntimeError, match=match):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

    def test_mixture_target_length_mismatch_raises(self):
        sample = _make_sample(8, 4, seed=9)
        sample["targets"][0]["target"] = np.zeros(7, dtype=np.float32)
        with pytest.raises(RuntimeError, match="length mismatch"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

    @pytest.mark.parametrize("tuple_length", [7, 8, 9])
    def test_tuple_samples_are_rejected(self, tuple_length):
        with pytest.raises(TypeError, match="canonical dict item"):
            tse_collate_fn([tuple(range(tuple_length))])

    def test_old_target_tuple_is_rejected(self):
        sample = _make_sample(8, 4, seed=10)
        sample["targets"] = [(np.zeros(8, dtype=np.float32),)]
        with pytest.raises(TypeError, match="canonical target dict"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

    def test_unknown_item_and_target_keys_are_rejected(self):
        sample = _make_sample(8, 4, seed=11)
        sample["enroll"] = np.zeros(4, dtype=np.float32)
        with pytest.raises(TypeError, match="unknown keys"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))
        sample = _make_sample(8, 4, seed=12)
        sample["targets"][0]["enroll"] = np.zeros(4, dtype=np.float32)
        with pytest.raises(TypeError, match="unknown keys"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))


class TestTseCollateMultiTarget:
    @staticmethod
    def _make_dict_sample(
        mix_len: int,
        n_targets: int,
        enroll_len: int,
        *,
        seed: int,
    ) -> dict:
        rng = np.random.default_rng(seed)
        targets = [
            make_tse_target(
                target=rng.standard_normal(mix_len).astype(np.float32) * 0.01,
                cues=_speaker_cues(
                    rng.standard_normal(enroll_len).astype(np.float32) * 0.01,
                    length=enroll_len,
                ),
                speaker_id=f"speaker-{seed}-{target_index}",
                spk_label=target_index,
            )
            for target_index in range(n_targets)
        ]
        return make_tse_sample(
            key=f"mixture-{seed}",
            mixture=rng.standard_normal((2, mix_len)).astype(np.float32) * 0.01,
            sr=16000,
            length=mix_len,
            targets=targets,
            geometry=_geometry(2),
            timebase=TrajectoryTimebase(16000, mix_len, mix_len, 1),
            provenance={"seed": seed},
        )

    def test_multi_target_packing_and_host_metadata(self):
        first = self._make_dict_sample(8, 2, 4, seed=1)
        second = self._make_dict_sample(8, 1, 4, seed=2)
        result = tse_collate_fn([first, second], cue_specs=_speaker_specs(4))

        assert result.mixture.shape == (2, 2, 8)
        assert result.target.shape == (3, 8)
        assert result.num_targets.tolist() == [2, 1]
        assert result.spk_labels.tolist() == [0, 1, 0]
        assert result.keys == ("mixture-1", "mixture-2")
        assert result.speaker_ids == ("speaker-1-0", "speaker-1-1", "speaker-2-0")
        assert result.geometry == (_geometry(2), _geometry(2))
        expected_timebase = TrajectoryTimebase(16000, 8, 8, 1)
        assert result.timebase == (expected_timebase, expected_timebase)
        assert result.provenance == ({"seed": 1}, {"seed": 2})

    def test_multi_target_sample_major_order(self):
        first = self._make_dict_sample(8, 2, 4, seed=3)
        second = self._make_dict_sample(8, 1, 4, seed=4)
        result = tse_collate_fn([first, second], cue_specs=_speaker_specs(4))

        np.testing.assert_allclose(result.target[0].numpy(), first["targets"][0]["target"])
        np.testing.assert_allclose(result.target[1].numpy(), first["targets"][1]["target"])
        np.testing.assert_allclose(result.target[2].numpy(), second["targets"][0]["target"])

    def test_absent_target_requires_zero_supervision(self):
        sample = self._make_dict_sample(8, 1, 4, seed=5)
        sample["targets"][0]["target_present"] = False
        with pytest.raises(RuntimeError, match="zero target supervision"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

        sample["targets"][0]["target"] = np.zeros(8, dtype=np.float32)
        result = tse_collate_fn([sample], cue_specs=_speaker_specs(4))
        assert result.target_present.tolist() == [False]
        assert torch.count_nonzero(result.target) == 0

    def test_channel_counts_must_match(self):
        with pytest.raises(RuntimeError, match="channel counts must match"):
            tse_collate_fn(
                [_make_sample(8, 4, channels=1), _make_sample(8, 4, channels=2, seed=1)],
                cue_specs=_speaker_specs(4),
            )

    def test_dict_sample_required_fields(self):
        with pytest.raises(TypeError, match="missing keys"):
            tse_collate_fn([{"mixture": np.zeros(10, dtype=np.float32)}])
        sample = _make_sample(8, 4)
        sample["targets"] = []
        with pytest.raises(TypeError, match="non-empty list"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))


class TestTSEBatch:
    def test_tuple_conversion_api_is_removed(self):
        assert not hasattr(TSEBatch, "from_tuple")
        assert not hasattr(TSEBatch, "to_tuple")

    def test_validate_rejects_inconsistent_target_cardinality(self):
        result = tse_collate_fn(
            [_make_sample(8, 4, seed=20)], cue_specs=_speaker_specs(4)
        )
        result.num_targets = torch.tensor([2], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"sum\(num_targets\)"):
            result.validate()

    def test_validate_requires_fixed_metadata_dtypes(self):
        result = tse_collate_fn(
            [_make_sample(8, 4, seed=21)], cue_specs=_speaker_specs(4)
        )
        result.sr = result.sr.to(torch.int64)
        with pytest.raises(TypeError, match="sr must use torch.int32"):
            result.validate()

    def test_validate_requires_one_device_for_all_batch_tensors(self):
        result = tse_collate_fn(
            [_make_sample(8, 4, seed=22)], cue_specs=_speaker_specs(4)
        )
        result.target = result.target.to("meta")
        with pytest.raises(ValueError, match="must be on one device"):
            result.validate()

        result = tse_collate_fn(
            [_make_sample(8, 4, seed=23)], cue_specs=_speaker_specs(4)
        )
        speaker = result.cues.speaker_waveform
        assert speaker is not None
        speaker.present = speaker.present.to("meta")
        with pytest.raises(ValueError, match="must be on one device"):
            result.validate()

    def test_geometry_uses_resolved_topology_not_profile_name(self):
        first = _make_sample(8, 4, seed=24, channels=2)
        second = _make_sample(8, 4, seed=25, channels=2)
        topology = second["geometry"].profile.topology
        second["geometry"] = ResolvedMicrophoneArray(
            MicrophoneArrayProfile("alternate-profile-name", topology),
            reference_channel=0,
        )
        result = tse_collate_fn([first, second], cue_specs=_speaker_specs(4))
        result.validate()

    def test_geometry_rejects_unknown_type_and_inconsistent_topology(self):
        sample = _make_sample(8, 4, seed=26, channels=2)
        sample["geometry"] = {"positions_m": [[0.0, 0.0, 0.0]]}
        with pytest.raises(TypeError, match="ResolvedMicrophoneArray"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

        first = _make_sample(8, 4, seed=27, channels=2)
        second = _make_sample(8, 4, seed=28, channels=2)
        second["geometry"] = _geometry(2, offset_m=0.01)
        with pytest.raises(ValueError, match="topology and reference channel"):
            tse_collate_fn([first, second], cue_specs=_speaker_specs(4))

    def test_multichannel_batch_requires_matching_geometry(self):
        sample = _make_sample(8, 4, seed=29, channels=2)
        sample["geometry"] = None
        with pytest.raises(ValueError, match="require resolved microphone geometry"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

        sample = _make_sample(8, 4, seed=30, channels=2)
        sample["geometry"] = _geometry(1)
        with pytest.raises(ValueError, match="channel count"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

    def test_timebase_requires_real_type_and_matches_audio_axes(self):
        sample = _make_sample(8, 4, seed=31)
        sample["timebase"] = {"sample_rate": 16000, "num_samples": 8}
        with pytest.raises(TypeError, match="TrajectoryTimebase"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

        sample = _make_sample(8, 4, seed=32)
        sample["timebase"] = TrajectoryTimebase(8000, 8, 8, 1)
        with pytest.raises(ValueError, match="sample_rate"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

        sample = _make_sample(8, 4, seed=33)
        sample["timebase"] = TrajectoryTimebase(16000, 7, 7, 1)
        with pytest.raises(ValueError, match="num_samples"):
            tse_collate_fn([sample], cue_specs=_speaker_specs(4))

    def test_to_recursively_moves_cue_tensors_and_keeps_host_metadata(self):
        result = tse_collate_fn(
            [_make_sample(8, 4, seed=34)], cue_specs=_speaker_specs(4)
        )
        moved = result.to("cpu")
        assert moved is not result
        assert moved.cues is not result.cues
        assert moved.cues.speaker_waveform is not result.cues.speaker_waveform
        assert moved.keys is result.keys
        moved.validate()

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="real pinned-memory assertion requires an available CUDA device",
    )
    def test_pin_memory_recurses_through_cues(self):
        result = tse_collate_fn(
            [_make_sample(8, 4, seed=23)], cue_specs=_speaker_specs(4)
        )
        pinned = result.pin_memory()
        speaker = pinned.cues.speaker_waveform
        assert pinned.mixture.is_pinned()
        assert pinned.target_present.is_pinned()
        assert speaker is not None
        assert speaker.value.is_pinned()
        assert speaker.present.is_pinned()
        assert pinned.keys is result.keys
        pinned.validate()

pytestmark = [pytest.mark.unit]
