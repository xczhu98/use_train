"""共享 offline multitrack lifecycle 与音频 primitive 合同。"""

from __future__ import annotations

import random

import pytest
import torch

from core.data.offline_multitrack import (
    BaseOfflineMultiTrackDataset,
    OfflineLifecycleOptions,
    ensure_finite,
    ensure_mono,
    frame_window_between_domains,
    resampled_num_frames,
    slice_or_pad,
)

pytestmark = [pytest.mark.unit]


class _PersistentRngDataset(BaseOfflineMultiTrackDataset):
    def __init__(self):
        super().__init__(
            full_length=1,
            filtered_indices=[0],
            lifecycle=OfflineLifecycleOptions(seed=7),
        )

    def _output_sr(self, real_index: int) -> int:
        return 16000

    def _source_length(self, real_index: int) -> int:
        return 1

    def _load_sample(self, real_index: int):
        return self.sample_rng(real_index).randint(0, 10**9)

    def _placeholder_sample(self, real_index: int):
        raise AssertionError("不应进入 placeholder")


class _DatasetAdapter(BaseOfflineMultiTrackDataset):
    def __init__(self, *, strict_load=True):
        self.fail = {1}
        self.loads = []
        super().__init__(
            full_length=4,
            filtered_indices=[3, 1, 2],
            lifecycle=OfflineLifecycleOptions(
                strict_load=strict_load,
                placeholder_num_samples=5,
                seed=7,
            ),
        )

    def _output_sr(self, real_index: int) -> int:
        return (8000, 16000, 16000, 16000)[real_index]

    def _source_length(self, real_index: int) -> int:
        return (2, 3, 4, 6)[real_index]

    def _load_sample(self, real_index: int):
        self.loads.append(real_index)
        if real_index in self.fail:
            raise OSError(f"broken-{real_index}")
        return real_index

    def _placeholder_sample(self, real_index: int):
        return ("placeholder", real_index, self.placeholder_num_samples)

    def _sample_description(self, real_index: int) -> str:
        return f"record={real_index}"


def test_filtered_mapping_caches_sampler_metadata_and_uses_real_indices():
    dataset = _DatasetAdapter()

    assert len(dataset) == 3
    assert dataset.get_srs() == [16000, 16000, 16000]
    assert dataset.get_source_length() == [6, 3, 4]

    dataset._source_length = lambda _index: (_ for _ in ()).throw(AssertionError("cache miss"))
    assert dataset.get_source_length() == [6, 3, 4]
    assert dataset[0] == 3
    assert dataset.loads == [3]


def test_strict_and_lenient_lifecycle_preserve_output_sr_bucket():
    strict = _DatasetAdapter(strict_load=True)
    with pytest.raises(RuntimeError, match=r"index=1.*record=1.*broken-1"):
        strict[1]

    lenient = _DatasetAdapter(strict_load=False)
    assert lenient[1] == 2
    assert lenient.loads == [1, 2]

    lenient.fail = {1, 2, 3}
    assert lenient[1] == ("placeholder", 1, 5)


def test_sample_rng_and_set_epoch_are_deterministic_and_idempotent():
    dataset = _DatasetAdapter()

    initial = dataset.sample_rng(2).randint(0, 10**9)
    dataset.set_epoch(4, base_seed=11)
    once = dataset.sample_rng(2).randint(0, 10**9)
    dataset.set_epoch(4, base_seed=11)
    twice = dataset.sample_rng(2).randint(0, 10**9)
    dataset.set_epoch(5, base_seed=11)
    next_epoch = dataset.sample_rng(2).randint(0, 10**9)

    assert initial != once
    assert once == twice
    assert once != next_epoch
    assert dataset.sample_rng(2, stream=1).randint(0, 10**9) != next_epoch
    assert isinstance(dataset.sample_rng(2), random.Random)


def test_persistent_worker_observes_epoch_and_base_seed_updates():
    dataset = _PersistentRngDataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        initial = next(iter(loader))
        dataset.set_epoch(4, base_seed=11)
        changed = next(iter(loader))
        expected = dataset.sample_rng(0).randint(0, 10**9)
        dataset.set_epoch(4, base_seed=11)
        repeated = next(iter(loader))
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert changed == expected
    assert changed == repeated
    assert initial != changed


def test_audio_primitives_make_mono_validate_finite_and_slice_or_pad():
    stereo = torch.tensor([[1.0, float("nan"), 5.0], [3.0, 4.0, 7.0]])
    mono = ensure_mono(stereo)
    assert mono.shape == (1, 3)
    assert mono[0, [0, 2]].tolist() == [2.0, 6.0]

    with pytest.raises(ValueError, match="NaN/Inf in mixture"):
        ensure_finite(mono, strict=True, track="mixture", sample="record=0")
    sanitized = ensure_finite(
        mono, strict=False, track="mixture", sample="record=0"
    )
    assert sanitized.isfinite().all()
    assert sanitized[0, 1] == 0

    wav = torch.arange(5, dtype=torch.float32).unsqueeze(0)
    assert slice_or_pad(wav, start=2, length=5).tolist() == [[2, 3, 4, 0, 0]]


def test_source_window_conversion_has_explicit_domains():
    assert resampled_num_frames(48001, 48000, 16000) == 16001
    assert frame_window_between_domains(
        start=24000,
        length=48000,
        source_sr=48000,
        output_sr=16000,
    ) == (8000, 16000)
