"""共享动态数据在真实离线 RIR 与 worker 生命周期下的随机性合同。"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from core.data.speech_scene import scene_rng
from tasks.se.data.dynamic_dataset import DynamicMixingDataset


@pytest.fixture
def offline_assets(tmp_path):
    sr, length = 8000, 1024
    speech_lines, noise_lines, rir_lines = [], [], []
    rng = np.random.default_rng(801)
    for index in range(8):
        for family, lines in (("speech", speech_lines), ("noise", noise_lines)):
            name = f"{family}_{index}.wav"
            waveform = rng.normal(0.0, 0.05, length).astype(np.float32)
            sf.write(tmp_path / name, waveform, sr, subtype="FLOAT")
            lines.append(f"{family}_{index} {sr} {length} {name}\n")
        rir = np.zeros(64, dtype=np.float32)
        rir[0], rir[8 + index], rir[40 + index] = 0.7, 0.3, -0.15
        name = f"rir_{index}.wav"
        sf.write(tmp_path / name, rir, sr, subtype="FLOAT")
        rir_lines.append(f"rir_{index} {sr} 64 {name}\n")
    for family, lines in (("speech", speech_lines), ("noise", noise_lines), ("rir", rir_lines)):
        (tmp_path / f"{family}.scp").write_text("".join(lines), encoding="utf-8")
    try:
        yield tmp_path
    finally:
        import shutil

        shutil.rmtree(tmp_path)


def make_dataset(root, *, split="train", reuse=False):
    dataset = DynamicMixingDataset(
        str(root / "speech.scp"), str(root / "noise.scp"), str(root / "rir.scp"),
        max_duration_seconds=1024 / 8000, filter_by_duration=False,
        simulation_cfg={
            "prob_reverberation": 1.0, "reuse_rir": reuse, "reuse_noise": reuse,
            "prob_target_only": 0.0, "prob_noise_only": 0.0, "prob_target_noise": 1.0,
            "snr_low": -3.0, "snr_high": 15.0,
            "source_volume_normalization_enabled": False,
            "peak_normalization_mode": "off", "prob_channel_enabled": False,
            "noisy_post_mix_degradations_enabled": False,
        },
    )
    dataset._simulation_split = split
    dataset.set_epoch(0, 1234)
    return dataset


def make_loader(dataset, *, workers=2, persistent=False, generator_seed=81):
    return DataLoader(
        dataset, batch_size=None, num_workers=workers,
        persistent_workers=persistent,
        multiprocessing_context="spawn" if persistent else None,
        generator=torch.Generator().manual_seed(generator_seed),
    )


def collect(loader):
    return [(item[0].numpy().copy(), item[1].numpy().copy()) for item in loader]


def assert_samples_equal(left, right):
    assert len(left) == len(right) == 8
    for actual, expected in zip(left, right):
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


@pytest.mark.integration
@pytest.mark.subprocess
@pytest.mark.parametrize("reuse", [False, True])
def test_nonpersistent_validation_and_epoch_resume(offline_assets, reuse):
    validation = make_dataset(offline_assets, split="validation", reuse=reuse)
    loader = make_loader(validation)
    first = collect(loader)
    assert_samples_equal(collect(loader), first)
    validation.set_epoch(9, 1234)
    assert_samples_equal(collect(loader), first)
    fresh_val = make_dataset(offline_assets, split="validation", reuse=reuse)
    assert_samples_equal(collect(make_loader(fresh_val, generator_seed=999)), first)

    training = make_dataset(offline_assets, reuse=reuse)
    loader = make_loader(training)
    epoch_zero = collect(loader)
    training.set_epoch(1, 1234)
    epoch_one = collect(loader)
    restored = make_dataset(offline_assets, reuse=reuse)
    restored.set_epoch(1, 1234)
    assert_samples_equal(collect(make_loader(restored, generator_seed=999)), epoch_one)
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(epoch_zero, epoch_one))
    if not reuse:
        restored.set_epoch(1, 1234)
        assert_samples_equal(collect(make_loader(restored, workers=0)), epoch_one)


@pytest.mark.unit
def test_scene_rng_split_epoch_seed_index_stream_contract(offline_assets):
    dataset = make_dataset(offline_assets)
    baseline = scene_rng(dataset, 0, 0).integers(0, 2**31, size=8)
    np.testing.assert_array_equal(dataset._get_rng(0).integers(0, 2**31, size=8), baseline)
    np.testing.assert_array_equal(scene_rng(dataset, 0, 0).integers(0, 2**31, size=8), baseline)
    for index, stream in ((1, 0), (0, 1)):
        assert not np.array_equal(scene_rng(dataset, index, stream).integers(0, 2**31, size=8), baseline)
    dataset.set_epoch(1, 1234)
    assert not np.array_equal(scene_rng(dataset, 0, 0).integers(0, 2**31, size=8), baseline)
    dataset.set_epoch(0, 4321)
    assert not np.array_equal(scene_rng(dataset, 0, 0).integers(0, 2**31, size=8), baseline)
    dataset._simulation_split = "validation"
    dataset.set_epoch(0, 1234)
    validation = scene_rng(dataset, 0, 0).integers(0, 2**31, size=8)
    assert not np.array_equal(validation, baseline)
    dataset.set_epoch(10, 1234)
    np.testing.assert_array_equal(scene_rng(dataset, 0, 0).integers(0, 2**31, size=8), validation)


@pytest.mark.integration
@pytest.mark.subprocess
@pytest.mark.parametrize("split", ["train", "validation"])
def test_spawn_persistent_workers_observe_seed_and_epoch(offline_assets, split):
    dataset = make_dataset(offline_assets, split=split, reuse=True)
    loader = make_loader(dataset, persistent=True)
    try:
        first = collect(loader)
        dataset.set_epoch(0, 4321)
        reseeded = collect(loader)
        fresh = make_dataset(offline_assets, split=split, reuse=True)
        fresh.set_epoch(0, 4321)
        assert_samples_equal(reseeded, collect(make_loader(fresh)))
        assert any(not np.array_equal(a[1], b[1]) for a, b in zip(first, reseeded))
        dataset.set_epoch(1, 4321)
        next_epoch = collect(loader)
        fresh.set_epoch(1, 4321)
        assert_samples_equal(next_epoch, collect(make_loader(fresh)))
        if split == "validation":
            assert_samples_equal(next_epoch, reseeded)
        else:
            assert any(not np.array_equal(a[1], b[1]) for a, b in zip(next_epoch, reseeded))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
