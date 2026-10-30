"""RIR 发布边界的独立快照、真实 replay 与整块复制预算。"""

from __future__ import annotations

import pickle
import sys
import tracemalloc

import numpy as np
import pytest
import soundfile as sf

from core.data.acoustic_scene_plan import (
    AcousticScenePlan, NoisePlan, RIRPlan, ResolvedAssets, SourceGainPlan,
)
from core.data.acoustic_scene_replay import replay_target_acoustic_scene
from tasks.se.data.dynamic_dataset import DynamicMixingDataset


pytestmark = pytest.mark.unit


def measure_peak(operation):
    tracemalloc.start()
    try:
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


@pytest.mark.parametrize("strided", [False, True])
def test_resolved_assets_convert_with_one_owned_snapshot(strided):
    storage = np.ones((2, 2**20), dtype=np.float64)
    source = storage[:, ::2] if strided else storage
    assets, peak = measure_peak(lambda: ResolvedAssets(rir=source))
    assert peak < assets.rir.nbytes * 1.25
    assert assets.rir.dtype == np.float32
    assert assets.rir.flags.c_contiguous
    assert not assets.rir.flags.writeable
    source[:] = 7.0
    np.testing.assert_array_equal(assets.rir, np.ones(source.shape, dtype=np.float32))
    assert not np.shares_memory(source, assets.rir)


def replay(plan, assets, waveform):
    return replay_target_acoustic_scene(
        waveform, scene_plan=plan, resolved_assets=assets, asset_role="scene",
        sample_rate=8000, num_samples=waveform.shape[-1],
        source_volume_normalization_enabled=False, source_volume_use_rms=True,
    )


def test_replay_validation_does_not_allocate_a_full_rir_copy(monkeypatch):
    rir = np.ones((1, 2**21), dtype=np.float32)
    assets = ResolvedAssets(rir=rir, role="scene", sample_rate=8000, num_samples=32)
    plan = AcousticScenePlan(
        sample_rate=8000, num_samples=32, need_noise=False,
        rir=RIRPlan(True, "rir", "rir.wav", False, False, False, None, False, 0.002, "early"),
        source_gain=SourceGainPlan.disabled(), noise=NoisePlan.disabled(),
    )
    monkeypatch.setattr(
        "core.data.acoustic_scene_replay.add_paired_reverberation",
        lambda waveform, *_args, **_kwargs: (waveform.copy(), waveform.copy()),
    )
    _, peak = measure_peak(lambda: replay(plan, assets, np.ones((1, 32), dtype=np.float32)))
    assert peak < assets.rir.nbytes * 0.5


@pytest.fixture
def acoustic_dataset(tmp_path):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(dir=tmp_path) as directory:
        root = Path(directory)

        def build(rir, *, normalize=False, target="early", capture=False, reuse=False):
            speech = np.zeros((1, 64), dtype=np.float32)
            speech[0, [0, 20, 40]] = [0.25, -0.125, 0.0625]
            sf.write(root / "speech.wav", speech[0], 8000, subtype="FLOAT")
            if rir.shape[0] > 1:
                name = "rir.pkl"
                with (root / name).open("wb") as stream:
                    pickle.dump({"rir": rir, "fs": 8000, "moving": True, "segment_sec": 0.004}, stream)
            else:
                name = "rir.wav"
                sf.write(root / name, rir[0], 8000, subtype="FLOAT")
            (root / "speech.scp").write_text("u 8000 64 speech.wav\n", encoding="utf-8")
            (root / "noise.scp").write_text("n 8000 64 speech.wav\n", encoding="utf-8")
            (root / "rir.scp").write_text(f"r 8000 {rir.shape[-1]} {name}\n", encoding="utf-8")
            dataset = DynamicMixingDataset(
                str(root / "speech.scp"), str(root / "noise.scp"), str(root / "rir.scp"),
                max_duration_seconds=64 / 8000, filter_by_duration=False,
                export_simulation_stems=capture,
                simulation_cfg={
                    "prob_reverberation": 1.0, "rir_normalize": normalize,
                    "reuse_rir": reuse, "target_type": target, "early_rir_sec": 0.002,
                    "segment_sec": 0.004, "prob_target_only": 1.0,
                    "prob_noise_only": 0.0, "prob_target_noise": 0.0,
                    "source_volume_normalization_enabled": False,
                    "peak_normalization_mode": "off", "prob_channel_enabled": False,
                    "noisy_post_mix_degradations_enabled": False,
                },
            )
            return dataset, speech

        yield build


def mix(dataset, speech):
    return dataset._mix_acoustic(
        speech, 8000, speech.shape[-1], speech.shape[-1],
        np.random.default_rng(55), False, resolved_asset_role="scene",
    )


def test_recorder_keeps_only_reader_copy_before_asset_publication(acoustic_dataset, monkeypatch):
    rir = np.ones((1, 2**20), dtype=np.float32)
    dataset, speech = acoustic_dataset(rir)
    monkeypatch.setattr(
        "core.data.dynamic_dataset_base.add_paired_reverberation",
        lambda waveform, *_args, **_kwargs: (waveform.copy(), waveform.copy()),
    )
    mix(dataset, speech)
    copies = []

    def profile(_frame, event, function):
        if event == "c_call" and getattr(function, "__name__", None) == "copy":
            owner = getattr(function, "__self__", None)
            if isinstance(owner, np.ndarray) and owner.nbytes == rir.nbytes:
                copies.append(owner.shape)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        output = mix(dataset, speech)
    finally:
        sys.setprofile(previous)
    assert len(copies) == 1, f"full RIR ndarray.copy allocations: {copies}"
    assert not output.resolved_assets.rir.flags.writeable
    np.testing.assert_array_equal(output.resolved_assets.rir, rir)


def test_forced_rir_is_borrowed_without_full_copy(acoustic_dataset, monkeypatch):
    rir = np.ones((1, 2**20), dtype=np.float32)
    dataset, speech = acoustic_dataset(rir)
    monkeypatch.setattr(
        "core.data.dynamic_dataset_base.add_paired_reverberation",
        lambda waveform, *_args, **_kwargs: (waveform.copy(), waveform.copy()),
    )
    formal = mix(dataset, speech)
    _, peak = measure_peak(lambda: dataset._mix_acoustic_rir(
        speech, 8000, np.random.default_rng(88), True,
        forced_rir_plan=formal.acoustic_scene_plan.rir,
        forced_rir_asset=formal.resolved_assets.rir,
    ))
    assert peak < rir.nbytes * 0.5


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("target", ["dry", "early"])
@pytest.mark.parametrize("capture", [False, True])
def test_real_rir_replay_and_cache_do_not_mutate_published_assets(
    acoustic_dataset, moving, normalize, target, capture,
):
    rir = np.zeros((2 if moving else 1, 48), dtype=np.float32)
    rir[0, [0, 8, 24]] = [0.5, 0.2, -0.1]
    if moving:
        rir[1, [0, 12, 28]] = [0.25, 0.15, 0.1]
    dataset, speech = acoustic_dataset(
        rir, normalize=normalize, target=target, capture=capture, reuse=True,
    )
    previous_assets = None
    for cache_hit in (False, True):
        output = mix(dataset, speech)
        plan, assets = output.acoustic_scene_plan, output.resolved_assets
        assert plan.rir.cache_hit is cache_hit
        assert plan.rir.moving is moving
        assert (output.c0_audit_stems is not None) is capture
        snapshot = assets.rir.tobytes()
        np.testing.assert_array_equal(assets.rir, rir)
        assert not assets.rir.flags.writeable
        assert not np.shares_memory(assets.rir, dataset._rir_reuse_cache[1])
        rendered = replay(plan, assets, speech)
        np.testing.assert_array_equal(rendered.noisy_speech, output.noisy_speech)
        np.testing.assert_array_equal(rendered.clean_target, output.clean_target)
        forced = dataset._mix_acoustic_impl(
            speech, 8000, speech.shape[-1], speech.shape[-1],
            np.random.default_rng(99), False,
            forced_scene_plan=plan, forced_resolved_assets=assets,
        )
        np.testing.assert_array_equal(forced.noisy_speech, output.noisy_speech)
        np.testing.assert_array_equal(forced.clean_target, output.clean_target)
        expected = np.zeros(speech.shape[-1], dtype=np.float32)
        filters = rir / np.max(np.abs(rir)) if normalize else rir
        for start in range(0, speech.shape[-1], 32 if moving else speech.shape[-1]):
            row = min(start // 32, filters.shape[0] - 1)
            segment = speech[0, start:start + (32 if moving else speech.shape[-1])]
            convolution = np.convolve(segment, filters[row])
            count = min(len(convolution), len(expected) - start)
            expected[start:start + count] += convolution[:count]
        np.testing.assert_allclose(rendered.noisy_speech[0], expected, atol=1e-7, rtol=1e-6)
        if target == "dry":
            np.testing.assert_array_equal(rendered.clean_target, speech)
        for array in (rendered.noisy_speech, rendered.mix_reference, rendered.clean_target):
            assert not np.shares_memory(array, assets.rir)
            array[:] = 999.0
        np.testing.assert_array_equal(replay(plan, assets, speech).noisy_speech, output.noisy_speech)
        assert assets.rir.tobytes() == snapshot
        if previous_assets is not None:
            np.testing.assert_array_equal(previous_assets.rir, rir)
        previous_assets = assets
    dataset._rir_reuse_cache[1][:] = -99.0
    output.noisy_speech[:] = -100.0
    np.testing.assert_array_equal(previous_assets.rir, rir)
