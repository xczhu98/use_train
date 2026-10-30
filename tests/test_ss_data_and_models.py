import copy
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from core.stft import build_stft_istft
from core.data.speech_scene import scene_rng
from tasks.ss.data import SSDynamicMixingDataset, SSOfflineDataset, ss_collate_fn
from tasks.ss.models import build_model


@pytest.mark.unit
def test_ss_offline_jsonl_and_collate_contract(tmp_path):
    sr = 8000
    mixture = tmp_path / "mixture.wav"
    source_a = tmp_path / "source_a.wav"
    source_b = tmp_path / "source_b.wav"
    waveform = torch.zeros(800, dtype=torch.float32).numpy()
    sf.write(mixture, waveform, sr)
    sf.write(source_a, waveform, sr)
    sf.write(source_b, waveform, sr)
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "uid": "u0",
                "fs": sr,
                "length": 800,
                "mixture": mixture.name,
                "sources": [
                    {"source_id": "a", "path": source_a.name, "length": 800},
                    {"source_id": "b", "path": source_b.name, "length": 800},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSOfflineDataset(manifest, num_sources=2)
    batch = ss_collate_fn([dataset[0]])
    batch.validate()
    assert batch.mixture.shape == (1, 800)
    assert batch.sources.shape == (1, 2, 800)


@pytest.mark.unit
@pytest.mark.parametrize('strict_load', [True, False])
def test_ss_offline_rejects_manifest_header_mismatch(tmp_path, strict_load):
    sr = 8000
    paths = [tmp_path / name for name in ("mix.wav", "s1.wav", "s2.wav")]
    for path in paths:
        sf.write(path, torch.zeros(800).numpy(), sr)
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "uid": "bad",
                "fs": sr,
                "length": 799,
                "mixture": paths[0].name,
                "sources": [
                    {"source_id": "s1", "path": paths[1].name, "length": 800},
                    {"source_id": "s2", "path": paths[2].name, "length": 800},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSOfflineDataset(manifest, num_sources=2, strict_load=strict_load)
    with pytest.raises(ValueError, match="manifest metadata mismatch"):
        dataset[0]


@pytest.mark.unit
def test_ss_offline_train_crop_is_epoch_deterministic(tmp_path):
    sr = 8000
    waveform = torch.arange(800, dtype=torch.float32).numpy() / 800.0
    paths = [tmp_path / name for name in ("mix.wav", "s1.wav", "s2.wav")]
    for path in paths:
        sf.write(path, waveform, sr, subtype="FLOAT")
    manifest = tmp_path / "crop.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "uid": "crop",
                "fs": sr,
                "length": 800,
                "mixture": paths[0].name,
                "sources": [
                    {"source_id": "s1", "path": paths[1].name, "length": 800},
                    {"source_id": "s2", "path": paths[2].name, "length": 800},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSOfflineDataset(
        manifest,
        num_sources=2,
        max_duration_seconds=0.05,
        training=True,
        seed=9,
    )
    assert dataset.get_source_length() == [400]
    first = dataset[0]
    assert first["length"] == 400
    assert torch.equal(first["mixture"], first["sources"][0])
    dataset.set_epoch(1, base_seed=17)
    second = dataset[0]
    assert not torch.equal(first["mixture"], second["mixture"])
    dataset.set_epoch(1, base_seed=17)
    assert torch.equal(second["mixture"], dataset[0]["mixture"])

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        collate_fn=lambda batch: batch[0],
    )
    try:
        dataset.set_epoch(2, base_seed=17)
        worker_first = next(iter(loader))
        dataset.set_epoch(3, base_seed=17)
        worker_second = next(iter(loader))
        assert not torch.equal(worker_first["mixture"], worker_second["mixture"])
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


@pytest.mark.unit
def test_ss_dynamic_source_pool_is_deterministic(tmp_path):
    sr = 8000
    paths = []
    for index in range(2):
        path = tmp_path / f"source_{index}.wav"
        sf.write(path, torch.ones(800, dtype=torch.float32).numpy() * (index + 1), sr)
        paths.append(path)
    source_pool = tmp_path / "source_pool.jsonl"
    source_pool.write_text(
        "\n".join(
            json.dumps(
                {
                    "uid": f"s{index}",
                    "speaker_id": f"speaker-{index}",
                    "fs": sr,
                    "length": 800,
                    "path": path.name,
                }
            )
            for index, path in enumerate(paths)
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSDynamicMixingDataset(
        str(source_pool), num_sources=2, sample_rate=sr, segment_seconds=0.1, seed=7
    )
    other = SSDynamicMixingDataset(
        str(source_pool), num_sources=2, sample_rate=sr, segment_seconds=0.1, seed=7
    )
    assert torch.equal(dataset[0]["mixture"], other[0]["mixture"])
    assert dataset[0]["sources"].__len__() == 2
    dataset.set_epoch(3, base_seed=11)
    other.set_epoch(3, base_seed=11)
    assert dataset[0]["source_ids"] == other[0]["source_ids"]


@pytest.mark.unit
def test_ss_mono_timeline_uses_dedicated_named_stream(tmp_path, monkeypatch):
    """mono SS 也必须在 shared renderer 前使用 timeline 子流 303。"""
    import tasks.ss.data.dynamic_dataset as dynamic_dataset

    sr = 8000
    paths = []
    for index in range(2):
        path = tmp_path / f"timeline_source_{index}.wav"
        sf.write(path, torch.ones(800, dtype=torch.float32).numpy() * (index + 1), sr)
        paths.append(path)
    source_pool = tmp_path / "timeline_pool.jsonl"
    source_pool.write_text(
        "\n".join(
            json.dumps(
                {
                    "uid": f"timeline-{index}",
                    "speaker_id": f"timeline-speaker-{index}",
                    "fs": sr,
                    "length": 800,
                    "path": path.name,
                }
            )
            for index, path in enumerate(paths)
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSDynamicMixingDataset(
        str(source_pool),
        num_sources=2,
        sample_rate=sr,
        segment_seconds=0.1,
        seed=7,
        simulation_cfg={
            "prob_target_only": 1.0,
            "prob_noise_only": 0.0,
            "prob_target_noise": 0.0,
            "prob_timeline": 1.0,
            "overlap_ratio_range": [0.5, 0.5],
            "timeline_mode_weights": {
                name: float(name == "head")
                for name in ("head", "tail", "middle", "crossing", "containment")
            },
        },
    )
    original = dynamic_dataset.place_scene_sources
    seen = {"called": False}

    def wrapped(source_waveforms, temporal, rng):
        probe = np.random.default_rng()
        probe.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
        expected = scene_rng(dataset, 0, 303)
        assert probe.random() == expected.random()
        seen["called"] = True
        return original(source_waveforms, temporal, rng)

    monkeypatch.setattr(dynamic_dataset, "place_scene_sources", wrapped)
    sample = dataset.get_sample_with_trace(0)
    assert seen["called"]
    assert sample.trace.metadata_dict()["scene"]["timeline"]


@pytest.mark.unit
def test_ss_dynamic_rejects_empty_decoded_source(tmp_path):
    sr = 8000
    empty = tmp_path / "empty.wav"
    valid = tmp_path / "valid.wav"
    sf.write(empty, torch.zeros(0).numpy(), sr)
    sf.write(valid, torch.ones(800).numpy(), sr)
    source_pool = tmp_path / "pool.jsonl"
    source_pool.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "uid": "empty",
                        "speaker_id": "speaker-empty",
                        "fs": sr,
                        "length": 1,
                        "path": empty.name,
                    }
                ),
                json.dumps(
                    {
                        "uid": "valid",
                        "speaker_id": "speaker-valid",
                        "fs": sr,
                        "length": 800,
                        "path": valid.name,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SSDynamicMixingDataset(
        str(source_pool), num_sources=2, sample_rate=sr, segment_seconds=0.1, seed=1
    )
    with pytest.raises(ValueError, match="manifest metadata mismatch|empty"):
        dataset[0]


@pytest.mark.unit
def test_ss_dynamic_samples_distinct_speakers(tmp_path):
    sr = 8000
    rows = []
    uid_to_speaker = {}
    for uid, speaker_id, value in (
        ("a0", "speaker-a", 0.1),
        ("a1", "speaker-a", 0.2),
        ("b0", "speaker-b", 0.3),
    ):
        path = tmp_path / f"{uid}.wav"
        sf.write(path, torch.full((800,), value).numpy(), sr)
        uid_to_speaker[uid] = speaker_id
        rows.append(
            json.dumps(
                {
                    "uid": uid,
                    "speaker_id": speaker_id,
                    "fs": sr,
                    "length": 800,
                    "path": path.name,
                }
            )
        )
    source_pool = tmp_path / "speaker_pool.jsonl"
    source_pool.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = SSDynamicMixingDataset(
        str(source_pool), num_sources=2, sample_rate=sr, segment_seconds=0.1
    )
    for index in range(len(dataset)):
        selected = dataset[index]["source_ids"]
        assert len({uid_to_speaker[uid] for uid in selected}) == 2


@pytest.mark.unit
def test_ss_dynamic_rejects_insufficient_distinct_speakers(tmp_path):
    sr = 8000
    rows = []
    for index in range(2):
        path = tmp_path / f"same_{index}.wav"
        sf.write(path, torch.ones(800).numpy(), sr)
        rows.append(
            json.dumps(
                {
                    "uid": f"same-{index}",
                    "speaker_id": "same-speaker",
                    "fs": sr,
                    "length": 800,
                    "path": path.name,
                }
            )
        )
    source_pool = tmp_path / "same_speaker.jsonl"
    source_pool.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="distinct speakers"):
        SSDynamicMixingDataset(
            str(source_pool), num_sources=2, sample_rate=sr, segment_seconds=0.1
        )


@pytest.mark.unit
def test_tiger_external_stft_output_uses_official_640_geometry():
    stft_cfg = {
        "n_fft": 640,
        "hop_length": 160,
        "win_length": 640,
        "window": "hann",
        "center": True,
    }
    model = build_model(
        {
            "type": "TIGER",
            "args": {
                "out_channels": 8,
                "in_channels": 16,
                "num_blocks": 1,
                "upsampling_depth": 2,
                "win": 640,
                "stride": 160,
                "sample_rate": 16000,
            },
        },
        stft_cfg=stft_cfg,
        num_sources=2,
    ).eval()
    stft = build_stft_istft(stft_cfg)
    mixture = torch.randn(1, 1280)
    with torch.no_grad():
        output = model(stft(mixture), torch.tensor([1280]))
    assert output["est_sources_spectra"].shape[:2] == (1, 2)
    assert output["est_sources_spectra"].shape[2] == 321
    with pytest.raises(RuntimeError, match="external-STFT contract"):
        model.forward_from_spectrum(
            torch.view_as_complex(stft(mixture).contiguous()), mixture.shape[-1]
        )


@pytest.mark.unit
def test_ss_model_builder_rejects_num_source_conflict():
    with pytest.raises(ValueError, match="conflicts"):
        build_model(
            {"type": "TDANet", "args": {"num_sources": 3}},
            num_sources=2,
        )


@pytest.mark.unit
def test_tdanet_rejects_input_too_short_for_any_output_frame():
    model = build_model(
        {
            "type": "TDANet",
            "args": {
                "out_channels": 8,
                "in_channels": 16,
                "num_blocks": 1,
                "upsampling_depth": 2,
                "sample_rate": 8000,
            },
        },
        num_sources=2,
    ).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 400))["est_sources"].shape == (1, 2, 400)
        with pytest.raises(RuntimeError, match="too short to produce any output frame"):
            model(torch.zeros(1, 0))
