"""WSJ0-mix-style mc_sim fixture generation through the shared spatial scene seams."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from core.data.simulation.rir_provider import SpatialRIRResult, SpatialRIRTrackSet
from tools.prepare_data_tools.generators.se.generate_mc_sim_testset import (
    generate_mc_sim_testset,
)
from tools.shared.scp_io import read_scp_dict

pytestmark = pytest.mark.unit


class _FakeSpatialProvider:
    def sample_scene_tracks(self, scene_plan, **_kwargs):
        positions = np.asarray(scene_plan.array.positions_m, dtype=np.float64)
        source_count = len(scene_plan.actors)
        channels = len(positions)
        rirs = np.zeros((source_count, channels, 128), dtype=np.float32)
        for source in range(source_count):
            for channel in range(channels):
                rirs[source, channel, channel + 1] = 1.0 + 0.2 * source + 0.1 * channel
                rirs[source, channel, 40 + channel] = 0.03
        room = dict(scene_plan.room)
        geometry = {
            "positions_m": positions.tolist(),
            "reference_channel": scene_plan.array.reference_channel,
        }
        provenance = {
            "scene_uid": scene_plan.identity["scene_uid"],
            "source_order": [actor.actor_uid for actor in scene_plan.actors],
            "room": room,
        }
        full = SpatialRIRResult(rirs, geometry, provenance, "fake-rir-v1")
        early = SpatialRIRResult(rirs.copy(), geometry, provenance, "fake-rir-v1")
        return SpatialRIRTrackSet(full=full, early=early)


def _write_scp(root: Path, stem: str, uid: str, samples: np.ndarray) -> Path:
    wav_path = root / f"{stem}.wav"
    sf.write(wav_path, samples, 8000, subtype="FLOAT")
    scp_path = root / f"{stem}.scp"
    scp_path.write_text(f"{uid} 8000 {len(samples)} {wav_path}\n", encoding="utf-8")
    return scp_path


def test_mc_sim_generator_publishes_scp_manifest_and_multichannel_oracle_target(
    tmp_path, monkeypatch
):
    target = 0.12 * np.sin(2 * np.pi * 220 * np.arange(8000) / 8000).astype(np.float32)
    interference = 0.10 * np.sin(2 * np.pi * 330 * np.arange(8000) / 8000).astype(np.float32)
    target_scp = _write_scp(tmp_path, "target", "target-u", target)
    interference_scp = _write_scp(tmp_path, "interference", "interference-u", interference)
    config = tmp_path / "spatial.yaml"
    config.write_text(
        "data:\n"
        "  mode: dynamic\n"
        "  sample_rate: 8000\n"
        "  train:\n"
        "    max_duration_seconds: 1.0\n"
        "    simulation_cfg:\n"
        "      spatial_enabled: true\n"
        "      rir_backend: online\n"
        "      array_profile: linear_2ch_80mm\n"
        "      reference_channel: 0\n",
        encoding="utf-8",
    )
    output = tmp_path / "mc_sim"

    manifest_path = generate_mc_sim_testset(
        config_path=config,
        target_scp=target_scp,
        interference_scp=interference_scp,
        output_dir=output,
        num_samples=1,
        seed=23,
        duration_sec=1.0,
        sir_range_db=(0.0, 0.0),
        provider=_FakeSpatialProvider(),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mixture_entries = read_scp_dict(output / "mixture.scp")
    assert read_scp_dict(output / "noisy.scp") == mixture_entries
    mixture_path = Path(mixture_entries["mc_sim_00000000"])
    clean_path = Path(read_scp_dict(output / "clean.scp")["mc_sim_00000000"])
    target_image_path = Path(read_scp_dict(output / "target_image.scp")["mc_sim_00000000"])
    mixture, mixture_sr = sf.read(mixture_path, dtype="float32", always_2d=True)
    clean, clean_sr = sf.read(clean_path, dtype="float32")
    target_image, target_image_sr = sf.read(target_image_path, dtype="float32", always_2d=True)

    assert manifest["domain"] == "wsj0-mix-style-two-source-spatial-se"
    assert manifest["items"][0]["target_uid"] == "target-u"
    assert manifest["items"][0]["interference_uid"] == "interference-u"
    assert manifest["items"][0]["sir_realized_db"] == pytest.approx(0.0, abs=1e-4)
    assert mixture.shape == target_image.shape == (8000, 2)
    assert clean.shape == (8000,)
    assert mixture_sr == clean_sr == target_image_sr == 8000
    np.testing.assert_allclose(clean, target_image[:, 0], atol=2e-6)

    from core.data.collate import collate_fn
    from tasks.se.data import build_dataset

    offline_dataset = build_dataset(
        {
            "data": {
                "mode": "offline",
                "sample_rate": 8000,
                "train": {
                    "clean_scp": str(output / "clean.scp"),
                    "noisy_scp": str(output / "noisy.scp"),
                    "spatial_enabled": True,
                    "array_profile": "linear_2ch_80mm",
                    "reference_channel": 0,
                    "channel_order": ["mic_0", "mic_1"],
                },
            }
        },
        mode="train",
    )
    offline_batch = collate_fn([offline_dataset[0]], allow_spatial_noisy=True)
    assert offline_batch.clean.shape == (1, 8000)
    assert offline_batch.noisy.shape == (1, 2, 8000)

    from types import SimpleNamespace

    from evaluate import eval_pipeline_se

    def skip_model_inference(ds_tag, _args, _config, _checkpoint, _eval_root, summaries, _scps):
        summaries[ds_tag] = {"status": "done"}

    monkeypatch.setattr(eval_pipeline_se, "_run_se_dataset", skip_model_inference)
    monkeypatch.setattr(
        eval_pipeline_se,
        "run_standard_evaluation",
        lambda **_kwargs: None,
    )
    summaries = {}
    args = SimpleNamespace(
        mc_sim_testset_root=str(output),
        step="evaluate",
        effective_config={"stft": {"n_fft": 256, "hop_length": 128, "win_length": 256}},
        audio_quality=False,
        device="cpu",
    )
    eval_pipeline_se.process_mc_sim(
        args, config, tmp_path / "unused.ckpt", tmp_path / "eval", summaries
    )
    assert summaries["mc_sim"]["oracle_mvdr"]["status"] == "done"
    assert (tmp_path / "eval" / "mc_sim" / "oracle_mvdr" / "enhanced" / "inf.scp").is_file()
