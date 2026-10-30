"""真实多任务训练入口 CPU smoke：数据、模型、loss、优化器、验证和 checkpoint。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from core.data.simulation.scene_manifest import build_track_identity

ROOT = Path(__file__).resolve().parents[1]
SR = 16000

pytestmark = pytest.mark.integration


def _audio(path: Path, sample_rate: int, length: int, frequency: float) -> None:
    time = np.arange(length, dtype=np.float32) / sample_rate
    waveform = (0.05 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")


def _scp(path: Path, rows: list[tuple[str, Path, int, int]]) -> None:
    path.write_text(
        "".join(f"{uid} {sample_rate} {length} {audio_path}\n" for uid, audio_path, sample_rate, length in rows),
        encoding="utf-8",
    )


def _common(exp_path: Path) -> dict:
    return {
        "meta": {
            "seed": 7,
            "use_amp": False,
            "use_ema": False,
            "compile_model": False,
        },
        "trainer": {
            "epochs": 1,
            "exp_path": str(exp_path),
            "accumulation_steps": 1,
            "log_interval": 1,
            "max_val_batches": 1,
            "validation_epoch_interval": 1,
            "validation_steps_interval": 0,
            "save_checkpoint_interval": 1,
            "save_curve_interval": 0,
            "save_top_k": 1,
            "val_metrics": {"enabled": False},
        },
        "distributed": {"backend": "gloo", "find_unused_parameters": False},
    }


def _se_config(root: Path, exp_path: Path) -> dict:
    clean = root / "se_clean.wav"
    noisy = root / "se_noisy.wav"
    _audio(clean, SR, 2048, 220.0)
    _audio(noisy, SR, 2048, 330.0)
    clean_scp = root / "se_clean.scp"
    noisy_scp = root / "se_noisy.scp"
    row = [("se0", clean, SR, 2048)]
    _scp(clean_scp, row)
    _scp(noisy_scp, [("se0", noisy, SR, 2048)])
    cfg = _common(exp_path)
    cfg.update(
        {
            "stft": {"n_fft": 512, "hop_length": 256, "win_length": 512, "window": "hann", "center": True},
            "model": {"type": "GTCRN", "args": {}},
            "data": {
                "mode": "offline",
                "sample_rate": SR,
                "train": {"clean_scp": str(clean_scp), "noisy_scp": str(noisy_scp)},
                "validation": {"clean_scp": str(clean_scp), "noisy_scp": str(noisy_scp)},
                "dataloader": {"batch_size": 1, "num_workers": 0},
            },
            "loss": {"type": "L1WavLoss"},
            "optimizer": {"type": "Adam", "args": {"lr": 0.001}},
            "scheduler": {"type": "ExponentialLR", "args": {"gamma": 1.0}},
        }
    )
    cfg["trainer"]["type"] = "SETrainer"
    return cfg


def _aec_config(root: Path, exp_path: Path) -> dict:
    near = root / "aec_near.wav"
    far = root / "aec_far.wav"
    mic = root / "aec_mic.wav"
    _audio(near, SR, 2048, 220.0)
    _audio(far, SR, 2048, 440.0)
    _audio(mic, SR, 2048, 660.0)
    paths = {}
    for name, audio_path in (("near", near), ("far", far), ("mic", mic)):
        manifest = root / f"aec_{name}.scp"
        _scp(manifest, [("aec0", audio_path, SR, 2048)])
        paths[name] = manifest
    cfg = _common(exp_path)
    cfg.update(
        {
            "stft": {"n_fft": 320, "hop_length": 160, "win_length": 320, "window": "hann", "center": True},
            "model": {"type": "DeepVQE", "args": {"variant": "small", "align_delay": 1}},
            "data": {
                "mode": "offline_aec",
                "sample_rate": SR,
                "task": "aec",
                "train": {"mic_scp": str(paths["mic"]), "farend_scp": str(paths["far"]), "nearend_scp": str(paths["near"])},
                "validation": {"mic_scp": str(paths["mic"]), "farend_scp": str(paths["far"]), "nearend_scp": str(paths["near"])},
                "dataloader": {"batch_size": 1, "num_workers": 0},
            },
            "loss": {"type": "L1WavLoss"},
            "optimizer": {"type": "Adam", "args": {"lr": 0.001}},
            "scheduler": {"type": "ExponentialLR", "args": {"gamma": 1.0}},
        }
    )
    cfg["trainer"]["type"] = "AECTrainer"
    return cfg


def _ss_config(root: Path, exp_path: Path) -> dict:
    mixture = root / "ss_mix.wav"
    source_a = root / "ss_source_a.wav"
    source_b = root / "ss_source_b.wav"
    for path, frequency in ((mixture, 220.0), (source_a, 220.0), (source_b, 440.0)):
        _audio(path, SR, 1024, frequency)
    manifest = root / "ss_samples.jsonl"
    record = {
        "uid": "ss0",
        "fs": SR,
        "length": 1024,
        "mixture": str(mixture),
        "sources": [
            {"source_id": "a", "path": str(source_a), "length": 1024},
            {"source_id": "b", "path": str(source_b), "length": 1024},
        ],
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    cfg = _common(exp_path)
    cfg.update(
        {
            "model": {
                "type": "TDANet",
                "args": {
                    "out_channels": 8,
                    "in_channels": 16,
                    "num_blocks": 1,
                    "upsampling_depth": 2,
                    "enc_kernel_size": 2,
                    "sample_rate": SR,
                },
            },
            "data": {
                "mode": "offline_ss",
                "sample_rate": SR,
                "train": {"manifest": str(manifest), "max_duration_seconds": 0.064},
                "validation": {"manifest": str(manifest), "max_duration_seconds": 0.064},
                "dataloader": {"batch_size": 1, "num_workers": 0},
            },
            "ss": {"num_sources": 2},
            "loss": {"type": "PITLoss", "args": {"pairwise_loss": "PairwiseNegSISDRLoss", "assignment": "exhaustive", "reduction": "mean"}},
            "optimizer": {"type": "Adam", "args": {"lr": 0.001}},
            "scheduler": {"type": "ExponentialLR", "args": {"gamma": 1.0}},
        }
    )
    cfg["meta"]["allow_no_stft"] = True
    cfg["trainer"]["type"] = "SSTrainer"
    return cfg


def _ssl_config(root: Path, exp_path: Path) -> dict:
    sr = 8000
    length = 128
    speech_paths = []
    for index, frequency in enumerate((240.0, 420.0)):
        path = root / f"ssl_speech_{index}.wav"
        _audio(path, sr, length, frequency)
        speech_paths.append(path)
    noise = root / "ssl_noise.wav"
    _audio(noise, sr, length, 700.0)
    speech_scp = root / "ssl_speech.scp"
    noise_scp = root / "ssl_noise.scp"
    _scp(speech_scp, [(f"source_{i}", path, sr, length) for i, path in enumerate(speech_paths)])
    _scp(noise_scp, [("noise0", noise, sr, length)])
    rir_entries = {}
    track_identities = {}
    for index, delay in enumerate((0, 1)):
        rir = np.zeros((2, 8), dtype=np.float32)
        rir[0, delay] = 1.0
        rir[1, delay + 1] = 0.8
        rir_path = root / f"ssl_rir_{index}.npy"
        direct_path = root / f"ssl_direct_{index}.npy"
        np.save(rir_path, rir)
        np.save(direct_path, rir)
        rir_entries[f"source_{index}"] = {"full": str(rir_path), "direct": str(direct_path), "early": str(direct_path)}
        track_identities[f"source_{index}"] = {
            "full": build_track_identity(rir_path),
            "direct": build_track_identity(direct_path),
            "early": build_track_identity(direct_path),
        }
    rir_manifest = root / "ssl_spatial_rir.json"
    rir_manifest.write_text(
        json.dumps(
            {
                "entries": rir_entries,
                "metadata": {
                    "schema_version": 1,
                    "sample_rate": sr,
                    "scene_uid": "entry-smoke",
                    "actor_order": ["source_0", "source_1"],
                    "room": {"room_uid": "smoke-room", "room_dim_m": [6, 6, 3]},
                    "actor_positions_m": {"source_0": [4.0, 3.0, 2.25], "source_1": [3.0, 4.0, 2.25]},
                    "generator_version": "entry-smoke-v1",
                    "track_identities": track_identities,
                    "geometry": {"positions_m": [[2.96, 3.0, 0.75], [3.04, 3.0, 0.75]], "reference_channel": 0, "array_profile": "linear_2ch_80mm"},
                },
            }
        ),
        encoding="utf-8",
    )
    simulation = {
        "rir_backend": "offline",
        "spatial_enabled": True,
        "array_profile": "linear_2ch_80mm",
        "reference_channel": 0,
        "prob_reverberation": 1.0,
        "prob_channel_enabled": False,
        "noisy_post_mix_degradations_enabled": False,
        "peak_normalization_mode": "off",
    }
    cfg = _common(exp_path)
    cfg.update(
        {
            "meta": {**cfg["meta"], "allow_no_stft": True},
            "stft": {"n_fft": 16, "hop_length": 8, "win_length": 16, "window": "hamming", "center": False},
            "model": {"type": "FixedArrayIPDNet", "args": {"input_size": 1, "hidden_size": 4, "max_track": 2, "n_fft": 16}},
            "data": {
                "mode": "dynamic_ssl",
                "sample_rate": sr,
                "train": {"speech_scp": str(speech_scp), "noise_scp": str(noise_scp), "rir_scp": str(rir_manifest), "segment_seconds": length / sr, "tracks": 2, "freqs": 9, "simulation_cfg": simulation},
                "validation": {"speech_scp": str(speech_scp), "noise_scp": str(noise_scp), "rir_scp": str(rir_manifest), "segment_seconds": length / sr, "tracks": 2, "freqs": 9, "simulation_cfg": simulation},
                "dataloader": {"batch_size": 1, "num_workers": 0},
            },
            "loss": {"type": "SSLSpatialMSELoss", "args": {"allow_no_stft": True}},
            "optimizer": {"type": "Adam", "args": {"lr": 0.001}},
            "scheduler": {"type": "ExponentialLR", "args": {"gamma": 1.0}},
        }
    )
    cfg["trainer"]["type"] = "SSLTrainer"
    return cfg


def _tse_config(root: Path, exp_path: Path) -> dict:
    mix = root / "tse_mix.wav"
    target = root / "tse_target.wav"
    mix2 = root / "tse_mix2.wav"
    target2 = root / "tse_target2.wav"
    enroll = root / "tse_enroll.wav"
    for path, frequency in ((mix, 220.0), (target, 220.0), (mix2, 240.0), (target2, 240.0), (enroll, 220.0)):
        _audio(path, SR, 2048 if path != enroll else 3200, frequency)
    samples = root / "tse_samples.jsonl"
    samples.write_text(
        "".join(
            json.dumps({"key": key, "spk": ["speaker0"], "mix": {"default": [str(m)]}, "src": {"speaker0": [str(t)]}}) + "\n"
            for key, m, t in (("tse0", mix, target), ("tse1", mix2, target2))
        ),
        encoding="utf-8",
    )
    cue_resource = root / "tse_cues.json"
    cue_resource.write_text(json.dumps({"speaker0": [{"utt_id": "enroll0", "path": str(enroll)}]}), encoding="utf-8")
    cues = root / "tse_cues.yaml"
    cues.write_text(yaml.safe_dump({"cues": {"audio": {"type": "raw", "guaranteed": True, "scope": "speaker", "policy": {"type": "fixed", "key": "spk_id", "resource": str(cue_resource)}}}}), encoding="utf-8")
    cfg = _common(exp_path)
    cfg.update(
        {
            "stft": {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann", "center": True},
            "model": {"type": "TSE_BSRNN_SPK", "args": {"sr": SR, "feature_dim": 32, "num_repeat": 2, "causal": False, "joint_training": True, "features": {"spkemb": {"enabled": True}}}},
            "data": {
                "mode": "offline_tse",
                "sample_rate": SR,
                "train": {
                    "samples_jsonl": str(samples),
                    "cues_yaml": str(cues),
                    "manifest_format": "jsonl",
                    "path_base": "manifest_relative",
                    "target_selection": "first",
                    "mix_variant": "default",
                    "cue_schema": "realtse_raw",
                    "enabled_cues": ["audio"],
                    "max_duration_seconds": 0.128,
                },
                "validation": {
                    "samples_jsonl": str(samples),
                    "cues_yaml": str(cues),
                    "manifest_format": "jsonl",
                    "path_base": "manifest_relative",
                    "target_selection": "first",
                    "mix_variant": "default",
                    "cue_schema": "realtse_raw",
                    "enabled_cues": ["audio"],
                    "max_duration_seconds": 0.128,
                },
                "dataloader": {"batch_size": 2, "num_workers": 0},
            },
            "loss": {"type": "CompositeLoss", "components": [{"type": "SISNRLoss", "weight": 1.0}]},
            "optimizer": {"type": "Adam", "args": {"lr": 0.001}},
            "scheduler": {"type": "ExponentialLR", "args": {"gamma": 1.0}},
        }
    )
    cfg["trainer"]["type"] = "TSETrainer"
    return cfg


_BUILDERS = {
    "se": _se_config,
    "aec": _aec_config,
    "ss": _ss_config,
    "ssl": _ssl_config,
    "tse": _tse_config,
}


@pytest.mark.parametrize("task", ["se", "aec", "ss", "ssl", "tse"])
def test_real_multitask_entry_cpu_epoch(task: str, tmp_path: Path) -> None:
    task_root = tmp_path / task
    task_root.mkdir()
    exp_path = task_root / "experiment"
    config_path = task_root / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(_BUILDERS[task](task_root, exp_path), sort_keys=False), encoding="utf-8")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    result = subprocess.run(
        [sys.executable, "train.py", "-C", str(config_path)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=110,
    )
    if result.returncode != 0:
        pytest.fail(f"{task} real train.py entry failed (exit={result.returncode}):\n{result.stdout}")
    checkpoints = list((exp_path / "checkpoints").glob("*/latest_model.tar"))
    assert len(checkpoints) == 1, f"{task} did not publish one checkpoint\n{result.stdout}"
    assert checkpoints[0].stat().st_size > 0
    assert "Training complete." in result.stdout
    assert "[Val] epoch 1" in result.stdout
