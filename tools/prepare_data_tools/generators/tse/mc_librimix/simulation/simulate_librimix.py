#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from WeSep (https://github.com/wenet-e2e/wesep).
# Modifications Copyright 2026 use_train contributors.
# 修改：可注入后端、输入输出校验与拒绝覆盖；保留上游卷积和归一化算法。
"""生成早期响应和完整混响的多通道 Libri2Mix 音频。"""

import argparse
import math
from pathlib import Path

from .create_scenarios import (
    load_config, read_scenarios, scenario_seed, selected_splits,
    source_inventory, validate_config,
)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import soundfile as sf
except (ImportError, OSError):
    sf = None
try:
    from scipy import signal
except ImportError:
    signal = None


def _require_audio():
    if np is None or sf is None or signal is None:
        raise RuntimeError("numpy, soundfile and scipy are required for waveform generation")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Spatialize fixed Libri2Mix s1/s2 waveforms with gpuRIR.", allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--librimix-root", required=True)
    parser.add_argument("--scenario-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="all")
    return parser.parse_args(argv)


def rotate_array(local_positions, center, rotation_deg):
    angle = math.radians(rotation_deg)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return local_positions @ rotation.T + center


def load_source(path, sample_rate):
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, got {rate} Hz: {path}")
    if waveform.shape[1] != 1:
        raise ValueError(f"Expected a mono Libri2Mix source: {path}")
    return waveform[:, 0]


def convolve_sources(sources, rirs, output_length):
    images = np.empty((len(sources), output_length, rirs.shape[1]),
                      dtype=np.float32)
    for source_index, source in enumerate(sources):
        for channel in range(rirs.shape[1]):
            images[source_index, :, channel] = signal.fftconvolve(
                source,
                rirs[source_index, channel],
                mode="full",
            )[:output_length]
    return images


def early_rirs(full_rirs, sources, microphones, config):
    sample_rate = config["audio"]["sample_rate"]
    rir_cfg = config["rir"]
    early_samples = round(rir_cfg["early_window_ms"] * sample_rate / 1000)
    early = full_rirs.copy()
    for source_index, source in enumerate(sources):
        for channel, microphone in enumerate(microphones):
            if rir_cfg["preserve_time_of_flight"]:
                distance = np.linalg.norm(source - microphone)
                arrival = round(distance / rir_cfg["sound_speed"] *
                                sample_rate)
            else:
                arrival = 0
            early[source_index, channel, arrival + early_samples:] = 0.0
    return early


def expected_paths(base_dir, name, outputs):
    paths = []
    if outputs["early_source"]:
        paths.extend(
            [base_dir / "s1_early" / name, base_dir / "s2_early" / name])
    if outputs["early_mixture"]:
        paths.append(base_dir / "mix_early" / name)
    if outputs["reverb_source"]:
        paths.extend(
            [base_dir / "s1_reverb" / name, base_dir / "s2_reverb" / name])
    if outputs["reverb_mixture"]:
        paths.append(base_dir / "mix_reverb" / name)
    return paths


def generate_waveforms(config, librimix_root, scenario_manifest, output_root,
                       split="all", *, backend=None):
    validate_config(config)
    splits = selected_splits(config, split)
    scenarios = read_scenarios(scenario_manifest)
    if any(row["split"] not in config["dataset"]["splits"] for row in scenarios):
        raise ValueError("Manifest contains unconfigured splits")
    scenarios = [row for row in scenarios if row["split"] in splits]
    inventory = source_inventory(librimix_root, splits)
    if {(r["split"], r["mixture_id"]) for r in scenarios} != {
            (name, key) for name, keys in inventory.items() for key in keys}:
        raise ValueError("Scenario manifest and Libri2Mix source IDs differ")
    for row in scenarios:
        if row["version"] != config["version"] or int(row["scenario_seed"]) != scenario_seed(
                config["dataset"]["seed"], config["version"], row["split"], row["mixture_id"]):
            raise ValueError("Scenario version/seed does not match config")
    _require_audio()
    sample_rate = config["audio"]["sample_rate"]
    local_mics = np.asarray(config["array"]["mic_positions"], dtype=np.float64)
    outputs = config["outputs"]
    input_root, output_root = Path(librimix_root), Path(output_root)
    for row in scenarios:
        filename = f"{row['mixture_id']}.wav"
        for path in expected_paths(output_root / row["split"], filename, outputs):
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
        infos = [sf.info(input_root / row["split"] / source / filename) for source in ("s1", "s2")]
        if any(info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0 for info in infos):
            raise ValueError(f"Expected nonempty mono {sample_rate} Hz sources: {filename}")
        if infos[0].frames != infos[1].frames:
            raise ValueError(f"Libri2Mix sources are not aligned: {filename}")
    if backend is None:
        try:
            import gpuRIR as backend
        except (ImportError, OSError) as error:
            raise RuntimeError("gpuRIR is required to generate the spatial dataset") from error
    stats = {"mixtures": len(scenarios), "wav_files": 0, "sample_rate": sample_rate,
             "channels": len(local_mics), "splits": {name: len(keys) for name, keys in inventory.items()}}
    for row in scenarios:
        split = row["split"]
        mixture_id = row["mixture_id"]
        filename = f"{mixture_id}.wav"
        base_dir = output_root / split
        first = load_source(input_root / split / "s1" / filename, sample_rate)
        second = load_source(input_root / split / "s2" / filename, sample_rate)
        if first.shape != second.shape or not len(first):
            raise ValueError(f"Libri2Mix sources are not aligned or empty: {mixture_id}")
        source_audio = [first, second]
        if not all(np.isfinite(audio).all() for audio in source_audio):
            raise ValueError(f"Nonfinite source waveform: {mixture_id}")
        room = np.array([row["room_x"], row["room_y"], row["room_z"]], dtype=np.float64)
        center = np.array([row["array_x"], row["array_y"], row["array_z"]], dtype=np.float64)
        microphones = rotate_array(local_mics, center, float(row["array_rotation_deg"]))
        sources = np.array([
            [row["source_1_x"], row["source_1_y"], row["source_1_z"]],
            [row["source_2_x"], row["source_2_y"], row["source_2_z"]],
        ], dtype=np.float64)
        margin = config["room"]["boundary_margin"]
        if any(np.any(points < margin - 1e-7) or np.any(points > room - margin + 1e-7)
               for points in (microphones, sources)):
            raise ValueError(f"Scenario geometry violates room margins: {mixture_id}")
        rt60 = float(row["rt60"])
        beta = backend.beta_SabineEstimation(room, rt60)
        image_order = backend.t2n(rt60, room)
        full_rirs = backend.simulateRIR(room, beta, sources, microphones,
                                        image_order, rt60, sample_rate)
        full_rirs = np.asarray(full_rirs, dtype=np.float32)
        if (full_rirs.ndim != 3 or full_rirs.shape[:2] != (2, len(local_mics))
                or full_rirs.shape[2] == 0 or not np.isfinite(full_rirs).all()):
            raise ValueError("Backend returned invalid or nonfinite RIRs")
        early = early_rirs(full_rirs, sources, microphones, config)
        length = len(first)
        early_images = convolve_sources(source_audio, early, length)
        reverb_images = convolve_sources(source_audio, full_rirs, length)
        wav_outputs = []
        if outputs["early_source"]:
            wav_outputs.extend([
                (base_dir / "s1_early" / filename, early_images[0]),
                (base_dir / "s2_early" / filename, early_images[1]),
            ])
        if outputs["early_mixture"]:
            wav_outputs.append((base_dir / "mix_early" / filename, early_images.sum(axis=0)))
        if outputs["reverb_source"]:
            wav_outputs.extend([
                (base_dir / "s1_reverb" / filename, reverb_images[0]),
                (base_dir / "s2_reverb" / filename, reverb_images[1]),
            ])
        if outputs["reverb_mixture"]:
            wav_outputs.append((base_dir / "mix_reverb" / filename, reverb_images.sum(axis=0)))
        if any(not np.isfinite(audio).all() for _, audio in wav_outputs):
            raise ValueError(f"Nonfinite convolved waveform: {mixture_id}")
        peak = max(float(np.max(np.abs(audio))) for _, audio in wav_outputs)
        scale = 1.0 / peak if peak > 1.0 else 1.0
        for path, audio in wav_outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                sf.write(stream, audio * scale, sample_rate, format="WAV", subtype="PCM_16")
            info = sf.info(path)
            if (info.frames, info.samplerate, info.channels, info.subtype) != (
                    length, sample_rate, len(local_mics), "PCM_16"):
                raise ValueError(f"Generated WAV validation failed: {path}")
            stats["wav_files"] += 1
    return stats


def main(argv=None):
    args = parse_args(argv)
    stats = generate_waveforms(load_config(args.config), args.librimix_root,
                               args.scenario_manifest, args.output_root, args.split)
    print(f"Generated {stats['mixtures']} mixtures / {stats['wav_files']} WAV files")


if __name__ == "__main__":
    main()
