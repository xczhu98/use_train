#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from WeSep (https://github.com/wenet-e2e/wesep).
# Modifications Copyright 2026 use_train contributors.
# 修改：包内 API、严格配置校验和独立命令行；保留上游采样算法。
"""为现有 Libri2Mix 音频创建确定性房间场景。"""

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
import math
from numbers import Real
from pathlib import Path
import re

try:
    import numpy as np
except ImportError:
    np = None


def _require_numpy():
    if np is None:
        raise RuntimeError("numpy is required to sample spatial scenarios")


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{label} requires exactly these fields: {names}")


def _number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{label} is outside its valid range")
    return float(value)


def _range(value, label, *, positive=False, nonnegative=False, vector=False):
    _keys(value, "min max", label)
    bounds = []
    for bound in ("min", "max"):
        values = value[bound] if vector else [value[bound]]
        if not isinstance(values, (list, tuple)) or len(values) != (3 if vector else 1):
            raise ValueError(f"{label}.{bound} must have three dimensions")
        bounds.append([_number(v, label, positive=positive, nonnegative=nonnegative)
                       for v in values])
    if any(low > high for low, high in zip(*bounds)):
        raise ValueError(f"{label}.min must not exceed max")
    return bounds


def validate_config(config):
    _keys(config, "version dataset room array sources audio rir outputs", "config")
    if not isinstance(config["version"], str) or not config["version"].strip():
        raise ValueError("version must be a nonempty string")
    dataset = config["dataset"]
    _keys(dataset, "seed splits max_sampling_attempts", "dataset")
    for key, minimum in (("seed", 0), ("max_sampling_attempts", 1)):
        if type(dataset[key]) is not int or dataset[key] < minimum:
            raise ValueError(f"dataset.{key} must be an integer >= {minimum}")
    splits = dataset["splits"]
    if (not isinstance(splits, list) or not splits
            or any(s not in ("train-100", "dev", "test") for s in splits)
            or len(set(splits)) != len(splits)):
        raise ValueError("dataset.splits must be unique train-100/dev/test names")
    room, array, sources = config["room"], config["array"], config["sources"]
    _keys(room, "dimensions rt60 boundary_margin", "room")
    dimensions = _range(room["dimensions"], "room.dimensions", positive=True, vector=True)
    _range(room["rt60"], "room.rt60", positive=True)
    margin = _number(room["boundary_margin"], "room.boundary_margin", nonnegative=True)
    _keys(array, "mic_positions height rotation_deg", "array")
    positions = array["mic_positions"]
    if not isinstance(positions, list) or not positions:
        raise ValueError("array.mic_positions must be a nonempty Nx3 array")
    for position in positions:
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError("array.mic_positions must be a nonempty Nx3 array")
        for coordinate in position:
            _number(coordinate, "array.mic_positions")
    if len({tuple(p) for p in positions}) != len(positions):
        raise ValueError("array.mic_positions must be distinct")
    height = _range(array["height"], "array.height", positive=True)
    _range(array["rotation_deg"], "array.rotation_deg")
    radius = max(math.sqrt(sum(x * x for x in p)) for p in positions)
    if any(d <= 2 * (margin + radius) for d in dimensions[0][:2]):
        raise ValueError("room dimensions do not fit the microphone array and margin")
    if height[0][0] + min(p[2] for p in positions) < margin or (
            height[1][0] + max(p[2] for p in positions) > dimensions[0][2] - margin):
        raise ValueError("array.height places microphones outside the room boundary")
    _keys(sources, "num_speakers azimuth_deg horizontal_distance height min_azimuth_difference_deg", "sources")
    if type(sources["num_speakers"]) is not int or sources["num_speakers"] != 2:
        raise ValueError("sources.num_speakers must be exactly 2")
    azimuth = _range(sources["azimuth_deg"], "sources.azimuth_deg")
    span = azimuth[1][0] - azimuth[0][0]
    if span > 360:
        raise ValueError("sources.azimuth_deg must span at most 360 degrees")
    _range(sources["horizontal_distance"], "sources.horizontal_distance", positive=True)
    source_height = _range(sources["height"], "sources.height", positive=True)
    if source_height[0][0] < margin or source_height[1][0] > dimensions[0][2] - margin:
        raise ValueError("sources.height exceeds the room boundary")
    separation = _number(sources["min_azimuth_difference_deg"], "sources.min_azimuth_difference_deg", nonnegative=True)
    if separation > min(180, span):
        raise ValueError("sources.min_azimuth_difference_deg is infeasible")
    _keys(config["audio"], "sample_rate", "audio")
    rate = config["audio"]["sample_rate"]
    if type(rate) is not int or rate <= 0:
        raise ValueError("audio.sample_rate must be a positive integer")
    rir = config["rir"]
    _keys(rir, "early_window_ms preserve_time_of_flight sound_speed", "rir")
    _number(rir["early_window_ms"], "rir.early_window_ms", positive=True)
    if round(rir["early_window_ms"] * rate / 1000) < 1:
        raise ValueError("rir.early_window_ms must cover at least one sample")
    _number(rir["sound_speed"], "rir.sound_speed", positive=True)
    if type(rir["preserve_time_of_flight"]) is not bool:
        raise ValueError("rir.preserve_time_of_flight must be boolean")
    _keys(config["outputs"], "early_source early_mixture reverb_source reverb_mixture", "outputs")
    if any(type(v) is not bool for v in config["outputs"].values()) or not any(config["outputs"].values()):
        raise ValueError("outputs must be booleans with at least one enabled output")
    return config


def load_config(path):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read the explicit simulation config") from error
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return validate_config(config)


def selected_splits(config, split):
    splits = config["dataset"]["splits"]
    if split == "all":
        return tuple(splits)
    if split not in splits:
        raise ValueError(f"Unknown or unconfigured split: {split}")
    return (split,)

FIELDS = [
    "version",
    "split",
    "mixture_id",
    "speaker_1",
    "speaker_2",
    "scenario_seed",
    "room_x",
    "room_y",
    "room_z",
    "rt60",
    "array_x",
    "array_y",
    "array_z",
    "array_rotation_deg",
    "source_1_x",
    "source_1_y",
    "source_1_z",
    "source_1_azimuth_deg",
    "source_1_elevation_deg",
    "source_1_horizontal_distance",
    "source_1_distance",
    "source_2_x",
    "source_2_y",
    "source_2_z",
    "source_2_azimuth_deg",
    "source_2_elevation_deg",
    "source_2_horizontal_distance",
    "source_2_distance",
    "azimuth_difference_deg",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create fixed spatial scenarios for Libri2Mix.", allow_abbrev=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--librimix-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="all")
    return parser.parse_args(argv)


@contextmanager
def open_csv(path, mode):
    if str(path).endswith(".gz"):
        if mode in ("w", "x"):
            with open(path, mode + "b") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, newline="", encoding="utf-8") as stream:
                        yield stream
            return
        with gzip.open(path, mode + "t", newline="", encoding="utf-8-sig") as stream:
            yield stream
        return
    with open(path, mode, newline="", encoding="utf-8-sig" if mode == "r" else "utf-8") as stream:
        yield stream


def speaker_ids(mixture_id):
    if not isinstance(mixture_id, str) or not re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+-[0-9]+", mixture_id):
        raise ValueError(f"Expected a two-speaker Libri2Mix ID, got: {mixture_id}")
    speakers = tuple(part.split("-")[0] for part in mixture_id.split("_"))
    if int(speakers[0]) == int(speakers[1]):
        raise ValueError(f"Libri2Mix speakers must be distinct: {mixture_id}")
    return speakers


def scenario_seed(global_seed, version, split, mixture_id):
    value = f"{version}:{global_seed}:{split}:{mixture_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def angular_difference(first, second):
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def sample_scenario(config, split, mixture_id):
    _require_numpy()
    version = config["version"]
    seed = scenario_seed(config["dataset"]["seed"], version, split, mixture_id)
    rng = np.random.default_rng(seed)
    room_cfg = config["room"]
    array_cfg = config["array"]
    source_cfg = config["sources"]
    margin = float(room_cfg["boundary_margin"])
    array_radius = np.linalg.norm(
        np.asarray(array_cfg["mic_positions"], dtype=np.float64),
        axis=1,
    ).max()

    # 采样完整房间，使声源和麦克风均处于边界内。
    for _ in range(config["dataset"]["max_sampling_attempts"]):
        room = rng.uniform(room_cfg["dimensions"]["min"],
                           room_cfg["dimensions"]["max"])
        rt60 = rng.uniform(room_cfg["rt60"]["min"], room_cfg["rt60"]["max"])
        xy_margin = margin + array_radius
        array_center = np.array([
            rng.uniform(xy_margin, room[0] - xy_margin),
            rng.uniform(xy_margin, room[1] - xy_margin),
            rng.uniform(array_cfg["height"]["min"],
                        array_cfg["height"]["max"]),
        ])
        rotation = rng.uniform(array_cfg["rotation_deg"]["min"],
                               array_cfg["rotation_deg"]["max"])

        sources = []
        for _ in range(source_cfg["num_speakers"]):
            azimuth = rng.uniform(source_cfg["azimuth_deg"]["min"],
                                  source_cfg["azimuth_deg"]["max"])
            horizontal = rng.uniform(
                source_cfg["horizontal_distance"]["min"],
                source_cfg["horizontal_distance"]["max"],
            )
            height = rng.uniform(source_cfg["height"]["min"],
                                 source_cfg["height"]["max"])
            global_angle = math.radians(azimuth + rotation)
            position = np.array([
                array_center[0] + horizontal * math.cos(global_angle),
                array_center[1] + horizontal * math.sin(global_angle),
                height,
            ])
            vertical = height - array_center[2]
            elevation = math.degrees(math.atan2(vertical, horizontal))
            distance = math.hypot(horizontal, vertical)
            sources.append(
                (position, azimuth, elevation, horizontal, distance))

        inside = all(
            np.all(item[0] >= margin) and np.all(item[0] <= room - margin)
            for item in sources)
        separation = angular_difference(sources[0][1], sources[1][1])
        if inside and separation >= source_cfg["min_azimuth_difference_deg"]:
            break
    else:
        raise RuntimeError(f"Failed to sample a valid room for {mixture_id}")

    spk1, spk2 = speaker_ids(mixture_id)
    row = {
        "version": version,
        "split": split,
        "mixture_id": mixture_id,
        "speaker_1": spk1,
        "speaker_2": spk2,
        "scenario_seed": seed,
        "room_x": room[0],
        "room_y": room[1],
        "room_z": room[2],
        "rt60": rt60,
        "array_x": array_center[0],
        "array_y": array_center[1],
        "array_z": array_center[2],
        "array_rotation_deg": rotation,
        "azimuth_difference_deg": separation,
    }
    for index, item in enumerate(sources, 1):
        position, azimuth, elevation, horizontal, distance = item
        row.update({
            f"source_{index}_x": position[0],
            f"source_{index}_y": position[1],
            f"source_{index}_z": position[2],
            f"source_{index}_azimuth_deg": azimuth,
            f"source_{index}_elevation_deg": elevation,
            f"source_{index}_horizontal_distance": horizontal,
            f"source_{index}_distance": distance,
        })
    return row


def source_inventory(librimix_root, splits):
    root = Path(librimix_root)
    inventory = {}
    for split in splits:
        directories = [root / split / source for source in ("s1", "s2")]
        for directory in directories:
            if not directory.is_dir():
                raise FileNotFoundError(directory)
        names = [{p.name for p in directory.glob("*.wav") if p.is_file()}
                 for directory in directories]
        if not names[0] or names[0] != names[1]:
            raise ValueError(f"Empty or inconsistent s1/s2 source IDs in {root / split}")
        inventory[split] = sorted(Path(name).stem for name in names[0])
        for mixture_id in inventory[split]:
            speaker_ids(mixture_id)
    return inventory


def build_scenarios(config, librimix_root, split="all"):
    validate_config(config)
    inventory = source_inventory(librimix_root, selected_splits(config, split))
    _require_numpy()
    return [sample_scenario(config, name, mixture_id)
            for name, mixtures in inventory.items() for mixture_id in mixtures]


def validate_rows(rows):
    if not rows:
        raise ValueError("Scenario manifest must not be empty")
    seen = set()
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Scenario fields do not match the upstream CSV schema")
        if row["split"] not in ("train-100", "dev", "test"):
            raise ValueError(f"Invalid scenario split: {row['split']}")
        speakers = speaker_ids(row["mixture_id"])
        if tuple(str(row[f"speaker_{i}"]) for i in (1, 2)) != speakers:
            raise ValueError("Scenario speaker IDs do not match mixture_id")
        key = (row["split"], row["mixture_id"])
        if key in seen:
            raise ValueError(f"Duplicate scenario: {key}")
        seen.add(key)
        if not isinstance(row["version"], str) or not row["version"].strip():
            raise ValueError("Scenario version must be nonempty")
        if not re.fullmatch(r"[0-9]+", str(row["scenario_seed"])):
            raise ValueError("Invalid scenario_seed")
        if int(row["scenario_seed"]) >= 2**64:
            raise ValueError("scenario_seed must fit uint64")
        for field in FIELDS[6:]:
            try:
                number = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid scenario value for {field}") from error
            if isinstance(row[field], bool) or not math.isfinite(number):
                raise ValueError(f"Nonfinite scenario value for {field}")
            if (field.startswith("room_") or field == "rt60" or field.endswith("distance")) and number <= 0:
                raise ValueError(f"Scenario {field} must be positive")
        if not 0 <= float(row["azimuth_difference_deg"]) <= 180:
            raise ValueError("Invalid azimuth separation")
        for i in (1, 2):
            if not -90 <= float(row[f"source_{i}_elevation_deg"]) <= 90:
                raise ValueError("Invalid elevation")
        for axis in "xyz":
            if any(not 0 <= float(row[f"{prefix}_{axis}"]) <= float(row[f"room_{axis}"])
                   for prefix in ("array", "source_1", "source_2")):
                raise ValueError("Scenario geometry lies outside the room")
    return rows


def read_scenarios(path):
    with open_csv(path, "r") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Scenario CSV header does not match the upstream schema")
        return validate_rows(list(reader))


def write_scenarios(rows, path):
    rows = validate_rows(list(rows))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_csv(output, "x") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.8f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def main(argv=None):
    args = parse_args(argv)
    rows = build_scenarios(load_config(args.config), args.librimix_root, args.split)
    write_scenarios(rows, args.output)
    print(f"Saved {len(rows)} fixed scenarios to {args.output}")


if __name__ == "__main__":
    main()
