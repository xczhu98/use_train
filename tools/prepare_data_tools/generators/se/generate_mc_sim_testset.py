"""用我方 SCP、scene plan 与 renderer 生成 WSJ0-mix 风格双源多通道 SE 评估集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from core.data.resource_paths import resolve_manifest_resource_path
from core.data.simulation.scene_renderer import render_speech_scene
from core.data.simulation.role_policy import AcousticActor, AcousticRole, RIRPolicy
from core.data.simulation.scene_sampler import prepare_scene_plan
from core.data.simulation.spatial_scene import SpatialSceneRenderer, SpatialSceneRequest
from core.data.simulation_configs import SimulationConfigs
from core.data.simulation_views import project_simulation_view
from core.utils.config import load_merged_config
from core.data.simulation_audio_utils import _resample_audio
from tools.shared.scp_io import parse_scp_fields, write_scp_entries


GENERATOR_ID = "se.mc_sim_testset"
GENERATOR_VERSION = "1.0"


def _read_audio_scp(path: str | Path) -> list[tuple[str, int, int, Path]]:
    scp = Path(path).expanduser().resolve()
    entries = []
    seen = set()
    with scp.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            uid, sample_rate, length, audio_path = parse_scp_fields(
                line, scp, line_number, allowed_columns=(4,)
            )
            if uid in seen:
                raise ValueError(f"{scp}:{line_number}: duplicate uid {uid!r}")
            seen.add(uid)
            entries.append(
                (
                    uid,
                    int(sample_rate),
                    int(length),
                    resolve_manifest_resource_path(audio_path, scp),
                )
            )
    if not entries:
        raise ValueError(f"SCP has no valid four-column entries: {scp}")
    return entries


def _load_mono_segment(entry, *, sample_rate: int, length: int, rng) -> np.ndarray:
    import soundfile as sf

    uid, declared_sr, declared_length, path = entry
    data, actual_sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] != 1:
        raise ValueError(f"{uid!r} must be mono, got C={data.shape[1]}: {path}")
    if int(actual_sr) != declared_sr or data.shape[0] != declared_length:
        raise ValueError(
            f"{uid!r} SCP metadata mismatch: declared ({declared_sr},{declared_length}), "
            f"actual ({actual_sr},{data.shape[0]})"
        )
    waveform = data[:, 0]
    if int(actual_sr) != sample_rate:
        waveform = _resample_audio(waveform, int(actual_sr), sample_rate)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size > length:
        start = int(rng.integers(0, waveform.size - length + 1))
        waveform = waveform[start : start + length]
    elif waveform.size < length:
        waveform = np.pad(waveform, (0, length - waveform.size))
    if waveform.shape != (length,) or not np.isfinite(waveform).all():
        raise ValueError(f"{uid!r} did not produce a finite mono {length}-sample segment")
    if float(np.mean(np.square(waveform.astype(np.float64)))) <= 1e-12:
        raise ValueError(f"{uid!r} produced a silent WSJ0-mix source segment")
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SceneAdapter:
    def __init__(self, simulation_cfg):
        self.sim_cfg = simulation_cfg
        self._simulation_view = project_simulation_view(simulation_cfg, "se")
        self.export_simulation_stems = False

    def _acoustic_plan_record(self):
        return None

    def _apply_channel_degradation_result(self, *_args, **_kwargs):
        return None


def generate_mc_sim_testset(
    *,
    config_path: str | Path,
    target_scp: str | Path,
    interference_scp: str | Path,
    output_dir: str | Path,
    num_samples: int,
    seed: int = 42,
    duration_sec: float | None = None,
    sir_range_db: tuple[float, float] = (-5.0, 5.0),
    provider: Any | None = None,
) -> Path:
    """Generate a static two-speaker multi-microphone evaluation set.

    Inputs are two disjoint standard four-column SCPs. The target and interference
    are independently selected/cropped, rendered as separate room actors by the
    shared spatial renderer, and mixed at the requested full-image SIR range.
    ``provider`` is injectable for deterministic unit fixtures; production uses
    the framework online RIR service.
    """
    import soundfile as sf

    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("num_samples must be a positive exact int")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative exact int")
    if (
        len(sir_range_db) != 2
        or not np.isfinite(sir_range_db).all()
        or sir_range_db[0] > sir_range_db[1]
    ):
        raise ValueError("sir_range_db must be an ordered finite pair")

    config = load_merged_config(config_path)
    data = config.get("data")
    if not isinstance(data, dict) or data.get("mode") != "dynamic":
        raise ValueError("mc_sim generator requires data.mode='dynamic'")
    train = data.get("train")
    if not isinstance(train, dict) or not isinstance(train.get("simulation_cfg"), dict):
        raise ValueError("mc_sim generator requires data.train.simulation_cfg")
    sample_rate = data.get("sample_rate")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("data.sample_rate must be a positive exact int")
    if duration_sec is None:
        duration_sec = float(train.get("max_duration_seconds", 4.0))
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive and finite")
    sample_length = int(round(float(duration_sec) * sample_rate))
    target_entries = _read_audio_scp(target_scp)
    interference_entries = _read_audio_scp(interference_scp)
    if len(target_entries) < num_samples or len(interference_entries) < num_samples:
        raise ValueError(
            "mc_sim requires at least num_samples unique target and interference utterances"
        )
    target_ids = {entry[0] for entry in target_entries}
    interference_ids = {entry[0] for entry in interference_entries}
    if target_ids & interference_ids:
        raise ValueError("target_scp and interference_scp must have disjoint utterance ids")

    simulation_values = dict(train["simulation_cfg"])
    simulation_values.update(
        spatial_enabled=True,
        rir_backend="online",
        target_type="early",
        prob_reverberation=1.0,
        prob_moving_rir=0.0,
        prob_array_rotation=0.0,
        prob_microphone_position_jitter=0.0,
        prob_wind_noise_sidechain=0.0,
        prob_colored_noise=0.0,
        prob_channel_enabled=False,
        noisy_post_mix_degradations_enabled=False,
        source_volume_normalization_enabled=False,
    )
    simulation_cfg = SimulationConfigs.from_dict(simulation_values, strict=True)
    if simulation_cfg.array_profile is None and simulation_cfg.custom_geometry is None:
        raise ValueError("mc_sim config requires an array_profile or custom_geometry")
    if provider is None:
        from core.data.simulation.rir_provider import OnlineRIRProvider

        provider = OnlineRIRProvider.from_simulation(simulation_cfg)
    adapter = _SceneAdapter(simulation_cfg)

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to replace an existing mc_sim output directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    for folder in ("mixture", "clean", "target_image"):
        (staged / folder).mkdir()
    mixture_entries = []
    clean_entries = []
    target_image_entries = []
    records = []
    spatial_view = adapter._simulation_view.shared.spatial
    try:
        for index in range(num_samples):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            target_entry = target_entries[index]
            interference_entry = interference_entries[(index + seed) % len(interference_entries)]
            target_wave = _load_mono_segment(
                target_entry, sample_rate=sample_rate, length=sample_length, rng=rng
            )
            interference_wave = _load_mono_segment(
                interference_entry, sample_rate=sample_rate, length=sample_length, rng=rng
            )
            source_uids = (f"target/{target_entry[0]}", f"interference/{interference_entry[0]}")
            scene_plan = prepare_scene_plan(
                provider=provider,
                task="se",
                sample_index=index,
                sample_rate=sample_rate,
                num_samples=sample_length,
                source_uids=source_uids,
                roles=("target", "interference"),
                spatial_view=spatial_view,
                simulation_cfg=simulation_cfg,
                rng=rng,
                moving=False,
            )
            rendered = SpatialSceneRenderer().render(
                SpatialSceneRequest(
                    sources=np.stack((target_wave, interference_wave)),
                    sample_rate=sample_rate,
                    provider=provider,
                    source_uids=source_uids,
                    mode="static",
                    track_kinds=("full", "early"),
                    early_boundary=int(round(simulation_cfg.early_rir_sec * sample_rate)),
                    reference_channel=simulation_cfg.reference_channel,
                    num_samples=sample_length,
                    normalize=True,
                    scene_plan=scene_plan,
                )
            )
            actors = (
                AcousticActor(
                    uid=source_uids[0],
                    role=AcousticRole.TARGET_SPEECH,
                    rir_policy=RIRPolicy.SPATIAL_SOURCE,
                    waveform=target_wave,
                ),
                AcousticActor(
                    uid=source_uids[1],
                    role=AcousticRole.INTERFERENCE_SPEECH,
                    rir_policy=RIRPolicy.SPATIAL_SOURCE,
                    waveform=interference_wave,
                ),
            )
            speech_scene = render_speech_scene(
                adapter,
                actors,
                sample_rate,
                rng,
                sir_range=sir_range_db,
                sir_mode="aggregate",
                need_noise=False,
                num_samples=sample_length,
                pre_rendered_spatial=rendered,
                sources_preplaced=True,
            )
            mixture = speech_scene.bundle.require("mixture").copy()
            clean = speech_scene.bundle.require("target/0").copy()
            target_image = rendered.early_tracks[0].copy()
            sir_fact = (speech_scene.facts.get("sir") or [{}])[0]
            sir_selected_db = sir_fact.get("selected_db")
            sir_realized_db = sir_fact.get("realized_db")
            if sir_selected_db is None or sir_realized_db is None:
                raise ValueError(f"mc_sim sample {index} did not realize its requested speech SIR")
            if (
                mixture.shape != (spatial_view.channels, sample_length)
                or clean.shape != (sample_length,)
                or target_image.shape != (spatial_view.channels, sample_length)
            ):
                raise ValueError(f"mc_sim sample {index} violated [C,T]/[T] waveform shapes")

            uid = f"mc_sim_{index:08d}"
            paths = {
                "mixture": staged / "mixture" / f"{uid}.wav",
                "clean": staged / "clean" / f"{uid}.wav",
                "target_image": staged / "target_image" / f"{uid}.wav",
            }
            sf.write(paths["mixture"], mixture.T, sample_rate, subtype="FLOAT")
            sf.write(paths["clean"], clean, sample_rate, subtype="FLOAT")
            sf.write(paths["target_image"], target_image.T, sample_rate, subtype="FLOAT")
            final_paths = {name: output / path.relative_to(staged) for name, path in paths.items()}
            mixture_entries.append((uid, sample_rate, sample_length, str(final_paths["mixture"])))
            clean_entries.append((uid, sample_rate, sample_length, str(final_paths["clean"])))
            target_image_entries.append(
                (uid, sample_rate, sample_length, str(final_paths["target_image"]))
            )
            records.append(
                {
                    "uid": uid,
                    "target_uid": target_entry[0],
                    "interference_uid": interference_entry[0],
                    "scene_uid": rendered.provenance["scene_uid"],
                    "sir_selected_db": float(sir_selected_db),
                    "sir_realized_db": float(sir_realized_db),
                    "geometry": dict(rendered.geometry),
                    "room": dict(scene_plan.room),
                    "payload_sha256": {
                        name: _sha256_file(path) for name, path in paths.items()
                    },
                }
            )

        write_scp_entries(mixture_entries, staged / "mixture.scp", four_col=True)
        write_scp_entries(mixture_entries, staged / "noisy.scp", four_col=True)
        write_scp_entries(clean_entries, staged / "clean.scp", four_col=True)
        write_scp_entries(target_image_entries, staged / "target_image.scp", four_col=True)
        protocol = {
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "sample_rate": sample_rate,
            "duration_sec": float(duration_sec),
            "sir_range_db": [float(value) for value in sir_range_db],
            "simulation_cfg": simulation_values,
        }
        config_digest = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "domain": "wsj0-mix-style-two-source-spatial-se",
            "seed": seed,
            "sample_rate": sample_rate,
            "duration_sec": float(duration_sec),
            "sir_range_db": [float(value) for value in sir_range_db],
            "config_sha256": config_digest,
            "source_scp_sha256": {
                "target": _sha256_file(Path(target_scp).expanduser().resolve()),
                "interference": _sha256_file(Path(interference_scp).expanduser().resolve()),
            },
            "scps_sha256": {
                name: _sha256_file(staged / f"{name}.scp")
                for name in ("mixture", "noisy", "clean", "target_image")
            },
            "reference_channel": int(simulation_cfg.reference_channel),
            "channel_order": [f"mic_{index}" for index in range(spatial_view.channels)],
            "array_profile": simulation_cfg.array_profile,
            "num_samples": len(records),
            "items": records,
        }
        (staged / "mc_sim_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, output)
        return output / "mc_sim_manifest.json"
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise
    finally:
        client = getattr(provider, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-scp", required=True)
    parser.add_argument("--interference-scp", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--sir-range-db", type=float, nargs=2, default=(-5.0, 5.0))
    args = parser.parse_args(argv)
    manifest = generate_mc_sim_testset(
        config_path=args.config,
        target_scp=args.target_scp,
        interference_scp=args.interference_scp,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        duration_sec=args.duration_seconds,
        sir_range_db=tuple(args.sir_range_db),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
