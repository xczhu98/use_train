#!/usr/bin/env python3
"""生成多麦克风 scene-level offline RIR bundle。

该入口只负责 ``C>1`` spatial scene；``C=1`` 继续使用
``generate_simulated_rirs`` 的 legacy mono 输出。生成结果包含每个 actor 的
full/direct/early ``[C,L]`` track 和逐 scene ``scenes.jsonl`` typed manifest。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.prepare_data_tools.generators.se.rir_config import GeneratorConfig
from tools.prepare_data_tools.generators.se.rir_plan import build_field_config
from tools.prepare_data_tools.generators.se.rir_output import save_moving_scene_result, save_scene_result
from tools.prepare_data_tools.generators.se.scene_generation import (
    generate_gpurir_moving_scene,
    generate_gpurir_scene,
    resolve_spatial_array,
    sample_spatial_moving_scene,
    sample_spatial_scene,
)

GENERATOR_VERSION = "se.scene-rir-v2.0"
GENERATOR_VERSION_MOVING = "se.scene-rir-v3.0"


def _build_rir_config(config: GeneratorConfig, sample_rate: int, field: str, *, mode: str = "static"):
    import argparse as _argparse

    namespace = _argparse.Namespace(**config.to_runtime_namespace())
    namespace.rt60_range = config.rt60_range_sec
    namespace.dist_range_near = config.near_distance_range_m
    namespace.dist_range_far = config.far_distance_range_m
    namespace.n_samples_near = config.near_rir_length_samples
    namespace.n_samples_far = config.far_rir_length_samples
    return build_field_config(namespace, field, sample_rate, mode=mode)


def generate_scene_dataset(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    num_scenes: int,
    sample_rate: int | None = None,
    field: str = "far",
    resume: bool = False,
    moving: bool = False,
) -> Path:
    """生成 typed scene manifest，返回 ``scenes.jsonl`` 路径。

    静态 scene 写 schema v2 manifest；``moving=True`` 时生成逐 actor 轨迹的
    moving scene（schema v3，track 布局 ``[P,A,C,L]``）。
    """
    config = GeneratorConfig.from_yaml(config_path)
    if moving and not config.motion_enabled:
        raise ValueError(
            "--moving requires motion.enabled=true in the generator config; "
            "scene moving trajectories reuse the motion ranges "
            "(n_segments_range / segment_duration_sec / speed_range_mps / trajectory_type)"
        )
    if config.backend != "gpurir":
        raise ValueError("scene-level spatial generator currently requires backend='gpurir'")
    if num_scenes < 1:
        raise ValueError("num_scenes must be positive")
    sample_rate = int(sample_rate or config.sample_rates_hz[0])
    rir_config = _build_rir_config(config, sample_rate, field, mode="moving" if moving else "static")
    array = resolve_spatial_array(rir_config)
    if array is None or array.channels <= 1:
        raise ValueError("scene-level generator requires array_profile/custom_geometry with C>1")
    from tools.prepare_data_tools.generators.se.rir_backends.gpurir import GPURIRBackend

    backend = GPURIRBackend()
    gpu_rir = backend._ensure_import()
    generator_version = GENERATOR_VERSION_MOVING if moving else GENERATOR_VERSION
    output = Path(output_dir).expanduser().resolve()
    if resume:
        output.mkdir(parents=True, exist_ok=True)
        work_output = output
    else:
        if output.exists():
            if any(output.iterdir()):
                raise ValueError(f"output directory exists and is not empty: {output}")
            output.rmdir()
        output.parent.mkdir(parents=True, exist_ok=True)
        work_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.scene-staging-", dir=output.parent))
    manifest_path = work_output / "scenes.jsonl"
    completed_uids: set[str] = set()
    if resume and manifest_path.is_file():
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # 半截/损坏行必须显式报错，否则续跑会静默丢掉已完成的 scene。
                raise ValueError(
                    f"--resume manifest 存在无法解析的行 {manifest_path}:{line_number}: {exc}"
                ) from exc
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"--resume manifest 记录缺少 metadata: {manifest_path}:{line_number}"
                )
            # generator_version 变化会改变 layout/轨迹语义，禁止跨版本续跑。
            record_version = metadata.get("generator_version")
            if record_version != generator_version:
                raise ValueError(
                    f"--resume manifest generator_version mismatch at "
                    f"{manifest_path}:{line_number}: expected {generator_version!r}, "
                    f"got {record_version!r}"
                )
            scene_uid = metadata.get("scene_uid")
            if isinstance(scene_uid, str):
                completed_uids.add(scene_uid)
    generated_count = 0
    skipped_count = 0
    try:
        for index in range(int(num_scenes)):
            if moving:
                scene = sample_spatial_moving_scene(rir_config, index, generator_version=generator_version)
            else:
                scene = sample_spatial_scene(rir_config, index, generator_version=generator_version)
            if scene.scene_uid in completed_uids:
                skipped_count += 1
                continue
            if moving:
                generated = generate_gpurir_moving_scene(scene, gpu_rir)
                record = save_moving_scene_result(
                    generated,
                    output_dir=work_output,
                    index=index,
                    shards=config.shards,
                )
            else:
                generated = generate_gpurir_scene(scene, gpu_rir)
                record = save_scene_result(
                    generated,
                    output_dir=work_output,
                    index=index,
                    shards=config.shards,
                )
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            completed_uids.add(scene.scene_uid)
            generated_count += 1
        summary = {
            "schema_version": 3 if moving else 2,
            "generator_id": "se.generate_scene_rirs",
            "generator_version": generator_version,
            "moving": bool(moving),
            "sample_rate": sample_rate,
            "scene_count": len(completed_uids),
            "generated_count": generated_count,
            "skipped_count": skipped_count,
            "array_profile": rir_config.array_profile,
            "channels": array.channels,
            "actor_roles": list(rir_config.actor_roles),
            "manifest": str(manifest_path),
        }
        (work_output / "scenes_meta.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        marker = work_output / "generation_complete.json"
        marker.write_text(json.dumps({"scene_count": len(completed_uids), "generator_version": generator_version}, indent=2), encoding="utf-8")
    except BaseException:
        if not resume:
            shutil.rmtree(work_output, ignore_errors=True)
        raise
    if not resume:
        os.replace(work_output, output)
        manifest_path = output / "scenes.jsonl"
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate typed multichannel scene RIR bundles")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-scenes", required=True, type=int)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--field", choices=("near", "far"), default="far")
    parser.add_argument("--moving", action="store_true", help="生成 moving scene（schema v3，[P,A,C,L] track）")
    parser.add_argument("--resume", action="store_true", help="在已有 scenes.jsonl 上按 scene_uid 续跑")
    args = parser.parse_args(argv)
    path = generate_scene_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        sample_rate=args.sample_rate,
        field=args.field,
        resume=args.resume,
        moving=args.moving,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
