#!/usr/bin/env python3
"""Standalone script to prepare standardized SCPs under the configured test root.

This is the same logic used by ``evaluate/eval_pipeline_common.py``; extracted here
so you can run it independently before any inference or evaluation.

Usage:
    # Prepare everything (VoiceBank, URGENT, AEC Official, DNS Challenge;
    # also AEC simulated when USE_TRAIN_TEST_DATA_ROOT is configured)
    python tools/prepare_data_tools/scp/prepare_testset_scps.py

    # Prepare selected testsets only
    python tools/prepare_data_tools/scp/prepare_testset_scps.py \
        --datasets voicebank,urgent

    # Custom paths
    python tools/prepare_data_tools/scp/prepare_testset_scps.py \
        --aec_testset_root /path/to/AEC_Challenge_test \
        --aec_subsets doubletalk \
        --urgent_noisy_dir /path/to/urgent/noisy_16k \
        --urgent_clean_dir /path/to/urgent/clean
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.utils.file_transaction import write_text_files_transactionally  # noqa: E402
from tools.prepare_data_tools._common import ensure_project_root, setup_logging  # noqa: E402
from tools.shared.scp_io import serialize_scp_entries  # noqa: E402
from evaluate.eval_pipeline_common import (
    DEFAULT_PATHS,
    DEFAULT_TEST_DATA_ROOT,
    prepare_testset_scps,
)

ensure_project_root()
logger = setup_logging("prepare_testset_scps")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare standardized testset SCPs under the configured test root"
    )
    p.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated: voicebank,urgent,aec_official,aec_sim,dns_challenge,all",
    )
    p.add_argument("--aec_testset_root", default=str(DEFAULT_PATHS["aec_testset_root"]))
    p.add_argument(
        "--aec_subsets",
        default="doubletalk",
        help="AEC official subsets (default: doubletalk). Use 'all' for all.",
    )
    p.add_argument("--voicebank_noisy_scp", default=str(DEFAULT_PATHS["voicebank_noisy_scp"]))
    p.add_argument("--voicebank_clean_scp", default=str(DEFAULT_PATHS["voicebank_clean_scp"]))
    p.add_argument("--urgent_noisy_dir", default=str(DEFAULT_PATHS["urgent_noisy_dir"]))
    p.add_argument("--urgent_clean_dir", default=str(DEFAULT_PATHS["urgent_clean_dir"]))
    p.add_argument("--dns_challenge_noisy_dir", default=str(DEFAULT_PATHS["dns_challenge_noisy_dir"]))
    p.add_argument(
        "--aec_sim_testset_root",
        default=os.environ.get("USE_TRAIN_TEST_DATA_ROOT"),
        help="Root dir of AEC simulated testset (7 scenario subdirs); "
        "默认取 USE_TRAIN_TEST_DATA_ROOT 环境变量，未设置且需要 aec_sim 时报错",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_TEST_DATA_ROOT),
        help="Where to write SCPs (defaults to USE_TRAIN_TEST_DATA_ROOT or repo data/test)",
    )
    return p.parse_args(argv)


from core.data.simulation.scene_names import AEC_SCENE_ORDER

AEC_SIM_SCENARIOS = list(AEC_SCENE_ORDER)
AEC_SIM_WAV_NAMES = ["mic", "nearend", "farend", "noisy_se"]


def resolve_aec_subsets(tag_str: str) -> list[str]:
    if tag_str.strip().lower() == "all":
        return ["doubletalk", "farend-singletalk", "nearend-singletalk"]
    return [s.strip() for s in tag_str.split(",") if s.strip()]


def _prepare_aec_sim_scps(src: Path, dst_root: Path) -> dict[str, Path]:
    """校验 AEC simulated testset，并事务发布全部非空场景 SCP。"""
    if not src.is_dir():
        raise ValueError(f"AEC sim testset root is missing or not a directory: {src}")

    dst = dst_root / "aec_sim_testset"
    result: dict[str, Path] = {}
    outputs: list[tuple[Path, str]] = []
    total_entries = 0
    for scen in AEC_SIM_SCENARIOS:
        scen_src = src / scen
        if not scen_src.is_dir():
            logger.warning("Scenario not found: %s, skipping", scen_src)
            continue

        samples = sorted(path for path in scen_src.iterdir() if path.is_dir())
        scen_dst = dst / scen
        counts: dict[str, int] = {}
        scenario_outputs: list[tuple[Path, str]] = []
        for wav_name in AEC_SIM_WAV_NAMES:
            entries = [
                (sample.name, str(wav.resolve()))
                for sample in samples
                if (wav := sample / f"{wav_name}.wav").is_file()
            ]
            counts[wav_name] = len(entries)
            scp = scen_dst / f"{wav_name}.scp"
            scenario_outputs.append((scp, serialize_scp_entries(entries)))
            result[f"{scen}/{wav_name}"] = scp
        if len(set(counts.values())) > 1:
            logger.warning(
                "Scenario %s: per-wav entry counts differ %s "
                "(missing wav files detected)",
                scen,
                counts,
            )
        if sum(counts.values()) == 0:
            logger.warning("Scenario contains zero valid wav entries: %s, skipping", scen_src)
            for wav_name in AEC_SIM_WAV_NAMES:
                result.pop(f"{scen}/{wav_name}", None)
            continue
        total_entries += sum(counts.values())
        outputs.extend(scenario_outputs)

    if total_entries == 0:
        raise ValueError(f"AEC sim testset contains zero valid wav entries: {src}")

    write_text_files_transactionally(
        [
            (path, lambda handle, payload=payload: handle.write(payload))
            for path, payload in outputs
        ],
        overwrite=True,
    )
    for path, payload in outputs:
        logger.info("Generated SCP: %s (%d entries)", path, len(payload.splitlines()))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tags = [t.strip() for t in args.datasets.split(",") if t.strip()]
    known_tags = {"voicebank", "urgent", "aec_official", "aec_sim", "dns_challenge", "all"}
    unknown = [t for t in tags if t not in known_tags]
    if unknown:
        logger.error("Unknown --datasets tag(s): %s (known: %s)", unknown, sorted(known_tags))
        return 1
    include_aec = "all" in tags or "aec_official" in tags
    include_aec_sim = "all" in tags or "aec_sim" in tags
    include_vb = "all" in tags or "voicebank" in tags
    include_urg = "all" in tags or "urgent" in tags
    include_dns = "all" in tags or "dns_challenge" in tags

    if include_aec_sim and not args.aec_sim_testset_root:
        if "aec_sim" in tags:
            logger.error(
                "--datasets 显式包含 aec_sim 但未提供 --aec_sim_testset_root，"
                "且环境变量 USE_TRAIN_TEST_DATA_ROOT 未设置"
            )
            return 1
        # 默认 all 场景：aec_sim 数据不可用时降级跳过，不阻塞其余评测集
        logger.warning(
            "USE_TRAIN_TEST_DATA_ROOT 未设置，跳过 aec_sim 评测集"
            "（可用 --aec_sim_testset_root 显式指定）"
        )
        include_aec_sim = False

    scp_paths = prepare_testset_scps(
        aec_testset_root=args.aec_testset_root if include_aec else None,
        aec_subsets=resolve_aec_subsets(args.aec_subsets) if include_aec else None,
        urgent_noisy_dir=args.urgent_noisy_dir if include_urg else None,
        urgent_clean_dir=args.urgent_clean_dir if include_urg else None,
        voicebank_noisy_scp=args.voicebank_noisy_scp if include_vb else None,
        voicebank_clean_scp=args.voicebank_clean_scp if include_vb else None,
        dns_challenge_noisy_dir=args.dns_challenge_noisy_dir if include_dns else None,
        output_root=args.output_dir,
    )

    if include_aec_sim:
        try:
            aec_sim_paths = _prepare_aec_sim_scps(
                Path(args.aec_sim_testset_root), Path(args.output_dir)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Cannot prepare AEC sim SCPs: %s", exc)
            return 1
        scp_paths["aec_sim"] = aec_sim_paths

    print("\n" + "=" * 60)
    print("PREPARED SCP FILES")
    print("=" * 60)
    for key, paths in scp_paths.items():
        print(f"\n  [{key}]")
        if isinstance(paths, dict):
            for sub_key, p in paths.items():
                print(f"    {sub_key}: {p}")
        else:
            print(f"    {paths}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
