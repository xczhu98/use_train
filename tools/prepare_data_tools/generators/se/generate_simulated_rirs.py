#!/usr/bin/env python3
"""统一的 RIR 仿真数据生成工具。

支持三种 backend：
  - rirgen : rir_generator（image-source）
  - fram   : FRAM-RIR（Fast Random Approximation）
  - gpurir : gpuRIR（GPU 加速 image-source）

默认使用 gpuRIR 生成 100000 条静态 RIR：近场/远场按 5:5 分配；两类场景
共用 0.1–1.0 s 的 RT60 范围。默认输出格式为 .pkl（静态
与移动一致，含完整元数据，推荐生产数据使用），可通过 --format 显式选择静态
FLOAT WAV 或 24-bit FLAC；FLAC 仅接受幅值位于 [-1, 1] 的 RIR，避免整数 PCM
静默截断。同时
自动输出训练框架可直接加载的 4 列 rirs.scp。SCP 中的音频路径相对于
manifest 文件目录，文件名与 uid 编码采样率、远/近场、RT60、距离、房间尺寸等元信息。
未设置 CUDA_VISIBLE_DEVICES 时默认使用全部可用 GPU（本机 8 卡），``--gpus``
可省略；``--validate`` 是严格发布门禁，异常 RIR 存在时不发布 manifest。

Usage:
    python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
        --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
        --output-dir /data/rir/my_gpurir_16k

    # 使用 FRAM profile 生成静态 RIR
    python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
        --config tools/prepare_data_tools/generators/se/rir_fram.yaml \
        --output-dir /data/rir/sim_rirs_fram

    # 查看压力测试 profile 的完整有效配置
    python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
        --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
        --dump-effective-config
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from tools.prepare_data_tools._common import parse_float_pair, setup_logging  # noqa: E402
from tools.prepare_data_tools.generators.se import rir_executor  # noqa: E402
from tools.prepare_data_tools.generators.se.rir_backends import (  # noqa: E402
    get_backend,
    list_available_backends,
)
from tools.prepare_data_tools.generators.se.rir_config import (  # noqa: E402
    GeneratorConfig,
    MIN_SOURCE_MICROPHONE_DISTANCE,
)
from tools.prepare_data_tools.generators.se.rir_manifest import (  # noqa: E402
    build_summary,
    publish_manifest,
)
from tools.prepare_data_tools.generators.se.rir_plan import (  # noqa: E402
    build_field_config,
    effective_mode,
    iter_gpu_tasks,
    iter_tasks,
    normalize_counts,
    normalize_segment_range,
    recommended_rir_length_samples,
    split_count,
)
from tools.prepare_data_tools.generators.se.rir_validation import validate_rirs  # noqa: E402

logger = setup_logging("generate_simulated_rirs")

GENERATOR_ID = "se.generate_simulated_rirs"
GENERATOR_VERSION_HISTORY = (
    "1.0",
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.6",
    "1.7",
    "1.8",
    "1.9",
    "1.10",
    "1.11",
    "1.12",
)
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]
DEFAULT_NEAR_RIRS = 50000
DEFAULT_FAR_RIRS = 50000
DEFAULT_TOTAL_RIRS = DEFAULT_NEAR_RIRS + DEFAULT_FAR_RIRS
DEFAULT_RT60_RANGE = (0.1, 1.0)
DEFAULT_STATIC_MOVING_RATIO = (1.0, 0.0)
DEFAULT_NEAR_FAR_RATIO = (5.0, 5.0)


# ---------------------------------------------------------------------------
# CLI 参数解析辅助
# ---------------------------------------------------------------------------

def _parse_ratio(value: str) -> tuple[float, float]:
    """解析 ``a,b`` 或 ``a:b`` 形式的非负比例。"""
    parts = value.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected two ratio values separated by ',' or ':', got {value!r}"
        )
    try:
        ratio = (float(parts[0].strip()), float(parts[1].strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ratio {value!r}") from exc
    return ratio


def _parse_int_pair(value: str) -> tuple[int, int]:
    parts = value.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two integer values separated by ',' or ':'")
    try:
        result = (int(parts[0].strip()), int(parts[1].strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer range {value!r}") from exc
    if result[0] < 1 or result[0] > result[1]:
        raise argparse.ArgumentTypeError("integer range must satisfy 1 <= lo <= hi")
    return result


def _parse_gpu_token(value: str) -> int | str:
    """解析裸编号、``slot:N``、``id:N``、UUID 或 MIG token。"""
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("GPU token must not be empty")
    for prefix in ("slot:", "id:"):
        if value.lower().startswith(prefix):
            number = value[len(prefix) :].strip()
            try:
                parsed = int(number)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid GPU token {value!r}; expected {prefix}N"
                ) from exc
            if parsed < 0:
                raise argparse.ArgumentTypeError("GPU values must be non-negative")
            return f"{prefix}{parsed}"
    try:
        return int(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# GPU 设备解析
# ---------------------------------------------------------------------------

def _parse_cuda_visible_tokens() -> list[int | str] | None:
    """读取 CUDA_VISIBLE_DEVICES；未设置时返回 None，禁用时返回空列表。"""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return None
    tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not tokens or any(
        token.lower() in {"-1", "none", "nodevfiles", "void"} for token in tokens
    ):
        return []
    return [_parse_gpu_token(token) for token in tokens]


_NVIDIA_SMI_TIMEOUT_SEC = 10


def _nvidia_smi(args: list[str]) -> str:
    """执行静态 nvidia-smi 查询并返回 stdout；命令缺失/超时/失败一律返回空串。"""
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def _detect_gpus() -> list[int | str]:
    """自动检测 GPU；未设置 CVD 时返回全部物理 GPU，设置时只返回可见设备。"""
    visible_devices = _parse_cuda_visible_tokens()
    if visible_devices is not None:
        return visible_devices
    try:
        result = _nvidia_smi(
            ["--query-gpu=index", "--format=csv,noheader,nounits"]
        )
        gpu_ids = [
            int(line.strip()) for line in result.strip().split("\n") if line.strip()
        ]
        return gpu_ids
    except Exception:
        return []


def _physical_gpu_token_map() -> dict[str, int]:
    """读取 GPU/MIG token 到物理 GPU 编号的映射。"""
    token_map: dict[str, int] = {}
    result = _nvidia_smi(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    for line in result.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            physical_id = int(parts[0])
        except ValueError:
            continue
        token_map[parts[1]] = physical_id

    result = _nvidia_smi(["-L"])
    for line in result.splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU "):
            prefix = stripped.split(":", 1)[0].split()
            try:
                current_gpu = int(prefix[1]) if len(prefix) == 2 else None
            except ValueError:
                current_gpu = None
            token = stripped.split("UUID:", 1)[1].split(")", 1)[0].strip()
            if current_gpu is not None and token:
                token_map[token] = current_gpu
        elif stripped.startswith("MIG "):
            # MIG 行的实例编号不等于物理 GPU 编号；从 MIG UUID 提取所属 GPU UUID 反查
            token = stripped.split("UUID:", 1)[1].split(")", 1)[0].strip()
            gpu_token = token.removeprefix("MIG-").split("/", 1)[0]
            if token and gpu_token in token_map:
                token_map[token] = token_map[gpu_token]
    return token_map


def _physical_to_visible_token(
    physical_id: int,
    visible: list[int | str],
    physical_token_map: dict[str, int] | None,
) -> int | str | None:
    """把物理 GPU 编号映射到 CVD 可见 token；不可见时返回 None。"""
    for visible_token in visible:
        if isinstance(visible_token, int) and visible_token == physical_id:
            return visible_token
    if physical_token_map is not None:
        for visible_token in visible:
            if (
                isinstance(visible_token, str)
                and physical_token_map.get(visible_token) == physical_id
            ):
                return visible_token
    return None


def _resolve_gpu_ids(requested: list[int | str] | None) -> list[int | str]:
    """解析 GPU；CVD 下裸整数是 visible slot，``id:N`` 才是物理编号。"""
    if requested is None:
        resolved = _detect_gpus()
    else:
        normalized = [_parse_gpu_token(str(token)) for token in requested]
        # 负数在 CVD 分支会触发 Python 负索引，必须在规范化后立即拒绝
        if any(isinstance(token, int) and token < 0 for token in normalized):
            raise ValueError("--gpus values must be non-negative")
        visible = _parse_cuda_visible_tokens()
        if visible is None:
            resolved = []
            physical_ids: list[int | str] | None = None
            for token in normalized:
                if isinstance(token, str) and token.startswith("slot:"):
                    if physical_ids is None:
                        physical_ids = _detect_gpus()
                    slot = int(token.removeprefix("slot:"))
                    if slot >= len(physical_ids):
                        raise ValueError(
                            f"GPU visible slot {slot} is unavailable; detected {physical_ids}"
                        )
                    resolved.append(physical_ids[slot])
                elif isinstance(token, str) and token.startswith("id:"):
                    resolved.append(int(token.removeprefix("id:")))
                else:
                    resolved.append(token)
        else:
            if not visible:
                raise ValueError("CUDA_VISIBLE_DEVICES exposes no GPU")
            resolved = []
            physical_token_map: dict[str, int] | None = None
            for token in normalized:
                if isinstance(token, str) and token.startswith("slot:"):
                    slot = int(token.removeprefix("slot:"))
                    if slot >= len(visible):
                        raise ValueError(
                            f"GPU visible slot {slot} is unavailable; visible devices are {visible}"
                        )
                    resolved.append(visible[slot])
                elif isinstance(token, str) and token.startswith("id:"):
                    physical_id = int(token.removeprefix("id:"))
                    match = _physical_to_visible_token(
                        physical_id, visible, physical_token_map
                    )
                    if match is None and physical_token_map is None:
                        # 物理编号不在 CVD 数字列表中时，才查 nvidia-smi 映射 UUID/MIG token
                        physical_token_map = _physical_gpu_token_map()
                        match = _physical_to_visible_token(
                            physical_id, visible, physical_token_map
                        )
                    if match is None:
                        raise ValueError(
                            f"physical GPU id:{physical_id} is not visible through CUDA_VISIBLE_DEVICES={visible}"
                        )
                    resolved.append(match)
                elif isinstance(token, int):
                    if token >= len(visible):
                        raise ValueError(
                            f"GPU visible slot {token} is unavailable; visible devices are {visible}"
                        )
                    resolved.append(visible[token])
                elif token in visible:
                    resolved.append(token)
                else:
                    raise ValueError(
                        f"GPU {token!r} is not visible through CUDA_VISIBLE_DEVICES={visible}"
                    )
    if any(isinstance(token, int) and token < 0 for token in resolved):
        raise ValueError("--gpus values must be non-negative")
    if len({str(token) for token in resolved}) != len(resolved):
        raise ValueError(f"GPU list contains duplicates: {resolved}")
    return resolved



def _build_parser() -> argparse.ArgumentParser:
    """构造 CLI 解析器，注册全部参数。"""
    parser = argparse.ArgumentParser(
        description="Generate simulated RIR datasets with unified backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog="""
Examples:
  # 默认 gpuRIR：100000 条，近/远场 5:5，仅静态 RIR
  python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
      --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
      --output-dir /data/rir/my_gpurir_16k

  # 使用 FRAM profile 生成静态 RIR
  python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
      --config tools/prepare_data_tools/generators/se/rir_fram.yaml \
      --output-dir /data/rir/sim_rirs_fram

  # 查看 profile 的完整有效配置
  python -m tools.prepare_data_tools.generators.se.generate_simulated_rirs \
      --config tools/prepare_data_tools/generators/se/rir_gpurir.yaml \
      --dump-effective-config
        """,
    )
    parser.add_argument(
        "--config",
        dest="rir_config_path",
        type=Path,
        default=None,
        help="RIR generator YAML 配置；配置文件与生成代码位于同一目录",
    )
    parser.add_argument(
        "--dump-effective-config",
        action="store_true",
        default=False,
        help="打印 --config 加载后的有效配置并退出",
    )
    parser.set_defaults(
        effective_config=None,
    )
    parser.add_argument(
        "--output-dir",
        dest="out_dir",
        required=False,
        type=Path,
        help="输出目录（必须不存在或为空）",
    )
    parser.add_argument(
        "--method",
        choices=["rirgen", "fram", "gpurir", "all"],
        default="gpurir",
        help="RIR 仿真方法（默认 gpurir；all 运行可用 backend，无 GPU 时跳过 gpurir）",
    )
    parser.add_argument(
        "--sample-rate-hz",
        dest="fs",
        type=int,
        nargs="+",
        default=[16000],
        help="目标采样率，可多值且不可重复，如 '--sample-rate-hz 16000 32000'（默认 16000）",
    )
    parser.add_argument(
        "--field",
        choices=["near", "far", "both"],
        default="both",
        help="近场 / 远场 / 全部（默认 both）",
    )
    parser.add_argument(
        "--near-rirs",
        dest="n_rirs_near",
        type=int,
        default=None,
        help="近场 RIR 数量；通常省略并由 total-rirs 和 near-far-ratio 推导",
    )
    parser.add_argument(
        "--far-rirs",
        dest="n_rirs_far",
        type=int,
        default=None,
        help="远场 RIR 数量；通常省略并由 total-rirs 和 near-far-ratio 推导",
    )
    parser.add_argument(
        "--total-rirs",
        dest="n_rirs_total",
        type=int,
        default=None,
        help="RIR 总数；field=both 时按 near-far-ratio 分配",
    )
    parser.add_argument(
        "--static-moving-ratio",
        dest="static_moving_ratio",
        type=_parse_ratio,
        default=DEFAULT_STATIC_MOVING_RATIO,
        help="静态:移动比例，如 7:3 或 7,3",
    )
    parser.add_argument(
        "--near-far-ratio",
        dest="near_far_ratio",
        type=_parse_ratio,
        default=DEFAULT_NEAR_FAR_RATIO,
        help="近场:远场比例，如 5:5 或 5,5",
    )
    parser.add_argument(
        "--near-rir-length-samples",
        dest="n_samples_near",
        type=int,
        default=None,
        help="近场 RIR 长度；未指定时按 RT60 和距离动态计算",
    )
    parser.add_argument(
        "--far-rir-length-samples",
        dest="n_samples_far",
        type=int,
        default=None,
        help="远场 RIR 长度；未指定时按 RT60 和距离动态计算",
    )
    parser.add_argument(
        "--rt60-range-sec",
        dest="rt60_range",
        type=parse_float_pair,
        default=None,
        help="近场/远场共享 RT60 范围，如 '0.2,1.3'（默认 0.2,1.3）",
    )
    parser.add_argument(
        "--near-distance-range-m",
        dest="dist_range_near",
        type=parse_float_pair,
        default=f"{MIN_SOURCE_MICROPHONE_DISTANCE},0.5",
        help=f"近场距离范围（米），最小值必须 >= {MIN_SOURCE_MICROPHONE_DISTANCE}，如 '0.05,0.5'",
    )
    parser.add_argument(
        "--far-distance-range-m",
        dest="dist_range_far",
        type=parse_float_pair,
        default="0.8,3.0",
        help="远场距离范围（米），如 '0.8,3.0'",
    )
    parser.add_argument(
        "--room-length-range-m",
        dest="room_length_range",
        type=parse_float_pair,
        default="3,10",
        help="房间长度范围（米）",
    )
    parser.add_argument(
        "--room-width-range-m",
        dest="room_width_range",
        type=parse_float_pair,
        default="3,10",
        help="房间宽度范围（米）",
    )
    parser.add_argument(
        "--room-height-range-m",
        dest="room_height_range",
        type=parse_float_pair,
        default="2.5,4.5",
        help="房间高度范围（米）",
    )
    parser.add_argument(
        "--source-wall-margin-m",
        dest="source_wall_margin_m",
        type=float,
        default=0.5,
        help="声源距墙最小距离（米，默认 0.5）",
    )
    parser.add_argument(
        "--microphone-wall-margin-m",
        dest="microphone_wall_margin_m",
        type=float,
        default=0.1,
        help="麦克风距墙最小距离（米，默认 0.1）",
    )
    parser.add_argument(
        "--source-height-range-m",
        dest="source_height_range_m",
        type=parse_float_pair,
        default=None,
        help="声源高度范围（米；默认使用内置高度策略）",
    )
    parser.add_argument(
        "--microphone-height-range-m",
        dest="microphone_height_range_m",
        type=parse_float_pair,
        default=None,
        help="麦克风高度范围（米；默认不单独限制高度）",
    )
    parser.add_argument(
        "--sound-speed-mps", dest="c", type=float, default=340.0, help="声速 m/s（默认 340）"
    )
    parser.add_argument("--workers", type=int, default=16, help="CPU backend 进程数")
    parser.add_argument(
        "--shards",
        type=int,
        default=10,
        help="每个 (method, fs, mode, field) 组合下的分片数（默认 10）",
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpu_token,
        nargs="+",
        default=None,
        help="gpuRIR 使用的 GPU token：裸整数（无 CVD 时=物理编号，有 CVD 时=可见 slot）、"
        "slot:N、id:N、UUID 或 MIG token（默认自动检测）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--format",
        choices=["auto", "wav", "pkl", "flac"],
        default="auto",
        help="输出格式（默认 auto：静态/移动均为 pkl，推荐生产使用；静态可显式 "
        "pkl、FLOAT wav 或 24-bit flac，FLAC 仅接受 [-1,1]，移动仅 pkl）",
    )
    parser.add_argument(
        "--write-metadata-json",
        dest="write_meta_json",
        action="store_true",
        help="为每条 RIR 额外写入 .json 元信息文件",
    )
    parser.add_argument(
        "--n-segments",
        dest="segments",
        type=int,
        default=None,
        help="固定移动轨迹段数；与 --n-segments-range 互斥",
    )
    parser.add_argument(
        "--n-segments-range",
        dest="segments_range",
        type=_parse_int_pair,
        default=None,
        help="移动轨迹段数范围（含首尾，默认 3,6）",
    )
    parser.add_argument(
        "--motion-enabled",
        dest="moving",
        action="store_true",
        help="生成移动声源 RIR；仅 gpurir 支持",
    )
    parser.add_argument(
        "--segment-duration-sec",
        dest="segment_sec",
        type=float,
        default=0.5,
        help="每段时间（秒，默认 0.5）",
    )
    parser.add_argument(
        "--speed-range-mps",
        dest="speed_range",
        type=parse_float_pair,
        default="0.5,1.5",
        help="声源移动速度范围 m/s，如 '0.5,1.5'",
    )
    parser.add_argument(
        "--trajectory-type",
        dest="traj_type",
        choices=["linear", "waypoint", "both"],
        default="both",
        help="轨迹类型（默认 both：每个场景随机选择）",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="严格发布门禁：检查采样率、长度、有限性和逐段峰值；异常时返回非零且不发布 manifest",
    )
    # 指纹默认跳过（60 万文件时 SHA-256 会多花约 5 分钟 CPU）；--inventory 显式开启
    parser.add_argument(
        "--inventory",
        dest="inventory",
        action="store_true",
        default=False,
        help="计算并记录每条 RIR 的 SHA-256 指纹（默认跳过；大数据集时较慢）",
    )
    parser.add_argument(
        "--no-inventory",
        dest="inventory",
        action="store_false",
        help="跳过 SHA-256 指纹（默认行为；inventory 中 sha256 为 null）",
    )
    parser.add_argument(
        "--validation-peak-threshold",
        dest="abnormal_threshold",
        type=float,
        default=1e-6,
        help="异常 RIR 最大幅值阈值（默认 1e-6）",
    )
    return parser


def _resolve_plan(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[list[str], list[str], list[int | str], dict[str, int], bool, Path]:
    """校验并规范化参数，推导执行计划。

    Returns:
        (backends, fields, gpu_ids, selected_counts, split_gpurir_static_moving, out_dir)
    """
    if args.out_dir is None:
        parser.error("--output-dir is required unless it is provided in --config")
    try:
        normalize_segment_range(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.fs or any(value <= 0 for value in args.fs):
        parser.error("--sample-rate-hz must contain positive integers")
    if len(set(args.fs)) != len(args.fs):
        parser.error("--sample-rate-hz must not contain duplicate sample rates")
    for name, value in {
        "--workers": args.workers,
        "--shards": args.shards,
    }.items():
        if value <= 0:
            parser.error(f"{name} must be > 0")
    for name, value in {
        "--near-rirs": args.n_rirs_near,
        "--far-rirs": args.n_rirs_far,
        "--total-rirs": args.n_rirs_total,
    }.items():
        if value is not None and value < 0:
            parser.error(f"{name} must be >= 0")
    for name, ratio in {
        "--static-moving-ratio": args.static_moving_ratio,
        "--near-far-ratio": args.near_far_ratio,
    }.items():
        if not all(math.isfinite(value) and value >= 0 for value in ratio) or sum(ratio) <= 0:
            parser.error(f"{name} must contain finite non-negative values with a positive sum")
    if not math.isfinite(args.c) or args.c <= 0.0:
        parser.error("--sound-speed-mps must be finite and > 0")
    if not math.isfinite(args.abnormal_threshold) or args.abnormal_threshold < 0.0:
        parser.error("--validation-peak-threshold must be finite and >= 0")

    args.rt60_range = args.rt60_range or DEFAULT_RT60_RANGE

    for name in (
        "rt60_range",
        "dist_range_near",
        "dist_range_far",
        "room_length_range",
        "room_width_range",
        "room_height_range",
    ):
        lo, hi = getattr(args, name)
        if not all(math.isfinite(value) for value in (lo, hi)) or lo > hi:
            parser.error(f"--{name} must be finite and satisfy lo <= hi")
        if name == "rt60_range" and lo <= 0.0:
            parser.error("--rt60-range-sec values must be > 0")
        if name.startswith("dist_range") and lo < MIN_SOURCE_MICROPHONE_DISTANCE:
            parser.error(
                f"--{name} lower bound must be >= {MIN_SOURCE_MICROPHONE_DISTANCE} m"
            )
        if name.startswith("room_") and lo < 1.0:
            parser.error(f"--{name} lower bound must be >= 1.0 m")
    if not args.allow_distance_overlap:
        near_lo, near_hi = args.dist_range_near
        far_lo, far_hi = args.dist_range_far
        if near_lo < far_hi and far_lo < near_hi:
            parser.error(
                "near and far distance ranges overlap; set allow_overlap=true in the profile"
            )

    explicit_field_counts = normalize_counts(args)
    split_gpurir_static_moving = (
        args.method == "gpurir" and not args.moving and not explicit_field_counts
    )

    for fs in args.fs:
        for field, distance_range in (
            ("near", args.dist_range_near),
            ("far", args.dist_range_far),
        ):
            explicit_samples = args.n_samples_near if field == "near" else args.n_samples_far
            recommended = recommended_rir_length_samples(
                fs, args.rt60_range, distance_range, args.c
            )
            if explicit_samples is None:
                continue
            if explicit_samples <= 0:
                parser.error(f"--{field}-rir-length-samples must be > 0")
            if explicit_samples < recommended:
                logger.warning(
                    "--%s-rir-length-samples=%d is shorter than the recommended %d samples "
                    "for fs=%d and RT60 up to %.2fs",
                    field,
                    explicit_samples,
                    recommended,
                    fs,
                    args.rt60_range[1],
                )

    if args.moving and args.method in {"rirgen", "fram"}:
        parser.error("--motion-enabled is only supported by the gpurir backend (got --method %s)" % args.method)
    selected_counts = {
        "near": args.n_rirs_near if args.field in ("near", "both") else 0,
        "far": args.n_rirs_far if args.field in ("far", "both") else 0,
    }
    moving_task_requested = (
        sum(selected_counts.values()) > 0
        and (args.moving or (split_gpurir_static_moving and args.static_moving_ratio[1] > 0))
    )
    if args.format in {"wav", "flac"} and moving_task_requested:
        parser.error(
            f"--format {args.format} cannot be used when effective moving tasks are requested"
        )
    # 数值形式校验无条件（nan/inf 无法写入 summary JSON）
    if not math.isfinite(args.segment_sec):
        parser.error("--segment-duration-sec must be finite")
    if not all(math.isfinite(value) for value in args.speed_range):
        parser.error("--speed-range-mps must be finite")
    # 移动参数语义下限校验（避免无效轨迹/退化场景）；纯静态任务不约束
    if moving_task_requested:
        if args.segments < 2:
            parser.error("--n-segments must be >= 2")
        if args.segment_sec <= 0:
            parser.error("--segment-duration-sec must be > 0")
        if not (0 < args.speed_range[0] < args.speed_range[1]):
            parser.error("--speed-range-mps must satisfy 0 < lo < hi")

    # 确定要运行的 backend
    available = list_available_backends()
    logger.info("Available backends: %s", available)

    if args.method == "all":
        # moving 场景只能使用 gpuRIR；普通 all 只生成各 backend 的静态 RIR。
        requested = ["gpurir"] if args.moving else ["rirgen", "fram", "gpurir"]
        backends = [backend for backend in requested if backend in available]
        if not backends:
            logger.error("No RIR backend is available. Please install at least one dependency.")
            for name in ["rirgen", "fram", "gpurir"]:
                logger.error("  - %s", get_backend(name).availability_error())
            # 与 worker 失败/非空目录一致，使用退出码 1（而非 argparse 的 2）
            raise SystemExit(1)
        logger.info("--method all will use: %s", backends)
    else:
        backends = [args.method]
        if args.method not in available:
            logger.error(
                "Backend '%s' is not available. Please install the required dependency.",
                args.method,
            )
            logger.error("  - %s", get_backend(args.method).availability_error())
            raise SystemExit(1)

    fields = []
    if args.field in ("near", "both"):
        fields.append("near")
    if args.field in ("far", "both"):
        fields.append("far")

    has_gpurir_work = "gpurir" in backends and sum(selected_counts.values()) > 0
    if has_gpurir_work:
        try:
            gpu_ids = _resolve_gpu_ids(args.gpus)
        except (argparse.ArgumentTypeError, ValueError) as exc:
            parser.error(str(exc))
        if not gpu_ids:
            if args.method == "all" and not args.moving and args.gpus is None:
                logger.warning(
                    "Skipping gpurir in --method all because no GPU is available"
                )
                backends.remove("gpurir")
                if not backends:
                    parser.error("no runnable RIR backend is available")
            else:
                parser.error("no GPU is available for gpurir generation")
    else:
        gpu_ids = []

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and not out_dir.is_dir():
        parser.error(f"output path exists and is not a directory: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.error(
            "%s already exists and is not empty; delete it before generating a new dataset",
            out_dir,
        )
        # 与 worker 失败/backend 缺失一致，使用退出码 1（而非 argparse 的 2）
        raise SystemExit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    return backends, fields, gpu_ids, selected_counts, split_gpurir_static_moving, out_dir


def _effective_format(args: argparse.Namespace) -> str:
    """取实际输出格式；auto 时静态/移动统一为 pkl。"""
    if args.format != "auto":
        return args.format
    return "pkl"



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw_argv or "-h" in raw_argv:
        parser.parse_args(raw_argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", dest="rir_config_path", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args(raw_argv)
    if bootstrap_args.rir_config_path is None:
        parser.error("--config is required")
    try:
        config_path = bootstrap_args.rir_config_path.expanduser().resolve()
        config = GeneratorConfig.from_yaml(config_path)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"failed to load RIR config: {exc}")
    defaults = config.to_runtime_namespace()
    if defaults:
        defaults["rir_config_path"] = config_path
        defaults["effective_config"] = config.to_dict()
        parser.set_defaults(**defaults)
    rt60_occurrences = sum(
        arg == "--rt60-range-sec" or arg.startswith("--rt60-range-sec=")
        for arg in raw_argv
    )
    if rt60_occurrences > 1:
        parser.error("--rt60-range-sec may be specified only once")
    args = parser.parse_args(raw_argv)
    if args.dump_effective_config:
        try:
            normalize_counts(args)
            normalize_segment_range(args)
            effective_config = config.with_runtime_overrides(args, args.out_dir)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(effective_config.to_dict(), ensure_ascii=False, indent=2))
        return 0
    (
        backends,
        fields,
        gpu_ids,
        selected_counts,
        split_gpurir_static_moving,
        out_dir,
    ) = _resolve_plan(args, parser)
    args.effective_config = config.with_runtime_overrides(args, out_dir).to_dict()

    all_entries: list[tuple[str, int, int, str]] = []
    planned_specs: list[dict[str, Any]] = []
    total_requested = 0  # 请求生成总数，用于末尾统计失败数
    failed = 0  # 显式统计 worker 失败数（不依赖 uid 去重后的数量差）
    start_time = time.time()

    for fs in args.fs:
        for backend in backends:
            for field in fields:
                n_field = selected_counts[field]
                if args.moving:
                    mode_counts = [("moving", n_field)]
                elif split_gpurir_static_moving:
                    static_count, moving_count = split_count(
                        n_field, args.static_moving_ratio
                    )
                    mode_counts = [("static", static_count), ("moving", moving_count)]
                else:
                    mode_counts = [("static", n_field)]

                for mode, n_rirs in mode_counts:
                    if n_rirs == 0:
                        continue
                    fmt = _effective_format(args)
                    cfg = build_field_config(args, field, fs, mode=mode)
                    runner_field = field
                    planned_specs.append(
                        {
                            "backend": backend,
                            "field": field,
                            "mode": mode,
                            "fs": fs,
                            "requested": n_rirs,
                            "format": fmt,
                        }
                    )
                    total_requested += n_rirs
                    if backend == "gpurir":
                        entries = rir_executor.run_gpu_backend(
                            field=runner_field,
                            n_rirs=n_rirs,
                            cfg=cfg,
                            output_dir=out_dir,
                            output_format=fmt,
                            write_metadata_json=args.write_meta_json,
                            gpu_ids=gpu_ids,
                            shards=args.shards,
                            mode=mode,
                        )
                    else:
                        entries = rir_executor.run_cpu_backend(
                            backend=backend,
                            field=runner_field,
                            n_rirs=n_rirs,
                            cfg=cfg,
                            output_dir=out_dir,
                            output_format=fmt,
                            write_metadata_json=args.write_meta_json,
                            workers=args.workers,
                            shards=args.shards,
                            mode=mode,
                        )
                    failed += max(0, n_rirs - len(entries))
                    all_entries.extend(entries)

    # 去重并按 uid 排序
    seen: set[str] = set()
    unique_entries: list[tuple[str, int, int, str]] = []
    for e in all_entries:
        if e[0] not in seen:
            seen.add(e[0])
            unique_entries.append(e)
    unique_entries.sort(key=lambda x: x[0])
    duplicate_entries = len(all_entries) - len(unique_entries)

    # worker 失败时不发布索引集合，避免下游把残缺生成树误认为完整数据集。
    if failed > 0:
        logger.error("Failed: %d/%d RIRs not generated", failed, total_requested)
        return 1
    if duplicate_entries > 0:
        logger.error(
            "Refusing to publish: %d duplicate RIR UID entries were generated",
            duplicate_entries,
        )
        return 1

    # --validate 是严格发布门禁：异常 RIR 存在时不发布 rirs.scp / rirs_meta.json。
    if args.validate:
        logger.info("Validating generated RIRs...")
        abnormal = validate_rirs(unique_entries, args.abnormal_threshold)
        if abnormal:
            logger.error(
                "Validation failed: %d/%d RIRs are abnormal", len(abnormal), len(unique_entries)
            )
            for line in abnormal[:20]:
                logger.error("  %s", line)
            if len(abnormal) > 20:
                logger.error("  ... and %d more", len(abnormal) - 20)
            return 1
        logger.info("Validation passed: no abnormal RIRs")

    summary = build_summary(
        output_dir=out_dir,
        args=args,
        backends=backends,
        fields=fields,
        gpu_ids=gpu_ids,
        unique_entries=unique_entries,
        planned_specs=planned_specs,
        failed=failed,
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        generator_version_history=GENERATOR_VERSION_HISTORY,
    )
    publish_manifest(out_dir, unique_entries, summary)


    elapsed = time.time() - start_time
    logger.info(
        "All done. Total: %d RIRs in %.2fs (%.4fs per RIR)",
        len(unique_entries),
        elapsed,
        elapsed / max(len(unique_entries), 1),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
