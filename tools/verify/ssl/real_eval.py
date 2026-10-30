"""在真实多通道录音上验证 SSL 方位输出。

该工具只负责离线验证，不是新的 ``data.mode`` 或训练入口。输入可以是带有
显式几何和时间标签的 JSONL 清单，也可以直接读取 upstream FN-SSL 使用的
LOCATA 目录。模型特征由录音观测计算；位置、距离和 VAD 标签只在指标阶段
使用。双麦克风默认只评估可辨识的水平半平面 ``[-pi/2, pi/2]``，不会把
前后/左右歧义折叠成一个看似更好的分数。

示例（已有本地 LOCATA 和模型权重时）：

.. code-block:: bash

    PYTHONPATH=. python tools/verify/ssl/real_eval.py \\
      --locata-root data/ssl/LOCATA --task 1 --array dicit \\
      --mic-indices 0,1 --config configs/ssl/fnssl_dynamic.yaml \\
      --checkpoint /path/to/model.pt --output /tmp/ssl-locata.json

LOCATA 的许可、下载和文件布局由数据准备脚本负责；本工具不会自动下载或
复制大型数据集。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from core.utils.config import load_merged_config
from evaluate.checkpoint_loading import CheckpointFamily, load_checkpoint
from tasks.ssl.config import resolve_ssl_num_tracks
from tasks.ssl.data.doa_mapping import DOAMappingConfig, DPIPDToDOAMapper
from tasks.ssl.data.projection import (
    SSLProjectionConfig,
    _reference_ipd_features,
    _raw_stft_real_imag,
    _stft,
)
from tasks.ssl.model_io import (
    ModelSpec,
    resolve_ssl_model_spec,
    validate_ssl_model_io,
    validate_ssl_model_output,
)
from tasks.ssl.models import build_model


_DICIT_MIC_POSITIONS = np.asarray(
    (
        (0.96, 0.00, 0.00),
        (0.64, 0.00, 0.00),
        (0.32, 0.00, 0.00),
        (0.16, 0.00, 0.00),
        (0.08, 0.00, 0.00),
        (0.04, 0.00, 0.00),
        (0.00, 0.00, 0.00),
        (0.96, 0.00, 0.32),
        (-0.04, 0.00, 0.00),
        (-0.08, 0.00, 0.00),
        (-0.16, 0.00, 0.00),
        (-0.32, 0.00, 0.00),
        (-0.64, 0.00, 0.00),
        (-0.96, 0.00, 0.00),
        (-0.96, 0.00, 0.32),
    ),
    dtype=np.float32,
)
_DUALCH_MIC_POSITIONS = np.asarray(((-0.04, 0.00, 0.00), (0.04, 0.00, 0.00)), dtype=np.float32)
_FLOAT_EPS = 1e-8


@dataclass(frozen=True)
class EvalRecord:
    """一条可审计的真实录音记录。

    ``label_times_s``、``azimuth_rad`` 和 ``vad`` 位于同一条标签时间轴。标签
    可以是稀疏的 LOCATA 定位时间戳；评估时会先插值到 STFT 帧，再按模型的
    ``time_stride`` 做左对齐分组。
    """

    key: str
    audio: Path
    sample_rate: int | None
    mic_positions: np.ndarray
    label_times_s: np.ndarray
    azimuth_rad: np.ndarray
    vad: np.ndarray
    coordinate_frame: str = "array_local_xyz"
    reference_channel: int = 0
    channel_indices: tuple[int, ...] | None = None

    def validate(self) -> "EvalRecord":
        if not self.key.strip():
            raise ValueError("evaluation record key must be non-empty")
        if not self.audio.is_file():
            raise FileNotFoundError(f"evaluation audio does not exist: {self.audio}")
        if self.sample_rate is not None and (type(self.sample_rate) is not int or self.sample_rate < 1000):
            raise ValueError("sample_rate must be an exact integer >= 1000 when provided")
        positions = np.asarray(self.mic_positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
            raise ValueError("mic_positions must have shape [C,3] with C >= 2")
        if not np.isfinite(positions).all():
            raise ValueError("mic_positions must be finite")
        times = np.asarray(self.label_times_s, dtype=np.float64)
        azimuth = np.asarray(self.azimuth_rad, dtype=np.float64)
        vad = np.asarray(self.vad, dtype=np.float64)
        if times.ndim != 1 or azimuth.ndim != 1 or vad.ndim != 1:
            raise ValueError("label_times_s, azimuth_rad and vad must be one-dimensional")
        if not (times.size == azimuth.size == vad.size) or times.size < 1:
            raise ValueError("evaluation labels must have equal non-empty lengths")
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
            raise ValueError("label_times_s must be finite and non-decreasing")
        if np.any(~np.isfinite(azimuth)) or np.any(~np.isfinite(vad)):
            raise ValueError("azimuth_rad and vad must be finite")
        if np.any(azimuth < -math.pi - _FLOAT_EPS) or np.any(azimuth > math.pi + _FLOAT_EPS):
            raise ValueError("azimuth_rad must use the canonical [-pi, pi] range")
        if np.any((vad < 0.0) | (vad > 1.0)):
            raise ValueError("vad must be in [0,1]")
        if type(self.reference_channel) is not int or not 0 <= self.reference_channel < positions.shape[0]:
            raise ValueError("reference_channel is outside mic_positions")
        if self.channel_indices is not None:
            indices = tuple(int(index) for index in self.channel_indices)
            if len(indices) != positions.shape[0] or min(indices, default=-1) < 0:
                raise ValueError("channel_indices must match mic_positions and be non-negative")
        else:
            indices = None
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            raise ValueError("coordinate_frame must be a non-empty string")
        return EvalRecord(
            key=self.key,
            audio=self.audio,
            sample_rate=self.sample_rate,
            mic_positions=np.ascontiguousarray(positions),
            label_times_s=times,
            azimuth_rad=azimuth,
            vad=vad,
            coordinate_frame=self.coordinate_frame,
            reference_channel=self.reference_channel,
            channel_indices=indices,
        )


def _as_array(value: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    return result


def record_from_mapping(value: Mapping[str, Any], *, base_dir: Path) -> EvalRecord:
    """从 JSONL 记录构造严格的单源评估记录。

    支持的标签字段为 ``labels.{times_s,azimuth_rad,vad}``，或同名顶层字段。
    不接受 ``source_distances`` 作为推理输入；该字段即使出现在 provenance
    中也不会被读取。
    """

    if not isinstance(value, Mapping):
        raise TypeError("JSONL evaluation item must be an object")
    raw_audio = value.get("audio", value.get("path", value.get("wav")))
    if not isinstance(raw_audio, str) or not raw_audio.strip():
        raise ValueError("evaluation item requires non-empty audio/path")
    audio = Path(raw_audio).expanduser()
    if not audio.is_absolute():
        audio = (base_dir / audio).resolve()
    geometry = value.get("geometry", value.get("mic_positions"))
    if isinstance(geometry, Mapping):
        positions = geometry.get("positions_m", geometry.get("mic_positions"))
        reference = geometry.get("reference_channel", value.get("reference_channel", 0))
    else:
        positions = geometry
        reference = value.get("reference_channel", 0)
    if positions is None:
        raise ValueError(f"evaluation item {raw_audio!r} requires geometry.positions_m")
    labels = value.get("labels", value)
    if not isinstance(labels, Mapping):
        raise TypeError("evaluation labels must be an object")
    azimuth = labels.get("azimuth_rad")
    if azimuth is None:
        raise ValueError("evaluation item requires labels.azimuth_rad in radians")
    times = labels.get("times_s", labels.get("label_times_s"))
    if times is None:
        if np.asarray(azimuth).ndim != 0:
            raise ValueError("vector azimuth labels require labels.times_s")
        times = [0.0]
    vad = labels.get("vad", labels.get("vad_probability"))
    if vad is None:
        vad = np.ones_like(np.asarray(azimuth, dtype=np.float64))
    azimuth_arr = _as_array(azimuth, name="azimuth_rad")
    times_arr = _as_array(times, name="label_times_s")
    vad_arr = _as_array(vad, name="vad")
    if azimuth_arr.size == 1 and times_arr.size > 1:
        azimuth_arr = np.repeat(azimuth_arr, times_arr.size)
    if vad_arr.size == 1 and times_arr.size > 1:
        vad_arr = np.repeat(vad_arr, times_arr.size)
    return EvalRecord(
        key=str(value.get("key", audio.stem)),
        audio=audio,
        sample_rate=value.get("sample_rate"),
        mic_positions=np.asarray(positions, dtype=np.float32),
        label_times_s=times_arr,
        azimuth_rad=azimuth_arr,
        vad=vad_arr,
        coordinate_frame=str(value.get("coordinate_frame", "array_local_xyz")),
        reference_channel=int(reference),
        channel_indices=(
            tuple(int(index) for index in value["channel_indices"])
            if value.get("channel_indices") is not None
            else None
        ),
    ).validate()


def iter_manifest(path: str | Path, *, max_items: int | None = None) -> Iterator[EvalRecord]:
    """遍历带显式 geometry/labels 的 JSONL 清单。"""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"evaluation manifest does not exist: {manifest}")
    count = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                yield record_from_mapping(value, base_dir=manifest.parent)
            except Exception as exc:
                raise ValueError(f"invalid evaluation manifest line {line_no}: {exc}") from exc
            count += 1
            if max_items is not None and count >= max_items:
                break


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _column(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"LOCATA metadata column {name!r} is missing or non-numeric") from exc


def _locata_geometry(array: str) -> np.ndarray:
    if array == "dicit":
        return _DICIT_MIC_POSITIONS.copy()
    if array in {"dualch", "dummy"}:
        return _DUALCH_MIC_POSITIONS.copy()
    raise ValueError(
        f"LOCATA array {array!r} has no registered local geometry; use JSONL with explicit geometry"
    )


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _locata_record(directory: Path, *, array: str, mic_indices: Sequence[int]) -> EvalRecord:
    array_pos_rows = _read_tsv(directory / f"position_array_{array}.txt")
    time_rows = _read_tsv(directory / "required_time.txt")
    if len(array_pos_rows) != len(time_rows) or not array_pos_rows:
        raise ValueError(f"LOCATA array/time metadata length mismatch: {directory}")
    required_time = (
        _column(time_rows, "hour") * 3600.0
        + _column(time_rows, "minute") * 60.0
        + _column(time_rows, "second")
    )
    required_time -= required_time[0]
    array_pos = np.column_stack([_column(array_pos_rows, axis) for axis in ("x", "y", "z")])
    rotation = np.empty((len(array_pos_rows), 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rotation[:, i, j] = _column(array_pos_rows, f"rotation_{i + 1}{j + 1}")
    source_files = sorted(directory.glob("position_source_*.txt"), key=lambda p: _natural_key(p.name))
    if len(source_files) != 1:
        raise ValueError(f"LOCATA evaluator supports exactly one source, got {len(source_files)}: {directory}")
    source_rows = _read_tsv(source_files[0])
    if not source_rows:
        raise ValueError(f"LOCATA source metadata is empty: {source_files[0]}")
    source_position = np.column_stack([_column(source_rows, axis) for axis in ("x", "y", "z")])
    source_times = np.linspace(required_time[0], required_time[-1], source_position.shape[0])
    array_interp = np.column_stack(
        [np.interp(source_times, required_time, array_pos[:, axis]) for axis in range(3)]
    )
    rotation_interp = np.empty((source_position.shape[0], 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rotation_interp[:, i, j] = np.interp(source_times, required_time, rotation[:, i, j])
    # 与 upstream LocataDataset 保持一致：全局位置差右乘阵列旋转矩阵。
    local_position = np.einsum("ni,nij->nj", source_position - array_interp, rotation_interp)
    azimuth = np.arctan2(local_position[:, 1], local_position[:, 0])
    vad = np.ones_like(azimuth)
    vad_files = sorted(directory.glob(f"VAD_{array}_*.txt"), key=lambda p: _natural_key(p.name))
    if vad_files:
        vad_rows = _read_tsv(vad_files[0])
        if vad_rows and "VAD" in vad_rows[0]:
            vad_values = _column(vad_rows, "VAD")
            source_stem = source_files[0].stem.removeprefix("position_")
            source_audio = directory / f"audio_{source_stem}.wav"
            vad_rate = 48000.0
            if source_audio.is_file():
                try:
                    import soundfile as sf

                    vad_rate = float(sf.info(str(source_audio)).samplerate)
                except Exception:
                    pass
            vad_times = np.arange(vad_values.size, dtype=np.float64) / vad_rate
            vad = np.interp(source_times, vad_times, vad_values, left=vad_values[0], right=vad_values[-1])
            vad = np.clip(vad, 0.0, 1.0)
    geometry = _locata_geometry(array)
    indices = tuple(int(index) for index in mic_indices)
    if not indices or min(indices) < 0 or max(indices) >= geometry.shape[0] or len(set(indices)) != len(indices):
        raise ValueError(f"invalid LOCATA mic_indices={indices!r} for {array!r}")
    selected_geometry = geometry[list(indices)]
    return EvalRecord(
        key=str(directory),
        audio=directory / f"audio_array_{array}.wav",
        sample_rate=None,
        mic_positions=selected_geometry,
        label_times_s=source_times,
        azimuth_rad=azimuth,
        vad=vad,
        coordinate_frame="array_local_xyz",
        reference_channel=0,
        channel_indices=indices,
    ).validate()


def iter_locata(
    root: str | Path,
    *,
    task: int,
    array: str,
    mic_indices: Sequence[int],
    max_items: int | None = None,
) -> Iterator[EvalRecord]:
    """读取 upstream LOCATA ``eval``/``dev`` 目录中的单源 recording。"""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"LOCATA root does not exist: {root_path}")
    bases = [root_path]
    for name in ("eval", "dev"):
        candidate = root_path / name
        if candidate.is_dir():
            bases.append(candidate)
    seen: set[Path] = set()
    yielded = 0
    for base in bases:
        task_dir = base / f"task{int(task)}"
        if not task_dir.is_dir():
            continue
        for recording in sorted(task_dir.iterdir(), key=lambda p: _natural_key(p.name)):
            directory = recording / array
            if directory in seen or not directory.is_dir():
                continue
            seen.add(directory)
            record = _locata_record(directory, array=array, mic_indices=mic_indices)
            yield record
            yielded += 1
            if max_items is not None and yielded >= max_items:
                return
    if not seen:
        raise FileNotFoundError(
            f"no LOCATA task{int(task)} / array {array!r} recordings under {root_path}"
        )


def parse_azimuth_domain(value: str) -> tuple[float, float]:
    """解析 ``halfplane`` 或显式 ``min_deg,max_deg`` 方位域。"""

    if value == "halfplane":
        return -math.pi / 2.0, math.pi / 2.0
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("azimuth domain must be 'halfplane' or '<min_deg>,<max_deg>'")
    lower, upper = (math.radians(float(part)) for part in parts)
    if not -math.pi <= lower < upper <= math.pi:
        raise ValueError("azimuth domain must satisfy -180 <= min < max <= 180 degrees")
    return lower, upper


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def compute_azimuth_metrics(
    prediction_rad: Sequence[float] | np.ndarray,
    target_rad: Sequence[float] | np.ndarray,
    *,
    target_vad: Sequence[float] | np.ndarray,
    prediction_valid: Sequence[bool] | np.ndarray,
    domain: tuple[float, float],
    vad_threshold: float = 0.5,
    accuracy_threshold_deg: float = 10.0,
) -> dict[str, Any]:
    """计算显式 VAD/域门控后的 circular azimuth 指标。"""

    prediction = np.asarray(prediction_rad, dtype=np.float64)
    target = np.asarray(target_rad, dtype=np.float64)
    vad = np.asarray(target_vad, dtype=np.float64)
    predicted_valid = np.asarray(prediction_valid, dtype=bool)
    if not (prediction.ndim == target.ndim == vad.ndim == predicted_valid.ndim == 1):
        raise ValueError("azimuth metrics expect one-dimensional frame arrays")
    if not (prediction.size == target.size == vad.size == predicted_valid.size):
        raise ValueError("azimuth metric arrays must have equal lengths")
    if not 0.0 <= vad_threshold <= 1.0:
        raise ValueError("vad_threshold must be in [0,1]")
    if not 0.0 < accuracy_threshold_deg <= 180.0:
        raise ValueError("accuracy_threshold_deg must be in (0,180]")
    lower, upper = domain
    target_in_domain = np.isfinite(target) & (target >= lower - _FLOAT_EPS) & (target <= upper + _FLOAT_EPS)
    target_active = np.isfinite(target) & (vad >= vad_threshold) & target_in_domain
    joint = target_active & predicted_valid & np.isfinite(prediction)
    errors = np.abs(_wrap_angle(prediction[joint] - target[joint]))
    metrics: dict[str, Any] = {
        "azimuth_domain_rad": [float(lower), float(upper)],
        "azimuth_domain_deg": [math.degrees(lower), math.degrees(upper)],
        "vad_threshold": float(vad_threshold),
        "accuracy_threshold_deg": float(accuracy_threshold_deg),
        "frames_total": int(target.size),
        "valid_vad_frames": int(target_active.sum()),
        "excluded_out_of_domain_frames": int((np.isfinite(target) & ~target_in_domain).sum()),
        "predicted_valid_frames": int((predicted_valid & target_active).sum()),
        "evaluated_frames": int(joint.sum()),
        "coverage": float(joint.sum() / target_active.sum()) if target_active.any() else None,
        "mae_deg": float(np.degrees(errors.mean())) if errors.size else None,
        "median_deg": float(np.degrees(np.median(errors))) if errors.size else None,
        "p90_deg": float(np.degrees(np.quantile(errors, 0.90))) if errors.size else None,
        "accuracy": float((np.degrees(errors) <= accuracy_threshold_deg).mean()) if errors.size else None,
    }
    return metrics


def _resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("resampling LOCATA audio requires scipy") from exc
    from math import gcd

    divisor = gcd(int(source_rate), int(target_rate))
    return np.asarray(
        resample_poly(waveform, target_rate // divisor, source_rate // divisor, axis=-1),
        dtype=np.float32,
    )


def _load_observation(record: EvalRecord, *, target_rate: int, mic_count: int) -> torch.Tensor:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("real SSL evaluation requires the soundfile package") from exc
    waveform, sample_rate = sf.read(str(record.audio), always_2d=True, dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32).T
    if record.channel_indices is None:
        if waveform.shape[0] < mic_count:
            raise ValueError(f"{record.key}: audio has {waveform.shape[0]} channels, expected {mic_count}")
        waveform = waveform[:mic_count]
    else:
        if max(record.channel_indices, default=-1) >= waveform.shape[0]:
            raise ValueError(
                f"{record.key}: requested audio channel {max(record.channel_indices)} "
                f"but file has {waveform.shape[0]} channels"
            )
        waveform = waveform[list(record.channel_indices)]
    if record.sample_rate is not None and int(record.sample_rate) != int(sample_rate):
        raise ValueError(f"{record.key}: manifest sample_rate disagrees with audio header")
    waveform = _resample_waveform(waveform, int(sample_rate), int(target_rate))
    if waveform.shape[-1] < 2:
        raise ValueError(f"{record.key}: audio is too short")
    return torch.from_numpy(np.ascontiguousarray(waveform))


def _model_output_stride(
    model: torch.nn.Module | None,
    spec: ModelSpec,
    *,
    config: Mapping[str, Any] | None = None,
) -> int:
    """返回真实评估前向的时间压缩倍率，并校验模型内部 stride 声明。"""
    config_model = {} if config is None else config.get("model", {})
    if not isinstance(config_model, Mapping):
        raise TypeError("model config must be a mapping")
    model_type = getattr(spec, "model_type", None) or config_model.get("type")
    raw_args = config_model.get("args", {})
    if not isinstance(raw_args, Mapping):
        raise TypeError("model.args must be a mapping")

    if model_type == "FixedArrayIPDNet":
        # 真实评估不传 offline_inference；该模型的普通 forward 保留每个输入帧，
        # downsample 只在显式离线压缩分支使用。
        return 1
    if model_type == "IPDNet2":
        stride = (
            getattr(model, "stride", None)
            if model is not None
            else raw_args.get("stride", raw_args.get("time_compression_ratio", 5))
        )
        ratio = (
            getattr(model, "time_compression_ratio", None)
            if model is not None
            else raw_args.get("time_compression_ratio", 5)
        )
        if type(stride) is not int or type(ratio) is not int or stride != ratio:
            raise ValueError(
                "IPDNet2 stride must equal time_compression_ratio; "
                f"got stride={stride!r}, time_compression_ratio={ratio!r}"
            )
        return ratio
    if model_type in {"FNSSL", "IPDNetFixed", "IPDNetVariable"}:
        stride = (
            getattr(model, "stride", None)
            if model is not None
            else raw_args.get("stride", 12)
        )
        if type(stride) is not int or stride < 1:
            raise ValueError(
                f"{model_type} must expose a positive integer model stride, got {stride!r}"
            )
        return stride
    raise ValueError(f"real SSL evaluation has no time-axis contract for {model_type!r}")


def _validate_model_time_stride(
    model: torch.nn.Module,
    spec: ModelSpec,
    projection: SSLProjectionConfig,
) -> int:
    model_stride = _model_output_stride(model, spec)
    if model_stride != projection.output_stride:
        raise ValueError(
            f"{spec.model_type} output stride {model_stride} conflicts with "
            f"ssl.time_stride/projection.output_stride={projection.output_stride}"
        )
    return model_stride


def _projection_from_config(
    config: Mapping[str, Any],
    spec: ModelSpec,
    *,
    sample_rate: int,
    model: torch.nn.Module | None = None,
) -> SSLProjectionConfig:
    stft = dict(config.get("stft", {}))
    ssl = dict(config.get("ssl", {}))
    feature_contract = spec.feature_contract
    if feature_contract == "multichannel_stft_real_imag":
        input_representation = "multichannel_stft_real_imag"
    elif feature_contract == "reference_ipd":
        input_representation = "reference_ipd"
    else:
        raise ValueError(f"unsupported model feature contract: {feature_contract!r}")
    if "output_stride" in ssl:
        raise ValueError("ssl.output_stride is retired; use ssl.time_stride")
    if "time_stride" in ssl:
        output_stride = ssl["time_stride"]
    else:
        output_stride = _model_output_stride(model, spec, config=config)
    if type(output_stride) is not int or output_stride < 1:
        raise ValueError("ssl.time_stride must be an exact positive integer")
    declared_sound_speed = ssl.get("sound_speed")
    if declared_sound_speed is None:
        data = config.get("data", {})
        if isinstance(data, Mapping):
            for split in ("train", "validation"):
                section = data.get(split)
                raw_simulation = section.get("simulation_cfg") if isinstance(section, Mapping) else None
                if isinstance(raw_simulation, Mapping):
                    from core.data.simulation_configs import SimulationConfigs

                    declared_sound_speed = float(
                        SimulationConfigs.from_dict(dict(raw_simulation), strict=True).gpurir_sound_speed_mps
                    )
                    break
    if declared_sound_speed is None:
        raise ValueError(
            "real SSL evaluation requires canonical simulation_cfg.gpurir_sound_speed_mps"
        )
    return SSLProjectionConfig(
        n_fft=int(stft.get("n_fft", 512)),
        hop_length=int(stft.get("hop_length", 256)),
        win_length=int(stft.get("win_length", stft.get("n_fft", 512))),
        window=str(stft.get("window", "hamming")),
        center=bool(stft.get("center", False)),
        reference_channel=int(ssl.get("reference_channel", 0)),
        input_representation=input_representation,
        target_representation="dpipd_near_field" if spec.field_mode == "near_field" else "dpipd_far_field",
        exclude_dc=bool(ssl.get("exclude_dc", False)),
        normalization=str(ssl.get("normalization", "none")),
        forgetting_norm_sample_length=int(ssl.get("forgetting_norm_sample_length", 298)),
        output_stride=output_stride,
        sound_speed=float(declared_sound_speed),
        pair_layout="reference_to_all",
        phase_sign=float(ssl.get("phase_sign", 1.0)),
        vad_mode=ssl.get("vad_mode", "direct_path"),
    )


def _align_labels(
    record: EvalRecord,
    *,
    frame_count: int,
    projection: SSLProjectionConfig,
    output_count: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_times = (
        np.arange(frame_count, dtype=np.float64) * projection.hop_length
        + (projection.win_length or projection.n_fft) / 2.0
    ) / sample_rate
    if record.label_times_s.size == 1:
        inside = np.ones(frame_count, dtype=bool)
        azimuth = np.full(frame_count, record.azimuth_rad[0], dtype=np.float64)
        vad = np.full(frame_count, record.vad[0], dtype=np.float64)
    else:
        inside = (frame_times >= record.label_times_s[0]) & (frame_times <= record.label_times_s[-1])
        unit = np.exp(1j * record.azimuth_rad)
        real = np.interp(frame_times, record.label_times_s, unit.real)
        imag = np.interp(frame_times, record.label_times_s, unit.imag)
        azimuth = np.angle(real + 1j * imag)
        vad = np.interp(frame_times, record.label_times_s, record.vad)
    stride = projection.output_stride
    expected_output_count = frame_count // stride
    if output_count != expected_output_count:
        raise ValueError(
            "model output time axis does not match the configured projection stride: "
            f"input_frames={frame_count}, stride={stride}, "
            f"expected_output_frames={expected_output_count}, actual_output_frames={output_count}"
        )
    usable = expected_output_count
    azimuth = azimuth[: usable * stride].reshape(usable, stride)
    vad = vad[: usable * stride].reshape(usable, stride)
    inside = inside[: usable * stride].reshape(usable, stride)
    return np.angle(np.mean(np.exp(1j * azimuth), axis=1)), np.mean(vad, axis=1), inside.all(axis=1)


def evaluate_records(
    records: Iterable[EvalRecord],
    *,
    model: torch.nn.Module,
    spec: ModelSpec,
    projection: SSLProjectionConfig,
    device: torch.device,
    sample_rate: int,
    azimuth_domain: tuple[float, float],
    azimuth_step_deg: float,
    vad_threshold: float,
    accuracy_threshold_deg: float,
    range_grid: Sequence[float] | None = None,
    offline_inference: bool = False,
) -> dict[str, Any]:
    """在记录上执行模型前向、DP-IPD 映射和显式指标计算。

    ``offline_inference`` 仅适用于非因果 ``IPDNetFixed``，默认关闭以保留整段
    前向路径。分块长度 ``n_seg`` 与单次分块数上限 ``offline_chunk_batch_size``
    读取模型配置并写入结果元数据；本入口在 inference_mode 下逐小批执行。
    """

    if type(offline_inference) is not bool:
        raise TypeError("offline_inference must be a bool")
    model_stride = _validate_model_time_stride(model, spec, projection)
    n_seg: int | None = None
    offline_chunk_batch_size: int | None = None
    if offline_inference:
        if spec.model_type != "IPDNetFixed" or spec.causal:
            raise ValueError(
                "offline_inference is supported only for IPDNetFixed with causal=false"
            )
        if projection.normalization != "global_mean":
            raise ValueError(
                "offline IPDNetFixed evaluation requires ssl.normalization='global_mean' "
                "computed over the complete observation"
            )
        if "offline_inference" not in inspect.signature(model.forward).parameters:
            raise TypeError("selected IPDNetFixed implementation does not expose offline_inference")
        configured_n_seg = getattr(model, "n_seg", None)
        if type(configured_n_seg) is not int or configured_n_seg < 1:
            raise ValueError("offline IPDNetFixed must expose a positive integer n_seg")
        if configured_n_seg % model_stride:
            raise ValueError(
                "IPDNetFixed n_seg must be a multiple of its output stride "
                f"({model_stride}), got {configured_n_seg}"
            )
        n_seg = configured_n_seg
        configured_chunk_batch_size = getattr(model, "offline_chunk_batch_size", None)
        if type(configured_chunk_batch_size) is not int or configured_chunk_batch_size < 1:
            raise ValueError(
                "offline IPDNetFixed must expose a positive integer offline_chunk_batch_size"
            )
        offline_chunk_batch_size = configured_chunk_batch_size
    elif spec.model_type == "IPDNetFixed":
        configured_n_seg = getattr(model, "n_seg", None)
        if type(configured_n_seg) is int and configured_n_seg > 0:
            n_seg = configured_n_seg

    if azimuth_step_deg <= 0.0 or azimuth_step_deg > 180.0:
        raise ValueError("azimuth_step_deg must be in (0,180]")
    lower, upper = azimuth_domain
    count = int(round(math.degrees(upper - lower) / azimuth_step_deg))
    azimuth_grid = np.linspace(lower, upper, count + 1, dtype=np.float64)
    per_item: list[dict[str, Any]] = []
    aggregate_prediction: list[np.ndarray] = []
    aggregate_target: list[np.ndarray] = []
    aggregate_vad: list[np.ndarray] = []
    aggregate_valid: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for record in records:
            record = record.validate()
            if projection.reference_channel >= record.mic_positions.shape[0]:
                raise ValueError(f"{record.key}: reference_channel is outside record geometry")
            observation = _load_observation(
                record,
                target_rate=sample_rate,
                mic_count=record.mic_positions.shape[0],
            )
            spectra = _stft(observation, projection)
            if projection.input_representation == "multichannel_stft_real_imag":
                features = _raw_stft_real_imag(spectra)
            elif projection.input_representation == "reference_ipd":
                features = _reference_ipd_features(spectra, projection.reference_channel)
            else:  # pragma: no cover - SSLProjectionConfig validates this
                raise ValueError(f"unsupported SSL input representation: {projection.input_representation!r}")
            features = features.unsqueeze(0).to(device=device, dtype=torch.float32)
            pair_indices = [
                [projection.reference_channel, index]
                for index in range(record.mic_positions.shape[0])
                if index != projection.reference_channel
            ]
            pair_index = torch.tensor(pair_indices, dtype=torch.int64, device=device).unsqueeze(0)
            kwargs: dict[str, Any]
            if spec.feature_contract == "multichannel_stft_real_imag":
                kwargs = {"multichannel_stft_real_imag": features}
            elif spec.feature_contract == "reference_ipd":
                kwargs = {"features": features}
            else:  # pragma: no cover - ModelSpec validates this
                raise ValueError(f"unsupported SSL ModelSpec feature contract: {spec.feature_contract!r}")
            if offline_inference:
                if int(features.shape[1]) < projection.output_stride:
                    raise ValueError(
                        f"{record.key}: offline IPDNetFixed requires at least one complete "
                        f"output stride ({projection.output_stride}) of STFT frames"
                    )
                kwargs["offline_inference"] = True
            if "pair_index" in inspect.signature(model.forward).parameters:
                kwargs["pair_index"] = pair_index
            max_tracks = getattr(model, "max_track", None)
            if type(max_tracks) is not int or max_tracks < 1:
                raise ValueError("SSL model must expose a positive integer max_track")
            input_frames = int(features.shape[1])
            expected_output_frames = input_frames // projection.output_stride
            raw_output = model(**kwargs)
            output_fields = validate_ssl_model_output(
                model,
                raw_output,
                expected_shape=(
                    1,
                    expected_output_frames,
                    int(features.shape[2]),
                    max_tracks,
                    int(pair_index.shape[1]),
                    2,
                ),
                expected_device=features.device,
            )
            output = output_fields["dpipd"]
            output_frames = int(output.shape[1])
            if output_frames != expected_output_frames:
                raise ValueError(
                    f"{record.key}: model output frames {output_frames} != input_frames // "
                    f"projection.output_stride ({input_frames} // {projection.output_stride} "
                    f"= {expected_output_frames})"
                )
            aligned_target, aligned_vad, aligned_inside = _align_labels(
                record,
                frame_count=input_frames,
                projection=projection,
                output_count=output_frames,
                sample_rate=sample_rate,
            )
            if aligned_target.size != output_frames:
                raise ValueError(
                    f"{record.key}: label/output alignment produced "
                    f"{aligned_target.size} frames for model {output_frames}"
                )
            mapping_config = DOAMappingConfig(
                n_fft=projection.n_fft,
                sample_rate=sample_rate,
                azimuth_grid=azimuth_grid,
                elevation_grid=(0.0,),
                range_grid=range_grid if spec.field_mode == "near_field" else None,
                field_mode=spec.field_mode,
                sound_speed=projection.sound_speed,
                exclude_dc=projection.exclude_dc,
                phase_sign=projection.phase_sign,
                elevation_convention="horizontal",
                angle_order="azimuth_elevation",
                coordinate_frame=record.coordinate_frame,
                vad_threshold=vad_threshold,
            )
            mapper = DPIPDToDOAMapper(mapping_config)
            mapping = mapper.map(
                output,
                geometry=torch.as_tensor(record.mic_positions, dtype=torch.float32, device=device).unsqueeze(0),
                pair_index=pair_index,
                pair_mask=torch.ones((1, pair_index.shape[1]), dtype=torch.bool, device=device),
                vad=torch.ones((1, output_frames, output.shape[3]), dtype=torch.float32, device=device),
                output_lengths=torch.tensor([output_frames], dtype=torch.int64, device=device),
                coordinate_frame=record.coordinate_frame,
                time_stride=projection.output_stride,
            )
            track_scores = mapping.scores.max(dim=-1).values
            selected_track = track_scores.argmax(dim=-1)
            selected_track = selected_track.unsqueeze(-1).unsqueeze(-1).expand(1, output_frames, 1, 2)
            prediction = mapping.doa.gather(2, selected_track).squeeze(2).detach().cpu().numpy()[0]
            prediction_valid = mapping.valid_mask.gather(
                2, selected_track[..., :1].squeeze(-1)
            ).squeeze(2).detach().cpu().numpy()[0]
            # ``angle_order=azimuth_elevation`` makes the first component azimuth.
            prediction = prediction[:, 0]
            prediction_valid = prediction_valid.astype(bool) & aligned_inside
            aligned_vad = np.where(aligned_inside, aligned_vad, 0.0)
            item_metrics = compute_azimuth_metrics(
                prediction,
                aligned_target,
                target_vad=aligned_vad,
                prediction_valid=prediction_valid,
                domain=azimuth_domain,
                vad_threshold=vad_threshold,
                accuracy_threshold_deg=accuracy_threshold_deg,
            )
            per_item.append(
                {
                    "key": record.key,
                    "audio": str(record.audio),
                    "input_frames": int(features.shape[1]),
                    "output_frames": output_frames,
                    "alignment": {
                        "label_time_axis": "record_label_times_s_interpolated_to_stft_centers",
                        "output_group": "left_aligned_stride_groups",
                        "time_stride": projection.output_stride,
                    },
                    "metrics": item_metrics,
                }
            )
            aggregate_prediction.append(prediction)
            aggregate_target.append(aligned_target)
            aggregate_vad.append(aligned_vad)
            aggregate_valid.append(prediction_valid)
    if not per_item:
        raise ValueError("real SSL evaluation received no records")
    aggregate = compute_azimuth_metrics(
        np.concatenate(aggregate_prediction),
        np.concatenate(aggregate_target),
        target_vad=np.concatenate(aggregate_vad),
        prediction_valid=np.concatenate(aggregate_valid),
        domain=azimuth_domain,
        vad_threshold=vad_threshold,
        accuracy_threshold_deg=accuracy_threshold_deg,
    )
    return {
        "schema": "ssl-real-eval-v1",
        "evidence": "real_recording_observation_with_external_azimuth_labels",
        "model_spec": spec.as_dict(),
        "inference": {
            "offline_inference": offline_inference,
            "n_seg": n_seg,
            "offline_chunk_batch_size": offline_chunk_batch_size,
        },
        "projection": {
            "sample_rate": sample_rate,
            "n_fft": projection.n_fft,
            "hop_length": projection.hop_length,
            "output_stride": projection.output_stride,
            "input_representation": projection.input_representation,
            "normalization": projection.normalization,
            "coordinate_frame": "record_declared_array_local_xyz",
        },
        "contract": {
            "single_source": True,
            "source_distances_used_for_inference": False,
            "azimuth_domain": "explicit_halfplane_or_cli_domain",
            "gt_vad_and_labels_used_only_for_metrics": True,
        },
        "items": per_item,
        "aggregate": aggregate,
    }


def _build_model(
    config_path: Path,
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, ModelSpec]:
    config = load_merged_config(config_path)
    if config.get("meta", {}).get("task") not in {None, "ssl"}:
        raise ValueError("evaluation config must declare meta.task=ssl")
    num_tracks = resolve_ssl_num_tracks(config)
    model_cfg = dict(config["model"])
    model_args = dict(model_cfg.get("args", {}))
    model_args["max_track"] = num_tracks
    model_cfg["args"] = model_args
    model = build_model(model_cfg, stft_cfg=config.get("stft"))
    spec = resolve_ssl_model_spec(model, config.get("model_spec"))
    validate_ssl_model_io(model, spec)
    load_checkpoint(
        model,
        checkpoint,
        family=CheckpointFamily.SINGLE_MODEL,
        map_location="cpu",
        purpose="SSL real-recording model-only evaluation",
    )
    model.to(device=device)
    return config, model, spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="带 geometry/labels 的 canonical JSONL")
    source.add_argument("--locata-root", type=Path, help="upstream LOCATA 根目录")
    parser.add_argument("--task", type=int, default=1, help="LOCATA task 编号")
    parser.add_argument("--array", default="dicit", help="LOCATA array 名称")
    parser.add_argument("--mic-indices", default="0,1", help="选择麦克风索引，例如 0,1")
    parser.add_argument("--config", type=Path, required=True, help="SSL 模型 YAML 配置")
    parser.add_argument("--checkpoint", type=Path, required=True, help="严格模型权重 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="JSON 结果路径；建议放在 /tmp 或 ignored data/ssl")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--azimuth-domain", default="halfplane", help="halfplane 或 min_deg,max_deg")
    parser.add_argument("--azimuth-step-deg", type=float, default=1.0)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--accuracy-threshold-deg", type=float, default=10.0)
    parser.add_argument("--range-grid", help="near-field range grid in meters, comma separated")
    parser.add_argument(
        "--offline-inference",
        action="store_true",
        help="仅对 causal=false 的 IPDNetFixed 启用官方 n_seg 分块推理（默认关闭）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_rate < 1000:
        parser.error("--sample-rate must be >= 1000")
    try:
        mic_indices = tuple(int(part.strip()) for part in args.mic_indices.split(",") if part.strip())
        if len(mic_indices) < 2:
            raise ValueError("--mic-indices must select at least two microphones")
        domain = parse_azimuth_domain(args.azimuth_domain)
        config, model, spec = _build_model(
            args.config.expanduser().resolve(),
            args.checkpoint.expanduser().resolve(),
            device=torch.device(args.device),
        )
        configured_rate = config.get("data", {}).get("sample_rate")
        if configured_rate is not None and int(configured_rate) != args.sample_rate:
            raise ValueError(
                f"--sample-rate={args.sample_rate} conflicts with config data.sample_rate={configured_rate}"
            )
        projection = _projection_from_config(
            config, spec, sample_rate=args.sample_rate, model=model
        )
        _validate_model_time_stride(model, spec, projection)
        if projection.reference_channel >= len(mic_indices):
            raise ValueError("projection reference_channel is outside selected microphones")
        records = (
            iter_manifest(args.manifest, max_items=args.max_items)
            if args.manifest is not None
            else iter_locata(
                args.locata_root,
                task=args.task,
                array=args.array,
                mic_indices=mic_indices,
                max_items=args.max_items,
            )
        )
        result = evaluate_records(
            records,
            model=model,
            spec=spec,
            projection=projection,
            device=torch.device(args.device),
            sample_rate=args.sample_rate,
            azimuth_domain=domain,
            azimuth_step_deg=args.azimuth_step_deg,
            vad_threshold=args.vad_threshold,
            accuracy_threshold_deg=args.accuracy_threshold_deg,
            range_grid=None if not args.range_grid else tuple(float(v) for v in args.range_grid.split(",")),
            offline_inference=args.offline_inference,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
