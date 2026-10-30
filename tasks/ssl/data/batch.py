"""SSL canonical batch 合同与帧轴补零校验。"""

from dataclasses import dataclass, replace
import re
from typing import Any

import torch


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HASH_FIELDS = ("rir_sha256", "direct_path_content_sha256", "manifest_sha256")

SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG = "multichannel_stft_real_imag"
SSL_INPUT_REFERENCE_IPD = "reference_ipd"
SSL_TARGET_DPIPD_FAR_FIELD = "dpipd_far_field"
SSL_TARGET_DPIPD_NEAR_FIELD = "dpipd_near_field"

# 旧的 raw_stft_ri 字符串不再作为合同别名接受。
SSL_TARGET_DIRECT_PATH_IPD_FAR_FIELD = SSL_TARGET_DPIPD_FAR_FIELD
SSL_TARGET_DIRECT_PATH_IPD_NEAR_FIELD = SSL_TARGET_DPIPD_NEAR_FIELD

_INPUT_REPRESENTATIONS = frozenset({SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG, SSL_INPUT_REFERENCE_IPD})
_TARGET_REPRESENTATIONS = frozenset(
    {
        SSL_TARGET_DPIPD_FAR_FIELD,
        SSL_TARGET_DPIPD_NEAR_FIELD,
    }
)
_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_frame_index(
    value: torch.Tensor,
    *,
    name: str,
    lengths: torch.Tensor,
    batch: int,
    frames: int,
) -> None:
    """校验帧索引的 dtype、有效区间单调性和补齐语义。"""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"SSL {name} must be a torch.Tensor")
    if value.shape != (batch, frames):
        frame_axis = "T_in" if name == "input_frame_index" else "T_out"
        raise ValueError(f"SSL {name} must have shape [B,{frame_axis}]")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"SSL {name} must use an integer dtype")
    if torch.any(value < 0):
        raise ValueError(f"SSL {name} must be non-negative")

    frame_ids = torch.arange(frames, device=value.device)[None, :]
    item_lengths = lengths.to(device=value.device, dtype=torch.long)
    valid = frame_ids < item_lengths[:, None]
    if frames > 1:
        adjacent_valid = valid[:, 1:]
        if bool(((value[:, 1:] <= value[:, :-1]) & adjacent_valid).any().item()):
            raise ValueError(f"SSL {name} must be strictly increasing on valid frames")
    if bool((value.masked_select(~valid) != 0).any().item()):
        raise ValueError(f"SSL {name} must be zero beyond each item's declared length")


def _provenance_integer(value: Any, *, name: str) -> int:
    """读取 provenance 中不允许隐式截断的整数标量。"""
    try:
        scalar = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SSL {name} must be an exact integer scalar") from exc
    if scalar.numel() != 1 or scalar.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"SSL {name} must be an exact integer scalar")
    return int(scalar.item())


@dataclass
class SSLBatch:
    features: torch.Tensor
    target_dpipd: torch.Tensor
    input_lengths: torch.Tensor
    output_lengths: torch.Tensor
    pair_index: torch.Tensor
    pair_mask: torch.Tensor
    geometry: torch.Tensor
    frame_origin: torch.Tensor
    time_stride: int
    # 输出时间轴单帧聚合的输入帧跨度；average_pool 语义下等于 time_stride。
    receptive_field: int = 1
    # 相对输出帧索引（池化组首帧）需要前瞻的输入帧数；当前左对齐平均
    # 池化登记 time_stride - 1，严格无前瞻时为 0。
    algorithmic_latency: int = 0
    # [B,T_out] bool，标识有效输出帧；output_lengths 之外的补齐帧必须为
    # False，有效区内必须为 True。
    valid_output_mask: torch.Tensor | None = None
    doa: torch.Tensor | None = None
    vad: torch.Tensor | None = None
    sample_rates: torch.Tensor | None = None
    provenance: tuple[dict[str, Any], ...] = ()
    source_distances: torch.Tensor | None = None
    input_representation: str = SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG
    target_representation: str = SSL_TARGET_DPIPD_FAR_FIELD
    coordinate_frame: str = "room_xyz_m"
    input_frame_index: torch.Tensor | None = None
    output_frame_index: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        """返回 features 的 canonical 主样本数。"""
        if not isinstance(self.features, torch.Tensor) or self.features.ndim < 1:
            raise ValueError("SSLBatch.features must expose a batch dimension")
        return int(self.features.shape[0])

    @property
    def targets(self) -> torch.Tensor:
        """兼容旧 trainer 读取；新代码必须使用 ``target_dpipd``。"""
        return self.target_dpipd

    @property
    def lengths(self) -> torch.Tensor:
        """兼容旧 trainer 读取；新代码必须区分输入和输出长度。"""
        return self.output_lengths

    @property
    def output_stride(self) -> int:
        """旧字段别名；时间基准统一由 ``time_stride`` 表达。"""
        return self.time_stride

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "SSLBatch":
        """把帧域 tensor 搬到目标设备，保留 provenance 映射。"""
        return replace(
            self,
            features=self.features.to(device=device, non_blocking=non_blocking),
            target_dpipd=self.target_dpipd.to(device=device, non_blocking=non_blocking),
            input_lengths=self.input_lengths.to(device=device, non_blocking=non_blocking),
            output_lengths=self.output_lengths.to(device=device, non_blocking=non_blocking),
            pair_index=self.pair_index.to(device=device, non_blocking=non_blocking),
            pair_mask=self.pair_mask.to(device=device, non_blocking=non_blocking),
            geometry=self.geometry.to(device=device, non_blocking=non_blocking),
            frame_origin=self.frame_origin.to(device=device, non_blocking=non_blocking),
            valid_output_mask=(
                None if self.valid_output_mask is None
                else self.valid_output_mask.to(device=device, non_blocking=non_blocking)
            ),
            doa=None if self.doa is None else self.doa.to(device=device, non_blocking=non_blocking),
            vad=None if self.vad is None else self.vad.to(device=device, non_blocking=non_blocking),
            sample_rates=(
                None if self.sample_rates is None
                else self.sample_rates.to(device=device, non_blocking=non_blocking)
            ),
            source_distances=(
                None
                if self.source_distances is None
                else self.source_distances.to(device=device, non_blocking=non_blocking)
            ),
            input_frame_index=(
                None if self.input_frame_index is None
                else self.input_frame_index.to(device=device, non_blocking=non_blocking)
            ),
            output_frame_index=(
                None if self.output_frame_index is None
                else self.output_frame_index.to(device=device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> "SSLBatch":
        """参与 DataLoader 的递归锁页内存。"""
        return replace(
            self,
            features=self.features.pin_memory(),
            target_dpipd=self.target_dpipd.pin_memory(),
            input_lengths=self.input_lengths.pin_memory(),
            output_lengths=self.output_lengths.pin_memory(),
            pair_index=self.pair_index.pin_memory(),
            pair_mask=self.pair_mask.pin_memory(),
            geometry=self.geometry.pin_memory(),
            frame_origin=self.frame_origin.pin_memory(),
            valid_output_mask=(
                None if self.valid_output_mask is None else self.valid_output_mask.pin_memory()
            ),
            doa=None if self.doa is None else self.doa.pin_memory(),
            vad=None if self.vad is None else self.vad.pin_memory(),
            sample_rates=None if self.sample_rates is None else self.sample_rates.pin_memory(),
            source_distances=(
                None
                if self.source_distances is None
                else self.source_distances.pin_memory()
            ),
            input_frame_index=(
                None if self.input_frame_index is None else self.input_frame_index.pin_memory()
            ),
            output_frame_index=(
                None if self.output_frame_index is None else self.output_frame_index.pin_memory()
            ),
        )

    def validate(self) -> None:
        """校验逐帧形状、元数据，以及 declared length 外的零填充。"""
        if self.input_representation not in _INPUT_REPRESENTATIONS:
            raise ValueError("SSL input_representation must be multichannel_stft_real_imag or reference_ipd")
        if self.target_representation not in _TARGET_REPRESENTATIONS:
            raise ValueError(
                "SSL target_representation must be one of "
                f"{sorted(_TARGET_REPRESENTATIONS)!r}"
            )
        if self.features.ndim != 4 or self.target_dpipd.ndim != 6:
            raise ValueError(
                "SSL batch expects features [B,T,F,D] and targets [B,T,F,K,P,2]"
            )
        if self.features.shape[0] != self.target_dpipd.shape[0] or self.features.shape[2] != self.target_dpipd.shape[2]:
            raise ValueError("SSL features and targets must share B and F")
        if self.target_dpipd.shape[-1] != 2:
            raise ValueError("SSL targets must use real/imag layout")
        if self.features.shape[-1] < 2 or self.features.shape[-1] % 2:
            raise ValueError("SSL features must contain real/imag pairs")
        batch = self.target_dpipd.shape[0]
        input_frames = self.features.shape[1]
        output_frames = self.target_dpipd.shape[1]
        tracks, pairs = self.target_dpipd.shape[3:5]
        if tracks < 1 or pairs < 1:
            raise ValueError("SSL target_dpipd must contain at least one track and pair")
        channels = self.features.shape[-1] // 2
        if self.input_representation == SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG:
            if channels < 2:
                raise ValueError("multichannel_stft_real_imag features require C>=2")
        elif channels < 1:
            raise ValueError("reference_ipd features require at least one pair")
        if not isinstance(self.pair_index, torch.Tensor) or self.pair_index.ndim != 3:
            raise ValueError("SSL pair_index must have shape [B,P,2]")
        if pairs != self.pair_index.shape[1]:
            raise ValueError("SSL pair_index and target pair dimensions must agree")
        if type(self.time_stride) is not int or self.time_stride < 1:
            raise ValueError("SSL time_stride must be an exact positive integer")
        if type(self.receptive_field) is not int or self.receptive_field < 1:
            raise ValueError("SSL receptive_field must be an exact positive integer")
        if type(self.algorithmic_latency) is not int or self.algorithmic_latency < 0:
            raise ValueError("SSL algorithmic_latency must be an exact non-negative integer")
        if self.algorithmic_latency > self.receptive_field - 1:
            raise ValueError("SSL algorithmic_latency must not exceed receptive_field - 1")
        if self.pair_index.shape != (batch, pairs, 2) or self.pair_index.dtype not in _INTEGER_DTYPES:
            raise ValueError("SSL pair_index must have shape [B,P,2] and integer dtype")
        if torch.any(self.pair_index < 0) or torch.any(self.pair_index >= channels if self.input_representation == SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG else self.pair_index >= channels + 1):
            raise ValueError("SSL pair_index contains an out-of-range microphone")
        geometry_channels = channels if self.input_representation == SSL_INPUT_MULTICHANNEL_STFT_REAL_IMAG else channels + 1
        # 当前 canonical projection 是 reference-to-all。pair 的方向有符号，
        # 因此仅检查 P 数量不足以防止把不同 reference 或重复 pair 混入 batch。
        if pairs != geometry_channels - 1:
            raise ValueError("SSL reference_to_all target must contain C-1 pairs")
        references = self.pair_index[:, :, 0]
        if not torch.equal(references, references[:, :1].expand_as(references)):
            raise ValueError("SSL reference_to_all pair_index must use one reference microphone per item")
        others = self.pair_index[:, :, 1]
        if torch.any(others == references):
            raise ValueError("SSL pair_index cannot contain self pairs")
        if torch.unique(others, dim=1).shape[1] != pairs:
            raise ValueError("SSL reference_to_all pair_index must not repeat target microphones")
        expected_others = torch.arange(geometry_channels, device=others.device).view(1, -1)
        expected_others = expected_others.expand(batch, -1)
        expected_others = expected_others.masked_fill(
            expected_others == references[:, :1], -1
        )
        expected_others = expected_others[expected_others >= 0].reshape(batch, pairs)
        if not torch.equal(torch.sort(others, dim=1).values, torch.sort(expected_others, dim=1).values):
            raise ValueError("SSL reference_to_all pair_index must enumerate every non-reference microphone")
        if not isinstance(self.pair_mask, torch.Tensor):
            raise ValueError("SSL pair_mask is required for the canonical batch contract")
        if self.pair_mask.shape != (batch, pairs) or self.pair_mask.dtype is not torch.bool:
            raise ValueError("SSL pair_mask must have shape [B,P] and bool dtype")
        if not self.pair_mask.any(dim=1).all():
            raise ValueError("SSL pair_mask must keep at least one pair per item")
        if not isinstance(self.geometry, torch.Tensor):
            raise ValueError("SSL geometry is required for the canonical batch contract")
        if self.geometry.shape != (batch, geometry_channels, 3):
            raise ValueError("SSL geometry must have shape [B,C,3]")
        if not torch.isfinite(self.geometry).all():
            raise ValueError("SSL geometry must be finite")
        if self.source_distances is not None:
            if self.target_representation != SSL_TARGET_DPIPD_NEAR_FIELD:
                raise ValueError(
                    "SSL source_distances require a near-field target or range-aware mapping"
                )
            if not isinstance(self.source_distances, torch.Tensor):
                raise TypeError("SSL source_distances must be a torch.Tensor")
            if self.source_distances.shape != (batch, output_frames, tracks):
                raise ValueError("SSL source_distances must have shape [B,T,K]")
            if not torch.isfinite(self.source_distances).all():
                raise ValueError("SSL source_distances must be finite")
        if (
            self.target_representation == SSL_TARGET_DPIPD_NEAR_FIELD
            and self.source_distances is None
        ):
            raise ValueError(
                "near-field direct-path IPD targets require source_distances [B,T,K]"
            )
        for name, values, maximum in (
            ("input_lengths", self.input_lengths, input_frames),
            ("output_lengths", self.output_lengths, output_frames),
        ):
            if values.ndim != 1 or values.shape[0] != batch:
                raise ValueError(f"SSL {name} must have shape [B]")
            if values.dtype not in _INTEGER_DTYPES:
                raise TypeError(f"SSL {name} must use an integer dtype")
            if torch.any(values < 1) or torch.any(values > maximum):
                raise ValueError(f"SSL {name} must be within the padded frame axis")
        if self.source_distances is not None:
            distance_frame_ids = torch.arange(
                output_frames, device=self.source_distances.device
            )[None, :]
            distance_lengths = self.output_lengths.to(
                device=self.source_distances.device, dtype=torch.long
            )
            valid_distance_frames = distance_frame_ids < distance_lengths[:, None]
            if bool((self.source_distances[valid_distance_frames] <= 0).any().item()):
                raise ValueError("SSL source_distances must be strictly positive")
        if not isinstance(self.valid_output_mask, torch.Tensor):
            raise ValueError("SSL canonical batch requires valid_output_mask")
        if self.valid_output_mask.shape != (batch, output_frames):
            raise ValueError("SSL valid_output_mask must have shape [B,T_out]")
        if self.valid_output_mask.dtype is not torch.bool:
            raise TypeError("SSL valid_output_mask must use a bool dtype")
        # 有效输出帧由 output_lengths 唯一决定：有效区内必须全 True，
        # 补齐帧必须全 False，禁止用 mask 夹带其他门控语义。
        expected_mask = torch.arange(
            output_frames, device=self.valid_output_mask.device
        )[None, :] < self.output_lengths.to(
            device=self.valid_output_mask.device, dtype=torch.long
        )[:, None]
        if not torch.equal(self.valid_output_mask, expected_mask):
            raise ValueError("SSL valid_output_mask must match output_lengths padding")
        if (
            not isinstance(self.frame_origin, torch.Tensor)
            or self.frame_origin.ndim != 1
            or self.frame_origin.shape[0] != batch
        ):
            raise ValueError("SSL frame_origin must have shape [B]")
        if self.frame_origin.dtype not in _INTEGER_DTYPES:
            raise TypeError("SSL frame_origin must use an integer dtype")
        if torch.any(self.frame_origin < 0):
            raise ValueError("SSL frame_origin must be non-negative")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame:
            raise ValueError("SSL coordinate_frame must be a non-empty string")
        if not isinstance(self.doa, torch.Tensor):
            raise ValueError("SSL canonical batch requires doa")
        if self.doa.shape != (batch, output_frames, tracks, 2) or not torch.isfinite(self.doa).all():
            raise ValueError("SSL doa must have shape [B,T,K,2] and be finite")
        if not isinstance(self.vad, torch.Tensor):
            raise ValueError("SSL canonical batch requires vad")
        if self.vad.shape != (batch, output_frames, tracks) or not torch.isfinite(self.vad).all():
            raise ValueError("SSL vad must have shape [B,T,K] and be finite")
        if torch.any((self.vad < 0.0) | (self.vad > 1.0)):
            raise ValueError("SSL vad must be in [0, 1]")
        if not isinstance(self.sample_rates, torch.Tensor):
            raise ValueError("SSL canonical batch requires sample_rates")
        if self.sample_rates.ndim != 1 or self.sample_rates.shape[0] != batch:
            raise ValueError("SSL sample_rates must have shape [B]")
        if self.sample_rates.dtype not in _INTEGER_DTYPES:
            raise TypeError("SSL sample_rates must use an integer dtype")
        if torch.any(self.sample_rates < 1000):
            raise ValueError("SSL sample_rates must be >= 1000")
        if not torch.equal(
            self.sample_rates, self.sample_rates[:1].expand_as(self.sample_rates)
        ):
            raise ValueError("SSL batch must not mix sample rates")
        if len(self.provenance) != batch:
            raise ValueError("SSL provenance must contain one mapping per batch item")
        for index, item in enumerate(self.provenance):
            if not isinstance(item, dict) or not isinstance(item.get("backend"), str) or not item["backend"]:
                raise ValueError(f"SSL provenance[{index}] must identify a backend")
            for field in _HASH_FIELDS:
                if field in item and (
                    not isinstance(item[field], str)
                    or _SHA256_RE.fullmatch(item[field]) is None
                ):
                    raise ValueError(
                        f"SSL provenance[{index}].{field} must be a strict lowercase 64-hex SHA-256 digest"
                    )
            if "coordinate_frame" in item:
                declared_coordinate_frame = item["coordinate_frame"]
                if not isinstance(declared_coordinate_frame, str) or not declared_coordinate_frame:
                    raise ValueError(
                        f"SSL provenance[{index}].coordinate_frame must be a non-empty string"
                    )
                if declared_coordinate_frame != self.coordinate_frame:
                    raise ValueError(
                        f"SSL provenance[{index}].coordinate_frame does not match batch coordinate_frame"
                    )
            if "sample_rate" in item:
                declared_sample_rate = _provenance_integer(
                    item["sample_rate"], name=f"provenance[{index}].sample_rate"
                )
                expected_sample_rate = int(self.sample_rates[index].item())
                if declared_sample_rate != expected_sample_rate:
                    raise ValueError(
                        f"SSL provenance[{index}].sample_rate does not match batch sample_rate"
                    )
            if "geometry" in item:
                declared_geometry = item["geometry"]
                if not isinstance(declared_geometry, dict):
                    raise ValueError(
                        f"SSL provenance[{index}].geometry must be a mapping"
                    )
                if "positions_m" not in declared_geometry:
                    raise ValueError(
                        f"SSL provenance[{index}].geometry requires positions_m"
                    )
                try:
                    declared_positions = torch.as_tensor(
                        declared_geometry["positions_m"], dtype=torch.float32
                    ).to(device=self.geometry.device)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"SSL provenance[{index}].geometry.positions_m must be numeric"
                    ) from exc
                expected_positions = self.geometry[index].to(
                    device=declared_positions.device, dtype=torch.float32
                )
                if (
                    declared_positions.shape != expected_positions.shape
                    or not torch.isfinite(declared_positions).all()
                    or not torch.allclose(
                        declared_positions, expected_positions, rtol=0.0, atol=1e-6
                    )
                ):
                    raise ValueError(
                        f"SSL provenance[{index}].geometry.positions_m does not match batch geometry"
                    )
                if "reference_channel" in declared_geometry:
                    declared_reference = _provenance_integer(
                        declared_geometry["reference_channel"],
                        name=f"provenance[{index}].geometry.reference_channel",
                    )
                    expected_reference = int(self.pair_index[index, 0, 0].item())
                    if declared_reference != expected_reference:
                        raise ValueError(
                            f"SSL provenance[{index}].geometry.reference_channel does not match pair_index"
                        )
            if (
                self.target_representation == SSL_TARGET_DPIPD_NEAR_FIELD
                and "source_distances" in item
            ):
                if self.source_distances is None:
                    raise ValueError(
                        f"SSL provenance[{index}].source_distances has no batch source_distances"
                    )
                # provenance 属于未补齐的 item；只与该 item 的有效输出帧比较，
                # batch 的尾部补零由下面的通用 padding 校验负责。
                output_length = int(self.output_lengths[index].item())
                expected_distances = self.source_distances[index, :output_length]
                try:
                    declared_distances = torch.as_tensor(
                        item["source_distances"], dtype=torch.float32
                    ).to(device=expected_distances.device)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"SSL provenance[{index}].source_distances must be numeric"
                    ) from exc
                expected_distances = expected_distances.to(
                    dtype=torch.float32, device=declared_distances.device
                )
                if (
                    declared_distances.shape != expected_distances.shape
                    or not torch.isfinite(declared_distances).all()
                    or not torch.allclose(
                        declared_distances, expected_distances, rtol=0.0, atol=1e-6
                    )
                ):
                    raise ValueError(
                        f"SSL provenance[{index}].source_distances does not match batch source_distances"
                    )
        if not torch.isfinite(self.features).all() or not torch.isfinite(self.targets).all():
            raise ValueError("SSL features and targets must be finite")
        if self.input_frame_index is not None:
            _validate_frame_index(
                self.input_frame_index,
                name="input_frame_index",
                lengths=self.input_lengths,
                batch=batch,
                frames=input_frames,
            )
        if self.output_frame_index is not None:
            _validate_frame_index(
                self.output_frame_index,
                name="output_frame_index",
                lengths=self.output_lengths,
                batch=batch,
                frames=output_frames,
            )
        if bool((self.input_lengths < input_frames).any().item()):
            frame_ids = torch.arange(input_frames, device=self.features.device)[None, :]
            padding = frame_ids >= self.input_lengths.to(
                device=self.features.device, dtype=torch.long
            )[:, None]
            padded_fields = [("features", self.features)]
            for name, value in padded_fields:
                nonzero_frames = value.reshape(batch, input_frames, -1).ne(0).any(dim=-1)
                if bool((nonzero_frames & padding.to(value.device)).any().item()):
                    raise ValueError(
                        f"SSL {name} must be zero beyond each item's input_lengths"
                    )
        if bool((self.output_lengths < output_frames).any().item()):
            frame_ids = torch.arange(output_frames, device=self.target_dpipd.device)[None, :]
            padding = frame_ids >= self.output_lengths.to(
                device=self.target_dpipd.device, dtype=torch.long
            )[:, None]
            padded_fields = [("target_dpipd", self.target_dpipd)]
            if self.doa is not None:
                padded_fields.append(("doa", self.doa))
            if self.vad is not None:
                padded_fields.append(("vad", self.vad))
            if self.source_distances is not None:
                padded_fields.append(("source_distances", self.source_distances))
            for name, value in padded_fields:
                nonzero_frames = value.reshape(batch, output_frames, -1).ne(0).any(dim=-1)
                if bool((nonzero_frames & padding.to(value.device)).any().item()):
                    raise ValueError(
                        f"SSL {name} must be zero beyond each item's output_lengths"
                    )
