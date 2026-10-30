"""AEC input/deployment frontend。

该模块统一封装可选 GCC-PHAT 时延对齐与可选 LinearAEC，并保持模型能力
校验留在调用侧。whole-config 仅接受顶层 ``aec_frontend`` 配置；直接调用代码
可构造 :class:`AECFrontendConfig`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tasks.aec.inference.streaming_delay_estimator import (
    AECStreamAlignConfig,
    create_delay_estimator,
)

if TYPE_CHECKING:
    from tasks.aec.data.linear_aec_frontend import LinearAECConfig

LinearAECFrontend = None

_INPUT_ROUTINGS = frozenset(
    {"mic_farend", "mic_farend_error", "error_farend"}
)
_ERROR_ROUTINGS = frozenset({"mic_farend_error", "error_farend"})


def _linear_aec_classes():
    from tasks.aec.data.linear_aec_frontend import LinearAECConfig, LinearAECFrontend as frontend

    return LinearAECConfig, frontend


def _strict_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _parse_delay_alignment(value: object) -> AECStreamAlignConfig:
    if isinstance(value, AECStreamAlignConfig):
        config = value
    else:
        config = AECStreamAlignConfig.from_dict(
            _strict_mapping(value, "delay_alignment")
        )
    if config.algorithm != "gcc_phat":
        raise ValueError(
            "delay_alignment.algorithm is fixed to 'gcc_phat' in the canonical "
            f"AEC frontend, got {config.algorithm!r}"
        )
    return config


def _parse_linear_aec(value: object) -> LinearAECConfig | None:
    if value is None:
        return None
    config_cls, _ = _linear_aec_classes()
    if isinstance(value, config_cls):
        config = value
    else:
        raw = _strict_mapping(value, "linear_aec")
        for name in ("gpu_batch", "use_res"):
            if name in raw and not isinstance(raw[name], bool):
                raise ValueError(
                    f"linear_aec.{name} must be a bool, got {raw[name]!r}"
                )
        config = config_cls.from_dict(raw)
    # backend:none 等价于禁用 frontend，避免创建无意义对象。
    if config.backend is None:
        return None
    return config


@dataclass
class AECFrontendConfig:
    """AEC input/deployment frontend 的 canonical typed config。

    ``delay_alignment.enabled`` 是唯一对齐开关；算法固定为现有 GCC-PHAT。
    ``input_routing`` 只验证基础信号依赖，具体模型 capability 由 Trainer 接线时
    校验。
    """

    delay_alignment: AECStreamAlignConfig = field(
        default_factory=AECStreamAlignConfig
    )
    linear_aec: LinearAECConfig | None = None
    input_routing: str = "mic_farend"

    def __post_init__(self) -> None:
        self.delay_alignment = _parse_delay_alignment(self.delay_alignment)
        self.linear_aec = _parse_linear_aec(self.linear_aec)
        if not isinstance(self.input_routing, str):
            raise ValueError(
                "input_routing must be a string, "
                f"got {self.input_routing!r}"
            )
        self.input_routing = self.input_routing.strip().lower()
        if self.input_routing not in _INPUT_ROUTINGS:
            raise ValueError(
                f"input_routing must be one of {sorted(_INPUT_ROUTINGS)}, "
                f"got {self.input_routing!r}"
            )

        has_linear_backend = (
            self.linear_aec is not None and self.linear_aec.backend is not None
        )
        if self.input_routing in _ERROR_ROUTINGS and not has_linear_backend:
            raise ValueError(
                f"input_routing={self.input_routing!r} requires a LinearAEC backend"
            )
        if self.input_routing == "mic_farend" and self.linear_aec is not None:
            raise ValueError(
                "input_routing='mic_farend' cannot configure linear_aec because "
                "the generated error would not be consumed"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "AECFrontendConfig":
        """严格解析 canonical mapping，未知键立即报错。"""
        if raw is None:
            return cls()
        values = _strict_mapping(raw, "aec_frontend")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"aec_frontend has unknown key(s) {unknown}; "
                f"expected subset of {sorted(known)}"
            )
        delay_alignment = values.get("delay_alignment", AECStreamAlignConfig())
        linear_aec = values.get("linear_aec")
        input_routing = values.get("input_routing", cls.input_routing)
        return cls(
            delay_alignment=delay_alignment,
            linear_aec=linear_aec,
            input_routing=input_routing,
        )

    def to_dict(self) -> dict[str, Any]:
        """返回可序列化的 canonical mapping。"""
        return {
            "delay_alignment": self.delay_alignment.to_dict(),
            "linear_aec": (
                asdict(self.linear_aec) if self.linear_aec is not None else None
            ),
            "input_routing": self.input_routing,
        }


def _legacy_align_paths(cfg: Mapping[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key == "aec_stream_align":
            paths.append(path)
        elif isinstance(value, Mapping):
            paths.extend(_legacy_align_paths(value, path))
    return paths


def resolve_aec_frontend_config(
    cfg: Mapping[str, Any] | AECFrontendConfig | None,
) -> AECFrontendConfig:
    """从 whole-config 的顶层 ``aec_frontend`` 解析 canonical 配置。

    ``aec_stream_align`` 与顶层 ``linear_aec`` 的所有旧位置均明确拒绝；
    唯一入口分别是 ``aec_frontend.delay_alignment.enabled`` 与
    ``aec_frontend.linear_aec``。
    """
    if cfg is None:
        return AECFrontendConfig()
    if isinstance(cfg, AECFrontendConfig):
        return cfg
    raw = _strict_mapping(cfg, "cfg")

    legacy_align_paths = _legacy_align_paths(raw)
    if legacy_align_paths:
        raise ValueError(
            "legacy aec_stream_align key(s) are not supported: "
            + ", ".join(legacy_align_paths)
            + "; configure aec_frontend.delay_alignment.enabled instead"
        )

    misplaced = sorted({"delay_alignment", "input_routing"}.intersection(raw))
    if misplaced:
        raise ValueError(
            f"top-level AEC frontend key(s) {misplaced} are not supported; "
            "configure them under aec_frontend"
        )

    if "linear_aec" in raw:
        raise ValueError(
            "top-level linear_aec is no longer supported; configure "
            "aec_frontend.linear_aec instead"
        )
    if "aec_frontend" in raw:
        return AECFrontendConfig.from_dict(
            _strict_mapping(raw["aec_frontend"], "aec_frontend")
        )
    return AECFrontendConfig()


def _validate_batch_inputs(
    mic: torch.Tensor,
    farend: torch.Tensor,
    lengths: torch.Tensor,
    sample_rate: int,
) -> list[int]:
    if not isinstance(mic, torch.Tensor) or not isinstance(farend, torch.Tensor):
        raise TypeError("mic and farend must be torch tensors")
    if mic.ndim != 2 or farend.ndim != 2:
        raise ValueError(
            f"mic and farend must have shape [B, T], got {mic.shape}, {farend.shape}"
        )
    if mic.shape != farend.shape:
        raise ValueError(f"mic/farend shape mismatch: {mic.shape} != {farend.shape}")
    if mic.device != farend.device:
        raise ValueError(f"mic/farend device mismatch: {mic.device} != {farend.device}")
    if not mic.is_floating_point() or not farend.is_floating_point():
        raise ValueError("mic and farend must use floating-point dtypes")
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
        raise ValueError("lengths must be a 1-D torch tensor")
    if lengths.numel() != mic.shape[0]:
        raise ValueError(
            f"lengths must have B={mic.shape[0]} entries, got {lengths.numel()}"
        )
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise ValueError("lengths must use an integer dtype")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
        raise ValueError(
            f"sample_rate must be a positive integer, got {sample_rate!r}"
        )

    resolved_lengths = [int(value) for value in lengths.detach().cpu().tolist()]
    total = mic.shape[1]
    for index, length in enumerate(resolved_lengths):
        if not 0 <= length <= total:
            raise ValueError(
                f"lengths[{index}] must be in [0, {total}], got {length}"
            )
    if total and resolved_lengths:
        # 设备端一次性完成有限性校验，避免逐 utterance 的 B 次 GPU→CPU 同步；
        # padding 尾部不属于有效样本，不参与检查。
        valid = torch.arange(total, device=mic.device).unsqueeze(0) < torch.as_tensor(
            resolved_lengths, device=mic.device
        ).unsqueeze(1)
        finite = (torch.isfinite(mic) & torch.isfinite(farend)) | ~valid
        if not bool(finite.all().item()):
            bad_rows = (~finite).any(dim=1).nonzero().flatten().tolist()
            first = int(bad_rows[0])
            name = (
                "mic"
                if not bool(torch.isfinite(mic[first, : resolved_lengths[first]]).all().item())
                else "farend"
            )
            raise ValueError(f"{name}[{first}] valid samples contain NaN or Inf")
    return resolved_lengths


class AECDeploymentFrontend:
    """对 `[B, T]` 波形运行 AEC input/deployment frontend。"""

    def __init__(
        self, config: AECFrontendConfig | Mapping[str, Any] | None = None
    ) -> None:
        self.config = (
            config
            if isinstance(config, AECFrontendConfig)
            else AECFrontendConfig.from_dict(config)
        )

    @torch.inference_mode()
    def process_batch(
        self,
        mic: torch.Tensor,
        farend: torch.Tensor,
        lengths: torch.Tensor,
        sample_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, Any]]:
        """处理有效样本并将所有 padding 尾部归零。"""
        valid_lengths = _validate_batch_inputs(mic, farend, lengths, sample_rate)
        aligned = torch.zeros_like(farend)
        delay_samples = [0 for _ in valid_lengths]
        delay_ms = [0.0 for _ in valid_lengths]
        identities: list[str | None] = [None for _ in valid_lengths]

        align_cfg = self.config.delay_alignment
        if align_cfg.enabled:
            estimator_cfg = align_cfg.to_estimator_config(sample_rate)
            # 每条 utterance 独立 estimator（既有合同：utterance 之间不得共享状态）；
            # 复用 + reset() 的省建开销方案待状态隔离验证后再评估。
            for index, length in enumerate(valid_lengths):
                if length == 0:
                    continue
                estimator = create_delay_estimator(estimator_cfg)
                mic_np = (
                    mic[index, :length].detach().to("cpu", torch.float32).numpy()
                )
                far_np = (
                    farend[index, :length].detach().to("cpu", torch.float32).numpy()
                )
                frames: list[np.ndarray] = []
                for start in range(0, length, align_cfg.frame_size):
                    end = min(start + align_cfg.frame_size, length)
                    estimator.push_ref(far_np[start:end])
                    estimator.push_mic(mic_np[start:end])
                    frames.append(estimator.get_aligned_ref_frame(end - start))
                aligned_np = np.concatenate(frames).astype(np.float32, copy=False)
                if aligned_np.shape != (length,) or not np.isfinite(aligned_np).all():
                    raise RuntimeError(
                        f"delay alignment produced invalid output for utterance {index}"
                    )
                aligned[index, :length] = torch.from_numpy(aligned_np).to(
                    device=farend.device, dtype=farend.dtype
                )
                utterance_delay_samples = int(estimator.delay_samples)
                utterance_delay_ms = float(estimator.delay_ms)
                if utterance_delay_samples < 0 or not math.isfinite(utterance_delay_ms):
                    raise RuntimeError(
                        f"delay estimator returned invalid delay for utterance {index}"
                    )
                delay_samples[index] = utterance_delay_samples
                delay_ms[index] = utterance_delay_ms
                identity = getattr(estimator, "backend_identity", None)
                if identity not in {"cpp", "python"}:
                    raise RuntimeError(
                        f"delay estimator returned invalid backend_identity {identity!r}"
                    )
                identities[index] = identity
        else:
            for index, length in enumerate(valid_lengths):
                if length:
                    aligned[index, :length] = farend[index, :length]

        error = self._process_linear_aec(mic, aligned, valid_lengths, sample_rate)
        if error is not None and not bool(torch.isfinite(error).all().item()):
            raise RuntimeError("LinearAEC produced NaN or Inf")
        error_mask = torch.full(
            (mic.shape[0],),
            error is not None,
            dtype=torch.bool,
            device=mic.device,
        )

        used_identities = sorted({item for item in identities if item is not None})
        backend_identity: str | None
        if not used_identities:
            backend_identity = None
        elif len(used_identities) == 1:
            backend_identity = used_identities[0]
        else:
            backend_identity = "mixed"
        linear_cfg = self.config.linear_aec
        facts: dict[str, Any] = {
            "input_routing": self.config.input_routing,
            "delay_alignment": {
                "enabled": align_cfg.enabled,
                "algorithm": "gcc_phat",
                "backend_identity": backend_identity,
                "backend_identity_per_utterance": identities,
                "delay_samples": delay_samples,
                "delay_ms": delay_ms,
            },
            "linear_aec": {
                "enabled": linear_cfg is not None,
                "backend_identity": self._linear_backend_identity(),
                "use_res": bool(linear_cfg.use_res) if linear_cfg else False,
            },
            "finite_policy": "fail_closed",
            "sanitized": False,
        }
        return aligned, error, error_mask, facts

    def _linear_backend_identity(self) -> str | None:
        config = self.config.linear_aec
        if config is None:
            return None
        if config.backend == "pfdkf_torch" or config.gpu_batch:
            return "pfdkf_torch"
        return config.backend

    def _process_linear_aec(
        self,
        mic: torch.Tensor,
        aligned: torch.Tensor,
        lengths: list[int],
        sample_rate: int,
    ) -> torch.Tensor | None:
        config = self.config.linear_aec
        if config is None:
            return None
        if config.backend is None:
            raise RuntimeError("validated LinearAEC config has no backend")

        error = torch.zeros_like(mic)
        if config.backend == "pfdkf_torch" or config.gpu_batch:
            self._process_torch_pfdkf(error, mic, aligned, lengths, config)
            return error

        frontend_cfg = config.to_frontend_dict()
        frontend_cls = LinearAECFrontend
        if frontend_cls is None:
            _, frontend_cls = _linear_aec_classes()
        for index, length in enumerate(lengths):
            if length == 0:
                continue
            # 每条 utterance 独立构造，避免 backend 状态或 cache 串扰。
            frontend = frontend_cls(frontend_cfg, sample_rate=sample_rate)
            mic_np = mic[index, :length].detach().to("cpu", torch.float32).numpy()
            ref_np = aligned[index, :length].detach().to("cpu", torch.float32).numpy()
            error_np = np.asarray(frontend.process(mic_np, ref_np), dtype=np.float32)
            if error_np.shape != (length,) or not np.isfinite(error_np).all():
                raise RuntimeError(
                    f"LinearAEC produced invalid output for utterance {index}"
                )
            error[index, :length] = torch.from_numpy(error_np).to(
                device=mic.device, dtype=mic.dtype
            )
        return error

    @staticmethod
    def _process_torch_pfdkf(
        output: torch.Tensor,
        mic: torch.Tensor,
        aligned: torch.Tensor,
        lengths: list[int],
        config: LinearAECConfig,
    ) -> None:
        from tasks.aec.data.pfdkf_torch import PFDKFTorch, PFDKFRESTorch

        model_cls = PFDKFRESTorch if config.use_res else PFDKFTorch
        kwargs = {
            "N": config.N,
            "M": config.M,
            "hop": config.hop,
            "A": config.A,
            "P_initial": config.P_initial,
            "keep_m_gate": config.keep_m_gate,
        }
        if config.use_res:
            kwargs["res"] = True
        model = model_cls(**kwargs).to(mic.device).eval()

        groups: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            if length:
                groups.setdefault(length, []).append(index)
        for length, indices in groups.items():
            index_tensor = torch.tensor(indices, device=mic.device, dtype=torch.long)
            mic_group = mic.index_select(0, index_tensor)[:, :length].to(torch.float32)
            ref_group = aligned.index_select(0, index_tensor)[:, :length].to(torch.float32)
            candidate = model.process(mic_group, ref_group)
            if candidate.shape != mic_group.shape or not bool(
                torch.isfinite(candidate).all().item()
            ):
                raise RuntimeError(
                    f"PFDKF Torch produced invalid output for length group {length}"
                )
            output[index_tensor, :length] = candidate.to(output.dtype)


__all__ = [
    "AECDeploymentFrontend",
    "AECFrontendConfig",
    "resolve_aec_frontend_config",
]
