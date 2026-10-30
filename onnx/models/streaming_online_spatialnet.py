"""OnlineSpatialNet 的多通道输入、单通道输出与稳定 cache ABI seam。

本模块不复制 NBSS 实现，也不把离线 SpatialNet 包装成伪流式模型。调用方必须提供
显式声明 ``streaming_spatial_abi_version=2`` 的因果 backend；wrapper 只负责把 backend
的 mapping cache 展平成 ONNX tensor I/O，并提供可重置的 PyTorch session。

ABI v2 的 cache 语义按分支变化：``mhsa`` 为逐层增量 K/V（外加全局 ``frame_index``
warm-up 计数），``ret`` 为 recurrent kv/scale 状态，``mamba`` 为 conv/ssm 状态；
``mamba`` 分支依赖 Triton/mamba_ssm 内核，不提供 ONNX 导出。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch
from torch import Tensor, nn


STREAMING_SPATIAL_ABI_VERSION = 2


class StreamingOnlineSpatialNetFlat(nn.Module):
    """将真实 OnlineSpatialNet backend 适配为稳定的 flat tensor ABI。

    backend 必须提供 ``CACHE_KEYS``、``init_cache(batch_size, device, dtype)`` 和
    ``forward_streaming(mixture_spec, cache)``。主输入固定为
    ``[B,C,F,T,2]`` microphone observation，主输出固定为 ``[B,F,T,2]`` 单通道谱。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("OnlineSpatialNet backend must be a torch.nn.Module")
        if getattr(model, "streaming_spatial_abi_version", None) != STREAMING_SPATIAL_ABI_VERSION:
            raise ValueError(
                "OnlineSpatialNet backend must explicitly declare "
                f"streaming_spatial_abi_version={STREAMING_SPATIAL_ABI_VERSION}; "
                "offline SpatialNet cannot be treated as a streaming model"
            )
        for method_name in ("init_cache", "forward_streaming"):
            if not callable(getattr(model, method_name, None)):
                raise TypeError(f"OnlineSpatialNet backend must implement {method_name}()")
        cache_keys = getattr(model, "CACHE_KEYS", None)
        if (
            not isinstance(cache_keys, Sequence)
            or isinstance(cache_keys, (str, bytes))
            or not cache_keys
            or any(not isinstance(name, str) or not name for name in cache_keys)
            or len(set(cache_keys)) != len(cache_keys)
        ):
            raise ValueError("OnlineSpatialNet CACHE_KEYS must be a non-empty unique string sequence")
        n_channel = getattr(model, "n_channel", None)
        n_fft = getattr(model, "n_fft", None)
        if isinstance(n_channel, bool) or not isinstance(n_channel, int) or n_channel < 1:
            raise ValueError("OnlineSpatialNet backend n_channel must be a positive integer")
        if isinstance(n_fft, bool) or not isinstance(n_fft, int) or n_fft < 2:
            raise ValueError("OnlineSpatialNet backend n_fft must be an integer >= 2")
        self.model = model
        self.CACHE_KEYS = tuple(cache_keys)
        self.n_channel = n_channel
        self.n_fft = n_fft
        self.hop_length = getattr(model, "hop_length", None)
        self.win_length = getattr(model, "win_length", None)
        self.target_fs = getattr(model, "target_fs", None)
        self.window = getattr(model, "window", "hann")
        self.center = getattr(model, "center", None)

    def _ordered_cache(
        self,
        values: object,
        *,
        label: str,
    ) -> tuple[Tensor, ...]:
        if not isinstance(values, Mapping):
            raise TypeError(f"OnlineSpatialNet {label} must be a mapping")
        if tuple(values) != self.CACHE_KEYS:
            raise ValueError(
                f"OnlineSpatialNet {label} names/order mismatch: "
                f"expected={self.CACHE_KEYS}, got={tuple(values)}"
            )
        tensors = tuple(values[name] for name in self.CACHE_KEYS)
        if any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError(f"OnlineSpatialNet {label} values must be tensors")
        return tensors

    def init_cache(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        values = self.model.init_cache(batch_size, torch.device(device), dtype)
        tensors = self._ordered_cache(values, label="cache initializer")
        return {name: value for name, value in zip(self.CACHE_KEYS, tensors)}

    def init_cache_flat(
        self,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> list[Tensor]:
        values = self.init_cache(batch_size, device, dtype)
        return [values[name] for name in self.CACHE_KEYS]

    def get_input_names(self) -> list[str]:
        return ["mixture_spec", *self.CACHE_KEYS]

    def get_output_names(self) -> list[str]:
        return ["enhanced_spec", *(f"{name}_out" for name in self.CACHE_KEYS)]

    def _cache_batch_axes(self) -> dict[str, tuple[int, str]]:
        one = self.init_cache(1, "cpu", torch.float32)
        two = self.init_cache(2, "cpu", torch.float32)
        axes: dict[str, tuple[int, str]] = {}
        for name in self.CACHE_KEYS:
            shape_one = tuple(one[name].shape)
            shape_two = tuple(two[name].shape)
            if len(shape_one) != len(shape_two):
                raise ValueError(f"OnlineSpatialNet cache {name!r} rank changes with batch size")
            changed = tuple(
                axis
                for axis, (first, second) in enumerate(zip(shape_one, shape_two))
                if first != second
            )
            if len(changed) != 1:
                raise ValueError(
                    f"OnlineSpatialNet cache {name!r} must expose exactly one batch-dependent axis"
                )
            axis = changed[0]
            if shape_one[axis] < 1 or shape_two[axis] != shape_one[axis] * 2:
                raise ValueError(
                    f"OnlineSpatialNet cache {name!r} batch axis is not linearly scalable"
                )
            axes[name] = (
                axis,
                "batch" if shape_one[axis] == 1 else "batch_flattened",
            )
        return axes

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes = {
            "mixture_spec": {0: "batch"},
            "enhanced_spec": {0: "batch"},
        }
        for name, (axis, label) in self._cache_batch_axes().items():
            axes[name] = {axis: label}
            axes[f"{name}_out"] = {axis: label}
        return axes

    def forward(self, mixture_spec: Tensor, *cache: Tensor) -> tuple[Tensor, ...]:
        if len(cache) != len(self.CACHE_KEYS):
            raise ValueError(
                f"expected {len(self.CACHE_KEYS)} OnlineSpatialNet cache tensors, got {len(cache)}"
            )
        if mixture_spec.ndim != 5 or mixture_spec.shape[-1] != 2:
            raise ValueError("OnlineSpatialNet expects mixture_spec shape [B,C,F,T,2]")
        if mixture_spec.shape[3] != 1:
            raise ValueError("OnlineSpatialNet streaming ABI accepts exactly one frame per step")
        if mixture_spec.shape[1] != self.n_channel:
            raise ValueError(
                f"OnlineSpatialNet expected C={self.n_channel}, got {mixture_spec.shape[1]}"
            )
        expected_freqs = self.n_fft // 2 + 1
        if mixture_spec.shape[2] != expected_freqs:
            raise ValueError(
                f"OnlineSpatialNet expected F={expected_freqs}, got {mixture_spec.shape[2]}"
            )
        cache_mapping = {name: value for name, value in zip(self.CACHE_KEYS, cache)}
        result = self.model.forward_streaming(mixture_spec, cache_mapping)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("OnlineSpatialNet forward_streaming() must return (output, cache)")
        output, next_cache = result
        next_values = self._ordered_cache(next_cache, label="updated cache")
        expected_output_shape = (
            mixture_spec.shape[0],
            mixture_spec.shape[2],
            mixture_spec.shape[3],
            2,
        )
        if not isinstance(output, Tensor) or tuple(output.shape) != expected_output_shape:
            raise ValueError(
                "OnlineSpatialNet must return mono spectrum [B,F,T,2], got "
                f"{getattr(output, 'shape', None)}"
            )
        return (output, *next_values)


class OnlineSpatialNetSession:
    """PyTorch chunk session；cache 仅在完整输出通过校验后提交。"""

    def __init__(
        self,
        model: StreamingOnlineSpatialNetFlat,
        *,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(model, StreamingOnlineSpatialNetFlat):
            raise TypeError("model must be StreamingOnlineSpatialNetFlat")
        self.model = model.to(device).eval()
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self._state: dict[str, Tensor] = {}
        self.reset()

    @property
    def state(self) -> Mapping[str, Tensor]:
        return MappingProxyType({name: value.detach().clone() for name, value in self._state.items()})

    def reset(self) -> None:
        initialized = self.model.init_cache(self.batch_size, self.device, self.dtype)
        self._state = {name: value.detach().clone() for name, value in initialized.items()}

    @torch.inference_mode()
    def step(self, mixture_spec: Tensor) -> Tensor:
        if not isinstance(mixture_spec, Tensor):
            raise TypeError("mixture_spec must be a torch.Tensor")
        if mixture_spec.device != self.device or mixture_spec.dtype != self.dtype:
            raise ValueError(
                "mixture_spec device/dtype must match session state: "
                f"expected=({self.device},{self.dtype}), "
                f"got=({mixture_spec.device},{mixture_spec.dtype})"
            )
        if mixture_spec.ndim != 5:
            raise ValueError("mixture_spec must have shape [B,C,F,T,2]")
        if mixture_spec.shape[0] != self.batch_size:
            raise ValueError(
                f"mixture_spec batch must be {self.batch_size}, got {mixture_spec.shape[0]}"
            )
        if mixture_spec.numel() == 0 or not torch.isfinite(mixture_spec).all():
            raise ValueError("mixture_spec must be non-empty and finite")
        values = tuple(self._state[name] for name in self.model.CACHE_KEYS)
        output, *next_values = self.model(mixture_spec, *values)
        if output.dtype != self.dtype or output.device != self.device or not torch.isfinite(output).all():
            raise RuntimeError("OnlineSpatialNet output changed dtype/device or contains non-finite values")
        pending: dict[str, Tensor] = {}
        for name, current, value in zip(self.model.CACHE_KEYS, values, next_values):
            if value.shape != current.shape or value.dtype != current.dtype or value.device != current.device:
                raise RuntimeError(f"OnlineSpatialNet cache {name!r} changed shape/dtype/device")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"OnlineSpatialNet cache {name!r} contains non-finite values")
            pending[name] = value.detach().clone()
        self._state = pending
        return output

    def process(self, chunks: Sequence[Tensor]) -> Tensor:
        if not chunks:
            raise ValueError("OnlineSpatialNet session requires at least one chunk")
        outputs = []
        for chunk in chunks:
            if chunk.ndim != 5:
                raise ValueError("OnlineSpatialNet chunks must have shape [B,C,F,T,2]")
            outputs.extend(self.step(chunk[:, :, :, index : index + 1, :]) for index in range(chunk.shape[3]))
        if not outputs:
            raise ValueError("OnlineSpatialNet chunks must contain at least one frame")
        return torch.cat(outputs, dim=2)
