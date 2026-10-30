"""SSL 的具名 runtime 输入边界。

`SSLBatch` 是数据层合同，`SSLRuntimeInputs` 是训练层合同。两者之间不能
通过位置参数猜测字段含义：输出帧长度、pair layout、几何和坐标系都必须
原样带到 loss/mapping 边界。模型本身只接收其 ``ModelSpec`` 声明的特征，
source distance 默认不会进入模型 forward。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch

from tasks.ssl.data.batch import SSLBatch


def _forward_accepts_pair_index(model: Any) -> bool:
    """判断模型 forward 是否显式声明 ``pair_index`` 形参。

    需要时先解开 DDP/``torch.compile`` 包装，避免包装器的 ``(*args, **kwargs)``
    签名掩盖真实模型的形参。
    """
    candidate = model
    while True:
        inner = getattr(candidate, "module", None)
        if inner is None:
            inner = getattr(candidate, "_orig_mod", None)
        if inner is None or inner is candidate:
            break
        candidate = inner
    forward = getattr(candidate, "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    return "pair_index" in parameters


def _unwrap_model_contract(model: Any) -> Any:
    """解开 DDP/``torch.compile`` 包装以读取模型声明的合同。

    Trainer 的 forward 句柄可能是 ``DDP(torch.compile(model))``；这些包装器
    不会透传 ``MODEL_SPEC``，若直接在包装器上读取会把官方多通道模型误判为
    旧的 ``features`` 输入，并丢失 ``IPDNetVariable`` 的 ``pair_index``。
    """
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "module", None)
        if inner is None:
            inner = getattr(current, "_orig_mod", None)
        if inner is None or inner is current:
            break
        current = inner
    return current


@dataclass(frozen=True)
class SSLRuntimeInputs:
    """把 canonical SSL batch 映射成具名 runtime/loss 输入。

    ``target_dpipd`` 使用 ``[B,T_out,F,K,P,2]``；``input_lengths`` 与
    ``output_lengths`` 分别属于输入和输出时间轴。保留 ``ipd_target`` 和
    ``lengths`` 属性作为短期只读兼容别名，禁止新代码依赖旧名称。
    """

    features: torch.Tensor
    target_dpipd: torch.Tensor
    pair_index: torch.Tensor
    pair_mask: torch.Tensor
    geometry: torch.Tensor
    source_distances: torch.Tensor | None
    input_lengths: torch.Tensor
    output_lengths: torch.Tensor
    frame_origin: torch.Tensor
    time_stride: int
    input_representation: str
    target_representation: str
    coordinate_frame: str
    doa: torch.Tensor | None
    vad: torch.Tensor | None
    sample_rates: torch.Tensor | None

    @property
    def ipd_target(self) -> torch.Tensor:
        """兼容旧 loss 包装器；新代码使用 ``target_dpipd``。"""
        return self.target_dpipd

    @property
    def lengths(self) -> torch.Tensor:
        """兼容旧 runtime 调用；loss 必须使用 output_lengths。"""
        return self.output_lengths

    @classmethod
    def from_batch(cls, batch: SSLBatch) -> "SSLRuntimeInputs":
        """验证 batch 后创建 runtime view。"""
        batch.validate()
        return cls._from_validated_batch(batch)

    @classmethod
    def _from_validated_batch(cls, batch: SSLBatch) -> "SSLRuntimeInputs":
        """创建 runtime view；调用方必须已在同步边界完成 batch 校验。"""
        # ``SSLBatch.validate`` 已经把这些字段登记为 canonical 必填合同；这里
        # 必须直接读取，不能把旧的 ``targets``/``lengths``/``output_stride``
        # 别名或默认坐标系重新带回生产 runtime。
        target = batch.target_dpipd
        input_lengths = batch.input_lengths
        output_lengths = batch.output_lengths
        pair_index = batch.pair_index
        pair_mask = batch.pair_mask
        geometry = batch.geometry
        frame_origin = batch.frame_origin
        source_distances = batch.source_distances
        time_stride = batch.time_stride
        if type(time_stride) is not int or time_stride < 1:
            raise ValueError("SSLBatch must expose a positive integer time_stride")
        input_representation = batch.input_representation
        target_representation = batch.target_representation
        coordinate_frame = batch.coordinate_frame
        if not isinstance(input_representation, str) or not input_representation:
            raise ValueError("SSLBatch input_representation must be a non-empty string")
        if not isinstance(target_representation, str) or not target_representation:
            raise ValueError("SSLBatch target_representation must be a non-empty string")
        if not isinstance(coordinate_frame, str) or not coordinate_frame:
            raise ValueError("SSLBatch coordinate_frame must be a non-empty string")
        return cls(
            features=batch.features,
            target_dpipd=target,
            pair_index=pair_index,
            pair_mask=pair_mask,
            geometry=geometry,
            source_distances=source_distances,
            input_lengths=input_lengths,
            output_lengths=output_lengths,
            frame_origin=frame_origin,
            time_stride=time_stride,
            input_representation=input_representation,
            target_representation=target_representation,
            coordinate_frame=coordinate_frame,
            doa=batch.doa,
            vad=batch.vad,
            sample_rates=batch.sample_rates,
        )

    def model_kwargs(self, model: Any | None = None) -> dict[str, Any]:
        """按 ``ModelSpec.feature_contract`` 选择唯一的模型输入名称。

        官方 variable-array IPDnet 的 pair-batched 合同要求 forward 显式接收
        canonical ``pair_index``；当模型 forward 声明了该形参时一并传入，
        其余模型不受影响。
        """
        raw_model = _unwrap_model_contract(model)
        spec = getattr(raw_model, "MODEL_SPEC", getattr(raw_model, "model_spec", None))
        feature_contract = getattr(spec, "feature_contract", None)
        if feature_contract == "multichannel_stft_real_imag":
            kwargs: dict[str, Any] = {"multichannel_stft_real_imag": self.features}
            if _forward_accepts_pair_index(model):
                kwargs["pair_index"] = self.pair_index
            return kwargs
        if feature_contract in (None, "reference_ipd"):
            return {"features": self.features}
        raise ValueError(f"unsupported SSL model feature_contract: {feature_contract!r}")

    def loss_kwargs(self) -> dict[str, Any]:
        """返回 DP-IPD 与所有共享 PIT 辅助项所需的完整 batch target context。

        assignment 由 DP-IPD loss 单独求解；DOA、VAD、range 不携带各自的
        permutation，也不在 runtime 中预先对齐，避免多头使用不一致的 source 对应关系。
        """
        return {
            "target_dpipd": self.target_dpipd,
            "input_lengths": self.input_lengths,
            "output_lengths": self.output_lengths,
            "pair_index": self.pair_index,
            "pair_mask": self.pair_mask,
            "geometry": self.geometry,
            "source_distances": self.source_distances,
            "frame_origin": self.frame_origin,
            "time_stride": self.time_stride,
            "input_representation": self.input_representation,
            "target_representation": self.target_representation,
            "coordinate_frame": self.coordinate_frame,
            "doa": self.doa,
            "vad": self.vad,
        }


__all__ = ["SSLRuntimeInputs"]
