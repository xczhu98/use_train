"""TDANet adapter with the SS task output contract."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.utils.registry import MODEL_REGISTRY
from core.models.io_contract import ModelIOContract, TensorField
from tasks.ss.models._third_party.tdanet import TDANet as _UpstreamTDANet


@MODEL_REGISTRY.register("TDANet")
class TDANet(_UpstreamTDANet):
    """Waveform-native TDANet returning ``{"est_sources": [B, K, T]}``."""

    input_domain = "waveform"
    output_domain = "waveform"
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(TensorField("mixture", "waveform", ("batch", "time")),),
        outputs=(
            TensorField(
                "est_sources",
                "waveform",
                ("batch", "source", "time"),
                multiple=True,
            ),
        ),
        forward_kind="estimate",
        frontend_owner="model",
    )

    def __init__(
        self,
        out_channels: int = 128,
        in_channels: int = 512,
        num_blocks: int = 16,
        upsampling_depth: int = 5,
        enc_kernel_size: int = 4,
        num_sources: int = 2,
        sample_rate: int = 8000,
    ) -> None:
        super().__init__(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            enc_kernel_size=enc_kernel_size,
            num_sources=num_sources,
            sample_rate=sample_rate,
        )

    def forward(self, mixture: torch.Tensor, lengths=None, *, fs=None):
        del fs
        if mixture.ndim != 2:
            raise ValueError(f"TDANet mixture must have shape [B, T], got {tuple(mixture.shape)}")
        output = super().forward(mixture)
        if output.ndim != 3 or output.shape[0] != mixture.shape[0]:
            raise RuntimeError(
                f"TDANet output must have shape [B, K, T], got {tuple(output.shape)}"
            )
        if output.shape[-1] == 0:
            raise RuntimeError(
                "TDANet input is too short to produce any output frame "
                f"(received {mixture.shape[-1]} samples, "
                f"enc_kernel_size={self.enc_kernel_size})"
            )
        if output.shape[-1] != mixture.shape[-1]:
            output = output[..., : mixture.shape[-1]]
            if output.shape[-1] < mixture.shape[-1]:
                output = torch.nn.functional.pad(output, (0, mixture.shape[-1] - output.shape[-1]))
        return {"est_sources": output}
