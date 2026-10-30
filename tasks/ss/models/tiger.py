"""TIGER adapter with framework-owned external STFT/iSTFT."""

from __future__ import annotations

import torch

from core.utils.registry import MODEL_REGISTRY
from core.models.io_contract import ModelIOContract, TensorField
from tasks.ss.models._third_party.tiger import TIGER as _UpstreamTIGER


@MODEL_REGISTRY.register("TIGER")
class TIGER(_UpstreamTIGER):
    """External-STFT TIGER returning complex source spectra."""

    input_domain = "spectrum"
    output_domain = "spectrum"
    IO_CONTRACT = ModelIOContract(
        tasks=frozenset({"ss"}),
        inputs=(
            TensorField(
                "mixture_spectra",
                "spectrum",
                ("batch", "frequency", "frame", "complex"),
            ),
        ),
        outputs=(
            TensorField(
                "est_sources_spectra",
                "spectrum",
                ("batch", "source", "frequency", "frame", "complex"),
                multiple=True,
            ),
        ),
        forward_kind="estimate",
        frontend_owner="runtime",
    )

    def __init__(
        self,
        out_channels: int = 128,
        in_channels: int = 256,
        num_blocks: int = 4,
        upsampling_depth: int = 5,
        att_n_head: int = 4,
        att_hid_chan: int = 4,
        att_kernel_size: int = 8,
        att_stride: int = 1,
        win: int = 640,
        stride: int = 160,
        num_sources: int = 2,
        sample_rate: int = 16000,
        n_fft=None,
        hop_length=None,
        win_length=None,
        window=None,
        center=None,
    ) -> None:
        if n_fft is not None and int(n_fft) != int(win):
            raise ValueError("TIGER external STFT n_fft must match model win")
        if hop_length is not None and int(hop_length) != int(stride):
            raise ValueError("TIGER external STFT hop_length must match model stride")
        if win_length is not None and int(win_length) != int(win):
            raise ValueError("TIGER external STFT win_length must match model win")
        if window is not None and str(window).lower() != "hann":
            raise ValueError("TIGER external STFT currently requires window='hann'")
        if center is not None and center is not True:
            raise ValueError("TIGER external STFT currently requires center=True")
        super().__init__(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            att_n_head=att_n_head,
            att_hid_chan=att_hid_chan,
            att_kernel_size=att_kernel_size,
            att_stride=att_stride,
            win=win,
            stride=stride,
            num_sources=num_sources,
            sample_rate=sample_rate,
        )
        if any(width <= 0 for width in self.band_width):
            raise ValueError(
                "TIGER band split produced an empty frequency band; use an STFT "
                "geometry compatible with the official 16 kHz recipe (640/160/640)"
            )

    def forward(self, mixture_spectra, lengths=None, *, fs=None):
        del fs
        if mixture_spectra.ndim != 4 or mixture_spectra.shape[-1] != 2:
            raise ValueError("TIGER expects mixture_spectra with shape [B, F, T, 2]")
        if lengths is None:
            raise ValueError("TIGER external-STFT forward requires lengths")
        complex_spec = torch.view_as_complex(mixture_spectra.float().contiguous())
        output_spec = self._separate_spectrum(complex_spec)
        return {"est_sources_spectra": torch.view_as_real(output_spec)}

    def forward_from_spectrum(self, spec, output_length):
        del spec, output_length
        raise RuntimeError(
            "TIGER production adapter enforces the external-STFT contract; "
            "iSTFT belongs to SSTrainer or SSExtractor"
        )
