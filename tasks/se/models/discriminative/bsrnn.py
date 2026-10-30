"""Unified BSRNN (Band-Split RNN) for speech enhancement.

Training-only interface: spectrum-in / spectrum-out. STFT/iSTFT and streaming
are handled under use_train/onnx.

Architecture:
  - BandSplit: split spectrum into subbands, each subband embedded independently
  - Time-dim LSTM: models intra-subband temporal relations (uni/bidirectional)
  - Freq-dim LSTM: models cross-subband spectral relations (bidirectional)
  - MaskDecoder: outputs complex mask + residual for signal separation

Usage:
  # Offline non-causal (e.g., URGENT2026 baseline; official repo conf uses num_channel=196)
  model = BSRNN(num_channel=196, causal=False, n_fft=960, hop_length=480, use_residual=True)
  # Offline causal (for comparison)
  model = BSRNN(num_channel=64, causal=True, n_fft=512, hop_length=256)

Run directly to test params and compute:
  python tasks/se/models/bsrnn.py [--sr 16000] [--dur 1.0]

论文与实现参考（与 ``bsrnn_impl.bsrnn_core`` 一致）:
  [1] J. Yu et al., "High fidelity speech enhancement with band-split RNN,"
      Proc. Interspeech, 2023.
  [2] J. Yu & Y. Luo, "Efficient monaural speech enhancement with universal sample rate
      band-split RNN," Proc. IEEE ICASSP, 2023.
"""

from __future__ import annotations

if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import torch

from tasks.se.models.base_model import BaseSEModel
from tasks.se.models.modules.bsrnn_impl import BSRNN as _BSRNNCore
from core.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("BSRNN")
class BSRNN(BaseSEModel):
    """Unified BSRNN: spectrum-in / spectrum-out. STFT/iSTFT is done by the caller (trainer/onnx).

    Args:
        num_channel: Embedding dimension per T-F bin. URGENT2026 official baseline=196. Default 192; 64 (Base), 16 (USR-Lite).
        num_layer: Number of time/freq LSTM layers. Default 6.
        target_fs: Internal processing sample rate. Default 48000.
        norm_type: Normalization. 'GN' (GroupNorm, espnet2), 'cLN' (ChannelwiseLayerNorm).
        causal: True=unidirectional time-LSTM, False=bidirectional (offline quality).
        hidden_factor: LSTM hidden = hidden_factor * num_channel. Default 2.
        use_residual: Add residual connection in mask decoder. Default False (USR paper).
    """

    def __init__(
        self,
        num_channel: int = 192,
        num_layer: int = 6,
        target_fs: int = 48000,
        norm_type: str = "GN",
        causal: bool = False,
        hidden_factor: int = 2,
        use_residual: bool = False,
        **kwargs,
    ):
        super().__init__()
        n_fft = kwargs.pop("n_fft", 960)
        hop_length = kwargs.pop("hop_length", 480)
        for key in ("win_length", "window", "center"):
            kwargs.pop(key, None)
        if kwargs:
            raise TypeError(f"BSRNN got unexpected keyword arguments: {sorted(kwargs)}")

        self.target_fs = target_fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.causal = causal

        input_dim = n_fft // 2 + 1
        self.bsrnn = _BSRNNCore(
            input_dim=input_dim,
            num_channel=num_channel,
            num_layer=num_layer,
            target_fs=target_fs,
            causal=causal,
            num_spk=1,
            norm_type=norm_type,
            hidden_factor=hidden_factor,
            use_residual=use_residual,
        )

    def forward(self, noisy_spec: torch.Tensor, lengths=None, fs: int = None) -> torch.Tensor:
        """Spectrum-in / spectrum-out. STFT/iSTFT must be done by the caller (trainer or inference).

        Args:
            noisy_spec: (B, F, T, 2) real-valued complex spectrum.
            lengths: optional (B,) valid lengths.
            fs: optional sampling rate for sub-Nyquist subband truncation.

        Returns:
            enhanced_spec: (B, F, T, 2).
        """
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "BSRNN accepts only spectrum input (B, F, T, 2). "
                "Do STFT in the caller (trainer/inference) and pass the result here."
            )
        if noisy_spec.shape[0] == 0 or noisy_spec.shape[2] == 0:
            raise ValueError("BSRNN requires non-empty batch and time dimensions")
        return self._forward_spectrum(noisy_spec, lengths, fs)

    def _forward_spectrum(
        self, noisy_spec: torch.Tensor, lengths=None, fs: int = None
    ) -> torch.Tensor:
        """Spectrum-in / spectrum-out. noisy_spec (B, F, T, 2) -> enhanced_spec (B, F, T, 2)."""
        if fs is None:
            fs = self.target_fs
        if isinstance(fs, torch.Tensor):
            fs = fs.item()
        fs = int(fs)
        feat = noisy_spec.permute(0, 2, 1, 3)
        enhanced_feat = self.bsrnn(feat, fs=fs)
        if enhanced_feat.ndim == 5:
            enhanced_feat = enhanced_feat[:, 0]
        return enhanced_feat.permute(0, 2, 1, 3)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证本模型参数量与算力。修改下面配置后运行: python tasks/se/models/bsrnn.py [--sr 48000] [--dur 1.0] [--device cpu]
    # -------------------------------------------------------------------------
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft
    import argparse

    args = argparse.ArgumentParser(parents=[make_profile_parser(default_sr=48000)], add_help=False).parse_args()

    # 完整配置：STFT、采样率、模型全部参数（与 YAML 对齐）
    SAMPLE_RATE = args.sr
    STFT_CFG = {
        "n_fft": 960,
        "hop_length": 480,
        "win_length": 960,
        "window": "hann",
    }
    MODEL_ARGS = {
        "num_channel": 196,
        "num_layer": 6,
        "n_fft": STFT_CFG["n_fft"],
        "hop_length": STFT_CFG["hop_length"],
        "target_fs": 48000,
        "causal": False,
        "use_residual": True,
        # "norm_type": "GN", "hidden_factor": 2  # 默认值，可按需覆盖
    }

    model = BSRNN(**MODEL_ARGS)
    stft_module = build_stft_istft(STFT_CFG)

    class _W(torch.nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m
        def forward(self, wav, **kw):
            spec = self._stft(wav)
            out_spec = self._m(spec, **kw)
            return self._stft.inverse(out_spec, wav.shape[-1])

    wrapped = _W(model, stft_module)
    fwd_kw = {"fs": SAMPLE_RATE} if model.target_fs != SAMPLE_RATE else {}
    print("\n[Config] STFT:", STFT_CFG, "| sample_rate:", SAMPLE_RATE, "| model args:", MODEL_ARGS)
    profile_model(
        wrapped,
        sample_rate=SAMPLE_RATE,
        duration=args.dur,
        batch_size=args.batch,
        device=args.device,
        repeats=args.repeats,
        forward_kwargs=fwd_kw,
        label="BSRNN",
    )
