"""Streaming DBGNet wrapper for frame-by-frame ONNX export.

Wraps the training-time ``DBGNet`` (``tasks/aec/models/discriminative/dbgnet.py``) into a
stateful streaming module that consumes one STFT frame at a time and produces:

- enhanced complex spectrogram ``(B, F, 1, 2)``
- optional frame-level state detector logits ``(B, 1, state_nclass)``

All temporal state (convolution history, GRU hidden states) is kept in an
explicit cache tuple so the ONNX graph is stateless. The Python streaming
module retains its existing batching math, while the formal flat ONNX
deployment contract is intentionally fixed to batch size 1.

``mask_domain`` 随底层 ``DBGNet`` 继承：``"mag"``（默认）逐帧复现实值 sigmoid 掩码
路径；``"com"``（DBG-Com）使用 2 通道 RI 输入与 tanh 复数掩码。cache 名字、顺序与
cache ABI 不随掩码域变化，但首层 ``mic_in`` / ``far_in`` 的通道数由输入通道数决定
（mag=1，com=2），因此不同掩码域的 cache 不可混用，切换时必须 cold reset。
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import torch
import torch.nn as nn

from tasks.aec.models.discriminative.dbgnet import DBGNet, _apply_complex_mask
from onnx.runtime_spec import dbgnet_cache_names, get_runtime_profile


class StreamingDBGNet(nn.Module):
    """Frame-by-frame streaming DBGNet.

    Args:
        dbgnet: A trained/non-trained ``DBGNet`` instance. The streaming wrapper
            reuses its weights and modules; no parameter copy is performed.
    """

    def __init__(
        self,
        dbgnet: DBGNet,
        *,
        window: str = "hann",
        center: bool | None = None,
        deployment_config: dict | None = None,
    ):
        super().__init__()
        self.dbgnet = dbgnet
        self.use_state_detector = bool(dbgnet.use_state_detector)
        self.state_nclass = int(dbgnet.state_nclass)
        self.n_fft = int(dbgnet.n_fft)
        self.hop_length = int(dbgnet.hop_length)
        self.win_length = int(dbgnet.win_length)
        self.nfreqs = self.n_fft // 2 + 1
        self.erb_low_bins = int(dbgnet.erb.low_bins)
        self.erb_bands = int(dbgnet.erb.erb_bands)
        self.target_fs = int(dbgnet.sample_rate)
        self.window = str(window)
        self.center = center
        self.deployment_config = deployment_config
        self.network_lookahead_frames = 0
        self.algorithmic_delay_frames = 0
        self.algorithmic_delay = 0

    @property
    def CACHE_KEYS(self) -> tuple[str, ...]:
        return dbgnet_cache_names(self.use_state_detector)

    def _validate_streaming_inputs(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
    ) -> None:
        """Eager 模式复用谱输入合同；导出由静态 ABI 与 runtime preflight 校验。"""
        self.dbgnet._validate_spectra_inputs(mic_spec, far_spec)
        # ONNX trace 中不得执行会固化的单帧 Python shape 判断；dtype/device 合同仍校验。
        if not torch.onnx.is_in_onnx_export() and mic_spec.shape[2] != 1:
            raise ValueError(
                "StreamingDBGNet requires exactly one frame, "
                f"got T={mic_spec.shape[2]}"
            )

    def _freq_sizes(self) -> tuple[int, int, int, int, int]:
        """Return encoder frequency sizes (f0..f4) from the underlying DBGNet model."""
        return tuple(self.dbgnet._freq_sizes)

    @property
    def _input_channels(self) -> int:
        """首层编码器输入通道数：``mag`` 为 1（幅度），``com`` 为 2（RI 实/虚）。"""
        return 1 if self.dbgnet.mask_domain == "mag" else 2

    @property
    def _mc(self) -> list[int]:
        """mic_channels from the actual DBGNet conv layers."""
        return [
            self.dbgnet.conv_mic1[0].out_channels,
            self.dbgnet.conv_mic2[0].out_channels,
            self.dbgnet.conv_mic3[0].out_channels,
            self.dbgnet.conv_mic4[0].out_channels,
        ]

    @property
    def _fc(self) -> list[int]:
        """far_channels from the actual DBGNet conv layers."""
        return [
            self.dbgnet.conv_far1[0].out_channels,
            self.dbgnet.conv_far2[0].out_channels,
            self.dbgnet.conv_far3[0].out_channels,
            self.dbgnet.conv_far4[0].out_channels,
        ]

    def _conv_step(
        self,
        conv_module: nn.Module,
        x: torch.Tensor,
        x_prev: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply one causal encoder conv step using the previous frame as history.

        Args:
            conv_module: DBGNet encoder block ``conv_mic*`` or ``conv_far*``.
            x: Current-frame input ``(B, C, F, 1)``.
            x_prev: Previous-frame input of the same shape, or ``None`` for zeros.

        Returns:
            Current-frame output ``(B, C_out, F_out, 1)``.
        """
        if x_prev is None:
            x_prev = torch.zeros_like(x)
        inp = torch.cat([x_prev, x], dim=-1)  # (B, C, F, 2)
        return conv_module(inp)

    def _ftg_step(
        self,
        ftg: nn.Module,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One streaming step of a ``_FTGBlock``.

        Args:
            ftg: The far-end tracking GRU block.
            x: Current-frame feature ``(B, C, F, 1)``.
            h: GRU hidden state ``(1, B*C, F)``.

        Returns:
            (y, h_new) where y has the same shape as x and h_new matches h.
        """
        b, c, f, t = x.shape
        y = x.flatten(0, 1).permute(0, 2, 1)  # (B*C, 1, F)
        y, h_new = ftg.gru(y, h)
        y = y.permute(0, 2, 1).view(b, c, f, t)
        return y + x, h_new

    def init_cache(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Return a zero-initialized cache dictionary for streaming inference.

        All channel / hidden sizes are derived from the actual DBGNet model
        instance, so this works for base, small, custom channel configs and both
        mask domains (``mic_in`` / ``far_in`` follow ``mask_domain``).
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        f0, f1, f2, f3, f4 = self._freq_sizes()
        mc = self._mc  # mic_channels
        fc = self._fc  # far_channels
        tgru_hidden = self.dbgnet.tgru.hidden_size  # = mic_channels[-1]
        cache: dict[str, torch.Tensor] = {}

        # 编码器卷积历史：每层缓存前一帧输入。首层缓存 ERB 输入，
        # gy1_in..gy3_in 与 gx1_in..gx3_in 缓存下一层消费的 gy/gx guidance。
        cache["mic_in"] = torch.zeros(
            batch_size, self._input_channels, f0, 1, device=device, dtype=dtype
        )
        cache["far_in"] = torch.zeros(
            batch_size, self._input_channels, f0, 1, device=device, dtype=dtype
        )
        cache["gy1_in"] = torch.zeros(batch_size, mc[0], f1, 1, device=device, dtype=dtype)
        cache["gy2_in"] = torch.zeros(batch_size, mc[1], f2, 1, device=device, dtype=dtype)
        cache["gy3_in"] = torch.zeros(batch_size, mc[2], f3, 1, device=device, dtype=dtype)
        cache["gx1_in"] = torch.zeros(batch_size, fc[0], f1, 1, device=device, dtype=dtype)
        cache["gx2_in"] = torch.zeros(batch_size, fc[1], f2, 1, device=device, dtype=dtype)
        cache["gx3_in"] = torch.zeros(batch_size, fc[2], f3, 1, device=device, dtype=dtype)

        # FTG GRU hidden states: (1, B * far_channels[i], freq_dim_at_level_i).
        cache["ftg_h1"] = torch.zeros(1, batch_size * fc[0], f1, device=device, dtype=dtype)
        cache["ftg_h2"] = torch.zeros(1, batch_size * fc[1], f2, device=device, dtype=dtype)
        cache["ftg_h3"] = torch.zeros(1, batch_size * fc[2], f3, device=device, dtype=dtype)
        cache["ftg_h4"] = torch.zeros(1, batch_size * fc[3], f4, device=device, dtype=dtype)

        # Bottleneck time GRU hidden state.
        cache["tgru_h"] = torch.zeros(
            1, batch_size * f4, tgru_hidden, device=device, dtype=dtype
        )

        # Decoder ConvTranspose2d spillover caches.
        # Each deconv has time kernel=2, stride=1, so the kernel[1] contribution
        # from the current frame must be added to the NEXT frame's output.
        # Shapes match the deconv output dimensions at each decoder level.
        cache["d4_spill"] = torch.zeros(batch_size, mc[2], f3, 1, device=device, dtype=dtype)
        cache["d3_spill"] = torch.zeros(batch_size, mc[1], f2, 1, device=device, dtype=dtype)
        cache["d2_spill"] = torch.zeros(batch_size, mc[0], f1, 1, device=device, dtype=dtype)
        cache["d1_spill"] = torch.zeros(batch_size, 1, f1, 1, device=device, dtype=dtype)

        if self.use_state_detector:
            state_hidden = self.dbgnet.state_head.detector.rnn.hidden_size
            cache["state_h"] = torch.zeros(
                1, batch_size, state_hidden, device=device, dtype=dtype
            )

        return cache

    def forward_streaming(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Process one STFT frame (dict-based API for Python streaming loops).

        Args:
            mic_spec: Microphone complex spectrogram ``(B, F, 1, 2)``.
            far_spec: Far-end complex spectrogram ``(B, F, 1, 2)``.
            cache: Mutable cache dictionary from ``init_cache``.

        Returns:
            (enhanced_spec, state_logits_or_None, updated_cache).
        """
        keys = self.CACHE_KEYS
        if not isinstance(cache, dict) or tuple(cache) != keys:
            got = tuple(cache) if isinstance(cache, dict) else type(cache).__name__
            raise ValueError(
                f"DBGNet cache schema mismatch: expected={keys}, got={got}"
            )
        inputs = [cache[k] for k in keys]
        outputs = self._forward_streaming_tuple(mic_spec, far_spec, *inputs)
        cache_offset = 2 if self.use_state_detector else 1
        new_cache = {k: outputs[i + cache_offset]
                     for i, k in enumerate(keys)}
        enhanced = outputs[0]
        self.dbgnet._validate_enhanced_output(enhanced)
        ps = outputs[1] if self.use_state_detector else None
        return enhanced, ps, new_cache

    def _forward_streaming_tuple(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        mic_in: torch.Tensor,
        far_in: torch.Tensor,
        gy1_in: torch.Tensor,
        gy2_in: torch.Tensor,
        gy3_in: torch.Tensor,
        gx1_in: torch.Tensor,
        gx2_in: torch.Tensor,
        gx3_in: torch.Tensor,
        ftg_h1: torch.Tensor,
        ftg_h2: torch.Tensor,
        ftg_h3: torch.Tensor,
        ftg_h4: torch.Tensor,
        tgru_h: torch.Tensor,
        d4_spill: torch.Tensor,
        d3_spill: torch.Tensor,
        d2_spill: torch.Tensor,
        d1_spill: torch.Tensor,
        state_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Tuple-based streaming step used by the ONNX-exportable flat wrapper.

        This is a copy of ``forward_streaming`` that avoids Python dict mutation so
        that ``torch.onnx.export`` traces a clean data-flow graph.
        """
        self._validate_streaming_inputs(mic_spec, far_spec)
        b = mic_spec.shape[0]

        if self.dbgnet.mask_domain == "mag":
            x_mic = self.dbgnet.erb.bm(self.dbgnet._magnitude(mic_spec)).unsqueeze(1)
            x_far = self.dbgnet.erb.bm(self.dbgnet._magnitude(far_spec)).unsqueeze(1)
        else:
            # DBG-Com：单帧 RI 实/虚分量分别 band-merge，得到 (B, 2, F_erb, 1)。
            x_mic = self.dbgnet._erb_bm_ri(mic_spec)
            x_far = self.dbgnet._erb_bm_ri(far_spec)
        x_far = self.dbgnet.fpb(x_far)

        y1 = self._conv_step(self.dbgnet.conv_mic1, x_mic, mic_in)
        x1 = self._conv_step(self.dbgnet.conv_far1, x_far, far_in)
        x1, ftg_h1 = self._ftg_step(self.dbgnet.ftg1, x1, ftg_h1)
        gy1, gx1, mg1 = self.dbgnet.slm1(y1, x1)

        y2 = self._conv_step(self.dbgnet.conv_mic2, gy1, gy1_in)
        x2 = self._conv_step(self.dbgnet.conv_far2, gx1, gx1_in)
        x2, ftg_h2 = self._ftg_step(self.dbgnet.ftg2, x2, ftg_h2)
        gy2, gx2, mg2 = self.dbgnet.slm2(y2, x2)

        y3 = self._conv_step(self.dbgnet.conv_mic3, gy2, gy2_in)
        x3 = self._conv_step(self.dbgnet.conv_far3, gx2, gx2_in)
        x3, ftg_h3 = self._ftg_step(self.dbgnet.ftg3, x3, ftg_h3)
        gy3, gx3, mg3 = self.dbgnet.slm3(y3, x3)

        y4 = self._conv_step(self.dbgnet.conv_mic4, gy3, gy3_in)
        x4 = self._conv_step(self.dbgnet.conv_far4, gx3, gx3_in)
        x4, ftg_h4 = self._ftg_step(self.dbgnet.ftg4, x4, ftg_h4)
        gy4, gx4, mg4 = self.dbgnet.slm4(y4, x4)

        state_feats = []
        if self.use_state_detector:
            # Use the refactored AECStateDetectorHead feature extractors directly.
            fe = self.dbgnet.state_head.feature_extractors
            state_feats.append(fe[0](mg1, gx1))
            state_feats.append(fe[1](mg2, gx2))
            state_feats.append(fe[2](mg3, gx3))
            state_feats.append(fe[3](mg4, gx4))

        y = torch.cat([gy4, gx4], dim=1)
        y = y.permute(0, 3, 2, 1).flatten(0, 1)
        y, _ = self.dbgnet.fgru(y)
        y = y.view(b, -1, *y.shape[1:])
        y = y.permute(0, 2, 1, 3).flatten(0, 1)
        y, tgru_h = self.dbgnet.tgru(y, tgru_h)
        y = y.view(b, -1, *y.shape[1:]).permute(0, 3, 1, 2)

        # 单帧 ConvTranspose2d 的 spillover 含 bias；必须剥离它以避免相邻帧重复计数。
        def _deconv_step(deconv_block, x, skip, spill):
            # x: (B, C_in, F_in, 1), skip: (B, C_skip, F_in, 1)
            # spill: (B, C_out, F_out, 1) — raw CT spillover (bias-free)
            m_enc = deconv_block.conv(torch.cat([skip, x], dim=1))
            skip_gated = m_enc * skip
            # Raw ConvTranspose2d (index 0 of deconv Sequential)
            ct_raw = deconv_block.deconv[0](torch.cat([skip_gated, x], dim=1))  # (B,C,F_out,2)
            ct_bias = deconv_block.deconv[0].bias.view(1, -1, 1, 1)  # (1, C_out, 1, 1)
            # Add spillover from previous frame to the first output frame
            ct_raw[:, :, :, :1] = ct_raw[:, :, :, :1] + spill
            # Spillover for next frame: second output frame MINUS the bias
            # (so bias isn't double-counted when added to next frame's first output)
            spill_new = ct_raw[:, :, :, 1:] - ct_bias
            # Apply BN + PReLU (indices 1 and 2 of deconv Sequential)
            out = deconv_block.deconv[1](ct_raw)  # BN
            out = deconv_block.deconv[2](out)      # PReLU
            # Keep only the first (corrected) time frame
            return out[:, :, :, :1], spill_new

        y, d4_spill = _deconv_step(self.dbgnet.deconv4, y, gy4, d4_spill)
        y, d3_spill = _deconv_step(self.dbgnet.deconv3, y, gy3, d3_spill)
        y, d2_spill = _deconv_step(self.dbgnet.deconv2, y, gy2, d2_spill)
        y, d1_spill = _deconv_step(self.dbgnet.deconv1, y, gy1, d1_spill)
        y = self.dbgnet.out_conv(y)

        if self.dbgnet.mask_domain == "mag":
            m = torch.sigmoid(self.dbgnet.erb.bs(y.squeeze(1)))
            enhanced = mic_spec * m[..., None]
        else:
            mask = torch.tanh(self.dbgnet._erb_bs_ri(y))
            enhanced = _apply_complex_mask(mic_spec, mask)

        if self.use_state_detector:
            z = torch.cat(state_feats, dim=-1)
            z, state_h = self.dbgnet.state_head.detector.rnn(z, state_h)
            ps = self.dbgnet.state_head.detector.linear(z)
            return (enhanced, ps, x_mic, x_far, gy1, gy2, gy3, gx1, gx2, gx3,
                    ftg_h1, ftg_h2, ftg_h3, ftg_h4, tgru_h,
                    d4_spill, d3_spill, d2_spill, d1_spill,
                    state_h)

        return (enhanced, x_mic, x_far, gy1, gy2, gy3, gx1, gx2, gx3,
                ftg_h1, ftg_h2, ftg_h3, ftg_h4, tgru_h,
                d4_spill, d3_spill, d2_spill, d1_spill)


class StreamingDBGNetFrontendAdapter(nn.Module):
    """将 DBGNet 三返回流式 API 适配为通用 AEC frontend 两返回 API。"""

    def __init__(self, inner: StreamingDBGNet):
        super().__init__()
        self.inner = inner
        for name in (
            "n_fft", "hop_length", "win_length", "target_fs", "window", "center",
            "use_state_detector", "state_nclass", "network_lookahead_frames",
            "algorithmic_delay_frames", "algorithmic_delay", "deployment_config",
        ):
            setattr(self, name, getattr(inner, name))

    @property
    def CACHE_KEYS(self) -> tuple[str, ...]:
        return self.inner.CACHE_KEYS

    def init_cache(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return self.inner.init_cache(batch_size, device, dtype)

    def forward(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        cache: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        enhanced, _state_logits, updated_cache = self.inner.forward_streaming(
            mic_spec, far_spec, cache
        )
        return enhanced, updated_cache


class StreamingDBGNetFlat(nn.Module):
    """Flattened batch-one ONNX deployment wrapper around ``StreamingDBGNet``.

    Cache tensors are passed as individual input/output arguments so that
    ``torch.onnx.export`` sees them as explicit ONNX inputs/outputs. Formal
    DBGNet ONNX artifacts intentionally expose only a static batch size of 1;
    the underlying Python wrapper keeps its existing batched cache math.
    """

    def __init__(self, inner: StreamingDBGNet):
        super().__init__()
        self.inner = inner
        for name in (
            "n_fft", "hop_length", "win_length", "nfreqs", "target_fs",
            "window", "center", "use_state_detector", "state_nclass",
            "network_lookahead_frames", "algorithmic_delay_frames",
            "algorithmic_delay", "deployment_config",
        ):
            setattr(self, name, getattr(inner, name))

    @property
    def CACHE_KEYS(self) -> tuple[str, ...]:
        return self.inner.CACHE_KEYS

    @property
    def _cache_keys(self) -> list[str]:
        return list(self.CACHE_KEYS)

    def init_cache(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        required_batch = get_runtime_profile("dbgnet").static_batch_size
        if batch_size != required_batch:
            raise ValueError(
                "StreamingDBGNetFlat formal ONNX deployment only supports "
                f"batch_size={required_batch}"
            )
        cache = self.inner.init_cache(batch_size, device, dtype)
        return tuple(cache[k] for k in self._cache_keys)

    def get_input_names(self) -> list[str]:
        return ["mic_spec", "far_spec"] + self._cache_keys

    def get_output_names(self) -> list[str]:
        names = ["enh"]
        if self.use_state_detector:
            names.append("state_probs")
        return names + [f"{k}_out" for k in self._cache_keys]

    def get_cache_config(self) -> dict[str, list[int]]:
        """返回 metadata 使用的权威 batch-one cache ABI。"""
        cache = self.init_cache(1, torch.device("cpu"), torch.float32)
        return {name: list(value.shape) for name, value in zip(self._cache_keys, cache)}

    @staticmethod
    def load_cache_cfg_from_onnx(onnx_path: str) -> dict[str, list[int]]:
        """按版本化 runtime metadata 声明顺序严格读取 DBGNet cache ABI。"""
        import json

        from onnx.runtime_spec import DBGNET_RUNTIME_METADATA_KEY, DBGNetRuntimeSpec
        from onnx.utils import parse_onnx_cache_config, read_onnx_metadata

        metadata = read_onnx_metadata(onnx_path)
        raw_runtime = metadata.get(DBGNET_RUNTIME_METADATA_KEY)
        if raw_runtime is None:
            raise RuntimeError(
                f"ONNX metadata is missing required {DBGNET_RUNTIME_METADATA_KEY!r}; "
                "legacy DBGNet artifacts must be re-exported"
            )
        try:
            runtime = DBGNetRuntimeSpec.from_metadata(json.loads(raw_runtime))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"ONNX metadata {DBGNET_RUNTIME_METADATA_KEY!r} is not valid JSON"
            ) from exc
        parsed = parse_onnx_cache_config(metadata, "dbgnet_cache_cfg")
        if set(parsed) != set(runtime.cache_names):
            raise RuntimeError(
                "DBGNet cache metadata names do not match versioned cache_names: "
                f"cache_cfg={tuple(parsed)}, cache_names={runtime.cache_names}"
            )
        return {name: parsed[name] for name in runtime.cache_names}

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """正式 DBGNet ONNX ABI 不声明未经验证的动态 batch 轴。"""
        return {}

    def _normalize_caches(
        self, caches: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """校验 cache 必须按 ``CACHE_KEYS`` 顺序展开为独立 tensor 参数。"""
        if len(caches) != len(self.CACHE_KEYS):
            raise ValueError(
                f"DBGNet flat cache count mismatch: expected {len(self.CACHE_KEYS)}, "
                f"got {len(caches)}"
            )
        return caches

    def forward(
        self,
        mic_spec: torch.Tensor,
        far_spec: torch.Tensor,
        *caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """ONNX-exportable forward.

        Uses the tuple-based streaming step to avoid dict mutation in the traced
        graph.
        """
        required_batch = get_runtime_profile("dbgnet").static_batch_size
        if (
            not torch.onnx.is_in_onnx_export()
            and mic_spec.shape[0] != required_batch
        ):
            raise ValueError(
                "StreamingDBGNetFlat formal ONNX deployment only supports "
                f"batch_size={required_batch}"
            )
        caches = self._normalize_caches(caches)
        outputs = self.inner._forward_streaming_tuple(
            mic_spec, far_spec, *caches
        )
        self.inner.dbgnet._validate_enhanced_output(outputs[0])
        return outputs
