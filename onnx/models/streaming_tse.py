"""TSE_BSRNN_SPK causal streaming 的稳定 flat tensor ABI。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from tasks.tse.model_io import TSEStreamingState


class StreamingTSEFlat(nn.Module):
    """将 ``TSEStreamingState`` 展平为 ONNX/TorchScript 可消费的 tensor 列表。

    enrollment condition 在一个 session 内作为 loopback cache 原样携带；只有
    ``reset``/重新初始化 state 才能替换目标说话人。speaker encoder/context
    condition 在 session 初始化时计算一次，不进入逐帧 separator 图。
    """

    CACHE_ABI_VERSION = 1
    CACHE_SEMANTICS = "fixed-enrollment-loopback-v1"

    def __init__(self, model: nn.Module, *, enrollment_samples: int = 48000) -> None:
        super().__init__()
        if not getattr(model, "causal", False):
            raise ValueError("StreamingTSEFlat requires a causal TSE model")
        if getattr(model, "features_config", {}).get("listen", {}).get("enabled", False):
            raise ValueError(
                "StreamingTSEFlat v1 does not support listen cue warm-up; "
                "use spkemb or tfmap+context causal variants"
            )
        tfmap = getattr(model, "features_config", {}).get("tfmap", {})
        if tfmap.get("enabled", False) and tfmap.get("type", "spec") != "spec":
            raise ValueError(
                "StreamingTSEFlat v1 supports only tfmap.type='spec'; "
                "embedding TF-map requires mixture-waveform frame encoding"
            )
        if isinstance(enrollment_samples, bool) or not isinstance(enrollment_samples, int):
            raise TypeError("enrollment_samples must be an integer")
        if enrollment_samples <= model.n_fft // 2:
            raise ValueError("enrollment_samples must exceed n_fft//2")
        self.model = model
        self.n_fft = int(model.n_fft)
        self.hop_length = int(model.hop_length)
        self.win_length = int(model.n_fft)
        self.target_fs = int(model.sample_rate)
        self.window = "hann"
        self.center = True
        self.enrollment_samples = enrollment_samples
        self.nband = int(model.sep_model.nband)
        self.separator_layers = len(model.sep_model.separator.separation)
        self.separator_hidden_size = int(model.sep_model.separator.feature_dim * 2)
        self.condition_names = self._condition_names()
        temporal = []
        for index in range(self.separator_layers):
            temporal.extend((f"separator_{index:02d}_h", f"separator_{index:02d}_c"))
        if model.features_config["usef"]["enabled"]:
            temporal.append("usef_history")
        self.temporal_cache_names = tuple(temporal)
        self.CACHE_KEYS = self.condition_names + self.temporal_cache_names

    def _condition_names(self) -> tuple[str, ...]:
        features = self.model.features_config
        names = []
        if any(features[name]["enabled"] for name in ("usef", "tfmap")):
            names.append("enroll_spectra")
        if features["context"]["enabled"]:
            names.append("context")
        if features["spkemb"]["enabled"]:
            names.append("spkemb")
        return tuple(names)

    def _dummy_enrollment(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype
    ) -> dict[str, Tensor | int]:
        values: dict[str, Tensor | int] = {}
        features = self.model.features_config
        if any(features[name]["enabled"] for name in ("usef", "tfmap")):
            frames = self.enrollment_samples // self.hop_length + 1
            values["enroll_spectra"] = torch.zeros(
                batch_size, self.n_fft // 2 + 1, frames, 2,
                device=device, dtype=dtype,
            )
        if (
            self.model.joint_training
            and any(features[name]["enabled"] for name in ("context", "spkemb"))
        ):
            values["enroll_wav"] = torch.zeros(
                batch_size, self.enrollment_samples, device=device, dtype=dtype
            )
            values["enroll_lengths"] = torch.full(
                (batch_size,),
                self.enrollment_samples,
                device=device,
                dtype=torch.int64,
            )
        if not self.model.joint_training and features["spkemb"]["enabled"]:
            embed_dim = int(self.model.features_config["spkemb"]["embed_dim"])
            values["spk_embedding"] = torch.zeros(
                batch_size, embed_dim, device=device, dtype=dtype
            )
        if not values:
            values["batch_size"] = batch_size
        return values

    def _flatten_state(
        self,
        state: TSEStreamingState,
    ) -> list[Tensor]:
        values = [state.enrollment[name] for name in self.condition_names]
        for hidden, cell in state.separator_states:
            values.extend((hidden, cell))
        if "usef_history" in self.temporal_cache_names:
            if state.usef_history is None:
                raise RuntimeError("usef_history is missing from TSE streaming state")
            values.append(state.usef_history)
        if len(values) != len(self.CACHE_KEYS):
            raise RuntimeError("TSE flat cache count drifted from CACHE_KEYS")
        return values

    def init_cache_flat(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        enrollment = self._dummy_enrollment(batch_size, device, dtype)
        with torch.no_grad():
            state = self.model.init_streaming_state(enrollment)
        return self._flatten_state(state)

    def init_cache_flat_from_enrollment(
        self, enrollment: Mapping[str, Tensor | int]
    ) -> list[Tensor]:
        if not isinstance(enrollment, Mapping):
            raise TypeError("enrollment must be a mapping")
        with torch.no_grad():
            state = self.model.init_streaming_state(enrollment)
        return self._flatten_state(state)

    def _state_from_flat(self, caches: tuple[Tensor, ...]) -> TSEStreamingState:
        if len(caches) != len(self.CACHE_KEYS):
            raise ValueError(
                f"expected {len(self.CACHE_KEYS)} TSE cache tensors, got {len(caches)}"
            )
        values = dict(zip(self.CACHE_KEYS, caches))
        condition = {name: values[name] for name in self.condition_names}
        if condition:
            batch_size = next(iter(condition.values())).shape[0]
        else:
            batch_size = int(values["separator_00_h"].shape[1] // self.nband)
        separator = []
        for index in range(self.separator_layers):
            separator.append(
                (
                    values[f"separator_{index:02d}_h"],
                    values[f"separator_{index:02d}_c"],
                )
            )
        usef_history = values.get("usef_history")
        state = TSEStreamingState(
            batch_size=batch_size,
            enrollment=condition,
            separator_states=tuple(separator),
            usef_history=usef_history,
        )
        return state

    def forward(self, mix_frame: Tensor, *caches: Tensor) -> tuple[Tensor, ...]:
        state = self._state_from_flat(tuple(caches))
        condition = state.enrollment
        spec_ri = mix_frame.permute(0, 3, 1, 2).contiguous()
        next_history = state.usef_history
        if self.model.features_config["usef"]["enabled"]:
            enroll_ri = condition["enroll_spectra"].permute(0, 3, 1, 2).contiguous()
            enroll_usef, mix_usef, next_history = self.model._stream_usef(
                enroll_ri, spec_ri, state.usef_history
            )
            spec_ri = self.model.spk_ft.usef.post(mix_usef, enroll_usef)
        if self.model.features_config["tfmap"]["enabled"]:
            enroll = condition["enroll_spectra"]
            enroll_mag = torch.linalg.vector_norm(enroll, dim=-1)
            mix_mag = torch.linalg.vector_norm(mix_frame, dim=-1)
            tfmap = self.model.spk_ft.tfmap.compute(enroll_mag, mix_mag)
            spec_ri = self.model.spk_ft.tfmap.post(spec_ri, tfmap.unsqueeze(1))

        subband_mix = self.model.sep_model.band_split(
            mix_frame.permute(0, 3, 1, 2).contiguous()
        )
        subband = self.model.sep_model.band_split(spec_ri)
        features = self.model.sep_model.subband_norm(subband)
        if self.model.features_config["context"]["enabled"]:
            if not self.model.multi_fuse:
                features = self.model.spk_ft.context.post(features, condition["context"])
        if self.model.features_config["spkemb"]["enabled"]:
            embedding = condition["spkemb"].unsqueeze(1).unsqueeze(3)
            if not self.model.multi_fuse:
                features = self.model.spk_ft.spkemb.post(features, embedding)
        if self.model.multi_fuse:
            if self.model.multi_fuse_feature == "spkemb":
                fuse_condition = embedding
                condition_transform = None
            else:
                fuse_condition = condition["context"]
                condition_transform = self.model.spk_ft.context.attend
            separated, separator_states = (
                self.model.sep_model.separator.forward_stream_conditioned(
                    features,
                    state.separator_states,
                    self.model.multi_fuse_layers,
                    fuse_condition,
                    condition_transform,
                )
            )
        else:
            separated, separator_states = self.model.sep_model.separator.forward_stream(
                features, state.separator_states
            )
        real_bands = []
        imag_bands = []
        masker = self.model.sep_model.band_masker
        for index, mask_func in enumerate(masker.mask):
            mask_values = mask_func(separated[:, index])
            batch_size, _, frames = mask_values.shape
            band_width = masker.band_width[index]
            mask_values = mask_values.view(
                batch_size, masker.nspk, 2, 2, band_width, frames
            )
            complex_mask = mask_values[:, :, 0] * torch.sigmoid(
                mask_values[:, :, 1]
            )
            mask_real = complex_mask[:, :, 0]
            mask_imag = complex_mask[:, :, 1]
            mix_real = subband_mix[index][:, 0].unsqueeze(1)
            mix_imag = subband_mix[index][:, 1].unsqueeze(1)
            real_bands.append(mix_real * mask_real - mix_imag * mask_imag)
            imag_bands.append(mix_real * mask_imag + mix_imag * mask_real)
        estimated = torch.stack(
            (torch.cat(real_bands, dim=2), torch.cat(imag_bands, dim=2)), dim=-1
        )[:, 0]
        next_state = TSEStreamingState(
            batch_size=state.batch_size,
            enrollment=condition,
            separator_states=tuple(separator_states),
            usef_history=next_history,
        )
        flattened = self._flatten_state(next_state)
        return (estimated, *flattened)

    def get_input_names(self) -> list[str]:
        return ["mix_frame", *self.CACHE_KEYS]

    def get_output_names(self) -> list[str]:
        return ["enhanced_frame", *(f"{name}_out" for name in self.CACHE_KEYS)]

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes: dict[str, dict[int, str]] = {
            "mix_frame": {0: "batch"},
            "enhanced_frame": {0: "batch"},
        }
        for name in self.CACHE_KEYS:
            if name == "enroll_spectra":
                value = {0: "batch", 2: "enrollment_time"}
            elif name == "context":
                value = {0: "batch", 2: "enrollment_time"}
            elif name.startswith("separator_"):
                value = {1: "batch_flattened"}
            else:
                value = {0: "batch"}
            axes[name] = dict(value)
            axes[f"{name}_out"] = dict(value)
        return axes

    @property
    def waveform_output_delay_samples(self) -> int:
        lookahead = self.n_fft // 2
        return max(
            lookahead + 1,
            lookahead + (lookahead // self.hop_length) * self.hop_length,
        )
