"""RT-SEMamba 注册、蒸馏、严格权重与显式 streaming state 测试。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest
import torch
import torch.nn as nn

from core.utils.safe_load import safe_torch_load
from tasks.se.models import build_model

try:
    from tasks.se.models.discriminative.rt_semamba import RT_SEMamba, RT_SEMambaDistill
    from tasks.se.trainer.rt_semamba_distill_trainer import RTSEMambaDistillTrainer
    from tools.model_tools.convert_rt_semamba_checkpoint import convert_checkpoint
except ImportError as exc:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ModuleNotFoundError) and current.name:
            missing_root = current.name.split(".", 1)[0]
            if missing_root == "mamba_ssm":
                pytest.skip(
                    f"RT-SEMamba 可选依赖不可用: {exc}",
                    allow_module_level=True,
                )
        current = current.__cause__
    raise

pytestmark = pytest.mark.unit


class _FakeMixer(nn.Module):
    """用确定性递推模拟 Mamba forward/step/cache 契约。"""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        layer_idx: int,
    ):
        super().__init__()
        del d_state, d_conv, expand
        self.d_model = d_model
        self.scale = nn.Parameter(torch.full((d_model,), 0.5 + layer_idx * 0.01))

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del max_seqlen, kwargs
        dtype = dtype or self.scale.dtype
        conv = torch.zeros(batch_size, self.d_model, 1, dtype=dtype, device=device)
        ssm = torch.zeros(batch_size, self.d_model, 1, dtype=dtype, device=device)
        return conv, ssm

    def step(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous = ssm_state.transpose(1, 2)
        output = hidden_states * self.scale + previous * 0.125
        next_conv = hidden_states.transpose(1, 2)
        next_ssm = output.transpose(1, 2)
        return output, next_conv, next_ssm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        conv, ssm = self.allocate_inference_cache(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        outputs = []
        for index in range(hidden_states.shape[1]):
            output, conv, ssm = self.step(
                hidden_states[:, index : index + 1], conv, ssm
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1)


def _fake_factory(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    layer_idx: int,
) -> nn.Module:
    return _FakeMixer(d_model, d_state, d_conv, expand, layer_idx)


def _small_model(*, layers: int = 1) -> RT_SEMamba:
    model = RT_SEMamba(
        num_tfmamba=layers,
        hid_feature=8,
        d_state=4,
        d_conv=3,
        expand=2,
        n_fft=400,
        hop_length=100,
        win_length=400,
        window="hamming",
        center=False,
        mixer_factory=_fake_factory,
    )
    model.eval()
    return model


def _spectrum(batch: int = 1, frames: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234 + frames)
    return torch.randn(batch, 201, frames, 2, generator=generator) * 0.2


def _state_numel(value) -> int:
    if torch.is_tensor(value):
        return value.numel()
    if is_dataclass(value):
        return sum(_state_numel(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, (tuple, list)):
        return sum(_state_numel(item) for item in value)
    return 0


def test_registry_build_and_spectrum_forward():
    model = build_model(
        {
            "type": "RT_SEMamba",
            "args": {
                "num_tfmamba": 1,
                "hid_feature": 8,
                "d_state": 4,
                "d_conv": 3,
                "expand": 2,
                "mixer_factory": _fake_factory,
            },
        },
        stft_cfg={
            "n_fft": 400,
            "hop_length": 100,
            "win_length": 400,
            "window": "hamming",
            "center": False,
        },
    )
    model.eval()
    noisy = _spectrum(frames=3)
    enhanced = model(noisy, torch.tensor([300]), fs=16000)
    assert isinstance(model, RT_SEMamba)
    assert enhanced.shape == noisy.shape
    assert torch.isfinite(enhanced).all()


def test_constructor_and_input_contracts_fail_closed():
    with pytest.raises(TypeError, match="unexpected arguments"):
        RT_SEMamba(mixer_factory=_fake_factory, misspelled_option=True)
    with pytest.raises(ValueError, match="requires n_fft=400"):
        RT_SEMamba(n_fft=512, mixer_factory=_fake_factory)

    model = _small_model()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 200, 2, 2))
    with pytest.raises(ValueError, match="16000"):
        model(_spectrum(frames=2), fs=48000)


def test_streaming_frames_chunks_and_offline_are_equivalent():
    # 固定曾触发幅度压缩舍入差异的权重，同时不污染其他用例的 RNG。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _small_model()
    noisy = _spectrum(frames=6)
    with torch.no_grad():
        offline = model(noisy, fs=16000)

        initial = model.init_streaming_state(1, device="cpu")
        initial_size = _state_numel(initial)
        frame_outputs = []
        frame_state = initial
        for index in range(noisy.shape[2]):
            output, frame_state = model.forward_stream_frame(
                noisy[:, :, index : index + 1], frame_state, fs=16000
            )
            frame_outputs.append(output)
        frames = torch.cat(frame_outputs, dim=2)

        chunk_state = model.init_streaming_state(1, device="cpu")
        first, chunk_state = model.forward_stream_chunk(
            noisy[:, :, :2], chunk_state, fs=16000
        )
        second, chunk_state = model.forward_stream_chunk(
            noisy[:, :, 2:], chunk_state, fs=16000
        )
        chunks = torch.cat((first, second), dim=2)

    torch.testing.assert_close(frames, offline, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(chunks, offline, rtol=2e-5, atol=2e-5)
    # abs/pow 对不同长度的张量存在末位差异，再经相位头放大；三个
    # 端到端比较使用相同容差。下面复用同一份预处理结果，严格核对状态递推。
    torch.testing.assert_close(chunks, frames, rtol=2e-5, atol=2e-5)
    assert _state_numel(frame_state) == initial_size
    assert _state_numel(chunk_state) == initial_size

    from tasks.se.models.discriminative.rt_semamba import _compressed_components

    with torch.no_grad():
        magnitude, phase, _ = _compressed_components(noisy, model.compress_factor)
        state = model.init_streaming_state(1, device="cpu")
        core_frames = []
        for index in range(noisy.shape[2]):
            result = model.network.forward_stream_frame(
                magnitude[:, :, index : index + 1],
                phase[:, :, index : index + 1],
                state,
            )
            state = result[3]
            core_frames.append(result[:3])
        core_chunk = model.network.forward_stream_chunk(
            magnitude, phase, model.init_streaming_state(1, device="cpu")
        )
    for field_index in range(3):
        expected = torch.cat([frame[field_index] for frame in core_frames], dim=2)
        torch.testing.assert_close(core_chunk[field_index], expected, rtol=0, atol=0)


def _compressed_input(noisy_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 ``(B,F,T,2)`` 复谱转成压缩幅度/相位，语义与模型内部 ``_compressed_components`` 一致。"""
    complex_spec = torch.view_as_complex(noisy_spec.float().contiguous())
    magnitude = complex_spec.abs().clamp_min(torch.finfo(torch.float32).eps)
    return magnitude, torch.angle(complex_spec)


def _real_small_model(*, layers: int = 1) -> RT_SEMamba:
    """真实 mamba_ssm Mamba backend 的小模型（依赖 mamba_ssm，module-level guard 已降级）。"""
    model = RT_SEMamba(
        num_tfmamba=layers,
        hid_feature=8,
        d_state=4,
        d_conv=4,
        expand=2,
        n_fft=400,
        hop_length=100,
        win_length=400,
        window="hamming",
        center=False,
    )
    model.eval()
    return model


def test_real_mamba_batch_matches_streaming_frames_and_chunks():
    """真实 mamba_ssm 下，batch 训练路径与 streaming 逐帧/分块路径数值一致。

    训练 batch 路径走 ``network.forward``（整段逐 token scan）；eval/streaming
    路径走 ``forward_stream_frame``/``forward_stream_chunk``（手写
    ``_mamba_step_reference`` 逐 token 递推）。逐中间量核验显示 Mamba scan、
    DenseEncoder、freq_mamba 均数值一致（差异 <= 1e-7，conv 历史由零初始化 cache
    递推与 batch 左侧 pad 等价）；唯一的合法差异来自 GPU 上 cudnn 对
    ``ConvTranspose1d``（TFMambaBlock.flinear）按 batch 维度选择不同 kernel
    （CPU 上相同输入输出 0 差异，CUDA 上 batch=8 vs batch=1 达 ~6e-4），并经
    atan2 相位回路放大到 1e-2 量级。因此 mag/com 使用 1e-3 容差、phase 使用
    2e-2 容差；真正的语义错位（帧错位/cache 污染）会产生 O(1) 差异，不会被掩盖。

    本机安装的 mamba_ssm（1.2.2）selective_scan 为 CUDA-only 实现、无 CPU
    fallback，故仅在 CUDA 可用时执行，否则跳过。
    """
    if not torch.cuda.is_available():
        pytest.skip("真实 mamba_ssm 的 selective_scan 为 CUDA-only，本机无可用 CUDA")
    # 真实 Mamba 的随机权重会决定 atan2 相位是否跨越 ±π 分支；隔离并固定
    # CPU/CUDA 初始化 RNG，避免此前测试推进全局 RNG 后偶发出现 2π 相位跳变。
    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = _real_small_model().to("cuda")
    noisy = _spectrum(frames=8).to("cuda")
    magnitude, phase = _compressed_input(noisy)
    with torch.no_grad():
        batch_mag, batch_pha, batch_com = model.network(magnitude, phase)

        frame_state = model.init_streaming_state(1, device="cuda")
        frame_outputs = []
        for index in range(noisy.shape[2]):
            frame_mag, frame_pha, frame_com, frame_state = (
                model.network.forward_stream_frame(
                    magnitude[:, :, index : index + 1],
                    phase[:, :, index : index + 1],
                    frame_state,
                )
            )
            frame_outputs.append((frame_mag, frame_pha, frame_com))
        frame_mag = torch.cat([output[0] for output in frame_outputs], dim=-1)
        frame_pha = torch.cat([output[1] for output in frame_outputs], dim=-1)
        frame_com = torch.cat([output[2] for output in frame_outputs], dim=2)

        chunk_state = model.init_streaming_state(1, device="cuda")
        first_mag, first_pha, first_com, chunk_state = (
            model.network.forward_stream_chunk(
                magnitude[:, :, :4], phase[:, :, :4], chunk_state
            )
        )
        second_mag, second_pha, second_com, _ = model.network.forward_stream_chunk(
            magnitude[:, :, 4:], phase[:, :, 4:], chunk_state
        )
        chunk_mag = torch.cat((first_mag, second_mag), dim=-1)
        chunk_pha = torch.cat((first_pha, second_pha), dim=-1)
        chunk_com = torch.cat((first_com, second_com), dim=2)

    mag_tolerance = dict(rtol=1e-3, atol=1e-3)
    com_tolerance = dict(rtol=1e-2, atol=1e-2)
    for streaming_mag, streaming_pha, streaming_com in (
        (frame_mag, frame_pha, frame_com),
        (chunk_mag, chunk_pha, chunk_com),
    ):
        torch.testing.assert_close(streaming_mag, batch_mag, **mag_tolerance)
        # 相位由 atan2 输出，对底层幅度噪声敏感，cudnn batch 相关浮点差异被放大
        # 到 1e-2 量级（实测最大 ~1.2e-2），故单独放宽到 2e-2，仍远小于相位量级 2π。
        torch.testing.assert_close(
            streaming_pha, batch_pha, rtol=2e-2, atol=2e-2
        )
        # complex = mag * exp(j*phase)，相位误差随幅度放大（实测最大 ~5.4e-3），
        # 容差放宽到 1e-2；语义错位会产生 O(1) 差异，不会被掩盖。
        torch.testing.assert_close(streaming_com, batch_com, **com_tolerance)
    torch.testing.assert_close(chunk_com, frame_com, rtol=2e-5, atol=1e-6)
    assert _state_numel(frame_state) == _state_numel(model.init_streaming_state(1, device="cuda"))
    torch.testing.assert_close(chunk_com, frame_com, rtol=2e-5, atol=1e-6)
    assert _state_numel(frame_state) == _state_numel(model.init_streaming_state(1, device="cuda"))


def test_streaming_is_causal_and_independent_states_do_not_cross_talk():
    # 隔离随机初始化，避免极小权重使两个独立 state 的差异落入 allclose 容差。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _small_model()
    noisy = _spectrum(frames=5)
    changed_future = noisy.clone()
    changed_future[:, :, 3:] += 10.0
    with torch.no_grad():
        original = model(noisy)
        changed = model(changed_future)
    torch.testing.assert_close(original[:, :, :3], changed[:, :, :3], rtol=1e-5, atol=4e-6)

    zero = torch.zeros(1, 201, 1, 2)
    one = torch.ones(1, 201, 1, 2) * 0.1
    state_a = model.init_streaming_state(1, device="cpu")
    state_b = model.init_streaming_state(1, device="cpu")
    with torch.no_grad():
        _, state_a = model.forward_stream_frame(zero, state_a)
        _, state_b = model.forward_stream_frame(one, state_b)
        output_a, _ = model.forward_stream_frame(zero, state_a)
        output_b, _ = model.forward_stream_frame(zero, state_b)
    assert not torch.allclose(output_a, output_b)


def _teacher_checkpoint(path, *, malformed: bool = False) -> None:
    teacher = _small_model(layers=8)
    state = teacher.network.state_dict()
    if malformed:
        state.pop(next(iter(state)))
    torch.save({"generator": state}, path)


def test_progressive_kd_freezes_teacher_ramps_and_backpropagates(tmp_path):
    checkpoint = tmp_path / "teacher.pth"
    _teacher_checkpoint(checkpoint)
    wrapper = RT_SEMambaDistill(
        teacher_checkpoint=str(checkpoint),
        teacher_layers=8,
        student_layers=1,
        ramp_steps=10,
        hid_feature=8,
        d_state=4,
        d_conv=3,
        expand=2,
        n_fft=400,
        hop_length=100,
        win_length=400,
        window="hamming",
        center=False,
        mixer_factory=_fake_factory,
    )
    wrapper.train()
    assert not wrapper.teacher.training
    assert all(not parameter.requires_grad for parameter in wrapper.teacher.parameters())

    noisy = _spectrum(frames=3)
    clean = _spectrum(frames=3) * 0.8
    loss_zero, info_zero = wrapper(noisy, clean, fs=16000, step=0)
    loss_full, info_full = wrapper(noisy, clean, fs=16000, step=10)
    torch.testing.assert_close(loss_zero.detach(), info_zero["task"])
    assert info_zero["kd_scale"].item() == 0.0
    assert info_full["kd_scale"].item() == 1.0
    assert torch.isfinite(loss_full)
    assert loss_full >= loss_zero

    loss_full.backward()
    assert any(parameter.grad is not None for parameter in wrapper.student.parameters())
    assert all(parameter.grad is None for parameter in wrapper.teacher.parameters())


def test_progressive_kd_rejects_incompatible_teacher_checkpoint(tmp_path):
    checkpoint = tmp_path / "bad_teacher.pth"
    _teacher_checkpoint(checkpoint, malformed=True)
    with pytest.raises(RuntimeError, match="incompatible"):
        RT_SEMambaDistill(
            teacher_checkpoint=str(checkpoint),
            teacher_layers=8,
            student_layers=1,
            hid_feature=8,
            d_state=4,
            d_conv=3,
            expand=2,
            window="hamming",
            mixer_factory=_fake_factory,
        )


def test_checkpoint_converter_strict_round_trip_and_rejects_bad_state(tmp_path):
    model = RT_SEMamba(num_tfmamba=1, window="hann")
    source = tmp_path / "official.pth"
    output = tmp_path / "converted.pth"
    torch.save({"generator": model.network.state_dict()}, source)
    converted = convert_checkpoint(source, "KD1", output)
    reloaded = safe_torch_load(
        output,
        map_location="cpu",
        purpose="test RT-SEMamba converted model-only checkpoint",
    )
    assert set(converted) == set(model.network.state_dict())
    assert set(reloaded) == set(converted)

    bad_source = tmp_path / "bad_official.pth"
    bad_state = dict(model.network.state_dict())
    bad_state.pop(next(iter(bad_state)))
    torch.save({"generator": bad_state}, bad_source)
    with pytest.raises(RuntimeError, match="missing"):
        convert_checkpoint(bad_source, "KD1", None)


class _RecordingSTFT:
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        batch = waveform.shape[0]
        value = waveform.mean(dim=-1).view(batch, 1, 1, 1)
        return value.expand(batch, 201, 2, 2).clone()


class _RecordingDistillModel:
    def __init__(self):
        self.call = None

    def __call__(self, noisy, clean, lengths, **kwargs):
        self.call = (noisy, clean, lengths, kwargs)
        return noisy.mean() * 0.0 + 1.0, {"kd_scale": noisy.new_tensor(0.5)}


def test_distill_trainer_converts_waveforms_and_passes_optimizer_step():
    trainer = object.__new__(RTSEMambaDistillTrainer)
    trainer.stft_module = _RecordingSTFT()
    trainer.model = _RecordingDistillModel()
    trainer.n_steps = 17
    trainer._model_accepts_fs = True
    noisy = torch.randn(2, 500)
    clean = torch.randn(2, 500)
    lengths = torch.tensor([500, 450])

    loss, diagnostics = trainer._distill_forward(noisy, clean, lengths, 16000)
    recorded_noisy, recorded_clean, recorded_lengths, kwargs = trainer.model.call
    assert recorded_noisy.shape == (2, 201, 2, 2)
    assert recorded_clean.shape == (2, 201, 2, 2)
    assert recorded_lengths is lengths
    assert kwargs == {"step": 17, "fs": 16000}
    assert loss.ndim == 0
    assert diagnostics["kd_scale"].item() == 0.5
