"""DCCRN 使用的实张量复数神经网络层。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_even(value: int, name: str) -> None:
    if value <= 0 or value % 2:
        raise ValueError(f"{name} must be a positive even integer, got {value}")


def _split_complex(inputs: torch.Tensor, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.shape[axis] % 2:
        raise ValueError(
            f"complex axis size must be even, got shape[{axis}]={inputs.shape[axis]}"
        )
    return torch.chunk(inputs, 2, dim=axis)


def complex_cat(inputs: Sequence[torch.Tensor], axis: int = 1) -> torch.Tensor:
    """沿复数通道轴拼接，保持 ``[全部实部, 全部虚部]`` 布局。"""
    if not inputs:
        raise ValueError("complex_cat requires at least one tensor")
    real_parts, imag_parts = zip(*(_split_complex(tensor, axis) for tensor in inputs))
    return torch.cat((torch.cat(real_parts, dim=axis), torch.cat(imag_parts, dim=axis)), dim=axis)


class ComplexPReLU(nn.Module):
    """分别对实部和虚部应用 PReLU。"""

    def __init__(self, complex_axis: int = 1) -> None:
        super().__init__()
        self.real_prelu = nn.PReLU()
        self.imag_prelu = nn.PReLU()
        self.complex_axis = complex_axis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(inputs, self.complex_axis)
        return torch.cat(
            (self.real_prelu(real), self.imag_prelu(imag)), dim=self.complex_axis
        )


class ComplexConv2d(nn.Module):
    """用两个实卷积实现复卷积 ``(Wr+jWi)(xr+jxi)``。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        causal: bool = True,
        complex_axis: int = 1,
    ) -> None:
        super().__init__()
        _require_even(in_channels, "in_channels")
        _require_even(out_channels, "out_channels")
        self.padding = padding
        self.causal = causal
        self.complex_axis = complex_axis
        conv_args = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding[0], 0),
            dilation=dilation,
            groups=groups,
        )
        self.real_conv = nn.Conv2d(in_channels // 2, out_channels // 2, **conv_args)
        self.imag_conv = nn.Conv2d(in_channels // 2, out_channels // 2, **conv_args)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        if self.real_conv.bias is not None:
            nn.init.zeros_(self.real_conv.bias)
        if self.imag_conv.bias is not None:
            nn.init.zeros_(self.imag_conv.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        time_padding = self.padding[1]
        if time_padding:
            right = 0 if self.causal else time_padding
            inputs = F.pad(inputs, (time_padding, right, 0, 0))
        real, imag = _split_complex(inputs, self.complex_axis)
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.imag_conv(real) + self.real_conv(imag)
        return torch.cat((real_out, imag_out), dim=self.complex_axis)


class ComplexConvTranspose2d(nn.Module):
    """用两个实转置卷积实现复转置卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (1, 1),
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        output_padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        complex_axis: int = 1,
    ) -> None:
        super().__init__()
        _require_even(in_channels, "in_channels")
        _require_even(out_channels, "out_channels")
        self.complex_axis = complex_axis
        conv_args = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.real_conv = nn.ConvTranspose2d(
            in_channels // 2, out_channels // 2, **conv_args
        )
        self.imag_conv = nn.ConvTranspose2d(
            in_channels // 2, out_channels // 2, **conv_args
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        if self.real_conv.bias is not None:
            nn.init.zeros_(self.real_conv.bias)
        if self.imag_conv.bias is not None:
            nn.init.zeros_(self.imag_conv.bias)

    def forward(
        self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            real, imag = _split_complex(inputs, self.complex_axis)
        else:
            real, imag = inputs
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.imag_conv(real) + self.real_conv(imag)
        return torch.cat((real_out, imag_out), dim=self.complex_axis)


class NaiveComplexLSTM(nn.Module):
    """共享实/虚 LSTM 权重的朴素复 LSTM。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        projection_dim: int | None = None,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _require_even(input_size, "input_size")
        _require_even(hidden_size, "hidden_size")
        if projection_dim is not None:
            _require_even(projection_dim, "projection_dim")
        self.real_lstm = nn.LSTM(
            input_size // 2, hidden_size // 2, bidirectional=bidirectional
        )
        self.imag_lstm = nn.LSTM(
            input_size // 2, hidden_size // 2, bidirectional=bidirectional
        )
        directions = 2 if bidirectional else 1
        if projection_dim is None:
            self.real_projection = None
            self.imag_projection = None
        else:
            self.real_projection = nn.Linear(
                hidden_size // 2 * directions, projection_dim // 2
            )
            self.imag_projection = nn.Linear(
                hidden_size // 2 * directions, projection_dim // 2
            )

    def forward(
        self, inputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            real, imag = _split_complex(inputs, -1)
        else:
            real, imag = inputs
        real_to_real = self.real_lstm(real)[0]
        real_to_imag = self.imag_lstm(real)[0]
        imag_to_real = self.real_lstm(imag)[0]
        imag_to_imag = self.imag_lstm(imag)[0]
        real_out = real_to_real - imag_to_imag
        imag_out = imag_to_real + real_to_imag
        if self.real_projection is not None and self.imag_projection is not None:
            real_out = self.real_projection(real_out)
            imag_out = self.imag_projection(imag_out)
        return real_out, imag_out

    def flatten_parameters(self) -> None:
        self.real_lstm.flatten_parameters()
        self.imag_lstm.flatten_parameters()


class ComplexBatchNorm(nn.Module):
    """对每个复通道执行二维协方差白化和可学习仿射变换。"""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        complex_axis: int = 1,
    ) -> None:
        super().__init__()
        _require_even(num_features, "num_features")
        if complex_axis != 1:
            raise ValueError("ComplexBatchNorm currently requires complex_axis=1")
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.complex_axis = complex_axis
        if affine:
            self.weight_rr = nn.Parameter(torch.ones(self.num_features))
            self.weight_ri = nn.Parameter(torch.zeros(self.num_features))
            self.weight_ii = nn.Parameter(torch.ones(self.num_features))
            self.bias_real = nn.Parameter(torch.zeros(self.num_features))
            self.bias_imag = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("weight_rr", None)
            self.register_parameter("weight_ri", None)
            self.register_parameter("weight_ii", None)
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        if track_running_stats:
            self.register_buffer("running_mean_real", torch.zeros(self.num_features))
            self.register_buffer("running_mean_imag", torch.zeros(self.num_features))
            self.register_buffer("running_var_rr", torch.ones(self.num_features))
            self.register_buffer("running_var_ri", torch.zeros(self.num_features))
            self.register_buffer("running_var_ii", torch.ones(self.num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean_real", None)
            self.register_buffer("running_mean_imag", None)
            self.register_buffer("running_var_rr", None)
            self.register_buffer("running_var_ri", None)
            self.register_buffer("running_var_ii", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(inputs, self.complex_axis)
        reduce_dims = tuple(index for index in range(real.ndim) if index != 1)
        shape = [1] * real.ndim
        shape[1] = self.num_features
        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            mean_real = real.mean(dim=reduce_dims, keepdim=True)
            mean_imag = imag.mean(dim=reduce_dims, keepdim=True)
        else:
            mean_real = self.running_mean_real.view(shape)
            mean_imag = self.running_mean_imag.view(shape)
        centered_real = real - mean_real
        centered_imag = imag - mean_imag
        if use_batch_stats:
            var_rr = centered_real.square().mean(dim=reduce_dims, keepdim=True)
            var_ri = (centered_real * centered_imag).mean(dim=reduce_dims, keepdim=True)
            var_ii = centered_imag.square().mean(dim=reduce_dims, keepdim=True)
            if self.training and self.track_running_stats:
                self.num_batches_tracked.add_(1)
                factor = (
                    1.0 / float(self.num_batches_tracked)
                    if self.momentum is None
                    else self.momentum
                )
                self.running_mean_real.lerp_(mean_real.reshape(-1), factor)
                self.running_mean_imag.lerp_(mean_imag.reshape(-1), factor)
                self.running_var_rr.lerp_(var_rr.reshape(-1), factor)
                self.running_var_ri.lerp_(var_ri.reshape(-1), factor)
                self.running_var_ii.lerp_(var_ii.reshape(-1), factor)
        else:
            var_rr = self.running_var_rr.view(shape)
            var_ri = self.running_var_ri.view(shape)
            var_ii = self.running_var_ii.view(shape)
        var_rr = var_rr + self.eps
        var_ii = var_ii + self.eps
        determinant_sqrt = torch.sqrt(torch.clamp(var_rr * var_ii - var_ri.square(), min=self.eps))
        trace_root = torch.sqrt(torch.clamp(var_rr + var_ii + 2 * determinant_sqrt, min=self.eps))
        inverse_scale = (determinant_sqrt * trace_root).reciprocal()
        whiten_rr = (determinant_sqrt + var_ii) * inverse_scale
        whiten_ii = (determinant_sqrt + var_rr) * inverse_scale
        whiten_ri = -var_ri * inverse_scale
        if self.affine:
            weight_rr = self.weight_rr.view(shape)
            weight_ri = self.weight_ri.view(shape)
            weight_ii = self.weight_ii.view(shape)
            transform_rr = weight_rr * whiten_rr + weight_ri * whiten_ri
            transform_ri = weight_rr * whiten_ri + weight_ri * whiten_ii
            transform_ir = weight_ri * whiten_rr + weight_ii * whiten_ri
            transform_ii = weight_ri * whiten_ri + weight_ii * whiten_ii
        else:
            transform_rr, transform_ri = whiten_rr, whiten_ri
            transform_ir, transform_ii = whiten_ri, whiten_ii
        output_real = transform_rr * centered_real + transform_ri * centered_imag
        output_imag = transform_ir * centered_real + transform_ii * centered_imag
        if self.affine:
            output_real = output_real + self.bias_real.view(shape)
            output_imag = output_imag + self.bias_imag.view(shape)
        return torch.cat((output_real, output_imag), dim=self.complex_axis)
