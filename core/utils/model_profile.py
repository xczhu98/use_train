"""轻量模型复杂度与延迟分析工具。

``profile_model`` 保留历史顶层返回键，并额外返回统一 ``complexity`` 合同：
``Params`` 明确 total/trainable/scope/source，``MACs`` 明确
value/backend/input/scope/coverage。FLOPs 不从 MACs 推算；没有独立可靠测量时固定写
``not measured``。MOS 不属于在线 profile，只能通过 ``evaluate/eval_pipeline_*.py``
离线评估。

可选依赖：
  pip install thop        # MACs 首选后端
  pip install torchinfo   # MACs 回退后端

用法：
  from core.utils.model_profile import profile_model
  profile_model(model, sample_rate=16000, duration=1.0)
"""

from __future__ import annotations

import argparse
import math
import time
import torch
import torch.nn as nn


NOT_MEASURED = "not measured"
MOS_OFFLINE_BOUNDARY = "offline only: evaluate/eval_pipeline_*.py"


# ─────────────────────────────────────────────
#  Internal utilities
# ─────────────────────────────────────────────

def _count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_complexity_contract(
    *,
    total_params: int,
    trainable_params: int,
    params_scope: str,
    params_source: str,
    macs: float | None,
    macs_backend: str,
    macs_input: dict,
    macs_scope: str,
    macs_coverage: str,
) -> dict:
    """构造共享复杂度头合同；FLOPs 与 MOS 边界不可由调用方省略。"""
    return {
        "Params": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "scope": str(params_scope),
            "source": str(params_source),
        },
        "MACs": {
            "value": None if macs is None else float(macs),
            "backend": str(macs_backend),
            "input": dict(macs_input),
            "scope": str(macs_scope),
            "coverage": str(macs_coverage),
        },
        "FLOPs": {
            "value": NOT_MEASURED,
            "source": "independent measurement required; no MACs conversion",
        },
        "MOS": {
            "value": NOT_MEASURED,
            "boundary": MOS_OFFLINE_BOUNDARY,
        },
    }


def _fmt_num(n: float, unit: str = "") -> str:
    """Format number with K/M/G suffix string."""
    for suffix, div in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n / div:.3f} {suffix}{unit}"
    return f"{n:.0f} {unit}"


class _ForwardWrapper(nn.Module):
    """把选定前向及其关键字参数包装为可被 profiler 遍历的模块。"""

    def __init__(self, model: nn.Module, extra_kwargs: dict, forward_fn=None):
        super().__init__()
        self.model = model
        self.extra_kwargs = extra_kwargs
        self.forward_fn = forward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self.forward_fn or self.model
        return fn(x, **self.extra_kwargs)


def _validate_profile_parameters(
    sample_rate: int,
    duration: float,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> None:
    """在分配输入和运行模型前拒绝无意义的 profile 参数。"""
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(f"duration must be positive, got {duration!r}")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative integer, got {warmup!r}")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError(f"repeats must be a positive integer, got {repeats!r}")


def _synchronize_cuda(device: str | torch.device) -> None:
    """对 ``cuda`` 与 ``cuda:N`` 使用同一显式设备同步语义。"""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


class _DualInputWrapper(nn.Module):
    """Dual-input (mic + far) wrapper.

    Splits (B, 2, T) into mic=(B,T) + far=(B,T) and calls model.forward_wav();
    model is registered as a real nn.Module submodule so thop can recursively traverse it.
    """
    def __init__(self, model: nn.Module, method: str = "forward_wav"):
        super().__init__()
        self.inner = model   # registered as submodule so thop can recurse
        self._method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mic, far = x[:, 0], x[:, 1]
        fn = getattr(self.inner, self._method)
        out = fn(mic, far)
        # forward_wav may return tuple (ne, ee), take first element
        return out[0] if isinstance(out, (tuple, list)) else out


# ─────────────────────────────────────────────
#  Main function
# ─────────────────────────────────────────────

def _profile_model(
    model: nn.Module,
    sample_rate: int = 16000,
    duration: float = 1.0,
    batch_size: int = 1,
    forward_kwargs: dict | None = None,
    forward_fn=None,
    dual_input: bool = False,
    device: str = "cpu",
    warmup: int = 3,
    repeats: int = 10,
    label: str = "",
    quiet: bool = False,
    params_scope: str = "model",
    params_source: str = "runtime model.parameters()",
    macs_scope: str = "profile forward path",
    profile_input: torch.Tensor | None = None,
    input_description: dict | None = None,
) -> dict:
    """统计模型 Params、MACs 与延迟，并返回统一复杂度合同。

    新参数均有默认值，历史调用与顶层返回键保持兼容。``profile_input`` 允许任务
    adapter 提供其真实张量输入，``input_description`` 记录命名输入。``MACs.coverage``
    必须披露后端局限；``FLOPs`` 不进行 MACs→FLOPs 换算，``MOS`` 仅记录离线边界。
    """
    _validate_profile_parameters(sample_rate, duration, batch_size, warmup, repeats)
    forward_kwargs = forward_kwargs or {}

    if not dual_input and "fs" not in forward_kwargs and hasattr(model, "target_fs"):
        if model.target_fs != sample_rate:
            forward_kwargs = {"fs": sample_rate, **forward_kwargs}

    model = model.to(device).eval()
    n_samples = int(sample_rate * duration)

    if profile_input is not None:
        if dual_input:
            raise ValueError("profile_input and dual_input cannot be combined")
        x = profile_input.to(device)
        _profile_target = _ForwardWrapper(model, forward_kwargs, forward_fn)
    elif dual_input:
        x = torch.zeros(batch_size, 2, n_samples, device=device)
        _profile_target = _DualInputWrapper(model)
    else:
        x = torch.zeros(batch_size, n_samples, device=device)
        _profile_target = _ForwardWrapper(model, forward_kwargs, forward_fn)
    _timing_fn = lambda: _profile_target(x)  # noqa: E731

    # ── Parameter count ──
    total_p, train_p = _count_params(model)

    # ── MACs (thop preferred, fallback torchinfo) ──
    macs: float | None = None
    macs_error: str | None = None
    macs_backend = NOT_MEASURED
    macs_coverage = NOT_MEASURED
    macs_str = "N/A"
    excluded_types: list[str] = []
    torchinfo_failure: list[str] = []

    # 以下模块由后端显式按零算子处理，必须在 coverage 中披露，不能冒充完整 MACs。
    _THOP_ZERO_OPS_CLASSES = (
        "Resample",
        "StftIstftModule",
        "Stft",
        "Istft",
        "ConvSTFT",
        "ConviSTFT",
    )

    def _try_torchinfo():
        try:
            from torchinfo import summary as ti_summary
            info = ti_summary(
                _profile_target,
                input_data=x,
                device=device,
                verbose=0,
            )
            return float(info.total_mult_adds)
        except ImportError:
            return None
        except Exception as exc:
            # 后端自身失败必须与“依赖未安装”区分，否则 MACs 缺失会被误报成缺依赖。
            torchinfo_failure.append(f"{type(exc).__name__}: {exc}")
            return None

    try:
        import warnings
        import torch as _torch
        from thop import profile as thop_profile

        with _torch.no_grad():
            _timing_fn()

        def _zero_ops(m, i, o):
            # thop 清理阶段执行 ``m._buffers.pop("total_ops")``，必须写入
            # ``_buffers`` 而非普通属性；否则 pop 抛 KeyError 中断 hook 清理，
            # 残留的 RNN/卷积统计 hook 会在后续计时阶段崩溃
            # （'GRU' object has no attribute 'total_ops'）。
            m._buffers["total_ops"] = _torch.zeros(1, dtype=_torch.float64)

        custom_ops: dict = {}
        for _, submod in model.named_modules():
            cls = type(submod)
            if cls.__name__ in _THOP_ZERO_OPS_CLASSES and cls not in custom_ops:
                custom_ops[cls] = _zero_ops
                excluded_types.append(cls.__name__)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs_raw, _ = thop_profile(
                _profile_target,
                inputs=(x,),
                verbose=False,
                custom_ops=custom_ops,
            )
        macs = float(macs_raw)
        macs_backend = "thop"
        macs_coverage = (
            "partial/unverified; module hooks may miss functional, complex, "
            "or unsupported custom operators"
        )
        macs_str = _fmt_num(macs, "MACs")
        if excluded_types:
            excluded = ", ".join(sorted(set(excluded_types)))
            macs_coverage += f"; zero-counted unsupported operators: {excluded}"
            macs_str += f"  (excl. {excluded})"
    except ImportError:
        macs = _try_torchinfo()
        if macs is not None:
            macs_backend = "torchinfo"
            macs_coverage = "torchinfo reported mult-adds; operator coverage backend-dependent"
            macs_str = _fmt_num(macs, "MACs")
        elif torchinfo_failure:
            macs_str = f"N/A (torchinfo failed: {torchinfo_failure[-1]})"
            macs_error = f"thop not installed; torchinfo failed: {torchinfo_failure[-1]}"
        else:
            macs_str = "N/A  (pip install thop)"
            macs_error = "thop/torchinfo not installed"
    except Exception as e:
        macs = _try_torchinfo()
        if macs is not None:
            macs_backend = "torchinfo"
            macs_coverage = "torchinfo fallback after thop failure; operator coverage backend-dependent"
            macs_str = _fmt_num(macs, "MACs")
        else:
            macs_str = f"N/A ({type(e).__name__}: {e})"
            macs_error = f"{type(e).__name__}: {e}"
            if torchinfo_failure:
                macs_error += f"; torchinfo also failed: {torchinfo_failure[-1]}"

    macs_input = input_description or {
        "waveform": {
            "shape": list(x.shape),
            "sample_rate_hz": int(sample_rate),
            "duration_seconds": float(duration),
        }
    }
    complexity = make_complexity_contract(
        total_params=total_p,
        trainable_params=train_p,
        params_scope=params_scope,
        params_source=params_source,
        macs=macs,
        macs_backend=macs_backend,
        macs_input=macs_input,
        macs_scope=macs_scope,
        macs_coverage=macs_coverage,
    )

    # ── Inference latency (RTF) ──
    latency_ms: float | None = None
    rtf: float | None = None
    rtf_error: str | None = None
    try:
        with torch.no_grad():
            for _ in range(warmup):
                _timing_fn()
            _synchronize_cuda(device)
            t0 = time.perf_counter()
            for _ in range(repeats):
                _timing_fn()
            _synchronize_cuda(device)
            latency_ms = (time.perf_counter() - t0) / repeats * 1000
        rtf = latency_ms / (duration * 1000)
    except Exception as exc:
        rtf_error = f"{type(exc).__name__}: {exc}"

    # ── Print (pure ASCII to avoid Windows terminal encoding) ──
    if not quiet:
        w = 62
        sep = "-" * w
        title = label or model.__class__.__name__
        native_sr = getattr(model, "target_fs", sample_rate)
        sr_note = f"  [native SR: {native_sr // 1000}kHz]" if native_sr != sample_rate else ""

        print(f"\n{sep}")
        print(f"  [{title}]")
        print(sep)
        print(f"  Input    : {sample_rate/1000:.0f}kHz x {duration:.1f}s  [B={batch_size}, T={n_samples}]{sr_note}")
        print(f"  Params   : {_fmt_num(total_p)}  (trainable: {_fmt_num(train_p)}; scope={params_scope}; source={params_source})")
        print(f"  MACs     : {macs_str}  (backend={macs_backend}; scope={macs_scope})")
        print(f"  FLOPs    : {NOT_MEASURED}")
        print(f"  MOS      : {NOT_MEASURED}  ({MOS_OFFLINE_BOUNDARY})")
        if rtf_error:
            print(f"  Latency  : N/A ({rtf_error})")
        else:
            print(f"  Latency  : {latency_ms:.2f} ms/frame  RTF={rtf:.4f}  ({device})")
        print(sep)

    return {
        "params": total_p,
        "trainable": train_p,
        "macs": macs,
        "macs_error": macs_error,
        "latency_ms": latency_ms,
        "rtf": rtf,
        "rtf_error": rtf_error,
        "complexity": complexity,
    }


def _restore_tensor_devices(
    model: nn.Module,
    parameter_devices: dict[str, torch.device],
    buffer_devices: dict[str, torch.device],
) -> None:
    """逐个恢复参数和 buffer 的原始设备，保留调用方的混合设备布局。"""
    for name, parameter in model.named_parameters():
        original_device = parameter_devices.get(name)
        if original_device is None:
            # profiling 期间新增的参数没有原始设备记录：跳过，避免 finally 中的
            # KeyError 掩盖真实异常并中断其余参数恢复。
            continue
        if parameter.device != original_device:
            parameter.data = parameter.data.to(original_device)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(original_device)
    for name, buffer in model.named_buffers():
        original_device = buffer_devices.get(name)
        if original_device is None:
            continue
        if buffer.device != original_device:
            module_name, _, buffer_name = name.rpartition(".")
            owner = model.get_submodule(module_name) if module_name else model
            owner._buffers[buffer_name] = buffer.to(original_device)


def profile_model(
    model: nn.Module,
    sample_rate: int = 16000,
    duration: float = 1.0,
    batch_size: int = 1,
    forward_kwargs: dict | None = None,
    forward_fn=None,
    dual_input: bool = False,
    device: str = "cpu",
    warmup: int = 3,
    repeats: int = 10,
    label: str = "",
    quiet: bool = False,
    params_scope: str = "model",
    params_source: str = "runtime model.parameters()",
    macs_scope: str = "profile forward path",
    profile_input: torch.Tensor | None = None,
    input_description: dict | None = None,
) -> dict:
    """统计模型复杂度且在返回或异常时恢复调用方的模型状态与设备。"""
    module_training = [(module, module.training) for module in model.modules()]
    parameter_devices = {
        name: parameter.device for name, parameter in model.named_parameters()
    }
    buffer_devices = {name: buffer.device for name, buffer in model.named_buffers()}

    try:
        return _profile_model(
            model,
            sample_rate=sample_rate,
            duration=duration,
            batch_size=batch_size,
            forward_kwargs=forward_kwargs,
            forward_fn=forward_fn,
            dual_input=dual_input,
            device=device,
            warmup=warmup,
            repeats=repeats,
            label=label,
            quiet=quiet,
            params_scope=params_scope,
            params_source=params_source,
            macs_scope=macs_scope,
            profile_input=profile_input,
            input_description=input_description,
        )
    finally:
        _restore_tensor_devices(model, parameter_devices, buffer_devices)
        for module, was_training in module_training:
            module.training = was_training


def make_profile_parser(default_sr: int = 16000, default_dur: float = 1.0) -> argparse.ArgumentParser:
    """Return standard argparse for model __main__ use."""
    p = argparse.ArgumentParser(description="Model parameter count and compute analysis")
    p.add_argument("--sr", type=int, default=default_sr,
                   help=f"Input sample rate Hz (default {default_sr})")
    p.add_argument("--dur", type=float, default=default_dur,
                   help=f"Input duration seconds (default {default_dur})")
    p.add_argument("--batch", type=int, default=1, help="Batch size (default 1)")
    p.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda' (default cpu)")
    p.add_argument("--repeats", type=int, default=10, help="Timing repeat count (default 10)")
    return p
