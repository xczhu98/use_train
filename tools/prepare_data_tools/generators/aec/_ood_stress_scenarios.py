#!/usr/bin/env python3
"""JAE-M03 OOD stress 场景 operators（纯函数，无数据集类依赖）。

审计发现训练仿真只覆盖七基本场景 + 静态 delay + 单配置 jitter，以下关键
真实场景没有显式覆盖：far-end 停止后 RIR tail、中途 RIR/path replacement、
buffer reset、reference 缺失/异常、近端泄漏进 reference、reference 极性反转、
采样时钟异步、codec 退化、AGC/DRC/limiter。本模块只在固定 OOD stress set
上补齐这些场景，**不改训练数据分布**。

每个 operator 统一签名::

    op(far, near, rir, sr, rng, **kw) -> dict

返回 dict 含以下键（音频全部为 1-D float32 numpy，等长；长度以 near 为准，
不足右侧补零、超出裁切）::

    mic     : 麦克风信号，默认 mic = near + echo（场景另有定义时除外）
    farend  : 模型参考通道（场景可篡改它：置零 / 反极性 / 混入泄漏等）
    nearend : oracle 近端 stem（监督目标）
    echo    : oracle 回声 stem（mic 中真实混入的回声分量）
    events  : list[dict]，场景事件；随机参数全部记录在事件内，可导出 meta.json

设计说明：
- 卷积统一用 scipy.signal.fftconvolve（requirements.txt 已含 scipy>=1.10）。
- 所有随机量都从传入的 rng（np.random.Generator）抽取；调用方按
  （master seed + 样本索引）派生 rng，保证固定 seed 逐样本可回放。
- **kw 可覆盖任一随机参数（如 d1_ms=...），便于单测固定边界条件。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import fftconvolve

from core.data.simulation.timebase_impairment import (
    apply_integer_delay,
    apply_linear_timebase_ratio,
)

__all__ = ["SCENARIOS", "fit_len", "convolve_trim"]


# ----------------------------------------------------------------------
# 公共工具
# ----------------------------------------------------------------------
def _as_f32_1d(x) -> np.ndarray:
    """转 1-D float32。"""
    return np.asarray(x, dtype=np.float32).ravel()


def _positive_finite(value, name: str) -> float:
    """校验场景正数参数，拒绝 NaN/Inf/非正值。"""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return parsed


def _strict_fraction(value, name: str) -> float:
    """校验真正中途切换的比例，拒绝非有限值和两个端点。"""
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1, got {value!r}")
    return parsed


def _finite(value, name: str) -> float:
    """校验可带符号参数为有限数。"""
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def fit_len(x: np.ndarray, n: int) -> np.ndarray:
    """等长处理：不足右侧补零，超出裁切。"""
    x = _as_f32_1d(x)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out


def convolve_trim(x: np.ndarray, rir: np.ndarray, n: int) -> np.ndarray:
    """fftconvolve 后裁到 n 点（回声路径仿真）。"""
    return fit_len(fftconvolve(_as_f32_1d(x), _as_f32_1d(rir)), n)


def _rms(x: np.ndarray) -> float:
    """均方根（float64 累加避免溢出）。"""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def _delay_zero(x: np.ndarray, d: int) -> np.ndarray:
    """保留 OOD facade，委托共享零填充整数延迟 mechanics。"""
    return apply_integer_delay(x, d)


def _prepare(far, near, rir):
    """统一预处理：转 float32、far 对齐到 near 长度，返回 (far, near, rir, n)。"""
    near = _as_f32_1d(near)
    n = len(near)
    far = fit_len(far, n)
    rir = _as_f32_1d(rir)
    return far, near, rir, n


def _result(farend, near, echo, events, mic=None):
    """组装返回 dict；默认 mic = near + echo。"""
    if mic is None:
        mic = near + echo
    return {
        "mic": _as_f32_1d(mic),
        "farend": _as_f32_1d(farend),
        "nearend": _as_f32_1d(near),
        "echo": _as_f32_1d(echo),
        "events": events,
    }


# ----------------------------------------------------------------------
# 场景 1：far-end 停止后 RIR tail
# ----------------------------------------------------------------------
def far_end_stop_rir_tail(far, near, rir, sr, rng, **kw):
    """far 在 t1（U[40%,60%] 长度）后归零；echo 保留 RIR tail 自然衰减。"""
    far, near, rir, n = _prepare(far, near, rir)
    t1_frac = _positive_finite(
        kw.get("t1_frac", rng.uniform(0.4, 0.6)), "t1_frac"
    )
    t1 = int(t1_frac * n)
    far_stop = far.copy()
    far_stop[t1:] = 0.0
    echo = convolve_trim(far_stop, rir, n)
    events = [{"type": "far_end_stop", "t_ms": 1000.0 * t1 / sr}]
    return _result(far_stop, near, echo, events)


# ----------------------------------------------------------------------
# 场景 2：中途 RIR / path replacement
# ----------------------------------------------------------------------
def mid_utterance_path_replace(far, near, rir, sr, rng, **kw):
    """前半 echo=conv(far,rir)，后半改用 rir2（25ms 循环移位或时间反转，
    制造可区分路径），t_switch 处 10ms 线性交叉淡化。"""
    far, near, rir, n = _prepare(far, near, rir)
    t_frac = _strict_fraction(
        kw.get("t_frac", rng.uniform(0.4, 0.6)), "t_frac"
    )
    t_switch = int(t_frac * n)
    rir2_kind = kw.get("rir2_kind")
    if rir2_kind is None:
        rir2_kind = "circular_shift_25ms" if int(rng.integers(2)) == 0 else "time_reverse"
    if rir2_kind == "circular_shift_25ms":
        rir2 = np.roll(rir, int(0.025 * sr))
    elif rir2_kind == "time_reverse":
        rir2 = rir[::-1].copy()
    else:
        raise ValueError(f"unknown rir2_kind: {rir2_kind}")
    echo1 = convolve_trim(far, rir, n)
    echo2 = convolve_trim(far, rir2, n)
    # t_switch 处 10ms 线性交叉淡化（以切换点为中心）
    w = max(2, int(0.010 * sr))
    lo = max(0, t_switch - w // 2)
    hi = min(n, t_switch + w // 2)
    echo = echo1.copy()
    echo[t_switch:] = echo2[t_switch:]
    if hi > lo:
        ramp = np.linspace(0.0, 1.0, hi - lo, dtype=np.float32)
        echo[lo:hi] = echo1[lo:hi] * (1.0 - ramp) + echo2[lo:hi] * ramp
    events = [{
        "type": "path_replace",
        "t_ms": 1000.0 * t_switch / sr,
        "rir2_kind": rir2_kind,
    }]
    return _result(far, near, echo, events)


# ----------------------------------------------------------------------
# 场景 3：buffer reset（延迟跳变，硬切）
# ----------------------------------------------------------------------
def buffer_reset(far, near, rir, sr, rng, **kw):
    """echo 前半延迟 d1=U[20,60]ms、后半延迟 d2=U[140,220]ms（零填充延迟），
    切换点硬切。"""
    far, near, rir, n = _prepare(far, near, rir)
    t_frac = _strict_fraction(
        kw.get("t_frac", rng.uniform(0.4, 0.6)), "t_frac"
    )
    t_switch = int(t_frac * n)
    d1_ms = _positive_finite(
        kw.get("d1_ms", rng.uniform(20.0, 60.0)), "d1_ms"
    )
    d2_ms = _positive_finite(
        kw.get("d2_ms", rng.uniform(140.0, 220.0)), "d2_ms"
    )
    d1 = int(round(d1_ms * sr / 1000.0))
    d2 = int(round(d2_ms * sr / 1000.0))
    base = convolve_trim(far, rir, n)
    echo = np.empty(n, dtype=np.float32)
    echo[:t_switch] = _delay_zero(base, d1)[:t_switch]
    echo[t_switch:] = _delay_zero(base, d2)[t_switch:]
    events = [{
        "type": "buffer_reset",
        "t_ms": 1000.0 * t_switch / sr,
        "d1_ms": d1_ms,
        "d2_ms": d2_ms,
    }]
    return _result(far, near, echo, events)


# ----------------------------------------------------------------------
# 场景 4：reference 缺失
# ----------------------------------------------------------------------
def reference_missing(far, near, rir, sr, rng, **kw):
    """输出 farend 在 [t0,t1) 置零（t0=U[30%,50%] 长度，窗长 U[0.5,1.0]s），
    mic / echo 不受影响。"""
    far, near, rir, n = _prepare(far, near, rir)
    t0_frac = _positive_finite(
        kw.get("t0_frac", rng.uniform(0.3, 0.5)), "t0_frac"
    )
    win_sec = _positive_finite(
        kw.get("win_sec", rng.uniform(0.5, 1.0)), "win_sec"
    )
    t0 = int(t0_frac * n)
    t1 = min(n, t0 + int(win_sec * sr))
    echo = convolve_trim(far, rir, n)
    farend = far.copy()
    farend[t0:t1] = 0.0
    events = [{
        "type": "reference_missing",
        "t0_ms": 1000.0 * t0 / sr,
        "t1_ms": 1000.0 * t1 / sr,
    }]
    return _result(farend, near, echo, events)


# ----------------------------------------------------------------------
# 场景 5：近端泄漏进 reference
# ----------------------------------------------------------------------
def nearend_leak_into_reference(far, near, rir, sr, rng, **kw):
    """farend = far + α·near（α=U[0.05,0.2]，按各自 RMS 归一后混合）。

    等价于 far/rms(far) + α·near/rms(near) 再乘回 rms(far)，保持 far 的
    幅度尺度，α 即为泄漏近端相对 far-end 的幅度比。
    """
    far, near, rir, n = _prepare(far, near, rir)
    alpha = _positive_finite(
        kw.get("alpha", rng.uniform(0.05, 0.2)), "alpha"
    )
    scale = _rms(far) / max(_rms(near), 1e-8)
    farend = far + alpha * scale * near
    echo = convolve_trim(far, rir, n)
    events = [{"type": "nearend_leak", "alpha": alpha}]
    return _result(farend, near, echo, events)


# ----------------------------------------------------------------------
# 场景 6：reference 极性反转
# ----------------------------------------------------------------------
def reference_polarity_inversion(far, near, rir, sr, rng, **kw):
    """farend = -far；echo 仍由 +far 卷积生成（mic 内回声不反转）。"""
    far, near, rir, n = _prepare(far, near, rir)
    echo = convolve_trim(far, rir, n)
    events = [{"type": "polarity_inversion"}]
    return _result(-far, near, echo, events)


# ----------------------------------------------------------------------
# 场景 7：采样时钟异步
# ----------------------------------------------------------------------
def clock_async_resample(far, near, rir, sr, rng, **kw):
    """echo 经线性插值重采样（比率 1±ppm，ppm=U[20,80] 随机符号）模拟
    采样时钟异步，之后裁回 / 补零到原长。

    kw 固定边界：显式传负 ppm 且未传 sign 时沿用负号（负漂移）；显式传
    sign 时以 sign 为准；ppm 为正或未传时符号仍随机。"""
    far, near, rir, n = _prepare(far, near, rir)
    ppm_val = _finite(kw.get("ppm", rng.uniform(20.0, 80.0)), "ppm")
    sign = kw.get("sign")
    if sign is None:
        # 调用方传 ppm=-50 期望负漂移，不应被随机翻正；正 ppm 维持原随机符号行为
        if ppm_val < 0:
            sign = -1.0
        else:
            sign = 1.0 if int(rng.integers(2)) == 0 else -1.0
    ppm = _finite(sign, "sign") * abs(ppm_val)
    echo = convolve_trim(far, rir, n)
    ratio = _positive_finite(1.0 + ppm * 1e-6, "resample_ratio")
    # 共享 primitive 保留原 np.interp 数值路径，生成资产 byte-compatible。
    echo_rs = apply_linear_timebase_ratio(echo, ratio).waveform
    events = [{"type": "clock_async", "ppm": ppm}]
    return _result(far, near, echo_rs, events)


# ----------------------------------------------------------------------
# 场景 8：codec 退化
# ----------------------------------------------------------------------
def _ideal_lowpass(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """FFT 理想低通（rfft 域高频置零）。"""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    X[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def _quantize_8bit(x: np.ndarray) -> np.ndarray:
    """8bit 均匀量化：round(x*128)/128。"""
    return (np.round(x * 128.0) / 128.0).astype(np.float32)


def codec_degrade(far, near, rir, sr, rng, **kw):
    """codec 退化：far 与 echo 分别经 3.4kHz 理想低通 + 8bit 量化。"""
    far, near, rir, n = _prepare(far, near, rir)
    cutoff = _positive_finite(kw.get("cutoff_hz", 3400.0), "cutoff_hz")
    far_deg = _quantize_8bit(_ideal_lowpass(far, cutoff, sr))
    echo = convolve_trim(far, rir, n)
    echo_deg = _quantize_8bit(_ideal_lowpass(echo, cutoff, sr))
    events = [{"type": "codec_degrade", "cutoff_hz": cutoff, "quant_bits": 8}]
    return _result(far_deg, near, echo_deg, events)


# ----------------------------------------------------------------------
# 场景 9：AGC / DRC / limiter
# ----------------------------------------------------------------------
def agc_drc_limiter(far, near, rir, sr, rng, **kw):
    """echo 过简易 DRC（20ms 滑动 RMS → 压缩增益 g=(rms/thr)^(1/ratio-1)，
    thr=-20dBFS，ratio=U[2,4]，增益 5ms 平滑）+ 0.95 硬限幅。"""
    far, near, rir, n = _prepare(far, near, rir)
    ratio = _positive_finite(
        kw.get("ratio", rng.uniform(2.0, 4.0)), "ratio"
    )
    thr = 10.0 ** (-20.0 / 20.0)  # -20 dBFS（幅度域）
    echo = convolve_trim(far, rir, n)
    # 20ms 滑动 RMS（centered）
    frame = max(1, int(0.020 * sr))
    sq = echo.astype(np.float64) ** 2
    rms = np.sqrt(np.convolve(sq, np.ones(frame) / frame, mode="same") + 1e-12)
    gain = (rms / thr) ** (1.0 / ratio - 1.0)
    # 数值保护：近零 RMS 处增益发散，限制在 [0.01, 100]
    gain = np.clip(gain, 1e-2, 1e2)
    # 5ms 增益平滑
    smooth = max(1, int(0.005 * sr))
    gain = np.convolve(gain, np.ones(smooth) / smooth, mode="same")
    echo_drc = np.clip(echo * gain.astype(np.float32), -0.95, 0.95).astype(np.float32)
    events = [{"type": "agc_drc", "ratio": ratio}]
    return _result(far, near, echo_drc, events)


# 场景注册表：key 即场景名
SCENARIOS: dict[str, Callable[..., dict]] = {
    "far_end_stop_rir_tail": far_end_stop_rir_tail,
    "mid_utterance_path_replace": mid_utterance_path_replace,
    "buffer_reset": buffer_reset,
    "reference_missing": reference_missing,
    "nearend_leak_into_reference": nearend_leak_into_reference,
    "reference_polarity_inversion": reference_polarity_inversion,
    "clock_async_resample": clock_async_resample,
    "codec_degrade": codec_degrade,
    "agc_drc_limiter": agc_drc_limiter,
}
