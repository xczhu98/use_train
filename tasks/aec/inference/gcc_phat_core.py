"""GCC-PHAT 延迟估计共享内核（numpy/scipy 纯实现，无 torch、无项目内部依赖）。

训练数据管道（``tasks/aec/data/dynamic_dataset.py``）与离线推理对齐
（``tasks/aec/inference/offline_alignment.py``）共用本模块，避免两份实现漂移。
本模块无 torch、无项目内部依赖，可按文件路径直接加载。

质量门口径与 ``tasks/aec/inference/streaming_delay_estimator.py`` 的
``_update_delay``（:327-331）完全一致：
``peak >= peak_threshold`` 且 ``peak >= mean / 10^(-prominence_db/20)``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


_GCC_TARGET_WORK_SR = 4000.0


def _finite_number(name: str, value: float) -> float:
    """把数值参数转成 float，并拒绝 NaN/Inf。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def clamp_gcc_delay_samples(
    tau_sec: float,
    sample_rate: int,
    *,
    bias_seconds: float = 0.0,
    max_delay_seconds: Optional[float] = None,
    min_delay_samples: int = 0,
) -> int:
    """把 GCC lag 转成实际施加的非负整数延迟。

    ``bias_seconds`` 表示估计器需要扣除的固定正偏差，因此统一采用
    ``effective_tau = tau_sec - bias_seconds``。训练、离线推理与预物化必须
    共用此函数，避免同一配置在不同路径上出现相反符号。
    """
    tau = _finite_number("tau_sec", tau_sec)
    bias = _finite_number("bias_seconds", bias_seconds)
    sr_value = _finite_number("sample_rate", sample_rate)
    if sr_value <= 0.0 or not sr_value.is_integer():
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    sr = int(sr_value)
    min_value = _finite_number("min_delay_samples", min_delay_samples)
    if min_value < 0.0 or not min_value.is_integer():
        raise ValueError(
            f"min_delay_samples must be a non-negative integer, got {min_delay_samples!r}"
        )
    min_samples = int(min_value)
    delay_samples = int(max(0.0, (tau - bias) * float(sr)))
    if max_delay_seconds is not None:
        max_delay = _finite_number("max_delay_seconds", max_delay_seconds)
        if max_delay < 0.0:
            raise ValueError(
                f"max_delay_seconds must be >= 0, got {max_delay_seconds!r}"
            )
        delay_samples = min(delay_samples, int(round(max_delay * float(sr))))
    if delay_samples <= min_samples:
        return 0
    return delay_samples


def resolve_gcc_downsample(target_sr: int, downsample: Optional[int] = None) -> tuple[int, float]:
    """解析 GCC 降采样倍率与实际工作采样率。

    未显式指定倍率时，选择最接近 4 kHz 的整数降采样倍率；非整倍数采样率
    使用 ``floor(target_sr / 4000 + 0.5)``，并至少保持倍率为 1，避免上采样。
    显式倍率优先，以兼容旧配置的自定义 GCC 口径。
    """
    sr_value = _finite_number("target_sr", target_sr)
    if sr_value <= 0.0 or not sr_value.is_integer():
        raise ValueError(f"target_sr must be a positive integer, got {target_sr!r}")
    sr = int(sr_value)
    if downsample is None:
        ds = max(1, int(np.floor(float(sr) / _GCC_TARGET_WORK_SR + 0.5)))
    else:
        ds_value = _finite_number("downsample", downsample)
        if ds_value < 1.0 or not ds_value.is_integer():
            raise ValueError(f"downsample must be a positive integer, got {downsample!r}")
        ds = int(ds_value)
    return ds, float(sr) / float(ds)


def gcc_phat_tau_with_confidence(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int,
    interp: int = 4,
    *,
    max_tau_seconds: Optional[float] = None,
    non_negative: bool = True,
) -> tuple[float, float, float]:
    """GCC-PHAT 延迟估计 + 峰值置信度。

    返回 ``(tau_sec, peak_mag, mean_mag)``：peak_mag / mean_mag 为搜索窗内
    |互相关| 的峰值与均值（口径与
    ``tasks/aec/inference/streaming_delay_estimator.py`` 的 ``_find_peak_in_cc``
    一致）。调用侧的质量门用 ``gcc_quality_ok``，表达式与
    ``streaming_delay_estimator.py`` 的 ``_update_delay``（:327-331）完全
    一致：peak >= peak_threshold 且 peak >= mean / 10^(-prominence_db/20)。
    """
    fs_value = _finite_number("fs", fs)
    if fs_value <= 0.0:
        raise ValueError(f"fs must be positive, got {fs!r}")
    interp_value = _finite_number("interp", interp)
    if interp_value < 1.0 or not interp_value.is_integer():
        raise ValueError(f"interp must be a positive integer, got {interp!r}")
    interp_int = int(interp_value)
    max_tau = None
    if max_tau_seconds is not None:
        max_tau = _finite_number("max_tau_seconds", max_tau_seconds)
        if max_tau < 0.0:
            raise ValueError(
                f"max_tau_seconds must be >= 0, got {max_tau_seconds!r}"
            )

    sig = np.asarray(sig, dtype=np.float32).ravel()
    refsig = np.asarray(refsig, dtype=np.float32).ravel()
    if not np.isfinite(sig).all() or not np.isfinite(refsig).all():
        raise ValueError("GCC inputs must contain only finite samples")
    if sig.size < 8 or refsig.size < 8:
        return 0.0, 0.0, 0.0
    n = int(sig.shape[0] + refsig.shape[0])
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    denom = np.abs(R) + 1e-15
    cc = np.fft.irfft(R / denom, n=(interp_int * n))
    max_shift = int(interp_int * n / 2)
    if max_tau is not None:
        cap = int(interp_int * fs_value * max_tau)
        max_shift = min(cap, max_shift)
    if non_negative or max_shift == 0:
        # 避免 max_shift=0 时 cc[-0:] 展开整段。
        cc = cc[: max_shift + 1]
        abs_cc = np.abs(cc)
        peak_idx = int(np.argmax(abs_cc))
        shift = peak_idx
    else:
        cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
        abs_cc = np.abs(cc)
        peak_idx = int(np.argmax(abs_cc))
        shift = peak_idx - max_shift
    tau_sec = float(shift) / float(interp_int * fs_value)
    peak_mag = float(abs_cc[peak_idx])
    mean_mag = float(abs_cc.mean()) if abs_cc.size else 0.0
    if not np.isfinite([tau_sec, peak_mag, mean_mag]).all():
        raise RuntimeError("GCC produced a non-finite estimate")
    return tau_sec, peak_mag, mean_mag


def gcc_phat_tau_seconds(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int,
    interp: int = 4,
    *,
    max_tau_seconds: Optional[float] = None,
    non_negative: bool = True,
) -> float:
    """Estimate time delay (seconds) between *sig* and *refsig* via GCC-PHAT.

    Returns the lag of the single peak of the phase-transform cross-correlation.
    This is a **coarse integer-sample** estimator; for dense multipath it may
    mis-lock to a strong reflection.

    Args:
        non_negative: If True, restrict the search to non-negative lags
            (sig lags refsig). This is the physically valid case for AEC,
            where echo cannot arrive before the farend signal.  Default is True.
    """
    # 委托给带置信度的实现取 tau（FFT 主体共用）。
    tau_sec, _, _ = gcc_phat_tau_with_confidence(
        sig,
        refsig,
        fs,
        interp=interp,
        max_tau_seconds=max_tau_seconds,
        non_negative=non_negative,
    )
    return tau_sec


def gcc_quality_ok(
    peak_mag: float,
    mean_mag: float,
    peak_threshold: float,
    prominence_db: float,
) -> bool:
    """GCC 质量门：与 tasks/aec/inference/streaming_delay_estimator.py:327-331 完全一致。

    peak >= peak_threshold 且 peak >= mean / 10^(-prominence_db/20)
    （默认 0.05 / 6dB）。
    """
    values = np.asarray(
        [peak_mag, mean_mag, peak_threshold, prominence_db], dtype=np.float64
    )
    if not np.isfinite(values).all():
        return False
    peak, mean, threshold, prominence = (float(value) for value in values)
    if peak < 0.0 or mean < 0.0 or not 0.0 < threshold <= 1.0 or prominence < 0.0:
        return False
    if peak < threshold:
        return False
    if peak < mean / (10.0 ** (-prominence / 20.0)):
        return False
    return True


def gcc_window_active(x: np.ndarray, fs: int) -> bool:
    """20ms 帧 RMS 能量门：窗活跃当且仅当 mean(r) > max(1e-4, max(r)·10^(-50/20))。

    绝对下限 1e-4 防止静音窗的 epsilon 能量误通过（实测教训：纯静音窗
    的数值噪声会骗过纯相对门限，导致 GCC 锁到随机 lag）。
    """
    frame = max(1, int(round(float(fs) * 0.02)))  # 20ms
    n_frames = int(x.size) // frame
    if n_frames <= 0:
        return False
    frames = x[: n_frames * frame].astype(np.float64, copy=False).reshape(n_frames, frame)
    r = np.sqrt(np.mean(frames ** 2, axis=1))
    return bool(r.mean() > max(1e-4, float(r.max()) * 10.0 ** (-50.0 / 20.0)))


def trajectory_tau_at(trajectory: tuple, t_sec: float) -> float:
    """查轨迹上距 ``t_sec`` 最近的窗的 tau。

    ``trajectory``: ``[(window_start_sec, tau_sec), ...]``（稀疏，1s hop）。
    """
    best = min(trajectory, key=lambda kv: abs(float(kv[0]) - float(t_sec)))
    return float(best[1])


@dataclass
class GccAlignEstimate:
    """GCC 精对齐估计结果（训练数据集 per-worker LRU 缓存的 value）。

    Attributes:
        tau_sec: 文件级 tau（秒）。constant 对即最终估计；时变对为各窗中值（兜底）。
        trajectory: 时变轨迹 ``((window_start_sec, tau_sec), ...)``；constant 为 None。
        time_varying: 全量扫描判定 IQR 超阈值为 True。
        low_confidence: 所有窗均未过质量门时为 True（tau=0，zero_pad 策略，样本保留）。
        mode: ``windowed_fast`` / ``windowed_full_scan`` / ``low_confidence``
            （数据集侧估计异常兜底另用 ``estimation_failed``）。
        peak: 参与聚合的窗的质量门 peak 幅值（fast 取 min，full_scan 取 median）。
        prominence: 线性 prominence = peak / mean（与 peak 相同的聚合口径）。
    """

    tau_sec: float
    trajectory: Optional[tuple]
    time_varying: bool
    low_confidence: bool
    mode: str
    peak: float
    prominence: float


def estimate_gcc_tau_windowed(
    wf: np.ndarray,
    we: np.ndarray,
    target_sr: int,
    *,
    downsample: Optional[int] = None,
    max_tau_seconds: float = 0.2,
    peak_threshold: float = 0.05,
    prominence_db: float = 6.0,
    tv_iqr_ms: float = 10.0,
    max_windows: int = 8,
) -> GccAlignEstimate:
    """窗式 GCC 延迟估计：head+tail @4k 快路径，必要时升级全量扫描。

    - 取 head 1s / tail 1s 两片段，resample_poly 降到 4k 的临时副本
      （仅用于估计，原始 16k 音频不碰）；
    - 各窗先过能量门（echo 与 far 同时活跃），再过 GCC + 质量门；
    - 双窗通过且 |Δ| < 5ms → tau = 均值；仅单窗通过 → 用该值；
      双窗分歧或全灭 → 升级全量扫描（见 ``gcc_full_scan``）。

    估计过程只使用 numpy（FFT / resample_poly），**不消耗任何 rng**。

    Args:
        wf: farend 参考信号（16k 等原始采样率）。
        we: echo/mic 信号（与 wf 同采样率；估计 echo 相对 farend 的滞后）。
        target_sr: wf/we 的采样率。
        downsample: 显式降采样因子；省略时自动选择最接近 4 kHz 的整数倍率。
            显式值优先，用于兼容旧配置的自定义 GCC 口径。
        max_tau_seconds: GCC 延迟搜索上限（秒）；0 表示仅搜索零延迟。
        peak_threshold / prominence_db: 质量门阈值（见 ``gcc_quality_ok``）。
        tv_iqr_ms: 时变判定 IQR 阈值（毫秒）。
        max_windows: 全量扫描有效窗数上限（均匀抽样）。
    """
    ds, fs_ds = resolve_gcc_downsample(target_sr, downsample)
    _max_tau = _finite_number("max_tau_seconds", max_tau_seconds)
    _peak_threshold = _finite_number("peak_threshold", peak_threshold)
    _prominence_db = _finite_number("prominence_db", prominence_db)
    _tv_iqr_ms = _finite_number("tv_iqr_ms", tv_iqr_ms)
    max_windows_value = _finite_number("max_windows", max_windows)
    if _max_tau < 0.0:
        raise ValueError(f"max_tau_seconds must be >= 0, got {max_tau_seconds!r}")
    if not 0.0 < _peak_threshold <= 1.0:
        raise ValueError(f"peak_threshold must be in (0, 1], got {peak_threshold!r}")
    if _prominence_db < 0.0:
        raise ValueError(f"prominence_db must be >= 0, got {prominence_db!r}")
    if _tv_iqr_ms <= 0.0:
        raise ValueError(f"tv_iqr_ms must be > 0, got {tv_iqr_ms!r}")
    if max_windows_value < 1.0 or not max_windows_value.is_integer():
        raise ValueError(f"max_windows must be a positive integer, got {max_windows!r}")
    wf = np.asarray(wf, dtype=np.float32).ravel()
    we = np.asarray(we, dtype=np.float32).ravel()
    if not np.isfinite(wf).all() or not np.isfinite(we).all():
        raise ValueError("GCC inputs must contain only finite samples")
    max_tau = _max_tau
    n16 = min(int(wf.size), int(we.size))
    win16 = min(n16, int(round(float(target_sr))))

    def _estimate_window(seg_wf16: np.ndarray, seg_we16: np.ndarray, w_start_sec: float):
        """单个 1s 窗：降采样 → 能量门 → GCC + 质量门；不通过返回 None。"""
        seg_wf = resample_poly(seg_wf16, 1, ds).astype(np.float32, copy=False)
        seg_we = resample_poly(seg_we16, 1, ds).astype(np.float32, copy=False)
        if not (gcc_window_active(seg_we, fs_ds) and gcc_window_active(seg_wf, fs_ds)):
            return None
        tau, peak, mean = gcc_phat_tau_with_confidence(
            seg_we,
            seg_wf,
            fs_ds,
            interp=4,  # 4k 窗上固定 interp=4（0.0625ms 量化），与流式估计器口径一致
            max_tau_seconds=max_tau,
            non_negative=True,
        )
        if not gcc_quality_ok(peak, mean, peak_threshold, prominence_db):
            return None
        return (float(w_start_sec), float(tau), float(peak), float(mean))

    def _prom(e: tuple) -> float:
        return float(e[2] / e[3]) if e[3] > 0 else 0.0

    ests: list[tuple] = []
    tail_off16 = n16 - win16
    for seg_wf16, seg_we16, w_start_sec in (
        (wf[:win16], we[:win16], 0.0),
        (
            wf[tail_off16 : tail_off16 + win16],
            we[tail_off16 : tail_off16 + win16],
            tail_off16 / float(target_sr),
        ),
    ):
        r = _estimate_window(seg_wf16, seg_we16, w_start_sec)
        if r is not None:
            ests.append(r)

    if len(ests) == 2 and abs(ests[0][1] - ests[1][1]) < 0.005:
        # 双窗通过且一致（|Δ| < 5ms）→ tau = 均值，constant
        return GccAlignEstimate(
            tau_sec=0.5 * (ests[0][1] + ests[1][1]),
            trajectory=None,
            time_varying=False,
            low_confidence=False,
            mode="windowed_fast",
            peak=float(min(ests[0][2], ests[1][2])),
            prominence=float(min(_prom(ests[0]), _prom(ests[1]))),
        )
    if len(ests) == 1:
        # 仅单窗通过 → 用该值，constant
        e = ests[0]
        return GccAlignEstimate(
            tau_sec=e[1],
            trajectory=None,
            time_varying=False,
            low_confidence=False,
            mode="windowed_fast",
            peak=e[2],
            prominence=_prom(e),
        )
    # 双窗分歧或双窗全灭 → 升级全量扫描
    return gcc_full_scan(
        wf,
        we,
        target_sr,
        ds,
        fs_ds,
        max_tau,
        peak_threshold=peak_threshold,
        prominence_db=prominence_db,
        tv_iqr_ms=tv_iqr_ms,
        max_windows=max_windows,
    )


def gcc_full_scan(
    wf: np.ndarray,
    we: np.ndarray,
    target_sr: int,
    ds: int,
    fs_ds: int,
    max_tau: Optional[float],
    *,
    peak_threshold: float = 0.05,
    prominence_db: float = 6.0,
    tv_iqr_ms: float = 10.0,
    max_windows: int = 8,
) -> GccAlignEstimate:
    """全量扫描：整文件 4k 瞬时副本上能量扫描所有 1s 窗（hop 1s）。

    - 活跃窗（echo 与 far 同时活跃）均匀抽样封顶 ``max_windows``；
    - 有效估计 ≥1 → tau_file = median；IQR > ``tv_iqr_ms`` →
      time_varying=True 并保留稀疏轨迹，否则 constant；
    - 全灭 → low_confidence=True, tau=0（zero_pad 策略，样本保留）。
    """
    n16 = min(int(wf.size), int(we.size))
    win16 = min(n16, int(round(1.0 * float(target_sr))))
    # 整文件降采样（瞬时副本，不进缓存）
    wf_ds = resample_poly(wf[:n16], 1, ds).astype(np.float32, copy=False)
    we_ds = resample_poly(we[:n16], 1, ds).astype(np.float32, copy=False)
    nds = min(int(wf_ds.size), int(we_ds.size))
    win_ds = max(64, int(round(win16 / ds)))
    starts = list(range(0, max(1, nds - win_ds + 1), win_ds)) or [0]
    active = [
        s
        for s in starts
        if gcc_window_active(we_ds[s : s + win_ds], fs_ds)
        and gcc_window_active(wf_ds[s : s + win_ds], fs_ds)
    ]
    # 均匀抽样封顶（确定性 np.linspace，不用 rng）
    max_w = max(1, int(max_windows))
    if len(active) > max_w:
        sel = sorted({int(round(i)) for i in np.linspace(0, len(active) - 1, max_w)})
        active = [active[i] for i in sel]
    ests: list[tuple] = []
    for s in active:
        tau, peak, mean = gcc_phat_tau_with_confidence(
            we_ds[s : s + win_ds],
            wf_ds[s : s + win_ds],
            fs_ds,
            interp=4,
            max_tau_seconds=max_tau,
            non_negative=True,
        )
        if gcc_quality_ok(peak, mean, peak_threshold, prominence_db):
            ests.append((s / float(fs_ds), float(tau), float(peak), float(mean)))
    if not ests:
        return GccAlignEstimate(
            tau_sec=0.0,
            trajectory=None,
            time_varying=False,
            low_confidence=True,
            mode="low_confidence",
            peak=0.0,
            prominence=0.0,
        )
    taus = np.array([e[1] for e in ests], dtype=np.float64)
    tau_file = float(np.median(taus))
    if taus.size >= 2:
        q25, q75 = np.percentile(taus, [25.0, 75.0])
        iqr_ms = float((q75 - q25) * 1000.0)
    else:
        iqr_ms = 0.0
    time_varying = iqr_ms > float(tv_iqr_ms)
    trajectory = (
        tuple((float(t0), float(tau)) for t0, tau, _, _ in ests) if time_varying else None
    )
    return GccAlignEstimate(
        tau_sec=tau_file,
        trajectory=trajectory,
        time_varying=bool(time_varying),
        low_confidence=False,
        mode="windowed_full_scan",
        peak=float(np.median([e[2] for e in ests])),
        prominence=float(
            np.median([e[2] / e[3] if e[3] > 0 else 0.0 for e in ests])
        ),
    )
