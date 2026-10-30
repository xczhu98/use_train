"""Post-mix noisy-only degradations (**Layer F**) for distortion / degradation simulation.

**Six-layer Dataset simulation pipeline (SE/AEC/TSE)** — Dataset 仿真仅包含
Layer A–F；**Layer D** 是 post-mix amplitude conditioning，固定按
D-normalize → D-guard 执行。只公开聚合的 ``layer_d_facts`` /
``simulation_by_layer['D']``，schema 严格为 ``stage_order`` / ``normalize`` /
``guard``。SE/TSE 的 ``guard`` 严格包含 ``roles`` / ``threshold`` / ``peak`` /
``scale`` / ``applied`` 五键；AEC 的 ``guard`` 保留 ``mic_related`` /
``farend`` 两个角色视图，但二者来自同一次五路 coupled guard。禁止公开
顶层 ``normalize``、独立 trace normalize 或顶层 farend guard。

``AECDeploymentFrontend`` 仅属于 AEC 模型输入/部署链路，由 Trainer/Inference
在 STFT 前运行，不属于 Dataset 仿真。

**This module** implements **Layer F** (DRC, codecs, clipping, EQ, …) gated by
``cfg.noisy_post_mix_degradations_enabled`` (default off until Layer-F experiments are enabled).
``post_eq`` 也由 AEC C-aec 的 echo-only ``random_eq`` 复用；它使用单向
peaking-biquad SOS 级联，保证平滑、因果且累计 dB 增益有界。

Optional dependencies (silently skipped if missing):
  - pedalboard: DRC, MP3, GSM codecs
  - torchaudio: Opus, AAC codecs
"""

from __future__ import annotations

import logging
import random
from numbers import Real
from typing import Optional

import librosa

from core.data.simulation.urgent_impl import mix_noise
from core.data.simulation_utils import rng_uniform_or_beta

_log = logging.getLogger(__name__)
import numpy as np
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# RNG helpers — use *rng* (np.random.Generator) when provided, else global random
# ---------------------------------------------------------------------------
def _rng_random(rng: Optional[np.random.Generator]) -> float:
    return float(rng.random()) if rng is not None else random.random()

def _rng_uniform(rng: Optional[np.random.Generator], a: float, b: float) -> float:
    return float(rng.uniform(a, b)) if rng is not None else random.uniform(a, b)

def _rng_randint(rng: Optional[np.random.Generator], a: int, b: int) -> int:
    """Inclusive [a, b]."""
    if rng is not None:
        return int(rng.integers(a, b + 1))
    return random.randint(a, b)

def _rng_choice(rng: Optional[np.random.Generator], seq):
    if rng is not None:
        return seq[int(rng.integers(0, len(seq)))]
    return random.choice(seq)

def _rng_choices_weighted(rng: Optional[np.random.Generator], seq, weights, k: int = 1):
    if rng is not None:
        total = sum(weights)
        p = [w / total for w in weights]
        idx = rng.choice(len(seq), size=k, p=p)
        return [seq[i] for i in idx]
    return random.choices(seq, weights=weights, k=k)

def _rng_normal(rng: Optional[np.random.Generator], loc: float, scale: float, size) -> np.ndarray:
    if rng is not None:
        return rng.normal(loc, scale, size)
    return np.random.normal(loc, scale, size)

# Lazy-loaded optional dependencies (cached at module level to avoid per-call overhead)
_pedalboard = None
_torchaudio_io = None


def _get_pedalboard():
    global _pedalboard
    if _pedalboard is None:
        try:
            import pedalboard as _pb
            _pedalboard = _pb
        except ImportError:
            _pedalboard = False
    return _pedalboard if _pedalboard is not False else None


def _get_torchaudio_io():
    global _torchaudio_io
    if _torchaudio_io is None:
        try:
            from torchaudio import io as _tio
            _torchaudio_io = _tio
        except ImportError:
            _torchaudio_io = False
    return _torchaudio_io if _torchaudio_io is not False else None


# ===================================================================
# P1-1  Hard / Soft Clipping  (speaker distortion)
# ===================================================================
def hard_clip_db(data: np.ndarray, threshold_db_min: float = -20,
                 threshold_db_max: float = -1,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    threshold = 10 ** (_rng_uniform(rng, threshold_db_min, threshold_db_max) / 20)
    return np.clip(data, -threshold, threshold)


def hard_clip_on_rate(data: np.ndarray, clip_rate_min: float = 0.01,
                      clip_rate_max: float = 0.2,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    clip_rate = _rng_uniform(rng, clip_rate_min, clip_rate_max)
    hist, bin_edges = np.histogram(np.abs(data), bins=1000)
    idx = np.cumsum(hist) > (1 - clip_rate) * len(data)
    if not np.any(idx):
        return data
    clip_threshold = bin_edges[:-1][idx][0]
    return np.clip(data, -clip_threshold, clip_threshold)


def soft_clip(data: np.ndarray, slope_min: float = 1.0,
              slope_max: float = 5.0,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    slope = _rng_uniform(rng, slope_min, slope_max)
    x_max = np.max(np.abs(data)) + 1e-8
    return x_max * data / (np.abs(x_max) ** slope + np.abs(data) ** slope + 1e-5) ** (1 / slope)


def sigmoid_clip(data: np.ndarray, slope_min: float = 1.0,
                 slope_max: float = 5.0,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    slope = _rng_uniform(rng, slope_min, slope_max)
    orig_rms = np.sqrt(np.mean(data ** 2) + 1e-8)
    out = 2 / (1 + np.exp(-slope * data)) - 1
    out_rms = np.sqrt(np.mean(out ** 2) + 1e-8)
    return out * (orig_rms / (out_rms + 1e-8))


def sox_clip(data: np.ndarray, threshold_db: float = -10.0) -> np.ndarray:
    """SoX-style soft clipping using tanh compression.
    
    Ported from se_train perturb.py.
    
    Args:
        data: Input audio (1D array)
        threshold_db: Threshold in dB (negative = below 0dBFS)
        
    Returns:
        Clipped audio (1D array)
    """
    threshold = 10 ** (threshold_db / 20)
    # Normalize to threshold, apply tanh, denormalize
    return threshold * np.tanh(data / threshold)


def pedal_clip(data: np.ndarray, sr: int,
               threshold_db: float = -10.0) -> np.ndarray:
    """Pedalboard-based clipping simulation.
    
    Uses pedalboard's Clip or Compressor if available, otherwise falls back.
    
    Args:
        data: Input audio (1D array)
        sr: Sample rate in Hz
        threshold_db: Threshold in dB
        
    Returns:
        Clipped audio (1D array)
    """
    pb = _get_pedalboard()
    if pb is None:
        # Fallback to sox-style clipping
        return sox_clip(data, threshold_db)
    
    try:
        # Use pedalboard's gain staging for clipping simulation
        threshold = 10 ** (threshold_db / 20)
        # Apply light compression then gain
        comp = pb.Compressor(threshold_db=-40, ratio=2.0, attack_ms=1, release_ms=100)
        gain = threshold * 2  # Boost to clip
        data_boosted = data * gain
        clipped = comp(data_boosted.astype(np.float32), sr)
        return np.clip(clipped, -1.0, 1.0).astype(np.float32)
    except Exception:
        _log.warning("pedal_clip failed, falling back to sox_clip", exc_info=True)
        return sox_clip(data, threshold_db)


def _fixed_rail_clipping_facts(
    clipped_mask: np.ndarray,
    sr: int,
) -> dict[str, int | float]:
    """汇总 fixed-rail 命中；帧为不重叠 20 ms，最后不足一帧也计一帧。"""
    flat_mask = np.asarray(clipped_mask, dtype=bool).reshape(-1)
    sample_count = int(np.count_nonzero(flat_mask))
    sample_total = int(flat_mask.size)

    # 20 ms 换算为整数 sample；仓库支持的常用 sample rate 均可整除。
    frame_length = max(1, int(sr * 20 / 1000))
    frame_total = (
        (sample_total + frame_length - 1) // frame_length if sample_total else 0
    )
    clipped_indices = np.flatnonzero(flat_mask)
    frame_count = int(np.unique(clipped_indices // frame_length).size)

    if sample_count:
        padded = np.pad(flat_mask.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1)
        longest_run = int(np.max(run_ends - run_starts))
    else:
        longest_run = 0

    return {
        "clipped_sample_count": sample_count,
        "clipped_sample_ratio": sample_count / sample_total if sample_total else 0.0,
        "clipped_frame_count": frame_count,
        "clipped_frame_ratio": frame_count / frame_total if frame_total else 0.0,
        "longest_clipped_run_samples": longest_run,
    }


def speaker_distortion(
    data: np.ndarray,
    sr: int,
    cfg,
    rng: Optional[np.random.Generator] = None,
    facts_out: Optional[dict] = None,
) -> np.ndarray:
    """执行一个 clipping variant，并记录 observation 级 realized facts。

    ``fixed_rail`` 的 clipped 统计只依据 pre-drive 后、``np.clip`` 前真正越过
    rail 的 waveform；``changed_sample_*`` 仍统计输入到输出的全部数值变化。
    """
    if facts_out is not None:
        facts_out.clear()
    source = np.asarray(data, dtype=np.float32)
    pre_peak = float(np.max(np.abs(source))) if source.size else 0.0
    variants = list(cfg["variants"])
    variant_probs = cfg["variant_probs"]
    probs = [variant_probs[variant] for variant in variants]
    variant = _rng_choices_weighted(rng, variants, probs, k=1)[0]
    pre_drive_db = None
    rail = None
    clipped_mask = None

    if variant == "fixed_rail":
        pre_drive_db = _rng_uniform(
            rng,
            cfg.get("pre_drive_db_min", 0.0),
            cfg.get("pre_drive_db_max", 0.0),
        )
        rail = 1.0
        driven = source * np.float32(10.0 ** (pre_drive_db / 20.0))
        clipped_mask = np.abs(driven) > rail
        output = np.clip(driven, -rail, rail)
    elif variant == "hard":
        hard_portion = cfg.get("hard_clip_portion", 0.5)
        use_configured = _rng_random(rng) < hard_portion
        use_rate = cfg.get("hard_clip_on_rate", True)
        if not use_configured:
            use_rate = not use_rate
        if use_rate:
            output = hard_clip_on_rate(
                source,
                cfg.get("clip_rate_min", 0.01),
                cfg.get("clip_rate_max", 0.2),
                rng=rng,
            )
        else:
            output = hard_clip_db(
                source,
                cfg.get("threshold_db_min", -20),
                cfg.get("threshold_db_max", -1),
                rng=rng,
            )
    elif variant == "soft":
        output = soft_clip(source, cfg.get("soft_slope_min", 1.0), cfg.get("soft_slope_max", 5.0), rng=rng)
    elif variant == "sigmoid":
        output = sigmoid_clip(source, cfg.get("sigmoid_slope_min", 1.0), cfg.get("sigmoid_slope_max", 5.0), rng=rng)
    elif variant == "sox":
        threshold_db = _rng_uniform(rng, cfg.get("sox_threshold_db_min", -15), cfg.get("sox_threshold_db_max", -5))
        output = sox_clip(source, threshold_db)
    elif variant == "pedal":
        threshold_db = _rng_uniform(rng, cfg.get("pedal_threshold_db_min", -15), cfg.get("pedal_threshold_db_max", -5))
        output = pedal_clip(source, sr, threshold_db)
    else:  # schema validator 保证不可达；保留 direct-call fail-closed。
        raise ValueError(f"unsupported clipping variant {variant!r}")

    output = np.asarray(output, dtype=np.float32)
    changed_count = int(np.count_nonzero(output != source))
    if facts_out is not None:
        rail_facts = (
            _fixed_rail_clipping_facts(clipped_mask, sr)
            if clipped_mask is not None
            else {
                "clipped_sample_count": None,
                "clipped_sample_ratio": None,
                "clipped_frame_count": None,
                "clipped_frame_ratio": None,
                "longest_clipped_run_samples": None,
            }
        )
        facts_out.update(
            {
                "variant": str(variant),
                "pre_drive_db": None if pre_drive_db is None else float(pre_drive_db),
                "rail": rail,
                "pre_peak": pre_peak,
                "post_peak": float(np.max(np.abs(output))) if output.size else 0.0,
                "changed_sample_count": changed_count,
                "changed_sample_ratio": changed_count / source.size if source.size else 0.0,
                **rail_facts,
                "applied": changed_count > 0,
            }
        )
    return output


# ===================================================================
# P1-2  Low-pass filter
# ===================================================================
def lowpass_filter(data: np.ndarray, sr: int,
                   min_cutoff: float = 1000, max_cutoff: float = 8000,
                   min_order: int = 4, max_order: int = 20,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    nyquist = sr / 2.0
    cutoff_hi = min(float(max_cutoff), np.nextafter(nyquist, 0.0))
    cutoff_lo = max(0.0, float(min_cutoff))
    if cutoff_hi <= 0.0 or cutoff_lo > cutoff_hi:
        _log.debug(
            "skip lowpass_filter: no feasible cutoff for sr=%s, configured=[%s, %s]",
            sr, min_cutoff, max_cutoff,
        )
        return data
    cutoff = _rng_uniform(rng, cutoff_lo, cutoff_hi)
    if _rng_random(rng) < 0.2:
        orig_len = len(data)
        spec = librosa.stft(data.astype(np.float32), n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        spec[freqs > cutoff] = 0
        return librosa.istft(spec, n_fft=2048, hop_length=512, length=orig_len)
    order = _rng_randint(rng, min_order, max_order)
    try:
        sos = scipy_signal.butter(order, cutoff, "lp", fs=sr, output="sos")
        return scipy_signal.sosfilt(sos, data).astype(np.float32)
    except Exception:
        _log.warning("lowpass_filter butter/sosfilt failed", exc_info=True)
        return data


# ===================================================================
# P1-3  Colored noise  (white / pink / brown / equalized)
# ===================================================================
def _gen_power_law_noise(
    n: int,
    sr: int,
    psd_exponent: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """按物理 Hz 频率轴生成 ``PSD ∝ f**(-psd_exponent)`` 噪声。"""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")

    white = _rng_normal(rng, 0, 1, n)
    spectrum = np.fft.rfft(white)
    freqs_hz = np.fft.rfftfreq(n, d=1.0 / float(sr))
    amplitude = np.zeros_like(freqs_hz)
    positive = freqs_hz > 0.0
    amplitude[positive] = freqs_hz[positive] ** (-0.5 * psd_exponent)
    spectrum *= amplitude
    noise = np.fft.irfft(spectrum, n=n)
    noise -= np.mean(noise)
    mx = float(np.max(np.abs(noise)))
    if mx > 0.0:
        noise /= mx
    return noise.astype(np.float32, copy=False)


def _gen_pink(
    n: int, sr: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    return _gen_power_law_noise(n, sr, 1.0, rng=rng)


def _gen_brown(
    n: int, sr: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    return _gen_power_law_noise(n, sr, 2.0, rng=rng)


def _gen_equalized(n: int, sr: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    white = _rng_normal(rng, 0, 1, n)
    nyq = sr / 2
    max_freq = max(200, nyq * 0.8)
    min_freq = min(100, max_freq * 0.5)
    num_bands = _rng_randint(rng, 1, 10)
    freqs = np.geomspace(min_freq, max_freq, num_bands)
    for i in range(num_bands):
        db = _rng_uniform(rng, -20, 20)
        try:
            b, a = scipy_signal.iirpeak(freqs[i], 1.0, fs=sr)
            adj = scipy_signal.lfilter(b, a, white)
            res = white - adj
            white = res + adj * 10 ** (db / 20)
        except Exception:
            _log.warning("_equalize_band iirpeak failed for band %s", i, exc_info=True)
            pass
    mx = np.max(np.abs(white))
    return white / mx if mx > 0 else white


def colored_noise(data: np.ndarray, sr: int,
                  snr_min: float = 10, snr_max: float = 50,
                  color_types=("white", "pink", "brown", "equalized"),
                  rng: Optional[np.random.Generator] = None,
                  audit_out: Optional[dict] = None) -> np.ndarray:
    """生成 colored noise，并复用 ordinary ``mix_noise`` 的 active-power 标定。"""
    color = _rng_choice(rng, color_types)
    n = len(data)
    if color == "white":
        noise = _rng_normal(rng, 0, 1, n)
    elif color == "pink":
        noise = _gen_pink(n, sr, rng=rng)
    elif color == "brown":
        noise = _gen_brown(n, sr, rng=rng)
    else:
        noise = _gen_equalized(n, sr, rng=rng)
    snr = _rng_uniform(rng, snr_min, snr_max)
    mix_audit = {} if audit_out is not None else None
    augmented, _ = mix_noise(
        np.asarray(data).reshape(1, -1),
        np.asarray(noise, dtype=np.float32).reshape(1, -1),
        snr=float(snr),
        rng=rng,
        audit_out=mix_audit,
    )
    if audit_out is not None:
        audit_out.update({"color": str(color), **mix_audit})
    return augmented[0]


# ===================================================================
# P1-4  DRC  (dynamic range compression via pedalboard)
# ===================================================================
def drc(data: np.ndarray, sr: int,
        threshold_db_min: float = -50, threshold_db_max: float = 0,
        ratio_min: float = 1, ratio_max: float = 20,
        attack_ms_min: float = 0.5, attack_ms_max: float = 5.0,
        release_ms_min: float = 50, release_ms_max: float = 1000,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    pb = _get_pedalboard()
    if pb is None:
        return data
    inst = pb.Compressor(
        threshold_db=_rng_uniform(rng, threshold_db_min, threshold_db_max),
        ratio=_rng_uniform(rng, ratio_min, ratio_max),
        attack_ms=_rng_uniform(rng, attack_ms_min, attack_ms_max),
        release_ms=_rng_uniform(rng, release_ms_min, release_ms_max),
    )
    return inst(data.astype(np.float32), sr)


# ===================================================================
# P1-5  Codecs  (MP3 / GSM / OPUS / OGG)
# ===================================================================
def codec_mp3(data: np.ndarray, sr: int,
              vbr_min: float = 0.5, vbr_max: float = 9.5,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    pb = _get_pedalboard()
    if pb is None:
        return data
    vbr = _rng_uniform(rng, vbr_min, vbr_max)
    return pb.MP3Compressor(vbr_quality=vbr)(data.astype(np.float32), sr)


def codec_gsm(data: np.ndarray, sr: int) -> np.ndarray:
    pb = _get_pedalboard()
    if pb is None:
        return data
    return pb.GSMFullRateCompressor()(data.astype(np.float32), sr)


def codec_opus(data: np.ndarray, sr: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """OPUS codec using torchaudio AudioEffector (no opuslib dependency)."""
    import torch
    tio = _get_torchaudio_io()
    if tio is None:
        return data
    qscale = _rng_randint(rng, 1, 10)
    try:
        eff = tio.AudioEffector(format="ogg", encoder="opus",
                                codec_config=tio.CodecConfig(qscale=qscale), pad_end=True)
        t = torch.from_numpy(data.astype(np.float32))
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # (T,) -> (T, 1); FFmpeg backend expects (T, C) to avoid channel_layout=0x0
        else:
            t = t.T  # (C, T) -> (T, C)
        out = eff.apply(t, sr).numpy()
        if out.ndim == 2:
            out = out[:, 0]
        if len(out) < len(data):
            out = np.pad(out, (0, len(data) - len(out)))
        return out[: len(data)]
    except Exception:
        _log.warning("codec_opus failed", exc_info=True)
        return data


def codec_aac(data: np.ndarray, sr: int,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """AAC codec using torchaudio AudioEffector."""
    import torch
    tio = _get_torchaudio_io()
    if tio is None:
        return data
    try:
        eff = tio.AudioEffector(format="adts", encoder="aac",
                                codec_config=tio.CodecConfig(qscale=_rng_randint(rng, 1, 5)),
                                pad_end=True)
        t = torch.from_numpy(data.astype(np.float32))
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # (T,) -> (T, 1); FFmpeg backend expects (T, C)
        else:
            t = t.T  # (C, T) -> (T, C)
        out = eff.apply(t, sr).numpy()
        if out.ndim == 2:
            out = out[:, 0]
        if len(out) < len(data):
            out = np.pad(out, (0, len(data) - len(out)))
        return out[: len(data)]
    except Exception:
        _log.warning("codec_aac failed", exc_info=True)
        return data


def apply_codec(data: np.ndarray, sr: int,
                codec_types=("mp3", "gsm", "opus"),
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    codec = _rng_choice(rng, codec_types)
    if codec == "mp3":
        return codec_mp3(data, sr, rng=rng)
    elif codec == "gsm":
        return codec_gsm(data, sr)
    elif codec == "opus":
        return codec_opus(data, sr, rng=rng)
    elif codec == "aac":
        return codec_aac(data, sr, rng=rng)
    return data


# ===================================================================
# P1-6  Band reject  (notch filter)
# ===================================================================
def band_reject(data: np.ndarray, sr: int,
                min_center_freq: float = 100, max_center_freq: float = 8000,
                min_freq_bw: float = 20, max_freq_bw: float = 500,
                use_stft: bool = True, max_n: int = 2,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    nyquist = sr / 2.0
    center_lo = max(0.0, float(min_center_freq))
    center_hi = min(float(max_center_freq), np.nextafter(nyquist, 0.0))
    if center_lo > center_hi:
        _log.debug(
            "skip band_reject: no feasible center frequency for sr=%s, configured=[%s, %s]",
            sr, min_center_freq, max_center_freq,
        )
        return data
    n = _rng_randint(rng, 1, max_n)
    if use_stft:
        orig_len = len(data)
        spec = librosa.stft(data.astype(np.float32), n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        for _ in range(n):
            cf = _rng_uniform(rng, center_lo, center_hi)
            bw = _rng_uniform(rng, min_freq_bw, max_freq_bw)
            bw = min(bw, cf / 2)
            mask = np.logical_and(freqs >= cf - bw / 2, freqs <= cf + bw / 2)
            spec[mask] = 0
        return librosa.istft(spec, n_fft=2048, hop_length=512, length=orig_len)
    else:
        for _ in range(n):
            cf = _rng_uniform(rng, center_lo, center_hi)
            q = _rng_uniform(rng, 1, 8)
            try:
                b, a = scipy_signal.iirnotch(cf, q, fs=sr)
                data = scipy_signal.lfilter(b, a, data).astype(np.float32)
            except Exception:
                _log.warning("band_reject iirnotch failed for cf=%s", cf, exc_info=True)
                pass
        return data


# ===================================================================
# P1-7  DC offset
# ===================================================================
def dc_offset(data: np.ndarray, min_offset: float = 0.001,
              max_offset: float = 0.2,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    offset = _rng_uniform(rng, min_offset, max_offset)
    if _rng_random(rng) < 0.5:
        offset = -offset
    return data + offset


# ===================================================================
# P1-8  Bit crush
# ===================================================================
def bit_crush(data: np.ndarray, bit_min: int = 4, bit_max: int = 16,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    bits = _rng_randint(rng, bit_min, bit_max)
    levels = 2 ** bits
    mx = np.max(np.abs(data)) + 1e-8
    normed = data / mx
    quantized = np.round(normed * (levels / 2)) / (levels / 2)
    return quantized * mx


# ===================================================================
# P1-9  Spectral time-frequency holes (se_train style)
# ===================================================================
def spectral_time_freq_holes(data: np.ndarray, sr: int,
                             frame_length: int = 1024,
                             hop_length: int = 256,
                             holes_num_min: int = 1,
                             holes_num_max: int = 20,
                             holes_width_freq_min: int = 1,
                             holes_width_freq_max: int = 5,
                             holes_width_time_min: int = 1,
                             holes_width_time_max: int = 5,
                             cutoff_freq: float = 10000,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply random spectral holes in time-frequency domain.
    
    This simulates frequency-selective fading or interference in transmission.
    Ported from se_train comm_distort_simu_dataset.py.
    
    Args:
        data: Input audio (1D array)
        sr: Sample rate in Hz
        frame_length: STFT frame length
        hop_length: STFT hop length
        holes_num_min: Minimum number of holes
        holes_num_max: Maximum number of holes
        holes_width_freq_min: Minimum hole width in frequency bins
        holes_width_freq_max: Maximum hole width in frequency bins
        holes_width_time_min: Minimum hole width in time frames
        holes_width_time_max: Maximum hole width in time frames
        cutoff_freq: Maximum frequency to apply holes (Hz)
        
    Returns:
        Audio with spectral holes (1D array)
    """
    import librosa
    
    if not np.isfinite(cutoff_freq) or cutoff_freq < 0.0:
        raise ValueError(f"cutoff_freq must be finite and non-negative, got {cutoff_freq!r}")
    if cutoff_freq <= 0.0 or sr <= 0:
        _log.debug("skip spectral_time_freq_holes: cutoff=%s, sr=%s", cutoff_freq, sr)
        return data
    orig_len = len(data)
    spec = librosa.stft(data.astype(np.float32), n_fft=frame_length, hop_length=hop_length)
    n_freq, n_time = spec.shape
    
    # Calculate frequency cutoff bin after clamping it to the current Nyquist.
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    cutoff_bin = min(np.searchsorted(freq_bins, min(cutoff_freq, sr / 2.0)), n_freq)
    
    # Number of holes to apply
    n_holes = _rng_randint(rng, holes_num_min, holes_num_max)
    
    for _ in range(n_holes):
        # Random hole dimensions
        hole_width_freq = _rng_randint(rng, holes_width_freq_min, holes_width_freq_max)
        hole_width_time = _rng_randint(rng, holes_width_time_min, holes_width_time_max)
        
        # Random position (within cutoff_freq for frequency)
        if cutoff_bin <= hole_width_freq:
            continue
        freq_start = _rng_randint(rng, 0, cutoff_bin - hole_width_freq)
        time_start = _rng_randint(rng, 0, max(0, n_time - hole_width_time))
        
        # Apply hole (zero out region)
        freq_end = min(freq_start + hole_width_freq, n_freq)
        time_end = min(time_start + hole_width_time, n_time)
        spec[freq_start:freq_end, time_start:time_end] = 0
    
    return librosa.istft(spec, n_fft=frame_length, hop_length=hop_length, length=orig_len)


# ===================================================================
# P1-10  WebRTC Noise Suppression / AGC (se_train style - simulated)
# ===================================================================
def webrtc_noise_suppression(data: np.ndarray, sr: int,
                              aggressiveness: int = 3) -> np.ndarray:
    """Simulated WebRTC NS via simple noise gating/spectral subtraction.
    
    Since actual WebRTC NS requires the webrtcvad/webrtc-noise-suppression libraries,
    this implementation provides a simplified simulation using spectral gating.
    For production use, consider using the actual WebRTC libraries.
    
    Args:
        data: Input audio (1D array)
        sr: Sample rate in Hz
        aggressiveness: Aggressiveness level 0-3 (3 = most aggressive)
        
    Returns:
        Noise-suppressed audio (1D array)
    """
    import librosa
    
    orig_len = len(data)
    
    # STFT parameters based on aggressiveness
    n_fft = 2048 if aggressiveness >= 2 else 1024
    hop = n_fft // 4
    
    # Compute STFT
    spec = librosa.stft(data.astype(np.float32), n_fft=n_fft, hop_length=hop)
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    
    # Estimate noise floor using median filter
    noise_floor = np.median(magnitude, axis=1, keepdims=True)
    
    # Aggressiveness determines threshold multiplier
    threshold_mult = [1.5, 2.0, 2.5, 3.5][min(aggressiveness, 3)]
    
    # Spectral subtraction with masking
    threshold = noise_floor * threshold_mult
    mask = magnitude > threshold
    suppressed = magnitude * mask.astype(np.float32)
    
    # ISTFT
    spec_out = suppressed * np.exp(1j * phase)
    return librosa.istft(spec_out, n_fft=n_fft, hop_length=hop, length=orig_len)


def webrtc_agc(data: np.ndarray, sr: int,
               target_dbfs_min: float = -31,
               target_dbfs_max: float = -3,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simulated WebRTC AGC via RMS-based volume normalization.
    
    Since actual WebRTC AGC requires the webrtcvad/webrtc-audio-processing libraries,
    this implementation provides a simplified simulation.
    
    Args:
        data: Input audio (1D array)
        sr: Sample rate in Hz
        target_dbfs_min: Minimum target dBFS
        target_dbfs_max: Maximum target dBFS
        
    Returns:
        AGC-processed audio (1D array)
    """
    target_dbfs = _rng_uniform(rng, target_dbfs_min, target_dbfs_max)
    target_linear = 10 ** (target_dbfs / 20)
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(data ** 2) + 1e-8)
    
    # Compute gain with limiting
    gain = target_linear / (current_rms + 1e-8)
    
    # Limit gain to prevent extreme amplification (WebRTC AGC-like behavior)
    max_gain = 100.0  # 40dB max gain
    min_gain = 0.01   # -40dB min gain
    gain = np.clip(gain, min_gain, max_gain)
    
    return data * gain


# ===================================================================
# P1-11  Bass boost (se_train style)
# ===================================================================
def bass_boost(data: np.ndarray, sr: int,
               highpass_cutoff_min: float = 500,
               highpass_cutoff_max: float = 2000,
               attenuation_min_db: float = -25,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Create relative bass emphasis by preserving lows and attenuating highs.

    From se_train ``comm_distort_simu_dataset.py``: "boosting of the bass
    frequencies by attenuating the other frequencies". Despite the historical
    ``bass_boost`` name, this implementation does not add low-frequency gain:
    it subtracts part of the high-pass component, leaving frequencies below the
    sampled cutoff near unity while attenuating content above it. It therefore
    approximates an optional user/vendor bass, loudness, or speaker-compensation
    EQ preset; it does not model the inherent bass roll-off of a small speaker.
    Room and placement coloration belongs primarily to the RIR simulation.

    Args:
        data: Input audio (1D array)
        sr: Sample rate in Hz
        highpass_cutoff_min: Minimum highpass cutoff in Hz
        highpass_cutoff_max: Maximum highpass cutoff in Hz
        attenuation_min_db: Lower bound of the sampled above-cutoff gain in dB;
            the upper bound is fixed at -5 dB.

    Returns:
        Audio with lows preserved and above-cutoff content attenuated, producing
        a larger low-to-high frequency ratio.
    """
    if not np.isfinite(attenuation_min_db) or attenuation_min_db > -5.0:
        raise ValueError(
            "attenuation_min_db must be finite and <= -5 dB, "
            f"got {attenuation_min_db!r}"
        )
    cutoff_lo = max(0.0, float(highpass_cutoff_min))
    cutoff_hi = min(float(highpass_cutoff_max), np.nextafter(sr / 2.0, 0.0))
    if cutoff_lo <= 0.0 or cutoff_lo > cutoff_hi:
        _log.debug(
            "skip bass_boost: no feasible highpass cutoff for sr=%s, configured=[%s, %s]",
            sr, highpass_cutoff_min, highpass_cutoff_max,
        )
        return data
    cutoff = _rng_uniform(rng, cutoff_lo, cutoff_hi)
    
    # Design highpass filter
    order = _rng_randint(rng, 2, 6)
    try:
        sos = scipy_signal.butter(order, cutoff, "hp", fs=sr, output="sos")
        filtered = scipy_signal.sosfilt(sos, data).astype(np.float32)
        
        # Attenuate the filtered component
        attenuation_db = _rng_uniform(rng, attenuation_min_db, -5)
        attenuation_linear = 10 ** (attenuation_db / 20)
        
        # Mix: original + attenuated high-pass = bass boost effect
        return data - filtered * (1 - attenuation_linear)
    except Exception:
        _log.warning("bass_boost failed", exc_info=True)
        return data


# ===================================================================
# P1-12  Spectral leakage (se_train style)
# ===================================================================
def spectral_leakage(data: np.ndarray, sr: int,
                       window_lengths: tuple = (1024, 2048, 4096),
                       max_time_shift: int = 20,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simulate spectral leakage via time-shifted STFT recombination.
    
    From se_train: simulating windowing artifacts in STFT processing.
    
    Args:
        data: Input audio (1D array)
        sr: Sample rate in Hz
        window_lengths: Possible STFT window lengths
        max_time_shift: Maximum time shift in samples
        
    Returns:
        Audio with spectral leakage artifacts (1D array)
    """
    import librosa
    
    orig_len = len(data)
    n_fft = _rng_choice(rng, window_lengths)
    hop = n_fft // 4
    
    # Compute STFT
    spec = librosa.stft(data.astype(np.float32), n_fft=n_fft, hop_length=hop)
    
    # Apply random time shifts to different frequency bands
    n_bands = _rng_randint(rng, 2, 6)
    n_freq, n_time = spec.shape
    
    band_size = max(1, n_freq // n_bands)
    for i in range(n_bands):
        freq_start = i * band_size
        freq_end = min((i + 1) * band_size, n_freq)
        
        # Random time shift for this band
        shift = _rng_randint(rng, -max_time_shift, max_time_shift)
        if shift != 0:
            spec[freq_start:freq_end] = np.roll(
                spec[freq_start:freq_end], shift, axis=1
            )
    
    # ISTFT
    return librosa.istft(spec, n_fft=n_fft, hop_length=hop, length=orig_len)


# ===================================================================
# P1-13  Loudness perturbation (se_train style)
# ===================================================================
def loudness_perturbation(data: np.ndarray,
                          min_factor: float = 0.1,
                          max_factor: float = 10.0,
                          max_n_intervals: int = 5,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply loudness perturbation with random gain factors over intervals.
    
    From se_train comm_distort_simu_dataset.py.
    
    Args:
        data: Input audio (1D array)
        min_factor: Minimum gain factor
        max_factor: Maximum gain factor
        max_n_intervals: Maximum number of intervals
        
    Returns:
        Loudness-perturbed audio (1D array)
    """
    n = len(data)
    n_intervals = _rng_randint(rng, 1, max_n_intervals)
    
    if n_intervals == 1:
        # Single gain factor
        factor = _rng_uniform(rng, min_factor, max_factor)
        return data * factor
    
    # Multiple intervals with smooth transitions
    interval_size = n // n_intervals
    result = np.zeros_like(data)
    
    factors = [_rng_uniform(rng, min_factor, max_factor) for _ in range(n_intervals)]
    
    for i in range(n_intervals):
        start = i * interval_size
        end = min((i + 1) * interval_size, n)
        
        if i == 0:
            # First interval: fade in
            fade_len = min(100, (end - start) // 4)
            if fade_len > 1:
                fade = np.linspace(0, 1, fade_len) ** 2
                result[start:start + fade_len] = data[start:start + fade_len] * factors[i] * fade
                result[start + fade_len:end] = data[start + fade_len:end] * factors[i]
            else:
                result[start:end] = data[start:end] * factors[i]
        elif i == n_intervals - 1:
            # Last interval: handle transition from previous
            prev_end = start
            fade_len = min(100, (end - start) // 4)
            if fade_len > 1 and prev_end > 0:
                fade_out = np.linspace(1, 0, fade_len) ** 2
                fade_in = np.linspace(0, 1, fade_len) ** 2
                
                # Cross-fade
                result[prev_end - fade_len:prev_end] = (
                    result[prev_end - fade_len:prev_end] * fade_out +
                    data[prev_end - fade_len:prev_end] * factors[i] * fade_in
                )
                result[prev_end:end] = data[prev_end:end] * factors[i]
            else:
                result[start:end] = data[start:end] * factors[i]
        else:
            # Middle interval
            prev_end = start
            fade_len = min(100, (end - start) // 4)
            if fade_len > 1 and prev_end > 0:
                fade_out = np.linspace(1, 0, fade_len) ** 2
                fade_in = np.linspace(0, 1, fade_len) ** 2
                
                # Cross-fade from previous
                result[prev_end - fade_len:prev_end] = (
                    result[prev_end - fade_len:prev_end] * fade_out +
                    data[prev_end - fade_len:prev_end] * factors[i] * fade_in
                )
                result[prev_end:end] = data[prev_end:end] * factors[i]
            else:
                result[start:end] = data[start:end] * factors[i]
    
    return result


# ===================================================================
# P0-4  Post-mix EQ  (applied to noisy after noise mixing)
# ===================================================================
def _design_post_eq_sos(
    sr: int,
    db_min: float = -5,
    db_max: float = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """设计 1--5 段 RBJ peaking biquad，并约束级联累计 dB 增益。"""
    if isinstance(sr, (bool, np.bool_)) or not isinstance(sr, Real):
        raise ValueError(f"post_eq sample rate must be a number, got {sr!r}")
    sample_rate = float(sr)
    if not np.isfinite(sample_rate) or sample_rate < 400.0:
        raise ValueError(
            "post_eq sample rate must be finite and at least 400 Hz so the "
            "100 Hz center-frequency margins are valid"
        )

    db_lo, db_hi = float(db_min), float(db_max)
    if not np.isfinite(db_lo) or not np.isfinite(db_hi):
        raise ValueError("post_eq db_min and db_max must be finite")
    if db_lo > db_hi:
        raise ValueError(
            f"post_eq db_min must be <= db_max, got {db_lo} > {db_hi}"
        )

    num_bands = _rng_randint(rng, 1, 5)
    min_freq, max_freq = 100.0, sample_rate / 2.0 - 100.0
    bands = [
        (
            _rng_uniform(rng, min_freq, max_freq),
            _rng_uniform(rng, 0.5, 3.0),
            _rng_uniform(rng, db_lo, db_hi),
        )
        for _ in range(num_bands)
    ]

    gains_db = np.asarray([band[2] for band in bands], dtype=np.float64)
    positive_total = float(np.sum(np.maximum(gains_db, 0.0)))
    negative_total = float(np.sum(np.maximum(-gains_db, 0.0)))
    positive_budget = max(0.0, db_hi)
    negative_budget = max(0.0, -db_lo)
    if positive_total > positive_budget:
        gains_db[gains_db > 0.0] *= positive_budget / positive_total
    if negative_total > negative_budget:
        gains_db[gains_db < 0.0] *= negative_budget / negative_total

    sos = np.empty((num_bands, 6), dtype=np.float64)
    for index, ((center_freq, q, _), gain_db) in enumerate(zip(bands, gains_db)):
        # RBJ Audio EQ Cookbook peaking-EQ 公式；归一化后 a0 固定为 1。
        amplitude = 10.0 ** (float(gain_db) / 40.0)
        omega = 2.0 * np.pi * center_freq / sample_rate
        alpha = np.sin(omega) / (2.0 * q)
        cos_omega = np.cos(omega)
        a0 = 1.0 + alpha / amplitude
        sos[index] = (
            (1.0 + alpha * amplitude) / a0,
            (-2.0 * cos_omega) / a0,
            (1.0 - alpha * amplitude) / a0,
            1.0,
            (-2.0 * cos_omega) / a0,
            (1.0 - alpha / amplitude) / a0,
        )

    if not np.isfinite(sos).all():
        raise ValueError("post_eq SOS design produced non-finite coefficients")
    return sos


def post_eq(data: np.ndarray, sr: int,
            db_min: float = -5, db_max: float = 5,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """施加平滑、稳定、因果且累计增益有界的随机 peaking EQ。

    随机生成 1--5 段中心频率 100 Hz 至 Nyquist-100 Hz、Q 0.5--3.0
    的 RBJ peaking biquad。正、负 gain 分别按 ``max(0, db_max)`` 与
    ``max(0, -db_min)`` 的累计 dB 预算同比缩放，再通过 ``sosfilt`` 单向级联，
    因而不会产生 centered STFT 的 pre-ringing，并保持输入长度与 float32 输出。
    """
    sos = _design_post_eq_sos(sr, db_min, db_max, rng=rng)
    return scipy_signal.sosfilt(sos, np.asarray(data, dtype=np.float32)).astype(
        np.float32, copy=False
    )


# ===================================================================
# P0-2  RMS-based random volume + volume clip
# ===================================================================
def _fast_active_rms(w: np.ndarray, sr: int = 16000,
                     frame_ms: int = 20, threshold_db: float = -50) -> float:
    """Fast voiced-segment RMS (frame-energy based, no librosa VAD)."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    if len(w) < frame_len:
        return float(np.sqrt(np.mean(w ** 2) + 1e-8))
    n_frames = len(w) // frame_len
    frames = w[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    threshold = 10 ** (threshold_db / 20)
    active = rms_per_frame[rms_per_frame > threshold]
    if len(active) == 0:
        return float(np.sqrt(np.mean(w ** 2) + 1e-8))
    return float(np.sqrt(np.mean(active ** 2)))


def draw_source_volume_target_db(
    rng: Optional[np.random.Generator],
    volume_min_dB: float,
    volume_max_dB: float,
    distribution: str = "uniform",
    beta_a: float = 2.0,
    beta_b: float = 2.0,
) -> float:
    """抽取有限 target dB；仅在极端跨度溢出时使用稳定凸组合。"""
    if distribution not in ("uniform", "beta"):
        raise ValueError(
            f"rng_uniform_or_beta: unknown distribution {distribution!r}; "
            "expected 'uniform' or 'beta'"
        )
    if distribution == "beta":
        for name, value in (("beta_a", beta_a), ("beta_b", beta_b)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(
                    f"source volume {name} must be a finite positive number, got {value!r}"
                )

    lo, hi = float(volume_min_dB), float(volume_max_dB)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("source volume target dB must be finite")
    if np.isfinite(hi - lo):
        target_db = rng_uniform_or_beta(
            rng, lo, hi, distribution, beta_a, beta_b
        )
    else:
        if distribution == "uniform":
            u = _rng_random(rng)
        elif distribution == "beta":
            u = (
                float(rng.beta(beta_a, beta_b))
                if rng is not None
                else random.betavariate(beta_a, beta_b)
            )
        else:
            raise ValueError(
                f"rng_uniform_or_beta: unknown distribution {distribution!r}; "
                "expected 'uniform' or 'beta'"
            )
        # ``lo + u * (hi - lo)`` 会先让跨度溢出；凸组合保持数学等价且有限。
        target_db = (1.0 - u) * lo + u * hi
    target_db = float(target_db)
    if not np.isfinite(target_db):
        raise ValueError("source volume target dB must be finite")
    return target_db


_FLOAT32_SAFE_MAGNITUDE = float(np.finfo(np.float32).max)


def source_volume_shared_max_abs(*signals: np.ndarray) -> float:
    """校验 C-0 stems 可安全表示为 finite float32，并返回共同峰值。"""
    shared_peak = 0.0
    for signal in signals:
        raw_signal = np.asarray(signal)
        if not np.isfinite(raw_signal).all():
            raise ValueError("source volume stem must contain only finite samples")
        with np.errstate(over="ignore", invalid="ignore"):
            signal_32 = np.asarray(raw_signal, dtype=np.float32)
        if not np.isfinite(signal_32).all():
            raise ValueError("source volume stem exceeds the finite float32 range")
        if signal_32.size:
            shared_peak = max(shared_peak, float(np.max(np.abs(signal_32))))
    return shared_peak


def scale_source_volume_stem(
    signal: np.ndarray,
    gain: float,
    *,
    stem_name: str,
) -> np.ndarray:
    """用已收紧的唯一 gain 缩放一个 C-0 stem，并对 float32 结果 fail-closed。"""
    raw_signal = np.asarray(signal)
    if not np.isfinite(raw_signal).all():
        raise ValueError(f"source volume {stem_name} stem must contain only finite samples")
    gain = float(gain)
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError(f"source volume shared gain must be finite and non-negative, got {gain!r}")

    with np.errstate(over="ignore", invalid="ignore"):
        signal_32 = np.asarray(raw_signal, dtype=np.float32)
    if not np.isfinite(signal_32).all():
        raise ValueError(
            f"source volume {stem_name} stem exceeds the finite float32 range"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(signal_32 * gain, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"source volume scaling produced non-finite {stem_name} stem")
    return out


def source_volume_normalize(
    signal: np.ndarray,
    volume_min_dB: float = -45,
    volume_max_dB: float = -10,
    use_rms: bool = True,
    sr: int = 16000,
    rng: Optional[np.random.Generator] = None,
    distribution: str = "uniform",
    beta_a: float = 2.0,
    beta_b: float = 2.0,
    *,
    forced_gain: float | None = None,
    shared_stem_max_abs: float | None = None,
    calibration_out: Optional[dict] = None,
) -> tuple[np.ndarray, float]:
    """按目标音量缩放单个 reference，并保持二元返回 API。

    默认从配置范围抽取目标 dB，并根据 ``signal``（runtime 使用 post-RIR early
    reference）计算 gain。``forced_gain`` 用于 TSE replay：直接复用已标定 gain，
    不抽取目标音量 RNG；此时第二个返回值是施加该 gain 后的 reference 电平。

    Args:
        distribution: "uniform" (default) or "beta". See
            ``core.data.simulation_utils.rng_uniform_or_beta`` for semantics.
        beta_a, beta_b: Only used when ``distribution == "beta"``.
        forced_gain: 非 ``None`` 时原样施加该 gain，且不消耗音量随机数；若 pipeline
            共同峰值表明该 gain 不安全则 fail-closed，禁止静默二次收紧。
        shared_stem_max_abs: pipeline 可传入 full/early/target 的共同峰值；普通候选
            gain 必要时可收紧，forced replay 只校验。直接调用默认语义不变。
        calibration_out: 非 positional sidecar，写入 target/reference/gain 标定信息。

    Returns:
        ``(scaled_signal, target_dB)``；保持二元组返回。
    """
    raw_signal = np.asarray(signal)
    if not np.isfinite(raw_signal).all():
        raise ValueError("source volume reference must contain only finite samples")
    signal_array = np.asarray(raw_signal, dtype=np.float32)
    if not np.isfinite(signal_array).all():
        raise ValueError("source volume reference exceeds the finite float32 range")

    signal_peak = float(np.max(np.abs(signal_array))) if signal_array.size else 0.0
    reference_all_zero = signal_peak == 0.0
    reference_near_silent = not reference_all_zero and signal_peak <= 1e-7
    if use_rms:
        cur_vol = _fast_active_rms(signal_array.ravel(), sr)
    else:
        cur_vol = signal_peak + 1e-8
    if not np.isfinite(cur_vol) or cur_vol < 0.0:
        raise ValueError("source volume reference level must be finite and non-negative")

    min_gain, max_gain = 1e-3, 1e3
    gain_clipped = False
    if forced_gain is None:
        target_db_requested = draw_source_volume_target_db(
            rng,
            volume_min_dB,
            volume_max_dB,
            distribution,
            beta_a,
            beta_b,
        )

        denominator = cur_vol + 1e-6
        log10_raw_gain = target_db_requested / 20.0 - float(np.log10(denominator))
        if log10_raw_gain > np.log10(max_gain):
            scale = max_gain
            gain_clipped = True
        elif log10_raw_gain < np.log10(min_gain):
            scale = min_gain
            gain_clipped = True
        else:
            target_vol = 10 ** (target_db_requested / 20.0)
            raw_scale = target_vol / denominator
            scale = float(np.clip(raw_scale, min_gain, max_gain))
            gain_clipped = scale != raw_scale
        target_db = target_db_requested
    else:
        scale = float(forced_gain)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(f"forced_gain must be finite and non-negative, got {forced_gain!r}")
        target_db = max(
            -240.0,
            20.0 * float(np.log10(max(cur_vol, 1e-12)))
            + 20.0 * float(np.log10(max(scale, 1e-12))),
        )
        target_db_requested = target_db

    shared_stem_limited = False
    if shared_stem_max_abs is None:
        safe_peak = signal_peak
    else:
        safe_peak = float(shared_stem_max_abs)
        if not np.isfinite(safe_peak) or safe_peak < 0.0:
            raise ValueError(
                "shared_stem_max_abs must be finite and non-negative, "
                f"got {shared_stem_max_abs!r}"
            )
    if safe_peak > 0.0:
        safe_magnitude = (
            _FLOAT32_SAFE_MAGNITUDE
            if shared_stem_max_abs is not None
            else float(np.finfo(np.float32).max)
        )
        float32_safe_gain = safe_magnitude / safe_peak
        if scale > float32_safe_gain:
            if forced_gain is not None and shared_stem_max_abs is not None:
                raise ValueError(
                    "forced source volume gain is unsafe for C-0 stems: "
                    f"gain={scale!r}, safe_max={float32_safe_gain!r}"
                )
            scale = float32_safe_gain
            gain_clipped = True
            shared_stem_limited = shared_stem_max_abs is not None

    out = scale_source_volume_stem(signal_array, scale, stem_name="reference")

    if shared_stem_limited:
        status, reason = "applied", "shared_stem_float32_limit"
    elif reference_all_zero:
        status, reason = "skipped", "reference_all_zero"
    elif reference_near_silent and gain_clipped:
        status, reason = "applied", "reference_near_silent_gain_clamped"
    elif reference_near_silent:
        status, reason = "applied", "reference_near_silent"
    elif gain_clipped:
        status, reason = "applied", "gain_clamped"
    else:
        status, reason = "applied", None

    if use_rms:
        after_vol = _fast_active_rms(out.ravel(), sr)
    else:
        after_vol = (float(np.max(np.abs(out))) if out.size else 0.0) + 1e-8
    reference_db_before = (
        -240.0
        if reference_all_zero
        else 20.0 * float(np.log10(max(cur_vol, 1e-12)))
    )
    reference_db_after = (
        -240.0
        if reference_all_zero
        else 20.0 * float(np.log10(max(after_vol, 1e-12)))
    )
    if calibration_out is not None:
        calibration_out.update(
            {
                "target_db": float(target_db),
                "target_db_requested": float(target_db_requested),
                "reference_db_before": reference_db_before,
                "reference_db_after": reference_db_after,
                "gain": float(scale),
                "gain_clipped": bool(gain_clipped),
                "forced_gain": forced_gain is not None,
                "reference": "early_rir",
                "status": status,
                "reason": reason,
            }
        )
    return out, float(target_db)


# ===================================================================
# Dispatcher  – apply all configured post-mix augmentations
# ===================================================================
def apply_post_augmentations(noisy: np.ndarray,
                             sr: int, cfg,
                             skip_clip: bool = False,
                             skip_codec: bool = False,
                             rng: Optional[np.random.Generator] = None,
                             applied_ops_out: Optional[list] = None,
                             clipping_facts_out: Optional[dict] = None,
                             sanitize_nonfinite: bool = False) -> np.ndarray:
    """Apply **Layer F**: noisy-only post-mix degradations (distortion / device-chain simulation).

    **Canonical training path** (``DynamicMixingDataset._run``): **Layer E** (URGENT2026
    channel degradation) runs **above** in ``_run``; then this function applies **only**
    Layer F items. Do not confuse this module with pipeline Layer D.

    Layer F execution order (when ``noisy_post_mix_degradations_enabled`` is True; noisy-only):
      F-6. DRC          (mic preamp compression)
      F-7. Codecs       (transmission encoding MP3/GSM/Opus)
      F-1. Clipping     (mic overload / device distortion)
      F-2. Post-mix EQ  (device frequency response)
      F-3. Lowpass filter (device bandwidth limit)
      F-4. Band reject  (resonance / notch)
      F-5. DC offset    (amplifier DC bias)
      F-8. Bit crush    (low-quality ADC)
      F-9. Spectral holes (time-freq dropouts)
      F-10. WebRTC NS   (noise suppression)
      F-11. WebRTC AGC  (auto gain control)
      F-12. Bass boost  (low-shelf emphasis)
      F-13. Spectral leakage (windowing artifacts)
      F-14. Loudness perturbation (overall level jitter)

    注：colored noise（彩色噪声）已迁至 **Layer C-3**（声学混合层，
    ``DynamicMixingDatasetBase._mix_acoustic``）——物理上属环境加性噪声而非
    设备链失真，此处不再施加、不占编号。编号与 `Simulation_Flow_Logic.md`
    §0.7 / 生产配置注释 / `SimulationConfigs` 字段注释统一（本列表按
    **执行顺序**排列，故编号非单调：DRC/Codecs 因依赖 pedalboard 最先执行）。

    Args:
        noisy:     1-D noisy waveform (the mic signal, including echo when applicable).
        sr:        Sample rate in Hz.
        cfg:       ``SimulationConfigs`` object with Layer-F fields.
        skip_clip: If True, skip only the Layer F clipping operation because
                   Layer E channel clipping was actually applied. Other F ops
                   remain enabled; AEC C-aec echo clipping must not set this gate.
        skip_codec: If True, skip codecs (already handled by Layer E urgent2026 codec).
        rng: Per-sample random generator for reproducibility. Falls back to
             global ``random`` when *None*.
        applied_ops_out: 可选 list；若提供，每个实际触发的 Layer F 操作名
             （如 ``"drc"`` / ``"clip"``）按执行顺序 append 进去，
             供调用方判定 realized 指标是否可分解（JAE-H03）。
        clipping_facts_out: 可选 dict；调用开始时清空，若抽中 clipping 则写入
             observation 级 variant、pre-drive、peak 与实际变更样本统计。
        sanitize_nonfinite: 仅 lenient contract 显式设为 True 时，将 Layer F 输出中的
             NaN/Inf 转为零；默认 False，让 strict/post-finite contract fail-fast。

    Returns:
        The degraded noisy waveform (1-D ``np.ndarray``).

    Note:
        Layer F is **noisy-only** by design — it simulates device-chain degradations
        (mic preamp, ADC, codec pipeline) that affect the recorded mic signal, not the
        training target. The ``clean`` target is not passed through this function.
    """

    if clipping_facts_out is not None:
        clipping_facts_out.clear()
    layer_f_on = getattr(cfg, "noisy_post_mix_degradations_enabled", False)

    if layer_f_on:
        # --- Layer F operations ---
        if _rng_random(rng) < cfg.prob_drc:
            _drc_thr_lo, _drc_thr_hi = cfg.drc_threshold_db_range
            _drc_ratio_lo, _drc_ratio_hi = cfg.drc_ratio_range
            _drc_atk_lo, _drc_atk_hi = cfg.drc_attack_ms_range
            _drc_rel_lo, _drc_rel_hi = cfg.drc_release_ms_range
            noisy = drc(noisy, sr, _drc_thr_lo, _drc_thr_hi,
                        _drc_ratio_lo, _drc_ratio_hi,
                        _drc_atk_lo, _drc_atk_hi,
                        _drc_rel_lo, _drc_rel_hi, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("drc")

        if not skip_codec and _rng_random(rng) < cfg.prob_codecs:
            noisy = apply_codec(noisy, sr, cfg.codecs_types, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("codec")

        if not skip_clip and _rng_random(rng) < cfg.prob_clip:
            # Build clip configuration including variant probabilities.
            # NOTE: the dict keys below (clip_rate_min/max, threshold_db_min/max, ...)
            # are speaker_distortion()'s own internal interface, unrelated to the
            # SimulationConfigs *_range attribute names unpacked on the right-hand
            # side -- do not rename these keys if the SimulationConfigs fields change.
            _clip_drive_lo, _clip_drive_hi = getattr(
                cfg, "clip_pre_drive_db_range", (0.0, 0.0)
            )
            _clip_rate_lo, _clip_rate_hi = getattr(cfg, "clip_rate_range", (0.01, 0.2))
            _clip_thr_lo, _clip_thr_hi = getattr(cfg, "clip_threshold_db_range", (-20, -1))
            _clip_soft_lo, _clip_soft_hi = getattr(cfg, "clip_soft_slope_range", (1.0, 5.0))
            _clip_sig_lo, _clip_sig_hi = getattr(cfg, "clip_sigmoid_slope_range", (1.0, 5.0))
            _clip_sox_lo, _clip_sox_hi = getattr(cfg, "clip_sox_threshold_db_range", (-15, -5))
            _clip_pedal_lo, _clip_pedal_hi = getattr(cfg, "clip_pedal_threshold_db_range", (-15, -5))
            clip_cfg = {
                "pre_drive_db_min": _clip_drive_lo,
                "pre_drive_db_max": _clip_drive_hi,
                "hard_clip_portion": getattr(cfg, "clip_hard_portion", 0.5),
                "hard_clip_on_rate": getattr(cfg, "clip_hard_on_rate", True),
                "clip_rate_min": _clip_rate_lo,
                "clip_rate_max": _clip_rate_hi,
                "threshold_db_min": _clip_thr_lo,
                "threshold_db_max": _clip_thr_hi,
                "soft_slope_min": _clip_soft_lo,
                "soft_slope_max": _clip_soft_hi,
                "sigmoid_slope_min": _clip_sig_lo,
                "sigmoid_slope_max": _clip_sig_hi,
                "sox_threshold_db_min": _clip_sox_lo,
                "sox_threshold_db_max": _clip_sox_hi,
                "pedal_threshold_db_min": _clip_pedal_lo,
                "pedal_threshold_db_max": _clip_pedal_hi,
                "variants": tuple(cfg.clip_variants),
                "variant_probs": dict(cfg.clip_variant_probs),
            }
            facts = {} if clipping_facts_out is None else clipping_facts_out
            noisy = speaker_distortion(
                noisy, sr, clip_cfg, rng=rng, facts_out=facts
            )
            if facts.get("applied", False) and applied_ops_out is not None:
                applied_ops_out.append("clip")

        if _rng_random(rng) < cfg.prob_post_eq:
            lo, hi = cfg.post_eq_db_range
            noisy = post_eq(noisy, sr, lo, hi, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("post_eq")

        if _rng_random(rng) < cfg.prob_lowpass:
            _lp_cutoff_lo, _lp_cutoff_hi = cfg.lowpass_cutoff_range
            _lp_order_lo, _lp_order_hi = cfg.lowpass_order_range
            noisy = lowpass_filter(noisy, sr, _lp_cutoff_lo, _lp_cutoff_hi,
                                   _lp_order_lo, _lp_order_hi, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("lowpass")

        if _rng_random(rng) < cfg.prob_band_reject:
            _br_freq_lo, _br_freq_hi = cfg.band_reject_center_freq_range
            _br_bw_lo, _br_bw_hi = cfg.band_reject_freq_bw_range
            noisy = band_reject(noisy, sr, _br_freq_lo, _br_freq_hi,
                                _br_bw_lo, _br_bw_hi,
                                cfg.band_reject_use_stft, cfg.band_reject_max_n, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("band_reject")

        if _rng_random(rng) < cfg.prob_dc_offset:
            _dc_lo, _dc_hi = cfg.dc_offset_range
            noisy = dc_offset(noisy, _dc_lo, _dc_hi, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("dc_offset")

        if _rng_random(rng) < cfg.prob_bit_crush:
            _bc_lo, _bc_hi = cfg.bit_crush_range
            noisy = bit_crush(noisy, _bc_lo, _bc_hi, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("bit_crush")

        if _rng_random(rng) < cfg.prob_spectral_holes:
            _sh_num_lo, _sh_num_hi = cfg.spectral_holes_num_range
            _sh_wf_lo, _sh_wf_hi = cfg.spectral_holes_width_freq_range
            _sh_wt_lo, _sh_wt_hi = cfg.spectral_holes_width_time_range
            noisy = spectral_time_freq_holes(
                noisy, sr,
                cfg.spectral_holes_frame_length,
                cfg.spectral_holes_hop_length,
                _sh_num_lo, _sh_num_hi,
                _sh_wf_lo, _sh_wf_hi,
                _sh_wt_lo, _sh_wt_hi,
                cfg.spectral_holes_cutoff_freq,
                rng=rng,
            )
            if applied_ops_out is not None:
                applied_ops_out.append("spectral_holes")

        # --- WebRTC NS / AGC (se_train style) ---
        if _rng_random(rng) < cfg.prob_webrtc_ns:
            level = _rng_choice(rng, cfg.webrtc_ns_levels)
            noisy = webrtc_noise_suppression(noisy, sr, level)
            if applied_ops_out is not None:
                applied_ops_out.append("webrtc_ns")

        if _rng_random(rng) < cfg.prob_webrtc_agc:
            _agc_lo, _agc_hi = cfg.webrtc_agc_target_dbfs_range
            noisy = webrtc_agc(noisy, sr, _agc_lo, _agc_hi, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("webrtc_agc")

        # --- Bass boost (se_train style) ---
        if _rng_random(rng) < cfg.prob_bass_boost:
            _bb_lo, _bb_hi = cfg.bass_boost_highpass_cutoff_range
            noisy = bass_boost(noisy, sr, _bb_lo,
                               _bb_hi, cfg.bass_boost_attenuation_min_db, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("bass_boost")

        # --- Spectral leakage (se_train style) ---
        if _rng_random(rng) < cfg.prob_spectral_leakage:
            noisy = spectral_leakage(noisy, sr, cfg.spectral_leakage_window_lengths,
                                     cfg.spectral_leakage_max_time_shift, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("spectral_leakage")

        # --- Loudness perturbation (se_train style) ---
        if _rng_random(rng) < cfg.prob_loudness_perturb:
            _loud_lo, _loud_hi = cfg.loudness_factor_range
            noisy = loudness_perturbation(noisy, _loud_lo,
                                          _loud_hi, cfg.loudness_max_n_intervals, rng=rng)
            if applied_ops_out is not None:
                applied_ops_out.append("loudness_perturb")

    if sanitize_nonfinite:
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(noisy, dtype=np.float32)
