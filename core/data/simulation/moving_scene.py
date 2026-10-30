"""公共移动声源 waveform 渲染，按发射时刻分段卷积并保留跨段声学尾部。"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from core.data.spatial import SpatialRenderResult


_SOURCE_BLOCK_SAMPLES = 4096


def _real_array(value, *, name: str, ndim: int) -> np.ndarray:
    """不复制完整 RIR bank，逐条检查真实数值和有限性。"""
    array = np.asarray(value)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    for index in np.ndindex(array.shape[:-1]):
        if not np.isfinite(array[index]).all():
            raise ValueError(f"{name} must be finite")
    return array


def _accumulate_segment_channels(
    output: np.ndarray, source: np.ndarray, rirs: np.ndarray, *, start: int,
) -> None:
    """一次复用 source FFT，将一个声源段卷积到全部麦克风通道。"""
    if rirs.size == 0:
        return
    rirs = np.asarray(rirs)
    if rirs.ndim != 2 or rirs.shape[0] != output.shape[0]:
        raise ValueError("channel RIR bank must have shape [C,L]")
    rirs = rirs[:, :output.shape[-1] - start]
    block_size = min(source.size, _SOURCE_BLOCK_SAMPLES)
    fft_size = next_fast_len(block_size + rirs.shape[-1] - 1, real=True)
    kernels = np.stack([
        rfft(np.asarray(rir, dtype=np.float64), n=fft_size)
        for rir in rirs
    ], axis=0)
    for offset in range(0, source.size, block_size):
        block = source[offset:offset + block_size]
        spectrum = rfft(np.asarray(block, dtype=np.float64), n=fft_size)
        convolution = irfft(kernels * spectrum[None, :], n=fft_size, axis=-1)
        left = start + offset
        count = min(block.size + rirs.shape[-1] - 1, output.shape[-1] - left)
        output[:, left:left + count] += convolution[:, :count]


def render_moving_spatial_scene(
    sources: np.ndarray,
    full_rirs: np.ndarray,
    *,
    direct_rirs: np.ndarray | None = None,
    frame_samples: np.ndarray,
    early_boundary: int,
    reference_channel: int = 0,
    num_samples: int | None = None,
    rts_rirs: np.ndarray | None = None,
) -> SpatialRenderResult:
    """把 ``sources[K,N]`` 和 ``RIRs[P,K,C,L]`` 渲染为 ``[K,C,T]``。

    ``frame_samples[P]`` 是每段的发射 sample 起点，必须为整数、从 0 开始、
    严格递增且小于 N；最后一段终止于 N。第 p 段使用第 p 个 RIR 完整卷积，
    从原始起点做 overlap-add。前段尾音即使穿过下一段边界也保留。这与本地
    gpuRIR ``simulateTrajectory`` 的分段语义一致，不插值 RIR、不移动声源
    waveform，也不依赖 CUDA 扩展。

    direct_rirs 可以省略；省略时不分配或计算 direct_tracks。传入的
    full/direct（若存在）必须共享 P/K/C 轴，L 可以不同。early 只取 full RIR 的
    ``[:early_boundary]`` taps；它不是 direct path。传入的直达延迟由 RIR
    的前导零自然保留，不做峰值对齐或 wrap。``early_boundary=0`` 产生零轨。

    ``num_samples=None`` 返回 N+max(L_full,L_direct)-1，保留所有尾部；显式
    长度必须 >=N，可以裁切尾部或补零。与静态 renderer 比较时应传入相同的
    ``num_samples``。输出及混合遵循 ``SpatialRenderResult`` 的 float32
    合同。除了输出 bank，每次仅处理一个段/source/channel，FFT 缓冲区不随
    P/K/C 或完整 source 长度增长。输入数组保持不变。
    """
    sources = _real_array(sources, name="sources [K,N]", ndim=2)
    full = _real_array(full_rirs, name="full_rirs [P,K,C,L]", ndim=4)
    direct = None if direct_rirs is None else _real_array(
        direct_rirs, name="direct_rirs [P,K,C,L]", ndim=4
    )
    points, tracks, channels, _ = full.shape
    if tracks != sources.shape[0]:
        raise ValueError("full_rirs source K must match sources")
    if direct is not None and direct.shape[:3] != full.shape[:3]:
        raise ValueError("direct_rirs must share full_rirs P/K/C axes and timebase")
    rts = None if rts_rirs is None else _real_array(rts_rirs, name="rts_rirs [P,K,C,L]", ndim=4)
    if rts is not None and rts.shape[:3] != full.shape[:3]:
        raise ValueError("rts_rirs must share full_rirs P/K/C axes and timebase")
    if type(early_boundary) is not int or early_boundary < 0:
        raise ValueError("early_boundary must be a non-negative exact int")
    if type(reference_channel) is not int or not 0 <= reference_channel < channels:
        raise ValueError("reference_channel must be an exact int in [0,C)")
    starts = np.asarray(frame_samples)
    if starts.ndim != 1 or starts.shape[0] != points or starts.dtype.kind not in "iu":
        raise ValueError("frame_samples must be an integer [P] array")
    source_length = sources.shape[1]
    if starts[0] != 0 or np.any(starts >= source_length) or np.any(starts < 0):
        raise ValueError("frame_samples must start at 0 and lie in [0,N)")
    # 直接比较可避免 unsigned diff 在倒序时下溢。
    if np.any(starts[1:] <= starts[:-1]):
        raise ValueError("frame_samples must be strictly increasing")
    rir_lengths = [full.shape[-1]]
    if direct is not None:
        rir_lengths.append(direct.shape[-1])
    if rts is not None:
        rir_lengths.append(rts.shape[-1])
    length = source_length + max(rir_lengths) - 1
    if num_samples is not None:
        if type(num_samples) is not int or num_samples < source_length:
            raise ValueError("num_samples must be an exact int >= source length N")
        length = num_samples
    output_shape = (tracks, channels, length)
    full_tracks = np.zeros(output_shape, dtype=np.float32)
    early_tracks = np.zeros_like(full_tracks)
    direct_tracks = None if direct is None else np.zeros_like(full_tracks)
    rts_tracks = None if rts is None else np.zeros_like(full_tracks)
    for point in range(points):
        start = int(starts[point])
        stop = int(starts[point + 1]) if point + 1 < points else source_length
        for track in range(tracks):
            segment = sources[track, start:stop]
            full_rir = full[point, track]
            _accumulate_segment_channels(full_tracks[track], segment, full_rir, start=start)
            _accumulate_segment_channels(
                early_tracks[track], segment, full_rir[:, :early_boundary], start=start,
            )
            if direct_tracks is not None:
                _accumulate_segment_channels(
                    direct_tracks[track], segment, direct[point, track], start=start,
                )
            if rts_tracks is not None:
                _accumulate_segment_channels(
                    rts_tracks[track], segment, rts[point, track], start=start,
                )
    return SpatialRenderResult(
        full_tracks=full_tracks,
        early_tracks=early_tracks,
        reference_tracks=full_tracks[:, reference_channel],
        mixture=full_tracks.sum(axis=0, dtype=np.float32),
        direct_tracks=direct_tracks,
        rts_tracks=rts_tracks,
        provenance={
            "renderer": "moving_segment_overlap_add",
            "renderer_version": 2,
            "frame_samples": [int(sample) for sample in starts],
        },
    )
