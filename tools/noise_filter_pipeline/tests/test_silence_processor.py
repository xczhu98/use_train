"""
SilenceProcessor 单元测试（合成音频，不依赖外部文件）。

覆盖：
- shorten_long_silences 边界行为（无静音原样返回、超长静默缩短、fade 不越界）
- is_silent_chunk 绝对能量阈值与可配置 silent_energy_threshold
"""
import numpy as np
import pytest

from noise_filter_pipeline.processors.silence_processor import (
    SilenceProcessor,
    compute_energy_per_frame,
    is_silent_chunk,
    shorten_long_silences,
)

pytestmark = pytest.mark.pipeline

SR = 16000


def _sine(duration, rms, freq=220.0, sr=SR):
    """生成指定 RMS 的正弦波（float32）。"""
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    amp = rms * np.sqrt(2.0)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration, sr=SR):
    return np.zeros(int(duration * sr), dtype=np.float32)


def test_compute_energy_per_frame_matches_legacy_loop():
    """向量化帧 RMS 必须保持旧循环的边界和数值语义。"""
    rng = np.random.RandomState(123)
    for sr, frame_duration, hop_duration in ((16000, 0.025, 0.010), (8000, 0.03, 0.007)):
        frame_samples = int(frame_duration * sr)
        hop_samples = int(hop_duration * sr)
        for length in (0, frame_samples - 1, frame_samples, frame_samples + hop_samples,
                       frame_samples + 2 * hop_samples + 3, 3 * frame_samples + 1):
            audio = rng.randn(length).astype(np.float32)
            expected_times = []
            expected_energies = []
            for i in range(0, len(audio) - frame_samples, hop_samples):
                frame = audio[i:i + frame_samples]
                expected_times.append(i / sr)
                expected_energies.append(np.sqrt(np.mean(frame ** 2)))
            actual_times, actual_energies = compute_energy_per_frame(
                audio, sr, frame_duration, hop_duration
            )
            assert actual_times == expected_times
            np.testing.assert_allclose(actual_energies, expected_energies, rtol=0, atol=1e-12)


def test_compute_energy_per_frame_block_boundary_matches_full_matrix():
    """帧数超过分块上限时，逐帧结果必须与全量矩阵实现完全一致。

    回归：帧 RMS 曾一次性物化整份 (帧数, 帧长) 平方矩阵（10 分钟 16k 音频
    约 15GB）；分块计算后块边界不得改变任何一帧的数值。
    """
    from noise_filter_pipeline.processors import silence_processor

    rng = np.random.RandomState(11)
    frame_samples, hop_samples, block = 400, 160, silence_processor._ENERGY_BLOCK_FRAMES
    # 帧数刻意超过一个块，且尾部留下非整块的余量
    frame_count = block + 123
    audio = rng.randn(frame_count * hop_samples + frame_samples).astype(np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(audio, frame_samples, axis=0)
    expected = np.sqrt(np.mean(windows[:len(audio) - frame_samples:hop_samples] ** 2, axis=1))

    times, energies = compute_energy_per_frame(audio, SR, 0.025, 0.010)

    assert len(energies) == frame_count > block
    assert times == [i * hop_samples / SR for i in range(frame_count)]
    assert np.array_equal(np.asarray(energies), expected)


# ---------------------------------------------------------------------------
# shorten_long_silences
# ---------------------------------------------------------------------------

def test_shorten_no_silence_returns_original():
    """无静音段时应原样返回（长度与内容均不变）。"""
    audio = _sine(2.0, rms=0.1)
    out = shorten_long_silences(audio, SR, use_random=False)
    assert len(out) == len(audio)
    assert np.array_equal(out, audio)


def test_shorten_long_silence_is_shortened():
    """超过 max_silence_duration 的静默段应被缩短到目标时长附近。"""
    tone = _sine(1.0, rms=0.1)
    silence = _silence(5.0)
    audio = np.concatenate([tone, silence, tone])

    target = 1.0
    out = shorten_long_silences(
        audio, SR,
        max_silence_duration=2.0,
        target_silence_duration=target,
        use_random=False,
    )

    expected = 2 * len(tone) + int(target * SR)
    # 静音边界按 10ms hop 量化，允许 ±0.1s 误差
    assert abs(len(out) - expected) < int(0.1 * SR)
    assert len(out) < len(audio) - int(3.0 * SR)  # 至少缩短 3 秒
    assert out.dtype == np.float32


def test_shorten_fade_does_not_touch_content_outside_silence():
    """fade 只作用于保留的静默段内部，不得改动静默段之前的内容。"""
    tone = _sine(1.0, rms=0.1)
    silence = _silence(5.0)
    audio = np.concatenate([tone, silence, tone])

    out = shorten_long_silences(
        audio, SR,
        max_silence_duration=2.0,
        target_silence_duration=1.0,
        use_random=False,
    )

    # 前 1s 内容段在静默区之前，fade 只允许落在保留的静默样本内（crossfade
    # 样本数 ≤ keep_start_samples），内容段必须逐样本不变
    assert np.array_equal(out[: len(tone)], tone)
    # 输出长度不得超过原音频（只删不增）
    assert len(out) <= len(audio)


def test_shorten_short_silence_kept_as_is():
    """短于 max_silence_duration 的静默段不缩短，输出长度不变。"""
    tone = _sine(1.0, rms=0.1)
    silence = _silence(1.0)
    audio = np.concatenate([tone, silence, tone])
    out = shorten_long_silences(audio, SR, max_silence_duration=2.0, use_random=False)
    # 长度只允许有帧量化级的误差
    assert abs(len(out) - len(audio)) < int(0.05 * SR)


def test_shorten_with_seed_is_reproducible():
    """同一 seed 的随机静音缩短必须逐样本一致。"""
    tone = _sine(1.0, rms=0.1)
    audio = np.concatenate([tone, _silence(5.0), tone])
    proc_a = SilenceProcessor(random_seed=123)
    proc_b = SilenceProcessor(random_seed=123)

    out_a = proc_a.process_audio(audio, SR)["processed_audio"]
    out_b = proc_b.process_audio(audio, SR)["processed_audio"]
    assert np.array_equal(out_a, out_b)


def test_silence_processor_rng_context_deterministic_single_thread():
    """确定性主锚：同一实例、固定调用顺序下，rng_context 相同 → 结果
    逐样本一致；rng_context 不同 → 结果不同。

    多线程测试（test_silence_processor_rng_per_call_derivation_thread_safe）
    依赖线程调度，存在假阴窗口，仅作辅助；本单线程断言是 per-call 派生
    随机流的主要回归锚。
    """
    tone = _sine(1.0, rms=0.1)
    audio = np.concatenate([tone, _silence(5.0), tone])

    proc = SilenceProcessor(random_seed=123)
    out_a1 = proc.process_audio(audio, SR, rng_context="rec_1")["processed_audio"]
    out_a2 = proc.process_audio(audio, SR, rng_context="rec_1")["processed_audio"]
    assert np.array_equal(out_a1, out_a2)

    # 不同 context 派生独立随机流：静音缩短的裁剪位置应不同（连续随机
    # 值，理论碰撞概率可忽略）
    out_b = proc.process_audio(audio, SR, rng_context="rec_2")["processed_audio"]
    assert not np.array_equal(out_a1, out_b)


def test_silence_processor_rng_per_call_derivation_thread_safe():
    """random_seed 下多线程共享 SilenceProcessor 实例必须确定可复现。

    回归：单一 self._rng 实例被多线程共享，并发调用交错消费随机流，
    同一批记录跨运行结果不一致。修复后按 seed + record 上下文 per-call
    派生独立随机流，与线程调度无关。
    """
    from concurrent.futures import ThreadPoolExecutor

    tone = _sine(1.0, rms=0.1)
    audio = np.concatenate([tone, _silence(5.0), tone])

    def run_pool():
        proc = SilenceProcessor(random_seed=123)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(proc.process_audio, audio, SR, rng_context=f"rec_{i}")
                for i in range(8)
            ]
            return [f.result()["processed_audio"] for f in futures]

    outs1 = run_pool()
    outs2 = run_pool()
    assert len(outs1) == 8
    for a, b in zip(outs1, outs2):
        assert np.array_equal(a, b)


def test_shorten_does_not_mutate_input():
    """shorten_long_silences 不得原地修改入参音频（fade 前必须拷贝 view）。

    用低电平噪声代替纯零静音：fade 区域样本非零，原地修改才能被观测到。
    """
    rng = np.random.RandomState(42)
    tone = _sine(1.0, rms=0.1)
    quiet = (rng.normal(0, 0.0005, 5 * SR)).astype(np.float32)
    audio = np.concatenate([tone, quiet, tone, quiet, tone])
    original = audio.copy()

    shorten_long_silences(
        audio, SR,
        max_silence_duration=2.0,
        target_silence_duration=1.0,
        use_random=False,
    )

    assert np.array_equal(audio, original)


# ---------------------------------------------------------------------------
# is_silent_chunk
# ---------------------------------------------------------------------------

def test_is_silent_chunk_low_level_sine_default_threshold():
    """RMS=0.002 的低电平正弦在默认阈值 0.005 下应判为静音（绝对阈值误杀场景）。"""
    audio = _sine(1.0, rms=0.002)
    assert is_silent_chunk(audio, SR) is True


def test_is_silent_chunk_low_level_sine_lowered_threshold():
    """同一段低电平正弦，threshold 调低到 0.001 后应判为非静音。"""
    audio = _sine(1.0, rms=0.002)
    assert is_silent_chunk(audio, SR, energy_threshold=0.001) is False


def test_is_silent_chunk_all_zeros():
    """全零音频必须判为静音。"""
    assert is_silent_chunk(_silence(1.0), SR) is True


def test_silence_processor_configurable_threshold():
    """SilenceProcessor 的 silent_energy_threshold 应透传到 is_silent_chunk。"""
    audio = _sine(1.0, rms=0.002)
    default_proc = SilenceProcessor()
    lowered_proc = SilenceProcessor(silent_energy_threshold=0.001)

    r_default = default_proc.process_audio(audio, SR)
    r_lowered = lowered_proc.process_audio(audio, SR)

    assert r_default["is_silent"] is True
    assert r_default["passed"] is False
    assert r_lowered["is_silent"] is False
    assert r_lowered["passed"] is True
    # process_audio 返回的音频必须保持 float32（VAD 进程池 pickle 体积）
    assert r_lowered["processed_audio"].dtype == np.float32


# ---------------------------------------------------------------------------
# DC offset 检查
# ---------------------------------------------------------------------------

def test_dc_offset_pedestal_rejected():
    """直流平台音频（|mean| > 阈值，解码伪影）应被拒绝。"""
    # 0.787 直流 + 微小波动，模拟 WavCaps 坏文件形态
    rng = np.random.RandomState(0)
    audio = (0.787 + rng.normal(0, 0.005, SR)).astype(np.float32)
    r = SilenceProcessor().process_audio(audio, SR)
    assert r["passed"] is False
    assert r["dc_offset"] > 0.5


def test_dc_offset_normal_audio_passes():
    """正常零均值音频不受 DC 检查影响。"""
    audio = _sine(2.0, rms=0.1)
    r = SilenceProcessor().process_audio(audio, SR)
    assert r["passed"] is True
    assert abs(r["dc_offset"]) < 0.01


def test_dc_offset_negative_pedestal_rejected():
    """负向直流平台同样被拒绝。"""
    audio = np.full(SR, -0.6, dtype=np.float32)
    assert SilenceProcessor().process_audio(audio, SR)["passed"] is False


def test_dc_offset_disabled_with_none():
    """dc_offset_threshold=None 时关闭检查。"""
    audio = np.full(SR, 0.8, dtype=np.float32)
    r = SilenceProcessor(dc_offset_threshold=None).process_audio(audio, SR)
    assert r["passed"] is True
