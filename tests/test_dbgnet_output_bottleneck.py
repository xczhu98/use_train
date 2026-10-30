"""JAE-M08 harness 纯函数单测：不加载真 checkpoint。

覆盖：
- oracle_crm 在合成复谱上应精确还原 near（SI-SDR > 40dB，表达上限 sanity check）；
- oracle_real_mask 的 SI-SDR 有限且显著低于 oracle_crm（实数 [0,1] mask 的表达瓶颈）；
- make_reference_variants 的 zeros/shifted 路径形状与内容正确。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.model_tools.dbgnet_output_bottleneck import (  # noqa: E402
    FAR_SHIFT_SAMPLES,
    _find_sample_dirs,
    _positive_int,
    make_reference_variants,
    oracle_crm_enh,
    oracle_real_mask_enh,
    sisdr_db,
)


def _synth_specs(seed: int = 0, F: int = 33, T: int = 50):
    """合成复谱：near = randn 复谱，mic = near + echo（echo 相位与 near 不正交）。"""
    rng = np.random.default_rng(seed)

    def randn_c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)

    near = randn_c(F, T)
    echo = randn_c(F, T)
    mic = near + echo
    return mic, near, echo


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1.5", ""])
def test_num_samples_rejects_non_positive_or_non_integer_text(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(value)


@pytest.mark.parametrize("value", [0, -1, 1.0, True, np.nan, np.inf])
def test_find_sample_dirs_rejects_non_positive_or_non_integer_limit(tmp_path, value):
    with pytest.raises(ValueError, match="num_samples 必须是正整数"):
        _find_sample_dirs(tmp_path, value)


def test_find_sample_dirs_recurses_canonical_nested_layout_in_stable_order(tmp_path):
    required = ("mic.wav", "nearend.wav", "farend.wav")
    expected = [
        tmp_path / "scenario_a" / "sample_001",
        tmp_path / "scenario_b" / "nested" / "sample_002",
    ]
    for sample_dir in reversed(expected):
        sample_dir.mkdir(parents=True)
        for name in required:
            (sample_dir / name).touch()
    incomplete = tmp_path / "scenario_a" / "sample_incomplete"
    incomplete.mkdir()
    (incomplete / "mic.wav").touch()

    assert _find_sample_dirs(tmp_path, 2) == expected
    assert _find_sample_dirs(tmp_path, 1) == expected[:1]


def test_sisdr_complex_spectra_rejects_non_time_shape_mismatch():
    ref = np.ones((2, 5), dtype=np.complex64)
    enh = np.ones((3, 5), dtype=np.complex64)
    with pytest.raises(ValueError, match="非时间维必须一致"):
        sisdr_db(ref, enh)


def test_sisdr_complex_spectra_align_time_before_real_imag_flattening():
    from core.metrics.intrusive import _INTRUSIVE_FNS

    ref = np.arange(10, dtype=np.float64).reshape(2, 5) + 1j * np.arange(10, 20).reshape(2, 5)
    enh = ref[:, :3] + (0.1 - 0.2j)
    ref_aligned = ref[:, :3]
    ref_vector = np.concatenate([ref_aligned.real.ravel(), ref_aligned.imag.ravel()])
    enh_vector = np.concatenate([enh.real.ravel(), enh.imag.ravel()])
    fn, _ = _INTRUSIVE_FNS["SI-SDR"]
    expected = float(fn(ref_vector, enh_vector, 16000))

    assert sisdr_db(ref, enh) == pytest.approx(expected)


def test_oracle_crm_recovers_near_above_40db():
    """对照 C：mic × (near / mic) 数学上精确还原 near → SI-SDR > 40dB。"""
    mic, near, _ = _synth_specs()
    enh = oracle_crm_enh(mic, near)
    v = sisdr_db(near, enh)
    assert np.isfinite(v)
    assert v > 40.0, f"oracle_crm 应精确还原 near，实际 SI-SDR={v:.2f}dB"


def test_oracle_real_mask_finite_and_below_crm():
    """对照 B：实数 [0,1] mask 不能旋转相位 → SI-SDR 有限且显著低于 CRM。"""
    mic, near, _ = _synth_specs()
    enh_b = oracle_real_mask_enh(mic, near)
    enh_c = oracle_crm_enh(mic, near)
    v_b = sisdr_db(near, enh_b)
    v_c = sisdr_db(near, enh_c)
    assert np.isfinite(v_b), "oracle_real_mask 的 SI-SDR 应有限"
    assert v_b < v_c, f"表达瓶颈：real mask ({v_b:.2f}dB) 应低于 CRM ({v_c:.2f}dB)"


def test_oracle_real_mask_bounded_gain():
    """oracle real mask 的等效增益必须在 [0,1]（与 sigmoid 输出同一 bound）。"""
    mic, near, _ = _synth_specs()
    enh_b = oracle_real_mask_enh(mic, near)
    gain = np.abs(enh_b) / (np.abs(mic) + 1e-8)
    assert gain.max() <= 1.0 + 1e-5
    assert gain.min() >= 0.0


def test_reference_variants_shapes():
    """对照 D：zeros-far 全零且形状不变；shifted 为循环移位；correct 原样。"""
    rng = np.random.default_rng(1)
    far = rng.standard_normal(16000).astype(np.float32)
    variants = make_reference_variants(far, FAR_SHIFT_SAMPLES)

    assert set(variants) == {"correct", "zeros", "shifted"}
    for v in variants.values():
        assert v.shape == far.shape
    assert np.array_equal(variants["correct"], far)
    assert np.count_nonzero(variants["zeros"]) == 0
    assert np.array_equal(variants["shifted"], np.roll(far, FAR_SHIFT_SAMPLES))
    # shifted 与原信号不同（随机信号循环移位 800 样点后不可能恒等）
    assert not np.array_equal(variants["shifted"], far)


def test_sisdr_db_real_waveform_identity():
    """SI-SDR 包装：实数波形完全相同 → 很大 dB；走 core.metrics 同一实现。"""
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(8000).astype(np.float64)
    assert sisdr_db(ref, ref.copy()) > 60.0
    noisy = ref + 0.1 * rng.standard_normal(8000)
    v = sisdr_db(ref, noisy)
    assert 0.0 < v < 40.0


def test_main_errors_without_checkpoint_and_config(monkeypatch):
    """checkpoint/config 默认值已移除（红线：路径不入版本控制），缺参必须 exit 2。"""
    from tools.model_tools import dbgnet_output_bottleneck as bob

    monkeypatch.setattr("sys.argv", ["prog", "--data-dir", "/tmp/x"])
    with pytest.raises(SystemExit) as exc_info:
        bob.main()
    assert exc_info.value.code == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

pytestmark = [pytest.mark.unit]
