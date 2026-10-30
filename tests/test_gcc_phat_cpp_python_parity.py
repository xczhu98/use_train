"""Verify C++ GccPhatEstimatorCpp and Python StreamingDelayEstimator agree.

The two implementations use different FFT back-ends (numpy/pocketfft vs C++
pocketfft) and may show tiny numerical differences on individual frames, but
after median smoothing they should agree within the expected jitter tolerance.
"""
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
module_path = _PROJECT_ROOT / "tasks" / "aec" / "inference" / "streaming_delay_estimator.py"
spec = importlib.util.spec_from_file_location("streaming_delay_estimator", module_path)
estimator_mod = importlib.util.module_from_spec(spec)
sys.modules["streaming_delay_estimator"] = estimator_mod
spec.loader.exec_module(estimator_mod)

StreamingDelayEstimator = estimator_mod.StreamingDelayEstimator
StreamingDelayEstimatorConfig = estimator_mod.StreamingDelayEstimatorConfig
create_delay_estimator = estimator_mod.create_delay_estimator

from tasks.aec.inference.matched_filter_delay_estimator import (
    MatchedFilterDelayEstimator,
    MatchedFilterDelayEstimatorConfig,
)


def _shaped_noise(rng, n_samples):
    ref = rng.normal(0, 1.0, n_samples).astype(np.float32)
    ref = np.convolve(ref, np.ones(8) / 8, mode="same").astype(np.float32)
    ref *= 0.5 / (np.std(ref) + 1e-12)
    return ref


def _run_estimator(estimator, ref, mic, frame_size):
    delays = []
    aligned = []
    for i in range(0, ref.size - frame_size + 1, frame_size):
        estimator.push_ref(ref[i : i + frame_size])
        estimator.push_mic(mic[i : i + frame_size])
        delays.append(estimator.delay_samples)
        aligned.append(estimator.get_aligned_ref_frame(frame_size))
    return np.asarray(delays), np.concatenate(aligned)


def _require_cpp(cfg):
    cpp = create_delay_estimator(cfg)
    assert cpp.backend_identity == "cpp", (
        "integration parity requires the freshly built delay_estimator_ext"
    )
    return cpp


def test_factory_exposes_stable_readonly_backend_identity():
    estimator = _require_cpp(StreamingDelayEstimatorConfig())

    assert estimator.backend_identity == "cpp"
    with pytest.raises(AttributeError):
        estimator.backend_identity = "python"


def test_cpp_noarg_defaults_match_python_defaults():
    extension = estimator_mod._load_delay_estimator_extension()
    default_cfg = StreamingDelayEstimatorConfig()
    assert default_cfg.acps_alpha == 0.8
    assert default_cfg.coarse_delay_ms == 50.0
    assert default_cfg.max_residual_delay_ms == 50.0
    assert default_cfg.min_observation_ms == 200.0
    py = StreamingDelayEstimator(default_cfg)
    cpp = extension.GccPhatEstimatorCpp()
    rng = np.random.default_rng(404)
    ref = _shaped_noise(rng, 2 * 16000)
    mic = np.zeros_like(ref)
    mic[640:] = 0.3 * ref[:-640]
    mic += rng.normal(0.0, 0.03, mic.size).astype(np.float32)

    py_delays, _ = _run_estimator(py, ref, mic, 160)
    cpp_delays, _ = _run_estimator(cpp, ref, mic, 160)
    np.testing.assert_allclose(cpp_delays, py_delays, atol=1)


def test_cpp_extension_is_fresh_native_dir_build():
    _require_cpp(StreamingDelayEstimatorConfig())
    import delay_estimator_ext

    native_dir = _PROJECT_ROOT / "tasks" / "aec" / "inference" / "_native"
    extension_path = Path(delay_estimator_ext.__file__).resolve()
    source_path = _PROJECT_ROOT / "tasks" / "aec" / "inference" / "csrc" / "delay_estimator_ext.cpp"
    assert extension_path.parent == native_dir, (
        f"expected native-dir extension, loaded {extension_path}; "
        "rebuild with: python tasks/aec/inference/csrc/setup.py build_ext --inplace --force"
    )
    assert extension_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns, (
        f"loaded stale extension {extension_path}; source is newer: {source_path}"
    )


def test_extension_validation_rejects_wrong_dir_and_stale_build(tmp_path):
    native_dir = tmp_path / "repo" / "_native"
    native_dir.mkdir(parents=True)
    source_path = tmp_path / "repo" / "delay_estimator_ext.cpp"
    extension_path = native_dir / "delay_estimator_ext.so"
    source_path.write_bytes(b"source")
    extension_path.write_bytes(b"extension")
    os.utime(extension_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(source_path, ns=(2_000_000_000, 2_000_000_000))

    with pytest.raises(ImportError, match="stale"):
        estimator_mod._validate_delay_estimator_extension(
            extension_path, source_path, native_dir
        )

    extension_path.touch()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    with pytest.raises(ImportError, match="non-native-dir"):
        estimator_mod._validate_delay_estimator_extension(
            extension_path, source_path, other_dir
        )


def test_factory_ignores_legacy_csrc_ahead_on_sys_path():
    csrc_dir = _PROJECT_ROOT / "tasks" / "aec" / "inference" / "csrc"
    script = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        from pathlib import Path

        root = Path({str(_PROJECT_ROOT)!r})
        csrc = Path({str(csrc_dir)!r})
        sys.path[:] = [str(csrc), str(root)] + [
            entry for entry in sys.path if entry not in {{str(csrc), str(root)}}
        ]
        module_path = root / "tasks" / "aec" / "inference" / "streaming_delay_estimator.py"
        spec = importlib.util.spec_from_file_location(
            "streaming_delay_estimator_import_order_test", module_path
        )
        estimator_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = estimator_mod
        spec.loader.exec_module(estimator_mod)
        estimator = estimator_mod.create_delay_estimator(
            estimator_mod.StreamingDelayEstimatorConfig()
        )
        loaded_path = Path(sys.modules["delay_estimator_ext"].__file__).resolve()

        assert loaded_path.parent == root / "tasks" / "aec" / "inference" / "_native", loaded_path
        assert estimator.backend_identity == "cpp"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=csrc_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_factory_rejects_preimported_foreign_extension():
    # 旧 tasks/aec/inference/csrc/*.so 已删除；用指向非根目录路径的 stub 模块
    # 覆盖 sys.modules 污染防护，避免再依赖旧二进制。
    script = textwrap.dedent(
        f"""
        import importlib.util
        import sys
        import types
        import warnings
        from pathlib import Path

        root = Path({str(_PROJECT_ROOT)!r})
        foreign = types.ModuleType("delay_estimator_ext")
        foreign.__file__ = str(root / "somewhere_else" / "delay_estimator_ext.so")
        sys.modules["delay_estimator_ext"] = foreign

        module_path = root / "tasks" / "aec" / "inference" / "streaming_delay_estimator.py"
        spec = importlib.util.spec_from_file_location(
            "streaming_delay_estimator_preimport_test", module_path
        )
        estimator_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = estimator_mod
        spec.loader.exec_module(estimator_mod)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator = estimator_mod.create_delay_estimator(
                estimator_mod.StreamingDelayEstimatorConfig()
            )

        assert estimator.backend_identity == "python"
        assert any("falling back" in str(item.message) for item in caught)
        assert sys.modules["delay_estimator_ext"] is foreign
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT / "tasks" / "aec" / "inference" / "csrc",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_each_update_delay_and_aligned_frame_match():
    _require_cpp(StreamingDelayEstimatorConfig(algorithm="gcc_phat"))

    sr = 16000
    duration_s = 4.0
    rng = np.random.default_rng(2024)
    ref = _shaped_noise(rng, int(duration_s * sr))

    cfg = StreamingDelayEstimatorConfig(
        sample_rate=sr,
        buffer_duration_ms=500.0,
        coarse_delay_ms=0.0,
        max_residual_delay_ms=300.0,
        update_interval_ms=50.0,
        smooth_alpha=0.3,
        peak_threshold=0.05,
        peak_prominence_db=6.0,
    )

    for true_delay_ms in [0.0, 25.0, 50.0, 87.5, 150.0, 250.0]:
        true_delay_samples = int(round(true_delay_ms * sr / 1000.0))
        echo = np.zeros_like(ref)
        if true_delay_samples > 0:
            echo[true_delay_samples:] = ref[:-true_delay_samples] * 0.3
        mic = echo + rng.normal(0, 0.05, ref.size).astype(np.float32)

        py_delays, py_aligned = _run_estimator(
            StreamingDelayEstimator(cfg), ref, mic, frame_size=800
        )
        cpp_delays, cpp_aligned = _run_estimator(
            _require_cpp(cfg), ref, mic, frame_size=800
        )

        np.testing.assert_allclose(py_delays, cpp_delays, atol=1)
        np.testing.assert_allclose(py_aligned, cpp_aligned, atol=1e-6)


@pytest.mark.parametrize("algorithm", ("gcc_phat", "matched_filter"))
def test_factory_cpp_backends_share_aligned_frame_argument_contract(algorithm):
    estimator = create_delay_estimator(
        StreamingDelayEstimatorConfig(algorithm=algorithm)
    )
    assert estimator.backend_identity == "cpp"
    assert estimator.get_aligned_ref_frame(np.int64(0)).shape == (0,)

    for frame_size, message in (
        (True, "integer"),
        (1.0, "integer"),
        (-1, "non-negative"),
        (estimator._aligned_ref_capacity + 1, "capacity"),
        (10**18, "capacity"),
    ):
        with pytest.raises(ValueError, match=message):
            estimator.get_aligned_ref_frame(frame_size)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"sr": 0},
        {"buffer_ms": 0.0},
        {"buffer_ms": np.nan},
        {"coarse_ms": -1.0},
        {"residual_ms": -1.0},
        {"update_ms": 0.0},
        {"smooth_alpha": 0.0},
        {"smooth_alpha": 1.01},
        {"ref_thresh_db": np.inf},
        {"peak_thresh": 0.0},
        {"peak_thresh": 1.01},
        {"prominence_db": -1.0},
        {"false_accept_prob": 0.0},
        {"false_accept_prob": 1.0},
        {"false_accept_prob": np.nan},
        {"acps_alpha": -0.01},
        {"acps_alpha": 1.0},
        {"min_obs_ms": 501.0, "buffer_ms": 500.0},
        {"eps": 0.0},
    ),
)
def test_cpp_constructor_rejects_invalid_config_matrix(kwargs):
    cpp_type = type(_require_cpp(StreamingDelayEstimatorConfig())._estimator)
    with pytest.raises(ValueError):
        cpp_type(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"sr": 0}, "sr must be > 0"),
        ({"sr": 8000}, "only support ds==4 and sr==16000"),
        ({"ds": 0}, "ds must be > 0"),
        ({"ds": 2}, "only support ds==4 and sr==16000"),
        ({"num_filters": 0}, "num_filters must be > 0"),
        ({"flen_subblocks": 0}, "flen_subblocks must be > 0"),
        ({"lag_shift_subblocks": 0}, "lag_shift_subblocks must be > 0"),
        ({"hist_window": 0}, "hist_window must be > 0"),
        ({"init_thresh": 0}, "init_thresh must be > 0"),
        ({"conv_thresh": 0}, "conv_thresh must be > 0"),
        ({"block_size": 0}, "block_size must be > 0"),
        ({"smoothing": np.nan}, "smoothing must be finite"),
        ({"smoothing": np.inf}, "smoothing must be finite"),
        ({"smoothing": -0.01}, "smoothing must be in [0, 1]"),
        ({"smoothing": 1.01}, "smoothing must be in [0, 1]"),
        ({"excitation_limit": np.nan}, "excitation_limit must be finite"),
        ({"excitation_limit": -np.inf}, "excitation_limit must be finite"),
        ({"excitation_limit": -0.001}, "excitation_limit must be >= 0"),
        (
            {"block_size": 66, "ds": 4},
            "block_size must be divisible by ds",
        ),
        (
            {"init_thresh": 21, "conv_thresh": 20},
            "init_thresh must be <= conv_thresh",
        ),
    ),
)
def test_cpp_matched_filter_constructor_rejects_invalid_config_matrix(
    kwargs, message
):
    extension = estimator_mod._load_delay_estimator_extension()

    with pytest.raises(ValueError, match=re.escape(message)):
        extension.MatchedFilterEstimatorCpp(**kwargs)


def test_matched_filter_cpp_huge_finite_rejects_atomically_and_recovers():
    extension = estimator_mod._load_delay_estimator_extension()
    est = extension.MatchedFilterEstimatorCpp(init_thresh=1, conv_thresh=2)
    control = extension.MatchedFilterEstimatorCpp(init_thresh=1, conv_thresh=2)
    valid = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    for candidate in (est, control):
        candidate.push_ref(valid)
        candidate.push_mic(valid)

    for method_name in ("push_ref", "push_mic"):
        before = est.get_aligned_ref_frame(64).copy()
        with pytest.raises(ValueError, match="normalized audio range"):
            getattr(est, method_name)(
                np.array([0.0, 1.0e20, -1.0e20], dtype=np.float32)
            )
        np.testing.assert_array_equal(est.get_aligned_ref_frame(64), before)

    for candidate in (est, control):
        candidate.push_ref(valid)
        candidate.push_mic(valid)
    assert est.delay_samples == control.delay_samples
    np.testing.assert_array_equal(
        est.get_aligned_ref_frame(64), control.get_aligned_ref_frame(64)
    )


def test_matched_filter_cpp_one_sided_pending_has_fixed_bound():
    extension = estimator_mod._load_delay_estimator_extension()
    est = extension.MatchedFilterEstimatorCpp()
    frame = np.zeros(100, dtype=np.float32)
    accepted = 0

    for _ in range(1000):  # 100k attempted one-sided samples.
        try:
            est.push_ref(frame)
            accepted += frame.size
        except ValueError as exc:
            assert "pending skew" in str(exc)

    assert accepted <= 16384
    assert 16384 - accepted < frame.size
    est.push_mic(np.zeros(accepted, dtype=np.float32))
    est.push_ref(np.zeros(64, dtype=np.float32))
    est.push_mic(np.zeros(64, dtype=np.float32))
    assert est.delay_samples == 0


def test_matched_filter_python_cpp_normal_delay_regression():
    extension = estimator_mod._load_delay_estimator_extension()
    cfg = MatchedFilterDelayEstimatorConfig(
        initial_threshold=2,
        converged_threshold=5,
    )
    py = MatchedFilterDelayEstimator(cfg)
    cpp = extension.MatchedFilterEstimatorCpp(init_thresh=2, conv_thresh=5)
    rng = np.random.default_rng(123)
    delay = 640
    ref = rng.normal(0.0, 0.2, 3 * 16000).astype(np.float32)
    mic = np.zeros_like(ref)
    mic[delay:] = 0.8 * ref[:-delay]

    for start in range(0, ref.size, 160):
        for estimator in (py, cpp):
            estimator.push_ref(ref[start : start + 160])
            estimator.push_mic(mic[start : start + 160])

    assert py.delay_samples == delay
    assert cpp.delay_samples == delay
    assert cpp.delay_samples == py.delay_samples


def test_cpp_valid_invalid_valid_matches_control():
    cfg = StreamingDelayEstimatorConfig(
        update_interval_ms=10.0,
        min_observation_ms=10.0,
        acps_alpha=0.0,
    )
    cpp = _require_cpp(cfg)
    control = _require_cpp(cfg)
    rng = np.random.default_rng(71)

    valid_frames = [_shaped_noise(rng, 160) for _ in range(12)]
    for frame in valid_frames[:6]:
        for estimator in (cpp, control):
            estimator.push_ref(frame)
            estimator.push_mic(frame)

    for push_name, bad_value in (
        ("push_ref", np.nan),
        ("push_ref", np.inf),
        ("push_mic", np.nan),
        ("push_mic", -np.inf),
    ):
        before = cpp.get_aligned_ref_frame(160).copy()
        bad = np.array([0.0, bad_value, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            getattr(cpp, push_name)(bad)
        np.testing.assert_array_equal(cpp.get_aligned_ref_frame(160), before)

    for frame in valid_frames[6:]:
        for estimator in (cpp, control):
            estimator.push_ref(frame)
            estimator.push_mic(frame)
        assert cpp.delay_samples == control.delay_samples
        np.testing.assert_array_equal(
            cpp.get_aligned_ref_frame(160), control.get_aligned_ref_frame(160)
        )


def test_prominence_filter_rejects_near_center_weak_multipath_candidate():
    """已复现双路径：弱近峰不能因靠近 search center 抢过高 prominence 峰。"""
    sr = 16000
    rng = np.random.default_rng(99)
    ref = _shaped_noise(rng, 5 * sr)
    mic = np.zeros_like(ref)
    near_delay, true_delay = 40 * 16, 220 * 16
    mic[near_delay:] += 0.21 * ref[:-near_delay]
    mic[true_delay:] += 0.40 * ref[:-true_delay]
    cfg = StreamingDelayEstimatorConfig(
        coarse_delay_ms=40.0,
        max_residual_delay_ms=300.0,
        update_interval_ms=50.0,
        smooth_alpha=1.0,
        acps_alpha=0.0,
        peak_threshold=0.03,
    )
    py_delays, _ = _run_estimator(StreamingDelayEstimator(cfg), ref, mic, 160)
    cpp_delays, _ = _run_estimator(_require_cpp(cfg), ref, mic, 160)
    np.testing.assert_allclose(cpp_delays, py_delays, atol=1)
    assert abs(py_delays[-1] - true_delay) <= 2
    assert abs(cpp_delays[-1] - true_delay) <= 2


def test_nearend_only_multiple_seeds_rejects_each_update_with_cpp_parity():
    # 显式 coarse=0：本测试语义是“永不锁定”，延迟必须停留在 0。
    cfg = StreamingDelayEstimatorConfig(
        buffer_duration_ms=500.0,
        coarse_delay_ms=0.0,
        max_residual_delay_ms=400.0,
        update_interval_ms=100.0,
        min_observation_ms=100.0,
        acps_alpha=0.9,
    )
    for seed in range(10):
        rng = np.random.default_rng(seed)
        ref = rng.normal(size=16000).astype(np.float32)
        mic = rng.normal(size=16000).astype(np.float32)
        py_delays, _ = _run_estimator(StreamingDelayEstimator(cfg), ref, mic, 160)
        cpp_delays, _ = _run_estimator(_require_cpp(cfg), ref, mic, 160)
        np.testing.assert_array_equal(cpp_delays, py_delays)
        assert np.all(py_delays == 0), (seed, py_delays)


def test_selected_lag_confidence_signed_shift_and_boundaries_match_cpp():
    extension = estimator_mod._load_delay_estimator_extension()
    rng = np.random.default_rng(193)
    source = rng.normal(size=512).astype(np.float32)
    cfg = StreamingDelayEstimatorConfig(acps_alpha=0.0)
    py = StreamingDelayEstimator(cfg)

    for lag in (37, -37):
        mic = np.zeros_like(source)
        if lag > 0:
            mic[lag:] = source[:-lag]
        else:
            mic[:lag] = source[-lag:]
        expected = py._selected_lag_confidence(
            mic, source, lag, candidate_count=9
        )
        actual = extension.selected_lag_confidence(
            mic, source, lag, 9, cfg.false_accept_prob
        )
        assert actual is expected is True

    low_variance = np.arange(512, dtype=np.float32)
    low_variance[1] = np.nextafter(low_variance[1], np.float32(np.inf))
    healthy = rng.normal(size=512).astype(np.float32)
    for mic, ref in (
        (low_variance * np.float32(1.0e4), healthy * np.float32(1.0e-4)),
        (healthy * np.float32(1.0e-4), low_variance * np.float32(1.0e4)),
    ):
        expected = py._selected_lag_confidence(mic, ref, 0, candidate_count=1)
        actual = extension.selected_lag_confidence(
            mic, ref, 0, 1, cfg.false_accept_prob
        )
        assert actual is expected is None

    diff_count = source.size - 1
    for threshold in (1.0, 1.01):
        false_accept_prob = 2.0 * np.exp(
            -0.5 * diff_count * threshold * threshold
        )
        boundary_cfg = StreamingDelayEstimatorConfig(
            acps_alpha=0.0,
            false_accept_prob=false_accept_prob,
        )
        expected = StreamingDelayEstimator(
            boundary_cfg
        )._selected_lag_confidence(source, source, 0, candidate_count=1)
        actual = extension.selected_lag_confidence(
            source, source, 0, 1, false_accept_prob
        )
        assert actual is expected is None


def test_positive_stationary_then_jump_delay_keeps_cpp_parity():
    sr = 16000
    segment_samples = 2 * sr
    first_delay = 320
    second_delay = 960
    rng = np.random.default_rng(811)
    ref = rng.normal(size=2 * segment_samples).astype(np.float32)
    mic = np.zeros_like(ref)
    mic[first_delay:segment_samples] = 0.4 * ref[: segment_samples - first_delay]
    second_ref = ref[segment_samples:]
    mic[segment_samples + second_delay :] = 0.4 * second_ref[:-second_delay]
    mic += rng.normal(0.0, 0.02, mic.size).astype(np.float32)
    cfg = StreamingDelayEstimatorConfig(
        max_residual_delay_ms=100.0,
        update_interval_ms=50.0,
        smooth_alpha=0.3,
        acps_alpha=0.9,
    )

    py_delays, _ = _run_estimator(StreamingDelayEstimator(cfg), ref, mic, 160)
    cpp_delays, _ = _run_estimator(_require_cpp(cfg), ref, mic, 160)
    np.testing.assert_allclose(cpp_delays, py_delays, atol=1)
    assert abs(py_delays[segment_samples // 160 - 1] - first_delay) <= 2
    assert abs(py_delays[-1] - second_delay) <= 2


def test_smooth_alpha_quarter_uses_even_history_median():
    # 显式 coarse=0：buffer 仅 16ms，默认 coarse=50ms 会把搜索窗钳到界外。
    cfg = StreamingDelayEstimatorConfig(
        sample_rate=16000,
        buffer_duration_ms=16.0,
        coarse_delay_ms=0.0,
        max_residual_delay_ms=15.0,
        update_interval_ms=16.0,
        min_observation_ms=16.0,
        smooth_alpha=0.25,
        acps_alpha=0.0,
        peak_threshold=0.01,
        peak_prominence_db=0.0,
    )
    py = StreamingDelayEstimator(cfg)
    cpp = _require_cpp(cfg)
    rng = np.random.default_rng(123)
    expected_medians = [16, 32, 48, 64, 96]

    for delay, expected in zip((16, 48, 80, 112, 144), expected_medians):
        ref = _shaped_noise(rng, 256)
        mic = np.zeros_like(ref)
        mic[delay:] = ref[:-delay]
        for estimator in (py, cpp):
            estimator.push_ref(ref)
            estimator.push_mic(mic)
        assert py.delay_samples == expected
        assert cpp.delay_samples == expected
        np.testing.assert_array_equal(
            cpp.get_aligned_ref_frame(256), py.get_aligned_ref_frame(256)
        )


pytestmark = [pytest.mark.integration]
