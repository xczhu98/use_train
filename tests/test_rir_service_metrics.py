"""受管理 gpuRIR service 指标合同。"""

import pytest

from core.data.simulation.rir_service import RIRServiceMetrics

pytestmark = pytest.mark.unit


def test_service_metrics_accumulate_cache_latency_queue_and_bytes(tmp_path):
    # 跨测试模块 helper 在用内导入：被依赖文件改名只影响本用例，不会连锁失败整个 collect
    from test_scene_level_rir_ipc import _fake_backend_factory, _scene, _server

    with _server(tmp_path, _fake_backend_factory) as client:
        first = client.generate_scene(_scene(), moving=False)
        second = client.generate_scene(_scene(), moving=False)

        metrics = client.metrics_snapshot()
        assert metrics["rir_request_count"] == 2
        assert metrics["rir_cache_hit_count"] == 1
        assert metrics["rir_cache_hit_rate"] == pytest.approx(0.5)
        assert metrics["rir_generation_count"] == 1
        assert metrics["rir_generation_latency_ms"] >= 0.0
        assert metrics["rir_queue_wait_ms"] >= 0.0
        assert metrics["rir_bytes_transferred"] > 0
        assert metrics["rir_request_kind_counts"]["scene_static"] == 2

        request_metrics = first["full"][1]["request_metrics"]
        assert request_metrics["cache_hit"] is False
        assert request_metrics["bytes_out"] > 0
        assert second["full"][1]["request_metrics"]["cache_hit"] is True
        assert "service_metrics" in second["full"][1]


def test_service_metrics_failure_and_timeout_counters_are_distinct():
    metrics = RIRServiceMetrics(device="fixture")
    metrics.record(
        request_kind="scene_static", track_kind="full_direct", cache_hit=False,
        queue_wait_ms=1.5, generation_latency_ms=2.5, bytes_in=10, bytes_out=20,
    )
    metrics.record(
        request_kind="scene_moving", track_kind="scene_moving_full_direct", cache_hit=False,
        queue_wait_ms=3.0, generation_latency_ms=0.0, bytes_in=5, bytes_out=0,
        failed=True, timed_out=True,
    )
    snapshot = metrics.snapshot()
    assert snapshot["rir_request_count"] == 2
    assert snapshot["rir_failure_count"] == 1
    assert snapshot["rir_timeout_count"] == 1
    assert snapshot["rir_bytes_transferred"] == 35
    assert snapshot["rir_service_gpu_id"] == "fixture"
