"""ONNX debug export、release preflight、cache ABI 与隔离边界合同。"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_release_preflight_requires_checkpoint_config_and_float32(tmp_path):
    from onnx.verify.verify_common import require_release_preflight

    common = dict(
        save_onnx=tmp_path / "model.onnx",
        thresholds={"mse_threshold": (1e-5, 1e-5)},
    )
    with pytest.raises(ValueError, match="checkpoint"):
        require_release_preflight(checkpoint=None, **common)
    with pytest.raises(ValueError, match="config"):
        require_release_preflight(
            checkpoint=tmp_path / "model.tar",
            config=None,
            require_config=True,
            **common,
        )
    with pytest.raises(ValueError, match="float32"):
        require_release_preflight(
            checkpoint=tmp_path / "model.tar",
            inference_dtype="float16",
            **common,
        )


@pytest.mark.parametrize("threshold", [True, 0.0, -1.0, float("nan"), float("inf"), 2e-5])
def test_release_preflight_rejects_invalid_or_relaxed_threshold(tmp_path, threshold):
    from onnx.verify.verify_common import require_release_preflight

    with pytest.raises(ValueError, match="mse_threshold"):
        require_release_preflight(
            save_onnx=tmp_path / "model.onnx",
            checkpoint=tmp_path / "model.tar",
            thresholds={"mse_threshold": (threshold, 1e-5)},
        )


def test_release_preflight_is_noop_without_save_target():
    from onnx.verify.verify_common import require_release_preflight

    require_release_preflight(
        save_onnx=None,
        checkpoint=None,
        config=None,
        require_config=True,
        inference_dtype="float16",
        thresholds={"mse_threshold": (math.inf, 1e-5)},
    )


@pytest.mark.parametrize(
    "statuses,counts,metrics",
    [
        (["PASS", "SKIP", "PASS"], [1, 1, 1], [0.0, 0.0, 0.0]),
        (["PASS", "PASS"], [1, 1], [0.0, 0.0]),
        (["PASS", "PASS", "PASS"], [1, 0, 1], [0.0, 0.0, 0.0]),
        (["PASS", "PASS", "PASS"], [1, 1, 1], [0.0, float("nan"), 0.0]),
        (["PASS", "PASS", "PASS"], [1, 1, 1], [0.0, float("inf"), 0.0]),
    ],
)
def test_release_results_fail_closed(statuses, counts, metrics):
    from onnx.verify.verify_common import release_results_pass

    assert not release_results_pass(
        statuses,
        compared_counts=counts,
        metrics=metrics,
        thresholds=[1e-5, 1e-5, 1e-5],
    )


def test_release_results_accept_only_complete_finite_threshold_pass():
    from onnx.verify.verify_common import release_results_pass

    assert release_results_pass(
        ["PASS", "PASS", "PASS"],
        compared_counts=[8, 8, 8],
        metrics=[1e-9, 2e-9, 3e-9],
        thresholds=[1e-5, 1e-5, 1e-5],
    )
    assert not release_results_pass(
        ["PASS", "PASS", "PASS"],
        compared_counts=[8, 8, 8],
        metrics=[1e-9, 2e-9, 3e-9, 4e-9],
        thresholds=[1e-5, 1e-5, 1e-5],
    )
    assert not release_results_pass(
        ["PASS", "PASS", "PASS"],
        compared_counts=[8, 8, 8],
        metrics=[1e-9, 2e-4, 3e-9],
        thresholds=[1e-5, 1e-5, 1e-5],
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"mse": 1e-5},
        {"mse": float("nan")},
        {"max_abs": float("inf")},
        {"compared_count": True},
        {"compared_count": 0},
        {"compared_count": 1.5},
        {"threshold": True},
        {"threshold": 0.0},
        {"threshold": float("nan")},
        {"status": "FAIL"},
    ],
)
def test_three_way_comparison_passed_revalidates_claimed_status(overrides):
    from onnx.verify.verify_common import ThreeWayComparison

    values = {
        "name": "offline_vs_ort",
        "mse": 1e-9,
        "max_abs": 1e-4,
        "compared_count": 8,
        "threshold": 1e-5,
        "status": "PASS",
    }
    values.update(overrides)
    assert not ThreeWayComparison(**values).passed


def test_three_way_comparison_accepts_complete_finite_result_below_threshold():
    from onnx.verify.verify_common import ThreeWayComparison

    assert ThreeWayComparison(
        name="offline_vs_ort",
        mse=1e-9,
        max_abs=1e-4,
        compared_count=8,
        threshold=1e-5,
        status="PASS",
    ).passed


def test_materialize_cache_shapes_validates_batch_and_static_axes():
    from onnx.utils import materialize_cache_shapes

    dynamic = [SimpleNamespace(name="cache", shape=["batch", 2, 3])]
    assert materialize_cache_shapes({"cache": [1, 2, 3]}, dynamic, 4) == {
        "cache": [4, 2, 3]
    }
    for batch_size in (True, 0, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            materialize_cache_shapes({"cache": [1, 2, 3]}, dynamic, batch_size)
    with pytest.raises(RuntimeError, match="batch template must be 1"):
        materialize_cache_shapes({"cache": [2, 2, 3]}, dynamic, 2)
    with pytest.raises(RuntimeError, match="static axis"):
        materialize_cache_shapes(
            {"cache": [1, 2, 4]},
            [SimpleNamespace(name="cache", shape=["batch", 2, 3])],
            2,
        )
    with pytest.raises(RuntimeError, match="only supports batch_size=1"):
        materialize_cache_shapes(
            {"cache": [1, 2]},
            [SimpleNamespace(name="cache", shape=[1, 2])],
            2,
        )


def test_cache_metadata_rejects_bool_dimension():
    from onnx.utils import parse_onnx_cache_config

    with pytest.raises(RuntimeError, match="invalid shape"):
        parse_onnx_cache_config({"cache_cfg": '{"cache":[true,2]}'}, "cache_cfg")


def test_require_cache_metadata_abi_checks_value_and_graph_shape():
    from onnx.verify.verify_common import require_cache_metadata_abi

    infos = [SimpleNamespace(name="cache", shape=["batch", 2])]
    require_cache_metadata_abi(
        {"cache_cfg": '{"cache":[1,2]}'},
        "cache_cfg",
        {"cache": [1, 2]},
        infos,
    )
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        require_cache_metadata_abi(
            {"cache_cfg": '{"cache":[1,3]}'},
            "cache_cfg",
            {"cache": [1, 2]},
            infos,
        )


def test_require_strict_runtime_spec_rejects_missing_runtime_metadata():
    from onnx.runtime_spec import RUNTIME_METADATA_KEY
    from onnx.verify.verify_common import require_strict_runtime_spec

    class CacheOnlySession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="spec", shape=["batch", 1, 1, 2]),
                SimpleNamespace(name="conv_cache", shape=["batch", 2]),
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="enh"), SimpleNamespace(name="conv_cache_out")]

        def get_modelmeta(self):
            return SimpleNamespace(
                custom_metadata_map={"gtcrn_cache_cfg": '{"conv_cache":[1,2]}'}
            )

    with pytest.raises(RuntimeError, match=RUNTIME_METADATA_KEY):
        require_strict_runtime_spec(CacheOnlySession(), "gtcrn")


def test_cagcrn_flat_wrapper_exposes_only_canonical_40_cache_abi():
    from tasks.aec.models.discriminative.cagcrn import CAGCRN
    from onnx.models.streaming_cagcrn import StreamingCAGCRN, StreamingCAGCRNFlat

    flat = StreamingCAGCRNFlat(StreamingCAGCRN(CAGCRN()).eval())

    assert len(flat.init_cache_flat()) == 40
    assert flat.get_input_names()[2:] == [f"cache_{index}" for index in range(40)]
    assert flat.get_output_names()[1:] == [f"out_cache_{index}" for index in range(40)]
    assert not hasattr(flat, "parse_delay_d_metadata")
    assert not hasattr(flat, "load_delay_d_from_onnx")


def test_pypi_onnx_scope_nested_restores_local_modules_and_path():
    import onnx.utils as local_utils
    from onnx.utils import pypi_onnx_scope

    original_path = list(sys.path)
    local_package = sys.modules["onnx"]
    local_utils_module = sys.modules["onnx.utils"]
    with pypi_onnx_scope() as external:
        assert external is sys.modules["onnx"]
        assert external is not local_package
        with pypi_onnx_scope() as nested:
            assert nested is external
    assert sys.path == original_path
    assert sys.modules["onnx"] is local_package
    assert sys.modules["onnx.utils"] is local_utils_module
    assert local_utils is local_utils_module


def test_generic_save_onnx_fails_before_model_construction(monkeypatch, tmp_path):
    import onnx.export as export_module

    args = SimpleNamespace(
        config=None,
        model="streaming_deepvqe",
        save_onnx=str(tmp_path / "release.onnx"),
        output=None,
    )
    monkeypatch.setattr(export_module, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="verify_3way_deepvqe.py"):
        export_module.main()


def test_debug_publish_rejects_unprepared_path_target_without_touching_targets(
    monkeypatch, tmp_path,
):
    import onnx.export as export_module

    # owned staging 固定在 _project_root/runs/onnx_debug/.tmp，必须隔离到 tmp_path，
    # 否则该测试会在真实仓库根重建 runs/onnx_debug/.tmp。
    monkeypatch.setattr(export_module, "_project_root", tmp_path)
    source = tmp_path / "source.onnx"
    source.write_bytes(b"new")
    target = tmp_path / "nested" / "debug.onnx"
    target.parent.mkdir()
    target.write_bytes(b"old")

    # 旧 Path-target API 只能通过 precheck 固定的 dirfd 发布：未 prepare 的目标
    # 必须在接触文件系统之前 fail-closed，且既不改写目标也不创建 staging。
    with pytest.raises(TypeError, match="prepared"):
        export_module._publish_debug_onnx(source, target)
    assert target.read_bytes() == b"old"
    assert source.read_bytes() == b"new"
    assert list(target.parent.iterdir()) == [target]
