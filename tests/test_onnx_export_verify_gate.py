"""onnx/export.py generic 导出与可选 smoke 门控单元测试。

覆盖点：
- ``--smoke``/``--verify`` 与 ``--no-smoke``/``--no-verify`` CLI 选项解析。
- generic exporter 拒绝 ``--save-onnx``，正式发布必须由专属 ``verify_3way`` 脚本完成。
- debug 导出先消费仓库内 owned 临时 artifact，再写入显式 debug 目标。
- 普通 direct 分支：smoke 返回 False 时 ``raise SystemExit(1)`` 且不打印 Done。
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

pytestmark = [pytest.mark.unit, pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed in this environment")]


@pytest.fixture()
def exp():
    """导入 onnx.export 模块（仓库根目录的 onnx/ 为本地包）。"""
    import onnx.export as m

    return m


def _make_args(**overrides):
    """构造 main() 所需的最小 args（走 streaming_gtcrn 直接导出分支）。"""
    defaults = dict(
        config=None,
        model="streaming_gtcrn",
        checkpoint=None,
        output="out.onnx",
        save_onnx=None,
        opset=17,
        verify=False,
        simplify=False,
        verify_streaming=False,
        mse_threshold=1e-4,
        export_state_detector=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FakeWrapper:
    """供 GTCRN metadata 注入分支使用的最小 wrapper（init_cache 返回空）。"""

    def init_cache(self, batch, device, dtype):
        return {}


def test_export_model_uses_registered_direct_exporter(monkeypatch, exp):
    """direct registry 是 export_model 的生产分派入口。"""
    import torch

    class _RegistryTestModel(torch.nn.Module):
        pass

    calls = []
    expected = (_FakeWrapper(), ("dummy",))

    def fake_exporter(model, output_path, opset):
        calls.append((model, output_path, opset))
        return expected

    monkeypatch.setitem(
        exp._STREAMING_EXPORTERS, "_RegistryTestModel", fake_exporter
    )
    model = _RegistryTestModel()

    result = exp.export_model(model, "debug.onnx", 16)

    assert result is expected
    assert calls == [(model, "debug.onnx", 16)]


def test_dbgnet_primary_output_name_is_canonical_enh():
    """DBGNet 忽略历史私有重命名属性，始终导出 canonical ``enh``。"""
    import torch

    from onnx.models.streaming_dbgnet import StreamingDBGNetFlat

    class _DummyStreamingDBGNet(torch.nn.Module):
        n_fft = 320
        hop_length = 160
        win_length = 320
        nfreqs = 161
        target_fs = 16000
        window = "hann"
        center = True
        use_state_detector = False
        state_nclass = 3
        network_lookahead_frames = 0
        algorithmic_delay_frames = 0
        algorithmic_delay = 0
        deployment_config = {}
        CACHE_KEYS = ()

    wrapper = StreamingDBGNetFlat(_DummyStreamingDBGNet())
    wrapper._primary_output_name = "enhanced"

    output_names = wrapper.get_output_names()
    dynamic_axes = wrapper.get_dynamic_axes()
    assert output_names[0] == "enh"
    assert "enhanced" not in dynamic_axes
    assert dynamic_axes == {}


@pytest.fixture()
def patched_main(monkeypatch, exp, tmp_path):
    """打桩 main() 的重型依赖：模型构建、导出、metadata 与发布。"""
    import torch

    import onnx.models.streaming_gtcrn as gtcrn_mod
    from onnx.verify import verify_common

    class _DummyModel(torch.nn.Module):
        pass

    def fake_export(_model, output, _opset):
        Path(output).write_bytes(b"verified-onnx")
        return _FakeWrapper(), ()

    monkeypatch.setattr(gtcrn_mod, "StreamingGTCRN", _DummyModel)
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    monkeypatch.setattr(exp, "export_model", fake_export)
    monkeypatch.setattr(exp, "write_streaming_runtime_metadata", lambda *a, **k: {})
    monkeypatch.setattr(verify_common, "_ONNX_RELEASE_ROOT", tmp_path)
    return exp


def test_parse_args_smoke_default_off_and_no_smoke(monkeypatch, exp):
    """默认关闭可选 smoke；``--smoke`` 可显式开启。"""
    monkeypatch.setattr("sys.argv", ["export.py", "-C", "x.yaml"])
    parsed = exp.parse_args()
    assert parsed.verify is False
    assert parsed.output == "runs/onnx_debug/model.onnx"
    assert parsed.save_onnx is None

    monkeypatch.setattr("sys.argv", ["export.py", "-C", "x.yaml", "--no-smoke"])
    assert exp.parse_args().verify is False

    monkeypatch.setattr("sys.argv", ["export.py", "-C", "x.yaml", "--smoke"])
    assert exp.parse_args().verify is True

    monkeypatch.setattr("sys.argv", ["export.py", "-C", "x.yaml", "--verify"])
    parsed = exp.parse_args()
    assert parsed.verify is True
    assert parsed.verify_streaming is False

    monkeypatch.setattr("sys.argv", ["export.py", "-C", "x.yaml", "--no-verify"])
    assert exp.parse_args().verify is False

    monkeypatch.setattr(
        "sys.argv",
        ["export.py", "-C", "x.yaml", "--save-onnx", "onnx/onnx_model/released.onnx"],
    )
    parsed = exp.parse_args()
    assert parsed.save_onnx == "onnx/onnx_model/released.onnx"


def test_verify_or_exit_raises_on_failure(monkeypatch, exp):
    """普通 direct 分支：verify_onnx 返回 False 时 SystemExit(1)。"""
    monkeypatch.setattr(exp, "verify_onnx", lambda *a, **k: False)
    with pytest.raises(SystemExit) as excinfo:
        exp._verify_or_exit(_make_args(), "out.onnx", _FakeWrapper(), ())
    assert excinfo.value.code == 1


def test_verify_or_exit_raises_on_verify_exception(monkeypatch, exp):
    """普通 direct 分支：verify_onnx 抛异常（如 ORT 加载失败）同样门控为 SystemExit(1)。"""
    def _boom(*a, **k):
        raise RuntimeError("ORT load failed")

    monkeypatch.setattr(exp, "verify_onnx", _boom)
    with pytest.raises(SystemExit) as excinfo:
        exp._verify_or_exit(_make_args(), "out.onnx", _FakeWrapper(), ())
    assert excinfo.value.code == 1


def test_verify_or_exit_passes_on_success(monkeypatch, exp):
    """普通 direct 分支：verify_onnx 返回 True 时不抛出。"""
    monkeypatch.setattr(exp, "verify_onnx", lambda *a, **k: True)
    exp._verify_or_exit(_make_args(), "out.onnx", _FakeWrapper(), ())


def test_main_gates_failed_smoke_and_skips_done(monkeypatch, caplog, patched_main, tmp_path):
    """smoke 失败：不调用正式发布 helper，且已有 debug 目标保持不变。"""
    exp = patched_main
    target = tmp_path / "runs" / "onnx_debug" / "out.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous-debug")
    verify_paths = []

    def fake_verify(path, *_args):
        verify_paths.append(Path(path))
        return False

    monkeypatch.setattr(exp, "parse_args", lambda: _make_args(output=str(target), verify=True))
    monkeypatch.setattr(exp, "verify_onnx", fake_verify)

    with caplog.at_level(logging.INFO, logger="export_onnx"):
        with pytest.raises(SystemExit) as excinfo:
            exp.main()
    assert excinfo.value.code == 1
    assert "Done." not in caplog.text
    assert len(verify_paths) == 1
    assert verify_paths[0] != target
    assert not verify_paths[0].exists()
    assert target.read_bytes() == b"previous-debug"


def test_main_prints_done_when_smoke_passes(monkeypatch, caplog, patched_main, tmp_path):
    """smoke 通过：先验证 owned 临时源，再原子写入 debug 目标。"""
    exp = patched_main
    target = tmp_path / "runs" / "onnx_debug" / "out.onnx"
    verify_paths = []

    def fake_verify(path, *_args):
        verify_paths.append(Path(path))
        return True

    monkeypatch.setattr(exp, "parse_args", lambda: _make_args(output=str(target), verify=True))
    monkeypatch.setattr(exp, "verify_onnx", fake_verify)

    with caplog.at_level(logging.INFO, logger="export_onnx"):
        exp.main()
    assert "Done." in caplog.text
    assert len(verify_paths) == 1
    assert verify_paths[0] != target
    assert target.read_bytes() == b"verified-onnx"
    assert not verify_paths[0].exists()


def test_main_runs_post_export_steps_on_same_owned_path(
    monkeypatch, patched_main, tmp_path,
):
    """metadata、simplify、verify 必须共同消费 owned 临时源，而非发布目标。"""
    exp = patched_main
    target = tmp_path / "runs" / "onnx_debug" / "out.onnx"
    seen = {"metadata": [], "simplify": [], "verify": []}

    def fake_metadata(path, *_args, **_kwargs):
        seen["metadata"].append(Path(path))

    def fake_simplify(path):
        seen["simplify"].append(Path(path))

    def fake_verify(path, *_args):
        seen["verify"].append(Path(path))
        return True

    monkeypatch.setattr(
        exp, "parse_args", lambda: _make_args(output=str(target), simplify=True, verify=True),
    )
    monkeypatch.setattr(exp, "write_streaming_runtime_metadata", fake_metadata)
    monkeypatch.setattr(exp, "simplify_onnx", fake_simplify)
    monkeypatch.setattr(exp, "verify_onnx", fake_verify)

    exp.main()

    assert seen["metadata"]
    assert seen["simplify"] == seen["verify"]
    assert seen["metadata"] == seen["verify"]
    assert seen["verify"][0] != target
    assert target.read_bytes() == b"verified-onnx"


@pytest.mark.parametrize("verify", [False, True])
def test_main_rejects_generic_save_target(monkeypatch, patched_main, tmp_path, verify):
    """generic exporter 无论是否 smoke 都不得接管正式发布参数。"""
    exp = patched_main
    target = tmp_path / "release.onnx"
    monkeypatch.setattr(
        exp,
        "parse_args",
        lambda: _make_args(
            output="model.onnx", save_onnx=str(target), verify=verify,
        ),
    )

    with pytest.raises(ValueError, match="generic onnx/export.py --save-onnx"):
        exp.main()
    assert not target.exists()


def test_main_without_smoke_writes_debug_output(monkeypatch, caplog, patched_main, tmp_path):
    """未启用 smoke 时，将 owned 产物写到显式 debug 目标。"""
    exp = patched_main
    target = tmp_path / "runs" / "onnx_debug" / "debug.onnx"
    calls = []

    def _spy(*a, **k):
        calls.append(a)
        return False

    monkeypatch.setattr(exp, "parse_args", lambda: _make_args(verify=False, output=str(target)))
    monkeypatch.setattr(exp, "verify_onnx", _spy)

    with caplog.at_level(logging.INFO, logger="export_onnx"):
        exp.main()
    assert calls == []
    assert "Done." in caplog.text
    assert target.read_bytes() == b"verified-onnx"


def test_main_verified_debug_output_skips_release_publisher(
    monkeypatch, caplog, patched_main, tmp_path,
):
    """带验证的 ``--output`` 写入 debug 目标，不调用正式发布 helper。"""
    exp = patched_main
    target = tmp_path / "runs" / "onnx_debug" / "debug-verified.onnx"
    verify_paths = []

    def fake_verify(path, *_args):
        verify_paths.append(Path(path))
        return True

    monkeypatch.setattr(exp, "parse_args", lambda: _make_args(output=str(target), verify=True))
    monkeypatch.setattr(exp, "verify_onnx", fake_verify)

    with caplog.at_level(logging.INFO, logger="export_onnx"):
        exp.main()

    assert verify_paths and verify_paths[0] != target
    assert target.read_bytes() == b"verified-onnx"
    assert not verify_paths[0].exists()
    assert "Debug/general ONNX exported to:" in caplog.text


def _patch_gtcrn_demo_roots(monkeypatch, gtcrn_module, tmp_path):
    """将 GTCRN demo 的 debug/release 根目录隔离到测试临时目录。"""
    debug_root = tmp_path / "runs" / "onnx_debug"
    release_root = tmp_path / "onnx" / "onnx_model"
    monkeypatch.setattr(gtcrn_module, "_GTCRN_DEMO_DEBUG_ROOT", debug_root)
    monkeypatch.setattr(gtcrn_module, "_GTCRN_RELEASE_ROOT", release_root)
    return debug_root, release_root


@pytest.fixture()
def gtcrn_demo_module():
    """导入 GTCRN demo 模块，不执行其 ``__main__`` 导出路径。"""
    import onnx.models.streaming_gtcrn as module

    return module


@pytest.mark.parametrize("target_kind", ["root", "child"])
def test_gtcrn_demo_rejects_release_root(
    monkeypatch, tmp_path, gtcrn_demo_module, target_kind,
):
    """demo 不能把 release-root 或其子路径当作 debug 输出。"""
    _debug_root, release_root = _patch_gtcrn_demo_roots(
        monkeypatch, gtcrn_demo_module, tmp_path,
    )
    target = release_root if target_kind == "root" else release_root / "published.onnx"

    with pytest.raises(ValueError, match="release root"):
        gtcrn_demo_module._validate_demo_output_path(target)
    assert not target.exists()


def test_gtcrn_demo_rejects_existing_target_without_overwrite(
    monkeypatch, tmp_path, gtcrn_demo_module,
):
    """demo 遇到已有 debug 目标必须失败，不能静默复用或覆盖。"""
    debug_root, _release_root = _patch_gtcrn_demo_roots(
        monkeypatch, gtcrn_demo_module, tmp_path,
    )
    target = debug_root / "existing.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"keep-me")

    with pytest.raises(ValueError, match="already exists"):
        gtcrn_demo_module._validate_demo_output_path(target)
    assert target.read_bytes() == b"keep-me"


@pytest.mark.parametrize("target_name", ["outside.onnx", "nested/not-an-onnx.txt"])
def test_gtcrn_demo_rejects_invalid_debug_path(
    monkeypatch, tmp_path, gtcrn_demo_module, target_name,
):
    """debug 目录之外或非 ONNX 后缀的路径必须 fail-closed。"""
    debug_root, _release_root = _patch_gtcrn_demo_roots(
        monkeypatch, gtcrn_demo_module, tmp_path,
    )
    target = tmp_path / target_name
    if target_name.startswith("nested/"):
        target = debug_root / "nested" / "not-an-onnx.txt"

    with pytest.raises(ValueError, match=r"debug directory|\.onnx file"):
        gtcrn_demo_module._validate_demo_output_path(target)
    assert not target.exists()


def test_gtcrn_demo_rejects_symlinked_debug_parent(
    monkeypatch, tmp_path, gtcrn_demo_module,
):
    """debug 子目录不能通过 symlink 将 demo 输出绕到 release 目录。"""
    debug_root, release_root = _patch_gtcrn_demo_roots(
        monkeypatch, gtcrn_demo_module, tmp_path,
    )
    debug_root.mkdir(parents=True)
    release_root.mkdir(parents=True)
    escaped_parent = debug_root / "release-link"
    escaped_parent.symlink_to(release_root, target_is_directory=True)
    target = escaped_parent / "escaped.onnx"

    with pytest.raises(ValueError, match=r"release root|parent must resolve"):
        gtcrn_demo_module._validate_demo_output_path(target)
    assert not target.exists()


def test_gtcrn_demo_main_and_simplified_use_owned_safe_publisher(
    monkeypatch, tmp_path, exp, gtcrn_demo_module,
):
    """GTCRN demo 主图与简化图都不得 direct-write 最终目标。"""
    debug_root, _release_root = _patch_gtcrn_demo_roots(
        monkeypatch, gtcrn_demo_module, tmp_path,
    )
    monkeypatch.setattr(gtcrn_demo_module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    target = debug_root / "demo.onnx"
    export_paths = []
    metadata_paths = []
    publish_pairs = []
    real_publish = exp._publish_debug_onnx

    def fake_export(_model, path, opset):
        assert opset == 17
        export_paths.append(Path(path))
        Path(path).write_bytes(b"main")
        return _FakeWrapper(), ()

    def fake_metadata(path, *_args):
        metadata_paths.append(Path(path))

    def spy_publish(source, prepared_target):
        publish_pairs.append((Path(source), prepared_target.display_path))
        return real_publish(source, prepared_target)

    fake_onnx = SimpleNamespace(
        load=lambda path: Path(path).read_bytes(),
        save=lambda model, path: Path(path).write_bytes(model),
    )
    fake_onnxsim = SimpleNamespace(simplify=lambda _model: (b"simple", True))
    monkeypatch.setattr(exp, "export_model", fake_export)
    monkeypatch.setattr(exp, "write_streaming_runtime_metadata", fake_metadata)
    monkeypatch.setattr(exp, "_publish_debug_onnx", spy_publish)
    monkeypatch.setattr("onnx.utils.pypi_onnx_scope", lambda: nullcontext(fake_onnx))
    monkeypatch.setitem(sys.modules, "onnxsim", fake_onnxsim)

    main_path, simple_path = gtcrn_demo_module._export_demo_onnx_artifacts(
        object(), target,
    )

    assert main_path == target
    assert simple_path == debug_root / "demo_simple.onnx"
    assert metadata_paths[:1] == export_paths
    assert len(metadata_paths) == 2
    assert all(path != final for path in export_paths for _source, final in publish_pairs)
    assert [final for _source, final in publish_pairs] == [main_path, simple_path]
    assert target.read_bytes() == b"main"
    assert simple_path.read_bytes() == b"simple"


def test_owned_debug_artifact_stays_outside_release_root(monkeypatch, tmp_path, exp):
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    artifact = exp._create_owned_onnx_artifact()
    try:
        assert artifact.display_path.is_relative_to(
            tmp_path / "runs" / "onnx_debug" / ".tmp"
        )
        assert not artifact.display_path.is_relative_to(tmp_path / "onnx" / "onnx_model")
    finally:
        artifact.close()


@pytest.mark.subprocess
def test_owned_debug_artifact_path_is_readable_by_metadata_subprocess(
    monkeypatch, tmp_path, exp,
):
    """owned 路径必须能被 metadata 子进程稳定访问。"""
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    artifact = exp._create_owned_onnx_artifact()
    try:
        Path(artifact).write_bytes(b"onnx")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                str(Path(artifact)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "onnx"
    finally:
        artifact.close()


def test_owned_debug_artifact_replacement_fails_closed(monkeypatch, tmp_path, exp):
    """owned 名称被替换后，不得发布攻击者控制的新 inode。"""
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    artifact = exp._create_owned_onnx_artifact()
    target = exp._prepare_debug_output_target(
        tmp_path / "runs" / "onnx_debug" / "target.onnx"
    )
    original_path = Path(artifact)
    moved_path = original_path.with_name(original_path.name + ".moved")
    original_path.write_bytes(b"verified")
    original_path.rename(moved_path)
    original_path.write_bytes(b"attacker")
    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            exp._publish_debug_onnx(artifact, target)
        assert not target.display_path.exists()
        assert artifact.display_path.read_bytes() == b"attacker"
    finally:
        artifact.close()
        target.close()


def test_owned_debug_artifact_rejects_symlinked_tmp_root(monkeypatch, tmp_path, exp):
    """owned ``.tmp`` 不能经 symlink 落入 release root。"""
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    debug_root = tmp_path / "runs" / "onnx_debug"
    release_root = tmp_path / "onnx" / "onnx_model"
    debug_root.mkdir(parents=True)
    release_root.mkdir(parents=True)
    (debug_root / ".tmp").symlink_to(release_root, target_is_directory=True)

    with pytest.raises((OSError, RuntimeError, ValueError)):
        exp._create_owned_onnx_artifact()
    assert list(release_root.iterdir()) == []


def test_debug_target_must_be_anchored_under_debug_root(monkeypatch, tmp_path, exp):
    """generic debug 输出不能写到显式 debug 根之外。"""
    monkeypatch.setattr(exp, "_project_root", tmp_path)

    with pytest.raises(ValueError, match="runs/onnx_debug"):
        exp._prepare_debug_output_target(tmp_path / "outside.onnx")
    assert not (tmp_path / "outside.onnx").exists()


def test_legacy_path_target_probe_never_publishes(monkeypatch, tmp_path, exp):
    """旧 Path-target seam 仅用于兼容故障注入，成功复制也必须 fail-closed。"""
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    source = tmp_path / "source.onnx"
    source.write_bytes(b"new")
    target = tmp_path / "legacy.onnx"

    with pytest.raises(TypeError, match="prepared"):
        exp._publish_debug_onnx(source, target)
    assert not target.exists()


def test_debug_publish_keeps_prechecked_parent_dirfd_during_symlink_swap(
    monkeypatch, tmp_path, exp,
):
    """目标 parent 在 precheck 后换链，发布仍只写稳定的已打开目录。"""
    monkeypatch.setattr(exp, "_project_root", tmp_path)
    stable_parent = tmp_path / "runs" / "onnx_debug" / "stable"
    escaped_parent = tmp_path / "onnx" / "onnx_model"
    stable_parent.mkdir(parents=True)
    escaped_parent.mkdir(parents=True)
    target = exp._prepare_debug_output_target(stable_parent / "debug.onnx")
    owned = exp._create_owned_onnx_artifact()
    Path(owned).write_bytes(b"new")

    moved_parent = stable_parent.with_name("stable-opened")
    stable_parent.rename(moved_parent)
    stable_parent.symlink_to(escaped_parent, target_is_directory=True)
    try:
        exp._publish_debug_onnx(owned, target)
    finally:
        target.close()
        owned.close()

    assert (moved_parent / "debug.onnx").read_bytes() == b"new"
    assert not (escaped_parent / "debug.onnx").exists()
