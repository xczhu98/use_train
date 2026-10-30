"""模型注册公共边界与三任务 build smoke 契约。"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch.nn as nn

from core.models import register_lazy_models, register_task_models
from core.models.build import build_model
from core.utils.registry import MODEL_REGISTRY


@pytest.fixture(autouse=True)
def _restore_shared_model_registry():
    """模型注册表是进程级单例：测试新增的注册在结束后必须撤回。

    只还原 ``_registry``（``_lazy_map`` 由任务包导入幂等建立，跨测试稳定）。
    新增条目对应的具体模块若在本测试中才首次导入，需要一并移出
    ``sys.modules``：否则后续 ``ensure_registered`` 命中模块缓存而不会重新
    执行注册装饰器，反而制造顺序敏感。
    """
    registry_before = dict(MODEL_REGISTRY._registry)
    modules_before = set(sys.modules)
    yield
    added_modules = {
        MODEL_REGISTRY._lazy_map[name]
        for name in MODEL_REGISTRY._registry
        if name not in registry_before and name in MODEL_REGISTRY._lazy_map
    }
    for module_path in added_modules:
        if module_path in sys.modules and module_path not in modules_before:
            del sys.modules[module_path]
    MODEL_REGISTRY._registry.clear()
    MODEL_REGISTRY._registry.update(registry_before)


def test_core_registry_facade_keeps_task_packages_lazy():
    """导入公共 facade 不得顺带导入任何任务模型包。"""
    script = (
        "import sys; import core.models; "
        "assert 'tasks.se.models' not in sys.modules; "
        "assert 'tasks.aec.models' not in sys.modules; "
        "assert 'tasks.tse.models' not in sys.modules; "
        "assert 'tasks.ss.models' not in sys.modules; "
        "assert 'tasks.ssl.models' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    ("task_package", "concrete_module"),
    [
        ("tasks.se.models", "tasks.se.models.discriminative.conformer"),
        ("tasks.aec.models", "tasks.aec.models.discriminative.deepvqe"),
        ("tasks.ss.models", "tasks.ss.models.tdanet"),
        ("tasks.tse.models", "tasks.tse.models.discriminative.tse_bsrnn_spk"),
        ("tasks.tse.models", "tasks.tse.models.discriminative.tse_modular"),
        ("tasks.ssl.models", "tasks.ssl.models.ipdnet"),
    ],
)
def test_task_model_package_import_does_not_load_concrete_models(
    task_package, concrete_module
):
    script = (
        f"import importlib, sys; importlib.import_module({task_package!r}); "
        f"assert {concrete_module!r} not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_register_task_models_rejects_unknown_task():
    with pytest.raises(KeyError, match="Unknown model task.*unknown"):
        register_task_models("unknown")


@pytest.mark.parametrize(
    ("task", "model_type"),
    [
        ("se", "Conformer"),
        ("aec", "DeepVQE"),
        ("ss", "TDANet"),
        ("tse", "TSE_BSRNN_SPK"),
        ("tse", "TSE_Modular"),
        ("ssl", "FixedArrayIPDNet"),
    ],
)
def test_register_task_models_only_registers_requested_model(task, model_type):
    register_task_models(task, only=model_type)
    assert model_type in MODEL_REGISTRY


def test_task_lazy_maps_are_exposed_through_shared_registry():
    import tasks.aec.models
    import tasks.se.models
    import tasks.ss.models
    import tasks.ssl.models
    import tasks.tse.models

    expected = {
        **tasks.se.models._MODEL_REGISTRY_MAP,
        **tasks.aec.models._AEC_MODEL_REGISTRY_MAP,
        **tasks.ss.models._MODEL_REGISTRY_MAP,
        **tasks.ssl.models._MODEL_REGISTRY_MAP,
        **tasks.tse.models._TSE_MODEL_REGISTRY_MAP,
    }
    assert {name: MODEL_REGISTRY._lazy_map[name] for name in expected} == expected
    assert set(tasks.aec.models._AEC_MODEL_REGISTRY_MAP) == tasks.aec.models.AEC_MODEL_TYPES
    assert set(tasks.ss.models._MODEL_REGISTRY_MAP) == tasks.ss.models.SS_MODEL_TYPES
    assert set(tasks.ssl.models._MODEL_REGISTRY_MAP) == tasks.ssl.models.SSL_MODEL_TYPES
    assert set(tasks.tse.models._TSE_MODEL_REGISTRY_MAP) == tasks.tse.models.TSE_MODEL_TYPES


def _decorated_model_names(models_dir: Path) -> set[str]:
    names: set[str] = set()
    for source_path in models_dir.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            register = node.func
            if not (
                isinstance(register, ast.Attribute)
                and register.attr == "register"
                and isinstance(register.value, ast.Name)
                and register.value.id == "MODEL_REGISTRY"
            ):
                continue
            name = node.args[0]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                names.add(name.value)
    return names


def test_every_decorated_model_has_exactly_one_task_lazy_map_entry():
    import tasks.aec.models
    import tasks.se.models
    import tasks.ss.models
    import tasks.ssl.models
    import tasks.tse.models

    root = Path(__file__).resolve().parents[1]
    assert _decorated_model_names(root / "tasks" / "se" / "models") == set(
        tasks.se.models._MODEL_REGISTRY_MAP
    )
    assert _decorated_model_names(root / "tasks" / "aec" / "models") == set(
        tasks.aec.models._AEC_MODEL_REGISTRY_MAP
    )
    assert _decorated_model_names(root / "tasks" / "ss" / "models") == set(
        tasks.ss.models._MODEL_REGISTRY_MAP
    )
    assert _decorated_model_names(root / "tasks" / "ssl" / "models") == set(
        tasks.ssl.models._MODEL_REGISTRY_MAP
    )
    assert _decorated_model_names(root / "tasks" / "tse" / "models") == set(
        tasks.tse.models._TSE_MODEL_REGISTRY_MAP
    )


def test_existing_task_register_functions_remain_compatible():
    from tasks.aec.models import register_aec_models
    from tasks.se.models import register_se_models
    from tasks.ss.models import register_ss_models
    from tasks.ssl.models import register_ssl_models
    from tasks.tse.models import register_tse_models

    register_se_models(only="Conformer")
    register_aec_models(only="DeepVQE")
    register_ss_models(only="TDANet")
    register_ssl_models(only="FixedArrayIPDNet")
    register_tse_models(only="TSE_BSRNN_SPK")

    assert {"Conformer", "DeepVQE", "TDANet", "FixedArrayIPDNet", "TSE_BSRNN_SPK"} <= set(
        MODEL_REGISTRY.keys()
    )


@pytest.mark.parametrize(
    ("task", "model_cfg", "stft_cfg", "expected_class"),
    [
        (
            "se",
            {
                "type": "Conformer",
                "args": {
                    "num_blocks": 1,
                    "d_model": 16,
                    "n_heads": 4,
                    "d_ff": 32,
                    "depthwise_conv_kernel_size": 3,
                },
            },
            {"n_fft": 512, "hop_length": 256, "win_length": 512, "window": "hann"},
            "Conformer",
        ),
        (
            "aec",
            {"type": "DeepVQE", "args": {"variant": "small", "align_delay": 2}},
            {"n_fft": 320, "hop_length": 160, "win_length": 320, "window": "hann"},
            "DeepVQE",
        ),
        (
            "ssl",
            {
                "type": "FixedArrayIPDNet",
                "args": {"input_size": 2, "hidden_size": 8, "max_track": 2, "n_fft": 64},
            },
            {
                "n_fft": 64,
                "hop_length": 32,
                "win_length": 64,
                "window": "hamming",
                "center": False,
            },
            "FixedArrayIPDNet",
        ),
        (
            "tse",
            {
                "type": "TSE_BSRNN_SPK",
                "args": {
                    "feature_dim": 32,
                    "num_repeat": 1,
                    "joint_training": False,
                    "features": {"spkemb": {"enabled": True}},
                },
            },
            {
                "n_fft": 512,
                "hop_length": 128,
                "win_length": 512,
                "window": "hann",
                "center": True,
            },
            "TSE_BSRNN_SPK",
        ),
    ],
)
def test_registry_build_smoke(task, model_cfg, stft_cfg, expected_class):
    register_task_models(task, only=model_cfg["type"])
    model = build_model(model_cfg, stft_cfg=stft_cfg)
    assert isinstance(model, nn.Module)
    assert type(model).__name__ == expected_class


class _CapturedStftModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("n_fft", 320),
        ("hop_length", 256),
        ("win_length", 320),
        ("window", "hamming"),
        ("center", False),
    ],
)
def test_build_model_rejects_explicit_stft_conflict(monkeypatch, key, value):
    monkeypatch.setattr(MODEL_REGISTRY, "ensure_registered", lambda _name: None)
    monkeypatch.setattr(
        MODEL_REGISTRY,
        "build",
        lambda _name, **kwargs: _CapturedStftModel(**kwargs),
    )

    with pytest.raises(
        ValueError,
        match=rf"{key} context conflict.*model args\.{key}.*effective stft\.{key}",
    ):
        build_model(
            {"type": "CapturedStft", "args": {key: value}},
            stft_cfg={"n_fft": 512, "hop_length": 128},
        )


def test_build_model_accepts_matching_stft_args_with_normalized_window(monkeypatch):
    monkeypatch.setattr(MODEL_REGISTRY, "ensure_registered", lambda _name: None)
    monkeypatch.setitem(MODEL_REGISTRY._registry, "CapturedStft", _CapturedStftModel)

    model = build_model(
        {
            "type": "CapturedStft",
            "args": {
                "n_fft": 512,
                "hop_length": 128,
                "win_length": 512,
                "window": "HANN",
                "center": True,
            },
        },
        stft_cfg={"n_fft": 512, "hop_length": 128},
    )

    assert model.kwargs == {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": "HANN",
        "center": True,
    }


def test_build_model_injects_resolved_stft_defaults(monkeypatch):
    monkeypatch.setattr(MODEL_REGISTRY, "ensure_registered", lambda _name: None)
    monkeypatch.setitem(MODEL_REGISTRY._registry, "CapturedStft", _CapturedStftModel)

    model = build_model(
        {"type": "CapturedStft", "args": {}},
        stft_cfg={"n_fft": 512, "hop_length": 128},
    )

    assert model.kwargs == {
        "n_fft": 512,
        "hop_length": 128,
        "win_length": 512,
        "window": "hann",
        "center": True,
    }


def test_build_model_preserves_partial_stft_config_behavior(monkeypatch):
    monkeypatch.setattr(MODEL_REGISTRY, "ensure_registered", lambda _name: None)
    monkeypatch.setitem(MODEL_REGISTRY._registry, "CapturedStft", _CapturedStftModel)

    model = build_model(
        {"type": "CapturedStft", "args": {}},
        stft_cfg={"window": "hann", "center": False},
    )

    assert model.kwargs == {"window": "hann", "center": False}


def test_se_alias_registration_and_build_contract():
    register_task_models("se", only="FastEnhancerB")

    alias_class = MODEL_REGISTRY.get("FastEnhancerB")
    canonical_class = MODEL_REGISTRY.get("FastEnhancer")
    assert alias_class is canonical_class

    model = build_model({"type": "FastEnhancerB", "args": {"variant": "T"}})
    assert isinstance(model, canonical_class)


def _missing_module(name: str) -> ModuleNotFoundError:
    error = ModuleNotFoundError(f"No module named {name!r}")
    error.name = name
    return error


def test_bulk_registration_skips_explicit_optional_dependency(caplog, monkeypatch):
    calls = []

    def fake_ensure_registered(model_type):
        calls.append(model_type)
        if model_type == "OptionalModel":
            raise _missing_module("optional_backend.submodule")

    monkeypatch.setattr(MODEL_REGISTRY, "ensure_registered", fake_ensure_registered)
    register_lazy_models(
        {
            "RequiredModel": "example.required",
            "OptionalModel": "example.optional",
            "FollowingModel": "example.following",
        },
        task="se",
        logger=__import__("logging").getLogger(__name__),
        optional_dependencies={
            "OptionalModel": frozenset({"optional_backend"})
        },
    )

    assert calls == ["RequiredModel", "OptionalModel", "FollowingModel"]
    assert "optional dependency 'optional_backend' is missing" in caplog.text


def test_bulk_registration_skips_wrapped_explicit_optional_dependency(
    caplog, monkeypatch
):
    missing = _missing_module("optional_backend")
    wrapped = ImportError("optional backend unavailable")
    wrapped.__cause__ = missing
    monkeypatch.setattr(
        MODEL_REGISTRY,
        "ensure_registered",
        lambda _model_type: (_ for _ in ()).throw(wrapped),
    )

    register_lazy_models(
        {"OptionalModel": "example.optional"},
        task="se",
        logger=__import__("logging").getLogger(__name__),
        optional_dependencies={
            "OptionalModel": frozenset({"optional_backend"})
        },
    )

    assert "optional dependency 'optional_backend' is missing" in caplog.text


def test_bulk_registration_does_not_swallow_undeclared_missing_module(monkeypatch):
    monkeypatch.setattr(
        MODEL_REGISTRY,
        "ensure_registered",
        lambda _model_type: (_ for _ in ()).throw(_missing_module("required_backend")),
    )

    with pytest.raises(ModuleNotFoundError, match="required_backend"):
        register_lazy_models({"Broken": "example.broken"}, task="se")


def test_bulk_registration_does_not_swallow_wrong_optional_dependency(monkeypatch):
    monkeypatch.setattr(
        MODEL_REGISTRY,
        "ensure_registered",
        lambda _model_type: (_ for _ in ()).throw(_missing_module("transitive_bug")),
    )

    with pytest.raises(ModuleNotFoundError, match="transitive_bug"):
        register_lazy_models(
            {"OptionalModel": "example.optional"},
            task="se",
            optional_dependencies={
                "OptionalModel": frozenset({"optional_backend"})
            },
        )


def test_requested_optional_model_remains_fail_closed(monkeypatch):
    monkeypatch.setattr(
        MODEL_REGISTRY,
        "ensure_registered",
        lambda _model_type: (_ for _ in ()).throw(_missing_module("optional_backend")),
    )

    with pytest.raises(ModuleNotFoundError, match="optional_backend"):
        register_lazy_models(
            {"OptionalModel": "example.optional"},
            task="se",
            only="OptionalModel",
            optional_dependencies={
                "OptionalModel": frozenset({"optional_backend"})
            },
        )


def test_optional_dependency_declaration_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model types.*Missing"):
        register_lazy_models(
            {"Known": "example.known"},
            task="se",
            optional_dependencies={"Missing": frozenset({"optional_backend"})},
        )


pytestmark = [pytest.mark.unit]
