"""历史生产 checkpoint loader 安全入口契约测试。"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import core.metrics.mos_models as mos_models
import core.utils.safe_load as safe_load
import evaluate.latency_verification as latency_verification
import tools.model_tools.export_pase_artifacts as pase_export
import tools.verify.infer_official_ckpts as official_ckpts

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_ROOTS = ("core", "tasks", "evaluate", "onnx", "tools")
_SAFE_LOAD_WRAPPER = Path("core/utils/safe_load.py")


def _bare_torch_load_calls(tree: ast.Module) -> list[int]:
    """用 AST 识别模块级裸调用及其别名绕过；注释和字符串不计入。"""
    torch_aliases = {"torch"}
    load_aliases = set()

    def scope_bindings(
        body: list[ast.stmt], args: ast.arguments | None = None
    ) -> set[str]:
        """收集当前作用域的绑定名，不进入嵌套作用域。

        函数内 import 绑定的 torch 本体/真实 loader 不进入遮蔽集合，改写
        ``torch_aliases`` / ``load_aliases``，避免漏检显式导入后的裸调用。
        """
        names = set()

        def add_target(target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    add_target(element)

        class BindingVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                names.add(node.name)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                names.add(node.name)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                return

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    add_target(target)
                self.visit(node.value)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                add_target(node.target)
                if node.value is not None:
                    self.visit(node.value)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                add_target(node.target)
                self.visit(node.value)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                add_target(node.target)
                self.visit(node.value)

            def visit_For(self, node: ast.For) -> None:
                add_target(node.target)
                self.generic_visit(node)

            visit_AsyncFor = visit_For

            def visit_With(self, node: ast.With) -> None:
                for item in node.items:
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)
                self.generic_visit(node)

            visit_AsyncWith = visit_With

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                if node.name is not None:
                    names.add(node.name)
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    # 函数内 `import torch` / `import torch as t` / `import torch.nn`
                    # 绑定的仍是 torch 本体，不构成遮蔽：登记为 loader 别名，
                    # 避免漏检显式导入后的 torch.load；torch 子模块别名仍按遮蔽处理。
                    if alias.name == "torch" or (
                        alias.asname is None and alias.name.startswith("torch.")
                    ):
                        torch_aliases.add(bound)
                        continue
                    names.add(bound)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    # 同理，函数内 `from torch import load` 绑定的是真实 loader。
                    if node.module == "torch" and alias.name == "load":
                        load_aliases.add(bound)
                        continue
                    names.add(bound)

        if args is not None:
            names.update(argument.arg for argument in args.posonlyargs)
            names.update(argument.arg for argument in args.args)
            names.update(argument.arg for argument in args.kwonlyargs)
            if args.vararg is not None:
                names.add(args.vararg.arg)
            if args.kwarg is not None:
                names.add(args.kwarg.arg)
        visitor = BindingVisitor()
        for statement in body:
            visitor.visit(statement)
        return names

    def is_torch_module(value: ast.expr, shadowed: set[str]) -> bool:
        return (
            isinstance(value, ast.Name)
            and value.id in torch_aliases
            and value.id not in shadowed
        )

    def is_torch_load(value: ast.expr, shadowed: set[str]) -> bool:
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "load"
            and is_torch_module(value.value, shadowed)
        ) or (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) >= 2
            and is_torch_module(value.args[0], shadowed)
            and isinstance(value.args[1], ast.Constant)
            and value.args[1].value == "load"
        )

    # 仅跟踪模块顶层绑定，避免把函数内同名局部变量误判为 torch loader。
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            for alias in node.names:
                if alias.name == "load":
                    load_aliases.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if is_torch_module(value, set()):
                    torch_aliases.add(target.id)
                elif is_torch_load(value, set()):
                    load_aliases.add(target.id)

    class BareTorchLoadVisitor(ast.NodeVisitor):
        def __init__(self):
            self.lines: list[int] = []
            self.scopes: list[tuple[str, set[str]]] = []

        def _shadowed_names(self) -> set[str]:
            if self.scopes and self.scopes[-1][0] == "function":
                # 类体绑定不会成为方法的词法闭包变量。
                return set().union(
                    *(names for kind, names in self.scopes if kind == "function")
                )
            return set().union(*(names for _, names in self.scopes))

        def visit_Call(self, node: ast.Call) -> None:
            shadowed = self._shadowed_names()
            direct_call = is_torch_load(node.func, shadowed)
            imported_call = (
                isinstance(node.func, ast.Name)
                and node.func.id in load_aliases
                and node.func.id not in shadowed
            )
            if direct_call or imported_call:
                self.lines.append(node.lineno)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if default is not None:
                    self.visit(default)
            self.scopes.append(("function", scope_bindings(node.body, node.args)))
            for statement in node.body:
                self.visit(statement)
            self.scopes.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword.value)
            self.scopes.append(("class", scope_bindings(node.body)))
            for statement in node.body:
                self.visit(statement)
            self.scopes.pop()

    visitor = BareTorchLoadVisitor()
    visitor.visit(tree)
    return sorted(visitor.lines)


@pytest.mark.unit
@pytest.mark.contract
def test_bare_torch_load_gate_ignores_function_local_shadowed_alias():
    """函数内同名局部变量覆盖模块 loader 别名时不应误报。"""
    source = (
        "from torch import load\n"
        "def inspect(path):\n"
        "    load = lambda value: value\n"
        "    return load(path)\n"
    )

    assert _bare_torch_load_calls(ast.parse(source)) == []


@pytest.mark.unit
@pytest.mark.contract
def test_bare_torch_load_gate_reports_unshadowed_function_alias_call():
    """函数内未遮蔽的模块 loader 别名仍必须被识别。"""
    source = "from torch import load\ndef restore(path):\n    return load(path)\n"

    assert _bare_torch_load_calls(ast.parse(source)) == [3]


@pytest.mark.unit
@pytest.mark.contract
def test_bare_torch_load_gate_reports_function_local_torch_import():
    """函数内显式 import torch 后的 torch.load 不得被当成遮蔽漏检。"""
    source = "def restore(path):\n    import torch\n    return torch.load(path)\n"
    aliased = "def restore(path):\n    import torch as t\n    return t.load(path)\n"

    assert _bare_torch_load_calls(ast.parse(source)) == [3]
    assert _bare_torch_load_calls(ast.parse(aliased)) == [3]


@pytest.mark.unit
@pytest.mark.contract
def test_bare_torch_load_gate_reports_function_local_loader_import():
    """函数内 `from torch import load` 后的调用同样必须被识别。"""
    source = "def restore(path):\n    from torch import load\n    return load(path)\n"

    assert _bare_torch_load_calls(ast.parse(source)) == [3]


def test_production_code_has_no_bare_torch_load_calls(parse_python_ast):
    """自有生产代码必须统一经过 safe_torch_load，封装本身除外。"""
    violations = []
    candidates = [_PROJECT_ROOT / "inference.py"]
    for root_name in _PRODUCTION_ROOTS:
        candidates.extend((_PROJECT_ROOT / root_name).rglob("*.py"))
    for path in candidates:
        relative = path.relative_to(_PROJECT_ROOT)
        if relative == _SAFE_LOAD_WRAPPER or "tests" in relative.parts:
            continue
        lines = _bare_torch_load_calls(parse_python_ast(path))
        violations.extend(f"{relative}:{line}" for line in lines)
    assert not violations, "裸 torch.load 调用:\n" + "\n".join(violations)


class _TinyLatencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def test_latency_loader_uses_safe_loader_for_embedded_and_explicit_config(
    monkeypatch, tmp_path
):
    """两种配置来源都必须经 FD 绑定安全入口，且不请求 unsafe pickle。"""
    import core.models.build as model_build
    import core.utils.config as config_utils

    model = _TinyLatencyModel()
    cfg = {"model": {"type": "tiny"}, "data": {"mode": "dynamic", "sample_rate": 8000}}
    checkpoint = tmp_path / "model.tar"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def fake_safe_load(stream, **kwargs):
        calls.append(
            {
                "readable": stream.readable(),
                "seekable": stream.seekable(),
                "kwargs": kwargs,
            }
        )
        return {"model": model.state_dict(), "config": cfg}

    monkeypatch.setattr(safe_load, "safe_torch_load", fake_safe_load)
    monkeypatch.setattr(model_build, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(config_utils, "load_merged_config", lambda _path: cfg)

    for config_path in (None, "config.yaml"):
        loaded, stft_mod, sample_rate = latency_verification.load_se_aec_model(
            str(checkpoint), config_path, torch.device("cpu")
        )
        assert loaded is model
        assert stft_mod is None
        assert sample_rate == 8000

    assert calls == [
        {
            "readable": True,
            "seekable": True,
            "kwargs": {
                "map_location": "cpu",
                "purpose": "latency verification model checkpoint",
            },
        },
        {
            "readable": True,
            "seekable": True,
            "kwargs": {
                "map_location": "cpu",
                "purpose": "latency verification model checkpoint",
            },
        },
    ]
    assert all("allow_unsafe" not in call["kwargs"] for call in calls)


def test_latency_loader_rejects_config_without_data_mode(monkeypatch, tmp_path):
    """缺少 data.mode 时不得静默推断任务，必须在配置边界报错。"""
    import core.models.build as model_build

    model = _TinyLatencyModel()
    checkpoint = tmp_path / "model.tar"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        safe_load,
        "safe_torch_load",
        lambda *_args, **_kwargs: {
            "model": model.state_dict(),
            "config": {"model": {"type": "tiny"}, "data": {"sample_rate": 8000}},
        },
    )
    monkeypatch.setattr(model_build, "build_model", lambda *_args, **_kwargs: model)

    with pytest.raises(ValueError, match=r"data\.mode"):
        latency_verification.load_se_aec_model(
            str(checkpoint), None, torch.device("cpu")
        )


def test_pase_train_loader_uses_safe_loader_without_unsafe_fallback(monkeypatch):
    calls = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"config": {}}

    monkeypatch.setattr(safe_load, "safe_torch_load", fake_safe_load)

    assert pase_export._load_train_ckpt("train.tar") == {"config": {}}
    assert calls == [("train.tar", {"purpose": "export training artifact"})]
    assert "allow_unsafe" not in calls[0][1]


class _TinyCFM(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def test_pase_artifact_verify_uses_safe_loader(monkeypatch):
    module_name = "tasks.se.models.pase_family.stupase.flow.cfm"
    fake_module = types.ModuleType(module_name)
    fake_module.CFM = _TinyCFM
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    calls = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"cfg": {"unused": True}, "model": {"weight": torch.ones(1)}}

    monkeypatch.setattr(safe_load, "safe_torch_load", fake_safe_load)

    pase_export._verify("cfm", "artifact.pt")

    assert calls == [
        (
            "artifact.pt",
            {
                "map_location": "cpu",
                "purpose": "verify exported PASE artifact",
            },
        )
    ]
    assert "allow_unsafe" not in calls[0][1]


def test_utmos_loader_uses_safe_loader_for_bundled_weights(monkeypatch, tmp_path):
    class _TinyUTMOS(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    module_names = (
        "speechmos",
        "speechmos.utmos22",
        "speechmos.utmos22.strong",
        "speechmos.utmos22.strong.model",
    )
    modules = [types.ModuleType(name) for name in module_names]
    modules[-1].UTMOS22Strong = _TinyUTMOS
    modules[0].utmos22 = modules[1]
    modules[1].strong = modules[2]
    modules[2].model = modules[3]
    for name, module in zip(module_names, modules):
        monkeypatch.setitem(sys.modules, name, module)

    (tmp_path / "SpeechMOS").mkdir()
    calls = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(mos_models, "safe_torch_load", fake_safe_load)

    scorer = mos_models._UTMOSScorer(str(tmp_path), device="cpu")

    assert scorer._model.weight.item() == pytest.approx(1.0)
    assert calls == [
        (
            str(tmp_path / "utmos22_strong_step7459_v1.pt"),
            {
                "map_location": "cpu",
                "purpose": "load bundled UTMOS weights",
            },
        )
    ]
    assert "allow_unsafe" not in calls[0][1]


def test_nisqa_checkpoint_loader_uses_safe_loader_and_validates_format(monkeypatch):
    calls = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"args": {"model": "NISQA_DIM"}, "model_state_dict": {}}

    monkeypatch.setattr(mos_models, "safe_torch_load", fake_safe_load)

    checkpoint = mos_models._load_nisqa_checkpoint("nisqa.tar")

    assert checkpoint["args"]["model"] == "NISQA_DIM"
    assert calls == [
        (
            "nisqa.tar",
            {
                "map_location": "cpu",
                "purpose": "load bundled NISQA weights",
            },
        )
    ]
    assert "allow_unsafe" not in calls[0][1]

    monkeypatch.setattr(
        mos_models,
        "safe_torch_load",
        lambda *_args, **_kwargs: {"args": {}},
    )
    with pytest.raises(ValueError, match="model_state_dict"):
        mos_models._load_nisqa_checkpoint("invalid.tar")


def test_safe_loader_never_falls_back_to_legacy_pickle_loading(monkeypatch, tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"placeholder")
    calls = []

    def legacy_torch_load(*args, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'weights_only'")
        raise AssertionError("default safe path must not invoke legacy pickle loading")

    monkeypatch.setattr(safe_load.torch, "load", legacy_torch_load)

    with pytest.raises(RuntimeError, match="Cannot safely load"):
        safe_load.safe_torch_load(checkpoint)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_safe_loader_requires_purpose_for_explicit_unsafe_loading(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "trusted.pt"
    checkpoint.write_bytes(b"placeholder")
    calls = []

    def recording_torch_load(*args, **kwargs):
        calls.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(safe_load.torch, "load", recording_torch_load)

    with pytest.raises(ValueError, match="non-empty purpose"):
        safe_load.safe_torch_load(checkpoint, allow_unsafe=True, purpose="   ")
    assert calls == []
    assert safe_load.safe_torch_load(
        checkpoint, allow_unsafe=True, purpose="verified legacy fixture"
    ) == {"model": {}}
    assert calls == [{"map_location": "cpu", "weights_only": False}]


def test_official_checkpoint_verifier_uses_safe_loader(monkeypatch):
    model = nn.Linear(1, 1)
    calls = []

    def fake_safe_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"model": model.state_dict()}

    monkeypatch.setattr(official_ckpts, "safe_torch_load", fake_safe_load)
    official_ckpts._load_spectrum_model(model, Path("official.tar"), "GTCRN")

    assert calls == [
        (
            Path("official.tar"),
            {
                "map_location": "cpu",
                "purpose": "verify official GTCRN checkpoint safely",
            },
        )
    ]
