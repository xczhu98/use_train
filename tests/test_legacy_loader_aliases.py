"""裸 ``torch.load`` AST 门禁别名绕过回归测试。"""

from __future__ import annotations

import ast

import pytest

from test_legacy_loader_contracts import _bare_torch_load_calls


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("source", "expected_lines"),
    [
        ("import torch\nloader = torch.load\nloader(path)\n", [3]),
        (
            "import torch as th\n"
            "loader = getattr(th, 'load')\n"
            "loader(path)\n",
            [3],
        ),
        ("import torch as th\ngetattr(th, 'load')(path)\n", [2]),
        ("import torch\ntorch_alias = torch\ntorch_alias.load(path)\n", [3]),
    ],
)
def test_bare_torch_load_gate_rejects_module_level_alias_bypasses(
    source, expected_lines
):
    """模块级 loader/getattr 别名均不能绕过生产代码安全加载门禁。"""
    assert _bare_torch_load_calls(ast.parse(source)) == expected_lines


@pytest.mark.unit
@pytest.mark.contract
def test_bare_torch_load_gate_ignores_function_local_shadowed_loader():
    """函数局部同名变量不应被误识别为模块级 torch.load 别名。"""
    source = "import torch\ndef load(path):\n    return path\nload(path)\n"

    assert _bare_torch_load_calls(ast.parse(source)) == []
