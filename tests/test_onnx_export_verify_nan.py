"""回归测试：onnx/export.py::verify_onnx 的 fail-closed 判定。

背景：旧逻辑 ``if mse >= mse_threshold: all_pass = False`` 在 MSE 为 NaN 时
恒为 False（NaN 与任何数比较均为 False），验证失败会被静默放过。
本测试保证：输出含 NaN、错误 shape 或 ONNX I/O ABI 额外字段时 verify_onnx 必须返回 False。

实现说明：端到端导出需要 PyPI ``onnx``（本地 ``onnx/`` 子包会
shadow 它），由项目 ``export_onnx()`` 在独立作用域中加载。测试显式使用
PyTorch 的 legacy exporter（``dynamo=False``），因此缺少仅供新 exporter
使用的 ``onnxscript`` 时仍可验证当前项目导出路径。验证仍在**独立子进程**中
执行，主进程零污染。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# 需要真实 onnxruntime：显式 ort marker，默认 `-m unit` 收集不会把它变成“绿但不测”
pytestmark = pytest.mark.ort

_SUBPROCESS_SCRIPT = '''
import atexit
import os
import shutil
import sys
import tempfile

sys.path.insert(0, {repo!r})

from onnx.export import verify_onnx
from onnx.utils import export_onnx

import torch
import torch.nn as nn


class _IdentityModel(nn.Module):
    def forward(self, x):
        return x

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]


class _NanModel(nn.Module):
    def forward(self, x):
        # ``0.0 * x`` 保证输入 x 出现在导出图里（否则全 nan 常量会被
        # 常量折叠，输入被优化掉，ORT 报 Invalid input name）。
        return torch.full_like(x, float("nan")) + 0.0 * x

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]


class _ExtraInputModel(nn.Module):
    def forward(self, x, extra=None):
        return x if extra is None else x + extra

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]


class _ExtraOutputModel(nn.Module):
    def forward(self, x):
        return x, x

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]


class _ShapeMismatchExportModel(nn.Module):
    def forward(self, x):
        return x[:, :1]

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]


class _ShapeMismatchReferenceModel(nn.Module):
    def forward(self, x):
        return x

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]


tmp = tempfile.mkdtemp()
# 子进程内创建：正常退出与断言失败路径都必须回收临时 ONNX，避免 /tmp 残留
atexit.register(shutil.rmtree, tmp, ignore_errors=True)
x = torch.zeros(1, 4)

path_i = os.path.join(tmp, "identity.onnx")
export_onnx(
    _IdentityModel().eval(), (x,), path_i,
    input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
)
assert verify_onnx(path_i, _IdentityModel().eval(), (x,), 1e-4) is True, "clean output must PASS"

path_n = os.path.join(tmp, "nan_model.onnx")
export_onnx(
    _NanModel().eval(), (x,), path_n,
    input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
)
assert verify_onnx(path_n, _NanModel().eval(), (x,), 1e-4) is False, "NaN output must FAIL (fail-closed)"

path_ei = os.path.join(tmp, "extra_input.onnx")
extra = torch.ones_like(x)
export_onnx(
    _ExtraInputModel().eval(), (x, extra), path_ei,
    input_names=["x", "extra"], output_names=["y"], opset_version=17, dynamo=False,
)
assert verify_onnx(path_ei, _ExtraInputModel().eval(), (x,), 1e-4) is False, "extra input must FAIL"

path_eo = os.path.join(tmp, "extra_output.onnx")
export_onnx(
    _ExtraOutputModel().eval(), (x,), path_eo,
    input_names=["x"], output_names=["y", "extra"], opset_version=17, dynamo=False,
)
assert verify_onnx(path_eo, _ExtraOutputModel().eval(), (x,), 1e-4) is False, "extra output must FAIL"

path_sm = os.path.join(tmp, "shape_mismatch.onnx")
export_onnx(
    _ShapeMismatchExportModel().eval(), (x,), path_sm,
    input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
)
assert verify_onnx(
    path_sm, _ShapeMismatchReferenceModel().eval(), (x,), 1e-4
) is False, "broadcastable shape mismatch must FAIL"

print("SUBPROCESS OK")
'''


def test_verify_onnx_fail_closed_for_nan():
    pytest.importorskip("onnxruntime")
    repo = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT.format(repo=repo)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    assert "SUBPROCESS OK" in result.stdout
