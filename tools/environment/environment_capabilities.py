"""运行环境能力探针。

探针只报告能力，不在 import 阶段强制安装或修改环境。调用方可以根据 report
决定跳过需要外部资源的测试，也可以用 ``require_capabilities`` 对正式门禁 fail-fast。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import socket
from typing import Iterable


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str


def _module(name: str) -> Capability:
    try:
        importlib.import_module(name)
    except Exception as exc:  # optional dependency/cache failures are capability failures
        return Capability(name, False, f"{type(exc).__name__}: {exc}")
    return Capability(name, True, "import ok")


def check_socket() -> Capability:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except OSError as exc:
        return Capability("socket", False, f"{type(exc).__name__}: {exc}")
    return Capability("socket", True, "loopback bind ok")


def capability_report() -> dict[str, Capability]:
    """返回 socket、Numba/Librosa、torchaudio 和 ONNX Runtime 能力。"""
    return {
        "socket": check_socket(),
        "numba": _module("numba"),
        "librosa": _module("librosa"),
        "torchaudio": _module("torchaudio"),
        "onnxruntime": _module("onnxruntime"),
    }


def require_capabilities(names: Iterable[str]) -> dict[str, Capability]:
    """要求指定能力可用；异常包含每个缺失能力的稳定名称和原因。"""
    report = capability_report()
    missing = [report[name] for name in names if name not in report or not report[name].available]
    if missing:
        details = "; ".join(f"{item.name}: {item.detail}" for item in missing)
        raise RuntimeError(f"required environment capabilities unavailable: {details}")
    return {name: report[name] for name in names}
