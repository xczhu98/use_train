"""RIR backend registry。"""

from tools.prepare_data_tools.generators.se.rir_backends.base import BaseRIRBackend
from tools.prepare_data_tools.generators.se.rir_backends.fram import FRAMRIRBackend
from tools.prepare_data_tools.generators.se.rir_backends.gpurir import GPURIRBackend
from tools.prepare_data_tools.generators.se.rir_backends.rirgen import RIRGeneratorBackend


BACKENDS: dict[str, type[BaseRIRBackend]] = {
    RIRGeneratorBackend.name: RIRGeneratorBackend,
    FRAMRIRBackend.name: FRAMRIRBackend,
    GPURIRBackend.name: GPURIRBackend,
}


def get_backend(name: str) -> BaseRIRBackend:
    """按注册名构造 backend。"""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS)}")
    return BACKENDS[name]()


def list_available_backends() -> list[str]:
    """返回当前环境依赖可用的 backend。"""
    return [name for name, backend_type in BACKENDS.items() if backend_type().is_available()]


__all__ = [
    "BACKENDS",
    "BaseRIRBackend",
    "FRAMRIRBackend",
    "GPURIRBackend",
    "RIRGeneratorBackend",
    "get_backend",
    "list_available_backends",
]
