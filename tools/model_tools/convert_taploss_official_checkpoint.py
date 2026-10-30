#!/usr/bin/env python3
"""将固定官方 TAPLoss checkpoint 转换为安全的 tensor-only state_dict。"""

from __future__ import annotations

import _codecs
import argparse
import hashlib
import io
import os
import pickletools
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.losses.tap import _TAPAcousticEstimator
from core.utils.safe_load import safe_torch_load

OFFICIAL_COMMIT = "d1fa133d80161bb837f2874d0b18eed89d5bf554"
OFFICIAL_SHA256 = "b6455bba8f282207f8dba8c38ef4c0bc410f76ab8f732d1d6f2de85ee05c4b72"
_EXPECTED_PICKLE_GLOBALS = frozenset(
    {
        "_codecs.encode",
        "collections.OrderedDict",
        "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar",
        "numpy.dtype",
        "numpy.ndarray",
        "torch.FloatStorage",
        "torch._utils._rebuild_tensor_v2",
    }
)
_FORBIDDEN_DYNAMIC_GLOBAL_OPCODES = frozenset(
    {"STACK_GLOBAL", "EXT1", "EXT2", "EXT4", "PERSID"}
)
_EXPECTED_TENSOR_COUNT = 28
_WORKER_FLAG = "--_taploss-conversion-worker"


class ConversionError(RuntimeError):
    """官方 TAPLoss checkpoint 不满足转换安全契约。"""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_symlink_components(
    path: Path,
    *,
    role: str,
    allow_missing_leaf: bool = False,
) -> Path:
    """逐级 lstat 路径，拒绝 leaf 或任意现存 ancestor 为 symlink。"""
    absolute_path = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute_path.anchor)
    parts = absolute_path.parts[1:] if absolute_path.anchor else absolute_path.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return absolute_path
            raise ConversionError(f"{role} path component does not exist: {current}") from None
        if stat.S_ISLNK(component_stat.st_mode):
            raise ConversionError(
                f"{role} path component must not be a symlink: {current}"
            )
    return absolute_path


def _read_bound_source(path: Path) -> bytes:
    """拒绝链接并一次读入 source；后续 SHA 与反序列化共用这些 bytes。"""
    path = _reject_symlink_components(path, role="source checkpoint")
    source_stat = path.lstat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ConversionError(f"source checkpoint must be a regular file: {path}")
    if source_stat.st_size == 0:
        raise ConversionError(f"source checkpoint must not be empty: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ConversionError(f"cannot open source checkpoint safely: {path}: {error}") from error
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ConversionError(f"source checkpoint must be a regular file: {path}")
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            source_stat.st_dev,
            source_stat.st_ino,
        ):
            raise ConversionError(f"source checkpoint changed while opening: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as source_file:
            source_bytes = source_file.read()
    finally:
        os.close(descriptor)
    if not source_bytes:
        raise ConversionError(f"source checkpoint must not be empty: {path}")
    return source_bytes


def _audit_pickle_globals(source_bytes: bytes) -> frozenset[str]:
    """静态提取 archive pickle globals，并要求与固定官方文件精确一致。"""
    try:
        with zipfile.ZipFile(io.BytesIO(source_bytes), "r") as archive:
            pickle_members = [
                name
                for name in archive.namelist()
                if name == "data.pkl" or name.endswith("/data.pkl")
            ]
            if len(pickle_members) != 1:
                raise ConversionError(
                    "source checkpoint must contain exactly one PyTorch data.pkl member"
                )
            pickle_payload = archive.read(pickle_members[0])
    except ConversionError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        raise ConversionError(f"source checkpoint is not a valid PyTorch ZIP archive: {error}") from error

    globals_found: set[str] = set()
    try:
        for opcode, argument, _position in pickletools.genops(pickle_payload):
            if opcode.name in _FORBIDDEN_DYNAMIC_GLOBAL_OPCODES:
                raise ConversionError(
                    f"source pickle contains forbidden dynamic opcode {opcode.name}"
                )
            if opcode.name == "GLOBAL":
                if not isinstance(argument, str) or " " not in argument:
                    raise ConversionError("source pickle contains malformed GLOBAL opcode")
                module_name, object_name = argument.split(" ", 1)
                globals_found.add(f"{module_name}.{object_name}")
    except ConversionError:
        raise
    except Exception as error:
        raise ConversionError(f"cannot statically audit source pickle: {error}") from error

    actual = frozenset(globals_found)
    if actual != _EXPECTED_PICKLE_GLOBALS:
        missing = sorted(_EXPECTED_PICKLE_GLOBALS - actual)
        unexpected = sorted(actual - _EXPECTED_PICKLE_GLOBALS)
        raise ConversionError(
            "source pickle globals do not exactly match the official allowlist; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return actual


def _expected_state_dict() -> dict[str, torch.Tensor]:
    return dict(_TAPAcousticEstimator().state_dict())


def _validate_state_dict(state_dict: object) -> dict[str, torch.Tensor]:
    """校验并复制为 plain direct state_dict，拒绝任意非 CPU Tensor 内容。"""
    if not isinstance(state_dict, Mapping):
        raise ConversionError("model_state_dict must be a mapping")
    if len(state_dict) != _EXPECTED_TENSOR_COUNT:
        raise ConversionError(
            f"model_state_dict must contain exactly {_EXPECTED_TENSOR_COUNT} tensors, "
            f"got {len(state_dict)}"
        )
    if any(not isinstance(key, str) for key in state_dict):
        raise ConversionError("model_state_dict keys must all be strings")

    expected = _expected_state_dict()
    actual_keys = set(state_dict)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        raise ConversionError(
            "model_state_dict keys do not match _TAPAcousticEstimator; "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )

    converted: dict[str, torch.Tensor] = {}
    for key, expected_tensor in expected.items():
        tensor = state_dict[key]
        if type(tensor) is not torch.Tensor:
            raise ConversionError(f"model_state_dict[{key!r}] must be a plain Tensor")
        if tensor.device.type != "cpu":
            raise ConversionError(
                f"model_state_dict[{key!r}] must be on CPU, got {tensor.device}"
            )
        if tensor.shape != expected_tensor.shape:
            raise ConversionError(
                f"model_state_dict[{key!r}] shape mismatch: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_tensor.dtype:
            raise ConversionError(
                f"model_state_dict[{key!r}] dtype mismatch: "
                f"expected {expected_tensor.dtype}, got {tensor.dtype}"
            )
        converted[key] = tensor.detach().clone().contiguous()

    verifier = _TAPAcousticEstimator()
    verifier.load_state_dict(converted, strict=True)
    return converted


def _legacy_numpy_global(name: str, target):
    """构造旧 numpy.core 名称代理，供 PyTorch 2.4 weights-only 匹配 pickle 名。"""
    def compatibility_global(*args):
        return target(*args)

    compatibility_global.__name__ = name
    compatibility_global.__qualname__ = name
    compatibility_global.__module__ = "numpy.core.multiarray"
    return compatibility_global


def _worker_convert(source_bytes: bytes) -> bytes:
    source_sha256 = _sha256_bytes(source_bytes)
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            _WORKER_FLAG,
            source_sha256,
        ],
        input=source_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ConversionError(f"isolated weights-only conversion failed: {detail}")
    if not process.stdout:
        raise ConversionError("isolated weights-only conversion produced no output")
    return process.stdout


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ConversionError("cannot write bound worker checkpoint bytes")
        view = view[written:]


def _load_worker_checkpoint(source_bytes: bytes) -> object:
    """从 worker 私有目录中的非链接普通文件安全反序列化。"""
    with tempfile.TemporaryDirectory(prefix="taploss-converter-worker-") as directory:
        private_directory = Path(directory)
        private_directory.chmod(0o700)
        source_path = private_directory / "sha-bound-source.pt"
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source_path, flags, 0o600)
        try:
            _write_all(descriptor, source_bytes)
            os.fsync(descriptor)
            path_stat = source_path.lstat()
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(path_stat.st_mode) or not stat.S_ISREG(
                opened_stat.st_mode
            ):
                raise ConversionError("worker checkpoint must be a regular file")
            if (path_stat.st_dev, path_stat.st_ino) != (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ):
                raise ConversionError("worker checkpoint path does not bind opened inode")
            if _sha256_bytes(source_path.read_bytes()) != _sha256_bytes(source_bytes):
                raise ConversionError("worker checkpoint bytes changed before loading")
            return safe_torch_load(
                source_path,
                map_location="cpu",
                purpose="convert SHA-pinned official TAPLoss checkpoint",
            )
        finally:
            os.close(descriptor)


def _worker_main(expected_sha256: str) -> int:
    try:
        source_bytes = sys.stdin.buffer.read()
        actual_sha256 = _sha256_bytes(source_bytes)
        if actual_sha256 != expected_sha256:
            raise ConversionError(
                "worker input SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        _audit_pickle_globals(source_bytes)
        reconstruct = _legacy_numpy_global(
            "_reconstruct", np._core.multiarray._reconstruct
        )
        scalar = _legacy_numpy_global("scalar", np._core.multiarray.scalar)
        compatibility_globals = [
            reconstruct,
            scalar,
            np.ndarray,
            np.dtype,
            _codecs.encode,
            type(np.dtype(np.float32)),
            type(np.dtype(np.float64)),
        ]
        # 仅在短命 worker 内注册，避免污染调用转换器的父进程。
        torch.serialization.add_safe_globals(compatibility_globals)
        checkpoint = _load_worker_checkpoint(source_bytes)
        if not isinstance(checkpoint, Mapping):
            raise ConversionError("official checkpoint must be a mapping")
        if "model_state_dict" not in checkpoint:
            raise ConversionError("official checkpoint is missing model_state_dict")
        converted = _validate_state_dict(checkpoint["model_state_dict"])
        output = io.BytesIO()
        torch.save(converted, output)
        sys.stdout.buffer.write(output.getvalue())
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


def _validate_output_path(output_path: Path) -> Path:
    if output_path.exists() or output_path.is_symlink():
        raise ConversionError(f"refusing to overwrite output path: {output_path}")
    output_path = _reject_symlink_components(
        output_path,
        role="output checkpoint",
        allow_missing_leaf=True,
    )
    parent = output_path.parent
    if not parent.is_dir():
        raise ConversionError(f"output parent directory is not a directory: {parent}")
    return output_path


def _publish_and_verify(output_bytes: bytes, output_path: Path) -> tuple[str, int, int]:
    output_path = _validate_output_path(output_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp-", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(output_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())
        try:
            os.link(temporary_path, output_path)
        except FileExistsError:
            raise ConversionError(f"refusing to overwrite output path: {output_path}") from None
        except OSError as error:
            raise ConversionError(f"cannot atomically publish output: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)

    try:
        reloaded = safe_torch_load(output_path, map_location="cpu")
        converted = _validate_state_dict(reloaded)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    output_sha256 = _sha256_bytes(output_bytes)
    tensor_count = len(converted)
    parameter_count = sum(tensor.numel() for tensor in converted.values())
    return output_sha256, tensor_count, parameter_count


def convert_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
) -> tuple[str, str, int, int]:
    """执行只接受固定官方 source SHA256 的 fail-closed 转换。"""
    source = Path(source_path)
    output = Path(output_path)
    _validate_output_path(output)
    source_bytes = _read_bound_source(source)
    source_sha256 = _sha256_bytes(source_bytes)
    if source_sha256 != OFFICIAL_SHA256:
        raise ConversionError(
            "source SHA256 mismatch: "
            f"expected {OFFICIAL_SHA256}, got {source_sha256}"
        )
    _audit_pickle_globals(source_bytes)
    output_bytes = _worker_convert(source_bytes)
    output_sha256, tensor_count, parameter_count = _publish_and_verify(
        output_bytes, output
    )
    return source_sha256, output_sha256, tensor_count, parameter_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the SHA-pinned official TAPLoss checkpoint into a direct "
            "tensor-only state_dict"
        )
    )
    parser.add_argument("--input", required=True, help="official TAP checkpoint path")
    parser.add_argument(
        "--output", required=True, help="new tensor-only state_dict output path"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        source_sha256, output_sha256, tensor_count, parameter_count = convert_checkpoint(
            args.input, args.output
        )
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"source_sha256={source_sha256}")
    print(f"output_sha256={output_sha256}")
    print(f"tensor_count={tensor_count}")
    print(f"parameter_count={parameter_count}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == _WORKER_FLAG:
        raise SystemExit(_worker_main(sys.argv[2]))
    raise SystemExit(main())
