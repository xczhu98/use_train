"""Linear AEC frontend for dataset-level preprocessing.

Supports multiple backends via a unified interface. Currently implemented:
  - speex: ctypes binding to system libspeexdsp.so.1 (MDF algorithm)
  - pfdkf: Partitioned-Block Frequency-Domain Kalman Filter

Optional disk cache (``cache_dir``): pre-computed error signals are saved as
``.npy`` files keyed by a hash of (mic, ref), the parsed config and the backend
implementation identity (Speex batch-helper C 源码/构建参数或 PFDKF 实现模块内容）。
后端实现修正后旧残差自动失效。Useful for validation-set repeatability and
cross-experiment reuse.

Usage:
    frontend = LinearAECFrontend(
        cfg={"backend": "speex", "frame_size": 256, "filter_length": 2048,
             "cache_dir": "/tmp/laec_cache"},
        sample_rate=16000,
    )
    err_time = frontend.process(mic_np, ref_np)  # both (T,) float32
"""

from __future__ import annotations

import abc
import hashlib
import hmac
import importlib
import json
import logging
import math
import os
import platform
import struct
import subprocess
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed config schema (linear_aec YAML 子键)
# ---------------------------------------------------------------------------

@dataclass
class LinearAECConfig:
    """``linear_aec`` YAML 子配置的严格 typed schema。

    长度类字段单位为采样点（``frame_size`` / ``filter_length`` / ``N`` / ``M`` /
    ``hop``），``A`` / ``P_initial`` / ``keep_m_gate`` 无量纲。未知键立即报错
    （含键名）。解析由 ``validate_aec_dataset_config`` 与
    ``DynamicMixingDataset.__init__`` 共用，保证两路径判定一致。
    """

    backend: Optional[str] = "speex"  # "speex" | "pfdkf" | "pfdkf_torch" | "none" | null
    frame_size: int = 256             # speex 帧大小（采样点）
    filter_length: int = 2048         # speex 滤波器长度（采样点）
    bulk_delay: int = 0               # speex bulk delay（采样点；当前仅支持 0）
    cache_dir: Optional[str] = None   # 磁盘缓存目录（None=禁用）
    cache_mem_max: int = 128          # 内存 LRU 缓存最大条目数
    gpu_batch: bool = False           # True: PFDKF 由批处理调用方执行
    N: int = 5                        # PFDKF 分区数
    M: int = 1024                     # PFDKF 滤波器长度（采样点）
    hop: int = 512                    # PFDKF 块 hop（采样点）
    A: float = 0.999                  # PFDKF 遗忘因子
    P_initial: float = 1.0            # PFDKF 初始协方差
    keep_m_gate: float = 0.5          # PFDKFRES 门限
    use_res: bool = False             # 是否使用 PFDKFRES 变体
    device: str = "cpu"               # pfdkf_torch 设备（"cpu" | "cuda"）

    @classmethod
    def from_dict(cls, d: dict) -> "LinearAECConfig":
        """从 YAML dict 解析并校验；未知键报错。"""
        if not isinstance(d, dict):
            raise ValueError(f"linear_aec must be a mapping, got {type(d).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(
                f"linear_aec has unknown key(s) {unknown}; "
                f"expected subset of {sorted(known)}"
            )
        backend = d.get("backend", cls.backend)
        if backend is not None:
            if type(backend) is not str:
                raise ValueError(
                    f"linear_aec.backend must be a string or null, got {backend!r}"
                )
            backend = backend.strip().lower()
            if backend == "none":
                backend = None
        if backend not in (None, "speex", "pfdkf", "pfdkf_torch"):
            raise ValueError(
                f"Unknown linear_aec backend {backend!r}. "
                f"Supported: ['speex', 'pfdkf', 'pfdkf_torch', None]"
            )

        values = {
            field.name: d.get(field.name, getattr(cls, field.name))
            for field in fields(cls)
        }
        for name in (
            "frame_size",
            "filter_length",
            "bulk_delay",
            "cache_mem_max",
            "N",
            "M",
            "hop",
        ):
            if type(values[name]) is not int:
                raise ValueError(
                    f"linear_aec.{name} must be an exact int, got {values[name]!r}"
                )
        for name in ("gpu_batch", "use_res"):
            if type(values[name]) is not bool:
                raise ValueError(
                    f"linear_aec.{name} must be a bool, got {values[name]!r}"
                )
        for name in ("A", "P_initial", "keep_m_gate"):
            value = values[name]
            if type(value) is bool or not isinstance(value, (int, float)):
                raise ValueError(
                    f"linear_aec.{name} must be a real number, got {value!r}"
                )
        if values["cache_dir"] is not None and type(values["cache_dir"]) is not str:
            raise ValueError(
                "linear_aec.cache_dir must be a string or null, "
                f"got {values['cache_dir']!r}"
            )
        if type(values["device"]) is not str:
            raise ValueError(
                f"linear_aec.device must be a string, got {values['device']!r}"
            )

        obj = cls(
            backend=backend,
            frame_size=values["frame_size"],
            filter_length=values["filter_length"],
            bulk_delay=values["bulk_delay"],
            cache_dir=values["cache_dir"],
            cache_mem_max=values["cache_mem_max"],
            gpu_batch=values["gpu_batch"],
            N=values["N"],
            M=values["M"],
            hop=values["hop"],
            A=float(values["A"]),
            P_initial=float(values["P_initial"]),
            keep_m_gate=float(values["keep_m_gate"]),
            use_res=values["use_res"],
            device=values["device"].strip().lower(),
        )
        for name in ("frame_size", "filter_length", "N", "M", "hop"):
            value = getattr(obj, name)
            if value < 1:
                raise ValueError(f"linear_aec.{name} must be >= 1, got {value}")
        for name in ("bulk_delay", "cache_mem_max"):
            value = getattr(obj, name)
            if value < 0:
                raise ValueError(f"linear_aec.{name} must be >= 0, got {value}")
        for name in ("A", "P_initial", "keep_m_gate"):
            value = getattr(obj, name)
            if not math.isfinite(value):
                raise ValueError(f"linear_aec.{name} must be finite, got {value}")
        if not 0.0 < obj.A <= 1.0:
            raise ValueError(f"linear_aec.A must be in (0, 1], got {obj.A}")
        if obj.P_initial <= 0:
            raise ValueError(f"linear_aec.P_initial must be > 0, got {obj.P_initial}")
        if not 0.0 <= obj.keep_m_gate <= 1.0:
            raise ValueError(
                f"linear_aec.keep_m_gate must be in [0, 1], got {obj.keep_m_gate}"
            )
        if obj.device not in ("cpu", "cuda"):
            raise ValueError(
                f"linear_aec.device must be 'cpu' or 'cuda', got {obj.device!r}"
            )
        if obj.backend in ("pfdkf", "pfdkf_torch") and obj.M != 2 * obj.hop:
            raise ValueError(
                f"linear_aec.M must equal 2 * hop for PFDKF, got M={obj.M}, hop={obj.hop}"
            )
        if obj.gpu_batch and obj.backend not in ("pfdkf", "pfdkf_torch"):
            raise ValueError(
                "linear_aec.gpu_batch requires backend 'pfdkf' or 'pfdkf_torch'"
            )
        if obj.bulk_delay != 0:
            raise ValueError(
                f"linear_aec.bulk_delay is not implemented; expected 0, got {obj.bulk_delay}"
            )
        return obj

    def to_frontend_dict(self, backend_override: Optional[str] = "keep") -> dict:
        """还原为 ``LinearAECFrontend`` 消费的 dict（键集合与解析前等价）。

        ``gpu_batch`` 是调用方调度标记，frontend 从不读取，故剔除。
        ``backend_override`` 非 ``"keep"`` 时覆盖 backend；None 表示禁用 frontend。
        """
        d = asdict(self)
        d.pop("gpu_batch")
        if backend_override != "keep":
            d["backend"] = backend_override
        return d


# ---------------------------------------------------------------------------
# Optional C batch wrapper (compiled lazily into a user cache)
# ---------------------------------------------------------------------------

_SPEEX_BATCH_C_SOURCE = r'''
#include <speex/speex_echo.h>
#include <speex/speex_types.h>
#include <stdlib.h>

void speex_process_batch(
    SpeexEchoState *st,
    const spx_int16_t *mic,
    const spx_int16_t *ref,
    spx_int16_t *out,
    int frame_size,
    int n_samples
) {
    int full_frames = n_samples / frame_size;
    int i;
    for (i = 0; i < full_frames; i++) {
        int offset = i * frame_size;
        speex_echo_cancellation(st, mic + offset, ref + offset, out + offset);
    }
    int rem = n_samples % frame_size;
    if (rem > 0) {
        int offset = full_frames * frame_size;
        spx_int16_t *mic_pad = (spx_int16_t*)malloc(frame_size * sizeof(spx_int16_t));
        spx_int16_t *ref_pad = (spx_int16_t*)malloc(frame_size * sizeof(spx_int16_t));
        spx_int16_t *out_pad = (spx_int16_t*)malloc(frame_size * sizeof(spx_int16_t));
        int j;
        for (j = 0; j < rem; j++) {
            mic_pad[j] = mic[offset + j];
            ref_pad[j] = ref[offset + j];
        }
        for (; j < frame_size; j++) {
            mic_pad[j] = 0;
            ref_pad[j] = 0;
        }
        speex_echo_cancellation(st, mic_pad, ref_pad, out_pad);
        for (j = 0; j < rem; j++) {
            out[offset + j] = out_pad[j];
        }
        free(mic_pad);
        free(ref_pad);
        free(out_pad);
    }
}
'''
_SPEEX_BATCH_BUILD_FLAGS = ("gcc", "-shared", "-fPIC", "-O2")
_SPEEX_BATCH_LINK_FLAGS = ("-lspeexdsp",)
_c_batch_lib = None


def _speex_batch_cache_dir() -> str:
    """返回只依赖标准库解析的用户级 helper cache 目录。"""
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "use_train", "linear_aec")


def _speex_batch_cache_paths() -> tuple[str, str]:
    """返回与源码、构建参数和 Linux ABI 身份绑定的产物及锁路径。"""
    identity = {
        "source": _SPEEX_BATCH_C_SOURCE,
        "build_flags": _SPEEX_BATCH_BUILD_FLAGS,
        "link_flags": _SPEEX_BATCH_LINK_FLAGS,
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_dir = _speex_batch_cache_dir()
    return (
        os.path.join(cache_dir, f"speex_batch-{digest}.so"),
        os.path.join(cache_dir, f"speex_batch-{digest}.lock"),
    )


def _speex_batch_digest_path(so_path: str) -> str:
    """返回记录构建产物内容摘要的 sidecar 路径（与 .so 同目录同名）。"""
    return f"{so_path}.sha256"


def _file_sha256(path: str) -> str:
    """流式计算文件内容摘要，避免一次性读入整个二进制。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _speex_batch_artifact_matches(so_path: str, digest_path: str) -> bool:
    """加载前校验 cache 内 .so 与构建时写入的内容摘要一致。"""
    try:
        with open(digest_path, "r", encoding="ascii") as digest_file:
            expected = digest_file.read().strip()
        if not expected:
            raise ValueError("empty digest sidecar")
        actual = _file_sha256(so_path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Speex batch helper %s has no verifiable digest (%s); rebuilding",
            so_path,
            exc,
        )
        return False
    if not hmac.compare_digest(expected, actual):
        logger.warning(
            "Speex batch helper digest mismatch for %s; rebuilding from source",
            so_path,
        )
        return False
    return True


def _write_speex_batch_digest(so_path: str, digest_path: str) -> None:
    """原子写入构建产物的内容摘要 sidecar，供下次加载前校验。"""
    digest = _file_sha256(so_path)
    tmp_fd, tmp_digest_path = tempfile.mkstemp(
        dir=os.path.dirname(digest_path), prefix=".speex_batch-", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="ascii") as digest_file:
            digest_file.write(digest)
            digest_file.flush()
            os.fsync(digest_file.fileno())
        os.replace(tmp_digest_path, digest_path)
    except BaseException:
        try:
            os.unlink(tmp_digest_path)
        except OSError:
            pass
        raise


def _load_speex_batch(path: str) -> bool:
    """加载并配置 batch helper；失败时保留纯 ctypes 回退。"""
    global _c_batch_lib
    try:
        import ctypes

        lib = ctypes.CDLL(path)
        lib.speex_process_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.speex_process_batch.restype = None
    except Exception as exc:
        logger.debug("Failed to load C batch wrapper %s: %s", path, exc)
        _c_batch_lib = None
        return False
    _c_batch_lib = lib
    logger.info("Loaded C batch wrapper: %s", path)
    return True


def _compile_speex_batch() -> bool:
    """在带 Linux 进程锁的内容寻址 cache 中原子构建 batch helper。"""
    global _c_batch_lib
    _c_batch_lib = None
    if platform.system() != "Linux":
        logger.debug("C batch wrapper is only built on Linux; using ctypes fallback")
        return False

    tmp_so_path: str | None = None
    try:
        import fcntl

        so_path, lock_path = _speex_batch_cache_paths()
        digest_path = _speex_batch_digest_path(so_path)
        cache_dir = os.path.dirname(so_path)
        source_dir = os.path.realpath(os.path.dirname(__file__))
        resolved_cache_dir = os.path.realpath(cache_dir)
        if os.path.commonpath([resolved_cache_dir, source_dir]) == source_dir:
            raise RuntimeError("Speex batch cache must be outside the source tree")
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(lock_fd, "a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if (
                os.path.isfile(so_path)
                and _speex_batch_artifact_matches(so_path, digest_path)
                and _load_speex_batch(so_path)
            ):
                return True

            with tempfile.TemporaryDirectory() as tmpdir:
                c_path = os.path.join(tmpdir, "speex_batch.c")
                with open(c_path, "w", encoding="utf-8") as source_file:
                    source_file.write(_SPEEX_BATCH_C_SOURCE)
                tmp_fd, tmp_so_path = tempfile.mkstemp(
                    dir=cache_dir, prefix=".speex_batch-", suffix=".tmp"
                )
                os.close(tmp_fd)
                cmd = [
                    *_SPEEX_BATCH_BUILD_FLAGS,
                    c_path,
                    *_SPEEX_BATCH_LINK_FLAGS,
                    "-o",
                    tmp_so_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    logger.debug("C batch wrapper compilation failed: %s", result.stderr)
                    return False
                with open(tmp_so_path, "rb") as compiled_file:
                    os.fsync(compiled_file.fileno())
                os.replace(tmp_so_path, so_path)
                tmp_so_path = None
                _write_speex_batch_digest(so_path, digest_path)
            return _load_speex_batch(so_path)
    except Exception as exc:
        logger.debug("C batch wrapper compilation exception: %s", exc)
        _c_batch_lib = None
        return False
    finally:
        if tmp_so_path is not None:
            try:
                os.unlink(tmp_so_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Base backend interface
# ---------------------------------------------------------------------------

class _AECBackend(abc.ABC):
    """Abstract LAEC backend."""

    def __init__(self, cfg: dict, sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate

    @abc.abstractmethod
    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Process one sample pair and return error waveform (float32)."""
        ...


# ---------------------------------------------------------------------------
# Speex backend (ctypes → libspeexdsp.so.1)
# ---------------------------------------------------------------------------

class _SpeexAECBackend(_AECBackend):
    """Speex MDF via ctypes.

    Optimisations over naive per-sample loop:
      1. Optional C batch wrapper (eliminates Python per-frame loop).
      2. Pre-allocated int16 frame buffers (eliminates per-frame np.zeros).
      3. Pointer-offset for full frames (eliminates per-frame data_as).
      4. Per-worker stderr suppression (eliminates per-sample dup/dup2).

    NOTE: We deliberately keep per-sample init/destroy instead of state-pooling
    + speex_echo_state_reset, because reset() does NOT fully restore the state
    to a fresh-init equivalent (verified by unit test diff ≈ 0.22).  Training
    speed is recovered by the C batch / ctypes fast-path above.
    """

    _LIB_NAME = "libspeexdsp.so.1"
    _SPEEX_ECHO_SET_SAMPLING_RATE = 24
    _SCALE = 32767.0

    _lib_cache = None
    _c_batch_available = None

    def __init__(self, cfg: dict, sample_rate: int) -> None:
        super().__init__(cfg, sample_rate)
        self.frame_size = int(cfg.get("frame_size", 256))
        self.filter_length = int(cfg.get("filter_length", 2048))
        self.bulk_delay = int(cfg.get("bulk_delay", 0))
        if self.bulk_delay != 0:
            raise ValueError(
                f"linear_aec.bulk_delay is not implemented; expected 0, got {self.bulk_delay}"
            )

        # class-level lazy library loading
        if _SpeexAECBackend._lib_cache is None:
            _SpeexAECBackend._lib_cache = self._load_library()
            self._setup_argtypes(_SpeexAECBackend._lib_cache)
        self._lib = _SpeexAECBackend._lib_cache

        # pre-allocated frame buffers (reused across calls)
        self._frame_buf_mic = np.empty(self.frame_size, dtype=np.int16)
        self._frame_buf_ref = np.empty(self.frame_size, dtype=np.int16)
        self._frame_buf_out = np.empty(self.frame_size, dtype=np.int16)

        # lazy C batch wrapper (compile once per process)
        if _SpeexAECBackend._c_batch_available is None:
            _SpeexAECBackend._c_batch_available = _compile_speex_batch()
        self._use_c_batch = bool(_SpeexAECBackend._c_batch_available)

        # NOTE: we no longer redirect stderr to /dev/null.  The old per-sample
        # dup/dup2 approach added syscall overhead and hid real errors.  Speex
        # "acting funny" messages are rare and harmless; if they appear they
        # will go to the default log stream where they are visible for debugging.

        logger.info(
            "[_SpeexAECBackend] frame_size=%d filter_length=%d sr=%d c_batch=%s",
            self.frame_size,
            self.filter_length,
            self.sample_rate,
            self._use_c_batch,
        )

    @classmethod
    def _load_library(cls):
        try:
            import ctypes
            return ctypes.CDLL(cls._LIB_NAME)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot load {cls._LIB_NAME}. "
                "Please install it: apt-get install -y libspeexdsp-dev"
            ) from exc

    @classmethod
    def _setup_argtypes(cls, lib) -> None:
        import ctypes
        lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_echo_state_init.restype = ctypes.c_void_p
        lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_echo_state_destroy.restype = None
        spx_int16_p = ctypes.POINTER(ctypes.c_int16)
        lib.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p, spx_int16_p, spx_int16_p, spx_int16_p,
        ]
        lib.speex_echo_cancellation.restype = None
        lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_echo_ctl.restype = ctypes.c_int

    def _process_c_batch(self, st, mic_i16, ref_i16, out_i16, n):
        """Fast path via C batch wrapper (no per-frame Python loop)."""
        import ctypes

        assert _c_batch_lib is not None
        _c_batch_lib.speex_process_batch(
            st,
            mic_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ref_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            out_i16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            self.frame_size,
            n,
        )

    def _process_ctypes(self, st, mic_i16, ref_i16, out_i16, n):
        """Optimized ctypes path with pointer arithmetic and pre-allocated buffers."""
        import ctypes

        frame_size = self.frame_size
        mic_base = mic_i16.ctypes.data
        ref_base = ref_i16.ctypes.data
        out_base = out_i16.ctypes.data
        itemsize = mic_i16.itemsize  # 2 for int16

        # full frames via pointer offset (zero-copy)
        full_end = (n // frame_size) * frame_size
        for offset in range(0, full_end, frame_size):
            mic_ptr = ctypes.cast(mic_base + offset * itemsize, ctypes.POINTER(ctypes.c_int16))
            ref_ptr = ctypes.cast(ref_base + offset * itemsize, ctypes.POINTER(ctypes.c_int16))
            out_ptr = ctypes.cast(out_base + offset * itemsize, ctypes.POINTER(ctypes.c_int16))
            self._lib.speex_echo_cancellation(st, mic_ptr, ref_ptr, out_ptr)

        # remainder via pre-allocated zero-padded buffer
        rem = n - full_end
        if rem > 0:
            self._frame_buf_mic[:rem] = mic_i16[full_end:]
            self._frame_buf_ref[:rem] = ref_i16[full_end:]
            self._frame_buf_mic[rem:] = 0
            self._frame_buf_ref[rem:] = 0
            self._lib.speex_echo_cancellation(
                st,
                self._frame_buf_mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                self._frame_buf_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                self._frame_buf_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            )
            out_i16[full_end:] = self._frame_buf_out[:rem]

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        import ctypes

        if mic.ndim != 1 or ref.ndim != 1:
            raise ValueError(f"Speex AEC expects 1-D inputs, got {mic.shape}, {ref.shape}")
        if len(mic) != len(ref):
            raise ValueError(f"Length mismatch: mic={len(mic)} ref={len(ref)}")
        if not np.isfinite(mic).all() or not np.isfinite(ref).all():
            raise ValueError("Speex AEC input contains NaN or Inf")

        # Joint peak scaling: if either mic or ref exceeds [-1, 1], scale both
        # down proportionally to avoid hard-clipping distortion.  Speex is a
        # linear system, so scaling both input channels by the same factor
        # preserves the relative level and the output can be scaled back.
        peak = max(np.abs(mic).max(), np.abs(ref).max(), 1e-8)
        scale = 1.0
        if peak > 1.0:
            scale = peak
            mic = mic / scale
            ref = ref / scale

        mic_i16 = (mic * self._SCALE).astype(np.int16)
        ref_i16 = (ref * self._SCALE).astype(np.int16)
        out_i16 = np.empty_like(mic_i16)
        n = len(mic_i16)

        st = self._lib.speex_echo_state_init(self.frame_size, self.filter_length)
        if not st:
            raise RuntimeError("speex_echo_state_init returned NULL")

        try:
            sr_ctype = ctypes.c_int(self.sample_rate)
            ret = self._lib.speex_echo_ctl(
                st, self._SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(sr_ctype)
            )
            if ret != 0:
                raise RuntimeError(f"speex_echo_ctl(SET_SAMPLING_RATE) failed: {ret}")

            if self._use_c_batch:
                try:
                    self._process_c_batch(st, mic_i16, ref_i16, out_i16, n)
                except Exception:
                    self._process_ctypes(st, mic_i16, ref_i16, out_i16, n)
            else:
                self._process_ctypes(st, mic_i16, ref_i16, out_i16, n)
        finally:
            self._lib.speex_echo_state_destroy(st)

        return out_i16.astype(np.float32) * (scale / self._SCALE)

    # -----------------------------------------------------------------------
    # Pickle support for multiprocessing DataLoader workers
    # -----------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # CDLL handle cannot be pickled; it will be reloaded on unpickle
        state.pop("_lib", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if _SpeexAECBackend._lib_cache is None:
            _SpeexAECBackend._lib_cache = self._load_library()
            self._setup_argtypes(_SpeexAECBackend._lib_cache)
        self._lib = _SpeexAECBackend._lib_cache



# ---------------------------------------------------------------------------
# PFDKF backend (Partitioned-Block Frequency-Domain Kalman Filter)
# ---------------------------------------------------------------------------

class _PFDKFBackend(_AECBackend):
    """PFDKF / PFDKFRES via pure NumPy.

    No external library dependencies; fully deterministic and pickle-safe.
    """

    def __init__(self, cfg: dict, sample_rate: int) -> None:
        super().__init__(cfg, sample_rate)
        from tasks.aec.data.pfdkf import PFDKF, PFDKFRES

        self.N = int(cfg.get("N", 5))
        self.M = int(cfg.get("M", 1024))
        self.hop = int(cfg.get("hop", 512))
        self.A = float(cfg.get("A", 0.999))
        self.P_initial = float(cfg.get("P_initial", 1.0))
        self.keep_m_gate = float(cfg.get("keep_m_gate", 0.5))
        self.use_res = bool(cfg.get("use_res", False))

        if self.use_res:
            self._cls = PFDKFRES
        else:
            self._cls = PFDKF

        # PFDKF processes input in hop-sized blocks; expose this as frame_size
        # so callers (e.g., length-mismatch guard) use the correct granularity.
        self.frame_size = self.hop

        logger.info(
            "[_PFDKFBackend] N=%d M=%d hop=%d sr=%d A=%.4f keep_m_gate=%.3f res=%s",
            self.N,
            self.M,
            self.hop,
            self.sample_rate,
            self.A,
            self.keep_m_gate,
            self.use_res,
        )

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if mic.ndim != 1 or ref.ndim != 1:
            raise ValueError(f"PFDKF expects 1-D inputs, got {mic.shape}, {ref.shape}")
        if len(mic) != len(ref):
            raise ValueError(f"Length mismatch: mic={len(mic)} ref={len(ref)}")
        if not np.isfinite(mic).all() or not np.isfinite(ref).all():
            raise ValueError("PFDKF input contains NaN or Inf")

        ft = self._cls(self.N, self.M, self.hop, self.A, self.P_initial, self.keep_m_gate)
        hop = self.hop
        num_block = min(len(mic), len(ref)) // hop
        e = np.zeros(num_block * hop, dtype=np.float32)

        for n in range(num_block):
            ref_n = ref[n * hop : (n + 1) * hop]
            mic_n = mic[n * hop : (n + 1) * hop]
            e_n, _ = ft.filt(ref_n, mic_n)
            ft.update()
            e[n * hop : (n + 1) * hop] = e_n

        # Process trailing partial block by zero-padding to hop length
        trailing = len(mic) - num_block * hop
        if trailing > 0:
            ref_n = np.pad(ref[num_block * hop :], (0, hop - trailing), mode="constant")
            mic_n = np.pad(mic[num_block * hop :], (0, hop - trailing), mode="constant")
            e_n, _ = ft.filt(ref_n, mic_n)
            ft.update()
            e = np.concatenate([e, e_n[:trailing]])

        return e.astype(np.float32)


# ---------------------------------------------------------------------------
# PFDKF Torch backend (PyTorch CPU / optional GPU)
# ---------------------------------------------------------------------------

class _PFDKFTorchBackend(_AECBackend):
    """PFDKF / PFDKFRES via PyTorch.

    Numerically equivalent to the NumPy reference; differences are at the
    float32 machine-epsilon level (~1e-7).  Supports both CPU and GPU
    execution via ``device`` config key.
    """

    def __init__(self, cfg: dict, sample_rate: int) -> None:
        super().__init__(cfg, sample_rate)
        from tasks.aec.data.pfdkf_torch import PFDKFTorch, PFDKFRESTorch

        self.N = int(cfg.get("N", 5))
        self.M = int(cfg.get("M", 1024))
        self.hop = int(cfg.get("hop", 512))
        self.A = float(cfg.get("A", 0.999))
        self.P_initial = float(cfg.get("P_initial", 1.0))
        self.keep_m_gate = float(cfg.get("keep_m_gate", 0.5))
        self.use_res = bool(cfg.get("use_res", False))
        self.device_str = str(cfg.get("device", "cpu")).strip().lower()

        if self.device_str == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "[_PFDKFTorchBackend] device=cuda requested but CUDA unavailable; falling back to cpu"
            )
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)

        _cls = PFDKFRESTorch if self.use_res else PFDKFTorch
        self._model = _cls(
            N=self.N,
            M=self.M,
            hop=self.hop,
            A=self.A,
            P_initial=self.P_initial,
            keep_m_gate=self.keep_m_gate,
        )
        if self.use_res:
            self._model.res = self.use_res
        self._model.to(self.device)
        self._model.eval()

        # PFDKF processes input in hop-sized blocks; expose this as frame_size
        # so callers (e.g., length-mismatch guard) use the correct granularity.
        self.frame_size = self.hop

        logger.info(
            "[_PFDKFTorchBackend] N=%d M=%d hop=%d sr=%d A=%.4f keep_m_gate=%.3f res=%s device=%s",
            self.N,
            self.M,
            self.hop,
            self.sample_rate,
            self.A,
            self.keep_m_gate,
            self.use_res,
            self.device,
        )

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if mic.ndim != 1 or ref.ndim != 1:
            raise ValueError(
                f"PFDKF Torch expects 1-D inputs, got {mic.shape}, {ref.shape}"
            )
        if len(mic) != len(ref):
            raise ValueError(f"Length mismatch: mic={len(mic)} ref={len(ref)}")
        if not np.isfinite(mic).all() or not np.isfinite(ref).all():
            raise ValueError("PFDKF Torch input contains NaN or Inf")

        mic_t = torch.from_numpy(mic).to(self.device)
        ref_t = torch.from_numpy(ref).to(self.device)

        with torch.no_grad():
            err_t = self._model.process(mic_t, ref_t)

        return err_t.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Backend implementation identity (LAEC disk/memory cache invalidation)
# ---------------------------------------------------------------------------

_SOURCE_DIGEST_CACHE: dict[str, str] = {}


def _module_source_digest(module_name: str) -> str:
    """返回后端实现模块源文件的内容摘要，使算法修正后旧缓存自动失效。"""
    cached = _SOURCE_DIGEST_CACHE.get(module_name)
    if cached is not None:
        return cached
    digest = "unavailable"
    try:
        module_file = getattr(importlib.import_module(module_name), "__file__", None)
        if module_file:
            digest = _file_sha256(module_file)
    except Exception as exc:  # pragma: no cover - 仅在源码不可读时触发
        logger.debug("Cannot digest backend source for %s: %s", module_name, exc)
    _SOURCE_DIGEST_CACHE[module_name] = digest
    return digest


def _backend_implementation_identity(backend: Optional[str]) -> dict:
    """描述后端算法实现身份，纳入 LAEC 缓存 key。

    ``backend`` 只描述算法名，不含实现版本；Speex 的 batch helper C 源码、
    构建参数以及 PFDKF 系列的实现模块内容都必须参与身份，否则实现修正后
    旧 ``.npy`` 残差会被静默复用，等于在代码变更后继续产出旧训练数据。
    """
    if backend == "speex":
        return {
            "backend": "speex",
            "helper_source": hashlib.sha256(
                _SPEEX_BATCH_C_SOURCE.encode("utf-8")
            ).hexdigest(),
            "build_flags": list(_SPEEX_BATCH_BUILD_FLAGS),
            "link_flags": list(_SPEEX_BATCH_LINK_FLAGS),
            "module": _module_source_digest(__name__),
        }
    if backend in ("pfdkf", "pfdkf_torch"):
        return {
            "backend": backend,
            "module": _module_source_digest(f"tasks.aec.data.{backend}"),
        }
    return {"backend": backend}


# ---------------------------------------------------------------------------
# Unified frontend
# ---------------------------------------------------------------------------

class LinearAECFrontend:
    """Dataset-level linear AEC frontend.

    Args:
        cfg: dict with at least ``backend`` key. Example:
            ``{"backend": "speex", "frame_size": 256, "filter_length": 2048}``
        sample_rate: audio sample rate (default 16000).
    """

    _CACHE_KEY_VERSION = 1
    _CACHE_KEY_EXCLUDED_FIELDS = frozenset({"cache_dir", "cache_mem_max", "gpu_batch"})

    def __init__(self, cfg: dict | None, sample_rate: int = 16000) -> None:
        if type(sample_rate) is not int or sample_rate < 1:
            raise ValueError(
                f"sample_rate must be an exact int > 0, got {sample_rate!r}"
            )
        self.sample_rate = sample_rate
        if cfg is None:
            self._backend: _AECBackend | None = None
            self._cache_dir = None
            self._cache_mem_max = 0
            self._cache_mem: OrderedDict[str, np.ndarray] = OrderedDict()
            self._cache_key_config = {}
            self._backend_identity = {}
            return

        cfg = LinearAECConfig.from_dict(cfg).to_frontend_dict()

        # Cache settings
        self._cache_dir = cfg["cache_dir"]
        self._cache_mem_max = cfg["cache_mem_max"]  # max entries in memory LRU
        self._cache_mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_key_config = {
            name: value for name, value in cfg.items()
            if name not in self._CACHE_KEY_EXCLUDED_FIELDS
        }
        # 后端实现身份参与缓存 key：只有算法名不足以在实现修正后失效旧残差。
        self._backend_identity = _backend_implementation_identity(cfg["backend"])
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)

        backend = cfg["backend"]
        if backend is None:
            self._backend = None
            logger.info("[LinearAECFrontend] disabled (no backend filter)")
        elif backend == "speex":
            self._backend = _SpeexAECBackend(cfg, sample_rate)
        elif backend == "pfdkf":
            self._backend = _PFDKFBackend(cfg, sample_rate)
        elif backend == "pfdkf_torch":
            self._backend = _PFDKFTorchBackend(cfg, sample_rate)

        # Expose backend frame size (speex: configured frame_size; PFDKF: hop)
        if self._backend is not None:
            self.frame_size = int(getattr(self._backend, "frame_size", cfg["frame_size"]))
        else:
            self.frame_size = cfg["frame_size"]

        if self._cache_dir:
            logger.info("[LinearAECFrontend] disk cache: %s", self._cache_dir)

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    def _hash_inputs(self, mic: np.ndarray, ref: np.ndarray) -> str:
        """Return a versioned, configuration- and implementation-aware cache key."""
        h = hashlib.blake2b(digest_size=16)
        cache_identity = {
            "version": self._CACHE_KEY_VERSION,
            "sample_rate": self.sample_rate,
            "config": self._cache_key_config,
            "backend_identity": self._backend_identity,
        }
        h.update(json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode())
        h.update(struct.pack("<ii", len(mic), len(ref)))
        # Sample a few points for quick entropy (avoids hashing giant arrays twice)
        if len(mic) > 0:
            step = max(1, len(mic) // 8)
            h.update(mic[::step].tobytes())
        if len(ref) > 0:
            step = max(1, len(ref) // 8)
            h.update(ref[::step].tobytes())
        h.update(mic.tobytes())
        h.update(ref.tobytes())
        return h.hexdigest()

    @staticmethod
    def _is_valid_cache_array(arr: np.ndarray, expected_length: int) -> bool:
        return arr.ndim == 1 and len(arr) == expected_length and np.isfinite(arr).all()

    def _cache_load(self, key: str, expected_length: int) -> np.ndarray | None:
        """Try validated memory then disk cache. Returns copy on hit."""
        if key in self._cache_mem:
            arr = self._cache_mem[key]
            if self._is_valid_cache_array(arr, expected_length):
                self._cache_mem.move_to_end(key)
                return arr.copy()
            logger.warning("Ignoring invalid LAEC memory cache entry: %s", key)
            del self._cache_mem[key]
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{key}.npy")
            if os.path.exists(path):
                try:
                    arr = np.load(path).astype(np.float32)
                except Exception as exc:
                    logger.warning("Ignoring unreadable LAEC cache file %s: %s", path, exc)
                    return None
                if not self._is_valid_cache_array(arr, expected_length):
                    logger.warning("Ignoring invalid LAEC cache file: %s", path)
                    return None
                self._cache_mem_put(key, arr)
                return arr.copy()
        return None

    def _cache_mem_put(self, key: str, arr: np.ndarray) -> None:
        """Insert into memory LRU, evict oldest if over limit."""
        if self._cache_mem_max <= 0:
            return
        if key in self._cache_mem:
            self._cache_mem[key] = arr.copy()
            self._cache_mem.move_to_end(key)
            return
        while len(self._cache_mem) >= self._cache_mem_max:
            self._cache_mem.popitem(last=False)
        self._cache_mem[key] = arr.copy()

    def _cache_save(self, key: str, arr: np.ndarray) -> None:
        """Save to memory and atomically publish an optional disk cache file."""
        self._cache_mem_put(key, arr)
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{key}.npy")
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self._cache_dir, prefix=f".{key}.", suffix=".tmp", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                    np.save(tmp, arr)
                    tmp.flush()
                    try:
                        os.fsync(tmp.fileno())
                    except OSError:
                        pass
                os.replace(tmp_path, path)
                tmp_path = None
            except Exception as exc:
                logger.warning("LAEC cache write failed: %s", exc)
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Return linear AEC residual (error) waveform as float32.

        If cache is configured, results are cached by content hash of (mic, ref).

        Args:
            mic: near-end microphone signal, shape (T,), float32.
            ref: far-end reference signal, shape (T,), float32.

        Returns:
            Error signal e = mic - estimated_echo, shape (T,), float32.
            When backend is None, the disabled frontend returns all-zeros.
        """
        mic = np.asarray(mic)
        ref = np.asarray(ref)
        if mic.ndim != 1 or ref.ndim != 1:
            raise ValueError(f"mic and ref must be 1-D, got {mic.shape} and {ref.shape}")
        if mic.shape != ref.shape:
            raise ValueError(f"mic/ref shape mismatch: {mic.shape} != {ref.shape}")
        if not np.issubdtype(mic.dtype, np.number) or not np.issubdtype(ref.dtype, np.number):
            raise TypeError("mic and ref must use numeric dtypes")
        if not np.isfinite(mic).all() or not np.isfinite(ref).all():
            raise ValueError("mic and ref must contain only finite values")
        mic = mic.astype(np.float32, copy=False)
        ref = ref.astype(np.float32, copy=False)
        if self._backend is None:
            return np.zeros_like(mic, dtype=np.float32)

        use_cache = self._cache_dir is not None or self._cache_mem_max > 0
        key: str | None = None
        if use_cache:
            key = self._hash_inputs(mic, ref)
            cached = self._cache_load(key, len(mic))
            if cached is not None:
                return cached

        result = self._backend.process(mic, ref)

        if key is not None:
            self._cache_save(key, result)

        return result

    # -----------------------------------------------------------------------
    # Pickle support for multiprocessing DataLoader workers
    # -----------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # _SpeexAECBackend handles its own CDLL unpickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
