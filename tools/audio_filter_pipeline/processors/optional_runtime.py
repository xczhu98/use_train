"""Pure-stdlib contract for local optional-runtime model assets."""
from contextlib import contextmanager
from dataclasses import dataclass, fields
import ctypes, errno, fcntl, hashlib, hmac, json, math, os, re, secrets, selectors, signal, stat, struct, subprocess, sys, tempfile, threading, time
try: import resource
except ImportError: resource = None
from urllib.parse import urlsplit
__all__ = [
    "OptionalRuntimeError", "OptionalRuntimeConfigError", "LocalAssetError",
    "AssetDigestMismatchError", "AssetScanLimits", "LocalModelAsset",
    "OwnedModelTreeSnapshot", "open_owned_model_tree_snapshot",
    "local_model_tree_sha256", "verify_local_model_asset", "OptionalWorkerError",
    "WorkerUnavailableError", "WorkerScriptDigestMismatchError", "WorkerTimeoutError",
    "WorkerOutputLimitError", "WorkerExitError", "WorkerCleanupError",
    "WorkerProtocolError", "WorkerRemoteError", "WorkerRemoteAssetError",
    "WorkerRuntimeVersionMismatchError", "WorkerLimits", "OptionalWorkerSpec",
    "OptionalWorkerSession", "run_optional_worker",
]

class OptionalRuntimeError(RuntimeError): pass
class OptionalRuntimeConfigError(OptionalRuntimeError): pass
class LocalAssetError(OptionalRuntimeError): pass
class AssetDigestMismatchError(LocalAssetError): pass
class OptionalWorkerError(OptionalRuntimeError): pass
class WorkerUnavailableError(OptionalWorkerError): pass
class WorkerScriptDigestMismatchError(WorkerUnavailableError): pass
class WorkerTimeoutError(OptionalWorkerError): pass
class WorkerOutputLimitError(OptionalWorkerError): pass
class WorkerExitError(OptionalWorkerError): pass
class WorkerCleanupError(OptionalWorkerError): pass
class WorkerProtocolError(OptionalWorkerError): pass

class WorkerRemoteError(OptionalWorkerError):
    code: str
    remote_message: str
    def __init__(self, code: str, remote_message: str):
        if type(code) is not str or type(remote_message) is not str:
            raise TypeError("remote error fields must be strings")
        self.code, self.remote_message = code, remote_message
        OptionalWorkerError.__init__(self, f"worker remote error {code}: {remote_message}")

class WorkerRemoteAssetError(WorkerRemoteError): pass
class WorkerRuntimeVersionMismatchError(WorkerRemoteError): pass

def _config_error(field): raise OptionalRuntimeConfigError(f"invalid {field}")
def _has_control(value): return any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)

@dataclass(frozen=True)
class AssetScanLimits:
    max_entries: int = 100_000
    max_total_bytes: int = 64 * 1024**3
    max_path_bytes: int = 4096
    read_chunk_bytes: int = 1024 * 1024
    max_depth: int = 128
    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value <= 0:
                _config_error(item.name)
        if self.max_path_bytes > 2**32 - 1: _config_error("max_path_bytes")
        if self.max_total_bytes > 2**64 - 1: _config_error("max_total_bytes")

@dataclass(frozen=True)
class LocalModelAsset:
    local_path: str
    version: str
    sha256: str
    def __post_init__(self):
        for name in ("local_path", "version", "sha256"):
            if type(getattr(self, name)) is not str:
                _config_error(name)
        path = self.local_path
        try:
            parsed = urlsplit(path)
        except ValueError:
            _config_error("local_path")
        canonical = os.path.normpath(os.path.abspath(path)) if path else ""
        if (not path or "\0" in path or _has_control(path) or parsed.scheme
                or parsed.netloc or "://" in path or not os.path.isabs(path)
                or path != canonical):
            _config_error("local_path")
        try:
            version_bytes = self.version.encode("utf-8", "strict")
        except UnicodeEncodeError:
            _config_error("version")
        if (not self.version or self.version != self.version.strip()
                or "\0" in self.version or _has_control(self.version)
                or not 1 <= len(version_bytes) <= 256):
            _config_error("version")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            _config_error("sha256")

@dataclass(frozen=True)
class WorkerLimits:
    timeout_ms: int; terminate_grace_ms: int; kill_wait_ms: int
    rlimit_cpu_seconds: int; rlimit_nofile: int; rlimit_fsize_bytes: int
    rlimit_as_bytes: int; max_request_bytes: int = 1024 * 1024
    max_stdout_bytes: int = 4 * 1024 * 1024; max_stderr_bytes: int = 256 * 1024
    max_worker_script_bytes: int = 1024 * 1024; max_json_depth: int = 32
    io_chunk_bytes: int = 64 * 1024
    def __post_init__(self):
        bounds = ((1, 86_400_000),) * 3 + ((1, 86_400), (8, 1_048_576)) + ((1, 2**63 - 1),) * 2 + ((1, 2**31 - 1),) * 4 + ((1, 128), (1, 1_048_576))
        for item, bound in zip(fields(self), bounds):
            value = getattr(self, item.name)
            if type(value) is not int or not bound[0] <= value <= bound[1]: _config_error(item.name)
@dataclass(frozen=True)
class OptionalWorkerSpec:
    python_executable: str; trusted_code_root: str; worker_script: str
    worker_script_sha256: str; runtime_name: str; runtime_version: str
    inherit_env: tuple[str, ...] = ()
    env_overrides: tuple[tuple[str, str], ...] = ()
    asset_env_paths: tuple[tuple[str, str], ...] = ()
    def __post_init__(self):
        for name in ("python_executable", "trusted_code_root", "worker_script"):
            value = getattr(self, name)
            if type(value) is not str: _config_error(name)
            try: parsed = urlsplit(value)
            except ValueError: _config_error(name)
            if not value or _has_control(value) or parsed.scheme or parsed.netloc or "://" in value or not os.path.isabs(value) or value != os.path.normpath(os.path.abspath(value)): _config_error(name)
        try: descendant = os.path.commonpath((self.trusted_code_root, self.worker_script))
        except ValueError: _config_error("worker_script")
        if descendant != self.trusted_code_root or self.worker_script == self.trusted_code_root: _config_error("worker_script")
        for name in ("worker_script_sha256", "runtime_name", "runtime_version"):
            if type(getattr(self, name)) is not str: _config_error(name)
        if re.fullmatch(r"[0-9a-f]{64}", self.worker_script_sha256) is None: _config_error("worker_script_sha256")
        try: name_bytes = self.runtime_name.encode()
        except UnicodeEncodeError: _config_error("runtime_name")
        try: version_bytes = self.runtime_version.encode()
        except UnicodeEncodeError: _config_error("runtime_version")
        if not 1 <= len(name_bytes) <= 128 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.runtime_name) is None: _config_error("runtime_name")
        if not 1 <= len(version_bytes) <= 256 or self.runtime_version != self.runtime_version.strip() or _has_control(self.runtime_version): _config_error("runtime_version")
        allowed = {"CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "MKL_NUM_THREADS", "NVIDIA_VISIBLE_DEVICES", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF"}
        if type(self.inherit_env) is not tuple or any(type(x) is not str or x not in allowed for x in self.inherit_env) or len(set(self.inherit_env)) != len(self.inherit_env) or self.inherit_env != tuple(sorted(self.inherit_env)): _config_error("inherit_env")
        overrides = self.env_overrides
        if (type(overrides) is not tuple or any(type(item) is not tuple or len(item) != 2 for item in overrides)
                or any(type(name) is not str or name not in allowed or type(value) is not str
                       or not value or _has_control(value) for name, value in overrides)
                or len({name for name, _ in overrides}) != len(overrides)
                or overrides != tuple(sorted(overrides))): _config_error("env_overrides")
        bindings = self.asset_env_paths
        if (type(bindings) is not tuple or any(type(item) is not tuple or len(item) != 2 for item in bindings)
                or any(type(name) is not str or type(relative) is not str
                       or re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name) is None
                       or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", relative) is None
                       for name, relative in bindings)
                or len({name for name, _ in bindings}) != len(bindings)
                or bindings != tuple(sorted(bindings))): _config_error("asset_env_paths")
def _canonical_json_bytes(value: object, *, max_depth: int, label: str) -> bytes:
    if type(max_depth) is not int or not 1 <= max_depth <= 128: raise OptionalRuntimeConfigError(f"{label} depth invalid")
    if type(label) is not str or not label: _config_error("label")
    active, parts, stack = set(), [], [("value", value, 0)]
    try:
        while stack:
            mode, item, depth = stack.pop()
            if mode == "raw": parts.append(item); continue
            if mode == "leave": active.remove(item); continue
            kind = type(item)
            if kind in (dict, list):
                if depth + 1 > max_depth: raise OptionalRuntimeConfigError(f"{label} depth exceeded")
                marker = id(item)
                if marker in active: raise OptionalRuntimeConfigError(f"{label} cycle")
                active.add(marker); parts.append("{" if kind is dict else "["); actions = []
                if kind is dict:
                    children = list(item.items())
                    for key, _ in children:
                        if type(key) is not str: raise OptionalRuntimeConfigError(f"{label} key invalid")
                        try: key.encode()
                        except UnicodeEncodeError: raise OptionalRuntimeConfigError(f"{label} key UTF-8 invalid")
                    children.sort(key=lambda pair: pair[0])
                else: children = list(enumerate(item))
                for index, (key, child) in enumerate(children):
                    if index: actions.append(("raw", ",", 0))
                    if kind is dict: actions.extend((("raw", json.dumps(key, ensure_ascii=False), 0), ("raw", ":", 0)))
                    actions.append(("value", child, depth + 1))
                stack.extend([("raw", "}" if kind is dict else "]", 0), ("leave", marker, 0), *reversed(actions)])
            else:
                if kind is int and not -2**63 <= item <= 2**63 - 1: raise OptionalRuntimeConfigError(f"{label} int64 invalid")
                elif kind is float and not math.isfinite(item): raise OptionalRuntimeConfigError(f"{label} finite float required")
                elif kind is str:
                    try: item.encode()
                    except UnicodeEncodeError: raise OptionalRuntimeConfigError(f"{label} UTF-8 invalid")
                elif kind not in (type(None), bool, int, float): raise OptionalRuntimeConfigError(f"{label} canonical JSON type invalid")
                parts.append(json.dumps(item, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        return "".join(parts).encode("utf-8", "strict")
    except OptionalRuntimeConfigError: raise
    except (RecursionError, RuntimeError, ValueError, TypeError, UnicodeError): raise OptionalRuntimeConfigError(f"{label} canonical JSON encoding invalid") from None
def _encode_worker_message(value: object, *, max_bytes: int, max_depth: int, label: str) -> bytes:
    if type(max_bytes) is not int or max_bytes <= 0: _config_error("max_bytes")
    result = _canonical_json_bytes(value, max_depth=max_depth, label=label) + b"\n"
    if len(result) > max_bytes: raise OptionalRuntimeConfigError(f"{label} size exceeds max bytes")
    return result
def _decode_worker_message(raw: bytes, *, max_bytes: int, max_depth: int, label: str) -> dict:
    if type(raw) is not bytes: _config_error("raw")
    if type(max_bytes) is not int or max_bytes <= 0: _config_error("max_bytes")
    if type(max_depth) is not int or not 1 <= max_depth <= 128: _config_error("max_depth")
    if type(label) is not str or not label: _config_error("label")
    def fail(reason): raise WorkerProtocolError(f"{label} {reason}")
    if not raw: fail("empty")
    if len(raw) > max_bytes: fail("size exceeds max bytes")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"): fail("framing")
    payload = raw[:-1]
    if not payload or b"\r" in payload or b"\n" in payload: fail("extra text")
    if payload.startswith(b"\xef\xbb\xbf"): fail("BOM")
    try: text = payload.decode("utf-8", "strict")
    except UnicodeDecodeError: fail("UTF-8")
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result: fail("duplicate key")
            result[key] = item
        return result
    def integer(token):
        value = int(token)
        if not -2**63 <= value <= 2**63 - 1: fail("int64")
        return value
    def floating(token):
        value = float(token)
        if not math.isfinite(value): fail("finite")
        return value
    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_int=integer, parse_float=floating, parse_constant=lambda _: fail("finite"))
        if type(result) is not dict: fail("root object required")
        canonical = _canonical_json_bytes(result, max_depth=max_depth, label=label)
    except WorkerProtocolError: raise
    except OptionalRuntimeConfigError as exc: fail(str(exc))
    except (ValueError, TypeError, RecursionError, UnicodeError): fail("invalid JSON")
    if canonical != payload: fail("non-canonical encoding")
    return result
def _worker_request_id(body: dict, *, max_depth: int) -> str:
    if type(body) is not dict or "request_id" in body: raise OptionalRuntimeConfigError("invalid worker request body")
    canonical = _canonical_json_bytes(body, max_depth=max_depth, label="worker request body")
    return hashlib.sha256(b"audio_filter_pipeline.optional_worker.request\0v1\0" + canonical).hexdigest()
def _build_worker_request(spec: OptionalWorkerSpec, asset: LocalModelAsset, operation: str,
                          payload: dict, *, nonce: str, limits: WorkerLimits) -> tuple[dict, bytes]:
    if not isinstance(spec, OptionalWorkerSpec): _config_error("spec")
    if not isinstance(asset, LocalModelAsset): _config_error("asset")
    if not isinstance(limits, WorkerLimits): _config_error("limits")
    if type(operation) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", operation) is None: _config_error("operation")
    if type(payload) is not dict: _config_error("payload")
    if type(nonce) is not str or re.fullmatch(r"[0-9a-f]{64}", nonce) is None: _config_error("nonce")
    encoded_payload = _canonical_json_bytes(payload, max_depth=limits.max_json_depth, label="worker payload")
    body = {"asset": {"local_path": asset.local_path, "sha256": asset.sha256, "version": asset.version}, "nonce": nonce, "operation": operation, "payload": json.loads(encoded_payload), "protocol": "audio_filter_pipeline.optional_worker.v1", "runtime": {"name": spec.runtime_name, "version": spec.runtime_version}, "worker_script_sha256": spec.worker_script_sha256}
    request = dict(body); request["request_id"] = _worker_request_id(body, max_depth=limits.max_json_depth)
    encoded = _encode_worker_message(request, max_bytes=limits.max_request_bytes, max_depth=limits.max_json_depth, label="worker request")
    return request, encoded
def _parse_worker_response(raw: bytes, *, request: dict, spec: OptionalWorkerSpec,
                           asset: LocalModelAsset, limits: WorkerLimits) -> dict:
    if type(raw) is not bytes: _config_error("raw")
    if type(request) is not dict: _config_error("request")
    if not isinstance(spec, OptionalWorkerSpec): _config_error("spec")
    if not isinstance(asset, LocalModelAsset): _config_error("asset")
    if not isinstance(limits, WorkerLimits): _config_error("limits")
    try:
        snapshot = json.loads(_canonical_json_bytes(request, max_depth=limits.max_json_depth, label="worker request"))
        if set(snapshot) != {"asset", "nonce", "operation", "payload", "protocol", "request_id", "runtime", "worker_script_sha256"} or type(snapshot.get("asset")) is not dict or set(snapshot["asset"]) != {"local_path", "sha256", "version"} or type(snapshot.get("runtime")) is not dict or set(snapshot["runtime"]) != {"name", "version"}: raise ValueError
        expected, _ = _build_worker_request(spec, asset, snapshot["operation"], snapshot["payload"], nonce=snapshot["nonce"], limits=limits)
        if snapshot != expected: raise ValueError
    except (OptionalRuntimeConfigError, KeyError, TypeError, ValueError, RuntimeError): raise OptionalRuntimeConfigError("invalid worker request") from None
    response = _decode_worker_message(raw, max_bytes=limits.max_stdout_bytes, max_depth=limits.max_json_depth, label="worker response")
    def protocol(reason): raise WorkerProtocolError(f"worker response {reason}")
    status = response.get("status"); runtime_value = response.get("runtime")
    common = {"asset", "nonce", "protocol", "request_id", "runtime", "status", "worker_script_sha256"}
    bindings = (("protocol", snapshot["protocol"]), ("request_id", snapshot["request_id"]), ("nonce", snapshot["nonce"]), ("asset", snapshot["asset"]), ("worker_script_sha256", snapshot["worker_script_sha256"]))
    if status not in ("ok", "error") or set(response) != common | ({"result"} if status == "ok" else {"error"}) or any(response.get(key) != value for key, value in bindings) or type(response.get("asset")) is not dict or set(response["asset"]) != {"local_path", "sha256", "version"} or type(runtime_value) is not dict or set(runtime_value) != {"name", "version"} or runtime_value["name"] != snapshot["runtime"]["name"]: protocol("schema or binding mismatch")
    observed = runtime_value["version"]
    if status == "ok":
        if observed != snapshot["runtime"]["version"] or type(response["result"]) is not dict: protocol("success schema or runtime mismatch")
        return response["result"]
    error = response["error"]
    if type(error) is not dict or set(error) != {"code", "message"}: protocol("error schema mismatch")
    code, message = error.get("code"), error.get("message")
    try: message_bytes = message.encode("utf-8", "strict")
    except (AttributeError, UnicodeEncodeError): protocol("error message invalid")
    allowed = {"asset_missing", "asset_digest_mismatch", "runtime_version_mismatch", "invalid_request", "unsupported_operation", "runtime_error"}
    if type(code) is not str or code not in allowed or type(message) is not str or not 1 <= len(message_bytes) <= 4096 or _has_control(message) or message != message.strip(): protocol("remote error invalid")
    if code == "runtime_version_mismatch":
        try: observed_bytes = observed.encode("utf-8", "strict")
        except (AttributeError, UnicodeEncodeError): protocol("runtime version invalid")
        if type(observed) is not str or observed == snapshot["runtime"]["version"] or observed != observed.strip() or _has_control(observed) or not 1 <= len(observed_bytes) <= 256: protocol("runtime version mismatch invalid")
        raise WorkerRuntimeVersionMismatchError(code, f"expected={snapshot['runtime']['version']} observed={observed}: {message}")
    if observed != snapshot["runtime"]["version"]: protocol("runtime version binding mismatch")
    if code in {"asset_missing", "asset_digest_mismatch"}: raise WorkerRemoteAssetError(code, message)
    raise WorkerRemoteError(code, message)

def _snapshot(value):
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)
def _binding(value): return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


@dataclass(frozen=True)
class _OwnedModelTreeManifestEntry:
    """Owned snapshot 中单个路径的不可变 metadata 身份。"""

    path: bytes
    file_type: int
    dev: int
    ino: int
    size: int
    mode: int
    uid: int
    gid: int
    mtime_ns: int
    ctime_ns: int
    children: tuple[bytes, ...]


def _snapshot_manifest_entry(path, value, children=()):
    return _OwnedModelTreeManifestEntry(
        path=path,
        file_type=stat.S_IFMT(value.st_mode),
        dev=value.st_dev,
        ino=value.st_ino,
        size=value.st_size,
        mode=stat.S_IMODE(value.st_mode),
        uid=value.st_uid,
        gid=value.st_gid,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        children=tuple(children),
    )


def _assert_snapshot_manifest_entry(expected, path, value, children=()):
    actual = _snapshot_manifest_entry(path, value, children)
    if actual != expected:
        raise LocalAssetError(
            "owned snapshot manifest mismatch for "
            + (os.fsdecode(path) if path else ".")
        )


def _verify_snapshot_manifest(root_fd, manifest, limits):
    """仅读取目录项和 stat metadata，复核 immutable snapshot manifest。"""
    expected_by_path = {entry.path: entry for entry in manifest}
    if len(expected_by_path) != len(manifest) or b"" not in expected_by_path:
        raise LocalAssetError("owned snapshot manifest is invalid")
    state = {"entries": 0, "bytes": 0, "visited": set()}

    def visit(fd, path, depth):
        before = os.fstat(fd)
        if not stat.S_ISDIR(before.st_mode):
            raise LocalAssetError("snapshot manifest directory changed type")
        identity = before.st_dev, before.st_ino
        if identity in state["visited"]:
            raise LocalAssetError("snapshot manifest repeated directory identity")
        state["visited"].add(identity)
        names = _names(fd)
        expected = expected_by_path.get(path)
        if expected is None:
            raise LocalAssetError("owned snapshot manifest has an unexpected directory")
        _assert_snapshot_manifest_entry(
            expected, path, before, (name_bytes for name_bytes, _ in names),
        )

        for name_bytes, name in names:
            child_path = name_bytes if not path else path + b"/" + name_bytes
            child_depth = depth + 1
            state["entries"] += 1
            if state["entries"] > limits.max_entries:
                raise LocalAssetError("snapshot manifest entry limit exceeded")
            if len(child_path) > limits.max_path_bytes or child_depth > limits.max_depth:
                raise LocalAssetError("snapshot manifest path or depth limit exceeded")
            child_expected = expected_by_path.get(child_path)
            if child_expected is None:
                raise LocalAssetError("owned snapshot manifest has an unexpected entry")
            pathname_before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(pathname_before.st_mode):
                raise LocalAssetError("snapshot manifest symlink rejected")
            is_dir = stat.S_ISDIR(pathname_before.st_mode)
            if not is_dir and not stat.S_ISREG(pathname_before.st_mode):
                raise LocalAssetError("snapshot manifest special file rejected")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            flags |= os.O_DIRECTORY if is_dir else os.O_NONBLOCK
            child_fd = os.open(name, flags, dir_fd=fd)
            try:
                opened = os.fstat(child_fd)
                if _binding(pathname_before) != _binding(opened):
                    raise LocalAssetError("snapshot manifest path binding changed")
                if is_dir:
                    visit(child_fd, child_path, child_depth)
                else:
                    state["bytes"] += opened.st_size
                    if state["bytes"] > limits.max_total_bytes:
                        raise LocalAssetError("snapshot manifest total bytes limit exceeded")
                    _assert_snapshot_manifest_entry(
                        child_expected, child_path, opened,
                    )
                pathname_after = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if is_dir:
                    after_names = _names(child_fd)
                    _assert_snapshot_manifest_entry(
                        child_expected,
                        child_path,
                        os.fstat(child_fd),
                        (child_name for child_name, _ in after_names),
                    )
                else:
                    _assert_snapshot_manifest_entry(
                        child_expected, child_path, os.fstat(child_fd),
                    )
                _assert_snapshot_manifest_entry(
                    child_expected,
                    child_path,
                    pathname_after,
                    child_expected.children if is_dir else (),
                )
            finally:
                error = _close(child_fd, None)
                if error is not None:
                    raise error

        after_names = _names(fd)
        _assert_snapshot_manifest_entry(
            expected,
            path,
            os.fstat(fd),
            (name_bytes for name_bytes, _ in after_names),
        )

    visit(root_fd, b"", 0)
    if state["entries"] + 1 != len(manifest):
        raise LocalAssetError("owned snapshot manifest entry set changed")


def _build_snapshot_manifest(root_fd, limits):
    """固定 snapshot 的完整 metadata/children 清单，不读取普通文件内容。"""
    entries = []
    state = {"entries": 0, "bytes": 0, "visited": set()}

    def visit(fd, path, depth):
        before = os.fstat(fd)
        if not stat.S_ISDIR(before.st_mode):
            raise LocalAssetError("snapshot manifest directory changed type")
        identity = before.st_dev, before.st_ino
        if identity in state["visited"]:
            raise LocalAssetError("snapshot manifest repeated directory identity")
        state["visited"].add(identity)
        names = _names(fd)
        directory_index = len(entries)
        entries.append(None)
        for name_bytes, name in names:
            child_path = name_bytes if not path else path + b"/" + name_bytes
            child_depth = depth + 1
            state["entries"] += 1
            if state["entries"] > limits.max_entries:
                raise LocalAssetError("snapshot manifest entry limit exceeded")
            if len(child_path) > limits.max_path_bytes or child_depth > limits.max_depth:
                raise LocalAssetError("snapshot manifest path or depth limit exceeded")
            pathname_before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(pathname_before.st_mode):
                raise LocalAssetError("snapshot manifest symlink rejected")
            is_dir = stat.S_ISDIR(pathname_before.st_mode)
            if not is_dir and not stat.S_ISREG(pathname_before.st_mode):
                raise LocalAssetError("snapshot manifest special file rejected")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            flags |= os.O_DIRECTORY if is_dir else os.O_NONBLOCK
            child_fd = os.open(name, flags, dir_fd=fd)
            try:
                opened = os.fstat(child_fd)
                if _binding(pathname_before) != _binding(opened):
                    raise LocalAssetError("snapshot manifest path binding changed")
                if is_dir:
                    visit(child_fd, child_path, child_depth)
                else:
                    state["bytes"] += opened.st_size
                    if state["bytes"] > limits.max_total_bytes:
                        raise LocalAssetError("snapshot manifest total bytes limit exceeded")
                    entries.append(_snapshot_manifest_entry(child_path, opened))
                pathname_after = os.stat(name, dir_fd=fd, follow_symlinks=False)
                opened_after = os.fstat(child_fd)
                child_names = _names(child_fd) if is_dir else ()
                children = (
                    tuple(name_bytes for name_bytes, _ in child_names)
                    if is_dir else ()
                )
                if _snapshot_manifest_entry(
                    child_path, pathname_after, children,
                ) != _snapshot_manifest_entry(
                    child_path, opened_after, children,
                ):
                    raise LocalAssetError("snapshot manifest path binding changed")
            finally:
                error = _close(child_fd, None)
                if error is not None:
                    raise error
        after_names = _names(fd)
        after = os.fstat(fd)
        if (
            _snapshot_manifest_entry(
                path, before, (name_bytes for name_bytes, _ in names),
            )
            != _snapshot_manifest_entry(
                path, after, (name_bytes for name_bytes, _ in after_names),
            )
        ):
            raise LocalAssetError("snapshot changed while building manifest")
        entries[directory_index] = _snapshot_manifest_entry(
            path, after, (name_bytes for name_bytes, _ in names),
        )

    visit(root_fd, b"", 0)
    return tuple(entries)


def _close(fd, primary):
    try:
        os.close(fd)
    except OSError as exc:
        if primary is None:
            primary = LocalAssetError("failed to close asset descriptor")
            primary.__cause__ = exc
    return primary

def _names(fd):
    result = []
    for name in os.listdir(fd):
        try: encoded = name.encode("utf-8", "strict")
        except UnicodeEncodeError as exc: raise LocalAssetError("name is not UTF-8") from exc
        result.append((encoded, name))
    result.sort()
    if any(result[i][0] == result[i - 1][0] for i in range(1, len(result))):
        raise LocalAssetError("duplicate encoded name")
    return result

def _write_all(fd, chunk):
    offset = 0
    while offset < len(chunk):
        written = os.write(fd, chunk[offset:])
        if written <= 0: raise LocalAssetError("snapshot write made no progress")
        offset += written


def _read_file(fd, before, limits, state, snapshot_fd=None):
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0: raise LocalAssetError("not regular file")
    state["bytes"] += before.st_size
    if state["bytes"] > limits.max_total_bytes: raise LocalAssetError("total bytes limit exceeded")
    digest, count = hashlib.sha256(), 0
    while True:
        chunk = os.read(fd, limits.read_chunk_bytes)
        if not chunk: break
        count += len(chunk)
        if count > before.st_size or count > limits.max_total_bytes:
            raise LocalAssetError("file changed or exceeded byte limit")
        digest.update(chunk)
        if snapshot_fd is not None: _write_all(snapshot_fd, chunk)
    if count != before.st_size or _snapshot(os.fstat(fd)) != _snapshot(before):
        raise LocalAssetError("file changed during scan")
    if snapshot_fd is not None:
        copied = os.fstat(snapshot_fd)
        if not stat.S_ISREG(copied.st_mode) or copied.st_size != count:
            raise LocalAssetError("snapshot file changed during copy")
        os.fchmod(snapshot_fd, 0o400)
    return before.st_size, digest.digest()


def _frame(fd, path, depth, parent, limits, state, snapshot_fd=None):
    before = os.fstat(fd)
    if not stat.S_ISDIR(before.st_mode): raise LocalAssetError("directory changed type")
    if state.get("require_snapshot_modes") and stat.S_IMODE(before.st_mode) != 0o500:
        raise LocalAssetError("snapshot directory permissions changed")
    identity = before.st_dev, before.st_ino
    if identity in state["visited"]: raise LocalAssetError("repeated directory identity")
    state["visited"].add(identity)
    snapshot_bindings = state.get("snapshot_bindings")
    if snapshot_bindings is not None and snapshot_fd is not None:
        snapshot_bindings[path] = _binding(os.fstat(snapshot_fd))
    return {"fd": fd, "path": path, "depth": depth, "parent": parent,
            "before": before, "names": _names(fd), "index": 0,
            "snapshot_fd": snapshot_fd}


def _scan_tree(root_fd, limits, state, entries, snapshot_root_fd=None):
    stack = [_frame(root_fd, b"", 0, None, limits, state, snapshot_root_fd)]
    root_before = stack[0]["before"]
    try:
        while stack:
            frame = stack[-1]
            if frame["index"] == len(frame["names"]):
                if (_names(frame["fd"]) != frame["names"] or
                        _snapshot(os.fstat(frame["fd"])) != _snapshot(frame["before"])):
                    raise LocalAssetError("directory changed during scan")
                if frame["snapshot_fd"] is not None:
                    os.fchmod(frame["snapshot_fd"], 0o500)
                parent = frame["parent"]
                if parent is not None:
                    after = os.stat(parent[1], dir_fd=parent[0], follow_symlinks=False)
                    if _snapshot(after) != _snapshot(parent[2]):
                        raise LocalAssetError("path binding changed")
                    stack.pop()
                    error = _close(frame["fd"], None)
                    if frame["snapshot_fd"] is not None:
                        error = _close(frame["snapshot_fd"], error)
                    if error: raise error
                else: stack.pop()
                continue
            name_bytes, name = frame["names"][frame["index"]]
            frame["index"] += 1
            path = name_bytes if not frame["path"] else frame["path"] + b"/" + name_bytes
            depth = frame["depth"] + 1
            state["entries"] += 1
            if state["entries"] > limits.max_entries: raise LocalAssetError("entry limit exceeded")
            if len(path) > limits.max_path_bytes or depth > limits.max_depth:
                raise LocalAssetError("path or depth limit exceeded")
            before = os.stat(name, dir_fd=frame["fd"], follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode): raise LocalAssetError("symlink rejected")
            is_dir = stat.S_ISDIR(before.st_mode)
            if not is_dir and not stat.S_ISREG(before.st_mode):
                raise LocalAssetError("special file rejected")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            flags |= os.O_DIRECTORY if is_dir else os.O_NONBLOCK
            child_fd = os.open(name, flags, dir_fd=frame["fd"])
            snapshot_fd = None
            try:
                opened = os.fstat(child_fd)
                if _binding(before) != _binding(opened): raise LocalAssetError("path binding changed")
                if (state.get("require_snapshot_modes") and not is_dir and
                        stat.S_IMODE(opened.st_mode) != 0o400):
                    raise LocalAssetError("snapshot file permissions changed")
                prefix = (b"D" if is_dir else b"F") + struct.pack(">I", len(path)) + path
                if frame["snapshot_fd"] is not None:
                    snapshot_bindings = state.get("snapshot_bindings")
                    if is_dir:
                        os.mkdir(name, 0o700, dir_fd=frame["snapshot_fd"])
                        created = os.stat(
                            name, dir_fd=frame["snapshot_fd"], follow_symlinks=False)
                        if snapshot_bindings is not None:
                            snapshot_bindings[path] = _binding(created)
                        snapshot_fd = os.open(
                            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=frame["snapshot_fd"])
                    else:
                        snapshot_fd = os.open(
                            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                            0o600, dir_fd=frame["snapshot_fd"])
                        if snapshot_bindings is not None:
                            snapshot_bindings[path] = _binding(os.fstat(snapshot_fd))
                if is_dir:
                    entries.append((path, prefix))
                    child = _frame(
                        child_fd, path, depth, (frame["fd"], name, before), limits,
                        state, snapshot_fd)
                else:
                    size, digest = _read_file(child_fd, opened, limits, state, snapshot_fd)
                    snapshot_bindings = state.get("snapshot_bindings")
                    if snapshot_bindings is not None and snapshot_fd is not None:
                        snapshot_bindings[path] = _binding(os.fstat(snapshot_fd))
                    entries.append((path, prefix + struct.pack(">Q", size) + digest))
                    after = os.stat(name, dir_fd=frame["fd"], follow_symlinks=False)
                    if _snapshot(after) != _snapshot(before): raise LocalAssetError("path binding changed")
            except BaseException as exc:
                error = _close(child_fd, exc)
                if snapshot_fd is not None: error = _close(snapshot_fd, error)
                raise error
            if is_dir: stack.append(child)
            else:
                error = _close(child_fd, None)
                if snapshot_fd is not None: error = _close(snapshot_fd, error)
                if error: raise error
    except BaseException as exc:
        error = exc
        for frame in reversed(stack):
            if frame["fd"] != root_fd: error = _close(frame["fd"], error)
            if (frame["snapshot_fd"] is not None and
                    frame["snapshot_fd"] != snapshot_root_fd):
                error = _close(frame["snapshot_fd"], error)
        raise error
    return root_before

def _open_root(local_path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    current = os.open("/", flags)
    try:
        for component in local_path.lstrip("/").split("/"):
            if not component: continue
            next_fd = os.open(component, flags | os.O_NOFOLLOW, dir_fd=current)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise LocalAssetError("root component is not a directory")
            except BaseException as exc:
                raise _close(next_fd, exc)
            old_fd, current = current, next_fd
            error = _close(old_fd, None)
            if error: raise error
        if os.path.realpath(f"/proc/self/fd/{current}") != local_path:
            raise LocalAssetError("root descriptor/path mismatch")
        return current
    except BaseException as exc:
        raise _close(current, exc)

def _revalidate_root(local_path, root_fd, before):
    reopened = _open_root(local_path)
    try:
        if (_snapshot(os.fstat(root_fd)) != _snapshot(before) or
                _snapshot(os.fstat(reopened)) != _snapshot(before) or
                os.path.realpath(f"/proc/self/fd/{root_fd}") != local_path):
            raise LocalAssetError("configured root binding changed")
    except BaseException as exc:
        raise _close(reopened, exc)
    error = _close(reopened, None)
    if error: raise error

def _digest_entries(entries):
    outer = hashlib.sha256(b"audio_filter_pipeline.local_model_tree\0v1\0")
    for _, encoding in sorted(entries): outer.update(encoding)
    return outer.hexdigest()


def _tree_digest(local_path, limits):
    root_fd = _open_root(local_path)
    try:
        entries, state = [], {"entries": 0, "bytes": 0, "visited": set()}
        root_before = _scan_tree(root_fd, limits, state, entries)
        result = _digest_entries(entries)
        _revalidate_root(local_path, root_fd, root_before)
    except BaseException as exc:
        raise _close(root_fd, exc)
    error = _close(root_fd, None)
    if error: raise error
    return result


def _add_asset_cleanup_note(primary, cleanup):
    note = ("owned model tree snapshot cleanup failed: " + repr(cleanup))[:512]
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        try:
            add_note(note)
            return
        except BaseException:
            pass
    try:
        setattr(primary, "__notes__", list(getattr(primary, "__notes__", ())) + [note])
    except BaseException:
        pass


def _snapshot_descriptor_removed(fd, identity):
    current = os.fstat(fd)
    if _binding(current) != identity:
        raise LocalAssetError("owned snapshot descriptor identity changed")
    return current.st_nlink == 0


def _snapshot_binding_matches(root_fd, parent_fd, name, identity):
    try:
        held = os.fstat(root_fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise LocalAssetError("owned snapshot path binding unavailable") from exc
    return (_binding(held) == identity and _binding(current) == identity and
            stat.S_ISDIR(held.st_mode) and stat.S_ISDIR(current.st_mode))


def _snapshot_renameat2():
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise LocalAssetError(
            "renameat2(RENAME_NOREPLACE) is required for owned snapshot cleanup"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    return renameat2


def _require_snapshot_rename_capability():
    """在创建任何 snapshot 路径前确认内核接受 RENAME_NOREPLACE。"""
    renameat2 = _snapshot_renameat2()
    probe = os.fsencode(".audio-filter-renameat2-probe-" + secrets.token_hex(16))
    if renameat2(-1, probe, -1, probe, 1) == 0:
        raise LocalAssetError("unexpected renameat2 capability probe result")
    if ctypes.get_errno() != errno.EBADF:
        raise LocalAssetError(
            "renameat2(RENAME_NOREPLACE) is unavailable for owned snapshot cleanup")


def _rename_snapshot_noreplace(source_fd, source_name, target_fd, target_name):
    """在固定目录 fd 间原子 rename，且绝不覆盖竞争 entry。"""
    renameat2 = _snapshot_renameat2()
    if renameat2(source_fd, os.fsencode(source_name), target_fd,
                 os.fsencode(target_name), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target_name)
    if error_number == errno.ENOENT:
        raise FileNotFoundError(error_number, os.strerror(error_number), source_name)
    raise OSError(error_number, os.strerror(error_number), source_name)


def _snapshot_name_matches(parent_fd, name, expected):
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return _binding(current) == expected


def _allocate_snapshot_quarantine(parent_fd):
    """分配 quarantine；factory 回滚可绕过被注入故障的公开 helper。"""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(128):
        container_name = ".audio-filter-model-cleanup-" + secrets.token_hex(16)
        try:
            os.mkdir(container_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        container_fd = quarantine_fd = None
        quarantine_name = None
        try:
            container_fd = os.open(container_name, flags, dir_fd=parent_fd)
            container_identity = _binding(os.fstat(container_fd))
            if (container_identity[2] != stat.S_IFDIR or
                    not _snapshot_name_matches(
                        parent_fd, container_name, container_identity) or
                    os.listdir(container_fd)):
                raise LocalAssetError("invalid snapshot cleanup container")
            quarantine_name = "quarantine-" + secrets.token_hex(16)
            os.mkdir(quarantine_name, 0o700, dir_fd=container_fd)
            quarantine_fd = os.open(quarantine_name, flags, dir_fd=container_fd)
            quarantine_identity = _binding(os.fstat(quarantine_fd))
            if (quarantine_identity[2] != stat.S_IFDIR or
                    not _snapshot_name_matches(
                        container_fd, quarantine_name, quarantine_identity) or
                    os.listdir(quarantine_fd)):
                raise LocalAssetError("invalid snapshot cleanup quarantine")
            # 外层容器保持不可遍历；内部目录只经预持有 fd 使用。
            os.fchmod(container_fd, 0)
            return {
                "container_name": container_name,
                "container_fd": container_fd,
                "container_identity": container_identity,
                "quarantine_name": quarantine_name,
                "quarantine_fd": quarantine_fd,
                "quarantine_identity": quarantine_identity,
            }
        except BaseException as primary:
            if container_fd is not None:
                try: os.fchmod(container_fd, 0o700)
                except BaseException as cleanup: _add_asset_cleanup_note(primary, cleanup)
            if container_fd is not None and quarantine_name is not None:
                try:
                    quarantine_stat = os.stat(
                        quarantine_name, dir_fd=container_fd,
                        follow_symlinks=False)
                    if (quarantine_fd is None or
                            _binding(os.fstat(quarantine_fd)) == _binding(quarantine_stat)):
                        os.rmdir(quarantine_name, dir_fd=container_fd)
                    else:
                        raise LocalAssetError(
                            "quarantine rollback lacks ownership evidence")
                except FileNotFoundError:
                    pass
                except BaseException as cleanup:
                    _add_asset_cleanup_note(primary, cleanup)
            if quarantine_fd is not None:
                try: os.close(quarantine_fd)
                except BaseException as cleanup: _add_asset_cleanup_note(primary, cleanup)
            if container_fd is not None:
                try: os.close(container_fd)
                except BaseException as cleanup: _add_asset_cleanup_note(primary, cleanup)
            try: os.rmdir(container_name, dir_fd=parent_fd)
            except BaseException as cleanup: _add_asset_cleanup_note(primary, cleanup)
            raise
    raise LocalAssetError("cannot allocate snapshot cleanup quarantine")


def _create_snapshot_quarantine(parent_fd):
    """创建 mode-0 外层容器和预持有的私有 quarantine fd。"""
    return _allocate_snapshot_quarantine(parent_fd)


def _open_snapshot_cleanup_entry(parent_fd, name, expected):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    flags |= os.O_DIRECTORY if expected[2] == stat.S_IFDIR else os.O_NONBLOCK
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        if _binding(os.fstat(fd)) != expected:
            raise LocalAssetError("snapshot cleanup binding changed")
        return fd
    except BaseException as exc:
        raise _close(fd, exc)


def _inspect_snapshot_cleanup_entry(parent_fd, name, expected):
    """破坏前重复核对 name，并以新 fd 固定实际 inode。"""
    if (not _snapshot_name_matches(parent_fd, name, expected) or
            not _snapshot_name_matches(parent_fd, name, expected)):
        raise LocalAssetError("snapshot cleanup binding changed")
    return _open_snapshot_cleanup_entry(parent_fd, name, expected)


def _remove_isolated_snapshot_contents(root_fd, path, bindings):
    """只在已隔离的 snapshot root 内按创建期 inode 清单递归删除。"""
    for _, name in _names(root_fd):
        name_bytes = name.encode("utf-8", "strict")
        child_path = name_bytes if not path else path + b"/" + name_bytes
        expected = bindings.get(child_path)
        if expected is None or expected[2] not in (stat.S_IFREG, stat.S_IFDIR):
            raise LocalAssetError("unexpected snapshot cleanup entry")
        child_fd = _inspect_snapshot_cleanup_entry(root_fd, name, expected)
        primary = None
        try:
            if expected[2] == stat.S_IFDIR:
                os.fchmod(child_fd, 0o700)
                _remove_isolated_snapshot_contents(child_fd, child_path, bindings)
                if (_binding(os.fstat(child_fd)) != expected or
                        not _snapshot_name_matches(root_fd, name, expected) or
                        os.listdir(child_fd)):
                    raise LocalAssetError("snapshot cleanup binding changed")
                os.rmdir(name, dir_fd=root_fd)
            else:
                if (_binding(os.fstat(child_fd)) != expected or
                        not _snapshot_name_matches(root_fd, name, expected)):
                    raise LocalAssetError("snapshot cleanup binding changed")
                os.unlink(name, dir_fd=root_fd)
            removed_stat = os.fstat(child_fd)
            if _binding(removed_stat) != expected or removed_stat.st_nlink != 0:
                raise LocalAssetError("snapshot cleanup removal could not be verified")
        except BaseException as exc:
            primary = exc
        error = _close(child_fd, primary)
        if error is not None: raise error
    if _names(root_fd):
        raise LocalAssetError("snapshot cleanup directory changed")


def _restore_snapshot_root(parent_fd, original_name, quarantine_fd,
                           claim_name, identity=None):
    try:
        moved = os.stat(
            claim_name, dir_fd=quarantine_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    moved_identity = _binding(moved)
    if identity is not None and moved_identity != identity:
        return False
    try:
        _rename_snapshot_noreplace(
            quarantine_fd, claim_name, parent_fd, original_name)
    except BaseException:
        return False
    return _snapshot_name_matches(parent_fd, original_name, moved_identity)


def _claim_snapshot_root(root_fd, parent_fd, cleanup_state, identity):
    """将 root 原子移入预持有的私有 quarantine，并重新打开核验。"""
    original_name = cleanup_state["original_name"]
    quarantine_fd = cleanup_state["quarantine_fd"]
    claim_name = cleanup_state["claim_name"]
    if (not _snapshot_name_matches(parent_fd, original_name, identity) or
            not _snapshot_name_matches(parent_fd, original_name, identity)):
        raise LocalAssetError("owned snapshot path binding changed")
    _rename_snapshot_noreplace(
        parent_fd, original_name, quarantine_fd, claim_name)
    # rename 已成功，后续核验无论如何失败都必须保留这一 provenance。
    cleanup_state["claimed"] = True
    try:
        claimed_fd = _open_snapshot_cleanup_entry(
            quarantine_fd, claim_name, identity)
        try:
            if _binding(os.fstat(root_fd)) != _binding(os.fstat(claimed_fd)):
                raise LocalAssetError("owned snapshot claim binding changed")
        finally:
            error = _close(claimed_fd, None)
            if error is not None:
                raise error
    except BaseException as exc:
        # 只有明确恢复了同一 inode 才允许再次尝试 claim。
        if _restore_snapshot_root(
                parent_fd, original_name, quarantine_fd, claim_name, identity):
            cleanup_state["claimed"] = False
        else:
            cleanup_state["terminal_provenance"] = (
                "snapshot claim verification failed and restoration failed; "
                "owned root remains in cleanup quarantine")
            _add_asset_cleanup_note(
                exc, LocalAssetError(cleanup_state["terminal_provenance"]))
        raise


def _cleanup_owned_snapshot_path(root_fd, parent_fd, cleanup_state, identity):
    """隔离 root 后删除；每次重试先用 held fd 对账已完成的 rmdir。"""
    if (not cleanup_state["path_removed"] and
            _snapshot_descriptor_removed(root_fd, identity)):
        cleanup_state["path_removed"] = True
        cleanup_state["claimed"] = False
    if not cleanup_state["path_removed"]:
        if not cleanup_state["claimed"]:
            _claim_snapshot_root(root_fd, parent_fd, cleanup_state, identity)
        quarantine_fd = cleanup_state["quarantine_fd"]
        claim_name = cleanup_state["claim_name"]
        try:
            if (_binding(os.fstat(root_fd)) != identity or
                    not _snapshot_name_matches(quarantine_fd, claim_name, identity)):
                raise LocalAssetError("owned snapshot path binding changed")
            os.fchmod(root_fd, 0o700)
            _remove_isolated_snapshot_contents(
                root_fd, b"", cleanup_state["bindings"])
            if (os.listdir(root_fd) or _binding(os.fstat(root_fd)) != identity or
                    not _snapshot_name_matches(quarantine_fd, claim_name, identity)):
                raise LocalAssetError("owned snapshot cleanup binding changed")
            os.rmdir(claim_name, dir_fd=quarantine_fd)
            if not _snapshot_descriptor_removed(root_fd, identity):
                raise LocalAssetError("owned snapshot root removal could not be verified")
            cleanup_state["path_removed"] = True
            cleanup_state["claimed"] = False
        except BaseException as exc:
            if cleanup_state.get("terminal_provenance") is not None:
                _add_asset_cleanup_note(
                    exc, LocalAssetError(cleanup_state["terminal_provenance"]))
            elif cleanup_state["claimed"] and _restore_snapshot_root(
                    parent_fd, cleanup_state["original_name"], quarantine_fd,
                    claim_name, identity):
                cleanup_state["claimed"] = False
            elif cleanup_state["claimed"]:
                _add_asset_cleanup_note(exc, LocalAssetError(
                    "owned snapshot remains claimed in cleanup quarantine"))
            raise
    container_fd = cleanup_state["container_fd"]
    quarantine_fd = cleanup_state["quarantine_fd"]
    quarantine_identity = cleanup_state["quarantine_identity"]
    if (not cleanup_state["quarantine_removed"] and
            _snapshot_descriptor_removed(quarantine_fd, quarantine_identity)):
        cleanup_state["quarantine_removed"] = True
    if not cleanup_state["quarantine_removed"]:
        quarantine_name = cleanup_state["quarantine_name"]
        os.fchmod(container_fd, 0o700)
        if (os.listdir(quarantine_fd) or
                not _snapshot_name_matches(
                    container_fd, quarantine_name, quarantine_identity)):
            raise LocalAssetError("owned snapshot quarantine binding changed")
        os.rmdir(quarantine_name, dir_fd=container_fd)
        if not _snapshot_descriptor_removed(quarantine_fd, quarantine_identity):
            raise LocalAssetError("snapshot quarantine removal could not be verified")
        cleanup_state["quarantine_removed"] = True
    container_identity = cleanup_state["container_identity"]
    if (not cleanup_state["container_removed"] and
            _snapshot_descriptor_removed(container_fd, container_identity)):
        cleanup_state["container_removed"] = True
    if not cleanup_state["container_removed"]:
        container_name = cleanup_state["container_name"]
        if (os.listdir(container_fd) or
                not _snapshot_name_matches(
                    parent_fd, container_name, container_identity)):
            raise LocalAssetError("owned snapshot cleanup container changed")
        os.rmdir(container_name, dir_fd=parent_fd)
        if not _snapshot_descriptor_removed(container_fd, container_identity):
            raise LocalAssetError("snapshot cleanup container removal could not be verified")
        cleanup_state["container_removed"] = True


_OWNED_MODEL_TREE_SNAPSHOT_TOKEN = object()


class OwnedModelTreeSnapshot:
    """PID-owned、descriptor-bound 的私有只读模型树快照。"""

    __slots__ = ("_token", "_source_path", "_path", "_sha256", "_limits",
                 "_manifest", "_root_fd", "_parent_fd", "_name", "_identity",
                 "_owner_pid", "_cleanup_state", "_closing", "_closed", "_lock")

    def __new__(cls, *args, **kwargs):
        raise TypeError("OwnedModelTreeSnapshot instances are created by open_owned_model_tree_snapshot")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("OwnedModelTreeSnapshot cannot be subclassed")

    def __setattr__(self, name, value):
        raise AttributeError("OwnedModelTreeSnapshot is immutable")

    def __reduce__(self):
        raise TypeError("OwnedModelTreeSnapshot cannot be serialized")

    def _release_inherited_descriptors(self):
        """fork child 只消费自身 fd 副本，绝不触碰父进程拥有的路径。"""
        state = self._cleanup_state
        descriptors = (
            ("object", "_root_fd", self._identity),
            ("state", "quarantine_fd", state["quarantine_identity"]),
            ("state", "container_fd", state["container_identity"]),
            ("object", "_parent_fd", state["parent_identity"]),
        )
        for owner, field, expected in descriptors:
            fd = getattr(self, field) if owner == "object" else state[field]
            if fd < 0:
                continue
            try:
                matches = _binding(os.fstat(fd)) == expected
            except OSError:
                matches = False
            if owner == "object": object.__setattr__(self, field, -1)
            else: state[field] = -1
            if matches:
                try: os.close(fd)
                except OSError: pass

    def _require_owner(self):
        if self._token is not _OWNED_MODEL_TREE_SNAPSHOT_TOKEN:
            raise LocalAssetError("invalid owned model tree snapshot")
        if os.getpid() != self._owner_pid:
            self._release_inherited_descriptors()
            raise LocalAssetError("owned model tree snapshot belongs to another process")

    def _raise_terminal(self):
        provenance = self._cleanup_state.get("terminal_provenance")
        if provenance is not None:
            raise LocalAssetError("owned snapshot cleanup provenance is terminal: " + provenance)
        if self._cleanup_state["fd_close_failed"]:
            raise LocalAssetError(
                "snapshot descriptor close result is terminal and uncertain")

    @property
    def source_path(self): return self._source_path

    @property
    def path(self): return self._path

    @property
    def sha256(self): return self._sha256

    @property
    def manifest(self): return self._manifest

    def _loader_path_locked(self):
        self._raise_terminal()
        if self._closed: raise LocalAssetError("owned model tree snapshot is closed")
        try:
            valid = self._root_fd >= 0 and _binding(os.fstat(self._root_fd)) == self._identity
        except OSError as exc:
            raise LocalAssetError("owned snapshot descriptor identity unavailable") from exc
        if not valid:
            raise LocalAssetError("owned snapshot descriptor identity changed")
        return f"/proc/self/fd/{self._root_fd}"

    @contextmanager
    def loader_path_lease(self):
        """在 context 生命周期内阻止并发 close，并租借 descriptor 路径。"""
        self._require_owner()
        with self._lock:
            self._cleanup_state["loader_leases"] += 1
            try:
                yield self._loader_path_locked()
            finally:
                self._cleanup_state["loader_leases"] -= 1

    @property
    def loader_path(self):
        """兼容属性；并发 loader 应改用 loader_path_lease()。"""
        with self.loader_path_lease() as path:
            return path

    def _verify_ready_locked(self):
        self._raise_terminal()
        if self._closed:
            raise LocalAssetError("owned model tree snapshot is closed")
        if self._cleanup_state["claimed"]:
            raise LocalAssetError("owned model tree snapshot cleanup is incomplete")
        if not _snapshot_binding_matches(
                self._root_fd, self._parent_fd, self._name, self._identity):
            raise LocalAssetError("owned snapshot path binding changed")

    def verify_manifest(self):
        """仅读取目录项/stat metadata，快速复核不可变 snapshot manifest。"""
        self._require_owner()
        with self._lock:
            self._verify_ready_locked()
            _verify_snapshot_manifest(self._root_fd, self._manifest, self._limits)
            if not _snapshot_binding_matches(
                    self._root_fd, self._parent_fd, self._name, self._identity):
                raise LocalAssetError("owned snapshot path binding changed")
            return self._sha256

    def verify(self):
        """完整重读内容 digest，并同时复核 immutable metadata manifest。"""
        self._require_owner()
        with self._lock:
            self._verify_ready_locked()
            entries = []
            state = {"entries": 0, "bytes": 0, "visited": set(),
                     "require_snapshot_modes": True}
            _scan_tree(self._root_fd, self._limits, state, entries)
            _verify_snapshot_manifest(self._root_fd, self._manifest, self._limits)
            if not _snapshot_binding_matches(
                    self._root_fd, self._parent_fd, self._name, self._identity):
                raise LocalAssetError("owned snapshot path binding changed")
            actual = _digest_entries(entries)
            if not hmac.compare_digest(actual, self._sha256):
                raise AssetDigestMismatchError(
                    f"owned snapshot digest mismatch: expected {self._sha256}, actual {actual}")
            return actual

    def close(self):
        """幂等清理 owned snapshot；路径故障可重试，close 故障终止。"""
        self._require_owner()
        with self._lock:
            self._raise_terminal()
            if self._closed: return
            if self._cleanup_state["loader_leases"]:
                raise LocalAssetError("owned model tree snapshot has an active loader path lease")
            if self._closing:
                raise LocalAssetError("owned model tree snapshot cleanup is in progress")
            object.__setattr__(self, "_closing", True)
            state = self._cleanup_state
            try:
                if not state["container_removed"]:
                    _cleanup_owned_snapshot_path(
                        self._root_fd, self._parent_fd, state, self._identity)
                primary = None
                descriptors = (
                    ("object", "_root_fd", "root descriptor"),
                    ("state", "quarantine_fd", "quarantine descriptor"),
                    ("state", "container_fd", "cleanup container descriptor"),
                    ("object", "_parent_fd", "parent descriptor"),
                )
                for owner, field, label in descriptors:
                    fd = getattr(self, field) if owner == "object" else state[field]
                    if fd < 0:
                        continue
                    # close(2) 后结果不确定；调用前消费 slot，之后绝不复用整数。
                    if owner == "object": object.__setattr__(self, field, -1)
                    else: state[field] = -1
                    try:
                        os.close(fd)
                    except OSError as exc:
                        error = LocalAssetError(f"failed to close snapshot {label}")
                        error.__cause__ = exc
                        if primary is None: primary = error
                        else: _add_asset_cleanup_note(primary, error)
                if primary is not None:
                    state["fd_close_failed"] = True
                    raise primary
                object.__setattr__(self, "_closed", True)
            finally:
                object.__setattr__(self, "_closing", False)

    def __enter__(self):
        self._require_owner()
        with self._lock:
            self._raise_terminal()
            if self._closed: raise LocalAssetError("owned model tree snapshot is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None: raise
            _add_asset_cleanup_note(exc, cleanup)
        return False

    def __del__(self):
        try:
            state = getattr(self, "_cleanup_state", None)
            if (getattr(self, "_token", None) is _OWNED_MODEL_TREE_SNAPSHOT_TOKEN and
                    not getattr(self, "_closed", True) and
                    not getattr(self, "_closing", True) and
                    state is not None and not state.get("fd_close_failed", True) and
                    os.getpid() == getattr(self, "_owner_pid", None)):
                self.close()
        except BaseException:
            pass


def _new_owned_model_tree_snapshot(source_path, path, sha256, limits, manifest,
                                   root_fd, parent_fd, name, identity,
                                   cleanup_state):
    result = object.__new__(OwnedModelTreeSnapshot)
    values = {
        "_token": _OWNED_MODEL_TREE_SNAPSHOT_TOKEN,
        "_source_path": source_path, "_path": path, "_sha256": sha256,
        "_limits": limits, "_manifest": manifest, "_root_fd": root_fd,
        "_parent_fd": parent_fd, "_name": name, "_identity": identity,
        "_owner_pid": os.getpid(), "_cleanup_state": cleanup_state,
        "_closing": False, "_closed": False, "_lock": threading.RLock(),
    }
    for field, value in values.items(): object.__setattr__(result, field, value)
    return result


def _factory_cleanup_snapshot_root(root_fd, parent_fd, name, identity, bindings,
                                   cleanup_state):
    """用已有或临时 quarantine 安全 claim 并删除 factory snapshot root。"""
    state = cleanup_state
    if state is None:
        state = _allocate_snapshot_quarantine(parent_fd)
        state.update({
            "original_name": name,
            "claim_name": "owned-snapshot-" + secrets.token_hex(16),
            "claimed": False,
            "path_removed": False,
            "quarantine_removed": False,
            "container_removed": False,
            "fd_close_failed": False,
            "terminal_provenance": None,
            "loader_leases": 0,
            "parent_identity": _binding(os.fstat(parent_fd)),
            "bindings": bindings,
        })
    _cleanup_owned_snapshot_path(root_fd, parent_fd, state, identity)
    return state


def _drain_factory_descriptors(primary, descriptors):
    """一次性消费所有局部 fd；单个 close 故障不得阻断后续 drain。"""
    for fd in descriptors:
        if fd is None or fd < 0:
            continue
        try:
            os.close(fd)
        except BaseException as cleanup:
            _add_asset_cleanup_note(primary, cleanup)


def _create_snapshot_root():
    """先持有 temp parent fd，再相对该 descriptor 创建并固定 root。"""
    parent_path = os.path.realpath(tempfile.gettempdir())
    parent_fd = _open_root(parent_path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        for _ in range(128):
            name = "audio-filter-model-tree-" + secrets.token_hex(16)
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            path = os.path.join(parent_path, name)
            root_fd = identity = local_state = None
            try:
                root_fd = os.open(name, flags, dir_fd=parent_fd)
                os.fchmod(root_fd, 0o700)
                root_stat = os.fstat(root_fd)
                identity = _binding(root_stat)
                if (not stat.S_ISDIR(root_stat.st_mode) or
                        stat.S_IMODE(root_stat.st_mode) != 0o700 or
                        not _snapshot_name_matches(parent_fd, name, identity) or
                        os.path.realpath(f"/proc/self/fd/{root_fd}") != path):
                    raise LocalAssetError("invalid owned snapshot directory")
                return path, name, parent_fd, root_fd, identity
            except BaseException as primary:
                if root_fd is not None and identity is not None:
                    try:
                        local_state = _factory_cleanup_snapshot_root(
                            root_fd, parent_fd, name, identity, {b"": identity}, None)
                    except BaseException as cleanup:
                        _add_asset_cleanup_note(primary, cleanup)
                else:
                    _add_asset_cleanup_note(primary, LocalAssetError(
                        "snapshot root was created but cannot be safely removed "
                        "because descriptor identity evidence is incomplete"))
                descriptors = [root_fd]
                if local_state is not None:
                    descriptors.extend((local_state.get("quarantine_fd", -1),
                                        local_state.get("container_fd", -1)))
                _drain_factory_descriptors(primary, descriptors)
                raise
        raise LocalAssetError("cannot allocate owned snapshot directory")
    except BaseException as primary:
        _drain_factory_descriptors(primary, [parent_fd])
        raise


def _open_owned_model_tree_snapshot(local_path, limits):
    snapshot_path = name = None
    parent_fd = root_fd = source_fd = identity = None
    snapshot_bindings = {}
    cleanup_state = None
    snapshot = None
    try:
        snapshot_path, name, parent_fd, root_fd, identity = _create_snapshot_root()
        snapshot_bindings[b""] = identity
        cleanup_state = _create_snapshot_quarantine(parent_fd)
        cleanup_state.update({
            "original_name": name,
            "claim_name": "owned-snapshot-" + secrets.token_hex(16),
            "claimed": False,
            "path_removed": False,
            "quarantine_removed": False,
            "container_removed": False,
            "fd_close_failed": False,
            "terminal_provenance": None,
            "loader_leases": 0,
            "parent_identity": _binding(os.fstat(parent_fd)),
            "bindings": snapshot_bindings,
        })
        source_fd = _open_root(local_path)
        entries, state = [], {"entries": 0, "bytes": 0, "visited": set(),
                              "snapshot_bindings": snapshot_bindings}
        source_before = _scan_tree(source_fd, limits, state, entries, root_fd)
        digest = _digest_entries(entries)
        _revalidate_root(local_path, source_fd, source_before)
        error = _close(source_fd, None)
        source_fd = None
        if error: raise error
        manifest = _build_snapshot_manifest(root_fd, limits)
        snapshot = _new_owned_model_tree_snapshot(
            local_path, snapshot_path, digest, limits, manifest,
            root_fd, parent_fd, name, identity, cleanup_state)
        root_fd = parent_fd = None
        snapshot.verify()
        return snapshot
    except BaseException as primary:
        local_state = cleanup_state
        if snapshot is not None:
            try:
                snapshot.close()
            except BaseException as cleanup:
                _add_asset_cleanup_note(primary, cleanup)
        elif root_fd is not None and parent_fd is not None and identity is not None:
            try:
                local_state = _factory_cleanup_snapshot_root(
                    root_fd, parent_fd, name, identity, snapshot_bindings,
                    cleanup_state)
            except BaseException as cleanup:
                _add_asset_cleanup_note(primary, cleanup)
        elif snapshot_path is not None:
            _add_asset_cleanup_note(primary, LocalAssetError(
                "snapshot cleanup skipped because ownership evidence is incomplete"))

        descriptors = [source_fd]
        if snapshot is None:
            descriptors.append(root_fd)
            if local_state is not None:
                descriptors.extend((local_state.get("quarantine_fd", -1),
                                    local_state.get("container_fd", -1)))
            descriptors.append(parent_fd)
        _drain_factory_descriptors(primary, descriptors)
        raise


def open_owned_model_tree_snapshot(
        local_path: str, *, limits: AssetScanLimits = AssetScanLimits()):
    """从同一次 source fd 遍历创建私有模型树快照并返回 owned handle。"""
    if type(local_path) is not str: _config_error("local_path")
    if not isinstance(limits, AssetScanLimits): _config_error("limits")
    capabilities = (os.name == "posix" and os.open in os.supports_dir_fd
                    and os.stat in os.supports_dir_fd and os.mkdir in os.supports_dir_fd
                    and os.unlink in os.supports_dir_fd and os.rmdir in os.supports_dir_fd
                    and os.path.isdir("/proc/self/fd") and all(hasattr(os, name) for name in
                    ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")))
    if not capabilities: raise LocalAssetError("descriptor-bound snapshotting unavailable")
    try:
        local_path.encode("utf-8", "strict")
        _require_snapshot_rename_capability()
        return _open_owned_model_tree_snapshot(local_path, limits)
    except LocalAssetError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        error = LocalAssetError("local model asset snapshot failed")
        error.__cause__ = exc
        for note in getattr(exc, "__notes__", ()):
            _add_asset_cleanup_note(error, note)
        raise error


def local_model_tree_sha256(local_path: str, *, limits: AssetScanLimits = AssetScanLimits()) -> str:
    if type(local_path) is not str: _config_error("local_path")
    if not isinstance(limits, AssetScanLimits): _config_error("limits")
    capabilities = (os.name == "posix" and os.open in os.supports_dir_fd
                    and os.stat in os.supports_dir_fd and os.path.isdir("/proc/self/fd")
                    and all(hasattr(os, name) for name in
                            ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")))
    if not capabilities: raise LocalAssetError("descriptor-bound scanning unavailable")
    try:
        local_path.encode("utf-8", "strict")
        return _tree_digest(local_path, limits)
    except LocalAssetError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise LocalAssetError("local model asset scan failed") from exc

def verify_local_model_asset(asset: LocalModelAsset, *, limits: AssetScanLimits = AssetScanLimits()) -> str:
    if not isinstance(asset, LocalModelAsset): _config_error("asset")
    if not isinstance(limits, AssetScanLimits): _config_error("limits")
    actual = local_model_tree_sha256(asset.local_path, limits=limits)
    if not hmac.compare_digest(actual, asset.sha256):
        raise AssetDigestMismatchError(
            f"local asset digest mismatch: expected {asset.sha256}, actual {actual}")
    return actual


_MFD_CLOEXEC = 1
_MFD_ALLOW_SEALING = 2
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 1
_F_SEAL_SHRINK = 2
_F_SEAL_GROW = 4
_F_SEAL_WRITE = 8
_REQUIRED_WORKER_SEALS = _F_SEAL_WRITE | _F_SEAL_GROW | _F_SEAL_SHRINK | _F_SEAL_SEAL


def _memfd_create(name):
    flags = _MFD_ALLOW_SEALING | _MFD_CLOEXEC
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)
    if sys.platform != "linux":
        raise OSError("Linux memfd_create unavailable")
    try:
        memfd_create = ctypes.CDLL(None, use_errno=True).memfd_create
    except AttributeError as exc:
        raise OSError("libc memfd_create unavailable") from exc
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    descriptor = memfd_create(name.encode("ascii"), flags)
    if descriptor < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return descriptor


def _require_worker_capabilities():
    os_names = ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK",
                "killpg", "waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    rlimit_names = ("RLIMIT_CORE", "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_FSIZE", "RLIMIT_AS")
    if (os.name != "posix" or sys.platform != "linux" or not os.path.isdir("/proc/self/fd") or
            os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd or
            any(not hasattr(os, name) for name in os_names) or resource is None or
            any(not hasattr(resource, name) for name in rlimit_names) or
            not hasattr(selectors, "DefaultSelector")):
        raise WorkerUnavailableError("optional worker executor capabilities unavailable")
    try:
        descriptor = _memfd_create("audio_filter_optional_probe")
    except (AttributeError, OSError, TypeError, UnicodeError) as exc:
        raise WorkerUnavailableError("optional worker executor capabilities unavailable") from exc
    error = _close_worker_fd(descriptor)
    if error:
        raise error

def _worker_full_snapshot(value):
    return (value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_mode & 0o7777,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _open_worker_path(path, *, directory=False, root_fd=None):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    parts = (path.lstrip("/") if root_fd is None else path).split("/")
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC) if root_fd is None else os.dup(root_fd)
    try:
        for index, component in enumerate(parts):
            if not component or component in (".", ".."): raise OSError("unsafe component")
            final = index == len(parts) - 1
            open_flags = flags | (os.O_DIRECTORY if directory or not final else os.O_NONBLOCK)
            next_fd = os.open(component, open_flags, dir_fd=current)
            before, current = current, next_fd
            error = _close_worker_fd(before)
            if error: raise error
            mode = os.fstat(current).st_mode
            if (directory or not final) and not stat.S_ISDIR(mode): raise OSError("not directory")
        return current
    except BaseException as exc: raise _close_worker_fd(current, exc)


def _read_worker_fd(fd, cap, chunk_size):
    os.lseek(fd, 0, os.SEEK_SET)
    result = bytearray()
    while True:
        try: chunk = os.read(fd, min(chunk_size, cap + 1 - len(result)))
        except InterruptedError: continue
        if not chunk: return bytes(result)
        result.extend(chunk)
        if len(result) > cap: raise WorkerUnavailableError("worker script size exceeds limit")


def _prepare_worker_descriptors(spec, limits):
    owned = []
    try:
        python_fd = _open_worker_path(spec.python_executable); owned.append(python_fd)
        python_stat = os.fstat(python_fd)
        if not stat.S_ISREG(python_stat.st_mode) or not python_stat.st_mode & 0o111: raise OSError("invalid executable")
        root_fd = _open_worker_path(spec.trusted_code_root, directory=True); owned.append(root_fd)
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode): raise OSError("invalid trusted root")
        root_before = _worker_full_snapshot(root_stat)
        relative = os.path.relpath(spec.worker_script, spec.trusted_code_root)
        script_fd = _open_worker_path(relative, root_fd=root_fd); owned.append(script_fd)
        script_stat = os.fstat(script_fd)
        if not stat.S_ISREG(script_stat.st_mode) or not 0 <= script_stat.st_size <= limits.max_worker_script_bytes: raise OSError("invalid script")
        expected_paths = ((python_fd, spec.python_executable), (root_fd, spec.trusted_code_root),
                          (script_fd, spec.worker_script))
        if any(" (deleted)" in os.path.realpath(f"/proc/self/fd/{fd}") or
               os.path.realpath(f"/proc/self/fd/{fd}") != path for fd, path in expected_paths):
            raise OSError("descriptor path mismatch")
        script_before = _worker_full_snapshot(script_stat)
        script_bytes = _read_worker_fd(script_fd, limits.max_worker_script_bytes, limits.io_chunk_bytes)
        if len(script_bytes) != script_stat.st_size or _worker_full_snapshot(os.fstat(script_fd)) != script_before: raise OSError("script changed")
        if not hmac.compare_digest(hashlib.sha256(script_bytes).hexdigest(), spec.worker_script_sha256):
            raise WorkerScriptDigestMismatchError("worker script digest mismatch")
        snapshot_fd = _memfd_create("audio_filter_optional_worker"); owned.append(snapshot_fd)
        offset = 0
        while offset < len(script_bytes):
            try: written = os.write(snapshot_fd, script_bytes[offset:offset + limits.io_chunk_bytes])
            except InterruptedError: continue
            if written <= 0: raise OSError("snapshot write failed")
            offset += written
        if not stat.S_ISREG(os.fstat(snapshot_fd).st_mode) or os.fstat(snapshot_fd).st_size != len(script_bytes): raise OSError("snapshot invalid")
        if hashlib.sha256(_read_worker_fd(snapshot_fd, limits.max_worker_script_bytes, limits.io_chunk_bytes)).hexdigest() != spec.worker_script_sha256: raise WorkerScriptDigestMismatchError("worker snapshot digest mismatch")
        fcntl.fcntl(snapshot_fd, _F_ADD_SEALS, _REQUIRED_WORKER_SEALS)
        if fcntl.fcntl(snapshot_fd, _F_GET_SEALS) & _REQUIRED_WORKER_SEALS != _REQUIRED_WORKER_SEALS: raise OSError("snapshot sealing failed")
        if hashlib.sha256(_read_worker_fd(snapshot_fd, limits.max_worker_script_bytes, limits.io_chunk_bytes)).hexdigest() != spec.worker_script_sha256: raise WorkerScriptDigestMismatchError("sealed snapshot digest mismatch")
        reopened = _open_worker_path(relative, root_fd=root_fd)
        try:
            if (_worker_full_snapshot(os.fstat(reopened)) != script_before or
                    os.path.realpath(f"/proc/self/fd/{reopened}") != spec.worker_script):
                raise OSError("script binding changed")
        except BaseException as exc: raise _close_worker_fd(reopened, exc)
        error = _close_worker_fd(reopened)
        if error: raise error
        owned.remove(script_fd); error = _close_worker_fd(script_fd)
        if error is not None: raise error
        return python_fd, root_fd, snapshot_fd, _worker_full_snapshot(python_stat), root_before
    except BaseException as exc:
        primary = exc
        if isinstance(exc, (OSError, ValueError)):
            primary = WorkerUnavailableError("trusted optional worker input unavailable"); primary.__cause__ = exc
            for note in getattr(exc, "__notes__", ()): _add_worker_note(primary, note)
        for fd in reversed(tuple(owned)):
            owned.remove(fd)
            primary = _close_worker_fd(fd, primary)
        raise primary

def _worker_close_once(fd):
    try: os.close(fd)
    except OSError as exc: return exc, True
    return None, True

def _close_worker_fd(fd, primary=None):
    error = None
    if fd >= 0:
        for _ in range(2):
            exc, released = _worker_close_once(fd)
            if error is None: error = exc
            if released: break
        if error is not None:
            if primary is None: primary = WorkerCleanupError("worker descriptor cleanup failed"); primary.__cause__ = error
            else: _add_worker_note(primary, "descriptor cleanup: " + repr(error))
    return primary
_OPTIONAL_WORKER_BOOTSTRAP = r'''import os,sys,resource

def fail(text):
 try: os.write(2,("bootstrap: "+text+"\n").encode("ascii","replace")[:512])
 except BaseException: pass
 raise SystemExit(120)
def number(value,positive=False):
 if not value.isdecimal(): fail("invalid arguments")
 result=int(value)
 if result < (1 if positive else 0): fail("invalid arguments")
 return result
def lower(which,value):
 soft,hard=resource.getrlimit(which); target=value
 if soft != resource.RLIM_INFINITY: target=min(target,soft)
 if hard != resource.RLIM_INFINITY: target=min(target,hard)
 resource.setrlimit(which,(target,target))
root_fd=None
try:
 if len(sys.argv)!=14: fail("invalid arguments")
 python_fd,root_fd,snapshot_fd=(number(x) for x in sys.argv[1:4])
 executable,root,script,digest=sys.argv[4:8]
 cpu,nofile,fsize,address,cap=(number(x,True) for x in sys.argv[8:13])
 persistent=number(sys.argv[13])
 if persistent not in (0,1): fail("invalid arguments")
 os.umask(0o077); lower(resource.RLIMIT_CORE,0); lower(resource.RLIMIT_CPU,cpu)
 lower(resource.RLIMIT_NOFILE,nofile); lower(resource.RLIMIT_FSIZE,fsize); lower(resource.RLIMIT_AS,address)
 import fcntl,hashlib
 F_GET_SEALS=1034
 required=1|2|4|8
 def target(fd): return os.path.realpath("/proc/self/fd/"+str(fd))
 if target(python_fd)!=executable or target(root_fd)!=root: fail("descriptor identity")
 if not __import__('stat').S_ISREG(os.fstat(python_fd).st_mode) or not __import__('stat').S_ISDIR(os.fstat(root_fd).st_mode): fail("descriptor type")
 if (os.stat("/proc/self/exe").st_dev,os.stat("/proc/self/exe").st_ino)!=(os.fstat(python_fd).st_dev,os.fstat(python_fd).st_ino): fail("executable identity")
 info=os.fstat(snapshot_fd)
 if not __import__('stat').S_ISREG(info.st_mode) or info.st_size>cap or fcntl.fcntl(snapshot_fd,F_GET_SEALS)&required!=required: fail("snapshot identity")
 os.lseek(snapshot_fd,0,0); pieces=[]; total=0
 while True:
  chunk=os.read(snapshot_fd,min(65536,cap+1-total))
  if not chunk: break
  pieces.append(chunk); total+=len(chunk)
  if total>cap: fail("snapshot size")
 data=b''.join(pieces)
 if hashlib.sha256(data).hexdigest()!=digest: fail("snapshot digest")
 os.close(python_fd); os.close(snapshot_fd); os.set_inheritable(root_fd,False)
 sys.path.insert(0,"/proc/self/fd/"+str(root_fd)); sys.argv=[script]+(["--persistent"] if persistent else [])
 scope={"__name__":"__main__","__file__":script,"__package__":None,"__spec__":None,"__builtins__":__builtins__}
 exec(compile(data,script,"exec"),scope,scope)
except SystemExit: raise
except BaseException as exc: fail(type(exc).__name__+":"+str(exc)[:256])
finally:
 if root_fd is not None:
  try: os.close(root_fd)
  except OSError: pass
'''

def _asset_environment_paths(spec, asset):
    result = {}
    for name, relative in spec.asset_env_paths:
        path = os.path.join(asset.local_path, relative)
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise LocalAssetError(f"required model asset subdirectory unavailable: {relative}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LocalAssetError(f"required model asset subdirectory invalid: {relative}")
        result[name] = path
    return result


def _worker_environment(spec, asset_paths):
    result = {"HOME": "/nonexistent", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
              "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "NO_PROXY": "*",
              "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
              "TO" + "RCH_HOME": "/nonexistent", "TRANSFORMERS_OFFLINE": "1", "TZ": "UTC",
              "XDG_CACHE_HOME": "/nonexistent", "no_proxy": "*"}
    result.update(asset_paths)
    for name in spec.inherit_env:
        if name in os.environ:
            value = os.environ[name]
            if type(value) is not str or "\0" in value or _has_control(value):
                raise WorkerUnavailableError("invalid inherited worker environment")
            result[name] = value
    result.update(spec.env_overrides)
    return result
def _add_worker_note(primary, note):
    add_note = getattr(primary, "add_note", None); note = note[:512]
    add_note(note) if add_note is not None else setattr(primary, "__notes__", list(getattr(primary, "__notes__", ())) + [note])

def _worker_pipe_close(selector, streams, name, primary):
    stream = streams.pop(name, None)
    if stream is None: return primary
    try:
        if selector is not None: selector.unregister(stream)
    except (KeyError, ValueError): pass
    except OSError as exc:
        if primary is None: primary = WorkerCleanupError("worker selector cleanup failed"); primary.__cause__ = exc
        else: _add_worker_note(primary, "selector cleanup: " + repr(exc))
    try: stream.close()
    except OSError as exc:
        if primary is None: primary = WorkerCleanupError("worker pipe cleanup failed"); primary.__cause__ = exc
        else: _add_worker_note(primary, "pipe cleanup: " + repr(exc))
    return primary


def _worker_output_primary(primary, name):
    if isinstance(primary, (KeyboardInterrupt, SystemExit, WorkerTimeoutError,
                            WorkerOutputLimitError)): return primary
    error = WorkerOutputLimitError(f"worker {name} exceeds limit")
    if primary is not None:
        cause = getattr(primary, "__cause__", None)
        _add_worker_note(error, "superseded: " + repr(primary) + (" caused by " + repr(cause) if cause else ""))
    return error


def _worker_wait(proc, limits, primary, label):
    end = time.monotonic() + limits.kill_wait_ms / 1000.0
    while True:
        try: return proc.wait(timeout=max(0, end - time.monotonic())), primary
        except (KeyboardInterrupt, SystemExit) as exc:
            if primary is not None: _add_worker_note(exc, "superseded: " + repr(primary))
            primary = exc
            if time.monotonic() >= end: return None, primary
        except BaseException as exc:
            cleanup = WorkerCleanupError("worker direct child could not be reaped"); cleanup.__cause__ = exc
            if primary is None: primary = cleanup
            else: _add_worker_note(primary, label + ": " + repr(cleanup))
            return None, primary


def _abort_worker_after_spawn(proc, limits, primary):
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try: stream.close()
        except OSError as exc: _add_worker_note(primary, "spawn-abort pipe: " + repr(exc))
    for sent in (signal.SIGTERM, signal.SIGKILL):
        try: os.killpg(proc.pid, sent)
        except ProcessLookupError: pass
        except OSError as exc: _add_worker_note(primary, "spawn-abort signal: " + repr(exc))
    _, primary = _worker_wait(proc, limits, primary, "spawn-abort reap")
    raise primary


def _run_worker_process(proc, encoded, limits, start):
    selector, streams = None, {"stdin": proc.stdin, "stdout": proc.stdout, "stderr": proc.stderr}
    buffers, offset, primary, child_exited = {"stdout": bytearray(), "stderr": bytearray()}, 0, None, False
    deadline = start + limits.timeout_ms / 1000.0
    try:
        selector = selectors.DefaultSelector()
        for stream in streams.values(): os.set_blocking(stream.fileno(), False)
        selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        while not child_exited:
            now = time.monotonic()
            if now >= deadline: primary = WorkerTimeoutError("optional worker timed out"); break
            for key, _ in selector.select(min(deadline - now, .1)):
                if time.monotonic() >= deadline: primary = WorkerTimeoutError("optional worker timed out"); break
                name, fd = key.data, key.fileobj.fileno()
                try:
                    if name == "stdin":
                        written = os.write(fd, encoded[offset:offset + limits.io_chunk_bytes])
                        if written <= 0: raise OSError("zero worker input write")
                        offset += written
                        if offset == len(encoded): primary = _worker_pipe_close(selector, streams, name, primary)
                    else:
                        cap = limits.max_stdout_bytes if name == "stdout" else limits.max_stderr_bytes
                        chunk = os.read(fd, min(limits.io_chunk_bytes, cap + 1 - len(buffers[name])))
                        if chunk:
                            buffers[name].extend(chunk)
                            if len(buffers[name]) > cap: primary = WorkerOutputLimitError(f"worker {name} exceeds limit"); break
                        else: primary = _worker_pipe_close(selector, streams, name, primary)
                except (BlockingIOError, InterruptedError): pass
                except BrokenPipeError: primary = _worker_pipe_close(selector, streams, name, primary)
                except OSError as exc:
                    primary = WorkerUnavailableError("optional worker pipe I/O failed"); primary.__cause__ = exc; break
            if primary is not None: break
            child_exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        if primary is not None or (child_exited and ("stdout" in streams or "stderr" in streams)):
            end = time.monotonic() + limits.terminate_grace_ms / 1000.0
        else: end = time.monotonic()
        primary = _worker_pipe_close(selector, streams, "stdin", primary)
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        except OSError as exc:
            if primary is None: primary = WorkerCleanupError("worker process-group cleanup failed"); primary.__cause__ = exc
            else: _add_worker_note(primary, "TERM cleanup: " + repr(exc))
        while time.monotonic() < end and ("stdout" in streams or "stderr" in streams):
            for key, _ in selector.select(min(end - time.monotonic(), .05)):
                name, fd = key.data, key.fileobj.fileno()
                if name == "stdin": continue
                cap = limits.max_stdout_bytes if name == "stdout" else limits.max_stderr_bytes
                try: chunk = os.read(fd, min(limits.io_chunk_bytes, cap + 1 - len(buffers[name])))
                except (BlockingIOError, InterruptedError): continue
                if chunk:
                    buffers[name].extend(chunk)
                    if len(buffers[name]) > cap: primary = _worker_output_primary(primary, name)
                else: primary = _worker_pipe_close(selector, streams, name, primary)
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        except OSError as exc:
            if primary is None: primary = WorkerCleanupError("worker process-group cleanup failed"); primary.__cause__ = exc
            else: _add_worker_note(primary, "KILL cleanup: " + repr(exc))
        kill_end = time.monotonic() + limits.kill_wait_ms / 1000.0
        while time.monotonic() < kill_end and ("stdout" in streams or "stderr" in streams):
            for key, _ in selector.select(min(kill_end - time.monotonic(), .05)):
                name = key.data
                if name == "stdin": continue
                cap = limits.max_stdout_bytes if name == "stdout" else limits.max_stderr_bytes
                try: chunk = os.read(key.fileobj.fileno(), min(limits.io_chunk_bytes, cap + 1 - len(buffers[name])))
                except (BlockingIOError, InterruptedError): continue
                if chunk:
                    buffers[name].extend(chunk)
                    if len(buffers[name]) > cap: primary = _worker_output_primary(primary, name)
                else: primary = _worker_pipe_close(selector, streams, name, primary)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)): primary = exc
        elif primary is not None: _add_worker_note(primary, "teardown secondary: " + repr(exc))
        elif isinstance(exc, OSError):
            primary = WorkerUnavailableError("optional worker selector failed"); primary.__cause__ = exc
        else: primary = exc
        try: os.killpg(proc.pid, signal.SIGTERM); os.killpg(proc.pid, signal.SIGKILL)
        except OSError as secondary: _add_worker_note(primary, "exception cleanup: " + repr(secondary))
    finally:
        for name in tuple(streams): primary = _worker_pipe_close(selector, streams, name, primary)
        try:
            if selector is not None: selector.close()
        except OSError as exc:
            if primary is None: primary = WorkerCleanupError("worker selector cleanup failed"); primary.__cause__ = exc
            else: _add_worker_note(primary, "selector close: " + repr(exc))
        returncode, primary = _worker_wait(proc, limits, primary, "final reap")
    if primary is not None: raise primary
    if returncode != 0: raise WorkerExitError(f"worker exited with status {returncode} (stderr_bytes={len(buffers['stderr'])})")
    return bytes(buffers["stdout"])


def _process_group_has_live_member(process_group_id):
    """只在已发送 TERM 后决定是否仍需 KILL，避免向已消失组重发信号。"""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return True
    for entry in entries:
        if not entry.isdecimal():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as stream:
                raw = stream.read()
            closing = raw.rfind(b")")
            fields = raw[closing + 2:].split()
            if len(fields) >= 3 and int(fields[2]) == process_group_id and fields[0] != b"Z":
                return True
        except (OSError, ValueError):
            continue
    return False


class OptionalWorkerSession:
    """在一个经验证的可选 worker 内串行执行多个严格绑定请求。"""

    def __init__(
        self,
        spec: OptionalWorkerSpec,
        asset: LocalModelAsset,
        *,
        limits: WorkerLimits,
        asset_limits: AssetScanLimits = AssetScanLimits(),
    ) -> None:
        if not isinstance(spec, OptionalWorkerSpec): _config_error("spec")
        if not isinstance(asset, LocalModelAsset): _config_error("asset")
        if not isinstance(limits, WorkerLimits): _config_error("limits")
        if not isinstance(asset_limits, AssetScanLimits): _config_error("asset_limits")
        self._spec, self._asset = spec, asset
        self._limits, self._asset_limits = limits, asset_limits
        self._proc = None
        self._process_group_id = None
        self._snapshot = None
        self._session_asset = None
        self._closed = False

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _start(self):
        if self._closed: raise WorkerUnavailableError("optional worker session is closed")
        if self._proc is not None: return
        snapshot = open_owned_model_tree_snapshot(self._asset.local_path, limits=self._asset_limits)
        try:
            if not hmac.compare_digest(snapshot.verify(), self._asset.sha256):
                raise AssetDigestMismatchError("owned model snapshot digest mismatch")
            model_fd = snapshot._root_fd
            if model_fd < 0 or not os.path.isdir(f"/proc/self/fd/{model_fd}"):
                raise LocalAssetError("owned model snapshot descriptor unavailable")
            session_asset = LocalModelAsset(
                f"/proc/self/fd/{model_fd}", self._asset.version, self._asset.sha256,
            )
            asset_paths = _asset_environment_paths(self._spec, session_asset)
            _require_worker_capabilities()
            prepared = _prepare_worker_descriptors(self._spec, self._limits)
        except BaseException:
            snapshot.close()
            raise
        self._snapshot, self._session_asset = snapshot, session_asset
        owned = list(prepared[:3])
        try:
            python_fd, root_fd, snapshot_fd = owned
            for path, fd, expected in ((self._spec.python_executable, python_fd, prepared[3]),
                                       (self._spec.trusted_code_root, root_fd, prepared[4])):
                reopened = _open_worker_path(path, directory=path == self._spec.trusted_code_root)
                try:
                    if (_worker_full_snapshot(os.fstat(fd)) != expected or
                            _worker_full_snapshot(os.fstat(reopened)) != expected or
                            os.path.realpath(f"/proc/self/fd/{reopened}") != path):
                        raise OSError("trusted path changed")
                except BaseException as exc: raise _close_worker_fd(reopened, exc)
                error = _close_worker_fd(reopened)
                if error: raise error
            argv = [self._spec.python_executable, "-I", "-B", "-c", _OPTIONAL_WORKER_BOOTSTRAP,
                    str(python_fd), str(root_fd), str(snapshot_fd), self._spec.python_executable,
                    self._spec.trusted_code_root, self._spec.worker_script, self._spec.worker_script_sha256,
                    str(self._limits.rlimit_cpu_seconds), str(self._limits.rlimit_nofile),
                    str(self._limits.rlimit_fsize_bytes), str(self._limits.rlimit_as_bytes),
                    str(self._limits.max_worker_script_bytes), "1"]
            proc = subprocess.Popen(argv, executable=f"/proc/self/fd/{python_fd}",
                pass_fds=(python_fd, root_fd, snapshot_fd, model_fd), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/", env=_worker_environment(self._spec, asset_paths),
                close_fds=True, start_new_session=True, restore_signals=True, bufsize=0)
            self._proc = proc
            self._process_group_id = proc.pid
            for fd in tuple(owned):
                error = _close_worker_fd(fd)
                owned.remove(fd)
                if error: raise error
        except BaseException as exc:
            primary = WorkerUnavailableError("optional worker session spawn failed") if isinstance(exc, (OSError, UnicodeError)) else exc
            if primary is not exc: primary.__cause__ = exc
            for fd in tuple(owned):
                owned.remove(fd); primary = _close_worker_fd(fd, primary)
            self.close()
            raise primary

    def _terminate(self):
        proc = self._proc
        process_group_id = self._process_group_id
        snapshot = self._snapshot
        self._proc = None
        self._process_group_id = None
        self._snapshot = None
        self._session_asset = None
        if proc is None:
            if snapshot is not None: snapshot.close()
            return
        primary = None
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try: stream.close()
            except OSError as exc:
                if primary is None: primary = WorkerCleanupError("worker session pipe cleanup failed"); primary.__cause__ = exc
                else: _add_worker_note(primary, "pipe cleanup: " + repr(exc))
        try: os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError: pass
        except OSError as exc:
            primary = WorkerCleanupError("worker session process-group cleanup failed"); primary.__cause__ = exc
        grace_deadline = time.monotonic() + self._limits.terminate_grace_ms / 1000.0
        while time.monotonic() < grace_deadline:
            if not _process_group_has_live_member(process_group_id): break
            time.sleep(min(.01, grace_deadline - time.monotonic()))
        if _process_group_has_live_member(process_group_id):
            try: os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError: pass
            except OSError as exc:
                if primary is None: primary = WorkerCleanupError("worker session process-group cleanup failed"); primary.__cause__ = exc
                else: _add_worker_note(primary, "KILL cleanup: " + repr(exc))
        try:
            proc.wait(timeout=self._limits.kill_wait_ms / 1000.0)
        except subprocess.TimeoutExpired as exc:
            if primary is None: primary = WorkerCleanupError("worker session child survived cleanup"); primary.__cause__ = exc
            else: _add_worker_note(primary, "final reap: " + repr(exc))
        except OSError as exc:
            if primary is None: primary = WorkerCleanupError("worker session child could not be reaped"); primary.__cause__ = exc
            else: _add_worker_note(primary, "final reap: " + repr(exc))
        if snapshot is not None:
            try: snapshot.close()
            except BaseException as exc:
                if primary is None: primary = exc
                else: _add_worker_note(primary, "model snapshot cleanup: " + repr(exc))
        if primary is not None: raise primary

    def close(self):
        if self._closed: return
        self._closed = True
        self._terminate()

    def call(self, operation: str, payload: dict) -> dict:
        self._start()
        proc = self._proc
        if proc is None: raise WorkerUnavailableError("optional worker session is unavailable")
        asset = self._session_asset
        if asset is None: raise WorkerUnavailableError("optional worker session asset is unavailable")
        request, encoded = _build_worker_request(
            self._spec, asset, operation, payload,
            nonce=secrets.token_bytes(32).hex(), limits=self._limits,
        )
        stdout, stderr, offset = bytearray(), bytearray(), 0
        deadline = time.monotonic() + self._limits.timeout_ms / 1000.0
        selector = selectors.DefaultSelector()
        primary = None
        try:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is None: raise WorkerUnavailableError("optional worker session pipe unavailable")
                os.set_blocking(stream.fileno(), False)
            selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
            selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
            selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
            while True:
                if proc.poll() is not None:
                    raise WorkerExitError("optional worker session exited before response")
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise WorkerTimeoutError("optional worker session timed out")
                for key, _ in selector.select(min(remaining, .1)):
                    name, fd = key.data, key.fileobj.fileno()
                    if name == "stdin":
                        try: written = os.write(fd, encoded[offset:offset + self._limits.io_chunk_bytes])
                        except (BlockingIOError, InterruptedError): continue
                        if written <= 0: raise WorkerUnavailableError("optional worker session input made no progress")
                        offset += written
                        if offset == len(encoded): selector.unregister(key.fileobj)
                        continue
                    try: chunk = os.read(fd, self._limits.io_chunk_bytes)
                    except (BlockingIOError, InterruptedError): continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if name == "stdout": raise WorkerExitError("optional worker session closed before response")
                        continue
                    buffer, cap = (stdout, self._limits.max_stdout_bytes) if name == "stdout" else (stderr, self._limits.max_stderr_bytes)
                    buffer.extend(chunk)
                    if len(buffer) > cap: raise WorkerOutputLimitError(f"worker {name} exceeds limit")
                    if name == "stdout" and b"\n" in stdout:
                        line, remainder = bytes(stdout).split(b"\n", 1)
                        if remainder: raise WorkerProtocolError("worker session response has extra text")
                        return _parse_worker_response(
                            line + b"\n", request=request, spec=self._spec,
                            asset=asset, limits=self._limits,
                        )
        except BaseException as exc:
            primary = exc
        finally:
            try: selector.close()
            except OSError as exc:
                if primary is None: primary = WorkerCleanupError("worker session selector cleanup failed"); primary.__cause__ = exc
                else: _add_worker_note(primary, "selector cleanup: " + repr(exc))
        if primary is not None:
            try: self.close()
            except BaseException as cleanup:
                _add_worker_note(primary, "session cleanup: " + repr(cleanup))
            raise primary
        raise WorkerUnavailableError("optional worker session ended without response")


def run_optional_worker(
    spec: OptionalWorkerSpec,
    asset: LocalModelAsset,
    operation: str,
    payload: dict,
    *,
    limits: WorkerLimits,
    asset_limits: AssetScanLimits = AssetScanLimits(),
) -> dict:
    if not isinstance(spec, OptionalWorkerSpec): _config_error("spec")
    if not isinstance(asset, LocalModelAsset): _config_error("asset")
    if not isinstance(limits, WorkerLimits): _config_error("limits")
    if not isinstance(asset_limits, AssetScanLimits): _config_error("asset_limits")
    verify_local_model_asset(asset, limits=asset_limits)
    asset_paths = _asset_environment_paths(spec, asset)
    _require_worker_capabilities()
    prepared = _prepare_worker_descriptors(spec, limits)
    owned, snapshots = list(prepared[:3]), prepared[3:]
    proc, primary = None, None
    try:
        nonce = secrets.token_bytes(32).hex()
        request, encoded = _build_worker_request(spec, asset, operation, payload, nonce=nonce, limits=limits)
        python_fd, root_fd, snapshot_fd = owned
        for path, fd, expected in ((spec.python_executable, python_fd, snapshots[0]),
                                   (spec.trusted_code_root, root_fd, snapshots[1])):
            reopened = _open_worker_path(path, directory=path == spec.trusted_code_root)
            try:
                if (_worker_full_snapshot(os.fstat(fd)) != expected or
                        _worker_full_snapshot(os.fstat(reopened)) != expected or
                        os.path.realpath(f"/proc/self/fd/{reopened}") != path):
                    raise OSError("trusted path changed")
            except BaseException as exc: raise _close_worker_fd(reopened, exc)
            error = _close_worker_fd(reopened)
            if error: raise error
        argv = [spec.python_executable, "-I", "-B", "-c", _OPTIONAL_WORKER_BOOTSTRAP,
                str(python_fd), str(root_fd), str(snapshot_fd), spec.python_executable,
                spec.trusted_code_root, spec.worker_script, spec.worker_script_sha256,
                str(limits.rlimit_cpu_seconds), str(limits.rlimit_nofile),
                str(limits.rlimit_fsize_bytes), str(limits.rlimit_as_bytes),
                str(limits.max_worker_script_bytes), "0"]
        start = time.monotonic()
        proc = subprocess.Popen(argv, executable=f"/proc/self/fd/{python_fd}",
            pass_fds=(python_fd, root_fd, snapshot_fd), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/", env=_worker_environment(spec, asset_paths),
            close_fds=True, start_new_session=True, restore_signals=True, bufsize=0)
        for fd in tuple(owned): primary = _close_worker_fd(fd, primary); owned.remove(fd)
        if primary is not None: _abort_worker_after_spawn(proc, limits, primary)
        raw = _run_worker_process(proc, encoded, limits, start)
        return _parse_worker_response(raw, request=request, spec=spec, asset=asset, limits=limits)
    except BaseException as exc:
        primary = exc
        if isinstance(exc, (OSError, UnicodeError)):
            primary = WorkerUnavailableError("optional worker spawn failed"); primary.__cause__ = exc
            for note in getattr(exc, "__notes__", ()): _add_worker_note(primary, note)
        for fd in tuple(owned):
            cleanup = _close_worker_fd(fd, primary); owned.remove(fd)
            if cleanup is not None: primary = cleanup
        raise primary
