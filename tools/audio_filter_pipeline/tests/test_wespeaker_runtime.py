"""Focused contracts for the isolated WeSpeaker optional runtime."""

import builtins
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import types
from dataclasses import replace

import pytest

from audio_filter_pipeline.processors import optional_runtime
from audio_filter_pipeline.processors.stages_v3 import wespeaker_runtime as runtime
from audio_filter_pipeline.processors.workers import wespeaker_worker as worker


pytestmark = pytest.mark.unit
_HEX = "a" * 64


def _request(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source-bytes")
    payload = {
        "source": {
            "local_path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size_bytes": source.stat().st_size, "sample_rate": 16_000, "frames": 12,
            "channels": 2, "start_sample": 2, "end_sample": 10,
        },
        "target_sample_rate": 16_000,
    }
    request = {
        "asset": {"local_path": str(tmp_path), "version": "model-v1", "sha256": _HEX},
        "nonce": "b" * 64, "operation": "extract_embedding_pcm_v1", "payload": payload,
        "protocol": "audio_filter_pipeline.optional_worker.v1", "request_id": "c" * 64,
        "runtime": {"name": "wespeaker", "version": "1.0.0"},
        "worker_script_sha256": "d" * 64,
    }
    return request, source


def _options(tmp_path):
    worker_path = os.path.abspath(worker.__file__)
    return runtime.WeSpeakerRuntimeOptions(
        model_path=str(tmp_path), model_version="model-v1", model_sha256=_HEX,
        runtime_version="1.0.0", python_executable=os.path.abspath(sys.executable),
        trusted_code_root=os.path.dirname(os.path.dirname(worker_path)), worker_script=worker_path,
        worker_script_sha256="b" * 64,
    )


def test_options_default_python_executable_resolves_symlinks(tmp_path, monkeypatch):
    interpreter_link = tmp_path / "python"
    interpreter_link.symlink_to(os.path.realpath(sys.executable))
    monkeypatch.setattr(runtime.sys, "executable", str(interpreter_link))

    options = runtime.WeSpeakerRuntimeOptions(
        model_path=str(tmp_path), model_version="model-v1", model_sha256=_HEX,
        runtime_version="1.0.0", worker_script_sha256="b" * 64,
    )

    assert options.python_executable == os.path.realpath(str(interpreter_link))
    runtime.WeSpeakerEmbeddingRuntime(options)


def test_runtime_session_keeps_configured_inner_python_executable(tmp_path, monkeypatch):
    captured = {}

    class Session:
        def __init__(self, spec, asset, *, limits, asset_limits):
            captured.update(spec=spec, asset=asset, limits=limits, asset_limits=asset_limits)

    options = _options(tmp_path)
    monkeypatch.setattr(runtime, "OptionalWorkerSession", Session)
    runtime.WeSpeakerEmbeddingRuntime(options).session()

    assert captured["spec"].python_executable == options.python_executable
    assert captured["asset"].local_path == options.model_path
    assert captured["limits"] == options.worker_limits
    assert captured["asset_limits"] == options.asset_limits


def test_optional_worker_session_serves_two_bound_requests_and_reaps(tmp_path):
    worker_script = tmp_path / "persistent_worker.py"
    worker_script.write_text(textwrap.dedent("""\
        import json
        import sys
        for line in sys.stdin.buffer:
            request = json.loads(line)
            response = {
                "asset": request["asset"],
                "nonce": request["nonce"],
                "protocol": request["protocol"],
                "request_id": request["request_id"],
                "result": {"asset_path": request["asset"]["local_path"], "echo": request["payload"]},
                "runtime": request["runtime"],
                "status": "ok",
                "worker_script_sha256": request["worker_script_sha256"],
            }
            sys.stdout.buffer.write(json.dumps(response, sort_keys=True, separators=(",", ":")).encode() + b"\\n")
            sys.stdout.buffer.flush()
    """), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    digest = optional_runtime.local_model_tree_sha256(str(model))
    spec = optional_runtime.OptionalWorkerSpec(
        os.path.realpath(sys.executable), str(tmp_path), str(worker_script),
        hashlib.sha256(worker_script.read_bytes()).hexdigest(), "wespeaker", "1.0.0",
    )
    limits = replace(runtime._DEFAULT_LIMITS, timeout_ms=2_000)

    session = optional_runtime.OptionalWorkerSession(
        spec, optional_runtime.LocalModelAsset(str(model), "model-v1", digest), limits=limits,
    )
    with session:
        first = session.call(
            "extract_embeddings_pcm_v1", {"sources": [], "target_sample_rate": 16_000},
        )
        second = session.call(
            "extract_embeddings_pcm_v1", {"sources": ["next"], "target_sample_rate": 16_000},
        )
        assert first["echo"] == {"sources": [], "target_sample_rate": 16_000}
        assert second["echo"] == {"sources": ["next"], "target_sample_rate": 16_000}
        assert first["asset_path"] == second["asset_path"]
        assert first["asset_path"].startswith("/proc/self/fd/")
        assert first["asset_path"] != str(model)
        process = session._proc
        snapshot = session._snapshot
        assert process is not None and process.poll() is None
        assert snapshot is not None and snapshot.verify() == digest

    assert process.poll() is not None
    with pytest.raises(optional_runtime.LocalAssetError, match="closed"):
        snapshot.verify()


def test_optional_worker_session_timeout_closes_and_reaps_process(tmp_path):
    worker_script = tmp_path / "hanging_persistent_worker.py"
    worker_script.write_text("import sys, time\nfor _line in sys.stdin.buffer: time.sleep(10)\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    spec = optional_runtime.OptionalWorkerSpec(
        os.path.realpath(sys.executable), str(tmp_path), str(worker_script),
        hashlib.sha256(worker_script.read_bytes()).hexdigest(), "wespeaker", "1.0.0",
    )
    session = optional_runtime.OptionalWorkerSession(
        spec,
        optional_runtime.LocalModelAsset(
            str(model), "model-v1", optional_runtime.local_model_tree_sha256(str(model)),
        ),
        limits=replace(runtime._DEFAULT_LIMITS, timeout_ms=50),
    )

    with pytest.raises(optional_runtime.WorkerTimeoutError):
        session.call("extract_embeddings_pcm_v1", {"sources": [], "target_sample_rate": 16_000})

    assert session._closed is True
    assert session._proc is None


def test_optional_worker_session_reaps_descendant_after_leader_exits(tmp_path):
    child_pid_path = tmp_path / "descendant.pid"
    worker_script = tmp_path / "orphaning_persistent_worker.py"
    worker_script.write_text(textwrap.dedent(f"""\
        import subprocess
        import sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        open({str(child_pid_path)!r}, "w", encoding="utf-8").write(str(child.pid))
        sys.stdin.buffer.readline()
    """), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    spec = optional_runtime.OptionalWorkerSpec(
        os.path.realpath(sys.executable), str(tmp_path), str(worker_script),
        hashlib.sha256(worker_script.read_bytes()).hexdigest(), "wespeaker", "1.0.0",
    )
    session = optional_runtime.OptionalWorkerSession(
        spec,
        optional_runtime.LocalModelAsset(
            str(model), "model-v1", optional_runtime.local_model_tree_sha256(str(model)),
        ),
        limits=replace(runtime._DEFAULT_LIMITS, timeout_ms=500),
    )

    with pytest.raises(optional_runtime.WorkerExitError):
        session.call("extract_embeddings_pcm_v1", {"sources": [], "target_sample_rate": 16_000})

    descendant_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


def test_runtime_session_preserves_body_error_when_snapshot_cleanup_fails(tmp_path, monkeypatch):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise optional_runtime.LocalAssetError("snapshot cleanup failed")

    adapter = runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path))
    session = runtime.WeSpeakerEmbeddingSession(adapter)
    monkeypatch.setattr(session, "_session", Session())

    with pytest.raises(ValueError, match="body failure") as raised:
        with session:
            raise ValueError("body failure")

    assert any("snapshot cleanup failed" in note for note in raised.value.__notes__)


def test_runtime_session_maps_primary_snapshot_cleanup_error(tmp_path, monkeypatch):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise optional_runtime.LocalAssetError("snapshot cleanup failed")

    adapter = runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path))
    session = runtime.WeSpeakerEmbeddingSession(adapter)
    monkeypatch.setattr(session, "_session", Session())

    with pytest.raises(runtime.WeSpeakerRuntimeUnavailableError) as raised:
        with session:
            pass

    assert isinstance(raised.value.__cause__, optional_runtime.LocalAssetError)


def test_gpu_binding_is_passed_only_as_worker_environment_override(tmp_path):
    options = runtime.WeSpeakerRuntimeOptions(**{**_options(tmp_path).__dict__, "cuda_visible_devices": "7"})
    adapter = runtime.WeSpeakerEmbeddingRuntime(options)

    assert adapter._spec.inherit_env == ()
    assert adapter._spec.env_overrides == (("CUDA_VISIBLE_DEVICES", "7"),)
    assert dict(adapter.provenance)["cuda_visible_devices"] == "7"


@pytest.mark.parametrize("value", ["", " 7", "7,8", "-1"])
def test_gpu_binding_rejects_nonphysical_single_gpu_selector(tmp_path, value):
    with pytest.raises(ValueError, match="cuda_visible_devices"):
        runtime.WeSpeakerRuntimeOptions(**{**_options(tmp_path).__dict__, "cuda_visible_devices": value})


def test_default_worker_limits_allow_wespeaker_batch_source_snapshots():
    """静态回归：32 个预打开 source fd 与 WeSpeaker 运行时依赖必须在 nofile 限制内。"""
    assert runtime._DEFAULT_LIMITS == optional_runtime.WorkerLimits(
        timeout_ms=30_000, terminate_grace_ms=500, kill_wait_ms=500,
        rlimit_cpu_seconds=60, rlimit_nofile=256,
        rlimit_fsize_bytes=128 * 1024 * 1024, rlimit_as_bytes=32 * 1024**3,
        max_request_bytes=1024 * 1024, max_stdout_bytes=4 * 1024 * 1024,
        max_stderr_bytes=256 * 1024, max_worker_script_bytes=1024 * 1024,
        max_json_depth=32, io_chunk_bytes=64 * 1024,
    )


def test_options_payload_roundtrips_nested_limits_through_json(tmp_path):
    options = replace(
        _options(tmp_path),
        worker_limits=optional_runtime.WorkerLimits(
            timeout_ms=1_234, terminate_grace_ms=234, kill_wait_ms=345,
            rlimit_cpu_seconds=12, rlimit_nofile=128, rlimit_fsize_bytes=4_096,
            rlimit_as_bytes=128 * 1024**2, max_request_bytes=2_048,
            max_stdout_bytes=8_192, max_stderr_bytes=4_096,
            max_worker_script_bytes=2_048, max_json_depth=8, io_chunk_bytes=1_024,
        ),
        asset_limits=optional_runtime.AssetScanLimits(
            max_entries=17, max_total_bytes=2**20, max_path_bytes=512,
            read_chunk_bytes=4_096, max_depth=9,
        ),
        cuda_visible_devices="7",
    )

    payload = options.to_payload()
    assert isinstance(payload["worker_limits"], dict)
    assert isinstance(payload["asset_limits"], dict)
    encoded = json.dumps(payload)
    restored = runtime.WeSpeakerRuntimeOptions.from_payload(json.loads(encoded))

    assert restored == options
    assert runtime.runtime_options_payload(options) == payload


def _row(tmp_path):
    return {
        "original_path": str(tmp_path / "source.wav"), "source_sha256": _HEX,
        "source_size_bytes": 10, "sample_rate": 16_000, "frames": 20, "channels": 2,
        "start_sample": 2, "end_sample": 18,
    }


def test_worker_and_parent_imports_do_not_load_optional_dependencies():
    assert worker.__all__ == ["handle_request", "main"]
    assert list(inspect.signature(worker.handle_request).parameters) == ["request", "installed_version"]
    code = (
        "import sys; before=set(sys.modules);"
        "import audio_filter_pipeline.processors.stages_v3.wespeaker_runtime;"
        "import audio_filter_pipeline.processors.workers.wespeaker_worker;"
        "bad={'torch','torchaudio','wespeaker','soundfile'};"
        "assert not any(name.split('.')[0] in bad for name in set(sys.modules)-before)"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, timeout=30,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )


def test_worker_rejects_payload_header_and_bounds_before_runtime_lookup(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    request["payload"]["source"]["end_sample"] = 13
    calls = []
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda _: calls.append(True) or "1.0.0")

    response = worker.handle_request(request)

    assert response["error"] == {"code": "invalid_request", "message": "Invalid request."}
    assert calls == []


@pytest.mark.parametrize("mutation", [
    lambda payload: payload["source"].__setitem__("local_path", "relative.wav"),
    lambda payload: payload["source"].__setitem__("channels", 0),
    lambda payload: payload.__setitem__("target_sample_rate", 0),
])
def test_worker_rejects_invalid_payload_values(tmp_path, mutation):
    request, _ = _request(tmp_path)
    mutation(request["payload"])
    assert worker.handle_request(request, installed_version="1.0.0")["error"]["code"] == "invalid_request"


def test_decode_resets_verified_fd_offset_before_opening_audio(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    source = request["payload"]["source"]
    observed = {}

    class Segment:
        shape = (8, 2)

    class Audio:
        samplerate = 16_000
        frames = 12
        channels = 2

        def __init__(self, stream):
            observed["offset_at_open"] = stream.tell()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def seek(self, offset):
            observed["seek"] = offset

        def read(self, frames, *, dtype, always_2d):
            observed["read"] = (frames, dtype, always_2d)
            return Segment()

    soundfile = types.ModuleType("soundfile")
    soundfile.SoundFile = Audio
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    fd, _ = worker._open_verified_source(source)
    try:
        assert os.lseek(fd, 0, os.SEEK_CUR) == source["size_bytes"]
        segment = worker._decode_exact_segment(fd, source)
    finally:
        os.close(fd)

    assert segment.shape == (8, 2)
    assert observed == {
        "offset_at_open": 0,
        "seek": 2,
        "read": (8, "float32", True),
    }


def test_worker_reads_exact_bounds_downmixes_equally_and_returns_embedding(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    observed = {}
    monkeypatch.setattr(worker, "_open_verified_source", lambda source: (42, ("baseline",)))
    monkeypatch.setattr(worker, "_verify_unchanged", lambda *args: None)
    monkeypatch.setattr(worker.os, "close", lambda fd: observed.setdefault("closed", fd))

    class Segment:
        shape = (8, 2)

    def decode(fd, source):
        observed["decoded"] = (fd, dict(source))
        return Segment()

    def resample(segment, source_rate, target_rate):
        observed["resample"] = (segment, source_rate, target_rate)
        return "pcm"

    def embed(path, pcm, sample_rate):
        observed["embedding"] = (path, pcm, sample_rate)
        return [0.25, -0.5]

    monkeypatch.setattr(worker, "_decode_exact_segment", decode)
    monkeypatch.setattr(worker, "_mono_and_resample", resample)
    monkeypatch.setattr(worker, "_embedding", embed)

    response = worker.handle_request(request, installed_version="1.0.0")

    assert response["status"] == "ok"
    assert response["result"] == {"schema": "audio_filter_pipeline.wespeaker_embedding.v1", "embedding": [0.25, -0.5]}
    assert observed["decoded"] == (42, request["payload"]["source"])
    assert observed["resample"][0].shape == (8, 2)
    assert observed["resample"][1:] == (16_000, 16_000)
    assert observed["embedding"] == (str(tmp_path), "pcm", 16_000)
    assert observed["closed"] == 42


@pytest.mark.parametrize("value", [[], [0.0, 0.0], [float("nan")], [[1.0]]])
def test_worker_rejects_empty_zero_nonfinite_or_multidimensional_embeddings(value):
    with pytest.raises(ValueError):
        worker._validate_embedding(value)


def test_worker_uses_official_api_with_private_temp_wav_and_cleans_up(tmp_path, monkeypatch):
    observed = {}

    class Model:
        def extract_embedding(self, path):
            observed["embedding_path"] = path
            observed["path_exists_during_extract"] = os.path.isfile(path)
            return [0.25, -0.5]

    def write(path, pcm, sample_rate, *, format, subtype):
        observed["write"] = (path, pcm, sample_rate, format, subtype)
        with open(path, "wb") as stream:
            stream.write(b"wav")

    wespeaker = types.ModuleType("wespeaker")
    wespeaker.load_model = lambda path: observed.setdefault("model_path", path) and Model()
    soundfile = types.ModuleType("soundfile")
    soundfile.write = write
    monkeypatch.setitem(sys.modules, "wespeaker", wespeaker)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(worker, "_require_nltk_data", lambda path: None)

    result = worker._embedding(str(tmp_path), "resampled-pcm", 16_000)

    assert result == [0.25, -0.5]
    assert observed["model_path"] == str(tmp_path)
    assert observed["write"][1:] == ("resampled-pcm", 16_000, "WAV", "FLOAT")
    assert observed["embedding_path"] == observed["write"][0]
    assert observed["embedding_path"].endswith("/segment.wav")
    assert observed["path_exists_during_extract"] is True
    assert not os.path.exists(observed["embedding_path"])


def test_worker_requires_g2p_en_nltk_zip_assets_without_download(tmp_path, monkeypatch):
    nltk_data = tmp_path / "nltk_data"
    nltk_data.mkdir()
    monkeypatch.setenv("NLTK_DATA", str(nltk_data))
    observed = {"paths": [], "downloads": []}
    nltk = types.ModuleType("nltk")
    nltk.data = types.SimpleNamespace(path=["ambient"], find=lambda path: observed["paths"].append(path))
    nltk.download = lambda *args, **kwargs: observed["downloads"].append((args, kwargs))
    monkeypatch.setitem(sys.modules, "nltk", nltk)

    worker._require_nltk_data(str(tmp_path))

    assert nltk.data.path == [str(nltk_data)]
    assert observed == {
        "paths": ["taggers/averaged_perceptron_tagger.zip", "corpora/cmudict.zip"],
        "downloads": [],
    }


def test_worker_missing_nltk_asset_fails_before_wespeaker_import(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    monkeypatch.setenv("NLTK_DATA", str(tmp_path / "nltk_data"))
    monkeypatch.setitem(sys.modules, "wespeaker", None)
    monkeypatch.setattr(worker, "_open_verified_source", lambda source: (42, ("baseline",)))
    monkeypatch.setattr(worker, "_decode_exact_segment", lambda *args: "segment")
    monkeypatch.setattr(worker, "_mono_and_resample", lambda *args: "pcm")
    monkeypatch.setattr(worker.os, "close", lambda fd: None)

    response = worker.handle_request(request, installed_version="1.0.0")

    assert response["error"] == {"code": "asset_missing", "message": "Required NLTK model asset is unavailable."}


def test_worker_batch_loads_once_extracts_in_input_order_and_writes_float_wav(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    first_source = dict(request["payload"]["source"])
    second_path = tmp_path / "second-source.wav"
    second_path.write_bytes(b"second-source-bytes")
    second_source = {
        **first_source,
        "local_path": str(second_path),
        "sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
        "size_bytes": second_path.stat().st_size,
    }
    request["operation"] = "extract_embeddings_pcm_v1"
    request["payload"] = {
        "sources": [first_source, second_source],
        "target_sample_rate": 16_000,
    }
    opened = iter([(41, ("first",)), (42, ("second",))])
    decoded, writes, loads, closed, source_events = [], [], [], [], []
    real_close = os.close

    def open_source(value):
        fd, baseline = next(opened)
        source_events.append(("open", value["local_path"], fd))
        return fd, baseline

    def close_source(fd):
        if fd in {41, 42}:
            closed.append(fd)
            source_events.append(("close", fd))
        else:
            real_close(fd)

    monkeypatch.setattr(worker, "_open_verified_source", open_source)
    monkeypatch.setattr(worker, "_decode_exact_segment", lambda fd, value: decoded.append(fd) or "segment")
    monkeypatch.setattr(worker, "_mono_and_resample", lambda *args: "pcm")
    monkeypatch.setattr(worker, "_verify_unchanged", lambda *args: None)
    monkeypatch.setattr(worker.os, "close", close_source)
    monkeypatch.setattr(worker, "_require_nltk_data", lambda path: None)

    class Model:
        def extract_embedding(self, path):
            return [float(len(writes)), 1.0]

    wespeaker = types.ModuleType("wespeaker")
    wespeaker.load_model = lambda path: loads.append(path) or Model()
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda path, pcm, rate, *, format, subtype: writes.append(
        (path, pcm, rate, format, subtype))
    monkeypatch.setitem(sys.modules, "wespeaker", wespeaker)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)

    response = worker.handle_request(request, installed_version="1.0.0")

    assert response["result"] == {
        "schema": "audio_filter_pipeline.wespeaker_embeddings.v1",
        "embeddings": [[1.0, 1.0], [2.0, 1.0]],
    }
    assert loads == [str(tmp_path)]
    assert decoded == [41, 42]
    assert [write[3:] for write in writes] == [("WAV", "FLOAT"), ("WAV", "FLOAT")]
    assert closed[-2:] == [41, 42]
    assert source_events == [
        ("open", first_source["local_path"], 41),
        ("close", 41),
        ("open", second_source["local_path"], 42),
        ("close", 42),
    ]


def test_worker_persistent_requests_reuse_one_model_and_preserve_source_checks(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    request["operation"] = "extract_embeddings_pcm_v1"
    request["payload"] = {"sources": [request["payload"]["source"]], "target_sample_rate": 16_000}
    loads, verified = [], []
    monkeypatch.setattr(worker, "_open_verified_source", lambda _source: (42, ("baseline",)))
    monkeypatch.setattr(worker, "_decode_exact_segment", lambda *_args: "segment")
    monkeypatch.setattr(worker, "_mono_and_resample", lambda *_args: "pcm")
    monkeypatch.setattr(worker, "_verify_unchanged", lambda *args: verified.append(args))
    monkeypatch.setattr(worker.os, "close", lambda _fd: None)
    model = object()
    monkeypatch.setattr(worker, "_load_model", lambda path: loads.append(path) or model)
    monkeypatch.setattr(worker, "_extract_embedding", lambda observed, *_args: [1.0] if observed is model else [])

    first, loaded, binding = worker._handle_request(request, installed_version="1.0.0")
    second, loaded_again, binding_again = worker._handle_request(
        request, installed_version="1.0.0", model=loaded, session_binding=binding,
    )

    assert first["status"] == second["status"] == "ok"
    assert loads == [str(tmp_path)]
    assert loaded_again is model
    assert binding_again == binding
    assert len(verified) == 2


def test_worker_batch_later_source_failure_closes_each_distinct_identity_once(
    tmp_path, monkeypatch,
):
    request, _ = _request(tmp_path)
    first_source = dict(request["payload"]["source"])
    second_path = tmp_path / "second-source.wav"
    second_path.write_bytes(b"second-source-bytes")
    second_source = {
        **first_source,
        "local_path": str(second_path),
        "sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
        "size_bytes": second_path.stat().st_size,
    }
    request["operation"] = "extract_embeddings_pcm_v1"
    request["payload"] = {
        "sources": [first_source, second_source],
        "target_sample_rate": 16_000,
    }
    descriptors = iter((41, 42))
    events = []

    def open_source(value):
        descriptor = next(descriptors)
        events.append(("open", value["local_path"], descriptor))
        return descriptor, (value["local_path"],)

    def decode(descriptor, _source):
        events.append(("decode", descriptor))
        if descriptor == 42:
            raise worker._SourceMismatch
        return "segment"

    monkeypatch.setattr(worker, "_open_verified_source", open_source)
    monkeypatch.setattr(worker, "_decode_exact_segment", decode)
    monkeypatch.setattr(worker, "_mono_and_resample", lambda *args: "pcm")
    monkeypatch.setattr(worker, "_verify_unchanged", lambda *args: None)
    monkeypatch.setattr(worker.os, "close", lambda fd: events.append(("close", fd)))
    monkeypatch.setattr(worker, "_load_model", lambda _path: object())
    monkeypatch.setattr(worker, "_extract_embedding", lambda *_args: [1.0])

    response = worker.handle_request(request, installed_version="1.0.0")

    assert response["error"] == {
        "code": "asset_digest_mismatch",
        "message": "Source asset verification failed.",
    }
    assert events == [
        ("open", first_source["local_path"], 41),
        ("decode", 41),
        ("close", 41),
        ("open", second_source["local_path"], 42),
        ("decode", 42),
        ("close", 42),
    ]


@pytest.mark.parametrize(("operation", "result_key", "schema"), [
    ("extract_embedding_pcm_v1", "embedding", "audio_filter_pipeline.wespeaker_embedding.v1"),
    ("extract_embeddings_pcm_v1", "embeddings", "audio_filter_pipeline.wespeaker_embeddings.v1"),
])
def test_worker_discards_wespeaker_stdout_for_single_and_batch_requests(
        tmp_path, monkeypatch, capsys, operation, result_key, schema):
    request, _ = _request(tmp_path)
    if operation == "extract_embeddings_pcm_v1":
        source = dict(request["payload"]["source"])
        request["operation"] = operation
        request["payload"] = {"sources": [source, dict(source)], "target_sample_rate": 16_000}
        opened = iter([(41, ("first",)), (42, ("second",))])
    else:
        opened = iter([(41, ("single",))])
    monkeypatch.setattr(worker, "_open_verified_source", lambda value: next(opened))
    monkeypatch.setattr(worker, "_decode_exact_segment", lambda *args: "segment")
    monkeypatch.setattr(worker, "_mono_and_resample", lambda *args: "pcm")
    monkeypatch.setattr(worker, "_verify_unchanged", lambda *args: None)
    monkeypatch.setattr(worker.os, "close", lambda fd: None)
    monkeypatch.setattr(worker, "_require_nltk_data", lambda path: None)

    class Model:
        def extract_embedding(self, _path):
            print("extract stdout")
            return [0.25, -0.5]

    wespeaker = types.ModuleType("wespeaker")

    def load_model(_path):
        print("load stdout")
        return Model()

    wespeaker.load_model = load_model
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda path, *_args, **_kwargs: open(path, "wb").close()
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    original_import = builtins.__import__

    def import_wespeaker(name, *args, **kwargs):
        if name == "wespeaker":
            print("import stdout")
            return wespeaker
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_wespeaker)

    response = worker.handle_request(request, installed_version="1.0.0")

    assert response["status"] == "ok"
    assert response["result"]["schema"] == schema
    assert response["result"][result_key] == ([0.25, -0.5] if result_key == "embedding"
                                               else [[0.25, -0.5], [0.25, -0.5]])
    assert capsys.readouterr().out == ""


def test_worker_rejects_invalid_batch_source_before_runtime_lookup(tmp_path, monkeypatch):
    request, _ = _request(tmp_path)
    request["operation"] = "extract_embeddings_pcm_v1"
    request["payload"] = {"sources": [request["payload"]["source"], {}], "target_sample_rate": 16_000}
    calls = []
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda _: calls.append(True) or "1.0.0")

    response = worker.handle_request(request)

    assert response["error"]["code"] == "invalid_request"
    assert calls == []


def test_worker_cleans_private_temp_wav_when_extraction_fails(tmp_path, monkeypatch):
    observed = {}

    class Model:
        def extract_embedding(self, path):
            observed["embedding_path"] = path
            raise RuntimeError("extract failed")

    def write(path, _pcm, _sample_rate, *, format, subtype):
        assert (format, subtype) == ("WAV", "FLOAT")
        with open(path, "wb") as stream:
            stream.write(b"wav")

    wespeaker = types.ModuleType("wespeaker")
    wespeaker.load_model = lambda path: observed.setdefault("model_path", path) and Model()
    soundfile = types.ModuleType("soundfile")
    soundfile.write = write
    monkeypatch.setitem(sys.modules, "wespeaker", wespeaker)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(worker, "_require_nltk_data", lambda path: None)

    with pytest.raises(RuntimeError, match="extract failed"):
        worker._embedding(str(tmp_path), "resampled-pcm", 16_000)

    assert observed["model_path"] == str(tmp_path)
    assert not os.path.exists(observed["embedding_path"])


def test_worker_source_binds_official_wespeaker_api_only():
    source = inspect.getsource(worker)
    assert "from wespeaker.cli.speaker import Speaker" not in source
    assert "Speaker(" not in source
    assert "import wespeaker" in source
    assert "wespeaker.load_model(asset_path)" in source
    assert "model.extract_embedding(wav_path)" in source


def test_worker_requires_exact_installed_version(tmp_path):
    request, _ = _request(tmp_path)
    response = worker.handle_request(request, installed_version="1.0.1")
    assert response["error"]["code"] == "runtime_version_mismatch"
    assert response["runtime"] == {"name": "wespeaker", "version": "1.0.1"}


def test_parent_batch_builds_strict_payload_maps_results_and_exposes_provenance(tmp_path, monkeypatch):
    captured = {}

    def invoke(spec, asset, operation, payload, *, limits, asset_limits):
        captured.update(operation=operation, payload=payload)
        return {"schema": "audio_filter_pipeline.wespeaker_embeddings.v1", "embeddings": [[1, -2.5], [3, 4]]}

    adapter = runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path))
    monkeypatch.setattr(runtime, "run_optional_worker", invoke)
    result = adapter.extract_batch([_row(tmp_path), _row(tmp_path)])

    assert result == ((1.0, -2.5), (3.0, 4.0))
    assert captured["operation"] == "extract_embeddings_pcm_v1"
    assert captured["payload"] == {"sources": [_source_payload_for_test(tmp_path)] * 2, "target_sample_rate": 16_000}
    assert dict(adapter.provenance) == {
        "capability": "wespeaker_optional_worker", "runtime": "wespeaker", "runtime_version": "1.0.0",
        "model_version": "model-v1", "model_sha256": _HEX, "target_sample_rate": 16_000,
    }
    with pytest.raises(TypeError):
        adapter.provenance["runtime"] = "other"


@pytest.mark.parametrize(("count", "batch_sizes"), [(33, [32, 1]), (64, [32, 32])])
def test_parent_batch_limits_worker_sources_and_preserves_input_order(tmp_path, monkeypatch, count, batch_sizes):
    calls = []

    def invoke(_spec, _asset, operation, payload, *, limits, asset_limits):
        assert operation == "extract_embeddings_pcm_v1"
        source_indexes = [int(source["local_path"].rsplit("-", 1)[1].removesuffix(".wav"))
                          for source in payload["sources"]]
        calls.append(source_indexes)
        return {
            "schema": "audio_filter_pipeline.wespeaker_embeddings.v1",
            "embeddings": [[index + 1.0] for index in source_indexes],
        }

    rows = []
    for index in range(count):
        row = _row(tmp_path)
        row["original_path"] = str(tmp_path / f"source-{index}.wav")
        rows.append(row)
    monkeypatch.setattr(runtime, "run_optional_worker", invoke)

    result = runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path)).extract_batch(rows)

    assert [len(call) for call in calls] == batch_sizes
    assert calls == [list(range(start, min(start + 32, count))) for start in range(0, count, 32)]
    assert result == tuple((index + 1.0,) for index in range(count))


def _source_payload_for_test(tmp_path):
    return {
        "local_path": str(tmp_path / "source.wav"), "sha256": _HEX, "size_bytes": 10,
        "sample_rate": 16_000, "frames": 20, "channels": 2, "start_sample": 2, "end_sample": 18,
    }


def test_parent_builds_strict_payload_and_validates_response(tmp_path, monkeypatch):
    captured = {}

    def invoke(spec, asset, operation, payload, *, limits, asset_limits):
        captured.update(spec=spec, asset=asset, operation=operation, payload=payload)
        return {"schema": "audio_filter_pipeline.wespeaker_embedding.v1", "embedding": [1, -2.5]}

    monkeypatch.setattr(runtime, "run_optional_worker", invoke)
    result = runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path))(_row(tmp_path))

    assert result == (1.0, -2.5)
    assert captured["operation"] == "extract_embedding_pcm_v1"
    assert captured["payload"] == {"source": {
        "local_path": str(tmp_path / "source.wav"), "sha256": _HEX, "size_bytes": 10,
        "sample_rate": 16_000, "frames": 20, "channels": 2, "start_sample": 2, "end_sample": 18,
    }, "target_sample_rate": 16_000}


@pytest.mark.parametrize(("raised", "error"), [
    (optional_runtime.AssetDigestMismatchError("bad"), runtime.WeSpeakerRuntimeAssetError),
    (optional_runtime.WorkerRemoteAssetError("asset_missing", "bad"), runtime.WeSpeakerRuntimeAssetError),
    (optional_runtime.WorkerUnavailableError("bad"), runtime.WeSpeakerRuntimeUnavailableError),
    (optional_runtime.WorkerTimeoutError("bad"), runtime.WeSpeakerRuntimeUnavailableError),
    (optional_runtime.WorkerProtocolError("bad"), runtime.WeSpeakerRuntimeRemoteError),
    (optional_runtime.WorkerRuntimeVersionMismatchError("runtime_version_mismatch", "bad"), runtime.WeSpeakerRuntimeVersionError),
    (optional_runtime.WorkerRemoteError("runtime_error", "bad"), runtime.WeSpeakerRuntimeRemoteError),
])
def test_parent_maps_optional_worker_failures(tmp_path, monkeypatch, raised, error):
    monkeypatch.setattr(runtime, "run_optional_worker", lambda *args, **kwargs: (_ for _ in ()).throw(raised))
    with pytest.raises(error):
        runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path))(_row(tmp_path))


@pytest.mark.parametrize("result", [
    {}, {"schema": "wrong", "embedding": [1.0]},
    {"schema": "audio_filter_pipeline.wespeaker_embedding.v1", "embedding": [0.0]},
])
def test_parent_rejects_worker_result_without_fallback(tmp_path, monkeypatch, result):
    monkeypatch.setattr(runtime, "run_optional_worker", lambda *args, **kwargs: result)
    with pytest.raises(runtime.WeSpeakerRuntimeRemoteError):
        runtime.WeSpeakerEmbeddingRuntime(_options(tmp_path))(_row(tmp_path))


def test_bounded_subprocess_drains_pipes_and_keeps_only_stderr_tail():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; data=sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(b'out:' + data); "
            "sys.stderr.buffer.write(b'0123456789abcdef')",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    stdout, stderr = runtime.run_bounded_subprocess(
        process,
        b"payload",
        timeout_seconds=2.0,
        stdout_limit_bytes=64,
        stderr_tail_bytes=5,
    )

    assert stdout == b"out:payload"
    assert stderr == b"bcdef"
    assert process.returncode == 0


def test_bounded_subprocess_terminates_process_group_on_stdout_limit():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 65536)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    with pytest.raises(runtime.BoundedSubprocessOutputLimitError) as raised:
        runtime.run_bounded_subprocess(
            process,
            timeout_seconds=2.0,
            stdout_limit_bytes=32,
            stderr_tail_bytes=8,
        )

    assert raised.value.stream == "stdout"
    assert len(raised.value.stdout) == 33
    assert process.returncode is not None


def test_bounded_subprocess_timeout_reaps_child_and_closes_pipes():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    with pytest.raises(runtime.BoundedSubprocessTimeoutError):
        runtime.run_bounded_subprocess(
            process,
            timeout_seconds=0.05,
            stdout_limit_bytes=32,
            stderr_tail_bytes=8,
        )

    assert process.returncode is not None
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed


def test_terminate_process_group_fails_if_child_survives_final_wait():
    signals = []

    class StubbornProcess:
        pid = 4321
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("stubborn-child", timeout)

    process = StubbornProcess()

    with pytest.raises(subprocess.TimeoutExpired, match="stubborn-child"):
        runtime.terminate_process_group(
            process,
            killpg=lambda pid, sig: signals.append((pid, sig)),
            terminate_wait_seconds=0.01,
            kill_wait_seconds=0.01,
        )

    assert signals == [
        (process.pid, runtime.signal.SIGTERM),
        (process.pid, runtime.signal.SIGKILL),
    ]
    assert process.poll() is None
