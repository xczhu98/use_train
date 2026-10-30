"""TSE 多 cue 文件推理与 artifact 绑定合同。

覆盖：

- 模型 IO 合同声明的 cue role 解析（speaker waveform/embedding、spatial static/track）；
- 文件 cue 与 inline cue JSON 的读取/校验，以及未声明 role、缺 role 的显式拒绝；
- UID/artifact v6 与 manifest v4 对全部 cue 文件、inline cue JSON 与 speaker 前端
  （checkpoint/fbank/preprocess）身份的版本化绑定；
- 公共 runner 的 reuse 在 cue 文件变化后被拒绝；
- 评测流水线能构造新 role，缺真实资源时给出清晰失败；
- streaming/ONNX 对新模态保持显式 unsupported。

真实资源仅在 tmp_path 内生成小模型与短音频，不做长训练或外部资源依赖。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from core.inference import (
    FileInferenceItem,
    InferenceSpec,
    ReusePolicy,
    run_inference,
)
from core.utils.inference_artifacts import stat_audio_identity
from core.models.build import build_model
from tasks.tse.inference import TSEExtractor, TSEInferenceAdapter
from tasks.tse.inference import artifacts as tse_artifacts
from tasks.tse.inference.adapter import cue_declaration
from tasks.tse.inference.extractor import (
    STREAMING_SUPPORTED_CUE_ROLES,
    declared_cue_roles,
    verify_model_checkpoint_paths,
)
from tasks.tse.inference.sidecar import publish_sidecars_from_report
from tasks.tse.models import register_tse_models

import evaluate.eval_pipeline_tse as pipeline

pytestmark = pytest.mark.unit

_SR = 16000


# ---------------------------------------------------------------------------
# 真实小模型与真实短音频 fixture
# ---------------------------------------------------------------------------
def _modular_cfg(*, spatial: str | None = "static") -> dict:
    """CPU 可跑的小尺寸 TSE_Modular 配置（speaker waveform + 可选 spatial）。"""
    cues: dict = {"speaker": {"representation": "waveform", "features": {"spkemb": {}}}}
    args: dict = {
        "backbone": "bsrnn",
        "separator": {"feature_dim": 8, "num_repeat": 2, "band_width": [16, 17]},
        "cues": cues,
        "composition": {},
        "sample_rate": _SR,
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
    }
    if spatial is not None:
        cues["spatial"] = {"representation": spatial, "features": {"ipd": {}, "sdf": {}}}
        args.update(
            n_channel=2,
            ref_channel=1,
            mic_positions_m=[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
            speed_of_sound_mps=340.0,
        )
    return {
        "stft": {"n_fft": 64, "hop_length": 16, "win_length": 64, "window": "hann"},
        "model": {"type": "TSE_Modular", "args": args},
        "data": {"sample_rate": _SR},
    }


def _save_ckpt(path: Path, cfg: dict) -> Path:
    register_tse_models(only="TSE_Modular")
    torch.manual_seed(0)
    model = build_model(cfg["model"], stft_cfg=cfg["stft"]).eval()
    torch.save({"model": model.state_dict(), "resolved_config": cfg}, path)
    return path


def _build_model(cfg: dict):
    register_tse_models(only="TSE_Modular")
    torch.manual_seed(0)
    return build_model(cfg["model"], stft_cfg=cfg["stft"]).eval()


def _write_mixture(path: Path, *, channels: int = 1, samples: int = _SR) -> Path:
    time = np.arange(samples) / _SR
    tone = 0.2 * np.sin(2 * np.pi * 220 * time)
    if channels == 1:
        sf.write(path, tone.astype(np.float32), _SR)
    else:
        stack = np.stack([tone, 0.2 * np.sin(2 * np.pi * 240 * time)])
        sf.write(path, stack.T.astype(np.float32), _SR)
    return path


def _write_enrollment(path: Path, *, samples: int = _SR) -> Path:
    time = np.arange(samples) / _SR
    sf.write(path, (0.3 * np.sin(2 * np.pi * 180 * time)).astype(np.float32), _SR)
    return path


def _write_static_spatial(path: Path, azimuth: float = 0.3, elevation: float = 0.1) -> Path:
    value = np.zeros(1, dtype=[("azimuth", np.float64), ("elevation", np.float64)])
    value["azimuth"][0] = azimuth
    value["elevation"][0] = elevation
    np.save(path, value)
    return path


def _write_track_spatial(path: Path, anchors=(0, 8000)) -> Path:
    value = np.zeros(
        2,
        dtype=[
            ("azimuth", np.float64),
            ("elevation", np.float64),
            ("sample_anchor", np.int64),
        ],
    )
    value["azimuth"] = [0.1, 0.4]
    value["elevation"] = [0.0, 0.2]
    value["sample_anchor"] = list(anchors)
    np.save(path, value)
    return path


def _extractor(tmp_path: Path, *, spatial: str | None = "static") -> TSEExtractor:
    cfg = _modular_cfg(spatial=spatial)
    return TSEExtractor(str(_save_ckpt(tmp_path / "model.tar", cfg)), device="cpu")


def _cue_item(
    tmp_path: Path,
    extractor: TSEExtractor,
    *,
    spatial: str | None = "static",
    include_spatial: bool = True,
    inline_spatial: bool = False,
    uid: str = "utt",
) -> FileInferenceItem:
    channels = 2 if spatial is not None else 1
    mixture = _write_mixture(tmp_path / "mix.wav", channels=channels)
    enrollment = _write_enrollment(tmp_path / "enroll.wav")
    inputs: dict = {"mixture": mixture, "speaker_waveform": enrollment}
    attributes: dict = {"dataset": "fixture"}
    if include_spatial:
        if inline_spatial:
            attributes["cues"] = {"spatial": {"azimuth": 0.3, "elevation": 0.1}}
        else:
            writer = _write_track_spatial if spatial == "track" else _write_static_spatial
            inputs["spatial"] = writer(tmp_path / f"spatial_{spatial}.npy")
    return FileInferenceItem(uid, inputs, tmp_path / f"{uid}.wav", attributes=attributes)


# ---------------------------------------------------------------------------
# 模型合同：role / representation / 每行维度
# ---------------------------------------------------------------------------
def test_declared_cue_roles_follow_model_contract(tmp_path: Path):
    static = _build_model(_modular_cfg(spatial="static"))
    roles = declared_cue_roles(static)
    assert roles["speaker_waveform"].representation == "waveform"
    assert roles["speaker_waveform"].row_rank == 1
    assert roles["spatial"].representation == "static"
    assert roles["spatial"].row_rank == 1
    assert roles["spatial"].required is True

    track = _build_model(_modular_cfg(spatial="track"))
    track_roles = declared_cue_roles(track)
    assert track_roles["spatial"].representation == "track"
    assert track_roles["spatial"].row_rank == 2
    assert "spatial_timebase" in track_roles["spatial"].fields


def test_embedding_representation_is_declared_as_embedding_role(tmp_path: Path):
    cfg = _modular_cfg(spatial=None)
    cfg["model"]["args"]["cues"] = {
        "speaker": {
            "representation": "embedding",
            "features": {"spkemb": {}},
            "encoder": {"embedding_dim": 8},
        }
    }
    roles = declared_cue_roles(_build_model(cfg))
    assert set(roles) == {"speaker_embedding"}
    assert roles["speaker_embedding"].representation == "embedding"
    assert roles["speaker_embedding"].row_rank == 1


# ---------------------------------------------------------------------------
# 文件 cue / inline cue 读取与显式拒绝
# ---------------------------------------------------------------------------
def test_legacy_enrollment_input_key_is_rejected(tmp_path: Path):
    """enrollment/enroll 别名已移除：旧输入键必须在 adapter 边界显式拒绝。"""
    extractor = _extractor(tmp_path, spatial=None)
    adapter = TSEInferenceAdapter(extractor)
    mixture = _write_mixture(tmp_path / "mix.wav")
    enrollment = _write_enrollment(tmp_path / "enroll.wav")

    with pytest.raises(ValueError, match="unknown TSE inference input role"):
        adapter.infer(
            FileInferenceItem(
                "a", {"mixture": mixture, "enrollment": enrollment}, tmp_path / "a.wav"
            )
        )


def test_extractor_without_declared_cue_roles_is_rejected(tmp_path: Path):
    """extractor 未声明 declared_cue_roles 时在 adapter 构造期失败（无旧入口）。"""

    class _LegacyExtractor:
        def extract_speech(self, mixture, enrollment):
            return np.zeros(160, dtype=np.float32)

        def extract_speech_with_cues(self, mixture, sr, cues):
            return np.zeros(160, dtype=np.float32)

    with pytest.raises(TypeError, match="declared_cue_roles"):
        TSEInferenceAdapter(_LegacyExtractor())


def test_adapter_reads_file_cues_and_binds_declaration(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)

    files, declaration = adapter.prepare_item(item)
    assert set(files) == {"speaker_waveform", "spatial"}
    assert declaration == {
        "speaker_waveform": {"kind": "file", "path": str(files["speaker_waveform"])},
        "spatial": {"kind": "file", "path": str(files["spatial"])},
    }
    output = adapter.infer(item)
    output.writer(tmp_path / "utt.wav")
    waveform, sample_rate = sf.read(tmp_path / "utt.wav")
    assert sample_rate == _SR
    assert waveform.ndim == 1 and np.isfinite(waveform).all()


def test_adapter_reads_track_spatial_npy(tmp_path: Path):
    extractor = _extractor(tmp_path, spatial="track")
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor, spatial="track")
    output = adapter.infer(item)
    output.writer(tmp_path / "track.wav")
    assert sf.info(tmp_path / "track.wav").frames == _SR


def test_static_spatial_npy_tolerates_optional_distance_field(tmp_path: Path):
    """MC-LibriMix 风格 NPY 含可选 distance 列：容忍解码但不进入条件。"""
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    value = np.zeros(
        1,
        dtype=[
            ("azimuth", np.float32),
            ("elevation", np.float32),
            ("distance", np.float32),
        ],
    )
    value["azimuth"][0] = 3.955119
    value["elevation"][0] = -0.12695357
    value["distance"][0] = 1.7784786
    np.save(tmp_path / "spatial_static.npy", value)
    item = FileInferenceItem(
        "dist",
        {
            "mixture": _write_mixture(tmp_path / "mix.wav", channels=2),
            "speaker_waveform": _write_enrollment(tmp_path / "enroll.wav"),
            "spatial": tmp_path / "spatial_static.npy",
        },
        tmp_path / "dist.wav",
    )
    output = adapter.infer(item)
    output.writer(tmp_path / "dist.wav")
    waveform, sample_rate = sf.read(tmp_path / "dist.wav")
    assert sample_rate == _SR
    assert waveform.ndim == 1 and np.isfinite(waveform).all()


def test_track_spatial_requires_structured_anchor_field(tmp_path: Path):
    extractor = _extractor(tmp_path, spatial="track")
    adapter = TSEInferenceAdapter(extractor)
    np.save(tmp_path / "plain.npy", np.zeros((2, 2), dtype=np.float64))
    item = FileInferenceItem(
        "bad",
        {
            "mixture": _write_mixture(tmp_path / "mix.wav", channels=2),
            "speaker_waveform": _write_enrollment(tmp_path / "enroll.wav"),
            "spatial": tmp_path / "plain.npy",
        },
        tmp_path / "bad.wav",
    )
    with pytest.raises(ValueError, match="sample_anchor"):
        adapter.infer(item)


def test_inline_cue_json_is_supported_for_declared_roles(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor, inline_spatial=True)
    files, declaration = adapter.prepare_item(item)
    assert set(files) == {"speaker_waveform"}
    assert declaration["spatial"]["kind"] == "inline"
    output = adapter.infer(item)
    output.writer(tmp_path / "inline.wav")
    assert sf.info(tmp_path / "inline.wav").frames == _SR


def test_missing_model_declared_role_is_rejected(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    mixture = _write_mixture(tmp_path / "mix.wav", channels=2)
    item = FileInferenceItem(
        "no-spatial", {"mixture": mixture}, tmp_path / "no-spatial.wav"
    )
    with pytest.raises(ValueError, match="missing model-declared cue roles.*spatial"):
        adapter.infer(item)


def test_unknown_input_role_is_rejected(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)
    bad = FileInferenceItem(
        "bad",
        {**dict(item.inputs), "visual": tmp_path / "spatial_static.npy"},
        tmp_path / "bad.wav",
    )
    with pytest.raises(ValueError, match="does not consume cue roles"):
        adapter.infer(bad)


def test_unknown_inline_role_is_rejected(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)
    bad = FileInferenceItem(
        "bad",
        dict(item.inputs),
        tmp_path / "bad.wav",
        attributes={"cues": {"unknown": {"value": 1}}},
    )
    with pytest.raises(ValueError, match="inline TSE cue role is not supported"):
        adapter.infer(bad)


def test_cue_representation_mismatch_is_rejected(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor, inline_spatial=True)
    item = FileInferenceItem(
        "bad",
        dict(item.inputs),
        tmp_path / "bad.wav",
        attributes={"cues": {"spatial": {"azimuth": [0.1, 0.2], "elevation": [0.0, 0.1], "sample_anchor": [0, 8000]}}},
    )
    with pytest.raises(ValueError, match="does not match model contract"):
        adapter.infer(item)


# ---------------------------------------------------------------------------
# 尚未被注册模型声明的模态：读取/校验边界单独覆盖
# ---------------------------------------------------------------------------
def test_visual_features_reader_decodes_structured_npy(tmp_path: Path):
    """visual features npy 必须同时携带 features 与秒级 timebase。"""
    from tasks.tse.inference.adapter import _decode_visual, _visual_cue

    dtype = np.dtype([("features", np.float64, (3,)), ("timebase", np.float64)])
    value = np.zeros(2, dtype=dtype)
    value["features"] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    value["timebase"] = [0.0, 0.02]
    path = tmp_path / "visual.npy"
    np.save(path, value)

    features, timebase = _decode_visual(np.load(path, allow_pickle=False))
    assert tuple(features.shape) == (2, 3)
    assert timebase.tolist() == [0.0, 0.02]

    spec = SimpleNamespace(representation="features")
    cue = _visual_cue(features, timebase, spec)
    assert cue.role == "visual" and cue.value.shape == (1, 2, 3)
    assert cue.length == 2 and cue.timebase.shape == (1, 2)

    plain = tmp_path / "plain.npy"
    np.save(plain, np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="structured npy"):
        _decode_visual(np.load(plain, allow_pickle=False))


def test_raw_video_reader_decodes_frames_and_timebase(tmp_path: Path):
    """raw video cue 由可选 torchvision.io 后端解码；缺失时显式失败。"""
    from tasks.tse.inference.adapter import _decode_video, _visual_cue

    av = pytest.importorskip("av", reason="raw video 编码需要可选 PyAV")
    path = tmp_path / "clip.mp4"
    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 8, 8, "yuv420p"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((8, 8, 3), index * 40, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as exc:  # pragma: no cover - 编码器缺失
        pytest.skip(f"raw video 编码不可用: {exc}")

    with path.open("rb") as handle:
        import io

        frames, timebase = _decode_video(io.BytesIO(handle.read()))
    assert frames.dtype == torch.uint8
    assert frames.shape[0] == 3 and frames.shape[1] == 3
    assert timebase.tolist() == pytest.approx([0.0, 0.1, 0.2])

    spec = SimpleNamespace(representation="raw")
    cue = _visual_cue(frames, timebase, spec)
    assert cue.role == "visual" and cue.length == 3


def test_textual_and_embedding_readers_build_typed_cues(tmp_path: Path):
    """textual phoneme JSON 与 speaker embedding 只做读取/校验，不猜测模型消费方式。"""
    from tasks.tse.inference.adapter import (
        _embedding_cue,
        _inline_cue,
        _textual_payload,
    )

    embedding = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    cue = _embedding_cue(
        embedding, SimpleNamespace(representation="embedding")
    )
    assert cue.role == "speaker_embedding" and cue.value.shape == (1, 8)

    textual = _inline_cue(
        "textual",
        SimpleNamespace(representation="phoneme_ids"),
        _textual_payload({"phoneme_ids": [3, 1, 4]}),
    )
    assert textual.role == "textual" and textual.value.dtype == torch.int64
    assert textual.value.tolist() == [[3, 1, 4]] and textual.length == 3

    with pytest.raises(ValueError, match="requires a 'phoneme_ids' key"):
        _textual_payload({"ids": [1]})
    with pytest.raises(ValueError, match="must be integers"):
        _inline_cue(
            "textual",
            SimpleNamespace(representation="phoneme_ids"),
            {"phoneme_ids": [1.5, 2]},
        )


# ---------------------------------------------------------------------------
# UID / artifact / manifest 版本化绑定
# ---------------------------------------------------------------------------
def test_uid_binds_cue_files_and_inline_cue_json(tmp_path: Path):
    spatial = _write_static_spatial(tmp_path / "spatial.npy")
    base = {"dataset": "ds", "mixture_utterance": "m1"}
    file_identity = tse_artifacts.make_uid(
        {**base, "cues": {"spatial": {"kind": "file", "path": str(spatial)}}}
    )
    other_file = tse_artifacts.make_uid(
        {
            **base,
            "cues": {"spatial": {"kind": "file", "path": str(tmp_path / "other.npy")}},
        }
    )
    inline = tse_artifacts.make_uid(
        {**base, "cues": {"spatial": {"kind": "inline", "value": {"azimuth": 0.3, "elevation": 0.1}}}}
    )
    inline_changed = tse_artifacts.make_uid(
        {**base, "cues": {"spatial": {"kind": "inline", "value": {"azimuth": 0.4, "elevation": 0.1}}}}
    )
    legacy = tse_artifacts.make_uid(
        {**base, "enrolment_speakers_utterance": "e1"}
    )

    assert len({file_identity, other_file, inline, inline_changed, legacy}) == 5
    assert tse_artifacts.UID_SCHEME.endswith("cues-v2-128")
    assert tse_artifacts.row_identity({**base, "cues": {"spatial": {"kind": "file", "path": str(spatial)}}})[
        "cues"
    ]["spatial"]["kind"] == "file"


def test_artifact_meta_binds_cue_files_inline_json_and_frontend(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)
    files, declaration = adapter.prepare_item(item)
    snapshot = tse_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": Path(item.inputs["mixture"]), **files}, cues=declaration
    )
    tse_artifacts.verify_artifact_meta(expected)

    assert expected["artifacts"]["cues"]["schema"] == tse_artifacts.CUE_BINDING_SCHEMA
    assert expected["artifacts"]["cues"]["roles"] == {
        "speaker_waveform": "file",
        "spatial": "file",
    }
    frontend = expected["artifacts"]["speaker_frontend"]
    assert frontend["schema"] == tse_artifacts.SPEAKER_FRONTEND_SCHEMA
    assert frontend["preprocess"]["algorithm"] == "sha256-json-v1"

    files["spatial"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="输入在输出发布前发生变化"):
        tse_artifacts.verify_artifact_meta(expected)


def test_inline_cue_binding_detects_tampered_value(tmp_path: Path):
    declaration = {"spatial": {"kind": "inline", "value": {"azimuth": 0.3}}}
    mixture = _write_mixture(tmp_path / "mix.wav")
    base = tse_artifacts._artifact_meta({"inputs": {}, "cues": None})
    expected = tse_artifacts.bind_input_artifacts(
        base, {"mixture": mixture}, cues=declaration
    )
    binding = expected["artifacts"]["cues"]
    inputs = expected["artifacts"]["inputs"]
    assert binding["roles"] == {"spatial": "inline"}
    tse_artifacts.verify_cue_binding(binding, inputs)

    binding["inline"]["spatial"]["value"] = {"azimuth": 0.9}
    with pytest.raises(ValueError, match="does not bind its value"):
        tse_artifacts.verify_cue_binding(binding, inputs)


def test_file_cue_must_be_bound_in_inputs(tmp_path: Path):
    inputs = {"mixture": tse_artifacts.stat_audio_artifact(_write_mixture(tmp_path / "mix.wav"))}
    binding = {"schema": tse_artifacts.CUE_BINDING_SCHEMA, "roles": {"spatial": "file"}, "inline": {}}
    with pytest.raises(ValueError, match="not bound in inputs"):
        tse_artifacts.verify_cue_binding(binding, inputs)


def test_manifest_v4_binds_cue_identity(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)
    files, declaration = adapter.prepare_item(item)
    snapshot = tse_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": Path(item.inputs["mixture"]), **files}, cues=declaration
    )
    output = tmp_path / "wav" / "utt.wav"
    output.parent.mkdir()
    output.write_bytes(b"wav")
    sidecar = output.with_name("utt.meta.json")
    sidecar.write_text(
        json.dumps({**expected, "output": stat_audio_identity(output)}), encoding="utf-8"
    )
    row = {
        "dataset": "ds",
        "mixture_utterance": "m1",
        "cues": {
            "spatial": {"kind": "file", "path": str(files["spatial"])},
            "speaker_waveform": {"kind": "file", "path": str(files["speaker_waveform"])},
        },
        "inference_status": "ok",
        "tse_path": str(output),
        "_inference_artifacts": expected,
    }
    manifest = tmp_path / "manifest.json"
    payload = tse_artifacts.write_inference_manifest([row], manifest, artifact_meta=snapshot.meta)
    assert payload["schema"] == "tse-inference-manifest-v4"
    assert set(payload["items"][0]["identity"]) == {"dataset", "mixture", "cues"}
    assert payload["items"][0]["artifacts"]["cues"] == expected["artifacts"]["cues"]
    assert tse_artifacts.verify_inference_manifest(manifest, artifact_meta=snapshot.meta) == payload

    payload["items"][0]["identity"]["cues"]["spatial"]["path"] = "/elsewhere.npy"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="UID does not bind item identity"):
        tse_artifacts.verify_inference_manifest(manifest, artifact_meta=snapshot.meta)


def test_speaker_frontend_identity_tracks_fbank_and_preprocess(tmp_path: Path):
    cfg = _modular_cfg(spatial=None)
    base = tse_artifacts.speaker_frontend_identity(cfg)
    assert base["schema"] == tse_artifacts.SPEAKER_FRONTEND_SCHEMA
    assert base["checkpoint"] is None

    changed = json.loads(json.dumps(cfg))
    changed["stft"]["n_fft"] = 128
    assert tse_artifacts.speaker_frontend_identity(changed)["preprocess"] != base["preprocess"]

    with_fbank = json.loads(json.dumps(cfg))
    with_fbank["model"]["args"]["cues"]["speaker"]["encoder"] = {
        "fbank": {"num_mel_bins": 40}
    }
    assert tse_artifacts.speaker_frontend_identity(with_fbank)["fbank"] != base["fbank"]
    assert base["fbank"] is None


def test_frontend_checkpoint_mismatch_fails_closed(tmp_path: Path):
    """模型自持 ``*_ckpt_path`` 必须与配置派生的前端身份一致。"""
    declared = tmp_path / "speaker.pt"
    declared.write_bytes(b"speaker-v1")
    other = tmp_path / "other.pt"
    other.write_bytes(b"speaker-v2")
    cfg = _modular_cfg(spatial=None)
    cfg["model"]["args"]["cues"]["speaker"]["encoder"] = {"checkpoint": str(declared)}
    frontend = tse_artifacts.speaker_frontend_identity(cfg)
    assert frontend["checkpoint"]["path"] == str(declared.resolve())

    def _model(path: Path):
        class _Model:
            speaker_encoder_ckpt_path = str(path)

        return _Model()

    matched = SimpleNamespace(
        _speaker_frontend=frontend,
        _model_checkpoint_paths=verify_model_checkpoint_paths(_model(declared)),
    )
    TSEExtractor._verify_frontend_checkpoint_identity(matched)

    mismatched = SimpleNamespace(
        _speaker_frontend=frontend,
        _model_checkpoint_paths=verify_model_checkpoint_paths(_model(other)),
    )
    with pytest.raises(RuntimeError, match="不一致"):
        TSEExtractor._verify_frontend_checkpoint_identity(mismatched)

    unbound = SimpleNamespace(
        _speaker_frontend=None,
        _model_checkpoint_paths=verify_model_checkpoint_paths(_model(other)),
    )
    with pytest.raises(RuntimeError, match="未被配置绑定"):
        TSEExtractor._verify_frontend_checkpoint_identity(unbound)


# ---------------------------------------------------------------------------
# 公共 runner 的 reuse 拒绝
# ---------------------------------------------------------------------------
def test_public_runner_rejects_reuse_after_cue_file_change(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)
    files, declaration = adapter.prepare_item(item)
    snapshot = tse_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": Path(item.inputs["mixture"]), **files}, cues=declaration
    )
    wav_dir = tmp_path

    def _run(decl):
        files_now, _ = adapter.prepare_item(item)
        current = snapshot.bind_inputs(
            {"mixture": Path(item.inputs["mixture"]), **files_now}, cues=decl
        )
        spec = InferenceSpec(
            config=tmp_path / "model.tar",
            checkpoint=tmp_path / "model.tar",
            output_dir=wav_dir,
            items=[item],
            artifacts=None,
            reuse=ReusePolicy(enabled=True),
        )
        report = run_inference(
            spec, adapter, artifact_fingerprint=_source_binding(snapshot)
        )
        return report, current

    first_report, first_expected = _run(declaration)
    assert [result.status for result in first_report.results] == ["ok"]
    publish_sidecars_from_report(first_report, {item.uid: first_expected}, output_dir=wav_dir)

    second_report, _ = _run(declaration)
    assert [result.status for result in second_report.results] == ["skipped"]

    _write_static_spatial(files["spatial"], azimuth=0.9)
    third_report, third_expected = _run(declaration)
    assert [result.status for result in third_report.results] == ["ok"]
    assert third_expected != first_expected


def _source_binding(snapshot) -> dict:
    """把 TSE artifact meta 交给公共 runner 的 source binding。"""
    from evaluate.eval_pipeline_tse import _public_inference_artifact_binding

    return _public_inference_artifact_binding(snapshot, {})


def test_sidecar_publication_requires_cue_binding(tmp_path: Path):
    extractor = _extractor(tmp_path)
    adapter = TSEInferenceAdapter(extractor)
    item = _cue_item(tmp_path, extractor)
    files, declaration = adapter.prepare_item(item)
    snapshot = tse_artifacts.TSEInferenceArtifactSnapshot.from_loaded_extractor(
        extractor, {"device": "cpu", "apply_vad": False}
    )
    expected = snapshot.bind_inputs(
        {"mixture": Path(item.inputs["mixture"]), **files}, cues=declaration
    )
    spec = InferenceSpec(
        config=tmp_path / "model.tar",
        checkpoint=tmp_path / "model.tar",
        output_dir=tmp_path,
        items=[item],
        artifacts=None,
        reuse=ReusePolicy(enabled=False),
    )
    report = run_inference(
        spec, adapter, artifact_fingerprint=_source_binding(snapshot)
    )
    publish_sidecars_from_report(report, {item.uid: expected}, output_dir=tmp_path)
    output = Path(report.results[0].output_path)
    sidecar = json.loads(output.with_name(output.stem + ".meta.json").read_text("utf-8"))
    assert sidecar["artifacts"]["cues"]["roles"]["spatial"] == "file"

    sidecar["artifacts"]["cues"]["roles"]["spatial"] = "inline"
    sidecar["fingerprint"] = tse_artifacts.fingerprint_json(sidecar["artifacts"])
    tampered = output.with_name(output.stem + ".meta.json")
    tampered.write_text(json.dumps(sidecar), encoding="utf-8")
    assert not snapshot.matches_sidecar(
        tse_artifacts.load_sidecar(tampered), expected, output
    )


# ---------------------------------------------------------------------------
# 评测流水线：新 role 的构造与缺资源失败
# ---------------------------------------------------------------------------
def _eval_args(tmp_path: Path, ckpt: Path, cue_root: Path) -> SimpleNamespace:
    meta = tmp_path / "fixture_meta.csv"
    meta.write_text(
        "mixture_utterance,enrolment_speakers_utterance,spatial_cue\n"
        "mix,enroll,spatial.npy\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        metrics="",
        device="cpu",
        output_dir=str(tmp_path / "eval"),
        meta=[str(meta)],
        max_samples=None,
        mapping_csv=None,
        model=str(ckpt),
        config=None,
        apply_vad=False,
        skip_inference=False,
        audio_root=str(tmp_path),
        cue_root=str(cue_root),
        ref_root=None,
        allow_partial=False,
    )


def test_eval_row_cue_inputs_resolve_declared_roles(tmp_path: Path):
    cue_root = tmp_path / "cues"
    cue_root.mkdir()
    spatial = _write_static_spatial(cue_root / "spatial.npy")
    _write_enrollment(tmp_path / "enroll.wav")
    args = SimpleNamespace(cue_root=str(cue_root), audio_root=str(tmp_path))
    roles = declared_cue_roles(_build_model(_modular_cfg(spatial="static")))
    row = {
        "enrolment_speakers_utterance": "enroll",
        "spatial_cue": "spatial.npy",
    }
    files, declaration = pipeline._row_cue_inputs(row, roles, args, {})
    assert files == {"speaker_waveform": tmp_path / "enroll.wav", "spatial": spatial}
    assert declaration["spatial"] == {"kind": "file", "path": str(spatial)}
    assert pipeline._uid_cue_identity(declaration) == {
        "spatial": {"kind": "file", "path": str(spatial)}
    }


def test_eval_row_cue_inputs_report_missing_resource(tmp_path: Path):
    _write_enrollment(tmp_path / "enroll.wav")
    args = SimpleNamespace(cue_root=None, audio_root=str(tmp_path))
    roles = declared_cue_roles(_build_model(_modular_cfg(spatial="static")))
    row = {"enrolment_speakers_utterance": "enroll", "spatial_cue": None}
    with pytest.raises(ValueError, match="meta 未提供模型声明的 cue 资源: spatial"):
        pipeline._row_cue_inputs(row, roles, args, {})


def test_eval_row_cue_inputs_report_unresolvable_file(tmp_path: Path):
    _write_enrollment(tmp_path / "enroll.wav")
    args = SimpleNamespace(cue_root=str(tmp_path), audio_root=str(tmp_path))
    roles = declared_cue_roles(_build_model(_modular_cfg(spatial="static")))
    row = {
        "enrolment_speakers_utterance": "enroll",
        "spatial_cue": "missing_spatial.npy",
    }
    with pytest.raises(FileNotFoundError, match="无法解析 cue 资源路径"):
        pipeline._row_cue_inputs(row, roles, args, {})


def test_eval_public_inference_constructs_cue_aware_items(tmp_path: Path, monkeypatch):
    cfg = _modular_cfg(spatial="static")
    ckpt = _save_ckpt(tmp_path / "model.tar", cfg)
    cue_root = tmp_path / "cues"
    cue_root.mkdir()
    _write_static_spatial(cue_root / "spatial.npy")
    _write_mixture(tmp_path / "mix.wav", channels=2)
    _write_enrollment(tmp_path / "enroll.wav")
    args = _eval_args(tmp_path, ckpt, cue_root)
    mapping = {
        "mix": str(tmp_path / "mix.wav"),
        "enroll": str(tmp_path / "enroll.wav"),
    }
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "_load_mapping", lambda _path: mapping)

    assert pipeline.main() == 0
    rows = pipeline._read_meta(args.meta)
    assert rows[0]["spatial_cue"] == "spatial.npy"
    manifest = json.loads(
        (Path(args.output_dir) / "tse_inference_manifest.json").read_text("utf-8")
    )
    assert manifest["ok"] == 1
    assert set(manifest["items"][0]["identity"]) == {"dataset", "mixture", "cues"}
    wav = Path(args.output_dir) / "wav" / f"{manifest['ok_uids'][0]}.wav"
    assert wav.is_file()


def test_eval_public_inference_fails_clearly_without_cue_resource(tmp_path: Path, monkeypatch):
    cfg = _modular_cfg(spatial="static")
    ckpt = _save_ckpt(tmp_path / "model.tar", cfg)
    cue_root = tmp_path / "cues"
    cue_root.mkdir()
    _write_mixture(tmp_path / "mix.wav", channels=2)
    _write_enrollment(tmp_path / "enroll.wav")
    args = _eval_args(tmp_path, ckpt, cue_root)
    args.meta = []
    meta = tmp_path / "no_cue_meta.csv"
    meta.write_text(
        "mixture_utterance,enrolment_speakers_utterance\nmix,enroll\n", encoding="utf-8"
    )
    args.meta = [str(meta)]
    mapping = {
        "mix": str(tmp_path / "mix.wav"),
        "enroll": str(tmp_path / "enroll.wav"),
    }
    monkeypatch.setattr(pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(pipeline, "_load_mapping", lambda _path: mapping)

    with pytest.raises(ValueError, match="meta 未提供模型声明的 cue 资源"):
        pipeline.main()


# ---------------------------------------------------------------------------
# streaming / ONNX 对新模态保持显式 unsupported
# ---------------------------------------------------------------------------
def test_streaming_and_onnx_do_not_claim_cue_modalities(tmp_path: Path):
    """cue-aware 模型不进入 streaming/ONNX 发布路径。

    ``TSE_Modular`` 明确 ``supports_streaming=False`` 且不提供
    ``forward_streaming``/``initialize_cache``，因此 ``onnx/export.py`` 的
    streaming 分支会显式报错；这里锁定该边界，防止把新模态假接进部署路径。
    """
    model = _build_model(_modular_cfg(spatial="static"))
    assert model.supports_streaming is False
    assert not hasattr(model, "forward_streaming")
    assert not hasattr(model, "initialize_cache")
    assert STREAMING_SUPPORTED_CUE_ROLES == frozenset(
        {"speaker_waveform", "speaker_embedding"}
    )
    assert "spatial" not in STREAMING_SUPPORTED_CUE_ROLES
