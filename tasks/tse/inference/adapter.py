"""TSE 文件级公共 inference adapter（cue-aware）。

Adapter 只消费 ``FileInferenceItem``：把 ``inputs`` 里的 mixture 与模型声明的
cue role 文件读成 typed cue，把 ``attributes['cues']`` 里的 inline cue JSON 纳入
同一条绑定路径，再交给 ``TSEExtractor`` 完成模型 I/O。它不处理 manifest、output
lock、``skip-existing`` 或 sidecar 发布——那些由 ``core.inference`` 公共 runner 与
``tasks.tse.inference.sidecar`` 负责。

支持的 cue role 与来源格式：

==================  ==========================  ==============================
role                文件                        inline JSON
==================  ==========================  ==============================
speaker_waveform    任意 torchaudio 音频        —
speaker_embedding   ``.npy`` 1-D float          ``{"values": [...]}``
spatial             ``.npy`` structured         static ``{"azimuth":..,"elevation":..}``
                    ``{azimuth,elevation[,      track ``{"azimuth":[..],"elevation":[..],
                    sample_anchor][,distance]}``  "sample_anchor":[..]}``
visual              ``.npy`` structured         ``{"features":[[..]],"timebase":[..]}``
                    ``{features,timebase}``
                    或 ``.mp4`` raw video
textual             ``.json``                   ``{"phoneme_ids": [..]}``
==================  ==========================  ==============================

spatial cue 的方向角约定与 ``tasks/tse/data/spatial_cues.py::azimuth_elevation_from_positions``
同源（TSE 唯一定义）：``azimuth = atan2(y, x)`` 弧度（数学惯例，array-local
坐标系），``elevation`` 从水平面起算、向上为正（弧度）。SSL 任务内部使用从
+Z 起算的极角口径（见 ``tasks/ssl/data/projection.py::_doa`` docstring），
两者相差 ``π/2 - 极角``，跨任务消费时必须显式换算，禁止直接互换。
可选 ``distance`` 字段（米）仅作容忍解码（与 MC-LibriMix 等含距离 NPY 资产对齐），
当前模型只消费 azimuth/elevation（track 另需 sample_anchor），distance 不进入条件。

``enrollment`` / ``enroll`` 兼容别名已移除：所有 cue 必须按模型 IO 合同声明的
canonical role 提供；extractor 未声明 ``declared_cue_roles`` 时在 adapter 构造期
显式失败，不存在 mixture+enrollment 旧入口。

streaming 与 ONNX 目前只覆盖 speaker 条件（``extractor.STREAMING_SUPPORTED_CUE_ROLES``）；
新模态不进入这两条发布路径。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from core.inference import FileInferenceItem, PreparedOutput
from tasks.tse.inference.artifacts import load_bound_payload
from tasks.tse.inference.extractor import TSECueInput

# role 别名（enrollment/enroll/spk_embedding）已随 _infer_legacy 一并移除；
# 所有 cue 输入必须使用模型 IO 合同声明的 canonical role。
_FILE_CUE_ROLES = frozenset(
    {"speaker_waveform", "speaker_embedding", "spatial", "visual", "textual"}
)
_INLINE_CUE_ROLES = frozenset(
    {"speaker_embedding", "spatial", "visual", "textual"}
)
_SPATIAL_FIELDS = frozenset({"azimuth", "elevation", "sample_anchor", "distance"})
_VISUAL_FIELDS = frozenset({"features", "timebase"})


def _numpy_loader(stream: Any) -> Any:
    return np.load(stream, allow_pickle=False)


def _json_loader(stream: Any) -> Any:
    try:
        return json.loads(stream.read().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"TSE textual cue is not valid UTF-8 JSON: {exc}") from exc


def _float_tensor(value: Any, name: str) -> Any:
    import torch

    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor-like") from exc
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN/Inf")
    return tensor.contiguous()


def _payload_sequence(payload: Mapping[str, Any], key: str, role: str) -> Any:
    if not isinstance(payload, Mapping):
        raise TypeError(f"inline cue {role!r} must be a mapping")
    if key not in payload:
        raise ValueError(f"inline cue {role!r} requires key {key!r}")
    value = payload[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"inline cue {role!r} key {key!r} must be a list")
    return value


def _decode_spatial(value: Any, representation: str) -> tuple[Any, Any | None]:
    """把 spatial npy 解码成 ``[2,P]`` 方向与可选 sample anchors。"""
    import torch

    if not isinstance(value, np.ndarray) or value.size < 1:
        raise ValueError("spatial cue must be a non-empty ndarray")
    if value.dtype.names is not None:
        unknown = sorted(set(value.dtype.names) - _SPATIAL_FIELDS)
        if unknown:
            raise ValueError(f"spatial cue has unknown fields: {unknown}")
        missing = sorted({"azimuth", "elevation"} - set(value.dtype.names))
        if missing:
            raise ValueError(f"spatial cue is missing fields: {missing}")
        if value.ndim != 1:
            raise ValueError(f"structured spatial cue must have shape [P], got {value.shape}")
        azimuth = np.asarray(value["azimuth"]).reshape(-1)
        elevation = np.asarray(value["elevation"]).reshape(-1)
        anchors = (
            np.asarray(value["sample_anchor"]).reshape(-1)
            if "sample_anchor" in value.dtype.names
            else None
        )
    else:
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("spatial cue must use a numeric dtype")
        matrix = np.asarray(value)
        if representation == "static" and matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != 2:
            raise ValueError(
                "spatial cue must have shape [P,2] in azimuth/elevation order, "
                f"got {value.shape}"
            )
        azimuth = matrix[:, 0]
        elevation = matrix[:, 1]
        anchors = None
    if not np.issubdtype(azimuth.dtype, np.number) or not np.issubdtype(
        elevation.dtype, np.number
    ):
        raise TypeError("spatial angles must be numeric radians")
    azimuth = azimuth.astype(np.float64)
    elevation = elevation.astype(np.float64)
    if not np.isfinite(azimuth).all() or not np.isfinite(elevation).all():
        raise ValueError("spatial angles must contain only finite radians")
    if representation == "static":
        if azimuth.size != 1:
            raise ValueError("static spatial cue must contain exactly one point")
        return (
            torch.tensor([float(azimuth[0]), float(elevation[0])], dtype=torch.float32),
            None,
        )
    if anchors is None:
        raise ValueError(
            "track spatial cue requires a structured npy with a sample_anchor field"
        )
    if not np.issubdtype(anchors.dtype, np.number):
        raise TypeError("spatial sample_anchor must be numeric")
    numeric = anchors.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("spatial sample_anchor must contain finite integer samples")
    if int(numeric[0]) != 0:
        raise ValueError("spatial track anchors must start at source sample 0")
    if numeric.size > 1 and bool((numeric[1:] <= numeric[:-1]).any()):
        raise ValueError("spatial track anchors must be strictly increasing")
    direction = torch.tensor(
        np.stack((azimuth, elevation)).astype(np.float32), dtype=torch.float32
    )
    return direction, torch.from_numpy(numeric.astype(np.int64))


def _decode_visual(value: Any) -> tuple[Any, Any]:
    """把 visual features npy 解码成 ``[F,D]`` 特征与 ``[F]`` 秒级 timebase。"""
    import torch

    if not isinstance(value, np.ndarray) or value.size < 1:
        raise ValueError("visual features cue must be a non-empty ndarray")
    if value.dtype.names is None:
        raise ValueError(
            "visual features cue must be a structured npy with "
            f"{sorted(_VISUAL_FIELDS)} fields"
        )
    unknown = sorted(set(value.dtype.names) - _VISUAL_FIELDS)
    if unknown:
        raise ValueError(f"visual cue has unknown fields: {unknown}")
    missing = sorted(_VISUAL_FIELDS - set(value.dtype.names))
    if missing:
        raise ValueError(f"visual cue is missing fields: {missing}")
    if value.ndim != 1:
        raise ValueError(f"structured visual cue must have shape [F], got {value.shape}")
    features = np.stack([np.asarray(item, dtype=np.float64) for item in value["features"]])
    timebase = np.asarray(value["timebase"], dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != timebase.shape[0]:
        raise ValueError("visual features and timebase must share one F axis")
    if not np.isfinite(features).all() or not np.isfinite(timebase).all():
        raise ValueError("visual features cue must contain only finite values")
    return (
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(timebase),
    )


def _decode_video(stream: Any) -> tuple[Any, Any]:
    """解码 raw video；缺失可选后端时显式失败，不做静默降级。"""
    import torch

    try:
        from torchvision.io import read_video
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "raw video cues require the optional torchvision.io backend"
        ) from exc
    try:
        frames, _, info = read_video(stream, pts_unit="sec", output_format="TCHW")
    except Exception as exc:  # pragma: no cover - 依赖具体解码后端
        raise RuntimeError(f"raw video cue cannot be decoded: {exc}") from exc
    if frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError(f"raw video cue must decode to [T,3,H,W], got {tuple(frames.shape)}")
    fps = float(info.get("video_fps") or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("raw video cue must report a positive video_fps")
    timebase = torch.arange(frames.shape[0], dtype=torch.float64) / fps
    return frames.to(torch.uint8), timebase


class TSEInferenceAdapter:
    """把 ``TSEExtractor`` 接入新的文件推理 Seam。

    Adapter 只消费 ``FileInferenceItem`` 并返回待发布内容；最终 output 路径由
    ``FileInferenceItem.output`` 和公共 runner 决定，Adapter 不参与 WAV reuse、
    manifest 或 sidecar 发布。
    """

    def __init__(self, extractor: Any) -> None:
        if not callable(getattr(extractor, "extract_speech_with_cues", None)):
            raise TypeError(
                "extractor must provide extract_speech_with_cues(mixture, sr, cues)"
            )
        if not callable(getattr(extractor, "declared_cue_roles", None)):
            raise TypeError(
                "extractor must implement declared_cue_roles(); "
                "the legacy mixture+enrollment entry has been removed"
            )
        self.extractor = extractor

    def declared_cue_roles(self) -> dict[str, Any]:
        """返回模型声明的 cue role；extractor 未声明时显式失败（无旧入口）。"""
        resolver = getattr(self.extractor, "declared_cue_roles", None)
        if not callable(resolver):
            raise TypeError(
                "extractor must implement declared_cue_roles(); "
                "the legacy mixture+enrollment entry has been removed"
            )
        roles = resolver()
        if not isinstance(roles, Mapping):
            raise TypeError("extractor.declared_cue_roles() must return a mapping")
        return dict(roles)

    def prepare_item(
        self, item: FileInferenceItem
    ) -> tuple[dict[str, Path], dict[str, Any] | None]:
        """解析 item 的 cue 声明，返回 ``(role→文件路径, canonical cue 声明)``。

        只做角色解析与校验，不读取 payload；评估流水线用它构造 expected artifact
        metadata，``infer`` 也复用同一解析结果。
        """
        roles = self.declared_cue_roles()
        mixture = item.inputs.get("mixture")
        if mixture is None:
            raise ValueError("TSE inference item requires a mixture path")
        files: dict[str, Path] = {}
        for raw_role, value in item.inputs.items():
            role = raw_role
            if role == "mixture":
                continue
            if role not in _FILE_CUE_ROLES:
                raise ValueError(
                    f"unknown TSE inference input role: {raw_role!r}; "
                    f"supported roles are mixture and {sorted(_FILE_CUE_ROLES)}"
                )
            if value is None:
                raise ValueError(f"TSE inference role {raw_role!r} must be a file path")
            if role in files and Path(value) != files[role]:
                raise ValueError(
                    f"TSE inference role {role!r} is declared twice with different paths"
                )
            files[role] = Path(value)
        inline = self._inline_declarations(item.attributes)
        unknown = sorted(set(files) - set(roles))
        unknown += sorted(set(inline) - set(roles))
        if unknown:
            raise ValueError(
                f"model.IO_CONTRACT does not consume cue roles: {sorted(set(unknown))}"
            )
        missing = sorted(
            role
            for role, spec in roles.items()
            if spec.required and role not in files and role not in inline
        )
        if missing:
            raise ValueError(
                "TSE inference item is missing model-declared cue roles: "
                f"{missing}; provide the role file in inputs or inline JSON in "
                "attributes['cues']"
            )
        return files, cue_declaration(files, inline)

    @staticmethod
    def _inline_declarations(attributes: Mapping[str, Any]) -> dict[str, Any]:
        """读取 ``attributes['cues']`` 的 inline cue JSON。"""
        raw = attributes.get("cues")
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError("attributes['cues'] must be a mapping of role to payload")
        inline: dict[str, Any] = {}
        for role, payload in raw.items():
            if role not in _INLINE_CUE_ROLES:
                raise ValueError(f"inline TSE cue role is not supported: {role!r}")
            if not isinstance(payload, Mapping):
                raise TypeError(f"inline TSE cue {role!r} must be a mapping")
            inline[role] = payload
        return inline

    def infer(self, item: FileInferenceItem) -> PreparedOutput:
        """按模型声明的 cue role 执行文件推理。"""
        mixture = item.inputs.get("mixture")
        if mixture is None:
            raise ValueError("TSE inference item requires a mixture path")
        roles = self.declared_cue_roles()
        waveform = self._infer_cues(item, roles)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError("TSE extractor must return finite mono waveform")
        sample_rate = int(getattr(self.extractor, "sample_rate", 16000))
        if sample_rate <= 0:
            raise ValueError("TSE extractor sample_rate must be positive")

        def writer(path: Path) -> None:
            try:
                import torchaudio
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("TSE file inference requires torchaudio") from exc
            import torch

            torchaudio.save(str(path), torch.from_numpy(waveform).unsqueeze(0), sample_rate)

        return PreparedOutput(item.uid, writer)

    def _infer_cues(self, item: FileInferenceItem, roles: Mapping[str, Any]) -> np.ndarray:
        """读取全部 cue 文件与 inline JSON，交给 extractor 的 cue 边界推理。"""
        import torchaudio

        files, _ = self.prepare_item(item)
        inline = self._inline_declarations(item.attributes)
        mixture_path = Path(item.inputs["mixture"])
        pcm_mix, sr_mix = load_bound_payload(
            mixture_path, torchaudio.load, label="mixture"
        )[0]

        cues: dict[str, Any] = {}
        for role, path in sorted(files.items()):
            cues[role] = self._read_cue_file(role, path, roles[role])
        for role, payload in sorted(inline.items()):
            cues[role] = _inline_cue(role, roles[role], payload)
        return np.asarray(
            self.extractor.extract_speech_with_cues(pcm_mix, sr_mix, cues),
            dtype=np.float32,
        )

    def _read_cue_file(self, role: str, path: Path, spec: Any) -> Any:
        """读取单个 cue 文件并转成 typed cue；不做 manifest/sidecar 判定。"""
        label = f"TSE cue {role}"
        if role == "speaker_waveform":
            import torchaudio

            pcm, sr = load_bound_payload(path, torchaudio.load, label=label)[0]
            return self.extractor.prepare_speaker_waveform_cue(pcm, sr)
        if role == "textual":
            payload = load_bound_payload(path, _json_loader, label=label)[0]
            return _inline_cue(role, spec, _textual_payload(payload))
        if role == "visual" and path.suffix.lower() == ".mp4":
            frames, timebase = load_bound_payload(
                path, _decode_video, label=label
            )[0]
            return _visual_cue(frames, timebase, spec)
        if role in {"speaker_embedding", "spatial"}:
            value = load_bound_payload(path, _numpy_loader, label=label)[0]
            if role == "speaker_embedding":
                return _embedding_cue(value, spec)
            return _spatial_cue(value, spec)
        if role == "visual":
            value = load_bound_payload(path, _numpy_loader, label=label)[0]
            features, timebase = _decode_visual(value)
            return _visual_cue(features, timebase, spec)
        raise ValueError(f"unsupported TSE cue role file: {role!r}")


def _embedding_cue(value: Any, spec: Any) -> Any:
    """1-D speaker embedding npy → ``[1,D]`` cue。"""
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.size < 1:
        raise ValueError("speaker embedding cue must be a non-empty 1-D ndarray")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("speaker embedding cue must be finite floating-point")
    if spec.representation != "embedding":
        raise ValueError("model contract does not consume speaker embedding cues")
    tensor = _float_tensor(value, "speaker embedding cue")
    if tensor.ndim != 1:
        raise ValueError("speaker embedding cue must be 1-D")
    return TSECueInput("speaker_embedding", "embedding", tensor[None], None, None)


def _spatial_cue(value: Any, spec: Any) -> Any:
    direction, anchors = _decode_spatial(value, spec.representation)
    if spec.representation == "static":
        return TSECueInput("spatial", "static", direction[None], None, None)
    if anchors is None:
        raise ValueError("track spatial cue requires sample anchors")
    return TSECueInput(
        "spatial",
        "track",
        direction[None],
        int(direction.shape[-1]),
        anchors[None],
    )


def _visual_cue(features: Any, timebase: Any, spec: Any) -> Any:
    if spec.representation == "raw":
        if features.ndim != 4:
            raise ValueError("raw visual cue must decode to [T,3,H,W]")
        return TSECueInput(
            "visual", "raw", features[None], int(features.shape[0]), timebase[None]
        )
    if features.ndim != 2:
        raise ValueError("visual features cue must have shape [F,D]")
    if tuple(timebase.shape) != (features.shape[0],):
        raise ValueError("visual features timebase must align with the F axis")
    return TSECueInput(
        "visual", "features", features[None], int(features.shape[0]), timebase[None]
    )


def _textual_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("textual cue JSON must be an object")
    if "phoneme_ids" in payload:
        return dict(payload)
    raise ValueError("textual cue JSON requires a 'phoneme_ids' key")


def _inline_cue(role: str, spec: Any, payload: Mapping[str, Any]) -> Any:
    """把 inline cue JSON 转成 typed cue，并按模型合同校验表示与维度。"""
    import torch

    if role == "speaker_embedding":
        values = _float_tensor(_payload_sequence(payload, "values", role), role)
        if values.ndim != 1:
            raise ValueError("inline speaker_embedding 'values' must be 1-D")
        return TSECueInput("speaker_embedding", "embedding", values[None], None, None)
    if role == "spatial":
        azimuth = payload.get("azimuth")
        elevation = payload.get("elevation")
        if isinstance(azimuth, (int, float)) and isinstance(elevation, (int, float)):
            tensor = _float_tensor([azimuth, elevation], role)
            return TSECueInput("spatial", "static", tensor[None], None, None)
        azimuth = _payload_sequence(payload, "azimuth", role)
        elevation = _payload_sequence(payload, "elevation", role)
        anchors = _payload_sequence(payload, "sample_anchor", role)
        if not (len(azimuth) == len(elevation) == len(anchors)):
            raise ValueError("inline spatial track fields must share one P axis")
        direction = _float_tensor(
            torch.stack(
                (
                    _float_tensor(azimuth, role),
                    _float_tensor(elevation, role),
                )
            ),
            role,
        )
        anchor_tensor = torch.as_tensor(anchors)
        if anchor_tensor.dtype == torch.bool or anchor_tensor.is_floating_point():
            anchor_tensor = anchor_tensor.to(torch.float64)
            if not bool(torch.equal(anchor_tensor, anchor_tensor.floor())):
                raise ValueError("inline spatial sample_anchor must be integers")
        anchors_int = anchor_tensor.to(torch.int64)
        if int(anchors_int[0]) != 0 or bool((anchors_int[1:] <= anchors_int[:-1]).any()):
            raise ValueError(
                "inline spatial sample_anchor must start at 0 and strictly increase"
            )
        return TSECueInput(
            "spatial", "track", direction[None], int(direction.shape[-1]), anchors_int[None]
        )
    if role == "visual":
        features = _float_tensor(_payload_sequence(payload, "features", role), role)
        if features.ndim != 2:
            raise ValueError("inline visual 'features' must be [F,D]")
        timebase = _float_tensor(_payload_sequence(payload, "timebase", role), role)
        if tuple(timebase.shape) != (features.shape[0],):
            raise ValueError("inline visual 'timebase' must align with the F axis")
        if bool((timebase < 0).any()) or bool((timebase[1:] < timebase[:-1]).any()):
            raise ValueError(
                "inline visual 'timebase' must be non-negative and non-decreasing"
            )
        return TSECueInput(
            "visual", "features", features[None], int(features.shape[0]), timebase[None]
        )
    if role == "textual":
        ids = torch.as_tensor(_payload_sequence(payload, "phoneme_ids", role))
        if ids.dtype == torch.bool or ids.is_floating_point():
            ids = ids.to(torch.float64)
            if not bool(torch.equal(ids, ids.floor())):
                raise ValueError("inline textual phoneme_ids must be integers")
        ids = ids.to(torch.int64)
        if ids.ndim != 1 or ids.numel() < 1:
            raise ValueError("inline textual phoneme_ids must be a non-empty 1-D list")
        return TSECueInput("textual", "phoneme_ids", ids[None], int(ids.numel()), None)
    raise ValueError(f"unsupported inline TSE cue role: {role!r}")


def cue_declaration(
    files: Mapping[str, Path],
    inline: Mapping[str, Any],
) -> dict[str, Any]:
    """构造 canonical cue 声明：文件 cue 绑路径，inline cue 绑 canonical JSON。"""
    declaration: dict[str, Any] = {
        role: {"kind": "file", "path": str(path)} for role, path in files.items()
    }
    for role, payload in inline.items():
        if role in declaration:
            raise ValueError(f"cue role {role!r} cannot be both a file and inline")
        declaration[role] = {"kind": "inline", "value": dict(payload)}
    return declaration


def build_inference_adapter(extractor: Any) -> TSEInferenceAdapter:
    """构造顶层文件推理使用的 TSE adapter。

    模型加载仍由 ``TSEExtractor`` 完成；descriptor factory 只负责把已加载的
    extractor 接到公共 ``InferenceSpec`` 生命周期，避免在 registry 内复制
    checkpoint、cue 或 sidecar 逻辑。
    """

    return TSEInferenceAdapter(extractor)
