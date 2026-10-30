"""任务无关的多通道 spatial simulation 基础合同。

本模块只提供不可变结构和纯投影函数，不读取配置、不抽样、不渲染，也不改变
现有任务的默认仿真语义。``C`` 表示 microphone channel 数，``K`` 表示
source/track 数；二者刻意保持独立。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from core.data.simulation.scene_spec import SceneSpec


class LayerPolicy(str, Enum):
    """Layer D-F 的执行粒度。"""

    SHARED = "shared"
    PER_CHANNEL = "per_channel"


@dataclass(frozen=True, slots=True)
class SpatialChannelLayout:
    """多通道布局合同；``reference_channel`` 使用零基索引。"""

    num_channels: int
    num_tracks: int
    reference_channel: int = 0

    def __post_init__(self) -> None:
        for name in ("num_channels", "num_tracks", "reference_channel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
        if self.num_channels < 1:
            raise ValueError("num_channels (C) must be positive")
        if self.num_tracks < 0:
            raise ValueError("num_tracks (K) must be non-negative")
        if not 0 <= self.reference_channel < self.num_channels:
            raise ValueError("reference_channel must be in [0, C)")

    @property
    def C(self) -> int:
        return self.num_channels

    @property
    def K(self) -> int:
        return self.num_tracks

    @property
    def reference(self) -> int:
        return self.reference_channel


@dataclass(frozen=True, slots=True)
class SpatialSourceSceneContract:
    """Layer A/B 的 source-scene 合同，与 C/K/reference 几何分离。

    ``scene`` 是 Layer B 的场景存在性描述；``source_names`` 是 Layer A
    实际准备的 source 顺序。``layout.K`` 必须等于 source 数量，避免把
    latent reference、enrollment 或 background 错当成监督 source。
    """

    scene: SceneSpec
    source_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scene, SceneSpec):
            raise TypeError("scene must be a SceneSpec")
        names = tuple(self.source_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("source_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("source_names must be unique")
        if len(names) != self.scene.source_count:
            raise ValueError("source_names length must equal scene.source_count")
        object.__setattr__(self, "source_names", names)

    @property
    def K(self) -> int:
        return len(self.source_names)


SpatialSourceScene = SpatialSourceSceneContract


@dataclass(frozen=True, slots=True)
class TrackProjection:
    """将命名 track 投影到 source index 和 microphone channel。"""

    track_to_source: tuple[tuple[str, int], ...]
    track_to_channel: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        try:
            source_items = tuple(self.track_to_source)
            channel_items = tuple(self.track_to_channel)
        except TypeError as exc:
            raise TypeError("track projections must be sequences of (name, index)") from exc
        for items in (source_items, channel_items):
            if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
                raise ValueError("track projections must contain (name, index) pairs")
        source = dict(source_items)
        channel = dict(channel_items)
        if len(source) != len(source_items) or len(channel) != len(channel_items):
            raise ValueError("track projections must not contain duplicate names")
        if set(source) != set(channel):
            raise ValueError("source and channel projections must contain the same tracks")
        for name, index in (*source.items(), *channel.items()):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("track names must be non-empty strings")
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
                raise ValueError("track projection indices must be non-negative integers")
        object.__setattr__(self, "track_to_source", tuple(source.items()))
        object.__setattr__(self, "track_to_channel", tuple(channel.items()))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, tuple[int, int]]) -> "TrackProjection":
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must be a mapping of track to (source, channel)")
        items = []
        for name, pair in mapping.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("each track projection must be a (source, channel) pair")
            items.append((name, pair))
        return cls(
            tuple((name, pair[0]) for name, pair in items),
            tuple((name, pair[1]) for name, pair in items),
        )

    @property
    def tracks(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.track_to_source)

    def source_index(self, track: str) -> int:
        return dict(self.track_to_source)[track]

    def channel_index(self, track: str) -> int:
        return dict(self.track_to_channel)[track]


@dataclass(frozen=True, slots=True)
class SpatialLayerPolicy:
    """Layer D、E、F 的 shared/per-channel 粒度声明。"""

    layer_d: LayerPolicy = LayerPolicy.SHARED
    layer_e: LayerPolicy = LayerPolicy.SHARED
    layer_f: LayerPolicy = LayerPolicy.PER_CHANNEL

    def __post_init__(self) -> None:
        for name in ("layer_d", "layer_e", "layer_f"):
            value = getattr(self, name)
            if not isinstance(value, LayerPolicy):
                try:
                    value = LayerPolicy(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be shared or per_channel") from exc
                object.__setattr__(self, name, value)


def project_channels(
    signal: Any,
    layout: SpatialChannelLayout,
    *,
    channel_axis: int = -2,
) -> np.ndarray:
    """返回 reference channel 的单通道投影，channel_axis 必须指向 C。"""
    array = np.asarray(signal)
    if array.ndim == 0:
        raise ValueError("signal must have a channel dimension")
    if not isinstance(channel_axis, int):
        raise TypeError("channel_axis must be an integer")
    if not -array.ndim <= channel_axis < array.ndim:
        raise ValueError("channel_axis is outside signal dimensions")
    if array.shape[channel_axis] != layout.C:
        raise ValueError(f"signal channel dimension must be C={layout.C}, got {array.shape}")
    return np.take(array, layout.reference, axis=channel_axis)


def project_tracks(
    tracks: Mapping[str, Any],
    projection: TrackProjection,
) -> dict[str, Any]:
    """按 TrackProjection 选择 track，保持 projection 声明的顺序。"""
    if not isinstance(tracks, Mapping):
        raise TypeError("tracks must be a mapping")
    missing = tuple(name for name in projection.tracks if name not in tracks)
    if missing:
        raise KeyError(f"missing projected tracks: {missing!r}")
    return {name: tracks[name] for name in projection.tracks}


def validate_projection(
    layout: SpatialChannelLayout,
    projection: TrackProjection,
) -> None:
    """验证 projection 没有超出 C/K 合同。"""
    if not isinstance(layout, SpatialChannelLayout):
        raise TypeError("layout must be a SpatialChannelLayout")
    if not isinstance(projection, TrackProjection):
        raise TypeError("projection must be a TrackProjection")
    sources = [index for _, index in projection.track_to_source]
    channels = [index for _, index in projection.track_to_channel]
    if any(index >= layout.K for index in sources):
        raise ValueError("track source index exceeds K")
    if any(index >= layout.C for index in channels):
        raise ValueError("track channel index exceeds C")


def _validate_rir_bank(rirs: np.ndarray, *, layout: SpatialChannelLayout) -> np.ndarray:
    rirs = np.asarray(rirs, dtype=np.float32)
    if rirs.ndim != 3 or rirs.shape[:2] != (layout.K, layout.C):
        raise ValueError(
            f"rirs must have shape [K, C, L] with K={layout.K}, C={layout.C}; "
            f"got {rirs.shape}"
        )
    if rirs.shape[-1] < 1 or not np.isfinite(rirs).all():
        raise ValueError("rirs must be finite and have a non-empty time axis")
    return rirs


def channel_count_from_geometry(geometry: Mapping[str, Any]) -> int:
    """从 resolved geometry metadata 推导 microphone channel 数。"""
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping")
    positions = geometry.get("positions_m")
    topology = geometry.get("topology")
    if positions is None and isinstance(topology, Mapping):
        positions = topology.get("positions_m")
    if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence) or not positions:
        raise ValueError("geometry must provide non-empty positions_m")
    return len(positions)


def load_offline_rir_bank(
    manifest: str | Path,
    source_keys: Sequence[str],
    *,
    geometry: Mapping[str, Any],
    return_metadata: bool = False,
    track: str = "full",
    include_internal_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Load a source-keyed JSON/SCP bank of ``[C, L]`` NumPy RIR arrays.

    The offline scene seam is intentionally explicit: every requested source key
    must have one finite ``.npy``/``.npz`` array, and ordinary audio manifests are
    never interpreted as microphone arrays.
    """
    num_channels = channel_count_from_geometry(geometry)
    if track not in {"full", "direct", "early"}:
        raise ValueError("track must be one of: full, direct, early")
    path = Path(manifest).expanduser()
    if not path.is_file():
        raise ValueError(f"offline spatial RIR manifest does not exist: {path}")
    entries: dict[str, str | dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = None
    manifest_metadata: dict[str, Any] = {}
    if isinstance(raw, dict):
        if "entries" in raw:
            manifest_metadata = raw.get("metadata", {})
            if not isinstance(manifest_metadata, dict):
                raise ValueError("offline spatial RIR metadata must be a mapping")
            raw_entries = raw["entries"]
        else:
            raw_entries = raw
        if not isinstance(raw_entries, dict):
            raise ValueError("offline spatial RIR JSON entries must be a mapping")
        for key, value in raw_entries.items():
            if not isinstance(key, str):
                raise ValueError("offline spatial RIR source keys must be strings")
            if not isinstance(value, (str, dict)):
                raise ValueError("offline spatial RIR entries must be paths or track mappings")
            if isinstance(value, dict):
                if "bundle_path" in value:
                    if not isinstance(value.get("bundle_path"), str) or type(value.get("actor_index")) is not int:
                        raise ValueError("bundle RIR entries require bundle_path and integer actor_index")
                    for track_name in ("full", "direct", "early"):
                        if track_name in value and not isinstance(value[track_name], str):
                            raise ValueError("bundle track keys must be strings")
                elif any(not isinstance(track_name, str) or not isinstance(track_path, str)
                         for track_name, track_path in value.items()):
                    raise ValueError("offline spatial RIR track mappings must map names to paths")
        entries = dict(raw_entries)
    elif raw is None:
        for line_number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number} must contain <source_key> <rir_path>")
            entries[fields[0]] = fields[1]
    else:
        raise ValueError("offline spatial RIR manifest must be a JSON mapping or SCP")
    missing = [key for key in source_keys if key not in entries]
    if missing:
        raise ValueError(f"offline spatial RIR manifest missing source keys: {missing!r}")
    entry_modes: dict[str, str] = {}
    entry_tracks: dict[str, dict[str, str]] = {}
    resolved_entries: dict[str, Any] = {}
    for key in source_keys:
        value = entries[key] if key in entries else None
        if isinstance(value, dict):
            entry_modes[key] = "tracks"
            if "bundle_path" in value:
                entry_tracks[key] = {name: value[name] for name in ("full", "direct", "early") if name in value}
                if track not in entry_tracks[key]:
                    raise ValueError(f"offline bundle RIR entry {key!r} is missing {track!r} track key")
                resolved_entries[key] = (value["bundle_path"], value[track], int(value["actor_index"]))
                continue
            entry_tracks[key] = dict(value)
            if track not in value:
                raise ValueError(f"offline spatial RIR entry {key!r} is missing {track!r} track")
            resolved_entries[key] = value[track]
        elif isinstance(value, str):
            entry_modes[key] = "full"
            if track != "full":
                raise ValueError(
                    f"offline full-only RIR manifest has no {track!r} track for source {key!r}"
                )
            resolved_entries[key] = value
        else:
            resolved_entries[key] = ""
    arrays = []
    for key in source_keys:
        resolved = resolved_entries[key]
        bundle_key = None
        actor_index = None
        if isinstance(resolved, tuple):
            rir_path = Path(resolved[0]).expanduser()
            bundle_key = resolved[1]
            actor_index = resolved[2]
        else:
            rir_path = Path(resolved).expanduser()
        if not rir_path.is_absolute():
            rir_path = path.parent / rir_path
        try:
            with np.load(rir_path, allow_pickle=False) as loaded:
                if isinstance(loaded, np.ndarray):
                    rir = loaded
                elif hasattr(loaded, "files"):
                    if bundle_key is not None:
                        if bundle_key not in loaded.files:
                            raise ValueError(f"bundle is missing track key {bundle_key!r}")
                        rir = loaded[bundle_key][int(actor_index)]
                    elif len(loaded.files) == 1:
                        rir = loaded[loaded.files[0]]
                    else:
                        raise ValueError("npz must declare a track key")
                else:
                    raise ValueError("npz must contain exactly one array")
        except (AttributeError, TypeError):
            rir = np.load(rir_path, allow_pickle=False)
        rir = np.asarray(rir, dtype=np.float32)
        if rir.ndim != 2 or rir.shape[0] != num_channels or rir.shape[1] < 1:
            raise ValueError(f"offline RIR for {key!r} must have shape [C,L], got {rir.shape}")
        if not np.isfinite(rir).all():
            raise ValueError(f"offline RIR for {key!r} contains NaN/Inf")
        arrays.append(rir)
    lengths = {array.shape[1] for array in arrays}
    if len(lengths) != 1:
        raise ValueError("offline spatial RIRs must have one common length")
    bank = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    manifest_metadata = {
        key: value
        for key, value in dict(manifest_metadata).items()
        if key not in {"_entry_modes", "_entry_tracks"}
    }
    if include_internal_metadata:
        manifest_metadata["_entry_modes"] = entry_modes
        manifest_metadata["_entry_tracks"] = entry_tracks
    return (bank, manifest_metadata) if return_metadata else bank


def load_offline_rir_bank_with_metadata(
    manifest: str | Path,
    source_keys: Sequence[str],
    *,
    geometry: Mapping[str, Any],
    track: str = "full",
    include_internal_metadata: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取 offline RIR bank 及 manifest 中的公开 metadata。"""
    return load_offline_rir_bank(
        manifest,
        source_keys,
        geometry=geometry,
        return_metadata=True,
        track=track,
        include_internal_metadata=include_internal_metadata,
    )


def _load_moving_actor_track(path: Path, track: str, bundle_key, actor_index, num_channels: int) -> np.ndarray:
    """读取单个 actor 的 moving track 文件，返回 ``[P, C, L]``。"""
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if isinstance(loaded, np.ndarray):
                array = loaded
            elif hasattr(loaded, "files"):
                if bundle_key is not None:
                    if bundle_key not in loaded.files:
                        raise ValueError(f"moving bundle is missing track key {bundle_key!r}")
                    # scene bundle 的 moving track 布局为 [P, A, C, L]。
                    array = loaded[bundle_key][:, int(actor_index)]
                elif len(loaded.files) == 1:
                    array = loaded[loaded.files[0]]
                else:
                    raise ValueError("moving npz must declare a track key")
            else:
                raise ValueError("moving npz must contain exactly one array")
    except (AttributeError, TypeError):
        array = np.load(path, allow_pickle=False)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] != num_channels or array.shape[2] < 1:
        raise ValueError(
            f"moving RIR track must have shape [P, C={num_channels}, L], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("moving RIR track contains NaN/Inf")
    return array


def load_offline_moving_rir_bank_with_metadata(
    manifest: str | Path,
    source_keys: Sequence[str],
    *,
    geometry: Mapping[str, Any],
    track: str = "full",
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取 typed moving scene 的逐 actor track，返回 ``[P, A, C, L]`` bank。

    只接受 typed scene manifest（``entries`` + ``metadata.moving: true``）；
    每个 actor 的 track 文件布局为 ``[P, C, L]``（或 scene bundle 内
    ``[P, A, C, L]`` 的 actor 切片）。actor 顺序严格按 ``source_keys``。
    """
    num_channels = channel_count_from_geometry(geometry)
    if track not in {"full", "direct", "early"}:
        raise ValueError("track must be one of: full, direct, early")
    path = Path(manifest).expanduser()
    if not path.is_file():
        raise ValueError(f"offline moving scene manifest does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("offline moving scene manifest must be a typed JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), dict):
        raise ValueError("offline moving scene manifest must declare typed 'entries'")
    manifest_metadata = raw.get("metadata", {})
    if not isinstance(manifest_metadata, dict):
        raise ValueError("offline moving scene metadata must be a mapping")
    if manifest_metadata.get("moving") is not True:
        raise ValueError("offline moving scene manifest requires metadata.moving=true")
    entries: dict[str, Any] = raw["entries"]
    tracks: list[np.ndarray] = []
    for key in source_keys:
        value = entries.get(key)
        bundle_key = None
        actor_index = None
        if isinstance(value, dict) and "bundle_path" in value:
            if not isinstance(value.get(track), str) or type(value.get("actor_index")) is not int:
                raise ValueError(f"moving bundle entry {key!r} is missing {track!r} track key")
            rir_path = Path(value["bundle_path"]).expanduser()
            bundle_key = value[track]
            actor_index = value["actor_index"]
        elif isinstance(value, dict):
            if not isinstance(value.get(track), str):
                raise ValueError(f"moving RIR entry {key!r} is missing {track!r} track")
            rir_path = Path(value[track]).expanduser()
        elif isinstance(value, str):
            rir_path = Path(value).expanduser()
        else:
            raise ValueError(f"offline moving scene manifest missing actor entry: {key!r}")
        if not rir_path.is_absolute():
            rir_path = path.parent / rir_path
        tracks.append(
            _load_moving_actor_track(rir_path, track, bundle_key, actor_index, num_channels)
        )
    frame_shapes = {(array.shape[0], array.shape[2]) for array in tracks}
    if len(frame_shapes) != 1:
        raise ValueError("offline moving scene tracks must share one [P, L] shape")
    bank = np.stack(tracks, axis=1).astype(np.float32, copy=False)
    return bank, manifest_metadata


def load_offline_spatial_rir_result(
    manifest: str | Path,
    source_keys: Sequence[str],
    *,
    geometry: Mapping[str, Any],
):
    """通过统一 typed provider 读取 offline spatial RIR bank。"""
    from core.data.simulation.rir_provider import OfflineRIRProvider

    keys = tuple(source_keys)
    return OfflineRIRProvider(manifest, geometry=geometry).get_rirs(
        sample_rate=0, source_count=len(keys), rng=None, source_uids=keys
    )


@dataclass(frozen=True, slots=True)
class SpatialRenderResult:
    """Layer C 的 full/direct/early/RTS 可分解渲染结果。"""

    full_tracks: np.ndarray
    early_tracks: np.ndarray
    reference_tracks: np.ndarray
    mixture: np.ndarray
    direct_tracks: np.ndarray | None = None
    geometry: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None
    rts_tracks: np.ndarray | None = None

    def __post_init__(self) -> None:
        full = np.asarray(self.full_tracks, dtype=np.float32)
        early = np.asarray(self.early_tracks, dtype=np.float32)
        reference = np.asarray(self.reference_tracks, dtype=np.float32)
        mixture = np.asarray(self.mixture, dtype=np.float32)
        direct = None if self.direct_tracks is None else np.asarray(self.direct_tracks, dtype=np.float32)
        rts = None if self.rts_tracks is None else np.asarray(self.rts_tracks, dtype=np.float32)
        if full.ndim != 3 or early.shape != full.shape:
            raise ValueError("full_tracks and early_tracks must have matching [K,C,T] shapes")
        if direct is not None and direct.shape != full.shape:
            raise ValueError("direct_tracks must match full_tracks [K,C,T] shape")
        if rts is not None and rts.shape != full.shape:
            raise ValueError("rts_tracks must match full_tracks [K,C,T] shape")
        if reference.shape != (full.shape[0], full.shape[2]):
            raise ValueError("reference_tracks must have shape [K,T]")
        if mixture.shape != (full.shape[1], full.shape[2]):
            raise ValueError("mixture must have shape [C,T]")
        values = (full, early, reference, mixture) + (() if direct is None else (direct,)) + (() if rts is None else (rts,))
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("rendered tracks must be finite")
        for name, value in (("geometry", self.geometry), ("provenance", self.provenance)):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")
        for name, value in (("full_tracks", full), ("early_tracks", early),
                            ("reference_tracks", reference), ("mixture", mixture),
                            ("direct_tracks", direct), ("rts_tracks", rts)):
            if value is not None:
                value = np.ascontiguousarray(value, dtype=np.float32)
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "geometry", dict(self.geometry or {}))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def has_direct_path(self) -> bool:
        return self.direct_tracks is not None


_DIRECT_CONVOLUTION_WORK_LIMIT = 1 << 18
_OVERLAP_ADD_LENGTH_RATIO = 8


def _convolve_truncated(source: np.ndarray, rir: np.ndarray, length: int) -> np.ndarray:
    """短输入保持直接卷积；长且失衡的输入使用 overlap-add。"""
    source = np.asarray(source, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)
    if source.size * rir.size <= _DIRECT_CONVOLUTION_WORK_LIMIT:
        return np.asarray(np.convolve(source, rir)[:length], dtype=np.float32)
    from scipy.signal import fftconvolve, oaconvolve

    shorter = min(source.size, rir.size)
    longer = max(source.size, rir.size)
    convolver = (
        oaconvolve
        if longer >= _OVERLAP_ADD_LENGTH_RATIO * shorter
        else fftconvolve
    )
    return np.asarray(convolver(source, rir, mode="full")[:length], dtype=np.float32)


def render_spatial_scene(
    sources: np.ndarray,
    rirs: np.ndarray,
    *,
    early_boundary: int,
    reference_channel: int = 0,
    num_samples: int | None = None,
    direct_rirs: np.ndarray | None = None,
    early_rirs: np.ndarray | None = None,
    rts_rirs: np.ndarray | None = None,
    geometry: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SpatialRenderResult:
    """Render full/early tracks and explicitly requested target RIR tracks."""
    sources = np.asarray(sources, dtype=np.float32)
    if sources.ndim != 2:
        raise ValueError(f"sources must have shape [K,T], got {sources.shape}")
    if type(early_boundary) is not int or early_boundary < 0:
        raise ValueError("early_boundary must be a non-negative exact int")
    layout = SpatialChannelLayout(rirs.shape[1], sources.shape[0], reference_channel) if np.asarray(rirs).ndim == 3 else None
    if layout is None:
        raise ValueError("rirs must have shape [K,C,L]")
    rirs = _validate_rir_bank(rirs, layout=layout)
    direct_bank = None
    if direct_rirs is not None:
        direct_bank = _validate_rir_bank(direct_rirs, layout=layout)
        if direct_bank.shape[:2] != rirs.shape[:2]:
            raise ValueError("direct_rirs must match full rirs source/channel axes")
    early_bank = None
    if early_rirs is not None:
        early_bank = _validate_rir_bank(early_rirs, layout=layout)
        if early_bank.shape[:2] != rirs.shape[:2]:
            raise ValueError("early_rirs must match full rirs source/channel axes")
    rts_bank = None
    if rts_rirs is not None:
        rts_bank = _validate_rir_bank(rts_rirs, layout=layout)
        if rts_bank.shape[:2] != rirs.shape[:2]:
            raise ValueError("rts_rirs must match full rirs source/channel axes")
    if not np.isfinite(sources).all():
        raise ValueError("sources must be finite")
    length = sources.shape[1] if num_samples is None else num_samples
    if type(length) is not int or length < 1:
        raise ValueError("num_samples must be a positive exact int")
    if length != sources.shape[1] and length < sources.shape[1]:
        raise ValueError("num_samples cannot truncate source alignment")
    full = np.zeros((layout.K, layout.C, length), dtype=np.float32)
    early = np.zeros_like(full)
    direct = None if direct_bank is None else np.zeros_like(full)
    rts = None if rts_bank is None else np.zeros_like(full)
    for k in range(layout.K):
        for c in range(layout.C):
            convolution = _convolve_truncated(sources[k], rirs[k, c], length)
            full[k, c, :convolution.size] = convolution
            early_rir = (
                early_bank[k, c]
                if early_bank is not None
                else rirs[k, c, :early_boundary]
            )
            if early_rir.size:
                early_conv = _convolve_truncated(sources[k], early_rir, length)
                early[k, c, :early_conv.size] = early_conv
            if direct is not None:
                direct_rir = direct_bank[k, c]
                direct_conv = _convolve_truncated(sources[k], direct_rir, length)
                direct[k, c, :direct_conv.size] = direct_conv
            if rts is not None:
                rts_conv = _convolve_truncated(sources[k], rts_bank[k, c], length)
                rts[k, c, :rts_conv.size] = rts_conv
    reference = full[:, layout.reference_channel, :]
    return SpatialRenderResult(
        full, early, reference, full.sum(axis=0, dtype=np.float32),
        direct_tracks=direct, geometry=geometry, provenance=provenance,
        rts_tracks=rts,
    )


def _validate_observation(observation: Any) -> np.ndarray:
    value = np.asarray(observation, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"observation must have shape [C,T], got {value.shape}")
    if value.shape[0] < 1 or value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError("observation must be finite with positive C and T")
    return value


def apply_observation_layer(
    observation: Any,
    decision: Any,
    *,
    policy: LayerPolicy | str,
) -> np.ndarray:
    """Apply one D/E/F decision to observation ``[C,T]`` only.

    ``shared`` accepts one scalar decision and ``per_channel`` accepts either
    ``[C]`` or a broadcastable ``[C,1]`` vector.  This seam never receives or
    modifies targets, making target invariance explicit.
    """
    value = _validate_observation(observation)
    try:
        policy = LayerPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy must be shared or per_channel") from exc
    factors = np.asarray(decision, dtype=np.float32)
    if not np.isfinite(factors).all():
        raise ValueError("decision must be finite")
    if policy is LayerPolicy.SHARED:
        if factors.size != 1:
            raise ValueError("shared decision must contain one value")
        factors = factors.reshape(1, 1)
    else:
        if factors.size != value.shape[0]:
            raise ValueError("per_channel decision must contain one value per channel")
        factors = factors.reshape(value.shape[0], 1)
    return (value * factors).astype(np.float32, copy=False)


def apply_spatial_layer_d(observation: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """Apply a joint Layer D decision to every channel without changing shape."""
    observation = _validate_observation(observation)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    return (observation * np.float32(scale)).astype(np.float32)


def apply_spatial_layer_ef(
    observation: np.ndarray,
    *,
    decision: bool = True,
    waveform=None,
    per_channel: bool = False,
) -> np.ndarray:
    """Apply shared E/F decision with optional per-channel waveform operation."""
    observation = _validate_observation(observation)
    if not isinstance(decision, (bool, np.bool_)):
        raise TypeError("decision must be boolean")
    if not decision or waveform is None:
        return observation.copy()
    if per_channel:
        output = np.stack([np.asarray(waveform(channel), dtype=np.float32) for channel in observation], axis=0)
    else:
        output = np.asarray(waveform(observation), dtype=np.float32)
    if output.shape != observation.shape or not np.isfinite(output).all():
        raise ValueError("Layer E/F waveform must preserve finite [C,T] shape")
    return output


def project_se_scene(observation: np.ndarray, target: np.ndarray, *, reference_channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Project a supplied spatial scene to SE ``noisy [C,T]`` and ``target [T]``."""
    observation = _validate_observation(observation)
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 1 or target.shape[0] != observation.shape[1] or not np.isfinite(target).all():
        raise ValueError("target must be finite and have shape [T]")
    if type(reference_channel) is not int or not 0 <= reference_channel < observation.shape[0]:
        raise ValueError("reference_channel must be in [0,C)")
    return observation.copy(), target.copy()


def project_ss_scene(observation: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a supplied spatial scene to SS ``mixture [C,T]`` and ``targets [K,T]``."""
    observation = _validate_observation(observation)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != observation.shape[1] or not np.isfinite(targets).all():
        raise ValueError("targets must be finite and have shape [K,T]")
    return observation.copy(), targets.copy()


def derive_rts_rir_bank(
    rirs: np.ndarray,
    *,
    reference_channel: int,
    original_t60_sec: float,
    target_t60_sec: float,
    sample_rate: int,
    time_after_max_sec: float = 0.002,
) -> np.ndarray:
    """Apply an RTS decay window, anchored to each source's reference-mic peak."""
    values = np.asarray(rirs, dtype=np.float32)
    if values.ndim < 2 or values.shape[-2] < 1 or values.shape[-1] < 1:
        raise ValueError("rirs must have shape [...,C,L] with positive C and L")
    if not np.isfinite(values).all():
        raise ValueError("rirs must be finite")
    if type(reference_channel) is not int or not 0 <= reference_channel < values.shape[-2]:
        raise ValueError("reference_channel must be an exact int in [0,C)")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive exact int")
    for name, value, allow_zero in (
        ("original_t60_sec", original_t60_sec, False),
        ("target_t60_sec", target_t60_sec, False),
        ("time_after_max_sec", time_after_max_sec, True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")
    original_t60_sec = float(original_t60_sec)
    target_t60_sec = float(target_t60_sec)
    if original_t60_sec <= target_t60_sec:
        return values.copy()

    reference = values[..., reference_channel, :]
    peak_indices = np.argmax(np.abs(reference), axis=-1)
    start_indices = peak_indices + int(float(time_after_max_sec) * sample_rate)
    samples = np.arange(values.shape[-1], dtype=np.int64)
    elapsed = np.maximum(samples - start_indices[..., None], 0)
    decay_rate = 3.0 / (target_t60_sec * sample_rate) - 3.0 / (
        original_t60_sec * sample_rate
    )
    decay = np.power(10.0, -decay_rate * elapsed)
    window = np.where(samples < start_indices[..., None], 1.0, decay)
    return np.ascontiguousarray(values * window[..., None, :], dtype=np.float32)


def derive_rir_tracks(
    rirs: np.ndarray,
    layout: SpatialChannelLayout,
    *,
    early_samples: int,
) -> dict[str, np.ndarray]:
    """Return full, early and reference RIR tracks without changing timing."""
    rirs = _validate_rir_bank(rirs, layout=layout)
    if type(early_samples) is not int or early_samples < 0:
        raise ValueError("early_samples must be a non-negative exact int")
    early = rirs.copy()
    early[:, :, early_samples:] = 0
    return {
        "full": rirs.copy(),
        "early": early,
        "reference": rirs[:, layout.reference].copy(),
    }


__all__ = [
    "LayerPolicy",
    "SpatialChannelLayout",
    "SpatialLayerPolicy",
    "SpatialRenderResult",
    "SpatialSourceScene",
    "SpatialSourceSceneContract",
    "TrackProjection",
    "channel_count_from_geometry",
    "apply_observation_layer",
    "apply_spatial_layer_d",
    "apply_spatial_layer_ef",
    "derive_rir_tracks",
    "derive_rts_rir_bank",
    "load_offline_rir_bank",
    "load_offline_rir_bank_with_metadata",
    "load_offline_spatial_rir_result",
    "project_channels",
    "project_se_scene",
    "project_ss_scene",
    "project_tracks",
    "render_spatial_scene",
    "validate_projection",
]
