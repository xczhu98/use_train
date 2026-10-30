"""WeSep manifest 与 cue 索引的只读导入适配器。"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MANIFEST_FORMAT_JSONL = "jsonl"
MANIFEST_FORMAT_PATH_LIST = "path_list"
PATH_BASE_MANIFEST_RELATIVE = "manifest_relative"
PATH_BASE_RECIPE_ROOT = "recipe_root"

_CUE_ORDER = ("audio", "spatial", "visual", "textual")
_CUE_FORMATS = {
    "audio": {("wav", "waveform"), ("npy", "spk_embedding")},
    "spatial": {("npy", "fields")},
    "visual": {("mp4", "raw_video"), ("npy", "muse_frontend")},
    "textual": {("json", "dae_keyword_phoneme")},
}
_DEFAULT_CUE_FORMATS = {
    "audio": ("wav", "waveform"),
    "spatial": ("npy", "fields"),
    "visual": ("mp4", "raw_video"),
}
_CUE_POLICIES = {
    "audio": {"random", "fixed"},
    "spatial": {"fixed"},
    "visual": {"fixed"},
    "textual": {"fixed"},
}
_COMMON_CUE_KEYS = {"type", "format", "scope", "guaranteed", "policy", "collate"}
_CUE_SPECIFIC_KEYS = {
    "audio": set(),
    "spatial": {"fields", "field_defaults"},
    "visual": set(),
    "textual": {"selection"},
}


@dataclass(frozen=True)
class WeSepCueSpec:
    """一个启用 cue 的索引声明，不读取实际 cue payload。"""

    name: str
    cue_type: str
    cue_format: str
    scope: str
    guaranteed: bool
    policy_type: str
    key_field: str
    resource_path: str
    format_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class WeSepCueRecord:
    """一个目标对应的 cue 索引结果。"""

    spec: WeSepCueSpec
    lookup_key: str
    present: bool
    items: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class WeSepTargetRecord:
    """按 manifest ``spk`` 顺序保留的目标与 cue 资源记录。"""

    speaker_id: str
    source_paths: tuple[str, ...]
    cues: tuple[WeSepCueRecord, ...]

    def cue(self, name: str) -> WeSepCueRecord:
        for cue_record in self.cues:
            if cue_record.spec.name == name:
                return cue_record
        raise KeyError(f"cue {name!r} is not enabled for speaker {self.speaker_id!r}")


@dataclass(frozen=True)
class WeSepSampleRecord:
    """不含解码波形的 canonical WeSep 样本资源记录。"""

    key: str
    manifest_path: str
    mix_variant: str
    mixture_paths: tuple[str, ...]
    targets: tuple[WeSepTargetRecord, ...]


@dataclass(frozen=True)
class _PathPolicy:
    mode: str
    recipe_root: str | None

    @classmethod
    def build(cls, mode: str, recipe_root: str | os.PathLike[str] | None) -> "_PathPolicy":
        if mode not in {PATH_BASE_MANIFEST_RELATIVE, PATH_BASE_RECIPE_ROOT}:
            raise ValueError(
                "path_base must be 'manifest_relative' or 'recipe_root', "
                f"got {mode!r}"
            )
        if mode == PATH_BASE_MANIFEST_RELATIVE:
            if recipe_root is not None:
                raise ValueError(
                    "recipe_root must be omitted when path_base='manifest_relative'"
                )
            return cls(mode=mode, recipe_root=None)
        if recipe_root is None:
            raise ValueError("recipe_root is required when path_base='recipe_root'")
        root = os.fspath(recipe_root)
        if not os.path.isabs(root):
            raise ValueError("recipe_root must be an absolute path")
        return cls(mode=mode, recipe_root=os.path.normpath(root))

    def resolve(self, value: str, carrier_path: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("resource paths must be non-empty strings")
        value = value.strip()
        if "://" in value:
            raise ValueError(f"only local resource paths are supported, got {value!r}")
        if os.path.isabs(value):
            return os.path.normpath(value)
        if self.mode == PATH_BASE_RECIPE_ROOT:
            assert self.recipe_root is not None
            base = self.recipe_root
        else:
            base = os.path.dirname(carrier_path)
        return os.path.abspath(os.path.join(base, value))


def _absolute_input_path(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    if not value:
        raise ValueError("input path must be non-empty")
    return os.path.abspath(value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_loads(text: str, source: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {source}: {error}") from error


def _load_json_file(path: str) -> Any:
    with open(path, encoding="utf-8") as stream:
        return _json_loads(stream.read(), path)


def _read_jsonl(path: str) -> list[tuple[Mapping[str, Any], str]]:
    records: list[tuple[Mapping[str, Any], str]] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = _json_loads(line, f"{path}:{line_number}")
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append((value, path))
    if not records:
        raise ValueError(f"manifest {path!r} contains no JSON records")
    return records


def _read_manifest_records(
    manifest_path: str,
    manifest_format: str,
    path_policy: _PathPolicy,
) -> list[tuple[Mapping[str, Any], str]]:
    if manifest_format == MANIFEST_FORMAT_JSONL:
        return _read_jsonl(manifest_path)
    if manifest_format != MANIFEST_FORMAT_PATH_LIST:
        raise ValueError(
            "manifest_format must be 'jsonl' or 'path_list', "
            f"got {manifest_format!r}"
        )

    referenced_paths: list[str] = []
    with open(manifest_path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = line.strip()
            if not value:
                continue
            if value.startswith(("{", "[", '"')):
                raise ValueError(
                    f"{manifest_path}:{line_number} contains JSON content but "
                    "manifest_format='path_list'"
                )
            referenced_paths.append(path_policy.resolve(value, manifest_path))
    if not referenced_paths:
        raise ValueError(f"path list {manifest_path!r} contains no JSONL paths")

    records: list[tuple[Mapping[str, Any], str]] = []
    for jsonl_path in referenced_paths:
        records.extend(_read_jsonl(jsonl_path))
    return records


@dataclass(frozen=True, eq=False)
class _FrozenMapping(Mapping[str, Any]):
    """可跨 spawn 重建的递归只读索引，不序列化内部 mappingproxy。"""

    _values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_values",
            MappingProxyType({str(key): _freeze(item) for key, item in self._values.items()}),
        )

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self):
        return type(self), (dict(self),)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{source} keys must be strings")
    return value


def _non_empty_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return value.strip()


def _path_sequence(
    value: Any,
    source: str,
    carrier_path: str,
    path_policy: _PathPolicy,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source} must be a non-empty path list")
    paths = []
    for index, path in enumerate(value):
        paths.append(
            path_policy.resolve(
                _non_empty_string(path, f"{source}[{index}]"), carrier_path
            )
        )
    return tuple(paths)


def _normalize_enabled_cues(
    enabled_cues: Mapping[str, bool] | Iterable[str],
) -> tuple[str, ...]:
    if isinstance(enabled_cues, Mapping):
        requested = []
        for name, enabled in enabled_cues.items():
            if not isinstance(name, str):
                raise TypeError("enabled cue names must be strings")
            if not isinstance(enabled, bool):
                raise TypeError(f"enabled_cues[{name!r}] must be bool")
            if name not in _CUE_FORMATS:
                raise ValueError(f"unknown cue in enabled_cues: {name!r}")
            if enabled:
                requested.append(name)
    else:
        if isinstance(enabled_cues, (str, bytes)):
            raise TypeError("enabled_cues must be a mapping or an iterable of names")
        requested = list(enabled_cues)
        if any(not isinstance(name, str) for name in requested):
            raise TypeError("enabled cue names must be strings")
        if len(set(requested)) != len(requested):
            raise ValueError("enabled_cues contains duplicate names")
        unknown = sorted(set(requested) - set(_CUE_FORMATS))
        if unknown:
            raise ValueError(f"unknown enabled cues: {unknown}")
    selected = set(requested)
    return tuple(name for name in _CUE_ORDER if name in selected)


def _load_yaml_mapping(path: str) -> Mapping[str, Any]:
    import yaml

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key {key!r} in {path}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    with open(path, encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=UniqueKeySafeLoader)
    if value is None:
        value = {}
    return _mapping(value, f"cues YAML {path!r}")


def _validate_format_metadata(name: str, config: Mapping[str, Any]) -> Mapping[str, Any]:
    allowed_keys = _COMMON_CUE_KEYS | _CUE_SPECIFIC_KEYS[name]
    unknown = sorted(set(config) - allowed_keys)
    if unknown:
        raise ValueError(f"unsupported {name} cue configuration keys: {unknown}")

    metadata: dict[str, Any] = {}
    if "collate" in config:
        metadata["collate"] = dict(_mapping(config["collate"], f"{name}.collate"))

    if name == "spatial":
        fields = config.get("fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
        ):
            raise ValueError("spatial cue requires a non-empty string list 'fields'")
        if len(set(fields)) != len(fields):
            raise ValueError("spatial cue fields must be unique")
        defaults = config.get("field_defaults", {})
        defaults = _mapping(defaults, "spatial.field_defaults")
        unknown_defaults = sorted(set(defaults) - set(fields))
        if unknown_defaults:
            raise ValueError(
                "spatial.field_defaults contains undeclared fields: "
                f"{unknown_defaults}"
            )
        metadata["fields"] = tuple(fields)
        metadata["field_defaults"] = dict(defaults)
    elif name == "textual":
        selection = _mapping(config.get("selection"), "textual.selection")
        selection_type = selection.get("type")
        if selection_type not in {"random", "fixed"}:
            raise ValueError(
                "textual.selection.type must be 'random' or 'fixed', "
                f"got {selection_type!r}"
            )
        if selection_type == "random":
            word_range = selection.get("word_range", [2, 6])
            if (
                not isinstance(word_range, list)
                or len(word_range) != 2
                or any(not isinstance(value, int) for value in word_range)
                or word_range[0] <= 0
                or word_range[1] < word_range[0]
            ):
                raise ValueError("textual random selection requires a valid word_range")
        metadata["selection"] = dict(selection)
    return _freeze(metadata)


def _load_cue_specs(
    cues_yaml: str,
    enabled_cues: tuple[str, ...],
    path_policy: _PathPolicy,
) -> tuple[WeSepCueSpec, ...]:
    root = _load_yaml_mapping(cues_yaml)
    cue_configs = _mapping(root.get("cues"), f"{cues_yaml}:cues")
    specs = []
    for name in enabled_cues:
        if name not in cue_configs:
            raise ValueError(f"enabled cue {name!r} is not declared in {cues_yaml}")
        config = _mapping(cue_configs[name], f"{cues_yaml}:cues.{name}")
        default_type, default_format = _DEFAULT_CUE_FORMATS.get(name, (None, None))
        cue_type = config.get("type", default_type)
        cue_format = config.get("format", default_format)
        if not isinstance(cue_type, str) or not isinstance(cue_format, str):
            raise ValueError(f"{name} cue must declare string type and format")
        if (cue_type, cue_format) not in _CUE_FORMATS[name]:
            raise ValueError(
                f"unsupported {name} cue type/format: "
                f"{cue_type!r}/{cue_format!r}"
            )
        scope = config.get("scope", "speaker")
        if scope != "speaker":
            raise ValueError(
                f"unsupported {name} cue scope {scope!r}; only 'speaker' is valid"
            )
        guaranteed = config.get("guaranteed", True)
        if not isinstance(guaranteed, bool):
            raise ValueError(f"{name}.guaranteed must be bool")

        policy = _mapping(config.get("policy"), f"{name}.policy")
        unknown_policy_keys = sorted(set(policy) - {"type", "key", "resource"})
        if unknown_policy_keys:
            raise ValueError(
                f"unsupported {name}.policy keys: {unknown_policy_keys}"
            )
        policy_type = policy.get("type")
        if policy_type not in _CUE_POLICIES[name]:
            raise ValueError(
                f"unsupported {name} cue policy.type {policy_type!r}"
            )
        key_field = policy.get("key")
        if key_field not in {"spk_id", "mix_spk_id"}:
            raise ValueError(f"unsupported {name} cue policy.key {key_field!r}")
        resource_path = path_policy.resolve(
            _non_empty_string(policy.get("resource"), f"{name}.policy.resource"),
            cues_yaml,
        )
        specs.append(
            WeSepCueSpec(
                name=name,
                cue_type=cue_type,
                cue_format=cue_format,
                scope=scope,
                guaranteed=guaranteed,
                policy_type=policy_type,
                key_field=key_field,
                resource_path=resource_path,
                format_metadata=_validate_format_metadata(name, config),
            )
        )
    return tuple(specs)


def _load_cue_resources(
    specs: Sequence[WeSepCueSpec],
) -> dict[str, Mapping[str, Any]]:
    resources = {}
    for spec in specs:
        value = _load_json_file(spec.resource_path)
        resources[spec.name] = _mapping(
            value, f"{spec.name} cue resource {spec.resource_path!r}"
        )
    return resources


def _validate_textual_item(
    item: Mapping[str, Any],
    spec: WeSepCueSpec,
    source: str,
) -> None:
    labels = item.get("phn_label")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{source}.phn_label must be a non-empty word list")
    for word_index, word in enumerate(labels):
        if not isinstance(word, list) or not word:
            raise ValueError(
                f"{source}.phn_label[{word_index}] must be a non-empty ID list"
            )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in word):
            raise ValueError(
                f"{source}.phn_label[{word_index}] must contain integer IDs"
            )
    selection = spec.format_metadata["selection"]
    if selection["type"] == "fixed":
        candidates = item.get("kw_candidate")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{source}.kw_candidate must be a non-empty index list")
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(labels)
            for index in candidates
        ):
            raise ValueError(f"{source}.kw_candidate contains an invalid word index")


def _normalize_cue_items(
    spec: WeSepCueSpec,
    lookup_key: str,
    value: Any,
    path_policy: _PathPolicy,
) -> tuple[Mapping[str, Any], ...]:
    source = f"{spec.name} cue {lookup_key!r}"
    if spec.name in {"audio", "visual"}:
        if not isinstance(value, list):
            raise ValueError(f"{source} must be a list of resource items")
        raw_items = value
    else:
        if not isinstance(value, Mapping):
            raise ValueError(f"{source} must be a resource item mapping")
        raw_items = [value]

    items = []
    for index, raw_item in enumerate(raw_items):
        item = dict(_mapping(raw_item, f"{source}[{index}]"))
        if spec.name == "textual":
            _validate_textual_item(item, spec, f"{source}[{index}]")
        else:
            item["path"] = path_policy.resolve(
                _non_empty_string(item.get("path"), f"{source}[{index}].path"),
                spec.resource_path,
            )
        if "duration_sec" in item:
            duration = item["duration_sec"]
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                raise ValueError(f"{source}[{index}].duration_sec must be numeric")
            if duration <= 0:
                raise ValueError(f"{source}[{index}].duration_sec must be positive")
        if spec.name == "visual" and "fps" in item:
            fps = item["fps"]
            if not isinstance(fps, (int, float)) or isinstance(fps, bool):
                raise ValueError(f"{source}[{index}].fps must be numeric")
            if fps <= 0:
                raise ValueError(f"{source}[{index}].fps must be positive")
        items.append(_freeze(item))
    return tuple(items)


def _attach_cues(
    sample_key: str,
    speaker_id: str,
    specs: Sequence[WeSepCueSpec],
    resources: Mapping[str, Mapping[str, Any]],
    path_policy: _PathPolicy,
) -> tuple[WeSepCueRecord, ...]:
    records = []
    for spec in specs:
        if spec.key_field == "spk_id":
            lookup_key = speaker_id
        else:
            lookup_key = f"{sample_key}::{speaker_id}"
        resource = resources[spec.name]
        if lookup_key not in resource:
            if spec.guaranteed:
                raise KeyError(
                    f"guaranteed {spec.name} cue is missing for {lookup_key!r}"
                )
            records.append(
                WeSepCueRecord(
                    spec=spec, lookup_key=lookup_key, present=False, items=()
                )
            )
            continue
        items = _normalize_cue_items(
            spec, lookup_key, resource[lookup_key], path_policy
        )
        if not items:
            if spec.guaranteed:
                raise ValueError(
                    f"guaranteed {spec.name} cue is empty for {lookup_key!r}"
                )
            records.append(
                WeSepCueRecord(
                    spec=spec, lookup_key=lookup_key, present=False, items=()
                )
            )
            continue
        records.append(
            WeSepCueRecord(
                spec=spec, lookup_key=lookup_key, present=True, items=items
            )
        )
    return tuple(records)


def _selection_for_sample(
    selection: str | Mapping[str, Sequence[str]],
    sample_key: str,
    speakers: Sequence[str],
) -> tuple[str, ...]:
    if selection == "all":
        return tuple(speakers)
    if selection == "first":
        return (speakers[0],)
    if not isinstance(selection, Mapping):
        raise ValueError(
            "target_selection must be 'all', 'first', or a sample-to-speakers mapping"
        )
    if sample_key not in selection:
        raise ValueError(f"target_selection has no entry for sample {sample_key!r}")
    selected = selection[sample_key]
    if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
        raise ValueError(
            f"target_selection[{sample_key!r}] must be a speaker sequence"
        )
    if not selected:
        raise ValueError(f"target_selection[{sample_key!r}] must not be empty")
    if any(not isinstance(speaker, str) or not speaker for speaker in selected):
        raise ValueError(
            f"target_selection[{sample_key!r}] contains an invalid speaker"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(
            f"target_selection[{sample_key!r}] contains duplicate speakers"
        )
    unknown = sorted(set(selected) - set(speakers))
    if unknown:
        raise ValueError(
            f"target_selection[{sample_key!r}] contains unknown speakers: {unknown}"
        )
    selected_set = set(selected)
    return tuple(speaker for speaker in speakers if speaker in selected_set)


def load_wesep_records(
    manifest_path: str | os.PathLike[str],
    *,
    manifest_format: str,
    path_base: str,
    mix_variant: str,
    target_selection: str | Mapping[str, Sequence[str]],
    recipe_root: str | os.PathLike[str] | None = None,
    cues_yaml: str | os.PathLike[str] | None = None,
    enabled_cues: Mapping[str, bool] | Iterable[str] = (),
) -> tuple[WeSepSampleRecord, ...]:
    """严格解析 WeSep samples/cues 索引，不解码或探测资源文件。

    ``manifest_format`` 明确区分 JSONL 内容与 JSONL 路径列表；文件后缀不参与
    判断。``path_base`` 明确选择逐载体文件相对路径或固定 ``recipe_root``，不做
    文件存在性回退。目标默认也不隐式选择，调用方必须传入 ``all``、``first``
    或按 sample key 指定 speaker 的映射。
    """

    manifest_path = _absolute_input_path(manifest_path)
    path_policy = _PathPolicy.build(path_base, recipe_root)
    mix_variant = _non_empty_string(mix_variant, "mix_variant")
    enabled = _normalize_enabled_cues(enabled_cues)
    if enabled and cues_yaml is None:
        raise ValueError("cues_yaml is required when cues are enabled")

    specs: tuple[WeSepCueSpec, ...] = ()
    resources: dict[str, Mapping[str, Any]] = {}
    if enabled:
        cues_yaml_path = _absolute_input_path(cues_yaml)
        specs = _load_cue_specs(cues_yaml_path, enabled, path_policy)
        resources = _load_cue_resources(specs)

    raw_records = _read_manifest_records(
        manifest_path, manifest_format, path_policy
    )
    result = []
    seen_keys = set()
    for index, (raw_sample, sample_manifest) in enumerate(raw_records):
        source = f"sample[{index}] in {sample_manifest}"
        key = _non_empty_string(raw_sample.get("key"), f"{source}.key")
        if key in seen_keys:
            raise ValueError(f"duplicate sample key: {key!r}")
        seen_keys.add(key)

        speakers_value = raw_sample.get("spk")
        if not isinstance(speakers_value, list) or not speakers_value:
            raise ValueError(f"{source}.spk must be a non-empty list")
        speakers = tuple(
            _non_empty_string(speaker, f"{source}.spk[{slot}]")
            for slot, speaker in enumerate(speakers_value)
        )
        if len(set(speakers)) != len(speakers):
            raise ValueError(f"{source}.spk contains duplicate speaker IDs")

        mix = _mapping(raw_sample.get("mix"), f"{source}.mix")
        if mix_variant not in mix:
            raise ValueError(
                f"{source}.mix has no requested variant {mix_variant!r}"
            )
        mixture_paths = _path_sequence(
            mix[mix_variant],
            f"{source}.mix[{mix_variant!r}]",
            sample_manifest,
            path_policy,
        )
        sources = _mapping(raw_sample.get("src"), f"{source}.src")
        selected_speakers = _selection_for_sample(
            target_selection, key, speakers
        )
        targets = []
        for speaker in selected_speakers:
            if speaker not in sources:
                raise ValueError(f"{source}.src has no target for speaker {speaker!r}")
            source_paths = _path_sequence(
                sources[speaker],
                f"{source}.src[{speaker!r}]",
                sample_manifest,
                path_policy,
            )
            targets.append(
                WeSepTargetRecord(
                    speaker_id=speaker,
                    source_paths=source_paths,
                    cues=_attach_cues(
                        key, speaker, specs, resources, path_policy
                    ),
                )
            )
        result.append(
            WeSepSampleRecord(
                key=key,
                manifest_path=sample_manifest,
                mix_variant=mix_variant,
                mixture_paths=mixture_paths,
                targets=tuple(targets),
            )
        )

    if isinstance(target_selection, Mapping):
        unknown_sample_keys = sorted(set(target_selection) - seen_keys)
        if unknown_sample_keys:
            raise ValueError(
                "target_selection contains unknown sample keys: "
                f"{unknown_sample_keys}"
            )
    return tuple(result)


__all__ = [
    "MANIFEST_FORMAT_JSONL",
    "MANIFEST_FORMAT_PATH_LIST",
    "PATH_BASE_MANIFEST_RELATIVE",
    "PATH_BASE_RECIPE_ROOT",
    "WeSepCueRecord",
    "WeSepCueSpec",
    "WeSepSampleRecord",
    "WeSepTargetRecord",
    "load_wesep_records",
]
