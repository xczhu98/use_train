"""OfflineDataset 发布后重载门禁。

支持 schema v2 的 SE 与 TSE production adapter。SE 只解析显式绑定 alias 下的
SCP/waveform inventory；TSE 直接构造生产 ``TSEOfflineDataset``，完整读取每条
样本并额外解码所有 enrollment candidate。AEC adapter 仍 fail-closed。正式发布
必须 full scan，probe 永不具发布资格。
"""

from __future__ import annotations

import json
import math
import shlex
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from tools.prepare_data_tools.generation_manifest import (
    ADAPTER_NONE,
    ADAPTER_SE,
    ADAPTER_TSE,
    GenerationManifest,
    GenerationManifestError,
    RequiredGate,
    ResourceReference,
    bind_generation_roots,
    load_generation_manifest,
    resolve_generation_resource,
    validate_generation_root,
    validate_manifest_files,
    validate_relative_resource_path,
)
from tools.shared.scp_io import parse_scp_fields

SCAN_FULL = "full"
SCAN_PROBE = "probe"


class OfflineReloadError(RuntimeError):
    """OfflineDataset 无法构造、完整读取或满足额外 adapter 检查。"""


@dataclass(frozen=True)
class GateReport:
    name: str
    adapter: str
    split: str
    scan_mode: str
    dataset_size: int
    scanned: int
    enrollment_candidates_scanned: int
    publish_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "adapter": self.adapter,
            "split": self.split,
            "scan_mode": self.scan_mode,
            "dataset_size": self.dataset_size,
            "scanned": self.scanned,
            "enrollment_candidates_scanned": self.enrollment_candidates_scanned,
            "publish_eligible": self.publish_eligible,
        }


@dataclass(frozen=True)
class ReloadReport:
    task: str
    scan_mode: str
    publish_eligible: bool
    gates: tuple[GateReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "scan_mode": self.scan_mode,
            "publish_eligible": self.publish_eligible,
            "gates": [gate.to_dict() for gate in self.gates],
        }


def run_offline_reload_gates(
    generation_root: str | Path,
    *,
    manifest: GenerationManifest | None = None,
    probe: int | None = None,
    root_bindings: Mapping[str, str | Path] | None = None,
) -> ReloadReport:
    """构造生产 OfflineDataset 并 full iterate；``probe`` 永不具发布资格。"""
    root = validate_generation_root(generation_root)
    if manifest is None:
        manifest = load_generation_manifest(root, root_bindings=root_bindings)
    roots = bind_generation_roots(root, manifest, root_bindings=root_bindings)
    validate_manifest_files(root, manifest, root_bindings=roots)
    if probe is not None and (type(probe) is not int or probe <= 0):
        raise OfflineReloadError(f"probe 必须是正整数，得到 {probe!r}")
    scan_mode = SCAN_PROBE if probe is not None else SCAN_FULL
    with ExitStack() as resources:
        reports = tuple(
            _run_gate(
                root, manifest, gate, probe=probe, roots=roots, resources=resources
            )
            for gate in manifest.required_gates
        )
    if manifest.task == "tse" and probe is None:
        reloaded = sum(report.dataset_size for report in reports)
        if reloaded != manifest.counts.published:
            raise OfflineReloadError(
                "TSE full reload sample count does not match GenerationManifest: "
                f"reloaded={reloaded}, published={manifest.counts.published}"
            )
    eligible = scan_mode == SCAN_FULL and all(report.publish_eligible for report in reports)
    return ReloadReport(
        task=manifest.task,
        scan_mode=scan_mode,
        publish_eligible=eligible,
        gates=reports,
    )


def require_publish_eligible(report: ReloadReport) -> None:
    """正式 publication seam 的 fail-closed 检查。"""
    if report.scan_mode != SCAN_FULL or not report.publish_eligible:
        raise OfflineReloadError("正式发布只接受通过 full scan 的 reload report")


def _run_gate(
    root: Path,
    manifest: GenerationManifest,
    gate: RequiredGate,
    *,
    probe: int | None,
    roots: Mapping[str, Path],
    resources: ExitStack,
) -> GateReport:
    if gate.adapter == ADAPTER_NONE:
        return GateReport(
            name=gate.name,
            adapter=gate.adapter,
            split=gate.split,
            scan_mode=SCAN_PROBE if probe is not None else SCAN_FULL,
            dataset_size=0,
            scanned=0,
            enrollment_candidates_scanned=0,
            publish_eligible=probe is None,
        )
    if gate.adapter not in {ADAPTER_SE, ADAPTER_TSE}:
        raise OfflineReloadError(
            f"GenerationManifest reload adapter 已下线: {gate.adapter!r}；"
            "仅支持 SE/TSE schema v2"
        )
    expected_task = "se" if gate.adapter == ADAPTER_SE else "tse"
    if manifest.task != expected_task:
        raise OfflineReloadError(
            f"task {manifest.task!r} 与 adapter {gate.adapter!r} 不匹配"
        )
    try:
        dataset = _build_dataset(
            root, manifest, gate, roots=roots, resources=resources
        )
        dataset_size = len(dataset)
        if dataset_size <= 0:
            raise OfflineReloadError(f"gate {gate.name!r} 构造出空 Dataset")
        limit = dataset_size if probe is None else min(probe, dataset_size)
        for index in range(limit):
            dataset[index]
        enrollment_count = _scan_tse_enrollment_candidates(
            dataset, probe=probe
        ) if gate.adapter == ADAPTER_TSE else 0
    except OfflineReloadError:
        raise
    except ImportError as exc:
        raise OfflineReloadError(
            f"adapter {gate.adapter!r} 缺少运行依赖: {exc}"
        ) from exc
    except Exception as exc:
        raise OfflineReloadError(
            f"gate {gate.name!r} adapter {gate.adapter!r} reload 失败: {exc}"
        ) from exc
    full = probe is None and limit == dataset_size
    return GateReport(
        name=gate.name,
        adapter=gate.adapter,
        split=gate.split,
        scan_mode=SCAN_FULL if probe is None else SCAN_PROBE,
        dataset_size=dataset_size,
        scanned=limit,
        enrollment_candidates_scanned=enrollment_count,
        publish_eligible=full,
    )


def _build_dataset(
    root: Path,
    manifest: GenerationManifest,
    gate: RequiredGate,
    *,
    roots: Mapping[str, Path],
    resources: ExitStack,
):
    del root
    if manifest.schema_version != 2:
        raise OfflineReloadError("仅支持 schema v2 reload adapter")
    if gate.adapter == ADAPTER_TSE:
        return _build_tse_dataset(gate, manifest=manifest, roots=roots)
    if gate.adapter != ADAPTER_SE:
        raise OfflineReloadError(f"不支持 reload adapter: {gate.adapter!r}")
    inputs = dict(gate.inputs)
    options = _plain_mapping(gate.options)
    if "strict_load" in options and options["strict_load"] is not True:
        raise OfflineReloadError("reload adapter 强制 strict_load=True")
    options["strict_load"] = True
    _require_input_keys(inputs, required={"clean_scp", "noisy_scp"})
    _validate_options(options, _SE_OPTIONS)
    inventory = set(manifest.inventory_by_resource())
    temporary_root = Path(resources.enter_context(
        tempfile.TemporaryDirectory(prefix="offline-reload-scp-")
    ))
    rewritten = {
        name: _rewrite_multi_root_se_scp(
            roots,
            inputs[name],
            inventory,
            destination=temporary_root / f"{name}.scp",
        )
        for name in ("clean_scp", "noisy_scp")
    }
    from core.data.offline_dataset import OfflineDataset

    return OfflineDataset(
        clean_scp=str(rewritten["clean_scp"]),
        noisy_scp=str(rewritten["noisy_scp"]),
        **options,
    )


def _build_tse_dataset(
    gate: RequiredGate,
    *,
    manifest: GenerationManifest,
    roots: Mapping[str, Path],
):
    """按公开 wesep manifest 构造生产 TSEOfflineDataset。"""
    inputs = dict(gate.inputs)
    options = _plain_mapping(gate.options)
    if "strict_load" in options and options["strict_load"] is not True:
        raise OfflineReloadError("TSE reload adapter 强制 strict_load=True")
    options["strict_load"] = True
    _require_input_keys(
        inputs,
        required={
            "samples_list",
            "samples_jsonl",
            "cues_yaml",
            "cues_resource",
            "source_audio_identity",
        },
    )
    _validate_options(options, _TSE_OPTIONS)
    production_reader_options = {
        "manifest_format": "path_list",
        "path_base": "manifest_relative",
        "target_selection": "all",
        "mix_variant": "default",
        "cue_schema": "realtse_raw",
        "enabled_cues": ["audio"],
    }
    for name, expected in production_reader_options.items():
        if name in options and options[name] != expected:
            raise OfflineReloadError(
                f"TSE reload option {name!r}={options[name]!r} conflicts with "
                f"production artifact schema {expected!r}"
            )
        options[name] = expected
    resolved = {
        name: resolve_generation_resource(
            roots[reference.root_alias],
            reference.path,
            location=f"TSE reload {name}",
        )
        for name, reference in inputs.items()
    }
    from tasks.tse.data.offline_dataset import TSEOfflineDataset

    dataset = TSEOfflineDataset(
        samples_list=str(resolved["samples_list"]),
        cues_yaml=str(resolved["cues_yaml"]),
        **options,
    )
    _verify_tse_carriers(dataset, resolved=resolved)
    _verify_tse_bound_audio(
        dataset,
        identity_path=resolved["source_audio_identity"],
        manifest=manifest,
        roots=roots,
    )
    return dataset


def _verify_tse_carriers(dataset: object, *, resolved: Mapping[str, Path]) -> None:
    """确保 raw.list、records 和 cues.yaml 均绑定 gate 声明的载体。"""
    raw_lines = [
        line.strip()
        for line in resolved["samples_list"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(raw_lines) != 1 or raw_lines[0].startswith("{"):
        raise OfflineReloadError("TSE production raw.list 必须只引用一个 samples.jsonl")
    raw_target = (resolved["samples_list"].parent / raw_lines[0]).resolve(strict=True)
    if raw_target != resolved["samples_jsonl"]:
        raise OfflineReloadError("TSE raw.list 未绑定 gate 声明的 samples_jsonl")

    records = getattr(dataset, "records", None)
    if not isinstance(records, tuple) or not records:
        raise OfflineReloadError("TSE reload dataset 缺少 canonical records")
    record_manifests = {
        Path(getattr(record, "manifest_path", "")).resolve(strict=True)
        for record in records
    }
    if record_manifests != {resolved["samples_jsonl"]}:
        raise OfflineReloadError(
            "TSE canonical records 未绑定 gate 声明的 samples_jsonl"
        )

    import yaml

    try:
        cues = yaml.safe_load(resolved["cues_yaml"].read_text(encoding="utf-8"))
        audio = cues["cues"]["audio"]
        policy = audio["policy"]
        resource_value = policy["resource"]
    except (OSError, UnicodeError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise OfflineReloadError(f"TSE cues.yaml schema 非法: {exc}") from exc
    if audio.get("type") != "raw" or not isinstance(resource_value, str):
        raise OfflineReloadError(
            "TSE production cues.yaml 必须声明 audio type=raw 与 resource"
        )
    resource_path = Path(resource_value)
    if not resource_path.is_absolute():
        resource_path = resolved["cues_yaml"].parent / resource_path
    if resource_path.resolve(strict=True) != resolved["cues_resource"]:
        raise OfflineReloadError("TSE cues.yaml 未绑定 gate 声明的 cues_resource")


def _selected_tse_resource_paths(
    dataset: object,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """从 canonical records 展开全部选中 mixture/target/cue 路径。"""
    records = getattr(dataset, "records", None)
    audio_cue_items = getattr(dataset, "_audio_cue_items", None)
    if not isinstance(records, tuple) or not records or not callable(audio_cue_items):
        raise OfflineReloadError("TSE reload dataset 缺少 canonical resource records")
    mixtures: list[str] = []
    targets: list[str] = []
    enrollments: list[str] = []
    for record in records:
        mixture_paths = getattr(record, "mixture_paths", None)
        target_records = getattr(record, "targets", None)
        if not isinstance(mixture_paths, tuple) or not mixture_paths:
            raise OfflineReloadError("TSE canonical record 缺少 mixture_paths")
        if not isinstance(target_records, tuple) or not target_records:
            raise OfflineReloadError("TSE canonical record 缺少 targets")
        if any(not isinstance(path, str) or not path for path in mixture_paths):
            raise OfflineReloadError("TSE canonical mixture_paths 非法")
        mixtures.extend(mixture_paths)
        for target in target_records:
            source_paths = getattr(target, "source_paths", None)
            if not isinstance(source_paths, tuple) or not source_paths:
                raise OfflineReloadError("TSE canonical target 缺少 source_paths")
            if any(not isinstance(path, str) or not path for path in source_paths):
                raise OfflineReloadError("TSE canonical target source_paths 非法")
            targets.extend(source_paths)
            present, items = audio_cue_items(record, target)
            if type(present) is not bool or not isinstance(items, tuple):
                raise OfflineReloadError("TSE canonical audio cue record 非法")
            if not present or not items:
                raise OfflineReloadError(
                    "TSE production reload requires one or more audio cue candidates"
                )
            for item in items:
                if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                    raise OfflineReloadError("TSE canonical audio cue candidate 非法")
                enrollments.append(item["path"])
    return tuple(mixtures), tuple(targets), tuple(enrollments)


def _verify_tse_bound_audio(
    dataset: object,
    *,
    identity_path: Path,
    manifest: GenerationManifest,
    roots: Mapping[str, Path],
) -> None:
    """复核外部 source 内容身份，并要求内部 target 全部进入 inventory。"""
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflineReloadError(f"TSE source audio identity 无法读取: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "files"}:
        raise OfflineReloadError("TSE source audio identity schema 非法")
    if payload["schema"] != "tse-source-audio-identity-v1" or not isinstance(
        payload["files"], list
    ):
        raise OfflineReloadError("TSE source audio identity schema 非法")
    declared: dict[str, set[str]] = {"mixture": set(), "enrollment": set()}
    seen: set[str] = set()
    from tools.prepare_data_tools.generation_manifest import sha256_file

    for index, item in enumerate(payload["files"]):
        if not isinstance(item, dict) or set(item) != {"path", "roles", "size", "sha256"}:
            raise OfflineReloadError(f"TSE source audio identity files[{index}] 非法")
        path = Path(item["path"])
        roles = item["roles"]
        if str(path.resolve(strict=True)) != str(path) or not isinstance(roles, list):
            raise OfflineReloadError(f"TSE source audio identity files[{index}] 路径/roles 非法")
        if str(path) in seen or not roles or any(role not in declared for role in roles):
            raise OfflineReloadError(f"TSE source audio identity files[{index}] 重复或 role 非法")
        seen.add(str(path))
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise OfflineReloadError(f"TSE source audio content identity changed: {path}")
        for role in roles:
            declared[role].add(str(path))
    mixture_paths, target_paths, enrollment_paths = _selected_tse_resource_paths(
        dataset
    )
    actual_mixtures = {
        str(Path(path).resolve(strict=True)) for path in mixture_paths
    }
    actual_enrollments = {
        str(Path(path).resolve(strict=True)) for path in enrollment_paths
    }
    if not actual_mixtures.issubset(declared["mixture"]) or not actual_enrollments.issubset(
        declared["enrollment"]
    ):
        raise OfflineReloadError("TSE source audio identity does not match dataset sources")

    inventory = set(manifest.inventory_by_resource())
    for path in target_paths:
        absolute = Path(path).resolve(strict=True)
        matches = []
        for alias, root in roots.items():
            try:
                relative = absolute.relative_to(root).as_posix()
            except ValueError:
                continue
            matches.append((alias, relative))
        if len(matches) != 1 or matches[0] not in inventory:
            raise OfflineReloadError(
                f"TSE target is outside declared GenerationManifest inventory: {absolute}"
            )


def _scan_tse_enrollment_candidates(dataset: object, *, probe: int | None) -> int:
    """解码全部（或 probe 数量）选中 enrollment，避免随机 policy 漏检。"""
    _, _, enrollment_paths = _selected_tse_resource_paths(dataset)
    unique_paths = list(dict.fromkeys(enrollment_paths))
    limit = len(unique_paths) if probe is None else min(probe, len(unique_paths))
    from core.data.offline_multitrack import ensure_finite, ensure_mono, load_audio

    for path in unique_paths[:limit]:
        waveform, sample_rate = load_audio(path)
        waveform = ensure_finite(
            ensure_mono(waveform),
            strict=True,
            track="enrollment",
            sample=path,
        )
        if sample_rate <= 0 or waveform.shape[-1] <= 0:
            raise OfflineReloadError(f"TSE enrollment candidate 为空或采样率非法: {path}")
    return limit


def _rewrite_multi_root_se_scp(
    roots: Mapping[str, Path],
    scp_reference: ResourceReference,
    inventory: set[tuple[str, str]],
    *,
    destination: Path,
) -> Path:
    scp_path = resolve_generation_resource(
        roots[scp_reference.root_alias], scp_reference.path, location="offline SCP"
    )
    rewritten: list[str] = []
    with scp_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            uid, fs, length, raw_audio = parse_scp_fields(
                line, scp_path, lineno, allowed_columns=(4,)
            )
            if raw_audio.startswith("/"):
                raise OfflineReloadError(
                    f"{scp_reference.path}:{lineno} 不允许绝对 waveform 路径"
                )
            if ":" not in raw_audio:
                raise OfflineReloadError(
                    f"{scp_reference.path}:{lineno} 缺少受控 root alias"
                )
            alias, relative = raw_audio.split(":", 1)
            if alias not in roots:
                raise OfflineReloadError(
                    f"{scp_reference.path}:{lineno} 使用未知 root alias: {alias!r}"
                )
            try:
                relative = validate_relative_resource_path(
                    relative,
                    location=f"{scp_reference.path}:{lineno} waveform path",
                )
                waveform = resolve_generation_resource(
                    roots[alias], relative, location="multi-root waveform"
                )
            except GenerationManifestError as exc:
                raise OfflineReloadError(str(exc)) from exc
            if (alias, relative) not in inventory:
                raise OfflineReloadError(
                    f"{scp_reference.path}:{lineno} waveform 未在 inventory 声明: "
                    f"{alias}:{relative}"
                )
            rewritten.append(
                " ".join(
                    shlex.quote(value)
                    for value in (uid, fs, length, str(waveform))
                )
            )
    if not rewritten:
        raise OfflineReloadError(f"{scp_reference.path}: SCP 没有有效记录")
    destination.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return destination


_SE_OPTIONS = frozenset(
    {
        "strict_load",
        "target_sr",
        "max_duration",
        "max_duration_seconds",
        "repeat_to_max_duration",
        "repeat_tile_silence_gap",
        "repeat_tile_silence_gap_range",
        "filter_by_duration",
        "filter_min_duration",
        "filter_max_duration",
        "placeholder_num_samples",
        "seed",
    }
)
_TSE_OPTIONS = frozenset(
    {
        "strict_load",
        "target_sr",
        "enroll_policy",
        "seed",
        "manifest_format",
        "path_base",
        "target_selection",
        "mix_variant",
        "cue_schema",
        "enabled_cues",
    }
)
def _validate_options(options: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise OfflineReloadError(f"reload options 包含未知字段: {unknown}")


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        return item

    return thaw(value)


def _require_input_keys(
    inputs: Mapping[str, str], *, required: set[str], optional: set[str] | None = None
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(inputs))
    unknown = sorted(set(inputs) - allowed)
    if missing or unknown:
        raise OfflineReloadError(
            f"adapter inputs 非法: missing={missing}, unknown={unknown}"
        )


__all__ = [
    "GateReport",
    "OfflineReloadError",
    "ReloadReport",
    "SCAN_FULL",
    "SCAN_PROBE",
    "require_publish_eligible",
    "run_offline_reload_gates",
]
