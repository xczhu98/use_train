"""多 GPU 分片推理编排。

该模块只负责把模型推理拆到独立 worker；指标、延迟、ONNX 和 badcase
仍由调用方使用单一 ``--device`` 执行。worker 的 manifest/scp 在父进程
完成严格校验后才会发布到正式输出目录。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils.inference_artifacts import (
    InferenceResult,
    build_input_inventory,
    build_run_manifest,
    fingerprint_file,
    fingerprint_json,
    load_run_manifest,
    inference_output_lock,
    publish_inference_metadata,
    stat_audio_identity,
)
from evaluate.eval_io import read_scp, validate_inference_artifacts
from core.inference import FileInferenceItem, InferenceReport, InferenceSpec, ReusePolicy
from evaluate.process_runner import EVAL_SUBPROCESS_TIMEOUT_S, death_signal_preexec
from tools.shared.scp_io import serialize_scp_entries, write_scp_entries

logger = logging.getLogger("multi_gpu_inference")


class MultiGPUInferenceError(RuntimeError):
    """多 GPU worker 或最终合并校验失败。"""


def validate_inference_devices(devices: str | Iterable[str]) -> list[str]:
    """校验并规范化推理设备列表；设备名称必须明确且不能重复。"""
    if isinstance(devices, str):
        values = [item.strip() for item in devices.split(",")]
    else:
        values = [str(item).strip() for item in devices]
    if not values or any(not item for item in values):
        raise ValueError("inference_devices 不能为空，格式为 cuda:0,cuda:1")
    if len(values) != len(set(values)):
        raise ValueError(f"inference_devices 存在重复设备: {values}")
    for device in values:
        if device == "cpu":
            continue
        if not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError(f"不支持的推理设备 {device!r}；请使用 cpu 或 cuda:N")
    return values


def normalize_inference_devices(
    devices: str | Iterable[str] | None, *, fallback: str = "cuda:0"
) -> list[str]:
    """将可选 CLI 值归一化，并独立校验指标阶段的单设备 fallback。"""
    fallback_values = validate_inference_devices(fallback)
    if len(fallback_values) != 1:
        raise ValueError("fallback device 必须是单个 cpu 或 cuda:N")
    return validate_inference_devices(
        fallback_values if devices is None else devices
    )


def shard_uids(uids: Iterable[str], num_shards: int) -> list[list[str]]:
    """按排序后的 UID 使用稳定 round-robin 分片。"""
    if num_shards <= 0:
        raise ValueError("num_shards 必须为正整数")
    ordered = sorted(str(uid) for uid in uids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("输入 UID 存在重复项")
    return [ordered[index::num_shards] for index in range(num_shards)]


def _resolve_map(path: str | Path) -> dict[str, Path]:
    return read_scp(path)


def _write_subset_scp(path: Path, mapping: Mapping[str, Path], uids: list[str]) -> None:
    entries = [(uid, str(mapping[uid])) for uid in uids]
    write_scp_entries(entries, path)


def _worker_command(
    mode: str,
    config: Path,
    checkpoint: Path,
    worker_dir: Path,
    device: str,
    input_paths: Mapping[str, Path],
    far_paths: Mapping[str, Path] | None = None,
    *,
    sr: int | None = None,
    no_stream_align: bool = False,
    allow_partial: bool = False,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "-C",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(worker_dir),
        "--device",
        device,
    ]
    if expected_effective_config_identity is not None:
        expected_sha256 = (
            expected_effective_config_identity
            if isinstance(expected_effective_config_identity, str)
            else expected_effective_config_identity["sha256"]
        )
        command += ["--expected-effective-config-sha256", str(expected_sha256)]
    if mode == "se":
        command += ["--input-scp", str(input_paths["noisy"])]
    elif mode == "aec_paired":
        command += [
            "--input-mic-scp",
            str(input_paths["mic"]),
            "--input-far-scp",
            str((far_paths or {})["farend"]),
        ]
    elif mode == "aec_zero_farend":
        command += ["--input-scp", str(input_paths["noisy"])]
    else:  # pragma: no cover - internal callers only
        raise ValueError(f"unknown worker mode: {mode}")
    if sr is not None:
        command += ["--sr", str(sr)]
    if no_stream_align:
        command.append("--aec_no_stream_align")
    if allow_partial:
        command.append("--allow-partial")
    return command


def _run_worker(command: list[str]) -> None:
    """启动单个 worker；worker 退出码必须明确反映其执行状态。

    本函数由 ThreadPoolExecutor 工作线程调用，禁止使用 preexec_fn（fork 死锁风险）；
    超时由 subprocess.run 直接杀子进程，其孙进程经 worker 主线程拉起处的 pdeathsig 链回收。
    """
    logger.info("启动推理 worker: device=%s", command[command.index("--device") + 1])
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=EVAL_SUBPROCESS_TIMEOUT_S,
        start_new_session=True,
    )
    if completed.returncode != 0:
        raise MultiGPUInferenceError(
            f"worker 失败 (exit={completed.returncode}, device={command[command.index('--device') + 1]})"
        )


def _compatible_fingerprint(first: dict, current: dict) -> bool:
    """比较 worker 内容指纹；允许每卡 requested/effective device 参数不同。"""
    left = deepcopy(first)
    right = deepcopy(current)
    for value in (left, right):
        parameters = value.get("parameters", {})
        parameters.pop("requested_device", None)
        parameters.pop("effective_device", None)
        value["sha256"] = fingerprint_json(
            {
                "algorithm": value["algorithm"],
                "files": value["files"],
                "parameters": parameters,
            }
        )
    return left == right


def _multi_fingerprint(source: dict, devices: list[str]) -> dict:
    """为合并结果绑定多卡编排源码和实际设备列表。"""
    result = deepcopy(source)
    files = result.setdefault("files", {})
    files["multi_gpu_inference"] = fingerprint_file(Path(__file__))
    parameters = result.setdefault("parameters", {})
    parameters.pop("requested_device", None)
    parameters.pop("effective_device", None)
    parameters["inference_devices"] = list(devices)
    payload = {
        "algorithm": result["algorithm"],
        "files": files,
        "parameters": parameters,
    }
    result["sha256"] = fingerprint_json(payload)
    return result


def _validate_artifact_fingerprint_current(artifact: Mapping[str, object]) -> None:
    """确认 worker 使用的每个 artifact 文件仍与当前文件内容一致。"""
    files = artifact.get("files")
    if not isinstance(files, Mapping) or not files:
        raise MultiGPUInferenceError("worker artifact fingerprint 缺少 files")
    for name, expected in files.items():
        if expected is None:
            continue
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise MultiGPUInferenceError(f"worker artifact fingerprint 非法: {name}")
        try:
            current = fingerprint_file(expected["path"])
        except (OSError, ValueError, RuntimeError) as exc:
            raise MultiGPUInferenceError(
                f"worker artifact 文件不可复核: {name}={expected['path']}"
            ) from exc
        if current != dict(expected):
            raise MultiGPUInferenceError(f"worker artifact 文件已变化: {name}")


def _validate_input_inventory_current(input_inventory: Mapping[str, object]) -> None:
    """在发布前重新检查输入 identity，避免推理期间输入被替换。"""
    items = input_inventory.get("items")
    if not isinstance(items, list):
        raise MultiGPUInferenceError("input inventory 缺少 items")
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("inputs"), Mapping):
            raise MultiGPUInferenceError("input inventory 结构非法")
        for role, expected in item["inputs"].items():
            if expected is None:
                continue
            if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
                raise MultiGPUInferenceError(f"input inventory 非法: {role}")
            try:
                current = stat_audio_identity(expected["path"])
            except (OSError, ValueError, RuntimeError) as exc:
                raise MultiGPUInferenceError(
                    f"input 在发布前不可复核: {expected['path']}"
                ) from exc
            if current != dict(expected):
                raise MultiGPUInferenceError(
                    f"input 在推理期间发生变化: {expected['path']}"
                )


def _copy_to_exclusive_temporary(source: Path, destination: Path) -> Path:
    """以不可预测 O_EXCL 文件名复制到正式目录，并在返回前 fsync。"""
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as output_stream:
            with source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _validate_published_results_current(results: Iterable[InferenceResult]) -> None:
    for result in results:
        if result.status not in {"ok", "skipped"}:
            continue
        path = Path(result.output_path or "")
        try:
            current = stat_audio_identity(path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise MultiGPUInferenceError(
                f"正式输出在 manifest 发布前不可复核: {result.uid}"
            ) from exc
        if current != result.output_identity:
            raise MultiGPUInferenceError(
                f"正式输出在 manifest 发布前发生变化: {result.uid}"
            )


def _validate_worker(
    worker_dir: Path,
    config: Path,
    checkpoint: Path,
    expected_inventory: dict,
    *,
    allow_partial: bool,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> tuple[dict, dict[str, InferenceResult]]:
    try:
        manifest, results = validate_inference_artifacts(
            worker_dir,
            config=config,
            checkpoint=checkpoint,
            expected_effective_config_identity=expected_effective_config_identity,
        )
        if manifest["input_inventory"] != expected_inventory:
            raise MultiGPUInferenceError("worker input inventory 与分片输入不一致")
        if manifest["failed_uids"] and not allow_partial:
            raise MultiGPUInferenceError(
                f"worker manifest 含失败 UID: {manifest['failed_uids'][:5]}"
            )
        worker_root = worker_dir.resolve()
        for result in results:
            if result.status not in {"ok", "skipped"}:
                continue
            output = Path(result.output_path or "")
            if output.is_symlink() or not output.is_file():
                raise MultiGPUInferenceError(
                    f"worker 输出必须是 worker 目录内的普通文件: {result.uid}"
                )
            try:
                output.resolve().relative_to(worker_root)
            except ValueError as exc:
                raise MultiGPUInferenceError(
                    f"worker 输出越出 worker 目录: {result.uid}"
                ) from exc
    except Exception as exc:
        raise MultiGPUInferenceError(f"worker manifest 校验失败 {worker_dir}: {exc}") from exc
    return manifest, {result.uid: result for result in results}


def _load_reusable_existing_results(
    output_dir: Path,
    input_inventory: Mapping[str, object],
    config: Path,
    checkpoint: Path,
    devices: list[str],
    expected_effective_config_identity: Mapping[str, object] | str | None,
) -> dict[str, InferenceResult]:
    """按公共 manifest 合同读取可复用的既有结果。

    这里是 multi-GPU 编排层唯一的复用判定入口。worker 只处理未复用 UID，
    从而避免父进程和 worker 对同一 UID 分别做 skip-existing 决策。
    """
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest, results = load_run_manifest(
            manifest_path, expected_input_inventory=input_inventory
        )
        if Path(manifest.get("config", "")).resolve() != config.resolve():
            return {}
        if Path(manifest.get("checkpoint", "")).resolve() != checkpoint.resolve():
            return {}
        artifact = manifest.get("artifact_fingerprint")
        if not isinstance(artifact, Mapping):
            return {}
        _validate_artifact_fingerprint_current(artifact)
        parameters = artifact.get("parameters", {})
        if not isinstance(parameters, Mapping):
            return {}
        if list(parameters.get("inference_devices", [])) != list(devices):
            return {}
        actual_identity = parameters.get("effective_config_identity")
        if expected_effective_config_identity is not None:
            expected_sha256 = (
                expected_effective_config_identity
                if isinstance(expected_effective_config_identity, str)
                else expected_effective_config_identity.get("sha256")
            )
            if not isinstance(actual_identity, Mapping) or actual_identity.get("sha256") != expected_sha256:
                return {}
        expected_paths = {
            result.uid: result.output_path
            for result in results
            if result.status in {"ok", "skipped"} and result.output_path
        }
        from core.utils.inference_artifacts import reusable_uids_from_manifest

        reusable = reusable_uids_from_manifest(
            manifest_path,
            expected_artifact_fingerprint=artifact,
            expected_input_inventory=input_inventory,
            expected_output_paths=expected_paths,
        )
        return {
            result.uid: InferenceResult(
                uid=result.uid,
                status="skipped",
                output_path=result.output_path,
                output_identity=result.output_identity,
            )
            for result in results
            if result.uid in reusable
        }
    except Exception as exc:
        logger.info("既有 multi-GPU manifest 不可复用，重新推理: %s", exc)
        return {}


def _publish_merged_unlocked(
    output_dir: Path,
    config: Path,
    checkpoint: Path,
    input_inventory: dict,
    manifests: list[dict],
    worker_results: Mapping[str, InferenceResult],
    devices: list[str],
    *,
    allow_partial: bool,
    worker_roots: Mapping[str, Path],
    reused_uids: set[str] | None = None,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> None:
    if not manifests:
        raise MultiGPUInferenceError("没有可合并的 worker manifest")
    reused_uids = set(reused_uids or ())
    identities = [
        manifest.get("artifact_fingerprint", {})
        .get("parameters", {})
        .get("effective_config_identity")
        for manifest in manifests
    ]
    if any(not isinstance(identity, Mapping) for identity in identities):
        raise MultiGPUInferenceError("worker manifest 缺少 effective config identity")
    first_identity = dict(identities[0])
    if any(dict(identity) != first_identity for identity in identities[1:]):
        raise MultiGPUInferenceError("worker effective config identity 不一致")
    if expected_effective_config_identity is not None:
        expected_sha256 = (
            expected_effective_config_identity
            if isinstance(expected_effective_config_identity, str)
            else expected_effective_config_identity.get("sha256")
        )
        if first_identity.get("sha256") != expected_sha256:
            raise MultiGPUInferenceError("worker effective config identity 与 parent 不一致")
    first = manifests[0]
    if any(
        manifest["config"] != first["config"]
        or manifest["checkpoint"] != first["checkpoint"]
        or not _compatible_fingerprint(first["artifact_fingerprint"], manifest["artifact_fingerprint"])
        for manifest in manifests[1:]
    ):
        raise MultiGPUInferenceError("worker config/checkpoint/artifact fingerprint 不一致")
    _validate_artifact_fingerprint_current(first["artifact_fingerprint"])
    _validate_input_inventory_current(input_inventory)

    expected_uids = [item["uid"] for item in input_inventory["items"]]
    if set(worker_results) != set(expected_uids) or len(worker_results) != len(expected_uids):
        raise MultiGPUInferenceError("worker UID 集合存在重复或缺失")
    if any(result.status not in {"ok", "skipped", "failed"} for result in worker_results.values()):
        raise MultiGPUInferenceError("worker 结果存在非法状态")
    if any(result.status == "failed" for result in worker_results.values()) and not allow_partial:
        raise MultiGPUInferenceError("worker 结果存在失败 UID，未启用 allow_partial")

    with tempfile.TemporaryDirectory(prefix="multi_gpu_merge_", dir=str(output_dir.parent)) as temp:
        stage = Path(temp)
        staged_by_uid: dict[str, InferenceResult] = {}
        for uid in expected_uids:
            result = worker_results[uid]
            if result.status == "failed":
                staged_by_uid[uid] = result
                continue
            if uid in reused_uids:
                staged_by_uid[uid] = result
                continue
            source_path = Path(result.output_path or "")
            worker_root = worker_roots.get(uid)
            if worker_root is None and uid not in reused_uids:
                raise MultiGPUInferenceError(f"worker UID 未绑定 worker 目录: {uid}")
            if source_path.is_symlink() or not source_path.is_file():
                raise MultiGPUInferenceError(f"worker 输出不存在或不是普通文件: {uid}")
            source = source_path.resolve()
            if worker_root is not None:
                worker_root = worker_root.resolve()
                try:
                    source.relative_to(worker_root)
                except ValueError as exc:
                    raise MultiGPUInferenceError(f"worker 输出越出 worker 目录: {uid}") from exc
            try:
                source_identity = stat_audio_identity(source_path)
            except (OSError, ValueError, RuntimeError) as exc:
                raise MultiGPUInferenceError(f"worker 输出无法复核: {uid}") from exc
            if source_identity != result.output_identity:
                raise MultiGPUInferenceError(f"worker 输出 identity 已变化: {uid}")
            destination = stage / f"uid-{uid.encode('utf-8').hex()}.wav"
            shutil.copy2(source_path, destination)
            if stat_audio_identity(source_path) != source_identity:
                raise MultiGPUInferenceError(f"worker 输出复制期间发生变化: {uid}")
            staged_by_uid[uid] = InferenceResult(
                uid=uid,
                status=result.status,
                output_path=str(destination.resolve()),
                output_identity=stat_audio_identity(destination),
            )
        artifact = _multi_fingerprint(first["artifact_fingerprint"], devices)
        staged_results = [staged_by_uid[uid] for uid in expected_uids]
        staged_manifest = build_run_manifest(
            config=config,
            checkpoint=checkpoint,
            artifact_fingerprint=artifact,
            input_inventory=input_inventory,
            results=staged_results,
            allow_partial=allow_partial,
            created_at=first["created_at"],
        )
        staged_scp = serialize_scp_entries(
            [(result.uid, result.output_path) for result in staged_results
             if result.status in {"ok", "skipped"}]
        )
        publish_inference_metadata(stage, scp_content=staged_scp, manifest=staged_manifest)

        output_dir.mkdir(parents=True, exist_ok=True)
        published_by_uid: dict[str, InferenceResult] = {
            uid: result for uid, result in worker_results.items() if result.status == "failed"
        }
        replacements: list[dict[str, object]] = []
        try:
            for uid in expected_uids:
                result = staged_by_uid[uid]
                if result.status == "failed":
                    continue
                if uid in reused_uids:
                    published_by_uid[uid] = result
                    continue
                destination = output_dir / Path(result.output_path).name
                temporary: Path | None = None
                replacement: dict[str, object] = {
                    "destination": destination,
                    "temporary": temporary,
                    "backup": None,
                    "installed": False,
                }
                replacements.append(replacement)
                if destination.is_symlink() or (
                    destination.exists() and not destination.is_file()
                ):
                    raise MultiGPUInferenceError(
                        f"正式输出目标必须是普通文件或不存在: {destination}"
                    )
                temporary = _copy_to_exclusive_temporary(
                    Path(result.output_path), destination
                )
                replacement["temporary"] = temporary
                if destination.exists():
                    # 备份用硬链接：destination 在中途始终存在，安装只需一次
                    # os.replace，避免"先搬走旧文件再装新文件"之间的空窗。
                    backup = stage / f"backup-{destination.name}"
                    os.link(destination, backup)
                    replacement["backup"] = backup
                os.replace(temporary, destination)
                replacement["installed"] = True
                published_by_uid[uid] = InferenceResult(
                    uid=uid,
                    status=result.status,
                    output_path=str(destination.resolve()),
                    output_identity=stat_audio_identity(destination),
                )
            final_results = [published_by_uid[uid] for uid in expected_uids]
            _validate_artifact_fingerprint_current(artifact)
            _validate_input_inventory_current(input_inventory)
            _validate_published_results_current(final_results)
            final_manifest = build_run_manifest(
                config=config,
                checkpoint=checkpoint,
                artifact_fingerprint=artifact,
                input_inventory=input_inventory,
                results=final_results,
                allow_partial=allow_partial,
                created_at=first["created_at"],
            )
            _validate_artifact_fingerprint_current(artifact)
            _validate_input_inventory_current(input_inventory)
            _validate_published_results_current(final_results)
            publish_inference_metadata(
                output_dir,
                scp_content=serialize_scp_entries(
                    [(result.uid, result.output_path) for result in final_results
                     if result.status in {"ok", "skipped"}]
                ),
                manifest=final_manifest,
            )
        except BaseException:
            rollback_errors: list[str] = []
            for replacement in reversed(replacements):
                destination = replacement["destination"]
                temporary = replacement["temporary"]
                backup = replacement["backup"]
                try:
                    if replacement["installed"] and isinstance(destination, Path) and destination.exists():
                        destination.unlink()
                    if isinstance(backup, Path) and backup.exists():
                        os.replace(backup, destination)
                    if isinstance(temporary, Path) and temporary.exists():
                        temporary.unlink()
                except Exception as exc:
                    rollback_errors.append(f"{destination}: {exc}")
            if rollback_errors:
                # 回滚异常不得覆盖原始发布错误，但必须显式暴露，不能静默通过。
                logger.error("multi-GPU 发布回滚失败: %s", "; ".join(rollback_errors[:8]))
            raise


def _publish_merged(
    output_dir: Path,
    config: Path,
    checkpoint: Path,
    input_inventory: dict,
    manifests: list[dict],
    worker_results: Mapping[str, InferenceResult],
    devices: list[str],
    *,
    allow_partial: bool = False,
    worker_roots: Mapping[str, Path] | None = None,
    reused_uids: set[str] | None = None,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> None:
    """在正式输出锁内发布合并后的 WAV、SCP 和 manifest。"""
    with inference_output_lock(output_dir) as locked_output:
        _publish_merged_unlocked(
            locked_output,
            config,
            checkpoint,
            input_inventory,
            manifests,
            worker_results,
            devices,
            allow_partial=allow_partial,
            worker_roots=worker_roots or {},
            reused_uids=reused_uids,
            expected_effective_config_identity=expected_effective_config_identity,
        )


def _build_file_inference_spec(
    *,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    input_maps: Mapping[str, Mapping[str, Path]],
    far_maps: Mapping[str, Mapping[str, Path]] | None,
    input_inventory: Mapping[str, object],
    allow_partial: bool,
    skip_existing: bool,
    expected_effective_config_identity: Mapping[str, object] | str | None,
) -> InferenceSpec:
    """把 multi-GPU 父级输入冻结为新的文件推理计划。

    该计划只在父进程构造；worker 接收其 UID 子集，不能重新解释 reuse。
    """
    items: list[FileInferenceItem] = []
    for entry in input_inventory.get("items", []):
        uid = str(entry["uid"])
        inputs: dict[str, Path] = {}
        for role, mapping in input_maps.items():
            inputs[role] = Path(mapping[uid])
        if far_maps:
            for role, mapping in far_maps.items():
                inputs[role] = Path(mapping[uid])
        items.append(
            FileInferenceItem(
                uid=uid,
                inputs=inputs,
                output=output_dir / f"{uid}.wav",
            )
        )
    parameters: dict[str, object] = {
        "execution": "multi_gpu",
        "effective_config_identity": (
            expected_effective_config_identity
            if expected_effective_config_identity is not None
            else None
        ),
    }
    return InferenceSpec(
        config=config,
        checkpoint=checkpoint,
        output_dir=output_dir,
        items=tuple(items),
        reuse=ReusePolicy(enabled=skip_existing),
        allow_partial=allow_partial,
        run_metadata=parameters,
    )


def _report_from_manifest(
    manifest: Mapping[str, object],
    results: Mapping[str, InferenceResult],
    *,
    manifest_path: Path,
    scp_path: Path,
) -> InferenceReport:
    """把已校验 worker manifest 投影为新的 InferenceReport。"""
    ordered = tuple(results[uid] for uid in manifest.get("expected_uids", []) if uid in results)
    failed = sum(result.status == "failed" for result in ordered)
    status = "failed" if failed == len(ordered) and ordered else "partial" if failed else "completed"
    return InferenceReport(
        status=status,
        results=ordered,
        manifest_path=manifest_path,
        scp_path=scp_path,
    )


def _run_multi(
    *,
    mode: str,
    config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    devices: str | Iterable[str],
    input_maps: Mapping[str, Mapping[str, Path]],
    input_inventory: dict,
    far_maps: Mapping[str, Mapping[str, Path]] | None = None,
    sr: int | None = None,
    no_stream_align: bool = False,
    skip_existing: bool = False,
    allow_partial: bool = False,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> Path:
    device_list = validate_inference_devices(devices)
    if len(device_list) < 2:
        raise ValueError("多 GPU 编排至少需要两个推理设备")
    config_path = Path(config).expanduser().resolve(strict=True)
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = _build_file_inference_spec(
        config=config_path,
        checkpoint=checkpoint_path,
        output_dir=output_path,
        input_maps=input_maps,
        far_maps=far_maps,
        input_inventory=input_inventory,
        allow_partial=allow_partial,
        skip_existing=skip_existing,
        expected_effective_config_identity=expected_effective_config_identity,
    )
    expected_uids = [item.uid for item in spec.items]
    if not expected_uids:
        raise ValueError("多 GPU 编排输入 UID 不能为空")
    reusable_results = (
        _load_reusable_existing_results(
            output_path,
            input_inventory,
            config_path,
            checkpoint_path,
            device_list,
            expected_effective_config_identity,
        )
        if skip_existing
        else {}
    )
    reused_uids = set(reusable_results)
    pending_uids = [uid for uid in expected_uids if uid not in reused_uids]
    shards = shard_uids(pending_uids, len(device_list))

    with tempfile.TemporaryDirectory(prefix="multi_gpu_workers_", dir=str(output_path.parent)) as temp:
        root = Path(temp)
        jobs: list[tuple[list[str], Path, dict]] = []
        for index, (device, uids) in enumerate(zip(device_list, shards)):
            if not uids:
                continue
            worker_dir = root / f"worker_{index:02d}"
            worker_dir.mkdir()
            subset_paths: dict[str, Path] = {}
            for role, mapping in input_maps.items():
                subset_scp = root / f"worker_{index:02d}_{role}.scp"
                _write_subset_scp(subset_scp, mapping, uids)
                subset_paths[role] = subset_scp
            far_paths = None
            if far_maps is not None:
                far_paths = {}
                for role, mapping in far_maps.items():
                    subset_scp = root / f"worker_{index:02d}_{role}.scp"
                    _write_subset_scp(subset_scp, mapping, uids)
                    far_paths[role] = subset_scp
            expected = build_input_inventory(
                (
                    uid,
                    {
                        **{role: input_maps[role][uid] for role in input_maps},
                        **({role: far_maps[role][uid] for role in far_maps} if far_maps else {}),
                    },
                )
                for uid in uids
            )
            jobs.append(
                (
                    _worker_command(
                        mode, config_path, checkpoint_path, worker_dir, device,
                        subset_paths, far_paths, sr=sr,
                        no_stream_align=no_stream_align,
                        allow_partial=allow_partial,
                        expected_effective_config_identity=expected_effective_config_identity,
                    ),
                    worker_dir,
                    expected,
                )
            )

        # 独立进程由 inference.py 管理模型生命周期；线程只负责并发等待子进程。
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                futures = [executor.submit(_run_worker, command) for command, _, _ in jobs]
                for future in futures:
                    future.result()

        manifests: list[dict] = []
        merged: dict[str, InferenceResult] = dict(reusable_results)
        worker_roots: dict[str, Path] = {}
        for _, worker_dir, expected in jobs:
            manifest, results = _validate_worker(
                worker_dir,
                config_path,
                checkpoint_path,
                expected,
                allow_partial=allow_partial,
                expected_effective_config_identity=expected_effective_config_identity,
            )
            worker_report = _report_from_manifest(
                manifest,
                results,
                manifest_path=worker_dir / "run_manifest.json",
                scp_path=worker_dir / "inf.scp",
            )
            manifests.append(manifest)
            for result in worker_report.results:
                uid = result.uid
                if uid in merged:
                    raise MultiGPUInferenceError(f"worker UID 重复: {uid}")
                merged[uid] = result
                worker_roots[uid] = worker_dir
        if not manifests:
            try:
                existing_manifest, _ = load_run_manifest(
                    output_path / "run_manifest.json",
                    expected_input_inventory=input_inventory,
                )
            except Exception as exc:
                raise MultiGPUInferenceError(
                    "所有 UID 都被标记为复用，但既有 manifest 无法重新读取"
                ) from exc
            manifests.append(existing_manifest)
        parent_report = InferenceReport(
            status=(
                "partial"
                if any(result.status == "failed" for result in merged.values())
                else "completed"
            ),
            results=tuple(merged[uid] for uid in expected_uids),
            manifest_path=output_path / "run_manifest.json",
            scp_path=output_path / "inf.scp",
        )
        _publish_merged(
            output_path,
            config_path,
            checkpoint_path,
            input_inventory,
            manifests,
            {result.uid: result for result in parent_report.results},
            device_list,
            allow_partial=allow_partial,
            worker_roots=worker_roots,
            reused_uids=reused_uids,
            expected_effective_config_identity=expected_effective_config_identity,
        )
    return output_path


def run_se_inference_multi_gpu(
    config: str | Path,
    checkpoint: str | Path,
    input_scp: str | Path,
    output_dir: str | Path,
    devices: str | Iterable[str],
    *,
    sr: int | None = None,
    skip_existing: bool = False,
    allow_partial: bool = False,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> Path:
    noisy = _resolve_map(input_scp)
    inventory = build_input_inventory((uid, {"noisy": noisy[uid]}) for uid in noisy)
    return _run_multi(
        mode="se", config=config, checkpoint=checkpoint, output_dir=output_dir,
        devices=devices, input_maps={"noisy": noisy}, input_inventory=inventory,
        sr=sr, skip_existing=skip_existing, allow_partial=allow_partial,
        expected_effective_config_identity=expected_effective_config_identity,
    )


def run_aec_paired_inference_multi_gpu(
    config: str | Path,
    checkpoint: str | Path,
    mic_scp: str | Path,
    far_scp: str | Path,
    output_dir: str | Path,
    devices: str | Iterable[str],
    *,
    no_stream_align: bool = False,
    skip_existing: bool = False,
    allow_partial: bool = False,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> Path:
    mic = _resolve_map(mic_scp)
    far = _resolve_map(far_scp)
    if set(mic) != set(far):
        raise MultiGPUInferenceError("AEC mic/farend UID 集合不一致")
    inventory = build_input_inventory(
        (uid, {"mic": mic[uid], "farend": far[uid]}) for uid in mic
    )
    return _run_multi(
        mode="aec_paired", config=config, checkpoint=checkpoint, output_dir=output_dir,
        devices=devices, input_maps={"mic": mic}, far_maps={"farend": far},
        input_inventory=inventory, no_stream_align=no_stream_align,
        skip_existing=skip_existing, allow_partial=allow_partial,
        expected_effective_config_identity=expected_effective_config_identity,
    )


def run_aec_zero_farend_inference_multi_gpu(
    config: str | Path,
    checkpoint: str | Path,
    input_scp: str | Path,
    output_dir: str | Path,
    devices: str | Iterable[str],
    *,
    skip_existing: bool = False,
    allow_partial: bool = False,
    expected_effective_config_identity: Mapping[str, object] | str | None = None,
) -> Path:
    noisy = _resolve_map(input_scp)
    inventory = build_input_inventory((uid, {"noisy": noisy[uid]}) for uid in noisy)
    return _run_multi(
        mode="aec_zero_farend", config=config, checkpoint=checkpoint, output_dir=output_dir,
        devices=devices, input_maps={"noisy": noisy}, input_inventory=inventory,
        skip_existing=skip_existing, allow_partial=allow_partial,
        expected_effective_config_identity=expected_effective_config_identity,
    )


def _validate_worker_device(device: str) -> None:
    """拒绝让显式 CUDA worker 静默降级到 CPU。"""
    if device == "cpu":
        return
    import torch

    if not torch.cuda.is_available():
        raise MultiGPUInferenceError(
            f"worker 请求 {device}，但当前进程没有可用 CUDA；禁止静默降级到 CPU"
        )
    index = int(device.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise MultiGPUInferenceError(
            f"worker 请求 {device}，当前可见 CUDA 设备数为 {torch.cuda.device_count()}"
        )


def _worker_main(args) -> int:
    from evaluate.eval_pipeline_common import (
        run_aec_inference_zero_farend,
    )

    try:
        _validate_worker_device(args.device)
    except MultiGPUInferenceError as exc:
        logger.error("%s", exc)
        return 2

    if args.worker_mode == "aec_zero_farend":
        run_aec_inference_zero_farend(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            input_scp=args.input_scp,
            output_dir=args.output_dir,
            device=args.device,
            allow_partial=args.allow_partial,
            expected_effective_config_identity=args.expected_effective_config_sha256,
        )
        return 0
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "inference.py"),
        "-C", args.config,
        "--checkpoint", args.checkpoint,
        "--output_dir", args.output_dir,
        "--device", args.device,
    ]
    if args.expected_effective_config_sha256 is not None:
        command += [
            "--expected-effective-config-sha256",
            args.expected_effective_config_sha256,
        ]
    if args.allow_partial:
        command.append("--allow-partial")
    if args.worker_mode == "se":
        command += ["--input_scp", args.input_scp]
        if args.sr is not None:
            command += ["--sr", str(args.sr)]
    else:
        command += ["--input_mic_scp", args.input_mic_scp, "--input_far_scp", args.input_far_scp]
        if args.aec_no_stream_align:
            command.append("--aec_no_stream_align")
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=EVAL_SUBPROCESS_TIMEOUT_S,
        start_new_session=True,
        preexec_fn=death_signal_preexec,
    )
    if completed.returncode != 0:
        return completed.returncode
    manifest = json.loads((Path(args.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("failed_uids") and not args.allow_partial:
        return 2
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="multi GPU inference worker")
    parser.add_argument(
        "--worker-mode", choices=["se", "aec_paired", "aec_zero_farend"], required=True
    )
    parser.add_argument("-C", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-effective-config-sha256")
    parser.add_argument("--input-scp")
    parser.add_argument("--input-mic-scp")
    parser.add_argument("--input-far-scp")
    parser.add_argument("--sr", type=int)
    parser.add_argument("--aec_no_stream_align", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    # 分片输入与 worker 模式绑定：缺失时给出用法错误，而不是把 None 拼进子进程命令。
    if args.worker_mode in {"se", "aec_zero_farend"} and not args.input_scp:
        parser.error(f"--worker-mode {args.worker_mode} requires --input-scp")
    if args.worker_mode == "aec_paired" and not (args.input_mic_scp and args.input_far_scp):
        parser.error("--worker-mode aec_paired requires --input-mic-scp and --input-far-scp")
    return _worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
