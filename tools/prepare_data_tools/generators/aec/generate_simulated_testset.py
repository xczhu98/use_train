#!/usr/bin/env python3
"""Simulate AEC testset for 7 scenarios using dynamic_aec pipeline config.

Reads SCP/json paths from config (speech, noise, rir, wind, farend/echo).
Builds one reusable Dataset from a publication-only config copy with
``strict_load=True``, binds the CLI seed with ``set_epoch(0, base_seed=seed)``,
then generates each scenario through the atomic
``get_sample_with_trace(index, force_branch=..., capture_stems=True)`` seam.
Branch probabilities remain training configuration, not a verification control plane.

Scenario semantics (mic = model input, nearend = supervision target):
  - prob_target_only          : mic = clean_target; nearend = clean_target
  - prob_noise_only           : mic = additive_noise; nearend = 0
  - prob_echo_only            : mic = echo; nearend = 0
  - prob_noise_echo_only      : mic = noise + echo; nearend = 0
  - prob_target_noise         : mic = full_RIR_speech + noise; nearend = clean_target
  - prob_target_echo          : mic = clean_target + echo; nearend = clean_target
  - prob_target_noise_echo    : mic = full_RIR_speech + noise + echo; nearend = clean_target

Saved tracks per sample:
  - mic.wav          : microphone signal (what the model sees at inference)
  - nearend.wav      : near-end clean target (supervision reference)
  - farend.wav       : far-end reference signal
  - noisy_se.wav     : clean + noise (SE-style reference, no echo; for SE-only eval)
  - echo.wav         : oracle echo stem (final_echo, echo branches only; for DT ERLE)
  - meta.json        : simulation metadata including branch info and layer params

NOTE: ``TracedSample.sample`` preserves the canonical final waveform contract:
its first five items are nearend/mic/farend/fs/length. Echo branches already
contain echo in ``mic`` before tuple assembly; assembly only aligns and packages
those signals. Intermediate stems and branch facts are consumed from the same
sealed ``SimulationTrace``, so no mutable sidecar can be paired with another sample.

Output structure::

    output_root/
      prob_target_only/
        sample_000/
          mic.wav
          nearend.wav
          farend.wav
          noisy_se.wav
          meta.json
        ...
      prob_noise_only/
        ...
      prob_echo_only/
        ...
      prob_noise_echo_only/
        ...
      prob_target_noise/
        ...
      prob_target_echo/
        ...
      prob_target_noise_echo/
        ...

Usage::

    python -m tools.prepare_data_tools.generators.aec.generate_simulated_testset \\
        -C configs/aec/microsoft_challenge/cagcrn/cagcrn_16k_16ms_delay80ms_loss_mse_sisnr_l1.yaml \\
        --output-dir output/aec_testset_sim \\
        --num-samples 10 \\
        --seed 42

发布状态：GenerationManifest v1 publication 已下线，非 ``--dry-run`` 调用在触碰输出目录前显式
fail-closed。迁移计划：AEC schema v2 reload 就绪后，本入口的发布改走
``tools.prepare_data_tools.offline_publication.publish_owned_offline_dataset_roots``；迁移完成前
下方的骨架保留仅供内部测试，不得作为生产发布路径。
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from tools.prepare_data_tools._common import _git_commit, ensure_project_root  # noqa: E402
from tools.shared.scp_io import read_scp  # noqa: E402

ensure_project_root()

# 生成器身份采用 canonical 脚本名；版本只随数据内容或正式发布语义变化。
GENERATOR_ID = "generate_simulated_testset.py"
# 2.1: 新增 echo.wav（oracle echo stem，JAE-H08）。
# 2.2: echo.wav 改用 final_echo（与实际混入 mic 的回声同源；layer_d_echo 不同源弃用）。
# 2.3: 发布改为 owned staging + 完整 inventory，source fingerprint 绑定实际内容。
# 2.4: 适配 AEC tuple/fs/length、审计 schema 与 Layer A fixed silence canvas 内容语义。
# 2.5: 采用 DynamicMixingDataset 的 post-RIR early target 音量标定语义。
# 2.6: 纳入 Phase 1C Layer E 结构化 metadata 与 Phase 1D ERL 行为/metadata 语义。
# 2.7: 接入 GenerationManifest、static gate、owned publication 与发布 receipt。
# 2.8: 固化 D-normalize trace、GPU/CPU clipping selector 与 AEC jitter 轨迹的发布内容语义。
# 2.9: random_eq 改为平滑、因果且累计增益有界的 peaking-biquad 级联。
# 2.10: 固化 post-mix D/E/F clipping 分层与 deterministic final guard 内容语义。
# 2.11: Layer D 发布 metadata 采用 normalize → guard 嵌套 facts。
# 2.12: Layer D guard 改为 canonical 五键，D-normalize 与 farend guard 独立发布。
# 2.13: Layer D 发布 schema 收敛为嵌套的 normalize → guard 聚合 facts。
# 2.14: Pink/Brown colored noise 改用基于物理 Hz 频率轴的功率律 PSD。
# 2.15: 仿真分层收敛为 Layer A-F；post-F realized 改为顶层审计节点。
# 2.19: 动态数据集离线 RNG 改为 seed/split/epoch/index/stream 派生，不再依赖
# DataLoader worker 生命周期，逐样本增强序列因此改变。
GENERATOR_VERSION_HISTORY = (
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.13", "2.14", "2.15", "2.16", "2.17", "2.18", "2.19"
)
GENERATOR_VERSION = GENERATOR_VERSION_HISTORY[-1]
_TRACE_BRANCH_MISSING = object()

# Dataset ``force_branch`` 接受的 7 个 canonical Layer-B 场景。
from core.data.simulation.scene_names import AEC_SCENE_ORDER, AEC_ECHO_SCENES

SCENARIOS = AEC_SCENE_ORDER

# 业务 manifest schema；GenerationManifest publication 已下线。
MANIFEST_SCHEMA_VERSION = 2
# 音频 payload 不参与业务 fingerprint、复用或发布校验；仅保留路径集合与类型。
_AUDIO_PAYLOAD_SUFFIXES = frozenset({".wav", ".flac", ".rir"})

# manifest 严格比对字段：任一不一致即拒绝复用（fail closed，JAE-C01）。
# git_commit 仅作谱系记录、不一致只告警：仓库任意提交都改变 HEAD，而测试集结构
# 实际由 config/source manifest/seed/样本数/生成器版本决定。
MANIFEST_STRICT_KEYS = (
    "manifest_schema_version",
    "generator_version",
    "config_fingerprint",
    "source_fingerprints",
    "seed",
    "num_samples",
)
OWNER_FILE = ".generate_simulated_testset.owner.json"
OWNER_SCHEMA = 1
PUBLICATION_OWNER_ID = "tasks/aec/generate_simulated_testset.py"
_PUBLICATION_CONTROL_FILES = frozenset()


def _file_fingerprint(path: str | os.PathLike) -> dict[str, Any]:
    """读取单个稳定非音频文件快照；读取期间变化时 fail closed。"""
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.suffix.lower() in _AUDIO_PAYLOAD_SUFFIXES:
        raise ValueError(f"禁止对音频 payload 计算 fingerprint: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"文件在扫描与读取之间被替换: {resolved}")
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        after_open = os.fstat(stream.fileno())
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_open = (
        after_open.st_dev,
        after_open.st_ino,
        after_open.st_size,
        after_open.st_mtime_ns,
    )
    if identity_before != identity_after or identity_before != identity_open:
        raise RuntimeError(f"文件在读取期间发生变化: {resolved}")
    return {
        "path": str(resolved),
        "device": before.st_dev,
        "inode": before.st_ino,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _resolve_manifest_entries(manifest: Path) -> list[Path]:
    """解析 source manifest 的路径集合，不打开或读取任何音频/RIR payload。"""
    suffix = manifest.suffix.lower()
    raw_paths: list[str] = []
    if suffix in {".scp", ".list"}:
        raw_paths = read_scp(manifest)
    elif suffix in {".json", ".jsonl"}:
        payloads: list[Any]
        if suffix == ".jsonl":
            payloads = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payloads = [json.loads(manifest.read_text(encoding="utf-8"))]

        for payload in payloads:
            if isinstance(payload, dict) and "pairs" in payload:
                unknown = set(payload) - {"pairs", "meta"}
                if unknown:
                    raise ValueError(
                        f"canonical echo-pair manifest has unknown keys: {sorted(unknown)!r}"
                    )
                if "meta" in payload and not isinstance(payload["meta"], dict):
                    raise ValueError("canonical echo-pair manifest meta must be an object")
                pairs = payload["pairs"]
                if not isinstance(pairs, list):
                    raise ValueError("canonical echo-pair manifest pairs must be a list")
                for index, pair in enumerate(pairs):
                    if not isinstance(pair, str) or not pair.strip():
                        raise ValueError(
                            "canonical echo-pair manifest "
                            f"pairs[{index}] must be a non-empty stereo path string"
                        )
                    raw_paths.append(pair.strip())
                continue

            def collect(value: Any, key: str = "") -> None:
                if isinstance(value, dict):
                    for child_key, child in value.items():
                        collect(child, str(child_key))
                elif isinstance(value, list):
                    for child in value:
                        collect(child, key)
                elif isinstance(value, str) and (
                    key.endswith("path")
                    or key in {"farend", "echo", "nearend", "mic", "rir"}
                ):
                    raw_paths.append(value)

            collect(payload)
    resolved: list[Path] = []
    for raw in raw_paths:
        candidate = Path(raw).expanduser()
        resolved.append(
            candidate.resolve(strict=True)
            if candidate.is_absolute()
            else (manifest.parent / candidate).resolve(strict=True)
        )
    return sorted(set(resolved), key=lambda path: str(path))


def _source_fingerprint(
    path: str | os.PathLike,
    *,
    resolve_entries: bool = True,
) -> dict[str, Any]:
    """记录 source manifest 内容及其路径集合，但绝不读取 entry payload。"""
    manifest = Path(path).expanduser().resolve(strict=True)
    before = _file_fingerprint(manifest)
    entries = _resolve_manifest_entries(manifest) if resolve_entries else []
    after = _file_fingerprint(manifest)
    if before != after:
        raise RuntimeError(f"source manifest 在扫描与复核之间发生变化: {manifest}")
    return {
        "manifest": before,
        "entries": [str(entry) for entry in entries],
    }


def _source_fingerprints(resources: dict[str, str]) -> dict[str, dict[str, Any]]:
    """构造全部结构资源快照；speech length sidecar 不按音频清单解析。"""
    return {
        name: _source_fingerprint(
            path,
            resolve_entries=name != "speech_length_file",
        )
        for name, path in resources.items()
    }


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    """记录业务发布文件集合；共享 publication 控制文件由 envelope 管理。"""
    files: dict[str, dict[str, Any]] = {}
    excluded = {"manifest.json", OWNER_FILE} | _PUBLICATION_CONTROL_FILES
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"inventory 拒绝符号链接: {path}")
        if not path.is_file() or path.name in excluded:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in _AUDIO_PAYLOAD_SUFFIXES:
            files[rel] = {"kind": "waveform"}
            continue
        fingerprint = _file_fingerprint(path)
        files[rel] = {
            "kind": "metadata",
            "size": fingerprint["size"],
            "sha256": fingerprint["sha256"],
        }
    return files


def _validate_published(root: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    manifest_path = root / "manifest.json"
    owner_path = root / OWNER_FILE
    if root.is_symlink() or not root.is_dir():
        return False, "发布根目录不是普通目录"
    if manifest_path.is_symlink() or owner_path.is_symlink():
        return False, "发布元数据不允许使用符号链接"
    if not manifest_path.is_file() or not owner_path.is_file():
        return False, "缺少 manifest.json 或 generator ownership marker"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"发布元数据不可读: {exc}"
    if not isinstance(manifest, dict):
        return False, "manifest 顶层必须是 mapping"
    if owner != {"generator": GENERATOR_ID, "schema_version": OWNER_SCHEMA}:
        return False, "generator ownership marker 不匹配"
    if manifest.get("generator") != GENERATOR_ID:
        return False, "manifest generator identity 不匹配"
    mismatched = [key for key in MANIFEST_STRICT_KEYS if manifest.get(key) != expected.get(key)]
    if mismatched:
        return False, "manifest 严格字段不一致: " + ", ".join(mismatched)
    declared = manifest.get("inventory")
    if not isinstance(declared, dict):
        return False, "manifest inventory 缺失或非法"
    try:
        actual = _inventory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"发布目录 inventory 不安全或不可读: {exc}"
    if actual != declared:
        return False, "manifest inventory 与真实文件集合或非音频 metadata 不一致"
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in _PUBLICATION_CONTROL_FILES
    }
    if actual_files != set(declared) | {"manifest.json", OWNER_FILE}:
        return False, "发布目录包含未声明业务文件或缺少声明文件"
    if manifest.get("git_commit") != expected.get("git_commit"):
        print(
            f"[WARN] 测试集生成时的 commit ({manifest.get('git_commit')}) 与当前 HEAD "
            f"({expected.get('git_commit')}) 不同；结构指纹一致，继续复用。"
        )
    return True, "manifest schema and file inventory verified"


def _is_owned(root: Path) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
        owner_path = root / OWNER_FILE
        if owner_path.is_symlink() or not owner_path.is_file():
            return False
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        return owner == {"generator": GENERATOR_ID, "schema_version": OWNER_SCHEMA}
    except (OSError, json.JSONDecodeError):
        return False


@contextmanager
def _output_lock(output_root: Path) -> Iterator[None]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.{GENERATOR_ID}.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _remove_owned_tree(path: Path) -> None:
    if not _is_owned(path):
        raise RuntimeError(f"拒绝清理 ownership 不匹配的目录: {path}")
    shutil.rmtree(path)


def _publish_staging(staging: Path, output_root: Path) -> None:
    backup = output_root.parent / f".{output_root.name}.{GENERATOR_ID}.backup"
    if backup.exists() or backup.is_symlink():
        if not _is_owned(backup):
            raise RuntimeError(f"backup ownership 不匹配，拒绝处理: {backup}")
        if output_root.exists() or output_root.is_symlink():
            _remove_owned_tree(backup)
        else:
            os.replace(backup, output_root)
    if output_root.exists() or output_root.is_symlink():
        if not _is_owned(output_root):
            raise RuntimeError(f"输出目录不属于 {GENERATOR_ID}，拒绝替换: {output_root}")
        os.replace(output_root, backup)
    try:
        os.replace(staging, output_root)
    except BaseException:
        if backup.exists() and not output_root.exists() and not output_root.is_symlink():
            os.replace(backup, output_root)
        raise
    if backup.exists():
        try:
            _remove_owned_tree(backup)
        except (OSError, RuntimeError) as exc:
            print(
                f"[WARN] 新集合已提交，但旧 owned backup 清理失败: {backup}: {exc}",
                file=sys.stderr,
            )


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _stable_scenario_seed(base_seed: int, scenario_name: str) -> int:
    """跨进程稳定的场景种子（Python str hash 受 PYTHONHASHSEED 影响，禁止再用）。"""
    digest = hashlib.sha256(scenario_name.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 10000


def _torch_to_numpy_f32(t) -> np.ndarray:
    """tensor/数组 → float32 一维 ndarray（sf.write 需要一维 numpy 数组）。"""
    if hasattr(t, "numpy"):
        return t.detach().cpu().numpy().astype(np.float32).ravel()
    return np.asarray(t, dtype=np.float32).ravel()


def _scalar_int(value: Any, name: str) -> int:
    """把 Python/NumPy/Torch 标量转换为 int，并拒绝非标量和非有限值。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"AEC sample {name} must be scalar, got shape={array.shape}")
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar) or not scalar.is_integer():
        raise ValueError(f"AEC sample {name} must be a finite integer, got {scalar!r}")
    return int(scalar)


def _build_seeded_dataset(dataset_cls, dataset_kwargs: dict[str, Any], base_seed: int):
    """以发布专用严格配置构造 dataset，并固定逐 index augmentation RNG。"""
    publication_kwargs = copy.deepcopy(dataset_kwargs)
    # 只修改生成器副本：训练 YAML 即使允许 fallback，发布也必须拒绝静音占位。
    publication_kwargs["strict_load"] = True
    publication_kwargs["export_simulation_stems"] = True
    dataset = dataset_cls(**publication_kwargs)
    dataset.set_epoch(0, base_seed=base_seed)
    return dataset


def _require_branch(value: Any, expected: str, field: str, index: int) -> None:
    """强制 branch 审计字段存在且与请求场景完全一致。"""
    if value != expected:
        raise RuntimeError(
            f"AEC forced sample idx {index} {field}={value!r}, expected branch={expected!r}"
        )


_CANONICAL_GUARD_KEYS = {"roles", "threshold", "peak", "scale", "applied"}
_CANONICAL_D_NORMALIZE_KEYS = {
    "mode",
    "target_peak",
    "scale",
    "preserves_decomposition",
}


def _require_exact_fact_keys(
    value: Any, *, expected: set[str], path: str
) -> dict[str, Any]:
    """要求发布事实 mapping 严格匹配 canonical 键集合。"""
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{path} must contain exactly {sorted(expected)}; got {sorted(actual)}"
        )
    return value


def _require_layer_d_facts(
    layers: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """只接受 AEC Layer A-F 分层及嵌套的 canonical Layer D facts。"""
    required_layers = {"A", "B", "C", "D", "E", "F"}
    unexpected_layers = sorted(set(layers) - required_layers)
    if unexpected_layers:
        raise RuntimeError(
            "simulation_by_layer must contain only Layers A-F; "
            f"unexpected={unexpected_layers}"
        )
    missing_layers = sorted(required_layers - set(layers))
    if missing_layers:
        missing_text = ", ".join(f"simulation_by_layer.{name}" for name in missing_layers)
        raise RuntimeError(
            "missing canonical simulation layer(s): "
            f"{missing_text}; expected Layers A-F"
        )
    invalid_layers = sorted(
        name for name in required_layers if not isinstance(layers.get(name), dict)
    )
    if invalid_layers:
        invalid_text = ", ".join(f"simulation_by_layer.{name}" for name in invalid_layers)
        raise RuntimeError(
            "canonical simulation layer(s) must be mappings: "
            f"{invalid_text}; expected Layers A-F"
        )
    if "D-normalize" in layers:
        raise RuntimeError("simulation_by_layer.D-normalize is forbidden")
    aec_context = layers.get("aec_context")
    if isinstance(aec_context, dict) and "farend_guard" in aec_context:
        raise RuntimeError("simulation_by_layer.aec_context.farend_guard is forbidden")
    layer_d = _require_exact_fact_keys(
        layers.get("D"),
        expected={"stage_order", "normalize", "guard"},
        path="simulation_by_layer.D",
    )
    if tuple(layer_d["stage_order"]) != ("D-normalize", "D-guard"):
        raise RuntimeError(
            "simulation_by_layer.D.stage_order must be "
            "('D-normalize', 'D-guard')"
        )
    normalize = _require_exact_fact_keys(
        layer_d["normalize"],
        expected=_CANONICAL_D_NORMALIZE_KEYS,
        path="simulation_by_layer.D.normalize",
    )
    guard = _require_exact_fact_keys(
        layer_d["guard"],
        expected={"mic_related", "farend"},
        path="simulation_by_layer.D.guard",
    )
    mic_related = _require_exact_fact_keys(
        guard["mic_related"],
        expected=_CANONICAL_GUARD_KEYS,
        path="simulation_by_layer.D.guard.mic_related",
    )
    farend = _require_exact_fact_keys(
        guard["farend"],
        expected=_CANONICAL_GUARD_KEYS,
        path="simulation_by_layer.D.guard.farend",
    )
    return normalize, mic_related, farend


def _validate_forced_sample(
    dataset: Any,
    scenario_name: str,
    index: int,
    sample: tuple | list,
    stems: Any,
    *,
    actual_branch: Any = _TRACE_BRANCH_MISSING,
) -> dict[str, Any]:
    """验证 forced sample 的分支、波形与 canonical stems 发布契约。"""
    if not isinstance(stems, dict):
        raise RuntimeError(
            f"AEC forced sample stems 缺失或非法: {scenario_name} idx {index}"
        )
    layers = stems.get("simulation_by_layer")
    if not isinstance(layers, dict):
        raise RuntimeError(
            f"AEC forced sample stems 缺少 simulation_by_layer: {scenario_name} idx {index}"
        )
    if "realized" in layers:
        raise RuntimeError(
            "AEC forced sample stems 使用旧 schema "
            "simulation_by_layer.realized；请迁移为顶层 realized"
        )
    try:
        _require_layer_d_facts(layers)
    except RuntimeError as exc:
        raise RuntimeError(
            f"AEC forced sample stems 仿真分层非法（仅允许 Layer A-F）: "
            f"{scenario_name} idx {index}; {exc}"
        ) from exc
    layer_b = layers["B"]
    realized_post_f = stems.get("realized")
    if not isinstance(realized_post_f, dict):
        raise RuntimeError(
            f"AEC forced sample stems 缺少顶层 realized（post-F 审计）: "
            f"{scenario_name} idx {index}"
        )
    realized_final = stems.get("realized_final")
    if not isinstance(realized_final, dict):
        raise RuntimeError(
            f"AEC forced sample stems 缺少 realized_final: {scenario_name} idx {index}"
        )

    _require_branch(
        layer_b.get("force_branch"), scenario_name,
        "simulation_by_layer.B.force_branch", index,
    )
    _require_branch(
        layer_b.get("branch"), scenario_name,
        "simulation_by_layer.B.branch", index,
    )
    _require_branch(
        realized_post_f.get("branch"), scenario_name,
        "realized.branch (post-F)", index,
    )
    _require_branch(
        realized_final.get("branch"), scenario_name,
        "realized_final.branch", index,
    )
    _require_branch(
        actual_branch
        if actual_branch is not _TRACE_BRANCH_MISSING
        else getattr(dataset, "_last_run_branch", None),
        scenario_name,
        "simulation_trace.branch",
        index,
    )

    nearend, mic, farend, _sample_rate, length = _unpack_aec_sample(sample)
    if length <= 0 or not any(np.any(waveform != 0.0) for waveform in (nearend, mic, farend)):
        raise RuntimeError(
            f"AEC forced sample 是静音 placeholder: {scenario_name} idx {index}"
        )
    for name, waveform in (("nearend", nearend), ("mic", mic), ("farend", farend)):
        if not np.all(np.isfinite(waveform)):
            raise RuntimeError(
                f"AEC forced sample {name} 含 NaN/Inf: {scenario_name} idx {index}"
            )

    required_arrays = (
        "layer_d_noisy",
        "layer_d_clean",
        "layer_d_noise",
        "layer_d_echo",
        "farend_ref",
        "final_noise",
        "final_clean",
        "final_noisy",
    )
    for stem_name in required_arrays:
        stem = stems.get(stem_name)
        if not isinstance(stem, np.ndarray) or stem.size < length:
            size = stem.size if isinstance(stem, np.ndarray) else None
            if stem_name == "layer_d_noisy":
                raise RuntimeError(
                    "AEC forced sample stems violate canonical Layer-D contract: "
                    f"layer_d_noisy requires size>={length}, got size={size}; "
                    f"{scenario_name} idx {index}"
                )
            raise RuntimeError(
                f"AEC forced sample stems 不完整: {stem_name} size={size}, "
                f"expected>={length}; {scenario_name} idx {index}"
            )
        if not np.all(np.isfinite(stem[:length])):
            raise RuntimeError(
                f"AEC forced sample stems 含 NaN/Inf: {stem_name}; "
                f"{scenario_name} idx {index}"
            )

    echo_scenarios = AEC_ECHO_SCENES
    final_echo = stems.get("final_echo")
    if scenario_name in echo_scenarios:
        if (
            not isinstance(final_echo, np.ndarray)
            or final_echo.size < length
            or not np.all(np.isfinite(final_echo[:length]))
            or not np.any(final_echo[:length] != 0.0)
        ):
            raise RuntimeError(
                f"AEC forced sample stems 缺少有效 final_echo: {scenario_name} idx {index}"
            )
    elif final_echo is not None:
        raise RuntimeError(
            f"AEC forced sample stems 含陈旧 final_echo: {scenario_name} idx {index}"
        )
    return stems


def _iter_forced_scenario_samples(
    dataset: Any,
    scenario_name: str,
    *,
    num_samples: int,
    base_seed: int,
) -> Iterator[tuple[int, Any, dict[str, Any]]]:
    """稳定选取 index，并通过 Dataset 验证 seam 强制单个 Layer-B 场景。"""
    total = len(dataset)
    if total <= 0:
        raise RuntimeError(f"Dataset is empty for {scenario_name}")
    scenario_rng = random.Random(_stable_scenario_seed(base_seed, scenario_name))
    indices = scenario_rng.sample(range(total), min(num_samples, total))
    if not indices:
        raise RuntimeError(f"未选择到样本: {scenario_name}")
    for index in indices:
        get_traced = getattr(dataset, "get_sample_with_trace", None)
        if callable(get_traced):
            traced = get_traced(
                index, force_branch=scenario_name, capture_stems=True
            )
            sample = traced.sample
            trace_stems = traced.trace.legacy_stems("scene")
            actual_branch = traced.trace.metadata.get("branch")
        else:
            dataset._last_simulation_stems = None
            dataset._last_run_branch = None
            sample = dataset._getitem_simulation_verify(
                index, force_branch=scenario_name
            )
            trace_stems = getattr(dataset, "_last_simulation_stems", None)
            actual_branch = getattr(dataset, "_last_run_branch", None)
        stems = _validate_forced_sample(
            dataset,
            scenario_name,
            int(index),
            sample,
            trace_stems,
            actual_branch=actual_branch,
        )
        yield int(index), sample, stems


def _unpack_aec_sample(sample: tuple | list) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """消费 AEC dataset tuple 的 canonical 前五项。"""
    supported_lengths = {5, 6, 7, 10, 11, 12, 13}
    size = len(sample) if isinstance(sample, (tuple, list)) else "non-tuple"
    if not isinstance(sample, (tuple, list)) or size not in supported_lengths:
        raise ValueError(
            "AEC dataset sample must contain 5, 6, 7, 10, 11, 12, or 13 "
            f"items; got {size}"
        )
    nearend_t, mic_t, farend_t, fs_t, lengths_t = sample[:5]
    nearend = _torch_to_numpy_f32(nearend_t)
    mic = _torch_to_numpy_f32(mic_t)
    farend = _torch_to_numpy_f32(farend_t)
    sample_rate = _scalar_int(fs_t, "fs")
    length = _scalar_int(lengths_t, "lengths")
    if sample_rate <= 0:
        raise ValueError(f"AEC sample fs must be positive, got {sample_rate}")
    if length < 0:
        raise ValueError(f"AEC sample lengths must be non-negative, got {length}")
    available = min(nearend.size, mic.size, farend.size)
    if length > available:
        raise ValueError(
            f"AEC sample lengths={length} exceeds shortest waveform length={available}"
        )
    return nearend[:length], mic[:length], farend[:length], sample_rate, length


def _finite_float_or_none(value: Any) -> float | None:
    """返回有限 float；缺失、非数值或 NaN/Inf 统一记为 ``None``。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _json_finite(value: Any) -> Any:
    """递归复制审计结构，并把 JSON 非有限浮点值规范化为 null。"""
    if isinstance(value, dict):
        return {str(key): _json_finite(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, np.generic):
        return _json_finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return copy.deepcopy(value)


def _build_sample_metadata(
    scenario_name: str,
    dataset_index: int,
    sample_rate: int,
    sample_length: int,
    stems: dict,
) -> dict[str, Any]:
    """从 canonical stems 契约构造无歧义的 sample metadata。"""
    raw_layers = stems.get("simulation_by_layer")
    if not isinstance(raw_layers, dict):
        raise RuntimeError("AEC stems 缺少 simulation_by_layer")
    if "realized" in raw_layers:
        raise RuntimeError(
            "AEC stems 使用旧 schema simulation_by_layer.realized；"
            "请迁移为顶层 realized"
        )
    _require_layer_d_facts(raw_layers)
    simulation_by_layer = _json_finite(raw_layers)
    layer_b = simulation_by_layer.get("B") or {}
    layer_c = simulation_by_layer.get("C") or {}
    layer_e = simulation_by_layer.get("E") or {}
    raw_realized = stems.get("realized")
    if not isinstance(raw_realized, dict):
        raise RuntimeError("AEC stems 缺少顶层 realized（post-F 审计）")
    realized = _json_finite(raw_realized)
    raw_realized_final = stems.get("realized_final")
    realized_final = (
        _json_finite(raw_realized_final)
        if isinstance(raw_realized_final, dict)
        else None
    )
    acoustic_audit = layer_c.get("acoustic_audit")
    if not isinstance(acoustic_audit, dict):
        acoustic_audit = None
    source_volume_calibration = layer_c.get("source_volume_calibration")
    if not isinstance(source_volume_calibration, dict):
        source_volume_calibration = None
    ser_drawn = _finite_float_or_none(layer_c.get("c3_ser_db_drawn"))
    candidate = (acoustic_audit or {}).get("candidate") or {}
    nominal = (acoustic_audit or {}).get("nominal") or {}
    snr_primary_candidate = _finite_float_or_none(candidate.get("primary_snr_db"))
    if snr_primary_candidate is None:
        snr_primary_candidate = _finite_float_or_none(
            layer_c.get("c1_c2_snr_primary_drawn_db")
        )
    snr_wind_candidate = _finite_float_or_none(candidate.get("wind_snr_db"))
    if snr_wind_candidate is None:
        snr_wind_candidate = _finite_float_or_none(
            layer_c.get("c1_c2_wind_noise_snr_candidate_drawn_db")
        )
    snr_selected_nominal = _finite_float_or_none(
        nominal.get("selected_primary_snr_db")
    )
    if snr_selected_nominal is None:
        snr_selected_nominal = _finite_float_or_none(
            layer_c.get("c1_c2_nominal_snr_db")
        )
    ser_realized = _finite_float_or_none((realized or {}).get("ser_db"))
    snr_realized = _finite_float_or_none((realized or {}).get("snr_db"))
    branch = layer_b.get("branch")
    if not isinstance(branch, str) or not branch:
        raise RuntimeError("AEC stems 缺少 simulation_by_layer.B.branch")
    if branch != scenario_name:
        raise RuntimeError(
            f"AEC stems simulation_by_layer.B.branch={branch!r}, "
            f"expected scenario={scenario_name!r}"
        )
    if "ser_applicable" in realized:
        ser_applicable = realized.get("ser_applicable") is True
    else:
        ser_applicable = branch in {"prob_target_echo", "prob_target_noise_echo"}
    if "snr_applicable" in realized:
        snr_applicable = realized.get("snr_applicable") is True
    else:
        snr_applicable = branch in {"prob_target_noise", "prob_target_noise_echo"}

    return {
        "scenario": scenario_name,
        "dataset_index": int(dataset_index),
        "sample_rate": int(sample_rate),
        "length_samples": int(sample_length),
        "branch": branch,
        "ser_db": ser_drawn if ser_applicable else None,
        "snr_db": snr_selected_nominal if snr_applicable else None,
        "ser_db_drawn": ser_drawn,
        "ser_applicable": ser_applicable,
        "ser_db_realized": ser_realized,
        "snr_db_drawn": snr_primary_candidate,
        "snr_db_primary_candidate": snr_primary_candidate,
        "snr_db_wind_candidate": snr_wind_candidate,
        "snr_db_selected_nominal": snr_selected_nominal,
        "snr_applicable": snr_applicable,
        "snr_db_realized": snr_realized,
        "reverb_enabled": layer_c.get("c0_reverb_sampled"),
        "need_echo": layer_c.get("need_echo_manifest"),
        "need_noise": layer_c.get("need_additive_noise"),
        "echo_pair_meta": layer_c.get("c3_disk_pair_load"),
        "channel_aug_count": layer_e.get("num_augmentations_drawn"),
        "acoustic_audit": acoustic_audit,
        "source_volume_calibration": source_volume_calibration,
        "realized": realized,
        "realized_final": realized_final,
        "simulation_by_layer": simulation_by_layer,
    }


def _finite_stats(values: list[Any]) -> dict[str, float | int | None]:
    """汇总有限值；空集合仍返回稳定 schema。"""
    finite = [value for raw in values if (value := _finite_float_or_none(raw)) is not None]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": float(np.mean(finite)),
    }


def _summarize_audit_metadata(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """生成 manifest 级 audit coverage、有限统计和 calibration 状态计数。"""
    total = len(metadata)
    acoustic_present = sum(isinstance(item.get("acoustic_audit"), dict) for item in metadata)
    realized_present = sum(isinstance(item.get("realized"), dict) for item in metadata)
    realized_final_present = sum(
        isinstance(item.get("realized_final"), dict) for item in metadata
    )
    calibration_present = sum(
        isinstance(item.get("source_volume_calibration"), dict) for item in metadata
    )
    decomposable_counts = {"true": 0, "false": 0, "missing": 0}
    calibration_status_counts: dict[str, int] = {}
    calibration_reason_counts: dict[str, int] = {}
    calibration_gain_clipped_counts = {"true": 0, "false": 0, "missing": 0}
    color_counts: dict[str, int] = {}
    for item in metadata:
        realized = item.get("realized")
        decomposable = realized.get("decomposable") if isinstance(realized, dict) else None
        key = "true" if decomposable is True else "false" if decomposable is False else "missing"
        decomposable_counts[key] += 1
        source_calibration = item.get("source_volume_calibration")
        status = (
            source_calibration.get("status")
            if isinstance(source_calibration, dict)
            else None
        )
        status_key = str(status) if isinstance(status, str) and status else "missing"
        calibration_status_counts[status_key] = (
            calibration_status_counts.get(status_key, 0) + 1
        )
        reason = (
            source_calibration.get("reason")
            if isinstance(source_calibration, dict)
            else None
        )
        reason_key = str(reason) if isinstance(reason, str) and reason else "none"
        calibration_reason_counts[reason_key] = (
            calibration_reason_counts.get(reason_key, 0) + 1
        )
        gain_clipped = (
            source_calibration.get("gain_clipped")
            if isinstance(source_calibration, dict)
            else None
        )
        gain_clipped_key = (
            "true" if gain_clipped is True
            else "false" if gain_clipped is False
            else "missing"
        )
        calibration_gain_clipped_counts[gain_clipped_key] += 1
        audit = item.get("acoustic_audit")
        colored = (
            ((audit.get("calibration") or {}).get("colored_noise"))
            if isinstance(audit, dict)
            else None
        )
        color = colored.get("color") if isinstance(colored, dict) else None
        if isinstance(color, str) and color:
            color_counts[color] = color_counts.get(color, 0) + 1

    def coverage(count: int) -> dict[str, float | int]:
        return {"count": count, "fraction": float(count / total) if total else 0.0}

    finite_stats = {
        field: _finite_stats([item.get(field) for item in metadata])
        for field in (
            "ser_db_drawn",
            "ser_db_realized",
            "snr_db_drawn",
            "snr_db_primary_candidate",
            "snr_db_wind_candidate",
            "snr_db_selected_nominal",
            "snr_db_realized",
        )
    }
    calibration_fields = {
        "source_volume_target_requested_db": "target_requested_db",
        "source_volume_realized_after_db": "realized_after_db",
        "source_volume_gain": "gain",
    }
    for summary_field, calibration_field in calibration_fields.items():
        finite_stats[summary_field] = _finite_stats(
            [
                calibration.get(calibration_field)
                if isinstance(calibration := item.get("source_volume_calibration"), dict)
                else None
                for item in metadata
            ]
        )

    return {
        "total_samples": total,
        "audit_coverage": {
            "acoustic_audit": coverage(acoustic_present),
            "source_volume_calibration": coverage(calibration_present),
            "realized": coverage(realized_present),
            "realized_final": coverage(realized_final_present),
            "ser_applicable": coverage(sum(item.get("ser_applicable") is True for item in metadata)),
            "snr_applicable": coverage(sum(item.get("snr_applicable") is True for item in metadata)),
            "decomposable": decomposable_counts,
        },
        "finite_stats": finite_stats,
        "source_volume_calibration": {
            "status_counts": dict(sorted(calibration_status_counts.items())),
            "reason_counts": dict(sorted(calibration_reason_counts.items())),
            "gain_clipped_counts": calibration_gain_clipped_counts,
        },
        "color_counts": dict(sorted(color_counts.items())),
    }


def _write_wav_f32(path: Path, waveform: Any, sample_rate: int, length: int) -> None:
    """按 canonical tuple length 写单声道 float32 WAV，短 stem 直接拒绝。"""
    array = _torch_to_numpy_f32(waveform)
    if array.size < length:
        raise ValueError(
            f"AEC export waveform {path.name} has {array.size} samples, expected {length}"
        )
    sf.write(str(path), array[:length], sample_rate, subtype="FLOAT")


def _export_noisy_se(stems: dict, sample_dir: Path, target_sr: int, length: int) -> None:
    """按 tuple length 导出 canonical Layer-D 无回声 SE 参考。"""
    arr_noisy_ld = stems.get("layer_d_noisy")
    if not isinstance(arr_noisy_ld, np.ndarray):
        raise RuntimeError(
            "AEC canonical Layer-D contract requires layer_d_noisy ndarray"
        )
    arr_echo_ld = stems.get("layer_d_echo")
    if isinstance(arr_echo_ld, np.ndarray):
        if min(arr_noisy_ld.size, arr_echo_ld.size) < length:
            raise ValueError("AEC noisy_se stems are shorter than tuple length")
        noisy_se = arr_noisy_ld[:length] - arr_echo_ld[:length]
        _write_wav_f32(sample_dir / "noisy_se.wav", noisy_se, target_sr, length)
    else:
        # 无回声分支（如 prob_target_noise）。
        _write_wav_f32(sample_dir / "noisy_se.wav", arr_noisy_ld, target_sr, length)


def _export_echo(stems: dict, sample_dir: Path, target_sr: int, length: int) -> None:
    """导出与最终 tuple 对齐的 oracle echo stem。

    ``final_echo`` 由 dataset 的实际 AEC disk context 在 tuple 组装时捕获，
    与 mic 使用同一 echo 源及窗口；Layer E/F 令 mic 不可分解时，其适用性由
    ``realized_final.decomposable`` 显式审计。无回声分支不导出。
    """
    arr_final_echo = stems.get("final_echo")
    if (
        isinstance(arr_final_echo, np.ndarray)
        and arr_final_echo.size > 0
        and float(np.abs(arr_final_echo).max()) > 0.0
    ):
        _write_wav_f32(sample_dir / "echo.wav", arr_final_echo, target_sr, length)


def _export_aec_sample(
    scenario_name: str,
    dataset_index: int,
    sample: tuple | list,
    stems: dict,
    sample_dir: Path,
) -> dict[str, Any]:
    """导出一个 canonical AEC sample，并返回与 WAV 对齐的严格 JSON metadata。"""
    nearend, mic, farend, sample_rate, sample_length = _unpack_aec_sample(sample)
    _write_wav_f32(sample_dir / "mic.wav", mic, sample_rate, sample_length)
    _write_wav_f32(sample_dir / "nearend.wav", nearend, sample_rate, sample_length)
    _write_wav_f32(sample_dir / "farend.wav", farend, sample_rate, sample_length)
    _export_noisy_se(stems, sample_dir, sample_rate, sample_length)
    _export_echo(stems, sample_dir, sample_rate, sample_length)
    if not (sample_dir / "noisy_se.wav").is_file():
        raise RuntimeError(f"noisy_se.wav 未生成: {scenario_name}/{sample_dir.name}")
    metadata = _build_sample_metadata(
        scenario_name,
        dataset_index,
        sample_rate,
        sample_length,
        stems,
    )
    _write_json(sample_dir / "meta.json", metadata)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate AEC testset for 7 scenarios.")
    parser.add_argument("-C", "--config", required=True, help="YAML config file (dynamic_aec)")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不写任何文件")
    parser.add_argument("--force", action="store_true",
                        help="删除已有场景目录并强制重新生成（manifest 缺失/不匹配时需要）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_samples <= 0:
        print("[ERROR] --num-samples 必须为正整数", file=sys.stderr)
        return 1
    if not args.dry_run:
        print(
            "[ERROR] GenerationManifest v1 publication 已下线；schema v2 reload 当前仅支持 SE；"
            "AEC offline publication 需要迁移后重新生成。",
            file=sys.stderr,
        )
        return 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_root = Path(os.path.abspath(os.fspath(Path(args.output_dir).expanduser())))
    config_path = Path(args.config).expanduser().resolve(strict=True)

    config_fingerprint = _file_fingerprint(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if _file_fingerprint(config_path) != config_fingerprint:
        print("[ERROR] config 在解析期间发生变化", file=sys.stderr)
        return 1
    if not isinstance(config, dict):
        print("[ERROR] YAML config 顶层必须是 mapping", file=sys.stderr)
        return 1
    data_cfg = config.get("data", {})
    train_cfg = data_cfg.get("train", {})
    if not isinstance(data_cfg, dict) or not isinstance(train_cfg, dict):
        print("[ERROR] data 与 data.train 必须是 mapping", file=sys.stderr)
        return 1
    target_sr = int(data_cfg.get("sample_rate", 16000))

    configured_paths = {
        "speech_scp": train_cfg.get("speech_scp"),
        "noise_scp": train_cfg.get("noise_scp"),
        "rir_scp": train_cfg.get("rir_scp"),
        "windnoise_scp": train_cfg.get("windnoise_scp"),
        "speech_length_file": train_cfg.get("speech_length_file"),
        "real_farend_echo_pair_json": train_cfg.get("real_farend_echo_pair_json"),
    }
    required_names = {
        "speech_scp",
        "noise_scp",
        "rir_scp",
        "real_farend_echo_pair_json",
    }
    missing = sorted(name for name in required_names if not configured_paths.get(name))
    if missing:
        print(
            "[ERROR] Missing required config key(s): "
            + ", ".join(f"data.train.{name}" for name in missing),
            file=sys.stderr,
        )
        return 1
    # windnoise_scp 可省略：分类 noise JSON/JSONL 可提供 embedded wind role。
    structural_resources: dict[str, str] = {}
    for name, path in configured_paths.items():
        if not path:
            continue
        try:
            structural_resources[name] = str(
                Path(path).expanduser().resolve(strict=True)
            )
        except OSError as exc:
            print(f"[ERROR] {name} not found: {path}: {exc}", file=sys.stderr)
            return 1

    dataset_cfg = {"data": copy.deepcopy(data_cfg)}
    for name, path in structural_resources.items():
        dataset_cfg["data"]["train"][name] = path
    if not isinstance(dataset_cfg["data"]["train"].get("simulation_cfg"), dict):
        print("[ERROR] Missing required config key: data.train.simulation_cfg", file=sys.stderr)
        return 1

    print(f"[dry-run] config: {config_path}")
    print(
        f"[dry-run] output: {output_root} | seed={args.seed} | "
        f"num_samples={args.num_samples} | sr={target_sr}"
    )
    for scenario_name in SCENARIOS:
        print(
            f"  [dry-run] Would generate up to {args.num_samples} samples in "
            f"{output_root / scenario_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
