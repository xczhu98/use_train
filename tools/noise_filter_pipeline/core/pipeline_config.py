"""
统一 Pipeline 配置。

新架构 Stage A-D 统一使用 UnifiedPipelineConfig。
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import yaml

from .registry import DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)


def _expand_env(value):
    """递归对配置中的字符串值做 expandvars + expanduser，提升可移植性。

    注意：未定义的 $VAR / ${VAR} 会按字面保留（expandvars 特性）；普通路径
    仍保留该占位语义，启用的 subprocess_daemon 的 python_path 则由下方
    preflight 明确拒绝未解析占位符。
    """
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _unresolved_env_vars(value: Any) -> List[str]:
    """返回字符串中仍未展开的环境变量名。"""
    if isinstance(value, str):
        return sorted(
            {
                match.group("braced") or match.group("bare")
                for match in _ENV_VAR_PATTERN.finditer(value)
            }
        )
    if isinstance(value, dict):
        names = set()
        for item in value.values():
            names.update(_unresolved_env_vars(item))
        return sorted(names)
    if isinstance(value, list):
        names = set()
        for item in value:
            names.update(_unresolved_env_vars(item))
        return sorted(names)
    return []


def validate_subprocess_daemon_python_path(
    config: Dict[str, Any], location: str
) -> str:
    """校验并展开 subprocess daemon 使用的 Python 解释器。

    ``python_path`` 可以是文件路径，也可以是当前 ``PATH`` 中的命令名。校验
    成功后将解析结果冻结为目标普通文件的绝对 realpath 并同步写回配置，避免
    通过 preflight 后在启动或重启时再次按可能已变化的 ``PATH`` 解析。
    """
    raw_path = config.get("python_path", "python")
    if raw_path is None:
        raise ValueError(
            f"{location}.python_path 未配置；请设置环境变量，或提供可执行的 Python 解释器"
        )
    try:
        path = os.fspath(raw_path)
    except TypeError as exc:
        raise ValueError(
            f"{location}.python_path 必须是字符串路径或命令名，实际为 {raw_path!r}"
        ) from exc
    if not isinstance(path, str):
        raise ValueError(
            f"{location}.python_path 必须是字符串路径或命令名，实际为 {raw_path!r}"
        )
    path = os.path.expanduser(os.path.expandvars(path)).strip()
    if not path:
        raise ValueError(
            f"{location}.python_path 不能为空；请设置环境变量，或提供可执行的 Python 解释器"
        )

    unresolved = sorted(
        {
            match.group("braced") or match.group("bare")
            for match in _ENV_VAR_PATTERN.finditer(path)
        }
    )
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(
            f"{location}.python_path 环境变量未解析（{names}）: {path!r}；"
            "请设置对应环境变量"
        )

    resolved = shutil.which(path)
    if resolved is None:
        raise ValueError(
            f"{location}.python_path 不是实际可执行文件: {path!r}"
            "（文件不存在，且该命令不在 PATH 中）；请提供可执行的 Python 解释器"
        )
    resolved = os.path.realpath(os.path.abspath(resolved))
    try:
        resolved_stat = os.stat(resolved)
    except OSError as exc:
        raise ValueError(
            f"{location}.python_path 无法读取已解析的可执行文件: {resolved!r}（{exc}）"
        ) from exc
    if not stat.S_ISREG(resolved_stat.st_mode) or not os.access(resolved, os.X_OK):
        detail = "路径不是普通文件" if not stat.S_ISREG(resolved_stat.st_mode) else "文件没有执行权限"
        raise ValueError(
            f"{location}.python_path 不是实际可执行文件: {resolved!r}（{detail}）；"
            "请提供可执行的 Python 解释器"
        )

    config["python_path"] = resolved
    return resolved



DEFAULT_STAGE_NAMES = ["A", "B", "C", "D"]

_STAGE_C_USER_KEYS = {"c1", "c2", "c3"}
_STAGE_C_C1_USER_KEYS = {
    "model_path",
    "batch_size",
    "device_map",
    "torch_dtype",
    "cache_dir",
    "max_retries",
    "max_new_tokens",
    "dry_run",
    "dry_run_distribution",
    "enable_checkpoint",
    "checkpoint_path",
    "checkpoint_save_interval",
    "manifest_write_batch_size",
    "num_workers",
    "gpu_ids",
    "max_consecutive_worker_infra_failures",
    "prompt",
    "allow_partial",
    "cleanup_intermediate",
}
_STAGE_C_C2_USER_KEYS = {
    "enable",
    "mode",
    "checkpoint",
    "label_dict",
    "sample_rate",
    "batch_size",
    "script_path",
    "python_path",
    "daemon_script_path",
    "daemon_timeout",
    "startup_timeout",
    "dry_run_labels",
    "cpu_threads",
    "interop_threads",
    "audio_workers",
    "timing_log_interval",
    "request_batches",
    "enable_checkpoint",
    "checkpoint_path",
    "checkpoint_save_interval",
    "manifest_write_batch_size",
    "num_gpus",
    "gpu_ids",
    "allow_partial",
    "cleanup_intermediate",
}
_STAGE_C_C3_USER_KEYS = {
    "enable",
    "policy_path",
    "allow_partial",
    "cleanup_intermediate",
}
_STAGE_C_INTERNAL_KEYS = {
    "c1": {"artifact_fingerprints", "input_sanitization", "execution_mode"},
    "c2": {
        "artifact_fingerprints",
        "input_sanitization",
        "execution_mode",
        "evidence_contract_version",
        "score_contract",
        "class_count",
    },
    "c3": {
        "artifact_fingerprints",
        "c1_identity",
        "c2_identity",
        "execution_mode",
        "publishable",
        "evidence_contract_version",
    },
}


def _validate_exact_mapping(
    value: Any,
    *,
    location: str,
    allowed_keys: set[str],
) -> Dict[str, Any]:
    """校验配置节点为 exact mapping，拒绝任何未声明键。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{location} 必须是 mapping")
    unknown = sorted(set(value).difference(allowed_keys))
    if unknown:
        raise ValueError(
            f"{location} 含未知或已移除配置键: "
            + ", ".join(f"{location}.{key}" for key in unknown)
        )
    return value


def validate_stage_c_config(
    stage_c: Any,
    *,
    allow_internal: bool = False,
) -> Dict[str, Any]:
    """校验 Stage C 用户配置；内部运行注入键必须显式开启。"""
    validated = _validate_exact_mapping(
        stage_c,
        location="stage_c",
        allowed_keys=_STAGE_C_USER_KEYS,
    )
    user_keys = {
        "c1": _STAGE_C_C1_USER_KEYS,
        "c2": _STAGE_C_C2_USER_KEYS,
        "c3": _STAGE_C_C3_USER_KEYS,
    }
    for name, allowed in user_keys.items():
        if allow_internal:
            allowed = allowed | _STAGE_C_INTERNAL_KEYS[name]
        _validate_exact_mapping(
            validated.get(name, {}),
            location=f"stage_c.{name}",
            allowed_keys=allowed,
        )
    c2 = validated.get("c2") or {}
    if "startup_timeout" in c2:
        startup_timeout = c2["startup_timeout"]
        if type(startup_timeout) is not int or startup_timeout <= 0:
            raise ValueError("stage_c.c2.startup_timeout 必须是严格正整数")
    return validated


MAX_STAGE_D_SNAPSHOT_NUM_WORKERS = 128

_STAGE_D_USER_KEYS = {
    "skip_d1",
    "num_workers",
    "snapshot_num_workers",
    "writer_lock_timeout",
    "d1",
    "d2",
}
_STAGE_D_D1_USER_KEYS = {
    "mode",
    "config_path",
    "apollo_config",
    "checkpoint",
    "apollo_checkpoint",
    "script_path",
    "python_path",
    "daemon_script_path",
    "daemon_timeout",
    "daemon_startup_timeout",
    "batch_size",
    "target_sr",
    "output_sr",
    "output_format",
    "chunk_sec",
    "overlap_sec",
    "overlap",
    "post_normalize",
    "enable_checkpoint",
    "checkpoint_path",
    "checkpoint_save_interval",
    "manifest_write_batch_size",
    "num_gpus",
    "gpu_ids",
    "allow_partial",
    "cleanup_intermediate",
}
_STAGE_D_D2_USER_KEYS = {
    "num_workers",
    "target_rms",
    "max_peak",
    "min_duration",
    "max_duration",
    "output_sr",
    "target_sr",
    "output_layout",
    "dataset_label_shard_size",
    "output_format",
    "enable_checkpoint",
    "checkpoint_path",
    "checkpoint_save_interval",
    "manifest_write_batch_size",
    "allow_partial",
    "cleanup_intermediate",
}


def _validate_stage_d_num_workers(
    config: Dict[str, Any],
    *,
    key: str,
    location: str,
    maximum: Optional[int] = None,
) -> None:
    """严格校验显式配置的 Stage D worker 数。"""
    if key not in config:
        return
    value = config[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location} 必须是严格正整数")
    if maximum is not None and value > maximum:
        raise ValueError(f"{location} 不得超过 {maximum}")


def resolve_stage_d_config(stage_d: Any) -> Dict[str, Any]:
    """返回 exact Stage D 配置副本，并拒绝拼写错误或废弃键。"""
    validated = _validate_exact_mapping(
        stage_d,
        location="stage_d",
        allowed_keys=_STAGE_D_USER_KEYS,
    )
    d1 = _validate_exact_mapping(
        validated.get("d1", {}),
        location="stage_d.d1",
        allowed_keys=_STAGE_D_D1_USER_KEYS,
    )
    d2 = _validate_exact_mapping(
        validated.get("d2", {}),
        location="stage_d.d2",
        allowed_keys=_STAGE_D_D2_USER_KEYS,
    )
    post_normalize = d1.get("post_normalize")
    post_normalize_config = None
    if post_normalize not in (None, False, True):
        post_normalize_config = _validate_exact_mapping(
            post_normalize,
            location="stage_d.d1.post_normalize",
            allowed_keys=_STAGE_D_D2_USER_KEYS,
        )
    skip_d1 = validated.get("skip_d1", False)
    if type(skip_d1) is not bool:
        raise ValueError("stage_d.skip_d1 必须是 bool")
    _validate_stage_d_num_workers(
        validated,
        key="num_workers",
        location="stage_d.num_workers",
    )
    _validate_stage_d_num_workers(
        validated,
        key="snapshot_num_workers",
        location="stage_d.snapshot_num_workers",
        maximum=MAX_STAGE_D_SNAPSHOT_NUM_WORKERS,
    )
    _validate_stage_d_num_workers(
        d2,
        key="num_workers",
        location="stage_d.d2.num_workers",
    )
    if post_normalize_config is not None:
        _validate_stage_d_num_workers(
            post_normalize_config,
            key="num_workers",
            location="stage_d.d1.post_normalize.num_workers",
        )
    writer_lock_timeout = validated.get("writer_lock_timeout", 30.0)
    if (
        isinstance(writer_lock_timeout, bool)
        or not isinstance(writer_lock_timeout, (int, float))
        or writer_lock_timeout < 0
        or not writer_lock_timeout < float("inf")
    ):
        raise ValueError("stage_d.writer_lock_timeout 必须是非负有限数值")
    resolved = dict(validated)
    resolved["writer_lock_timeout"] = float(writer_lock_timeout)
    resolved["skip_d1"] = skip_d1
    resolved["d1"] = dict(d1)
    resolved["d2"] = dict(d2)
    return resolved


def resolve_stage_d_final_stage(stage_d: Any) -> tuple[str, str]:
    """从 exact Stage D 配置返回最终子阶段名与 manifest 前缀。"""
    resolved = resolve_stage_d_config(stage_d)
    merged_d1 = not resolved["skip_d1"] and bool(
        resolved["d1"].get("post_normalize")
    )
    stage_name = "D1" if merged_d1 else "D2"
    return stage_name, stage_name.lower()


# 默认输出根目录指向正式部署路径；环境变量仍可显式 override。
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "NOISE_FILTER_OUTPUT_ROOT", "/data_nvme0n1/data/noise_filter/"
)


@dataclass
class UnifiedPipelineConfig:
    """新架构统一配置。--stage A/B/C/D 命令使用本类。"""

    # 全局
    output_root: str = DEFAULT_OUTPUT_ROOT
    # 默认注册表路径（相对 use_train 根解析）；noise.yaml 仅作为便携模板。
    registry_path: str = DEFAULT_REGISTRY_PATH
    pipeline_version: str = "4.1.0"
    seed: Optional[int] = None

    # 阶段开关
    stages: List[str] = field(default_factory=lambda: DEFAULT_STAGE_NAMES.copy())

    # 各阶段配置（具体内容由对应 Processor 解释）
    stage_a: Dict[str, Any] = field(default_factory=dict)
    stage_b: Dict[str, Any] = field(default_factory=dict)
    stage_c: Dict[str, Any] = field(default_factory=dict)
    stage_d: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        active_stages: Optional[Iterable[str]] = None,
    ) -> "UnifiedPipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 支持在配置值中使用 $VAR / ${VAR} / ~，提升跨机器可移植性；具体
        # 展开与 daemon preflight 统一由 from_dict 完成，避免两条加载路径漂移。
        return cls.from_dict(data, active_stages=active_stages)

    def validate_subprocess_daemon_paths(
        self, active_stages: Optional[Iterable[str]] = None
    ) -> None:
        """校验本次阶段计划中实际启用的 subprocess daemon 解释器。

        ``active_stages=None`` 使用配置声明的 stages；显式空 iterable 表示本次
        仅读取配置、不运行任何 stage。D1 的 ``skip_d1: true`` 与 C2 的
        ``enable: false`` 都不会触发对应 daemon preflight。
        """
        planned_stages = self.stages if active_stages is None else active_stages
        enabled_stages = {str(stage).upper() for stage in planned_stages}

        stage_c = self.stage_c or {}
        c2 = stage_c.get("c2") or {}
        if (
            "C" in enabled_stages
            and bool(c2.get("enable", True))
            and c2.get("mode", "dry_run") == "subprocess_daemon"
        ):
            validate_subprocess_daemon_python_path(c2, "stage_c.c2")

        stage_d = self.stage_d or {}
        d1 = stage_d.get("d1") or {}
        if (
            "D" in enabled_stages
            and not bool(stage_d.get("skip_d1", False))
            and d1.get("mode", "dry_run") == "subprocess_daemon"
        ):
            validate_subprocess_daemon_python_path(d1, "stage_d.d1")

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        active_stages: Optional[Iterable[str]] = None,
    ) -> "UnifiedPipelineConfig":
        data = _expand_env(data)
        if not isinstance(data, dict):
            raise ValueError("pipeline config 必须是 mapping")
        validate_stage_c_config(data.get("stage_c", {}))
        stage_d = resolve_stage_d_config(data.get("stage_d", {}))
        output_root = data.get("output_root")
        if not output_root:
            output_root = DEFAULT_OUTPUT_ROOT
        config = cls(
            output_root=str(output_root),
            registry_path=data.get("registry_path") or cls.registry_path,
            pipeline_version=data.get("pipeline_version", cls.pipeline_version),
            seed=data.get("seed"),
            # YAML 写 `stages: null` 时 data.get 返回 None 而非默认值，需显式回退
            stages=data.get("stages") or DEFAULT_STAGE_NAMES.copy(),
            stage_a=data.get("stage_a", {}),
            stage_b=data.get("stage_b", {}),
            stage_c=data.get("stage_c", {}),
            stage_d=stage_d,
        )
        # 用户 YAML 的 Stage C/Stage D 都接受 exact schema；运行期派生
        # identity 由 orchestrator 在加载后注入，不能从用户配置伪造。
        config.validate_subprocess_daemon_paths(active_stages=active_stages)
        return config

    def resolve_registry_path(self, base_dir: str) -> str:
        """解析并预检 Stage A 注册表路径。"""
        unresolved = _unresolved_env_vars(self.registry_path)
        if unresolved:
            names = ", ".join(unresolved)
            raise ValueError(
                f"Stage A registry_path 环境变量未解析（{names}）: "
                f"{self.registry_path!r}；请设置对应环境变量"
            )
        p = Path(self.registry_path)
        resolved = p if p.is_absolute() else Path(base_dir) / p
        resolved = resolved.resolve()
        if not resolved.is_file():
            return str(resolved)
        with resolved.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle) or {}
        unresolved = _unresolved_env_vars(registry)
        if unresolved:
            names = ", ".join(unresolved)
            raise ValueError(
                f"Stage A registry 条目环境变量未解析（{names}）: "
                f"{resolved}；请设置对应环境变量"
            )
        return str(resolved)

    def get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """获取某阶段配置字典。"""
        return getattr(self, f"stage_{stage_name.lower()}", {}) or {}
