"""
数据集注册表加载与查询。

默认注册表为 config/noise.yaml（便携模板，部署时复制并写入真实数据根）。
"""
from __future__ import annotations

import os
import re
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 默认使用便携模板 noise.yaml；部署时复制该文件并写入真实数据根目录。
DEFAULT_REGISTRY_PATH = "tools/noise_filter_pipeline/config/noise.yaml"


VALID_TYPES = ("directory", "json", "csv", "huggingface")
# 注意：filename 策略从未在 Stage A 的 _resolve_split 中实现，不属于合法集合；
# 旧配置误用 filename 时 __post_init__ 会单独报"未实现"的 ValueError
VALID_SPLIT_STRATEGIES = ("directory", "ratio", "none")
_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# 音频扩展名集合：Stage A 目录扫描与 validate/scan 脚本统一引用本常量，
# 避免各侧重复定义导致口径漂移
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass
class DatasetEntry:
    """单个数据集注册项。"""
    id: str
    name: str
    enabled: bool = True
    type: str = "directory"
    path: str = ""
    input_pattern: str = "**/*.wav"
    label_source: str = "filename"
    label_field: Optional[str] = None
    label_column: Optional[str] = None
    # audio_field 默认 None：json 的回退默认值由 _load 在 registry 侧填充
    #（未显式配置时按旧行为 "audio"，显式 null 仍表示未指定，读取侧回退
    # "audio_path"）；huggingface 由 ingest 路径回退 "audio"，csv 走
    # audio_column，均不受此处影响
    audio_field: Optional[str] = None
    audio_column: Optional[str] = None
    # label_separator 显式配置 null 表示不对标签串做拆分（Stage A 原样返回整串）
    label_separator: Optional[str] = ","
    label_mapping: Optional[str] = None
    splits: List[str] = field(default_factory=lambda: ["train"])
    split_strategy: str = "none"
    split_ratio: Optional[List[float]] = None
    license: str = "unknown"
    category_priorities: Optional[List[str]] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not _DATASET_ID_RE.fullmatch(self.id) or "__" in self.id:
            raise ValueError(
                f"数据集 id 不安全: {self.id!r}；只允许字母、数字、单个 _ 和 -"
            )
        if self.type not in VALID_TYPES:
            raise ValueError(f"数据集 {self.id}: 不支持的 type={self.type}")
        # 旧配置可能写 split_strategy: filename；该策略从未实现过，
        # 单独报错指明未实现，避免被误认为合法但行为缺失
        if self.split_strategy == "filename":
            raise ValueError(
                f"数据集 {self.id}: split_strategy=filename 未实现，"
                "请改用 directory / ratio / none"
            )
        if self.split_strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError(f"数据集 {self.id}: 不支持的 split_strategy={self.split_strategy}")
        # ratio 策略要求比例与 splits 一一对应且总和为 1，否则 idx/total
        # 切分结果静默偏离配置意图
        if self.split_strategy == "ratio":
            if self.split_ratio is None or len(self.split_ratio) != len(self.splits):
                raise ValueError(
                    f"数据集 {self.id}: split_strategy=ratio 要求 split_ratio 长度 "
                    f"等于 splits 长度（{len(self.splits)}），实际 split_ratio={self.split_ratio}"
                )
            if abs(sum(self.split_ratio) - 1.0) >= 1e-6:
                raise ValueError(
                    f"数据集 {self.id}: split_ratio 之和必须为 1.0，实际为 "
                    f"{sum(self.split_ratio)}"
                )

    def label_mapping_path(self, registry_dir: str) -> Optional[str]:
        """解析标签映射文件绝对路径。

        约定（registry_dir 由调用方传入，Stage A 与 validate_dataset_registry
        统一传注册表文件所在目录，即 ``DatasetRegistry._dir``）：
          - 绝对路径：直接使用。
          - 相对路径且含目录分隔符：相对于 registry_dir 解析。
          - 仅文件名：默认在 registry_dir/labels/ 下查找。
        """
        if not self.label_mapping:
            return None
        p = Path(self.label_mapping)
        if p.is_absolute():
            return str(p)
        if os.path.sep not in self.label_mapping:
            return str(Path(registry_dir) / "labels" / self.label_mapping)
        return str(Path(registry_dir) / self.label_mapping)


class DatasetRegistry:
    """数据集注册表。"""

    def __init__(self, path: str):
        self.path = path
        # _dir 统一取绝对路径，保证 resolve_path 对相对条目路径产出绝对路径
        self._dir = os.path.dirname(os.path.abspath(path))
        self.entries: Dict[str, DatasetEntry] = {}
        self._load()

    def resolve_path(self, path: str) -> str:
        """将条目路径解析为绝对路径。

        口径约定：相对路径一律按 registry 文件所在目录解析；绝对路径原样
        返回。Stage A 与 validate_dataset_registry 共用本方法，避免各自
        实现导致同一 registry 在两侧解析结果不一致。
        """
        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(self._dir) / p)

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            logger.warning("注册表文件不存在: %s", self.path)
            return

        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for raw in data.get("datasets", []):
            entry = DatasetEntry(
                id=raw["id"],
                name=raw.get("name", raw["id"]),
                enabled=raw.get("enabled", True),
                type=raw.get("type", "directory"),
                path=raw.get("path", ""),
                input_pattern=raw.get("input_pattern", "**/*.wav"),
                label_source=raw.get("label_source", "filename"),
                label_field=raw.get("label_field"),
                label_column=raw.get("label_column"),
                audio_field=(
                    "audio"
                    if raw.get("type", "directory") == "json" and "audio_field" not in raw
                    else raw.get("audio_field")
                ),
                audio_column=raw.get("audio_column"),
                label_separator=raw.get("label_separator", ","),
                label_mapping=raw.get("label_mapping"),
                splits=raw.get("splits", ["train"]),
                split_strategy=raw.get("split_strategy", "none"),
                split_ratio=raw.get("split_ratio"),
                license=raw.get("license", "unknown"),
                category_priorities=raw.get("category_priorities"),
                notes=raw.get("notes", ""),
                extra={k: v for k, v in raw.items() if k not in {
                    "id", "name", "enabled", "type", "path", "input_pattern",
                    "label_source", "label_field", "label_column", "label_separator",
                    "audio_field", "audio_column",
                    "label_mapping", "splits", "split_strategy", "split_ratio",
                    "license", "category_priorities", "notes"
                }},
            )
            if entry.id in self.entries:
                raise ValueError(f"数据集注册表存在重复 id: {entry.id}")
            self.entries[entry.id] = entry

        logger.info("加载数据集注册表: %d 个数据集", len(self.entries))

    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        return self.entries.get(dataset_id)

    def list_enabled(self) -> List[DatasetEntry]:
        return [e for e in self.entries.values() if e.enabled]

    def list_all(self) -> List[DatasetEntry]:
        return list(self.entries.values())

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self.entries
