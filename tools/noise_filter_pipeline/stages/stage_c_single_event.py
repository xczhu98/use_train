"""Stage C：固定执行 C1 语义分流 → C2 AudioSet top-5 事实 → C3 准入。

C1 ``speech`` 是终态拒绝，编排层只把 C1 ``passed`` 的 ``noise/babble``
候选交给 C2；C2 仅持久化固定 top-5 证据（``audioset_top5``，排序自完整
527 sigmoid 内部排名），不消费 policy；C3 绑定 policy 与 C1/C2 identity，
委托共享 core evaluator 输出三种业务结果。C3 决策无阈值、无 calibration
依赖，approved 门禁已删除（链路直接可发布）。

三层 checkpoint 依赖彼此隔离：C1 只绑定 Qwen model/prompt，C2 只绑定
AudioTag checkpoint/label space/事实契约，C3 绑定 C1/C2 identity 与 policy。
policy 变化只会失效 C3，不触发昂贵的 C1/C2。
Stage C 输入 seam 默认拒绝预置 C1-C4、旧证据或新派生准入证据，不做清理重算。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from ..core.execution_contract import build_stage_c_execution_contract
from ..core.manifest_io import (
    sanitize_stage_c_input_record,
    sanitize_stage_c2_input_record,
)
from ..core.manifest_per_dataset import link_stage_output, split_manifest_by_dataset
from ..core.pipeline_config import UnifiedPipelineConfig, validate_stage_c_config
from ..core.run_lock import stage_c_run_lock
from ..core.stage_orchestrator import StageOrchestratorBase
from ..core.stats import write_stage_stats
from ..processors.qwen_single_event import QwenSingleEventProcessor
from ..processors.audiotag_classifier import AudioTagClassifierProcessor
from ..processors.noise_label_filter import NoiseAdmissionProcessor
from core.data.noise_admission_policy import (
    CLASSIFIER_LABEL_COUNT,
    EVIDENCE_CONTRACT_VERSION,
    SCORE_CONTRACT,
    default_voice_policy_path,
)

logger = logging.getLogger(__name__)


class StageCOrchestrator(StageOrchestratorBase):
    """Stage C orchestrator。"""

    def __init__(self, config: UnifiedPipelineConfig):
        self.config = config
        self.stage_cfg = validate_stage_c_config(config.get_stage_config("C"))
        c3_cfg = self.stage_cfg.get("c3", {}) or {}
        self.policy_path = self._validate_canonical_policy_path(
            c3_cfg.get("policy_path")
        )

    @staticmethod
    def _validate_canonical_policy_path(policy_path: Any) -> Path:
        """拒绝不指向仓库 canonical policy 的自定义路径。"""
        canonical = Path(default_voice_policy_path()).resolve()
        candidate = canonical if not policy_path else Path(policy_path)
        if not candidate.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / candidate
        candidate = candidate.resolve()
        try:
            same_file = candidate.is_file() and os.path.samefile(candidate, canonical)
        except OSError:
            same_file = False
        if not same_file:
            raise ValueError(
                "Stage C3 policy_path 必须解析为仓库 canonical policy: "
                f"{canonical}；实际为 {candidate}"
            )
        return canonical

    def _resolve_artifact_paths(self) -> Dict[str, Optional[Path]]:
        """按 C1/C2/C3 各自消费边界解析运行资产。"""
        repo_root = Path(__file__).resolve().parents[3]

        def resolve(path: Any) -> Optional[Path]:
            if not path:
                return None
            resolved = Path(path)
            return resolved if resolved.is_absolute() else repo_root / resolved

        c1_cfg = self.stage_cfg.get("c1", {}) or {}
        c2_cfg = self.stage_cfg.get("c2", {}) or {}
        # 键名只取 canonical 形式：audiotag_label_dict/audiotag_checkpoint/qwen_model
        # 已被 validate_stage_c_config 的 exact-key 白名单显式拒绝，不保留读取分支
        return {
            "voice_policy_sha256": resolve(self.policy_path) or default_voice_policy_path(),
            "audiotag_label_dict_sha256": resolve(c2_cfg.get("label_dict")),
            "audiotag_checkpoint_sha256": resolve(c2_cfg.get("checkpoint")),
            "qwen_model_sha256": resolve(c1_cfg.get("model_path")),
        }

    def _validate_required_artifacts(
        self, entries: Dict[str, Optional[Path]]
    ) -> None:
        """按 C1/C2/C3 独立模式聚合校验正式运行所需资产。"""
        c1_cfg = self.stage_cfg.get("c1", {}) or {}
        c2_cfg = self.stage_cfg.get("c2", {}) or {}
        required = {"voice_policy_sha256": "voice policy"}
        if not bool(c1_cfg.get("dry_run", False)):
            required["qwen_model_sha256"] = "C1 Qwen model"
        if bool(c2_cfg.get("enable", True)):
            required["audiotag_label_dict_sha256"] = "C2 AudioTag label dict"
        if (
            bool(c2_cfg.get("enable", True))
            and c2_cfg.get("mode", "dry_run") == "subprocess_daemon"
        ):
            required["audiotag_checkpoint_sha256"] = "C2 AudioTag checkpoint"

        missing = [
            (label, entries[key])
            for key, label in required.items()
            if entries[key] is None or not entries[key].exists()
        ]
        if missing:
            details = "\n".join(f"  - {label}: {path}" for label, path in missing)
            raise FileNotFoundError(
                "Stage C 必需运行资产缺失（已聚合检查）:\n" + details
            )

    def _compute_artifact_fingerprints(self) -> Dict[str, str]:
        """校验必需资产后计算指纹；仅非必需缺失资产可使用 unset 占位。"""
        from core.data.noise_admission import sha256_file, sha256_path

        entries = self._resolve_artifact_paths()
        # 必须先完成所有 exists 检查，避免某个大模型目录已开始哈希后才发现
        # 另一个正式运行资产缺失。
        self._validate_required_artifacts(entries)

        fingerprints: Dict[str, str] = {}
        for key, path in entries.items():
            if path is None or not path.exists():
                fingerprints[key] = f"unset:{key}"
                logger.warning(
                    "Stage C 非必需资产指纹 %s 路径缺失（%s），使用确定性占位；"
                    "checkpoint 指纹不会随该资产内容变化", key, path,
                )
                continue
            fingerprints[key] = sha256_path(path) if path.is_dir() else sha256_file(path)
        return fingerprints

    def run(
        self,
        input_manifest: str,
        output_dir: str,
        output_manifest: str,  # 保留以对齐 StageRunner 统一接口；per-dataset 架构下
                               # 本阶段输出 = output_dir 下 c_manifest_<dataset_id>.jsonl
        global_pbar: Optional[tqdm] = None,
    ) -> str:
        # 必需资产检查必须先于 output_dir 创建和 processor 构造。
        c1_cfg = dict(self.stage_cfg.get("c1", {}))
        c2_cfg = dict(self.stage_cfg.get("c2", {}))
        c3_cfg = dict(self.stage_cfg.get("c3", {}))
        if self.policy_path:
            c3_cfg["policy_path"] = self.policy_path

        from ..processors.qwen_single_event import DEFAULT_PROMPT

        c1_cfg.setdefault("prompt", DEFAULT_PROMPT)
        if not isinstance(c1_cfg.get("prompt"), str) or not c1_cfg["prompt"].strip():
            raise ValueError("C1 prompt 必须是非空字符串")

        artifact_fingerprints = self._compute_artifact_fingerprints()
        import hashlib

        c1_prompt_sha256 = hashlib.sha256(
            c1_cfg["prompt"].encode("utf-8")
        ).hexdigest()
        c1_identity = {
            "qwen_model_sha256": artifact_fingerprints["qwen_model_sha256"],
            "c1_prompt_sha256": c1_prompt_sha256,
        }
        c2_identity = {
            "audiotag_label_dict_sha256": artifact_fingerprints[
                "audiotag_label_dict_sha256"
            ],
            "audiotag_checkpoint_sha256": artifact_fingerprints[
                "audiotag_checkpoint_sha256"
            ],
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "score_contract": SCORE_CONTRACT,
            "class_count": CLASSIFIER_LABEL_COUNT,
            "top_k": 5,
        }
        execution_contract = build_stage_c_execution_contract(c1_cfg, c2_cfg)
        c1_cfg["artifact_fingerprints"] = c1_identity
        # seam v4 改为 fail-closed，必须使旧“清理后重算”checkpoint 失效。
        c1_cfg["input_sanitization"] = {"version": 4, "mode": "fail_closed"}
        c1_cfg["execution_mode"] = execution_contract["execution_mode"]["c1"]
        c2_cfg["artifact_fingerprints"] = c2_identity
        c2_cfg["execution_mode"] = execution_contract["execution_mode"]["c2"]
        c2_cfg["evidence_contract_version"] = EVIDENCE_CONTRACT_VERSION
        c2_cfg["score_contract"] = SCORE_CONTRACT
        c2_cfg["class_count"] = CLASSIFIER_LABEL_COUNT
        c2_cfg["input_sanitization"] = {
            "version": 4,
            "owner": "C2",
            "mode": "fail_closed",
        }
        # 发布 identity 为五项资产 SHA；事实契约与 policy identity 使用独立字段，
        # 避免把非哈希配置混进 artifact_fingerprints。fingerprint 不再包含
        # calibration（决策无阈值、无校准依赖）。
        c3_cfg["artifact_fingerprints"] = {
            **c1_identity,
            "audiotag_label_dict_sha256": artifact_fingerprints[
                "audiotag_label_dict_sha256"
            ],
            "audiotag_checkpoint_sha256": artifact_fingerprints[
                "audiotag_checkpoint_sha256"
            ],
            "voice_policy_sha256": artifact_fingerprints["voice_policy_sha256"],
        }
        c3_cfg["c1_identity"] = c1_identity
        c3_cfg["c2_identity"] = c2_identity
        c3_cfg["evidence_contract_version"] = EVIDENCE_CONTRACT_VERSION
        c3_cfg.update(execution_contract)
        validate_stage_c_config(
            {"c1": c1_cfg, "c2": c2_cfg, "c3": c3_cfg},
            allow_internal=True,
        )

        with stage_c_run_lock(output_dir):
            progress_callback = global_pbar.update if global_pbar is not None else None

            # C1/C2/C3 按 eligible 数动态扩充全局 total，避免重复计数超 100%
            def total_callback(n: int) -> None:
                if global_pbar is not None:
                    global_pbar.total = (global_pbar.total or 0) + n
                    global_pbar.refresh()

            # C1: Qwen 单事件判定（仅处理 B4 passed；B4 rejected 记录直接 skipped 透传，
            # 避免浪费 C1 Qwen / D1 Apollo 算力。原 C1 单标签过滤已删除——多标签
            # 原始标签不影响可懂度判定，人声由 C3 标签过滤拦截）
            c1_manifest = self._run_substage(
                processor=QwenSingleEventProcessor(c1_cfg),
                input_manifest=input_manifest,
                output_dir=os.path.join(output_dir, "C1"),
                prev_stage="B4",
                progress_callback=progress_callback,
                total_callback=total_callback,
                input_transform=sanitize_stage_c_input_record,
            )

            # C2: 仅 C1 noise/babble（status=passed）进入；speech/reject/failed 均跳过。
            # C2-owned seam 保留本轮 C1，但拒绝任何预置 C2-C4 或派生准入证据。
            c2_manifest = self._run_substage(
                processor=AudioTagClassifierProcessor(c2_cfg),
                input_manifest=c1_manifest,
                output_dir=os.path.join(output_dir, "C2"),
                prev_stage="C1",
                progress_callback=progress_callback,
                total_callback=total_callback,
                input_transform=sanitize_stage_c2_input_record,
            )

            # C3: 最终准入（固定直接消费 C2 输出）
            c3_manifest = self._run_substage(
                processor=NoiseAdmissionProcessor(c3_cfg),
                input_manifest=c2_manifest,
                output_dir=os.path.join(output_dir, "C3"),
                prev_stage="C2",
                progress_callback=progress_callback,
                total_callback=total_callback,
            )

            # 汇总：C3 输出（含 skipped 透传记录）落到 stage 根目录的
            # c_manifest_<dataset_id>.jsonl，供下游按"stage 根目录"统一约定读取。
            # 目录化输出（新）硬链接 per-dataset 文件；legacy merged 单文件（旧运行
            # 产物复用）按 dataset_id 拆分写入。增量场景下本次未涉及的数据集文件
            # 保留不动。
            if os.path.isdir(c3_manifest):
                link_stage_output(c3_manifest, output_dir, "c")
            else:
                split_manifest_by_dataset(
                    src_path=c3_manifest,
                    dst_base=os.path.join(output_dir, "c_manifest.jsonl"),
                )

            stats = write_stage_stats(
                manifest_path=output_dir,
                stage_name="C",
                output_path=os.path.join(output_dir, "c_stats.json"),
            )
            # stats 的单遍扫描已含总数与聚合 passed（C3 passed 的记录其聚合状态
            # 即 passed），不再为一条日志重复解析整个 manifest
            logger.info(
                "Stage C 完成：%d / %d 条通过 C3，输出 %s",
                stats["status"]["passed"], stats["total_records"], output_dir,
            )
            return output_dir

