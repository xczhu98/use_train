"""
Stage Orchestrator 公共基类。

把 Stage C/D/E 中重复的子阶段调度逻辑（eligible/ineligible 拆分、skipped 透传、
结果合并）抽象到这里，避免三份相同代码。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .manifest_io import (
    _iter_manifest_files,
    atomic_replace_durable,
    config_hash,
    validate_input_path,
)
from .manifest_per_dataset import validate_dataset_id
from .pipeline_config import validate_subprocess_daemon_python_path
from .processor_base import Processor
from .run_lock import substage_writer_lock

logger = logging.getLogger(__name__)

_MARKER_SCHEMA = "stage_orchestrator_complete"
_MARKER_VERSION = 4
_IDENTITY_ERROR_SAMPLE_LIMIT = 5
_IDENTITY_ERROR_TEXT_LIMIT = 160


def _bounded_identity_text(value: Any) -> str:
    """截断不可信身份字段，避免异常信息无界增长。"""
    text = repr(value)
    if len(text) <= _IDENTITY_ERROR_TEXT_LIMIT:
        return text
    return text[: _IDENTITY_ERROR_TEXT_LIMIT - 3] + "..."


def _make_stage_pass_checker(prev_stage: str):
    """构造轻量「prev_stage 是否 passed」判定器。

    本管线自己写出的 manifest 中，stage 条目形如
    ``"B1": {"status": "passed", "timestamp": ...}``（status 恒为首个键），
    因此首选正则快查（免整行 JSON 解析）；stage 键存在但正则未命中
    （如第三方 manifest 键序不同）时回退整行 json.loads，保证正确性。
    """
    pattern = re.compile(
        r'"%s"\s*:\s*\{\s*"status"\s*:\s*"passed"' % re.escape(prev_stage)
    )
    stage_key = f'"{prev_stage}"'

    def is_passed(line: str) -> bool:
        if stage_key not in line:
            return False
        if pattern.search(line):
            return True
        # 键序异常或空白差异：整行解析兜底
        try:
            rec = json.loads(line)
        except Exception:
            return False
        return rec.get("stages", {}).get(prev_stage, {}).get("status") == "passed"

    return is_passed


def _extract_dataset_id(line: str, ds_re: re.Pattern) -> str:
    """从记录行解析顶层 dataset_id，非法或缺失时返回 ``unknown``。

    不直接信任首个正则匹配：第三方 manifest 可能先出现嵌套 dataset_id，
    将其误作顶层值会把记录路由到错误的 per-dataset 文件。
    ``ds_re`` 参数为兼容现有调用保留。
    """
    del ds_re
    try:
        rec = json.loads(line)
    except (TypeError, ValueError):
        return "unknown"
    if not isinstance(rec, dict):
        return "unknown"
    value = rec.get("dataset_id")
    return str(value) if value else "unknown"


class StageOrchestratorBase:
    """提供 `_run_substage` 公共实现，供 Stage C/D/E 的 orchestrator 继承。"""

    @staticmethod
    def _preflight_processor(processor: Processor) -> None:
        """在创建子阶段输出前校验 subprocess daemon 的解释器。

        处理器可能来自直接构造的 ``UnifiedPipelineConfig``，绕过 YAML
        加载阶段；因此这里保留同一项 fail-fast 校验。禁用的 C2 不启动
        daemon，不应因其遗留占位配置阻断运行；D1 的 skip_d1 路径不会构造
        D1 处理器，天然不会进入此校验。GPU 空列表仍由 daemon pool 展开。
        """
        config = processor.config or {}
        if config.get("mode") != "subprocess_daemon":
            return
        if not bool(config.get("enable", True)):
            return
        python_path = validate_subprocess_daemon_python_path(
            config, f"{processor.stage_name} config"
        )
        # Processor 在进入统一 orchestrator 前已经构造；同步更新其 daemon
        # pool，确保 preflight 通过后 Popen 不会继续使用旧的占位符。
        if hasattr(processor, "python_path"):
            processor.python_path = python_path
        daemon_pool = getattr(processor, "_daemon_pool", None)
        if daemon_pool is not None:
            bind_executable = getattr(daemon_pool, "bind_python_executable", None)
            if callable(bind_executable):
                bind_executable(python_path)
            elif hasattr(daemon_pool, "python_path"):
                daemon_pool.python_path = python_path

    def _run_substage(
        self,
        processor: Processor,
        input_manifest: str,
        output_dir: str,
        prev_stage: Optional[str] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        total_callback: Optional[Callable[[int], None]] = None,
        input_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        writer_lock_timeout: float = 30.0,
    ) -> str:
        """在完整子阶段区间持有 output_dir 粒度的排它 writer 锁。"""
        with substage_writer_lock(output_dir, timeout=writer_lock_timeout):
            self._preflight_processor(processor)
            return self._run_substage_locked(
                processor,
                input_manifest,
                output_dir,
                prev_stage=prev_stage,
                progress_callback=progress_callback,
                total_callback=total_callback,
                input_transform=input_transform,
            )

    def _run_substage_locked(
        self,
        processor: Processor,
        input_manifest: str,
        output_dir: str,
        prev_stage: Optional[str] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        total_callback: Optional[Callable[[int], None]] = None,
        input_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> str:
        """
        运行一个已持 writer 锁的子处理器。

        若指定 `prev_stage`，则只让上一阶段 passed 的记录进入处理；
        其余记录标记为 skipped 并透传到输出，保证 manifest 行数不变。
        `input_transform` 在严格解析每条输入后、eligible 判定与落盘前执行；
        未提供时保留原始行透传快路径。

        全程流式处理，不将 eligible/ineligible 全量驻留内存。
        拆分阶段一次遍历输入 manifest，同时写出 eligible 与 ineligible 两份文件
        （ineligible 在写入时就地标记 skipped）；合并阶段对 processor 输出与
        ineligible 文件做原始行透传拷贝（两者都是完整 JSONL，不再做任何
        JSON 解析/序列化），ineligible 段不触发 progress_callback。

        进度记账约定（避免 C/D 多子阶段重复计数导致全局进度条超过 100%）：
          - 拆分阶段不触发 progress_callback；
          - 拆分完成后通过 total_callback 把本子阶段 eligible 数计入全局 total；
          - 合并阶段每写出一条 eligible 结果触发一次 progress_callback。
        这样每个子阶段 total 增量与 callback 次数严格相等，全局进度条收敛到 100%。
        """
        os.makedirs(output_dir, exist_ok=True)
        stage_name = processor.stage_name

        # 仅当完整 marker 验证输入、配置与输出一致时复用 merged 输出。
        merged_path = os.path.join(output_dir, f"{stage_name.lower()}_merged.jsonl")
        marker_path = merged_path + ".complete"
        if self._merged_output_reusable(merged_path, marker_path, input_manifest, processor):
            logger.info("%s: 复用已完成的 merged 输出 %s，跳过拆分与处理", stage_name, merged_path)
            return merged_path

        # 目录化输出（新）：子阶段输出 = output_dir 下的 per-dataset 文件集合
        # （eligible 由 processor 写 <stage>_manifest_<ds>.jsonl，ineligible 在拆分后
        # 按 dataset 追加到同一组文件），不再落 merged 单文件——merged 本就是
        # per-dataset 文件的 concat（百万级记录时每个子阶段 ~4GB 纯冗余写）。
        # 幂等标记为 <stage>_output.complete（含输入/输出文件集合指纹）。
        dir_marker_path = os.path.join(output_dir, f"{stage_name.lower()}_output.complete")
        incomplete_path = os.path.join(output_dir, f"{stage_name.lower()}_output.incomplete")
        if os.path.isfile(incomplete_path):
            # 上轮有可重试失败被中止：不得复用目录输出，进入重处理
            # （processor 级 checkpoint 会恢复已完成记录）
            logger.info("%s: 检测到未完成标记 %s，重新处理", stage_name, incomplete_path)
        elif self._dir_output_reusable(output_dir, dir_marker_path, input_manifest, processor):
            logger.info("%s: 复用已完成的目录化输出 %s，跳过拆分与处理", stage_name, output_dir)
            return output_dir

        # 进入重处理：先失效所有旧完成凭据。若本轮随后失败，旧 marker 不得
        # 对部分重写或残留输出重新授权复用。
        for stale in (merged_path, marker_path, dir_marker_path, incomplete_path):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass
            except OSError as e:
                if stale in (marker_path, dir_marker_path):
                    raise RuntimeError(f"{stage_name} 无法失效旧完成 marker {stale}: {e}") from e
                logger.warning("%s 清理旧产物失败 %s: %s", stage_name, stale, e)

        eligible_path = os.path.join(output_dir, f"{stage_name.lower()}_input_eligible.jsonl")
        ineligible_path = os.path.join(output_dir, f"{stage_name.lower()}_input_ineligible.jsonl")

        eligible_count = 0
        ineligible_count = 0
        bad_line_count = 0
        allow_partial = bool(processor.config.get("allow_partial", False))
        # 本轮输入的 dataset 集合（拆分阶段逐行正则提取，用于处理后识别陈旧
        # per-dataset 文件——不依赖 mtime/inode 等元数据：高速连续写入下
        # mtime 精度可能同 tick 无法区分重写，inode 也可能被复用）
        eligible_datasets: set = set()
        input_datasets: set = set()
        # 百万行场景只保留身份最小映射，不驻留 AudioRecord/整行对象。
        # record_id 是全管线主键，重复时无法定义一进一出的守恒关系，立即失败。
        eligible_identities: Dict[str, str] = {}
        ds_re = re.compile(r'"dataset_id":\s*"([^"]+)"')

        # 单次遍历输入 manifest，eligible/ineligible 分别落盘；
        # ineligible 在写入时就地标记 skipped，合并阶段无需再读输入 manifest。
        # 性能：eligible 行原始透传（免 JSON 解析/序列化往返）；
        # ineligible 行直接改 dict 而绕过 dataclass/asdict 往返（大 manifest 下
        # 单条 24KB × 百万级时这是分钟级差异）。拆分进度定期打日志。
        # 输入可能为 per-dataset 目录（多个 *_manifest_*.jsonl）或单文件。
        # snapshot 完成集合闭合与逐文件身份验证后，拆分必须复用该集合；禁止
        # 在两者之间独立枚举，否则新增文件会只进入 marker 而未交给 processor。
        input_snapshot_before, input_files = self._input_snapshot(
            input_manifest, include_files=True
        )
        total_bytes = sum(item["identity"][2] for item in input_snapshot_before[1])
        read_bytes = 0
        t0 = time.monotonic()
        last_log = t0
        lines_read = 0
        is_passed = _make_stage_pass_checker(prev_stage) if prev_stage else None

        with open(eligible_path, "w", encoding="utf-8") as eligible_f, open(
            ineligible_path, "w", encoding="utf-8"
        ) as ineligible_f:
            for input_file in input_files:
                # 二进制模式读取以精确统计字节数（text 模式 len(line) 是字符数，
                # 标签含中文时进度百分比与 ETA 会系统性失真；与
                # read_manifest_with_progress 的口径保持一致）
                with open(input_file, "rb") as src:
                    for raw in src:
                        read_bytes += len(raw)
                        # errors="replace"：进程被杀后末行的截断多字节字符
                        # （中文标签 ensure_ascii=False）replace 成 U+FFFD 后由
                        # 下方首尾结构检查拦截，不得以 UnicodeDecodeError 中断
                        # stage（allow_partial 才有机会生效）
                        line = raw.decode("utf-8", errors="replace")
                        line_stripped = line.strip()
                        if not line_stripped:
                            continue
                        lines_read += 1
                        # 截断行保留既有“结构不完整”分类与日志契约；首尾完整后
                        # 再确认整行是合法 JSON object，防止宽松 reader 静默吞行。
                        if not (
                            line_stripped.startswith("{") and line_stripped.endswith("}")
                        ):
                            bad_line_count += 1
                            if not allow_partial:
                                raise ValueError(
                                    f"{stage_name} 拆分遇到结构不完整的坏行"
                                    "（疑似截断写入）；如需跳过坏行请设置 "
                                    "allow_partial: true"
                                )
                            if bad_line_count <= 5 or bad_line_count % 10000 == 0:
                                logger.warning(
                                    "%s 拆分时跳过结构不完整的非法 JSON 行（累计 %d 行）: %.100s",
                                    stage_name, bad_line_count, line_stripped,
                                )
                            continue
                        try:
                            parsed_input = json.loads(line_stripped)
                            if not isinstance(parsed_input, dict):
                                raise ValueError("JSON 值不是 object")
                        except (json.JSONDecodeError, ValueError) as e:
                            bad_line_count += 1
                            if not allow_partial:
                                raise ValueError(
                                    f"{stage_name} 拆分遇到非法 JSON object；"
                                    "如需跳过坏行请设置 allow_partial: true"
                                ) from e
                            if bad_line_count <= 5 or bad_line_count % 10000 == 0:
                                logger.warning(
                                    "%s 拆分时跳过非法 JSON 行（累计 %d 行）: %s",
                                    stage_name, bad_line_count, e,
                                )
                            continue
                        if input_transform is not None:
                            parsed_input = input_transform(parsed_input)
                            if not isinstance(parsed_input, dict):
                                raise TypeError(
                                    f"{stage_name} input_transform 必须返回 JSON object"
                                )
                            line_stripped = json.dumps(parsed_input, ensure_ascii=False)
                            line = line_stripped + "\n"
                        if is_passed is None or is_passed(line_stripped):
                            record_id = parsed_input.get("record_id")
                            dataset_id = parsed_input.get("dataset_id")
                            if not isinstance(record_id, str) or not record_id:
                                raise ValueError(
                                    f"{stage_name} eligible 记录缺少非空 record_id"
                                )
                            if not isinstance(dataset_id, str) or not dataset_id:
                                raise ValueError(
                                    f"{stage_name} eligible 记录 {record_id!r} 缺少非空 dataset_id"
                                )
                            if record_id in eligible_identities:
                                raise ValueError(
                                    f"{stage_name} eligible record_id 重复: {record_id!r}"
                                )
                            eligible_identities[record_id] = dataset_id
                            eligible_count += 1
                            eligible_datasets.add(dataset_id)
                            input_datasets.add(dataset_id)
                            eligible_f.write(line if line.endswith("\n") else line_stripped + "\n")
                        else:
                            ineligible_count += 1
                            input_datasets.add(_extract_dataset_id(line_stripped, ds_re))
                            # 快速路径：文本级注入 skipped 标记（免整行 JSON 解析/序列化往返）。
                            # 使用条件（任一不满足即回退 parse+dump）：
                            #   1. '"stages": {' 恰好出现一次（防止嵌套 stages 键或字符串值中
                            #      含有锚点导致注入位置错误）；
                            #   2. 本 stage 键未出现过（防 resume 重跑产生重复键；键若出现在
                            #      锚点之前同样回退，避免误判）。
                            ts_now = datetime.now(timezone.utc).isoformat()
                            stages_anchor = '"stages": {'
                            stage_key = f'"{stage_name}"'
                            pos = line_stripped.find(stages_anchor)
                            stage_key_pos = line_stripped.find(stage_key)
                            if (
                                line_stripped.count(stages_anchor) == 1
                                and pos >= 0
                                and (stage_key_pos < 0 or stage_key_pos > pos)
                            ):
                                inject_at = pos + len(stages_anchor)
                                # 空 stages dict（锚点后只隔空白就是 '}'）时注入不得带尾逗号，
                                # 否则产生 `"C1": {...},}` 非法 JSON，记录会在合并阶段被静默丢弃
                                rest = line_stripped[inject_at:]
                                trailing_comma = "" if rest.lstrip()[:1] == "}" else ","
                                injection = (
                                    f' {json.dumps(stage_name, ensure_ascii=False)}: '
                                    f'{{"status": "skipped", "timestamp": "{ts_now}", '
                                    f'"reason": "prev_stage_{prev_stage}_not_passed"}}{trailing_comma}'
                                )
                                new_line = line_stripped[:inject_at] + injection + rest
                                ineligible_f.write(new_line + "\n")
                            else:
                                try:
                                    rec = json.loads(line_stripped)
                                except json.JSONDecodeError as e:
                                    bad_line_count += 1
                                    if not allow_partial:
                                        raise ValueError(
                                            f"{stage_name} 拆分遇到非法 JSON 行；"
                                            "如需跳过坏行请设置 allow_partial: true"
                                        ) from e
                                    if bad_line_count <= 5 or bad_line_count % 10000 == 0:
                                        logger.warning(
                                            "%s 拆分时跳过非法 JSON 行（累计 %d 行）: %s",
                                            stage_name, bad_line_count, e,
                                        )
                                    continue
                                entry = {
                                    "status": "skipped",
                                    "timestamp": ts_now,
                                    "reason": f"prev_stage_{prev_stage}_not_passed",
                                }
                                rec.setdefault("stages", {})[stage_name] = entry
                                ineligible_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        now = time.monotonic()
                        if now - last_log >= 30:
                            last_log = now
                            elapsed = now - t0
                            rate = lines_read / max(elapsed, 1e-6)
                            pct = read_bytes / max(total_bytes, 1) * 100
                            eta_s = (total_bytes - read_bytes) / max(read_bytes / max(elapsed, 1e-6), 1e-6)
                            # 用 print：本管线未配置 logging，INFO 级日志不可见
                            print(
                                f"  [{stage_name} 拆分中] {lines_read} 行 ({pct:.0f}%), "
                                f"eligible={eligible_count}, ineligible={ineligible_count}, "
                                f"{rate:.0f} 行/s, 预计剩余 {eta_s:.0f}s",
                                flush=True,
                            )

        # eligible 侧坏行守恒校验：每行必须落入 eligible / ineligible / bad 之一；
        # 不等说明有坏行被 eligible 透传吞掉（记录将静默丢失），严格模式直接中止。
        # is_passed 为 None（无 prev_stage）时 eligible == lines_read，恒成立不受影响
        if not allow_partial and (
            eligible_count + ineligible_count + bad_line_count != lines_read
        ):
            raise ValueError(
                f"{stage_name} 拆分行数不守恒：eligible={eligible_count} + "
                f"ineligible={ineligible_count} + bad_lines={bad_line_count} != "
                f"lines_read={lines_read}，存在被 eligible 透传吞掉的坏行"
            )

        # 把本子阶段的工作量计入全局 total（与合并阶段的 callback 次数一一对应）
        if total_callback is not None:
            total_callback(eligible_count)

        logger.info(
            "%s: eligible=%d ineligible=%d bad_lines=%d (prev_stage=%s, 拆分耗时 %.1fs)",
            stage_name,
            eligible_count,
            ineligible_count,
            bad_line_count,
            prev_stage,
            time.monotonic() - t0,
        )

        # processor 输出只包含 eligible 记录（per-dataset 目录：多个 *_manifest_*.jsonl）。
        # fingerprint_input 传稳定的上游输入：eligible 是本次运行刚生成的中间文件，
        # 其 mtime 每次必变，直接进 checkpoint 指纹会让崩溃恢复断点必然失效全量重算。
        # processor 依赖旧文件恢复 checkpoint，不能预删。先快照每个已有
        # per-dataset 文件的 stat 身份，处理后要求所有 eligible dataset 文件
        # 存在且相较本快照确实发生变化。
        pre_run_files = self._snapshot_substage_files(output_dir, stage_name)
        pre_run_names = set(pre_run_files)
        proc_output = processor.process(
            eligible_path, output_dir, fingerprint_input=input_manifest
        )
        input_snapshot_after = self._input_snapshot(input_manifest)
        if input_snapshot_after != input_snapshot_before:
            raise RuntimeError(
                f"{stage_name} processor 处理期间输入 manifest 内容或 identity 发生变化，拒绝发布"
            )
        self._validate_eligible_files_rewritten(
            output_dir, stage_name, eligible_datasets, pre_run_files
        )
        proc_files = _iter_manifest_files(proc_output)

        # 可重试的基础设施失败不可伪装成完成：写未完成标记（阻止目录输出被复用），
        # 保留 processor 输出现场供 checkpoint 恢复，并中止 stage，
        # 确保下游无法消费不完整产物
        has_retryable_failures = getattr(processor, "has_retryable_failures", None)
        if callable(has_retryable_failures) and has_retryable_failures():
            with open(incomplete_path, "w", encoding="utf-8") as f:
                f.write(datetime.now(timezone.utc).isoformat() + "\n")
            raise RuntimeError(f"{stage_name} 存在可重试处理失败，未写完成 marker")

        # 陈旧 per-dataset 文件清理（确定性规则，不依赖 mtime/inode 等元数据：
        # 高速连续写入下 mtime 精度可能同 tick 无法区分重写，inode 也可能被复用）。
        # 规则见 _prune_stale_substage_files docstring。
        self._prune_stale_substage_files(
            output_dir, stage_name, pre_run_names, input_datasets, eligible_datasets
        )
        # 陈旧文件已删除，重新 glob 保证后续 stat/读取不报 FileNotFoundError
        proc_files = _iter_manifest_files(proc_output)

        # eligible 结果行流式统计 + 全局进度回调（纯读一遍，不再落 merged）：
        # 每行触发一次 progress_callback（与拆分阶段 total_callback 一一对应）。
        # 必须在 ineligible 追加之前统计——追加后 ineligible 行与 eligible 行
        # 共处同一 per-dataset 文件，无法再区分（进度回调会超发）。
        # processor 输出逐行严格解析；截断、非法 JSON、非 object 均立即失败，
        # 不得过滤后以少行结果继续发布完成 marker。
        eligible_lines = 0
        # 直接消费输入身份映射，避免百万行时再复制一整张 dict 峰值翻倍。
        remaining_identities = eligible_identities
        identity_error_count = 0
        identity_error_samples = []
        total_bytes = sum(f.stat().st_size for f in proc_files)
        read_bytes = 0
        t0 = time.monotonic()
        last_log = t0
        for path in proc_files:
            with open(path, "rb") as src:
                for line_number, raw in enumerate(src, 1):
                    read_bytes += len(raw)
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        parsed_output = json.loads(stripped)
                        if not isinstance(parsed_output, dict):
                            raise ValueError("JSON 值不是 object")
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                        raise ValueError(
                            f"{stage_name} processor 输出非法: {path}:{line_number}: {e}"
                        ) from e
                    eligible_lines += 1
                    record_id = parsed_output.get("record_id")
                    dataset_id = parsed_output.get("dataset_id")
                    expected_dataset = remaining_identities.pop(record_id, None)
                    if expected_dataset is None:
                        identity_error_count += 1
                        if len(identity_error_samples) < _IDENTITY_ERROR_SAMPLE_LIMIT:
                            identity_error_samples.append(
                                f"{path.name}:{line_number} 未知或重复 "
                                f"record_id={_bounded_identity_text(record_id)}"
                            )
                    elif dataset_id != expected_dataset:
                        identity_error_count += 1
                        if len(identity_error_samples) < _IDENTITY_ERROR_SAMPLE_LIMIT:
                            identity_error_samples.append(
                                f"{path.name}:{line_number} "
                                f"record_id={_bounded_identity_text(record_id)} "
                                f"dataset_id={_bounded_identity_text(dataset_id)} != "
                                f"{_bounded_identity_text(expected_dataset)}"
                            )
                    if progress_callback is not None:
                        progress_callback()
                    now = time.monotonic()
                    if now - last_log >= 30:
                        last_log = now
                        elapsed = now - t0
                        rate = eligible_lines / max(elapsed, 1e-6)
                        pct = read_bytes / max(total_bytes, 1) * 100
                        eta_s = (total_bytes - read_bytes) / max(read_bytes / max(elapsed, 1e-6), 1e-6)
                        # 用 print：本管线未配置 logging，INFO 级日志不可见
                        print(
                            f"  [{stage_name} 汇总中] {eligible_lines} 行 ({pct:.0f}%), "
                            f"{rate:.0f} 行/s, 预计剩余 {eta_s:.0f}s",
                            flush=True,
                        )

        # 先保留既有行数错误契约，再校验同数量下的身份多重集。
        if eligible_lines != eligible_count:
            raise RuntimeError(
                f"{stage_name} eligible 输出行数 {eligible_lines} != "
                f"eligible_count {eligible_count}，拒绝发布不完整输出"
            )
        # 计数相等仍可能“丢 A、重复 B”或篡改 dataset_id，必须在追加
        # ineligible 前拒绝。
        if identity_error_count or remaining_identities:
            missing_count = len(remaining_identities)
            details = list(identity_error_samples)
            if remaining_identities:
                missing_samples = [
                    _bounded_identity_text(value)
                    for value in list(remaining_identities)[:_IDENTITY_ERROR_SAMPLE_LIMIT]
                ]
                details.append(f"缺失 record_id 示例={missing_samples}")
            raise RuntimeError(
                f"{stage_name} eligible 身份不守恒: errors={identity_error_count}, "
                f"missing={missing_count}; " + "; ".join(details)
            )

        # ineligible 按 dataset_id 路由追加到同一组 per-dataset 文件（原始行透传，
        # 免 JSON 解析/序列化往返；dataset_id 用行内正则提取，失败回退 json.loads）。
        # 必须在 eligible 统计之后（见上方注释），在 marker 指纹计算之前
        appended_lines, bad_merge_lines = self._append_ineligible_per_dataset(
            output_dir, stage_name, ineligible_path
        )
        if bad_merge_lines != 0 or appended_lines != ineligible_count:
            raise RuntimeError(
                f"{stage_name} ineligible 追加不守恒: appended_lines={appended_lines}, "
                f"ineligible_count={ineligible_count}, bad_merge_lines={bad_merge_lines}"
            )
        lines_written = eligible_lines + appended_lines

        # 写出完成 marker（含输入/配置指纹与逐文件内容 SHA-256），
        # 下次运行重新流式校验内容后才允许幂等复用。
        self._write_dir_marker(
            dir_marker_path,
            input_manifest,
            processor,
            lines_written,
            output_dir,
            input_snapshot=input_snapshot_before,
        )

        # 合并验证通过（目录输出 + marker 均已落盘）后清理拆分中间产物：
        # eligible/ineligible 只是过程文件，此后无任何消费者（resume 走
        # marker，processor 级 resume 走各自输出 manifest）；
        # 失败路径不走到这里，现场保留。
        if processor.config.get("cleanup_intermediate", True):
            freed = 0
            for path in (eligible_path, ineligible_path):
                try:
                    freed += os.path.getsize(path)
                    os.remove(path)
                except OSError as e:
                    logger.warning("%s 清理中间产物失败 %s: %s", stage_name, path, e)
            if freed:
                logger.info("%s: 已清理拆分中间产物 %.1f GB", stage_name, freed / 1e9)
        return output_dir

    # ------------------------------------------------------------------
    # 目录化输出（per-dataset 文件集合即子阶段输出）
    # ------------------------------------------------------------------
    @staticmethod
    def _snapshot_substage_files(output_dir: str, stage_name: str) -> Dict[str, tuple]:
        """快照已有 per-dataset 文件的 ``(inode, size, mtime_ns)`` 身份。"""
        stem = stage_name.lower()
        snapshot: Dict[str, tuple] = {}
        for name in os.listdir(output_dir):
            if not (name.startswith(f"{stem}_manifest_") and name.endswith(".jsonl")):
                continue
            stat = os.stat(os.path.join(output_dir, name))
            snapshot[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return snapshot

    @staticmethod
    def _validate_eligible_files_rewritten(
        output_dir: str,
        stage_name: str,
        eligible_datasets: set,
        pre_run_files: Dict[str, tuple],
    ) -> None:
        """确认每个 eligible dataset 的输出存在且相较处理前确实重写。"""
        stem = stage_name.lower()
        errors = []
        for dataset_id in eligible_datasets:
            name = f"{stem}_manifest_{dataset_id}.jsonl"
            path = os.path.join(output_dir, name)
            try:
                stat = os.stat(path)
            except OSError as e:
                errors.append(f"{name} 不存在或不可 stat: {e}")
                continue
            current = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if pre_run_files.get(name) == current:
                errors.append(f"{name} 未重写（stat 身份未变化）")
        if errors:
            raise RuntimeError(
                f"{stage_name} eligible per-dataset 输出未重写: " + "; ".join(errors)
            )

    @staticmethod
    def _prune_stale_substage_files(
        output_dir: str,
        stage_name: str,
        pre_run_names: set,
        input_datasets: set,
        eligible_datasets: set,
    ) -> int:
        """按确定性规则删除陈旧 per-dataset 文件，返回删除数。

        对处理前已存在的文件按文件名解析 dataset，两条删除规则：
        - 数据集不在本轮输入中：旧运行的残留，留着会随目录读取错误流入下游；
        - 数据集在输入中但零 eligible：其记录全部在 ineligible 中，删除后由
          调用方追加重建（截断语义，避免旧记录重复）。
        数据集 eligible>0 时 processor 必定写文件（passed/rejected/restore 都
        落盘；全 infra-failed 会被 has_retryable_failures 中止），因此处理前
        存在的 eligible 文件必已被本轮重写，保留即可。
        残余风险：eligible 行被 processor 静默丢弃（proc 实际未写该数据集文件）
        时旧文件会被误保留——该场景本身就是 processor 数据丢失 bug，
        不在此处防护。处理前不存在、本轮新建的文件一律保留。
        """
        stem = stage_name.lower()
        manifest_prefix = f"{stem}_manifest_"
        removed = 0
        errors = []
        for name in pre_run_names:
            ds_from_name = name[len(manifest_prefix):-len(".jsonl")]
            if ds_from_name in eligible_datasets:
                continue  # 本轮必已重写
            if ds_from_name not in input_datasets:
                reason = "数据集不在本轮输入中"
            else:
                reason = "数据集本轮零 eligible"
            try:
                os.remove(os.path.join(output_dir, name))
                removed += 1
                logger.info("%s 清理陈旧 per-dataset 文件 %s（%s）", stage_name, name, reason)
            except OSError as e:
                errors.append(f"{name}: {e}")
        if errors:
            raise RuntimeError(
                f"{stage_name} 清理陈旧 per-dataset 文件失败: " + "; ".join(errors)
            )
        return removed

    @staticmethod
    def _append_ineligible_per_dataset(
        output_dir: str, stage_name: str, ineligible_path: str
    ) -> tuple:
        """把 ineligible 原始行按 dataset_id 追加到 <stage>_manifest_<ds>.jsonl。

        返回 (appended_lines, bad_lines)。逐行解析 JSON 并只读取顶层
        dataset_id，避免嵌套同名字段把记录路由到错误文件；非法行跳过并限频告警。
        """
        stem = stage_name.lower()
        handles: Dict[str, Any] = {}
        appended = 0
        bad = 0
        try:
            with open(ineligible_path, "rb") as src:
                for raw in src:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    if not (stripped.startswith(b"{") and stripped.endswith(b"}")):
                        bad += 1
                        if bad <= 5 or bad % 10000 == 0:
                            logger.warning(
                                "%s ineligible 追加时跳过不完整行（累计 %d 行，疑似截断写入）",
                                stage_name, bad,
                            )
                        continue
                    text = stripped.decode("utf-8", errors="replace")
                    try:
                        rec = json.loads(text)
                    except json.JSONDecodeError:
                        bad += 1
                        if bad <= 5 or bad % 10000 == 0:
                            logger.warning(
                                "%s ineligible 追加时跳过非法 JSON 行（累计 %d 行）",
                                stage_name, bad,
                            )
                        continue
                    ds = validate_dataset_id(
                        rec.get("dataset_id") or "unknown"
                        if isinstance(rec, dict) else "unknown"
                    )
                    path = os.path.join(output_dir, f"{stem}_manifest_{ds}.jsonl")
                    h = handles.get(path)
                    if h is None:
                        h = open(path, "a", encoding="utf-8")
                        handles[path] = h
                    h.write(text + "\n")
                    appended += 1
        finally:
            for h in handles.values():
                h.close()
        return appended, bad

    # ------------------------------------------------------------------
    # 子阶段幂等（目录化输出复用）
    # ------------------------------------------------------------------
    def _dir_output_reusable(
        self,
        output_dir: str,
        dir_marker_path: str,
        input_manifest: str,
        processor: Processor,
    ) -> bool:
        """以逐文件 SHA-256 判断目录化输出是否可直接复用。"""
        stage_name = processor.stage_name
        if not os.path.isfile(dir_marker_path):
            return False
        try:
            with open(dir_marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if not (
            isinstance(marker, dict)
            and marker.get("schema") == _MARKER_SCHEMA
            and marker.get("version") == _MARKER_VERSION
            and marker.get("output_kind") == "per_dataset_dir"
            and self._valid_content_digests(marker.get("content_digests"))
            and self._valid_content_digests(marker.get("input_fingerprints"))
        ):
            return False
        try:
            input_fingerprints, _ = self._input_content_fingerprints(input_manifest)
            current_snapshot = self._input_snapshot(input_manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False
        if not (
            marker.get("input_fingerprints") == input_fingerprints
            and (
                "input_identities" not in marker
                or marker.get("input_identities") == current_snapshot[1]
            )
            and marker.get("config_hash") == config_hash(processor.config)
        ):
            return False
        try:
            current_digests, line_count = self._content_digests(output_dir)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            logger.info("%s: 目录输出内容校验失败，重新处理: %s", stage_name, e)
            return False
        return (
            marker.get("content_digests") == current_digests
            and marker.get("line_count") == line_count
        )

    @staticmethod
    def _valid_content_digests(value: Any) -> bool:
        """校验 marker 中按文件名排序的 SHA-256 摘要结构。"""
        if not isinstance(value, list):
            return False
        names = []
        for item in value:
            if not isinstance(item, dict) or set(item) != {
                "name",
                "sha256",
                "line_count",
            }:
                return False
            name = item["name"]
            digest = item["sha256"]
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(item["line_count"]) is not int
                or item["line_count"] < 0
            ):
                return False
            names.append(name)
        return names == sorted(names) and len(names) == len(set(names))

    @staticmethod
    def _snapshot_content(
        path: str, *, validate_json: bool, include_files: bool = False
    ) -> tuple:
        """安全快照内容，返回按相对文件名排序的 SHA-256 与非空行数。

        扫描前后同时核对 fd 与路径身份；正常并发 writer 若原地修改、原子替换
        或增删文件，本次快照均失效，调用方不得据此发布或复用完成 marker。
        ``include_files`` 供拆分入口复用闭合后的同一文件集合。
        """
        validate_input_path(path, directory=os.path.isdir(path))
        root = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
        files = _iter_manifest_files(path)
        initial_names = [os.path.relpath(str(p), root) for p in files]
        if initial_names != sorted(initial_names):
            raise ValueError("manifest 文件顺序不稳定")
        digests = []
        stable_identities = []
        line_count = 0
        for manifest_path, relative_name in zip(files, initial_names):
            validate_input_path(manifest_path)
            path_before = os.stat(manifest_path, follow_symlinks=False)
            digest = hashlib.sha256()
            file_line_count = 0
            with open(manifest_path, "rb") as f:
                fd_before = os.fstat(f.fileno())
                if (path_before.st_dev, path_before.st_ino) != (
                    fd_before.st_dev,
                    fd_before.st_ino,
                ):
                    raise OSError(f"扫描前文件已被替换: {relative_name}")
                for raw in f:
                    digest.update(raw)
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    if validate_json:
                        data = json.loads(stripped.decode("utf-8"))
                        if not isinstance(data, dict):
                            raise ValueError(f"{relative_name} JSON 值不是 object")
                    file_line_count += 1
                    line_count += 1
                fd_after = os.fstat(f.fileno())
            path_after = os.stat(manifest_path, follow_symlinks=False)
            before_identity = (
                fd_before.st_dev,
                fd_before.st_ino,
                fd_before.st_size,
                fd_before.st_mtime_ns,
                fd_before.st_ctime_ns,
            )
            after_identity = (
                fd_after.st_dev,
                fd_after.st_ino,
                fd_after.st_size,
                fd_after.st_mtime_ns,
                fd_after.st_ctime_ns,
            )
            current_identity = (
                path_after.st_dev,
                path_after.st_ino,
                path_after.st_size,
                path_after.st_mtime_ns,
                path_after.st_ctime_ns,
            )
            if before_identity != after_identity or after_identity != current_identity:
                raise OSError(f"扫描期间文件发生变化: {relative_name}")
            stable_identities.append(current_identity)
            digests.append(
                {
                    "name": relative_name,
                    "sha256": digest.hexdigest(),
                    "line_count": file_line_count,
                }
            )
        final_files = _iter_manifest_files(path)
        final_names = [os.path.relpath(str(p), root) for p in final_files]
        if final_names != initial_names:
            raise OSError("扫描期间 manifest 文件集合发生变化")
        for manifest_path, relative_name, expected in zip(
            final_files, final_names, stable_identities
        ):
            final_stat = os.stat(manifest_path, follow_symlinks=False)
            current = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            )
            if current != expected:
                raise OSError(f"扫描期间文件发生变化: {relative_name}")
        snapshot = (digests, line_count)
        if include_files:
            return snapshot, final_files, stable_identities
        return snapshot

    @staticmethod
    def _content_digests(path: str) -> tuple:
        """严格校验 JSONL，并返回强内容指纹与记录数。"""
        return StageOrchestratorBase._snapshot_content(path, validate_json=True)

    @staticmethod
    def _input_snapshot(path: str, *, include_files: bool = False) -> tuple:
        """捕获输入快照，并可返回其已闭合验证的同一文件集合。"""
        (digests, _), files, stable_identities = (
            StageOrchestratorBase._snapshot_content(
                path, validate_json=False, include_files=True
            )
        )
        identities = [
            {
                "name": digest["name"],
                "identity": [identity[0], identity[1], identity[2]],
            }
            for digest, identity in zip(digests, stable_identities)
        ]
        snapshot = (digests, identities)
        if include_files:
            return snapshot, files
        return snapshot

    @staticmethod
    def _input_content_fingerprints(path: str) -> tuple:
        """快照输入原始内容；允许 ``allow_partial`` 路径中的坏行参与哈希。"""
        return StageOrchestratorBase._snapshot_content(path, validate_json=False)

    def _write_dir_marker(
        self,
        dir_marker_path: str,
        input_manifest: str,
        processor: Processor,
        line_count: int,
        output_dir: str,
        input_snapshot: Optional[tuple] = None,
    ) -> None:
        """写目录化输出完成 marker（绑定拆分时输入内容与 identity）。"""
        if input_snapshot is None:
            input_snapshot = self._input_snapshot(input_manifest)
        input_fingerprints, input_identities = input_snapshot
        content_digests, scanned_line_count = self._content_digests(output_dir)
        if scanned_line_count != line_count:
            raise RuntimeError(
                f"{processor.stage_name} marker 行数 {scanned_line_count} != {line_count}"
            )
        marker = {
            "schema": _MARKER_SCHEMA,
            "version": _MARKER_VERSION,
            "output_kind": "per_dataset_dir",
            "input_manifest": os.path.abspath(input_manifest),
            "input_fingerprints": input_fingerprints,
            "input_identities": input_identities,
            "config_hash": config_hash(processor.config),
            "line_count": line_count,
            "content_digests": content_digests,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        marker_tmp = dir_marker_path + ".tmp"
        with open(marker_tmp, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
            # marker 是复用凭据：掉电后丢失会整段重算，tmp 与目录都需 fsync
            f.flush()
            os.fsync(f.fileno())
        atomic_replace_durable(marker_tmp, dir_marker_path)

    # ------------------------------------------------------------------
    # 子阶段幂等（merged 输出复用）
    # ------------------------------------------------------------------
    def _merged_output_reusable(
        self,
        merged_path: str,
        marker_path: str,
        input_manifest: str,
        processor: Processor,
    ) -> bool:
        """以逐文件 SHA-256 判断 legacy merged 输出是否可直接复用。"""
        stage_name = processor.stage_name
        if not os.path.isfile(merged_path) or not os.path.isfile(marker_path):
            return False
        try:
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if not (
            isinstance(marker, dict)
            and marker.get("schema") == _MARKER_SCHEMA
            and marker.get("version") == _MARKER_VERSION
            and marker.get("output_kind") == "merged_file"
            and self._valid_content_digests(marker.get("content_digests"))
            and self._valid_content_digests(marker.get("input_fingerprints"))
        ):
            return False
        try:
            input_fingerprints, _ = self._input_content_fingerprints(input_manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False
        if not (
            marker.get("input_fingerprints") == input_fingerprints
            and marker.get("config_hash") == config_hash(processor.config)
        ):
            return False
        return self._merged_output_matches_marker(merged_path, marker, stage_name)

    def _merged_output_matches_marker(
        self, merged_path: str, marker: Dict[str, Any], stage_name: str
    ) -> bool:
        """流式校验 merged JSONL，并比较 marker 中的内容摘要与行数。"""
        try:
            current_digests, line_count = self._content_digests(merged_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            logger.info("%s: merged 内容校验失败，重新处理: %s", stage_name, e)
            return False
        return (
            marker.get("content_digests") == current_digests
            and marker.get("line_count") == line_count
        )

    @staticmethod
    def _input_fingerprint(input_manifest: str) -> Dict[str, Any]:
        """输入 manifest 指纹：单文件或 per-dataset 目录（文件集合）。

        用于兼容调用方的轻量输入元数据校验；正式 marker 复用使用
        ``_input_content_fingerprints`` 的内容 SHA-256 与 identity 复验。
        """
        files = _iter_manifest_files(input_manifest)
        if not files:
            return {"size": 0, "mtime_ns": 0, "count": 0, "names": ""}
        return {
            "size": sum(f.stat().st_size for f in files),
            "mtime_ns": max(f.stat().st_mtime_ns for f in files),
            "count": len(files),
            "names": ",".join(f.name for f in files),
        }

    @staticmethod
    def _write_marker(
        marker_path: str,
        input_manifest: str,
        processor: Processor,
        line_count: int,
        merged_path: str,
    ) -> None:
        """写 legacy merged 完成 marker（marker v4 + 输入/输出内容指纹）。"""
        input_fingerprints, _ = StageOrchestratorBase._input_content_fingerprints(
            input_manifest
        )
        content_digests, scanned_line_count = StageOrchestratorBase._content_digests(
            merged_path
        )
        if scanned_line_count != line_count:
            raise RuntimeError(
                f"{processor.stage_name} marker 行数 {scanned_line_count} != {line_count}"
            )
        marker = {
            "schema": _MARKER_SCHEMA,
            "version": _MARKER_VERSION,
            "output_kind": "merged_file",
            "input_manifest": os.path.abspath(input_manifest),
            "input_fingerprints": input_fingerprints,
            "config_hash": config_hash(processor.config),
            "line_count": line_count,
            "content_digests": content_digests,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        marker_tmp = marker_path + ".tmp"
        with open(marker_tmp, "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace_durable(marker_tmp, marker_path)
