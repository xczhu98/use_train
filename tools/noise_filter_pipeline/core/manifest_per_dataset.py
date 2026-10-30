"""
per-dataset manifest 读写与断点初始化（增量/全量共存的核心）。

per-dataset 架构：每个 stage 的 manifest 按 ``dataset_id`` 分文件
（``<stage>_manifest_<dataset_id>.jsonl``），增量跑只更新输入涉及的数据集
文件，旧数据集文件永久保留——下游断点恢复 / 全量重建时据此跳过已处理数据。

本模块提供：
  - PerDatasetManifestWriter：按 dataset_id 路由的缓冲写入器；
  - split_manifest_by_dataset：stage 结尾把 merged 输出汇总为 per-dataset 文件；
  - find_legacy_single_manifests / reject_legacy_single_manifest：遗留单文件
    （旧代码产物 ``<stage>_manifest.jsonl``）的检测与显式拒绝；
  - link_per_dataset_manifests：子阶段输出链接到 stage 根目录；
  - init_checkpointed_output：统一初始化「writer + checkpoint + prev_records」，
    消除各 processor 的重复样板。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .checkpoint import CheckpointTracker
from .manifest_io import (
    AudioRecord,
    _iter_manifest_files,
    append_manifest,
    build_checkpoint_fingerprint,
    link_or_copy_file,
    read_manifest,
    read_records_by_id,
)

logger = logging.getLogger(__name__)


def validate_dataset_id(dataset_id: Any) -> str:
    """校验可安全用于 per-dataset 文件名的 dataset_id。"""
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id in (".", "..")
        or "/" in dataset_id
        or "\\" in dataset_id
        or os.path.basename(dataset_id) != dataset_id
    ):
        raise ValueError(f"非法 dataset_id，不能用于输出文件名: {dataset_id!r}")
    return dataset_id


class PerDatasetManifestWriter:
    """按 dataset_id 分文件的 manifest 写入器（per-dataset manifest）。

    输出文件 ``<base_stem>_<dataset_id>.jsonl``（如 ``b_manifest_dns5.jsonl``），
    不同数据集互不覆盖：增量跑只更新输入涉及的 dataset 文件，旧数据集文件
    保留，供下游断点恢复 / 全量重建时跳过已处理数据。

    - 首次写入某 dataset 时截断其旧文件（覆盖当次结果，避免重复记录）；
    - close 默认**不删除**遗留单文件 ``<base>``（如 ``b_manifest.jsonl``）：
      增量跑时旧单文件保留，其内容仍完整（下游 per-dataset 读取不会消费它，
      StageRunner 检测到该文件会显式拒绝启动，见 reject_legacy_single_manifest）；
    - 仅 ``remove_legacy_base=True``（本 stage 自己的 merged→per-dataset 拆分）
      时，close 删除 merged 单文件，避免新旧格式记录同时存在；
    - 线程安全，可在多线程 producer 中使用。
    """

    def __init__(
        self,
        base_path: str,
        batch_size: int = 100,
        remove_legacy_base: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self.base_path = base_path
        self.batch_size = int(batch_size)
        self._remove_legacy_base = remove_legacy_base
        self._dir = os.path.dirname(base_path) or "."
        self._stem = os.path.splitext(os.path.basename(base_path))[0]
        self._buffers: Dict[str, List[AudioRecord]] = {}
        self._truncated: Set[str] = set()
        self._lock = threading.Lock()
        # 写盘串行锁：缓冲 swap 用 _lock（持锁期间不做文件 I/O），append_manifest
        # 落盘段用 _write_lock 串行化——否则并发 flush（主线程 + 后台 flush 线程）
        # 写同一 dataset 文件时，>8KB 记录被拆成多个 write syscall，线程间交错
        # 会撕裂 JSONL 行
        self._write_lock = threading.Lock()
        self._closed = False
        self._wrote_any = False

    def _path_for(self, dataset_id: str) -> str:
        dataset_id = validate_dataset_id(dataset_id)
        return os.path.join(self._dir, f"{self._stem}_{dataset_id}.jsonl")

    def append(self, rec: AudioRecord) -> None:
        ds = validate_dataset_id(rec.dataset_id or "unknown")
        should_flush = False
        with self._lock:
            if self._closed:
                raise RuntimeError("PerDatasetManifestWriter 已关闭")
            buf = self._buffers.setdefault(ds, [])
            buf.append(rec)
            self._wrote_any = True
            should_flush = len(buf) >= self.batch_size
        if should_flush:
            self.flush(ds)

    def extend(self, records: Iterable[AudioRecord]) -> None:
        for rec in records:
            self.append(rec)

    def flush(self, dataset_id: Optional[str] = None) -> None:
        # 锁内（_lock）：收集待写批次并 swap 缓冲（截断标记也在锁内记录）；
        # 锁外（_write_lock）：写盘段串行化，既不持 _lock 做文件 I/O 阻塞并发
        # append，也防止并发 flush 的 write syscall 交错撕裂行
        batches: List[Tuple[str, List[AudioRecord]]] = []
        with self._lock:
            ids = [dataset_id] if dataset_id else list(self._buffers.keys())
            for ds in ids:
                buf = self._buffers.get(ds)
                if not buf:
                    continue
                if ds not in self._truncated:
                    # 首次写该 dataset：截断旧文件，保证当次结果覆盖而非追加
                    path = self._path_for(ds)
                    if os.path.exists(path):
                        os.remove(path)
                    self._truncated.add(ds)
                batches.append((ds, buf))
                self._buffers[ds] = []
        with self._write_lock:
            for ds, batch in batches:
                append_manifest(iter(batch), self._path_for(ds))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            ids = list(self._buffers.keys())
        # 先在开放状态下 flush，避免 close 与 append 交错导致尾部记录遗失。
        self.flush()
        batches: List[Tuple[str, List[AudioRecord]]] = []
        with self._lock:
            if self._closed:
                return
            # flush 后仍可能有并发 append 写入的缓冲；在同一把 _lock 内 swap 出来
            # 并切换 closed（此后 append 一律拒绝），确保任何已接受的记录均在
            # 本批中、均会持久化；落盘放到 _write_lock 段，不持 _lock 做文件 I/O
            for ds in ids + [ds for ds in self._buffers if ds not in ids]:
                buf = self._buffers.get(ds)
                if not buf:
                    continue
                if ds not in self._truncated:
                    path = self._path_for(ds)
                    if os.path.exists(path):
                        os.remove(path)
                    self._truncated.add(ds)
                self._buffers[ds] = []
                batches.append((ds, buf))
            self._closed = True
        with self._write_lock:
            for ds, batch in batches:
                append_manifest(iter(batch), self._path_for(ds))
        # 仅本 stage 自己的 merged→per-dataset 拆分后删除 merged 单文件；
        # 增量写必须保留，否则旧数据记录会被误删（见 __init__ 注释）
        if self._remove_legacy_base and self._wrote_any and os.path.isfile(self.base_path):
            os.remove(self.base_path)

    def __enter__(self) -> "PerDatasetManifestWriter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
            return
        # 异常退出仍需刷出普通 writer 已接受的记录，但迁移 writer 不得把
        # “部分拆分”误判为成功并删除 legacy 源。
        remove_legacy_base = self._remove_legacy_base
        self._remove_legacy_base = False
        try:
            self.close()
        finally:
            self._remove_legacy_base = remove_legacy_base


def find_legacy_single_manifests(directory: str) -> List[str]:
    """返回目录下的遗留单文件 manifest（``<stage>_manifest.jsonl``，无 dataset 后缀）。

    per-dataset 架构下 stage 根目录只应有 ``<stage>_manifest_<dataset_id>.jsonl``；
    命中单文件即说明该目录由旧代码写出，或人工把单文件放进了 stage 根目录。
    """
    if not os.path.isdir(directory):
        return []
    return sorted(
        str(fp) for fp in Path(directory).glob("*_manifest.jsonl") if fp.is_file()
    )


def reject_legacy_single_manifest(directory: str, *, stage_label: str) -> None:
    """遗留单文件 manifest 存在时显式拒绝，并给出手工迁移指引。

    per-dataset 架构只读取 ``<stage>_manifest_<dataset_id>.jsonl``，旧单文件既不会
    被消费也不会被自动拆分/删除；静默忽略会让旧记录无声消失，自动拆分又会隐式
    改写用户数据，因此这里统一 fail-fast，交由人工处理。
    """
    found = find_legacy_single_manifests(directory)
    if not found:
        return
    raise RuntimeError(
        f"{stage_label} 检测到遗留单文件 manifest（旧代码格式）：{', '.join(found)}；"
        "per-dataset 架构不再自动拆分或删除该文件，也不会把它计入输入。"
        "请人工确认后二选一："
        "(1) 该单文件内容仍需保留时，先拆分为 per-dataset 文件"
        "（split_manifest_by_dataset(<单文件>, <同名单文件路径>) 会写出 "
        "<stage>_manifest_<dataset_id>.jsonl），确认拆分结果完整后删除单文件；"
        "(2) 内容已由 per-dataset 文件覆盖时，备份后删除该单文件。"
        "处理完成后重跑本 stage。"
    )


def split_manifest_by_dataset(
    src_path: str,
    dst_base: str,
    batch_size: int = 1000,
    remove_legacy_base: bool = False,
) -> int:
    """把单文件 manifest 按 dataset_id 拆分为 per-dataset 文件（``<dst_base>_<id>.jsonl``）。

    用于 stage 结尾把子阶段 merged 输出（含 skipped 透传记录）拆分为
    per-dataset 文件落到 stage 根目录，供下游按目录读取；增量场景下
    本次未涉及的数据集文件保留不动。返回拆分的记录数。

    remove_legacy_base=True 用于 stage B 把本轮自己产出的 merged 输出拆分后删除该
    merged 文件；其余场景（C/D 汇总、人工迁移）保留源文件，避免误删用户数据。
    """
    if not os.path.isfile(src_path):
        return 0
    writer = PerDatasetManifestWriter(
        dst_base, batch_size=batch_size, remove_legacy_base=remove_legacy_base
    )
    count = 0
    # 拆分可能涉及百 GB 级单文件，全程无输出会让用户误判卡死；
    # 定期打印进度（条数/速率/耗时）。
    t0 = time.monotonic()
    last_log = t0
    with writer, open(src_path, "r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                if not isinstance(data, dict):
                    raise ValueError("JSON 值不是 object")
                rec = AudioRecord.from_dict(data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(
                    f"manifest 拆分遇到非法记录: {src_path}:{line_number}: {e}"
                ) from e
            writer.append(rec)
            count += 1
            now = time.monotonic()
            if now - last_log >= 30:
                last_log = now
                elapsed = now - t0
                logger.info(
                    "per-dataset 拆分中（%s -> %s*）：%d 条, %.0f 条/s, 已耗时 %.0fs",
                    src_path, dst_base, count, count / max(elapsed, 1e-6), elapsed,
                )
    return count


def link_per_dataset_manifests(
    src_dir: str,
    src_prefix: str,
    dst_dir: str,
    dst_prefix: str,
) -> int:
    """把 src_dir 下 ``<src_prefix>_manifest_*.jsonl`` 汇总到
    dst_dir 下 ``<dst_prefix>_manifest_<dataset_id>.jsonl``。

    用于 stage 结尾把子阶段 per-dataset 输出硬链接/拷贝到 stage 根目录，
    供下游按"stage 根目录"统一约定读取。返回处理的文件数。
    增量场景下 dst 中本次未涉及的数据集文件保留不动。
    """
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for f in sorted(Path(src_dir).glob(f"{src_prefix}_manifest_*.jsonl")):
        name = f.name
        ds = name[len(f"{src_prefix}_manifest_"):-len(".jsonl")]
        dst = os.path.join(dst_dir, f"{dst_prefix}_manifest_{ds}.jsonl")
        link_or_copy_file(str(f), dst)
        n += 1
    return n


def link_stage_output(
    src_dir: str,
    dst_dir: str,
    dst_prefix: str,
) -> int:
    """把子阶段目录化输出（per-dataset 文件集合）链接到 stage 根目录。

    src_dir 下应只有 ``<substage>_manifest_*.jsonl`` 一类前缀（如 C3 目录下的
    ``c3_manifest_*.jsonl``）；自动推断该前缀并调用
    ``link_per_dataset_manifests(src_dir, <substage>, dst_dir, dst_prefix)``。
    前缀不唯一（目录混入多个子阶段产物）时 fail-fast，避免错误链接。
    """
    prefixes = {
        f.name.split("_manifest_")[0]
        for f in Path(src_dir).glob("*_manifest_*.jsonl")
    }
    if not prefixes:
        logger.warning("link_stage_output: %s 下无 per-dataset manifest", src_dir)
        return 0
    if len(prefixes) != 1:
        raise ValueError(
            f"link_stage_output: {src_dir} 下存在多个 manifest 前缀 {sorted(prefixes)}，"
            "无法确定子阶段来源"
        )
    return link_per_dataset_manifests(src_dir, prefixes.pop(), dst_dir, dst_prefix)


def init_checkpointed_output(
    output_manifest: str,
    output_dir: str,
    local_config: Dict[str, Any],
    input_manifest: str,
    *,
    enable_checkpoint: bool,
    checkpoint_path: Optional[str],
    save_interval: int,
    manifest_batch_size: int,
    stage_label: str,
    dry_run: bool = False,
    fingerprint_input: Optional[str] = None,
) -> Tuple[PerDatasetManifestWriter, CheckpointTracker, Dict[str, AudioRecord]]:
    """统一初始化「per-dataset writer + checkpoint + prev_records 恢复索引」。

    消除 B/C1/C2/C3/D1/D2 各 processor 中重复的样板：创建 writer（必须先于
    checkpoint 创建，作为 pre-save callback 保证 checkpoint 落盘前 manifest
    已 flush）、构造带配置指纹（本阶段 + 上游）的 CheckpointTracker、从输出
    目录下所有 per-dataset 文件恢复已处理记录（只物化 checkpoint 确认过的）。

    返回 (writer, ckpt, prev_records)；调用方负责 ``with writer:`` 生命周期。
    """
    # manifest_writer 必须先创建，以便作为 CheckpointTracker 的 pre-save callback，
    # 确保 checkpoint 落盘前 manifest 缓冲已刷出。
    writer = PerDatasetManifestWriter(output_manifest, batch_size=manifest_batch_size)
    default_ckpt = "checkpoint.dry_run.json" if dry_run else "checkpoint.json"
    ckpt = CheckpointTracker(
        checkpoint_path=checkpoint_path or os.path.join(output_dir, default_ckpt),
        enabled=enable_checkpoint,
        save_interval=save_interval,
        pre_save_callbacks=[writer.flush],
        # 配置指纹：本阶段 + 上游配置（见 build_checkpoint_fingerprint），
        # 上游配置变更会失效旧断点、全量重算，避免恢复基于旧输入的结果；
        # fingerprint_input 供子阶段传入稳定上游产物（eligible 中间文件的
        # mtime 每次运行都变，直接用于指纹会让断点必然失效）
        config_hash=build_checkpoint_fingerprint(
            local_config, input_manifest, stat_input=fingerprint_input
        ),
    )
    # 重跑前先索引上次输出：checkpoint 命中时恢复带本阶段标记的记录，
    # 避免 resume 后记录退回"未处理"状态；只物化 ckpt 确认过的记录
    prev_records = read_records_by_id(
        output_dir, predicate=lambda rec: ckpt.is_done(rec.record_id)
    )
    if prev_records:
        logger.info(
            "%s 断点续跑：从 %s 恢复 %d 条已处理记录",
            stage_label, output_dir, len(prev_records),
        )
    return writer, ckpt, prev_records
